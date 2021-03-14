import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

from model.network import deeplabv3plus_resnet50
from model.aggregate import aggregate_wbg_channel as aggregate
from model.losses import LossComputer, iou_hooks_to_be_used
from util.log_integrator import Integrator
from util.image_saver import pool_pairs


class S2MModel:
    def __init__(self, para, logger=None, save_path=None, local_rank=0, world_size=1):
        self.para = para
        self.local_rank = local_rank

        self.S2M = nn.parallel.DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(
                deeplabv3plus_resnet50(num_classes=1, output_stride=16, pretrained_backbone=False)
            ).cuda(),
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Setup logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.train_integrator.add_hook(iou_hooks_to_be_used)
        self.loss_computer = LossComputer(para)

        self.train()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.S2M.parameters()), lr=para['lr'], weight_decay=1e-7)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, para['steps'], para['gamma'])

        # Logging info
        self.report_interval = 50
        self.save_im_interval = 800
        self.save_model_interval = 20000
        if para['debug']:
            self.report_interval = self.save_im_interval = 1

    def do_pass(self, data, it=0):
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        Fs = data['rgb']
        Ss = data['seg']
        Rs = data['srb']
        Ms = data['gt']

        inputs = torch.cat([Fs, Ss, Rs], 1)
        prob = torch.sigmoid(self.S2M(inputs))
        logits, mask = aggregate(prob)

        out['logits'] = logits
        out['mask'] = mask

        if self._do_log or self._is_train:
            losses = self.loss_computer.compute({**data, **out}, it)

            # Logging
            if self._do_log:
                self.integrator.add_dict(losses)
                if self._is_train:
                    if it % self.save_im_interval == 0 and it != 0:
                        if self.logger is not None:
                            images = {**data, **out}
                            size = (384, 384)
                            self.logger.log_cv2('train/pairs', pool_pairs(images, size=size), it)

        if self._is_train:
            if (it) % self.report_interval == 0 and it != 0:
                if self.logger is not None:
                    self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                    self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval, it)
                self.last_time = time.time()
                self.train_integrator.finalize('train', it)
                self.train_integrator.reset_except_hooks()

            if it % self.save_model_interval == 0 and it != 0:
                if self.logger is not None:
                    self.save(it)

            # Backward pass
            self.optimizer.zero_grad() 
            losses['total_loss'].backward() 
            self.optimizer.step()
            self.scheduler.step()

    def save(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % it)
        torch.save(self.S2M.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': it,
            'network': self.S2M.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.S2M.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        map_location = 'cuda:%d' % self.local_rank
        self.S2M.module.load_state_dict(torch.load(path, map_location={'cuda:0': map_location}))
        print('Network weight loaded:', path)

    def load_deeplab(self, path):
        map_location = 'cuda:%d' % self.local_rank

        cur_dict = self.S2M.module.state_dict()
        src_dict = torch.load(path, map_location={'cuda:0': map_location})['model_state']

        for k in list(src_dict.keys()):
            if type(src_dict[k]) is not int:
                if src_dict[k].shape != cur_dict[k].shape:
                    print('Reloading: ', k)
                    if 'bias' in k:
                        # Reseting the class prob bias
                        src_dict[k] = torch.zeros_like((src_dict[k][0:1]))
                    elif src_dict[k].shape[1] != 3:
                        # Reseting the class prob weight
                        src_dict[k] = torch.zeros_like((src_dict[k][0:1]))
                        nn.init.orthogonal_(src_dict[k])
                    else:
                        # Adding the mask and scribbles channel
                        pads = torch.zeros((64,3,7,7), device=src_dict[k].device)
                        nn.init.orthogonal_(pads)
                        src_dict[k] = torch.cat([src_dict[k], pads], 1)

        self.S2M.module.load_state_dict(src_dict)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.S2M.train()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.S2M.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.S2M.eval()
        return self

