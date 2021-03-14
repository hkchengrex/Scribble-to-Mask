import torch
import torch.nn.functional as F
import torch.nn as nn
from util.tensor_util import compute_tensor_iu


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

iou_hooks_to_be_used = [
    get_iou_hook,
]

class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=10000, end_warm=30000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it >= self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class LossComputer:
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE()

    def compute(self, data, it):
        losses = {}

        losses['total_loss'], losses['p'] = self.bce(data['logits'], data['cls_gt'], it)

        total_i, total_u = compute_tensor_iu(data['mask']>0.5, data['gt']>0.5)
        losses['hide_iou/i'] = total_i
        losses['hide_iou/u'] = total_u

        return losses
