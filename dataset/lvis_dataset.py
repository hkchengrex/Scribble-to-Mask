from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.lvis import LVIS
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed


class LVISTransformDataset(Dataset):
    """
    Apply random transform on LVIS images.
    """
    def __init__(self, root, lvis: LVIS):
        self.root = root

        self.lvis = lvis
        # We only want the large enough ones
        self.ann_list = list(lvis.get_ann_ids(area_rng=[50*50, float('inf')]))

        print('%d annotations out of %d used in LVIS.' % (len(self.ann_list), len(self.lvis.anns)))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.0), shear=10, resample=Image.BILINEAR, fillcolor=im_mean),
            transforms.Resize(480, Image.BILINEAR),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.0), shear=10, resample=Image.NEAREST, fillcolor=0),
            transforms.Resize(480, Image.NEAREST),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img_name, gt = self.lvis.load_img_ann(self.ann_list[idx])

        im = Image.open(path.join(self.root, img_name)).convert('RGB')
        gt = Image.fromarray(gt*255)

        # from_zero - no previous mask
        from_zero = np.random.rand() < 0.33
        if from_zero:
            another_gt = False
        else:
            # another_gt - loads another GT mask from the 
            # same image and do something with it
            another_gt = (np.random.rand() < 0.75)

        if another_gt:
            image_id = self.lvis.anns[self.ann_list[idx]]['image_id']
            a_gt_list = list(self.lvis.get_ann_ids(img_ids=[image_id], area_rng=[30*30, float('inf')]))
            a_gt_id = np.random.choice(a_gt_list)

            # make sure we don't pick the same one
            if a_gt_id == self.ann_list[idx]:
                another_gt = False
            else:
                _, a_gt = self.lvis.load_img_ann(a_gt_id)
                a_gt = Image.fromarray(a_gt*255)

        sequence_seed = np.random.randint(2147483647)

        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        im = self.im_lone_transform(im)
        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)
        if another_gt:
            reseed(sequence_seed)
            a_gt = self.gt_dual_transform(a_gt)

        gt_np = np.array(gt)
        if from_zero:
            seg = np.zeros_like(gt_np)
        else:
            iou_max = 0.95
            iou_min = 0.4
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            seg = perturb_mask(gt_np, iou_target=iou_target)

        if another_gt:
            a_gt_np = np.array(a_gt)
            add_extra_n_srb = np.random.rand() < 0.5
            if add_extra_n_srb:
                # Consider positive scribbles from another instance (that turns into negative here)
                another_n_srb, _ = get_scribble(np.zeros_like(seg), a_gt_np, True)
            else:
                # Add another GT to current seg and work from there
                a_seg = perturb_mask(a_gt_np)
                seg = ((seg.astype(np.float32) + a_seg.astype(np.float32)).clip(0, 255)).astype(np.uint8)
                
        # Generate scribbles
        p_srb, n_srb = get_scribble(seg, gt_np, from_zero=from_zero)
        if another_gt and add_extra_n_srb:
            n_srb = (n_srb + another_n_srb).clip(0, 1)

        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)

        p_srb = torch.from_numpy(p_srb)
        n_srb = torch.from_numpy(n_srb)
        srb = torch.stack([p_srb, n_srb], 0).float()
        seg = self.final_gt_transform(seg)

        info = {}
        info['name'] = img_name

        # Class label version of GT
        cls_gt = (gt>0.5).long().squeeze(0)

        data = {
            'rgb': im,
            'gt': gt,
            'cls_gt': cls_gt,
            'seg': seg,
            'srb': srb,
            'info': info
        }

        return data


    def __len__(self):
        return len(self.ann_list)
