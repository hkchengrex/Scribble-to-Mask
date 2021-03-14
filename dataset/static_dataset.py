import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.mask_perturb import perturb_mask
from dataset.gen_scribble import get_scribble
from dataset.reseed import reseed


class StaticTransformDataset(Dataset):
    """
    Apply random transform on static images.

    Method 0 - FSS style (class/1.jpg class/1.png)
    Method 1 - Others style (XXX.jpg XXX.png)
    """
    def __init__(self, root, method=0):
        self.root = root
        self.method = method

        if method == 0:
            # Get images
            self.im_list = []
            classes = os.listdir(self.root)
            for c in classes:
                imgs = os.listdir(path.join(root, c))
                jpg_list = [im for im in imgs if 'jpg' in im[-3:].lower()]

                joint_list = [path.join(root, c, im) for im in jpg_list]
                self.im_list.extend(joint_list)

        elif method == 1:
            self.im_list = [path.join(self.root, im) for im in os.listdir(self.root) if '.jpg' in im]

        print('%d images found in %s' % (len(self.im_list), root))

        self.im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, resample=Image.BILINEAR, fillcolor=im_mean),
            transforms.Resize(480, Image.BILINEAR),
            transforms.RandomCrop((480, 480), pad_if_needed=True, fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.8,1.25), shear=10, resample=Image.NEAREST, fillcolor=0),
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
        im = Image.open(self.im_list[idx]).convert('RGB')

        if self.method == 0:
            gt = Image.open(self.im_list[idx][:-3]+'png').convert('L')
        else:
            gt = Image.open(self.im_list[idx].replace('.jpg','.png')).convert('L')

        sequence_seed = np.random.randint(2147483647)

        reseed(sequence_seed)
        im = self.im_dual_transform(im)
        im = self.im_lone_transform(im)
        reseed(sequence_seed)
        gt = self.gt_dual_transform(gt)

        gt_np = np.array(gt)
        if np.random.rand() < 0.33:
            # from_zero - no previous mask
            seg = np.zeros_like(gt_np)
            from_zero = True
        else:
            iou_max = 0.95
            iou_min = 0.4
            iou_target = np.random.rand()*(iou_max-iou_min) + iou_min
            seg = perturb_mask(gt_np, iou_target=iou_target)
            from_zero = False

        # Generate scribbles
        p_srb, n_srb = get_scribble(seg, gt_np, from_zero=from_zero)

        im = self.final_im_transform(im)
        gt = self.final_gt_transform(gt)

        p_srb = torch.from_numpy(p_srb)
        n_srb = torch.from_numpy(n_srb)
        srb = torch.stack([p_srb, n_srb], 0).float()
        seg = self.final_gt_transform(seg)

        info = {}
        info['name'] = self.im_list[idx]

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
        return len(self.im_list)
