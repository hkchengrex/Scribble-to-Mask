import os
from os import path
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

from model.network import deeplabv3plus_resnet50 as S2M
from model.aggregate import aggregate_wbg_channel as aggregate
from dataset.range_transform import im_normalization
from util.tensor_util import pad_divide_by


class InteractiveManager:
    def __init__(self, model, image, mask):
        self.model = model

        if args.cpu:
            self.image = im_normalization(TF.to_tensor(image)).unsqueeze(0)
            self.mask = TF.to_tensor(mask).unsqueeze(0)
        else:
            self.image = im_normalization(TF.to_tensor(image)).unsqueeze(0).cuda()
            self.mask = TF.to_tensor(mask).unsqueeze(0).cuda()

        h, w = self.image.shape[-2:]
        self.image, self.pad = pad_divide_by(self.image, 16)
        self.mask, _ = pad_divide_by(self.mask, 16)
        self.last_mask = None

        # Positive and negative scribbles
        self.p_srb = np.zeros((h, w), dtype=np.uint8)
        self.n_srb = np.zeros((h, w), dtype=np.uint8)

        # Used for drawing
        self.pressed = False
        self.last_ex = self.last_ey = None
        self.positive_mode = True
        self.need_update = True

    def mouse_down(self, ex, ey):
        self.last_ex = ex
        self.last_ey = ey
        self.pressed = True
        if self.positive_mode:
            cv2.circle(self.p_srb, (ex, ey), radius=3, color=(1), thickness=-1)
        else:
            cv2.circle(self.n_srb, (ex, ey), radius=3, color=(1), thickness=-1)
        self.need_update = True

    def mouse_move(self, ex, ey):
        if not self.pressed:
            return
        if self.positive_mode:
            cv2.line(
                self.p_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=3
            )
        else:
            cv2.line(
                self.n_srb, (self.last_ex, self.last_ey), (ex, ey), (1), thickness=3
            )
        self.need_update = True
        self.last_ex = ex
        self.last_ey = ey

    def mouse_up(self):
        self.pressed = False

    def run_s2m(self):
        # Convert scribbles to tensors
        if args.cpu:
            Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float()
            Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float()
        else:
            Rsp = torch.from_numpy(self.p_srb).unsqueeze(0).unsqueeze(0).float().cuda()
            Rsn = torch.from_numpy(self.n_srb).unsqueeze(0).unsqueeze(0).float().cuda()
        Rs = torch.cat([Rsp, Rsn], 1)
        Rs, _ = pad_divide_by(Rs, 16)

        # Use the network to do stuff
        inputs = torch.cat([self.image, self.mask, Rs], 1)
        _, mask = aggregate(torch.sigmoid(net(inputs)))

        # We don't overwrite current mask until commit
        self.last_mask = mask
        np_mask = (mask.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8)

        if self.pad[2] + self.pad[3] > 0:
            np_mask = np_mask[self.pad[2] : -self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            np_mask = np_mask[:, self.pad[0] : -self.pad[1]]

        return np_mask

    def commit(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        if self.last_mask is not None:
            self.mask = self.last_mask

    def clean_up(self):
        self.p_srb.fill(0)
        self.n_srb.fill(0)
        self.mask.zero_()
        self.last_mask = None


parser = ArgumentParser()
parser.add_argument("--image", default="ust_cat.jpg")
parser.add_argument("--model", default="saves/s2m.pth")
parser.add_argument("--mask", default=None)
parser.add_argument("--cpu", default=False)
args = parser.parse_args()

# network stuff
net = S2M()
if args.cpu:
    net.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    net = net.eval()
else:
    net.load_state_dict(torch.load(args.model))
    net = net.cuda().eval()
torch.set_grad_enabled(False)

# Reading stuff
image = cv2.imread(args.image, cv2.IMREAD_COLOR)
h, w = image.shape[:2]
if args.mask is None:
    mask = np.zeros((h, w), dtype=np.uint8)
else:
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)

manager = InteractiveManager(net, image, mask)


def mouse_callback(event, x, y, *args):
    if event == cv2.EVENT_LBUTTONDOWN:
        manager.mouse_down(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        manager.mouse_up()
    elif event == cv2.EVENT_MBUTTONDOWN:
        manager.positive_mode = not manager.positive_mode
        if manager.positive_mode:
            print("Entering positive scribble mode.")
        else:
            print("Entering negative scribble mode.")

    # Draw
    if event == cv2.EVENT_MOUSEMOVE:
        manager.mouse_move(x, y)


def comp_image(image, mask, p_srb, n_srb):
    color_mask = np.zeros_like(image, dtype=np.uint8)
    color_mask[:, :, 2] = 1
    if len(mask.shape) == 2:
        mask = mask[:, :, None]
    comp = (image * 0.5 + color_mask * mask * 0.5).astype(np.uint8)
    comp[p_srb > 0.5, :] = np.array([0, 255, 0], dtype=np.uint8)
    comp[n_srb > 0.5, :] = np.array([255, 0, 0], dtype=np.uint8)

    return comp


# OpenCV setup
cv2.namedWindow("S2M demo")
cv2.setMouseCallback("S2M demo", mouse_callback)

print(
    "Usage: python interactive.py --image <image> --model <model> [Optional: --mask initial_mask]"
)
print("This GUI is rudimentary; the network is naively designed.")
print("Mouse Left - Draw scribbles")
print("Mouse middle key - Switch positive/negative")
print("Key f - Commit changes, clear scribbles")
print("Key r - Clear everything")
print("Key d - Switch between overlay/mask view")
print("Key s - Save masks into a temporary output folder (./output/)")

display_comp = True
while 1:
    if manager.need_update:
        np_mask = manager.run_s2m()
        if display_comp:
            display = comp_image(image, np_mask, manager.p_srb, manager.n_srb)
        else:
            display = np_mask
        manager.need_update = False

    cv2.imshow("S2M demo", display)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("f"):
        manager.commit()
        manager.need_update = True
    elif k == ord("s"):
        print("saved")
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/%s" % path.basename(args.image), np_mask)
    elif k == ord("d"):
        display_comp = not display_comp
        manager.need_update = True
    elif k == ord("r"):
        manager.clean_up()
        manager.need_update = True
    elif k == 27:
        break

cv2.destroyAllWindows()
