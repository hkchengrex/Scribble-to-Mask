# MiVOS (CVPR 2021) - Scribble To Mask

[Ho Kei Cheng](https://hkchengrex.github.io/), Yu-Wing Tai, Chi-Keung Tang

[[arXiv]](https://arxiv.org/abs/2103.07941) [[Paper PDF]](https://arxiv.org/pdf/2103.07941.pdf) [[Project Page]](https://hkchengrex.github.io/MiVOS/)

A simplistic network that turns scribbles to mask. It supports multi-object segmentation using soft-aggregation. Don't expect SOTA results from this model!

![Ex1](https://imgur.com/HesuB4x.gif) ![Ex2](https://imgur.com/NmCrCE1.gif)

## Overall structure and capabilities

| | [MiVOS](https://github.com/hkchengrex/MiVOS) | [Mask-Propagation](https://github.com/hkchengrex/Mask-Propagation)| [Scribble-to-Mask](https://github.com/hkchengrex/Scribble-to-Mask)  |
| ------------- |:-------------:|:-----:|:-----:|
| DAVIS/YouTube semi-supervised evaluation | :x: | :heavy_check_mark: | :x: |
| DAVIS interactive evaluation | :heavy_check_mark: | :x: | :x: |
| User interaction GUI tool | :heavy_check_mark: | :x: | :x: |
| Dense Correspondences | :x: | :heavy_check_mark: | :x: |
| Train propagation module | :x: | :heavy_check_mark: | :x: |
| Train S2M (interaction) module | :x: | :x: | :heavy_check_mark: |
| Train fusion module | :heavy_check_mark: | :x: | :x: |
| Generate more synthetic data | :heavy_check_mark: | :x: | :x: |

## Requirements

The package versions shown here are the ones that I used. You might not need the exact versions.

- PyTorch `1.6.0`
- torchvision `0.7.0`
- opencv-contrib `4.2.0`
- davis-interactive (<https://github.com/albertomontesg/davis-interactive>)
- gitpython for training
- gdown for downloading pretrained models

Refer to the official [PyTorch guide](<https://pytorch.org/>) for installing PyTorch/torchvision. The rest can be installed by:

`pip install opencv-contrib-python gitpython gdown`

## Pretrained model

[Download](https://drive.google.com/file/d/1HKwklVey3P2jmmdmrACFlkXtcvNxbKMM/view?usp=sharing) and put the model in `./saves/`. Alternatively use the provided `download_model.py`.

[[OneDrive Mirror]](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hkchengad_connect_ust_hk/EjHifAlvYUFPlEG2qBr-GGQBb1XyzxUvizJiQKBf8te2Cw?e=a6mxKz)

## Interactive GUI

`python interactive.py --image <image>`

Controls:

```bash
Mouse Left - Draw scribbles
Mouse middle key - Switch positive/negative
Key f - Commit changes, clear scribbles
Key r - Clear everything
Key d - Switch between overlay/mask view
Key s - Save masks into a temporary output folder (./output/)
```

## Known issues

The model almost always needs to focus on at least one object. It is very difficult to erase all existing masks from an image using scribbles.

## Training

### Datasets

1. Download and extract [LVIS](https://www.lvisdataset.org/dataset) training set.
2. Download and extract [a set of static image segmentation datasets](https://drive.google.com/file/d/1wUJq3HcLdN-z1t4CsUhjeZ9BVDb9YKLd/view?usp=sharing). These are already downloaded for you if you used the `download_datasets.py` in [Mask-Propagation](https://github.com/hkchengrex/Mask-Propagation).

```bash
├── lvis
│   ├── lvis_v1_train.json
│   └── train2017
├── Scribble-to-Mask
└── static
    ├── BIG_small
    └── ...
```

### Commands

Use the `deeplabv3plus_resnet50` pretrained model provided [here](https://github.com/VainF/DeepLabV3Plus-Pytorch).

`CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 train.py --id s2m --load_deeplab <path_to_deeplab.pth>`

## Credit

Deeplab implementation and pretrained model: <https://github.com/VainF/DeepLabV3Plus-Pytorch>.

## Citation

Please cite our paper if you find this repo useful!

```bibtex
@inproceedings{MiVOS_2021,
  title={Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2021}
}
```

Contact: <hkchengrex@gmail.com>
