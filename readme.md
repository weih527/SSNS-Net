# Semi-Supervised Neuron Segmentation via Reinforced Consistency Learning

Wei Huang, Chang Chen, Zhiwei Xiong(*), Yueyi Zhang, Xuejin Chen, Xiaoyan Sun, Feng Wu

*Corresponding Author

University of Science and Technology of China (USTC)



## Introduction

This repository is the **official implementation** of the paper, "Semi-Supervised Neuron Segmentation via Reinforced Consistency Learning", where more visual results and implementation details are presented.



## Installation

This code was tested with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. It is worth mentioning that, besides some commonly used image processing packages, you also need to install some special post-processing packages for neuron segmentation, such as [waterz](https://github.com/funkey/waterz) and [elf](https://github.com/constantinpape/elf).

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows,

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v5.4
```

or

```shell
docker pull renwu527/auto-emseg:v5.4
```



## Dataset

| Datasets   | Sizes                        | Resolutions | Species | Download (Processed) |
| ---------- | ---------------------------- | ----------- | ----------- | ----------- |
| [AC3/AC4 ](https://software.rc.fas.harvard.edu/lichtman/vast/AC3AC4Package.zip)   | 1024x1024x256, 1024x1024x100 | 6x6x30 nm^3 | Mouse | [BaiduYun](https://pan.baidu.com/s/1sSTkh7g9tccb_uZOvySQqQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |
| [CREMI](https://cremi.org/)      | 1250x1250x125 (x3)           | 4x4x40 nm^3 | Drosophila | [BaiduYun](https://pan.baidu.com/s/1q-irVm5aoSXL5eQiqyYs1w) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |
| [Kasthuri15](https://lichtman.rc.fas.harvard.edu/vast/Thousands_6nm_spec_lossless.vsv) | 10747x12895x1850             | 6x6x30 nm^3 | Mouse | [BaiduYun](https://pan.baidu.com/s/136Eml2gBHYIklVPP0MI_kQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1JAdoKchlWrHnbTXvnFn6pWWwx6VIiMH3?usp=sharing) |



## Training stage

Take the training on the AC3 dataset as an example.

### 1. Pre-training

```shell
python pre_training.py -c=pretraining_snemi3d
```

### 2. Consistency learning

Weight Sharing (WS)

```shell
python main.py -c=seg_snemi3d_d5_u200
```

EMA

```shell
python main_ema.py -c=seg_snemi3d_d5_1024_u200_ema
```



## Validation stage

Take the validation on the AC3 dataset as an example.

### 1. Predict affinities

```shell
 python inference.py -c=seg_snemi3d_d5_1024_u200 -mn=seg_ac3_d5_1024_u200_WS -id=seg_ac3_d5_1024_u200_WS -m=snemi3d-ac3
```

### 2. Evaluate on Waterz

```shell
python2 evaluate_waterz.py -mn=seg_ac3_d5_1024_u200_WS -id=seg_ac3_d5_1024_u200_WS -m=snemi3d-ac3
```

### 3. Evaluate on LMC

```shell
python evaluate_lmc.py -mn=seg_ac3_d5_1024_u200_WS -id=seg_ac3_d5_1024_u200_WS -m=snemi3d-ac3
```



## Model Zoo

We provide the trained models on the AC3 dataset at BaiduYun and GoogleDrive, including the pre-trained model and the segmentation models on different numbers of labeled (\*L) and unlabeled (\*U) sections (1024x1024).

| Methods       | Models                         | Download                                                     |
| ------------- | ------------------------------ | ------------------------------------------------------------ |
| pre-training  | pretraining_snemi3d.ckpt       | [BaiduYun](https://pan.baidu.com/s/1kxor7JbLFZuEoCRClD_DVw) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 5L+200U (WS)  | seg_ac3_d5_1024_u200_WS.ckpt   | [BaiduYun](https://pan.baidu.com/s/1Is0qpJn1XsoxVxGcKl5M_Q) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 5L+200U (EMA) | seg_ac3_d5_1024_u200_EMA.ckpt  | [BaiduYun](https://pan.baidu.com/s/1GEEYw-hFD4v0Nir4vPeWIQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 10L           | seg_kasthuri15_d10.ckpt        | [BaiduYun](https://pan.baidu.com/s/1NIh7ZU2IJsLLRSedlHLlFQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 10L+200U      | seg_kasthuri15_d10_u200.ckpt   | [BaiduYun](https://pan.baidu.com/s/1kAKhct5Y7t_J2TmIuuPhsQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 50L+200U      | seg_kasthuri15_d50_u200.ckpt   | [BaiduYun](https://pan.baidu.com/s/1JzwltZhAgQ1-xbrFGMUIwg) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 100L          | seg_kasthuri15_d100.ckpt       | [BaiduYun](https://pan.baidu.com/s/1NCUXuUQELMrUZ4r_fFh21w) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 100L+200U     | seg_kasthuri15_d100_u200.ckpt  | [BaiduYun](https://pan.baidu.com/s/1HdjVuu8ic3CQtWtV8yzsYA) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |
| 100L+1000U    | seg_kasthuri15_d100_u1000.ckpt | [BaiduYun](https://pan.baidu.com/s/1jYJsCFZxbylVTO6ECTjSMw) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/1pr7SQE6Kuog4oNat0vbCCd3Pa2OUgQgR?usp=sharing) |



## More visual results on the Kasthuri15 dataset

To demonstrate the generalizability performance of our method on the large-scale EM data, we test our models on the Kasthuri15 dataset. The quantitative results can be found in our paper. Here, we provide more visual results on the Subset3 dataset to qualitatively demonstrate the superiority of our semi-supervised method compared with the existing supervised method with full labeled data (100L).

![id5884](./images/id5884.png)

![id1455](./images/id1455.png)

![id6913](./images/id6913.png)

Left images are the results of the supervised method (100L), while right images are the results of our semi-supervised method (100L+1000U), where blue and red arrows represent split and merge errors, respectively.



## Related Projects

[funkey/waterz](https://github.com/funkey/waterz)

[constantinpape/elf](https://github.com/constantinpape/elf)



## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (weih527@mail.ustc.edu.cn).

