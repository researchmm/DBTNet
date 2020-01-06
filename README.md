# DBTNet
MXNet version of the code for our NeurIPS'19 paper ["Learning Deep Bilinear Transformation for Fine-grained Image Representation"](http://papers.nips.cc/paper/8680-learning-deep-bilinear-transformation-for-fine-grained-image-representation.pdf)

Bilinear feature transformation has shown the state-of-the-art performance in learning fine-grained image representations. The proposed DBTNet can deeply integrate bilinear features into CNN to learn fine-grained image representations.

# Framework
![alt text](https://user-images.githubusercontent.com/35843017/71718086-15bc9600-2e55-11ea-98b9-14d9295c46be.jpg)

# Main Results


| Method           |Dimension| CUB-200-2011 | Stanford-Car | Aircraft|
| -----------------|:-------:|:------------:|:------------:|:-------:|
| Compact Bilinear | 14k     |   81.6       | 88.6         |81.6     |
| Kernel Pooling   | 14k     |   84.7       | 91.1         |85.7     |
| iSQRT-COV        | 8k      |   87.3       | 91.7         |89.5     |
| iSQRT-COV        | 32k     |   88.1       | 92.8         |90.0     |
| DBTNet-50 (ours) | 2k      |   87.5       | 94.1         |91.2     |
| DBTNet-101 (ours)| 2k      |   88.1       | 94.5         |91.6     |

# Prerequisites
MXNet   1.3.1

GluonCV 0.3.0

# Quick Start

## Prepare the data:

download the imagenet data:
```
cd data/imagenet/
wget https://australiav100data.blob.core.windows.net/heliang/imagenet_train.rec
wget https://australiav100data.blob.core.windows.net/heliang/imagenet_train.idx
wget https://australiav100data.blob.core.windows.net/heliang/imagenet_val.rec
wget https://australiav100data.blob.core.windows.net/heliang/imagenet_val.idx
```
download the CUB-200-2011 dataset:
```
cd data/
wget https://australiav100data.blob.core.windows.net/heliang/cub.tar
tar -xvf cub.tar
```

## Train the model on ImageNet dataset:
```
cd code/
bash train_imagenet_dbt.sh
```

## Fine-tune the model on CUB-200-2011 dataset:
The ImageNet pretrained [model](https://australiav100data.blob.core.windows.net/heliang/dbt_imagenet.params) is available.

```
cd code/
bash ft_cub_dbt.sh
```

# Pytorch Version
On going. Welcome to reimplement and share the DBT code in pytorch.

# Citation
If any part of our paper and code is helpful to your work, please generously cite with:

```
@incollection{NIPS2019_8680,
title = {Learning Deep Bilinear Transformation for Fine-grained Image Representation},
author = {Zheng, Heliang and Fu, Jianlong and Zha, Zheng-Jun and Luo, Jiebo},
booktitle = {Advances in Neural Information Processing Systems 32},
pages = {4279--4288},
year = {2019}
```
