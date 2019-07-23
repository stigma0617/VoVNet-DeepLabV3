## VoVNet.DeepLabV3

This is a pytorch implementation of [DeepLabV3](https://arxiv.org/abs/1706.05587) with [VoVNet](https://arxiv.org/abs/1904.09730) Backbone Networks. This code based on pytorch implementation of [DeepLabV3.pytorch](https://github.com/chenxi116/DeepLabv3.pytorch).


### Highlights

 - Memory efficient
 - Better performance
 - Faster speed

### Comparison with ResNet & DenseNet backbones

 - For fair comparison, **totally SAME training setup** except for backbone
 - 50 epoch
 - 0.007 base_lr
 - 16 batch size
 - same ASSP module & parameters
 - V100 GPU
 - pytorch 1.1.0a0+3752916
 - CUDA v10
 - cuDnn v7.3


|   Backbone   |  mIoU | inference   time (ms) | Memory usage (MB) | Energy  Efficiency  (J/frame) | DOWNLOAD |
|:------------:|:-----:|:---------------------:|:-----------------:|:-----------------------------:| :---:    |
|   ResNet-50  | 74.27 |           24          |        2193       |              4.1              | [link](https://www.dropbox.com/s/djru33qnugcw0p7/deeplab_resnet50_pascal_v3_bn_lr7e-3_epoch50.pth)
| DenseNet-201 | 75.63 |           50          |        3945       |               7               | [link](https://www.dropbox.com/s/xsqbfbbpoh479kl/deeplab_densenet201_pascal_v3_bn_lr7e-3_epoch50.pth)
|**VoV-39**    |**75.71**|     **19**          |        **1901**   |              **3.1**          | [link](https://www.dropbox.com/s/oqqozntgrowmfb1/deeplab_vovnet39_pascal_v3_bn_lr7e-3_epoch50.pth)
|  ResNet-101  | 76.81 |           32          |        2865       |              15.8             | [link](https://www.dropbox.com/s/vqpifempofpujvr/deeplab_resnet101_pascal_v3_bn_lr7e-3_epoch50.pth)
| DenseNet-161 | 76.13 |           49          |        4523       |              8.3              | [link](https://www.dropbox.com/s/t3au8o1n7u6ijk6/deeplab_densenet161_pascal_v3_bn_lr7e-3_epoch50.pth)
|**VoV-57**    |**77.4**|     **25**          |        **2251**   |              **4.2**            | [link](https://www.dropbox.com/s/ykfj7enw9x5fhv6/deeplab_vovnet57_pascal_v3_bn_lr7e-3_epoch50.pth)


### ImageNet pretrained weight

- [VoVNet-39](https://www.dropbox.com/s/s7f4vyfybyc9qpr/vovnet39_statedict_norm.pth)
- [VoVNet-57](https://www.dropbox.com/s/b826phjle6kbamu/vovnet57_statedict_norm.pth)


### Preparation

```bash
git clone https://github.com/stigma0617/VoVNet-DeepLabV3.git
cd VoVNet-DeepLabV3

mkdir -p data/pretrained
cd data/pretrained
wget https://www.dropbox.com/s/b826phjle6kbamu/vovnet57_statedict_norm.pth
wget https://www.dropbox.com/s/s7f4vyfybyc9qpr/vovnet39_statedict_norm.pth
```

#### PASCAL VOC 2012 Dataset

```bash

cd ~/VoVNet-DeeplabV3/data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
cd VOCdevkit/VOC2012/
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip
```

### Training

Specifying a backbone network with ```--backbone```,

For VoVNet-39, ```--backbone vovnet39```
```bash
python main.py --train --exp bn_lr7e-3 --epochs 50 --base_lr 0.007 --backbone vovnet39
```

### Evaluation
use the same command except delete --train
```bash
wget https://www.dropbox.com/s/oqqozntgrowmfb1/deeplab_vovnet39_pascal_v3_bn_lr7e-3_epoch50.pth -P data/
python main.py --exp bn_lr7e-3 --epochs 50 --base_lr 0.007 --backbone vovnet39
```
