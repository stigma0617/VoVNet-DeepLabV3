import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from collections import OrderedDict
import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152','vovnet39', 'vovnet57','vovnet39', 'densenet161', 'densenet201' ]


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

_GN = False


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        # return super(Conv2d, self).forward(x)
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ASPP(nn.Module):

    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C
        self._depth = depth
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, conv=None, norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = conv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, num_groups=None, weight_std=False, beta=False):
        self.inplanes = 64
        self.norm = lambda planes, momentum=0.05: nn.BatchNorm2d(planes, momentum=momentum) if num_groups is None else nn.GroupNorm(num_groups, planes)
        self.conv = Conv2d if weight_std else nn.Conv2d

        super(ResNet, self).__init__()
        if not beta:
            self.conv1 = self.conv(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        else:
            self.conv1 = nn.Sequential(
                self.conv(3, 64, 3, stride=2, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False),
                self.conv(64, 64, 3, stride=1, padding=1, bias=False))
        self.bn1 = self.norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1) #ywlee
        self.aspp = ASPP(512 * block.expansion, 256, num_classes, conv=self.conv, norm=self.norm)

        for m in self.modules():
            if isinstance(m, self.conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or dilation != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, dilation=max(1, dilation/2), bias=False),
                self.norm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=max(1, dilation/2), conv=self.conv, norm=self.norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, conv=self.conv, norm=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = nn.Upsample(size, mode='bilinear', align_corners=True)(x)
        return x


def resnet50(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        if num_groups and weight_std:
            pretrained_dict = torch.load('data/R-50-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, num_groups=None, weight_std=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_groups=num_groups, weight_std=weight_std, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        if num_groups and weight_std:
            pretrained_dict = torch.load('data/R-101-GN-WS.pth.tar')
            overlap_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
            assert len(overlap_dict) == 312
        elif not num_groups and not weight_std:
            pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
            overlap_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        else:
            raise ValueError('Currently only support BN or GN+WS')
        model_dict.update(overlap_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

#################################### VoVNet ##################################

def conv3x3(in_channels, out_channels, module_name, postfix, dilation=1, stride=1, groups=1, kernel_size=3):
    """3x3 convolution with padding"""
    return [
        (f'{module_name}_{postfix}/conv',
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation,  groups=groups, bias=False)),
        (f'{module_name}_{postfix}/norm',
            nn.GroupNorm(num_groups=_num_groups, num_channels=out_channels, eps=1e-5)
            if _GN else nn.BatchNorm2d(out_channels)
        ),
        (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))
    ]


def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """3x3 convolution with padding"""
    return [
        (f'{module_name}_{postfix}/conv',
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                   bias=False)),
        (f'{module_name}_{postfix}/norm',
            nn.GroupNorm(num_groups=_num_groups, num_channels=out_channels, eps=1e-5)
            if _GN else nn.BatchNorm2d(out_channels)
         ),
        (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))
    ]


class _OSA_module(nn.Module):
    
    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block,  module_name, dilation=1,  identity=False):
        super(_OSA_module, self).__init__()
        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i, dilation))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):

        identity_feat = x
            
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, in_ch, stage_ch, concat_ch,  block_per_stage, layer_per_block, stage_num, stride, dilation):
        super(_OSA_stage, self).__init__()

        # if not stage_num==2:
        if stride == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, module_name, dilation))
        
        for i in range(block_per_stage-1):
            module_name = f'OSA{stage_num}_{i+2}'
            self.add_module(module_name, _OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, module_name, dilation, identity=True))


class VoVNet(nn.Module):

    def __init__(self, 
                 config_stage_ch, 
                 config_concat_ch, 
                 block_per_stage, 
                 layer_per_block, 
                 stride_list,
                 dilation_list, 
                 num_classes, 
                 num_groups=None, 
                 weight_std=False, 
                 beta=False):


        super(VoVNet, self).__init__()

        global _GN 
        global _num_groups
        if num_groups is not None:
            _GN = True
            _num_groups = num_groups

        # Stem module
        stem =  conv3x3(3,   64, 'stem', '1', stride=2)
        stem += conv3x3(64,  64, 'stem', '2', stride=1)
        stem += conv3x3(64, 128, 'stem', '3', stride=2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4): #num_stages
            name = 'stage%d' % (i+2)
            self.stage_names.append(name)
            self.add_module(name,_OSA_stage(in_ch_list[i],
                                            config_stage_ch[i], 
                                            config_concat_ch[i], 
                                            block_per_stage[i],
                                            layer_per_block,
                                            i+2,
                                            stride_list[i],
                                            dilation_list[i]))
 
        # ASPP module
        self.aspp = ASPP(config_concat_ch[-1], 256, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        
        size = (x.shape[2], x.shape[3])
        x = self.stem(x)
        
        for name in self.stage_names:
            x  = getattr(self, name)(x)

        x = self.aspp(x)

        return nn.Upsample(size, mode='bilinear', align_corners=True)(x)


def vovnet57(pretrained=False, **kwarg):

    r"""VoVNet-57 model from `"An Energy and GPU-Computation Efficient Backbone Networks"
    In version 2, we add identity mapping

    Args:
        pretrained (bool) : If True, returns a model pre-trained on ImageNet

    """
    model = VoVNet(config_stage_ch=[128, 160, 192, 224], 
                   config_concat_ch=[256, 512, 768, 1024], 
                   block_per_stage=[1,1,4,3], 
                   layer_per_block=5,
                   stride_list=[1,2,2,1],
                   dilation_list = [1,1,1,2],
                   **kwarg)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load('/home/ywlee/DeepLabv3.pytorch/data/pretrained/VOV57v4_onlyStateDict_norm.pth.tar')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def vovnet39(pretrained=False, **kwarg):

    r"""VoVNet-39 model from `"An Energy and GPU-Computation Efficient Backbone Networks"
    In version 2, we add identity mapping

    Args:
        pretrained (bool) : If True, returns a model pre-trained on ImageNet

    """
    model = VoVNet(config_stage_ch=[128, 160, 192, 224], 
                   config_concat_ch=[256, 512, 768, 1024], 
                   block_per_stage=[1,1,2,2], 
                   layer_per_block=5,
                   stride_list=[1,2,2,1],
                   dilation_list = [1,1,1,2],
                   **kwarg)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load('/home/ywlee/DeepLabv3.pytorch/data/pretrained/VOV39v4_onlyStateDict_norm.pth.tar')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model



class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=dilation,
                                           dilation=dilation, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, 
    			 growth_rate=32, 
                 block_config=(6, 12, 24, 16),
                 num_init_features=64, 
                 bn_size=4, 
                 drop_rate=0, 
                 stride_list=None,
                 dilation_list=[], 
                 num_classes=21, 
                 num_groups=None, 
                 weight_std=False, 
                 beta=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                # drop_rate=drop_rate)
                                drop_rate=drop_rate, dilation=dilation_list[i]) #ywlee
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    stride=stride_list[i])
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # ASPP layer
        self.aspp = ASPP(num_features, 256, num_classes)


        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        features = self.features(x)
        out = F.relu(features, inplace=True)
        x = self.aspp(out)
        return nn.Upsample(size, mode='bilinear', align_corners=True)(x)


def _load_state_dict(model, model_url, progress=True):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    model_dict = model.state_dict()
    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        if 'classifier' in key:
            del state_dict[key]
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


def _densenet(arch, growth_rate, block_config, num_init_features, stride_list, dilation_list, pretrained, **kwargs):
    
    model = DenseNet(growth_rate=growth_rate, 
    				 block_config=block_config, 
    				 num_init_features=num_init_features, 
    				 stride_list=stride_list, 
    				 dilation_list=dilation_list, 
    				 **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch])
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _densenet(arch='densenet201', 
                     growth_rate=32, 
                     block_config=(6, 12, 48, 32), 
                     num_init_features=64,
                     stride_list=[2,2,1],
                     dilation_list=[1,1,1,2], 
                     pretrained=pretrained, 
                     **kwargs)

def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _densenet(arch='densenet161', 
                     growth_rate=48, 
                     block_config=(6, 12, 36, 24), 
                     num_init_features=96,
                     stride_list=[2,2,1],
                     dilation_list=[1,1,1,2], 
                     pretrained=pretrained, 
                     **kwargs)