'''
ResNet in PyTorch.absFor Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385

Note: cifar_resnet18 constructs the same model with that from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

import torch.nn as nn
from builder import ConvBuilder
from constants import RESNET50_ORIGIN_DEPS_FLATTENED, resnet_bottleneck_origin_deps_flattened, rc_convert_flattened_deps, rc_origin_deps_flattened
import torch
import numpy as np
from utils.misc import efficient_torchvision_style_normalize
import torch.nn.functional as F

class BottleneckBranch(nn.Module):

    def __init__(self, builder:ConvBuilder, in_channels, deps, stride=1):
        super(BottleneckBranch, self).__init__()
        assert len(deps) == 3
        self.conv1 = builder.Conv2dBNReLU(in_channels, deps[0], kernel_size=1,stride=1)
        self.conv2 = builder.Conv2dBNReLU(deps[0], deps[1], kernel_size=3, stride=stride, padding=1)
        self.conv3 = builder.Conv2dBN(deps[1], deps[2], kernel_size=1,stride=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class ResNetBottleneckStage(nn.Module):

    #   stage_deps:     3n+1 (first is the projection),  n is the num of blocks

    def __init__(self, builder:ConvBuilder, in_planes, stage_deps, stride=1):
        super(ResNetBottleneckStage, self).__init__()
        print('building stage: in {}, deps {}'.format(in_planes, stage_deps))
        assert (len(stage_deps) - 1) % 3 == 0
        self.num_blocks = (len(stage_deps) - 1) // 3
        stage_out_channels = stage_deps[3]
        for i in range(2, self.num_blocks):
            assert stage_deps[3 * i] == stage_out_channels

        self.relu = builder.ReLU()

        self.projection = builder.Conv2dBN(in_channels=in_planes, out_channels=stage_deps[0], kernel_size=1, stride=1)
        self.align_opr = builder.ResNetAlignOpr(channels=stage_deps[0])
        
        for i in range(self.num_blocks):
            in_c = in_planes if i == 0 else stage_out_channels
            block_stride = stride if i == (self.num_blocks-1) else 1
            self.__setattr__('block{}'.format(i), BottleneckBranch(builder=builder,
                            in_channels=in_c, deps=stage_deps[1+i*3: 4+i*3], stride=block_stride))
            if i > 0:
                if block_stride > 1:
                    self.__setattr__('inner_sc{}'.format(i),builder.Maxpool2d(1,block_stride))
                else:
                    self.__setattr__('inner_sc{}'.format(i),self.align_opr)

    def forward(self, x):
        proj = self.align_opr(self.projection(x))
        out = proj + self.align_opr(self.block0(x))
        out = self.relu(out)
        for i in range(1, self.num_blocks):
            b = self.align_opr(self.__getattr__('block{}'.format(i))(out))
            out = self.__getattr__('inner_sc{}'.format(i))(out)
            out = self.relu(out)
            out = out + b
        return out


class SBottleneckResNet(nn.Module):
    def __init__(self, builder:ConvBuilder, num_blocks, num_classes=1000, deps=None):
        super(SBottleneckResNet, self).__init__()
        # self.mean_tensor = torch.from_numpy(np.array([0.485, 0.456, 0.406])).reshape(1, 3, 1, 1).cuda().type(torch.cuda.FloatTensor)
        # self.std_tensor = torch.from_numpy(np.array([0.229, 0.224, 0.225])).reshape(1, 3, 1, 1).cuda().type(torch.cuda.FloatTensor)

        # self.mean_tensor = torch.from_numpy(np.array([0.406, 0.456, 0.485])).reshape(1, 3, 1, 1).cuda().type(
        #     torch.cuda.FloatTensor)
        # self.std_tensor = torch.from_numpy(np.array([0.225, 0.224, 0.229])).reshape(1, 3, 1, 1).cuda().type(
        #     torch.cuda.FloatTensor)

        # self.mean_tensor = torch.from_numpy(np.array([0.5, 0.5, 0.5])).reshape(1, 3, 1, 1).cuda().type(
        #     torch.cuda.FloatTensor)
        # self.std_tensor = torch.from_numpy(np.array([0.5, 0.5, 0.5])).reshape(1, 3, 1, 1).cuda().type(
        #     torch.cuda.FloatTensor)

        if deps is None:
            if num_blocks == [3,4,6,3]:
                deps = RESNET50_ORIGIN_DEPS_FLATTENED
            elif num_blocks == [3,4,23,3]:
                deps = resnet_bottleneck_origin_deps_flattened(101)
            else:
                raise ValueError('???')
            
        self.deps = deps
        self.num_blocks = num_blocks
        self.conv1 = builder.Conv2dBNReLU(3, deps[0], kernel_size=7, stride=2, padding=3)
        self.maxpool = builder.Maxpool2d(kernel_size=3, stride=2, padding=1)
        #   every stage has  num_block * 3 + 1
        nls = [n*3+1 for n in num_blocks]    # num layers in each stage
        self.stage1 = ResNetBottleneckStage(builder=builder, in_planes=deps[0], stage_deps=deps[1: nls[0]+1])
        self.stage2 = ResNetBottleneckStage(builder=builder, in_planes=deps[nls[0]],
                                            stage_deps=deps[nls[0]+1: nls[0]+1+nls[1]], stride=2)
        self.stage3 = ResNetBottleneckStage(builder=builder, in_planes=deps[nls[0]+nls[1]],
                                            stage_deps=deps[nls[0]+nls[1]+1: nls[0]+1+nls[1]+nls[2]], stride=2)
        self.stage4 = ResNetBottleneckStage(builder=builder, in_planes=deps[nls[0] + nls[1] + nls[2]],
                                            stage_deps=deps[nls[0] + nls[1] + nls[2] + 1: nls[0] + 1 + nls[1] + nls[2] + nls[3]],
                                            stride=2)
        self.gap = builder.GAP(kernel_size=7)
        self.fc = builder.Linear(deps[-1], num_classes)
        self.num_classes = num_classes


    def forward(self, x):

        out = self.conv1(x)
        out = self.maxpool(out)
        # out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = self.fc(out)
        return out


def swresnet50v2(cfg, builder, num_classes = 1000,pretrained=None):
    return SBottleneckResNet(builder, [3,4,6,3], num_classes=num_classes, deps=cfg.deps)

def swresnet101v2(cfg, builder, num_classes = 1000,pretrained=None):
    return SBottleneckResNet(builder, [3,4,23,3], num_classes=num_classes, deps=cfg.deps)

def swresnet152v2(cfg, builder, num_classes = 1000,pretrained=None):
    return SBottleneckResNet(builder, [3,8,36,3], num_classes=num_classes, deps=cfg.deps)

def swresnet50v2_phi2(cfg, builder, num_classes = 1000,pretrained=None):
    deps = [53,212,53,53,212,53,53,212,424,106,106,424,106,106,424,106,106,424, 848,
                                    212, 212, 848,212,212,848,212, 212, 848, 212, 212, 848, 1692, 423, 423, 1692,423, 423, 1692]
    return SBottleneckResNet(builder, [2,3,4,2], num_classes=num_classes, deps=deps)

def swresnet50v2_phi3(cfg, builder, num_classes = 1000,pretrained=None):
    deps = [48,192,48,48,192,48,48,192,384,96,96,384,96,96,384, 768, 192, 192, 768,192,192,768,192, 192, 768, 1540, 385, 385, 1540,385, 385, 1540]
    return SBottleneckResNet(builder, [2,2,3,2], num_classes=num_classes, deps=deps)
