import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, Optional, List
import functools

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AttnBasicBlock(nn.Module):
    """
    结构：3×3 → 3×3
    通道数：保持 planes
    用于浅层网络（ResNet-18/34）


    """
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            base_width: int = 64,
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # x:[N,C,H,W]
        # 默认 stride 为1，通过卷积核在局部区域滑动，提取纹理、边缘、模式等局部特征；
        out = self.conv1(x)  # 卷积 out:[N,C,H,W]
        out = self.ln1(out)  # BN: [N,C,H,W]
        out = self.relu(out)  # [N,C,H,W]

        out = self.conv2(out)  # 卷积 out:[N,C,H,W]
        out = self.ln2(out)  # [N,C,H,W]

        if self.downsample is not None:
            identity = self.downsample(x)  # [N, planes, H/stride, W/stride] 步长不确定，为1时与原shape一致

        out += identity  # 残差相加: [N, planes, H/stride, W/stride]
        out = self.relu(out)  # ReLU: [N, planes, H/stride, W/stride]

        return out  # 输出: [N, planes, H/stride, W/stride]


class AttnBottleneck(nn.Module):
    """
    结构：1×1 → 3×3 → 1×1
    通道数：扩展为 planes * 4
    用于深层网络（ResNet-50/101/152）

    这种设计大幅度降低了参数量和计算量，同时保持了模型表达能力。
    适合构建 更深层的 Encoder，捕获更抽象、更高级的特征。
    """
    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            base_width: int = 64,
    ) -> None:
        super(AttnBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))

        self.conv1 = conv1x1(inplanes, width)
        self.ln1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=stride)
        self.ln2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.ln3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # [N, inplanes, H, W]    inplanes=C

        out = self.conv1(x)  # 1x1 卷积降维 → [N, width, H, W]
        out = self.ln1(out)  # BN → [N, width, H, W]
        out = self.relu(out)  # ReLU → [N, width, H, W]

        out = self.conv2(out)  # 3x3 卷积提取特征 → [N, width, H/stride, W/stride]
        out = self.ln2(out)  # BN → [N, width, H/stride, W/stride]
        out = self.relu(out)  # ReLU → [N, width, H/stride, W/stride]

        out = self.conv3(out)  # 1x1 卷积升维 → [N, planes*4, H/stride, W/stride]
        out = self.ln3(out)  # BN → [N, planes*4, H/stride, W/stride]

        if self.downsample is not None:
            identity = self.downsample(x)  # [N, planes*4, H/stride, W/stride]

        out += identity  # 残差连接 → [N, planes*4, H/stride, W/stride]
        out = self.relu(out)  # ReLU → [N, planes*4, H/stride, W/stride]

        return out  # 输出: [N, planes*4, H/stride, W/stride]


