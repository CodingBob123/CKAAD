from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, List, Optional
import functools
from model.DySample import DySample

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class DeBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        base_width: int = 64
    ) -> None:
        super(DeBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        if base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if stride == 2:
            # self.conv1 = deconv2x2(inplanes, planes, stride)  # 原转置卷积
            self.conv1 = DySample(inplanes, scale=2)  # DySample动态上采样
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # [N, inplanes, H,   W]

        out = self.conv1(x)  # DySample/deconv2x2, s=2 → [N, planes, 2H, 2W]    conv3x3, stride=1 → [N, planes, H, W] 下面保持不变
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out  #                      [N, planes,   2H,  2W]  /  [N, planes,   H,   W]


class DeBottleneck(nn.Module):
    """
    DeBottleneck = “瓶颈式残差上采样块”：
    1×1（降维到 width）→ 3×3 或 2×2 deconv（保持/放大空间）→ 1×1（升维到 planes*4）+ 残差相加；

    与论文思路完全一致：用残差转置卷积逐级上采样，重建多尺度特征。

    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        base_width: int = 64
    ) -> None:
        super(DeBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            # self.conv2 = deconv2x2(width, width, stride)  # 原转置卷积
            self.conv2 = DySample(width, scale=2)  # DySample动态上采样
        else:
            self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # 输入张量：x 的形状为 [N, inplanes, H, W]
        # 其中 N 是批次大小，inplanes 是输入通道数，H 和 W 是空间维度
        
        identity = x        # 保存原始输入作为残差连接
                      # 形状：[N, inplanes, H, W]
        
        # 第一个卷积层：1x1 卷积，用于降维
        out = self.conv1(x) # 1x1卷积：inplanes → width
                       # 形状：[N, width, H, W]
        
        # 第一个批归一化和激活
        out = self.bn1(out) # 批归一化，形状不变
                       # 形状：[N, width, H, W]
        out = self.relu(out) # ReLU激活，形状不变
                        # 形状：[N, width, H, W]

        # 第二个卷积层：根据stride决定是3x3卷积还是DySample上采样
        out = self.conv2(out)  # 如果stride=1：3x3卷积，空间不变
                          # 如果stride=2：DySample上采样2倍
                          # 形状：stride=1时 [N, width, H, W]
                          #       stride=2时 [N, width, 2H, 2W]
        
        # 第二个批归一化和激活
        out = self.bn2(out)    # 批归一化，形状不变
                          # 形状：stride=1时 [N, width, H, W]
                          #       stride=2时 [N, width, 2H, 2W]
        out = self.relu(out)   # ReLU激活，形状不变
                          # 形状：stride=1时 [N, width, H, W]
                          #       stride=2时 [N, width, 2H, 2W]

        # 第三个卷积层：1x1卷积，用于升维
        out = self.conv3(out)  # 1x1卷积：width → planes*4
                          # 形状：stride=1时 [N, planes*4, H, W]
                          #       stride=2时 [N, planes*4, 2H, 2W]
        
        out = self.bn3(out)    # 批归一化，形状不变
                          # 形状：stride=1时 [N, planes*4, H, W]
                          #       stride=2时 [N, planes*4, 2H, 2W]

        # 残差连接：如果需要上采样，则对identity进行相应的上采样
        if self.upsample is not None:
            identity = self.upsample(x)   # 对原始输入进行上采样
                                    # 形状：stride=1时 [N, planes*4, H, W]
                                    #       stride=2时 [N, planes*4, 2H, 2W]

        # 残差相加
        out += identity   # 元素级相加，形状必须相同
                    # 形状：stride=1时 [N, planes*4, H, W]
                    #       stride=2时 [N, planes*4, 2H, 2W]
        
        # 最终激活
        out = self.relu(out)   # ReLU激活，形状不变
                          # 形状：stride=1时 [N, planes*4, H, W]
                          #       stride=2时 [N, planes*4, 2H, 2W]

        return out   # 返回最终结果
                # 形状：stride=1时 [N, planes*4, H, W]
                #       stride=2时 [N, planes*4, 2H, 2W]


class DeResNet(nn.Module):
    """
    输入张量: [N, 2048, H, W]
        │
        ▼
    ┌───────────────────────┐
    │ 第0层 (3个DeBottleneck块) │
    │ inplanes=2048, planes=256 │
    └───────────────────────┘
        │
        ▼
    [N, 1024, 2H, 2W]
        │
        ▼
    ┌───────────────────────┐
    │ 第1层 (4个DeBottleneck块) │
    │ inplanes=1024, planes=128 │
    └───────────────────────┘
        │
        ▼
    [N, 512, 4H, 4W]
        │
        ▼
    ┌───────────────────────┐
    │ 第2层 (6个DeBottleneck块) │
    │ inplanes=512, planes=64  │
    └───────────────────────┘
        │
        ▼
    [N, 256, 8H, 8W]
    """
    """
    参数对应关系（通用形式）
    层索引	output_channels[::-1]	layers (使用前3个)	实际输入通道 (inplanes)	实际输出通道 (planes×expansion)	块数量
    0	C₃	L₁	C₃×2×E	C₃×E	L₁
    1	C₂	L₂	C₂×2×E	C₂×E	L₂
    2	C₁	L₃	C₁×2×E	C₁×E	L₃
    张量形状变化（通用形式）
    层索引	输入形状	第一个块输出形状	最终输出形状	上采样倍率
    0	[N, C_in, H, W]	[N, C₃×E, 2H, 2W]	[N, C₃×E, 2H, 2W]	2倍
    1	[N, C₃×E, 2H, 2W]	[N, C₂×E, 4H, 4W]	[N, C₂×E, 4H, 4W]	2倍
    2	[N, C₂×E, 4H, 4W]	[N, C₁×E, 8H, 8W]	[N, C₁×E, 8H, 8W]	2倍
    输入张量: [N, C_in, H, W]
        │
        ▼
    ┌───────────────────────────┐
    │ 第0层 (L₁个DeBottleneck块)  │
    │ inplanes=C₃×2×E, planes=C₃ │
    └───────────────────────────┘
        │
        ▼
    [N, C₃×E, 2H, 2W]
        │
        ▼
    ┌───────────────────────────┐
    │ 第1层 (L₂个DeBottleneck块)  │
    │ inplanes=C₂×2×E, planes=C₂ │
    └───────────────────────────┘
        │
        ▼
    [N, C₂×E, 4H, 4W]
        │
        ▼
    ┌───────────────────────────┐
    │ 第2层 (L₃个DeBottleneck块)  │
    │ inplanes=C₁×2×E, planes=C₁ │
    └───────────────────────────┘
        │
        ▼
    [N, C₁×E, 8H, 8W]
    """
    def __init__(
        self,
        block: Type[Union[DeBasicBlock, DeBottleneck]],
        layers: List[int],
        output_channels: List[int],
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        width_per_group: int = 64
    ) -> None:
        super(DeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        self._norm_layer = norm_layer

        self.dilation = 1
        self.base_width = width_per_group
        deconv_layers = []
        
        for o_channel, layer in zip(output_channels[::-1], layers):
            deconv_layers.append(self._make_layer(block, o_channel * 2 * block.expansion, o_channel, layer, 2))
        self.layers = nn.ModuleList(deconv_layers)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block: Type[Union[DeBasicBlock, DeBottleneck]], inplanes: int, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # 原转置卷积上采样 (通道数会改变)
            # upsample = nn.Sequential(
            #     deconv2x2(inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )

            # DySample上采样 + 1x1卷积调整通道数
            # 注意：这里scale=stride是为了与转置卷积行为一致
            upsample = nn.Sequential(
                DySample(inplanes, scale=stride),
                conv1x1(inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, upsample, norm_layer, self.base_width))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            outputs.append(x)
        return outputs[::-1]
        """
        [
            [N, C₁×E, 8H, 8W],   # 最低层特征（分辨率最高）
            [N, C₂×E, 4H, 4W],   # 中间层特征
            [N, C₃×E, 2H, 2W]    # 最高层特征（分辨率最低）
        ]
        """


class Decoder(nn.Module):
    def __init__(self, backbone='wide_resnet50_2', output_channels=[64, 128, 256]):
        super(Decoder, self).__init__()
        if backbone == 'resnet18':
            self.backbone = DeResNet(DeBasicBlock, [2, 2, 2, 2], output_channels)
        elif backbone == 'resnet34':
            self.backbone = DeResNet(DeBasicBlock, [3, 4, 6, 3], output_channels)
        elif backbone == 'resnet50':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 6, 3], output_channels)
        elif backbone == 'resnet101':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 23, 3], output_channels)
        elif backbone == 'resnet152':
            self.backbone = DeResNet(DeBottleneck, [3, 8, 36, 3], output_channels)
        elif backbone == 'wide_resnet50_2':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 6, 3], output_channels, width_per_group=64 * 2)
        elif backbone == 'wide_resnet101_2':
            self.backbone = DeResNet(DeBottleneck, [3, 4, 23, 3], output_channels, width_per_group=64 * 2)
            
    def forward(self, x):
        return self.backbone(x)
    