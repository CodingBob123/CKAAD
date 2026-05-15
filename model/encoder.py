import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, Optional, List
import functools
from model.Efficient_CA_complex import CoordAtt_ECA
from model.ECANet import ECAAttention
from model.SEAAttention import Sea_Attention


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
        out = self.ln1(out)  #  BN: [N,C,H,W]
        out = self.relu(out) #  [N,C,H,W]

        out = self.conv2(out)  # 卷积 out:[N,C,H,W]
        out = self.ln2(out)    # [N,C,H,W]

        if self.downsample is not None:
            identity = self.downsample(x)  # [N, planes, H/stride, W/stride] 步长不确定，为1时与原shape一致

        out += identity   # 残差相加: [N, planes, H/stride, W/stride]
        out = self.relu(out)   # ReLU: [N, planes, H/stride, W/stride]

        return out   # 输出: [N, planes, H/stride, W/stride]


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
        self.conv2 = conv3x3(width, width, stride)
        self.ln2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.ln3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x          # [N, inplanes, H, W]    inplanes=C

        out = self.conv1(x)   # 1x1 卷积降维 → [N, width, H, W]
        out = self.ln1(out)   # BN → [N, width, H, W]
        out = self.relu(out)  # ReLU → [N, width, H, W]

        out = self.conv2(out) # 3x3 卷积提取特征 → [N, width, H/stride, W/stride]
        out = self.ln2(out)   # BN → [N, width, H/stride, W/stride]
        out = self.relu(out)  # ReLU → [N, width, H/stride, W/stride]

        out = self.conv3(out) # 1x1 卷积升维 → [N, planes*4, H/stride, W/stride]
        out = self.ln3(out)   # BN → [N, planes*4, H/stride, W/stride]

        if self.downsample is not None:
            identity = self.downsample(x)   # [N, planes*4, H/stride, W/stride]

        out += identity       # 残差连接 → [N, planes*4, H/stride, W/stride]
        out = self.relu(out)  # ReLU → [N, planes*4, H/stride, W/stride]

        return out   # 输出: [N, planes*4, H/stride, W/stride]


class BoundaryPreservingBlock(nn.Module):
    """边界保持对齐模块 - 适用于Encoder第二分支
    通过水平(1x3) + 垂直(3x1) 卷积增强边界特征感知能力
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(BoundaryPreservingBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d

        self.block = nn.Sequential(
            # 3×3 下采样卷积
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            # 水平边界增强 (1×3)
            nn.Conv2d(out_channels // 2, out_channels // 2, (1, 3), padding=(0, 1), bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            # 垂直边界增强 (3×1)
            nn.Conv2d(out_channels // 2, out_channels // 2, (3, 1), padding=(1, 0), bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            # 1×1 输出投影
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FusionLayer(nn.Module):
    def __init__(self,
                 block: Type[Union[AttnBasicBlock, AttnBottleneck]],
                 layers: int,
                 input_channels: List[int] = [64, 128, 256],
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 width_per_group: int = 64,
                 enable_enhancement: bool = False,
                 ):
        super(FusionLayer, self).__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
            
        self._norm_layer = norm_layer
        self.dilation = 1
        self.base_width = width_per_group
        
        conv_layers = []   #  原文一开始使用的是三层预训练特征块，每一块转换通道用的卷积层列表（共三个列表）
        for i, input_channel in enumerate(input_channels):
            # 第二分支（index=1）启用边界保持对齐
            if enable_enhancement and i == 1:
                conv_layers.append(self._make_boundary_conv_layer(
                    input_channel * block.expansion,
                    input_channels[-1] * block.expansion
                ))
            else:
                conv_layers.append(self._make_conv_layer(block, input_channel * block.expansion, input_channels[-1]))
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # 为每个分支添加坐标注意力模块（在预训练特征对齐前先进行 CA 增强）
        branch_channels = [c * block.expansion for c in input_channels]
        self.coord_atts = nn.ModuleList([
            CoordAtt_ECA(inp=ch, oup=ch) for ch in branch_channels
        ])
        
        # 拼接后的总通道数：分支数 × 对齐后的通道数
        inplanes_after_concat = input_channels[-1] * block.expansion * len(input_channels)
        
        # ECA 通道注意力：作用于拼接后的融合特征，进行通道重标定
        self.eca_attention = ECAAttention(kernel_size=3)
        
        # SEA 结构感知注意力：增强融合特征的轴向长程依赖和局部细节
        self.sea_attention = Sea_Attention(dim=inplanes_after_concat, key_dim=64, num_heads=8, attn_ratio=2)
        
        self.encode_layer1 = self._make_layer(block, inplanes_after_concat, input_channels[-1] * 2, layers, stride=2)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_conv_layer(self, block, inplanes: int, out_planes: int) -> nn.Sequential:
        """ _make_conv_layer 会不断用 3×3 stride=2 的卷积调整，直到输入通道和目标通道对齐。"""
        layers = []
        norm_layer = self._norm_layer
        while out_planes * block.expansion != inplanes:
            layers.append(
                nn.Sequential(conv3x3(in_planes=inplanes, out_planes=inplanes * 2, stride=2),
                              norm_layer(inplanes * 2),
                              nn.ReLU(inplace=True))
            )
            inplanes = inplanes * 2
            
        return nn.Sequential(*layers)

    def _make_boundary_conv_layer(self, inplanes: int, out_planes: int) -> nn.Sequential:
        """使用 BoundaryPreservingBlock 构建第二分支的边界保持对齐序列"""
        layers = []
        norm_layer = self._norm_layer
        while out_planes != inplanes:
            layers.append(
                BoundaryPreservingBlock(inplanes, inplanes * 2, stride=2, norm_layer=norm_layer)
            )
            inplanes = inplanes * 2
        return nn.Sequential(*layers)

    def _make_layer(self, block: Type[Union[AttnBasicBlock, AttnBottleneck]], inplanes: int, planes: int, blocks: int,
                    stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer, base_width=self.base_width))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer, base_width=self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        输入是 [x1, x2, x3]，
        x1: [B, 64*exp, H1, W1]
        x2: [B,128*exp, H2, W2]
        x3: [B,256*exp, H3, W3]
        （exp = 1 for BasicBlock, 4 for Bottleneck）

        """
        # 1) 在原始预训练特征上先进行坐标注意力增强（对齐前增强）
        ca_features = [self.coord_atts[i](xi) for i, xi in enumerate(x)]

        # 2) 对每一个分支进行特征通道/尺度对齐到最后一个尺度
        features = [self.conv_layers[i](fi) for i, fi in enumerate(ca_features)]  # → 每个 [B, 256*exp, H3, W3]

        # 3) 将对齐后的三个分支在通道维度上拼接
        fused = torch.cat(features, dim=1)   # → [B, 3*256*exp, H3, W3]

        # 4) 应用 SEA 结构感知注意力（轴向长程依赖 + 局部细节增强）
        fused = self.sea_attention(fused)

        # 5) 对拼接后的融合特征应用 ECA 通道注意力（通道重标定）
        fused = self.eca_attention(fused)

        # 6) 送入后续编码层进行特征压缩和抽象
        output = self.encode_layer1(fused)   # → [B, 512*exp, H3/2, W3/2]

        return output.contiguous()


class Encoder(nn.Module):
    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256],
                 attn_block_num=3, enable_enhancement=False, disable_same=False) -> None:
        super(Encoder, self).__init__()
        self.disable_same = disable_same
        self.expansion = 4
        if backbone == 'resnet18':
            self.fusion_layer = FusionLayer(AttnBasicBlock, 2, input_channels, enable_enhancement=enable_enhancement)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.fusion_layer = FusionLayer(AttnBasicBlock, attn_block_num, input_channels, enable_enhancement=enable_enhancement)
            self.expansion = 1
            
        elif backbone == 'resnet50':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, enable_enhancement=enable_enhancement)
        elif backbone == 'resnet101':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, enable_enhancement=enable_enhancement)
        elif backbone == 'resnet152':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, enable_enhancement=enable_enhancement)
        elif backbone == 'wide_resnet50_2':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2, enable_enhancement=enable_enhancement)
        elif backbone == 'wide_resnet101_2':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2, enable_enhancement=enable_enhancement)
            
    def forward(self, x):
        if self.disable_same:
            return x[-1]
        return self.fusion_layer(x)
