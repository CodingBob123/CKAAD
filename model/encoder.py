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

# 这个类是一个残差快(Residual Block)，它是深度残差网络(ResNet)的基本构建的单元。
class AttnBasicBlock(nn.Module):
    expansion: int = 1  # 扩展因子，BasicBlock 的扩展因子为1

    def __init__(
        self,
        inplanes: int,        # 输入通道数
        planes: int,          # 中间层通道数
        stride: int = 1,      # 步长
        downsample: Optional[nn.Module] = None,  # 下采样层，用于调整残差连接的尺寸
        norm_layer: Optional[Callable[..., nn.Module]] = None,  # 归一化层类型
        base_width: int = 64, # 基础宽度
    ) -> None:
        super(AttnBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 默认使用 BatchNorm2d
        if base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        # 第一个卷积层：3x3 卷积
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = norm_layer(planes)  # 归一化层
        self.relu = nn.ReLU(inplace=True)  # 激活函数
        
        # 第二个卷积层：3x3 卷积
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = norm_layer(planes)  # 归一化层
        
        self.downsample = downsample  # 下采样层
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 1. 保存输入x，作为残差分支

        # 主分支：两个卷积层
        out = self.conv1(x)      # 第一个卷积
        out = self.ln1(out)      # 第一个归一化
        out = self.relu(out)     # 第一个激活

        out = self.conv2(out)    # 第二个卷积
        out = self.ln2(out)      # 第二个归一化

        if self.downsample is not None:
            identity = self.downsample(x)  # 2. 如果需要，调整identity的shape

        # 这个类已经通过下面这个语句实现了残差连接(skip-connection)的功能：
        out += identity  # 3. 残差连接：主分支输出与identity相加
        out = self.relu(out)     # 最终激活

        return out


class AttnBottleneck(nn.Module):
    """
    Bottleneck 残差块，用于更深的网络
    结构：1x1 conv -> 3x3 conv -> 1x1 conv
    扩展因子为4，意味着输出通道数是输入通道数的4倍
    """
    expansion: int = 4  # 扩展因子，Bottleneck 的扩展因子为4

    def __init__(
        self,
        inplanes: int,        # 输入通道数
        planes: int,          # 中间层通道数
        stride: int = 1,      # 步长
        downsample: Optional[nn.Module] = None,  # 下采样层
        norm_layer: Optional[Callable[..., nn.Module]] = None,  # 归一化层类型
        base_width: int = 64, # 基础宽度
    ) -> None:
        super(AttnBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))  # 计算实际宽度
        
        # 第一个卷积层：1x1 卷积，用于降维
        self.conv1 = conv1x1(inplanes, width)
        self.ln1 = norm_layer(width)
        
        # 第二个卷积层：3x3 卷积，主要特征提取
        self.conv2 = conv3x3(width, width, stride)
        self.ln2 = norm_layer(width)
        
        # 第三个卷积层：1x1 卷积，用于升维
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.ln3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x  # 保存输入作为残差连接

        # 主分支：三个卷积层
        out = self.conv1(x)      # 1x1 卷积降维
        out = self.ln1(out)      # 归一化
        out = self.relu(out)     # 激活

        out = self.conv2(out)    # 3x3 卷积特征提取
        out = self.ln2(out)      # 归一化
        out = self.relu(out)     # 激活

        out = self.conv3(out)    # 1x1 卷积升维
        out = self.ln3(out)      # 归一化

        if self.downsample is not None:
            identity = self.downsample(x)  # 调整残差连接的尺寸

        out += identity  # 残差连接
        out = self.relu(out)     # 最终激活

        return out


class FusionLayer(nn.Module):
    """
    特征融合层：将多层特征下采样到相同尺寸，然后拼接并通过 ResNet Block 处理
    这是 Encoder 的核心组件
    """
    def __init__(self,
                 block: Type[Union[AttnBasicBlock, AttnBottleneck]],  # 使用的残差块类型
                 layers: int,                                           # 残差块层数
                 input_channels: List[int] = [64, 128, 256],          # 输入各层的通道数
                 norm_layer: Optional[Callable[..., nn.Module]] = None, # 归一化层类型
                 width_per_group: int = 64,                            # 每组宽度
                 ):
        super(FusionLayer, self).__init__()
        if norm_layer is None:
            # 使用 InstanceNorm2d 作为归一化层，并启用仿射变换
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
            
        self._norm_layer = norm_layer
        self.dilation = 1
        self.base_width = width_per_group
        
        # 为每层特征创建下采样卷积层
        conv_layers = []
        for input_channel in input_channels:
            # 为每个输入通道创建下采样层，将特征下采样到最小尺寸
            conv_layers.append(self._make_conv_layer(block, input_channel * block.expansion, input_channels[-1]))
        self.conv_layers = nn.ModuleList(conv_layers)  # 将列表转换为 ModuleList
        
        # 创建编码层：将拼接后的特征通过 ResNet Block 处理
        # 输入通道数：最后一层通道数 * 扩展因子 * 层数
        # 输出通道数：最后一层通道数 * 2
        self.encode_layer1 = self._make_layer(block, 
                                            input_channels[-1] * block.expansion * len(input_channels), 
                                            input_channels[-1] * 2, 
                                            layers, 
                                            stride=2)
        
        # 初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_conv_layer(self, block, inplanes: int, out_planes: int) -> nn.Sequential:
        """
        创建下采样卷积层，将特征下采样到目标尺寸
        Args:
            block: 残差块类型，用于获取扩展因子
            inplanes: 输入通道数
            out_planes: 目标通道数
        Returns:
            Sequential: 下采样层序列
        """
        layers = []
        norm_layer = self._norm_layer
        
        # 通过循环添加卷积层，直到达到目标通道数
        while out_planes * block.expansion != inplanes:
            layers.append(
                nn.Sequential(
                    conv3x3(in_planes=inplanes, out_planes=inplanes * 2, stride=2),  # 3x3 卷积下采样
                    norm_layer(inplanes * 2),  # 归一化
                    nn.ReLU(inplace=True)      # 激活函数
                )
            )
            inplanes = inplanes * 2  # 通道数翻倍
            
        return nn.Sequential(*layers)  # 将层列表转换为 Sequential

    def _make_layer(self, block: Type[Union[AttnBasicBlock, AttnBottleneck]], 
                    inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        创建 ResNet 层，包含多个残差块
        Args:
            block: 残差块类型
            inplanes: 输入通道数
            planes: 中间层通道数
            blocks: 残差块数量
            stride: 步长
        Returns:
            Sequential: ResNet 层
        """
        norm_layer = self._norm_layer
        downsample = None
        
        # 如果需要调整残差连接的尺寸，创建下采样层
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),  # 1x1 卷积调整尺寸
                norm_layer(planes * block.expansion),                  # 归一化
            )

        layers = []
        # 添加第一个残差块（可能需要下采样）
        layers.append(block(inplanes, planes, stride, downsample, norm_layer, base_width=self.base_width))
        inplanes = planes * block.expansion  # 更新输入通道数
        
        # 添加剩余的残差块
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer, base_width=self.base_width))

        return nn.Sequential(*layers)  # 将层列表转换为 Sequential

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播过程
        Args:
            x: 输入特征列表，每个元素对应一层特征
        Returns:
            Tensor: 编码后的特征（潜在变量 Z）
        """
        # 1. 对每层特征进行下采样处理
        feature = [self.conv_layers[i](xi) for i, xi in enumerate(x)]
        
        # 2. 在通道维度上拼接所有特征
        feature = torch.cat(feature, dim=1)
        
        # 3. 通过 ResNet Block 处理，形成潜在变量 Z
        output = self.encode_layer1(feature)

        return output.contiguous()  # 确保内存连续


class Encoder(nn.Module):
    """
    编码器：将多层特征融合并编码为潜在变量 Z
    这是整个编码过程的主控制器
    """
    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256], attn_block_num=3) -> None:
        super(Encoder, self).__init__()
        self.expansion = 4  # 默认扩展因子
        
        # 根据不同的 backbone 创建不同的 FusionLayer
        if backbone == 'resnet18':
            self.fusion_layer = FusionLayer(AttnBasicBlock, 2, input_channels)
            self.expansion = 1  # ResNet18 使用 BasicBlock，扩展因子为1
        elif backbone == 'resnet34':
            self.fusion_layer = FusionLayer(AttnBasicBlock, attn_block_num, input_channels)
            self.expansion = 1
            
        elif backbone == 'resnet50':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels)
        elif backbone == 'resnet101':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels)
        elif backbone == 'resnet152':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels)
        elif backbone == 'wide_resnet50_2':
            # Wide ResNet 使用更大的基础宽度
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2)
        elif backbone == 'wide_resnet101_2':
            self.fusion_layer = FusionLayer(AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2)
            
    def forward(self, x):
        """
        编码器前向传播
        Args:
            x: 输入特征列表
        Returns:
            Tensor: 潜在变量 Z
        """
        return self.fusion_layer(x)  # 直接返回 FusionLayer 的输出
