import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Callable, Union, Optional, List
import functools

# 导入必要的组件 (根据您原本的目录结构)
# from model.SENetv2 import SEAttention
# 为了代码独立运行，这里保留 SEAttention 的占位引用或假设已导入
from model.SENetv2 import SEAttention


# ==========================================
# 基础卷积组件 (保持不变)
# ==========================================
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ==========================================
# 基础ResNet模块 (AttnBasicBlock, AttnBottleneck)
# (代码逻辑保持不变，此处省略具体实现以节省篇幅)
# ==========================================
class AttnBasicBlock(nn.Module):
    # ... (保持原代码内容不变) ...
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, base_width: int = 64) -> None:
        super(AttnBasicBlock, self).__init__()
        # ... (标准ResNet BasicBlock实现) ...
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class AttnBottleneck(nn.Module):
    # ... (保持原代码内容不变) ...
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None, base_width: int = 64) -> None:
        super(AttnBottleneck, self).__init__()
        # ... (标准ResNet Bottleneck实现) ...
        if norm_layer is None: norm_layer = nn.BatchNorm2d
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
        identity = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.ln3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ==========================================
# 新增组件：混合边界块 (MixedBoundaryBlock)
# 用于 Feature 2 的优化
# ==========================================
class MixedBoundaryBlock(nn.Module):
    """
    混合边界感知模块 (Hybrid Boundary Block)
    结合了：
    1. 标准 3x3 卷积：捕获圆形、斜向、不规则边界 (解决 Screw, Nut, Pill 掉点)
    2. 非对称卷积 (1x3 + 3x1)：强化水平/垂直纹理 (保持 Grid, Carpet 优势)
    """

    def __init__(self, in_channels, out_channels, stride, norm_layer):
        super().__init__()
        mid_channels = out_channels // 2

        # 1. 降维与下采样
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 分支A: 标准 3x3 (全向感知)
        self.branch_standard = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 3. 分支B: 非对称卷积 (极值纹理感知)
        self.branch_asym = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, (1, 3), padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, (3, 1), padding=(1, 0), bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 4. 融合输出
        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_in(x)
        # 特征叠加：同时保留全向轮廓和锐利边缘
        feat_std = self.branch_standard(x)
        feat_asym = self.branch_asym(x)
        return self.conv_out(feat_std + feat_asym)


# ==========================================
# 新增组件：带门控的残差混合块 (GatedResidualFusion)
# ==========================================
class GatedResidualFusion(nn.Module):
    """
    带门控的残差混合块
    公式: Output = F_raw + alpha * Sigmoid(Gate(F_raw)) * F_enhanced
    特点:
    1. alpha初始化为0: 保证训练初期完全回退到ResNet基准，防止性能崩塌。
    2. Gate门控: 空间注意力机制，自动判断哪些区域需要"修补"，哪些区域保持原样。
    """

    def __init__(self, channels):
        super().__init__()

        # 门控网络: 生成 [0, 1] 的空间权重图
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 可学习的全局缩放因子，初始化为 0
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, f_raw, f_enhanced):
        """
        f_raw: 原始特征 (经过Shortcut投影后)
        f_enhanced: 对齐/增强后的特征
        """
        # 计算门控权重
        gate = self.gate_net(f_raw)

        # 残差混合
        return f_raw + self.alpha * (gate * f_enhanced)


# ==========================================
# 核心组件：静态对齐块 (StaticAlignmentBlock) - 优化后
# ==========================================
class StaticAlignmentBlock(nn.Module):
    """
    改进后的静态对齐块
    包含:
    1. 针对性的特征提取 (优化后的 Feature 1 & 2)
    2. Gated Residual Fusion 融合机制
    """

    def __init__(self, in_channels, out_channels, stride=2, branch_type='feature1', norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.branch_type = branch_type

        # 1. Shortcut路径: 确保原始特征(f_raw)与输出维度一致，用于残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels)
            )

        # 2. 构建特征增强路径 (f_enhanced)
        if branch_type == 'feature1':
            # Feature1: 纹理感知 (优化: 移除dilation)
            self.alignment = self._build_texture_aware_alignment(in_channels, out_channels, stride, norm_layer)
        elif branch_type == 'feature2':
            # Feature2: 边界保持 (优化: 使用MixedBoundaryBlock)
            self.alignment = self._build_boundary_preserving_alignment(in_channels, out_channels, stride, norm_layer)
        elif branch_type == 'feature3':
            # Feature3: 全局信息 (保持不变)
            self.alignment = self._build_global_preserving_alignment(in_channels, out_channels, stride, norm_layer)
        else:
            self.alignment = self._build_default_alignment(in_channels, out_channels, stride, norm_layer)

        # 3. 门控残差融合模块
        self.fusion = GatedResidualFusion(out_channels)

    def _build_texture_aware_alignment(self, in_channels, out_channels, stride, norm_layer):
        """
        [优化] Feature1: 纹理感知
        改动: 移除空洞卷积(dilation=2 -> dilation=1)，防止网格效应
        作用: 更好保留 Pill, Capsule 等物体的微小划痕/斑点细节
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            # 改为标准卷积，专注于局部细节
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, stride=1, padding=1, dilation=1, bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def _build_boundary_preserving_alignment(self, in_channels, out_channels, stride, norm_layer):
        """
        [优化] Feature2: 边界感知
        改动: 使用 MixedBoundaryBlock (混合卷积)
        作用:
          - 3x3分支: 修复 Screw, Nut 的圆形边界感知
          - 非对称分支: 保持 Carpet, Grid, Wood 的纹理感知
        """
        return MixedBoundaryBlock(in_channels, out_channels, stride, norm_layer)

    def _build_global_preserving_alignment(self, in_channels, out_channels, stride, norm_layer):
        """Feature3: 全局信息保持 (保持原样，这对所有类别都有效)"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels // 2, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 2, 1, bias=False),
            norm_layer(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def _build_default_alignment(self, in_channels, out_channels, stride, norm_layer):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. 获取对齐到目标维度的原始特征 (Identity/Shortcut)
        f_raw = self.shortcut(x)

        # 2. 获取增强特征 (Aligned/Enhanced)
        f_enhanced = self.alignment(x)

        # 3. 执行带门控的残差融合
        return self.fusion(f_raw, f_enhanced)


class StaticAlignmentLayer(nn.Module):
    """静态对齐层 - 逻辑保持不变，只是调用的 Block 变强了"""

    def __init__(self, inplanes, target_planes, branch_type, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.branch_type = branch_type
        layers = []
        current_planes = inplanes
        while current_planes < target_planes:
            layers.append(
                StaticAlignmentBlock(
                    in_channels=current_planes,
                    out_channels=current_planes * 2,
                    stride=2,
                    branch_type=branch_type,
                    norm_layer=norm_layer
                )
            )
            current_planes *= 2
        self.alignment_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.alignment_layers(x)


class StaticEnhancedFusionLayer(nn.Module):
    """
    静态增强融合层
    实现逻辑:
    1. 三分支并行提取与增强 (使用 StaticAlignmentLayer)
    2. SE-Net 融合
    """

    def __init__(self,
                 block: Type[Union[AttnBasicBlock, AttnBottleneck]],
                 layers: int,
                 input_channels: List[int] = [64, 128, 256],
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 width_per_group: int = 64,
                 enable_branch_enhancement: Union[bool, List[bool]] = False):
        super(StaticEnhancedFusionLayer, self).__init__()

        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)

        self._norm_layer = norm_layer
        self.dilation = 1
        self.base_width = width_per_group

        # 处理增强开关逻辑 (保持不变)
        if isinstance(enable_branch_enhancement, bool):
            self.enable_branch_enhancement = [enable_branch_enhancement] * len(input_channels)
        elif isinstance(enable_branch_enhancement, list):
            # ... (边界检查逻辑省略，保持原样) ...
            if len(enable_branch_enhancement) > len(input_channels):
                self.enable_branch_enhancement = enable_branch_enhancement[:len(input_channels)]
            elif len(enable_branch_enhancement) < len(input_channels):
                padding = len(input_channels) - len(enable_branch_enhancement)
                self.enable_branch_enhancement = enable_branch_enhancement + [False] * padding
            else:
                self.enable_branch_enhancement = enable_branch_enhancement
        else:
            raise ValueError("enable_branch_enhancement type error")

        # ========== 静态对齐层构建 ==========
        alignment_layers = []
        for i, input_channel in enumerate(input_channels):
            target_planes = input_channels[-1] * block.expansion
            current_planes = input_channel * block.expansion

            if self.enable_branch_enhancement[i]:
                # 启用增强：使用改进后的 StaticAlignmentLayer (内含 GatedResidualFusion)
                branch_type = ['feature1', 'feature2', 'feature3'][i]
                alignment_layers.append(
                    StaticAlignmentLayer(
                        inplanes=current_planes,
                        target_planes=target_planes,
                        branch_type=branch_type,
                        norm_layer=norm_layer
                    )
                )
            else:
                # 不启用增强：使用默认的简单卷积对齐 (原始CKAAD策略)
                alignment_layers.append(
                    self._make_default_conv_layer(block, current_planes, target_planes)
                )
        self.alignment_layers = nn.ModuleList(alignment_layers)

        # SENet融合机制
        self.senet = SEAttention(channel=input_channels[-1] * block.expansion, reduction=16)

        # 后续编码层
        senet_fused_channel = input_channels[-1] * block.expansion
        self.encode_layer1 = self._make_layer(
            block, senet_fused_channel, input_channels[-1] * 2, layers, stride=2
        )

        # 权重初始化 (保持不变)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None: nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None: nn.init.constant_(m.bias, 0)

    def _make_default_conv_layer(self, block, inplanes: int, out_planes: int) -> nn.Sequential:
        """默认卷积对齐层 - 原始CKAAD的方法 (保持不变)"""
        layers = []
        norm_layer = self._norm_layer
        while out_planes != inplanes:
            layers.append(
                nn.Sequential(
                    conv3x3(in_planes=inplanes, out_planes=inplanes * 2, stride=2),
                    norm_layer(inplanes * 2),
                    nn.ReLU(inplace=True)
                )
            )
            inplanes = inplanes * 2
        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1) -> nn.Sequential:
        """构建编码层 (保持不变)"""
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

    def forward(self, x: List[Tensor]) -> Tensor:
        # 1) 对齐 (这里会调用带残差和门控的Block)
        aligned_features = [self.alignment_layers[i](fi) for i, fi in enumerate(x)]

        # 2) SENet 融合
        fused = self.senet(aligned_features[0], aligned_features[1], aligned_features[2])

        # 3) 编码
        output = self.encode_layer1(fused)

        return output.contiguous()


class Encoder(nn.Module):
    """
    Encoder类 - 接口保持与之前一致
    """

    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256], attn_block_num=3,
                 enable_enhancement=[False, False, False]):
        super(Encoder, self).__init__()

        self.expansion = 4
        # 根据骨干网络配置 (保持不变，省略部分重复代码)
        if backbone == 'wide_resnet50_2':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2,
                enable_branch_enhancement=enable_enhancement
            )
        elif backbone == 'resnet18':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBasicBlock, 2, input_channels, enable_branch_enhancement=enable_enhancement)
            self.expansion = 1
        # ... 其他 backbone 配置保持不变 ...
        else:
            # 默认处理 (简化)
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels,
                enable_branch_enhancement=enable_enhancement
            )

    def forward(self, x):
        return self.fusion_layer(x)


if __name__ == "__main__":
    # 简单的测试代码
    print("初始化改进版Encoder (含 MixedBoundaryBlock 和 GatedResidualFusion)...")
    try:
        inputs = [
            torch.randn(2, 256, 64, 64),
            torch.randn(2, 512, 32, 32),
            torch.randn(2, 1024, 16, 16)
        ]
        # 启用全部分支增强
        encoder = Encoder(backbone='wide_resnet50_2', enable_enhancement=[True, True, True])
        output = encoder(inputs)
        print(f"前向传播成功，输出尺寸: {output.shape}")
    except Exception as e:
        print(f"出错: {e}")