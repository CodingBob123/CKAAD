# 多分支CA+特征拼接后ECA+SEA attention增强
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from typing import Type, Callable, Union, Optional, List
import functools
from model.SENetv2 import SEAttention
from model.Efficient_CA_complex import CoordAtt_ECA
from model.ECANet import ECAAttention
from model.SEAAttention import Sea_Attention
from model.deformConv import SAMDeformConv2d


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
        self.conv2 = conv3x3(width, width, stride=stride)
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


# =========================================================================
# 新增模块 1: ECASBF (ECA-based Selective Branch Fusion)
# 作用: 极度轻量化的多分支动态融合仲裁者
# =========================================================================
class ECASBF(nn.Module):
    def __init__(self, channels, num_branches=2, kernel_size=3):
        super(ECASBF, self).__init__()
        self.num_branches = num_branches
        
        # 1. 全局信息聚合 (GAP)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. ECA 交互 (利用1D卷积进行跨通道交互，无降维)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        
        # 3. 权重生成 (1x1 卷积生成多分支权重)
        self.weight_gen = nn.Conv1d(channels, channels * num_branches, kernel_size=1, groups=1, bias=True)
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, branches):
        # branches: List[Tensor] -> [B, C, H, W]
        batch_size, channels, height, width = branches[0].shape
        
        # Stack & Sum
        branch_stack = torch.stack(branches, dim=1) # [B, num_branches, C, H, W]
        U = torch.sum(branch_stack, dim=1)          # [B, C, H, W]
        
        # Global Descriptor & Interaction
        S = self.avg_pool(U).view(batch_size, 1, channels) # [B, 1, C]
        Z = self.eca_conv(S)                               # [B, 1, C]
        Z = Z.permute(0, 2, 1)                             # [B, C, 1]
        
        # Generate Weights
        weights = self.weight_gen(Z)                       # [B, num_branches*C, 1]
        weights = weights.view(batch_size, self.num_branches, channels) # [B, num_branches, C]
        
        # Softmax
        attention_vectors = F.softmax(weights, dim=1)      # [B, num_branches, C]
        
        # Fusion
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1) # [B, nb, C, 1, 1]
        V = torch.sum(branch_stack * attention_vectors, dim=1) # [B, C, H, W]
        
        return V


# =========================================================================
# 新增模块 2: DualPathBoundaryBlock (双路边界竞争模块)
# 作用: 同时运行"保守卷积"和"混合增强卷积"，利用 ECASBF 择优输出
# =========================================================================
class DualPathBoundaryBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, norm_layer, fusion_weight=0.5):
        super(DualPathBoundaryBlock, self).__init__()

        # 融合权重控制参数
        # fusion_weight=0.0: 完全使用保守路径 (适合zipper, tile等结构化类别)
        # fusion_weight=1.0: 完全使用激进路径 (适合复杂拓扑类别)
        # fusion_weight=0.5: 动态融合 (原始ECASBF行为)
        self.fusion_weight = fusion_weight

        # --- Path A: 保守型旧模块 (Standard 3x3) ---
        # 负责 Metal_nut, Tile 等简单纹理/刚性结构
        self.path_conservative = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)
        )

        # --- Path B: 激进型新模块 (Mixed Asymmetric) ---
        # 负责 Cable, Transistor 等复杂拓扑/线条结构
        # (完全复刻您原文件中 feature2 的逻辑)
        ch_per_branch = out_planes // 3
        # 确保总通道数对齐
        self.path_aggressive_branches = nn.ModuleDict({
            'horizontal': nn.Sequential(
                nn.Conv2d(in_planes, ch_per_branch, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
                nn.AvgPool2d((stride, 1), stride=(stride, 1)),
                norm_layer(ch_per_branch),
                nn.ReLU(inplace=True)
            ),
            'vertical': nn.Sequential(
                nn.Conv2d(in_planes, ch_per_branch, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
                nn.AvgPool2d((1, stride), stride=(1, stride)),
                norm_layer(ch_per_branch),
                nn.ReLU(inplace=True)
            ),
            'diagonal': nn.Sequential(
                nn.Conv2d(in_planes, ch_per_branch, 3, stride=stride, padding=1, bias=False),
                norm_layer(ch_per_branch),
                nn.ReLU(inplace=True)
            )
        })
        # 融合层: 将 H+V+D (Total Ch) -> out_planes
        total_ch = ch_per_branch * 3
        self.path_aggressive_fusion = nn.Sequential(
            nn.Conv2d(total_ch, out_planes, 1, bias=False),
            norm_layer(out_planes),
            nn.ReLU(inplace=True)
        )

        # --- Arbiter: 内部仲裁者 ---
        # 只有在fusion_weight=0.5时才使用动态仲裁
        if fusion_weight == 0.5:
            self.arbiter = ECASBF(channels=out_planes, num_branches=2, kernel_size=3)
        else:
            self.arbiter = None

    def forward(self, x):
        # 1. 计算 Path A (保守路径)
        feat_cons = self.path_conservative(x)

        # 2. 计算 Path B (激进路径 - 三个子分支拼接 + 融合)
        h = self.path_aggressive_branches['horizontal'](x)
        v = self.path_aggressive_branches['vertical'](x)
        d = self.path_aggressive_branches['diagonal'](x)
        feat_agg_raw = torch.cat([h, v, d], dim=1)
        feat_agg = self.path_aggressive_fusion(feat_agg_raw)

        # 3. 根据fusion_weight进行可控融合
        if self.fusion_weight == 0.0:
            # 完全使用保守路径 - 适合zipper, tile等结构化类别
            out = feat_cons
        elif self.fusion_weight == 1.0:
            # 完全使用激进路径 - 适合复杂拓扑类别
            out = feat_agg
        elif self.fusion_weight == 0.5:
            # 动态融合 (原始ECASBF行为)
            out = self.arbiter([feat_cons, feat_agg])
        else:
            # 线性插值融合 - 在0-1之间平滑过渡
            out = (1 - self.fusion_weight) * feat_cons + self.fusion_weight * feat_agg

        return out



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
            target_channels = input_channels[-1] * block.expansion
            current_channels = input_channel * block.expansion

            # 如果启用增强且是第二个分支（feature2），使用DualPathBoundaryBlock进行优化
            if enable_enhancement and i == 1:
                conv_layers.append(self._make_enhanced_conv_layer(current_channels, target_channels))
            else:
                # 其他分支使用原始的对齐方式
                conv_layers.append(self._make_conv_layer(block, current_channels, target_channels))
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # SEAttention 版本：添加SEAttention模块，通道数为最后一层的通道数乘以扩展系数 
        # self.se_attention = SEAttention(channel=input_channels[-1] * block.expansion, reduction=16)
        # 修改encode_layer1的输入通道数，现在直接使用SE模块的输出，不再拼接
        # self.encode_layer1 = self._make_layer(block, input_channels[-1] * block.expansion, input_channels[-1] * 2, layers, stride=2)
        
        # 为每个分支添加坐标注意力模块（在原始预训练特征上先进行CA增强）
        branch_channels = [c * block.expansion for c in input_channels]
        self.coord_atts = nn.ModuleList([
            CoordAtt_ECA(inp=ch, oup=ch) for ch in branch_channels
        ])
        
        # 拼接后的总通道数：分支数 × 对齐后的通道数（均为 input_channels[-1] * block.expansion）
        ca_aligned_channel = input_channels[-1] * block.expansion
        inplanes_after_concat = ca_aligned_channel * len(input_channels)
        
        # 初始化 ECAAttention：作用于拼接后的特征
        self.eca_attention = ECAAttention(kernel_size=3)

        # ========== 结构感知模块：SEAttention ==========
        # SEAFORMER: Squeeze-Enhanced Axial Transformer
        # 包含两个关键组件：
        # 1. Squeeze Axial Attention: 轴向注意力，建模行/列方向长程依赖
        # 2. Detail Enhancement Kernel: 细节增强核，提升局部边界和形状感知
        # 优势：
        # - 轴向建模：补足多尺度方向感知的全局性不足
        # - 细节增强：通过3x3卷积提升边界清晰度
        # - 与CoordAtt互补：CoordAtt是分支级行列感知，SEAttention是融合后全局行列建模
        # - 与ECA配合：ECA负责通道选择，SEAttention负责空间结构建模
        # 输入通道数：拼接后的总通道数 inplanes_after_concat
        # 例如：wide_resnet50_2时为 256*4*3 = 3072
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
        while out_planes != inplanes:
            layers.append(
                nn.Sequential(conv3x3(in_planes=inplanes, out_planes=inplanes * 2, stride=2),
                              norm_layer(inplanes * 2),
                              nn.ReLU(inplace=True))
            )
            inplanes = inplanes * 2

        return nn.Sequential(*layers)

    def _make_enhanced_conv_layer(self, inplanes: int, out_planes: int) -> nn.Sequential:
        """ 为第二个分支创建增强的对齐层，使用DualPathBoundaryBlock进行优化 """
        layers = []
        norm_layer = self._norm_layer
        current_planes = inplanes

        while out_planes != current_planes:
            # 计算stride：只有在需要改变空间尺寸时才使用stride=2
            stride = 2 if out_planes > current_planes else 1

            # 使用DualPathBoundaryBlock进行边界感知对齐
            layers.append(
                DualPathBoundaryBlock(
                    in_planes=current_planes,
                    out_planes=min(out_planes, current_planes * 2),
                    stride=stride,
                    norm_layer=norm_layer,
                    fusion_weight=0.5  # 使用动态融合
                )
            )
            current_planes = min(out_planes, current_planes * 2)

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
        # 1) 在原始预训练特征上先进行坐标注意力增强（不进行多尺度对齐）
        ca_features = [self.coord_atts[i](xi) for i, xi in enumerate(x)]
        
        # 2) 对每一个分支进行特征通道/尺度对齐到最后一个尺度
        features = [self.conv_layers[i](fi) for i, fi in enumerate(ca_features)]  # 每个 → [B, 256*exp, H3, W3]
        
        # SEAttention版本：使用SEAttention模块处理三个对齐后的特征（保留占位，暂不启用）
        # se_output = self.se_attention(features[0], features[1], features[2])  # → [B, 256*exp, H3, W3]
        
        # 3) 将对齐后的三个分支在通道维度上拼接
        fused = torch.cat(features, dim=1)
        
        # 3.5) 对拼接后的特征应用 ECA 通道注意力
        # ECA：通道维度特征选择，选择哪些通道重要
        fused = self.eca_attention(fused)

        # 3.6) 应用SEAttention增强结构感知能力
        # SEAttention包含：
        # - Squeeze Axial Attention: 建模行/列方向长程依赖
        # - Detail Enhancement Kernel: 通过局部卷积增强边界和细节
        # 优势：
        # - 补足轴向（行/列）方向的结构感知
        # - 提升局部边界和形状的清晰度
        # - 与CoordAtt形成分支级→全局级的互补
        # - 在特征压缩前进行结构增强，保留更多空间信息
        fused = self.sea_attention(fused)

        # 4) 送入后续编码层进行特征压缩和抽象
        output = self.encode_layer1(fused)  # → [B, 512*exp, H3/2, W3/2]

        return output.contiguous()


class Encoder(nn.Module):
    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256], attn_block_num=3, enable_enhancement=False) -> None:
        super(Encoder, self).__init__()
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
        return self.fusion_layer(x)


# ========== 测试函数 ==========
def test_enhanced_encoder():
    """
    测试增强编码器的功能 - 第二分支优化功能

    测试配置说明:
    - enable_enhancement=False: 使用原始的对齐方式
    - enable_enhancement=True: 对第二分支启用DualPathBoundaryBlock优化

    使用方法:
    1. 命令行参数: --enable_enhancement True
    2. 代码调用: Encoder(enable_enhancement=True)
    3. 脚本配置: 在mvtec.sh中为不同类别设置不同值
    """
    print("="*80)
    print("测试增强编码器 - 第二分支优化功能")
    print("="*80)

    # 测试配置
    test_configs = [
        ("基准线（无分支增强）", False),
        ("启用第二分支优化", True),
    ]

    # 简化的输入特征（模拟Wide ResNet-50-2的输出）
    x1 = torch.randn(2, 256, 64, 64)    # layer1: [B, 256, 64, 64]
    x2 = torch.randn(2, 512, 32, 32)    # layer2: [B, 512, 32, 32]
    x3 = torch.randn(2, 1024, 16, 16)   # layer3: [B, 1024, 16, 16]
    inputs = [x1, x2, x3]

    print("输入特征尺寸:")
    for i, feat in enumerate(inputs):
        print(f"  Feature{i+1}: {feat.shape}")

    results = []

    for config_name, enhancement_config in test_configs:
        print(f"\n🧪 测试配置: {config_name}")
        print(f"   第二分支增强: {enhancement_config}")

        # 创建编码器
        encoder = Encoder(
            backbone='wide_resnet50_2',
            attn_block_num=3,
            enable_enhancement=enhancement_config
        )

        # 前向传播
        with torch.no_grad():
            output = encoder(inputs)

        # 统计参数量
        total_params = sum(p.numel() for p in encoder.parameters())

        # 计算量统计 (使用thop库)
        try:
            from thop import profile
            macs, _ = profile(encoder, inputs=(inputs,), verbose=False)
            macs_g = macs / 1e9
        except ImportError:
            macs_g = 0.0

        print(f"   输出尺寸: {output.shape}")
        print(f"   参数量: {total_params/1e6:.2f}M, 计算量: {macs_g:.3f}G")

        results.append((config_name, enhancement_config, total_params, macs_g, output.shape))

    # 输出对比表格
    print(f"\n{'='*80}")
    print("实验结果对比表")
    print(f"{'='*80}")
    print("<25")
    print("-" * 80)

    for config_name, config, params, macs_g, out_shape in results:
        print("<25")

    print("\n✅ 增强编码器测试完成！")
    print("   实现了第二分支的优化功能")
    print("   支持灵活的增强配置")
    print("   保持了原始的注意力机制")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 运行测试
        test_enhanced_encoder()
    else:
        print("=" * 60)
        print("增强编码器 - 多分支CA+特征拼接后ECA+SEA attention增强")
        print("=" * 60)
        print("这是一个增强版本的编码器实现：")
        print("1. 支持第二分支的边界感知优化")
        print("2. 保持原始的多分支融合策略")
        print("3. 支持灵活的增强配置开关")
        print("\n使用方法：")
        print("  python model/encoder_mixed.py --test")
        print("\n或者在代码中调用：")
        print("  from model.encoder_mixed import Encoder")
        print("  encoder = Encoder(enable_enhancement=True)")