# 静态设置 + 二分支双通道优化 + ECASBF优化 + 原始CKAAD策略版本
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Callable, Union, Optional, List
import functools

# 导入必要的组件
from model.Efficient_CA_complex import CoordAtt_ECA
# from model.ECANet import ECAAttention
from model.SEAAttention import Sea_Attention
from model.SENetv2 import SEAttention

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
        out = self.conv1(x)  # 卷积 out:[N,C,H,W]
        out = self.ln1(out)  #  BN: [N,C,H,W]
        out = self.relu(out) #  [N,C,H,W]
        out = self.conv2(out)  # 卷积 out:[N,C,H,W]
        out = self.ln2(out)    # [N,C,H,W]

        if self.downsample is not None:
            identity = self.downsample(x)  # [N, planes, H/stride, W/stride]

        out += identity   # 残差相加: [N, planes, H/stride, W/stride]
        out = self.relu(out)   # ReLU: [N, planes, H/stride, W/stride]

        return out   # 输出: [N, planes, H/stride, W/stride]


class AttnBottleneck(nn.Module):
    """
    结构：1×1 → 3×3 → 1×1
    通道数：扩展为 planes * 4
    用于深层网络（ResNet-50/101/152）
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
        identity = x          # [N, inplanes, H, W]

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
    def __init__(self, in_planes, out_planes, stride, norm_layer):
        super(DualPathBoundaryBlock, self).__init__()
        
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
        self.arbiter = ECASBF(channels=out_planes, num_branches=2, kernel_size=3)

    def forward(self, x):
        # 1. 计算 Path A
        feat_cons = self.path_conservative(x)
        
        # 2. 计算 Path B (三个子分支拼接 + 融合)
        h = self.path_aggressive_branches['horizontal'](x)
        v = self.path_aggressive_branches['vertical'](x)
        d = self.path_aggressive_branches['diagonal'](x)
        feat_agg_raw = torch.cat([h, v, d], dim=1)
        feat_agg = self.path_aggressive_fusion(feat_agg_raw)
        
        # 3. 动态融合 (二选一或软加权)
        # 输入: [Feature_Old, Feature_New]
        out = self.arbiter([feat_cons, feat_agg])
        
        return out


class StaticAlignmentBlock(nn.Module):
    """静态对齐块 - 根据分支类型使用不同的对齐策略"""

    def __init__(self, in_channels, out_channels, stride=2, branch_type='feature1', norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.branch_type = branch_type

        if branch_type == 'feature1':
            # Feature1: 纹理感知的对齐
            self.alignment = self._build_texture_aware_alignment(in_channels, out_channels, stride, norm_layer)
        elif branch_type == 'feature2':
            # Feature2: 边界保持的对齐
            self.alignment = DualPathBoundaryBlock(in_channels, out_channels, stride, norm_layer)
        elif branch_type == 'feature3':
            # Feature3: 全局信息保持的对齐
            self.alignment = self._build_global_preserving_alignment(in_channels, out_channels, stride, norm_layer)
        else:
            # 默认对齐方式 - 原始CKAAD的方法
            self.alignment = self._build_default_alignment(in_channels, out_channels, stride, norm_layer)

    def _build_texture_aware_alignment(self, in_channels, out_channels, stride, norm_layer):
        """纹理感知的对齐 - 适用于Feature1，加入坐标注意力开头"""
        return nn.Sequential(
            # 在开头添加坐标注意力模块
            CoordAtt_ECA(inp=in_channels, oup=in_channels),
            # 多尺度纹理感知卷积
            nn.Conv2d(in_channels, out_channels//2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels//2),
            nn.ReLU(inplace=True),
            # 空洞卷积增强纹理感知
            nn.Conv2d(out_channels//2, out_channels//2, 3, stride=1, padding=2, dilation=2, bias=False),
            norm_layer(out_channels//2),
            nn.ReLU(inplace=True),
            # 最终输出卷积
            nn.Conv2d(out_channels//2, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    # def _build_boundary_preserving_alignment(self, in_channels, out_channels, stride, norm_layer):
    #     """双分支优化 - 适用于Feature2"""
    #     return nn.Sequential(
    #         # 边界感知卷积
    #         nn.Conv2d(in_channels, out_channels//2, 3, stride=stride, padding=1, bias=False),
    #         norm_layer(out_channels//2),
    #         nn.ReLU(inplace=True),
    #         # 水平和垂直边界增强
    #         nn.Conv2d(out_channels//2, out_channels//2, (1, 3), padding=(0, 1), bias=False),
    #         norm_layer(out_channels//2),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_channels//2, out_channels//2, (3, 1), padding=(1, 0), bias=False),
    #         norm_layer(out_channels//2),
    #         nn.ReLU(inplace=True),
    #         # 最终输出卷积
    #         nn.Conv2d(out_channels//2, out_channels, 1, bias=False),
    #         norm_layer(out_channels),
    #         nn.ReLU(inplace=True)
    #     )
    #     pass

    def _build_global_preserving_alignment(self, in_channels, out_channels, stride, norm_layer):
        """全局信息保持的对齐 - 适用于Feature3"""
        # 重新设计：真正的全局信息保持策略

        # 通道数计算
        mid_channels = out_channels // 2
        global_channels = out_channels // 4

        # 创建一个自定义模块而不是简单的Sequential
        class GlobalPreservingAlign(nn.Module):
            def __init__(self, in_channels, out_channels, stride, norm_layer, mid_channels, global_channels):
                super().__init__()
                self.stride = stride

                # 在开头添加坐标注意力模块
                self.coord_attention = CoordAtt_ECA(inp=in_channels, oup=in_channels)

                # 局部特征分支：下采样获得局部特征
                self.local_branch = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, 3, stride=stride, padding=1, bias=False),
                    norm_layer(mid_channels),
                    nn.ReLU(inplace=True)
                )

                # 全局上下文分支：从原始输入中提取全局信息
                self.global_branch = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),  # 全局池化： [B, in_channels, 1, 1]
                    nn.Conv2d(in_channels, global_channels, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(global_channels, mid_channels, 1, bias=False),
                    norm_layer(mid_channels),
                    nn.ReLU(inplace=True)
                )

                # 融合层
                self.fusion = nn.Sequential(
                    nn.Conv2d(mid_channels * 2, out_channels, 1, bias=False),
                    norm_layer(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                # 首先应用坐标注意力
                x = self.coord_attention(x)

                # 局部分支：下采样
                local_feat = self.local_branch(x)  # [B, mid_channels, H/stride, W/stride]

                # 全局分支：提取全局上下文并广播到与local_feat相同的空间尺寸
                global_context = self.global_branch(x)  # [B, mid_channels, 1, 1]
                # 将全局上下文广播到与local_feat相同的空间尺寸
                global_feat = global_context.expand(-1, -1, local_feat.size(2), local_feat.size(3))

                # 拼接局部特征和全局上下文
                combined = torch.cat([local_feat, global_feat], dim=1)  # [B, mid_channels*2, H/stride, W/stride]

                # 融合
                output = self.fusion(combined)  # [B, out_channels, H/stride, W/stride]

                return output

        return GlobalPreservingAlign(in_channels, out_channels, stride, norm_layer, mid_channels, global_channels)

    def _build_default_alignment(self, in_channels, out_channels, stride, norm_layer):
        """默认对齐方式 - 原始CKAAD的方法，加入坐标注意力开头"""
        return nn.Sequential(
            # 在开头添加坐标注意力模块
            CoordAtt_ECA(inp=in_channels, oup=in_channels),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.alignment(x)


class StaticAlignmentLayer(nn.Module):
    """静态对齐层 - 针对特定分支的完整对齐"""

    def __init__(self, inplanes, target_planes, branch_type, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.branch_type = branch_type

        # 构建对齐序列
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
    静态增强融合层 - 使用静态设置的三分支特征对齐 + 原始CKAAD策略
    保持纯净状态，只使用原始的注意力机制
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

        # ========== 处理分支增强开关参数 ==========
        # enable_branch_enhancement 控制每个分支（feature1, feature2, feature3）是否启用增强对齐
        # 支持两种输入格式：
        # 1. 单个布尔值：应用到所有分支
        # 2. 布尔值列表：分别为每个分支指定是否启用增强

        if isinstance(enable_branch_enhancement, bool):
            # 如果输入是单个布尔值，将其扩展为与分支数量相同的列表
            # 例如：enable_branch_enhancement=True → [True, True, True]
            # 例如：enable_branch_enhancement=False → [False, False, False]
            self.enable_branch_enhancement = [enable_branch_enhancement] * len(input_channels)

        elif isinstance(enable_branch_enhancement, list):
            # 如果输入是列表，需要确保长度与分支数量匹配

            if len(enable_branch_enhancement) > len(input_channels):
                # 列表过长：截取前面的元素，丢弃多余的元素
                # 例如：input_channels=[64,128,256]（3个分支），但收到[True,False,True,False]（4个元素）
                # 结果：[True,False,True]（只使用前3个）
                print(f"Warning: enable_branch_enhancement长度({len(enable_branch_enhancement)})大于输入通道数({len(input_channels)}), 使用前{len(input_channels)}个元素")
                self.enable_branch_enhancement = enable_branch_enhancement[:len(input_channels)]

            elif len(enable_branch_enhancement) < len(input_channels):
                # 列表过短：用False填充缺失的元素
                # 例如：input_channels=[64,128,256]（3个分支），但收到[True,False]（2个元素）
                # 结果：[True,False,False]（第3个分支默认不启用增强）
                print(f"Warning: enable_branch_enhancement长度({len(enable_branch_enhancement)})小于输入通道数({len(input_channels)}), 用False填充")
                padding_length = len(input_channels) - len(enable_branch_enhancement)
                self.enable_branch_enhancement = enable_branch_enhancement + [False] * padding_length

            else:
                # 长度正好匹配：直接使用
                # 例如：input_channels=[64,128,256]，收到[True,False,True] → 直接使用
                self.enable_branch_enhancement = enable_branch_enhancement

        else:
            # 输入类型不支持：抛出错误
            raise ValueError("enable_branch_enhancement必须是bool类型或bool类型的列表")

        # ========== 坐标注意力模块（原始CKAAD策略）==========
        # branch_channels = [c * block.expansion for c in input_channels]
        # self.coord_atts = nn.ModuleList([
        #     CoordAtt_ECA(inp=ch, oup=ch) for ch in branch_channels
        # ])

        # ========== 静态对齐层 ==========
        alignment_layers = []
        for i, input_channel in enumerate(input_channels):
            target_planes = input_channels[-1] * block.expansion
            current_planes = input_channel * block.expansion

            # 根据是否启用增强选择对齐策略
            if self.enable_branch_enhancement[i]:
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
                # 默认对齐方式 - 原始CKAAD的方法
                alignment_layers.append(
                    self._make_default_conv_layer(block, current_planes, target_planes)
                )
        self.alignment_layers = nn.ModuleList(alignment_layers)

        # ========== SENet融合后的通道数 ==========
        # SENet融合后输出单个特征，其通道数等于输入的单个特征通道数
        senet_fused_channel = input_channels[-1] * block.expansion

        # ========== SENet融合机制 ==========
        # 三个对齐后的特征都是 input_channels[-1] * block.expansion = 256 * 4 = 1024 通道
        # SENet会学习每个特征分支的通道注意力权重，实现三个特征的智能融合
        self.senet = SEAttention(channel=input_channels[-1] * block.expansion, reduction=16)

        # SEA注意力：输入通道数等于SENet融合后的通道数
        # input_channels[-1] * block.expansion = 256 * 4 = 1024
        self.axileAttention = Sea_Attention(
            dim=senet_fused_channel,  # 1024
            key_dim=64,               # 每个头的维度
            num_heads=8,              # 注意力头数：1024/64/8=2，符合attn_ratio=2
            attn_ratio=2
        )

        # ========== 后续编码层 ==========
        # 输入是SENet融合后的特征，通道数为 senet_fused_channel
        self.encode_layer1 = self._make_layer(
            block, senet_fused_channel, input_channels[-1] * 2, layers, stride=2
        )

        # ========== 权重初始化 ==========
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_default_conv_layer(self, block, inplanes: int, out_planes: int) -> nn.Sequential:
        """默认卷积对齐层 - 原始CKAAD的方法"""
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

    def forward(self, x: List[Tensor]) -> Tensor:
        """
        前向传播 - 使用之前改进版CKAAD策略：
        1. 坐标注意力增强
        2. 静态特征对齐
        3. 特征拼接
        4. ECA注意力
        5. SEA注意力
        6. 后续编码

        # 1) 在原始预训练特征上先进行坐标注意力增强
        ca_features = [self.coord_atts[i](xi) for i, xi in enumerate(x)]

        # 2) 对每一个分支进行静态特征通道/尺度对齐
        aligned_features = [self.alignment_layers[i](fi) for i, fi in enumerate(ca_features)]

        # 3) 将对齐后的三个分支在通道维度上拼接
        fused = torch.cat(aligned_features, dim=1)

        # 4) 对拼接后的特征应用 ECA 通道注意力
        fused = self.eca_attention(fused)

        # 5) 应用SEA注意力增强结构感知能力
        fused = self.sea_attention(fused)

        # 6) 送入后续编码层进行特征压缩和抽象
        output = self.encode_layer1(fused)
        """
        
        """
        前向传播 - 使用SENet融合策略：
        1. 静态特征对齐
        2. SENet特征融合（三个对齐后的特征通过SE注意力进行融合）
        3. 后续编码层
        """
    

        # 1) 对每一个分支进行静态特征通道/尺度对齐
        aligned_features = [self.alignment_layers[i](fi) for i, fi in enumerate(x)]

        # 2) 使用SENet进行三个对齐特征的融合
        # SENet会学习每个特征分支的通道注意力权重，并进行加权融合
        fused = self.senet(aligned_features[0], aligned_features[1], aligned_features[2])

        # 3) 使用SEA_attention 对整个特征拼接后的整体进行结构感知
        fused = self.axileAttention(fused)

        # 4) 送入后续编码层进行特征压缩和抽象
        output = self.encode_layer1(fused)

        return output.contiguous()


class Encoder(nn.Module):
    """
    兼容性Encoder类 - 封装StaticEnhancedFusionLayer以保持与现有代码的兼容性
    使用静态设置的三分支特征对齐 + 原始CKAAD策略
    """

    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256], attn_block_num=3,
                 enable_enhancement=[False, False, False]):
        super(Encoder, self).__init__()

        self.expansion = 4  # 默认bottleneck expansion

        # 根据骨干网络类型配置融合层
        if backbone == 'resnet18':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBasicBlock, 2, input_channels,
                enable_branch_enhancement=enable_enhancement
            )
            self.expansion = 1
        elif backbone == 'resnet34':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBasicBlock, attn_block_num, input_channels,
                enable_branch_enhancement=enable_enhancement
            )
            self.expansion = 1
        elif backbone == 'resnet50':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels,
                enable_branch_enhancement=enable_enhancement
            )
        elif backbone == 'resnet101':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels,
                enable_branch_enhancement=enable_enhancement
            )
        elif backbone == 'resnet152':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels,
                enable_branch_enhancement=enable_enhancement
            )
        elif backbone == 'wide_resnet50_2':
            # Wide ResNet使用更宽的卷积
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2,
                enable_branch_enhancement=enable_enhancement
            )
        elif backbone == 'wide_resnet101_2':
            self.fusion_layer = StaticEnhancedFusionLayer(
                AttnBottleneck, attn_block_num, input_channels, width_per_group=64 * 2,
                enable_branch_enhancement=enable_enhancement
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 多分支特征列表 [feat1, feat2, feat3]

        Returns:
            融合后的特征张量
        """
        return self.fusion_layer(x)


# ========== 测试函数 ==========
def test_static_enhanced_encoder():
    """
    测试静态增强编码器的功能
    """
    print("="*80)
    print("测试静态增强编码器 - 静态对齐 + 原始CKAAD策略")
    print("="*80)

    # 测试配置
    test_configs = [
        ("基准线（无分支增强）", False),
        ("仅分支1增强", [True, False, False]),
        ("仅分支2增强", [False, True, False]),
        ("仅分支3增强", [False, False, True]),
        ("全部分支增强", [True, True, True]),
    ]

    # 简化的输入特征（模拟Wide ResNet-50-2的输出）
    x1 = torch.randn(2, 256, 64, 64)    # layer1: [B, 256, 64, 64]
    x2 = torch.randn(2, 512, 32, 32)    # layer2: [B, 512, 32, 32]
    x3 = torch.randn(2, 1024, 16, 16)   # layer3: [B, 1024, 16, 16]
    inputs = [x1, x2, x3]

    print(f"输入特征尺寸:")
    for i, feat in enumerate(inputs):
        print(f"  Feature{i+1}: {feat.shape}")

    results = []

    for config_name, enhancement_config in test_configs:
        print(f"\n🧪 测试配置: {config_name}")
        print(f"   配置: {enhancement_config}")

        # 创建编码器
        encoder = Encoder(
            backbone='wide_resnet50_2',
            attn_block_num=3,
            enable_branch_enhancement=enhancement_config
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
    print("<30")
    print("-" * 80)

    for config_name, config, params, macs_g, out_shape in results:
        config_str = str(config)
        print("<30")

    print(f"\n✅ 静态增强编码器测试完成！")
    print(f"   实现了静态设置的三分支特征对齐")
    print(f"   保持了原始CKAAD的注意力机制策略")
    print(f"   支持灵活的分支增强配置")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 运行测试
        test_static_enhanced_encoder()
    else:
        print("=" * 60)
        print("静态增强编码器 - 静态对齐 + 原始CKAAD策略")
        print("=" * 60)
        print("这是一个纯净版本的编码器实现：")
        print("1. 使用静态设置的三分支特征对齐")
        print("2. 特征拼接后使用原始CKAAD策略（ECA + SEA）")
        print("3. 不加入其他注意力机制，保持纯净状态")
        print("4. 支持三分支是否优化的策略开关")
        print("\n使用方法：")
        print("  python model/encoder_static_alignment.py --test")
        print("\n或者在代码中调用：")
        print("  from model.encoder_static_alignment import Encoder")
        print("  encoder = Encoder(enable_branch_enhancement=[True, False, False])")
