import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Type, Callable, Union, Optional, List
import functools

# 注意：增强模块的实现已集成在本文件中，无需额外导入
# from texture_enhancement import MultiScaleTextureBlock
# from boundary_enhancement import LocalBoundaryEnhancementBlock
# from global_consistency import GlobalConsistencyBlock


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class EnhancedAlignmentBlock(nn.Module):
    """
    增强的对齐块 - 在每一步对齐过程中都集成相应的增强功能
    这才是真正按照您想法实现的版本！
    """
    def __init__(self, 
                 in_planes: int, 
                 out_planes: int, 
                 stride: int = 2,
                 branch_type: str = 'feature1',  # 'feature1', 'feature2', 'feature3'
                 norm_layer = nn.InstanceNorm2d,
                 enable_enhancement: bool = True):
        super().__init__()
        
        self.branch_type = branch_type
        self.enable_enhancement = enable_enhancement
        
        # 基础卷积对齐
        self.basic_conv = conv3x3(in_planes=in_planes, out_planes=out_planes, stride=stride)
        self.norm = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 根据分支类型，在对齐的同时进行针对性增强
        if self.enable_enhancement:
            if branch_type == 'feature1':
                # Feature1: 在对齐过程中增强纹理感知
                self.texture_aware_alignment = self._build_texture_aware_conv(
                    in_planes, out_planes, stride, norm_layer
                )
            elif branch_type == 'feature2':
                # Feature2: 在对齐过程中增强边界感知
                self.boundary_aware_alignment = self._build_boundary_aware_conv(
                    in_planes, out_planes, stride, norm_layer
                )
            elif branch_type == 'feature3':
                # Feature3: 在对齐过程中保持全局信息
                self.global_aware_alignment = self._build_global_aware_conv(
                    in_planes, out_planes, stride, norm_layer
                )
    
    def _build_texture_aware_conv(self, in_planes, out_planes, stride, norm_layer):
        """构建纹理感知的对齐卷积"""
        return nn.ModuleDict({
            'multi_scale_conv': nn.ModuleList([
                # 不同膨胀率的卷积，在下采样的同时感知多尺度纹理
                nn.Sequential(
                    nn.Conv2d(in_planes, out_planes//4, 3, stride=stride, padding=dil, dilation=dil, bias=False),
                    norm_layer(out_planes//4),
                    nn.ReLU(inplace=True)
                ) for dil in [1, 2, 3]
            ]),
            'global_branch': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes, out_planes//4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes//4, out_planes//4, 1, bias=False),
                norm_layer(out_planes//4),
                nn.ReLU(inplace=True)
            ),
            'fusion': nn.Sequential(
                nn.Conv2d(out_planes, out_planes, 1, bias=False),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
            )
        })
    
    def _build_boundary_aware_conv(self, in_planes, out_planes, stride, norm_layer):
        """构建边界感知的对齐卷积"""
        # 确保通道数可以被3整除
        ch_per_branch = out_planes // 3
        total_ch = ch_per_branch * 3

        return nn.ModuleDict({
            'horizontal_conv': nn.Sequential(
                # 水平边界检测 + 下采样
                nn.Conv2d(in_planes, ch_per_branch, (1, 7), stride=(1, stride),
                         padding=(0, 3), bias=False),
                nn.AvgPool2d((stride, 1), stride=(stride, 1)),  # 垂直方向下采样
                norm_layer(ch_per_branch),
                nn.ReLU(inplace=True)
            ),
            'vertical_conv': nn.Sequential(
                # 垂直边界检测 + 下采样
                nn.Conv2d(in_planes, ch_per_branch, (7, 1), stride=(stride, 1),
                         padding=(3, 0), bias=False),
                nn.AvgPool2d((1, stride), stride=(1, stride)),  # 水平方向下采样
                norm_layer(ch_per_branch),
                nn.ReLU(inplace=True)
            ),
            'diagonal_conv': nn.Sequential(
                # 对角边界检测 + 下采样
                nn.Conv2d(in_planes, ch_per_branch, 3, stride=stride, padding=1, bias=False),
                norm_layer(ch_per_branch),
                nn.ReLU(inplace=True)
            ),
            'fusion': nn.Sequential(
                nn.Conv2d(total_ch, out_planes, 1, bias=False),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
            )
        })
    
    def _build_global_aware_conv(self, in_planes, out_planes, stride, norm_layer):
        """构建全局感知的对齐卷积"""
        return nn.ModuleDict({
            'local_conv': nn.Sequential(
                # 局部特征 + 下采样
                nn.Conv2d(in_planes, out_planes//2, 3, stride=stride, padding=1, bias=False),
                norm_layer(out_planes//2),
                nn.ReLU(inplace=True)
            ),
            'global_context': nn.Sequential(
                # 全局上下文
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes, out_planes//4, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes//4, out_planes//4, 1, bias=False),
                norm_layer(out_planes//4),
                nn.ReLU(inplace=True)
            ),
            'multi_scale_pool': nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_planes, out_planes//8, 1, bias=False),
                    norm_layer(out_planes//8),
                    nn.ReLU(inplace=True)
                ) for scale in [2, 4]
            ]),
            'fusion': nn.Sequential(
                nn.Conv2d(out_planes, out_planes, 1, bias=False),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
            )
        })
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        if not self.enable_enhancement:
            # 不启用增强时，使用基础对齐
            out = self.basic_conv(x)
            out = self.norm(out)
            out = self.relu(out)
            return out
        
        # 启用增强时，根据分支类型使用不同的增强对齐策略
        if self.branch_type == 'feature1':
            # 纹理感知的对齐
            multi_scale_feats = []
            for conv in self.texture_aware_alignment['multi_scale_conv']:
                multi_scale_feats.append(conv(x))
            
            # 全局分支 - 先上采样再应用最后的归一化
            # 先执行除了最后norm和relu的所有操作
            global_feat = x
            for i, module in enumerate(self.texture_aware_alignment['global_branch']):
                if i < len(self.texture_aware_alignment['global_branch']) - 2:  # 跳过最后两个(norm和relu)
                    global_feat = module(global_feat)

            # 上采样到目标尺寸
            _, _, H_out, W_out = multi_scale_feats[0].shape
            global_feat = F.interpolate(global_feat, (H_out, W_out), mode='bilinear', align_corners=False)

            # 应用最后的归一化和激活
            global_feat = self.texture_aware_alignment['global_branch'][-2](global_feat)  # norm
            global_feat = self.texture_aware_alignment['global_branch'][-1](global_feat)  # relu
            
            # 拼接并融合
            combined = torch.cat(multi_scale_feats + [global_feat], dim=1)
            out = self.texture_aware_alignment['fusion'](combined)
            
        elif self.branch_type == 'feature2':
            # 边界感知的对齐
            h_feat = self.boundary_aware_alignment['horizontal_conv'](x)
            v_feat = self.boundary_aware_alignment['vertical_conv'](x)
            d_feat = self.boundary_aware_alignment['diagonal_conv'](x)
            
            # 拼接并融合
            combined = torch.cat([h_feat, v_feat, d_feat], dim=1)
            out = self.boundary_aware_alignment['fusion'](combined)
            
        elif self.branch_type == 'feature3':
            # 全局感知的对齐
            local_feat = self.global_aware_alignment['local_conv'](x)
            
            # 全局上下文
            global_ctx = self.global_aware_alignment['global_context'](x)
            _, _, H_out, W_out = local_feat.shape
            global_ctx = global_ctx.expand(-1, -1, H_out, W_out)
            
            # 多尺度池化特征
            multi_scale_feats = []
            for pool_conv in self.global_aware_alignment['multi_scale_pool']:
                pool_feat = pool_conv(x)
                pool_feat = F.interpolate(pool_feat, (H_out, W_out), mode='bilinear', align_corners=False)
                multi_scale_feats.append(pool_feat)
            
            # 拼接并融合
            combined = torch.cat([local_feat, global_ctx] + multi_scale_feats, dim=1)
            out = self.global_aware_alignment['fusion'](combined)
            
        else:
            # 默认情况
            out = self.basic_conv(x)
            out = self.norm(out)
            out = self.relu(out)
        
        return out


class TrulyIntegratedConvLayer(nn.Module):
    """
    真正集成的卷积对齐层 - 完全按照您的想法实现
    在对齐过程的每一步都集成相应的增强功能
    """
    def __init__(self, 
                 block, 
                 inplanes: int, 
                 out_planes: int, 
                 branch_index: int,
                 norm_layer,
                 enable_enhancements: bool = True):
        super().__init__()
        
        self.branch_index = branch_index
        self.enable_enhancements = enable_enhancements
        
        # 根据分支索引确定分支类型
        branch_types = ['feature1', 'feature2', 'feature3']
        branch_type = branch_types[min(branch_index, len(branch_types)-1)]
        
        # 构建增强的对齐层序列 - 每一步都集成增强
        self.alignment_blocks = nn.ModuleList()
        
        current_planes = inplanes
        while out_planes * block.expansion != current_planes:
            self.alignment_blocks.append(
                EnhancedAlignmentBlock(
                    in_planes=current_planes,
                    out_planes=current_planes * 2,
                    stride=2,
                    branch_type=branch_type,
                    norm_layer=norm_layer,
                    enable_enhancement=enable_enhancements
                )
            )
            current_planes = current_planes * 2
    
    def forward(self, x):
        """每一步对齐都集成了相应的增强功能"""
        for alignment_block in self.alignment_blocks:
            x = alignment_block(x)
        return x


class CorrectedImprovedFusionLayer(nn.Module):
    """
    修正版的改进融合层 - 按照您的正确想法实现
    真正在特征对齐过程中集成增强功能，而不是先对齐再增强
    """
    def __init__(self,
                 block: Type[Union[nn.Module, nn.Module]],  # AttnBasicBlock or AttnBottleneck
                 layers: int,
                 input_channels: List[int] = [64, 128, 256],
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 width_per_group: int = 64,
                 enable_enhancements: bool = True):
        super(CorrectedImprovedFusionLayer, self).__init__()
        
        if norm_layer is None:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
            
        self._norm_layer = norm_layer
        self.dilation = 1
        self.base_width = width_per_group
        self.enable_enhancements = enable_enhancements
        
        # ========== 核心修正：真正集成的对齐层 ==========
        # 这里才是按照您想法的正确实现！
        conv_layers = []
        for i, input_channel in enumerate(input_channels):
            conv_layers.append(
                TrulyIntegratedConvLayer(
                    block=block,
                    inplanes=input_channel * block.expansion,
                    out_planes=input_channels[-1],
                    branch_index=i,
                    norm_layer=norm_layer,
                    enable_enhancements=enable_enhancements
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # ========== 保持原始的后续逻辑 ==========
        # 坐标注意力模块 (如果需要)
        branch_channels = [c * block.expansion for c in input_channels]
        # self.coord_atts = nn.ModuleList([...])  # 根据需要启用
        
        # 拼接后的通道数
        ca_aligned_channel = input_channels[-1] * block.expansion
        inplanes_after_concat = ca_aligned_channel * len(input_channels)
        
        # ECA和SEA注意力 (根据需要启用)
        # self.eca_attention = ECAAttention(kernel_size=3)
        # self.sea_attention = Sea_Attention(...)
        
        # 后续编码层
        self.encode_layer1 = self._make_layer(
            block, inplanes_after_concat, input_channels[-1] * 2, layers, stride=2
        )
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """保持与原始代码一致的_make_layer实现"""
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
        修正版的前向传播 - 真正在对齐过程中集成增强
        """
        
        # ========== 核心改进：在对齐过程中同时进行增强 ==========
        # 这里每个 conv_layer 都是 TrulyIntegratedConvLayer，
        # 在对齐的每一步都集成了相应的增强功能！
        features = [self.conv_layers[i](xi) for i, xi in enumerate(x)]
        
        # ========== 后续逻辑保持不变 ==========
        # 特征拼接
        fused = torch.cat(features, dim=1)
        
        # 可选的注意力机制
        # fused = self.eca_attention(fused)  # 如需要
        # fused = self.sea_attention(fused)  # 如需要
        
        # 后续编码
        output = self.encode_layer1(fused)
        
        return output.contiguous()


# ========== 使用示例 - 展示真正的集成效果 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("修正版：真正在对齐过程中集成增强的FusionLayer")
    print("=" * 60)
    
    # 简化的block类型用于测试
    class SimpleBlock(nn.Module):
        expansion = 4
        def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, base_width=64):
            super().__init__()
            self.conv = nn.Conv2d(inplanes, planes * self.expansion, 1)
        def forward(self, x):
            return self.conv(x)
    
    # 创建修正版的融合层
    fusion_layer = CorrectedImprovedFusionLayer(
        block=SimpleBlock,
        layers=3,
        input_channels=[64, 128, 256],
        enable_enhancements=True  # 启用在对齐过程中的增强
    )
    
    # 模拟输入
    x1 = torch.randn(2, 64*4, 64, 64)    # Feature1
    x2 = torch.randn(2, 128*4, 32, 32)   # Feature2  
    x3 = torch.randn(2, 256*4, 16, 16)   # Feature3
    
    inputs = [x1, x2, x3]
    
    print("输入特征:")
    for i, xi in enumerate(inputs):
        print(f"  Feature{i+1}: {xi.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = fusion_layer(inputs)
    
    print(f"\n输出特征: {output.shape}")
    
    # 验证增强是真正集成在对齐过程中的
    print(f"\n验证：真正的集成架构")
    print("每个分支的对齐层都包含增强功能：")
    for i, conv_layer in enumerate(fusion_layer.conv_layers):
        print(f"  分支{i+1}: {len(conv_layer.alignment_blocks)} 个增强对齐块")
        for j, block in enumerate(conv_layer.alignment_blocks):
            print(f"    对齐块{j+1}: 分支类型={block.branch_type}, 增强启用={block.enable_enhancement}")
    
    # 参数量和计算量分析
    print(f"\n=== 参数量和计算量分析 ===")

    # 参数量统计
    total_params = sum(p.numel() for p in fusion_layer.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 计算量统计 (使用thop库)
    try:
        from thop import profile
        macs, params = profile(fusion_layer, inputs=(inputs,), verbose=False)
        print(f"计算量 (MACs): {macs/1e9:.2f}G")
        print(f"每张图片的计算量: {macs/inputs[0].shape[0]/1e9:.3f}G")
    except ImportError:
        print("thop库未安装，跳过计算量分析")
        print("如需计算量统计，请安装: pip install thop")

    print("\n✅ 这个版本才真正符合您的想法：在对齐过程中同时进行增强！")
