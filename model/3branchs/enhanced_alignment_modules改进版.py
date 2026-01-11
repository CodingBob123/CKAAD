import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Callable, Union, Optional, List
from torch import Tensor

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
        self.conv2 = conv3x3(width, width, stride)
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


class EnhancedAlignmentModule(nn.Module):
    """
    改进的特征对齐模块 - 在对齐过程中同时进行针对性增强
    这种设计比先增强再对齐更高效，信息丢失更少
    """
    def __init__(self, 
                 inplanes: int, 
                 target_planes: int,
                 branch_type: str = 'feature1',  # 'feature1', 'feature2', 'feature3'
                 norm_layer = nn.InstanceNorm2d):
        super().__init__()
        
        self.branch_type = branch_type
        self.inplanes = inplanes
        self.target_planes = target_planes
        
        # 根据分支类型选择不同的对齐策略
        if branch_type == 'feature1':
            # Feature1: 纹理感知的对齐
            self.alignment = self._build_texture_aware_alignment(inplanes, target_planes, norm_layer)
        elif branch_type == 'feature2':  
            # Feature2: 边界保持的对齐
            self.alignment = self._build_boundary_preserving_alignment(inplanes, target_planes, norm_layer)
        elif branch_type == 'feature3':
            # Feature3: 全局信息保持的对齐  
            self.alignment = self._build_global_preserving_alignment(inplanes, target_planes, norm_layer)
        else:
            # 默认对齐方式
            self.alignment = self._build_default_alignment(inplanes, target_planes, norm_layer)
    
    def _build_texture_aware_alignment(self, inplanes, target_planes, norm_layer):
        """纹理感知的对齐 - 适用于Feature1"""
        layers = []
        current_planes = inplanes
        
        while current_planes < target_planes:
            # 使用多尺度卷积进行对齐，保持纹理信息
            layers.append(
                MultiScaleAlignmentBlock(
                    in_channels=current_planes,
                    out_channels=current_planes * 2,
                    stride=2,
                    dilation_rates=[1, 2],  # 多尺度纹理感知
                    norm_layer=norm_layer
                )
            )
            current_planes *= 2
            
        return nn.Sequential(*layers)
    
    def _build_boundary_preserving_alignment(self, inplanes, target_planes, norm_layer):
        """边界保持的对齐 - 适用于Feature2"""
        layers = []
        current_planes = inplanes
        
        while current_planes < target_planes:
            # 使用边界感知卷积进行对齐
            layers.append(
                BoundaryPreservingAlignmentBlock(
                    in_channels=current_planes,
                    out_channels=current_planes * 2,
                    stride=2,
                    norm_layer=norm_layer
                )
            )
            current_planes *= 2
            
        return nn.Sequential(*layers)
    
    def _build_global_preserving_alignment(self, inplanes, target_planes, norm_layer):
        """全局信息保持的对齐 - 适用于Feature3"""
        layers = []
        current_planes = inplanes
        
        while current_planes < target_planes:
            # 使用全局信息保持卷积进行对齐
            layers.append(
                GlobalPreservingAlignmentBlock(
                    in_channels=current_planes,
                    out_channels=current_planes * 2,
                    stride=2,
                    norm_layer=norm_layer
                )
            )
            current_planes *= 2
            
        return nn.Sequential(*layers)
    
    def _build_default_alignment(self, inplanes, target_planes, norm_layer):
        """默认对齐方式 - 原始CKAAD的方法"""
        layers = []
        current_planes = inplanes
        
        while current_planes < target_planes:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(current_planes, current_planes * 2, 3, stride=2, padding=1, bias=False),
                    norm_layer(current_planes * 2),
                    nn.ReLU(inplace=True)
                )
            )
            current_planes *= 2
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.alignment(x)


class MultiScaleAlignmentBlock(nn.Module):
    """多尺度对齐块 - 用于纹理感知对齐"""
    def __init__(self, in_channels, out_channels, stride=2, dilation_rates=[1, 2], norm_layer=nn.InstanceNorm2d):
        super().__init__()
        
        # 主干卷积
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度纹理分支
        self.texture_branches = nn.ModuleList()
        branch_channels = out_channels // (2 * len(dilation_rates))
        
        for dilation in dilation_rates:
            self.texture_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, 3, 
                             stride=stride, padding=dilation, dilation=dilation, bias=False),
                    norm_layer(branch_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 主干特征
        main_feat = self.main_conv(x)
        
        # 多尺度纹理特征
        texture_feats = [branch(x) for branch in self.texture_branches]
        
        # 拼接所有特征
        combined = torch.cat([main_feat] + texture_feats, dim=1)
        
        # 融合
        output = self.fusion(combined)
        
        return output


class BoundaryPreservingAlignmentBlock(nn.Module):
    """边界保持对齐块 - 用于边界感知对齐"""
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        
        # 主干卷积
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 边界检测分支
        self.boundary_conv = nn.Sequential(
            # 使用深度可分离卷积检测边界
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            norm_layer(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 局部细节保持分支
        self.detail_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, 1, stride=1, bias=False),
            norm_layer(out_channels//4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((None, None))  # 将在forward中设置目标尺寸
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 主干特征
        main_feat = self.main_conv(x)
        
        # 边界特征
        boundary_feat = self.boundary_conv(x)
        
        # 细节特征 (下采样到目标尺寸)
        detail_feat = self.detail_conv[:-1](x)  # 除了池化的所有层
        _, _, H, W = main_feat.shape
        detail_feat = F.adaptive_avg_pool2d(detail_feat, (H, W))
        
        # 拼接所有特征
        combined = torch.cat([main_feat, boundary_feat, detail_feat], dim=1)
        
        # 融合
        output = self.fusion(combined)
        
        return output


class GlobalPreservingAlignmentBlock(nn.Module):
    """全局信息保持对齐块 - 用于全局感知对齐"""
    def __init__(self, in_channels, out_channels, stride=2, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        
        # 主干卷积
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, stride=stride, padding=1, bias=False),
            norm_layer(out_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 全局上下文分支
        self.global_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels//4, 1, bias=False),
            norm_layer(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度全局池化
        self.multi_global = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels//8, 1, bias=False),
                norm_layer(out_channels//8),
                nn.ReLU(inplace=True)
            ) for scale in [2, 4]
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 主干特征
        main_feat = self.main_conv(x)
        _, _, H_out, W_out = main_feat.shape
        
        # 全局上下文特征
        global_feat = self.global_conv(x)
        global_feat = global_feat.expand(-1, -1, H_out, W_out)
        
        # 多尺度全局特征
        multi_global_feats = []
        for multi_conv in self.multi_global:
            multi_feat = multi_conv(x)
            multi_feat = F.interpolate(multi_feat, (H_out, W_out), mode='bilinear', align_corners=False)
            multi_global_feats.append(multi_feat)
        
        # 拼接所有特征
        combined = torch.cat([main_feat, global_feat] + multi_global_feats, dim=1)
        
        # 融合
        output = self.fusion(combined)
        
        return output


# 使用示例
class ImprovedFusionLayer(nn.Module):
    """改进的融合层 - 使用增强的对齐模块"""
    def __init__(self, input_channels=[64, 128, 256], target_channel=256):
        super().__init__()
        
        # 为每个分支创建针对性的对齐模块
        self.alignments = nn.ModuleList()
        branch_types = ['feature1', 'feature2', 'feature3']
        
        for i, (in_ch, branch_type) in enumerate(zip(input_channels, branch_types)):
            # 计算实际输入通道数 (考虑expansion)
            actual_in_ch = in_ch * 4  # 假设expansion=4
            target_ch = target_channel * 4
            
            self.alignments.append(
                EnhancedAlignmentModule(
                    inplanes=actual_in_ch,
                    target_planes=target_ch,
                    branch_type=branch_type
                )
            )
    
    def forward(self, features):
        # 使用增强的对齐模块处理每个特征
        aligned_features = [
            alignment(feat) for alignment, feat in zip(self.alignments, features)
        ]
        
        # 拼接对齐后的特征
        fused = torch.cat(aligned_features, dim=1)
        
        return fused


if __name__ == "__main__":
    # 测试改进的对齐模块
    fusion_layer = ImprovedFusionLayer()
    
    # 模拟输入
    x1 = torch.randn(2, 256, 64, 64)    # Feature1
    x2 = torch.randn(2, 512, 32, 32)    # Feature2  
    x3 = torch.randn(2, 1024, 16, 16)   # Feature3
    
    inputs = [x1, x2, x3]
    output = fusion_layer(inputs)
    
    print(f"输出特征尺寸: {output.shape}")
    
    # 对比参数量
    from thop import profile
    macs, params = profile(fusion_layer, inputs=(inputs,), verbose=False)
    print(f"参数量: {params/1e6:.2f}M, 计算量: {macs/1e9:.2f}G")
