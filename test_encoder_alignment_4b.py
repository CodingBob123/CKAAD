#!/usr/bin/env python3
"""
独立的测试脚本，用于测试encoder_alignment_4b.py中的编码器实现
"""

import torch
import torch.nn as nn

def test_dual_path_boundary_block():
    """测试DualPathBoundaryBlock的形状对齐和潜在bug"""
    print("="*80)
    print("测试DualPathBoundaryBlock")
    print("="*80)

    class ECASBF(nn.Module):
        def __init__(self, channels, num_branches=2, kernel_size=3):
            super().__init__()
            self.num_branches = num_branches
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.eca_conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
            self.weight_gen = nn.Conv1d(channels, channels * num_branches, kernel_size=1, groups=1, bias=True)

        def forward(self, branches):
            batch_size, channels, height, width = branches[0].shape
            branch_stack = torch.stack(branches, dim=1)
            U = torch.sum(branch_stack, dim=1)
            S = self.avg_pool(U).view(batch_size, 1, channels)
            Z = self.eca_conv(S)
            Z = Z.permute(0, 2, 1)
            weights = self.weight_gen(Z)
            weights = weights.view(batch_size, self.num_branches, channels)
            attention_vectors = torch.softmax(weights, dim=1)
            attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
            V = torch.sum(branch_stack * attention_vectors, dim=1)
            return V

    class DualPathBoundaryBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride, norm_layer):
            super().__init__()
            self.path_conservative = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
            )
            ch_per_branch = out_planes // 3
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
            total_ch = ch_per_branch * 3
            self.path_aggressive_fusion = nn.Sequential(
                nn.Conv2d(total_ch, out_planes, 1, bias=False),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
            )
            self.arbiter = ECASBF(channels=out_planes, num_branches=2, kernel_size=3)

        def forward(self, x):
            feat_cons = self.path_conservative(x)
            h = self.path_aggressive_branches['horizontal'](x)
            v = self.path_aggressive_branches['vertical'](x)
            d = self.path_aggressive_branches['diagonal'](x)
            feat_agg_raw = torch.cat([h, v, d], dim=1)
            feat_agg = self.path_aggressive_fusion(feat_agg_raw)
            out = self.arbiter([feat_cons, feat_agg])
            return out

    # 测试1: 基本功能
    print("\n1. 基本功能测试:")
    block = DualPathBoundaryBlock(in_planes=256, out_planes=512, stride=2, norm_layer=nn.InstanceNorm2d)
    x = torch.randn(2, 256, 32, 32)
    print(f"输入形状: {x.shape}")

    try:
        output = block(x)
        print(f"输出形状: {output.shape}")
        expected_shape = (2, 512, 16, 16)  # stride=2, H/2=16
        if output.shape == expected_shape:
            print("✓ 形状对齐正确")
        else:
            print(f"✗ 形状不匹配: 期望{expected_shape}, 实际{output.shape}")
    except Exception as e:
        print(f"✗ 前向传播错误: {e}")

    # 测试2: 不同out_planes值，特别是不能被3整除的情况
    print("\n2. 不同输出通道数测试:")
    test_out_planes = [256, 512, 768, 1024, 257, 511, 769]  # 包括不能被3整除的

    for out_planes in test_out_planes:
        try:
            block = DualPathBoundaryBlock(in_planes=256, out_planes=out_planes, stride=2, norm_layer=nn.InstanceNorm2d)
            output = block(x)
            ch_per_branch = out_planes // 3
            total_ch = ch_per_branch * 3

            print(f"  out_planes={out_planes}: ch_per_branch={ch_per_branch}, total_ch={total_ch}, 输出={output.shape}")

            if output.shape[1] != out_planes:
                print(f"    ⚠️  通道数不匹配! 期望{out_planes}, 实际{output.shape[1]}")
            elif total_ch != out_planes:
                print(f"    ⚠️  总通道计算问题: {total_ch} != {out_planes}")
            else:
                print("    ✓ 正常"

        except Exception as e:
            print(f"  ✗ out_planes={out_planes}: 错误 - {e}")

def test_static_alignment_layer():
    """测试StaticAlignmentLayer的循环逻辑"""
    print("\n" + "="*80)
    print("测试StaticAlignmentLayer循环逻辑")
    print("="*80)

    class StaticAlignmentBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=2, branch_type='feature1', norm_layer=nn.InstanceNorm2d):
            super().__init__()
            self.alignment = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.alignment(x)

    class StaticAlignmentLayer(nn.Module):
        def __init__(self, inplanes, target_planes, branch_type, norm_layer=nn.InstanceNorm2d):
            super().__init__()
            layers = []
            current_planes = inplanes
            iteration_count = 0
            max_iterations = 10  # 防止无限循环

            print(f"  构建对齐层: inplanes={inplanes}, target_planes={target_planes}")

            while current_planes < target_planes and iteration_count < max_iterations:
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
                iteration_count += 1
                print(f"    迭代 {iteration_count}: current_planes={current_planes}")

            if iteration_count >= max_iterations:
                print(f"    ⚠️  达到最大迭代次数 ({max_iterations})，可能存在无限循环风险")

            self.alignment_layers = nn.Sequential(*layers)
            print(f"  最终层数: {len(self.alignment_layers)}")

        def forward(self, x):
            return self.alignment_layers(x)

    # 测试不同场景
    test_cases = [
        (256, 1024, "正常情况: 256->512->1024"),
        (512, 1024, "起始等于目标: 512->1024"),
        (128, 2048, "需要多次倍增: 128->256->512->1024->2048"),
        (1024, 512, "目标小于起始: 不应该发生"),
    ]

    for inplanes, target_planes, desc in test_cases:
        print(f"\n测试: {desc}")
        try:
            layer = StaticAlignmentLayer(inplanes, target_planes, 'feature1')
            x = torch.randn(2, inplanes, 64, 64)
            output = layer(x)

            print(f"  输入: {x.shape}")
            print(f"  输出: {output.shape}")

            # 计算期望的输出形状
            if inplanes < target_planes:
                num_layers = len(layer.alignment_layers)
                expected_h = 64 // (2 ** num_layers)
                expected_w = expected_h
                expected_channels = inplanes * (2 ** num_layers)
                expected_shape = (2, expected_channels, expected_h, expected_w)
                print(f"  期望: {expected_shape}")

                if output.shape == expected_shape:
                    print("  ✓ 形状匹配")
                else:
                    print("  ✗ 形状不匹配")
            else:
                print("  ✓ 输入已达到或超过目标，无需对齐")

        except Exception as e:
            print(f"  ✗ 错误: {e}")

def test_se_attention():
    """测试SEAttention的形状对齐"""
    print("\n" + "="*80)
    print("测试SEAttention")
    print("="*80)

    class SEAttention(nn.Module):
        def __init__(self, channel=512, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, 3*channel, bias=False),
            )

        def forward(self, x1, x2, x3):
            B, C, H, W = x1.size()
            x = x1 + x2 + x3
            y = self.avg_pool(x).view(B, C)
            y = self.fc(y).view(B, 3*C, 1, 1)
            weight1 = torch.sigmoid(y[:,:C,:,:])
            weight2 = torch.sigmoid(y[:, C:2*C, :, :])
            weight3 = torch.sigmoid(y[:, 2*C:, :, :])
            out = x1 * weight1 + x2 * weight2 + x3 * weight3
            return out

    # 测试SEAttention
    se = SEAttention(channel=1024, reduction=16)
    x1 = torch.randn(2, 1024, 16, 16)
    x2 = torch.randn(2, 1024, 16, 16)
    x3 = torch.randn(2, 1024, 16, 16)

    print(f"输入形状: {x1.shape}, {x2.shape}, {x3.shape}")

    try:
        output = se(x1, x2, x3)
        print(f"输出形状: {output.shape}")
        if output.shape == x1.shape:
            print("✓ SEAttention形状对齐正确")
        else:
            print("✗ SEAttention形状不匹配")
    except Exception as e:
        print(f"✗ SEAttention错误: {e}")

if __name__ == "__main__":
    test_dual_path_boundary_block()
    test_static_alignment_layer()
    test_se_attention()
    print("\n🎉 所有测试完成！")
