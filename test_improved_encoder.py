#!/usr/bin/env python3
"""
测试改进的编码器功能 - 带门控残差学习的版本
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_improved_encoder():
    """测试改进的编码器功能"""
    print("="*80)
    print("残差通道与卷积改进版Encoder - 静态对齐 + SENet融合")
    print("="*80)

    # 动态导入，避免循环依赖
    from model.encoder_improved_alignment import Encoder

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
    print("<30")
    print("-" * 80)

    for config_name, config, params, macs_g, out_shape in results:
        config_str = str(config)
        print("<30")

    print(f"\n✅ 改进版编码器测试完成！")
    print(f"   实现了带门控残差学习的静态特征对齐")
    print(f"   使用SENet进行三个特征的智能融合")
    print(f"   支持灵活的分支增强配置")

    return results


if __name__ == "__main__":
    # 运行测试
    test_improved_encoder()
