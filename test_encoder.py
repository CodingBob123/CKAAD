#!/usr/bin/env python3
"""
测试改进的编码器功能
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_improved_encoder():
    """测试改进的编码器功能"""
    print("=== 测试改进的编码器 ===")

    # 动态导入，避免循环依赖
    from model.encoder import ImprovedEncoder

    # 创建编码器实例
    encoder = ImprovedEncoder(backbone='wide_resnet50_2', attn_block_num=3)

    # 模拟预训练模型输出的三分支特征
    # 这些尺寸对应Wide ResNet-50-2的layer1, layer2, layer3输出
    x1 = torch.randn(2, 256, 64, 64)    # layer1: [B, 256, 64, 64]
    x2 = torch.randn(2, 512, 32, 32)    # layer2: [B, 512, 32, 32]
    x3 = torch.randn(2, 1024, 16, 16)   # layer3: [B, 1024, 16, 16]

    inputs = [x1, x2, x3]

    print(f"输入特征尺寸:")
    for i, feat in enumerate(inputs):
        print(f"  Feature{i+1}: {feat.shape}")

    # 前向传播
    output = encoder(inputs)

    print(f"\n输出特征尺寸: {output.shape}")
    print(f"输出通道数: {output.shape[1]} (期望: 512 * {encoder.expansion} = {512 * encoder.expansion})")
    print(f"空间尺寸: {output.shape[2]}x{output.shape[3]} (从16x16下采样到8x8)")

    # 参数量分析
    print("\n=== 参数量分析 ===")
    try:
        from thop import profile
        macs, params = profile(encoder, inputs=(inputs,), verbose=False)
        print(f"总参数量: {params/1e6:.2f}M")
        print(f"计算量 (MACs): {macs/1e9:.2f}G")
    except ImportError:
        print("thop库未安装，跳过参数量分析")

    return output


if __name__ == "__main__":
    # 运行测试
    test_improved_encoder()
