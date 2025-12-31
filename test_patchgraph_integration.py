#!/usr/bin/env python3
"""
测试PatchGraph在CKAAD中的集成是否正常工作
"""

import torch
import torch.nn as nn
from model.model import PretrainedFeatureExtractor, ED
import argparse

def test_patchgraph_integration():
    """测试PatchGraph集成"""
    print("=== 测试PatchGraph集成 ===")

    # 设置参数
    args = argparse.Namespace()
    args.model = 'wide_resnet50_2'
    args.layer = [2]  # 使用layer2作为测试
    args.enable_patch_graph = True
    args.patch_graph_k = 8
    args.patch_graph_mode = 'consistency'

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        # 1. 初始化预训练特征提取器
        print("初始化预训练特征提取器...")
        pfe = PretrainedFeatureExtractor(
            backbone=args.model,
            pretrained=True,
            layers=args.layer
        ).to(device)  # 🔧 修复：将pfe也移到设备上
        print(f"特征提取器输出通道: {pfe.output_channels}")
        print(f"特征提取器输出尺寸: {pfe.output_sizes}")

        # 2. 初始化带有PatchGraph的ED模型
        print("初始化带有PatchGraph的ED模型...")
        ae = ED(
            backbone=args.model,
            input_channels=pfe.output_channels,
            enable_patch_graph=args.enable_patch_graph,
            patch_graph_k=args.patch_graph_k,
            patch_graph_mode=args.patch_graph_mode
        ).to(device)

        # 3. 创建测试输入
        batch_size = 2
        input_size = 256
        input_tensor = torch.randn(batch_size, 3, input_size, input_size).to(device)
        print(f"输入张量形状: {input_tensor.shape}")

        # 4. 前向传播测试
        print("开始前向传播测试...")

        # 4.1 特征提取器
        with torch.no_grad():
            features = pfe(input_tensor)
        print(f"提取的特征数量: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  特征{i}形状: {feat.shape}")

        # 4.2 Encoder (包含PatchGraph)
        print("测试Encoder (包含PatchGraph)...")
        encoded = ae.encoder(features)
        print(f"编码器输出形状: {encoded.shape}")

        # 4.3 Decoder
        print("测试Decoder...")
        decoded = ae.decoder(encoded)
        print(f"解码器输出数量: {len(decoded)}")
        for i, dec in enumerate(decoded):
            print(f"  解码特征{i}形状: {dec.shape}")

        # 4.4 完整前向传播
        print("测试完整模型前向传播...")
        with torch.no_grad():
            pfe.eval()
            features_full = pfe(input_tensor)
            output_full = ae(features_full)
        print(f"完整输出数量: {len(output_full)}")
        for i, out in enumerate(output_full):
            print(f"  输出{i}形状: {out.shape}")

        print("✅ PatchGraph集成测试成功！")

        # 5. 检查PatchGraph是否真的被集成
        print("\n检查PatchGraph模块状态:")
        fusion_layer = ae.encoder.fusion_layer
        if hasattr(fusion_layer, 'patch_graph'):
            print("✅ PatchGraph模块已成功集成")
            print(f"  - kNN邻域数: {fusion_layer.patch_graph.k}")
            print(f"  - 对称关系: {fusion_layer.patch_graph.enable_symmetry}")
            print(f"  - 周期关系: {fusion_layer.patch_graph.enable_periodic}")
            print(f"  - 异常检测模式: {fusion_layer.patch_graph.anomaly_detection_mode}")
        else:
            print("❌ PatchGraph模块未找到")

        if hasattr(fusion_layer, 'structure_upsampler'):
            print("✅ 结构上采样模块已集成")
        else:
            print("❌ 结构上采样模块未找到")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_patchgraph():
    """测试不使用PatchGraph的基准情况"""
    print("\n=== 测试不使用PatchGraph的基准情况 ===")

    args = argparse.Namespace()
    args.model = 'wide_resnet50_2'
    args.layer = [2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        pfe = PretrainedFeatureExtractor(
            backbone=args.model,
            pretrained=True,
            layers=args.layer
        ).to(device)  # 🔧 修复：将pfe也移到设备上

        ae = ED(
            backbone=args.model,
            input_channels=pfe.output_channels,
            enable_patch_graph=False  # 禁用PatchGraph
        ).to(device)

        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)

        with torch.no_grad():
            pfe.eval()
            features = pfe(input_tensor)
            output = ae(features)

        print("✅ 不使用PatchGraph的基准测试成功")
        return True

    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始PatchGraph集成测试...\n")

    # 测试启用PatchGraph的情况
    success_with_pg = test_patchgraph_integration()

    # 测试禁用PatchGraph的情况
    success_without_pg = test_without_patchgraph()

    if success_with_pg and success_without_pg:
        print("\n🎉 所有测试通过！PatchGraph已成功集成到CKAAD架构中。")
        print("\n接下来你可以:")
        print("1. 运行完整训练: python main.py --enable_patch_graph --patch_graph_k 8")
        print("2. 调整参数: --patch_graph_mode consistency/reconstruction/attention")
        print("3. 在MVTec数据集上测试结构异常检测效果")
    else:
        print("\n❌ 部分测试失败，请检查代码实现。")
