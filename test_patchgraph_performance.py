#!/usr/bin/env python3
"""
测试PatchGraph性能优化效果
"""

import torch
import time
from model.patch_graph import PatchGraph, create_patch_graph_for_mvtec

def test_performance_comparison():
    """比较启用/禁用对称和周期关系的性能"""
    print("=== PatchGraph性能测试 ===")

    # 设置测试参数
    batch_size = 4
    channels = 512
    H, W = 32, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"测试配置: batch_size={batch_size}, channels={channels}, HxW={H}x{W}")
    print(f"设备: {device}")

    # 创建测试特征
    features = torch.randn(batch_size, channels, H, W, device=device)

    # 测试配置
    configs = [
        {"name": "轻量版", "enable_symmetry": False, "enable_periodic": False},
        {"name": "对称版", "enable_symmetry": True, "enable_periodic": False},
        {"name": "周期版", "enable_symmetry": False, "enable_periodic": True},
        {"name": "完整版", "enable_symmetry": True, "enable_periodic": True},
    ]

    results = []

    for config in configs:
        print(f"\n测试配置: {config['name']}")

        # 创建PatchGraph实例
        patch_graph = PatchGraph(
            k=8,
            enable_symmetry=config['enable_symmetry'],
            enable_periodic=config['enable_periodic'],
            symmetry_type='axial',
            periodic_directions=4,
            use_local_window=True,
            window_radius=2,
            anomaly_detection_mode='consistency'
        ).to(device)

        # 预热
        with torch.no_grad():
            _ = patch_graph(features)

        # 正式测试
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = patch_graph(features)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # 毫秒
        print(f"平均推理时间: {avg_time:.2f}ms")
        results.append({
            'config': config['name'],
            'time_ms': avg_time,
            'enable_symmetry': config['enable_symmetry'],
            'enable_periodic': config['enable_periodic']
        })

    # 输出性能对比
    print("\n" + "="*50)
    print("性能对比结果:")
    print("="*50)

    baseline_time = results[0]['time_ms']  # 轻量版作为基准

    for result in results:
        speedup = baseline_time / result['time_ms']
        print(f"{result['config']:8} | {result['time_ms']:6.2f}ms | {speedup:.1f}x | "
              f"对称: {result['enable_symmetry']}, 周期: {result['enable_periodic']}")

    print(f"\n性能提升: 完整版 vs 轻量版 = {baseline_time / results[-1]['time_ms']:.1f}x 变慢")
    # 验证功能正确性
    print("\n" + "="*50)
    print("功能验证:")
    print("="*50)

    # 创建两个相同的PatchGraph实例
    pg1 = create_patch_graph_for_mvtec(enable_symmetry=False, enable_periodic=False)
    pg2 = create_patch_graph_for_mvtec(enable_symmetry=True, enable_periodic=True)

    # 测试输出shape一致性
    with torch.no_grad():
        out1 = pg1(features)
        out2 = pg2(features)

    print(f"输出shape: {out1.shape}")
    print(f"Shape一致性: {out1.shape == out2.shape}")

    # 测试基本数值合理性
    print(f"输出范围: [{out1.min().item():.3f}, {out1.max().item():.3f}]")
    print(f"输出均值: {out1.mean().item():.3f}")
    print(f"输出标准差: {out1.std().item():.3f}")

    print("\n✅ 性能测试完成！")

if __name__ == "__main__":
    test_performance_comparison()
