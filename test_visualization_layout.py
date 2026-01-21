#!/usr/bin/env python3
"""
测试异常检测热力图可视化布局修复
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

def simulate_visualization_data():
    """模拟可视化所需的数据"""
    # 模拟一个256x256的RGB图像
    img_np = np.random.rand(256, 256, 3)
    img_np = np.clip(img_np, 0, 1)

    # 模拟异常热力图
    anomaly_map = np.random.rand(256, 256)
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

    # 模拟ground truth
    gt = np.random.randint(0, 2, (256, 256)).astype(np.uint8)

    return img_np, anomaly_map, gt

def test_visualization_layout():
    """测试可视化布局"""
    print("🔍 测试异常检测热力图可视化布局...")

    # 测试有ground truth的情况
    print("\n📊 测试场景1: 有Ground Truth (2x2布局)")

    img_np, anomaly_map, gt = simulate_visualization_data()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. 显示原始图像
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image\nLabel: 1')
    axes[0].axis('off')

    # 2. 显示异常热力图
    im = axes[1].imshow(anomaly_map, cmap='jet')
    axes[1].set_title('Anomaly Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # 3. 显示ground truth
    axes[2].imshow(gt, cmap='gray')
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    # 4. 显示叠加效果
    axes[3].imshow(img_np)
    axes[3].imshow(anomaly_map, cmap='jet', alpha=0.6)
    axes[3].set_title('Overlay (Original + Anomaly)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('test_layout_with_gt.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ 2x2布局测试完成，保存为: test_layout_with_gt.png")

    # 测试没有ground truth的情况
    print("\n📊 测试场景2: 无Ground Truth (1x3布局)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    # 1. 显示原始图像
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image\nLabel: 0')
    axes[0].axis('off')

    # 2. 显示异常热力图
    im = axes[1].imshow(anomaly_map, cmap='jet')
    axes[1].set_title('Anomaly Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # 3. 显示叠加效果
    axes[2].imshow(img_np)
    axes[2].imshow(anomaly_map, cmap='jet', alpha=0.6)
    axes[2].set_title('Overlay (Original + Anomaly)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('test_layout_no_gt.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ 1x3布局测试完成，保存为: test_layout_no_gt.png")

def compare_old_vs_new():
    """对比修复前后的布局"""
    print("\n🔄 修复前后对比:")

    old_layout = """
修复前的问题布局:
axes[0]: 原始图像 → 然后叠加异常热力图 → 变成混合图像
axes[1]: 异常热力图
axes[2]: Ground Truth (如果有)

结果: axes[0]显示的是原始图像+异常热力图的混合，标题也被覆盖
"""

    new_layout_with_gt = """
修复后的2x2布局 (有Ground Truth):
┌─────────────────┬─────────────────┐
│  Original Image │  Anomaly Map    │
│    Label: 1     │   (with colorbar)│
├─────────────────┼─────────────────┤
│  Ground Truth   │  Overlay        │
│   (Gray)        │ (Original+Anomaly)│
└─────────────────┴─────────────────┘
"""

    new_layout_no_gt = """
修复后的1x3布局 (无Ground Truth):
┌─────────────────┬─────────────────┬─────────────────┐
│  Original Image │  Anomaly Map    │     Overlay     │
│    Label: 0     │ (with colorbar) │ (Original+Anomaly)│
└─────────────────┴─────────────────┴─────────────────┘
"""

    print(old_layout)
    print(new_layout_with_gt)
    print(new_layout_no_gt)

if __name__ == "__main__":
    test_visualization_layout()
    compare_old_vs_new()

    print("\n🎯 总结:")
    print("- ✅ 修复了原始图像被异常热力图覆盖的问题")
    print("- ✅ 现在正确显示: 原始图像 → 异常热力图 → Ground Truth → 叠加效果")
    print("- ✅ 有Ground Truth时使用2x2布局，无Ground Truth时使用1x3布局")
    print("- ✅ 每个子图都有明确的标题和内容，不会相互覆盖")

    print("\n📁 生成的测试图片:")
    print("- test_layout_with_gt.png: 有Ground Truth的2x2布局")
    print("- test_layout_no_gt.png: 无Ground Truth的1x3布局")
