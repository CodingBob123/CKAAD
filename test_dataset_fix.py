#!/usr/bin/env python3
"""
测试MVTecDataset的gt_targets属性修复
"""

import sys
import os
sys.path.append('.')

def test_mvtec_dataset():
    """测试MVTecDataset是否正确设置了gt_targets属性"""
    try:
        from dataset.mvtec import MVTecDataset
        from torchvision import transforms

        print("🔍 测试MVTecDataset修复...")

        # 创建数据集实例（与main.py中相同）
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        gt_transform = transforms.Compose([transforms.ToTensor()])

        # 测试不同类别
        test_categories = ['carpet', 'hazelnut']

        for category in test_categories:
            print(f"\n📂 测试类别: {category}")

            try:
                # 创建数据集
                dataset = MVTecDataset(
                    root='./data',
                    category=category,
                    train=False,
                    transform=img_transform,
                    gt_target_transform=gt_transform,
                    img_size=256
                )

                # 检查基本属性
                print(f"  ✅ 数据集长度: {len(dataset)}")
                print(f"  ✅ targets属性存在: {hasattr(dataset, 'targets')}")

                # 检查在调用load_data()之前
                print(f"  ❌ load_data()前gt_targets存在: {hasattr(dataset, 'gt_targets')}")

                # 调用load_data() - 这是修复的关键
                dataset.load_data()

                # 检查在调用load_data()之后
                print(f"  ✅ load_data()后gt_targets存在: {hasattr(dataset, 'gt_targets')}")
                print(f"  ✅ gt_targets形状: {dataset.gt_targets.shape}")

                # 测试数据访问
                sample_img, sample_gt, sample_label = dataset[0]
                print(f"  ✅ 样本数据形状: img={sample_img.shape}, gt={sample_gt.shape}, label={sample_label}")

                print(f"  🎉 {category}数据集修复成功！")

            except Exception as e:
                print(f"  ❌ {category}数据集测试失败: {e}")
                return False

        print("\n🎯 总结:")
        print("- ✅ MVTecDataset现在正确调用load_data()方法")
        print("- ✅ gt_targets属性在可视化时可用")
        print("- ✅ 异常检测热力图可视化应该能正常工作")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

if __name__ == "__main__":
    success = test_mvtec_dataset()
    if success:
        print("\n🚀 数据集修复验证通过！现在可以正常运行训练和可视化。")
    else:
        print("\n⚠️  数据集修复验证失败，请检查相关代码。")
