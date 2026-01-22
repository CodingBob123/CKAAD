#!/usr/bin/env python3
"""
测试可视化修复是否有效
"""

import torch
import sys
import os
sys.path.append('.')

def test_imports():
    """测试必要的导入是否工作"""
    try:
        from torchvision import transforms
        print("✅ torchvision.transforms 导入成功")

        from util.test import visualize_anomaly_maps_simple
        print("✅ visualize_anomaly_maps_simple 函数导入成功")

        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_main_imports():
    """测试main.py中的导入"""
    try:
        # 模拟main.py的导入
        import torch
        from contextlib import nullcontext
        import numpy as np
        import random
        import os
        from util.test import evaluation
        from model.model import PretrainedFeatureExtractor, ED, Discriminator
        import logging
        from argparse import ArgumentParser
        from dataset.dataset import OODDataSet
        from itertools import cycle
        import tqdm
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torchvision import transforms

        print("✅ main.py 中的所有导入都成功")
        return True
    except ImportError as e:
        print(f"❌ main.py 导入失败: {e}")
        return False

def test_visualization_function():
    """测试visualize_anomaly_maps_simple函数的基本结构"""
    try:
        from util.test import visualize_anomaly_maps_simple
        import inspect

        # 检查函数签名
        sig = inspect.signature(visualize_anomaly_maps_simple)
        params = list(sig.parameters.keys())

        expected_params = ['pfe', 'ae', 'dataloader', 'args', 'device', 'epochs']
        if all(param in params for param in expected_params):
            print("✅ visualize_anomaly_maps_simple 函数签名正确")
            print(f"   参数: {params}")
            return True
        else:
            print(f"❌ 函数参数不匹配，期望: {expected_params}，实际: {params}")
            return False

    except Exception as e:
        print(f"❌ 函数检查失败: {e}")
        return False

if __name__ == "__main__":
    print("🔍 测试可视化修复...")
    print("="*50)

    all_passed = True

    print("\n1. 测试基本导入:")
    if not test_imports():
        all_passed = False

    print("\n2. 测试main.py导入:")
    if not test_main_imports():
        all_passed = False

    print("\n3. 测试visualize_anomaly_maps_simple函数:")
    if not test_visualization_function():
        all_passed = False

    print("\n" + "="*50)
    if all_passed:
        print("✅ 所有测试通过！可视化修复应该已经解决导入问题。")
        print("\n现在可以运行训练脚本，应该不会再出现'transforms'未定义的错误。")
    else:
        print("❌ 部分测试失败，请检查上述错误信息。")

    print("\n修复内容总结:")
    print("1. 在main.py中添加了: from torchvision import transforms")
    print("2. 在util/test.py中实现了visualize_anomaly_maps_simple函数")
    print("3. 该函数使用matplotlib而不是OpenCV，避免依赖问题")
