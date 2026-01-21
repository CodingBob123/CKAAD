#!/usr/bin/env python3
"""
演示训练过程中定期绘制异常热力图的功能
"""

def show_visualization_options():
    """展示可视化选项"""
    print("🎨 异常检测热力图绘制选项")
    print("="*50)

    options = {
        "当前默认行为": [
            "✅ 只在训练结束后绘制一次热力图",
            "📁 保存位置: ./results/{dataset}_{category}_final_epoch_{epochs}/"
        ],

        "新增定期绘制功能": [
            "✅ 在训练过程中每N轮定期绘制热力图",
            "⚙️  控制参数: --enable_epoch_viz --viz_interval N",
            "📁 保存位置: ./results/{dataset}_{category}_epoch_{current_epoch}/"
        ]
    }

    for category, items in options.items():
        print(f"\n🔧 {category}:")
        for item in items:
            print(f"   {item}")

    print("\n" + "="*50)
    print("💡 使用方法:")
    print("1. 保持默认行为（只在训练结束时绘制）:")
    print("   python main.py --dataset mvtec --normal carpet")
    print("")
    print("2. 启用定期绘制（每8轮绘制一次）:")
    print("   python main.py --dataset mvtec --normal carpet --enable_epoch_viz --viz_interval 8")
    print("")
    print("3. 自定义绘制间隔（每5轮绘制一次）:")
    print("   python main.py --dataset mvtec --normal carpet --enable_epoch_viz --viz_interval 5")

def analyze_visualization_workflow():
    """分析可视化工作流程"""
    print("\n🔄 可视化工作流程分析")
    print("="*50)

    workflow = {
        "训练前准备": [
            "设置matplotlib后端为'Agg'（无GUI）",
            "准备数据变换和数据集"
        ],

        "训练中定期评估 (每eval_epoch轮)": [
            "计算性能指标 (AUROC, F1, ACC, PRO)",
            "如果启用 --enable_epoch_viz 且到达viz_interval:",
            "  → 创建可视化数据集（前5个样本）",
            "  → 调用 visualize_anomaly_maps_simple()",
            "  → 保存到 ./results/{dataset}_{category}_epoch_{epoch}/"
        ],

        "训练结束": [
            "保存最终损失曲线到 ./pic/",
            "进行最终异常热力图可视化",
            "保存到 ./results/{dataset}_{category}_final_epoch_{epochs}/"
        ]
    }

    for phase, steps in workflow.items():
        print(f"\n📍 {phase}:")
        for step in steps:
            print(f"   {step}")

def show_file_structure():
    """展示文件结构"""
    print("\n📁 输出文件结构")
    print("="*50)

    structure = """
./results/
├── carpet_final_epoch_200/          # 训练结束时的最终结果
│   ├── 000_label_0.png             # 正常样本
│   ├── 001_label_1.png             # 异常样本
│   └── ...
├── carpet_epoch_8/                  # 第8轮的结果
│   ├── 000_label_0.png
│   └── ...
├── carpet_epoch_16/                 # 第16轮的结果
│   ├── 000_label_0.png
│   └── ...
└── carpet_epoch_200/                # 第200轮的结果
    ├── 000_label_0.png
    └── ...

./pic/
└── loss_curve_final_n_carpet_a_1_s_0.jpg  # 损失曲线
"""

    print(structure)

if __name__ == "__main__":
    show_visualization_options()
    analyze_visualization_workflow()
    show_file_structure()

    print("\n🎯 总结:")
    print("- 默认行为：只在训练结束后绘制一次")
    print("- 新功能：可选择在训练过程中定期绘制")
    print("- 灵活控制：通过命令行参数调整绘制频率")
    print("- 文件管理：不同阶段的结果保存在不同目录")
