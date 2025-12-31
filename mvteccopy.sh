#!/bin/bash
# ============================================================
# MVTec AD 快速测试脚本 (集成PatchGraph结构感知模块)
# ============================================================
# 该脚本用于快速测试单个类别的训练效果
# 使用较小配置以加快测试速度
# 已集成PatchGraph模块用于结构异常检测

# ==================== 全局配置参数 ====================
labeled_anomaly_ratio=0.05          # 标签异常样本比例
labeled_anomaly_class_num=1         # 标签异常类别数量
patch_graph_k=8                     # PatchGraph kNN邻域数量
patch_graph_mode="consistency"      # PatchGraph异常检测模式

# ==================== 测试类别配置 ====================
# 快速测试只选择一个类别 (cable)
# 如需测试其他类别，请修改下面的循环
for normal in 'cable'
do
    echo "=========================================="
    echo "快速测试类别: $normal (PatchGraph已启用)"
    echo "=========================================="

    # ==================== 快速测试配置 ====================
    # 使用最小配置以加快测试速度
    case $normal in 'carpet')
            epochs=10              # carpet样本少，仍需较多轮数
            eval_epoch=1
            ;;
        *)
            epochs=1               # 快速测试只需1轮
            eval_epoch=1           # 每轮评估
            ;;
    esac

    # 学习率配置
    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)
            lr=5e-03
            ;;
    esac

    # ==================== 快速测试命令 ====================
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --dataset mvtec \
        --batch_size 16 \
        --lr ${lr} \
        --d_lr 1e-04 \
        --adv_conf 0.02 \
        --epochs ${epochs} \
        --normal $normal \
        --seed 111 \
        --img_size 128 \
        --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
        --labeled_anomaly_class 0 \
        --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
        --log_dir ./log \
        --model wide_resnet50_2 \
        --eval_epoch ${eval_epoch} \
        --layer 1 \
        --use_amp \
        \
        # ==================== PatchGraph结构感知模块 ====================
        --enable_patch_graph True \
        --patch_graph_k ${patch_graph_k} \
        --patch_graph_mode ${patch_graph_mode} \
        

    echo "快速测试完成: $normal"
    echo "提示: 如需完整训练，请使用 mvtec.sh 脚本"
    echo ""

done

echo "=========================================="
echo "快速测试完成！"
echo "=========================================="
echo ""
echo "如需对比测试 (不使用PatchGraph):"
echo "CUDA_VISIBLE_DEVICES=0 python main.py --enable_patch_graph False [其他参数...]"
echo ""
echo "如需调整PatchGraph参数:"
echo "--patch_graph_k 16        # 增加邻域数量"
echo "--patch_graph_mode reconstruction    # 切换检测模式"