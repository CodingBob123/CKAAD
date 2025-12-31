#!/bin/bash
# ============================================================
# MVTec AD 全数据集训练脚本 (集成PatchGraph结构感知模块)
# ============================================================
# 该脚本用于在MVTec AD数据集上训练CKAAD模型
# 已集成PatchGraph模块，提升对结构异常的检测能力

# ==================== 全局配置参数 ====================
labeled_anomaly_ratio=0.05          # 标签异常样本比例 (5%)
labeled_anomaly_class_num=1         # 标签异常类别数量
patch_graph_k=8                     # PatchGraph kNN邻域数量
patch_graph_mode="consistency"      # PatchGraph异常检测模式 (consistency/reconstruction/attention)

# ==================== 数据集类别配置 ====================
# 注释掉bottle类别（因为它通常需要特殊处理）
# for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
for normal in 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo "=========================================="
    echo "开始训练类别: $normal"
    echo "=========================================="

    # ==================== 类别特定配置 ====================
    # carpet类别的样本较少，使用较少的训练轮数
    case $normal in 'carpet')
            epochs=10              # 训练轮数
            eval_epoch=1           # 每轮都评估
            ;;
        *)
            epochs=104             # 标准训练轮数
            eval_epoch=8           # 每8轮评估一次
            ;;
    esac

    # transistor类别的学习率需要更保守
    case $normal in 'transistor')
            lr=1e-03               # 学习率
            ;;
        *)
            lr=5e-03               # 标准学习率
            ;;
    esac

    # ==================== 核心训练命令 ====================
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --dataset mvtec \
        --batch_size 16 \
        --lr ${lr} \
        --d_lr 1e-04 \
        --adv_conf 0.02 \
        --epochs ${epochs} \
        --normal $normal \
        --seed 111 \
        --img_size 256 \
        --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
        --labeled_anomaly_class 0 \
        --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
        --log_dir ./log \
        --model wide_resnet50_2 \
        --eval_epoch ${eval_epoch} \
        --layer 1 2 3 \
        \
        # ==================== 可视化配置 ====================
        --eval_visualize \
        --eval_viz_samples 5 \
        --eval_viz_freq 1 \
        \
        # ==================== 重构损失配置 ====================
        --recon_loss_type combined \
        --loss_alpha 1.0 \
        --loss_beta 0 \
        --loss_gamma 0 \
        \
        # ==================== 性能优化 ====================
        --use_amp \
        \
        # ==================== PatchGraph结构感知模块 ====================
        --enable_patch_graph True \
        --patch_graph_k ${patch_graph_k} \
        --patch_graph_mode ${patch_graph_mode} \
        --use_amp

    echo "类别 $normal 训练完成"
    echo ""

done

echo "=========================================="
echo "所有类别训练完成！"
echo "=========================================="