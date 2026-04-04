#!/bin/bash
# EERM 模块第一轮实验脚本
# 目标：基于已训练好的 CKAAD checkpoint，训练 EERM 可靠性调制模块
#
# 实验步骤：
# 1. 使用已有 CKAAD checkpoint，验证 EERM 推理链路
# 2. 冻结主干，训练 EERM 模块
# 3. 多版本对比评估（V7-EERM vs V6-Mean-E-diff vs V2-Input-energy）

labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
checkpoint_interval=160
checkpoint_mode=best
checkpoint_dir=/hy-tmp/checkpoints

# 实验类别（使用已有的 checkpoint）
for normal in 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper' 'carpet'
do
    echo "============================================"
    echo "[EERM Exp] Processing: $normal"
    echo "============================================"

    # checkpoint 路径
    ckpt_path="/hy-tmp/checkpoints/mvtec/${normal}/n_${normal}_a_0_s_111/best.pth"

    if [ ! -f "$ckpt_path" ]; then
        echo "[Warning] Checkpoint not found: $ckpt_path"
        echo "Skipping $normal..."
        continue
    fi

    # ===========================================================
    # 阶段 1: 验证 EERM 推理链路（跳过训练，直接评估）
    # ===========================================================
    echo ""
    echo "[Phase 1] Verifying EERM inference pipeline..."
    echo ">>> CMD: python main.py --skip_training --fusion_mode eerm --checkpoint_path $ckpt_path"

    phase1_args=(
        --dataset mvtec
        --batch_size 16
        --epochs 1
        --normal $normal
        --seed 111
        --img_size 256
        --labeled_anomaly_class_num ${labeled_anomaly_class_num}
        --labeled_anomaly_class 0
        --labeled_anomaly_ratio ${labeled_anomaly_ratio}
        --log_dir ./log_eerm_phase1
        --model wide_resnet50_2
        --eval_epoch 1
        --layer 1 2 3
        --checkpoint_dir ${checkpoint_dir}
        --use_amp
        # --energy_diff_mode

        # EERM 核心参数
        --eerm_mode ckaad                    # 不训练，只用 CKAAD 主干
        --fusion_mode eerm                   # 使用 EERM 融合模式
        --eerm_extra_channels 3              # 5通道：Err + E + delta_l1 + delta_cos + delta_gap
        --eerm_hidden_channels 32
        --skip_training                      # 跳过训练，直接评估
        --checkpoint_path ${ckpt_path}       # 加载已有 checkpoint
    )

    CUDA_VISIBLE_DEVICES=0 python main.py "${phase1_args[@]}"

    if [ $? -ne 0 ]; then
        echo "[Error] Phase 1 failed for $normal"
        continue
    fi

    # ===========================================================
    # 阶段 2: 冻结 CKAAD 主干，训练 EERM 模块
    # ===========================================================
    echo ""
    echo "[Phase 2] Training EERM module (backbone frozen)..."

    case $normal in 'carpet')
        epochs=20
        eval_epoch=10
        ;;
        *)
        epochs=40
        eval_epoch=8
        ;;
    esac

    phase2_args=(
        --dataset mvtec
        --batch_size 16
        --epochs ${epochs}
        --normal $normal
        --seed 111
        --img_size 256
        --labeled_anomaly_class_num ${labeled_anomaly_class_num}
        --labeled_anomaly_class 0
        --labeled_anomaly_ratio ${labeled_anomaly_ratio}
        --log_dir ./log_eerm_phase2
        --model wide_resnet50_2
        --eval_epoch ${eval_epoch}
        --layer 1 2 3
        --checkpoint_interval ${checkpoint_interval}
        --checkpoint_mode ${checkpoint_mode}
        --checkpoint_dir ${checkpoint_dir}
        --use_amp
        --energy_diff_mode

        # EERM 核心参数
        --eerm_mode eerm                     # 冻结主干，训练 EERM
        --enable_eerm_training               # 启用 EERM 损失
        --fusion_mode eerm                    # 使用 EERM 融合模式
        --eerm_extra_channels 3              # 额外通道数
        --eerm_hidden_channels 32            # 隐藏层通道数
        --eerm_lr 1e-4                       # EERM 学习率
        --lambda_good 1.0                    # 正常样本抑制损失权重
        --lambda_keep 1.0                    # 异常峰值保留损失权重
        --eerm_tau 0.7                       # 峰值保留阈值
        --eerm_k_ratio 0.01                  # Top-K 比例

        # 加载已有 checkpoint（训练 CKAAD 时的权重）
        --checkpoint_path ${ckpt_path}
    )

    echo ">>> CMD: python main.py ${phase2_args[*]}"
    CUDA_VISIBLE_DEVICES=0 python main.py "${phase2_args[@]}"

    if [ $? -ne 0 ]; then
        echo "[Error] Phase 2 failed for $normal"
        continue
    fi

    echo "[EERM Exp] Done: $normal"
    echo "============================================"
    echo ""
done

# EERM Checkpoint 输出路径
# /hy-tmp/checkpoints/eerm/{dataset}/{normal}/n_{normal}_a_{class}_s_{seed}/
echo "EERM Checkpoints will be saved to: /hy-tmp/checkpoints/eerm/"
