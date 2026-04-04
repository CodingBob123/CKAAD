#!/bin/bash
# EERM 单类别调试脚本
# 用于快速验证 EERM 模块是否正常工作

normal='wood'
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
checkpoint_dir=/hy-tmp/checkpoints
ckpt_path="/hy-tmp/checkpoints/mvtec/${normal}/n_${normal}_a_0_s_111/best.pth"

echo "============================================"
echo "[EERM Debug] Testing on: $normal"
echo "Checkpoint: $ckpt_path"
echo "============================================"

if [ ! -f "$ckpt_path" ]; then
    echo "[Error] Checkpoint not found: $ckpt_path"
    exit 1
fi

# 阶段 1: 验证 EERM 推理链路
echo ""
echo "[Phase 1] Verifying EERM inference pipeline..."

CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset mvtec \
    --batch_size 8 \
    --epochs 1 \
    --normal $normal \
    --seed 111 \
    --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log_eerm_debug \
    --model wide_resnet50_2 \
    --eval_epoch 1 \
    --layer 1 2 3 \
    --checkpoint_dir ${checkpoint_dir} \
    --use_amp \
    --energy_diff_mode \
    --eerm_mode ckaad \
    --fusion_mode eerm \
    --eerm_extra_channels 3 \
    --eerm_hidden_channels 32 \
    --skip_training \
    --checkpoint_path ${ckpt_path}

echo "[EERM Debug] Checkpoints will be saved to: /hy-tmp/checkpoints/eerm/"