#!/bin/bash
#
# test_checkpoint.sh
# -----------
# 快速测试 checkpoint 功能：训练极短 epoch → 保存 checkpoint → 验证加载/跳过训练/list 功能
#
# 用法：
#   bash test_checkpoint.sh           # 运行完整测试流程
#   bash test_checkpoint.sh list      # 仅列出已有 checkpoint
#   bash test_checkpoint.sh skip      # 仅测试 skip_training（需已有 checkpoint）
#   bash test_checkpoint.sh resume    # 仅测试 resume（需已有 checkpoint）
#

set -e

DATASET=mvtec
NORMAL=carpet
TEST_EPOCHS=3           # 测试用极短 epoch 数
CKPT_INTERVAL=1         # 每个 epoch 都保存，方便快速验证
EVAL_EPOCH=1
DEVICE=${CUDA_VISIBLE_DEVICES:-0}

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
run_main() {
    CUDA_VISIBLE_DEVICES=$DEVICE python main.py "$@"
}

log() {
    echo ""
    echo "========================================"
    echo "$*"
    echo "========================================"
}

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
ACTION="${1:-full}"

# 1) 列出已有 checkpoint（训练前）
log "[Step 1/4] 列出已有 checkpoint"
run_main \
    --dataset $DATASET \
    --normal $NORMAL \
    --seed 111 \
    --epochs 1 \
    --img_size 256 \
    --model wide_resnet50_2 \
    --layer 2 \
    --enable_enhancement False \
    --batch_size 4 \
    --labeled_anomaly_ratio 0.0 \
    --labeled_anomaly_class_num 0 \
    --labeled_anomaly_class 0 \
    --log_dir ./log \
    --eval_epoch $EVAL_EPOCH \
    --lr 5e-03 \
    --d_lr 1e-04 \
    --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --use_amp \
    --checkpoint_interval $CKPT_INTERVAL \
    --checkpoint_mode best \
    --list_checkpoints
echo "[Step 1/4] 完成"

# 仅 list 模式（不训练）
if [ "$ACTION" = "list" ]; then
    echo "[list 模式] 测试完毕，退出"
    exit 0
fi

# 2) 训练极短 epoch，产生 checkpoint
log "[Step 2/4] 开始训练 (${TEST_EPOCHS} epochs) 并自动保存 checkpoint"
run_main \
    --dataset $DATASET \
    --normal $NORMAL \
    --seed 111 \
    --epochs $TEST_EPOCHS \
    --img_size 256 \
    --model wide_resnet50_2 \
    --layer 2 \
    --enable_enhancement False \
    --batch_size 4 \
    --labeled_anomaly_ratio 0.0 \
    --labeled_anomaly_class_num 0 \
    --labeled_anomaly_class 0 \
    --log_dir ./log \
    --eval_epoch $EVAL_EPOCH \
    --lr 5e-03 \
    --d_lr 1e-04 \
    --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --use_amp \
    --enable_recon_energy_viz \
    --enable_multi_scale_viz \
    --checkpoint_interval $CKPT_INTERVAL \
    --checkpoint_mode best
echo "[Step 2/4] 训练完成"

# 3) 列出刚才生成的 checkpoint
log "[Step 3/4] 列出训练后新增的 checkpoint"
run_main \
    --dataset $DATASET \
    --normal $NORMAL \
    --seed 111 \
    --epochs 1 \
    --img_size 256 \
    --model wide_resnet50_2 \
    --layer 2 \
    --enable_enhancement False \
    --batch_size 4 \
    --labeled_anomaly_ratio 0.0 \
    --labeled_anomaly_class_num 0 \
    --labeled_anomaly_class 0 \
    --log_dir ./log \
    --eval_epoch $EVAL_EPOCH \
    --lr 5e-03 \
    --d_lr 1e-04 \
    --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --use_amp \
    --checkpoint_interval 999 \
    --checkpoint_mode best \
    --list_checkpoints
echo "[Step 3/4] 完成"

# 仅 skip_training 模式
if [ "$ACTION" = "skip" ] || [ "$ACTION" = "full" ]; then
    # 自动找到 best checkpoint 路径
    CKPT_PATH="./checkpoints/${DATASET}/${NORMAL}/n_${NORMAL}_a_0_s_111/best.pth"
    log "[Step 4/4a] 测试 --skip_training（跳过训练直接评估）"
    echo ">>> 使用 checkpoint: ${CKPT_PATH}"
    run_main \
        --dataset $DATASET \
        --normal $NORMAL \
        --seed 111 \
        --epochs 1 \
        --img_size 256 \
        --model wide_resnet50_2 \
        --layer 2 \
        --enable_enhancement False \
        --batch_size 4 \
        --labeled_anomaly_ratio 0.0 \
        --labeled_anomaly_class_num 0 \
        --labeled_anomaly_class 0 \
        --log_dir ./log \
        --eval_epoch $EVAL_EPOCH \
        --lr 5e-03 \
        --d_lr 1e-04 \
        --adv_conf 0.02 \
        --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
        --use_amp \
        --checkpoint_interval 999 \
        --checkpoint_mode best \
        --skip_training \
        --checkpoint_path "${CKPT_PATH}"
    echo "[Step 4/4a] 完成"
fi

# 仅 resume 模式
if [ "$ACTION" = "resume" ] || [ "$ACTION" = "full" ]; then
    CKPT_PATH="./checkpoints/${DATASET}/${NORMAL}/n_${NORMAL}_a_0_s_111/best.pth"
    log "[Step 4/4b] 测试 --resume（从 checkpoint 恢复训练，多训练 1 epoch）"
    echo ">>> 使用 checkpoint: ${CKPT_PATH}"
    run_main \
        --dataset $DATASET \
        --normal $NORMAL \
        --seed 111 \
        --epochs $((TEST_EPOCHS + 1)) \
        --img_size 256 \
        --model wide_resnet50_2 \
        --layer 2 \
        --enable_enhancement False \
        --batch_size 4 \
        --labeled_anomaly_ratio 0.0 \
        --labeled_anomaly_class_num 0 \
        --labeled_anomaly_class 0 \
        --log_dir ./log \
        --eval_epoch $EVAL_EPOCH \
        --lr 5e-03 \
        --d_lr 1e-04 \
        --adv_conf 0.02 \
        --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
        --use_amp \
        --enable_recon_energy_viz \
        --enable_multi_scale_viz \
        --checkpoint_interval $CKPT_INTERVAL \
        --checkpoint_mode best \
        --resume \
        --checkpoint_path "${CKPT_PATH}"
    echo "[Step 4/4b] 完成"
fi

log "全部测试完成！"
echo ""
echo ">>> checkpoint 保存位置: ./checkpoints/${DATASET}/${NORMAL}/n_${NORMAL}_a_0_s_111/"
echo "    - best.pth   : 最佳指标 checkpoint"
echo "    - final.pth  : 训练结束 checkpoint"
echo "    - epoch_*.pth: 定期保存的 checkpoint"
echo ""
echo ">>> 推理脚本使用方式（训练完成后）："
echo "    python inference_example.py --dataset $DATASET --normal $NORMAL --checkpoint_mode best"
