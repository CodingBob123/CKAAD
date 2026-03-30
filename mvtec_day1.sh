#!/bin/bash
# Day 1: Recon baseline 训练脚本
# 目标：纯重建误差基线，每 8 轮测试并保存 checkpoint，不使用能量图融合
# 用于验证"能量图引导与筛选"创新点是否有效

labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
checkpoint_interval=80          # 每 多少 个 epoch 保存一次 checkpoint
checkpoint_mode=best         # 加载最新保存的 checkpoint
# checkpoint_dir=./hy-tmp/checkpoints  # checkpoint 存储根目录；设为空或不设置则默认保存在项目根目录的 ./checkpoints 下

for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
# for normal in 'capsule' 'carpet' 'grid' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo "[Day1 Baseline] Training: $normal"

    case $normal in 'bottle'|'cable'|'capsule')
            lr=15e-04
            d_lr=5e-04
            ;;
        'transistor')
            lr=1e-03
            ;;
        *)
            lr=5e-03
            ;;
    esac

    case $normal in 'carpet')
            epochs=20
            eval_epoch=8
            ;;
        *)
            epochs=160
            eval_epoch=8
            ;;
    esac

    case $normal in 'toothbrush'|'transistor'|'wood')
            enable_enhancement=False
            ;;
        *)
            enable_enhancement=False
            ;;
    esac

    # 基础命令：关闭能量图融合，使用 --recon_only 跳过 D(input) 能量图融合
    base_args=(
        --dataset mvtec
        --batch_size 8
        --lr ${lr}
        --d_lr 1e-04
        --adv_conf 0.02
        --epochs ${epochs}
        --normal $normal
        --seed 111
        --img_size 256
        --labeled_anomaly_class_num ${labeled_anomaly_class_num}
        --labeled_anomaly_class 0
        --labeled_anomaly_ratio ${labeled_anomaly_ratio}
        --log_dir ./log_day1
        --model wide_resnet50_2
        --eval_epoch ${eval_epoch}
        --layer 1 2 3
        --enable_enhancement ${enable_enhancement}
        # --recon_only                    # 核心：跳过能量图融合，使用纯重建误差图
        --checkpoint_interval ${checkpoint_interval}
        --checkpoint_mode ${checkpoint_mode}
        --use_amp
        --energy_diff_mode
    )

    echo ">>> CMD: python main.py ${base_args[*]}"
    CUDA_VISIBLE_DEVICES=0 python main.py "${base_args[@]}"

    echo "[Day1 Baseline] Done: $normal"
    echo "======================================"
done

echo "[Day1] All categories completed!"
echo "Checkpoints saved in: ./log_day1/"