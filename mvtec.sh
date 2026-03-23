#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
# checkpoint 配置（新增功能）
checkpoint_interval=10          # 每 N 个 epoch 保存一次 checkpoint；0=仅保存 final 和 best
checkpoint_mode=best            # 自动加载策略：best / latest / final
# resume=false                  # 取消注释以从 checkpoint 恢复训练（会加载 optimizer 状态）
# skip_training=false           # 取消注释以跳过训练直接评估（需配合 checkpoint_path 使用）
# checkpoint_path=              # 显式指定 checkpoint 路径，不指定则自动按 mode 搜索

# for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
for normal in 'capsule' 'carpet' 'grid' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo $normal

    case $normal in 'carpet')
            epochs=20
            eval_epoch=1
            ;;
        *)
            epochs=160
            eval_epoch=8
            ;;
    esac

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

    # 根据类别设置enable_enhancement
    case $normal in 'toothbrush'|'transistor'|'wood')
            enable_enhancement=True
            ;;
        *)
            enable_enhancement=False
            ;;
    esac

    # 构建基础命令参数（不变）
    base_args=(
        --dataset mvtec
        --batch_size 16
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
        --log_dir ./log
        --model wide_resnet50_2
        --eval_epoch ${eval_epoch}
        --layer 1 2 3
        --enable_enhancement ${enable_enhancement}
        --gate_k 10.0
        --gate_te 0.5
        --gate_sigma 0.0
        --enable_recon_energy_viz
        --enable_multi_scale_viz
        --use_amp
        # ---- checkpoint 相关参数（新增） ----
        --checkpoint_interval ${checkpoint_interval}
        --checkpoint_mode ${checkpoint_mode}
    )

    # 可选：resume 恢复训练（加载权重 + optimizer 状态）
    if [ "${resume}" = "true" ]; then
        if [ -n "${checkpoint_path}" ]; then
            base_args+=(--resume --checkpoint_path "${checkpoint_path}")
        else
            echo "[WARN] --resume requires --checkpoint_path to be set. Skipping resume."
        fi
    fi

    # 可选：跳过训练直接评估（加载 checkpoint 后立即评估）
    if [ "${skip_training}" = "true" ]; then
        if [ -n "${checkpoint_path}" ]; then
            base_args+=(--skip_training --checkpoint_path "${checkpoint_path}")
        else
            echo "[WARN] --skip_training requires --checkpoint_path to be set. Skipping skip_training."
        fi
    fi

    # 打印即将执行的命令（方便调试）
    echo ">>> CMD: python main.py ${base_args[*]}"

    # 执行训练/评估
    CUDA_VISIBLE_DEVICES=0 python main.py "${base_args[@]}"

    # --enable_enhancement False True False --feature2_fusion_weight ${feature2_fusion_weight} --use_amp
done