#!/bin/bash
# EERM Phase 2 单独执行脚本
# 目标：冻结 CKAAD 主干，训练 EERM 可靠性调制模块
#
# 使用方法：
#   bash mvtec_eerm_phase2.sh bottle      # 单类别
#   bash mvtec_eerm_phase2.sh all        # 所有已有 checkpoint 的类别

normal=${1:-bottle}
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
checkpoint_interval=40
checkpoint_mode=best
checkpoint_dir=/hy-tmp/checkpoints

if [ "$normal" == "all" ]; then
    # 处理所有已有 checkpoint 的类别
    for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
    do
        ckpt_path="/hy-tmp/checkpoints/mvtec/${normal}/n_${normal}_a_0_s_111/best.pth"

        if [ ! -f "$ckpt_path" ]; then
            echo "[Warning] Checkpoint not found: $ckpt_path"
            echo "Skipping $normal..."
            continue
        fi

        echo "============================================"
        echo "[EERM Phase 2] Training on: $normal"
        echo "============================================"

        case $normal in 'carpet'|'grid'|'leather')
            epochs=20
            eval_epoch=5
            ;;
            *)
            epochs=40
            eval_epoch=8
            ;;
        esac

        CUDA_VISIBLE_DEVICES=0 python main.py \
            --dataset mvtec \
            --batch_size 16 \
            --epochs ${epochs} \
            --normal ${normal} \
            --seed 111 \
            --img_size 256 \
            --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
            --labeled_anomaly_class 0 \
            --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
            --log_dir ./log_eerm_phase2 \
            --model wide_resnet50_2 \
            --eval_epoch ${eval_epoch} \
            --layer 1 2 3 \
            --checkpoint_interval ${checkpoint_interval} \
            --checkpoint_mode ${checkpoint_mode} \
            --checkpoint_dir ${checkpoint_dir} \
            --use_amp \
            --energy_diff_mode \
            --eerm_mode eerm \
            --enable_eerm_training \
            --fusion_mode eerm \
            --eerm_extra_channels 3 \
            --eerm_hidden_channels 32 \
            --eerm_lr 1e-4 \
            --lambda_good 1.0 \
            --lambda_keep 1.0 \
            --eerm_tau 0.7 \
            --eerm_k_ratio 0.01 \
            --checkpoint_path ${ckpt_path}

        if [ $? -ne 0 ]; then
            echo "[Error] Phase 2 failed for $normal"
            continue
        fi

        echo "[Done] $normal"
        echo ""
    done
else
    # 单类别模式
    ckpt_path="/hy-tmp/checkpoints/mvtec/${normal}/n_${normal}_a_0_s_111/best.pth"

    if [ ! -f "$ckpt_path" ]; then
        echo "[Error] Checkpoint not found: $ckpt_path"
        echo ""
        echo "Available checkpoints:"
        ls -la /hy-tmp/checkpoints/mvtec/*/n_*_a_0_s_111/best.pth 2>/dev/null || echo "  None found"
        exit 1
    fi

    echo "============================================"
    echo "[EERM Phase 2] Training on: $normal"
    echo "Checkpoint: $ckpt_path"
    echo "============================================"

    case $normal in 'carpet'|'grid'|'leather')
        epochs=20
        eval_epoch=5
        ;;
        *)
        epochs=40
        eval_epoch=8
        ;;
    esac

    echo "Epochs: ${epochs}, Eval every: ${eval_epoch} epochs"
    echo ""

    CUDA_VISIBLE_DEVICES=0 python main.py \
        --dataset mvtec \
        --batch_size 16 \
        --epochs ${epochs} \
        --normal ${normal} \
        --seed 111 \
        --img_size 256 \
        --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
        --labeled_anomaly_class 0 \
        --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
        --log_dir ./log_eerm_phase2 \
        --model wide_resnet50_2 \
        --eval_epoch ${eval_epoch} \
        --layer 1 2 3 \
        --checkpoint_interval ${checkpoint_interval} \
        --checkpoint_mode ${checkpoint_mode} \
        --checkpoint_dir ${checkpoint_dir} \
        --use_amp \
        --energy_diff_mode \
        --eerm_mode eerm \
        --enable_eerm_training \
        --fusion_mode eerm \
        --eerm_extra_channels 3 \
        --eerm_hidden_channels 32 \
        --eerm_lr 1e-4 \
        --lambda_good 1.0 \
        --lambda_keep 1.0 \
        --eerm_tau 0.7 \
        --eerm_k_ratio 0.01 \
        --checkpoint_path ${ckpt_path}

    if [ $? -eq 0 ]; then
        echo ""
        echo "============================================"
        echo "[Success] EERM Phase 2 completed for $normal"
        echo "============================================"
        echo "EERM checkpoints saved to:"
        echo "  /hy-tmp/checkpoints/eerm/mvtec/${normal}/n_${normal}_a_0_s_111/"
    fi
fi

echo ""
echo "EERM Phase 2 script finished."
