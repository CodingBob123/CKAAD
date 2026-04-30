#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1

# ========== AFS 参数说明 ==========
# AFS auto-half: 对于 wide_resnet50_2 --layer 1 2 3:
#   layer1: 256 → 128 | layer2: 512 → 256 | layer3: 1024 → 512
# 可手动指定: --afs_select_planes 128 256 512
#
# 如需启用合成异常协同训练（可选），取消下面注释：
#   --use_synthetic_anomaly --anomaly_perturbation noise \
#   --anomaly_ratio 0.3 --anomaly_noise_std 0.15 --multi_scale_anomaly
#
# 如需使用 SimpleNet 风格全局多级噪声扰动：
#   --use_synthetic_anomaly --anomaly_perturbation simplenet_noise \
#   --anomaly_noise_std 0.05 --anomaly_mix_noise 3 \
#   说明: mix_noise=3 表示噪声 std 按 1.1^k 递增 (k=0,1,2)，
#         每样本随机选一级，实现多级噪声策略
#
# 如需使用硬增强擦除（特征均值填充）：
#   --use_synthetic_anomaly --anomaly_perturbation hard_erase \
#   说明: 基于Perlin掩码的不规则形状擦除，掩码区域用特征均值填充，
#         区别于 erase 的归零操作，模拟图像级硬增强遮挡效果

# ========== 像素级异常合成（PatchGuard + OCR-GAN）参数 ==========
# 如需启用像素级异常合成（与 Perlin 合成异常协同训练），添加：
#   --use_pixel_anomaly --pixel_anomaly_mode mixed \
#   --anomaly_strategy progressive \
#   --pixel_anomaly_prob 0.4 --perlin_anomaly_prob 0.35
#
# 如需仅使用像素级异常（不使用 Perlin），添加：
#   --use_pixel_anomaly --pixel_anomaly_mode mixed \
#   （不添加 --use_synthetic_anomaly）
#
# pixel_anomaly_mode 选项:
#   patchguard : 前景感知 cut-and-paste（需 foreground_mask 目录）
#   cutpaste   : OCR-GAN CutPaste（仅ColorJitter，无需前景mask）
#   cutout     : OCR-GAN Cutout（随机擦除，无需前景mask）
#   mixed      : 随机选择上述三种

for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo $normal

    case $normal in 'carpet')
            epochs=10
            eval_epoch=1
            ;;
        *)
            epochs=104
            eval_epoch=8
            ;;
    esac

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)
            lr=5e-03
            ;;
    esac

CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --batch_size 16 \
 --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
--normal $normal --seed 111 --img_size 256 \
--labeled_anomaly_class_num ${labeled_anomaly_class_num} \
--labeled_anomaly_class 0 \
--labeled_anomaly_ratio ${labeled_anomaly_ratio} \
--log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3 \
--recon_loss_type combined --loss_alpha 1.0 --loss_beta 0 --loss_gamma 0 \
--use_rrs --rrs_anomaly_samples 20 --rrs_lr 1e-3 --rrs_loss_weight 1.0 \
--use_afs --afs_init_bsn 50
done
