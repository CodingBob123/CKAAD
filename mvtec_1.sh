#!/bin/bash
# ======================================================================
# CKAAD + 多模态异常合成控制器 — MVTec AD 完整训练脚本
# ======================================================================
# 推荐参数配置说明（基于 UnifiedAnomalyController 的渐进式策略）
# ======================================================================
#
# 异常合成模块              | 参数/模式                      | 推荐值       | 说明
# --------------------------|--------------------------------|--------------|------------------------------------------
# [控制器] UnifiedAnomaly   | --anomaly_strategy             | progressive  | 渐进式: 前期Perlin→后期Pixel→混合泛化
# [像素级] PixelAnomaly(总) | --use_pixel_anomaly            | (启用)       | 激活 PatchGuard + CutPaste + Cutout
# [像素] 子模式选择          | --pixel_anomaly_mode           | mixed        | 随机混合三种像素级异常
# [像素] PatchGuard 概率    | --pixel_patchguard_prob         | 0.5          | 前景感知 cut-and-paste（需前景 mask）
# [像素] CutPaste 概率      | --pixel_cutpaste_prob           | 0.3          | OCR-GAN 风格局部自粘贴
# [像素] Cutout 概率        | --pixel_cutout_prob             | 0.2          | 随机方块擦除
# [像素] 每样本像素异常概率  | --pixel_anomaly_prob            | 0.4          | 40% 样本走像素级异常
# [特征级] Perlin 合成       | --use_synthetic_anomaly         | (启用)       | 激活特征空间 Perlin 扰动
# [Perlin] 扰动类型          | --anomaly_perturbation         | simplenet_noise | 多级噪声策略（最鲁棒）
# [Perlin] 噪声强度          | --anomaly_noise_std            | 0.05         | 基础噪声标准差
# [Perlin] 噪声级数          | --anomaly_mix_noise            | 3            | 3 级噪声按 1.1^k 递增
# [Perlin] 多尺度面积        | --multi_scale_anomaly          | (启用)       | 训练中动态变化异常面积
# [Perlin] 每样本 Perlin 概率| --perlin_anomaly_prob           | 0.35         | 35% 样本走 Perlin 特征扰动
# [保留正常] 25% 样本保持纯正常（无任何异常合成），提升 AE 重建鲁棒性
# [AFS] 异常感知特征选择     | --use_afs --afs_init_bsn 50    | (启用)       | 自动半通道选择（256→128 512→256 1024→512）
# [RRS] 残差分割网络         | --use_rrs --rrs_anomaly_samples 20 | (启用)  | 用测试集真实异常训练分割分支
# [消融对照 1] 仅 Pixel      | --use_pixel_anomaly（不设 use_synthetic_anomaly）| 所有样本 100% 像素异常
# [消融对照 2] 仅 Perlin     | --use_synthetic_anomaly（不设 use_pixel_anomaly） | 所有样本 100% 特征扰动
# [消融对照 3] Raw Elastic   | 原版 labeled_anomaly_ratio=0.05（不使用任何合成） | ElasticTransform 弹性形变
# ======================================================================

# ======================================================================
# 全局参数
# ======================================================================
labeled_anomaly_ratio=0          # 必须 =0：避免 AnomalyDataset 包装，确保数据集返回 foreground_mask
labeled_anomaly_class_num=1

# ======================================================================
# 合成异常参数（控制器 + 像素级 + 特征级 Perlin）
# ======================================================================

# === 控制器策略 ===
# 'prob'        : 固定概率分配（简单直接）
# 'adapt'       : 根据类别是否有前景 mask 自适应调整概率
# 'progressive' : 按 epoch 进度渐进变化（推荐：前期 Perlin 简单噪声 → 后期 PatchGuard 复杂结构）
anomaly_strategy=progressive

# === 像素级异常（PixelAnomalyGenerator） ===
pixel_anomaly_mode=mixed          # mixed: 随机选择 PatchGuard/CutPaste/Cutout
pixel_patchguard_prob=0.5         # mixed 内 PatchGuard 概率
pixel_cutpaste_prob=0.3           # mixed 内 CutPaste 概率
pixel_cutout_prob=0.2             # mixed 内 Cutout 概率
pixel_anomaly_prob=0.4            # 控制器中像素级异常的出现概率

# === 特征级 Perlin 异常（PerlinAnomalyGenerator） ===
anomaly_perturbation=simplenet_noise  # 多级噪声扰动（最鲁棒的 Perlin 模式）
anomaly_noise_std=0.05             # 基础噪声强度
anomaly_mix_noise=3                # 3 级噪声 (1.1^0, 1.1^1, 1.1^2)
perlin_anomaly_prob=0.35           # 控制器中 Perlin 异常的出现概率
# 注: 剩余 1 - 0.40 - 0.35 = 25% 样本保持正常（无异常合成）

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
--use_afs --afs_init_bsn 50 \
\
--use_pixel_anomaly \
--pixel_anomaly_mode ${pixel_anomaly_mode} \
--pixel_patchguard_prob ${pixel_patchguard_prob} \
--pixel_cutpaste_prob ${pixel_cutpaste_prob} \
--pixel_cutout_prob ${pixel_cutout_prob} \
--pixel_anomaly_prob ${pixel_anomaly_prob} \
\
--use_synthetic_anomaly \
--anomaly_perturbation ${anomaly_perturbation} \
--anomaly_noise_std ${anomaly_noise_std} \
--anomaly_mix_noise ${anomaly_mix_noise} \
--multi_scale_anomaly \
--perlin_anomaly_prob ${perlin_anomaly_prob} \
\
--anomaly_strategy ${anomaly_strategy}
done
