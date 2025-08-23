#!/bin/bash

# 改进的CKAAD训练脚本示例

echo "开始改进的CKAAD训练..."

# 基础配置 - 使用Wasserstein GAN损失
echo "=== 配置1: Wasserstein GAN损失 ==="
python improved_training.py \
    --dataset mvtec \
    --normal carpet \
    --model wide_resnet50_2 \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --d_lr 1e-05 \
    --loss_type wasserstein \
    --lambda_gp 10.0 \
    --lambda_adv 1.0 \
    --lambda_recon 10.0 \
    --n_critic 5 \
    --log_dir ./improved_log/wgan/

echo "=== 配置2: 添加感知损失 ==="
python improved_training.py \
    --dataset mvtec \
    --normal carpet \
    --model wide_resnet50_2 \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --d_lr 1e-05 \
    --loss_type wasserstein \
    --lambda_gp 10.0 \
    --lambda_adv 1.0 \
    --lambda_recon 10.0 \
    --lambda_perceptual 1.0 \
    --use_perceptual \
    --n_critic 5 \
    --log_dir ./improved_log/wgan_perceptual/

echo "=== 配置3: 添加一致性损失 ==="
python improved_training.py \
    --dataset mvtec \
    --normal carpet \
    --model wide_resnet50_2 \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --d_lr 1e-05 \
    --loss_type wasserstein \
    --lambda_gp 10.0 \
    --lambda_adv 1.0 \
    --lambda_recon 10.0 \
    --lambda_consistency 1.0 \
    --use_consistency \
    --n_critic 5 \
    --log_dir ./improved_log/wgan_consistency/

echo "=== 配置4: 完整配置（所有损失） ==="
python improved_training.py \
    --dataset mvtec \
    --normal carpet \
    --model wide_resnet50_2 \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --d_lr 1e-05 \
    --loss_type wasserstein \
    --lambda_gp 10.0 \
    --lambda_adv 1.0 \
    --lambda_recon 10.0 \
    --lambda_perceptual 1.0 \
    --lambda_consistency 1.0 \
    --lambda_diversity 0.1 \
    --use_perceptual \
    --use_consistency \
    --use_diversity \
    --n_critic 5 \
    --log_dir ./improved_log/wgan_full/

echo "=== 配置5: 相对判别器损失 ==="
python improved_training.py \
    --dataset mvtec \
    --normal carpet \
    --model wide_resnet50_2 \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --d_lr 1e-05 \
    --loss_type relativistic \
    --lambda_adv 1.0 \
    --lambda_recon 10.0 \
    --lambda_perceptual 1.0 \
    --use_perceptual \
    --n_critic 1 \
    --log_dir ./improved_log/relativistic/

echo "=== 配置6: 特征匹配损失 ==="
python improved_training.py \
    --dataset mvtec \
    --normal carpet \
    --model wide_resnet50_2 \
    --batch_size 16 \
    --epochs 200 \
    --lr 0.005 \
    --d_lr 1e-05 \
    --loss_type feature_matching \
    --lambda_gp 10.0 \
    --lambda_fm 1.0 \
    --lambda_adv 1.0 \
    --lambda_recon 10.0 \
    --n_critic 5 \
    --log_dir ./improved_log/feature_matching/

echo "所有训练配置完成！"
