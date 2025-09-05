#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1

# 选择要运行的数据集类别
for normal in 'bottle'
do
    echo $normal

    # 根据不同的数据集类别设置训练轮数和评估频率
    case $normal in 'carpet')
            epochs=10
            eval_epoch=1
            ;;
        *)  
            epochs=200
            eval_epoch=10
            ;;
    esac

    # 根据不同的数据集类别设置学习率
    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)  
            lr=5e-03
            ;;
    esac

    # ======================================================================
    # 以下是不同的判别器配置选项，使用时取消注释您需要的配置，注释其他配置
    # ======================================================================

    # 配置1：原始WGAN-GP判别器（使用原始main.py）
    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3

    # 配置2：增强的图像级判别器（带多头注意力和位置编码）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode image --use_attention --use_spectral_norm --use_position_encoding

    # 配置3：增强的补丁级判别器（带多头注意力和位置编码）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode patch --use_attention --use_spectral_norm --use_position_encoding --margin 3.0

    # 配置4：增强的图像级判别器（不带注意力，仅使用谱归一化和位置编码）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode image --use_spectral_norm --use_position_encoding

    # 配置5：增强的补丁级判别器（不带注意力，仅使用谱归一化和位置编码）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode patch --use_spectral_norm --use_position_encoding --margin 3.0

    # 配置6：增强的图像级判别器（仅使用多头注意力，不使用谱归一化）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode image --use_attention --use_position_encoding --lambda_gp 0

    # 配置7：增强的补丁级判别器（仅使用多头注意力，不使用谱归一化）
    CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    --discriminator_mode patch --use_attention --use_position_encoding --lambda_gp 0
done 