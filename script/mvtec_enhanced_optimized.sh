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
    # 以下是针对显存不足问题优化的配置选项
    # ======================================================================

    # 配置1：原始WGAN-GP判别器（使用原始main.py，作为基准）
    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3

    # 配置2：增强的补丁级判别器（减小批量大小为8）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 8 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode patch --use_attention --use_position_encoding --lambda_gp 0

    # 配置3：增强的补丁级判别器（减小图像尺寸为128）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 128 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode patch --use_attention --use_position_encoding --lambda_gp 0

    # 配置4：增强的补丁级判别器（减小批量大小为4，保持原图像尺寸）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 4 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode patch --use_attention --use_position_encoding --lambda_gp 0

    # 配置5：增强的图像级判别器（使用多头注意力，但减小批量大小为8）
    # CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 8 \
    #  --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    # --normal $normal --seed 111 --img_size 256 \
    # --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    # --labeled_anomaly_class 0 \
    # --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    # --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 3 \
    # --discriminator_mode image --use_attention --use_position_encoding --lambda_gp 0

    # 配置6：只使用较浅的网络层（减少特征图数量，仅使用层1和2）
    CUDA_VISIBLE_DEVICES=0 python main_enhanced.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --use_amp --eval_epoch ${eval_epoch} --layer 1 2 \
    --discriminator_mode patch --use_attention --use_position_encoding --lambda_gp 0
done 