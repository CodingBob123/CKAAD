#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
# for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
for normal in 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo $normal

    # 基于训练日志分析优化epoch设置
    case $normal in 'carpet')
            epochs=20  # carpet快速收敛，保持10个epoch
            eval_epoch=1
            ;;
        'transistor')
            epochs=160  # transistor收敛较慢，增加到90个epoch
            eval_epoch=8
            ;;
        'wood')
            epochs=160  # wood收敛中等，设置为85个epoch
            eval_epoch=8
            ;;
        *)  # 其他类别基于bottle表现调整
            epochs=150  # 大部分类别在70个epoch内收敛良好
            eval_epoch=8
            ;;
    esac

    # 基于训练稳定性优化学习率
    case $normal in 'transistor')
            lr=8e-04  # transistor保持较低学习率
            ;;
        'wood')
            lr=15e-04  # wood轻微降低学习率提高稳定性
            ;;
        *)  # 其他类别进一步降低学习率
            lr=15e-04  # 降低到1.5e-3，提高训练稳定性
            ;;
    esac

    # 基于损失曲线分析调整判别器学习率
    case $normal in 'carpet')
            d_lr=3e-04  # carpet判别器学习率略低
            ;;
        'transistor')
            d_lr=8e-04  # transistor判别器学习率较高以匹配主学习率
            ;;
        *)  # 其他类别保持平衡
            d_lr=5e-04
            ;;
    esac

    CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr ${d_lr} --adv_conf 0.008 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3
done