#!/bin/bash
labeled_anomaly_ratio=0.04
labeled_anomaly_class_num=1
for normal in 'cable' 'capsule' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill'
do
    echo "Training $normal with improved reconstruction loss..."

    case $normal in 'carpet')
            epochs=50
            eval_epoch=10
            ;;
        *)  
            epochs=50
            eval_epoch=10
            ;;
    esac

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)  
            lr=5e-03
            ;;
    esac

    # 使用改进的训练脚本，包含图像重建损失
    CUDA_VISIBLE_DEVICES=0 python ./main_improved.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./improved_log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3 \
    --reconstruction_type adaptive --alpha 1.0 --beta 0.5
done
