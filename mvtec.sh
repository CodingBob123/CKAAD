#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
# for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
for normal in 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
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
    --eval_visualize --eval_viz_samples 5 --eval_viz_freq 1 \
    --recon_loss_type combined --loss_alpha 1.0 --loss_beta 0 --loss_gamma 0
done