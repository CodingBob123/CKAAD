#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
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


    CUDA_VISIBLE_DEVICES=0 python main.py --dataset mvtec --batch_size 16 \
    --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3 \
    --struct_loss_alpha 0.005 --struct_warmup_epochs 10 --struct_ramp_epochs 30 \
    # --enable_enhancement False True False --feature2_fusion_weight ${feature2_fusion_weight} --use_amp
    --enable_enhancement True --enable_epoch_viz --viz_interval 10 --viz_samples_per_type 3 --use_amp
    # --enable_enhancement False --use_amp
done