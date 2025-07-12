#!/bin/bash
labeled_anomaly_ratio=0.04
labeled_anomaly_class_num=1
for normal in 'cable' 'capsule' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill'
do
    echo $normal

    case $normal in 'carpet')
            epochs=120
            eval_epoch=20
            ;;
        *)  
            epochs=120
            eval_epoch=20
            ;;
    esac

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)  
            lr=5e-03
            ;;
    esac

    CUDA_VISIBLE_DEVICES=0 python ./main.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3
done
