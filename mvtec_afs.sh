#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
for normal in 'bottle'
do
    echo $normal

    case $normal in 'carpet')
            epochs=1
            eval_epoch=1
            ;;
        *)  
            epochs=1
            eval_epoch=1
            ;;
    esac

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)  
            lr=5e-03
            ;;
    esac

    # AFS模块训练（不使用SDAS/DTD数据，使用简化合成）
    CUDA_VISIBLE_DEVICES=0 python ./main.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3 \
    --use_afs --afs_init_bsn 200 --afs_planes 64
done
