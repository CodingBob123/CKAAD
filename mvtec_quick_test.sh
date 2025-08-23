#!/bin/bash
labeled_anomaly_ratio=0.04
labeled_anomaly_class_num=1

# 只选择几个类别进行快速测试
for normal in 'carpet' 'cable' 'capsule'
do
    echo "Quick test training $normal..."

    epochs=50
    eval_epoch=10

    case $normal in 'transistor')
            lr=1e-03
            ;;
        *)  
            lr=5e-03
            ;;
    esac

    # 使用原始训练脚本
    CUDA_VISIBLE_DEVICES=0 python ./main.py --dataset mvtec --batch_size 16 \
     --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${epochs} \
    --normal $normal --seed 111 --img_size 256 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./quick_test_log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 1 2 3
done
