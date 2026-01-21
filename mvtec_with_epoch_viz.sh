#!/bin/bash
# MVTec AD 数据集训练脚本 - 启用定期异常热力图绘制
# 在训练过程中每8轮定期绘制异常检测热力图，便于监控训练进度

labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
# for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
for normal in 'capsule' 'carpet' 'grid' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
do
    echo "Training $normal (with epoch visualization every 8 epochs)"

    case $normal in 'carpet')
            epochs=10
            eval_epoch=1
            viz_interval=2  # carpet训练轮数少，每2轮绘制一次
            ;;
        *)
            epochs=160
            eval_epoch=8
            viz_interval=8  # 其他类别每8轮绘制一次
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
    --enable_enhancement False False False --use_amp \
    --enable_epoch_viz --viz_interval ${viz_interval}
    # 新增参数：
    # --enable_epoch_viz: 启用训练过程中定期绘制异常热力图
    # --viz_interval: 绘制间隔轮数
done
