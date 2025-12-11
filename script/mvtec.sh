#!/bin/bash

# ============================================
# CKAAD 训练脚本 - MVTec AD 数据集
# 模型: wide_resnet50_2 + SEAttention
# 特征层: layer1, layer2, layer3
# ============================================

# 全局配置
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
model='wide_resnet50_2'
img_size=256
batch_size=16
seed=111

# MVTec AD 所有类别
categories=(
    'bottle' 'cable' 'capsule' 'carpet' 'grid'
    'hazelnut' 'leather' 'metal_nut' 'pill' 'screw'
    'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
)

# 遍历所有类别进行训练
for normal in "${categories[@]}"
do
    echo "=========================================="
    echo "开始训练类别: $normal"
    echo "=========================================="

    # 根据不同类别设置训练轮数
    case $normal in 
        'carpet'|'grid'|'leather'|'tile'|'wood')
            # 纹理类别：训练更多轮次
            epochs=200
            eval_epoch=10
            ;;
        'bottle'|'cable'|'capsule'|'hazelnut'|'metal_nut'|'pill'|'toothbrush'|'zipper')
            # 物体类别：标准训练轮次
            epochs=150
            eval_epoch=10
            ;;
        'screw'|'transistor')
            # 复杂类别：需要更多训练
            epochs=250
            eval_epoch=10
            ;;
        *)  
            # 默认配置
            epochs=200
            eval_epoch=10
            ;;
    esac

    # 根据不同类别设置学习率
    case $normal in 
        'transistor'|'screw')
            # 复杂类别使用较小学习率
            lr=1e-03
            d_lr=5e-05
            ;;
        'grid'|'carpet')
            # 纹理类别使用较大学习率
            lr=8e-03
            d_lr=1e-04
            ;;
        *)  
            # 默认学习率
            lr=5e-03
            d_lr=1e-04
            ;;
    esac

    # 根据不同类别设置对抗损失权重
    case $normal in 
        'screw'|'transistor'|'pill')
            # 小物体使用较大对抗权重
            adv_conf=0.03
            ;;
        *)  
            # 默认对抗权重
            adv_conf=0.02
            ;;
    esac

    # 执行训练
    CUDA_VISIBLE_DEVICES=0 python ../main.py \
        --dataset mvtec \
        --batch_size ${batch_size} \
        --lr ${lr} \
        --d_lr ${d_lr} \
        --adv_conf ${adv_conf} \
        --epochs ${epochs} \
        --normal ${normal} \
        --seed ${seed} \
        --img_size ${img_size} \
        --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
        --labeled_anomaly_class 0 \
        --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
        --log_dir ./log \
        --model ${model} \
        --eval_epoch ${eval_epoch} \
        --layer 1 2 3 \
        --topk 100

    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 类别 ${normal} 训练完成！"
    else
        echo "❌ 类别 ${normal} 训练失败！"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "🎉 所有类别训练完成！"
echo "=========================================="
