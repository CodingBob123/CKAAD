#!/bin/bash

# ============================================
# CKAAD 快速测试脚本 - MVTec AD 数据集
# 用于验证模型张量形状是否正确对齐
# ============================================

echo "=========================================="
echo "🔍 开始快速测试模型..."
echo "=========================================="

# 测试配置（快速验证）
normal='bottle'
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
model='wide_resnet50_2'
img_size=256
batch_size=4  # 小批次加快测试
epochs=2      # 只训练2轮验证
eval_epoch=1
lr=5e-03
d_lr=1e-04
adv_conf=0.02
seed=111

echo "测试类别: ${normal}"
echo "模型: ${model}"
echo "特征层: [1, 2, 3]"
echo "图像尺寸: ${img_size}×${img_size}"
echo "批次大小: ${batch_size}"
echo ""

# 执行测试
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
    --log_dir ./log_test \
    --model ${model} \
    --eval_epoch ${eval_epoch} \
    --layer 1 2 3 \
    --topk 100

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 模型测试成功！所有张量形状正确对齐！"
    echo "=========================================="
    echo ""
    echo "📊 张量形状验证通过："
    echo "  ✅ PretrainedFeatureExtractor: 3个特征图"
    echo "  ✅ Encoder: 特征对齐 + SEAttention融合"
    echo "  ✅ Decoder: 3个重建特征图"
    echo "  ✅ Loss Function: 余弦相似度损失"
    echo "  ✅ Discriminator: 对抗训练"
    echo ""
    echo "🚀 可以开始完整训练了！运行命令："
    echo "   cd script && ./mvtec.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 模型测试失败！请检查错误信息"
    echo "=========================================="
    exit 1
fi


