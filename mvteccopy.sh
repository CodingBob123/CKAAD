# #!/bin/bash

# # ============================================================================
# # CKAAD 训练脚本 - 支持渐进式实验配置
# # ============================================================================
# #
# # 功能特性：
# # - 支持渐进式分支增强实验
# # - 自动参数配置优化
# # - 多GPU并行训练支持
# # - 完善的日志和错误处理
# #
# # 使用方法：
# # 1. 修改 EXPERIMENT_MODE 变量选择实验类型
# # 2. 调整 CATEGORIES 选择训练类别
# # 3. 根据需要修改其他配置参数
# # 4. 运行脚本: ./mvteccopy.sh
# #
# # 实验模式说明：
# # - baseline: 基准线实验（无增强）
# # - single_branch: 单分支增强实验
# # - multi_branch: 多分支增强实验
# # - full: 全增强实验

# # ------------------------- 全局配置 -------------------------
# # 基础参数
# DATASET="mvtec"
# BATCH_SIZE=16
# IMG_SIZE=128
# SEED=111
# MODEL="wide_resnet50_2"
# LOG_DIR="./log"

# # 训练参数
# D_LR=1e-04
# ADV_CONF=0.02
# LABELED_ANOMALY_RATIO=0.05
# LABELED_ANOMALY_CLASS_NUM=1
# LABELED_ANOMALY_CLASS=0

# # 类别特定的参数
# declare -A EPOCHS_CONFIG=(
#     ["carpet"]=10
#     ["default"]=200  # 默认轮数
# )

# declare -A LR_CONFIG=(
#     ["transistor"]=1e-03
#     ["default"]=5e-03  # 默认学习率
# )

# declare -A EVAL_EPOCH_CONFIG=(
#     ["carpet"]=1
#     ["default"]=8  # 默认评估间隔
# )

# # ------------------------- 实验配置 -------------------------
# # 选择实验模式
# EXPERIMENT_MODE="single_branch"  # 可选: baseline, single_branch, multi_branch, full

# # 渐进式实验配置
# case $EXPERIMENT_MODE in
#     "baseline")
#         echo "=== 运行基准线实验（无增强）==="
#         ENABLE_ENHANCEMENT_FLAGS=("False" "False" "False")
#         ;;
#     "single_branch")
#         echo "=== 运行单分支增强实验 ==="
#         ENABLE_ENHANCEMENT_FLAGS=("False" "True" "False")  # 只启用分支2（边界增强）
#         ;;
#     "multi_branch")
#         echo "=== 运行多分支增强实验 ==="
#         ENABLE_ENHANCEMENT_FLAGS=("True" "True" "False")  # 分支1+2
#         ;;
#     "full")
#         echo "=== 运行全增强实验 ==="
#         ENABLE_ENHANCEMENT_FLAGS=("True" "True" "True")   # 全增强
#         ;;
#     *)
#         echo "错误：未知的实验模式 $EXPERIMENT_MODE"
#         exit 1
#         ;;
# esac

# # ------------------------- GPU配置 -------------------------
# # 支持多GPU并行训练
# GPU_DEVICES="0"  # 可以设置为 "0,1" 来使用多GPU
# NUM_GPUS=$(echo $GPU_DEVICES | tr ',' '\n' | wc -l)

# # ------------------------- 类别列表 -------------------------
# # 完整MVTec AD类别列表
# # CATEGORIES=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')

# # 测试用类别（可以根据需要修改）
# CATEGORIES=('transistor')

# # ------------------------- 工具函数 -------------------------
# get_epochs() {
#     local category=$1
#     echo ${EPOCHS_CONFIG[$category]:-${EPOCHS_CONFIG["default"]}}
# }

# get_lr() {
#     local category=$1
#     echo ${LR_CONFIG[$category]:-${LR_CONFIG["default"]}}
# }

# get_eval_epoch() {
#     local category=$1
#     echo ${EVAL_EPOCH_CONFIG[$category]:-${EVAL_EPOCH_CONFIG["default"]}}
# }

# log_experiment_info() {
#     local category=$1
#     local epochs=$2
#     local lr=$3
#     local eval_epoch=$4

#     echo "=========================================="
#     echo "开始训练类别: $category"
#     echo "实验模式: $EXPERIMENT_MODE"
#     echo "增强配置: $ENABLE_ENHANCEMENT_FLAGS"
#     echo "训练轮数: $epochs"
#     echo "学习率: $lr"
#     echo "评估间隔: $eval_epoch"
#     echo "GPU设备: $GPU_DEVICES"
#     echo "随机种子: $SEED"
#     echo "=========================================="
# }

# run_training() {
#     local category=$1
#     local epochs=$(get_epochs $category)
#     local lr=$(get_lr $category)
#     local eval_epoch=$(get_eval_epoch $category)

#     log_experiment_info $category $epochs $lr $eval_epoch

#     # 构建完整的命令
#     local cmd="CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py"
#     cmd="$cmd --dataset $DATASET"
#     cmd="$cmd --batch_size $BATCH_SIZE"
#     cmd="$cmd --lr $lr"
#     cmd="$cmd --d_lr $D_LR"
#     cmd="$cmd --adv_conf $ADV_CONF"
#     cmd="$cmd --epochs $epochs"
#     cmd="$cmd --normal $category"
#     cmd="$cmd --seed $SEED"
#     cmd="$cmd --img_size $IMG_SIZE"
#     cmd="$cmd --labeled_anomaly_class_num $LABELED_ANOMALY_CLASS_NUM"
#     cmd="$cmd --labeled_anomaly_class $LABELED_ANOMALY_CLASS"
#     cmd="$cmd --labeled_anomaly_ratio $LABELED_ANOMALY_RATIO"
#     cmd="$cmd --log_dir $LOG_DIR"
#     cmd="$cmd --model $MODEL"
#     cmd="$cmd --eval_epoch $eval_epoch"
#     cmd="$cmd --layer 1"  # 选择第一层特征

#     # 添加渐进式实验配置（如果需要的话）
#     if [ "$EXPERIMENT_MODE" != "baseline" ]; then
#         # 将布尔数组转换为命令行参数格式
#         # 例如：("False" "True" "False") -> --enable_enhancement False True False
#         enhancement_args=""
#         for flag in "${ENABLE_ENHANCEMENT_FLAGS[@]}"; do
#             enhancement_args="$enhancement_args $flag"
#         done
#         cmd="$cmd --enable_enhancement$enhancement_args"
#     fi

#     # 添加性能优化选项
#     cmd="$cmd --use_amp"  # 启用混合精度训练
#     # cmd="$cmd --compile"  # 如需要可启用torch.compile

#     echo "执行命令: $cmd"
#     echo ""

#     # 执行训练
#     eval $cmd

#     local exit_code=$?
#     if [ $exit_code -eq 0 ]; then
#         echo "✅ 类别 $category 训练完成"
#     else
#         echo "❌ 类别 $category 训练失败 (退出码: $exit_code)"
#         return $exit_code
#     fi

#     echo ""
# }

# # ------------------------- 主训练循环 -------------------------
# echo "🚀 开始 CKAAD 训练实验"
# echo "实验模式: $EXPERIMENT_MODE"
# echo "待训练类别数量: ${#CATEGORIES[@]}"
# echo "GPU设备: $GPU_DEVICES"
# echo "=========================================="
# echo ""

# total_categories=${#CATEGORIES[@]}
# completed=0
# failed=0

# for category in "${CATEGORIES[@]}"; do
#     echo "进度: [$((completed+1))/$total_categories] 开始训练 $category"

#     if run_training "$category"; then
#         ((completed++))
#         echo "进度更新: ✅ $completed/$total_categories 成功, ❌ $failed/$total_categories 失败"
#     else
#         ((failed++))
#         echo "进度更新: ✅ $completed/$total_categories 成功, ❌ $failed/$total_categories 失败"
#         echo "⚠️  继续训练其他类别..."
#     fi

#     echo "=========================================="
# done

# # ------------------------- 训练总结 -------------------------
# echo ""
# echo "🎉 训练实验完成！"
# echo "总类别数: $total_categories"
# echo "成功完成: $completed"
# echo "训练失败: $failed"

# if [ $failed -eq 0 ]; then
#     echo "✅ 所有训练任务均成功完成！"
# else
#     echo "⚠️  有 $failed 个训练任务失败，请检查日志。"
# fi

# echo ""
# echo "📊 实验配置摘要:"
# echo "  - 实验模式: $EXPERIMENT_MODE"
# echo "  - 增强配置: $ENABLE_ENHANCEMENT_FLAGS"
# echo "  - 模型: $MODEL"
# echo "  - 图像尺寸: ${IMG_SIZE}x${IMG_SIZE}"
# echo "  - 批大小: $BATCH_SIZE"
# echo "  - 日志目录: $LOG_DIR"

# # 可选：生成训练报告
# echo ""
# echo "💡 提示:"
# echo "  - 查看训练日志: $LOG_DIR"
# echo "  - 查看损失曲线: ./pic/"
# echo "  - 如需修改实验配置，请编辑脚本开头的变量"


#!/bin/bash
labeled_anomaly_ratio=0.05
labeled_anomaly_class_num=1
# for normal in 'bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
# for normal in 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
for normal in 'cable'
do
    echo $normal

    case $normal in 'carpet')
            epochs=10
            eval_epoch=1
            ;;
        *)  
            epochs=2
            eval_epoch=2
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
    --normal $normal --seed 111 --img_size 128 \
    --labeled_anomaly_class_num ${labeled_anomaly_class_num} \
    --labeled_anomaly_class 0 \
    --labeled_anomaly_ratio ${labeled_anomaly_ratio} \
    --log_dir ./log --model wide_resnet50_2 --eval_epoch ${eval_epoch} --layer 2 \
    --enable_enhancement False True False --use_amp
done