#!/bin/bash
# ==============================================================
# analyze_maps.sh  — 能量图 & 重建误差后处理批量运行脚本
# ==============================================================
#
# 用法示例：
#   bash analyze_maps.sh                          # 使用默认参数
#   bash analyze_maps.sh --sample_dir xxx --idx 20 --show
#   bash analyze_maps.sh --batch --rule percentile_threshold --params "target_map=energy_fused,quantile=0.60"
#
# --------------------------------------------------------------
# 参数说明（可通过命令行或下方默认变量覆盖）：
#
#   --sample_dir      样本目录路径
#   --idx             单个样本编号（与 --batch 互斥）
#   --batch           批量模式，遍历整个目录
#   --rule            应用的规则（可多个，空格分隔）
#                     clip / fuse / gaussian / bilateral /
#                     morphological / adaptive_threshold /
#                     cross_suppress / edge_suppress /
#                     energy_norm / percentile_threshold
#   --params          规则参数，格式：key1=val1,key2=val2
#   --target_map      percentile_threshold 等规则的目标图像
#   --quantile        分位阈值（0~1）
#   --pipeline        是否应用完整流水线（--apply_pipeline）
#   --output_dir      输出目录
#   --show            是否实时显示图片
#   --save            是否保存图片
#   --stats           是否打印统计信息
#   --overlay         是否显示叠加对比图
#   --compare_idx     对比两个样本（如 "20 30"）
#
# ==============================================================

set -e  # 遇错即停
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --------------------------------------------------------------
# 默认参数（可根据需求修改）
# --------------------------------------------------------------
DEFAULT_SAMPLE_DIR="testSet/bottle/n_bottle_a_0_s_111"
DEFAULT_OUTPUT_DIR="output/maps_analysis"
DEFAULT_IDX=""
DEFAULT_BATCH="false"
DEFAULT_RULES=""
DEFAULT_PARAMS=""
DEFAULT_TARGET_MAP="recon_error"
DEFAULT_QUANTILE="0.60"
DEFAULT_PIPELINE="false"
DEFAULT_SHOW="false"
DEFAULT_SAVE="false"
DEFAULT_STATS="false"
DEFAULT_OVERLAY="false"

# --------------------------------------------------------------
# 辅助函数
# --------------------------------------------------------------

log_info() {
    echo -e "\033[34m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[33m[WARN]\033[0m $1"
}

log_success() {
    echo -e "\033[32m[OK]\033[0m $1"
}

log_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

show_help() {
    grep "^# " "$0" | sed 's/^# //'
}

# 检查依赖
check_deps() {
    log_info "检查 Python 环境..."
    if ! command -v python &> /dev/null; then
        log_error "未找到 python，请先安装 Python 环境"
        exit 1
    fi

    log_info "检查 analyze_maps.py 是否存在..."
    SCRIPT_PATH="${PROJECT_ROOT}/analyze_maps.py"
    if [[ ! -f "${SCRIPT_PATH}" ]]; then
        log_error "未找到 analyze_maps.py，路径：${SCRIPT_PATH}"
        exit 1
    fi
    log_success "依赖检查通过"
}

# --------------------------------------------------------------
# 解析命令行参数
# --------------------------------------------------------------

show_usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --sample_dir DIR      样本目录（默认: ${DEFAULT_SAMPLE_DIR}）"
    echo "  --idx NUM             单个样本编号（0起始）"
    echo "  --batch               批量模式（遍历目录中所有样本）"
    echo "  --rule RULES          应用规则（可多个，空格分隔）"
    echo "  --params PARAMS       规则参数，格式：key1=val1,key2=val2"
    echo "  --target_map MAP      目标图像类型（默认: ${DEFAULT_TARGET_MAP}）"
    echo "                        可选: energy_layer1 / energy_layer2 /"
    echo "                              energy_layer3 / recon_error /"
    echo "                              energy_fused"
    echo "  --quantile Q          分位阈值 0~1（默认: ${DEFAULT_QUANTILE}）"
    echo "  --pipeline            应用完整推荐流水线"
    echo "  --output_dir DIR      输出目录（默认: ${DEFAULT_OUTPUT_DIR}）"
    echo "  --show                实时显示图片"
    echo "  --save                保存图片到 output_dir"
    echo "  --stats               打印统计信息"
    echo "  --overlay             显示叠加对比图"
    echo "  --compare_idx A B     对比两个样本（如: 20 30）"
    echo "  --help                显示本帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --idx 20 --show"
    echo "  $0 --batch --rule percentile_threshold --params target_map=energy_fused,quantile=0.60 --save"
    echo "  $0 --pipeline --save --output_dir output/my_result"
    echo "  $0 --compare_idx 20 30 --show"
}

# 解析参数
SAMPLE_DIR="${DEFAULT_SAMPLE_DIR}"
IDX="${DEFAULT_IDX}"
BATCH="${DEFAULT_BATCH}"
RULES="${DEFAULT_RULES}"
PARAMS="${DEFAULT_PARAMS}"
TARGET_MAP="${DEFAULT_TARGET_MAP}"
QUANTILE="${DEFAULT_QUANTILE}"
USE_PIPELINE="${DEFAULT_PIPELINE}"
SHOW="${DEFAULT_SHOW}"
SAVE="${DEFAULT_SAVE}"
STATS="${DEFAULT_STATS}"
OVERLAY="${DEFAULT_OVERLAY}"
COMPARE_IDX=""
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sample_dir)
            SAMPLE_DIR="$2"
            shift 2
            ;;
        --idx)
            IDX="$2"
            shift 2
            ;;
        --batch)
            BATCH="true"
            shift
            ;;
        --rule)
            RULES="$2"
            shift 2
            ;;
        --params)
            PARAMS="$2"
            shift 2
            ;;
        --target_map)
            TARGET_MAP="$2"
            shift 2
            ;;
        --quantile)
            QUANTILE="$2"
            shift 2
            ;;
        --pipeline)
            USE_PIPELINE="true"
            shift
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --show)
            SHOW="true"
            shift
            ;;
        --save)
            SAVE="true"
            shift
            ;;
        --stats)
            STATS="true"
            shift
            ;;
        --overlay)
            OVERLAY="true"
            shift
            ;;
        --compare_idx)
            COMPARE_IDX="$2 $3"
            shift 3
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 自动将 target_map 和 quantile 合并到 PARAMS（如果使用了 percentile_threshold 规则）
if [[ "${RULES}" == *"percentile_threshold"* ]]; then
    if [[ -n "${PARAMS}" ]]; then
        PARAMS="${PARAMS},target_map=${TARGET_MAP},quantile=${QUANTILE}"
    else
        PARAMS="target_map=${TARGET_MAP},quantile=${QUANTILE}"
    fi
fi

# --------------------------------------------------------------
# 构建 python 命令
# --------------------------------------------------------------

build_python_cmd() {
    local idx_arg=""
    local batch_arg=""
    local rule_arg=""
    local params_arg=""
    local pipeline_arg=""
    local output_arg=""
    local show_arg=""
    local save_arg=""
    local stats_arg=""
    local overlay_arg=""
    local compare_arg=""

    # 样本目录
    [[ -n "${SAMPLE_DIR}" ]] && sample_dir_arg="--sample_dir ${SAMPLE_DIR}"

    # 编号
    if [[ -n "${IDX}" ]]; then
        idx_arg="--sample_idx ${IDX}"
    elif [[ -n "${COMPARE_IDX}" ]]; then
        compare_arg="--compare_idx ${COMPARE_IDX}"
    fi

    # 批量模式
    [[ "${BATCH}" == "true" ]] && batch_arg="# batch mode: no --sample_idx, process all"

    # 规则
    if [[ -n "${RULES}" ]]; then
        rule_arg="--rule ${RULES}"
    fi

    # 参数
    [[ -n "${PARAMS}" ]] && params_arg="--params ${PARAMS}"

    # 流水线
    [[ "${USE_PIPELINE}" == "true" ]] && pipeline_arg="--apply_pipeline"

    # 输出目录
    [[ -n "${OUTPUT_DIR}" ]] && output_arg="--output_dir ${OUTPUT_DIR}"

    # 可视化选项
    [[ "${SHOW}" == "true" ]] && show_arg="--show"
    [[ "${SAVE}" == "true" ]] && save_arg="--save"
    [[ "${STATS}" == "true" ]] && stats_arg="--stats"
    [[ "${OVERLAY}" == "true" ]] && overlay_arg="--overlay"

    echo "${sample_dir_arg} ${idx_arg} ${compare_arg} ${rule_arg} ${params_arg} ${pipeline_arg} ${output_arg} ${show_arg} ${save_arg} ${stats_arg} ${overlay_arg}"
}

# --------------------------------------------------------------
# 主执行流程
# --------------------------------------------------------------

main() {
    echo ""
    echo "============================================================"
    echo "  analyze_maps.sh  —  批量后处理脚本"
    echo "============================================================"
    echo ""
    log_info "项目根目录: ${PROJECT_ROOT}"
    log_info "样本目录:   ${SAMPLE_DIR}"
    log_info "输出目录:   ${OUTPUT_DIR}"

    if [[ -n "${IDX}" ]]; then
        log_info "处理模式:   单样本 #${IDX}"
    elif [[ -n "${COMPARE_IDX}" ]]; then
        log_info "处理模式:   对比样本 ${COMPARE_IDX}"
    elif [[ "${BATCH}" == "true" ]]; then
        log_info "处理模式:   批量（遍历所有样本）"
    else
        log_info "处理模式:   默认前3个样本"
    fi

    if [[ -n "${RULES}" ]]; then
        log_info "应用规则:   ${RULES}"
    elif [[ "${USE_PIPELINE}" == "true" ]]; then
        log_info "应用规则:   [完整流水线] 平滑→裁剪→融合→交叉抑制→边缘感知"
    fi

    if [[ -n "${PARAMS}" ]]; then
        log_info "规则参数:   ${PARAMS}"
    fi

    echo ""

    # 检查依赖
    check_deps

    # 构建完整命令
    PYTHON_CMD="python ${PROJECT_ROOT}/analyze_maps.py $(build_python_cmd)"

    log_info "执行命令:"
    echo "  ${PYTHON_CMD}"
    echo ""

    # 执行
    eval ${PYTHON_CMD}

    log_success "处理完成！"
    echo ""
}

main
