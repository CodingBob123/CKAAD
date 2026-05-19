#!/bin/bash

set -e

# =====================================================================
# 实验参数
# =====================================================================
SARRNET_GPU=${SARRNET_GPU:-0}
SARRNET_SEED=${SARRNET_SEED:-111}
SARRNET_EPOCHS=${SARRNET_EPOCHS:-200}
SARRNET_EVAL_EPOCH=${SARRNET_EVAL_EPOCH:-8}
SARRNET_IMG_SIZE=${SARRNET_IMG_SIZE:-256}
SARRNET_BATCH_SIZE=${SARRNET_BATCH_SIZE:-16}
SARRNET_MODEL=${SARRNET_MODEL:-wide_resnet50_2}
SARRNET_LAYERS=${SARRNET_LAYERS:-"1 2 3"}
SARRNET_CKPT_DIR=${SARRNET_CKPT_DIR:-/hy-tmp/checkpoints/}

# 实验状态目录（供 run_experiment 使用）
EXPERIMENT_STATUS_DIR="${EXPERIMENT_STATUS_DIR:-./.experiment_status}"

sarrnet_category_extra_args() {
    local category="$1"
    local extra_args=""
    case ${category} in
        cable|toothbrush|transistor|wood)
            extra_args="${extra_args} --enable_enhancement"
            ;;
    esac
    echo "${extra_args}"
}

sarrnet_category_lr() {
    local category="$1"
    if [ "${category}" = "transistor" ]; then
        echo "1e-03"
    else
        echo "5e-03"
    fi
}

sarrnet_base_cmd() {
    local dataset="$1"
    local category="$2"
    local exp_name="$3"
    local lr="$4"

    echo "python ../main.py --dataset ${dataset} --batch_size ${SARRNET_BATCH_SIZE} --lr ${lr} --d_lr 1e-04 --adv_conf 0.02 --epochs ${SARRNET_EPOCHS} --normal ${category} --seed ${SARRNET_SEED} --img_size ${SARRNET_IMG_SIZE} --labeled_anomaly_class_num 1 --labeled_anomaly_class 0 --labeled_anomaly_ratio 0 --log_dir ./log --exp_name ${exp_name} --ckpt_dir ${SARRNET_CKPT_DIR} --model ${SARRNET_MODEL} --eval_epoch ${SARRNET_EVAL_EPOCH} --layer ${SARRNET_LAYERS} --recon_loss_type combined --loss_alpha 1.0 --loss_beta 0 --loss_gamma 0 --save_best"
}

# =====================================================================
# run_experiment — 错误恢复包装器
# =====================================================================
# 功能：
#   1. 子实验失败不会终止整个脚本（暂时关闭 set -e）
#   2. 创建 done / fail 标记文件，支持断点续跑
#   3. 执行完毕后自动清理 GPU 缓存
#   4. 通过 print_experiment_summary 输出汇总
#
# 用法（调用者需在 for 循环中逐条调用）：
#   run_experiment <exp_group> <variant> <dataset> <category> \
#       "CUDA_VISIBLE_DEVICES=0 python ... <全部参数>"
#
# 断点续跑：
#   - 标记文件存放在 ${EXPERIMENT_STATUS_DIR}（默认 ./.experiment_status）
#   - 清除单个标记：rm -f .experiment_status/done_<name>
#   - 清除全部：rm -rf .experiment_status
# =====================================================================

run_experiment() {
    local exp_group="$1"     # 实验组名，如 "table8_cres"
    local variant="$2"       # 变体名，如 "recon_map"
    local dataset="$3"       # 数据集，如 "mvtec"
    local category="$4"      # 类别，如 "bottle"
    shift 4
    local full_cmd="$*"      # 完整的可执行命令字符串

    mkdir -p "${EXPERIMENT_STATUS_DIR}"

    local safe_name="${dataset}_${category}_${exp_group}_${variant}"
    local done_marker="${EXPERIMENT_STATUS_DIR}/done_${safe_name}"
    local fail_marker="${EXPERIMENT_STATUS_DIR}/fail_${safe_name}"

    # --- 跳过已完成实验 ---
    if [ -f "${done_marker}" ]; then
        echo "[SKIP] ${exp_group}/${variant}: ${dataset}/${category}（已完成）"
        return 0
    fi

    # --- 清除旧的失败标记（如果有） ---
    rm -f "${fail_marker}" 2>/dev/null

    echo ""
    echo "================================================================================"
    echo "  RUN: ${exp_group}/${variant}: ${dataset}/${category}"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"

    # --- 暂时关闭 set -e，使子实验失败不会导致脚本退出 ---
    set +e
    (
        set -e
        eval "${full_cmd}"
    )
    local exit_code=$?
    set -e

    # --- 记录结果 ---
    if [ ${exit_code} -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: ${exp_group}/${variant}: ${dataset}/${category}"
        touch "${done_marker}"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAIL (exit=${exit_code}):"
        echo "        ${exp_group}/${variant}: ${dataset}/${category}"
        echo "        exit_code=${exit_code}" > "${fail_marker}"
    fi

    # --- GPU 内存清理 ---
    python -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
" 2>/dev/null || true
    # 短暂等待，确保显存释放
    sleep 2

    return ${exit_code}
}

# =====================================================================
# print_experiment_summary — 实验汇总报告
# =====================================================================
# 在所有实验循环结束后调用，输出通过/失败统计
# =====================================================================

print_experiment_summary() {
    local passed=0 failed=0 total=0
    if [ -d "${EXPERIMENT_STATUS_DIR}" ]; then
        passed=$(find "${EXPERIMENT_STATUS_DIR}" -maxdepth 1 -name 'done_*' 2>/dev/null | wc -l)
        failed=$(find "${EXPERIMENT_STATUS_DIR}" -maxdepth 1 -name 'fail_*' 2>/dev/null | wc -l)
        total=$((passed + failed))
    fi

    echo ""
    echo "========================================"
    echo "  Experiment Summary"
    echo "  Status dir: ${EXPERIMENT_STATUS_DIR}/"
    echo "----------------------------------------"
    printf "  Total:   %3d\n" "${total}"
    printf "  Passed:  %3d   \342\234\205\n" "${passed}"    # ✅
    printf "  Failed:  %3d   \342\235\214\n" "${failed}"    # ❌
    echo "----------------------------------------"
    if [ ${failed} -gt 0 ]; then
        echo "  Failed experiments:"
        find "${EXPERIMENT_STATUS_DIR}" -maxdepth 1 -name 'fail_*' \
            -printf '    - %f\n' | sed 's/^    - fail_/    - /'
        echo ""
        echo "  Re-run failed:"
        echo "    rm -f ${EXPERIMENT_STATUS_DIR}/fail_*"
        echo "    # 然后重新运行脚本即可（已通过的会自动跳过）"
    fi
    echo "  Re-run ALL:  rm -rf ${EXPERIMENT_STATUS_DIR}"
    echo "========================================"
}
