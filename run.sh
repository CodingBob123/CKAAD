#!/bin/bash
# =============================================================
# 用法：
#   ./run.sh <idx> <map1> [map2...] [--quantile N] [--save]
#
# 参数：
#   <idx>         样本编号或范围，如 20 或 20-30
#   <mapN>        recon_error energy_layer1 energy_layer2 energy_layer3
#   --quantile N  阈值分位值，默认 0.6
#   --save        保存图片，不加则弹窗
#
# 示例：
#   ./run.sh 20 recon_error
#   ./run.sh 20 recon_error energy_layer1 energy_layer2
#   ./run.sh 30 energy_layer2 energy_layer3 0.5
#   ./run.sh 20-30 recon_error --quantile 0.7
#   ./run.sh 20-30 recon_error energy_layer1 --save
#   ./run.sh 20-30 recon_error energy_layer1 energy_layer2 --quantile 0.5 --save
# =============================================================

SAMPLE_DIR="testSet/bottle/n_bottle_a_0_s_111"

show_help() {
    echo ""
    echo "用法: $0 <idx> <map1> [map2...] [--quantile N] [--save]"
    echo ""
    echo "  idx         样本编号或范围，如 20 或 20-30"
    echo "  mapN        recon_error energy_layer1 energy_layer2 energy_layer3"
    echo "  --quantile  阈值分位值，默认 0.6"
    echo "  --save      保存图片，不加则弹窗"
    echo ""
    echo "示例:"
    echo "  $0 20 recon_error"
    echo "  $0 20 recon_error energy_layer1 energy_layer2"
    echo "  $0 30 energy_layer2 energy_layer3 0.5"
    echo "  $0 20-30 recon_error --quantile 0.7"
    echo "  $0 20-30 recon_error energy_layer1 energy_layer2 --save"
    echo ""
}

# ---- 参数解析 ----
IDX=""
MAPS=()
QUANTILE="0.6"
SAVE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help; exit 0 ;;
        --save)
            SAVE="yes"; shift ;;
        --quantile)
            QUANTILE="$2"; shift 2 ;;
        -*)
            echo "[错误] 未知参数: $1"; show_help; exit 1 ;;
        *)
            if [[ -z "$IDX" ]]; then
                IDX="$1"
            else
                MAPS+=("$1")
            fi
            shift ;;
    esac
done

# ---- 校验参数 ----
if [[ -z "$IDX" || ${#MAPS[@]} -eq 0 ]]; then
    show_help; exit 1
fi

# 如果最后一个 map 参数其实是数字，说明用户用了 ./run.sh 30 e2 e3 0.5 的简写形式
LAST="${MAPS[-1]}"
if [[ "$LAST" =~ ^[0-9]*\.?[0-9]+$ ]] && [[ ${#MAPS[@]} -gt 1 ]]; then
    QUANTILE="$LAST"
    MAPS=("${MAPS[@]:0:${#MAPS[@]}-1}")
fi

# ---- 解析样本范围 ----
if [[ "$IDX" == *"-"* ]]; then
    START=$(echo "$IDX" | cut -d'-' -f1)
    END=$(echo "$IDX" | cut -d'-' -f2)
else
    START="$IDX"; END="$IDX"
fi

# ---- 生成 --rule 和 --params ----
RULE_ARGS=""
PARAMS_ARGS=""
for m in "${MAPS[@]}"; do
    RULE_ARGS="$RULE_ARGS percentile_threshold"
    PARAMS_ARGS="$PARAMS_ARGS --params target_map=$m,quantile=$QUANTILE"
done

# ---- 生成规则参数 ----
RULE_ARGS=""
PARAMS_ARGS=""
for m in "${MAPS[@]}"; do
    RULE_ARGS="$RULE_ARGS percentile_threshold"
    PARAMS_ARGS="$PARAMS_ARGS --params target_map=$m,quantile=$QUANTILE"
done

# ---- 执行 ----
run_one() {
    local i="$1"
    local PY="python analyze_maps.py --sample_dir $SAMPLE_DIR --sample_idx $i --rule $RULE_ARGS $PARAMS_ARGS"
    if [[ -n "$SAVE" ]]; then
        echo "[保存] idx=$i  maps=${MAPS[*]}  quantile=$QUANTILE"
        $PY --save --output_dir "output/maps_analysis"
    else
        echo "[显示] idx=$i  maps=${MAPS[*]}  quantile=$QUANTILE"
        $PY --show
    fi
}

if [[ "$START" != "$END" ]]; then
    echo "[批量] 范围: $START ~ $END  共 $((END - START + 1)) 个"
    for i in $(seq "$START" "$END"); do
        run_one "$i"
    done
else
    run_one "$IDX"
fi
