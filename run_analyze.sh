#!/bin/bash
# =============================================================
# analyze_maps.py 快速运行脚本
# =============================================================
# 单样本：./run_analyze.sh [样本编号] [命令]
# 批量：  ./run_analyze.sh batch [起始] [终止] [命令]
#
# 示例：
#   ./run_analyze.sh 20 show              # 查看样本 20 原始 map
#   ./run_analyze.sh 20 thresh_all_save    # 样本 20 四图阈值，保存
#   ./run_analyze.sh 0 100 batch_save       # 批量 0~99，阈值保存
#   ./run_analyze.sh 20 help               # 查看帮助
# =============================================================

SAMPLE_DIR="testSet/bottle/n_bottle_a_0_s_111"

# ---- 单独样本调用 ----
RUN() {
    python analyze_maps.py --sample_dir "$SAMPLE_DIR" --sample_idx "$1" "${@:2}"
}

# ---- 批量调用（不传 sample_idx，改用 --range） ----
# 用法: RUN_BATCH start stop [args...]
RUN_BATCH() {
    local start="$1"; local stop="$2"; shift 2
    python analyze_maps.py --sample_dir "$SAMPLE_DIR" --range "$start" "$stop" "$@"
}

# ---- 帮助信息 ----
show_help() {
    echo ""
    echo "  用法:"
    echo "    $0 [样本编号] [命令]           # 单样本"
    echo "    $0 [起始] [终止] batch [命令]  # 批量（不含终止）"
    echo ""
    echo "  样本编号默认: 20"
    echo ""
    echo "  单样本命令:"
    echo "    show              - 仅显示原始 map"
    echo "    stats             - 显示统计信息"
    echo "    all               - 应用全套流水线"
    echo ""
    echo "    recon60           - 重建误差 值域 60% 阈值（显示）"
    echo "    recon70           - 重建误差 值域 70% 阈值（显示）"
    echo "    e1_60             - energy_layer1 值域 60% 阈值（显示）"
    echo "    e2_60             - energy_layer2 值域 60% 阈值（显示）"
    echo "    e3_60             - energy_layer3 值域 60% 阈值（显示）"
    echo ""
    echo "    thresh_all        - 四张图同一规则 60% 阈值（显示）"
    echo "    thresh_all_save   - 同上，保存模式"
    echo ""
    echo "    thresh [map] [q]      - 自定义阈值（显示）"
    echo "    thresh_save [map] [q] - 自定义阈值（保存）"
    echo ""
    echo "  批量命令（不写样本编号，写起始/终止）:"
    echo "    batch              - 批量 0~100，60% 阈值，显示"
    echo "    batch_save         - 批量 0~100，60% 阈值，保存"
    echo ""
    echo "  示例:"
    echo "    $0 20 show"
    echo "    $0 20 thresh_all_save"
    echo "    $0 0 200 batch_save"
    echo "    $0 50 80 thresh_all_save"
    echo ""
}

# ============================================================
# 主逻辑
# ============================================================

# 第一个参数是否为数字（用于判断是单样本还是批量）
is_number() { [[ "$1" =~ ^-?[0-9]+$ ]]; }

if is_number "$1" && is_number "$2"; then
    # -------- 批量模式：./run_analyze.sh start stop [cmd] --------
    START="$1"; STOP="$2"; CMD="${3:-}"; shift 3
    echo "[批量模式] 编号 $START ~ $((STOP-1))"

    case "$CMD" in
      ""|batch)
        echo "[批量] 60% 阈值，显示模式"
        RUN_BATCH "$START" "$STOP" --rule percentile_threshold --params "target_map=all,quantile=0.6" --show
        ;;
      batch_save)
        echo "[批量] 60% 阈值，保存模式"
        RUN_BATCH "$START" "$STOP" --rule percentile_threshold --params "target_map=all,quantile=0.6" --save --output_dir "output/maps_analysis"
        ;;
      thresh_all)
        echo "[批量] thresh_all，显示模式"
        RUN_BATCH "$START" "$STOP" --rule percentile_threshold --params "target_map=all,quantile=0.6" --show
        ;;
      thresh_all_save)
        echo "[批量] thresh_all_save，保存模式"
        RUN_BATCH "$START" "$STOP" --rule percentile_threshold --params "target_map=all,quantile=0.6" --save --output_dir "output/maps_analysis"
        ;;
      all)
        echo "[批量] 全套流水线，显示模式"
        RUN_BATCH "$START" "$STOP" --apply_pipeline --show
        ;;
      *)
        echo "[提示] 未知批量命令: $CMD"
        echo "输入 '$0 help' 查看"
        ;;
    esac

elif is_number "$1"; then
    # -------- 单样本模式：./run_analyze.sh [idx] [cmd] --------
    SAMPLE_IDX="$1"; CMD="${2:-}"

    case "$CMD" in
      ""|show)
        echo "[显示] 原始 map (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show
        ;;
      help|-h|--help)
        show_help
        ;;
      all)
        echo "[流水线] 平滑 → 裁剪 → 融合 → 交叉抑制 → 边缘感知 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --apply_pipeline --show
        ;;
      stats)
        echo "[统计] (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --stats --show
        ;;
      # ---------- 单独阈值（显示） ----------
      recon60)
        echo "[规则] recon_error 60% 阈值 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=recon_error,quantile=0.6"
        ;;
      recon70)
        echo "[规则] recon_error 70% 阈值 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=recon_error,quantile=0.7"
        ;;
      e1_60)
        echo "[规则] energy_layer1 60% 阈值 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=energy_layer1,quantile=0.6"
        ;;
      e2_60)
        echo "[规则] energy_layer2 60% 阈值 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=energy_layer2,quantile=0.6"
        ;;
      e3_60)
        echo "[规则] energy_layer3 60% 阈值 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=energy_layer3,quantile=0.6"
        ;;
      # ---------- 单独阈值（保存） ----------
      recon60_save)
        echo "[规则] recon_error 60% 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=recon_error,quantile=0.6"
        ;;
      recon70_save)
        echo "[规则] recon_error 70% 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=recon_error,quantile=0.7"
        ;;
      e1_60_save)
        echo "[规则] energy_layer1 60% 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=energy_layer1,quantile=0.6"
        ;;
      e2_60_save)
        echo "[规则] energy_layer2 60% 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=energy_layer2,quantile=0.6"
        ;;
      e3_60_save)
        echo "[规则] energy_layer3 60% 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=energy_layer3,quantile=0.6"
        ;;
      # ---------- 自定义阈值 ----------
      thresh)
        MAP="${3:-recon_error}"; Q="${4:-0.6}"
        echo "[规则] $MAP ${Q} 阈值（显示）(样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=$MAP,quantile=$Q"
        ;;
      thresh_save)
        MAP="${3:-recon_error}"; Q="${4:-0.6}"
        echo "[规则] $MAP ${Q} 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=$MAP,quantile=$Q"
        ;;
      # ---------- 四图同一规则 ----------
      thresh_all)
        echo "[规则] target_map=all，四张图 60% 阈值（显示）(样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --show --rule percentile_threshold --params "target_map=all,quantile=0.6"
        ;;
      thresh_all_save)
        echo "[规则] target_map=all，四张图 60% 阈值 → 保存 (样本 #$SAMPLE_IDX)"
        RUN "$SAMPLE_IDX" --save --output_dir "output/maps_analysis" --rule percentile_threshold --params "target_map=all,quantile=0.6"
        ;;
      *)
        echo "[提示] 未知命令: $CMD"
        echo "输入 '$0 help' 查看"
        ;;
    esac

else
    show_help
fi
