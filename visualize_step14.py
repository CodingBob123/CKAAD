"""
======================================================================
Rule 13 + Rule 14 流水线可视化脚本
======================================================================
功能：
  1. 加载 MapDataset 数据（能量图层 + 重构误差）
  2. 依次执行 Rule 13（morph_energy_reliability_fusion）和 Rule 14（reliability_calibrate）
  3. 以热力图形式展示原始 map 与处理结果的对比

热力图排版（4行 N+2 列）：
  行0：原始 recon_error + 各 energy_layer（每列独立 colorbar）
  行1：各层 *_morph_mask（Rule 5 产物）+ 可靠性 q 标签
  行2：Rule 13 融合结果 energy_fused_reliable + 融合 mask energy_reliability_mask
  行3：Rule 14 校准结果 recon_error_calibrated + 校准增益图 + GT（如果有）

用法示例：
  python visualize_step14.py --sample_idx 20 --save
  python visualize_step14.py --sample_idx 20 --show --sample_dir testSet/bottle/n_bottle_a_0_s_111
  python visualize_step14.py --sample_idx 20 --save --output_dir output/step14_vis
  python visualize_step14.py --range 0 10 --save --output_dir output/batch_step14
======================================================================
"""

import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ==============================================================
#                        数据加载器
# ==============================================================

class MapDataset:
    """从样本目录中加载所有 map 文件（npy 格式）。"""

    MAP_TYPES = ["energy_layer1", "energy_layer2", "energy_layer3", "recon_error"]

    def __init__(self, sample_dir: str):
        self.sample_dir = Path(sample_dir)
        if not self.sample_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.sample_dir}")
        self._index_map: Dict[int, Dict[str, Path]] = {}
        self._metadata: Optional[dict] = None
        self._scan_files()

    def _scan_files(self):
        for map_type in self.MAP_TYPES:
            map_dir = self.sample_dir / map_type
            if not map_dir.exists():
                continue
            for f in map_dir.glob(f"{map_type}_*.npy"):
                idx = self._parse_index(f.name)
                if idx is not None:
                    self._index_map.setdefault(idx, {})[map_type] = f
        meta_path = self.sample_dir / "metadata.npy"
        if meta_path.exists():
            self._metadata = np.load(meta_path, allow_pickle=True).item()

    @staticmethod
    def _parse_index(filename: str) -> Optional[int]:
        m = re.search(r"_(\d+)\.npy$", filename)
        return int(m.group(1)) if m else None

    @property
    def indices(self) -> List[int]:
        return sorted(self._index_map.keys())

    @property
    def num_samples(self) -> int:
        return len(self._index_map)

    def get_all_maps(self, idx: int) -> Dict[str, np.ndarray]:
        """加载编号 idx 对应的所有原始 map（能量图 + 重构误差）。"""
        maps: Dict[str, np.ndarray] = {}
        if idx not in self._index_map:
            return maps
        for mtype, fpath in self._index_map[idx].items():
            arr = np.load(fpath)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr.squeeze(0)
            maps[mtype] = arr.astype(np.float32)
        return maps

    def get_gt(self, idx: int) -> Optional[np.ndarray]:
        gt_dir = self.sample_dir / "ground_truth"
        if not gt_dir.exists():
            return None
        for ext in ("png", "jpg", "npy"):
            for f in gt_dir.glob(f"*_{idx:04d}.{ext}"):
                gt = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if gt is not None:
                    return (gt > 127).astype(np.uint8)
        return None

    def get_label(self, idx: int) -> str:
        if self._metadata and "labels" in self._metadata:
            labels = self._metadata["labels"]
            if 0 <= idx < len(labels):
                return str(labels[idx])
        return "unknown"


# ==============================================================
#                     规则13：形态学可靠性融合
# ==============================================================

def morph_energy_reliability_fusion(
    maps: Dict[str, np.ndarray],
    energy_layers: tuple = ("energy_layer1", "energy_layer2", "energy_layer3"),
    area_min: int = 50,
    small_area_threshold: int = 50,
    small_ratio_max: float = 0.6,
    total_area_min_ratio: float = 0.001,
    total_area_max_ratio: float = 0.35,
    connectivity: int = 8,
    remove_small_components: bool = True,
    layer_params: Dict[str, Dict] = None,
) -> Dict[str, np.ndarray]:
    """规则13 = Step2：morph_mask 连通域分析 → 四重可靠性判据 → 融合可靠能量图。"""
    result = {k: v.copy() for k, v in maps.items()}
    reliability: Dict[str, Dict] = {}
    H, W = None, None

    lp = layer_params or {}
    for layer in energy_layers:
        per = lp.get(layer, {})
        p_area_min = int(per.get("area_min", area_min))
        p_small_area_threshold = int(per.get("small_area_threshold", small_area_threshold))
        p_small_ratio_max = float(per.get("small_ratio_max", small_ratio_max))
        p_total_area_min_ratio = float(per.get("total_area_min_ratio", total_area_min_ratio))
        p_total_area_max_ratio = float(per.get("total_area_max_ratio", total_area_max_ratio))
        p_connectivity = int(per.get("connectivity", connectivity))

        mask_key = f"{layer}_morph_mask"
        if mask_key not in maps:
            reliability[layer] = {"valid": False, "q": 0.0, "reason": "missing_morph_mask"}
            continue

        mask = np.squeeze(maps[mask_key])
        if mask.ndim != 2:
            reliability[layer] = {"valid": False, "q": 0.0, "reason": "invalid_mask_shape"}
            continue

        H, W = mask.shape
        binary = (mask > 0).astype(np.uint8)

        try:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=p_connectivity, ltype=cv2.CV_32S,
            )
        except Exception as e:
            reliability[layer] = {"valid": False, "q": 0.0, "reason": f"cc_error: {e}"}
            continue

        areas_raw = []
        small_count_raw = 0
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx][cv2.CC_STAT_AREA])
            areas_raw.append(area)
            if area < p_small_area_threshold:
                small_count_raw += 1

        total_raw = len(areas_raw)
        total_area_raw = int(np.sum(areas_raw)) if areas_raw else 0
        max_area_raw = int(np.max(areas_raw)) if areas_raw else 0
        small_ratio_raw = small_count_raw / (total_raw + 1e-8)
        total_area_ratio_raw = total_area_raw / float(H * W)

        valid = (
            total_raw > 0
            and max_area_raw >= p_area_min
            and small_ratio_raw <= p_small_ratio_max
            and p_total_area_min_ratio <= total_area_ratio_raw <= p_total_area_max_ratio
        )
        q = 1.0 if valid else 0.0

        reliability[layer] = {
            "valid": valid,
            "q": q,
            "total_components": total_raw,
            "small_component_count": small_count_raw,
            "small_ratio": small_ratio_raw,
            "areas": areas_raw,
            "max_area": max_area_raw,
            "total_area": total_area_raw,
            "total_area_ratio": total_area_ratio_raw,
            "params": {
                "area_min": p_area_min,
                "small_area_threshold": p_small_area_threshold,
                "small_ratio_max": p_small_ratio_max,
                "total_area_min_ratio": p_total_area_min_ratio,
                "total_area_max_ratio": p_total_area_max_ratio,
                "connectivity": p_connectivity,
            },
        }

        if q == 1.0 and remove_small_components:
            keep_mask = np.zeros_like(binary)
            for label_idx in range(1, num_labels):
                area = int(stats[label_idx][cv2.CC_STAT_AREA])
                if area >= p_small_area_threshold:
                    keep_mask[labels == label_idx] = 1
            cleaned_mask = keep_mask

            e_layer = np.squeeze(maps[layer])
            e_cleaned = e_layer * keep_mask
            if maps[layer].ndim == 3:
                e_cleaned = np.expand_dims(e_cleaned, axis=-1)
            result[f"{layer}_morph_filtered"] = e_cleaned.astype(np.float32)

            num_labels_clean, _, stats_clean, _ = cv2.connectedComponentsWithStats(
                cleaned_mask.astype(np.uint8), connectivity=p_connectivity, ltype=cv2.CV_32S,
            )
            areas_clean = []
            small_count_clean = 0
            for label_idx in range(1, num_labels_clean):
                area = int(stats_clean[label_idx][cv2.CC_STAT_AREA])
                areas_clean.append(area)
                if area < p_small_area_threshold:
                    small_count_clean += 1

            result[f"{layer}_morph_cleaned"] = cleaned_mask.astype(np.float32)
            result[f"{layer}_cc_stats_cleaned"] = {
                "small_component_count": small_count_clean,
                "all_areas": areas_clean,
                "area_threshold": p_small_area_threshold,
                "total_components": len(areas_clean),
                "removed_small_components": True,
            }
        else:
            result[f"{layer}_morph_cleaned"] = np.zeros((H, W), dtype=np.float32)
            result[f"{layer}_cc_stats_cleaned"] = {
                "small_component_count": 0,
                "all_areas": [],
                "area_threshold": p_small_area_threshold,
                "total_components": 0,
                "removed_small_components": False,
            }

        result[f"{layer}_cc_stats"] = {
            "small_component_count": small_count_raw,
            "all_areas": areas_raw,
            "area_threshold": p_small_area_threshold,
            "total_components": total_raw,
            "removed_small_components": False,
        }

    valid_maps = []
    weights = []
    reliability_mask = np.zeros((H or 1, W or 1), dtype=np.float32)

    for layer in energy_layers:
        if layer not in maps:
            continue
        rel = reliability.get(layer, {})
        q = rel.get("q", 0.0)
        if q <= 0:
            continue
        filtered_key = f"{layer}_morph_filtered"
        if filtered_key in result:
            e = np.squeeze(result[filtered_key]).astype(np.float32)
        else:
            e = np.squeeze(maps[layer]).astype(np.float32)
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)

        mask_key = f"{layer}_morph_cleaned"
        if mask_key in result:
            m = np.squeeze(result[mask_key]).astype(np.float32)
            reliability_mask = np.maximum(reliability_mask, m)
        else:
            m = np.ones((H, W), dtype=np.float32)

        valid_maps.append(e_norm * m)
        weights.append(q)

    result["energy_reliability_dict"] = reliability

    if len(valid_maps) == 0:
        result["energy_fused_reliable"] = np.zeros((H or 1, W or 1), dtype=np.float32)
        result["energy_reliability_mask"] = reliability_mask
        return result

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / (weights_arr.sum() + 1e-8)

    fused = np.zeros((H, W), dtype=np.float32)
    for e, w in zip(valid_maps, weights_arr):
        fused += w * e

    result["energy_fused_reliable"] = fused
    result["energy_reliability_mask"] = reliability_mask

    return result


# ==============================================================
#                     规则14：可靠性能量引导校准
# ==============================================================

def reliability_calibrate(
    maps: Dict[str, np.ndarray],
    recon_key: str = "recon_error",
    lambda_enhance: float = 0.25,
    lambda_suppress: float = 0.0,
    use_suppression: bool = False,
    use_energy_diff: bool = True,
    energy_layers: Tuple[str, ...] = ("energy_layer1", "energy_layer2", "energy_layer3"),
) -> Dict[str, np.ndarray]:
    """规则14 = Step3：可靠性能量引导的重构误差校准。

    公式: S(p) = R(p) * [1 + λ * Q(p) * ΔE(p)]
    """
    result = {k: v.copy() for k, v in maps.items()}

    if recon_key not in maps:
        print("警告: 无重建误差图，跳过规则14校准")
        return result

    reliability = maps.get("energy_reliability_dict", {})
    if not reliability:
        print("警告: maps 中无 energy_reliability_dict（规则13未执行），跳过规则14校准")
        result["recon_error_calibrated"] = maps.get(recon_key, np.zeros((1, 1)))
        result["calibration_gain"] = np.ones_like(result["recon_error_calibrated"])
        return result

    recon = np.squeeze(maps[recon_key]).astype(np.float32)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

    energy_layer_names = []
    for base in energy_layers:
        if use_energy_diff:
            diff_key = base.replace("energy_layer", "energy_diff_layer")
            if diff_key in maps:
                energy_layer_names.append((base, diff_key))
        if base not in [e[0] for e in energy_layer_names]:
            if base in maps:
                energy_layer_names.append((base, base))

    valid_energy = []
    valid_weights = []
    valid_masks = []

    for base_name, energy_key in energy_layer_names:
        q = reliability.get(base_name, {}).get("q", 0.0)
        if q <= 0:
            continue

        e = np.squeeze(maps[energy_key]).astype(np.float32)
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)

        mask_key = f"{base_name}_morph_mask"
        if mask_key in maps:
            m = np.squeeze(maps[mask_key]).astype(np.float32)
            m = (m > 0).astype(np.float32)
        else:
            m = np.ones_like(e_norm, dtype=np.float32)

        valid_energy.append(e_norm * m)
        valid_weights.append(q)
        valid_masks.append(m)

    if len(valid_energy) == 0:
        result["recon_error_calibrated"] = recon_norm
        result["energy_reliability_mask"] = np.zeros_like(recon_norm)
        result["calibration_gain"] = np.ones_like(recon_norm)
        return result

    weights = np.array(valid_weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    energy_fused = np.zeros_like(recon_norm, dtype=np.float32)
    reliability_mask = np.zeros_like(recon_norm, dtype=np.float32)

    for e, m, w in zip(valid_energy, valid_masks, weights):
        energy_fused += w * e
        reliability_mask = np.maximum(reliability_mask, m)

    energy_fused = (energy_fused - energy_fused.min()) / (energy_fused.max() - energy_fused.min() + 1e-8)

    gain = 1.0 + lambda_enhance * reliability_mask * energy_fused
    calibrated = recon_norm * gain

    if use_suppression and lambda_suppress > 0:
        suppress_weight = 1.0 - lambda_suppress * reliability_mask * (1.0 - energy_fused)
        suppress_weight = np.clip(suppress_weight, 0.7, 1.0)
        calibrated = calibrated * suppress_weight

    calibrated = (calibrated - calibrated.min()) / (calibrated.max() - calibrated.min() + 1e-8)

    result["recon_error_calibrated"] = calibrated
    result["energy_fused_reliable"] = energy_fused
    result["energy_reliability_mask"] = reliability_mask
    result["calibration_gain"] = gain

    return result


# ==============================================================
#                     可视化核心
# ==============================================================

def plot_step13_14_heatmap(
    maps: Dict[str, np.ndarray],
    rule13_result: Dict[str, np.ndarray],
    rule14_result: Dict[str, np.ndarray],
    reliability: Dict[str, Dict],
    sample_idx: int,
    gt: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    fig_dpi: int = 150,
) -> None:
    """
    四行热力图布局，展示 Rule 13 + Rule 14 完整流水线：

    行0：原始 recon_error + 各 energy_layer（每列独立 JET colorbar）
    行1：各层 *_morph_mask（Rule 5 产物）+ 可靠性 q 标签
    行2：Rule 13 融合结果 energy_fused_reliable + energy_reliability_mask
    行3：Rule 14 校准结果 + 校准增益图 + GT（如果有）

    参数：
        maps             : MapDataset.get_all_maps() 返回的原始 map 字典
        rule13_result    : morph_energy_reliability_fusion() 返回的字典
        rule14_result    : reliability_calibrate() 返回的字典
        reliability      : energy_reliability_dict（来自 rule13_result）
        sample_idx       : 样本编号
        gt               : 真实标注（可选）
        save_path        : 保存路径（可选）
        fig_dpi          : 输出 DPI
    """
    energy_layers = ["energy_layer1", "energy_layer2", "energy_layer3"]
    available_energy = [e for e in energy_layers if e in maps]

    # 列数：recon_error + N 个 energy_layer + (可选 GT 列)
    n_energy = len(available_energy)
    n_cols = 1 + n_energy
    has_gt = gt is not None

    fig = plt.figure(figsize=(4 * n_cols + (0.5 if has_gt else 0), 4 * 4 + 1.5))
    fig.suptitle(
        f"Sample #{sample_idx:04d}  —  Rule 13 + Rule 14 Pipeline",
        fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(
        4, n_cols + (1 if has_gt else 0),
        figure=fig, hspace=0.45, wspace=0.3,
    )

    # ── 辅助函数：在指定 GridSpec 位置绘制热力图 cell ─────────────
    def heat_cell(
        row: int, col: int,
        arr: np.ndarray,
        title: str,
        cmap: str = "jet",
        show_cbar: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        fmt: str = ".3f",
        gt_mask: Optional[np.ndarray] = None,
    ) -> None:
        """在 fig 的 (row, col) 位置绘制一张热力图。"""
        ax = fig.add_subplot(gs[row, col])
        arr_2d = np.squeeze(arr)
        v_lo = vmin if vmin is not None else float(arr_2d.min())
        v_hi = vmax if vmax is not None else float(arr_2d.max())
        im = ax.imshow(arr_2d, cmap=cmap, vmin=v_lo, vmax=v_hi)
        ax.set_title(f"{title}\n[{v_lo:{fmt}} ~ {v_hi:{fmt}}]", fontsize=8)
        ax.axis("off")
        if show_cbar:
            plt.colorbar(im, ax=ax, fraction=0.046)
        if gt_mask is not None:
            gt_c = np.squeeze(gt_mask).astype(np.uint8)
            contours, _ = cv2.findContours(gt_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ax.contour(contours, levels=[0.5], colors=["white"], linewidths=[1.2])
            ax.imshow(np.ma.masked_where(gt_c == 0, gt_c * 0.3),
                      cmap="gray", vmin=0, vmax=1, alpha=0.25)

    def text_cell(row: int, col: int, title: str, lines: List[Tuple[str, str]]) -> None:
        """在指定位置绘制文本标签 cell（用于可靠性 q 值展示）。"""
        ax = fig.add_subplot(gs[row, col])
        ax.text(0.5, 0.85, title, ha="center", va="top",
                fontsize=8, transform=ax.transAxes)
        for i, (k, v) in enumerate(lines):
            color = "green" if "VALID" in v else ("orange" if "WARN" in v else "red")
            ax.text(0.05, 0.65 - i * 0.22, k, ha="left", va="center",
                    fontsize=7.5, color="white", transform=ax.transAxes,
                    fontfamily="monospace")
            ax.text(0.55, 0.65 - i * 0.22, v, ha="left", va="center",
                    fontsize=7.5, color=color, fontweight="bold",
                    transform=ax.transAxes, fontfamily="monospace")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ────────────────────────────────────────────────────────────
    # 行0：原始 map（recon_error + energy_layer*）
    # ────────────────────────────────────────────────────────────
    heat_cell(0, 0, maps["recon_error"], "原始 recon_error", gt_mask=gt)
    for c, layer in enumerate(available_energy, start=1):
        heat_cell(0, c, maps[layer], f"原始 {layer}")

    # ────────────────────────────────────────────────────────────
    # 行1：Rule 5 *_morph_mask + 各层 q 标签
    # ────────────────────────────────────────────────────────────
    for c, layer in enumerate(available_energy, start=1):
        morph_key = f"{layer}_morph_mask"
        if morph_key in maps:
            heat_cell(1, c, maps[morph_key], f"Rule5 {layer}\n_morph_mask",
                      cmap="hot", vmin=0, vmax=1)
        else:
            # 无 morph_mask 时画纯黑格子占位
            arr = np.zeros_like(maps["recon_error"])
            heat_cell(1, c, arr, f"Rule5 {layer}\n(no mask)", cmap="gray",
                      vmin=0, vmax=1)

        # q 值文本框（col0 行1 展示融合 mask 概览）
        rel = reliability.get(layer, {})
        q = rel.get("q", 0.0)
        valid = rel.get("valid", False)
        status = "VALID" if valid else "DISCARD"
        text_lines = [
            ("q", f"{q:.1f}"),
            ("status", status),
            ("CCs", str(rel.get("total_components", 0))),
            ("maxA", str(rel.get("max_area", 0))),
            ("s_ratio", f"{rel.get('small_ratio', 0):.2f}"),
            ("area%", f"{rel.get('total_area_ratio', 0)*100:.2f}%"),
        ]
        text_cell(1, c, f"Rel {layer}", text_lines)

    # col0 行1：展示 energy_reliability_mask（各层 cleaned mask 取并集）
    if "energy_reliability_mask" in rule13_result:
        heat_cell(1, 0, rule13_result["energy_reliability_mask"],
                  "Rule13 融合 Mask\n(reliability_mask)", cmap="bone", vmin=0, vmax=1)

    # ────────────────────────────────────────────────────────────
    # 行2：Rule 13 融合结果 + 清理后的 mask
    # ────────────────────────────────────────────────────────────
    if "energy_fused_reliable" in rule13_result:
        heat_cell(2, 0, rule13_result["energy_fused_reliable"],
                  "Rule13 融合能量图\n(energy_fused)", cmap="jet")
    for c, layer in enumerate(available_energy, start=1):
        cleaned_key = f"{layer}_morph_cleaned"
        if cleaned_key in rule13_result:
            heat_cell(2, c, rule13_result[cleaned_key],
                      f"Rule13 {layer}\n_cleaned_mask",
                      cmap="hot", vmin=0, vmax=1)

    # ────────────────────────────────────────────────────────────
    # 行3：Rule 14 校准结果 + 增益图 + GT
    # ────────────────────────────────────────────────────────────
    if "recon_error_calibrated" in rule14_result:
        heat_cell(3, 0, rule14_result["recon_error_calibrated"],
                  "Rule14 校准异常图\nS(p)", cmap="jet", gt_mask=gt)

    if "calibration_gain" in rule14_result:
        # 增益图放在 recon_error 列右侧第一列（col1），若无 energy layer 则 col1 可能溢出
        gain_col = min(1, n_cols - 1)
        heat_cell(3, gain_col, rule14_result["calibration_gain"],
                  "Rule14 校准增益\nG(p)", cmap="coolwarm",
                  vmin=float(rule14_result["calibration_gain"].min()),
                  vmax=float(rule14_result["calibration_gain"].max()),
                  fmt=".2f")

    if "energy_reliability_mask" in rule14_result:
        mask_col = min(2, n_cols - 1) if n_cols > 2 else n_cols - 1
        heat_cell(3, mask_col, rule14_result["energy_reliability_mask"],
                  "Rule14 可靠 Mask\nQ(p)", cmap="bone", vmin=0, vmax=1)

    # GT 列
    if has_gt:
        gt_col = n_cols
        ax = fig.add_subplot(gs[0, gt_col])
        ax.imshow(gt, cmap="gray")
        ax.set_title("Ground Truth", fontsize=8)
        ax.axis("off")
        # GT 也画在最后一行便于对齐
        ax_last = fig.add_subplot(gs[3, gt_col])
        ax_last.imshow(gt, cmap="Reds_r")
        ax_last.set_title("GT (overlay)", fontsize=8)
        ax_last.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=fig_dpi, bbox_inches="tight",
                     facecolor=fig.get_facecolor())
        print(f"[保存] {save_path}")
    plt.close(fig)


# ==============================================================
#                     命令行接口
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rule 13 + Rule 14 流水线可视化脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单样本可视化
  python visualize_step14.py --sample_idx 20 --save

  # 指定数据目录
  python visualize_step14.py --sample_idx 20 --sample_dir testSet/bottle/n_bottle_a_0_s_111 --save

  # 批量处理
  python visualize_step14.py --range 0 20 --save --output_dir output/step14_batch

  # 自定义参数
  python visualize_step14.py --sample_idx 20 --rule13_area_min 100 --rule13_small_ratio_max 0.5 --rule14_lambda_enhance 0.3 --save
        """,
    )
    parser.add_argument("--sample_dir", type=str,
                        default="testSet/bottle/n_bottle_a_0_s_111",
                        help="样本目录路径")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="单个样本编号（0起始）")
    parser.add_argument("--compare_idx", type=int, nargs=2, default=None,
                        help="对比两个样本（如 20 30）")
    parser.add_argument("--range", type=int, nargs=2, default=None,
                        metavar=("START", "STOP"),
                        help="批量范围，左闭右开。例如 --range 0 10 → 处理 0~9")
    parser.add_argument("--show", action="store_true", help="实时显示图片")
    parser.add_argument("--save", action="store_true", help="保存图片到 output_dir")
    parser.add_argument("--output_dir", type=str, default="output/step14_vis",
                        help="输出目录")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI，默认 150")

    # ── Rule 13 参数 ──
    r13 = parser.add_argument_group("Rule 13 (morph_energy_reliability_fusion) 参数")
    r13.add_argument("--rule13_area_min", type=int, default=50,
                     help="连通域最小面积阈值，默认 50")
    r13.add_argument("--rule13_small_thresh", type=int, default=50,
                     help="小连通域面积阈值，默认 50")
    r13.add_argument("--rule13_small_ratio_max", type=float, default=0.6,
                     help="小连通域占比上限，默认 0.6")
    r13.add_argument("--rule13_area_min_ratio", type=float, default=0.001,
                     help="总面积比下限，默认 0.001")
    r13.add_argument("--rule13_area_max_ratio", type=float, default=0.35,
                     help="总面积比上限，默认 0.35")
    r13.add_argument("--rule13_connectivity", type=int, default=8,
                     help="连通域连通性（4或8），默认 8")

    # ── Rule 14 参数 ──
    r14 = parser.add_argument_group("Rule 14 (reliability_calibrate) 参数")
    r14.add_argument("--rule14_lambda_enhance", type=float, default=0.25,
                     help="保守增强系数 λ，默认 0.25")
    r14.add_argument("--rule14_lambda_suppress", type=float, default=0.0,
                     help="软抑制系数（消融用），默认 0.0")
    r14.add_argument("--rule14_use_suppression", action="store_true",
                     help="启用软抑制（默认关闭）")
    r14.add_argument("--rule14_use_energy_diff", action="store_true", default=True,
                     help="优先使用能量差图 ΔE（默认 True）")

    return parser.parse_args()


def resolve_path(p: str) -> Path:
    if os.path.isabs(p):
        return Path(p)
    root = Path("/home/bobbystone/CKAAD")
    candidate = root / p
    if candidate.exists():
        return candidate
    return Path(p)


def main():
    args = parse_args()

    sample_dir = resolve_path(args.sample_dir)
    print(f"[INFO] 数据目录: {sample_dir}")
    if not sample_dir.exists():
        print(f"[ERROR] 目录不存在: {sample_dir}")
        return

    dataset = MapDataset(str(sample_dir))
    print(f"[INFO] 共 {dataset.num_samples} 个样本，编号: {dataset.indices[0]} ~ {dataset.indices[-1]}")

    if args.output_dir:
        out_dir = resolve_path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 输出目录: {out_dir}")

    if args.sample_idx is not None:
        indices = [args.sample_idx]
    elif args.compare_idx is not None:
        indices = args.compare_idx
    elif args.range is not None:
        indices = list(range(args.range[0], args.range[1]))
    else:
        indices = dataset.indices[:3]
        print(f"[INFO] 未指定样本，默认分析: {indices}")

    for pos, idx in enumerate(indices, 1):
        if idx not in dataset.indices:
            print(f"[WARN] 编号 {idx} 不存在，跳过")
            continue

        label = dataset.get_label(idx)
        gt = dataset.get_gt(idx)
        maps = dataset.get_all_maps(idx)

        print(f"\n{'='*60}")
        print(f"  样本 #{idx:04d}  |  标签={label}  |  GT={'有' if gt is not None else '无'}"
              f"  [{pos}/{len(indices)}]")
        print(f"{'='*60}")

        if not maps:
            print(f"[WARN] 样本 {idx} 无可用 map，跳过")
            continue

        # Step 1: Rule 13 — 形态学可靠性融合（需要 *_morph_mask）
        print("  [Rule13] morph_energy_reliability_fusion ...")
        rule13_result = morph_energy_reliability_fusion(
            maps,
            area_min=args.rule13_area_min,
            small_area_threshold=args.rule13_small_thresh,
            small_ratio_max=args.rule13_small_ratio_max,
            total_area_min_ratio=args.rule13_area_min_ratio,
            total_area_max_ratio=args.rule13_area_max_ratio,
            connectivity=args.rule13_connectivity,
        )

        reliability = rule13_result.get("energy_reliability_dict", {})
        print("  Rule13 可靠性:")
        for layer, info in reliability.items():
            q = info.get("q", 0.0)
            status = "VALID" if info.get("valid") else "DISCARD"
            print(f"    {layer}: q={q:.1f} | {status} | "
                  f"CC={info.get('total_components', 0)} | "
                  f"max_area={info.get('max_area', 0)} | "
                  f"area_ratio={info.get('total_area_ratio', 0):.4f}")

        # Step 2: Rule 14 — 可靠性能量引导校准
        print(f"  [Rule14] reliability_calibrate ...")
        rule14_result = reliability_calibrate(
            rule13_result,
            lambda_enhance=args.rule14_lambda_enhance,
            lambda_suppress=args.rule14_lambda_suppress,
            use_suppression=args.rule14_use_suppression,
            use_energy_diff=args.rule14_use_energy_diff,
        )

        calibrated = rule14_result.get("recon_error_calibrated")
        gain = rule14_result.get("calibration_gain")
        rel_mask = rule14_result.get("energy_reliability_mask")
        print(f"  Rule14 校准完成:")
        print(f"    λ_enhance={args.rule14_lambda_enhance}, "
              f"λ_suppress={args.rule14_lambda_suppress}, "
              f"use_suppression={args.rule14_use_suppression}, "
              f"use_energy_diff={args.rule14_use_energy_diff}")
        if calibrated is not None:
            print(f"    校准后: mean={float(calibrated.mean()):.4f}, "
                  f"std={float(calibrated.std()):.4f}, max={float(calibrated.max()):.4f}")
        if gain is not None:
            print(f"    增益 G(p): min={float(gain.min()):.4f}, max={float(gain.max()):.4f}")
        if rel_mask is not None:
            nonzero = int(np.count_nonzero(rel_mask))
            total = rel_mask.size
            print(f"    可靠区域覆盖: {nonzero}/{total} ({100*nonzero/total:.1f}%)")

        # ── 可视化 ──
        save_path = None
        if args.save and args.output_dir:
            save_path = str(out_dir / f"sample_{idx:04d}_step13_14.png")

        plot_step13_14_heatmap(
            maps=maps,
            rule13_result=rule13_result,
            rule14_result=rule14_result,
            reliability=reliability,
            sample_idx=idx,
            gt=gt,
            save_path=save_path,
            fig_dpi=args.dpi,
        )

        print(f"  可视化完成{'，已保存' if save_path else '（未保存）'}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
