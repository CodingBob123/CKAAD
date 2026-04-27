"""
======================================================================
创新点二 Step1~Step3 流水线可视化脚本
======================================================================

三步流水线：
  Step1 (Rule5):  形态学滤波（开运算+闭运算）→ *_morph_mask
  Step2:          连通域可靠性分析 → 能量层 q 值 / energy_reliability_dict
  Step3 (Rule14): 可靠性能量引导重构误差校准 → recon_error_calibrated

配色方案：
  - 所有热力图统一使用 cv2.COLORMAP_JET（比 matplotlib jet 更标准）
  - GT 叠加用白色轮廓 + 半透明红色填充

排版（4行 N+2 列）：
  行0：原始 maps（recon_error + energy_layer1/2/3）
  行1：Step1 处理后（_morph_mask 二值图）
  行2：Step2 可靠性（各层 q 值标签 + 融合能量图）
  行3：Step3 校准结果（校准异常图 + 增益图 + GT）

用法示例：
  # 单样本可视化
  python viz_step123.py --sample_idx 20 --save

  # 指定数据目录
  python viz_step123.py --sample_idx 20 --sample_dir testSet/bottle/n_bottle_a_0_s_111 --save

  # 批量处理
  python viz_step123.py --range 0 10 --save --output_dir output/step123_batch

  # 自定义 Step1 参数
  python viz_step123.py --sample_idx 20 --morph_threshold 0.5 --morph_kernel 5 --save

  # 自定义 Step2/Step3 参数
  python viz_step123.py --sample_idx 20 --area_min 100 --lambda_enhance 0.3 --save
======================================================================
"""

import os
import re
import argparse
import numpy as np
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

# 配置中文字体支持（优先使用 Noto Serif CJK）
_fonts_candidates = [
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_fonts_found = [f for f in _fonts_candidates if os.path.exists(f)]
if _fonts_found:
    fm.fontManager.addfont(_fonts_found[0])
    _prop = fm.FontProperties(fname=_fonts_found[0])
    _font_name = _prop.get_name()
    plt.rcParams["font.family"] = _font_name
    plt.rcParams["axes.unicode_minus"] = False
else:
    _font_name = "DejaVu Sans"
    _prop = None

# ==============================================================
#                        配色工具
# ==============================================================

def arr_to_heatmap(arr: np.ndarray) -> np.ndarray:
    """
    将 float ndarray 归一化到 [0, 255]，用 cv2.COLORMAP_JET 渲染为 RGB 热力图。

    与 util/test.py 中 cvt2heatmap() 逻辑完全一致，确保所有图的配色标准统一。
    """
    arr = np.squeeze(arr).astype(np.float32)
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max - arr_min < 1e-8:
        gray = np.zeros(arr.shape, dtype=np.uint8)
    else:
        gray = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_gt_on_heatmap(
    heatmap: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.30,
    contour_color: Tuple[int, int, int] = (255, 255, 255),
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    在热力图上叠加 GT mask。

    参数：
        heatmap:  RGB 热力图（由 arr_to_heatmap 生成），shape [H, W, 3]
        gt:        二值 mask，shape [H, W]，值为 0 或 1
        alpha:     GT 填充的透明度
        contour_color: 轮廓颜色 (R, G, B)
        contour_thickness: 轮廓线宽
    返回：
        叠加后的 RGB 图像
    """
    result = heatmap.copy().astype(np.float32)
    gt_u8 = (np.squeeze(gt) > 0.5).astype(np.uint8)

    h, w = gt_u8.shape[:2]
    if result.shape[:2] != gt_u8.shape:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    fill_color = np.array([255, 0, 0], dtype=np.float32)
    for c in range(3):
        result[:, :, c] = np.where(
            gt_u8 == 1,
            (1 - alpha) * result[:, :, c] + alpha * fill_color[2 - c] * 255,
            result[:, :, c],
        )

    contours, _ = cv2.findContours(
        gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, contour_color, contour_thickness)
    return np.clip(result, 0, 255).astype(np.uint8)


def normalized_array(arr: np.ndarray) -> np.ndarray:
    """归一化到 [0, 1]，避免除零。"""
    arr = np.squeeze(arr).astype(np.float32)
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


# ==============================================================
#                        数据加载器
# ==============================================================

class MapDataset:
    """
    从样本目录中加载所有 map 文件（npy 格式）。

    目录结构：
        sample_dir/
            energy_layer1/energy_layer1_0000.npy
            energy_layer2/...
            energy_layer3/...
            recon_error/...
            ground_truth/...
            metadata.npy
    """

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
        """加载编号 idx 对应的所有原始 map。"""
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
        """从 metadata.npy 加载 GT mask。"""
        if self._metadata is None:
            return None
        indices = self._metadata.get("indices", [])
        gts = self._metadata.get("gts", None)
        if not indices or gts is None:
            return None
        try:
            pos = indices.index(idx)
            gt = gts[pos, 0]  # shape: (H, W)
            return (gt > 0.5).astype(np.uint8)
        except (ValueError, IndexError):
            return None

    def get_label(self, idx: int) -> str:
        if self._metadata and "labels" in self._metadata:
            labels = self._metadata["labels"]
            if 0 <= idx < len(labels):
                return str(labels[idx])
        return "unknown"


# ==============================================================
#                   Step1: 形态学滤波（Rule5）
# ==============================================================

def step1_morphological_filter(
    maps: Dict[str, np.ndarray],
    threshold: float = 0.6,
    kernel_size: int = 3,
    operation: str = "both",
) -> Dict[str, np.ndarray]:
    """
    Step1：形态学滤波（规则5：开运算+闭运算）

    对每个能量层和重建误差图：
      1. 二值化（阈值分割）
      2. 开运算（去除小噪点）/ 闭运算（填充小空洞）
      3. 用 morph_mask 加权原图

    参数：
        maps:       原始 map 字典
        threshold:  二值化阈值（相对值），默认 0.6
        kernel_size: 形态学卷积核大小，默认 3
        operation:  "open" | "close" | "both"，默认 "both"
    返回：
        包含原图 + *_morph_mask 的字典
    """
    result = {k: v.copy() for k, v in maps.items()}
    struct = generate_binary_structure(2, 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)

    for name, arr in maps.items():
        arr_2d = np.squeeze(arr)
        if arr_2d.ndim != 2:
            result[name] = arr.copy()
            continue

        if threshold is None:
            t = arr_2d.mean() + arr_2d.std()
        else:
            t = arr_2d.min() + threshold * (arr_2d.max() - arr_2d.min())

        binary = arr_2d > t

        if operation in ("open", "both"):
            binary = binary_opening(binary, structure=struct)
        if operation in ("close", "both"):
            binary = binary_closing(binary, structure=struct)

        mask = binary.astype(np.float32)
        if arr.ndim == 3:
            mask = np.expand_dims(mask, axis=-1)
        result[name] = arr * mask
        result[f"{name}_morph_mask"] = mask

    return result


# ==============================================================
#              Step2: 连通域可靠性分析
# ==============================================================

def step2_energy_reliability(
    maps: Dict[str, np.ndarray],
    energy_layers: Tuple[str, ...] = ("energy_layer1", "energy_layer2", "energy_layer3"),
    area_min: int = 50,
    small_area_threshold: int = 50,
    small_ratio_max: float = 0.6,
    total_area_min_ratio: float = 0.001,
    total_area_max_ratio: float = 0.35,
    connectivity: int = 8,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Step2：连通域可靠性分析

    对每个能量层的 *_morph_mask 做连通域统计，评估其可靠性并计算 q 值。

    可靠性判据（四重约束）：
      1. 总连通域数 > 0
      2. 最大连通域面积 >= area_min
      3. 小连通域占比 <= small_ratio_max
      4. 总面积比在 [total_area_min_ratio, total_area_max_ratio] 区间内

    同时对可靠层做 morph_filtered（去除小连通域）和融合。

    参数：
        maps:  Step1 输出（包含 *_morph_mask）
        energy_layers: 要分析的能量层列表
        其他参数：各判据阈值

    返回：
        (更新后的 maps 字典, 可靠性字典)
    """
    result = {k: v.copy() for k, v in maps.items()}
    reliability: Dict[str, Dict] = {}
    H, W = None, None

    for layer in energy_layers:
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

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=connectivity, ltype=cv2.CV_32S,
        )

        areas_raw = []
        small_count_raw = 0
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx][cv2.CC_STAT_AREA])
            areas_raw.append(area)
            if area < small_area_threshold:
                small_count_raw += 1

        total_raw = len(areas_raw)
        total_area_raw = int(np.sum(areas_raw)) if areas_raw else 0
        max_area_raw = int(np.max(areas_raw)) if areas_raw else 0
        small_ratio_raw = small_count_raw / (total_raw + 1e-8)
        total_area_ratio_raw = total_area_raw / float(H * W)

        valid = (
            total_raw > 0
            and max_area_raw >= area_min
            and small_ratio_raw <= small_ratio_max
            and total_area_min_ratio <= total_area_ratio_raw <= total_area_max_ratio
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
        }

        if q == 1.0:
            keep_mask = np.zeros_like(binary)
            for label_idx in range(1, num_labels):
                area = int(stats[label_idx][cv2.CC_STAT_AREA])
                if area >= small_area_threshold:
                    keep_mask[labels == label_idx] = 1

            e_layer = np.squeeze(maps[layer])
            e_cleaned = e_layer * keep_mask
            if maps[layer].ndim == 3:
                e_cleaned = np.expand_dims(e_cleaned, axis=-1)
            result[f"{layer}_morph_filtered"] = e_cleaned.astype(np.float32)
            result[f"{layer}_morph_cleaned"] = keep_mask.astype(np.float32)
        else:
            result[f"{layer}_morph_filtered"] = np.zeros((H, W), dtype=np.float32)
            result[f"{layer}_morph_cleaned"] = np.zeros((H, W), dtype=np.float32)

    # ── 融合可靠能量图 ──
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
        e_norm = normalized_array(e)

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
        return result, reliability

    weights_arr = np.array(weights, dtype=np.float32)
    weights_arr = weights_arr / (weights_arr.sum() + 1e-8)

    fused = np.zeros((H, W), dtype=np.float32)
    for e, w in zip(valid_maps, weights_arr):
        fused += w * e

    result["energy_fused_reliable"] = fused
    result["energy_reliability_mask"] = reliability_mask

    return result, reliability


# ==============================================================
#              Step3: 可靠性能量引导校准
# ==============================================================

def step3_reliability_calibrate(
    maps: Dict[str, np.ndarray],
    recon_key: str = "recon_error",
    lambda_enhance: float = 0.25,
    lambda_suppress: float = 0.0,
    use_suppression: bool = False,
    use_energy_diff: bool = True,
    energy_layers: Tuple[str, ...] = ("energy_layer1", "energy_layer2", "energy_layer3"),
) -> Dict[str, np.ndarray]:
    """
    Step3：可靠性能量引导的重构误差校准

    公式: S(p) = R(p) * [1 + λ * Q(p) * ΔE(p)]

    参数：
        maps:           Step2 输出（含 energy_reliability_dict）
        recon_key:      重构误差图 key
        lambda_enhance: 保守增强系数，默认 0.25
        lambda_suppress: 软抑制系数（消融用），默认 0.0
        use_suppression: 是否启用软抑制，默认 False
        use_energy_diff: 是否优先使用能量差图，默认 True
    """
    result = {k: v.copy() for k, v in maps.items()}

    if recon_key not in maps:
        print("警告: 无重建误差图，跳过 Step3 校准")
        return result

    reliability = maps.get("energy_reliability_dict", {})
    if not reliability:
        print("警告: maps 中无 energy_reliability_dict（Step2 未执行），跳过 Step3 校准")
        result["recon_error_calibrated"] = maps.get(recon_key, np.zeros((1, 1)))
        result["calibration_gain"] = np.ones_like(result["recon_error_calibrated"])
        return result

    recon = np.squeeze(maps[recon_key]).astype(np.float32)
    recon_norm = normalized_array(recon)

    # ── 确定使用哪类能量图 ──
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
        e_norm = normalized_array(e)

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

    energy_fused = normalized_array(energy_fused)

    # 保守增强
    gain = 1.0 + lambda_enhance * reliability_mask * energy_fused
    calibrated = recon_norm * gain

    # 可选：轻微软抑制
    if use_suppression and lambda_suppress > 0:
        suppress_weight = 1.0 - lambda_suppress * reliability_mask * (1.0 - energy_fused)
        suppress_weight = np.clip(suppress_weight, 0.7, 1.0)
        calibrated = calibrated * suppress_weight

    calibrated = normalized_array(calibrated)

    result["recon_error_calibrated"] = calibrated
    result["energy_fused_reliable"] = energy_fused
    result["energy_reliability_mask"] = reliability_mask
    result["calibration_gain"] = gain

    return result


# ==============================================================
#                   可视化核心函数
# ==============================================================

def plot_pipeline_cv2_heatmap(
    maps: Dict[str, np.ndarray],
    step1_result: Dict[str, np.ndarray],
    reliability: Dict[str, Dict],
    step3_result: Dict[str, np.ndarray],
    sample_idx: int,
    gt: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    fig_dpi: int = 150,
) -> None:
    """
    绘制创新点二 Step1~Step3 完整流水线对比图。

    所有热力图统一使用 cv2.COLORMAP_JET 渲染。

    排版（4行 N+1 列，N = 1 + len(available_energy)）：
      行0（原始）：recon_error + energy_layer1/2/3
      行1（Step1）：各层 _morph_mask 二值图
      行2（Step2）：融合可靠能量图 + 各层可靠性标签
      行3（Step3）：校准异常图 + 增益图 + GT
    """

    energy_layers = ["energy_layer1", "energy_layer2", "energy_layer3"]
    available_energy = [e for e in energy_layers if e in maps]
    n_energy = len(available_energy)
    has_gt = gt is not None
    n_cols = 1 + n_energy + (1 if has_gt else 0)

    base_w = 3.0
    base_h = 2.8
    fig_h = base_h * 4 + 1.5
    fig_w = base_w * n_cols + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle(
        f"Sample #{sample_idx:04d}  —  Step1~Step3 Pipeline  (cv2 JET)",
        fontsize=13, fontweight="bold", color="white", y=0.99,
        fontproperties=fm.FontProperties(fname=_fonts_found[0]) if _fonts_found else None,
    )
    gs = gridspec.GridSpec(
        4, n_cols,
        figure=fig,
        hspace=0.45,
        wspace=0.30,
        top=0.93,
        bottom=0.05,
        left=0.05,
        right=0.95,
    )

    # ── 辅助：绘制 cv2 热力图 cell ────────────────────────────────
    def heat_cell(
        row: int, col: int,
        arr: np.ndarray,
        title: str,
        show_cbar: bool = True,
        show_gt: bool = False,
        gt_arr: Optional[np.ndarray] = None,
    ) -> None:
        """在 (row, col) 位置绘制 cv2 JET 热力图 cell。"""
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#2a2a2a")

        heatmap = arr_to_heatmap(arr)
        if show_gt and gt_arr is not None:
            heatmap = overlay_gt_on_heatmap(heatmap, gt_arr)

        ax.imshow(heatmap, aspect="equal")
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        title_lines = title.split("\n")
        title_str = "\n".join(title_lines)
        ax.set_title(f"{title_str}\n[{vmin:.3f} ~ {vmax:.3f}]",
                     fontsize=7.5, color="white", pad=3)
        ax.axis("off")

        if show_cbar:
            cbar = plt.colorbar(
                matplotlib.cm.ScalarMappable(
                    cmap="jet",
                    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                ),
                ax=ax, fraction=0.046, pad=0.04,
            )
            cbar.ax.tick_params(colors="white", labelsize=6)
            cbar.outline.set_edgecolor("white")
            cbar.ax.yaxis.set_tick_params(color="white")

    def binary_cell(
        row: int, col: int,
        arr: np.ndarray,
        title: str,
    ) -> None:
        """在 (row, col) 位置绘制二值 mask cell（黑白）。"""
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#2a2a2a")
        arr_2d = np.squeeze(arr).astype(np.float32)
        ax.imshow(arr_2d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title}\n[0.000 ~ 1.000]",
                     fontsize=7.5, color="white", pad=3)
        ax.axis("off")

    def text_cell(
        row: int, col: int,
        title: str,
        lines: List[Tuple[str, str]],
        bg_color: str = "#1a1a1a",
    ) -> None:
        """在 (row, col) 位置绘制文本标签 cell（显示 q 值等信息）。"""
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(bg_color)
        ax.text(0.5, 0.90, title, ha="center", va="top",
                fontsize=8, color="white", fontweight="bold",
                transform=ax.transAxes,
                fontproperties=_prop if _prop else None)

        for i, (k, v) in enumerate(lines):
            color = "#00e676" if "VALID" in v else ("#ff9800" if "WARN" in v else "#ff5252")
            ax.text(0.05, 0.72 - i * 0.20, f"{k}:", ha="left", va="center",
                    fontsize=7.5, color="#b0b0b0", transform=ax.transAxes,
                    fontproperties=_prop if _prop else None)
            ax.text(0.55, 0.72 - i * 0.20, v, ha="left", va="center",
                    fontsize=7.5, color=color, fontweight="bold",
                    transform=ax.transAxes,
                    fontproperties=_prop if _prop else None)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ── 行0：原始 maps ─────────────────────────────────────────
    heat_cell(0, 0, maps["recon_error"], "原始\nrecon_error",
              show_gt=True, gt_arr=gt)
    for c, layer in enumerate(available_energy, start=1):
        heat_cell(0, c, maps[layer], f"原始\n{layer}")

    # ── 行1：Step1 形态学 mask ─────────────────────────────────
    binary_cell(1, 0, step1_result.get("recon_error_morph_mask",
             step1_result.get("recon_error", maps["recon_error"])),
             "Step1\nrecon_morph")
    for c, layer in enumerate(available_energy, start=1):
        morph_key = f"{layer}_morph_mask"
        if morph_key in step1_result:
            binary_cell(1, c, step1_result[morph_key], f"Step1\n{layer}_morph")
        else:
            arr = np.zeros_like(maps["recon_error"])
            binary_cell(1, c, arr, f"Step1\n{layer}_morph\n(missing)")

    # ── 行2：Step2 可靠性分析 ──────────────────────────────────
    if "energy_fused_reliable" in step3_result:
        heat_cell(2, 0, step3_result["energy_fused_reliable"],
                  "Step2 融合\n可靠能量图 E*")
    elif "energy_fused_reliable" in step1_result:
        heat_cell(2, 0, step1_result["energy_fused_reliable"],
                  "Step2 融合\n可靠能量图 E*")
    else:
        arr = np.zeros_like(maps["recon_error"])
        heat_cell(2, 0, arr, "Step2 融合\n可靠能量图 E*\n(unavailable)")

    for c, layer in enumerate(available_energy, start=1):
        rel = reliability.get(layer, {})
        q = rel.get("q", 0.0)
        valid = rel.get("valid", False)
        status = "VALID" if valid else "DISCARD"
        text_lines = [
            ("q", f"{q:.1f}"),
            ("status", status),
            ("CCs", str(rel.get("total_components", 0))),
            ("maxA", str(rel.get("max_area", 0))),
            ("sm%",
             f"{rel.get('small_ratio', 0) * 100:.1f}%"),
            ("area%",
             f"{rel.get('total_area_ratio', 0) * 100:.2f}%"),
        ]
        bg = "#0d3310" if valid else "#330d10"
        text_cell(2, c, f"Step2 {layer}\n可靠性", text_lines, bg_color=bg)

    # ── 行3：Step3 校准结果 ────────────────────────────────────
    if "recon_error_calibrated" in step3_result:
        heat_cell(3, 0, step3_result["recon_error_calibrated"],
                  "Step3 校准\n异常图 S(p)",
                  show_gt=True, gt_arr=gt)
    else:
        arr = np.zeros_like(maps["recon_error"])
        heat_cell(3, 0, arr, "Step3 校准\n异常图 S(p)\n(unavailable)")

    # 校准增益图
    if "calibration_gain" in step3_result:
        gain_arr = step3_result["calibration_gain"]
        gain_heatmap = arr_to_heatmap(gain_arr)
        ax = fig.add_subplot(gs[3, 1])
        ax.set_facecolor("#2a2a2a")
        ax.imshow(gain_heatmap, aspect="equal")
        vmin, vmax = float(gain_arr.min()), float(gain_arr.max())
        ax.set_title(f"Step3 校准\n增益 G(p)\n[{vmin:.3f}~{vmax:.3f}]",
                     fontsize=7.5, color="white", pad=3)
        ax.axis("off")
        cbar = plt.colorbar(
            matplotlib.cm.ScalarMappable(
                cmap="jet",
                norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
            ),
            ax=ax, fraction=0.046, pad=0.04,
        )
        cbar.ax.tick_params(colors="white", labelsize=6)
        cbar.outline.set_edgecolor("white")
    elif n_cols > 1:
        arr = np.zeros_like(maps["recon_error"])
        heat_cell(3, 1, arr, "Step3 校准\n增益 G(p)\n(unavailable)",
                  show_cbar=False)

    # 可靠性 Mask
    if "energy_reliability_mask" in step3_result and n_cols > 2:
        binary_cell(3, 2, step3_result["energy_reliability_mask"],
                    "Step2 可靠\nMask Q(p)")
    elif "energy_reliability_mask" in step1_result and n_cols > 2:
        binary_cell(3, 2, step1_result["energy_reliability_mask"],
                    "Step2 可靠\nMask Q(p)")

    # GT 列（如果有）
    if gt is not None:
        gt_col = n_cols - 1
        # 在每行最后一个位置添加一个 GT 小图
        for row in range(4):
            ax = fig.add_subplot(gs[row, gt_col])
            ax.set_facecolor("#2a2a2a")
            if row == 3:
                ax.imshow(gt, cmap="Reds")
                ax.set_title("Ground\nTruth", fontsize=7.5, color="white", pad=3)
            else:
                ax.imshow(gt, cmap="gray")
                ax.set_title("GT", fontsize=7, color="white", pad=3)
            ax.axis("off")

    # ── 保存 ────────────────────────────────────────────────────
    if save_path:
        plt.savefig(
            save_path, dpi=fig_dpi, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"[保存] {save_path}")
    plt.close(fig)


def print_pipeline_summary(
    maps: Dict[str, np.ndarray],
    step1_result: Dict[str, np.ndarray],
    reliability: Dict[str, Dict],
    step3_result: Dict[str, np.ndarray],
    sample_idx: int,
) -> None:
    """打印 Step1~Step3 流水线处理摘要。"""
    print(f"\n{'=' * 65}")
    print(f"  Sample #{sample_idx:04d}  —  Step1~Step3 Pipeline Summary")
    print(f"{'=' * 65}")

    # Step2 可靠性
    print(f"\n  [Step2] 能量层可靠性:")
    for layer in ["energy_layer1", "energy_layer2", "energy_layer3"]:
        if layer not in maps:
            continue
        rel = reliability.get(layer, {})
        q = rel.get("q", 0.0)
        valid = rel.get("valid", False)
        status = "VALID" if valid else "DISCARD"
        print(
            f"    {layer}: q={q:.1f} | {status} | "
            f"CC={rel.get('total_components', 0)} | "
            f"max_area={rel.get('max_area', 0)} | "
            f"area_ratio={rel.get('total_area_ratio', 0):.4f} | "
            f"small_ratio={rel.get('small_ratio', 0):.3f}"
        )

    # Step3 校准统计
    if "recon_error_calibrated" in step3_result:
        orig = maps.get("recon_error")
        proc = step3_result["recon_error_calibrated"]
        if orig is not None and proc is not None:
            print(f"\n  [Step3] 校准前后统计:")
            print(
                f"    原始 recon:  mean={float(orig.mean()):.4f}  "
                f"std={float(orig.std()):.4f}  max={float(orig.max()):.4f}"
            )
            print(
                f"    Step3 校准:  mean={float(proc.mean()):.4f}  "
                f"std={float(proc.std()):.4f}  max={float(proc.max()):.4f}"
            )
    if "calibration_gain" in step3_result:
        gain = step3_result["calibration_gain"]
        print(
            f"\n  [Step3] 增益 G(p): "
            f"min={float(gain.min()):.4f}  "
            f"max={float(gain.max()):.4f}  "
            f"mean={float(gain.mean()):.4f}"
        )
    if "energy_reliability_mask" in step3_result:
        q_mask = step3_result["energy_reliability_mask"]
        nonzero = int(np.count_nonzero(q_mask))
        total = q_mask.size
        print(
            f"  [Step2] 可靠区域覆盖: {nonzero}/{total} "
            f"({100 * nonzero / total:.1f}%)"
        )
    print(f"{'=' * 65}\n")


# ==============================================================
#                        命令行接口
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="创新点二 Step1~Step3 流水线可视化脚本（cv2 JET 热力图）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单样本可视化（默认参数）
  python viz_step123.py --sample_idx 20 --save

  # 指定数据目录并显示
  python viz_step123.py --sample_idx 20 --sample_dir testSet/bottle/n_bottle_a_0_s_111 --show

  # 批量处理保存到指定目录
  python viz_step123.py --range 0 10 --save --output_dir output/step123_batch

  # 自定义 Step1 参数
  python viz_step123.py --sample_idx 20 --morph_threshold 0.5 --morph_kernel 5 --save

  # 自定义 Step2 参数
  python viz_step123.py --sample_idx 20 --area_min 100 --small_ratio_max 0.5 --save

  # 自定义 Step3 参数（消融实验）
  python viz_step123.py --sample_idx 20 --lambda_enhance 0.3 --lambda_suppress 0.1 --use_suppression --save
        """,
    )
    parser.add_argument(
        "--sample_dir", type=str,
        default="testSet/bottle/n_bottle_a_0_s_111",
        help="样本目录路径",
    )
    parser.add_argument(
        "--sample_idx", type=int, default=None,
        help="单个样本编号（0起始）",
    )
    parser.add_argument(
        "--compare_idx", type=int, nargs=2, default=None,
        help="对比两个样本（如 20 30）",
    )
    parser.add_argument(
        "--range", type=int, nargs=2, default=None,
        metavar=("START", "STOP"),
        help="批量范围，左闭右开。例如 --range 0 10 → 处理 0~9",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="实时显示图片（需图形界面支持）",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="保存图片到 output_dir",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/step123_vis",
        help="输出目录",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="输出图片 DPI，默认 150",
    )
    parser.add_argument(
        "--no_gt", action="store_true",
        help="不显示 GT 叠加",
    )

    # ── Step1 参数 ──
    s1 = parser.add_argument_group("Step1 (形态学滤波 Rule5) 参数")
    s1.add_argument(
        "--morph_threshold", type=float, default=0.6,
        help="二值化阈值（相对值），默认 0.6",
    )
    s1.add_argument(
        "--morph_kernel", type=int, default=3,
        help="形态学卷积核大小，默认 3",
    )
    s1.add_argument(
        "--morph_operation", type=str, default="both",
        choices=["open", "close", "both"],
        help="形态学操作类型，默认 both（开+闭）",
    )

    # ── Step2 参数 ──
    s2 = parser.add_argument_group("Step2 (连通域可靠性分析) 参数")
    s2.add_argument(
        "--area_min", type=int, default=50,
        help="最大连通域面积下限，默认 50",
    )
    s2.add_argument(
        "--small_thresh", type=int, default=50,
        help="小连通域面积阈值，默认 50",
    )
    s2.add_argument(
        "--small_ratio_max", type=float, default=0.6,
        help="小连通域占比上限，默认 0.6",
    )
    s2.add_argument(
        "--area_min_ratio", type=float, default=0.001,
        help="总面积比下限，默认 0.001",
    )
    s2.add_argument(
        "--area_max_ratio", type=float, default=0.35,
        help="总面积比上限，默认 0.35",
    )
    s2.add_argument(
        "--connectivity", type=int, default=8,
        help="连通性（4或8），默认 8",
    )

    # ── Step3 参数 ──
    s3 = parser.add_argument_group("Step3 (可靠性能量引导校准 Rule14) 参数")
    s3.add_argument(
        "--lambda_enhance", type=float, default=0.25,
        help="保守增强系数 λ，默认 0.25",
    )
    s3.add_argument(
        "--lambda_suppress", type=float, default=0.0,
        help="软抑制系数（消融用），默认 0.0",
    )
    s3.add_argument(
        "--use_suppression", action="store_true",
        help="启用软抑制（默认关闭）",
    )
    s3.add_argument(
        "--use_energy_diff", action="store_true", default=True,
        help="优先使用能量差图 ΔE（默认 True）",
    )
    s3.add_argument(
        "--no_energy_diff", action="store_true",
        help="禁用能量差图，回退使用普通能量图",
    )

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
    print(
        f"[INFO] 共 {dataset.num_samples} 个样本，"
        f"编号: {dataset.indices[0]} ~ {dataset.indices[-1]}"
    )

    if args.output_dir:
        out_dir = resolve_path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 输出目录: {out_dir}")

    # 确定处理范围
    if args.sample_idx is not None:
        indices = [args.sample_idx]
    elif args.compare_idx is not None:
        indices = args.compare_idx
    elif args.range is not None:
        indices = list(range(args.range[0], args.range[1]))
    else:
        indices = dataset.indices[:3]
        print(f"[INFO] 未指定样本，默认分析: {indices}")

    use_energy_diff = args.use_energy_diff and not args.no_energy_diff

    for pos, idx in enumerate(indices, 1):
        if idx not in dataset.indices:
            print(f"[WARN] 编号 {idx} 不存在，跳过")
            continue

        label = dataset.get_label(idx)
        gt = dataset.get_gt(idx) if not args.no_gt else None
        maps = dataset.get_all_maps(idx)

        print(f"\n{'=' * 65}")
        print(
            f"  样本 #{idx:04d}  |  标签={label}  |  "
            f"GT={'有' if gt is not None else '无'}  [{pos}/{len(indices)}]"
        )
        print(f"{'=' * 65}")

        if not maps:
            print(f"[WARN] 样本 {idx} 无可用 map，跳过")
            continue

        # ── Step1: 形态学滤波 ───────────────────────────────────
        print(f"  [Step1] morphological_filter ...")
        step1_result = step1_morphological_filter(
            maps,
            threshold=args.morph_threshold,
            kernel_size=args.morph_kernel,
            operation=args.morph_operation,
        )

        # ── Step2: 连通域可靠性分析 ──────────────────────────────
        print(f"  [Step2] energy_reliability ...")
        step1_with_rel, reliability = step2_energy_reliability(
            step1_result,
            area_min=args.area_min,
            small_area_threshold=args.small_thresh,
            small_ratio_max=args.small_ratio_max,
            total_area_min_ratio=args.area_min_ratio,
            total_area_max_ratio=args.area_max_ratio,
            connectivity=args.connectivity,
        )

        # ── Step3: 可靠性能量引导校准 ────────────────────────────
        print(f"  [Step3] reliability_calibrate ...")
        step3_result = step3_reliability_calibrate(
            step1_with_rel,
            lambda_enhance=args.lambda_enhance,
            lambda_suppress=args.lambda_suppress,
            use_suppression=args.use_suppression,
            use_energy_diff=use_energy_diff,
        )

        print(f"  Step3 完成: λ_enhance={args.lambda_enhance}, "
              f"λ_suppress={args.lambda_suppress}, "
              f"use_suppression={args.use_suppression}, "
              f"use_energy_diff={use_energy_diff}")

        # ── 打印统计摘要 ────────────────────────────────────────
        print_pipeline_summary(maps, step1_result, reliability, step3_result, idx)

        # ── 可视化 ──────────────────────────────────────────────
        save_path = None
        if args.save and args.output_dir:
            save_path = str(out_dir / f"sample_{idx:04d}_step123.png")

        plot_pipeline_cv2_heatmap(
            maps=maps,
            step1_result=step1_result,
            reliability=reliability,
            step3_result=step3_result,
            sample_idx=idx,
            gt=gt,
            save_path=save_path,
            fig_dpi=args.dpi,
        )

        print(f"  可视化完成{'，已保存' if save_path else '（未保存）'}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
