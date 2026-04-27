"""
==============================================================
指标计算工具类 —— AnomalyMetrics
==============================================================

对标 util/test.py 中的三个核心指标计算函数：
  - Image-AUROC（图像级异常分类AUC）
  - Pixel-AUROC（像素级异常分割AUC）
  - PRO        （per-region overlap，区域级重叠率）

所有函数封装在 AnomalyMetrics 类中，可直接实例化后反复调用。

用法示例：
    calc = AnomalyMetrics(topk=100)
    metrics = calc.compute(map_np, gt_np)         # 计算单个 map
    metrics = calc.compute_multi(maps_dict, gt_np)  # 批量计算多个 map
    calc.print_table(metrics)                       # 打印汇总表
==============================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from skimage import measure
from statistics import mean
from typing import Dict, List, Optional, Tuple


class AnomalyMetrics:
    """
    统一封装 Image-AUROC、Pixel-AUROC、PRO 三个指标的计算逻辑。

    用法：
        calc = AnomalyMetrics(topk=100)
        results = calc.compute(anomaly_map, gt_mask)
        # results = {'Image-AUROC': 0.xxx, 'Pixel-AUROC': 0.xxx, 'PRO': 0.xxx}

        # 批量计算多个 map
        maps = {
            'recon':   recon_maps,
            'energy':  energy_maps,
            'fused':  fused_maps,
        }
        results = calc.compute_multi(maps, gt_masks)
        calc.print_table(results)
    """

    def __init__(self, topk: int = 100, pro_num_th: int = 200):
        """
        参数:
            topk:       计算 Image-level score 时取每张图 topk 像素的均值作为图像得分。
            pro_num_th: PRO 曲线采样的阈值数量。
        """
        self.topk = topk
        self.pro_num_th = pro_num_th

    # ==========================================================
    #  核心计算函数
    # ==========================================================

    def compute_image_auroc(
        self,
        amap: np.ndarray,
        gt_binary: np.ndarray,
    ) -> float:
        """
        计算 Image-level AUROC。

        参数:
            amap:     np.ndarray [N, H, W] 异常得分图
            gt_binary: np.ndarray [N, H, W] Ground Truth 二值掩码（0/1）

        返回:
            float, Image-AUROC 值
        """
        N = amap.shape[0]
        # 每张图取 topk 像素均值作为图像得分
        sample_scores = np.array([
            np.sort(amap[i].reshape(-1))[-min(self.topk, amap[i].size) :].mean()
            for i in range(N)
        ])
        # 图像级标签：有异常区域则为 1
        sample_labels = (gt_binary.max(axis=(1, 2)) > 0).astype(int)
        if len(np.unique(sample_labels)) < 2:
            return 0.5
        return roc_auc_score(sample_labels, sample_scores)

    def compute_pixel_auroc(
        self,
        amap: np.ndarray,
        gt_binary: np.ndarray,
    ) -> float:
        """
        计算 Pixel-level AUROC。

        参数:
            amap:     np.ndarray [N, H, W] 异常得分图
            gt_binary: np.ndarray [N, H, W] Ground Truth 二值掩码（0/1）

        返回:
            float, Pixel-AUROC 值
        """
        gt_flat = gt_binary.astype(int).reshape(-1)
        score_flat = amap.reshape(-1)
        if len(np.unique(gt_flat)) < 2:
            return 0.5
        return roc_auc_score(gt_flat, score_flat)

    def compute_pro(
        self,
        amap: np.ndarray,
        gt_binary: np.ndarray,
    ) -> float:
        """
        计算 Per-Region Overlap (PRO)，对齐 util/test.py 的 compute_pro 逻辑。

        在 self.pro_num_th 个阈值下：
            1. 将 amap 二值化得到 binary_amaps
            2. 对每个 GT 连通区域，计算该区域被覆盖的比率
            3. 计算 FPR（误报率）
            4. 在 FPR ∈ [0, 0.3] 区间内对 PRO-FPR 曲线积分得到 PRO-AUC

        参数:
            amap:     np.ndarray [N, H, W] 异常得分图
            gt_binary: np.ndarray [N, H, W] Ground Truth 二值掩码

        返回:
            float, PRO 分数（0~1 之间）
        """
        amap = np.asarray(amap, dtype=np.float64)
        gt_binary = np.asarray(gt_binary, dtype=np.int32)

        assert amap.shape == gt_binary.shape
        assert set(gt_binary.flatten()) == {0, 1}

        min_th = float(amap.min())
        max_th = float(amap.max())
        if max_th - min_th < 1e-8:
            return 0.0
        delta = (max_th - min_th) / self.pro_num_th

        rows = []
        for th in np.arange(min_th, max_th, delta):
            binary_amaps = (amap > th).astype(np.bool_)

            pros = []
            for bi in range(gt_binary.shape[0]):
                mask = gt_binary[bi]
                binary_amap = binary_amaps[bi]
                for region in measure.regionprops(measure.label(mask)):
                    coords = region.coords
                    tp = binary_amap[coords[:, 0], coords[:, 1]].sum()
                    pros.append(tp / region.area)

            inverse_masks = 1 - gt_binary
            fp = int(np.logical_and(inverse_masks, binary_amaps).sum())
            total_neg = int(inverse_masks.sum())
            fpr = float(fp / total_neg) if total_neg > 0 else 0.0
            rows.append({"pro": mean(pros) if pros else 0.0, "fpr": fpr, "threshold": float(th)})

        df = pd.DataFrame(rows)
        df = df[df["fpr"] < 0.3]
        if df["fpr"].max() > 0:
            df["fpr"] = df["fpr"] / df["fpr"].max()

        if df.empty:
            return 0.0

        from sklearn.metrics import auc
        return float(auc(df["fpr"].values, df["pro"].values))

    # ==========================================================
    #  高级封装：一次性计算三个指标
    # ==========================================================

    def compute(
        self,
        amap: np.ndarray,
        gt: np.ndarray,
    ) -> Dict[str, float]:
        """
        对单个异常图计算 Image-AUROC、Pixel-AUROC、PRO。

        参数:
            amap: np.ndarray [N, H, W] 异常得分图
            gt:   np.ndarray [N, H, W] Ground Truth 掩码（0 或 1，float 或 int）

        返回:
            dict: {'Image-AUROC': float, 'Pixel-AUROC': float, 'PRO': float}
        """
        gt_binary = self._to_binary(gt)

        image_auroc = self.compute_image_auroc(amap, gt_binary)
        pixel_auroc = self.compute_pixel_auroc(amap, gt_binary)
        pro_score = self.compute_pro(amap, gt_binary)

        return {
            "Image-AUROC": round(image_auroc, 6),
            "Pixel-AUROC": round(pixel_auroc, 6),
            "PRO": pro_score,
        }

    def compute_multi(
        self,
        maps_dict: Dict[str, np.ndarray],
        gt: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        批量对多个 map 计算指标。

        参数:
            maps_dict: dict of {name: amap}，例如 {'recon': recon_map, 'fused': fused_map}
            gt:        np.ndarray [N, H, W] Ground Truth 掩码

        返回:
            dict of {name: {'Image-AUROC': ..., 'Pixel-AUROC': ..., 'PRO': ...}}
        """
        results = {}
        for name, amap in maps_dict.items():
            results[name] = self.compute(amap, gt)
        return results

    # ==========================================================
    #  辅助 & 打印工具
    # ==========================================================

    @staticmethod
    def _to_binary(gt: np.ndarray) -> np.ndarray:
        """将 GT 转为 0/1 二值数组。"""
        gt_b = gt.copy()
        gt_b[gt_b > 0.5] = 1
        gt_b[gt_b <= 0.5] = 0
        return gt_b

    @staticmethod
    def print_table(
        results: Dict[str, Dict[str, float]],
        title: str = "Anomaly Detection Metrics",
    ) -> pd.DataFrame:
        """
        打印格式化的指标表格，并返回 DataFrame。

        参数:
            results: compute_multi() 或 compute() 的返回值
        """
        df = pd.DataFrame(results).T
        df.index.name = "Map"
        df = df.reset_index()

        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(
            df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        print(f"{'=' * 60}\n")
        return df

    @staticmethod
    def auroc_only(
        amap: np.ndarray,
        gt: np.ndarray,
    ) -> Tuple[float, float]:
        """
        快速计算 Image-AUROC 和 Pixel-AUROC（不含 PRO），用于快速调试。

        参数:
            amap: np.ndarray [N, H, W]
            gt:   np.ndarray [N, H, W]

        返回:
            (Image-AUROC, Pixel-AUROC)
        """
        calc = AnomalyMetrics()
        gt_b = calc._to_binary(gt)
        img_auc = calc.compute_image_auroc(amap, gt_b)
        pix_auc = calc.compute_pixel_auroc(amap, gt_b)
        return round(img_auc, 6), round(pix_auc, 6)

    @staticmethod
    def compare(
        results_before: Dict[str, float],
        results_after: Dict[str, float],
    ) -> None:
        """
        打印处理前后的指标对比表（差值 = after - before）。

        参数:
            results_before: 处理前指标 dict
            results_after:  处理后指标 dict
        """
        print(f"\n{'=' * 65}")
        print(f"  {'Metric':<15} {'Before':>12} {'After':>12} {'Delta':>12} {'%Change':>10}")
        print(f"{'-' * 65}")
        for key in results_before:
            b = results_before[key]
            a = results_after.get(key, b)
            delta = a - b
            pct = (delta / b * 100) if b != 0 else 0.0
            sign = "+" if delta > 0 else ""
            print(
                f"  {key:<15} {b:>12.4f} {a:>12.4f} {sign}{delta:>11.4f} {pct:>+9.2f}%"
            )
        print(f"{'=' * 65}\n")


# ==============================================================
#                     直接调用入口（命令行用法）
# ==============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用 AnomalyMetrics 计算指标")
    parser.add_argument("--npy_dir", type=str, required=True, help=".npy 文件目录")
    parser.add_argument("--gt_name", type=str, default="gt.npy", help="GT 文件名")
    parser.add_argument("--map_prefix", type=str, default="", help="map 文件名前缀过滤")
    parser.add_argument("--topk", type=int, default=100, help="Image-level topk")
    args = parser.parse_args()

    import os
    from glob import glob

    npy_dir = args.npy_dir
    gt_path = os.path.join(npy_dir, args.gt_name)
    map_files = sorted(glob(os.path.join(npy_dir, f"{args.map_prefix}*.npy")))
    map_files = [f for f in map_files if os.path.basename(f) != args.gt_name]

    if not map_files:
        print(f"[ERROR] 在 {npy_dir} 中未找到任何 .npy map 文件")
        exit(1)

    gt = np.load(gt_path) if os.path.exists(gt_path) else None
    maps_dict = {}

    for f in map_files:
        name = os.path.splitext(os.path.basename(f))[0]
        maps_dict[name] = np.load(f)

    if gt is not None:
        calc = AnomalyMetrics(topk=args.topk)
        results = calc.compute_multi(maps_dict, gt)
        calc.print_table(results)
    else:
        print("[WARN] 未找到 GT 文件，仅打印各 map 的统计信息")
        for name, arr in maps_dict.items():
            print(f"  {name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")
