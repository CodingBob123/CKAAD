"""
==============================================================
testSet/bottle 能量图层 & 重建误差 分析与后处理脚本
==============================================================

功能：
  1. 按编号批量读取同一张图像的所有 map（energy_layer1/2/3, recon_error）
  2. 对单个编号的 map 进行拼接可视化对比
  3. 实现多种后处理规则来降低伪影
  4. 输出处理前后的对比图与统计报告

用法示例：
  python analyze_maps.py --sample_idx 20 --show
  python analyze_maps.py --sample_idx 20 --apply_rules all --save
  python analyze_maps.py --sample_idx 20 --apply_rules threshold --params threshold=0.3 --save
  python analyze_maps.py --compare_idx 20 50 --save
==============================================================
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
# ==============================================================
#                        数据加载器
# ==============================================================

class MapDataset:
    """
    读取 testSet/bottle 下单个样本目录中的所有 map 文件。

    目录结构：
        sample_dir/
            energy_layer1/
                energy_layer1_0000.npy
                ...
            energy_layer2/
                energy_layer2_0000.npy
                ...
            energy_layer3/
            recon_error/
            overlay/
            metadata.npy
    """

    MAP_TYPES = ["energy_layer1", "energy_layer2", "energy_layer3", "recon_error"]

    def __init__(self, sample_dir: str):
        self.sample_dir = Path(sample_dir)
        if not self.sample_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.sample_dir}")

        self._index_map: Dict[int, Dict[str, np.ndarray]] = {}
        self._metadata: Optional[dict] = None
        self._scan_files()

    def _scan_files(self):
        """扫描目录，构建 index -> {map_type: path} 的映射。"""
        for map_type in self.MAP_TYPES:
            map_dir = self.sample_dir / map_type
            if not map_dir.exists():
                continue
            for f in map_dir.glob(f"{map_type}_*.npy"):
                idx = self._parse_index(f.name)
                if idx is not None:
                    if idx not in self._index_map:
                        self._index_map[idx] = {}
                    self._index_map[idx][map_type] = f

        # 加载 metadata
        meta_path = self.sample_dir / "metadata.npy"
        if meta_path.exists():
            self._metadata = np.load(meta_path, allow_pickle=True).item()

    @staticmethod
    def _parse_index(filename: str) -> Optional[int]:
        """从文件名中提取编号，如 energy_layer1_0023.npy -> 23"""
        m = re.search(r"_(\d+)\.npy$", filename)
        return int(m.group(1)) if m else None

    @property
    def indices(self) -> List[int]:
        """所有可用的编号。"""
        return sorted(self._index_map.keys())

    @property
    def num_samples(self) -> int:
        return len(self._index_map)

    def get_map(self, idx: int, map_type: str) -> Optional[np.ndarray]:
        """
        读取指定编号和 map 类型的 .npy 文件。
        """
        path = self._index_map.get(idx, {}).get(map_type)
        if path is None:
            return None
        return np.load(path)

    def get_all_maps(self, idx: int) -> Dict[str, np.ndarray]:
        """
        读取指定编号的所有可用 map。
        返回 dict: {map_type: array}
        """
        result = {}
        for map_type in self.MAP_TYPES:
            arr = self.get_map(idx, map_type)
            if arr is not None:
                result[map_type] = arr
        return result

    def get_label(self, idx: int) -> Optional[int]:
        """返回该编号对应的标签（从 metadata 获取）。"""
        if self._metadata is None:
            return None
        indices = self._metadata.get("indices", [])
        labels = self._metadata.get("labels", [])
        if not indices:
            return None
        try:
            pos = indices.index(idx)
            return labels[pos]
        except ValueError:
            return None

    def get_gt(self, idx: int) -> Optional[np.ndarray]:
        """返回该编号对应的 ground truth mask。"""
        if self._metadata is None:
            return None
        indices = self._metadata.get("indices", [])
        gts = self._metadata.get("gts", None)
        if not indices or gts is None:
            return None
        try:
            pos = indices.index(idx)
            return gts[pos, 0]  # shape: (H, W)
        except (ValueError, IndexError):
            return None

    def get_stats(self, idx: int) -> Dict[str, Dict[str, float]]:
        """返回各 map 的统计信息（min, max, mean, std）。"""
        maps = self.get_all_maps(idx)
        stats = {}
        for name, arr in maps.items():
            stats[name] = {
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
            }
        return stats


# ==============================================================
#                        后处理规则
# ==============================================================

class PostProcessRules:
    """
    各种后处理规则，用于降低能量图 / 重建误差图中的伪影。
    每个规则都接受一个 dict {map_type: ndarray}，返回处理后的 dict。
    """

    # ──────────────────────────────────────────────────────────
    # 规则 1：全局阈值裁剪（去除极端值噪声）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def clip_by_percentile(
        maps: Dict[str, np.ndarray],
        low_pct: float = 1.0,
        high_pct: float = 99.0,
    ) -> Dict[str, np.ndarray]:
        result = {}
        for name, arr in maps.items():
            lo = np.percentile(arr, low_pct)
            hi = np.percentile(arr, high_pct)
            result[name] = np.clip(arr, lo, hi)
        return result

    # ──────────────────────────────────────────────────────────
    # 规则 2：能量图层的加权融合（融合多层能量信息）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def fuse_energy_layers(
        maps: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        if weights is None:
            weights = {"energy_layer1": 0.2, "energy_layer2": 0.3, "energy_layer3": 0.5}

        result = {k: v.copy() for k, v in maps.items()}

        # 构建融合图（仅能量层）
        fuse = None
        for layer in ["energy_layer1", "energy_layer2", "energy_layer3"]:
            if layer in maps:
                norm = (maps[layer] - maps[layer].min()) / (maps[layer].max() - maps[layer].min() + 1e-8)
                w = weights.get(layer, 1.0)
                if fuse is None:
                    fuse = w * norm
                else:
                    fuse += w * norm

        if fuse is not None:
            result["energy_fused"] = fuse

        return result

    # ──────────────────────────────────────────────────────────
    # 规则 3：高斯平滑（去除高频噪声）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def gaussian_smooth(
        maps: Dict[str, np.ndarray],
        sigma: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            print("警告: scipy 未安装，跳过高斯平滑")
            return maps

        result = {}
        for name, arr in maps.items():
            result[name] = gaussian_filter(arr, sigma=sigma)
        return result

    # ──────────────────────────────────────────────────────────
    # 规则 4：双边滤波（保边去噪）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def bilateral_filter(
        maps: Dict[str, np.ndarray],
        sigma_color: float = 0.1,
        sigma_spatial: float = 5,
    ) -> Dict[str, np.ndarray]:
        try:
            from skimage.restoration import denoise_bilateral
        except ImportError:
            print("警告: skimage 未安装，跳过双边滤波")
            return maps

        result = {}
        for name, arr in maps.items():
            # 先归一化到 [0, 1]，再双边滤波
            arr_min, arr_max = arr.min(), arr.max()
            arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8)
            denoised = denoise_bilateral(arr_norm, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
            result[name] = denoised * (arr_max - arr_min) + arr_min
        return result

    # ──────────────────────────────────────────────────────────
    # 规则 5：形态学操作（开运算去除小噪点，闭运算填充孔洞）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def morphological_filter(
        maps: Dict[str, np.ndarray],
        threshold: float = None,
        kernel_size: int = 3,
        operation: str = "both",  # 取值: "open" | "close" | "both"
    ) -> Dict[str, np.ndarray]:
        try:
            from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
        except ImportError:
            print("警告: scipy 未安装，跳过形态学操作")
            return maps

        result = {}
        struct = generate_binary_structure(2, 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)

        for name, arr in maps.items():
            # 步骤1：降维（去掉单维度，如 (H,W,1) → (H,W)）
            arr_2d = np.squeeze(arr)  # 关键修复：移除多余维度
            if arr_2d.ndim != 2:
                print(f"警告: {name} 不是2维数组（shape={arr.shape}），跳過形态学操作")
                result[name] = arr
                continue

            # 步骤2：根据阈值二值化
            if threshold is None:
                t = arr_2d.mean() + arr_2d.std()
            else:
                t = arr_2d.min() + threshold * (arr_2d.max() - arr_2d.min())

            binary = arr_2d > t

            # 步骤3：确保结构元素维度和输入一致（2维）
            if struct.ndim != binary.ndim:
                struct = generate_binary_structure(binary.ndim, 1)

            if operation in ("open", "both"):
                binary = binary_opening(binary, structure=struct)
            if operation in ("close", "both"):
                binary = binary_closing(binary, structure=struct)

            # 步骤4：用处理后的 mask 对原图做加权（还原原维度）
            mask = binary.astype(float)
            if arr.ndim == 3:  # 若原数组是3维，mask 扩维
                mask = np.expand_dims(mask, axis=-1)
            result[name] = arr * mask
            result[f"{name}_morph_mask"] = mask

        return result

    # ──────────────────────────────────────────────────────────
    # 规则 6：动态阈值（结合局部均值与全局统计）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def adaptive_threshold(
        maps: Dict[str, np.ndarray],
        window_size: int = 15,
        multiplier: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        try:
            from scipy.ndimage import uniform_filter
        except ImportError:
            print("警告: scipy 未安装，跳过自适应阈值")
            return maps

        result = {}
        for name, arr in maps.items():
            local_mean = uniform_filter(arr, size=window_size)
            local_sq_mean = uniform_filter(arr ** 2, size=window_size)
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

            threshold_map = local_mean + multiplier * local_std
            masked = np.where(arr > threshold_map, arr, 0)
            result[name] = masked
            result[f"{name}_adaptive_mask"] = (arr > threshold_map).astype(float)

        return result

    # ──────────────────────────────────────────────────────────
    # 规则 7：重建误差与能量图交叉抑制（能量高但重建误差低的区域 = 伪影）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def cross_suppress_artifact(
        maps: Dict[str, np.ndarray],
        energy_threshold_quantile: float = 0.9,
        recon_threshold_quantile: float = 0.3,
        suppression_factor: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """
        思路：
          - 能量图层的高值区域通常指示结构/特征敏感区域
          - 重建误差图的高值区域通常指示异常区域
          - 伪影特征：能量高 但 重建误差低 → 说明不是真正的异常，只是模型对某些结构的过度敏感
          - 对这类区域进行抑制
        """
        result = {k: v.copy() for k, v in maps.items()}

        energy_layers = [v for k, v in maps.items() if "energy_layer" in k]
        if not energy_layers:
            return result

        # 取多层能量的平均
        energy_mean = np.mean(energy_layers, axis=0)
        energy_norm = (energy_mean - energy_mean.min()) / (energy_mean.max() - energy_mean.min() + 1e-8)

        recon = maps.get("recon_error")
        if recon is None:
            return result

        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

        # 找到伪影区域：能量高（>高阈值），重建误差低（<低阈值）
        energy_hi_mask = energy_norm > np.percentile(energy_norm, energy_threshold_quantile * 100)
        recon_lo_mask = recon_norm < np.percentile(recon_norm, recon_threshold_quantile * 100)
        artifact_mask = energy_hi_mask & recon_lo_mask

        # 伪影区域在重建误差中做抑制
        suppressed = recon.copy()
        suppressed[artifact_mask] *= suppression_factor
        result["recon_error"] = suppressed
        result["artifact_mask"] = artifact_mask.astype(float)

        return result

    # ──────────────────────────────────────────────────────────
    # 规则 8：边缘感知抑制（与 GT 边缘重叠区域可能是伪影）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def edge_aware_suppress(
        maps: Dict[str, np.ndarray],
        gt: Optional[np.ndarray] = None,
        edge_weight: float = 0.5,
        suppress_radius: int = 3,
    ) -> Dict[str, np.ndarray]:
        if gt is None:
            return maps

        try:
            from scipy.ndimage import binary_dilation, distance_transform_edt
        except ImportError:
            print("警告: scipy 未安装，跳过边缘感知抑制")
            return maps

        result = {k: v.copy() for k, v in maps.items()}

        # 提取 GT 边缘
        gt_edges = gt > 0
        # 膨胀边缘区域
        dilated = binary_dilation(gt_edges, iterations=suppress_radius)

        # 边缘区域权重衰减
        for name in ["recon_error", "energy_layer1", "energy_layer2", "energy_layer3"]:
            if name in maps:
                edge_mask = dilated.astype(float)
                # 非边缘区域保持 1，边缘区域降到 edge_weight
                weight_map = np.where(edge_mask > 0, edge_weight, 1.0)
                result[name] = maps[name] * weight_map

        return result

    # ──────────────────────────────────────────────────────────
    # 规则 9：按能量响应归一化重建误差（抑制正常区域，保留异常区域）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def energy_normalized_recon(
        maps: Dict[str, np.ndarray],
        layer_weights: Optional[Dict[str, float]] = None,
        quantile_clip: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        result = {k: v.copy() for k, v in maps.items()}

        if layer_weights is None:
            layer_weights = {"energy_layer1": 0.25, "energy_layer2": 0.35, "energy_layer3": 0.4}

        # 加权融合能量层
        energy_fused = None
        for layer, w in layer_weights.items():
            if layer in maps:
                norm = (maps[layer] - maps[layer].min()) / (maps[layer].max() - maps[layer].min() + 1e-8)
                energy_fused = (energy_fused * w / (sum(layer_weights.values()) + 1e-8) + norm * w) \
                    if energy_fused is not None else norm * w

        if energy_fused is None or "recon_error" not in maps:
            return result

        recon = maps["recon_error"]

        # 用能量响应作为因子：能量低（正常区域）→ 抑制误差，能量高（敏感区域）→ 保持误差
        energy_factor = (energy_fused - energy_fused.min()) / (energy_fused.max() - energy_fused.min() + 1e-8)
        # 抑制低能量区的误差
        normalized_recon = recon * (1.0 + energy_factor)

        # 裁剪极端值
        hi = np.percentile(normalized_recon, quantile_clip * 100)
        normalized_recon = np.clip(normalized_recon, 0, hi)

        result["recon_error_energy_norm"] = normalized_recon
        return result

    # ──────────────────────────────────────────────────────────
    # 规则 10：热力图 60% 分位阈值（二值化）  是规则5的子集，后续常用规则5了
    #   - 计算重建误差图的 60% 分位数作为阈值
    #   - 高于该阈值 → 保持不变
    #   - 低于该阈值 → 像素值置为 0
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def percentile_threshold(
        maps: Dict[str, np.ndarray],
        target_map: str = "recon_error",
        quantile: float = 0.60,
    ) -> Dict[str, np.ndarray]:
        result = {k: v.copy() for k, v in maps.items()}

        arr = maps.get(target_map)
        if arr is None:
            print(f"警告: 未找到目标 map '{target_map}'，跳过 percentile_threshold")
            return result

        arr_min, arr_max = arr.min(), arr.max()
        threshold = arr_min + quantile * (arr_max - arr_min)
        # 高于阈值保持不变，低于阈值置为 0
        masked = np.where(arr > threshold, arr, 0.0)
        result[target_map] = masked
        result[f"{target_map}_p{int(quantile * 100)}_mask"] = (arr > threshold).astype(float)
        result[f"{target_map}_p{int(quantile * 100)}_threshold"] = np.array(threshold)

        return result

    # ──────────────────────────────────────────────────────────
    # 规则 11：连通域分析（统计小面积连通域，可选移除小连通域）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def connected_component_analysis(
        maps: Dict[str, np.ndarray],
        target_map: str = "recon_error",
        binary_threshold: Optional[float] = None,
        area_threshold: int = 50,
        remove_small_components: bool = True,
        connectivity: int = 8,
        skip_binarization: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        对指定map进行连通域分析，统计小面积连通域数量，并可选移除小连通域以降低伪影。
        当 target_map="all" 时，会对 maps 中所有可用的 2D map 逐一执行分析。
        统计信息会存入 maps 的 {target_map}_cc_stats 键中（单个map）或
        {map_name}_cc_stats 键中（target_map=all），包含：
        - small_component_count: 面积小于阈值的连通域数量
        - all_areas: 所有连通域（非背景）的面积列表
        - area_threshold: 使用的面积阈值
        - binary_threshold: 二值化使用的阈值

        参数：
            maps: 输入的map字典
            target_map: 要分析的目标map名称（如 recon_error），传 "all" 则遍历所有map
            binary_threshold: 二值化阈值（None则自动计算：mean + std）
            area_threshold: 连通域面积阈值（小于该值视为小连通域）
            remove_small_components: 是否移除小连通域（置0）
            connectivity: 连通域分析的连通性（4/8，默认8连通）
            skip_binarization: 是否跳过二值化步骤（True时假设输入已经是二值化的）
        """
        result = {k: v.copy() for k, v in maps.items()}

        # 确定要处理的 map 列表
        if target_map == "all":
            target_maps = [name for name, arr in maps.items()
                           if np.squeeze(arr).ndim == 2]
        else:
            target_maps = [target_map] if target_map in result else []

        if not target_maps:
            if target_map == "all":
                print(f"警告: 没有找到可用的 2D map，跳过连通域分析")
            else:
                print(f"警告: 未找到目标 map '{target_map}'，跳过连通域分析")
            return result

        for tmap in target_maps:
            arr = np.squeeze(result[tmap])
            if arr.ndim != 2:
                print(f"警告: {tmap} 不是2维数组（shape={result[tmap].shape}），跳过")
                continue

            # 步骤1：二值化（如果输入已经是二值化的，可跳过）
            if skip_binarization:
                binary_mask = (arr > 0).astype(np.uint8)
                bt = None
            else:
                if binary_threshold is None:
                    bt = arr.mean() + arr.std()
                else:
                    bt = arr.min() + binary_threshold * (arr.max() - arr.min())
                binary_mask = (arr > bt).astype(np.uint8)

            # 步骤2：连通域分析
            try:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary_mask,
                    connectivity=connectivity,
                    ltype=cv2.CV_32S
                )
            except Exception as e:
                print(f"警告: {tmap} 连通域分析失败 - {e}，跳过")
                continue

            # 步骤3：统计面积 & 小连通域数量
            small_component_count = 0
            all_areas = []
            for label_idx in range(1, num_labels):
                area = stats[label_idx][cv2.CC_STAT_AREA]
                all_areas.append(area)
                if area < area_threshold:
                    small_component_count += 1

            # 步骤4：（可选）移除小连通域
            if remove_small_components:
                keep_mask = np.zeros_like(binary_mask)
                for label_idx in range(1, num_labels):
                    area = stats[label_idx][cv2.CC_STAT_AREA]
                    if area >= area_threshold:
                        keep_mask[labels == label_idx] = 1
                processed_arr = arr * keep_mask
                if len(result[tmap].shape) == 3:
                    processed_arr = np.expand_dims(processed_arr, axis=-1)
                result[tmap] = processed_arr

            # 步骤5：记录统计信息
            cc_stats = {
                "small_component_count": small_component_count,
                "all_areas": all_areas,
                "area_threshold": area_threshold,
                "binary_threshold": bt,
                "total_components": num_labels - 1,
                "removed_small_components": remove_small_components,
            }
            result[f"{tmap}_cc_stats"] = cc_stats
            print(f"[连通域分析] {tmap} - 总连通域数: {num_labels-1} | 小连通域数(<{area_threshold}): {small_component_count}")

        return result

    # ==============================================================
    # 规则 13：形态学掩码可靠性评估与能量融合（Step2 合一版）
    # ==============================================================
    @staticmethod
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
    ) -> Dict[str, np.ndarray]:
        """
        规则13 = Step2 完整流程：对规则5生成的 *_morph_mask 做连通域分析 →
        用原始 mask 的统计做四重可靠性判据 → q=1 则清理并融合，q=0 则直接丢弃。

        核心逻辑（与规则11的 skip_binarization=True 模式对齐）：
          1. 对每个 *_morph_mask 做连通域分析（原始 mask，不做清理）
          2. 在原始统计上做四重可靠性判据，计算 q 值
          3. q=1 → 移除小连通域得到 cleaned_mask → 用 cleaned_mask 加权融合能量图
          4. q=0 → 该层直接丢弃，统计和 mask 均保留供诊断

        设计理念：原始 mask 的响应质量决定了该层是否可信。如果一张图本身就是
        杂乱噪声铺满，即使清理后看起来干净，其根基也不可信。

        输出键：
          - {layer}_cc_stats       : 原始 morph_mask 的连通域统计（q 判据基于此）
          - {layer}_cc_stats_cleaned : 清理后 cleaned_mask 的统计（仅 q=1 时有效）
          - {layer}_morph_cleaned  : 清理后的 binary mask（仅 q=1 时有效）
          - {layer}_morph_filtered : 清理后的能量图（小连通域区域置零，与规则11行为一致）
          - energy_reliability_dict : 每层 q 值及完整统计字典
          - energy_reliability_mask : 各层 cleaned_mask 取并集（仅 q=1 层）
          - energy_fused_reliable  : 最终融合结果
        """
        result = {k: v.copy() for k, v in maps.items()}
        reliability: Dict[str, Dict] = {}
        H, W = None, None

        # ── 第一步：逐层连通域分析 + 原始统计判 q ─────────────────
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

            try:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary, connectivity=connectivity, ltype=cv2.CV_32S,
                )
            except Exception as e:
                reliability[layer] = {"valid": False, "q": 0.0, "reason": f"cc_error: {e}"}
                continue

            # ── 原始 mask 连通域统计（q 判据基于此）────────────────
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

            # ── 四重可靠性判据（基于原始 mask 统计）──────────────
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

            # ── q=1 时才清理小连通域，并将清理结果写回 map ─────────
            if q == 1.0 and remove_small_components:
                keep_mask = np.zeros_like(binary)
                for label_idx in range(1, num_labels):
                    area = int(stats[label_idx][cv2.CC_STAT_AREA])
                    if area >= small_area_threshold:
                        keep_mask[labels == label_idx] = 1
                cleaned_mask = keep_mask

                # 清理后的 morph_mask（覆盖原 map，与规则11行为一致）
                e_layer = np.squeeze(maps[layer])
                e_cleaned = e_layer * keep_mask
                if maps[layer].ndim == 3:
                    e_cleaned = np.expand_dims(e_cleaned, axis=-1)
                result[f"{layer}_morph_filtered"] = e_cleaned.astype(np.float32)

                # 清理后重新统计（诊断用）
                num_labels_clean, _, stats_clean, _ = cv2.connectedComponentsWithStats(
                    cleaned_mask.astype(np.uint8), connectivity=connectivity, ltype=cv2.CV_32S,
                )
                areas_clean = []
                small_count_clean = 0
                for label_idx in range(1, num_labels_clean):
                    area = int(stats_clean[label_idx][cv2.CC_STAT_AREA])
                    areas_clean.append(area)
                    if area < small_area_threshold:
                        small_count_clean += 1

                result[f"{layer}_morph_cleaned"] = cleaned_mask.astype(np.float32)
                result[f"{layer}_cc_stats_cleaned"] = {
                    "small_component_count": small_count_clean,
                    "all_areas": areas_clean,
                    "area_threshold": small_area_threshold,
                    "binary_threshold": None,
                    "total_components": len(areas_clean),
                    "removed_small_components": True,
                }
            else:
                # q=0 时不生成 cleaned_mask，但保留清理后的空版本供索引安全
                result[f"{layer}_morph_cleaned"] = np.zeros((H, W), dtype=np.float32)
                result[f"{layer}_cc_stats_cleaned"] = {
                    "small_component_count": 0,
                    "all_areas": [],
                    "area_threshold": small_area_threshold,
                    "binary_threshold": None,
                    "total_components": 0,
                    "removed_small_components": False,
                }

            # 原始 mask 统计写入 result
            result[f"{layer}_cc_stats"] = {
                "small_component_count": small_count_raw,
                "all_areas": areas_raw,
                "area_threshold": small_area_threshold,
                "binary_threshold": None,
                "total_components": total_raw,
                "removed_small_components": False,
            }

        # ── 第二步：融合可靠能量图（仅 q=1 的层参与，权重均为 1/q=1）──
        valid_maps = []
        weights = []
        reliability_mask = np.zeros((H, W), dtype=np.float32)

        for layer in energy_layers:
            if layer not in maps:
                continue

            rel = reliability.get(layer, {})
            q = rel.get("q", 0.0)
            if q <= 0:
                continue

            # 融合时优先用清理后的 map（q=1 时已生成），回退到原始 map
            filtered_key = f"{layer}_morph_filtered"
            if filtered_key in result:
                e = np.squeeze(result[filtered_key]).astype(np.float32)
            else:
                e = np.squeeze(maps[layer]).astype(np.float32)
            e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)

            # cleaned_mask 用于空间约束和 mask 累积
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
            result["energy_fused_reliable"] = np.zeros((H, W), dtype=np.float32)
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
    # 规则 12：新增：能量图筛选与补足机制
    # ==============================================================
    @staticmethod
    def energy_recon_compensate(
        maps: Dict[str, np.ndarray],
        energy_high_threshold_quantile: float = 0.9,  # 能量图高阈值分位数
        recon_low_threshold_quantile: float = 0.1,    # 重建误差低阈值分位数
        small_weight: float = 0.2,                     # 压低权重
        compensate_factor: float = 1.5,                # 补足因子
        cc_area_threshold: int = 50,                   # 连通域面积阈值
    ) -> Dict[str, np.ndarray]:
        """
        能量图筛选和补足机制：
        1. 压低：能量图值低 + 重建误差值高的区域（小权重乘以该区域像素）
        2. 补足：能量图高阈值筛选后连通域面积达标 + 重建误差值低的区域（补足像素）
        """
        result = {k: v.copy() for k, v in maps.items()}
        if "recon_error" not in maps:
            print("警告: 无重建误差图，跳过能量图补足/压低")
            return result

        # 步骤1：融合能量层（用于统一判断）
        energy_layers = [v for k, v in maps.items() if "energy_layer" in k]
        if not energy_layers:
            print("警告: 无能量图层，跳过能量图补足/压低")
            return result
        energy_mean = np.mean(energy_layers, axis=0)
        energy_norm = (energy_mean - energy_mean.min()) / (energy_mean.max() - energy_mean.min() + 1e-8)
        recon = maps["recon_error"]
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

        # 步骤2：计算阈值
        energy_high_thr = np.percentile(energy_norm, energy_high_threshold_quantile * 100)
        recon_low_thr = np.percentile(recon_norm, recon_low_threshold_quantile * 100)
        recon_high_thr = np.percentile(recon_norm, (1 - recon_low_threshold_quantile) * 100)

        # 步骤3：1. 压低区域：能量低 + 重建误差高
        energy_low_mask = energy_norm < (1 - energy_high_threshold_quantile)  # 能量低阈值（反向高阈值）
        recon_high_mask = recon_norm > recon_high_thr                        # 重建误差高阈值
        suppress_mask = energy_low_mask & recon_high_mask
        result["recon_error"][suppress_mask] *= small_weight
        result["suppress_mask"] = suppress_mask.astype(float)  # 记录压低掩码

        # 步骤4：2. 补足区域：能量高 + 重建误差低 + 连通域面积达标
        # 先筛选能量高+重建误差低的原始掩码
        energy_high_mask = energy_norm > energy_high_thr
        recon_low_mask = recon_norm < recon_low_thr
        candidate_mask = energy_high_mask & recon_low_mask

        # 对候选掩码做连通域分析，筛选面积达标的区域
        if candidate_mask.sum() > 0:
            try:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                    candidate_mask.astype(np.uint8),
                    connectivity=8,
                    ltype=cv2.CV_32S
                )
                # 筛选面积达标连通域
                compensate_mask = np.zeros_like(candidate_mask)
                for label_idx in range(1, num_labels):
                    area = stats[label_idx][cv2.CC_STAT_AREA]
                    if area >= cc_area_threshold:
                        compensate_mask[labels == label_idx] = 1

                # 补足逻辑：用能量图归一化值 * 补足因子 * 重建误差均值 补足
                recon_mean = recon.mean()
                compensate_vals = energy_norm[compensate_mask] * compensate_factor * recon_mean
                result["recon_error"][compensate_mask] = compensate_vals
                result["compensate_mask"] = compensate_mask.astype(float)  # 记录补足掩码
                print(f"[补足机制] 压低区域像素数: {suppress_mask.sum()} | 补足区域像素数: {compensate_mask.sum()}")
            except Exception as e:
                print(f"警告: 补足区域连通域分析失败 - {e}，跳过补足")
        else:
            result["compensate_mask"] = np.zeros_like(candidate_mask)
            print(f"[补足机制] 无满足条件的补足候选区域")

        return result

    # ──────────────────────────────────────────────────────────
    # 一键应用所有规则（流水线）
    # ──────────────────────────────────────────────────────────
    @classmethod
    def apply_pipeline(
        cls,
        maps: Dict[str, np.ndarray],
        gt: Optional[np.ndarray] = None,
        sigma_smooth: float = 1.5,
    ) -> Dict[str, np.ndarray]:
        """推荐的处理流水线：平滑 -> 裁剪 -> 融合 -> 交叉抑制。"""
        step = cls.gaussian_smooth(maps, sigma=sigma_smooth)
        step = cls.clip_by_percentile(step, low_pct=2.0, high_pct=98.0)
        step = cls.fuse_energy_layers(step)
        step = cls.cross_suppress_artifact(step)
        if gt is not None:
            step = cls.edge_aware_suppress(step, gt=gt)
        return step


# ==============================================================
#                        创新点二Step2
# ==============================================================
def energy_map_reliability_from_morph_masks(
    maps,
    energy_layers=("energy_layer1", "energy_layer2", "energy_layer3"),
    area_min=50,
    small_area_threshold=50,
    small_ratio_max=0.6,
    total_area_min_ratio=0.001,
    total_area_max_ratio=0.35,
    connectivity=8,
):
    """
    设计一个函数，用于接收能量图集，返回能量图是否可用的判据建议
    """
    reliability = {}
    H, W = None, None

    for layer in energy_layers:
        mask_key = f"{layer}_morph_mask"
        if mask_key not in maps:
            reliability[layer] = {
                "valid": False,
                "q": 0.0,
                "reason": "missing_morph_mask",
            }
            continue

        mask = np.squeeze(maps[mask_key])
        if mask.ndim != 2:
            reliability[layer] = {
                "valid": False,
                "q": 0.0,
                "reason": "invalid_mask_shape",
            }
            continue

        H, W = mask.shape
        binary = (mask > 0).astype(np.uint8)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=connectivity,
            ltype=cv2.CV_32S,
        )

        areas = []
        small_count = 0
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx][cv2.CC_STAT_AREA])
            areas.append(area)
            if area < small_area_threshold:
                small_count += 1

        total_components = len(areas)
        total_area = int(np.sum(areas)) if areas else 0
        max_area = int(np.max(areas)) if areas else 0

        small_ratio = small_count / (total_components + 1e-8)
        total_area_ratio = total_area / float(H * W)

        valid = (
            total_components > 0
            and max_area >= area_min
            and small_ratio <= small_ratio_max
            and total_area_min_ratio <= total_area_ratio <= total_area_max_ratio
        )

        q = 1.0 if valid else 0.0

        reliability[layer] = {
            "valid": valid,
            "q": q,
            "total_components": total_components,
            "small_component_count": small_count,
            "small_ratio": small_ratio,
            "areas": areas,
            "max_area": max_area,
            "total_area": total_area,
            "total_area_ratio": total_area_ratio,
        }

    return reliability


def fuse_valid_energy_maps(maps, reliability, energy_layers=("energy_layer1", "energy_layer2", "energy_layer3")):
    """
    不是所有的能量图都参与，只让形态结构可靠的能量图参与融合过程
    """
    valid_maps = []
    weights = []

    for layer in energy_layers:
        if layer not in maps:
            continue

        q = reliability.get(layer, {}).get("q", 0.0)
        if q <= 0:
            continue

        e = maps[layer]
        e = (e - e.min()) / (e.max() - e.min() + 1e-8)

        mask_key = f"{layer}_morph_mask"
        if mask_key in maps:
            e = e * np.squeeze(maps[mask_key])

        valid_maps.append(e)
        weights.append(q)

    if len(valid_maps) == 0:
        return None

    weights = np.array(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    fused = np.zeros_like(valid_maps[0], dtype=np.float32)
    for e, w in zip(valid_maps, weights):
        fused += w * e

    return fused



# ==============================================================
#                        创新点二Step3
# ==============================================================
def reliability_guided_recon_calibration(
    maps,
    reliability,
    recon_key="recon_error",
    lambda_enhance=0.25,
    lambda_suppress=0.0,
    use_suppression=False,
    use_energy_diff=True,
):
    """
    Step3: 可靠性能量引导的重构误差校准

    公式: S(p) = R(p) * [1 + λ * Q(p) * ΔE(p)]

    其中 ΔE(p) = D(F_in(p)) - D(F_out(p)) 为能量差图，比普通能量图具有更强的理论支撑：
      - 正常 patch：ΔE ≈ 0（输入与输出都偏正常）
      - 异常 patch：ΔE > 0（输入偏异常但被自编码器修复为正常）

    参数：
        maps: 包含能量图/能量差图的字典
        reliability: Step2 输出的可靠性字典，key 名为 "energy_layer1" 等
        recon_key: 重构误差图的 key
        lambda_enhance: 保守增强系数，默认 0.25
        lambda_suppress: 软抑制系数，默认 0.0（消融实验用）
        use_suppression: 是否启用软抑制，默认 False
        use_energy_diff: 是否优先使用能量差图（ΔE），默认 True。
                          若为 True 但 energy_diff_layer* 不存在，则回退到普通能量图。
    """
    result = {k: v.copy() for k, v in maps.items()}

    if recon_key not in maps:
        print("警告: 无重建误差图，跳过 Step3 校准")
        return result

    recon = np.squeeze(maps[recon_key]).astype(np.float32)
    recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

    # ── 确定使用哪类能量图 ──
    # 优先找 energy_diff_layer*，回退到普通 energy_layer*
    energy_layer_names = []
    for base in ("energy_layer1", "energy_layer2", "energy_layer3"):
        if use_energy_diff:
            diff_key = base.replace("energy_layer", "energy_diff_layer")
            if diff_key in maps:
                energy_layer_names.append((base, diff_key))
        if not energy_layer_names or (not use_energy_diff and energy_layer_names and energy_layer_names[-1][0] != base):
            if base not in [e[0] for e in energy_layer_names]:
                if base in maps:
                    energy_layer_names.append((base, base))

    if use_energy_diff and not any(base != diff for base, diff in energy_layer_names):
        print("提示: 未找到 energy_diff_layer*，Step3 将回退使用普通能量图")

    valid_energy = []
    valid_weights = []
    valid_masks = []

    for base_name, energy_key in energy_layer_names:
        q = reliability.get(base_name, {}).get("q", 0.0)
        if q <= 0:
            continue

        e = np.squeeze(maps[energy_key]).astype(np.float32)
        e_norm = (e - e.min()) / (e.max() - e.min() + 1e-8)

        # morph_mask 的 key 固定为 "energy_layer*_morph_mask"（由规则5生成）
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

    # ── 保守增强：只在可靠能量区域提高重构误差信任度 ──
    # S(p) = R(p) * [1 + λ * Q(p) * ΔE(p)]
    gain = 1.0 + lambda_enhance * reliability_mask * energy_fused

    calibrated = recon_norm * gain

    # ── 可选：轻微软抑制（误差高但能量弱的区域）──
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
#              创新点二 Step1-Step3 完整流水线可视化
# ==============================================================
def plot_step23_pipeline(
    maps: Dict[str, np.ndarray],
    step1_result: Dict[str, np.ndarray],
    reliability: Dict[str, Dict],
    step3_result: Dict[str, np.ndarray],
    sample_idx: int,
    save_path: Optional[str] = None,
    gt: Optional[np.ndarray] = None,
):
    """
    绘制创新点二 Step1~Step3 完整流水线的对比图。

    固定排版：4行 N列（N = 1 + len(available_energy)，每行每列位置严格对齐）
      行0: 原始 recon_error + 三层原始 energy_layer
      行1: Step1 规则5处理后的 recon_error_morph + 三层 energy_layer_morph_mask
      行2: Step2 融合可靠能量图 + 三层可靠性标签 + 可靠性Mask
      行3: Step3 校准后异常图 + 校准增益图 + GT（如果有）
    """
    energy_layers = ["energy_layer1", "energy_layer2", "energy_layer3"]
    available_energy = [e for e in energy_layers if e in maps]
    n_cols = len(available_energy) + 1  # col0=recon_error, col1~=energy_layers

    fig = plt.figure(figsize=(4 * n_cols, 4 * 4 + 1))
    fig.suptitle(f"Sample #{sample_idx:04d} — Step1~Step3 Pipeline", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(4, n_cols, figure=fig, hspace=0.4, wspace=0.25)

    col_labels = ["recon_error"] + available_energy

    def make_cell(gs_row, gs_col, arr, title, cmap="jet", show_cbar=True, fmt=".3f"):
        """在指定 GridSpec 位置创建热力图 cell。
        直接传入标量数组，让 matplotlib 用 cmap 渲染，colorbar 自动对齐。
        """
        ax = fig.add_subplot(gs[gs_row, gs_col])
        arr_2d = np.squeeze(arr)
        im = ax.imshow(arr_2d, cmap=cmap)  # 标量数组 + jet colormap，colorbar 自动正确
        vmin, vmax = float(arr_2d.min()), float(arr_2d.max())
        ax.set_title(f"{title}\n[{vmin:{fmt}}~{vmax:{fmt}}]", fontsize=7.5)
        ax.axis("off")
        if show_cbar:
            plt.colorbar(im, ax=ax, fraction=0.046)
        return ax

    # ── 行0: 原始 map ──
    for c, label in enumerate(col_labels):
        arr = np.squeeze(maps[label])
        make_cell(0, c, arr, f"原始 {label}")

    # ── 行1: Step1 规则5 处理后 ──
    for c, label in enumerate(col_labels):
        if label == "recon_error":
            morph_key = "recon_error_morph_mask"
            raw = step1_result.get("recon_error", maps["recon_error"])
        else:
            morph_key = f"{label}_morph_mask"
            raw = step1_result.get(label, maps.get(label, np.zeros_like(maps["recon_error"])))
        arr = np.squeeze(step1_result.get(morph_key, raw))
        make_cell(1, c, arr, f"Step1 {label}\n_morph")

    # ── 行2: Step2 可靠性可视化 ──
    # col0: 融合可靠能量图 E*
    if "energy_fused_reliable" in step3_result:
        arr = np.squeeze(step3_result["energy_fused_reliable"])
        make_cell(2, 0, arr, "Step2 融合可靠\n能量图 E*")
    # col1~: 各层可靠性标签
    for c, layer in enumerate(available_energy, start=1):
        ax = fig.add_subplot(gs[2, c])
        rel = reliability.get(layer, {})
        q = rel.get("q", 0.0)
        valid = rel.get("valid", False)
        total = rel.get("total_components", 0)
        max_a = rel.get("max_area", 0)
        s_ratio = rel.get("small_ratio", 0.0)
        color = "green" if valid else "red"
        ax.text(0.5, 0.7, f"q={q:.1f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.45, f"{'VALID' if valid else 'DISCARD'}",
                ha="center", va="center", fontsize=9, color=color, transform=ax.transAxes)
        ax.text(0.5, 0.2, f"CC={total}  max={max_a}\nsmall_ratio={s_ratio:.2f}",
                ha="center", va="center", fontsize=7, transform=ax.transAxes)
        ax.set_title(f"Step2 {layer}\n可靠性", fontsize=7.5)
        ax.axis("off")

    # ── 行3: Step3 校准结果 ──
    # col0: 校准后异常图 S(p)
    if "recon_error_calibrated" in step3_result:
        arr = np.squeeze(step3_result["recon_error_calibrated"])
        make_cell(3, 0, arr, "Step3 校准后\n异常图 S(p)")
    # col1: 校准增益 G(p)
    if "calibration_gain" in step3_result:
        arr = np.squeeze(step3_result["calibration_gain"])
        make_cell(3, 1, arr, "Step3 校准增益\nG(p)", cmap="coolwarm")
    # col2: 可靠性 Mask Q
    if "energy_reliability_mask" in step3_result and n_cols >= 3:
        arr = np.squeeze(step3_result["energy_reliability_mask"])
        make_cell(3, 2, arr, "Step2 可靠性\nMask Q", cmap="gray")
    # 最后列: GT
    if gt is not None and n_cols >= 4:
        ax = fig.add_subplot(gs[3, n_cols - 1])
        ax.imshow(gt, cmap="gray")
        ax.set_title("Ground Truth", fontsize=7.5)
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Step1~Step3 流水线图已保存: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_reliability_table(
    reliability: Dict[str, Dict],
    sample_idx: int,
    save_path: Optional[str] = None,
):
    """打印 Step2 能量图可靠性统计表格。"""
    print(f"\n{'='*70}")
    print(f"  样本 #{sample_idx:04d} — Step2 能量图可靠性统计")
    print(f"{'='*70}")
    print(f"{'能量层':<18} {'可用':<6} {'q':<5} {'总CC数':<8} {'小CC数':<8} "
          f"{'最大面积':<10} {'总面积':<10} {'总面积比':<10} {'小CC比':<8}")
    print(f"{'-'*70}")
    for layer, info in reliability.items():
        valid = info.get("valid", False)
        q = info.get("q", 0.0)
        total_cc = info.get("total_components", 0)
        small_cc = info.get("small_component_count", 0)
        max_area = info.get("max_area", 0)
        total_area = info.get("total_area", 0)
        total_ratio = info.get("total_area_ratio", 0.0)
        small_ratio = info.get("small_ratio", 0.0)
        status = "是" if valid else "否"
        print(f"{layer:<18} {status:<6} {q:<5.1f} {total_cc:<8} {small_cc:<8} "
              f"{max_area:<10} {total_area:<10} {total_ratio:<10.4f} {small_ratio:<8.4f}")
    print(f"{'='*70}\n")
    if save_path:
        with open(save_path, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"  样本 #{sample_idx:04d} — Step2 能量图可靠性统计\n")
            f.write(f"{'='*70}\n")
            for layer, info in reliability.items():
                valid = info.get("valid", False)
                q = info.get("q", 0.0)
                total_cc = info.get("total_components", 0)
                small_cc = info.get("small_component_count", 0)
                max_area = info.get("max_area", 0)
                total_area = info.get("total_area", 0)
                total_ratio = info.get("total_area_ratio", 0.0)
                small_ratio = info.get("small_ratio", 0.0)
                status = "是" if valid else "否"
                f.write(f"{layer:<18} {status:<6} {q:<5.1f} {total_cc:<8} {small_cc:<8} "
                        f"{max_area:<10} {total_area:<10} {total_ratio:<10.4f} {small_ratio:<8.4f}\n")


# ==============================================================
#                        可视化工具（cv2 热力图配色）
# ==============================================================

def to_heatmap(arr: np.ndarray) -> np.ndarray:
    """
    将任意值域的 float ndarray 归一化到 [0, 255] uint8，
    再用 cv2.COLORMAP_JET 转为 BGR 热力图，最终转回 RGB 返回。
    与 util/test.py 中 cvt2heatmap() 逻辑完全一致。
    """
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        gray = np.zeros_like(arr, dtype=np.uint8)
    else:
        # 归一化到 [0, 255]
        gray = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
    # cv2.COLORMAP_JET: 输入 BGR Gray，输出 BGR 彩色
    heatmap_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # 转 RGB（matplotlib 用 RGB）
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def plot_all_maps(
    maps: Dict[str, np.ndarray],
    sample_idx: int,
    title_prefix: str = "",
    save_path: Optional[str] = None,
    vminmax: Optional[Dict[str, Tuple[float, float]]] = None,
    gt: Optional[np.ndarray] = None,
    processed_maps: Optional[Dict[str, np.ndarray]] = None,
    processed_title: str = "After Post-Processing",
):
    """
    可视化同一个编号的所有 map。
    上排：原始 map（JET 热力图 + colorbar 显示真实值域）；
    下排（如提供）：处理后的 map。
    """
    if vminmax is None:
        vminmax = {}

    map_types = [k for k in ["energy_layer1", "energy_layer2", "energy_layer3", "recon_error"] if k in maps]

    n_cols = len(map_types)
    n_rows = 2 if processed_maps else 1
    if gt is not None:
        n_cols += 1

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 0.5))
    fig.suptitle(f"{title_prefix} Sample #{sample_idx:04d}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig,
                           hspace=0.45, wspace=0.3,
                           height_ratios=[1] * n_rows + [0.08])

    # ── 上排：原始热力图 ──
    for col, mtype in enumerate(map_types):
        ax = fig.add_subplot(gs[0, col])
        arr = maps[mtype]
        vrange = vminmax.get(mtype, (float(arr.min()), float(arr.max())))

        # 直接传标量数组，让 matplotlib 用 jet 做归一化，colorbar 刻度自动正确
        im = ax.imshow(arr, cmap="jet", vmin=vrange[0], vmax=vrange[1])
        ax.set_title(f"{mtype}\n[{vrange[0]:.3f} ~ {vrange[1]:.3f}]", fontsize=9)
        ax.axis("off")

        # colorbar 挂在底部分离行
        cax = fig.add_subplot(gs[n_rows, col])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("value", fontsize=7)

    # GT 列
    if gt is not None:
        ax = fig.add_subplot(gs[0, n_cols - 1])
        ax.imshow(gt, cmap="gray")
        ax.set_title("Ground Truth", fontsize=9)
        ax.axis("off")

    # ── 下排：处理后热力图 ──
    if processed_maps:
        for col, mtype in enumerate(map_types):
            ax = fig.add_subplot(gs[1, col])
            key = mtype if mtype in processed_maps else mtype
            if key in processed_maps:
                arr = processed_maps[key]
                vrange = vminmax.get(mtype, (float(arr.min()), float(arr.max())))
                im = ax.imshow(arr, cmap="jet", vmin=vrange[0], vmax=vrange[1])
                ax.set_title(f"{mtype} (processed)\n[{vrange[0]:.3f} ~ {vrange[1]:.3f}]", fontsize=9)
                ax.axis("off")

                cax = fig.add_subplot(gs[n_rows, col])
                cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label("value", fontsize=7)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison_overlay(
    maps: Dict[str, np.ndarray],
    sample_idx: int,
    save_path: Optional[str] = None,
):
    """将各 map 叠加在同一张图上做横向对比（JET 热力图）。"""
    map_types = [k for k in ["energy_layer1", "energy_layer2", "energy_layer3", "recon_error"] if k in maps]

    fig, axes = plt.subplots(1, len(map_types), figsize=(4 * len(map_types), 4))
    if len(map_types) == 1:
        axes = [axes]

    fig.suptitle(f"Sample #{sample_idx:04d} — Map Comparison (JET)", fontsize=13, fontweight="bold")

    for ax, mtype in zip(axes, map_types):
        arr = maps[mtype]
        im = ax.imshow(arr, cmap="jet")
        ax.set_title(f"{mtype}\n[{arr.min():.3f} ~ {arr.max():.3f}]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"对比图已保存: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_stats_table(stats: Dict[str, Dict[str, float]], sample_idx: int):
    """打印统计表格。"""
    print(f"\n{'='*60}")
    print(f"  样本 #{sample_idx:04d} — 各 Map 统计信息")
    print(f"{'='*60}")
    print(f"{'Map Type':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"{'-'*60}")
    for mtype, s in stats.items():
        print(f"{mtype:<20} {s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
    print(f"{'='*60}\n")


# ==============================================================
#                        命令行入口
# ==============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="分析 testSet/bottle 中的能量图层与重建误差，并应用后处理规则降低伪影。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看单个样本的所有 map
  python analyze_maps.py --sample_dir testSet/bottle/n_bottle_a_0_s_111 --sample_idx 20 --show

  # 应用全部流水线并保存
  python analyze_maps.py --sample_dir testSet/bottle/n_bottle_a_0_s_111 --sample_idx 20 --apply_pipeline --save --output_dir output/

  # 仅使用高斯平滑 + 阈值裁剪
  python analyze_maps.py --sample_dir testSet/bottle/n_bottle_a_0_s_111 --sample_idx 20 --rule clip --rule gaussian_smooth --params sigma=2.0,low_pct=2,high_pct=98 --show

  # 对比两个样本
  python analyze_maps.py --sample_dir testSet/bottle/n_bottle_a_0_s_111 --compare_idx 20 30 --show

  # 批量处理编号 20~49（共30张）
  python analyze_maps.py --sample_dir testSet/bottle/n_bottle_a_0_s_111 --range 20 50 --rule percentile_threshold --params "target_map=all,quantile=0.6" --save --output_dir output/batch_q60

  # 批量，流水线，显示进度
  python analyze_maps.py --sample_dir testSet/bottle/n_bottle_a_0_s_111 --range 0 100 --apply_pipeline --show
        """,
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="testSet/bottle/n_bottle_a_0_s_111",
        help="样本目录路径（相对于项目根目录或绝对路径）",
    )
    parser.add_argument("--sample_idx", type=int, default=None, help="要分析的单个样本编号（0起始）")
    parser.add_argument("--compare_idx", type=int, nargs=2, default=None, help="对比两个样本（如 20 30）")
    parser.add_argument("--range", type=int, nargs=2, default=None,
                        metavar=("START", "STOP"),
                        help="批量范围，左闭右开。例如 --range 20 50 → 处理 20~49")
    parser.add_argument("--show", action="store_true", help="实时显示图片")
    parser.add_argument("--save", action="store_true", help="保存图片到 output_dir")
    parser.add_argument("--output_dir", type=str, default="output/maps_analysis", help="输出目录")
    parser.add_argument("--stats", action="store_true", help="打印统计信息")

    # 规则选择
    parser.add_argument("--apply_pipeline", action="store_true", help="应用推荐的全套后处理流水线")
    parser.add_argument(
        "--rule", type=str, nargs="+",
        choices=[
            "clip", "fuse", "gaussian", "bilateral",
            "morphological", "adaptive_threshold",
            "cross_suppress", "edge_suppress", "energy_norm",
            "percentile_threshold", "connected_component",
            "morph_reliability_fusion",
        ],
        default=[],
        help="指定要应用的后处理规则（可多个）",
    )
    parser.add_argument(
        "--params", type=str, nargs="+",
        default=[],
        help="规则参数块，格式：key1=val1,key2=val2（每个 --params 对应一个 --rule）",
    )

    # 可视化选项
    parser.add_argument("--overlay", action="store_true", help="显示叠加对比图")

    # ── 创新点二 Step1-Step3 流水线 ──
    parser.add_argument("--apply_step23", action="store_true",
                       help="应用创新点二完整流水线: 规则5(Step1) → 可靠性验证(Step2) → 校准(Step3)")
    parser.add_argument("--step23_lambda_enhance", type=float, default=0.25,
                       help="Step3 保守增强系数 λ，默认 0.25")
    parser.add_argument("--step23_lambda_suppress", type=float, default=0.0,
                       help="Step3 软抑制系数（消融用），默认 0.0")
    parser.add_argument("--step23_use_suppression", action="store_true",
                       help="Step3 是否启用软抑制（默认关闭，仅消融实验）")
    parser.add_argument("--step23_use_energy_diff", action="store_true", default=True,
                       help="Step3 是否优先使用能量差图 ΔE（默认 True）")
    parser.add_argument("--step23_area_min", type=int, default=50,
                       help="Step2 最大连通域面积下限，默认 50")
    parser.add_argument("--step23_small_thresh", type=int, default=50,
                       help="Step2 小连通域面积阈值，默认 50")
    parser.add_argument("--step23_small_ratio_max", type=float, default=0.6,
                       help="Step2 小连通域比例上限，默认 0.6")
    parser.add_argument("--step23_area_min_ratio", type=float, default=0.001,
                       help="Step2 总面积比下限，默认 0.001")
    parser.add_argument("--step23_area_max_ratio", type=float, default=0.35,
                       help="Step2 总面积比上限，默认 0.35")
    parser.add_argument("--step23_show_reliability", action="store_true",
                       help="打印 Step2 可靠性统计表格")

    return parser.parse_args()


def resolve_path(p: str) -> Path:
    if os.path.isabs(p):
        return Path(p)
    return Path("/home/bobbystone/CKAAD") / p


def main():
    args = parse_args()

    sample_dir = resolve_path(args.sample_dir)
    print(f"[INFO] 加载数据集: {sample_dir}")

    dataset = MapDataset(sample_dir)
    print(f"[INFO] 共发现 {dataset.num_samples} 个样本，编号范围: {dataset.indices[0]} ~ {dataset.indices[-1]}")

    if args.output_dir:
        out_dir = resolve_path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 输出目录: {out_dir}")

    # 确定要分析的编号列表（优先级：单个 > compare > 批量范围 > 默认前3个）
    if args.sample_idx is not None:
        indices = [args.sample_idx]
    elif args.compare_idx is not None:
        indices = args.compare_idx
    elif args.range is not None:
        start, stop = args.range
        indices = list(range(start, stop))
    else:
        indices = dataset.indices[:3]  # 默认前3个
        print(f"[INFO] 未指定编号，默认分析: {indices}")

    # 解析每个 params 块
    # args.params 是一个 list，每个元素是一个 "key1=val1,key2=val2" 字符串
    param_blocks = []
    for block in args.params:
        p = {}
        for kv in block.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    p[k] = float(v)
                except ValueError:
                    p[k] = v
        param_blocks.append(p)

    # ── 主循环：逐个编号处理 ──
    total = len(indices)
    for pos, idx in enumerate(indices, 1):
        if idx not in dataset.indices:
            print(f"[WARN] 编号 {idx} 不存在，跳过")
            continue

        label = dataset.get_label(idx)
        gt = dataset.get_gt(idx)
        maps = dataset.get_all_maps(idx)

        # 批量时打进度（每10张或首尾）
        if total > 1:
            step = (args.range[1] - args.range[0]) if args.range else 1
            report_every = max(step * 10, 1)
            if pos == 1 or pos % report_every == 0 or pos == total:
                print(f"\n[Batch] 进度 {pos}/{total}  (样本 #{idx:04d})")

        print(f"\n[{'='*60}")
        print(f"  处理样本 #{idx:04d}  |  标签={label}  |  GT可用={'是' if gt is not None else '否'}")
        print(f"{'='*60}")

        if args.stats:
            plot_stats_table(dataset.get_stats(idx), idx)

        # ── 应用后处理规则 ──
        processed = None
        step1_result = None
        step2_reliability = None
        step3_result = None

        if args.apply_pipeline:
            print("[Pipeline] 应用推荐流水线: 平滑 → 裁剪 → 融合 → 交叉抑制 → 边缘感知")
            # 从全局 param_blocks 取第一个 block（向后兼容）
            p = param_blocks[0] if param_blocks else {}
            processed = PostProcessRules.apply_pipeline(
                maps, gt=gt, sigma_smooth=p.get("sigma_smooth", 1.5)
            )
        elif args.apply_step23:
            print("[Step1-3] 应用创新点二完整流水线:")
            print("  Step1: 形态学筛选（规则5: 开运算+闭运算）")
            step1_result = PostProcessRules.morphological_filter(
                maps, operation="both", threshold=0.6
            )
            print("  Step2: 能量图可靠性验证（连通域统计）")
            step2_reliability = energy_map_reliability_from_morph_masks(
                step1_result,
                area_min=args.step23_area_min,
                small_area_threshold=args.step23_small_thresh,
                small_ratio_max=args.step23_small_ratio_max,
                total_area_min_ratio=args.step23_area_min_ratio,
                total_area_max_ratio=args.step23_area_max_ratio,
            )
            if args.step23_show_reliability:
                plot_reliability_table(step2_reliability, idx)

            print("  Step3: 可靠性能量引导的校准")
            step3_result = reliability_guided_recon_calibration(
                step1_result,
                step2_reliability,
                lambda_enhance=args.step23_lambda_enhance,
                lambda_suppress=args.step23_lambda_suppress,
                use_suppression=args.step23_use_suppression,
                use_energy_diff=args.step23_use_energy_diff,
            )
            print(f"  Step3 完成: λ_enhance={args.step23_lambda_enhance}, "
                  f"λ_suppress={args.step23_lambda_suppress}, "
                  f"use_energy_diff={args.step23_use_energy_diff}")

        elif args.rule:
            print(f"[Rules] 应用规则: {args.rule}")
            processed = maps.copy()
            for i, rule in enumerate(args.rule):
                p = param_blocks[i] if i < len(param_blocks) else {}
                if rule == "clip":
                    processed = PostProcessRules.clip_by_percentile(
                        processed,
                        low_pct=p.get("low_pct", 1.0),
                        high_pct=p.get("high_pct", 99.0),
                    )
                elif rule == "fuse":
                    processed = PostProcessRules.fuse_energy_layers(processed)
                elif rule == "gaussian":
                    processed = PostProcessRules.gaussian_smooth(
                        processed, sigma=p.get("sigma", 1.5)
                    )
                elif rule == "bilateral":
                    processed = PostProcessRules.bilateral_filter(
                        processed,
                        sigma_color=p.get("sigma_color", 0.1),
                        sigma_spatial=p.get("sigma_spatial", 5),
                    )
                elif rule == "morphological":
                    processed = PostProcessRules.morphological_filter(
                        processed,
                        threshold=p.get("threshold"),
                        kernel_size=int(p.get("kernel_size", 3)),
                        operation=p.get("operation", "both"),
                    )
                elif rule == "adaptive_threshold":
                    processed = PostProcessRules.adaptive_threshold(
                        processed,
                        window_size=int(p.get("window_size", 15)),
                        multiplier=p.get("multiplier", 1.5),
                    )
                elif rule == "cross_suppress":
                    processed = PostProcessRules.cross_suppress_artifact(processed)
                elif rule == "edge_suppress":
                    processed = PostProcessRules.edge_aware_suppress(processed, gt=gt)
                elif rule == "energy_norm":
                    processed = PostProcessRules.energy_normalized_recon(processed)
                elif rule == "percentile_threshold":
                    processed = PostProcessRules.percentile_threshold(
                        processed,
                        target_map=p.get("target_map", "recon_error"),
                        quantile=p.get("quantile", 0.6),
                    )
                elif rule == "connected_component":
                    processed = PostProcessRules.connected_component_analysis(
                        processed,
                        target_map=p.get("target_map", "recon_error"),
                        binary_threshold=p.get("binary_threshold"),
                        area_threshold=int(p.get("area_threshold", 50)),
                        remove_small_components=p.get("remove_small_components", "True").lower() == "true",
                        connectivity=int(p.get("connectivity", 8)),
                        skip_binarization=p.get("skip_binarization", "False").lower() == "true",
                    )
                elif rule == "morph_reliability_fusion":
                    processed = PostProcessRules.morph_energy_reliability_fusion(
                        processed,
                        area_min=int(p.get("area_min", 50)),
                        small_area_threshold=int(p.get("small_area_threshold", 50)),
                        small_ratio_max=float(p.get("small_ratio_max", 0.6)),
                        total_area_min_ratio=float(p.get("total_area_min_ratio", 0.001)),
                        total_area_max_ratio=float(p.get("total_area_max_ratio", 0.35)),
                        connectivity=int(p.get("connectivity", 8)),
                        remove_small_components=p.get("remove_small_components", "True").lower() == "true",
                    )
                    # 规则13输出 Step2 统计信息
                    rel_dict = processed.get("energy_reliability_dict", {})
                    print(f"  [规则13] 能量图可靠性:")
                    for layer, info in rel_dict.items():
                        q = info.get("q", 0.0)
                        status = "VALID" if info.get("valid") else "DISCARD"
                        print(f"    {layer}: q={q:.1f} | {status} | "
                              f"CC={info.get('total_components', 0)}, max_area={info.get('max_area', 0)}, "
                              f"small_ratio={info.get('small_ratio', 0):.3f}, "
                              f"area_ratio={info.get('total_area_ratio', 0):.4f}")

        # ── 保存路径（提前定义，供所有可视化使用）──
        save_base = str(resolve_path(args.output_dir) / f"sample_{idx:04d}") if args.save else None

        # ── 创新点二 Step1-Step3 流水线可视化 ──
        if args.apply_step23 and step1_result is not None and step3_result is not None:
            pipeline_path = save_base + "_step123_pipeline.png" if save_base else None
            plot_step23_pipeline(
                maps=maps,
                step1_result=step1_result,
                reliability=step2_reliability,
                step3_result=step3_result,
                sample_idx=idx,
                save_path=pipeline_path,
                gt=gt,
            )
            # Step3 统计对比
            orig = maps.get("recon_error")
            proc = step3_result.get("recon_error_calibrated")
            if orig is not None and proc is not None:
                print("\n  Step3 校准前后重构误差统计对比:")
                print(f"    原始: mean={orig.mean():.4f}, std={orig.std():.4f}, max={orig.max():.4f}")
                print(f"    Step3后: mean={proc.mean():.4f}, std={proc.std():.4f}, max={proc.max():.4f}")
                if "calibration_gain" in step3_result:
                    gain = step3_result["calibration_gain"]
                    print(f"    校准增益: min={gain.min():.4f}, max={gain.max():.4f}, mean={gain.mean():.4f}")
                if "energy_reliability_mask" in step3_result:
                    q_mask = step3_result["energy_reliability_mask"]
                    print(f"    可靠区域占比: {(q_mask > 0).sum() / q_mask.size * 100:.2f}%")

        # ── 普通可视化 ──
        if args.save and args.output_dir:
            out_dir = resolve_path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

        # 多 map 可视化（上下对比）
        plot_all_maps(
            maps, idx,
            title_prefix="",
            save_path=save_base + "_comparison.png" if save_base else None,
            gt=gt,
            processed_maps=processed,
        )

        # 叠加对比
        if args.overlay:
            plot_comparison_overlay(
                maps, idx,
                save_path=save_base + "_overlay.png" if save_base else None,
            )

        # 打印处理前后统计对比
        if processed is not None:
            print("\n  处理前后重建误差统计对比:")
            orig = maps.get("recon_error")
            proc = processed.get("recon_error")
            if orig is not None and proc is not None:
                print(f"    原始: mean={orig.mean():.4f}, std={orig.std():.4f}, max={orig.max():.4f}")
                print(f"    处理后: mean={proc.mean():.4f}, std={proc.std():.4f}, max={proc.max():.4f}")

    print("\n[完成]")


if __name__ == "__main__":
    main()
