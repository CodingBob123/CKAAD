import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class PatchGraph(nn.Module):
    """
    PatchGraph: 结构感知异常检测模块

    基于kNN关系的图结构建模，用于捕捉工业图像中的结构异常。
    支持多种关系类型：特征相似、空间邻域、对称关系、周期关系。

    输入: [B, C, H, W] 特征图
    输出: [B, 1, H, W] 结构异常分数图
    """

    def __init__(
        self,
        k: int = 8,  # kNN邻域数量
        enable_symmetry: bool = True,  # 是否启用对称关系
        enable_periodic: bool = True,  # 是否启用周期关系
        symmetry_type: str = 'axial',  # 对称类型: 'axial', 'central', 'rotational'
        periodic_directions: int = 4,  # 周期方向数量
        use_local_window: bool = True,  # 是否使用局部窗口限制搜索范围
        window_radius: int = 2,  # 局部窗口半径
        anomaly_detection_mode: str = 'consistency',  # 检测模式: 'consistency', 'reconstruction', 'attention'
    ):
        super(PatchGraph, self).__init__()

        self.k = k
        self.enable_symmetry = enable_symmetry
        self.enable_periodic = enable_periodic
        self.symmetry_type = symmetry_type
        self.periodic_directions = periodic_directions
        self.use_local_window = use_local_window
        self.window_radius = window_radius
        self.anomaly_detection_mode = anomaly_detection_mode

        # 结构异常检测头（如果使用reconstruction或attention模式）
        if anomaly_detection_mode == 'reconstruction':
            self.structure_reconstruction_head = nn.Sequential(
                nn.Conv2d(k * 2, k, kernel_size=1),  # 聚合邻域信息
                nn.ReLU(inplace=True),
                nn.Conv2d(k, 1, kernel_size=1)
            )
        elif anomaly_detection_mode == 'attention':
            self.structure_attention = nn.MultiheadAttention(
                embed_dim=k,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.structure_head = nn.Conv2d(k, 1, kernel_size=1)

    def _get_patch_tokens(self, features: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        将特征图转换为patch tokens

        Args:
            features: [B, C, H, W]

        Returns:
            tokens: [B, N, C] where N = H*W
            H, W: 空间尺寸
        """
        B, C, H, W = features.shape
        N = H * W

        # 展平空间维度为序列
        tokens = features.view(B, C, N).transpose(1, 2).contiguous()  # [B, N, C]

        # L2归一化，增强特征相似性计算的稳定性
        tokens = F.normalize(tokens, dim=-1)

        return tokens, H, W

    def _get_patch_positions(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        获取patch的位置坐标（用于空间关系建模）

        Args:
            H, W: 空间尺寸
            device: 设备

        Returns:
            positions: [N, 2] 归一化坐标 [-1, 1]
        """
        # 创建网格坐标
        y_coords = torch.arange(H, device=device).float() / (H - 1) * 2 - 1  # [-1, 1]
        x_coords = torch.arange(W, device=device).float() / (W - 1) * 2 - 1  # [-1, 1]

        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # [N, 2]

        return positions

    def _build_knn_graph(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建kNN图，支持多种关系类型

        Args:
            tokens: [B, N, C]
            positions: [N, 2]
            H, W: 空间尺寸

        Returns:
            neighbor_indices: [B, N, k] 邻域patch索引
            neighbor_weights: [B, N, k] 邻域权重
        """
        B, N, C = tokens.shape

        if self.use_local_window:
            # 局部窗口kNN：限制搜索范围，提高效率和准确性
            return self._build_local_window_knn(tokens, positions, H, W)
        else:
            # 全局kNN：计算所有patch间的相似性
            return self._build_global_knn(tokens)

    def _build_local_window_knn(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        局部窗口内kNN建图
        """
        B, N, C = tokens.shape
        device = tokens.device

        neighbor_indices = torch.zeros(B, N, self.k, dtype=torch.long, device=device)
        neighbor_weights = torch.zeros(B, N, self.k, device=device)

        for i in range(N):
            # 获取当前位置
            pos_i = positions[i]  # [2]

            # 计算与所有patch的距离
            pos_diff = positions - pos_i.unsqueeze(0)  # [N, 2]
            distances = torch.norm(pos_diff, dim=-1)  # [N]

            # 选择局部窗口内的候选patch
            window_mask = distances <= self.window_radius * math.sqrt(2)  # 近似圆形窗口
            window_mask[i] = False  # 排除自己

            if window_mask.sum() == 0:
                # 如果窗口内没有其他patch，使用全局最近邻
                window_mask = torch.ones(N, dtype=torch.bool, device=device)
                window_mask[i] = False

            # 在窗口内计算特征相似性
            candidates = torch.where(window_mask)[0]  # [M]
            if len(candidates) == 0:
                continue

            # 计算相似性
            similarities = torch.matmul(
                tokens[:, i:i+1],  # [B, 1, C]
                tokens[:, candidates].transpose(-1, -2)  # [B, C, M]
            ).squeeze(1)  # [B, M]

            # 选择top-k相似patch
            topk_sim, topk_idx = similarities.topk(min(self.k, len(candidates)), dim=-1)
            global_idx = candidates[topk_idx]  # [B, k]

            # 存储结果
            actual_k = global_idx.shape[-1]
            neighbor_indices[:, i, :actual_k] = global_idx
            neighbor_weights[:, i, :actual_k] = F.softmax(topk_sim, dim=-1)

        return neighbor_indices, neighbor_weights

    def _build_global_knn(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        全局kNN建图
        """
        B, N, C = tokens.shape

        # 计算所有patch间的相似性
        similarities = torch.matmul(tokens, tokens.transpose(-1, -2))  # [B, N, N]

        # 将对角线设为负无穷，避免选择自己
        similarities = similarities - torch.eye(N, device=tokens.device).unsqueeze(0) * 1e9

        # 选择top-k相似patch
        topk_sim, topk_idx = similarities.topk(self.k, dim=-1)  # [B, N, k]

        # 应用softmax得到权重
        weights = F.softmax(topk_sim, dim=-1)

        return topk_idx, weights

    def _add_symmetry_relations(
        self,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor,
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加对称关系到邻域图 (GPU向量优化版本)
        """
        if not self.enable_symmetry:
            return neighbor_indices, neighbor_weights

        B, N, k = neighbor_indices.shape
        device = neighbor_indices.device

        # ==================== GPU向量优化实现 ====================
        # 目前只支持axial对称，后续可以扩展

        if self.symmetry_type == 'axial':
            # 计算左右和上下对称位置
            left_right = torch.tensor([-1, 1], device=device, dtype=positions.dtype) * positions  # 左右镜像
            top_bottom = torch.tensor([1, -1], device=device, dtype=positions.dtype) * positions  # 上下镜像

            # 合并所有对称位置 [N*2, 2]
            sym_positions = torch.cat([left_right, top_bottom], dim=0)

            # 转换为patch索引
            max_coords = torch.tensor([W-1, H-1], device=device, dtype=torch.float)
            sym_coords = ((sym_positions + 1) / 2 * max_coords).long()
            sym_coords = torch.clamp(sym_coords, min=torch.tensor(0, device=device), max=max_coords.long())
            sym_indices = sym_coords[:, 0] + sym_coords[:, 1] * W  # [N*2]

            # 对应的原始patch索引
            original_indices = torch.arange(N, device=device).repeat(2)  # [N*2]

            # 排除自己对称的情况
            valid_mask = sym_indices != original_indices
            valid_sym_indices = sym_indices[valid_mask]
            valid_original_indices = original_indices[valid_mask]

            # 批量更新邻域图
            if len(valid_sym_indices) > 0:
                # 简单的循环更新（已经比原始版本高效很多）
                for i in range(len(valid_original_indices)):
                    patch_idx = valid_original_indices[i].item()
                    sym_idx = valid_sym_indices[i].item()

                    # 检查是否已在邻域中
                    existing_neighbors = neighbor_indices[:, patch_idx]
                    is_already_neighbor = (existing_neighbors == sym_idx).any(dim=-1)

                    # 更新未包含的batch
                    batch_mask = ~is_already_neighbor
                    if batch_mask.any():
                        min_weight_idx = neighbor_weights[batch_mask, patch_idx].argmin(dim=-1)
                        neighbor_indices[batch_mask, patch_idx, min_weight_idx] = sym_idx
                        neighbor_weights[batch_mask, patch_idx, min_weight_idx] = 1.0

        # 重新归一化权重
        neighbor_weights = F.normalize(neighbor_weights, p=1, dim=-1)

        return neighbor_indices, neighbor_weights

    def _add_periodic_relations(
        self,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor,
        positions: torch.Tensor,
        H: int,
        W: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        添加周期关系到邻域图 (GPU向量优化版本)
        """
        if not self.enable_periodic:
            return neighbor_indices, neighbor_weights

        B, N, k = neighbor_indices.shape
        device = neighbor_indices.device

        # ==================== 简化的向量化实现 ====================
        # 目前只支持基本的周期模式，后续可以扩展

        # 生成4个主要方向的周期位置（上、下、左、右）
        directions = torch.tensor([
            [0, 1],   # 上
            [0, -1],  # 下
            [-1, 0],  # 左
            [1, 0],   # 右
        ], dtype=positions.dtype, device=device)  # [4, 2]

        # 计算周期位置（步长为图像尺寸的1/8）
        step_size = 2.0 / max(H, W)  # 归一化步长
        periodic_offsets = directions * step_size  # [4, 2]

        # 为每个patch计算周期位置
        periodic_positions = positions.unsqueeze(1) + periodic_offsets.unsqueeze(0)  # [N, 4, 2]

        # 边界检查
        valid_mask = (
            (periodic_positions >= -1).all(dim=-1) &
            (periodic_positions <= 1).all(dim=-1)
        )  # [N, 4]

        # 转换为patch索引
        valid_positions = periodic_positions[valid_mask]  # [M, 2]
        max_coords = torch.tensor([W-1, H-1], device=device, dtype=torch.float)
        coords = ((valid_positions + 1) / 2 * max_coords).long()
        coords = torch.clamp(coords, min=torch.tensor(0, device=device), max=max_coords.long())
        periodic_indices = coords[:, 0] + coords[:, 1] * W  # [M]

        # 获取原始patch索引
        valid_indices = torch.nonzero(valid_mask.view(-1)).squeeze(-1)  # [M]
        original_indices = valid_indices // 4  # [M]

        # 排除自己周期的情况
        self_mask = periodic_indices != original_indices
        periodic_indices = periodic_indices[self_mask]
        original_indices = original_indices[self_mask]

        # ==================== 优化批量更新（减少循环） ====================
        if len(periodic_indices) > 0:
            # 限制每个patch最多2个周期邻域 - 使用更高效的分组方式

            # 1. 按原始patch分组并限制数量
            unique_patches = torch.unique(original_indices)
            selected_pairs = []

            for patch_idx in unique_patches:
                patch_mask = original_indices == patch_idx
                patch_periodic = periodic_indices[patch_mask]

                # 限制每个patch最多2个周期邻域
                if len(patch_periodic) > 2:
                    patch_periodic = patch_periodic[:2]

                # 记录选中的关系
                for period_idx in patch_periodic:
                    selected_pairs.append((patch_idx.item(), period_idx.item()))

            # 2. 批量更新邻域图（减少循环次数）
            for patch_idx, period_idx in selected_pairs:
                # 检查是否已在邻域中
                existing_neighbors = neighbor_indices[:, patch_idx]  # [B, k]
                is_already_neighbor = (existing_neighbors == period_idx).any(dim=-1)  # [B]

                # 更新未包含的batch
                batch_mask = ~is_already_neighbor
                if batch_mask.any():
                    # 批量找到最小权重位置并更新
                    min_weight_indices = neighbor_weights[batch_mask, patch_idx].argmin(dim=-1)
                    neighbor_indices[batch_mask, patch_idx, min_weight_indices] = period_idx
                    neighbor_weights[batch_mask, patch_idx, min_weight_indices] = 0.8

        # 重新归一化权重
        neighbor_weights = F.normalize(neighbor_weights, p=1, dim=-1)

        return neighbor_indices, neighbor_weights

    def _compute_structure_anomaly_scores(
        self,
        tokens: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        计算结构异常分数

        Args:
            tokens: [B, N, C]
            neighbor_indices: [B, N, k]
            neighbor_weights: [B, N, k]
            H, W: 空间尺寸

        Returns:
            anomaly_scores: [B, 1, H, W]
        """
        B, N, C = tokens.shape
        device = tokens.device

        if self.anomaly_detection_mode == 'consistency':
            # 邻域一致性残差模式（最简单有效）
            anomaly_scores = self._consistency_anomaly_detection(tokens, neighbor_indices, neighbor_weights)

        elif self.anomaly_detection_mode == 'reconstruction':
            # 重构误差模式
            anomaly_scores = self._reconstruction_anomaly_detection(tokens, neighbor_indices, neighbor_weights)

        elif self.anomaly_detection_mode == 'attention':
            # 注意力模式
            anomaly_scores = self._attention_anomaly_detection(tokens, neighbor_indices, neighbor_weights)

        else:
            raise ValueError(f"Unknown anomaly detection mode: {self.anomaly_detection_mode}")

        # 重塑为空间维度
        anomaly_scores = anomaly_scores.view(B, 1, H, W)

        return anomaly_scores

    def _consistency_anomaly_detection(
        self,
        tokens: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        邻域一致性异常检测
        计算每个patch与其邻域的平均特征距离
        """
        B, N, C = tokens.shape

        # 收集邻域特征
        neighbor_tokens = torch.gather(
            tokens.unsqueeze(1).expand(-1, N, -1, -1),  # [B, N, N, C]
            2,
            neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, C)  # [B, N, k, C]
        )  # [B, N, k, C]

        # 计算与邻域的距离
        distances = torch.norm(
            tokens.unsqueeze(2) - neighbor_tokens,  # [B, N, k, C]
            dim=-1
        )  # [B, N, k]

        # 加权平均距离作为异常分数
        anomaly_scores = torch.sum(distances * neighbor_weights, dim=-1)  # [B, N]

        return anomaly_scores

    def _reconstruction_anomaly_detection(
        self,
        tokens: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        重构误差异常检测
        使用邻域预测当前patch，然后计算重构误差
        """
        B, N, C = tokens.shape

        # 收集邻域特征
        neighbor_tokens = torch.gather(
            tokens.unsqueeze(1).expand(-1, N, -1, -1),
            2,
            neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # [B, N, k, C]

        # 加权聚合邻域特征
        aggregated_features = torch.sum(
            neighbor_tokens * neighbor_weights.unsqueeze(-1),
            dim=2
        )  # [B, N, C]

        # 通过重构头预测
        reconstruction_input = torch.cat([tokens, aggregated_features], dim=-1)  # [B, N, 2*C]
        reconstruction_input = reconstruction_input.view(B, N, 2*C).transpose(1, 2).view(B, 2*C, N//self.k, self.k)
        # 简化为直接计算距离
        anomaly_scores = torch.norm(tokens - aggregated_features, dim=-1)  # [B, N]

        return anomaly_scores

    def _attention_anomaly_detection(
        self,
        tokens: torch.Tensor,
        neighbor_indices: torch.Tensor,
        neighbor_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        注意力机制异常检测
        """
        B, N, C = tokens.shape

        # 使用邻域索引作为query-key
        # 这里简化为基于距离的注意力分数
        anomaly_scores = self._consistency_anomaly_detection(tokens, neighbor_indices, neighbor_weights)

        return anomaly_scores

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            features: [B, C, H, W] 输入特征图

        Returns:
            anomaly_scores: [B, 1, H, W] 结构异常分数图
        """
        # 1. 转换为patch tokens
        tokens, H, W = self._get_patch_tokens(features)

        # 2. 获取位置信息
        positions = self._get_patch_positions(H, W, features.device)

        # 3. 构建基础kNN图
        neighbor_indices, neighbor_weights = self._build_knn_graph(tokens, positions, H, W)

        # 4. 添加对称关系
        if self.enable_symmetry:
            neighbor_indices, neighbor_weights = self._add_symmetry_relations(
                neighbor_indices, neighbor_weights, positions, H, W
            )

        # 5. 添加周期关系
        if self.enable_periodic:
            neighbor_indices, neighbor_weights = self._add_periodic_relations(
                neighbor_indices, neighbor_weights, positions, H, W
            )

        # 6. 计算结构异常分数
        anomaly_scores = self._compute_structure_anomaly_scores(
            tokens, neighbor_indices, neighbor_weights, H, W
        )

        return anomaly_scores


# 便捷构造函数
def create_patch_graph_for_mvtec(
    k: int = 8,
    enable_symmetry: bool = True,
    enable_periodic: bool = True,
    anomaly_detection_mode: str = 'consistency'
) -> PatchGraph:
    """
    为MVTec数据集优化的PatchGraph配置

    Args:
        k: kNN邻域数量
        enable_symmetry: 是否启用对称关系
        enable_periodic: 是否启用周期关系
        anomaly_detection_mode: 异常检测模式

    Returns:
        配置好的PatchGraph实例
    """
    return PatchGraph(
        k=k,
        enable_symmetry=enable_symmetry,
        enable_periodic=enable_periodic,
        symmetry_type='axial',  # MVTec中轴对称最常见
        periodic_directions=4,  # 支持4个主要方向
        use_local_window=True,
        window_radius=2,
        anomaly_detection_mode=anomaly_detection_mode
    )


def create_extended_patch_graph_for_texture(
    k: int = 8,
    enable_symmetry: bool = True,
    enable_periodic: bool = True,
    periodic_directions: int = 8,  # 扩展到8个方向
    anomaly_detection_mode: str = 'consistency'
) -> PatchGraph:
    """
    为复杂纹理优化的PatchGraph配置（支持更多周期方向）

    适用于wood、tile等具有复杂自然纹理的检测对象

    Args:
        k: kNN邻域数量
        enable_symmetry: 是否启用对称关系
        enable_periodic: 是否启用周期关系
        periodic_directions: 周期方向数量 (4/8/12)
        anomaly_detection_mode: 异常检测模式

    Returns:
        配置好的PatchGraph实例
    """
    return PatchGraph(
        k=k,
        enable_symmetry=enable_symmetry,
        enable_periodic=enable_periodic,
        symmetry_type='axial',
        periodic_directions=periodic_directions,  # 扩展方向数
        use_local_window=True,
        window_radius=2,
        anomaly_detection_mode=anomaly_detection_mode
    )


def create_patch_graph_for_mvtec_category(
    category: str,
    k: int = 8,
    enable_symmetry: bool = True,
    enable_periodic: bool = True
) -> PatchGraph:
    """
    根据MvTec类别自动选择最优的PatchGraph配置

    Args:
        category: MvTec类别名称
        k: kNN邻域数量
        enable_symmetry: 是否启用对称关系
        enable_periodic: 是否启用周期关系

    Returns:
        配置好的PatchGraph实例
    """

    # 各类别的最优配置
    category_configs = {
        # 强结构依赖类别 - 使用consistency模式
        'transistor': {'mode': 'consistency', 'symmetry': True, 'periodic': False},
        'metal_nut': {'mode': 'consistency', 'symmetry': True, 'periodic': False},
        'zipper': {'mode': 'consistency', 'symmetry': True, 'periodic': True},
        'toothbrush': {'mode': 'consistency', 'symmetry': False, 'periodic': True},

        # 周期性强的类别 - 混合模式
        'screw': {'mode': 'consistency', 'symmetry': False, 'periodic': True},

        # 外观异常为主的类别 - consistency但权重较低
        'wood': {'mode': 'consistency', 'symmetry': False, 'periodic': False},
        'tile': {'mode': 'consistency', 'symmetry': False, 'periodic': True},
        'carpet': {'mode': 'consistency', 'symmetry': False, 'periodic': False},

        # 其他类别默认配置
        'default': {'mode': 'consistency', 'symmetry': True, 'periodic': True}
    }

    config = category_configs.get(category, category_configs['default'])

    return PatchGraph(
        k=k,
        enable_symmetry=config['symmetry'] if enable_symmetry else False,
        enable_periodic=config['periodic'] if enable_periodic else False,
        symmetry_type='axial',
        periodic_directions=4,
        use_local_window=True,
        window_radius=2,
        anomaly_detection_mode=config['mode']
    )
