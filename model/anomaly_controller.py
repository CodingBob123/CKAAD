"""
三层异常合成策略控制器 — UnifiedAnomalyController。

管理像素级异常（PixelAnomalyGenerator）、特征级异常（PerlinAnomalyGenerator）
和真实异常（anomaly_dataloader）的灵活组合与样本选择策略。

每张图像独立选择异常生成方法，支持：
  - 策略 A：按概率随机选择
  - 策略 B：按类别前景 mask 可用性自适应
  - 策略 C：按 epoch 进度渐进切换
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from model.pixel_anomaly import PixelAnomalyGenerator, has_foreground_mask


class UnifiedAnomalyController:
    """三层异常合成策略控制器。

    在 CKAAD 训练循环中，决定每个 batch 中哪些样本使用哪种异常合成方法。
    """

    def __init__(
        self,
        args,
        num_epochs: int,
        dataset: str = "mvtec",
        class_name: str = "carpet",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            args: 命令行参数（包含 use_synthetic_anomaly, use_pixel_anomaly 等）
            num_epochs: 总训练 epoch 数
            dataset: 数据集名称
            class_name: 类别名称
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.num_epochs = num_epochs
        self.dataset = dataset
        self.class_name = class_name
        self.has_fg_mask = has_foreground_mask(dataset, class_name)

        # 配置
        self.use_perlin = args.use_synthetic_anomaly
        self.use_pixel = getattr(args, "use_pixel_anomaly", False)
        self.use_real = not self.use_perlin  # 未启用合成异常时使用真实异常

        # 策略模式
        self.strategy = getattr(args, "anomaly_strategy", "prob")
        # 'prob' : 策略 A — 按概率
        # 'adapt': 策略 B — 自适应（按类别）
        # 'progressive': 策略 C — 渐进式

        # 各方法基础概率（策略 A）
        self.pixel_prob = getattr(args, "pixel_anomaly_prob", 0.4)
        self.perlin_prob = getattr(args, "perlin_anomaly_prob", 0.35)
        self.normal_prob = 1.0 - self.pixel_prob - self.perlin_prob  # 保持正常的比例

        self._validate_probs()

        # PixelAnomalyGenerator 实例（由外部初始化后传入）
        self.pixel_gen: Optional[PixelAnomalyGenerator] = None

        # PerlinAnomalyGenerator 实例（由外部传入）
        self.perlin_gen = None

        # 日志统计
        self._batch_counter = 0
        self._pixel_count = 0
        self._perlin_count = 0
        self._normal_count = 0

    def _validate_probs(self):
        """确保概率之和不超过 1.0。"""
        total = self.pixel_prob + self.perlin_prob
        if total > 1.0:
            scale = 1.0 / total
            self.pixel_prob *= scale
            self.perlin_prob *= scale
            self.normal_prob = 0.0
        else:
            self.normal_prob = 1.0 - total

    def set_pixel_generator(self, pixel_gen: PixelAnomalyGenerator):
        """设置 PixelAnomalyGenerator 实例。"""
        self.pixel_gen = pixel_gen

    def set_perlin_generator(self, perlin_gen):
        """设置 PerlinAnomalyGenerator 实例。"""
        self.perlin_gen = perlin_gen

    def _get_effective_probs(self, epoch: int) -> Tuple[float, float, float]:
        """根据 epoch 和策略返回有效的 (pixel_prob, perlin_prob, normal_prob)。"""
        if self.strategy == "prob":
            # 策略 A：固定概率
            return self.pixel_prob, self.perlin_prob, self.normal_prob

        elif self.strategy == "adapt":
            # 策略 B：自适应
            if self.has_fg_mask:
                # 有前景 mask → 提高像素级异常概率
                p_pixel = min(self.pixel_prob * 1.5, 0.6)
                p_perlin = max(self.perlin_prob * 0.7, 0.2)
            else:
                # 无前景 mask → 回退 CutPaste（像素级可用） + Perlin
                p_pixel = self.pixel_prob * 0.5  # CutPaste/Cutout 仍可用
                p_perlin = self.perlin_prob * 1.2
            p_normal = 1.0 - p_pixel - p_perlin
            return p_pixel, p_perlin, max(p_normal, 0.0)

        elif self.strategy == "progressive":
            # 策略 C：渐进式
            progress = epoch / max(self.num_epochs, 1)
            if progress < 0.3:
                # 前期：更多 Perlin（简单噪声）
                p_pixel = self.pixel_prob * 0.3
                p_perlin = self.perlin_prob * 1.5
            elif progress < 0.7:
                # 中期：混合
                p_pixel = self.pixel_prob
                p_perlin = self.perlin_prob
            else:
                # 后期：更多 Pixel（复杂结构异常）
                p_pixel = self.pixel_prob * 1.5
                p_perlin = self.perlin_prob * 0.5

            total = p_pixel + p_perlin
            if total > 1.0:
                p_pixel /= total
                p_perlin /= total
                p_normal = 0.0
            else:
                p_normal = 1.0 - total

            return p_pixel, p_perlin, p_normal

        else:
            return self.pixel_prob, self.perlin_prob, self.normal_prob

    def _decide_per_sample(self, epoch: int, batch_size: int) -> torch.Tensor:
        """为 batch 中每张图像决策异常方法。

        Returns:
            决策向量 [B]，值含义：
                0 = 保持正常（不做异常合成）
                1 = 像素级异常（PixelAnomalyGenerator）
                2 = 特征级异常（PerlinAnomalyGenerator）
        """
        p_pixel, p_perlin, p_normal = self._get_effective_probs(epoch)
        probs = torch.tensor([p_normal, p_pixel, p_perlin])  # 正常 / 像素 / Perlin
        decisions = torch.multinomial(probs, batch_size, replacement=True)
        return decisions

    def _print_stats(self, decisions: torch.Tensor):
        """打印当前 batch 的决策统计。"""
        self._batch_counter += 1
        self._pixel_count += (decisions == 1).sum().item()
        self._perlin_count += (decisions == 2).sum().item()
        self._normal_count += (decisions == 0).sum().item()

        if self._batch_counter % 10 == 0:
            total = self._pixel_count + self._perlin_count + self._normal_count
            if total > 0:
                self.logger.info(
                    "[AnomalyController] pixel=%.1f%% perlin=%.1f%% normal=%.1f%%",
                    self._pixel_count / total * 100,
                    self._perlin_count / total * 100,
                    self._normal_count / total * 100,
                )

    @torch.no_grad()
    def __call__(
        self,
        normal_img: torch.Tensor,
        normal_raw: List[torch.Tensor],
        foreground_masks: Optional[torch.Tensor],
        pfe: torch.nn.Module,
        epoch: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """在训练循环中调用，为当前 batch 生成异常特征和 mask。

        Args:
            normal_img: [B, 3, H, W] 原始正常图像
            normal_raw: PFE 提取的正常特征列表（多尺度）
            foreground_masks: [B, 1, H, W] 前景 mask（可 None）
            pfe: 预训练特征提取器（用于像素级异常的二次特征提取）
            epoch: 当前 epoch

        Returns:
            anomaly_features: 异常特征列表，与 normal_raw 形状一致
            anomaly_masks_list: 异常 mask 列表，与 normal_raw 形状一致（特征空间尺度）
            anomaly_size: 异常样本数
        """
        B = normal_img.shape[0]
        device = normal_img.device

        # 如果不需要像素异常，走原有 Perlin 路径
        if not self.use_pixel:
            if self.use_perlin and self.perlin_gen is not None:
                return self.perlin_gen(normal_raw)
            else:
                # 无合成异常——返回空
                empty_feats = [torch.zeros_like(f[:0]) for f in normal_raw]
                empty_masks = [torch.zeros_like(f[:0, :1]) for f in normal_raw]
                return empty_feats, empty_masks, 0

        # 为 batch 中每张图像做决策
        decisions = self._decide_per_sample(epoch, B)
        self._print_stats(decisions)

        has_pixel = (decisions == 1).any().item()
        has_perlin = (decisions == 2).any().item()

        # 收集各方法的索引
        pixel_indices = (decisions == 1).nonzero(as_tuple=True)[0]
        perlin_indices = (decisions == 2).nonzero(as_tuple=True)[0]

        # ---- 像素级异常分支 ----
        anomaly_features_list = []
        anomaly_masks_list_list = []  # list of lists (per scale)

        if has_pixel and self.pixel_gen is not None:
            pixel_imgs = normal_img[pixel_indices]
            pixel_fg = foreground_masks[pixel_indices] if foreground_masks is not None else None

            # 生成像素级异常图像 + mask
            pixel_anomaly, pixel_masks = self.pixel_gen(pixel_imgs, pixel_fg)

            # 提取异常特征
            pixel_raw = pfe(pixel_anomaly)  # list of [N_pixel, C, H, W]

            # 下采样 pixel_masks 到各特征尺度
            for scale_idx, feat in enumerate(pixel_raw):
                _, _, fh, fw = feat.shape
                if (fh, fw) != pixel_masks.shape[2:]:
                    mask_down = F.interpolate(
                        pixel_masks, size=(fh, fw), mode="nearest"
                    )
                else:
                    mask_down = pixel_masks
                anomaly_features_list.append(feat)
                anomaly_masks_list_list.append(mask_down)

        # ---- 特征级异常分支 ----
        if has_perlin and self.perlin_gen is not None:
            # 提取 Perlin 要处理的正常特征子集
            perlin_norm = [f[perlin_indices] for f in normal_raw]
            perlin_ano, perlin_masks = self.perlin_gen(perlin_norm)

            if not has_pixel:
                # 只有 Perlin，直接返回
                return perlin_ano, perlin_masks, perlin_indices.numel()

            # 合并 Pixel 和 Perlin 结果
            for scale_idx in range(len(perlin_ano)):
                feat_cat = torch.cat(
                    [anomaly_features_list[scale_idx], perlin_ano[scale_idx]], dim=0
                )
                mask_cat = torch.cat(
                    [anomaly_masks_list_list[scale_idx], perlin_masks[scale_idx]], dim=0
                )
                # 更新为 cat 后的结果
                anomaly_features_list[scale_idx] = feat_cat
                anomaly_masks_list_list[scale_idx] = mask_cat

        elif has_pixel:
            # 只有像素异常
            pass  # anomaly_features_list 已经包含

        # 如果没有开启任何合成异常
        if not has_pixel and not has_perlin:
            empty_feats = [torch.zeros_like(f[:0]) for f in normal_raw]
            empty_masks = [torch.zeros_like(f[:0, :1]) for f in normal_raw]
            return empty_feats, empty_masks, 0

        anomaly_size = anomaly_features_list[0].shape[0] if anomaly_features_list else 0

        return anomaly_features_list, anomaly_masks_list_list, anomaly_size
