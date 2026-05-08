"""
轻量级异常生成模块 - Perlin噪声特征扰动

核心思路：
1. 生成平滑随机噪声掩码（形状自然，模拟裂纹/划痕/凹坑）
2. 直接在CKAAD特征空间中对正常特征进行局部扰动
3. 无需任何可学习参数，即插即用

与之前的扩散模型方案对比：
    扩散模型（已删除） | Perlin扰动（当前）
    -------------------+------------------
    1800万+参数        | 0参数
    需要单独预训练     | 无需训练
    像素空间操作       | 特征空间直接操作
    生成一张图几秒     | 生成一批毫秒级

Perlin噪声 → 高斯模糊随机噪声 → 自适应阈值 → 自然形状二值掩码

参考：DRAEM (CVPR 2021) 使用Perlin噪声生成合成异常的思想，
      但直接在CKAAD的特征空间实现，无需回到像素空间。
"""

import torch
import torch.nn.functional as F
import numpy as np


class PerlinAnomalyGenerator:
    """基于平滑噪声掩码的轻量级异常生成器。

    直接在CKAAD的特征空间（多尺度ResNet特征图）上生成合成异常。
    无需任何训练，即插即用。

    用法：
        gen = PerlinAnomalyGenerator(anomaly_ratio=0.3)

        # 在CKAAD训练循环中：
        normal_inputs = pfe(normal_img)  # [f1, f2, f3]
        anomaly_inputs, masks = gen(normal_inputs)
    """

    def __init__(self, anomaly_ratio=0.3, perturbation='noise',
                 noise_std=0.15, kernel_size=65, sigma=12.0,
                 mix_noise=1):
        """
        Args:
            anomaly_ratio: 异常区域占图像比例 (0~1)
            perturbation: 扰动方式
                'noise'          - 掩码区域加高斯噪声（模拟纹理异常）
                'shuffle'        - 跨batch混洗特征（模拟结构异常）
                'erase'          - 掩码区域归零（模拟缺失缺陷）
                'simplenet_noise' - 全图多级高斯噪声（SimpleNet风格扰动，无空间掩码）
                'hard_erase'     - 掩码区域用特征均值填充（模拟硬增强遮挡，区别于erase的归零）
            noise_std: 噪声强度（仅 perturbation='noise' 或 'simplenet_noise' 时有效）
            kernel_size: 高斯模糊核大小（越大掩码越平滑）
            sigma: 高斯模糊sigma（越大掩码过渡越自然）
            mix_noise: 噪声级数（仅 'simplenet_noise'，≥1，噪声 std 按 1.1^k 递增，
                       每样本随机选一级，k=0,...,mix_noise-1，类似 SimpleNet 多级噪声策略）
        """
        self.anomaly_ratio = anomaly_ratio
        self.perturbation = perturbation
        self.noise_std = noise_std
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.mix_noise = mix_noise

    def __call__(self, features_list, anomaly_ratio=None):
        """在特征空间中生成合成异常。

        Args:
            features_list: 列表 [f1, f2, ...]，每个 fi: [B, Ci, Hi, Wi]
            anomaly_ratio: 可选，覆盖初始化时的比例

        Returns:
            anomaly_features: 扰动后的特征列表，形状同 features_list
            masks_list:      每个尺度的二值掩码列表 [B, 1, Hi, Wi]
        """
        ratio = anomaly_ratio if anomaly_ratio is not None else self.anomaly_ratio
        device = features_list[0].device
        batch_size = features_list[0].shape[0]

        anomaly_features = []
        masks_list = []

        # 最高分辨率特征图的尺寸决定掩码生成尺度
        ref_h, ref_w = features_list[0].shape[2:]

        for feat in features_list:
            _, c, h, w = feat.shape

            # 1. 在最高分辨率上生成掩码，再下采样到当前尺度
            if self.perturbation == 'simplenet_noise':
                # SimpleNet noise perturbs the whole feature map, so supervision should be full-map too.
                mask = torch.ones(batch_size, 1, h, w, device=device)
            else:
                mask = self._generate_mask(
                    batch_size, ref_h, ref_w, device, ratio
                )
                if (h, w) != (ref_h, ref_w):
                    mask = F.interpolate(
                        mask, size=(h, w), mode='nearest'
                    )

            # 2. 对特征进行扰动
            feat_anom = self._perturb(feat, mask)

            anomaly_features.append(feat_anom)
            masks_list.append(mask)

        return anomaly_features, masks_list

    def _perturb(self, feat, mask):
        """对特征施加扰动。"""
        if self.perturbation == 'noise':
            noise = torch.randn_like(feat) * self.noise_std
            return feat + noise * mask

        elif self.perturbation == 'shuffle':
            B = feat.shape[0]
            idx = torch.randperm(B, device=feat.device)
            shuffled = feat[idx]
            return feat * (1 - mask) + shuffled * mask

        elif self.perturbation == 'erase':
            return feat * (1 - mask)

        elif self.perturbation == 'simplenet_noise':
            # SimpleNet 风格全局多级高斯噪声（无空间掩码）
            B, C, H, W = feat.shape
            device = feat.device
            # 每样本随机选一个噪声级数
            noise_idxs = torch.randint(0, self.mix_noise, torch.Size([B]), device=device)
            noise_one_hot = F.one_hot(noise_idxs, num_classes=self.mix_noise)  # (B, K)
            # 为每种噪声级数生成噪声张量
            noise_stack = torch.stack([
                torch.normal(0, self.noise_std * (1.1 ** k), feat.shape).to(device)
                for k in range(self.mix_noise)
            ], dim=1)  # (B, K, C, H, W)
            # 每样本只选择其对应的噪声级数
            noise = (noise_stack * noise_one_hot.view(B, self.mix_noise, 1, 1, 1)).sum(1)
            return feat + noise

        elif self.perturbation == 'hard_erase':
            # 硬增强风格擦除：掩码区域用特征均值填充
            # 区别于 erase 的归零，这里用每个样本/通道的空间均值作为"硬填充值"
            # 模拟图像级 Random Erasing 中用固定值填充的思想
            feat_mean = feat.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            return feat * (1 - mask) + feat_mean * mask

        else:
            raise ValueError(f"Unknown perturbation: {self.perturbation}")

    def _generate_mask(self, batch_size, h, w, device, ratio):
        """生成平滑随机二值掩码。

        流程：随机噪声 → 高斯模糊(产生连续过渡) → 自适应阈值(控制异常比例)
        """
        # 生成随机噪声
        noise = torch.rand(batch_size, 1, h, w, device=device)

        # 高斯模糊（产生平滑、自然的形状）
        kernel = self._gaussian_kernel(self.kernel_size, self.sigma, device)
        pad = self.kernel_size // 2
        noise_pad = F.pad(noise, [pad] * 4, mode='reflect')
        smooth = F.conv2d(noise_pad, kernel)

        # 自适应阈值：使得 mask 中 1 的比例约为 ratio
        flat = smooth.view(batch_size, -1)
        k = max(1, int(flat.size(1) * (1 - ratio)))
        threshold = torch.kthvalue(flat, k, dim=1).values
        threshold = threshold.view(batch_size, 1, 1, 1)
        mask = (smooth > threshold).float()

        return mask

    @staticmethod
    def _gaussian_kernel(size, sigma, device):
        """生成2D高斯卷积核。"""
        center = size // 2
        coords = torch.arange(size, dtype=torch.float32, device=device) - center
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d[None, None, :, :]


class MultiScaleAnomalyGenerator:
    """多尺度异常生成器：在不同异常比例下同时生成。

    用于训练时产生多样化的异常样本，提高CKAAD判别器的鲁棒性。

    用法：
        gen = MultiScaleAnomalyGenerator()
        # 每次调用随机选择一个异常比例
        anomaly_inputs, masks = gen(normal_inputs)
    """

    def __init__(self, ratios=(0.05, 0.1, 0.2, 0.35, 0.5),
                 perturbation='noise', noise_std=0.15, mix_noise=1):
        """
        Args:
            ratios: 候选异常比例列表，每次随机选取一个
            perturbation: 扰动方式
            noise_std: 噪声强度
            mix_noise: 噪声级数（仅 'simplenet_noise'）
        """
        self.generators = [
            PerlinAnomalyGenerator(
                anomaly_ratio=r, perturbation=perturbation,
                noise_std=noise_std, mix_noise=mix_noise
            ) for r in ratios
        ]

    def __call__(self, features_list):
        """随机选择一个比例生成异常。"""
        gen = np.random.choice(self.generators)
        return gen(features_list)
