"""
针对 Tile 数据集的改进方案

核心思想：
1. 能量归一化（但保持 margin 比例）
2. 频域辅助分析（不使用显式注意力）
3. 轻量级正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FrequencyAwareLoss(nn.Module):
    """
    频域感知的能量损失

    利用傅里叶分析捕获纹理周期性
    """

    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size

    def extract_frequency_features(self, x):
        """
        提取频域特征

        参数:
            x: 图像 [N, 3, H, W]

        返回:
            freq_features: 频域特征
        """
        # 转为灰度
        if x.dim() == 4 and x.size(1) == 3:
            gray = x.mean(dim=1, keepdim=True)  # [N, 1, H, W]
        else:
            gray = x

        B, C, H, W = gray.shape

        # 分块处理（提取局部频域特征）
        patches = F.unfold(gray, kernel_size=self.patch_size, stride=self.patch_size // 2)
        # [N, 1 * patch_size * patch_size, num_patches]

        num_patches = patches.shape[-1]
        patches = patches.view(B, 1, self.patch_size, self.patch_size, num_patches)
        # [N, 1, patch_size, patch_size, num_patches]

        # 对每个块进行 FFT
        freq_features = []
        for b in range(B):
            for p in range(num_patches):
                patch = patches[b, 0, :, :, p]  # [patch_size, patch_size]
                fft = torch.fft.rfft2(patch, dim=[0, 1])
                magnitude = torch.abs(fft)  # [patch_size, patch_size//2+1]

                # 只取低频成分（代表整体纹理结构）
                h, w = magnitude.shape
                low_freq = magnitude[:h//4, :w//4]

                freq_features.append(low_freq.flatten())

        freq_features = torch.stack(freq_features)  # [N*num_patches, feature_dim]

        return freq_features

    def forward(self, normal_images, anomaly_images, normal_features, anomaly_features):
        """
        计算频域一致性损失

        正常样本的频域特征应该一致
        异常样本的频域特征应该不一致
        """
        # 提取频域特征
        normal_freq = self.extract_frequency_features(normal_images)
        anomaly_freq = self.extract_frequency_features(anomaly_images)

        # 计算正常样本内部的频域一致性
        # 使用相似度矩阵
        normal_sim = torch.mm(normal_freq, normal_freq.t())
        normal_consistency = 1 - torch.std(normal_sim - torch.eye_like(normal_sim))

        # 异常样本与正常样本的频域差异
        anomaly_diff = torch.abs(normal_freq.mean(dim=0) - anomaly_freq.mean(dim=0))

        return normal_consistency, anomaly_diff.mean()


class RelativeEnergyWithMarginScaling(nn.Module):
    """
    带 Margin 缩放的相对能量

    核心改进：
    - 能量归一化
    - margin 也相应缩放（保持区分度）
    """

    def __init__(self, initial_margin=5.0, momentum=0.1):
        super().__init__()

        self.initial_margin = initial_margin
        self.momentum = momentum

        # 正常样本能量的统计
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))
        self.register_buffer('margin', torch.tensor(initial_margin))

    def update_statistics(self, normal_energy):
        """更新正常样本的能量统计"""
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * normal_energy.detach().mean()
            self.running_std = (1 - self.momentum) * self.running_std + \
                              self.momentum * normal_energy.detach().std()

    def get_scaled_margin(self):
        """获取缩放后的 margin"""
        return self.initial_margin / (self.running_std + 1e-8)

    def calculate_loss(self, normal_energy, anomaly_energy, recon_energy):
        """
        计算相对能量损失

        参数:
            normal_energy: 正常样本的能量 [N]
            anomaly_energy: 异常样本的能量 [N]
            recon_energy: 重构样本的能量 [N]
        """
        # 更新统计
        self.update_statistics(normal_energy)

        # 相对能量（相对于正常样本分布）
        rel_normal = (normal_energy - self.running_mean) / (self.running_std + 1e-8)
        rel_anomaly = (anomaly_energy - self.running_mean) / (self.running_std + 1e-8)
        rel_recon = (recon_energy - self.running_mean) / (self.running_std + 1e-8)

        # 获取缩放后的 margin
        scaled_margin = self.get_scaled_margin()

        # 损失
        normal_loss = torch.abs(rel_normal).mean()  # → 0
        anomaly_loss = F.relu(scaled_margin - rel_anomaly).mean()  # → margin
        recon_loss = F.relu(scaled_margin - rel_recon).mean()  # → margin

        return normal_loss + anomaly_loss + recon_loss, scaled_margin.item()


class FeatureSelectionRegularizer(nn.Module):
    """
    特征选择正则化

    轻量级方法：
    - 不添加新层
    - 只在损失函数中引导特征选择
    - 抑制与异常无关的背景特征
    """

    def __init__(self):
        super().__init__()

    def forward(self, normal_features, anomaly_features):
        """
        计算特征选择正则化损失

        目标：
        - 正常样本的特征应该集中（低方差）
        - 异常样本的特征应该分散（高方差）
        """
        # 展平特征
        normal_flat = [f.view(f.size(0), -1) for f in normal_features]
        anomaly_flat = [f.view(f.size(0), -1) for f in anomaly_features]

        # 计算每个尺度的特征方差
        normal_variance = sum([f.std() for f in normal_flat]) / len(normal_flat)
        anomaly_variance = sum([f.std() for f in anomaly_flat]) / len(anomaly_flat)

        # 正则化：正常样本方差应该小，异常样本方差应该大
        variance_ratio = anomaly_variance / (normal_variance + 1e-8)

        # 鼓励方差比增大
        reg_loss = 1 / (variance_ratio + 1e-8)

        return reg_loss


class TileOptimizedCKAAD(nn.Module):
    """
    针对 Tile 优化的 CKAAD

    组合使用：
    1. 相对能量（带 margin 缩放）
    2. 频域分析辅助
    3. 特征选择正则化
    """

    def __init__(self, base_model, args):
        super().__init__()

        # 基础模型
        self.ed = base_model.ed
        self.discriminator = base_model.discriminator

        # 改进模块
        self.energy_module = RelativeEnergyWithMarginScaling(
            initial_margin=args.margin,
            momentum=0.1
        )

        self.frequency_loss = FrequencyAwareLoss(patch_size=32)

        self.feature_reg = FeatureSelectionRegularizer()

    def forward(self, x):
        return self.ed(x)

    def calculate_total_loss(self, normal_inputs, anomaly_inputs, outputs,
                             normal_images, anomaly_images, gamma=0.5):
        """
        计算总损失
        """
        # 1. 重建损失
        recon_loss = self._cosine_reconstruction_loss(normal_inputs, outputs)

        # 2. 获取能量
        normal_energy = self._get_energy(normal_inputs)
        anomaly_energy = self._get_energy(anomaly_inputs)
        recon_energy = self._get_energy(outputs)

        # 3. 相对能量损失（带 margin 缩放）
        energy_loss, current_margin = self.energy_module.calculate_loss(
            normal_energy, anomaly_energy, recon_energy
        )

        # 4. 频域损失
        normal_freq_consistency, anomaly_freq_diff = self.frequency_loss(
            normal_images, anomaly_images,
            normal_inputs[-1] if isinstance(normal_inputs[-1], torch.Tensor) else normal_inputs[-1],
            anomaly_inputs[-1] if isinstance(anomaly_inputs[-1], torch.Tensor) else anomaly_inputs[-1]
        )
        freq_loss = anomaly_freq_diff

        # 5. 特征选择正则化
        feature_reg_loss = self.feature_reg(normal_inputs, anomaly_inputs)

        # 6. 总损失
        total_loss = recon_loss + \
                    energy_loss + \
                    0.01 * freq_loss + \
                    0.001 * feature_reg_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'energy_loss': energy_loss,
            'freq_loss': freq_loss,
            'feature_reg': feature_reg_loss,
            'current_margin': current_margin
        }

    def _get_energy(self, features):
        """获取样本的能量"""
        score = self.discriminator(features)
        energy = torch.abs(score.view(-1))
        return energy

    def _cosine_reconstruction_loss(self, inputs, outputs):
        """余弦相似度重建损失"""
        cos_sim = nn.CosineSimilarity(dim=1)
        loss = 0
        for i in range(len(inputs)):
            a = inputs[i].view(inputs[i].size(0), -1)
            b = outputs[i].view(outputs[i].size(0), -1)
            loss += torch.mean(1 - cos_sim(a, b))
        return loss