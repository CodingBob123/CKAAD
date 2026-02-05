"""
CKAAD 模型改进版本 - 针对 Tile 等复杂背景数据集

主要改进：
1. 相对能量计算（带 Margin 缩放）
2. 频域分析辅助模块
3. 特征选择正则化

代码风格尽量保持与原 model.py 一致
"""

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder_mixed import Encoder
from model.decoder import Decoder
import numpy as np
import math


class RelativeEnergyDiscriminator(nn.Module):
    """
    带相对能量计算的判别器

    核心改进：
    1. 能量归一化（相对于正常样本分布）
    2. Margin 自动缩放（保持区分度）
    3. 统计量累积更新
    """

    def __init__(self, input_sizes=[64, 32, 16], input_channels=[64, 128, 256],
                 expansion=4, initial_margin=5.0, momentum=0.1):
        """
        参数:
            input_sizes: 输入特征图的空间尺寸列表，默认 [64, 32, 16]
            input_channels: 输入特征图的基础通道数列表，默认 [64, 128, 256]
            expansion: 通道扩展系数，默认 4
            initial_margin: 初始边界值，默认 5.0
            momentum: 统计量更新动量，默认 0.1
        """
        super(RelativeEnergyDiscriminator, self).__init__()

        # 存储扩展系数
        self.expansion = expansion
        self.initial_margin = initial_margin
        self.momentum = momentum

        # 将基础通道数乘以扩展系数
        input_channels = [c * self.expansion for c in input_channels]

        # 第一部分：特征对齐层
        layers = []
        for s, c in zip(input_sizes, input_channels):
            layer = []
            while s > input_sizes[-1]:
                layer.append(nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3,
                              padding=1, stride=2, bias=False),
                    nn.InstanceNorm2d(c * 2),
                    nn.LeakyReLU(0.1, inplace=True),
                ))
                s = s // 2
                c = c * 2
            layers.append(nn.Sequential(*layer))

        self.layers = nn.ModuleList(layers)

        # 第二部分：共享特征处理层
        layers = []
        in_channels = input_channels[-1] * len(input_channels)
        out_channels = input_channels[-1]
        size = input_sizes[-1]

        while size > 2:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, padding=1, stride=2, bias=False),
                nn.InstanceNorm2d(input_channels[-1]),
                nn.LeakyReLU(0.1, inplace=True)
            ))
            in_channels = out_channels
            size = size // 2

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=2, padding=0, stride=2, bias=False))
        self.layer1 = nn.Sequential(*layers)

        # 第三部分：分类层
        self.cls_layer = nn.Sequential(
            nn.Linear(input_channels[-1], input_channels[-1] // 4, bias=False),
            nn.InstanceNorm1d(input_channels[-1] // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(input_channels[-1] // 4, 1, bias=False)
        )

        # 正常样本能量的统计量（用于相对能量计算）
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))
        self.register_buffer('margin', torch.tensor(initial_margin))

    def forward(self, x):
        """
        前向传播（与原版相同）

        参数:
            x: 输入特征图列表

        返回:
            score: 判别分数 [N, 1]
        """
        x = [self.layers[i](xi) for i, xi in enumerate(x)]
        x = torch.cat(x, dim=1)
        z = self.layer1(x)
        z = z.view(z.size(0), -1)
        score = self.cls_layer(z)
        return score

    def get_energy(self, x):
        """
        获取样本的能量值

        参数:
            x: 输入特征图列表

        返回:
            energy: 能量值 [N]
        """
        score = self(x)
        energy = torch.abs(score.view(-1))
        return energy

    def update_statistics(self, normal_energy):
        """
        更新正常样本的能量统计量

        参数:
            normal_energy: 正常样本的能量值 [N]
        """
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * normal_energy.detach().mean()
            self.running_std = (1 - self.momentum) * self.running_std + \
                              self.momentum * normal_energy.detach().std()

    def get_scaled_margin(self):
        """
        获取缩放后的 margin

        返回:
            scaled_margin: 缩放后的边界值
        """
        return self.initial_margin / (self.running_std + 1e-8)

    def calculate_relative_loss(self, x, label_value):
        """
        计算相对能量损失

        核心改进：
        - 能量相对于正常样本分布归一化
        - margin 也相应缩放

        参数:
            x: 输入特征图列表
            label_value: 标签值（0=正常，1=异常）

        返回:
            loss: 损失值
            metrics: 包含能量统计信息的字典
        """
        # 获取能量
        energy = self.get_energy(x)

        # 更新统计量（仅使用正常样本）
        if label_value == 0:
            self.update_statistics(energy)

        # 计算相对能量
        scaled_margin = self.get_scaled_margin()
        relative_energy = (energy - self.running_mean) / (self.running_std + 1e-8)

        # 计算损失
        if label_value == 0:
            # 正常样本：相对能量应该接近 0
            loss = torch.abs(relative_energy).mean()
        else:
            # 异常样本：相对能量应该超过 margin
            loss = F.relu(scaled_margin - relative_energy).mean()

        # 返回损失和统计信息
        metrics = {
            'energy_mean': self.running_mean.item(),
            'energy_std': self.running_std.item(),
            'scaled_margin': scaled_margin.item(),
            'relative_energy_mean': relative_energy.mean().item()
        }

        return loss, metrics

    def calculate_loss(self, x, label_value, margin=5.0):
        """
        兼容原版接口的计算损失函数

        如果模型处于训练状态，使用相对能量计算
        否则使用原始的固定 margin 方式

        参数:
            x: 输入特征图列表
            label_value: 标签值（0=正常，1=异常）
            margin: 保留参数（向后兼容）

        返回:
            loss: 损失值
        """
        if self.training:
            # 训练时使用相对能量
            loss, _ = self.calculate_relative_loss(x, label_value)
            return loss
        else:
            # 推理时使用原始方式
            return self.calculate_original_loss(x, label_value, margin)

    def calculate_original_loss(self, x, label_value, margin=5.0):
        """
        原始的能量损失计算（保留向后兼容）

        参数:
            x: 输入特征图列表
            label_value: 标签值
            margin: 边界值

        返回:
            loss: 损失值
        """
        score = self(x)
        score = torch.abs(score.view(-1))
        label = torch.ones_like(score) * label_value
        loss = ((1 - label) * score + label * (margin - score).clamp_(min=0.)).mean()
        return loss


class FrequencyAwareModule(nn.Module):
    """
    频域分析辅助模块

    用于提取和分析图像的频域特征
    特别适合 Tile 等纹理图像
    """

    def __init__(self, patch_size=32, stride=16):
        """
        参数:
            patch_size: 分块大小，默认 32
            stride: 分块步长，默认 16（50% 重叠）
        """
        super(FrequencyAwareModule, self).__init__()

        self.patch_size = patch_size
        self.stride = stride

    def extract_frequency_features(self, x):
        """
        提取图像的频域特征

        参数:
            x: 图像张量 [N, 3, H, W]

        返回:
            freq_features: 频域特征列表
            patches_info: 分块信息
        """
        # 转为灰度
        if x.dim() == 4 and x.size(1) == 3:
            gray = x.mean(dim=1, keepdim=True)  # [N, 1, H, W]
        else:
            gray = x

        B, C, H, W = gray.shape

        # 计算 padding
        ph = self.patch_size
        pw = self.patch_size
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw

        if pad_h > 0 or pad_w > 0:
            gray = F.pad(gray, (0, pad_w, 0, pad_h))

        # Unfold 提取分块
        patches = F.unfold(gray, kernel_size=ph, stride=self.stride)
        # [N, 1 * ph * ph, num_patches]

        num_patches = patches.shape[-1]
        patches = patches.view(B, 1, ph, ph, num_patches)
        # [N, 1, ph, ph, num_patches]

        freq_features = []
        for b in range(B):
            for p in range(num_patches):
                patch = patches[b, 0, :, :, p]  # [ph, ph]

                # FFT
                fft = torch.fft.rfft2(patch, dim=[0, 1])
                magnitude = torch.abs(fft)  # [ph, ph//2+1]

                # 低频成分（整体纹理结构）
                h, w = magnitude.shape
                low_freq = magnitude[:h//4, :w//4]

                freq_features.append(low_freq.flatten())

        freq_features = torch.stack(freq_features)  # [N*num_patches, feature_dim]

        return freq_features, (B, num_patches)

    def compute_frequency_statistics(self, x):
        """
        计算频域统计信息

        参数:
            x: 图像张量 [N, 3, H, W]

        返回:
            stats: 频域统计字典
        """
        # 转为灰度
        if x.dim() == 4 and x.size(1) == 3:
            gray = x.mean(dim=1, keepdim=True)
        else:
            gray = x

        B, C, H, W = gray.shape

        stats = {
            'mean_magnitude': [],
            'std_magnitude': [],
            'low_freq_ratio': [],
            'high_freq_energy': []
        }

        for b in range(B):
            img = gray[b, 0]  # [H, W]

            # FFT
            fft = torch.fft.rfft2(img, dim=[0, 1])
            magnitude = torch.abs(fft)  # [H, W//2+1]

            # 统计量
            stats['mean_magnitude'].append(magnitude.mean().item())
            stats['std_magnitude'].append(magnitude.std().item())

            # 低频比例
            h, w = magnitude.shape
            low_freq = magnitude[:h//4, :w//4]
            total = magnitude.sum()
            low_freq_ratio = low_freq.sum() / (total + 1e-8)
            stats['low_freq_ratio'].append(low_freq_ratio.item())

            # 高频能量（边缘和细节）
            high_freq = magnitude[h//4:, :]
            stats['high_freq_energy'].append(high_freq.mean().item())

        return {k: torch.tensor(v) for k, v in stats.items()}

    def forward(self, x):
        """
        前向传播

        参数:
            x: 图像张量 [N, 3, H, W]

        返回:
            stats: 频域统计字典
        """
        return self.compute_frequency_statistics(x)


class FrequencyConsistencyLoss(nn.Module):
    """
    频域一致性损失

    正常样本的频域特征应该一致
    异常样本的频域特征应该不一致
    """

    def __init__(self, patch_size=32):
        super(FrequencyConsistencyLoss, self).__init__()

        self.patch_size = patch_size
        self.freq_module = FrequencyAwareModule(patch_size=patch_size)

    def forward(self, normal_images, anomaly_images):
        """
        计算频域一致性损失

        参数:
            normal_images: 正常图像 [N, 3, H, W]
            anomaly_images: 异常图像 [N, 3, H, W]

        返回:
            loss: 频域一致性损失
            metrics: 统计信息字典
        """
        # 提取频域特征
        normal_freq = self.freq_module.extract_frequency_features(normal_images)[0]
        anomaly_freq = self.freq_module.extract_frequency_features(anomaly_images)[0]

        # 正常样本内部的频域一致性（使用类内方差）
        normal_mean = normal_freq.mean(dim=0)
        normal_variance = torch.var(normal_freq, dim=0).mean()

        # 异常样本与正常样本的频域差异
        anomaly_mean = anomaly_freq.mean(dim=0)
        anomaly_diff = torch.abs(anomaly_mean - normal_mean).mean()

        # 损失：正常样本方差小，异常样本差异大
        loss = normal_variance + 1.0 / (anomaly_diff + 1e-8)

        metrics = {
            'normal_freq_variance': normal_variance.item(),
            'anomaly_freq_diff': anomaly_diff.item()
        }

        return loss, metrics


class FeatureSelectionRegularizer(nn.Module):
    """
    特征选择正则化

    轻量级方法引导特征向有意义的方向学习
    """

    def __init__(self):
        super(FeatureSelectionRegularizer, self).__init__()

    def forward(self, normal_features, anomaly_features):
        """
        计算特征选择正则化损失

        参数:
            normal_features: 正常样本特征列表
            anomaly_features: 异常样本特征列表

        返回:
            loss: 正则化损失
            ratio: 方差比
        """
        def compute_variance(features):
            """计算特征的方差"""
            flat_features = [f.view(f.size(0), -1) for f in features]
            variances = [f.std() for f in flat_features]
            return sum(variances) / len(variances) if variances else 0

        normal_var = compute_variance(normal_features)
        anomaly_var = compute_variance(anomaly_features)

        # 正常样本方差应该小，异常样本方差应该大
        variance_ratio = anomaly_var / (normal_var + 1e-8)

        # 鼓励方差比增大
        loss = 1 / (variance_ratio + 1e-8)

        return loss, variance_ratio.item()


class ED(nn.Module):
    """
    编码器-解码器（与原版相同，保留兼容性）
    """

    def __init__(self, backbone='resnet18', input_channels=[64, 128, 256],
                 enable_enhancement=[False, False, False], feature2_fusion_weight=0.5):
        super(ED, self).__init__()
        self.encoder = Encoder(backbone=backbone, input_channels=input_channels,
                              enable_enhancement=enable_enhancement,
                              feature2_fusion_weight=feature2_fusion_weight)
        self.decoder = Decoder(backbone=backbone, output_channels=input_channels)

    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return o


class CKAADv2(nn.Module):
    """
    CKAAD 改进版本 - 针对 Tile 等复杂背景数据集

    组合使用：
    1. 相对能量判别器（带 margin 缩放）
    2. 频域分析辅助
    3. 特征选择正则化
    """

    def __init__(self, backbone='wide_resnet50_2', input_channels=[64, 128, 256],
                 enable_enhancement=[False, False, False],
                 feature2_fusion_weight=0.5, initial_margin=5.0, gamma=0.5):
        """
        参数:
            backbone: 骨干网络
            input_channels: 输入通道数列表
            enable_enhancement: 是否启用增强
            feature2_fusion_weight: 特征融合权重
            initial_margin: 初始边界值
            gamma: 权重因子
        """
        super(CKAADv2, self).__init__()

        # 编码器-解码器
        self.ed = ED(backbone=backbone, input_channels=input_channels,
                     enable_enhancement=enable_enhancement,
                     feature2_fusion_weight=feature2_fusion_weight)

        # 改进的判别器（相对能量）
        self.discriminator = RelativeEnergyDiscriminator(
            input_sizes=[64, 32, 16],
            input_channels=input_channels,
            expansion=4,
            initial_margin=initial_margin,
            momentum=0.1
        )

        # 频域分析模块
        self.frequency_loss = FrequencyConsistencyLoss(patch_size=32)

        # 特征正则化
        self.feature_reg = FeatureSelectionRegularizer()

        # 权重参数
        self.gamma = gamma

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入图像 [N, 3, H, W]

        返回:
            outputs: 重构特征列表
        """
        outputs = self.ed(x)
        return outputs

    def get_discriminator_energy(self, features):
        """
        获取判别器能量

        参数:
            features: 特征图列表

        返回:
            energy: 能量值 [N]
        """
        return self.discriminator.get_energy(features)

    def calculate_loss(self, normal_inputs, anomaly_inputs, outputs,
                       normal_images=None, anomaly_images=None):
        """
        计算总损失

        参数:
            normal_inputs: 正常样本特征列表
            anomaly_inputs: 异常样本特征列表
            outputs: 重构特征列表
            normal_images: 正常样本图像（可选，用于频域分析）
            anomaly_images: 异常样本图像（可选，用于频域分析）

        返回:
            losses: 包含各项损失的字典
        """
        losses = {}

        # 1. 重建损失（余弦相似度）
        recon_loss = self._cosine_reconstruction_loss(normal_inputs, outputs)
        losses['recon_loss'] = recon_loss

        # 2. 判别器损失（相对能量）
        dis_loss, dis_metrics = self._calculate_discriminator_loss(
            normal_inputs, anomaly_inputs, outputs
        )
        losses['dis_loss'] = dis_loss
        losses['dis_metrics'] = dis_metrics

        # 3. 频域损失（如果提供了图像）
        if normal_images is not None and anomaly_images is not None:
            freq_loss, freq_metrics = self.frequency_loss(normal_images, anomaly_images)
            losses['freq_loss'] = freq_loss
            losses['freq_metrics'] = freq_metrics
        else:
            losses['freq_loss'] = torch.tensor(0.0)
            losses['freq_metrics'] = {'normal_freq_variance': 0, 'anomaly_freq_diff': 0}

        # 4. 特征正则化
        reg_loss, reg_ratio = self.feature_reg(normal_inputs, anomaly_inputs)
        losses['feature_reg'] = reg_loss
        losses['feature_reg_ratio'] = reg_ratio

        # 5. 总损失
        total_loss = recon_loss + dis_loss + 0.01 * losses['freq_loss'] + 0.001 * reg_loss
        losses['total_loss'] = total_loss

        return losses

    def _calculate_discriminator_loss(self, normal_inputs, anomaly_inputs, outputs):
        """
        计算判别器损失

        参数:
            normal_inputs: 正常样本特征
            anomaly_inputs: 异常样本特征
            outputs: 重构特征

        返回:
            loss: 总损失
            metrics: 统计信息
        """
        # 分离梯度
        normal_inputs_detach = [i.detach() for i in normal_inputs]
        anomaly_inputs_detach = [i.detach() for i in anomaly_inputs]
        outputs_detach = [o.detach() for o in outputs]

        # 计算各部分损失
        true_label = 0
        fake_label = 1

        # 正常样本
        normal_loss, normal_metrics = self.discriminator.calculate_relative_loss(
            normal_inputs_detach, true_label
        )

        # 异常样本
        anomaly_loss, anomaly_metrics = self.discriminator.calculate_relative_loss(
            anomaly_inputs_detach, fake_label
        )

        # 重构样本
        recon_loss, recon_metrics = self.discriminator.calculate_relative_loss(
            outputs_detach, fake_label
        )

        # 合并指标
        metrics = {
            **normal_metrics,
            **{f'anomaly_{k}': v for k, v in anomaly_metrics.items()},
            **{f'recon_{k}': v for k, v in recon_metrics.items()}
        }

        # 总损失
        loss = normal_loss + (1 - self.gamma) * anomaly_loss + self.gamma * recon_loss

        return loss, metrics

    def _cosine_reconstruction_loss(self, inputs, outputs):
        """
        计算余弦相似度重建损失

        参数:
            inputs: 输入特征列表
            outputs: 输出特征列表

        返回:
            loss: 重建损失
        """
        cos_sim = nn.CosineSimilarity(dim=1)
        loss = 0
        for i in range(len(inputs)):
            a = inputs[i].view(inputs[i].size(0), -1)
            b = outputs[i].view(outputs[i].size(0), -1)
            loss += torch.mean(1 - cos_sim(a, b))
        return loss

    def get_energy_map(self, features):
        """
        获取能量图（用于可视化）

        参数:
            features: 特征图列表

        返回:
            energy_map: 能量图
        """
        # 对于 GlobalDiscriminator，返回标量能量
        energy = self.discriminator.get_energy(features)
        return energy


# 兼容层：保持与原版接口兼容
class Discriminator(RelativeEnergyDiscriminator):
    """
    兼容原版 Discriminator 名称
    """
    pass


def cosine_similarity_loss(inputs, outputs):
    """
    计算余弦相似度损失（工具函数）

    参数:
        inputs: 输入特征列表
        outputs: 输出特征列表

    返回:
        loss: 重建损失
    """
    cos_sim = nn.CosineSimilarity(dim=1)
    loss = 0
    for i in range(len(inputs)):
        a = inputs[i].view(inputs[i].size(0), -1)
        b = outputs[i].view(outputs[i].size(0), -1)
        loss += torch.mean(1 - cos_sim(a, b))
    return loss

