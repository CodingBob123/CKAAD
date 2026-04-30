"""
FeatureAdapter — 基于 SimpleNet Projection 的特征适配模块。

目的：
在 AFS 输出后、AE 输入前，对每个尺度的特征图在通道维进行线性投影，
使特征表达更适合后续自编码器重建。

设计参考 SimpleNet 的 Projection 类（详见 simplenet.py 第59-87行）：
- 每层特征图使用独立的 Linear 投影（保持通道数不变）
- Xavier 正态分布初始化权重

用法（需在 main.py 中取消注释相应代码）：
    feature_adapter = FeatureAdapter(in_channels_list, n_layers=1)
    adapted_feats = feature_adapter(inputs)  # inputs = AFS 输出
"""

import torch
import torch.nn as nn


def init_weight(m):
    """Xavier 正态分布初始化。"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class FeatureAdapter(nn.Module):
    """多尺度特征适配器。

    对 AFS 输出的每个尺度特征图 [B, C, H, W]，在通道维应用独立的 Linear 投影。
    投影是逐像素的（即对每个空间位置独立进行），保持通道数 C 不变。

    Args:
        in_channels_list: 各尺度特征图的输入通道数列表（含 expansion），
                          与 AFS 实际输出通道数一致。
        n_layers: 每层投影的 Linear 层数。默认 1（单层 Linear）。
                  增加层数可增强适配能力，但也会引入更多参数。
    """

    def __init__(self, in_channels_list, n_layers=1):
        super(FeatureAdapter, self).__init__()

        self.n_layers = n_layers
        self.num_layers = len(in_channels_list)

        self.projections = nn.ModuleList()
        for c in in_channels_list:
            layers = []
            _in = c
            for i in range(n_layers):
                layers.append(nn.Linear(_in, c))
                _in = c
            proj = nn.Sequential(*layers)
            proj.apply(init_weight)
            self.projections.append(proj)

    def forward(self, features_list):
        """前向传播。

        Args:
            features_list: AFS 输出的多尺度特征列表，
                           [f1, f2, ...]，每个 fi: [B, Ci, Hi, Wi]

        Returns:
            adapted_list: 适配后的特征列表，形状与输入相同
        """
        adapted = []
        for i, feat in enumerate(features_list):
            B, C, H, W = feat.shape
            # [B, H, W, C] -> [B*H*W, C]
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C)
            # Linear 投影
            feat_proj = self.projections[i](feat_flat)
            # [B, H, W, C] -> [B, C, H, W]
            feat_out = feat_proj.reshape(B, H, W, C).permute(0, 3, 1, 2)
            adapted.append(feat_out)

        return adapted
