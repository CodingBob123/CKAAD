"""
FeatureAdapter — 基于 1x1 Conv2d 的特征适配模块。
（原设计基于 SimpleNet Projection 的 Linear 投影，已将 Linear 替换为 Conv2d 1x1，
  避免不必要的 permute/reshape 操作，语义更自然、性能更优。）

目的：
在 AFS 输出后、AE 输入前，对每个尺度的特征图在通道维进行 1x1 卷积投影，
使特征表达更适合后续自编码器重建。

设计要点：
- 每层特征图使用独立的 Conv2d(1x1) 投影（保持通道数不变）
- n_layers > 1 时，层间插入 ReLU 激活，使投影具有非线性表达能力
- Xavier 正态分布初始化权重
"""

import torch.nn as nn


def init_weight(m):
    """Xavier 正态分布初始化。"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class FeatureAdapter(nn.Module):
    """多尺度特征适配器（Conv2d 1x1 版）。

    对 AFS 输出的每个尺度特征图 [B, C, H, W]，应用独立的 1x1 卷积投影。
    投影是逐像素的（即对每个空间位置独立进行），保持通道数 C 不变。

    Args:
        in_channels_list: 各尺度特征图的输入通道数列表（含 expansion），
                          与 AFS 实际输出通道数一致。
        n_layers: 每层投影的 Conv2d 层数。默认 1（单层 Conv2d）。
                  当 >1 时，层间插入 ReLU 激活函数，提供非线性表达能力。
    """

    def __init__(self, in_channels_list, n_layers=1):
        super(FeatureAdapter, self).__init__()

        self.n_layers = n_layers
        self.num_layers = len(in_channels_list)

        self.projections = nn.ModuleList()
        for c in in_channels_list:
            layers = []
            for i in range(n_layers):
                layers.append(nn.Conv2d(c, c, kernel_size=1, bias=True))
                if i < n_layers - 1:  # 层间插入 ReLU（最后一层后不加）
                    layers.append(nn.ReLU(inplace=True))
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
            # Conv2d(1x1) 直接处理 [B, C, H, W]，无需 permute/reshape
            feat_out = self.projections[i](feat)
            adapted.append(feat_out)

        return adapted
