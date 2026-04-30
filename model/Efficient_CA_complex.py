import math
import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt_ECA(nn.Module):
    """
    ECA + Coordinate Attention 融合版（完整体现 ECA 创新）

    - 仍然使用坐标注意力的 H / W 方向池化和拼接结构；
    - 中间不降维：通道始终保持 C；
    - 在 H 分支和 W 分支上，分别使用 ECA 风格的 1D Conv
      在通道维上建模“局部跨通道交互”（自适应 k）。

    输入:
        x: [B, C, H, W]
    输出:
        out: [B, C, H, W]   （默认要求 inp == oup）
    """

    def __init__(self, inp, oup=None, gamma: float = 2.0, b: float = 1.0, k_size: int = None):
        super(CoordAtt_ECA, self).__init__()
        if oup is None:
            oup = inp
        # 为了用残差乘法 identity * a_h * a_w，一般要求 oup == inp
        assert oup == inp, "当前实现假设 out_channels == in_channels，方便做残差乘法"

        self.inp = inp
        self.oup = oup

        # 1) 坐标方向的自适应池化（跟原始 CoordAtt 一样）
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (B,C,H,W) -> (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # (B,C,H,W) -> (B,C,1,W)

        # 2) 中间的 1×1 卷积头：不降维，C -> C
        self.conv_head = nn.Conv2d(
            in_channels=inp,
            out_channels=inp,     # 不降维
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn_head = nn.BatchNorm2d(inp)
        self.act = h_swish()

        # 3) ECA：自适应计算 1D 卷积核大小 k（如果没手动指定）
        if k_size is None:
            # 按 ECA-Net 论文常见经验公式：
            # k = |(log2(C) + b) / gamma|，再取最近的奇数
            t = int(abs((math.log2(inp) + b) / gamma))
            k_size = t if t % 2 else t + 1
            if k_size < 1:
                k_size = 1
        self.k_size = k_size

        # H 分支上的 ECA：对每个 (batch, 行) 的通道向量做 1D Conv
        self.eca_h = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        # W 分支上的 ECA：对每个 (batch, 列) 的通道向量做 1D Conv
        self.eca_w = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        identity = x
        B, C, H, W = x.size()

        # ------------------------------------------------------------
        # Step 1: 坐标池化（保留 H / W 方向的空间位置信息）
        # ------------------------------------------------------------
        # 沿 W 池化，保留 H: [B,C,H,W] -> [B,C,H,1]
        x_h = self.pool_h(x)
        # 沿 H 池化，保留 W: [B,C,H,W] -> [B,C,1,W] -> [B,C,W,1]
        x_w = self.pool_w(x)
        x_w = x_w.permute(0, 1, 3, 2)   # [B,C,1,W] -> [B,C,W,1]

        # ------------------------------------------------------------
        # Step 2: 在“高度维”拼接 H/W 两个分支，并做 1×1 Conv 头（不降维）
        # ------------------------------------------------------------
        # 拼接后: y: [B, C, H+W, 1]
        y = torch.cat([x_h, x_w], dim=2)

        # 1×1 Conv + BN + 激活，通道仍然是 C
        y = self.conv_head(y)           # [B,C,H+W,1]
        y = self.bn_head(y)
        y = self.act(y)

        # 按照 H / W 拆回两个分支：
        # y_h: [B,C,H,1]，y_w_tmp: [B,C,W,1]
        y_h, y_w_tmp = torch.split(y, [H, W], dim=2)
        # y_w: [B,C,1,W]
        y_w = y_w_tmp.permute(0, 1, 3, 2)

        # ------------------------------------------------------------
        # Step 3: 在每个 H / W 位置上，用 ECA 的 1D Conv 处理“通道维”
        # ------------------------------------------------------------

        # ===== H 分支 =====
        # y_h: [B,C,H,1] -> [B,C,H]
        f_h = y_h.squeeze(-1)          # [B,C,H]
        # 变为 [B,H,C]，把“行 H”看成 batch 的一部分
        f_h = f_h.permute(0, 2, 1).contiguous()   # [B,H,C]
        # 合并 batch 和行： [B*H,1,C]
        f_h = f_h.view(B * H, 1, C)

        # 在通道维上做 1D Conv（ECA）：局部跨通道交互
        a_h = self.eca_h(f_h)          # [B*H,1,C]
        a_h = self.sigmoid(a_h)        # 激活到 [0,1]

        # reshape 回 [B,C,H,1]
        a_h = a_h.view(B, H, C)        # [B,H,C]
        a_h = a_h.permute(0, 2, 1).unsqueeze(-1)  # [B,C,H,1]

        # ===== W 分支 =====
        # y_w: [B,C,1,W] -> [B,C,W]
        f_w = y_w.squeeze(2)           # [B,C,W]
        # 变为 [B,W,C]，把“列 W”看成 batch 的一部分
        f_w = f_w.permute(0, 2, 1).contiguous()   # [B,W,C]
        # 合并 batch 和列： [B*W,1,C]
        f_w = f_w.view(B * W, 1, C)

        # ECA Conv1d
        a_w = self.eca_w(f_w)          # [B*W,1,C]
        a_w = self.sigmoid(a_w)

        # reshape 回 [B,C,1,W]
        a_w = a_w.view(B, W, C)        # [B,W,C]
        a_w = a_w.permute(0, 2, 1).unsqueeze(2)   # [B,C,1,W]

        # ------------------------------------------------------------
        # Step 4: 将注意力应用到原特征上（坐标感知 + 通道局部交互）
        # ------------------------------------------------------------
        # 广播相乘：A_h 控制每一行，A_w 控制每一列
        out = identity * a_h * a_w     # [B,C,H,W]

        return out


# ------------------------ 简单自测 ------------------------
if __name__ == '__main__':
    x = torch.randn(1, 512, 7, 7)
    model = CoordAtt_ECA(inp=512, oup=512)
    y = model(x)
    print("input shape :", x.shape)
    print("output shape:", y.shape)