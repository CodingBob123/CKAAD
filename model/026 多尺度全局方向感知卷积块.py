import torch
import torch.nn as nn

"""
    多尺度全局方向感知卷积块：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1Zay1BPE4w/
        作用位置：任何单一特征处理时，或者任何即插即用模块中。
        主要功能（写作要点）：①多尺度响应与尺度一致性；②多方向特征感知与方向一致性；③空间结构感知。
        代码层面：在保持特征维度不变的前提下，强化对不同空间方向（水平、垂直）和尺度（局部细节至全局结构）信息的表征能力，提升特征的判别性与鲁棒性。
"""


class MultiScaleDirectionalConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        # 大卷积核的尺寸
        kernel_size = 63
        padding = kernel_size // 2

        # 输入卷积：通道对齐 + 激活
        self.input_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        # 输出卷积：恢复通道
        self.output_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        # 四个方向/尺度的深度卷积
        self.conv_horizontal = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            stride=1,
            groups=channels
        )  # 横向卷积

        self.conv_vertical = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=1,
            groups=channels
        )  # 纵向卷积

        self.conv_large = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=1,
            groups=channels
        )  # 大核卷积

        self.conv_pointwise = nn.Conv2d(
            channels, channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=channels
        )  # 逐点卷积

        # 激活函数
        self.act = nn.ReLU()

    def forward(self, x):
        # Step1: 输入特征预处理
        feat = self.input_proj(x)

        # Step2: 四个分支分别计算
        out_horizontal = self.conv_horizontal(feat)   # 横向卷积特征
        out_vertical   = self.conv_vertical(feat)     # 纵向卷积特征
        out_large      = self.conv_large(feat)        # 大核卷积特征
        out_pointwise  = self.conv_pointwise(feat)    # 逐点卷积特征

        # Step3: 残差相加
        out = x + out_horizontal + out_vertical + out_large + out_pointwise
        # Step4: 激活
        out = self.act(out)
        # Step5: 输出投影
        return self.output_proj(out)

if __name__ == '__main__':
    input_tensor = torch.randn(1, 32, 50, 50)
    model = MultiScaleDirectionalConvBlock(channels=32)
    output_tensor = model(input_tensor)
    print(f"输入张量形状: {input_tensor.shape}")
    print(f"输出张量形状: {output_tensor.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")
