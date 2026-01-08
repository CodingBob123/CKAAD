from typing import Union, Tuple                    # 引入类型注解，用于函数参数和返回值约束
import torch                                       # 导入 PyTorch 主库
import torch.nn as nn                              # 导入神经网络模块
import torchvision                                 # 导入 torchvision，用于可变形卷积算子

""" 
    即插即用的可变形卷积模块（便携版，无需编译）：
        写作思路与代码讲解：https://www.bilibili.com/video/BV1SUqoBiE1J/
        作用位置：任何单一特征处理时/任何普通卷积，或者任何即插即用模块中。
        主要功能（写作要点）：①直接替换标准卷积层，提升几何变换（如尺度、变形）特征捕捉能力。
        代码层面：在标准卷积的基础上，加入
               1、Offset predictor：从输入特征图预测每个采样点的二维偏移量，使采样位置能够自适应调整到更相关区域。
               2、Mask predictor：动态调整采样点对输出特征的权重系数（类似于注意力机制，抑制无关区域、增强感兴趣区域）。
"""

def to_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    # 将单个整数或二元组统一转换为 (height, width) 的二元形式
    if isinstance(x, tuple):                       # 判断输入是否已经是元组
        assert len(x) == 2                          # 强制保证元组长度为 2
        return x                                   # 若合法则直接返回
    return (x, x)                                  # 若为整数，则在高和宽维度上复用

class SAMDeformConv2d(nn.Module):
    # 结构自适应调制可变形卷积模块（DCNv2 风格实现）
    def __init__(
        self,
        in_channels: int,                          # 输入特征图通道数
        out_channels: int,                         # 输出特征图通道数
        kernel_size: Union[int, Tuple[int, int]] = 3,  # 卷积核尺寸
        stride: Union[int, Tuple[int, int]] = 1,       # 卷积步长
        padding: Union[int, Tuple[int, int]] = 1,      # 卷积填充大小
        bias: bool = False,                        # 是否使用卷积偏置
    ) -> None:
        super().__init__()                         # 初始化父类 nn.Module

        kH, kW = to_2tuple(kernel_size)            # 解析卷积核高宽
        sH, sW = to_2tuple(stride)                 # 解析步长高宽
        pH, pW = to_2tuple(padding)                # 解析填充高宽

        self.ks = (kH, kW)                          # 保存卷积核尺寸
        self.stride = (sH, sW)                      # 保存步长参数
        self.padding = (pH, pW)                     # 保存填充参数

        k2 = kH * kW                                # 计算每个卷积核的采样点数量

        self.offset_conv = nn.Conv2d(               # 定义偏移量预测分支
            in_channels=in_channels,                # 输入通道数与主分支一致
            out_channels=2 * k2,                    # 每个采样点预测 (dx, dy)
            kernel_size=(kH, kW),                   # 使用与主卷积一致的核尺寸
            stride=(sH, sW),                        # 使用与主卷积一致的步长
            padding=(pH, pW),                       # 使用与主卷积一致的填充
            bias=True,                              # 偏移预测需要偏置项
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)  # 初始化偏移权重为 0
        nn.init.constant_(self.offset_conv.bias, 0.0)    # 初始化偏移偏置为 0

        self.mask_conv = nn.Conv2d(                 # 定义调制系数预测分支
            in_channels=in_channels,                # 输入通道数
            out_channels=1 * k2,                    # 每个采样点预测一个权重
            kernel_size=(kH, kW),                   # 卷积核尺寸
            stride=(sH, sW),                        # 步长
            padding=(pH, pW),                       # 填充
            bias=True,                              # 使用偏置项
        )
        nn.init.constant_(self.mask_conv.weight, 0.0)    # 初始化调制权重为 0
        nn.init.constant_(self.mask_conv.bias, 0.0)      # 初始化调制偏置为 0

        self.proj = nn.Conv2d(                      # 定义实际用于卷积计算的权重存储层
            in_channels=in_channels,                # 输入通道数
            out_channels=out_channels,              # 输出通道数
            kernel_size=(kH, kW),                   # 卷积核尺寸
            stride=(sH, sW),                        # 步长
            padding=(pH, pW),                       # 填充
            bias=bias,                              # 是否启用偏置
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)                # 预测每个位置的采样偏移量 创新点1
        mask = 2.0 * torch.sigmoid(self.mask_conv(x))  # 预测调制权重并映射到 (0, 2)  创新点2

        y = torchvision.ops.deform_conv2d(          # 调用官方可变形卷积算子
            input=x,                                # 输入特征图
            offset=offset,                          # 空间偏移量
            weight=self.proj.weight,                # 卷积核权重
            bias=self.proj.bias,                    # 卷积偏置
            stride=self.stride,                     # 步长参数
            padding=self.padding,                   # 填充参数
            mask=mask,                              # 调制系数
        )
        return y                                    # 返回可变形卷积输出结果

if __name__ == "__main__":
    model = SAMDeformConv2d(                        # 实例化可变形卷积模块
        in_channels=32,                             # 设置输入通道数
        out_channels=32                             # 设置输出通道数
    )
    x = torch.rand(1, 32, 50, 50)                   # 构造随机输入张量
    y = model(x)                                    # 前向推理
    print(f"输入张量形状: {x.shape}")               # 输出输入特征图尺寸
    print(f"输出张量形状: {y.shape}")               # 输出结果特征图尺寸
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：微创新·代码无误")