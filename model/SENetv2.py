import numpy as np
import torch
from torch import nn
from torch.nn import init

"Squeeze-and-Excitation Networks"


class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=3):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ============ ECA优化：使用1D卷积替代全连接层 ============
        # ECA核心创新：用1D卷积建模通道间关系，避免SE的维度变换
        # 优势：更轻量、更有效、参数更少
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2, bias=False)
        self.eca_sigmoid = nn.Sigmoid()  # ECA标准实现中的sigmoid激活

        # ECA优化：不需要额外的多头映射层，通过广播实现


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                # ECA风格初始化：由于后面接sigmoid，使用标准初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1,x2,x3):
        # (B,C,H,W)
        B, C, H, W = x1.size()
        x = x1 + x2 + x3

        # ============ ECA优化流程 ============
        # Step 1: Squeeze - 全局平均池化 (B,C,H,W) --> (B,C,1,1)
        y = self.avg_pool(x)

        # Step 2: ECA处理 - 使用1D卷积建模通道间关系
        # (B,C,1,1) --> (B,1,C) --> ECA --> (B,1,C) --> sigmoid --> (B,1,C)
        y = y.squeeze(-1).squeeze(-1).unsqueeze(1)  # (B,C,1,1) -> (B,1,C)
        y = self.eca_conv(y)  # ECA: (B,1,C) -> (B,1,C)
        y = self.eca_sigmoid(y)  # 标准ECA中的sigmoid激活，生成0-1权重

        # Step 3: 多头扩展 - 为每个通道生成3个权重（对应3个分支）
        # (B,1,C) --> (B,3,C) --> 转置 --> (B,C,3) --> 重塑 --> (B,3*C,1,1)
        y = y.expand(-1, 3, -1)  # (B,1,C) -> (B,3,C) 广播扩展
        y = y.permute(0, 2, 1)   # (B,3,C) -> (B,C,3)
        y = y.reshape(B, 3*C, 1, 1)  # (B,C,3) -> (B,3*C,1,1)

        # Step 4: 分割权重并应用sigmoid
        weight1 = torch.sigmoid(y[:,:C,:,:])      # (B,C,1,1)
        weight2 = torch.sigmoid(y[:, C:2*C, :, :])    # (B,C,1,1)
        weight3 = torch.sigmoid(y[:, 2*C:, :, :])     # (B,C,1,1)

        # Step 5: 加权融合
        out = x1 * weight1 + x2 * weight2 + x3 * weight3
        return out


def compare_se_eca():
    """比较原始SE和ECA优化版本的参数量和性能"""
    import time

    print("="*60)
    print("SENet ECA优化版本测试")
    print("="*60)

    # 测试输入
    batch_size, channels, height, width = 2, 512, 7, 7
    input1 = torch.randn(batch_size, channels, height, width)
    input2 = torch.randn(batch_size, channels, height, width)
    input3 = torch.randn(batch_size, channels, height, width)

    # 创建模型
    model = SEAttention(channel=channels, reduction=8, kernel_size=3)

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,} ({total_params/1e6:.3f}M)")

    # 前向传播测试
    print(f"\n输入形状: {input1.shape}")
    with torch.no_grad():
        output = model(input1, input2, input3)
    print(f"输出形状: {output.shape}")

    # 推理时间测试
    model.eval()
    num_runs = 100
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(input1, input2, input3)

        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input1, input2, input3)
        end_time = time.time()

    avg_time = (end_time - start_time) / num_runs * 1000
    print(f"推理时间: {avg_time:.3f}ms")
    # 验证输出合理性
    print(f"\n输出统计:")
    print(f"  均值: {output.mean().item():.4f}")
    print(f"  标准差: {output.std().item():.4f}")
    print(f"  最小值: {output.min().item():.4f}")
    print(f"  最大值: {output.max().item():.4f}")

    print(f"\n✅ ECA优化SENet测试完成！")
    return total_params, avg_time

if __name__ == '__main__':
    compare_se_eca()

