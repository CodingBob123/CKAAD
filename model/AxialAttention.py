import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Axial Attention Module (轴向注意力模块)
参考论文: "Axial Attention in Multidimensional Transformers" (ICML 2019)

作用:
    - 分别对Height和Width维度进行自注意力计算
    - 捕捉行列方向的长程依赖关系，增强结构感知能力
    - 与CoordAtt形成互补：CoordAtt是"硬权重"（通过池化+卷积），Axial是"软权重"（通过注意力机制）
    
优势:
    - 计算复杂度为O(HW(H+W))，比全局注意力O(H²W²)更高效
    - 特别适合捕捉"条带状"、"连通"结构（如道路、血管、裂缝、缺陷边缘等）
    - 可以堆叠多个Block逐步增强结构感知能力

插入位置建议:
    - 在FusionLayer的输出后（encode_layer1之后）
    - 或在Decoder的上采样层之间
"""


class AxialAttention(nn.Module):
    """
    轴向注意力模块
    
    参数:
        dim: 输入特征的通道数
        heads: 多头注意力的头数，默认为8
        dim_head: 每个注意力头的维度，默认为64
        qk_scale: 注意力分数的缩放因子，默认为None（使用默认的1/sqrt(d_k)）
    """
    def __init__(self, dim, heads=8, dim_head=64, qk_scale=None):
        super(AxialAttention, self).__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        
        self.heads = heads
        self.dim_head = dim_head
        self.scale = qk_scale if qk_scale is not None else (dim_head ** -0.5)
        
        # 用于Height维度的Q、K、V投影
        self.to_qkv_h = nn.Linear(dim, dim_head * heads * 3, bias=False)
        # 用于Width维度的Q、K、V投影
        self.to_qkv_w = nn.Linear(dim, dim_head * heads * 3, bias=False)
        
        # 输出投影层
        self.to_out_h = nn.Linear(dim_head * heads, dim)
        self.to_out_w = nn.Linear(dim_head * heads, dim)
        
        # LayerNorm用于稳定训练
        self.norm_h = nn.LayerNorm(dim)
        self.norm_w = nn.LayerNorm(dim)
        
        # 残差连接的权重（可学习，初始化为0，方便渐进训练）
        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播
        
        输入:
            x: [B, C, H, W] 特征图
        
        输出:
            out: [B, C, H, W] 增强后的特征图
        
        流程:
            1. Height-wise Attention: 对每一行内的所有位置进行注意力计算
            2. Width-wise Attention: 对每一列内的所有位置进行注意力计算
            3. 残差连接 + 归一化
        """
        B, C, H, W = x.shape
        residual = x
        
        # ========== Height-wise Attention (行方向注意力) ==========
        # 重塑为 [B, W, C, H]，使得每一行(H维度)成为一个序列
        x_h = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x_h = x_h.view(B * W, C, H)  # [B*W, C, H]
        x_h = x_h.permute(0, 2, 1).contiguous()  # [B*W, H, C]
        
        # LayerNorm
        x_h_norm = self.norm_h(x_h)  # [B*W, H, C]
        
        # 计算Q、K、V
        qkv_h = self.to_qkv_h(x_h_norm)  # [B*W, H, 3*dim_head*heads]
        qkv_h = qkv_h.chunk(3, dim=-1)  # 拆分为Q、K、V，每个为 [B*W, H, dim_head*heads]
        q_h, k_h, v_h = map(lambda t: t.view(B * W, H, self.heads, self.dim_head).permute(0, 2, 1, 3), qkv_h)
        # q_h, k_h, v_h: [B*W, heads, H, dim_head]
        
        # 计算注意力分数和加权后的值
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale  # [B*W, heads, H, H]
        attn_h = F.softmax(attn_h, dim=-1)
        out_h = (attn_h @ v_h)  # [B*W, heads, H, dim_head]
        
        # 重塑并投影
        out_h = out_h.permute(0, 2, 1, 3).contiguous()  # [B*W, H, heads, dim_head]
        out_h = out_h.view(B * W, H, self.heads * self.dim_head)  # [B*W, H, dim_head*heads]
        out_h = self.to_out_h(out_h)  # [B*W, H, C]
        
        # 残差连接
        out_h = self.gamma_h * out_h + x_h  # [B*W, H, C]
        
        # 恢复形状
        out_h = out_h.permute(0, 2, 1).contiguous()  # [B*W, C, H]
        out_h = out_h.view(B, W, C, H)  # [B, W, C, H]
        out_h = out_h.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W]
        
        # ========== Width-wise Attention (列方向注意力) ==========
        # 使用Height注意力后的结果作为输入
        # 重塑为 [B, H, C, W]，使得每一列(W维度)成为一个序列
        x_w = out_h.permute(0, 2, 1, 3).contiguous()  # [B, H, C, W]
        x_w = x_w.view(B * H, C, W)  # [B*H, C, W]
        x_w = x_w.permute(0, 2, 1).contiguous()  # [B*H, W, C]
        
        # LayerNorm
        x_w_norm = self.norm_w(x_w)  # [B*H, W, C]
        
        # 计算Q、K、V
        qkv_w = self.to_qkv_w(x_w_norm)  # [B*H, W, 3*dim_head*heads]
        qkv_w = qkv_w.chunk(3, dim=-1)  # 拆分为Q、K、V
        q_w, k_w, v_w = map(lambda t: t.view(B * H, W, self.heads, self.dim_head).permute(0, 2, 1, 3), qkv_w)
        # q_w, k_w, v_w: [B*H, heads, W, dim_head]
        
        # 计算注意力分数和加权后的值
        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale  # [B*H, heads, W, W]
        attn_w = F.softmax(attn_w, dim=-1)
        out_w = (attn_w @ v_w)  # [B*H, heads, W, dim_head]
        
        # 重塑并投影
        out_w = out_w.permute(0, 2, 1, 3).contiguous()  # [B*H, W, heads, dim_head]
        out_w = out_w.view(B * H, W, self.heads * self.dim_head)  # [B*H, W, dim_head*heads]
        out_w = self.to_out_w(out_w)  # [B*H, W, C]
        
        # 残差连接
        out_w = self.gamma_w * out_w + x_w  # [B*H, W, C]
        
        # 恢复形状
        out_w = out_w.permute(0, 2, 1).contiguous()  # [B*H, C, W]
        out_w = out_w.view(B, H, C, W)  # [B, H, C, W]
        out_w = out_w.permute(0, 2, 1, 3).contiguous()  # [B, C, H, W]
        
        # 最终的残差连接（与原始输入）
        out = out_w + residual
        
        return out


if __name__ == '__main__':
    # 测试代码
    # 模拟输入: [batch_size=2, channels=512, height=16, width=16]
    input_tensor = torch.randn(2, 512, 16, 16)
    print(f"输入张量形状: {input_tensor.shape}")
    
    # 创建AxialAttention模块
    model = AxialAttention(dim=512, heads=8, dim_head=64)
    
    # 前向传播
    output_tensor = model(input_tensor)
    print(f"输出张量形状: {output_tensor.shape}")
    print("AxialAttention模块测试通过！")

