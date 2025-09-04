import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math


def get_positional_encoding_2d(d_model, height, width, device):
    """
    Create 2D positional encoding for adding spatial information
    """
    # Create position indices
    y_pos = torch.arange(height, device=device).unsqueeze(1).expand(height, width).float()
    x_pos = torch.arange(width, device=device).unsqueeze(0).expand(height, width).float()
    
    # Normalize positions to [0, 1]
    y_pos = y_pos / (height - 1)
    x_pos = x_pos / (width - 1)
    
    # Scale to [-1, 1]
    y_pos = y_pos * 2 - 1
    x_pos = x_pos * 2 - 1
    
    # Expand to batch dimension and concatenate
    pos_encoding = torch.stack([y_pos, x_pos], dim=0).unsqueeze(0)
    
    # Expand to match d_model if needed (optional)
    if d_model > 2:
        zeros = torch.zeros((1, d_model-2, height, width), device=device)
        pos_encoding = torch.cat([pos_encoding, zeros], dim=1)
    
    return pos_encoding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, head_dim=64, use_spectral_norm=True):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Total dimension for all heads
        self.total_head_dim = num_heads * head_dim
        
        # Linear projections
        if use_spectral_norm:
            self.q_proj = spectral_norm(nn.Conv2d(in_channels, self.total_head_dim, kernel_size=1, bias=False))
            self.k_proj = spectral_norm(nn.Conv2d(in_channels, self.total_head_dim, kernel_size=1, bias=False))
            self.v_proj = spectral_norm(nn.Conv2d(in_channels, self.total_head_dim, kernel_size=1, bias=False))
            self.out_proj = spectral_norm(nn.Conv2d(self.total_head_dim, in_channels, kernel_size=1, bias=False))
        else:
            self.q_proj = nn.Conv2d(in_channels, self.total_head_dim, kernel_size=1, bias=False)
            self.k_proj = nn.Conv2d(in_channels, self.total_head_dim, kernel_size=1, bias=False)
            self.v_proj = nn.Conv2d(in_channels, self.total_head_dim, kernel_size=1, bias=False)
            self.out_proj = nn.Conv2d(self.total_head_dim, in_channels, kernel_size=1, bias=False)
        
        # Temperature parameter for attention scaling
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Project inputs to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim, height * width)
        k = k.view(batch_size, self.num_heads, self.head_dim, height * width)
        v = v.view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Transpose for matrix multiplication
        q = q.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        k = k.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        v = v.transpose(2, 3)  # [B, num_heads, H*W, head_dim]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax for attention weights
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        
        # Reshape back to original format
        out = out.transpose(2, 3).contiguous()  # [B, num_heads, head_dim, H*W]
        out = out.view(batch_size, self.total_head_dim, height, width)
        
        # Final projection
        out = self.out_proj(out)
        
        # Residual connection
        return out + x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, use_spectral_norm=True, num_heads=4):
        super(SpatialAttentionBlock, self).__init__()
        
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.attention = MultiHeadSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            head_dim=in_channels // num_heads,
            use_spectral_norm=use_spectral_norm
        )
        
    def forward(self, x):
        # Apply instance normalization
        norm_x = self.norm(x)
        
        # Apply attention
        out = self.attention(norm_x)
        
        return out 