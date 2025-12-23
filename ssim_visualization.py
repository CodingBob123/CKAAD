"""
SSIM损失计算原理可视化示例
用于理解SSIM的三个组件：亮度、对比度、结构相似性
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def create_window(window_size, channel):
    """创建高斯窗口"""
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
                              for x in range(window_size)])
        return gauss/gauss.sum()
    
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def compute_ssim_components(pred, target, window_size=11):
    """
    计算SSIM的三个组件，用于可视化理解
    
    返回:
        luminance: 亮度相似性
        contrast: 对比度相似性
        structure: 结构相似性
        ssim: 完整的SSIM值
    """
    channel = pred.size(1)
    window = create_window(window_size, channel).to(pred.device)
    
    # 计算均值
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算三个组件
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * torch.sqrt(sigma1_sq * sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C2/2) / (torch.sqrt(sigma1_sq * sigma2_sq) + C2/2)
    
    # 完整SSIM
    ssim = luminance * contrast * structure
    
    return luminance.mean().item(), contrast.mean().item(), structure.mean().item(), ssim.mean().item()

def visualize_ssim_components():
    """可视化SSIM的三个组件"""
    print("=" * 60)
    print("SSIM损失计算原理可视化")
    print("=" * 60)
    
    # 示例1: 完全相同的情况
    print("\n【示例1】完全相同的情况")
    pred1 = torch.randn(1, 1, 32, 32)
    target1 = pred1.clone()
    l1, c1, s1, ssim1 = compute_ssim_components(pred1, target1)
    print(f"亮度相似性: {l1:.6f} (应该接近1.0)")
    print(f"对比度相似性: {c1:.6f} (应该接近1.0)")
    print(f"结构相似性: {s1:.6f} (应该接近1.0)")
    print(f"完整SSIM: {ssim1:.6f} (应该接近1.0)")
    
    # 示例2: 亮度不同但结构相同
    print("\n【示例2】亮度不同但结构相同")
    pred2 = torch.randn(1, 1, 32, 32)
    target2 = pred2 + 10.0  # 整体亮度+10
    l2, c2, s2, ssim2 = compute_ssim_components(pred2, target2)
    print(f"亮度相似性: {l2:.6f} (可能较低，因为亮度不同)")
    print(f"对比度相似性: {c2:.6f} (应该接近1.0，对比度相同)")
    print(f"结构相似性: {s2:.6f} (应该接近1.0，结构相同)")
    print(f"完整SSIM: {ssim2:.6f} (应该较高，因为结构相同)")
    
    # 示例3: 结构不同
    print("\n【示例3】结构不同")
    pred3 = torch.randn(1, 1, 32, 32)
    target3 = torch.randn(1, 1, 32, 32)  # 完全不同的随机图像
    l3, c3, s3, ssim3 = compute_ssim_components(pred3, target3)
    print(f"亮度相似性: {l3:.6f}")
    print(f"对比度相似性: {c3:.6f}")
    print(f"结构相似性: {s3:.6f} (应该较低，因为结构不同)")
    print(f"完整SSIM: {ssim3:.6f} (应该较低)")
    
    # 示例4: 对比度不同但结构相同
    print("\n【示例4】对比度不同但结构相同")
    pred4 = torch.randn(1, 1, 32, 32)
    target4 = pred4 * 0.5  # 对比度减半
    l4, c4, s4, ssim4 = compute_ssim_components(pred4, target4)
    print(f"亮度相似性: {l4:.6f} (可能较低)")
    print(f"对比度相似性: {c4:.6f} (应该较低，因为对比度不同)")
    print(f"结构相似性: {s4:.6f} (应该接近1.0，结构相同)")
    print(f"完整SSIM: {ssim4:.6f} (应该中等)")
    
    print("\n" + "=" * 60)
    print("关键观察：")
    print("1. 结构相似性(s)是最关键的组件，衡量局部结构模式的相关性")
    print("2. 即使亮度或对比度不同，只要结构相似，SSIM仍然较高")
    print("3. 这就是为什么SSIM对结构错误敏感，而对亮度/对比度变化不敏感")
    print("=" * 60)

def compare_ssim_vs_l2():
    """对比SSIM损失和L2损失的差异"""
    print("\n" + "=" * 60)
    print("SSIM损失 vs L2损失的对比")
    print("=" * 60)
    
    # 创建两张图像：结构相同但亮度不同
    base = torch.randn(1, 1, 32, 32)
    pred = base
    target = base + 5.0  # 整体亮度+5
    
    # 计算L2损失
    l2_loss = F.mse_loss(pred, target).item()
    
    # 计算SSIM损失（直接使用main.py中的实现）
    def ssim_loss(pred, target, window_size=11, size_average=True):
        channel = pred.size(1)
        window = create_window(window_size, channel).to(pred.device)
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return (1 - ssim_map.mean()) / 2 if size_average else (1 - ssim_map) / 2
    
    ssim_loss_value = ssim_loss(pred, target).item()
    
    print(f"\n情况：两张图像结构完全相同，但亮度相差5.0")
    print(f"L2损失: {l2_loss:.6f} (较大，因为像素值差异大)")
    print(f"SSIM损失: {ssim_loss_value:.6f} (较小，因为结构相同)")
    print(f"\n结论：SSIM更关注结构而非像素值！")

if __name__ == '__main__':
    visualize_ssim_components()
    compare_ssim_vs_l2()

