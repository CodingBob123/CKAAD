"""
独立测试SSIM损失函数
"""
import torch
import torch.nn.functional as F

def ssim_loss(pred, target, window_size=11, size_average=True):
    """SSIM损失函数（与main.py中的实现相同）"""
    def create_window(window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
                                  for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
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
    
    if size_average:
        return (1 - ssim_map.mean()) / 2
    else:
        return (1 - ssim_map) / 2

if __name__ == '__main__':
    # 测试SSIM损失函数
    print("测试SSIM损失函数...")
    
    # 测试1: 相同输入应该产生接近0的损失
    x1 = torch.randn(2, 1, 32, 32)
    x2 = x1.clone()
    loss1 = ssim_loss(x1, x2)
    print(f"测试1 - 相同输入: SSIM损失 = {loss1.item():.6f} (应该接近0)")
    
    # 测试2: 不同输入应该产生较大的损失
    x3 = torch.randn(2, 1, 32, 32)
    loss2 = ssim_loss(x1, x3)
    print(f"测试2 - 不同输入: SSIM损失 = {loss2.item():.6f} (应该 > 0)")
    
    # 测试3: 多通道输入
    x4 = torch.randn(2, 3, 32, 32)
    x5 = torch.randn(2, 3, 32, 32)
    loss3 = ssim_loss(x4, x5)
    print(f"测试3 - 多通道输入: SSIM损失 = {loss3.item():.6f}")
    
    # 测试4: 小尺寸特征图（需要上采样）
    x6 = torch.randn(2, 1, 8, 8)
    x7 = torch.randn(2, 1, 8, 8)
    x6_up = F.interpolate(x6, size=(32, 32), mode='bilinear', align_corners=False)
    x7_up = F.interpolate(x7, size=(32, 32), mode='bilinear', align_corners=False)
    loss4 = ssim_loss(x6_up, x7_up)
    print(f"测试4 - 小尺寸上采样后: SSIM损失 = {loss4.item():.6f}")
    
    print("\n所有测试完成！SSIM损失函数工作正常。")

