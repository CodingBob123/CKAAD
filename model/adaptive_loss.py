import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveReconstructionLoss(nn.Module):
    """自适应重建损失函数"""
    
    def __init__(self, initial_alpha=1.0, initial_beta=0.5, adaptation_rate=0.01):
        super(AdaptiveReconstructionLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.beta = nn.Parameter(torch.tensor(initial_beta))
        self.adaptation_rate = adaptation_rate
        
        # 记录历史损失
        self.feature_loss_history = []
        self.pixel_loss_history = []
        
    def forward(self, normal_inputs, normal_outputs, recon_img, normal_img):
        """
        计算自适应重建损失
        Args:
            normal_inputs: 原始特征
            normal_outputs: 重建特征
            recon_img: 重建图像
            normal_img: 原始图像
        """
        # 特征重建损失（余弦相似度）
        cos_loss = torch.nn.CosineSimilarity()
        feature_loss = 0
        for item in range(len(normal_inputs)):
            feature_loss += torch.mean(1-cos_loss(
                normal_inputs[item].view(normal_outputs[item].shape[0], -1),
                normal_outputs[item].view(normal_outputs[item].shape[0], -1)
            ))
        
        # 像素重建损失
        if recon_img.shape[-2:] != normal_img.shape[-2:]:
            recon_img = F.interpolate(recon_img, size=normal_img.shape[-2:], 
                                    mode='bilinear', align_corners=False)
        pixel_loss = F.mse_loss(recon_img, normal_img)
        
        # 记录历史损失
        self.feature_loss_history.append(feature_loss.item())
        self.pixel_loss_history.append(pixel_loss.item())
        
        # 自适应权重调整
        if len(self.feature_loss_history) > 10:
            self._adapt_weights()
        
        # 总损失
        total_loss = self.alpha * feature_loss + self.beta * pixel_loss
        
        return total_loss, {
            'feature_loss': feature_loss.item(),
            'pixel_loss': pixel_loss.item(),
            'alpha': self.alpha.item(),
            'beta': self.beta.item()
        }
    
    def _adapt_weights(self):
        """自适应调整权重"""
        if len(self.feature_loss_history) < 10:
            return
            
        # 计算最近10个epoch的平均损失
        recent_feature_loss = torch.mean(torch.tensor(self.feature_loss_history[-10:]))
        recent_pixel_loss = torch.mean(torch.tensor(self.pixel_loss_history[-10:]))
        
        # 归一化损失
        total_loss = recent_feature_loss + recent_pixel_loss
        if total_loss > 0:
            feature_ratio = recent_feature_loss / total_loss
            pixel_ratio = recent_pixel_loss / total_loss
            
            # 调整权重
            with torch.no_grad():
                self.alpha.data = torch.clamp(
                    self.alpha + self.adaptation_rate * (0.5 - feature_ratio), 
                    min=0.1, max=2.0
                )
                self.beta.data = torch.clamp(
                    self.beta + self.adaptation_rate * (0.5 - pixel_ratio), 
                    min=0.1, max=2.0
                )

class MultiScaleReconstructionLoss(nn.Module):
    """多尺度重建损失函数"""
    
    def __init__(self, scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        super(MultiScaleReconstructionLoss, self).__init__()
        self.scales = scales
        self.weights = weights
        
    def forward(self, recon_img, normal_img):
        """
        计算多尺度重建损失
        Args:
            recon_img: 重建图像
            normal_img: 原始图像
        """
        total_loss = 0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                # 下采样到指定尺度
                target_size = (int(normal_img.shape[-2] * scale), 
                             int(normal_img.shape[-1] * scale))
                scaled_recon = F.interpolate(recon_img, size=target_size, 
                                           mode='bilinear', align_corners=False)
                scaled_normal = F.interpolate(normal_img, size=target_size, 
                                            mode='bilinear', align_corners=False)
            else:
                scaled_recon = recon_img
                scaled_normal = normal_img
            
            # 计算该尺度的损失
            scale_loss = F.mse_loss(scaled_recon, scaled_normal)
            total_loss += weight * scale_loss
        
        return total_loss

class PerceptualReconstructionLoss(nn.Module):
    """感知重建损失函数"""
    
    def __init__(self, vgg_model=None, lambda_perceptual=1.0):
        super(PerceptualReconstructionLoss, self).__init__()
        self.vgg_model = vgg_model
        self.lambda_perceptual = lambda_perceptual
        
    def forward(self, recon_img, normal_img):
        """
        计算感知重建损失
        Args:
            recon_img: 重建图像
            normal_img: 原始图像
        """
        # 基础像素损失
        pixel_loss = F.mse_loss(recon_img, normal_img)
        
        # 感知损失
        perceptual_loss = 0
        if self.vgg_model is not None:
            try:
                recon_features = self.vgg_model(recon_img)
                normal_features = self.vgg_model(normal_img)
                perceptual_loss = F.mse_loss(recon_features, normal_features)
            except:
                # 如果VGG模型不可用，使用简单的L1损失
                perceptual_loss = F.l1_loss(recon_img, normal_img)
        
        total_loss = pixel_loss + self.lambda_perceptual * perceptual_loss
        
        return total_loss, {
            'pixel_loss': pixel_loss.item(),
            'perceptual_loss': perceptual_loss.item()
        }
