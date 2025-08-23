import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class AdvancedGANLosses:
    """高级GAN损失函数集合"""
    
    @staticmethod
    def wasserstein_loss(real_scores, fake_scores, real_labels=1.0, fake_labels=-1.0):
        """
        Wasserstein GAN损失
        Args:
            real_scores: 真实样本的判别器分数
            fake_scores: 生成样本的判别器分数
            real_labels: 真实样本标签
            fake_labels: 生成样本标签
        """
        real_loss = -torch.mean(real_scores * real_labels)
        fake_loss = -torch.mean(fake_scores * fake_labels)
        return real_loss + fake_loss
    
    @staticmethod
    def gradient_penalty(discriminator, real_samples, fake_samples, lambda_gp=10.0):
        """
        梯度惩罚项，用于WGAN-GP
        Args:
            discriminator: 判别器网络
            real_samples: 真实样本
            fake_samples: 生成样本
            lambda_gp: 梯度惩罚权重
        """
        batch_size = real_samples[0].size(0)
        device = real_samples[0].device
        
        # 创建插值样本
        alpha = torch.rand(batch_size, 1, 1, 1).to(device)
        interpolated_samples = []
        
        for real, fake in zip(real_samples, fake_samples):
            interpolated = alpha * real + (1 - alpha) * fake
            interpolated.requires_grad_(True)
            interpolated_samples.append(interpolated)
        
        # 计算判别器对插值样本的输出
        d_interpolated = discriminator(interpolated_samples)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated_samples,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        
        # 计算梯度惩罚
        gradient_penalty = 0
        for grad in gradients:
            gradient_norm = grad.view(batch_size, -1).norm(2, dim=1)
            gradient_penalty += torch.mean((gradient_norm - 1) ** 2)
        
        return lambda_gp * gradient_penalty
    
    @staticmethod
    def relativistic_loss(real_scores, fake_scores, real_labels=1.0, fake_labels=0.0):
        """
        相对判别器损失 (Relativistic GAN)
        Args:
            real_scores: 真实样本的判别器分数
            fake_scores: 生成样本的判别器分数
        """
        real_loss = F.binary_cross_entropy_with_logits(
            real_scores - fake_scores.mean(0, keepdim=True), 
            torch.ones_like(real_scores)
        )
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_scores - real_scores.mean(0, keepdim=True), 
            torch.zeros_like(fake_scores)
        )
        return real_loss + fake_loss
    
    @staticmethod
    def feature_matching_loss(real_features, fake_features):
        """
        特征匹配损失，用于稳定训练
        Args:
            real_features: 真实样本的特征
            fake_features: 生成样本的特征
        """
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.mse_loss(real_feat, fake_feat)
        return loss
    
    @staticmethod
    def perceptual_loss(real_img, fake_img, vgg_model=None):
        """
        感知损失，使用预训练的VGG网络
        Args:
            real_img: 真实图像
            fake_img: 生成图像
            vgg_model: 预训练的VGG模型
        """
        if vgg_model is None:
            # 如果没有提供VGG模型，使用简单的L1损失
            return F.l1_loss(real_img, fake_img)
        
        real_features = vgg_model(real_img)
        fake_features = vgg_model(fake_img)
        
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.mse_loss(real_feat, fake_feat)
        return loss
    
    @staticmethod
    def consistency_loss(real_img, fake_img, discriminator):
        """
        一致性损失，确保判别器对相似输入给出相似输出
        Args:
            real_img: 真实图像
            fake_img: 生成图像
            discriminator: 判别器
        """
        real_scores = discriminator(real_img)
        fake_scores = discriminator(fake_img)
        
        # 计算分数的一致性
        consistency = F.mse_loss(real_scores, fake_scores)
        return consistency
    
    @staticmethod
    def diversity_loss(fake_samples, temperature=1.0):
        """
        多样性损失，鼓励生成器产生多样化的输出
        Args:
            fake_samples: 生成样本列表
            temperature: 温度参数
        """
        if len(fake_samples) < 2:
            return torch.tensor(0.0)
        
        diversity = 0
        for i in range(len(fake_samples)):
            for j in range(i+1, len(fake_samples)):
                # 计算样本间的余弦相似度
                sim = F.cosine_similarity(
                    fake_samples[i].view(fake_samples[i].size(0), -1),
                    fake_samples[j].view(fake_samples[j].size(0), -1),
                    dim=1
                )
                diversity += torch.mean(sim)
        
        # 最小化相似度，最大化多样性
        return diversity / (len(fake_samples) * (len(fake_samples) - 1) / 2)
    
    @staticmethod
    def cycle_consistency_loss(real_img, reconstructed_img):
        """
        循环一致性损失，用于确保重建质量
        Args:
            real_img: 原始图像
            reconstructed_img: 重建图像
        """
        return F.l1_loss(real_img, reconstructed_img)
    
    @staticmethod
    def identity_loss(real_img, identity_img):
        """
        身份损失，确保生成器保持输入的身份信息
        Args:
            real_img: 真实图像
            identity_img: 身份映射图像
        """
        return F.l1_loss(real_img, identity_img)

class ImprovedDiscriminatorLoss:
    """改进的判别器损失函数"""
    
    def __init__(self, loss_type='wasserstein', lambda_gp=10.0, lambda_fm=1.0):
        self.loss_type = loss_type
        self.lambda_gp = lambda_gp
        self.lambda_fm = lambda_fm
    
    def __call__(self, discriminator, real_samples, fake_samples, real_features=None, fake_features=None):
        """
        计算判别器损失
        Args:
            discriminator: 判别器网络
            real_samples: 真实样本
            fake_samples: 生成样本
            real_features: 真实样本特征（用于特征匹配）
            fake_features: 生成样本特征（用于特征匹配）
        """
        real_scores = discriminator(real_samples)
        fake_scores = discriminator(fake_samples)
        
        if self.loss_type == 'wasserstein':
            d_loss = AdvancedGANLosses.wasserstein_loss(real_scores, fake_scores)
            # 添加梯度惩罚
            gp_loss = AdvancedGANLosses.gradient_penalty(
                discriminator, real_samples, fake_samples, self.lambda_gp
            )
            return d_loss + gp_loss
        
        elif self.loss_type == 'relativistic':
            d_loss = AdvancedGANLosses.relativistic_loss(real_scores, fake_scores)
            return d_loss
        
        elif self.loss_type == 'feature_matching':
            d_loss = AdvancedGANLosses.wasserstein_loss(real_scores, fake_scores)
            if real_features is not None and fake_features is not None:
                fm_loss = AdvancedGANLosses.feature_matching_loss(real_features, fake_features)
                return d_loss + self.lambda_fm * fm_loss
            return d_loss
        
        else:
            # 默认使用Wasserstein损失
            return AdvancedGANLosses.wasserstein_loss(real_scores, fake_scores)

class ImprovedGeneratorLoss:
    """改进的生成器损失函数"""
    
    def __init__(self, lambda_adv=1.0, lambda_recon=10.0, lambda_perceptual=1.0, 
                 lambda_consistency=1.0, lambda_diversity=0.1):
        self.lambda_adv = lambda_adv
        self.lambda_recon = lambda_recon
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency
        self.lambda_diversity = lambda_diversity
    
    def __call__(self, discriminator, real_samples, fake_samples, real_img=None, fake_img=None,
                 real_features=None, fake_features=None, vgg_model=None):
        """
        计算生成器损失
        Args:
            discriminator: 判别器网络
            real_samples: 真实样本
            fake_samples: 生成样本
            real_img: 真实图像
            fake_img: 生成图像
            real_features: 真实特征
            fake_features: 生成特征
            vgg_model: VGG模型（用于感知损失）
        """
        fake_scores = discriminator(fake_samples)
        
        # 对抗损失
        adv_loss = -torch.mean(fake_scores)
        
        # 重建损失
        recon_loss = 0
        if real_features is not None and fake_features is not None:
            recon_loss = AdvancedGANLosses.feature_matching_loss(real_features, fake_features)
        
        # 感知损失
        perceptual_loss = 0
        if real_img is not None and fake_img is not None and vgg_model is not None:
            perceptual_loss = AdvancedGANLosses.perceptual_loss(real_img, fake_img, vgg_model)
        
        # 一致性损失
        consistency_loss = 0
        if real_samples is not None and fake_samples is not None:
            consistency_loss = AdvancedGANLosses.consistency_loss(real_samples, fake_samples, discriminator)
        
        # 多样性损失
        diversity_loss = 0
        if isinstance(fake_samples, list) and len(fake_samples) > 1:
            diversity_loss = AdvancedGANLosses.diversity_loss(fake_samples)
        
        total_loss = (self.lambda_adv * adv_loss + 
                     self.lambda_recon * recon_loss + 
                     self.lambda_perceptual * perceptual_loss + 
                     self.lambda_consistency * consistency_loss + 
                     self.lambda_diversity * diversity_loss)
        
        return total_loss, {
            'adv_loss': adv_loss.item(),
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'perceptual_loss': perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss,
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss
        }
