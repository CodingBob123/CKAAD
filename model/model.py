from torchvision import models
import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
import numpy as np
import math


class PretrainedFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, layers=[2], image_size=256):
        super(PretrainedFeatureExtractor, self).__init__()
        weight = None
        self.expansion = 1
        default_channels = [64, 128, 256, 512]
        default_output_sizes = [image_size // 4, image_size // 8, image_size // 16, image_size // 32]
        if backbone == 'resnet18':
            weight = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weight)
        elif backbone == 'resnet34':
            weight = models.ResNet34_Weights.DEFAULT
            self.backbone = models.resnet34(weights=weight)
        elif backbone == 'resnet50':
            weight = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weight)
            self.expansion = 4
        elif backbone == 'resnet101':
            weight = models.ResNet101_Weights.DEFAULT
            self.backbone = models.resnet101(weights=weight)
            self.expansion = 4
            
        elif backbone == 'resnet152':
            weight = models.ResNet152_Weights.DEFAULT
            self.backbone = models.resnet152(weights=weight)
            self.expansion = 4
            
        elif backbone == 'wide_resnet50_2':
            if pretrained:
                weight = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            self.backbone = models.wide_resnet50_2(weights=weight)
            self.expansion = 4
            
        elif backbone == 'wide_resnet101_2':
            if pretrained:
                weight = models.Wide_ResNet101_2_Weights.IMAGENET1K_V1
            self.backbone = models.wide_resnet101_2(weights=weight)
            self.expansion = 4
        
        self.output_channels = []
        self.output_sizes = []
        for layer in layers:
            self.output_channels.append(default_channels[layer - 1])
            self.output_sizes.append(default_output_sizes[layer - 1])
        self.output_layers = layers
        
    def forward(self, x):
        """
        
         层	            操作	                输出形状	        通道数变化	        空间尺寸变化
        输入	    -	                [N, 3, 256, 256]	-	-
        conv1	7×7卷积, stride=2	[N, 64, 128, 128]	            3→64	        256→128   # 这里就是预训练模型得到的特征层数，这里就是预训练模型得到的特征层数
        maxpool	3×3池化, stride=2	[N, 64, 64, 64]	                64→64	        128→64   # 这里就是预训练模型得到的特征层数，这里就是预训练模型得到的特征层数
        layer1	3×Bottleneck	    [N, 256, 64, 64]	            64→256	        64→64
        layer2	4×Bottleneck	    [N, 512, 32, 32]	            256→512	        64→32
        layer3	6×Bottleneck	    [N, 1024, 16, 16]	            512→1024	    32→16
        layer4	3×Bottleneck	    [N, 2048, 8, 8]	                1024→2048	    16→8
        
        """
        #  x: [N, 3, 256, 256]
        x = self.backbone.conv1(x)  # 7x7卷积，stride=2  [N, 64, 128, 128]
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  # 3x3最大池化，stride=2  [N, 64, 64, 64]  

        # 这里就是预训练模型得到的特征层数
        x1 = self.backbone.layer1(x)  # [N, 256, 64, 64]
        x2 = self.backbone.layer2(x1)  # [N, 512, 32, 32]
        x3 = self.backbone.layer3(x2)  # [N, 1024, 16, 16]
        x4 = self.backbone.layer4(x3)  # [N, 2048, 8, 8]
        outputs = []
        if 1 in self.output_layers:
            outputs.append(x1)
        if 2 in self.output_layers:
            outputs.append(x2)
        if 3 in self.output_layers:
            outputs.append(x3)   
        if 4 in self.output_layers:
            outputs.append(x4) 
        return outputs


class ED(nn.Module):
    def __init__(self, backbone='resnet18', input_channels=[64, 128, 256]):
        super(ED, self).__init__()
        self.encoder = Encoder(backbone=backbone, input_channels=input_channels)
        self.decoder = Decoder(backbone=backbone, output_channels=input_channels)
    
    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return o
    
class Discriminator(nn.Module):
    def __init__(self, input_sizes=[64, 32, 16], input_channels=[64, 128, 256], expansion=4):
        """
        初始化判别器网络
        
        参数:
            input_sizes: 输入特征图的空间尺寸列表，默认 [64, 32, 16]
            input_channels: 输入特征图的基础通道数列表，默认 [64, 128, 256]
            expansion: 通道扩展系数，默认 4
        
        假设输入三个特征图 x1, x2, x3，形状分别为:
            x1: [N, 256, 64, 64] (256 = 64*expansion)
            x2: [N, 512, 32, 32] (512 = 128*expansion)
            x3: [N, 1024, 16, 16] (1024 = 256*expansion)
        """
        super(Discriminator, self).__init__()
        
        # 存储扩展系数
        self.expansion = expansion
        
        # 将基础通道数乘以扩展系数，得到实际通道数
        # 例如: [64, 128, 256] → [256, 512, 1024] (当 expansion=4)
        input_channels = [c * self.expansion for c in input_channels]
        
        # 第一部分：构建特征对齐层，将不同尺寸的特征图下采样到相同的空间尺寸
        layers = []
        for s, c in zip(input_sizes, input_channels):
            layer = []
            # 如果当前尺寸大于目标尺寸(input_sizes[-1] 16)，则进行下采样
            while s > input_sizes[-1]:
                # 添加一个下采样块: 3x3卷积(stride=2) + 实例归一化 + LeakyReLU
                layer.append(nn.Sequential(
                    # 每次下采样：空间尺寸减半，通道数翻倍
                    # 例如: [N, c, s, s] → [N, c*2, s/2, s/2]
                    nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.InstanceNorm2d(c * 2),
                    nn.LeakyReLU(0.1, inplace=True),
                ))
                s = s // 2  # 更新尺寸（减半）
                c = c * 2   # 更新通道数（翻倍）
            layers.append(nn.Sequential(*layer))
        
        # 存储特征对齐层
        # 对于默认参数:
        # layers[0]: 包含两个下采样层，将 [N, 256, 64, 64] → [N, 1024, 16, 16]
        # layers[1]: 包含一个下采样层，将 [N, 512, 32, 32] → [N, 1024, 16, 16]
        # layers[2]: 空序列，[N, 1024, 16, 16] 已经是目标尺寸
        self.layers = nn.ModuleList(layers)
        
        # 第二部分：构建共享特征处理层
        layers = []
        # 初始输入通道数是所有特征通道数之和（因为后面会拼接）
        # 例如: 1024 * 3 = 3072 通道
        in_channels = input_channels[-1] * len(input_channels)
        # 输出通道数等于最后一个特征的通道数
        # 例如: 1024 通道
        out_channels = input_channels[-1]
        # 从最小的空间尺寸开始
        # 例如: size = 16
        size = input_sizes[-1]
        
        # 逐步下采样，直到尺寸小于2
        while size > 2:
            layers.append(nn.Sequential(
                # 每次下采样：空间尺寸减半，通道数保持不变
                # 第一次: [N, 3072, 16, 16] → [N, 1024, 8, 8]
                # 第二次: [N, 1024, 8, 8] → [N, 1024, 4, 4]
                # 第三次: [N, 1024, 4, 4] → [N, 1024, 2, 2]
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2, bias=False),
                nn.InstanceNorm2d(input_channels[-1]),
                nn.LeakyReLU(0.1, inplace=True)
            ))
            in_channels = out_channels  # 更新输入通道数
            size = size // 2            # 更新尺寸（减半）
        
        # 最后一个下采样层，将 [N, 1024, 2, 2] → [N, 1024, 1, 1]
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=0, stride=2, bias=False))
        self.layer1 = nn.Sequential(*layers)
        
        # 第三部分：构建分类层
        self.cls_layer = nn.Sequential(
            # 第一个线性层: [N, 1024] → [N, 256]
            nn.Linear(input_channels[-1], input_channels[-1] // 4, bias=False),
            nn.InstanceNorm1d(input_channels[-1] // 4),
            nn.LeakyReLU(0.1, inplace=True),
            # 第二个线性层: [N, 256] → [N, 1]
            nn.Linear(input_channels[-1] // 4, 1, bias=False)
        )
    
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图列表，例如 [x1, x2, x3]
               x1: [N, 256, 64, 64]
               x2: [N, 512, 32, 32]
               x3: [N, 1024, 16, 16]
        
        返回:
            score: 判别分数，形状为 [N, 1]
        """
        # 步骤1: 对每个特征图应用对应的下采样层，使所有特征图具有相同的尺寸
        # 输出: x = [x1', x2', x3']
        #       x1': [N, 1024, 16, 16] (经过两次下采样)
        #       x2': [N, 1024, 16, 16] (经过一次下采样)
        #       x3': [N, 1024, 16, 16] (无需下采样)
        x = [self.layers[i](xi) for i, xi in enumerate(x)]
        
        # 步骤2: 在通道维度上拼接所有特征
        # 输出: [N, 3072, 16, 16] (3*1024 = 3072通道)
        x = torch.cat(x, dim=1)
        
        # 步骤3: 应用共享特征处理层，逐步下采样
        # 输出: [N, 1024, 1, 1]
        z = self.layer1(x)
        
        # 步骤4: 展平特征
        # 输出: [N, 1024]
        z = z.view(z.size(0), -1)
        
        # 步骤5: 应用分类层
        # 输出: [N, 1]
        score = self.cls_layer(z)
        
        return score
    
    def calculate_loss(self, x, label_value, margin=5.0):
        """
        计算损失函数
        
        参数:
            x: 输入特征图列表
            label_value: 目标标签值（0表示真实，1表示虚假）
            margin: 边界值，默认5.0
            
        返回:
            loss: 计算得到的损失值
        """
        # 获取判别分数
        # 输入: 特征图列表
        # 输出: [N, 1]
        score = self(x)
        
        # 取绝对值并展平
        # 输出: [N]
        score = torch.abs(score.view(-1))
        
        # 创建与score相同形状的标签
        # 输出: [N]
        label = torch.ones_like(score) * label_value
        
        # 计算损失
        # 对于真实样本(label=0): loss = score
        # 对于虚假样本(label=1): loss = max(0, margin - score)
        loss = ((1 - label) * score + label * (margin - score).clamp_(min=0.)).mean()
        
        return loss
    

class Discriminator(nn.Module):
    def __init__(self, input_sizes=[64, 32, 16], input_channels=[64, 128, 256], expansion=4):
        """
        初始化细粒度级别的判别器网络
        
        参数:
            input_sizes: 输入特征图的空间尺寸列表，默认 [64, 32, 16]
            input_channels: 输入特征图的基础通道数列表，默认 [64, 128, 256]
            expansion: 通道扩展系数，默认 4
        
        假设输入三个特征图 x1, x2, x3，形状分别为:
            x1: [N, 256, 64, 64] (256 = 64*expansion)
            x2: [N, 512, 32, 32] (512 = 128*expansion)
            x3: [N, 1024, 16, 16] (1024 = 256*expansion)
        """
        super(Discriminator, self).__init__()
        
        # 存储扩展系数
        self.expansion = expansion
        
        # 将基础通道数乘以扩展系数，得到实际通道数
        # 例如: [64, 128, 256] → [256, 512, 1024] (当 expansion=4)
        input_channels = [c * self.expansion for c in input_channels]
        
        # 创建层列表和位置嵌入列表
        layers = []
        positional_embeds = []
        
        # 为每个输入特征图创建对应的处理层
        for s, c in zip(input_sizes, input_channels):
            # 每个特征图使用三个1x1卷积层进行处理
            # 这种设计保持空间尺寸不变，仅在通道维度上进行变换
            layers.append(
                nn.Sequential(
                    # 第一个1x1卷积: [N, c, s, s] → [N, c, s, s]
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    # 第二个1x1卷积: [N, c, s, s] → [N, c, s, s]
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    # 第三个1x1卷积: [N, c, s, s] → [N, 1, s, s]
                    # 将通道数降为1，生成逐像素的判别分数
                    nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
                )
            )
            
            # 为每个特征图创建可学习的位置嵌入
            # 形状: [1, c, s, s]
            positional_embeds.append(nn.Parameter(torch.randn(1, c, s, s), requires_grad=True))
        
        # 存储层和位置嵌入
        self.layers = nn.ModuleList(layers)
        self.positional_embeds = nn.ParameterList(positional_embeds)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图列表，例如 [x1, x2, x3]
               x1: [N, 256, 64, 64]
               x2: [N, 512, 32, 32]
               x3: [N, 1024, 16, 16]
        
        返回:
            scores: 每个特征图的判别分数列表
                   scores[0]: [N, 64*64] (从x1生成的分数，展平的64x64像素分数)
                   scores[1]: [N, 32*32] (从x2生成的分数，展平的32x32像素分数)
                   scores[2]: [N, 16*16] (从x3生成的分数，展平的16x16像素分数)
        """
        # 获取批次大小
        b = x[0].size(0)
        
        # 步骤1: 对每个特征图进行通道维度上的归一化
        # 这有助于稳定训练并使模型对特征幅度不敏感
        # 输入: x = [x1, x2, x3]
        # 输出: x = [x1_norm, x2_norm, x3_norm]
        #       每个张量的通道维度L2范数为1
        x = [torch.nn.functional.normalize(xi, dim=1) for xi in x]
        
        # 步骤2: 对每个特征图应用对应的卷积层，并将输出展平为二维张量
        # 输入: x = [x1_norm, x2_norm, x3_norm]
        # 处理过程:
        #   x1_norm: [N, 256, 64, 64] → [N, 1, 64, 64] → [N, 64*64]
        #   x2_norm: [N, 512, 32, 32] → [N, 1, 32, 32] → [N, 32*32]
        #   x3_norm: [N, 1024, 16, 16] → [N, 1, 16, 16] → [N, 16*16]
        # 输出: scores = [score1, score2, score3]
        # -1 是一个特殊值，表示PyTorch应该自动计算这个维度的大小，以保证总元素数量不变
        scores = [self.layers[i](xi).view(b, -1) for i, xi in enumerate(x)]
        
        # 返回所有特征图的分数列表
        return scores
    
    def calculate_loss(self, x, label_value, margin=5.0):
        """
        计算损失函数
        
        参数:
            x: 输入特征图列表
            label_value: 目标标签值（0表示真实，1表示虚假）
            margin: 边界值，默认5.0
            
        返回:
            loss: 计算得到的损失值
        """
        # 获取所有特征图的判别分数
        # scores = [score1, score2, score3]
        scores = self(x)
        
        loss = 0
        # 对每个特征图的分数单独计算损失，然后累加
        for score in scores:
            # 取分数的绝对值并展平为一维张量
            # 例如: [N, 64*64] → [N*64*64]
            score = torch.abs(score.view(-1))
            
            # 创建与score相同形状的标签
            # 输出: [N*64*64]
            label = torch.ones_like(score) * label_value
            
            # 计算损失并累加
            # 对于真实样本(label=0): loss += score
            # 对于虚假样本(label=1): loss += max(0, margin - score)
            loss += ((1 - label) * score + label * (margin - score).clamp_(min=0.)).mean()
            
        return loss
            