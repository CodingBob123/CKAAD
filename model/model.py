from torchvision import models
import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
import numpy as np
import math
from torch.nn.utils import spectral_norm


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
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
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
        super(Discriminator, self).__init__()
        self.expansion = expansion
        input_channels = [c * self.expansion for c in input_channels]
        layers = []
        for s, c in zip(input_sizes, input_channels):
            layer = []
            while s > input_sizes[-1]:
                layer.append(nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1, stride=2, bias=False)),
                    nn.InstanceNorm2d(c * 2),
                    nn.LeakyReLU(0.1, inplace=True),
                ))
                s = s // 2
                c = c * 2
            layers.append(nn.Sequential(*layer))
        self.layers = nn.ModuleList(layers)
        
        layers = []
        in_channels = input_channels[-1] * len(input_channels)
        out_channels = input_channels[-1]
        size = input_sizes[-1]
        while size > 2:
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2, bias=False)),
                nn.InstanceNorm2d(input_channels[-1]),
                nn.LeakyReLU(0.1, inplace=True)
            ))
            in_channels = out_channels
            size = size // 2
        
        layers.append(spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding=0, stride=2, bias=False)))
        self.layer1 = nn.Sequential(*layers)
        
        self.cls_layer = nn.Sequential(
            spectral_norm(nn.Linear(input_channels[-1], input_channels[-1] // 4, bias=False)),
            nn.InstanceNorm1d(input_channels[-1] // 4),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Linear(input_channels[-1] // 4, 1, bias=False))
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """使用正交初始化来改善谱归一化的效果"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = [self.layers[i](xi) for i, xi in enumerate(x)]
        x = torch.cat(x, dim=1)
        z = self.layer1(x)
        z = z.view(z.size(0), -1)
        score = self.cls_layer(z)
        return score
            
    def extract_features(self, x):
        """提取中间层特征用于特征匹配损失"""
        features = []
        x_processed = [self.layers[i](xi) for i, xi in enumerate(x)]
        features.extend(x_processed)
        
        x_cat = torch.cat(x_processed, dim=1)
        for i, layer in enumerate(self.layer1[:-1]):  # 除了最后一层
            x_cat = layer(x_cat)
            features.append(x_cat)
            
        return features
            