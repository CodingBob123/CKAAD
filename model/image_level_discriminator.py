import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from model.attention_utils import SpatialAttentionBlock, get_positional_encoding_2d


class ImageLevelDiscriminator(nn.Module):
    def __init__(self, input_sizes=[64, 32, 16], input_channels=[64, 128, 256], expansion=4, 
                 use_spectral_norm=True, use_attention=True, use_position_encoding=True):
        super(ImageLevelDiscriminator, self).__init__()
        self.expansion = expansion
        self.use_position_encoding = use_position_encoding
        
        # Apply expansion to input channels
        input_channels = [c * self.expansion for c in input_channels]
        
        # Initial feature downsampling layers for each scale
        layers = []
        for s, c in zip(input_sizes, input_channels):
            layer = []
            while s > input_sizes[-1]:
                conv = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=3, padding=1, stride=2, bias=False)
                if use_spectral_norm:
                    conv = spectral_norm(conv)
                    
                layer.append(nn.Sequential(
                    conv,
                    nn.InstanceNorm2d(c * 2),
                    nn.LeakyReLU(0.1, inplace=False)
                ))
                s = s // 2
                c = c * 2
            layers.append(nn.Sequential(*layer))
        self.layers = nn.ModuleList(layers)
        
        # Multi-head attention blocks for each scale after downsampling
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                SpatialAttentionBlock(input_channels[-1], use_spectral_norm=use_spectral_norm)
                for _ in range(len(input_sizes))
            ])
        else:
            self.attention_blocks = None
        
        # Feature pyramid for further processing
        layers = []
        in_channels = input_channels[-1] * len(input_channels)
        out_channels = input_channels[-1]
        size = input_sizes[-1]
        
        while size > 2:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=3, padding=1, stride=2, bias=False)
            if use_spectral_norm:
                conv = spectral_norm(conv)
                
            layers.append(nn.Sequential(
                conv,
                nn.InstanceNorm2d(input_channels[-1]),
                nn.LeakyReLU(0.1, inplace=False)
            ))
            in_channels = out_channels
            size = size // 2
        
        # Final 2x2 convolution to 1x1
        final_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                             kernel_size=2, padding=0, stride=2, bias=False)
        if use_spectral_norm:
            final_conv = spectral_norm(final_conv)
        layers.append(final_conv)
        
        self.layer1 = nn.Sequential(*layers)
        
        # Classification head
        fc1 = nn.Linear(input_channels[-1], input_channels[-1] // 4, bias=False)
        fc2 = nn.Linear(input_channels[-1] // 4, 1, bias=False)
        if use_spectral_norm:
            fc1 = spectral_norm(fc1)
            fc2 = spectral_norm(fc2)
            
        self.cls_layer = nn.Sequential(
            fc1,
            nn.InstanceNorm1d(input_channels[-1] // 4),
            nn.LeakyReLU(0.1, inplace=False),
            fc2,
            nn.Softplus()  # Ensure non-negative energy output
        )

    def forward(self, x):
        batch_size = x[0].size(0)
        
        # Process each scale with position encoding if enabled
        processed_features = []
        for i, xi in enumerate(x):
            if self.use_position_encoding:
                # Create and add positional encoding
                pos_enc = get_positional_encoding_2d(
                    2, xi.size(2), xi.size(3), xi.device
                ).expand(batch_size, -1, -1, -1)
                
                # Concatenate position encoding in channel dimension
                xi = torch.cat([xi, pos_enc], dim=1)
                
                # Project back to original channel dimension with 1x1 conv
                if not hasattr(self, f'pos_proj_{i}'):
                    self.register_module(
                        f'pos_proj_{i}', 
                        spectral_norm(nn.Conv2d(xi.size(1), x[i].size(1), kernel_size=1, bias=False))
                    )
                pos_proj = getattr(self, f'pos_proj_{i}')
                xi = pos_proj(xi)
            
            # Apply downsampling layers
            xi = self.layers[i](xi)
            
            # Apply attention if enabled
            if self.attention_blocks is not None:
                xi = self.attention_blocks[i](xi)
                
            processed_features.append(xi)
        
        # Concatenate features from all scales
        x = torch.cat(processed_features, dim=1)
        
        # Apply feature pyramid
        z = self.layer1(x)
        z = z.view(z.size(0), -1)
        
        # Apply classification head
        score = self.cls_layer(z)
        
        return score 