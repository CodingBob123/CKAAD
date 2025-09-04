import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from model.attention_utils import SpatialAttentionBlock, get_positional_encoding_2d


class PatchLevelDiscriminator(nn.Module):
    def __init__(self, input_sizes=[64, 32, 16], input_channels=[64, 128, 256], expansion=4,
                 use_spectral_norm=True, use_attention=True, use_position_encoding=True, margin=5.0):
        super(PatchLevelDiscriminator, self).__init__()
        self.expansion = expansion
        self.use_position_encoding = use_position_encoding
        self.margin = margin
        
        # Apply expansion to input channels
        input_channels = [c * self.expansion for c in input_channels]
        
        # Create patch-level discriminator layers for each scale
        layers = []
        positional_embeds = []
        
        for s, c in zip(input_sizes, input_channels):
            # Create position encodings for each scale
            if use_position_encoding:
                positional_embeds.append(nn.Parameter(torch.randn(1, c, s, s), requires_grad=True))
            else:
                positional_embeds.append(None)
            
            # Create patch discriminator with 1x1 convs
            patch_conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, bias=False)
            patch_conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, stride=1, padding=0, bias=False)
            patch_conv3 = nn.Conv2d(in_channels=c, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
            
            if use_spectral_norm:
                patch_conv1 = spectral_norm(patch_conv1)
                patch_conv2 = spectral_norm(patch_conv2)
                patch_conv3 = spectral_norm(patch_conv3)
            
            # Create attention block if enabled
            if use_attention:
                attention_block = SpatialAttentionBlock(c, use_spectral_norm=use_spectral_norm)
                layers.append(
                    nn.Sequential(
                        attention_block,
                        patch_conv1,
                        nn.LeakyReLU(0.1, inplace=False),
                        patch_conv2,
                        nn.LeakyReLU(0.1, inplace=False),
                        patch_conv3,
                        nn.Softplus()  # Ensure non-negative energy output
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        patch_conv1,
                        nn.LeakyReLU(0.1, inplace=False),
                        patch_conv2,
                        nn.LeakyReLU(0.1, inplace=False),
                        patch_conv3,
                        nn.Softplus()  # Ensure non-negative energy output
                    )
                )
                
        self.layers = nn.ModuleList(layers)
        self.positional_embeds = nn.ParameterList(positional_embeds) if use_position_encoding else None

    def forward(self, x):
        b = x[0].size(0)
        
        # Apply channel normalization to each scale
        x = [F.normalize(xi, dim=1) for xi in x]
        
        # Apply position encoding if enabled
        if self.use_position_encoding and self.positional_embeds is not None:
            x = [xi + self.positional_embeds[i] for i, xi in enumerate(x)]
        
        # Apply patch discriminator to each scale
        scores = [self.layers[i](xi).view(b, -1) for i, xi in enumerate(x)]
        
        return scores

    def calculate_loss(self, x, label_value, margin=None):
        """
        Calculate bilateral hinge loss for patch-level discriminator
        label_value: 0 for anomaly/fake, 1 for normal/real
        margin: margin for hinge loss, defaults to self.margin
        """
        if margin is None:
            margin = self.margin
            
        scores = self(x)
        loss = 0
        
        for score in scores:
            # We don't use abs() to preserve sign information
            score = score.view(-1)  # Flatten to (B*H*W)
            label = torch.ones_like(score) * label_value
            
            # Bilateral hinge loss:
            # - For normal samples (label=1): penalize (margin - score) if score < margin
            # - For anomaly samples (label=0): penalize score directly
            loss += ((1 - label) * score + label * F.relu(margin - score)).mean()
            
        return loss 