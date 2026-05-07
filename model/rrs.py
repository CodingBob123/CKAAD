"""
RRS (Reconstruction Residual Selection) module adapted from RealNet for CKAAD.

Core idea:
  1. Compute L2-normalized per-channel MSE residuals between input features and reconstructed features
     (equivalent to channel-wise decomposition of cosine distance, scale-invariant)
  2. Align multi-scale residuals to the highest resolution
  3. Select the most discriminative channels via topk (max/mean modes)
  4. Decode selected channels into a 2-class segmentation output (normal/anomaly)

Reference: RealNet (CVPR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_channels):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class RRS(nn.Module):
    """
    Reconstruction Residual Selection module adapted for CKAAD.

    Takes multi-scale reconstruction residuals as input,
    computed as L2-normalized per-channel MSE: (norm(inp) - norm(out))^2.
    This is equivalent to channel-wise cosine distance decomposition.
    selects the most discriminative channels via topk (max/mean modes),
    and decodes them into a 2-class segmentation output.

    Args:
        layer_channels: list of channels per layer, e.g. [256, 512, 1024]
        layer_strides:  list of stride per layer, e.g. [4, 8, 16]
        modes: residual selection strategies, e.g. ['max', 'mean']
        mode_numbers: channels to select per mode, e.g. [256, 256]
        num_residual_layers: number of residual blocks in decoder1
        stop_grad: whether to detach input residuals before selection
    """

    def __init__(self,
                 layer_channels,
                 layer_strides,
                 modes=['max', 'mean'],
                 mode_numbers=None,
                 num_residual_layers=2,
                 stop_grad=False,
                 ):
        super(RRS, self).__init__()

        self.num_layers = len(layer_channels)
        self.layer_channels = layer_channels
        self.layer_strides = layer_strides
        self.modes = modes
        self.stop_grad = stop_grad
        self.num_residual_layers = num_residual_layers

        # Auto-set mode_numbers if not provided
        total_channels = sum(layer_channels)
        if mode_numbers is None:
            per_mode = min(total_channels // len(modes), 256)
            mode_numbers = [per_mode] * len(modes)
        self.mode_numbers = mode_numbers
        self.total_select_number = sum(self.mode_numbers)

        # Upsampling layers: align all residuals to the smallest stride (highest resolution)
        align_stride = min(layer_strides)
        for i in range(self.num_layers):
            scale_factor = layer_strides[i] / align_stride
            self.add_module(f"layer{i}_upsample",
                            nn.UpsamplingBilinear2d(scale_factor=scale_factor))

        # Total channels after concatenation
        align_inplane = sum(layer_channels)

        # BN for index selection (no learnable affine params, used only for ranking)
        self.bn_idx = nn.BatchNorm2d(align_inplane, momentum=0.9, affine=False)

        # Decoder
        self.decoder1 = nn.Sequential(
            ResidualStack(self.total_select_number, self.num_residual_layers),
            nn.Conv2d(self.total_select_number, 128, 3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(8, 4, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(4, 2, 3, padding=1, bias=True),
        )

    @torch.no_grad()
    def select_ano_index(self, residual, mode, k):
        """
        Select top-k channel indices based on spatial aggregation mode.

        Args:
            residual: [B, C, H, W]
            mode: 'max' or 'mean'
            k: number of channels to select

        Returns:
            idxs: [B, k] channel indices
        """
        B, C, W, H = residual.size()
        residual = residual.view((B, C, W * H))
        if mode == 'max':
            residual, _ = torch.max(residual, dim=-1)
        elif mode == 'mean':
            residual = torch.mean(residual, dim=-1)
        else:
            raise ValueError(f"mode must be in [max, mean], got {mode}")
        _, idxs = torch.topk(residual, dim=1, largest=True, k=k, sorted=True)
        return idxs

    def compute_residuals(self, inputs, outputs):
        """
        Compute L2-normalized per-channel MSE residuals.

        L2 normalize features first, then compute per-channel MSE.
        Mathematically: ||norm(x) - norm(y)||^2 = 2(1 - cos(x,y))
        This is equivalent to a channel-wise decomposition of cosine distance,
        preserving channel-level variation while being scale-invariant.

        Args:
            inputs:  list of [B, C_i, H_i, W_i] - original features from PFE
            outputs: list of [B, C_i, H_i, W_i] - reconstructed features from AE

        Returns:
            list of [B, C_i, H_i, W_i] - per-channel normalized MSE residuals
        """
        residuals = []
        for inp, out in zip(inputs, outputs):
            inp_norm = F.normalize(inp, dim=1, eps=1e-6)
            out_norm = F.normalize(out, dim=1, eps=1e-6)
            residual = (inp_norm - out_norm) ** 2  # per-channel normalized MSE
            residuals.append(residual)
        return residuals

    def _align_features(self, features):
        return [getattr(self, f"layer{i}_upsample")(features[i])
                for i in range(self.num_layers)]

    def _select_indices(self, residual_cat):
        residual_idx = self.bn_idx(residual_cat)
        B, _, H, W = residual_cat.size()
        selected_idxs = []
        for mode, mode_n in zip(self.modes, self.mode_numbers):
            idxs = self.select_ano_index(residual_idx, mode, mode_n)
            selected_idxs.append(idxs.view((B, mode_n, 1, 1)).repeat(1, 1, H, W))
        return selected_idxs

    def rrs_cosine_map(self, inputs, outputs, out_size, amap_mode='add'):
        """Compute baseline-style cosine map on RRS-selected channels."""
        residuals = self.compute_residuals(inputs, outputs)
        if self.stop_grad:
            residuals = [r.detach() for r in residuals]

        residual_cat = torch.cat(self._align_features(residuals), dim=1)
        input_cat = torch.cat(self._align_features(inputs), dim=1)
        output_cat = torch.cat(self._align_features(outputs), dim=1)

        B, _, H, W = residual_cat.size()
        if amap_mode == 'mul':
            anomaly_map = torch.ones([B, 1, out_size, out_size], device=residual_cat.device)
        else:
            anomaly_map = torch.zeros([B, 1, out_size, out_size], device=residual_cat.device)

        for idxs in self._select_indices(residual_cat):
            selected_inputs = torch.gather(input_cat, dim=1, index=idxs)
            selected_outputs = torch.gather(output_cat, dim=1, index=idxs)
            a_map = 1 - F.cosine_similarity(selected_inputs, selected_outputs)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)

            if amap_mode == 'mul':
                anomaly_map *= a_map
            elif amap_mode == 'max':
                anomaly_map = torch.max(anomaly_map, a_map)
            else:
                anomaly_map += a_map

        return anomaly_map

    def forward(self, inputs, outputs, image=None):
        """
        Forward pass of RRS module.

        Args:
            inputs:  list of [B, C_i, H_i, W_i] - original features from PFE
            outputs: list of [B, C_i, H_i, W_i] - reconstructed features from AE
            image:   [B, 3, H_img, W_img] - original image (for output size alignment)

        Returns:
            dict with:
                'logit': [B, 2, H, W] - 2-class segmentation logits
                'anomaly_score': [B, 1, H, W] - anomaly probability map
        """
        # 1. Compute per-channel MSE residuals
        residuals = self.compute_residuals(inputs, outputs)

        # 2. Stop gradient if needed (prevents RRS loss from affecting AE)
        if self.stop_grad:
            residuals = [r.detach() for r in residuals]

        # 3. Upsample all residuals to the smallest stride (highest resolution)
        aligned = self._align_features(residuals)

        # 4. Concatenate along channel dimension
        residual_cat = torch.cat(aligned, dim=1)  # [B, sum(C_i), H_align, W_align]

        # 5. BN for index selection (does not affect gradient flow to original residual)
        B, C, H, W = residual_cat.size()

        # 6. Channel selection (core of RRS)
        residual_choose = []
        for idxs in self._select_indices(residual_cat):
            residual_choose.append(
                torch.gather(residual_cat, dim=1, index=idxs))

        selected = torch.cat(residual_choose, dim=1)  # [B, total_select, H, W]

        # 7. Decode
        decoded = self.decoder1(selected)
        decoded = self.decoder2(decoded)

        # Upsample 2x
        upsample_size = (decoded.size(-1) * 2,) * 2
        decoded = F.interpolate(decoded, upsample_size, mode='bilinear', align_corners=True)
        logit = self.decoder3(decoded)

        # 8. Upsample to image size if provided
        if image is not None:
            _, _, ht, wt = image.size()
            logit = F.interpolate(logit, (ht, wt), mode='bilinear', align_corners=True)

        # 9. Compute anomaly score
        pred = torch.softmax(logit, dim=1)
        anomaly_score = pred[:, 1, :, :].unsqueeze(1)  # [B, 1, H, W]

        return {'logit': logit, 'anomaly_score': anomaly_score}
