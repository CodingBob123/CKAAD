import torch
from typing import Dict, List


class AFSGate(torch.nn.Module):
    """Channel gating using AFS selected indices.
    Keeps tensor shapes unchanged by zero-masking unselected channels.
    """
    def __init__(self, layer_numbers: List[int], inplanes: Dict[int, int], selected_indices: Dict[int, torch.Tensor]):
        super().__init__()
        self.layer_numbers = list(layer_numbers)
        self.inplanes = inplanes
        # register buffers for masks per layer
        masks = {}
        for ln in self.layer_numbers:
            c = inplanes[ln]
            mask = torch.zeros(c, dtype=torch.float32)
            idx = selected_indices[ln].long().clamp(min=0, max=c-1)
            mask[idx] = 1.0
            masks[str(ln)] = mask
        # store as buffers to move with .to(device)
        for k, v in masks.items():
            self.register_buffer(f"mask_{k}", v)

    def _get_mask(self, ln: int, device: torch.device, dtype: torch.dtype, h: int, w: int):
        base = getattr(self, f"mask_{ln}")  # [C]
        m = base.to(device=device, dtype=dtype).view(1, -1, 1, 1)
        return m

    @torch.no_grad()
    def apply_list(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply gate to list of feature maps ordered by self.layer_numbers.
        feats[i]: [B,C,H,W] corresponds to layer_numbers[i].
        """
        out = []
        for i, ln in enumerate(self.layer_numbers):
            x = feats[i]
            b, c, h, w = x.shape
            m = self._get_mask(ln, x.device, x.dtype, h, w)
            out.append(x * m)
        return out 