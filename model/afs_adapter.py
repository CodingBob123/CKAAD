import torch
from typing import Dict, List


class AFSBackboneAdapter(torch.nn.Module):
    """Adapter that wraps PretrainedFeatureExtractor to provide
    - .device attribute
    - .backbone(inputs, train=True) -> {'feats': {...}, 'gt_feats': {...}}
      where each is dict: {idx: {'feat': Tensor[B,C,H,W]}}
    """
    def __init__(self, pfe: torch.nn.Module, layer_numbers: List[int], device: torch.device):
        super().__init__()
        self.pfe = pfe
        self.layer_numbers = list(layer_numbers)
        self.device = device
        # map from layer idx (1..4) to position in PFE outputs list
        self.idx_to_pos: Dict[int, int] = {ln: i for i, ln in enumerate(self.layer_numbers)}

    @torch.no_grad()
    def backbone(self, inputs: Dict, train: bool = True):
        image = inputs["image"].to(self.device)
        feats_list = self.pfe(image)
        feats = {ln: {"feat": feats_list[self.idx_to_pos[ln]]} for ln in self.layer_numbers}
        output = {"feats": feats}
        if train and ("gt_image" in inputs):
            gt_image = inputs["gt_image"].to(self.device)
            gt_feats_list = self.pfe(gt_image)
            gt_feats = {ln: {"feat": gt_feats_list[self.idx_to_pos[ln]]} for ln in self.layer_numbers}
            output["gt_feats"] = gt_feats
        return output 