import torch
import numpy as np
from torch.utils.data import Dataset


class AFSInitDataset(Dataset):
    """Wrap an existing dataset (returning image tensor as first element)
    to produce dict inputs for AFS index initialization:
      - image: synthetic anomalous image tensor [C,H,W]
      - gt_image: original clean image tensor [C,H,W]
      - mask: anomaly mask tensor [1,H,W] with values in {0,1}
    """
    def __init__(self, base_dataset, anomaly_ratio_range=(0.05, 0.20)):
        super().__init__()
        self.base = base_dataset
        self.anomaly_ratio_range = anomaly_ratio_range
        # mimic RealNet API expected by AFS.init_idxs
        self.anomaly_types = {"synthetic": 1.0}

    def __len__(self):
        return len(self.base)

    def _random_rect_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.float32)
        area = h * w
        target_ratio = np.random.uniform(*self.anomaly_ratio_range)
        rect_area = max(1, int(area * target_ratio))
        # pick random rectangle dimensions
        rect_h = max(1, int(np.sqrt(rect_area)))
        rect_w = max(1, int(rect_area / rect_h))
        rect_h = min(rect_h, h)
        rect_w = min(rect_w, w)
        y = np.random.randint(0, max(1, h - rect_h + 1))
        x = np.random.randint(0, max(1, w - rect_w + 1))
        mask[y:y + rect_h, x:x + rect_w] = 1.0
        return mask

    def _jitter(self, img_t):
        # img_t: [C,H,W], normalized tensor
        noise = torch.randn_like(img_t) * 0.25
        return torch.clamp(img_t + noise, -3.0, 3.0)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if isinstance(sample, (list, tuple)):
            img = sample[0]
        else:
            img = sample
        # ensure tensor
        assert isinstance(img, torch.Tensor), "Base dataset must return tensor image as first element"
        gt_image = img.clone()
        c, h, w = img.shape
        mask_np = self._random_rect_mask(h, w)
        mask = torch.from_numpy(mask_np).float().unsqueeze(0)
        anomaly_region = self._jitter(img)
        image = img * (1 - mask) + anomaly_region * mask
        return {"image": image, "gt_image": gt_image, "mask": mask} 