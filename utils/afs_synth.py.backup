import os
import random
import math
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    # t: [C,H,W] normalized by ImageNet
    c, h, w = t.shape
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    img = (t.detach().cpu().numpy() * std + mean)
    img = np.clip(img, 0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return img


def _to_tensor_image(img: np.ndarray) -> torch.Tensor:
    # img: HxWx3 uint8 (RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return img


def _morphology(mask: np.ndarray, close_k: int, open_k: int) -> np.ndarray:
    if close_k > 0:
        kernel = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_k > 0:
        kernel = np.ones((open_k, open_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _target_foreground_mask(img: np.ndarray, dataset: str, subclass: str) -> np.ndarray:
    # returns np.uint8 mask in {0,1}
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if dataset == 'mvtec':
        if subclass in ['carpet', 'leather', 'tile', 'wood', 'cable', 'transistor']:
            mask = np.ones_like(img_gray, dtype=np.uint8)
            return mask
        if subclass == 'pill':
            _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask = (th > 0).astype(np.uint8)
            return _morphology(mask, 6, 6)
        elif subclass in ['hazelnut', 'metal_nut', 'toothbrush']:
            _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
            mask = (th > 0).astype(np.uint8)
            return _morphology(mask, 6, 6)
        elif subclass in ['bottle','capsule','grid','screw','zipper']:
            _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            bg = (th > 0).astype(np.uint8)
            fg = (1 - bg).astype(np.uint8)
            return fg
        else:
            # fallback
            return np.ones_like(img_gray, dtype=np.uint8)
    elif dataset == 'visa':
        if subclass in ['capsules']:
            return np.ones_like(img_gray, dtype=np.uint8)
        if subclass in ['pcb1', 'pcb2', 'pcb3', 'pcb4']:
            _, th = cv2.threshold(img[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
            mask = (th > 0).astype(np.uint8)
            return _morphology(mask, 8, 3)
        else:
            _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            mask = (th > 0).astype(np.uint8)
            return _morphology(mask, 3, 3)
    elif dataset == 'btad':
        if subclass in ['02']:
            return np.ones_like(img_gray, dtype=np.uint8)
        _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = (th > 0).astype(np.uint8)
        return _morphology(mask, 15, 6)
    else:
        return np.ones_like(img_gray, dtype=np.uint8)


def _lerp_np(x, y, w):
    return (y - x) * w + x


def _rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    def tile_grads(s1, s2):
        return np.repeat(np.repeat(gradients[s1[0]:s1[1], s2[0]:s2[1]], d[0], axis=0), d[1], axis=1)
    def dot(grad, shift):
        return (np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * _lerp_np(_lerp_np(n00, n10, t[..., 0]), _lerp_np(n01, n11, t[..., 0]), t[..., 1])


def _rotate_affine(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _generate_perlin_mask(resize: Tuple[int, int], min_scale: int, max_scale: int, threshold: float) -> np.ndarray:
    scalex = 2 ** np.random.randint(min_scale, max_scale)
    scaley = 2 ** np.random.randint(min_scale, max_scale)
    noise = _rand_perlin_2d((resize[0], resize[1]), (scalex, scaley))
    angle = np.random.uniform(-90, 90)
    noise = _rotate_affine(noise, angle)
    mask = (noise > threshold).astype(np.uint8)
    return mask


def _load_file_list(directory: Optional[str]) -> List[str]:
    if directory is None:
        return []
    patterns = ["*/*", "*"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(directory, p)))
    return [f for f in files if os.path.isfile(f)]


class AFSSynthesizer:
    def __init__(self,
                 resize_hw: Tuple[int, int],
                 dataset_name: str,
                 subclass: str,
                 sdas_dir: Optional[str] = None,
                 dtd_dir: Optional[str] = None,
                 sdas_transparency_range: Tuple[float, float] = (0.4, 0.8),
                 dtd_transparency_range: Tuple[float, float] = (0.6, 0.9),
                 perlin_scale: int = 6,
                 min_perlin_scale: int = 0,
                 perlin_noise_threshold: float = 0.5,
                 ):
        self.resize = resize_hw
        self.dataset = dataset_name
        self.subclass = subclass
        self.sdas_files = _load_file_list(sdas_dir)
        self.dtd_files = _load_file_list(dtd_dir)
        self.sdas_range = sdas_transparency_range
        self.dtd_range = dtd_transparency_range
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_th = perlin_noise_threshold

    def _sample_sdas(self) -> np.ndarray:
        if not self.sdas_files:
            return None
        path = random.choice(self.sdas_files)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize[1], self.resize[0]))
        return img.astype(np.float32)

    def _sample_dtd(self) -> np.ndarray:
        if not self.dtd_files:
            return None
        path = random.choice(self.dtd_files)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize[1], self.resize[0]))
        return img.astype(np.float32)

    def _anomaly_source(self, img: np.ndarray, mask: np.ndarray, anomaly_type: str) -> np.ndarray:
        if anomaly_type == 'sdas' and self.sdas_files:
            src = self._sample_sdas()
            factor = float(np.random.uniform(*self.sdas_range))
        elif anomaly_type == 'dtd' and self.dtd_files:
            src = self._sample_dtd()
            factor = float(np.random.uniform(*self.dtd_range))
        else:
            # fallback: use image itself with heavy blur as texture source
            src = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            factor = 0.7
        mask_exp = np.expand_dims(mask.astype(np.float32), axis=2)
        blended = factor * (mask_exp * src) + (1 - factor) * (mask_exp * img)
        out = ((1.0 - mask_exp) * img) + blended
        return out

    def synthesize(self, img_t: torch.Tensor, anomaly_type: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # gt_image tensor (normalized)
        gt_image = img_t.detach().clone()
        img = _to_numpy_image(img_t)  # HxWx3 uint8
        img = img.astype(np.float32)

        # masks
        fg = _target_foreground_mask(img.astype(np.uint8), self.dataset, self.subclass)
        perlin = _generate_perlin_mask(self.resize, self.min_perlin_scale, self.perlin_scale, self.perlin_th)
        mask = (perlin * fg).astype(np.uint8)

        # choose source
        if anomaly_type is None:
            if self.sdas_files and self.dtd_files:
                anomaly_type = random.choice(['sdas', 'dtd'])
            elif self.sdas_files:
                anomaly_type = 'sdas'
            elif self.dtd_files:
                anomaly_type = 'dtd'
            else:
                anomaly_type = 'fallback'

        ano_img = self._anomaly_source(img, mask, anomaly_type).astype(np.uint8)

        image_t = _to_tensor_image(ano_img)
        mask_t = torch.from_numpy(mask.astype(np.float32) / 255.0 if mask.max() > 1 else mask.astype(np.float32)).unsqueeze(0)
        return image_t, gt_image, mask_t

    def batch_synthesize(self, batch_t: torch.Tensor, anomaly_type: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = batch_t.size(0)
        outs = [self.synthesize(batch_t[i], anomaly_type) for i in range(b)]
        images = torch.stack([o[0] for o in outs], dim=0)
        gts = torch.stack([o[1] for o in outs], dim=0)
        masks = torch.stack([o[2] for o in outs], dim=0)
        return images.to(batch_t.device), gts.to(batch_t.device), masks.to(batch_t.device) 