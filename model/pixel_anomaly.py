"""
统一的像素级异常生成模块 — 融合 PatchGuard + OCR-GAN CutPaste/Cutout。

提供三种模式并在 'mixed' 模式下按概率随机选择：
  - 'patchguard'：前景感知 Cut-and-Paste（7 种增强 + 旋转 + 精准 mask）
  - 'cutpaste'  ：OCR-GAN CutPaste（ColorJitter + 自粘贴 + mask 反推）
  - 'cutout'    ：OCR-GAN Cutout（随机擦除 + mask 反推）

所有模式统一输出 (anomaly_imgs, anomaly_masks)，均为 [B, C, H, W] torch Tensor。
"""

import io
import math
import random
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import noise
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
from torchvision import transforms


# fmt: off
PATCHGUARD_BOUNDS = {
    "mvtec": {
        "toothbrush": (14, 39), "cable": (14, 65), "screw": (9, 35),
        "transistor": (14, 86), "capsule": (14, 39), "bottle": (14, 65),
        "hazelnut": (14, 65), "metal_nut": (14, 65), "pill": (14, 39),
        "zipper": (14, 65),
        "wood": (14, 86), "carpet": (14, 86), "grid": (14, 86),
        "leather": (14, 86), "tile": (14, 86),
    },
    "visa": {
        "candle": (14, 86), "capsules": (6, 86), "cashew": (10, 86),
        "chewinggum": (14, 86), "fryum": (6, 15), "macaroni1": (14, 86),
        "macaroni2": (14, 86), "pcb1": (14, 86), "pcb2": (14, 86),
        "pcb3": (14, 86), "pcb4": (14, 86), "pipe_fryum": (14, 86),
    },
}
# fmt: on

# =====================================================================
# 辅助：是否属于有前景 mask 的类别
# =====================================================================
MVTEC_FG_CLASSES = {
    "bottle", "cable", "capsule", "hazelnut",
    "metal_nut", "pill", "screw", "toothbrush", "zipper",
}


def has_foreground_mask(dataset_name: str, class_name: str) -> bool:
    """检查给定类别是否有预计算的前景 mask。"""
    if dataset_name == "mvtec":
        return class_name in MVTEC_FG_CLASSES
    return False


# =====================================================================
# 随机增强（从 PatchGuard 移植）
# =====================================================================
class RandomAugmentations:
    """PatchGuard 的 7 种增强 + ColorJitter，分 light/medium/heavy 三个等级。"""

    def __init__(self):
        self.param_ranges = {
            "brightness": {"light": (0.1, 0.1), "medium": (0.4, 0.4), "heavy": (0.8, 1.0)},
            "contrast": {"light": (0.1, 0.1), "medium": (0.4, 0.4), "heavy": (0.8, 1.0)},
            "saturation": {"light": (0.1, 0.1), "medium": (0.4, 0.4), "heavy": (0.8, 1.0)},
            "hue": {"light": (0.1, 0.1), "medium": (0.3, 0.3), "heavy": (0.5, 0.5)},
            "elastic_alpha": {"light": (10, 20), "medium": (20, 40), "heavy": (40, 100)},
            "torn_lines": {"light": (1, 3), "medium": (5, 10), "heavy": (10, 20)},
            "perlin_scale": {"light": (20, 50), "medium": (10, 20), "heavy": (5, 10)},
            "perlin_threshold": {"light": (200, 255), "medium": (150, 200), "heavy": (128, 150)},
            "swirl_strength": {"light": (0.5, 1.0), "medium": (1.0, 1.5), "heavy": (1.5, 2.0)},
            "erase_ratio": {"light": (0.01, 0.05), "medium": (0.05, 0.1), "heavy": (0.1, 0.2)},
            "erase_rects": {"light": (1, 2), "medium": (2, 3), "heavy": (3, 5)},
            "blur_radius": {"light": (0.2, 0.5), "medium": (0.8, 1.5), "heavy": (2.0, 3.0)},
            "jpeg_quality": {"light": (30, 50), "medium": (20, 30), "heavy": (1, 20)},
        }
        self.num_augmentations = {"light": 1, "medium": 2, "heavy": 4}
        self.augmentations = {
            "light": [self.gaussian_blur, self.elastic_transform, self.swirl_distortion],
            "medium": [
                self.gaussian_blur, self.swirl_distortion,
                self.jpeg_artifacts, self.elastic_transform,
            ],
            "heavy": [
                self.elastic_transform, self.torn_paper_effect, self.jpeg_artifacts,
                self.perlin_noise_mask, self.swirl_distortion,
                self.random_erasing, self.gaussian_blur,
            ],
        }

    def apply(self, image: Image.Image, level: str = "medium") -> Image.Image:
        image_np = np.array(image)
        n_aug = random.randint(0, self.num_augmentations[level])
        selected = random.sample(self.augmentations[level], n_aug)
        selected.insert(random.randint(0, len(selected)), self.color_transformation)
        for aug in selected:
            image_np = aug(image_np, level)
        return Image.fromarray(image_np)

    # ---- 各增强方法 ----
    def elastic_transform(self, image: np.ndarray, level: str) -> np.ndarray:
        alpha = random.uniform(*self.param_ranges["elastic_alpha"][level])
        sigma = 3.0
        rs = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((rs.rand(*shape[:2]) * 2 - 1), sigma, mode="reflect") * alpha
        dy = gaussian_filter((rs.rand(*shape[:2]) * 2 - 1), sigma, mode="reflect") * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).flatten(), (x + dx).flatten()
        out = np.zeros_like(image)
        for c in range(shape[2]):
            out[..., c] = map_coordinates(image[..., c], indices, order=1, mode="reflect").reshape(shape[:2])
        return out

    def torn_paper_effect(self, image: np.ndarray, level: str) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        n = random.randint(*self.param_ranges["torn_lines"][level])
        for _ in range(n):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
            cv2.line(img, (x1, y1), (x2, y2), [random.choice([0, 255]) for _ in range(3)], thickness=1)
        return img

    def perlin_noise_mask(self, image: np.ndarray, level: str) -> np.ndarray:
        scale = random.uniform(*self.param_ranges["perlin_scale"][level])
        thresh = random.randint(*self.param_ranges["perlin_threshold"][level])
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                mask[i, j] = noise.pnoise2(i / scale, j / scale, octaves=6)
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255
        img = image.copy()
        img[mask > thresh] = np.random.randint(0, 255, 3)
        return img

    def color_transformation(self, image: np.ndarray, level: str) -> np.ndarray:
        b = random.uniform(*self.param_ranges["brightness"][level])
        c = random.uniform(*self.param_ranges["contrast"][level])
        s = random.uniform(*self.param_ranges["saturation"][level])
        h = random.uniform(*self.param_ranges["hue"][level])
        t = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
        return np.array(t(Image.fromarray(image)))

    def swirl_distortion(self, image: np.ndarray, level: str) -> np.ndarray:
        strength = random.uniform(*self.param_ranges["swirl_strength"][level])
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        y, x = np.indices((h, w))
        x, y = x - cx, y - cy
        dist = np.sqrt(x**2 + y**2)
        angle = strength * np.exp(-dist**2 / (2 * (min(h, w) // 3) ** 2))
        nx = cx + x * np.cos(angle) - y * np.sin(angle)
        ny = cy + x * np.sin(angle) + y * np.cos(angle)
        mx = np.clip(nx, 0, w - 1).astype(np.float32)
        my = np.clip(ny, 0, h - 1).astype(np.float32)
        return cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR)

    def random_erasing(self, image: np.ndarray, level: str) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        ratio = random.uniform(*self.param_ranges["erase_ratio"][level])
        n_rects = random.randint(*self.param_ranges["erase_rects"][level])
        for _ in range(n_rects):
            area = int(ratio * h * w)
            aspect = random.uniform(0.3, 3.3)
            eh = int(np.sqrt(area / aspect))
            ew = int(aspect * eh)
            x = random.randint(0, max(0, w - ew))
            y = random.randint(0, max(0, h - eh))
            img[y : y + eh, x : x + ew] = np.random.randint(0, 255, 3)
        return img

    def gaussian_blur(self, image: np.ndarray, level: str) -> np.ndarray:
        r = random.uniform(*self.param_ranges["blur_radius"][level])
        return np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=r)))

    def jpeg_artifacts(self, image: np.ndarray, level: str) -> np.ndarray:
        q = random.randint(*self.param_ranges["jpeg_quality"][level])
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="JPEG", quality=q)
        return np.array(Image.open(buf))


# =====================================================================
# OCR-GAN CutPaste（加 mask 输出）
# =====================================================================

def _cutpaste_mask(img: torch.Tensor, from_box, to_box) -> torch.Tensor:
    """从粘贴位置反推 mask。"""
    B, C, H, W = img.shape
    mask = torch.zeros(B, 1, H, W, device=img.device, dtype=img.dtype)
    for b in range(B):
        x1, y1, x2, y2 = to_box[b]
        mask[b, 0, y1:y2, x1:x2] = 1.0
    return mask


@torch.no_grad()
def cutpaste_batch(
    imgs: torch.Tensor,
    area_ratio: Tuple[float, float] = (0.02, 0.15),
    aspect_ratio: float = 0.3,
    jitter_strength: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 batch 中每张图像执行 CutPaste。

    Args:
        imgs: [B, 3, H, W] torch tensor (像素值 0~1，CHW 格式)
        area_ratio: patch 面积占比范围
        aspect_ratio: patch 宽高比最小值
        jitter_strength: ColorJitter 强度

    Returns:
        augmented_imgs: [B, 3, H, W] 增强后的图像
        masks: [B, 1, H, W] 异常区域 mask
    """
    B, C, H, W = imgs.shape
    device = imgs.device

    # 转换为 PIL 处理（ColorJitter 和 paste 操作）
    imgs_pil = [transforms.ToPILImage()(imgs[b].cpu().clamp(0, 1)) for b in range(B)]

    from_boxes, to_boxes = [], []
    aug_pil_list = []
    patch_pil_list = []

    for b in range(B):
        pil = imgs_pil[b]
        pw = random.uniform(*area_ratio) * W * H
        aspect = random.uniform(aspect_ratio, 1.0 / aspect_ratio)
        cut_w = int(round(math.sqrt(pw * aspect)))
        cut_h = int(round(math.sqrt(pw / aspect)))
        cut_w, cut_h = min(cut_w, W), min(cut_h, H)
        if cut_w < 2 or cut_h < 2:
            # 失败回退：使用最小 patch
            cut_w, cut_h = max(2, int(W * 0.02)), max(2, int(H * 0.02))

        fx = random.randint(0, W - cut_w)
        fy = random.randint(0, H - cut_h)
        tx = random.randint(0, W - cut_w)
        ty = random.randint(0, H - cut_h)

        patch = pil.crop((fx, fy, fx + cut_w, fy + cut_h))
        if jitter_strength > 0:
            cj = transforms.ColorJitter(
                brightness=jitter_strength,
                contrast=jitter_strength,
                saturation=jitter_strength,
                hue=jitter_strength,
            )
            patch = cj(patch)

        aug = pil.copy()
        aug.paste(patch, (tx, ty))

        aug_pil_list.append(aug)
        patch_pil_list.append(patch)
        from_boxes.append((fx, fy, fx + cut_w, fy + cut_h))
        to_boxes.append((tx, ty, tx + cut_w, ty + cut_h))

    # 转回 tensor
    aug_tensor_list = [transforms.ToTensor()(p) for p in aug_pil_list]
    augmented = torch.stack(aug_tensor_list, dim=0).to(device)

    # 从粘贴位置生成 mask
    masks = torch.zeros(B, 1, H, W, device=device, dtype=augmented.dtype)
    for b in range(B):
        x1, y1, x2, y2 = to_boxes[b]
        masks[b, 0, y1:y2, x1:x2] = 1.0

    return augmented, masks


# =====================================================================
# OCR-GAN Cutout（加 mask 输出）
# =====================================================================

@torch.no_grad()
def cutout_batch(
    imgs: torch.Tensor,
    n_holes: int = 1,
    hole_size: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 batch 中每张图像执行 Cutout（随机擦除）。

    Args:
        imgs: [B, 3, H, W] torch tensor
        n_holes: 每张图擦除的方块数
        hole_size: 每个方块的边长（像素）

    Returns:
        erased_imgs: [B, 3, H, W] 擦除后的图像
        masks: [B, 1, H, W] 擦除位置 mask
    """
    B, C, H, W = imgs.shape
    device = imgs.device
    out = imgs.clone()
    masks = torch.zeros(B, 1, H, W, device=device, dtype=imgs.dtype)

    for b in range(B):
        for _ in range(n_holes):
            cy = random.randint(0, H - 1)
            cx = random.randint(0, W - 1)
            y1 = max(0, cy - hole_size // 2)
            y2 = min(H, cy + hole_size // 2)
            x1 = max(0, cx - hole_size // 2)
            x2 = min(W, cx + hole_size // 2)
            out[b, :, y1:y2, x1:x2] = 0.0
            masks[b, 0, y1:y2, x1:x2] = 1.0

    return out, masks


# =====================================================================
# PixelAnomalyGenerator — 统一像素级异常生成器
# =====================================================================

class PixelAnomalyGenerator:
    """统一的像素级异常生成器，融合 PatchGuard + OCR-GAN 方法。

    支持模式：
      - 'patchguard' : 前景感知 cut-and-paste（需要 foreground_mask）
      - 'cutpaste'   : OCR-GAN CutPaste（仅 ColorJitter，无需 foreground_mask）
      - 'cutout'     : OCR-GAN Cutout（随机擦除，无需 foreground_mask）
      - 'mixed'      : 按概率在三种子模式中随机选择

    统一输出：
        (anomaly_imgs, anomaly_masks) 均为 [B, 3, H, W] / [B, 1, H, W] torch tensor
    """

    def __init__(
        self,
        dataset: str = "mvtec",
        class_name: str = "carpet",
        mode: str = "mixed",
        patchguard_prob: float = 0.5,
        cutpaste_prob: float = 0.3,
        cutout_prob: float = 0.2,
        max_attempts: int = 50,
    ):
        """
        Args:
            dataset: 数据集名称（用于获取 bounds）
            class_name: 类别名称（用于获取 bounds）
            mode: 'patchguard' / 'cutpaste' / 'cutout' / 'mixed'
            patchguard_prob: mixed 模式下 patchguard 子模式概率
            cutpaste_prob: mixed 模式下 cutpaste 子模式概率
            cutout_prob: mixed 模式下 cutout 子模式概率
            max_attempts: PatchGuard 采样坐标的最大尝试次数（默认 50，原版 250）
        """
        self.mode = mode
        self.patchguard_prob = patchguard_prob
        self.cutpaste_prob = cutpaste_prob
        self.cutout_prob = cutout_prob
        self.max_attempts = max_attempts

        # 从 PatchGuard bounds 获取 patch 尺寸范围
        ds_bounds = PATCHGUARD_BOUNDS.get(dataset, {})
        self.lower_bound, self.upper_bound = ds_bounds.get(class_name, (14, 86))

        self.augmentor = RandomAugmentations()

        # 子模式概率归一化
        total = patchguard_prob + cutpaste_prob + cutout_prob
        if total > 0:
            self.pg_prob = patchguard_prob / total
            self.cp_prob = cutpaste_prob / total
            self.co_prob = cutout_prob / total
        else:
            self.pg_prob = 1.0 / 3
            self.cp_prob = 1.0 / 3
            self.co_prob = 1.0 / 3

    def _choose_mode(self) -> str:
        """根据配置选择子模式。"""
        if self.mode != "mixed":
            return self.mode
        r = random.random()
        if r < self.pg_prob:
            return "patchguard"
        elif r < self.pg_prob + self.cp_prob:
            return "cutpaste"
        else:
            return "cutout"

    @torch.no_grad()
    def __call__(
        self,
        imgs: torch.Tensor,
        foreground_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对 batch 中每张图像生成像素级异常。

        Args:
            imgs: [B, 3, H, W] 正常图像（归一化后，0~1 范围）
            foreground_masks: [B, 1, H, W] 前景 mask（1=前景，0=背景）
                              如果为 None，全部视为前景（全 1 mask）

        Returns:
            anomaly_imgs: [B, 3, H, W] 异常图像
            anomaly_masks: [B, 1, H, W] 异常区域二值 mask（1=异常）
        """
        B, C, H, W = imgs.shape
        device = imgs.device

        # 默认全 1 mask
        if foreground_masks is None:
            fg_masks = torch.ones(B, 1, H, W, device=device, dtype=imgs.dtype)
        else:
            fg_masks = foreground_masks.to(device)

        # 根据模式选择处理方法
        if self.mode == "mixed":
            # 每张图独立选择子模式
            return self._mixed_call(imgs, fg_masks)
        elif self.mode == "patchguard":
            return self._patchguard_call(imgs, fg_masks)
        elif self.mode == "cutpaste":
            return cutpaste_batch(imgs)
        elif self.mode == "cutout":
            return cutout_batch(imgs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _mixed_call(self, imgs, fg_masks):
        """mixed 模式：batch 内每张图独立选择子模式。"""
        B, C, H, W = imgs.shape
        device = imgs.device
        all_anomaly = []
        all_masks = []

        for b in range(B):
            single_img = imgs[b : b + 1]  # [1, C, H, W]
            single_fg = fg_masks[b : b + 1]
            mode = self._choose_mode()

            if mode == "patchguard":
                a_img, a_mask = self._patchguard_call(single_img, single_fg)
            elif mode == "cutpaste":
                a_img, a_mask = cutpaste_batch(single_img)
            else:
                a_img, a_mask = cutout_batch(single_img)

            all_anomaly.append(a_img)
            all_masks.append(a_mask)

        return torch.cat(all_anomaly, dim=0), torch.cat(all_masks, dim=0)

    # ============ PatchGuard 核心方法 ============

    def _patchguard_call(self, imgs, fg_masks):
        """PatchGuard 模式核心实现。"""
        B, C, H, W = imgs.shape
        device = imgs.device
        out_imgs = []
        out_masks = []

        for b in range(B):
            img_t = imgs[b].cpu()
            img_pil = transforms.ToPILImage()(img_t.clamp(0, 1))
            fg_np = fg_masks[b, 0].cpu().numpy().astype(np.uint8)

            # 采样坐标和 patch 尺寸
            x1, y1, x2, y2, pw, ph = self._sample_coordinate_shape(fg_np)
            pw, ph = int(pw), int(ph)

            # 裁 patch
            patch = img_pil.crop((x1, y1, x1 + pw, y1 + ph))

            # 随机增强
            level = np.random.choice(["light", "medium", "heavy"], p=[0.2, 0.2, 0.6])
            patch = self.augmentor.apply(patch, level)

            # 旋转
            patch, rot_mask = self._rotate(patch, pw, ph)

            # 计算最终 mask
            mask_np = np.ones((ph, pw), dtype=np.uint8)
            mask_np = cv2.resize(mask_np, (pw, ph), interpolation=cv2.INTER_CUBIC)
            mask_np = self._intersect_masks(mask_np, rot_mask)

            # 粘贴
            aug = img_pil.copy()
            aug.paste(patch, (x2, y2), mask=Image.fromarray(mask_np))

            # 生成全图 mask
            full_mask = Image.fromarray(np.zeros((H, W), dtype=np.uint8)).convert("L")
            full_mask.paste(Image.fromarray(mask_np), (x2, y2))

            out_imgs.append(transforms.ToTensor()(aug))
            out_masks.append(transforms.ToTensor()(full_mask))

        return torch.stack(out_imgs).to(device), torch.stack(out_masks).to(device)

    def _rotate(self, patch, width, height, min_angle=-90, max_angle=90):
        angle = random.uniform(min_angle, max_angle)
        patch = patch.convert("RGBA").rotate(angle, expand=True)
        patch = patch.resize((width, height), resample=Image.BICUBIC)
        mask = patch.split()[-1]
        return patch.convert("RGB"), np.array(mask)

    @staticmethod
    def _intersect_masks(mask1_np, mask2_np):
        return np.logical_and(mask1_np, mask2_np).astype(np.uint8) * 255

    def _expand_mask(self, mask, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=10)

    def _get_max_shape(self, x, y, fg_mask):
        h, w = fg_mask.shape
        mw = 0
        for i in range(x, w):
            if fg_mask[y, i] == 1:
                mw += 1
            else:
                break
        mh = 0
        for j in range(y, h):
            if fg_mask[j, x] == 1:
                mh += 1
            else:
                break
        return mw, mh

    def _sample_patch_size(self, fg_mask, x1, y1, x2, y2, max_w, max_h):
        for _ in range(10):
            pw = random.randint(0, max_w)
            ph = random.randint(0, max_h)
            src = fg_mask[y1 : y1 + ph, x1 : x1 + pw]
            dst = fg_mask[y2 : y2 + ph, x2 : x2 + pw]
            if src.size > 0 and dst.size > 0 and np.all(src == 1) and np.all(dst == 1):
                break
        else:
            pw = ph = 0
        return pw, ph

    def _sample_coordinate_shape(self, fg_mask):
        """在前景 mask 上采样两个坐标和 patch 尺寸。"""
        fg_mask = self._expand_mask(fg_mask)
        coords = np.column_stack(np.where(fg_mask == 1))
        if len(coords) == 0:
            return 0, 0, 0, 0, self.lower_bound, self.lower_bound

        for _ in range(self.max_attempts):
            idx1 = random.randint(0, len(coords) - 1)
            idx2 = random.randint(0, len(coords) - 1)
            y1, x1 = coords[idx1]
            y2, x2 = coords[idx2]
            mw1, mh1 = self._get_max_shape(x1, y1, fg_mask)
            mw2, mh2 = self._get_max_shape(x2, y2, fg_mask)
            max_w, max_h = min(mw1, mw2), min(mh1, mh2)
            pw, ph = self._sample_patch_size(fg_mask, x1, y1, x2, y2, max_w, max_h)
            if pw >= self.lower_bound and ph >= self.lower_bound:
                return x1, y1, x2, y2, pw, ph

        return 0, 0, 0, 0, self.lower_bound, self.lower_bound
