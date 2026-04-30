"""dataset"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms

class MVTecDataset(Dataset):
    def __init__(self, root, category, train=True, transform=None, gt_target_transform=None,
                 img_size=256, return_foreground_mask=False):
        super(MVTecDataset, self).__init__()
        self.category = category
        self.train = train
        if train:
            self.img_path = os.path.join(root, 'mvtec', category, 'train')
        else:
            self.img_path = os.path.join(root, 'mvtec', category, 'test')
            self.gt_path = os.path.join(root, 'mvtec', category, 'ground_truth')
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.img_size = img_size
        self.return_foreground_mask = return_foreground_mask

        # 检查 foreground_mask 目录是否存在（训练模式且需要返回时才检查）
        self.fg_mask_dir = None
        if train and return_foreground_mask:
            fg_dir = os.path.join(root, 'mvtec', category, 'train', 'foreground_mask')
            if os.path.isdir(fg_dir):
                self.fg_mask_dir = fg_dir

        self.process_data()  
        
    def get_initial_data(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_types
    
    def process_data(self):
        self.img_paths, self.gt_paths, tot_types = self.get_initial_data()
        data = []
        types = []
        type_set = list(set(tot_types))
        type_set.remove('good')
        type_set.sort()
        self.types_set = ['good'] + type_set
        for i, _ in enumerate(self.img_paths):
            data.append(i)
            types.append(self.types_set.index(tot_types[i]))
        self.data = np.stack(data) 
        self.targets = np.array(types).astype(np.int32)
    
    def load_data(self):
        data = []
        gt_targets = []
        for i in self.data:
            img_path = self.img_paths[i]
            gt_path = self.gt_paths[i]
            img = np.array(Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)
            if self.gt_paths[i] == 0:
                gt = np.zeros((img.shape[-2], img.shape[-2]), dtype=np.uint8)
            else:
                gt = np.array(Image.open(gt_path).resize((self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)
            data.append(img)
            gt_targets.append(gt)
        self.data = np.stack(data)
        self.gt_targets = np.stack(gt_targets) 
             
    def __len__(self):
        return len(self.data)

    def _load_foreground_mask(self, idx):
        """加载前景 mask。如果目录不存在或文件不存在，返回全 1 mask。"""
        if self.fg_mask_dir is None:
            return torch.ones((1, self.img_size, self.img_size), dtype=torch.float32)

        img_path = self.img_paths[idx]
        base_name = os.path.basename(img_path)
        fg_path = os.path.join(self.fg_mask_dir, base_name)

        if not os.path.exists(fg_path):
            return torch.ones((1, self.img_size, self.img_size), dtype=torch.float32)

        fg = Image.open(fg_path).convert('L')
        fg = fg.resize((self.img_size, self.img_size), Image.NEAREST)
        return transforms.ToTensor()(fg)  # [1, H, W]

    def __getitem__(self, idx):
        
        img, gt, label = self.data[idx], self.gt_targets[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)

        if self.return_foreground_mask:
            fg_mask = self._load_foreground_mask(idx)
            return img, gt, label, fg_mask
        return img, gt, label
    


class AnomalyDataset(Dataset):
    def __init__(self, normal_dataset):
        super(AnomalyDataset, self).__init__()
        self.normal_dataset = normal_dataset
        self.transform = self.normal_dataset.transform
        self.gt_target_transform = self.normal_dataset.gt_target_transform
        self.normal_dataset.transform = None
        self.normal_dataset.gt_target_transform = None
        self.category = self.normal_dataset.category
        self.img_size = self.normal_dataset.img_size
        
        self.anomaly_transform = transforms.ElasticTransform(alpha=200.0)
        # self.ano
        
    # def mask_img(self, img, mask_ratios = (0.05, 0.20)):
    #     h, w = img.shape[:2]
    #     mask_ratio = np.random.uniform(mask_ratios[0], mask_ratios[1])
    #     mask_h = int(h * mask_ratio)
    #     mask_w = int(w * mask_ratio)
    #     start_y = np.random.randint(0, h - mask_h)
    #     start_x = np.random.randint(0, w - mask_w)
    #     img[start_y:start_y+mask_h, start_x:start_x+mask_w] = int(255 / 2)
    #     return img
   
    def get_anomaly_data(self, idx):
        img, gt, label = self.normal_dataset[idx]
        # img, gt = self.normal_dataset.data[idx], self.normal_dataset.gt_targets[idx]
        anomaly_img = np.array(self.anomaly_transform(Image.fromarray(img.copy())))
        # anomaly_img = img.copy()
        # anomaly_img = self.mask_img(img)
        anomaly_gt = (np.zeros_like(gt) * 255).astype(np.int32)
        anomaly_label = 1
        return img, anomaly_img, anomaly_gt, anomaly_label
    
    
    def __len__(self):
        return len(self.normal_dataset)
    
    def __getitem__(self, idx):
        img, anomaly_img, anomaly_gt, anomaly_label = self.get_anomaly_data(idx)
        
        if self.transform is not None:
            img = self.transform(img)
            anomaly_img = self.transform(anomaly_img)
        if self.gt_target_transform is not None:
            anomaly_gt = self.gt_target_transform(anomaly_gt.copy())
        return img, anomaly_img, anomaly_gt, anomaly_label


class MVTecAnomalyWithMask(Dataset):
    """
    Load anomaly images from MVTec test set with pixel-level GT masks.
    Used for RRS training with SegmentCrossEntropyLoss.

    Returns:
        (anomaly_img, gt_mask, label) where:
            anomaly_img: transformed anomaly image tensor
            gt_mask: binary GT mask tensor (1=anomaly, 0=normal)
            label: 1 (anomaly)
    """

    def __init__(self, root, category, transform=None, gt_transform=None, img_size=256, max_samples=None):
        super(MVTecAnomalyWithMask, self).__init__()
        self.category = category
        self.transform = transform
        self.gt_transform = gt_transform
        self.img_size = img_size

        # Load anomaly image paths and corresponding GT mask paths from test set
        test_path = os.path.join(root, 'mvtec', category, 'test')
        gt_path = os.path.join(root, 'mvtec', category, 'ground_truth')

        self.img_paths = []
        self.gt_paths = []

        if os.path.exists(test_path):
            defect_types = os.listdir(test_path)
            for defect_type in sorted(defect_types):
                if defect_type == 'good':
                    continue  # skip normal test images

                img_dir = os.path.join(test_path, defect_type)
                gt_dir = os.path.join(gt_path, defect_type)

                if not os.path.isdir(img_dir):
                    continue

                img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
                gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png"))) if os.path.isdir(gt_dir) else []

                # Pair images with masks (MVTec GT masks have _mask suffix)
                for img_file in img_files:
                    # Try to find corresponding GT mask
                    base_name = os.path.splitext(os.path.basename(img_file))[0]
                    # GT mask naming: 001_mask.png or 001.png
                    gt_candidates = [
                        os.path.join(gt_dir, f"{base_name}_mask.png"),
                        os.path.join(gt_dir, f"{base_name}.png"),
                    ]
                    gt_file = None
                    for candidate in gt_candidates:
                        if os.path.exists(candidate):
                            gt_file = candidate
                            break

                    if gt_file is not None:
                        self.img_paths.append(img_file)
                        self.gt_paths.append(gt_file)

        # Limit number of samples if specified
        if max_samples is not None and max_samples < len(self.img_paths):
            indices = np.random.choice(len(self.img_paths), max_samples, replace=False)
            indices.sort()
            self.img_paths = [self.img_paths[i] for i in indices]
            self.gt_paths = [self.gt_paths[i] for i in indices]

        print(f"MVTecAnomalyWithMask: loaded {len(self.img_paths)} anomaly images with GT masks for category '{category}'")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert('RGB').resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)
        gt = np.array(Image.open(self.gt_paths[idx]).convert('L').resize(
            (self.img_size, self.img_size), resample=Image.BILINEAR), dtype=np.uint8)

        # Binarize GT mask: any non-zero pixel is anomaly
        gt = (gt > 0).astype(np.uint8)

        label = 1  # always anomaly

        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return img, gt, label
    

        