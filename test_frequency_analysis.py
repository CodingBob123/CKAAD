"""
频域分析测试脚本 - 针对 Tile 数据集

功能：
1. 读取图像并提取频域特征
2. 可视化频域分析结果
3. 评估频域方法对异常检测的有效性

使用方法：
    python test_frequency_analysis.py --data_dir /path/to/mvtec/tile --visualize
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
import warnings
warnings.filterwarnings('ignore')


def load_image(path, size=(256, 256)):
    """
    加载并预处理图像

    参数:
        path: 图像路径
        size: 目标大小

    返回:
        img: 预处理后的图像 [3, H, W]
    """
    img = Image.open(path).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
    return img


def compute_2d_fft(image):
    """
    计算二维傅里叶变换

    参数:
        image: 灰度图像 [H, W]

    返回:
        magnitude: 频域幅值 [H, W]
        phase: 相位 [H, W]
    """
    # 确保是 numpy 数组
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        image = image.mean(axis=0)

    # 中心化
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 幅值和相位
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)

    return magnitude, phase, fshift


def compute_frequency_features(image, patch_size=32):
    """
    计算图像的频域特征（分块版本）

    参数:
        image: 图像 [H, W] 或 [3, H, W]
        patch_size: 分块大小

    返回:
        features: 频域特征字典
    """
    # 转为灰度
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        image = image.mean(axis=0)

    H, W = image.shape
    features = {}

    # 全局 FFT
    magnitude, phase, fshift = compute_2d_fft(image)

    # 统计量
    features['mean_magnitude'] = magnitude.mean()
    features['std_magnitude'] = magnitude.std()
    features['max_magnitude'] = magnitude.max()
    features['min_magnitude'] = magnitude[magnitude > 0].min()

    # 低频能量比例
    crow, ccol = H // 2, W // 2
    radius = min(H, W) // 8
    mask = np.zeros_like(magnitude)
    y, x = np.ogrid[:H, :W]
    mask[(y - crow)**2 + (x - ccol)**2 <= radius**2] = 1
    low_freq_energy = (magnitude * mask).sum()
    total_energy = magnitude.sum()
    features['low_freq_ratio'] = low_freq_energy / (total_energy + 1e-8)

    # 高频能量比例
    high_freq_mask = 1 - mask
    features['high_freq_energy'] = (magnitude * high_freq_mask).sum()

    # 频谱质心（频率分布的"中心"）
    freq_y, freq_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    centroid_y = (freq_y * magnitude).sum() / (magnitude.sum() + 1e-8)
    centroid_x = (freq_x * magnitude).sum() / (magnitude.sum() + 1e-8)
    features['spectral_centroid'] = np.sqrt((centroid_y - H//2)**2 + (centroid_x - W//2)**2)

    # 分块频域特征
    if patch_size > 0:
        patch_features = []
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                patch = image[i:i+patch_size, j:j+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    patch_mag, _, _ = compute_2d_fft(patch)
                    patch_features.append(patch_mag.mean())

        features['patch_variance'] = np.var(patch_features)
        features['patch_mean'] = np.mean(patch_features)

    return features


def detect_texture_anomalies(image, threshold=2.0):
    """
    基于频域的纹理异常检测

    参数:
        image: 图像 [H, W] 或 [3, H, W]
        threshold: 异常阈值（标准差倍数）

    返回:
        anomaly_map: 异常热力图 [H, W]
    """
    # 转为灰度
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3:
        gray = image.mean(axis=0)
    else:
        gray = image

    H, W = gray.shape

    # 全局 FFT 统计
    magnitude, _, _ = compute_2d_fft(gray)
    global_std = magnitude.std()
    global_mean = magnitude.mean()

    # 分块处理
    patch_size = 32
    stride = 16
    anomaly_map = np.zeros((H, W))
    count_map = np.zeros((H, W))

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = gray[i:i+patch_size, j:j+patch_size]
            patch_mag, _, _ = compute_2d_fft(patch)

            # 局部统计
            local_mean = patch_mag.mean()
            local_std = patch_mag.std()

            # 与全局比较
            z_score = (local_mean - global_mean) / (global_std + 1e-8)

            # 异常分数
            if abs(z_score) > threshold:
                anomaly_score = abs(z_score) - threshold
            else:
                anomaly_score = 0

            # 累加到异常图
            anomaly_map[i:i+patch_size, j:j+patch_size] += anomaly_score
            count_map[i:i+patch_size, j:j+patch_size] += 1

    # 平均
    count_map[count_map == 0] = 1
    anomaly_map = anomaly_map / count_map

    return anomaly_map


def visualize_frequency_analysis(images, titles, save_path=None):
    """
    可视化频域分析结果

    参数:
        images: 图像字典 {name: image_array}
        titles: 标题字典 {name: title}
        save_path: 保存路径
    """
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(5*n, 15))

    for i, (name, img) in enumerate(images.items()):
        # 原图
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 3:
            img_show = img.transpose(1, 2, 0)
        else:
            img_show = img

        axes[0, i].imshow(np.clip(img_show, 0, 1))
        axes[0, i].set_title(f'{titles[name]}: 原图')
        axes[0, i].axis('off')

        # 转为灰度
        if img.ndim == 3:
            gray = img.mean(axis=0)
        else:
            gray = img

        # FFT 幅值（对数变换）
        magnitude, _, _ = compute_2d_fft(gray)
        magnitude_log = np.log1p(magnitude)
        magnitude_log = magnitude_log / magnitude_log.max()

        axes[1, i].imshow(magnitude_log, cmap='viridis')
        axes[1, i].set_title(f'{titles[name]}: 频域幅值')
        axes[1, i].axis('off')

        # 频域特征
        features = compute_frequency_features(gray)
        text = f"低频比例: {features['low_freq_ratio']:.3f}\n"
        text += f"谱质心: {features['spectral_centroid']:.1f}\n"
        if 'patch_variance' in features:
            text += f"分块方差: {features['patch_variance']:.3f}"

        axes[2, i].text(0.1, 0.5, text, fontsize=12, transform=axes[2, i].transAxes,
                       verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat'))
        axes[2, i].set_title(f'{titles[name]}: 频域统计')
        axes[2, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"频域分析结果已保存到: {save_path}")

    plt.close()


def visualize_anomaly_detection(normal_img, anomaly_img, anomaly_gt=None, save_path=None):
    """
    可视化异常检测结果

    参数:
        normal_img: 正常图像
        anomaly_img: 异常图像
        anomaly_gt: 异常真值掩码（可选）
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # 正常图像
    if isinstance(normal_img, torch.Tensor):
        normal_img = normal_img.cpu().numpy()
    if normal_img.ndim == 3:
        normal_show = normal_img.transpose(1, 2, 0)
    else:
        normal_show = normal_img
    normal_show = np.clip(normal_show, 0, 1)

    axes[0, 0].imshow(normal_show)
    axes[0, 0].set_title('正常图像', fontsize=14)
    axes[0, 0].axis('off')

    # 异常图像
    if isinstance(anomaly_img, torch.Tensor):
        anomaly_img = anomaly_img.cpu().numpy()
    if anomaly_img.ndim == 3:
        anomaly_show = anomaly_img.transpose(1, 2, 0)
    else:
        anomaly_show = anomaly_img
    anomaly_show = np.clip(anomaly_show, 0, 1)

    axes[0, 1].imshow(anomaly_show)
    axes[0, 1].set_title('异常图像', fontsize=14)
    axes[0, 1].axis('off')

    # 异常真值
    if anomaly_gt is not None:
        if isinstance(anomaly_gt, torch.Tensor):
            anomaly_gt = anomaly_gt.cpu().numpy()
        axes[0, 2].imshow(anomaly_gt, cmap='gray')
        axes[0, 2].set_title('异常真值', fontsize=14)
        axes[0, 2].axis('off')

    # 正常图像频域
    if normal_img.ndim == 3:
        normal_gray = normal_img.mean(axis=0)
    else:
        normal_gray = normal_img
    normal_mag, _, _ = compute_2d_fft(normal_gray)
    normal_mag_log = np.log1p(normal_mag)
    normal_mag_log = normal_mag_log / normal_mag_log.max()

    axes[0, 3].imshow(normal_mag_log, cmap='viridis')
    axes[0, 3].set_title('正常图像频域', fontsize=14)
    axes[0, 3].axis('off')

    # 异常图像频域
    if anomaly_img.ndim == 3:
        anomaly_gray = anomaly_img.mean(axis=0)
    else:
        anomaly_gray = anomaly_img
    anomaly_mag, _, _ = compute_2d_fft(anomaly_gray)
    anomaly_mag_log = np.log1p(anomaly_mag)
    anomaly_mag_log = anomaly_mag_log / anomaly_mag_log.max()

    axes[1, 0].imshow(anomaly_mag_log, cmap='viridis')
    axes[1, 0].set_title('异常图像频域', fontsize=14)
    axes[1, 0].axis('off')

    # 频域差异
    diff_mag = np.abs(anomaly_mag - normal_mag)
    diff_mag_log = np.log1p(diff_mag)
    if diff_mag_log.max() > 0:
        diff_mag_log = diff_mag_log / diff_mag_log.max()

    axes[1, 1].imshow(diff_mag_log, cmap='hot')
    axes[1, 1].set_title('频域差异', fontsize=14)
    axes[1, 1].axis('off')

    # 异常热力图（基于频域）
    heatmap = detect_texture_anomalies(anomaly_img, threshold=2.0)
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    axes[1, 2].imshow(heatmap_normalized, cmap='hot')
    axes[1, 2].set_title('频域异常热力图', fontsize=14)
    axes[1, 2].axis('off')

    # 在原图上叠加热力图
    if anomaly_img.ndim == 3:
        overlay = anomaly_img.transpose(1, 2, 0).copy()
    else:
        overlay = anomaly_img.copy()
    overlay = np.clip(overlay, 0, 1)

    # 创建 RGB 热力图
    heatmap_color = plt.cm.hot(heatmap_normalized)[:, :, :3]
    overlay = (overlay * 0.5 + heatmap_color * 0.5)

    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title('异常热力图叠加', fontsize=14)
    axes[1, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"异常检测结果已保存到: {save_path}")

    plt.close()

    return heatmap_normalized


def compare_normal_anomaly_frequency(data_dir, num_samples=5, save_dir=None):
    """
    比较正常和异常样本的频域特征

    参数:
        data_dir: MVTec 数据集目录
        num_samples: 比较的样本数量
        save_dir: 保存目录
    """
    # 查找正常和异常图像
    normal_dir = os.path.join(data_dir, 'train', 'good')
    anomaly_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(normal_dir):
        print(f"正常样本目录不存在: {normal_dir}")
        return

    # 获取正常样本
    normal_files = [f for f in os.listdir(normal_dir) if f.endswith('.png')]
    normal_files = normal_files[:num_samples]

    # 获取异常样本（遍历所有缺陷类型）
    anomaly_files = []
    for item in os.listdir(anomaly_dir):
        item_path = os.path.join(anomaly_dir, item)
        if os.path.isdir(item) and item != 'good':
            files = [os.path.join(item, f) for f in os.listdir(item_path) if f.endswith('.png')]
            anomaly_files.extend(files[:2])  # 每个类型取 2 个

    anomaly_files = anomaly_files[:num_samples]

    print(f"找到 {len(normal_files)} 个正常样本")
    print(f"找到 {len(anomaly_files)} 个异常样本")

    # 计算特征
    normal_features = []
    anomaly_features = []

    print("\n处理正常样本...")
    for f in normal_files:
        path = os.path.join(normal_dir, f)
        img = load_image(path)
        features = compute_frequency_features(img.numpy())
        normal_features.append(features)

    print("处理异常样本...")
    for f in anomaly_files:
        path = os.path.join(anomaly_dir, f)
        img = load_image(path)
        features = compute_frequency_features(img.numpy())
        anomaly_features.append(features)

    # 汇总统计
    print("\n" + "="*60)
    print("频域特征统计对比")
    print("="*60)

    keys = ['low_freq_ratio', 'spectral_centroid', 'patch_variance', 'mean_magnitude']
    for key in keys:
        normal_vals = [f.get(key, 0) for f in normal_features]
        anomaly_vals = [f.get(key, 0) for f in anomaly_features]

        print(f"\n{key}:")
        print(f"  正常样本: 均值={np.mean(normal_vals):.4f}, 标准差={np.std(normal_vals):.4f}")
        print(f"  异常样本: 均值={np.mean(anomaly_vals):.4f}, 标准差={np.std(anomaly_vals):.4f}")

    # 可视化对比
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 选择代表性样本
        if normal_files and anomaly_files:
            # 正常样本
            normal_img = load_image(os.path.join(normal_dir, normal_files[0]))
            # 异常样本
            anomaly_img = load_image(os.path.join(anomaly_dir, anomaly_files[0]))

            # 频域分析可视化
            visualize_frequency_analysis(
                {'normal': normal_img.numpy(), 'anomaly': anomaly_img.numpy()},
                {'normal': '正常样本', 'anomaly': '异常样本'},
                save_path=os.path.join(save_dir, 'frequency_analysis.png')
            )

            # 异常检测可视化
            visualize_anomaly_detection(
                normal_img.numpy(),
                anomaly_img.numpy(),
                save_path=os.path.join(save_dir, 'anomaly_detection.png')
            )

            print(f"\n可视化结果已保存到: {save_dir}")


def test_single_image(image_path, save_dir=None):
    """
    测试单张图像的频域分析

    参数:
        image_path: 图像路径
        save_dir: 保存目录
    """
    img = load_image(image_path)

    print("="*60)
    print("单张图像频域分析")
    print("="*60)

    # 计算频域特征
    if img.ndim == 3:
        gray = img.numpy().mean(axis=0)
    else:
        gray = img.numpy()

    features = compute_frequency_features(gray)
    print("\n频域特征:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}")

    # 检测异常
    heatmap = detect_texture_anomalies(gray, threshold=2.0)

    # 可视化
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原图
        img_show = img.numpy().transpose(1, 2, 0)
        img_show = np.clip(img_show, 0, 1)
        axes[0].imshow(img_show)
        axes[0].set_title('输入图像')
        axes[0].axis('off')

        # 频域
        magnitude, _, _ = compute_2d_fft(gray)
        magnitude_log = np.log1p(magnitude)
        magnitude_log = magnitude_log / magnitude_log.max()
        axes[1].imshow(magnitude_log, cmap='viridis')
        axes[1].set_title('频域幅值')
        axes[1].axis('off')

        # 异常热力图
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        axes[2].imshow(heatmap_normalized, cmap='hot')
        axes[2].set_title('频域异常热力图')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'single_image_analysis.png'), dpi=150)
        plt.close()

        print(f"\n分析结果已保存到: {os.path.join(save_dir, 'single_image_analysis.png')}")


def main():
    parser = argparse.ArgumentParser(description='频域分析测试脚本')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='MVTec 数据集目录')
    parser.add_argument('--image_path', type=str, default=None,
                        help='单张图像路径')
    parser.add_argument('--save_dir', type=str, default='./frequency_analysis_results',
                        help='结果保存目录')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='比较的样本数量')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化结果')

    args = parser.parse_args()

    if args.visualize or args.data_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.data_dir:
        # 比较正常和异常样本
        compare_normal_anomaly_frequency(
            args.data_dir,
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )

    if args.image_path:
        # 测试单张图像
        test_single_image(args.image_path, save_dir=args.save_dir)

    if not args.data_dir and not args.image_path:
        parser.print_help()
        print("\n请提供 --data_dir 或 --image_path 参数")
        print("\n示例用法:")
        print("  python test_frequency_analysis.py --data_dir /path/to/mvtec/tile --visualize")
        print("  python test_frequency_analysis.py --image_path /path/to/image.png --visualize")


if __name__ == '__main__':
    main()

