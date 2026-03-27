"""
能量图引导与筛选 - 多版本评估工具

提供 5 个版本的评估函数：
  V1. Recon-only       : 纯重建误差图（原始基线）
  V2. Input-energy      : D(input) 能量图软门控融合（当前版本）
  V3. E1-diff          : D(input) - D(output) 第一层能量差软门控
  V4. E2-diff          : D(input) - D(output) 第二层能量差软门控
  V5. E3-diff          : D(input) - D(output) 第三层能量差软门控
  V6. Mean-E-diff      : 三层能量差分别计算后平均融合

每个版本均返回 (metrics, anomaly_maps)，用于打印汇总表。
"""

import torch
import numpy as np
from util.test import (
    cal_anomaly_map,
    cal_energy_map,
    soft_gate_fuse,
    sigmoid_np,
    minmax_norm_per_sample,
    quantile_clip,
    high_tail_compress,
)
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter


# ============================================================================
# 能量差计算（核心工具）
# ============================================================================

def cal_energy_diff_maps(discriminator, inputs, outputs, out_size):
    """
    计算多层能量差图：D(input) - D(output)

    判别器对正常特征输出低能量，对异常/重建特征输出高能量。
    因此：
        - 正常区域：input 和 output 都偏正常，差值小
        - 异常区域：input 偏异常，output 被拉回正常，差值大

    参数:
        discriminator: 训练好的判别器
        inputs:  list of torch.Tensor，输入特征列表 [x1, x2, x3]
        outputs: list of torch.Tensor，重建特征列表 [x1', x2', x3']
        out_size: int，输出图的尺寸

    返回:
        diff_maps: list of torch.Tensor [[N,1,H,W], ...]
                   三层能量差图，已上采样到 out_size
    """
    # 计算 input 和 output 的能量图
    energy_in = cal_energy_map(discriminator, inputs, out_size)   # [N,1,H,W] x 3
    energy_out = cal_energy_map(discriminator, outputs, out_size)  # [N,1,H,W] x 3

    # 每层做差
    diff_maps = []
    for i in range(len(energy_in)):
        diff_maps.append(energy_in[i] - energy_out[i])

    return diff_maps


# ============================================================================
# V1: Recon-only（纯重建基线）
# ============================================================================

def eval_recon_only(encoder, ed, discriminator, dataloader, device, args):
    """
    V1. Recon-only：不做任何融合，直接用重建误差图作为异常评分。
    返回 metrics 和 anomaly_maps（与原论文测试流程一致）。
    """
    encoder.eval()
    ed.eval()
    if discriminator is not None:
        discriminator.eval()

    pixel_gt_list, pixel_score_list = [], []
    sample_gt_list, sample_score_list = [], []
    all_maps = []

    with torch.no_grad():
        for img, gt, label in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = ed(inputs)
            gt = gt.squeeze(1).cpu().numpy()
            anomaly_map = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='add')
            all_maps.append(anomaly_map)
            gt_binary = (gt > 0.5).astype(int).reshape(-1)
            pixel_gt_list.append(gt_binary)
            pixel_score_list.append(anomaly_map.reshape(-1))
            sample_gt_list.append(np.max(gt.astype(int), axis=-1))
            sample_score_list.append(
                torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)),
                           args.topk, dim=-1)[0].mean(axis=-1).numpy())

    gt_arr = np.concatenate(pixel_gt_list)
    score_arr = np.concatenate(pixel_score_list)
    from util.test import calculate_metrics
    return {
        'Pixel': calculate_metrics(score_arr, gt_arr, acc=False),
        'Image': calculate_metrics(np.concatenate(sample_score_list),
                                  np.concatenate(sample_gt_list), acc=True),
    }, np.concatenate(all_maps)


# ============================================================================
# V2: Input-energy（当前软门控版本）
# ============================================================================

def eval_input_energy_fusion(encoder, ed, discriminator, dataloader, device, args):
    """
    V2. Input-energy：使用 D(input) 能量图做软门控融合。
    与当前 evaluation_pixel 的逻辑一致。
    """
    encoder.eval()
    ed.eval()
    if discriminator is not None:
        discriminator.eval()

    pixel_gt_list, pixel_score_list = [], []
    sample_gt_list, sample_score_list = [], []
    all_maps = []

    with torch.no_grad():
        for img, gt, label in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = ed(inputs)
            gt = gt.squeeze(1).cpu().numpy()
            recon_map = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='add')
            energy_maps = cal_energy_map(discriminator, inputs, img.shape[-1])
            final_map = soft_gate_fuse(
                recon_map=recon_map,
                energy_maps_t=energy_maps,
                k=args.gate_k,
                Te=args.gate_te,
                smooth_sigma=args.gate_sigma,
                recon_norm_quantile_low=args.recon_norm_quantile_low,
                recon_norm_quantile_high=args.recon_norm_quantile_high,
                recon_compress=args.recon_compress,
                fuse_output_norm=False,
            )
            anomaly_map = final_map
            all_maps.append(anomaly_map)
            gt_binary = (gt > 0.5).astype(int).reshape(-1)
            pixel_gt_list.append(gt_binary)
            pixel_score_list.append(anomaly_map.reshape(-1))
            sample_gt_list.append(np.max(gt.astype(int), axis=-1))
            sample_score_list.append(
                torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)),
                           args.topk, dim=-1)[0].mean(axis=-1).numpy())

    gt_arr = np.concatenate(pixel_gt_list)
    score_arr = np.concatenate(pixel_score_list)
    from util.test import calculate_metrics
    return {
        'Pixel': calculate_metrics(score_arr, gt_arr, acc=False),
        'Image': calculate_metrics(np.concatenate(sample_score_list),
                                  np.concatenate(sample_gt_list), acc=True),
    }, np.concatenate(all_maps)


# ============================================================================
# V3-V5: 单层能量差融合
# ============================================================================

def _eval_energy_diff_single_layer(encoder, ed, discriminator, dataloader,
                                    device, args, layer_idx: int):
    """
    V3/V4/V5. 单层能量差融合：仅使用指定层级的 D(input) - D(output) 做软门控。

    参数:
        layer_idx: 0 / 1 / 2，对应第一/二/三层
    """
    encoder.eval()
    ed.eval()
    if discriminator is not None:
        discriminator.eval()

    pixel_gt_list, pixel_score_list = [], []
    sample_gt_list, sample_score_list = [], []
    all_maps = []

    with torch.no_grad():
        for img, gt, label in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = ed(inputs)
            gt = gt.squeeze(1).cpu().numpy()

            # 计算能量差
            diff_maps = cal_energy_diff_maps(discriminator, inputs, outputs, img.shape[-1])
            diff_layer = diff_maps[layer_idx]  # [N, 1, H, W]

            # 能量差归一化到 [0, 1]（per-sample）
            diff_np = diff_layer.detach().cpu().numpy()  # [N,1,H,W]
            diff_stacked = diff_np.squeeze(1)  # [N,H,W]
            diff_normed = np.zeros_like(diff_stacked, dtype=np.float32)
            for i in range(diff_stacked.shape[0]):
                d = diff_stacked[i]
                d_min, d_max = d.min(), d.max()
                if d_max - d_min > 1e-12:
                    diff_normed[i] = (d - d_min) / (d_max - d_min)
                else:
                    diff_normed[i] = np.zeros_like(d)

            # 可选高斯平滑
            if args.gate_sigma > 0:
                for i in range(diff_normed.shape[0]):
                    diff_normed[i] = scipy_gaussian_filter(
                        diff_normed[i], sigma=args.gate_sigma)

            # 重建误差图
            recon_map = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='add')

            # 重建误差图归一化：截断 -> 压缩 -> minmax
            recon_clipped = quantile_clip(
                recon_map,
                quantile_low=args.recon_norm_quantile_low,
                quantile_high=args.recon_norm_quantile_high,
            )
            recon_compressed = high_tail_compress(recon_clipped, method=args.recon_compress)
            recon_normed = minmax_norm_per_sample(recon_compressed)

            # 计算门控权重（能量差越大，门控越开放）
            gate = sigmoid_np(args.gate_k * (diff_normed - args.gate_te))

            # 融合
            anomaly_map = (recon_normed.astype(np.float32)
                          * gate.astype(np.float32))
            all_maps.append(anomaly_map)

            gt_binary = (gt > 0.5).astype(int).reshape(-1)
            pixel_gt_list.append(gt_binary)
            pixel_score_list.append(anomaly_map.reshape(-1))
            sample_gt_list.append(np.max(gt.astype(int), axis=-1))
            sample_score_list.append(
                torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)),
                           args.topk, dim=-1)[0].mean(axis=-1).numpy())

    gt_arr = np.concatenate(pixel_gt_list)
    score_arr = np.concatenate(pixel_score_list)
    from util.test import calculate_metrics
    return {
        'Pixel': calculate_metrics(score_arr, gt_arr, acc=False),
        'Image': calculate_metrics(np.concatenate(sample_score_list),
                                  np.concatenate(sample_gt_list), acc=True),
    }, np.concatenate(all_maps)


def eval_e1_diff(encoder, ed, discriminator, dataloader, device, args):
    """V3. 第一层能量差融合（D(input) - D(output)，layer 0）"""
    return _eval_energy_diff_single_layer(encoder, ed, discriminator,
                                           dataloader, device, args, layer_idx=0)


def eval_e2_diff(encoder, ed, discriminator, dataloader, device, args):
    """V4. 第二层能量差融合（D(input) - D(output)，layer 1）"""
    return _eval_energy_diff_single_layer(encoder, ed, discriminator,
                                           dataloader, device, args, layer_idx=1)


def eval_e3_diff(encoder, ed, discriminator, dataloader, device, args):
    """V5. 第三层能量差融合（D(input) - D(output)，layer 2）"""
    return _eval_energy_diff_single_layer(encoder, ed, discriminator,
                                           dataloader, device, args, layer_idx=2)


# ============================================================================
# V6: 三层能量差平均融合
# ============================================================================

def eval_mean_e_diff(encoder, ed, discriminator, dataloader, device, args):
    """
    V6. 三层能量差平均融合：每层分别计算 D(input) - D(output)，
    各自上采样到原图大小后做等权平均，再用平均能量差做软门控。
    """
    encoder.eval()
    ed.eval()
    if discriminator is not None:
        discriminator.eval()

    pixel_gt_list, pixel_score_list = [], []
    sample_gt_list, sample_score_list = [], []
    all_maps = []

    with torch.no_grad():
        for img, gt, label in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = ed(inputs)
            gt = gt.squeeze(1).cpu().numpy()

            # 计算三层能量差
            diff_maps = cal_energy_diff_maps(discriminator, inputs, outputs, img.shape[-1])

            # 分别归一化每层能量差
            diff_normed_list = []
            for diff_layer in diff_maps:
                diff_np = diff_layer.detach().cpu().numpy().squeeze(1)  # [N,H,W]
                diff_normed = np.zeros_like(diff_np, dtype=np.float32)
                for i in range(diff_np.shape[0]):
                    d = diff_np[i]
                    d_min, d_max = d.min(), d.max()
                    if d_max - d_min > 1e-12:
                        diff_normed[i] = (d - d_min) / (d_max - d_min)
                    else:
                        diff_normed[i] = np.zeros_like(d)
                if args.gate_sigma > 0:
                    for i in range(diff_normed.shape[0]):
                        diff_normed[i] = scipy_gaussian_filter(
                            diff_normed[i], sigma=args.gate_sigma)
                diff_normed_list.append(diff_normed)

            # 三层等权平均
            diff_avg = np.stack(diff_normed_list, axis=1).mean(axis=1)  # [N, H, W]

            # 重建误差图归一化
            recon_map = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='add')
            recon_clipped = quantile_clip(
                recon_map,
                quantile_low=args.recon_norm_quantile_low,
                quantile_high=args.recon_norm_quantile_high,
            )
            recon_compressed = high_tail_compress(recon_clipped, method=args.recon_compress)
            recon_normed = minmax_norm_per_sample(recon_compressed)

            # 门控 + 融合
            gate = sigmoid_np(args.gate_k * (diff_avg - args.gate_te))
            anomaly_map = (recon_normed.astype(np.float32)
                          * gate.astype(np.float32))
            all_maps.append(anomaly_map)

            gt_binary = (gt > 0.5).astype(int).reshape(-1)
            pixel_gt_list.append(gt_binary)
            pixel_score_list.append(anomaly_map.reshape(-1))
            sample_gt_list.append(np.max(gt.astype(int), axis=-1))
            sample_score_list.append(
                torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)),
                           args.topk, dim=-1)[0].mean(axis=-1).numpy())

    gt_arr = np.concatenate(pixel_gt_list)
    score_arr = np.concatenate(pixel_score_list)
    from util.test import calculate_metrics
    return {
        'Pixel': calculate_metrics(score_arr, gt_arr, acc=False),
        'Image': calculate_metrics(np.concatenate(sample_score_list),
                                  np.concatenate(sample_gt_list), acc=True),
    }, np.concatenate(all_maps)


# ============================================================================
# 汇总打印
# ============================================================================

VERSIONS = [
    ('V1-Recon-only',       eval_recon_only),
    ('V2-Input-energy',     eval_input_energy_fusion),
    ('V3-E1-diff',          eval_e1_diff),
    ('V4-E2-diff',          eval_e2_diff),
    ('V5-E3-diff',          eval_e3_diff),
    ('V6-Mean-E-diff',      eval_mean_e_diff),
]


def run_all_versions(encoder, ed, discriminator, dataloader, device, args):
    """
    一次性跑全部 6 个版本，返回 results_dict
    results_dict: {name: {'Pixel': {...}, 'Image': {...}}}
    """
    results = {}
    for name, fn in VERSIONS:
        metrics, _ = fn(encoder, ed, discriminator, dataloader, device, args)
        results[name] = metrics
    return results


def print_comparison_table(results: dict, version_order=None):
    """
    打印横向对比表
    """
    if version_order is None:
        version_order = [name for name, _ in VERSIONS]

    header = "{:<22} {:>12} {:>10} {:>13} {:>8} {:>8}".format(
        "Version", "Pixel-AUROC", "PRO", "Image-AUROC", "F1", "ACC")
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("  能量图引导与筛选 - 多版本评估汇总表")
    print("=" * len(header))
    print(header)
    print(sep)

    for name in version_order:
        if name not in results:
            continue
        m = results[name]
        pixel = m.get('Pixel', {})
        image = m.get('Image', {})
        print("{:<22} {:>12.4f} {:>10.4f} {:>13.4f} {:>8.4f} {:>8.4f}".format(
            name,
            pixel.get('AUROC', 0.0),
            pixel.get('PRO', 0.0),
            image.get('AUROC', 0.0),
            image.get('F1', 0.0),
            image.get('ACC', 0.0)))

    print(sep)
    print("=" * len(header) + "\n")
