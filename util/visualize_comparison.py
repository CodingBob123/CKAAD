import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 从项目模块导入计算函数
from util.test import cal_anomaly_map, cal_energy_map, soft_gate_fuse, sigmoid_np, cal_energy_diff_map
from scipy.ndimage import gaussian_filter


# =============================================================================
# 统一可视化入口（使用缓存的中间结果，避免重复计算）
# =============================================================================

def visualize_recon_energy_unified(encoder, ed, discriminator, dataloader,
                                    args, device, epochs,
                                    cached_maps=None, cached_imgs_denorm=None,
                                    enable_stats=True):
    """
    统一可视化：统计对比 + 重建误差 vs 能量图对比
    所有输出共享同一次数据前向传递（通过 cached_maps 传入），
    彻底消除各可视化函数分别重复遍历 dataloader 的冗余。

    参数:
        encoder: 预训练特征提取器
        ed: 编码器-解码器模型
        discriminator: 判别器模型
        dataloader: 数据加载器（用于获取 ground truth / labels / 原图）
        args: 命令行参数
        device: 计算设备
        epochs: 当前 epoch 数
        cached_maps: dict 或 None，包含以下键:
            - recon_maps: np.ndarray [N, H, W] 重建误差图（来自 evaluation 返回）
            - energy_maps: list of torch.Tensor [[N,1,H,W], ...] 多尺度能量图
            - final_maps: np.ndarray [N, H, W] 融合后的异常图
        cached_imgs_denorm: torch.Tensor 或 None，[N,3,H,W] 反归一化后的图像
            若为 None，则在函数内部重新计算
        enable_stats: bool，是否打印统计信息
    """
    encoder.eval()
    ed.eval()
    discriminator.eval()

    # 获取空间尺寸
    if hasattr(discriminator, 'get_spatial_sizes'):
        spatial_sizes = discriminator.get_spatial_sizes()
    else:
        spatial_sizes = [64, 32, 16]

    # 图像反归一化 transform（可视化分支需要）
    img_transform = transforms.Compose([
        transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
                           std=(1/0.229, 1/0.224, 1/0.225))
    ])

    # ---------------------------------------------------------------
    # 提前检查：无缓存 map 时，直接打印警告并返回
    # ---------------------------------------------------------------
    if cached_maps is None:
        print("[visualize_recon_energy_unified] cached_maps is None, skipping all visualizations.")
        return None

    # Recon-only 模式下跳过可视化（无能量图数据）
    # ---------------------------------------------------------------
    if cached_maps.get('energy_maps') is None:
        print("[visualize_recon_energy_unified] energy_maps is None (recon_only mode), skipping visualization.")
        return None

    # ---------------------------------------------------------------
    # 计算统计信息
    # ---------------------------------------------------------------
    recon_maps = cached_maps['recon_maps']
    energy_maps = cached_maps['energy_maps']
    energy_maps_np = [em.cpu().numpy() for em in energy_maps]
    energy_avg = np.mean(np.stack([em.squeeze(1) for em in energy_maps_np]), axis=0)
    stats = {
        'recon': {'values': [], 'means': [], 'stds': [], 'maxs': []},
        'energy': {'values': [], 'means': [], 'stds': [], 'maxs': []},
        'correlation': []
    }
    for i in range(recon_maps.shape[0]):
        r = recon_maps[i]
        e = energy_avg[i]
        stats['recon']['values'].extend(r.flatten())
        stats['recon']['means'].append(r.mean())
        stats['recon']['stds'].append(r.std())
        stats['recon']['maxs'].append(r.max())
        stats['energy']['values'].extend(e.flatten())
        stats['energy']['means'].append(e.mean())
        stats['energy']['stds'].append(e.std())
        stats['energy']['maxs'].append(e.max())
        corr = np.corrcoef(r.flatten(), e.flatten())[0, 1]
        stats['correlation'].append(corr)

    # ---------------------------------------------------------------
    # 仅打印统计信息（不生成图）
    # ---------------------------------------------------------------
    if enable_stats:
        print("=" * 70)
        print("重建误差图 vs 能量图 统计对比")
        print("=" * 70)
        print(f"\n重建误差图 (Reconstruction Error Map):")
        print(f"  全局均值: {np.mean(stats['recon']['means']):.6f}")
        print(f"  全局标准差: {np.mean(stats['recon']['stds']):.6f}")
        print(f"  全局最大值: {np.mean(stats['recon']['maxs']):.6f}")
        print(f"  总体像素范围: [{np.min(stats['recon']['values']):.6f}, {np.max(stats['recon']['values']):.6f}]")
        print(f"\n能量图 (Energy Map):")
        print(f"  全局均值: {np.mean(stats['energy']['means']):.6f}")
        print(f"  全局标准差: {np.mean(stats['energy']['stds']):.6f}")
        print(f"  全局最大值: {np.mean(stats['energy']['maxs']):.6f}")
        print(f"  总体像素范围: [{np.min(stats['energy']['values']):.6f}, {np.max(stats['energy']['values']):.6f}]")
        print(f"\n重建误差 vs 能量图 相关系数:")
        print(f"  平均相关系数: {np.mean(stats['correlation']):.6f}")
        print(f"  相关系数标准差: {np.std(stats['correlation']):.6f}")
        print(f"  相关系数范围: [{np.min(stats['correlation']):.6f}, {np.max(stats['correlation']):.6f}]")
        print("=" * 70)

    # ---------------------------------------------------------------
    # 生成可视化图像（遍历 dataloader 获取图像 / 标签 / GT）
    # ---------------------------------------------------------------
    # result_path = '/hy-tmp/results/{}_{}_recon_vs_energy_epoch_{}'.format(
    result_path = './viz_results_day1/{}_{}_epoch_{}'.format(
        args.dataset, args.normal, epochs)
    os.makedirs(result_path, exist_ok=True)

    # 若传入预拼接的图像缓存，则可以按索引切片取出当前 batch
    use_cached_imgs = cached_imgs_denorm is not None

    with torch.no_grad():
        type_to_count = {}  # 跟踪每个类型已处理的样本数量
        sample_idx = 0

        for data in dataloader:
            imgs = data[0].to(device)
            labels = data[-1]
            gt_batch = data[1] if len(data) >= 3 else None
            batch_size = imgs.size(0)

            # 获取当前 batch 的反归一化图像
            if use_cached_imgs:
                imgs_denorm = cached_imgs_denorm[sample_idx:sample_idx + batch_size]
            else:
                imgs_denorm = img_transform(imgs)

            for i in range(batch_size):
                label = labels[i]
                type_name = (dataloader.dataset.types_set[label.item()]
                             if hasattr(dataloader.dataset, 'types_set')
                             else f"label_{label.item()}")

                # 每类型只绘制指定数量的样本（viz_samples_per_type）
                # if type_to_count.get(type_name, 0) >= args.viz_samples_per_type:
                if type_to_count.get(type_name, 0) >= 2:  # 每个异常类型各2张
                    sample_idx += 1
                    continue
                type_to_count[type_name] = type_to_count.get(type_name, 0) + 1

                img = imgs_denorm[i]
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)

                gt = (gt_batch[i].squeeze(0).cpu().numpy()
                      if gt_batch is not None else None)

                recon_vis = cached_maps['recon_maps'][sample_idx]
                final_vis = cached_maps['final_maps'][sample_idx]
                energy_maps_vis = cached_maps['energy_maps']
                n_scales = len(energy_maps_vis)

                # ------------------------------------------------------------
                # 新布局设计（2行 x 9列，以3层能量图为例）
                # 行0（热力图）：Orig(col0) | GT(col1) | Recon(col2) | E1(col3) | E2(col4) | E3(col5) | Final(col6) | Overlay(col7) | [空(col8)]
                # 行1（分布图）：[空(col0)] | [空(col1)] | Recon Dist(col2) | E1 Dist(col3) | E2 Dist(col4) | E3 Dist(col5) | Final Dist(col6) | [空(col7)] | [空(col8)]
                # ------------------------------------------------------------
                num_cols = 9
                fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))

                # ---- 行0：热力图 ----
                col = 0
                axes[0, col].imshow(img_np)
                axes[0, col].set_title(f'Original Image\nType: {type_name}', fontsize=9, fontweight='bold')
                axes[0, col].axis('off')

                col = 1
                if gt is not None:
                    axes[0, col].imshow(gt, cmap='gray')
                    axes[0, col].set_title('Ground Truth', fontsize=9, fontweight='bold')
                else:
                    axes[0, col].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=9)
                    axes[0, col].set_title('Ground Truth', fontsize=9)
                axes[0, col].axis('off')

                col = 2
                im_recon = axes[0, col].imshow(recon_vis, cmap='jet')
                axes[0, col].set_title('Recon Error Map', fontsize=9, fontweight='bold')
                axes[0, col].axis('off')
                plt.colorbar(im_recon, ax=axes[0, col], shrink=0.8)

                for j in range(n_scales):
                    col = 3 + j
                    ev = energy_maps_vis[j][sample_idx, 0].cpu().numpy()
                    im_e = axes[0, col].imshow(ev, cmap='jet')
                    axes[0, col].set_title(
                        f'Energy Scale {j+1}\n{spatial_sizes[j]}x{spatial_sizes[j]}',
                        fontsize=9, fontweight='bold')
                    axes[0, col].axis('off')
                    plt.colorbar(im_e, ax=axes[0, col], shrink=0.8)

                col = 3 + n_scales
                im_final = axes[0, col].imshow(final_vis, cmap='jet')
                axes[0, col].set_title('Final Fused Map', fontsize=9, fontweight='bold')
                axes[0, col].axis('off')
                plt.colorbar(im_final, ax=axes[0, col], shrink=0.8)

                col = 3 + n_scales + 1
                axes[0, col].imshow(img_np)
                axes[0, col].imshow(final_vis, cmap='jet', alpha=0.5)
                axes[0, col].set_title('Fused + Original\nOverlay', fontsize=9, fontweight='bold')
                axes[0, col].axis('off')

                # ---- 行1：分布直方图（与行0热力图列对齐） ----
                # 列0-1：空（与 Original / GT 对齐）
                axes[1, 0].axis('off')
                axes[1, 1].axis('off')

                # 列2：Recon Error 分布
                col = 2
                axes[1, col].hist(recon_vis.flatten(), bins=50, color='blue', alpha=0.7)
                axes[1, col].set_title('Recon Error\nDistribution', fontsize=9)
                axes[1, col].set_xlabel('Value')
                axes[1, col].set_ylabel('Frequency')

                # 列3~：各 Energy Scale 分布
                for j in range(n_scales):
                    col = 3 + j
                    ev = energy_maps_vis[j][sample_idx, 0].cpu().numpy()
                    axes[1, col].hist(ev.flatten(), bins=50, color='orange', alpha=0.7)
                    axes[1, col].set_title(f'Energy Scale {j+1}\nDistribution', fontsize=9)
                    axes[1, col].set_xlabel('Value')
                    axes[1, col].set_ylabel('Frequency')

                # Final Fused 分布
                col = 3 + n_scales
                axes[1, col].hist(final_vis.flatten(), bins=50, color='green', alpha=0.7)
                axes[1, col].set_title('Final Fused\nDistribution', fontsize=9)
                axes[1, col].set_xlabel('Value')
                axes[1, col].set_ylabel('Frequency')

                # 剩余列置空
                for col in range(3 + n_scales + 1, num_cols):
                    axes[1, col].axis('off')

                plt.suptitle(f'Recon vs Energy - {type_name} (Epoch {epochs})',
                            fontsize=13, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.savefig(os.path.join(result_path, f'{type_name}.png'),
                            dpi=150, bbox_inches='tight')
                plt.close()

                sample_idx += 1

    print(f"\nVisualization saved to: {result_path}")
    print(f"Total samples visualized: {sum(type_to_count.values())} ({', '.join(sorted(type_to_count.keys()))})")

    return stats


# =============================================================================
# 能量差模式可视化（V3-V6：D(input) - D(output)）
# =============================================================================

def visualize_energy_diff_unified(encoder, ed, discriminator, dataloader,
                                   args, device, epochs,
                                   cached_maps=None, cached_imgs_denorm=None,
                                   enable_stats=True):
    """
    能量差模式可视化：重建误差 vs 输入/输出/差值能量图对比
    用于 V3-V6 的能量差模式可视化

    新布局（4行 x N列）：
    - 行0（主图）：Orig | GT | Recon | E_in | E_out | E_diff | Final | Overlay
    - 行1（E_in 分布）：各尺度输入能量图分布
    - 行2（E_out 分布）：各尺度输出能量图分布
    - 行3（E_diff 分布）：各尺度能量差图分布

    参数:
        encoder: 预训练特征提取器
        ed: 编码器-解码器模型
        discriminator: 判别器模型
        dataloader: 数据加载器
        args: 命令行参数
        device: 计算设备
        epochs: 当前 epoch 数
        cached_maps: dict，必须包含以下键（能量差模式）:
            - recon_maps: np.ndarray [N, H, W] 重建误差图
            - energy_maps: list of torch.Tensor [[N,1,H,W], ...] 能量差图
            - energy_in_maps: list of torch.Tensor [[N,1,H,W], ...] 输入能量图
            - energy_out_maps: list of torch.Tensor [[N,1,H,W], ...] 输出能量图
            - final_maps: np.ndarray [N, H, W] 融合后的异常图
        cached_imgs_denorm: torch.Tensor 或 None，[N,3,H,W] 反归一化后的图像
        enable_stats: bool，是否打印统计信息
    """
    encoder.eval()
    ed.eval()
    discriminator.eval()

    # 获取空间尺寸
    if hasattr(discriminator, 'get_spatial_sizes'):
        spatial_sizes = discriminator.get_spatial_sizes()
    else:
        spatial_sizes = [64, 32, 16]

    # 图像反归一化 transform
    img_transform = transforms.Compose([
        transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
                           std=(1/0.229, 1/0.224, 1/0.225))
    ])

    # ---------------------------------------------------------------
    # 提前检查
    # ---------------------------------------------------------------
    if cached_maps is None:
        print("[visualize_energy_diff_unified] cached_maps is None, skipping.")
        return None

    energy_diff_maps = cached_maps.get('energy_maps')
    energy_in_maps = cached_maps.get('energy_in_maps')
    energy_out_maps = cached_maps.get('energy_out_maps')

    if energy_diff_maps is None or energy_in_maps is None or energy_out_maps is None:
        print("[visualize_energy_diff_unified] energy_maps/energy_in_maps/energy_out_maps is None, skipping.")
        return None

    recon_maps = cached_maps['recon_maps']
    final_maps = cached_maps['final_maps']
    n_scales = len(energy_diff_maps)

    # ---------------------------------------------------------------
    # 计算统计信息
    # ---------------------------------------------------------------
    # 计算能量差平均值
    energy_diff_np = [em.cpu().numpy() for em in energy_diff_maps]
    energy_diff_avg = np.mean(np.stack([em.squeeze(1) for em in energy_diff_np]), axis=0)

    stats = {
        'recon': {'means': [], 'stds': [], 'maxs': []},
        'energy_in': {'means': [], 'stds': [], 'maxs': []},
        'energy_out': {'means': [], 'stds': [], 'maxs': []},
        'energy_diff': {'means': [], 'stds': [], 'maxs': []},
        'correlation_diff': []
    }

    for i in range(recon_maps.shape[0]):
        r = recon_maps[i]
        e_in_avg = np.mean([em[i, 0].cpu().numpy() for em in energy_in_maps], axis=0)
        e_out_avg = np.mean([em[i, 0].cpu().numpy() for em in energy_out_maps], axis=0)
        e_diff_avg = energy_diff_avg[i]

        stats['recon']['means'].append(r.mean())
        stats['recon']['stds'].append(r.std())
        stats['recon']['maxs'].append(r.max())

        stats['energy_in']['means'].append(e_in_avg.mean())
        stats['energy_in']['stds'].append(e_in_avg.std())
        stats['energy_in']['maxs'].append(e_in_avg.max())

        stats['energy_out']['means'].append(e_out_avg.mean())
        stats['energy_out']['stds'].append(e_out_avg.std())
        stats['energy_out']['maxs'].append(e_out_avg.max())

        stats['energy_diff']['means'].append(e_diff_avg.mean())
        stats['energy_diff']['stds'].append(e_diff_avg.std())
        stats['energy_diff']['maxs'].append(e_diff_avg.max())

        corr = np.corrcoef(r.flatten(), e_diff_avg.flatten())[0, 1]
        stats['correlation_diff'].append(corr)

    if enable_stats:
        print("=" * 70)
        print("能量差模式 - 统计对比 (V3-V6)")
        print("=" * 70)
        print(f"\n重建误差图:")
        print(f"  均值: {np.mean(stats['recon']['means']):.6f}, 标准差: {np.mean(stats['recon']['stds']):.6f}, 最大值: {np.mean(stats['recon']['maxs']):.6f}")
        print(f"\n输入能量图 E_in:")
        print(f"  均值: {np.mean(stats['energy_in']['means']):.6f}, 标准差: {np.mean(stats['energy_in']['stds']):.6f}, 最大值: {np.mean(stats['energy_in']['maxs']):.6f}")
        print(f"\n输出能量图 E_out:")
        print(f"  均值: {np.mean(stats['energy_out']['means']):.6f}, 标准差: {np.mean(stats['energy_out']['stds']):.6f}, 最大值: {np.mean(stats['energy_out']['maxs']):.6f}")
        print(f"\n能量差图 E_diff = E_in - E_out:")
        print(f"  均值: {np.mean(stats['energy_diff']['means']):.6f}, 标准差: {np.mean(stats['energy_diff']['stds']):.6f}, 最大值: {np.mean(stats['energy_diff']['maxs']):.6f}")
        print(f"\n重建误差 vs 能量差 相关系数:")
        print(f"  平均: {np.mean(stats['correlation_diff']):.6f}, 标准差: {np.std(stats['correlation_diff']):.6f}")
        print("=" * 70)

    # ---------------------------------------------------------------
    # 生成可视化图像
    # ---------------------------------------------------------------
    # result_path = '/hy-tmp/results/{}_{}_recon_vs_energy_epoch_{}'.format(
    result_path = './viz_results_day1/{}_{}_energy_diff_epoch_{}'.format(
        args.dataset, args.normal, epochs)
    os.makedirs(result_path, exist_ok=True)

    use_cached_imgs = cached_imgs_denorm is not None

    with torch.no_grad():
        type_to_count = {}
        sample_idx = 0

        for data in dataloader:
            imgs = data[0].to(device)
            labels = data[-1]
            gt_batch = data[1] if len(data) >= 3 else None
            batch_size = imgs.size(0)

            if use_cached_imgs:
                imgs_denorm = cached_imgs_denorm[sample_idx:sample_idx + batch_size]
            else:
                imgs_denorm = img_transform(imgs)

            for i in range(batch_size):
                label = labels[i]
                type_name = (dataloader.dataset.types_set[label.item()]
                             if hasattr(dataloader.dataset, 'types_set')
                             else f"label_{label.item()}")

                if type_to_count.get(type_name, 0) >= 2:  # 每个异常类型各2张
                    sample_idx += 1
                    continue
                type_to_count[type_name] = type_to_count.get(type_name, 0) + 1

                img = imgs_denorm[i]
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)

                gt = (gt_batch[i].squeeze(0).cpu().numpy()
                      if gt_batch is not None else None)

                recon_vis = recon_maps[sample_idx]
                final_vis = final_maps[sample_idx]

                # ------------------------------------------------------------
                # 新布局设计（4行 x 11列，以3层能量图为例）
                # 行0（主图热力图）：Orig | GT | Recon | E_in1 | E_in2 | E_in3 | E_out1 | E_out2 | E_out3 | Final | Overlay
                # 行1（E_in 分布）：各尺度输入能量图分布
                # 行2（E_out 分布）：各尺度输出能量图分布
                # 行3（E_diff 分布）：各尺度能量差图分布
                # ------------------------------------------------------------
                num_cols = 3 + n_scales * 3 + 2 + 1  # Orig + GT + Recon + 3*E_in + 3*E_out + Final + Overlay
                num_cols = 3 + n_scales * 3 + 2  # Orig + GT + Recon + 3*E_in + 3*E_out + Final + Overlay (去掉最后一列空位)

                # 简化布局：3 + n_scales*3 + 2 = 14列（Orig, GT, Recon, E_in*3, E_out*3, Final, Overlay）
                num_cols = 3 + n_scales * 3 + 2
                fig, axes = plt.subplots(4, num_cols, figsize=(4 * num_cols, 16))

                # ---- 行0：主图热力图 ----
                # 列0：原始图像
                col = 0
                axes[0, col].imshow(img_np)
                axes[0, col].set_title(f'Original\n{type_name}', fontsize=8, fontweight='bold')
                axes[0, col].axis('off')

                # 列1：Ground Truth
                col = 1
                if gt is not None:
                    axes[0, col].imshow(gt, cmap='gray')
                    axes[0, col].set_title('Ground Truth', fontsize=8, fontweight='bold')
                else:
                    axes[0, col].text(0.5, 0.5, 'No GT', ha='center', va='center', fontsize=8)
                    axes[0, col].set_title('Ground Truth', fontsize=8)
                axes[0, col].axis('off')

                # 列2：重建误差图
                col = 2
                im_recon = axes[0, col].imshow(recon_vis, cmap='jet')
                axes[0, col].set_title('Recon\nError', fontsize=8, fontweight='bold')
                axes[0, col].axis('off')
                plt.colorbar(im_recon, ax=axes[0, col], shrink=0.8)

                # 列3~5：输入能量图 E_in (3 scales)
                for j in range(n_scales):
                    col = 3 + j
                    e_in = energy_in_maps[j][sample_idx, 0].cpu().numpy()
                    im_e = axes[0, col].imshow(e_in, cmap='jet')
                    axes[0, col].set_title(f'E_in S{j+1}\n{spatial_sizes[j]}x{spatial_sizes[j]}',
                                          fontsize=8, fontweight='bold')
                    axes[0, col].axis('off')
                    plt.colorbar(im_e, ax=axes[0, col], shrink=0.8)

                # 列6~8：输出能量图 E_out (3 scales)
                for j in range(n_scales):
                    col = 3 + n_scales + j
                    e_out = energy_out_maps[j][sample_idx, 0].cpu().numpy()
                    im_e = axes[0, col].imshow(e_out, cmap='jet')
                    axes[0, col].set_title(f'E_out S{j+1}\n{spatial_sizes[j]}x{spatial_sizes[j]}',
                                          fontsize=8, fontweight='bold')
                    axes[0, col].axis('off')
                    plt.colorbar(im_e, ax=axes[0, col], shrink=0.8)

                # 列9：Final 融合图
                col = 3 + n_scales * 2
                im_final = axes[0, col].imshow(final_vis, cmap='jet')
                axes[0, col].set_title('Final\nFused', fontsize=8, fontweight='bold')
                axes[0, col].axis('off')
                plt.colorbar(im_final, ax=axes[0, col], shrink=0.8)

                # 列10：Overlay 叠加图
                col = 3 + n_scales * 2 + 1
                axes[0, col].imshow(img_np)
                axes[0, col].imshow(final_vis, cmap='jet', alpha=0.5)
                axes[0, col].set_title('Fused\nOverlay', fontsize=8, fontweight='bold')
                axes[0, col].axis('off')

                # ---- 行1：E_in 各尺度分布 ----
                axes[1, 0].axis('off')
                axes[1, 1].axis('off')
                axes[1, 2].axis('off')

                for j in range(n_scales):
                    col = 3 + j
                    e_in = energy_in_maps[j][sample_idx, 0].cpu().numpy()
                    axes[1, col].hist(e_in.flatten(), bins=50, color='blue', alpha=0.7)
                    axes[1, col].set_title(f'E_in S{j+1}\nDist', fontsize=7)
                    axes[1, col].set_xlabel('Value')
                    axes[1, col].set_ylabel('Freq')

                for j in range(n_scales):
                    col = 3 + n_scales + j
                    axes[1, col].axis('off')

                axes[1, 3 + n_scales * 2].axis('off')
                axes[1, 3 + n_scales * 2 + 1].axis('off')

                # ---- 行2：E_out 各尺度分布 ----
                axes[2, 0].axis('off')
                axes[2, 1].axis('off')
                axes[2, 2].axis('off')

                for j in range(n_scales):
                    col = 3 + j
                    axes[2, col].axis('off')

                for j in range(n_scales):
                    col = 3 + n_scales + j
                    e_out = energy_out_maps[j][sample_idx, 0].cpu().numpy()
                    axes[2, col].hist(e_out.flatten(), bins=50, color='orange', alpha=0.7)
                    axes[2, col].set_title(f'E_out S{j+1}\nDist', fontsize=7)
                    axes[2, col].set_xlabel('Value')
                    axes[2, col].set_ylabel('Freq')

                axes[2, 3 + n_scales * 2].axis('off')
                axes[2, 3 + n_scales * 2 + 1].axis('off')

                # ---- 行3：E_diff 各尺度分布 ----
                axes[3, 0].axis('off')
                axes[3, 1].axis('off')
                axes[3, 2].axis('off')

                for j in range(n_scales):
                    col = 3 + j
                    axes[3, col].axis('off')

                for j in range(n_scales):
                    col = 3 + n_scales + j
                    axes[3, col].axis('off')

                # E_diff 分布在 Final 和 Overlay 列
                col = 3 + n_scales * 2
                e_diff = energy_diff_maps[j][sample_idx, 0].cpu().numpy()  # 使用最后一层
                # 显示所有尺度的 E_diff 平均的分布
                e_diff_avg_sample = np.mean([energy_diff_maps[k][sample_idx, 0].cpu().numpy()
                                            for k in range(n_scales)], axis=0)
                axes[3, col].hist(e_diff_avg_sample.flatten(), bins=50, color='red', alpha=0.7)
                axes[3, col].set_title('E_diff Avg\nDist', fontsize=7)
                axes[3, col].set_xlabel('Value')
                axes[3, col].set_ylabel('Freq')

                axes[3, 3 + n_scales * 2 + 1].axis('off')

                plt.suptitle(f'Energy Diff Visualization (V3-V6) - {type_name} (Epoch {epochs})',
                            fontsize=12, fontweight='bold', y=1.01)
                plt.tight_layout()
                plt.savefig(os.path.join(result_path, f'{type_name}.png'),
                            dpi=150, bbox_inches='tight')
                plt.close()

                sample_idx += 1

    print(f"\nEnergy Diff Visualization saved to: {result_path}")
    print(f"Total samples visualized: {sum(type_to_count.values())} ({', '.join(sorted(type_to_count.keys()))})")
