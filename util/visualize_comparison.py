import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 从项目模块导入计算函数
from util.test import cal_anomaly_map, cal_energy_map, soft_gate_fuse, sigmoid_np
from scipy.ndimage import gaussian_filter


# =============================================================================
# 统一可视化入口（使用缓存的中间结果，避免重复计算）
# =============================================================================

def visualize_recon_energy_unified(encoder, ed, discriminator, dataloader,
                                    args, device, epochs,
                                    cached_maps=None, cached_imgs_denorm=None,
                                    enable_stats=True, enable_comparison=True,
                                    enable_multiscale=True):
    """
    统一可视化：统计对比 + 重建误差 vs 能量图对比 + 多尺度能量图可视化
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
        cached_maps: dict 或 None，可选，包含以下键:
            - recon_maps: np.ndarray [N, H, W] 重建误差图（来自 evaluation 返回）
            - energy_maps: list of torch.Tensor [[N,1,H,W], ...] 多尺度能量图
            - final_maps: np.ndarray [N, H, W] 融合后的异常图
        cached_imgs_denorm: torch.Tensor 或 None，[N,3,H,W] 反归一化后的图像
            若为 None，则在函数内部重新计算
        enable_stats: bool，是否打印统计信息（对应 compare_recon_energy_statistics）
            注意：stats 总是计算并返回，enable_stats 仅控制是否打印
        enable_comparison: bool，是否生成重建 vs 能量对比图
        enable_multiscale: bool，是否生成多尺度能量叠加图
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

    # ---------------------------------------------------------------
    # 总是计算 stats（用于返回；enable_stats 仅控制是否打印）
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

    # 仅当不需要任何可视化时，才在这里打印并返回
    if not enable_comparison and not enable_multiscale:
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
        return stats

    # ---------------------------------------------------------------
    # 需要生成可视化图像（遍历 dataloader 获取图像 / 标签 / GT）
    # ---------------------------------------------------------------
    prefix_comparison = './results/{}_{}_recon_vs_energy_epoch_{}'.format(
        args.dataset, args.normal, epochs)
    prefix_multiscale = './results/{}_{}_multiscale_energy_epoch_{}'.format(
        args.dataset, args.normal, epochs)

    if enable_comparison:
        os.makedirs(prefix_comparison, exist_ok=True)
    if enable_multiscale:
        os.makedirs(prefix_multiscale, exist_ok=True)

    # 若传入预拼接的图像缓存，则可以按索引切片取出当前 batch
    use_cached_imgs = cached_imgs_denorm is not None

    with torch.no_grad():
        seen_types = set()
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

                # 跳过已处理过的类型，每个类型只画一张
                if type_name in seen_types:
                    sample_idx += 1
                    continue
                seen_types.add(type_name)

                img = imgs_denorm[i]
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)

                gt = (gt_batch[i].squeeze(0).cpu().numpy()
                      if gt_batch is not None else None)

                recon_vis = cached_maps['recon_maps'][sample_idx]
                final_vis = cached_maps['final_maps'][sample_idx]
                energy_maps_vis = cached_maps['energy_maps']
                n_scales = len(energy_maps_vis)

                # ---- 重建误差 vs 能量图对比 ----
                if enable_comparison:
                    num_cols = 4 + n_scales
                    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))

                    axes[0, 0].imshow(img_np)
                    axes[0, 0].set_title(f'Original Image\nType: {type_name}', fontsize=10, fontweight='bold')
                    axes[0, 0].axis('off')

                    im1 = axes[0, 1].imshow(recon_vis, cmap='jet')
                    axes[0, 1].set_title('Recon Error Map\n(Reconstruction)', fontsize=10, fontweight='bold')
                    axes[0, 1].axis('off')
                    plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

                    for j in range(n_scales):
                        ev = energy_maps_vis[j][sample_idx, 0].cpu().numpy()
                        im = axes[0, 2 + j].imshow(ev, cmap='jet')
                        axes[0, 2 + j].set_title(
                            f'Energy Map (Scale {j+1})\n{spatial_sizes[j]}x{spatial_sizes[j]}',
                            fontsize=10, fontweight='bold')
                        axes[0, 2 + j].axis('off')
                        plt.colorbar(im, ax=axes[0, 2 + j], shrink=0.8)

                    im_f = axes[0, -1].imshow(final_vis, cmap='jet')
                    axes[0, -1].set_title('Final Fused Map\n(Soft Gate)', fontsize=10, fontweight='bold')
                    axes[0, -1].axis('off')
                    plt.colorbar(im_f, ax=axes[0, -1], shrink=0.8)

                    if gt is not None:
                        axes[1, 0].imshow(gt, cmap='gray')
                        axes[1, 0].set_title('Ground Truth', fontsize=10, fontweight='bold')
                        axes[1, 0].axis('off')
                    else:
                        axes[1, 0].hist(recon_vis.flatten(), bins=50, color='blue', alpha=0.7)
                        axes[1, 0].set_title('Recon Error\nDistribution', fontsize=10)
                        axes[1, 0].set_xlabel('Value')
                        axes[1, 0].set_ylabel('Frequency')

                    for j in range(n_scales):
                        ev = energy_maps_vis[j][sample_idx, 0].cpu().numpy()
                        axes[1, 1 + j].hist(ev.flatten(), bins=50, color='orange', alpha=0.7)
                        axes[1, 1 + j].set_title(f'Energy Scale {j+1}\nDistribution', fontsize=10)
                        axes[1, 1 + j].set_xlabel('Value')
                        axes[1, 1 + j].set_ylabel('Frequency')

                    axes[1, -1].hist(final_vis.flatten(), bins=50, color='green', alpha=0.7)
                    axes[1, -1].set_title('Final Fused\nDistribution', fontsize=10)
                    axes[1, -1].set_xlabel('Value')
                    axes[1, -1].set_ylabel('Frequency')

                    plt.suptitle(f'Recon vs Energy - {type_name} (Epoch {epochs})',
                                fontsize=14, fontweight='bold', y=1.02)
                    plt.tight_layout()
                    plt.savefig(os.path.join(prefix_comparison, f'{type_name}.png'),
                                dpi=150, bbox_inches='tight')
                    plt.close()

                # ---- 多尺度能量图叠加可视化 ----
                if enable_multiscale:
                    fig, axes = plt.subplots(2, n_scales + 1,
                                            figsize=(5 * (n_scales + 1), 10))

                    axes[0, 0].imshow(img_np)
                    axes[0, 0].set_title(f'Original\n{type_name}', fontsize=10, fontweight='bold')
                    axes[0, 0].axis('off')

                    for j in range(n_scales):
                        ev = energy_maps_vis[j][sample_idx, 0].cpu().numpy()
                        im = axes[0, j + 1].imshow(ev, cmap='jet', vmin=0, vmax=ev.max())
                        axes[0, j + 1].set_title(
                            f'Energy Scale {j+1}\n{spatial_sizes[j]}x{spatial_sizes[j]}',
                            fontsize=10, fontweight='bold')
                        axes[0, j + 1].axis('off')
                        plt.colorbar(im, ax=axes[0, j + 1], shrink=0.8)

                    axes[1, 0].imshow(img_np)
                    axes[1, 0].set_title('Original', fontsize=10, fontweight='bold')
                    axes[1, 0].axis('off')

                    for j in range(n_scales):
                        ev = energy_maps_vis[j][sample_idx, 0].cpu().numpy()
                        axes[1, j + 1].imshow(img_np)
                        im = axes[1, j + 1].imshow(ev, cmap='jet', alpha=0.6)
                        axes[1, j + 1].set_title(f'Overlay Scale {j+1}', fontsize=10, fontweight='bold')
                        axes[1, j + 1].axis('off')

                    plt.suptitle(f'Multi-Scale Energy - {type_name} (Epoch {epochs})',
                                fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(prefix_multiscale, f'{type_name}_multi_scale.png'),
                                dpi=150, bbox_inches='tight')
                    plt.close()

                sample_idx += 1

    if enable_comparison:
        print(f"\nVisualization saved to: {prefix_comparison}")
        print(f"Total samples visualized: {len(seen_types)} ({', '.join(sorted(seen_types))})")
    if enable_multiscale:
        print(f"Multi-scale energy visualization saved to: {prefix_multiscale}")

    return stats


# =============================================================================
# 以下为旧版独立可视化函数（保持向后兼容，内部逻辑不变）
# =============================================================================

def visualize_recon_vs_energy(encoder, ed, discriminator, dataloader, args, device, epochs):
    """
    可视化重建误差图与能量图的对比（独立版本，独立计算缓存）

    功能：
    1. 显示原始图像
    2. 显示重建误差图 (recon_map)
    3. 显示多尺度能量图 (energy_map)
    4. 显示融合后的异常图
    5. 如果有ground truth也显示

    参数:
        encoder: 预训练特征提取器
        ed: 编码器-解码器模型
        discriminator: 判别器模型
        dataloader: 数据加载器
        args: 命令行参数
        device: 计算设备
        epochs: 当前epoch数
    """
    encoder.eval()
    ed.eval()
    discriminator.eval()

    # 获取空间尺寸
    if hasattr(discriminator, 'get_spatial_sizes'):
        spatial_sizes = discriminator.get_spatial_sizes()
    else:
        spatial_sizes = [64, 32, 16]

    with torch.no_grad():
        cnt = 0
        for data in dataloader:
            imgs = data[0].to(device)
            inputs = encoder(imgs)
            outputs = ed(inputs)
            labels = data[-1]

            # 1. 计算重建误差图
            recon_map = cal_anomaly_map(inputs, outputs, imgs.shape[-1], amap_mode='add')  # [N, H, W]

            # 2. 计算多尺度能量图
            energy_maps = cal_energy_map(discriminator, inputs, imgs.shape[-1])  # [[N,1,H,W], ...]

            # 3. 计算融合结果（可视化时开启 fuse_output_norm 以提升热力图对比度）
            final_map = soft_gate_fuse(
                recon_map=recon_map,
                energy_maps_t=energy_maps,
                k=args.gate_k,
                Te=args.gate_te,
                smooth_sigma=args.gate_sigma,
                recon_norm_quantile_low=args.recon_norm_quantile_low,
                recon_norm_quantile_high=args.recon_norm_quantile_high,
                recon_compress=args.recon_compress,
                fuse_output_norm=True
            )

            # 4. 反变换图像
            img_transform = transforms.Compose([
                transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
                                   std=(1/0.229, 1/0.224, 1/0.225))
            ])
            imgs_denorm = img_transform(imgs)

            # 5. 创建结果目录
            result_path = './results/{}_{}_recon_vs_energy_epoch_{}'.format(
                args.dataset, args.normal, epochs)
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=True)

            for i in range(imgs.size(0)):
                img = imgs_denorm[i]
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)

                # 获取标签
                label = labels[i]
                if hasattr(dataloader.dataset, 'types_set'):
                    type_name = dataloader.dataset.types_set[label.item()]
                else:
                    type_name = f"label_{label.item()}"

                # 获取ground truth
                if len(data) >= 3:
                    gt = data[1][i].squeeze(0).cpu().numpy()
                else:
                    gt = None

                # 创建子图
                num_cols = 4 + len(energy_maps)  # 原图 + 重建 + 多个能量图 + 融合
                fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10))

                # 第一行：原图、重建误差图、各尺度能量图
                # 原图
                axes[0, 0].imshow(img_np)
                axes[0, 0].set_title(f'Original Image\nType: {type_name}', fontsize=10, fontweight='bold')
                axes[0, 0].axis('off')

                # 重建误差图
                recon_vis = recon_map[i]
                im1 = axes[0, 1].imshow(recon_vis, cmap='jet')
                axes[0, 1].set_title('Recon Error Map\n(Reconstruction)', fontsize=10, fontweight='bold')
                axes[0, 1].axis('off')
                plt.colorbar(im1, ax=axes[0, 1], shrink=0.8)

                # 各尺度能量图
                for j, energy_map in enumerate(energy_maps):
                    energy_vis = energy_map[i, 0].cpu().numpy()
                    scale = spatial_sizes[j]
                    im = axes[0, 2 + j].imshow(energy_vis, cmap='jet')
                    axes[0, 2 + j].set_title(f'Energy Map (Scale {j+1})\n{scale}x{scale}', fontsize=10, fontweight='bold')
                    axes[0, 2 + j].axis('off')
                    plt.colorbar(im, ax=axes[0, 2 + j], shrink=0.8)

                # 融合图
                final_vis = final_map[i]
                im_f = axes[0, -1].imshow(final_vis, cmap='jet')
                axes[0, -1].set_title('Final Fused Map\n(Soft Gate)', fontsize=10, fontweight='bold')
                axes[0, -1].axis('off')
                plt.colorbar(im_f, ax=axes[0, -1], shrink=0.8)

                # 第二行：各图的像素值分布直方图
                # 重建误差图分布
                axes[1, 0].hist(recon_vis.flatten(), bins=50, color='blue', alpha=0.7)
                axes[1, 0].set_title('Recon Error\nDistribution', fontsize=10)
                axes[1, 0].set_xlabel('Value')
                axes[1, 0].set_ylabel('Frequency')

                # 各尺度能量图分布
                for j, energy_map in enumerate(energy_maps):
                    energy_vis = energy_map[i, 0].cpu().numpy()
                    axes[1, 1 + j].hist(energy_vis.flatten(), bins=50, color='orange', alpha=0.7)
                    axes[1, 1 + j].set_title(f'Energy Scale {j+1}\nDistribution', fontsize=10)
                    axes[1, 1 + j].set_xlabel('Value')
                    axes[1, 1 + j].set_ylabel('Frequency')

                # 融合图分布
                axes[1, -1].hist(final_vis.flatten(), bins=50, color='green', alpha=0.7)
                axes[1, -1].set_title('Final Fused\nDistribution', fontsize=10)
                axes[1, -1].set_xlabel('Value')
                axes[1, -1].set_ylabel('Frequency')

                # 如果有ground truth，在右下角显示
                if gt is not None:
                    axes[1, 0].clear()
                    axes[1, 0].imshow(gt, cmap='gray')
                    axes[1, 0].set_title('Ground Truth', fontsize=10, fontweight='bold')
                    axes[1, 0].axis('off')

                plt.suptitle(f'Recon vs Energy Comparison - Sample {cnt} (Epoch {epochs})',
                            fontsize=14, fontweight='bold', y=1.02)
                plt.tight_layout()

                # 保存图片
                filename = f'{cnt:03d}_{type_name}.png'
                save_path = os.path.join(result_path, filename)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                cnt += 1

                # 打印统计信息
                print(f"Sample {cnt}:")
                print(f"  Recon Map   - Min: {recon_vis.min():.4f}, Max: {recon_vis.max():.4f}, Mean: {recon_vis.mean():.4f}")
                for j, energy_map in enumerate(energy_maps):
                    e = energy_map[i, 0].cpu().numpy()
                    print(f"  Energy S{j+1} ({spatial_sizes[j]}x{spatial_sizes[j]}) - Min: {e.min():.4f}, Max: {e.max():.4f}, Mean: {e.mean():.4f}")
                print(f"  Final Fuse  - Min: {final_vis.min():.4f}, Max: {final_vis.max():.4f}, Mean: {final_vis.mean():.4f}")
                print("-" * 60)

    print(f"\nVisualization saved to: {result_path}")
    print(f"Total samples visualized: {cnt}")


def visualize_multi_scale_energy(encoder, discriminator, dataloader, args, device, epochs):
    """
    专门可视化多尺度能量图的对比（独立版本，独立计算缓存）

    显示：
    1. 原始图像
    2. 各尺度能量图（插值到原图大小后）
    3. 各尺度能量图叠加在原图上
    """
    encoder.eval()
    discriminator.eval()

    with torch.no_grad():
        for data in dataloader:
            imgs = data[0].to(device)
            inputs = encoder(imgs)
            labels = data[-1]

            # 获取原始图像（反变换）
            img_transform = transforms.Compose([
                transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
                                   std=(1/0.229, 1/0.224, 1/0.225))
            ])
            imgs_denorm = img_transform(imgs)

            # 计算多尺度能量图
            energy_maps = cal_energy_map(discriminator, inputs, imgs.shape[-1])

            # 获取spatial_sizes
            if hasattr(discriminator, 'get_spatial_sizes'):
                spatial_sizes = discriminator.get_spatial_sizes()
            else:
                spatial_sizes = [64, 32, 16]

            for i in range(imgs.size(0)):
                img = imgs_denorm[i].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                label = labels[i]

                if hasattr(dataloader.dataset, 'types_set'):
                    type_name = dataloader.dataset.types_set[label.item()]
                else:
                    type_name = f"label_{label.item()}"

                # 创建子图
                n_scales = len(energy_maps)
                fig, axes = plt.subplots(2, n_scales + 1, figsize=(5 * (n_scales + 1), 10))

                # 第一行：各尺度能量图热力图
                axes[0, 0].imshow(img)
                axes[0, 0].set_title(f'Original\n{type_name}', fontsize=10, fontweight='bold')
                axes[0, 0].axis('off')

                for j, energy_map in enumerate(energy_maps):
                    e = energy_map[i, 0].cpu().numpy()
                    im = axes[0, j + 1].imshow(e, cmap='jet', vmin=0, vmax=e.max())
                    axes[0, j + 1].set_title(f'Energy Scale {j+1}\n{spatial_sizes[j]}x{spatial_sizes[j]}', fontsize=10, fontweight='bold')
                    axes[0, j + 1].axis('off')
                    plt.colorbar(im, ax=axes[0, j + 1], shrink=0.8)

                # 第二行：能量图叠加效果
                axes[1, 0].imshow(img)
                axes[1, 0].set_title('Original', fontsize=10, fontweight='bold')
                axes[1, 0].axis('off')

                for j, energy_map in enumerate(energy_maps):
                    e = energy_map[i, 0].cpu().numpy()
                    axes[1, j + 1].imshow(img)
                    im = axes[1, j + 1].imshow(e, cmap='jet', alpha=0.6)
                    axes[1, j + 1].set_title(f'Overlay Scale {j+1}', fontsize=10, fontweight='bold')
                    axes[1, j + 1].axis('off')

                plt.suptitle(f'Multi-Scale Energy Maps - Sample (Epoch {epochs})',
                           fontsize=14, fontweight='bold')
                plt.tight_layout()

                result_path = './results/{}_{}_multiscale_energy_epoch_{}'.format(
                    args.dataset, args.normal, epochs)
                if not os.path.exists(result_path):
                    os.makedirs(result_path, exist_ok=True)

                filename = f'{type_name}_multi_scale.png'
                save_path = os.path.join(result_path, filename)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                break  # 只可视化一个batch

    print(f"Multi-scale energy visualization saved to: {result_path}")


def compare_recon_energy_statistics(encoder, ed, discriminator, dataloader, args, device):
    """
    统计对比重建误差图和能量图的数值特征（独立版本，独立计算缓存）

    输出：
    1. 各图的值域、均值、方差
    2. 重建误差 vs 能量的相关系数
    3. 像素级别的差异图
    """
    encoder.eval()
    ed.eval()
    discriminator.eval()

    stats = {
        'recon': {'values': [], 'means': [], 'stds': [], 'maxs': []},
        'energy': {'values': [], 'means': [], 'stds': [], 'maxs': []},
        'correlation': []
    }

    with torch.no_grad():
        for data in dataloader:
            imgs = data[0].to(device)
            inputs = encoder(imgs)
            outputs = ed(inputs)

            # 计算重建误差图
            recon_map = cal_anomaly_map(inputs, outputs, imgs.shape[-1], amap_mode='add')

            # 计算能量图（平均多尺度）
            energy_maps = cal_energy_map(discriminator, inputs, imgs.shape[-1])
            energy_maps_np = [em.cpu().numpy() for em in energy_maps]
            energy_avg = np.mean(np.stack([em.squeeze(1) for em in energy_maps_np]), axis=0)

            # 收集统计信息
            for i in range(recon_map.shape[0]):
                r = recon_map[i]
                e = energy_avg[i]

                stats['recon']['values'].extend(r.flatten())
                stats['recon']['means'].append(r.mean())
                stats['recon']['stds'].append(r.std())
                stats['recon']['maxs'].append(r.max())

                stats['energy']['values'].extend(e.flatten())
                stats['energy']['means'].append(e.mean())
                stats['energy']['stds'].append(e.std())
                stats['energy']['maxs'].append(e.max())

                # 计算相关系数
                corr = np.corrcoef(r.flatten(), e.flatten())[0, 1]
                stats['correlation'].append(corr)

    # 打印汇总统计
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

    return stats
