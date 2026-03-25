# utils/test.py
import torch
import numpy as np
from dataset.mvtec import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve
import cv2
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from scipy.ndimage import gaussian_filter
from statistics import mean
import os
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model.model import Discriminator

def transform_invert(img_, transform_train):
    """
    reverse transfrom 
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None]) 
    return img_


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    batch = 1 if len(fs_list[0].size()) == 2 else fs_list[0].size(0)
    if amap_mode == 'mul':
        anomaly_map = torch.ones([batch, 1, out_size, out_size], device=fs_list[0].device)
    else:
        anomaly_map = torch.zeros([batch, 1, out_size, out_size], device=fs_list[0].device)
    a_map_list = []
    for i in range(len(ft_list)):
        
        fs = fs_list[i]
        ft = ft_list[i]
        
        a_map = 1 - F.cosine_similarity(fs, ft) # a_map: batch * H * W
        a_map = torch.unsqueeze(a_map, dim=1) # a_map: batch(1)  * 1  * H * W
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        elif amap_mode == 'max':
            anomaly_map = torch.max(anomaly_map, a_map)
        else:
            anomaly_map += a_map
    anomaly_map = anomaly_map.cpu().numpy()
    anomaly_map_list = []
    for i in range(len(anomaly_map)):
        amap = gaussian_filter(anomaly_map[i], sigma=4)
        anomaly_map_list.append(amap)
    anomaly_map = np.vstack(anomaly_map_list)
    return anomaly_map


def cal_energy_map(discriminator, feat_list, out_size):
    """
    生成判别器的能量图（与异常图对应）

    参数:
        discriminator: 训练好的判别器模型
        feat_list: 特征图列表 [x1, x2, x3]，例如 inputs 或 outputs
        out_size: 输出图的尺寸（通常为原始图像尺寸）

    返回:
        energy_maps: 能量图列表 [[N,1,H,W], [N,1,H,W], [N,1,H,W]]
                     每个元素对应一个尺度的能量图，已经上采样到 out_size
    """
    scores_list = discriminator(feat_list)  # list: [N, Hi*Wi] x 3

    # 从判别器获取实际的空间尺寸
    if hasattr(discriminator, 'get_spatial_sizes'):
        spatial_sizes = discriminator.get_spatial_sizes()
    else:
        # 兼容旧版本：使用默认尺寸
        spatial_sizes = [64, 32, 16]

    energy_maps = []

    for i, score in enumerate(scores_list):
        # 1) 取绝对值，与训练时的计算一致
        s = score.abs()  # [N, Hi*Wi]

        # 2) 获取批次大小和空间尺寸
        N = s.shape[0]
        Hi = Wi = spatial_sizes[i]

        # 3) reshape 回空间图 [N, 1, Hi, Wi]
        s = s.view(N, 1, Hi, Wi)

        # 4) 上采样到原图大小
        s = F.interpolate(s, size=out_size, mode="bilinear", align_corners=True)

        energy_maps.append(s)

    return energy_maps  # 返回列表，每个元素是 [N, 1, H, W]

def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def minmax_norm_per_sample(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    x: [N, 1, H, W] or [N, H, W]
    对每张图单独做min-max归一化，避免不同batch/不同类尺度差异太大
    """
    if x.ndim == 4:
        x_ = x[:, 0]  # [N,H,W]
    else:
        x_ = x
    N = x_.shape[0]
    out = np.zeros_like(x_, dtype=np.float32)
    for i in range(N):
        a_min = x_[i].min()
        a_max = x_[i].max()
        out[i] = (x_[i] - a_min) / (a_max - a_min + eps)
    return out  # [N,H,W]


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def soft_gate_fuse(recon_map: np.ndarray,
                   energy_maps_t: list,
                   k: float = 10.0,
                   Te: float = 0.5,
                   smooth_sigma: float = 0.0) -> np.ndarray:
    """
    软门控融合机制：将重建误差图与多尺度能量图融合

    融合公式: final = recon_map * sigmoid(k * (energy_norm - Te))

    参数:
        recon_map: np.ndarray [N,H,W] 重建误差图（cal_anomaly_map输出）
        energy_maps_t: list of torch.Tensor [[N,1,H,W], [N,1,H,W], [N,1,H,W]]
                       多尺度能量图列表（cal_energy_map输出）
        k: float = 10.0 门控曲线的陡峭程度
        Te: float = 0.5 能量图的阈值（建议做per-sample minmax归一化后使用）
        smooth_sigma: float = 0.0 高斯平滑的sigma值，0表示不平滑

    返回:
        final_map: np.ndarray [N,H,W] 融合后的异常图
    """
    # 1) 将所有能量图转换并拼接
    energy_maps_np = []
    for energy_map_t in energy_maps_t:
        energy_map = energy_map_t.detach().cpu().numpy()  # [N,1,H,W]
        energy_maps_np.append(energy_map)

    # 2) 堆叠成 [N, 3, H, W] 然后 squeeze 成 [N, H, W]
    energy_stacked = np.stack([em.squeeze(1) for em in energy_maps_np], axis=1)  # [N, 3, H, W]
    energy_stacked = energy_stacked.transpose(0, 2, 3, 1)  # [N, H, W, 3] for per-sample processing

    # 3) 对每个样本、每个尺度的能量图进行 per-sample minmax 归一化
    energy_norm = np.zeros_like(energy_stacked)  # [N, H, W, 3]
    for n in range(energy_stacked.shape[0]):
        for c in range(energy_stacked.shape[3]):
            em = energy_stacked[n, :, :, c]
            em_min, em_max = em.min(), em.max()
            if em_max - em_min > 1e-12:
                energy_norm[n, :, :, c] = (em - em_min) / (em_max - em_min)
            else:
                energy_norm[n, :, :, c] = em

    energy_norm = energy_norm.transpose(0, 3, 1, 2)  # [N, 3, H, W]

    # 4) 可选：在门控之前对能量图进行高斯平滑
    if smooth_sigma and smooth_sigma > 0:
        for c in range(energy_norm.shape[1]):
            energy_norm[:, c] = np.stack([
                gaussian_filter(energy_norm[n, c], sigma=smooth_sigma)
                for n in range(energy_norm.shape[0])
            ], axis=0)

    # 5) 对多尺度能量图进行平均融合
    energy_avg = energy_norm.mean(axis=1)  # [N, H, W]

    # 6) 计算门控权重 g in (0,1)
    gate = sigmoid_np(k * (energy_avg - Te))  # [N,H,W]

    # 7) 融合：final = recon * gate
    final = recon_map.astype(np.float32) * gate.astype(np.float32)

    return final


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def calculate_metrics(scores, labels, acc=True):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
    best_threshold = thresholds[np.argmax(f1_scores)]
    binary_predictions = np.where(scores >= best_threshold, 1, 0)

    TP = np.sum((binary_predictions == 1) & (labels == 1))
    TN = np.sum((binary_predictions == 0) & (labels == 0))
    FP = np.sum((binary_predictions == 1) & (labels == 0))
    FN = np.sum((binary_predictions == 0) & (labels == 1))
    ACC = (TP + TN) / (TP + TN + FP + FN)
    auroc_score = roc_auc_score(labels, scores)
    if acc:
        res = {
            'AUROC': auroc_score,
            'F1': np.max(f1_scores),
            'ACC': ACC,
        }
    else:
        res = {
            'AUROC': auroc_score,
        }
    return res

def evaluation(encoder, ed, discriminator, dataloader, device, args, return_maps=False):
    """
    统一评估接口

    参数:
        encoder: 预训练特征提取器
        ed: 编码器-解码器模型
        discriminator: 训练好的判别器模型（用于软门控融合）
        dataloader: 数据加载器
        device: 计算设备
        args: 命令行参数
        return_maps: bool，是否返回中间 map（用于避免可视化时的重复计算）

    返回:
        metrics: 评估指标字典
        若 return_maps=True，额外返回 (recon_maps, energy_maps, final_maps, all_gts, anomaly_maps)
    """
    if args.dataset in ['mvtec', 'visa', 'btad']:
        return evaluation_pixel(encoder, ed, discriminator, dataloader, device, args, return_maps)
    else:
        # semantic 模式不使用判别器，仅返回 metrics（不支持 return_maps）
        result = evaluation_semantic(encoder, ed, dataloader, device, args)
        if return_maps:
            return result, None, None, None, None, None
        return result

def evaluation_semantic(encoder, ed, dataloader, device, args):
    encoder.eval()
    ed.eval()
    gt_list = []
    sample_score_list = []
    metric_dict = {}
    
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = ed(inputs)
            gt_list.append(label != int(args.normal))
            anomaly_map = cal_anomaly_map(inputs, outputs, out_size=img.size(-1), amap_mode='add')
            if args.dataset in ['isic']:
                fea_score = anomaly_map.reshape(img.size(0), -1).mean(axis=-1)
            else:
                fea_score = torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)), args.topk, dim=-1)[0].numpy().mean(axis=-1)
            sample_score_list.append(fea_score)
        gt_list = torch.cat(gt_list).cpu().numpy()
        sample_score_list = np.concatenate(sample_score_list)
        metric_dict['Image'] = calculate_metrics(sample_score_list, gt_list)
    return metric_dict

def evaluation_pixel(encoder, ed, discriminator, dataloader, device, args, return_maps=False):
    """
    像素级异常检测评估

    参数:
        encoder: 预训练特征提取器
        ed: 编码器-解码器模型
        discriminator: 训练好的判别器模型
        dataloader: 数据加载器
        device: 计算设备
        args: 命令行参数
        return_maps: bool，是否返回所有批次的中间 map（避免可视化时重复计算）

    返回:
        metrics: 评估指标字典
        若 return_maps=True，额外返回 (recon_maps, energy_maps_list, final_maps, all_gts)
        - recon_maps: np.ndarray [N_total, H, W]，重建误差图
        - energy_maps_list: list of torch.Tensor [[N,1,H,W], ...]，每尺度一个，共 3 个尺度
        - final_maps: np.ndarray [N_total, H, W]，融合后的异常图
        - all_gts: np.ndarray [N_total, H, W]，ground truth mask
    """
    encoder.eval()
    ed.eval()

    # 如果传入了判别器，设置其为评估模式
    if discriminator is not None:
        discriminator.eval()
    pixel_gt_list = []
    pixel_score_list = []
    sample_gt_list = []
    sample_score_list = []
    aupro_list = []
    all_gts = []
    all_maps = []

    # 仅在需要返回 map 时才收集
    if return_maps:
        all_recon_maps = []
        all_energy_maps = []   # 每 batch 一个 torch.Tensor [N, 1, H, W]
        all_final_maps = []
        all_anomaly_maps = []

    with torch.no_grad():
        for img, gt, label in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = ed(inputs)
            gt = gt.squeeze(1)
            # 软门控机制的加入
            recon_map = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='add')
            energy_maps = cal_energy_map(discriminator, inputs, img.shape[-1])
            final_map = soft_gate_fuse(
                recon_map=recon_map,
                energy_maps_t=energy_maps,
                k=args.gate_k,
                Te=args.gate_te,
                smooth_sigma=args.gate_sigma
            )
            anomaly_map = final_map
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            # 收集用于返回 map
            if return_maps:
                all_recon_maps.append(recon_map)
                # energy_maps: list of [N,1,H,W]，对每个尺度逐个拼接
                if all_energy_maps:
                    for j in range(len(energy_maps)):
                        all_energy_maps[j] = torch.cat([all_energy_maps[j], energy_maps[j]], dim=0)
                else:
                    # 第一次：初始化各尺度的累积张量
                    all_energy_maps = [em.clone() for em in energy_maps]
                all_final_maps.append(final_map)
                all_anomaly_maps.append(anomaly_map)

            all_gts.append(gt.cpu().numpy())
            all_maps.append(anomaly_map)
            pixel_gt_list.append(gt.cpu().numpy().astype(int).reshape(-1))  # 扁平向量，一维
            pixel_score_list.append(anomaly_map.reshape(-1))
            sample_gt_list.append(np.max(gt.reshape(gt.size(0), -1).cpu().numpy().astype(int), axis=-1))
            sample_score = torch.topk(torch.from_numpy(anomaly_map.reshape(img.size(0), -1)), args.topk, dim=-1)[0].numpy().mean(axis=-1)
            sample_score_list.append(sample_score)
            label = gt.reshape(gt.shape[0], -1).max(axis=-1)[0]
            if len(gt[label.bool()]) > 0:
                anomaly_map_filtered = anomaly_map[label.bool()]
                gt_filtered = gt[label.bool()]
                for am, g in zip(anomaly_map_filtered, gt_filtered):
                    aupro_list.append(compute_pro(g.unsqueeze(dim=0).cpu().numpy().astype(int), am.reshape(1, *am.shape)))

        pixel_gt_list = np.concatenate(pixel_gt_list).reshape(-1)
        pixel_score_list = np.concatenate(pixel_score_list).reshape(-1)
        sample_gt_list = np.concatenate(sample_gt_list)
        sample_score_list = np.concatenate(sample_score_list)
        pixel_aupro = round(np.mean(aupro_list), 6)
        all_gts = np.concatenate(all_gts)
        all_maps = np.concatenate(all_maps)
        metrics = {}
        metrics['Pixel'] = calculate_metrics(pixel_score_list, pixel_gt_list, False)
        metrics['Pixel']['PRO'] = pixel_aupro
        metrics['Image'] = calculate_metrics(sample_score_list, sample_gt_list, True)

    if return_maps:
        recon_maps = np.concatenate(all_recon_maps, axis=0)
        final_maps = np.concatenate(all_final_maps, axis=0)
        anomaly_maps = np.concatenate(all_anomaly_maps, axis=0)
        # energy_maps 已在循环中拼接，all_energy_maps[j] 就是第 j 尺度的完整 torch.Tensor
        return metrics, recon_maps, all_energy_maps, final_maps, all_gts, anomaly_maps

    return metrics

# def visualize(pfe, ae, dataloader: MVTecDataset, args, transform, device, postfix=""):
#     pfe.eval()
#     ae.eval()
#     with torch.no_grad():
#         cnt = 0
#         for data in dataloader:
#             imgs = data[0].to(device)
#             inputs = pfe(imgs)
#             outputs = ae(inputs)
#             labels = data[-1]
#             anomaly_maps = cal_anomaly_map(inputs, outputs, imgs.shape[-1], amap_mode='a')
            
#             imgs = transform_invert(imgs, transform)
            
#             if len(data) == 3:
#                 gts = data[1].squeeze(1)
#                 pack = zip(imgs, anomaly_maps, gts, labels)
#             else:
#                 pack = zip(imgs, anomaly_maps, labels)
#             for p in pack:
#                 ano_map = cvt2heatmap((p[1] / 2) * 255)
#                 img = cv2.cvtColor((p[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
#                 result_path = './results/' + args.dataset +'_' + args.normal + postfix
#                 if not os.path.exists(result_path):
#                     os.makedirs(result_path)
#                 if len(data) == 3:
#                     gt = cv2.cvtColor((p[2].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
#                     res = np.vstack((img, gt, ano_map))
#                 else:
#                     res = np.vstack((img, ano_map))
#                 cv2.imwrite(result_path + '/' + str(cnt) + '_' + str(p[-1].item()) + '.png', res)
#                 cnt += 1


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def visualize_anomaly_maps_simple(pfe, ae, dataloader, args, device, epochs,
                                  anomaly_maps=None):
    """
    使用matplotlib进行异常检测热力图可视化（不依赖OpenCV）
    支持类型平衡采样，每个异常类型选择固定数量的样本

    参数:
        anomaly_maps: np.ndarray [N, H, W] 或 None。若为 None，则在函数内部重新计算前向。
    """
    pfe.eval()
    ae.eval()

    with torch.no_grad():
        cnt = 0
        global_offset = 0  # 跟踪缓存 anomaly_maps 中的全局样本索引

        for data in dataloader:
            imgs = data[0].to(device)
            labels = data[-1]
            batch_size = imgs.size(0)

            # 仅在未传入缓存时才重新计算前向
            if anomaly_maps is None:
                inputs = pfe(imgs)
                outputs = ae(inputs)
                batch_anomaly_maps = cal_anomaly_map(inputs, outputs, imgs.shape[-1], amap_mode='add')

            # 反变换图像
            img_transform = transforms.Compose([
                transforms.Normalize(mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
                                   std=(1/0.229, 1/0.224, 1/0.225))
            ])
            imgs_denorm = img_transform(imgs)

            # 创建结果目录
            result_path = './results/{}_{}_final_epoch_{}'.format(args.dataset, args.normal, epochs)
            if not os.path.exists(result_path):
                os.makedirs(result_path, exist_ok=True)

            for i, (img, label) in enumerate(zip(imgs_denorm, labels)):
                # 获取当前样本的异常图：缓存优先，否则按需计算
                if anomaly_maps is not None:
                    amap = anomaly_maps[global_offset + i]
                else:
                    amap = batch_anomaly_maps[i]
                amap = amap.squeeze()

                # 转换为numpy数组
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)  # 确保值在[0,1]范围内

                # 如果有ground truth
                if len(data) >= 3:
                    gt = data[1][i].squeeze(0).cpu().numpy()
                else:
                    gt = None

                # 获取图像的异常类型名称（如果可用）
                if hasattr(dataloader.dataset, 'types_set'):
                    type_name = dataloader.dataset.types_set[label.item()]
                else:
                    type_name = f"label_{label.item()}"

                # 创建统一的子图布局 - 所有图像使用相同的尺寸和布局
                if gt is not None:
                    # 有ground truth：2x2网格布局
                    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                    axes = axes.flatten()
                else:
                    # 无ground truth：1x3布局
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes = axes.flatten()

                # 设置统一的图像显示参数
                img_height, img_width = img_np.shape[:2]

                # 1. 显示原始图像 - 确保尺寸正确
                axes[0].imshow(img_np, aspect='equal')
                axes[0].set_title(f'Original Image\nType: {type_name}', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                # 设置坐标轴范围确保图像不变形
                axes[0].set_xlim(0, img_width)
                axes[0].set_ylim(img_height, 0)

                # 2. 显示异常热力图 - 与原始图像尺寸完全匹配
                im = axes[1].imshow(amap, cmap='jet', aspect='equal',
                                   extent=[0, img_width, img_height, 0])
                axes[1].set_title('Anomaly Map', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                # 添加颜色条
                cbar = plt.colorbar(im, ax=axes[1], shrink=0.8, aspect=20)
                cbar.ax.tick_params(labelsize=10)

                # 3. 显示ground truth（如果有的话）
                if gt is not None:
                    axes[2].imshow(gt, cmap='gray', aspect='equal',
                                  extent=[0, img_width, img_height, 0])
                    axes[2].set_title('Ground Truth', fontsize=12, fontweight='bold')
                    axes[2].axis('off')

                    # 4. 显示叠加效果（原始图像 + 异常热力图）- 确保完全重合
                    axes[3].imshow(img_np, aspect='equal',
                                  extent=[0, img_width, img_height, 0])
                    axes[3].imshow(amap, cmap='jet', alpha=0.6, aspect='equal',
                                  extent=[0, img_width, img_height, 0])
                    axes[3].set_title('Overlay (Original + Anomaly)', fontsize=12, fontweight='bold')
                    axes[3].axis('off')
                else:
                    # 3. 显示叠加效果（原始图像 + 异常热力图）- 确保完全重合
                    axes[2].imshow(img_np, aspect='equal',
                                  extent=[0, img_width, img_height, 0])
                    axes[2].imshow(amap, cmap='jet', alpha=0.6, aspect='equal',
                                  extent=[0, img_width, img_height, 0])
                    axes[2].set_title('Overlay (Original + Anomaly)', fontsize=12, fontweight='bold')
                    axes[2].axis('off')

                # 调整布局，确保所有子图大小一致
                plt.tight_layout(pad=2.0, h_pad=1.0, w_pad=1.0)

                # 保存图像 - 使用更具描述性的文件名
                filename = f'{cnt:03d}_{type_name}.png'
                save_path = os.path.join(result_path, filename)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

                cnt += 1
            # 每个 batch 结束后更新全局偏移量
            global_offset += batch_size
