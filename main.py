import torch
import numpy as np
import random
import os
import csv
from util.test import evaluation, evaluation_pixel  # , visualize  # [VIS-DISABLED] 可视化函数导入
from model.model import PretrainedFeatureExtractor, ED, Discriminator
from model.rrs import RRS
from model.perlin_anomaly import PerlinAnomalyGenerator, MultiScaleAnomalyGenerator
from model.pixel_anomaly import PixelAnomalyGenerator
from model.anomaly_controller import UnifiedAnomalyController
from model.afs import AFS_Adapted
# =====================================================================
# 【FeatureAdapter 缝合点 ①】-- 导入
# 取消下面的注释以启用 FeatureAdapter（AFS → AE 之间插入通道适配）：
# from model.feature_adapter import FeatureAdapter
# =====================================================================
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm
# [VIS-DISABLED] matplotlib导入及后端设置
# import matplotlib
# matplotlib.use('Agg')  # 设置非GUI后端，避免WSL图形界面问题
# import matplotlib.pyplot as plt


def parse_args():
    
    parser = ArgumentParser(description='Pytorch implemention of Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning')

    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset')
    
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    parser.add_argument('--lr', type=float, default=0.005, help='learning tate')

    parser.add_argument('--epochs', type=int, default=200, help='training epoch')
    
    parser.add_argument('--img_size', type=int, default=32, help='img size')
    
    parser.add_argument('--normal', type=str, default='carpet', help='normal class')
    
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=0, help='labeled anomaly class num')
    
    parser.add_argument('--labeled_anomaly_class', type=int, default=1, help='labeled anomaly class')
    
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.0, help='labeled anomaly ratio')
    
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')

    parser.add_argument('--exp_name', type=str, default='default', help='experiment name for logs and checkpoints')

    parser.add_argument('--ckpt_dir', type=str, default='/hy-tmp/checkpoints/', help='checkpoint root dir')

    parser.add_argument('--save_best', action='store_true', help='save best checkpoint during evaluation')
    
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet152'])
    
    parser.add_argument('--eval_epoch', type=int, default=1, help='every eval_epoch to eval')
    
    parser.add_argument('--layer', nargs='+', type=int, default=[2], help='choose pretrain resnet layer to reconstruct')
    
    parser.add_argument('--d_lr', type=float, default=1e-05, help='discriminator learning rate')
    
    parser.add_argument('--adv_conf', type=float, default=0.02, help='adversial loss conf')

    parser.add_argument('--topk', type=int, default=100, help='calculate topk values')

    parser.add_argument('--eval_anomaly_map_source', type=str, default='recon',
                        choices=['recon', 'rrs', 'rrs_cos'],
                        help='anomaly map source for evaluation: recon uses AE reconstruction residual, rrs uses RRS anomaly_score, rrs_cos uses RRS-selected channels with cosine residual')

    # [VIS-DISABLED] 评估可视化相关参数
    # parser.add_argument('--eval_visualize', action='store_true', help='whether to visualize anomaly maps during evaluation')
    # parser.add_argument('--eval_viz_samples', type=int, default=5, help='number of samples to visualize during evaluation')
    # parser.add_argument('--eval_viz_freq', type=int, default=1, help='frequency of evaluation visualization (every N eval_epochs)')

    parser.add_argument('--recon_loss_type', type=str, default='cosine', choices=['cosine', 'mse', 'perceptual', 'ssim', 'combined'], help='reconstruction loss type')
    parser.add_argument('--loss_alpha', type=float, default=0.7, help='weight for cosine loss in combined loss')
    parser.add_argument('--loss_beta', type=float, default=0.2, help='weight for pixel loss in combined loss')
    parser.add_argument('--loss_gamma', type=float, default=0.1, help='weight for structure loss in combined loss')

    # RRS module arguments
    parser.add_argument('--use_rrs', action='store_true', help='enable RRS module for anomaly segmentation')
    parser.add_argument('--rrs_loss_weight', type=float, default=1.0, help='weight for RRS segmentation loss')
    parser.add_argument('--rrs_anomaly_samples', type=int, default=None, help='max number of anomaly samples to use for RRS training (None=all)')
    parser.add_argument('--rrs_stop_grad', action='store_true', help='stop gradient from RRS to AE')
    parser.add_argument('--rrs_lr', type=float, default=1e-3, help='RRS learning rate')

    # AFS (Anomaly-aware Feature Selection) module
    parser.add_argument('--use_afs', action='store_true',
                        help='enable AFS (Anomaly-aware Feature Selection)')
    parser.add_argument('--afs_select_planes', nargs='+', type=int, default=None,
                        help='output channels (with expansion) for AFS, '
                             'e.g. [256, 512] for layers=[2,3] with wide_resnet50_2')
    parser.add_argument('--afs_init_bsn', type=int, default=50,
                        help='number of batches for AFS initialization')

    # Synthetic anomaly generation (lightweight Perlin-based)
    parser.add_argument('--use_synthetic_anomaly', action='store_true',
                        help='use Perlin noise feature perturbation instead of loaded anomaly images')
    parser.add_argument('--anomaly_ratio', type=float, default=0.3,
                        help='anomaly area ratio for synthetic anomaly (0~1)')
    parser.add_argument('--anomaly_perturbation', type=str, default='noise',
                        choices=['noise', 'shuffle', 'erase', 'simplenet_noise', 'hard_erase'],
                        help='feature perturbation type')
    parser.add_argument('--anomaly_noise_std', type=float, default=0.15,
                        help='noise std for synthetic anomaly perturbation')
    parser.add_argument('--anomaly_mix_noise', type=int, default=1,
                        help='noise levels for simplenet_noise perturbation '
                             '(noise std scaled by 1.1^k per level, samples pick one randomly)')
    parser.add_argument('--multi_scale_anomaly', action='store_true',
                        help='randomly vary anomaly ratio during training for diversity')

    # Pixel-level anomaly synthesis (PatchGuard + OCR-GAN)
    parser.add_argument('--use_pixel_anomaly', action='store_true',
                        help='use pixel-level anomaly synthesis (PatchGuard/CutPaste/Cutout)')
    parser.add_argument('--pixel_anomaly_mode', type=str, default='mixed',
                        choices=['patchguard', 'cutpaste', 'cutout', 'mixed'],
                        help='pixel anomaly mode')
    parser.add_argument('--pixel_anomaly_prob', type=float, default=0.4,
                        help='probability of pixel anomaly per sample')
    parser.add_argument('--perlin_anomaly_prob', type=float, default=0.35,
                        help='probability of Perlin anomaly per sample')
    parser.add_argument('--anomaly_strategy', type=str, default='prob',
                        choices=['prob', 'adapt', 'progressive'],
                        help='sample selection strategy for anomaly synthesis')
    parser.add_argument('--pixel_patchguard_prob', type=float, default=0.5,
                        help='within pixel mixed mode, patchguard probability')

    # Encoder 第二分支增强对齐
    parser.add_argument('--enable_enhancement', action='store_true',
                        help='enable boundary-preserving alignment for the second branch of encoder')
    parser.add_argument('--pixel_cutpaste_prob', type=float, default=0.3,
                        help='within pixel mixed mode, cutpaste probability')
    parser.add_argument('--pixel_cutout_prob', type=float, default=0.2,
                        help='within pixel mixed mode, cutout probability')
    parser.add_argument('--pixel_max_attempts', type=int, default=50,
                        help='max attempts for PatchGuard coordinate sampling')

    # =================================================================
    # 【FeatureAdapter 缝合点 ②】-- 参数解析
    # 取消下面的注释以启用 FeatureAdapter：
    # parser.add_argument('--use_feature_adapter', action='store_true',
    #                     help='enable FeatureAdapter between AFS and AE')
    # parser.add_argument('--feature_adapter_layers', type=int, default=1,
    #                     help='number of Linear layers per scale in FeatureAdapter')
    # =================================================================
    
    return parser.parse_args()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def get_res_str(metrics):
    score_res_str = ""
    for key, value in metrics.items():
        for item, v in value.items():
            score_res_str += "{}_{}: {:.6f} ".format(key, item, v) 
    return score_res_str


def get_best_score(metrics):
    image_auroc = metrics.get('Image', {}).get('AUROC', 0.0)
    pixel_auroc = metrics.get('Pixel', {}).get('AUROC', 0.0)
    pixel_pro = metrics.get('Pixel', {}).get('PRO', 0.0)
    return image_auroc + pixel_auroc + pixel_pro


def get_ckpt_dir(args):
    return os.path.join(args.ckpt_dir, args.exp_name, args.dataset,
                        args.normal, 'seed_{}'.format(args.seed))


def save_checkpoint(path, epoch, args, metrics, score, pfe, ae, discriminator,
                    afs=None, rrs=None, ae_optimizer=None,
                    discriminator_optimizer=None, rrs_optimizer=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'args': vars(args),
        'metrics': metrics,
        'best_score': score,
        'pfe': pfe.state_dict(),
        'ae': ae.state_dict(),
        'discriminator': discriminator.state_dict(),
        'afs': afs.state_dict() if afs is not None else None,
        'rrs': rrs.state_dict() if rrs is not None else None,
        'ae_optimizer': ae_optimizer.state_dict() if ae_optimizer is not None else None,
        'discriminator_optimizer': discriminator_optimizer.state_dict() if discriminator_optimizer is not None else None,
        'rrs_optimizer': rrs_optimizer.state_dict() if rrs_optimizer is not None else None,
    }
    torch.save(checkpoint, path)


def append_eval_csv(path, epoch, metrics, score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    row = {
        'epoch': epoch,
        'image_auroc': metrics.get('Image', {}).get('AUROC', ''),
        'image_f1': metrics.get('Image', {}).get('F1', ''),
        'image_acc': metrics.get('Image', {}).get('ACC', ''),
        'pixel_auroc': metrics.get('Pixel', {}).get('AUROC', ''),
        'pixel_pro': metrics.get('Pixel', {}).get('PRO', ''),
        'score': score,
    }
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def loss_function(a, b, loss_type='cosine', alpha=0.7, beta=0.2, gamma=0.1):
    """
    结构感知重建损失函数

    Args:
        a: 原始特征列表
        b: 重建特征列表
        loss_type: 损失类型 ('cosine', 'mse', 'perceptual', 'ssim', 'combined')
        alpha: 余弦相似度损失权重
        beta: 像素级损失权重
        gamma: 结构损失权重
    """
    if loss_type == 'cosine':
        # 原始余弦相似度损失
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            loss += torch.mean(1-cos_loss(a[item].view(b[item].shape[0], -1),
                                          b[item].view(b[item].shape[0], -1)))
        return loss

    elif loss_type == 'mse':
        # 均方误差损失 - 像素级重建
        mse_loss = torch.nn.MSELoss()
        loss = 0
        for item in range(len(a)):
            loss += mse_loss(a[item], b[item])
        return loss

    elif loss_type == 'perceptual':
        # 感知损失 - 考虑高层语义特征
        loss = 0
        for item in range(len(a)):
            # L1距离在特征空间
            loss += torch.mean(torch.abs(a[item] - b[item]))
        return loss

    elif loss_type == 'ssim':
        # SSIM结构相似性损失
        loss = 0
        for item in range(len(a)):
            # 简化的SSIM计算（实际应用中建议使用pytorch-msssim库）
            mu_a = torch.mean(a[item], dim=[2, 3], keepdim=True)
            mu_b = torch.mean(b[item], dim=[2, 3], keepdim=True)
            sigma_a = torch.var(a[item], dim=[2, 3], keepdim=True)
            sigma_b = torch.var(b[item], dim=[2, 3], keepdim=True)
            sigma_ab = torch.mean((a[item] - mu_a) * (b[item] - mu_b), dim=[2, 3], keepdim=True)

            c1 = 0.01 ** 2
            c2 = 0.03 ** 2

            numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
            denominator = (mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2)
            ssim = numerator / denominator

            loss += torch.mean(1 - ssim)
        return loss

    elif loss_type == 'combined':
        # 组合损失：余弦 + MSE + 结构感知
        cos_loss = torch.nn.CosineSimilarity()
        mse_loss = torch.nn.MSELoss()

        total_loss = 0
        for item in range(len(a)):
            # 余弦相似度损失（特征级）
            cosine_loss = torch.mean(1 - cos_loss(a[item].view(b[item].shape[0], -1),
                                                 b[item].view(b[item].shape[0], -1)))

            # MSE损失（像素级）
            mse = mse_loss(a[item], b[item])

            # 梯度损失（结构感知）
            grad_loss = gradient_loss(a[item], b[item])

            # 加权组合
            combined = alpha * cosine_loss + beta * mse + gamma * grad_loss
            total_loss += combined

        return total_loss

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def gradient_loss(a, b):
    """
    梯度损失：感知图像结构变化
    """
    def gradient_x(img):
        return torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])

    def gradient_y(img):
        return torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

    grad_a_x = gradient_x(a)
    grad_a_y = gradient_y(a)
    grad_b_x = gradient_x(b)
    grad_b_y = gradient_y(b)

    loss_x = torch.nn.functional.mse_loss(grad_a_x, grad_b_x)
    loss_y = torch.nn.functional.mse_loss(grad_a_y, grad_b_y)

    return (loss_x + loss_y) / 2


# [VIS-DISABLED] visualize_evaluation_anomaly_maps() 函数
# def visualize_evaluation_anomaly_maps(pfe, ae, dataloader, args, device, epoch, phase="test", afs=None):
#     """
#     在评估时可视化anomaly maps，用于监控训练过程中的异常检测效果
#
#     Args:
#         pfe: 预训练特征提取器
#         ae: 自编码器
#         dataloader: 数据加载器
#         args: 参数
#         device: 设备
#         epoch: 当前epoch
#         phase: 阶段名称 ("valid" 或 "test")
#     """
#     import matplotlib.pyplot as plt
#     from util.test import cal_anomaly_map
#
#     pfe.eval()
#     ae.eval()
#
#     # 创建保存目录
#     result_path = f'./eval_results/{args.dataset}_{args.normal}_epoch_{epoch}_{phase}'
#     os.makedirs(result_path, exist_ok=True)
#
#     sample_count = 0
#     normal_count = 0
#     abnormal_count = 0
#
#     with torch.no_grad():
#         for batch_data in dataloader:
#             if len(batch_data) == 3:
#                 imgs, gts, labels = batch_data
#             else:
#                 imgs, labels = batch_data
#                 gts = None
#
#             # 只可视化指定数量的样本，且保持正常/异常样本的平衡
#             batch_normal_count = (labels == 0).sum().item()
#             batch_abnormal_count = (labels > 0).sum().item()
#
#             if normal_count >= args.eval_viz_samples // 2 and batch_normal_count > 0:
#                 # 如果正常样本已经够了，跳过包含正常样本的batch
#                 continue
#             if abnormal_count >= args.eval_viz_samples // 2 and batch_abnormal_count > 0:
#                 # 如果异常样本已经够了，跳过包含异常样本的batch
#                 continue
#
#             imgs = imgs.to(device)
#             inputs_raw = pfe(imgs)
#             if afs is not None:
#                 inputs = afs(inputs_raw)
#             else:
#                 inputs = inputs_raw
#             outputs = ae(inputs)
#
#             # 计算anomaly map
#             anomaly_maps = cal_anomaly_map(inputs, outputs, imgs.shape[-1], amap_mode='add')
#
#             # 反归一化图像以便显示
#             imgs_np = imgs.cpu().numpy()
#             mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
#             std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
#             imgs_np = imgs_np * std + mean
#             imgs_np = np.clip(imgs_np, 0, 1)
#
#             for i in range(len(imgs)):
#                 current_label = labels[i].item()
#
#                 # 检查是否还需要这种类型的样本
#                 if current_label == 0 and normal_count >= args.eval_viz_samples // 2:
#                     continue
#                 if current_label > 0 and abnormal_count >= args.eval_viz_samples // 2:
#                     continue
#
#                 fig, axes = plt.subplots(1, 3 if gts is not None else 2, figsize=(12, 4))
#
#                 # 原始图像
#                 img_display = np.transpose(imgs_np[i], (1, 2, 0))
#                 axes[0].imshow(img_display)
#                 axes[0].set_title(f'Original Image\nLabel: {"Abnormal" if current_label > 0 else "Normal"}')
#                 axes[0].axis('off')
#
#                 # Anomaly Map
#                 if len(anomaly_maps.shape) == 4:  # (batch, 1, H, W)
#                     anomaly_map = anomaly_maps[i, 0]  # 取第一个通道
#                 elif len(anomaly_maps.shape) == 3:  # (batch, H, W)
#                     anomaly_map = anomaly_maps[i]
#                 elif len(anomaly_maps.shape) == 2 and anomaly_maps.shape[0] > 1:  # (batch, features) - 多个样本的特征
#                     # 这种情况通常是(batch, H*W)，需要reshape每个样本
#                     H = int(np.sqrt(anomaly_maps.shape[1]))  # 假设是正方形
#                     if H * H == anomaly_maps.shape[1]:  # 确保是完美正方形
#                         anomaly_map = anomaly_maps[i].reshape(H, H)
#                     else:
#                         # 如果不是完美正方形，保持为一维
#                         anomaly_map = anomaly_maps[i]
#                 else:  # 其他情况，包括单一样本的一维数组
#                     anomaly_map = anomaly_maps[i]
#
#                 # 确保是二维数组用于imshow
#                 if anomaly_map.ndim == 1:
#                     # 如果是一维数组，将其reshape为二维用于可视化
#                     total_size = len(anomaly_map)
#                     # 找到最接近的正方形尺寸
#                     size = int(np.sqrt(total_size))
#                     if size * size > total_size:
#                         size -= 1
#
#                     target_size = size * size
#                     if target_size <= total_size:
#                         # 截断到正方形
#                         anomaly_map = anomaly_map[:target_size].reshape(size, size)
#                     else:
#                         # 不应该发生，但为了安全
#                         anomaly_map = anomaly_map.reshape(-1, 1)  # 保持为一维但作为列向量
#                 elif anomaly_map.ndim > 2:
#                     # 如果是三维或更高维，压缩到二维
#                     anomaly_map = anomaly_map.squeeze()
#                     # 如果压缩后还是一维，reshape为二维
#                     if anomaly_map.ndim == 1:
#                         size = int(np.sqrt(len(anomaly_map)))
#                         if size * size <= len(anomaly_map):
#                             anomaly_map = anomaly_map[:size*size].reshape(size, size)
#                         else:
#                             anomaly_map = anomaly_map.reshape(-1, 1)
#
#                 im = axes[1].imshow(anomaly_map, cmap='jet', vmin=0, vmax=anomaly_map.max())
#                 axes[1].set_title('Anomaly Map')
#                 axes[1].axis('off')
#
#                 # 添加colorbar
#                 plt.colorbar(im, ax=axes[1], shrink=0.8)
#
#                 # Ground Truth (如果有的话)
#                 if gts is not None:
#                     gt = gts[i].squeeze().cpu().numpy()
#                     axes[2].imshow(gt, cmap='gray')
#                     axes[2].set_title('Ground Truth')
#                     axes[2].axis('off')
#
#                 plt.tight_layout()
#                 sample_type = "normal" if current_label == 0 else "abnormal"
#                 save_path = os.path.join(result_path, f'sample_{sample_count:02d}_{sample_type}_label_{current_label}.png')
#                 plt.savefig(save_path, dpi=150, bbox_inches='tight')
#                 plt.close()
#
#                 sample_count += 1
#                 if current_label == 0:
#                     normal_count += 1
#                 else:
#                     abnormal_count += 1
#
#                 # 检查是否已经收集了足够的样本
#                 if sample_count >= args.eval_viz_samples:
#                     break
#
#             if sample_count >= args.eval_viz_samples:
#                 break
#
#     print(f"Evaluation visualization saved to: {result_path} ({sample_count} samples)")


# [VIS-DISABLED] visualize_anomaly_maps_simple() 函数
# def visualize_anomaly_maps_simple(pfe, ae, dataloader, args, device, epochs, afs=None):
#     """
#     使用matplotlib进行anomaly map可视化的简化版本
#     不依赖opencv，使用numpy和matplotlib
#     """
#     import matplotlib.pyplot as plt
#     from util.test import cal_anomaly_map
#
#     pfe.eval()
#     ae.eval()
#
#     result_path = './results/{}_{}_final_epoch_{}'.format(args.dataset, args.normal, epochs)
#     os.makedirs(result_path, exist_ok=True)
#
#     with torch.no_grad():
#         cnt = 0
#         for batch_data in dataloader:
#             if len(batch_data) == 3:
#                 imgs, gts, labels = batch_data
#             else:
#                 imgs, labels = batch_data
#                 gts = None
#
#             imgs = imgs.to(device)
#             inputs_raw = pfe(imgs)
#             if afs is not None:
#                 inputs = afs(inputs_raw)
#             else:
#                 inputs = inputs_raw
#             outputs = ae(inputs)
#
#             # 计算anomaly map
#             anomaly_maps = cal_anomaly_map(inputs, outputs, imgs.shape[-1], amap_mode='add')
#
#             # 反归一化图像以便显示
#             imgs_np = imgs.cpu().numpy()
#             mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
#             std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
#             imgs_np = imgs_np * std + mean
#             imgs_np = np.clip(imgs_np, 0, 1)
#
#             for i in range(len(imgs)):
#                 fig, axes = plt.subplots(1, 3 if gts is not None else 2, figsize=(12, 4))
#
#                 # 原始图像
#                 img_display = np.transpose(imgs_np[i], (1, 2, 0))
#                 axes[0].imshow(img_display)
#                 axes[0].set_title(f'Original Image\nLabel: {"Abnormal" if labels[i] > 0 else "Normal"}')
#                 axes[0].axis('off')
#
#                 # Anomaly Map
#                 anomaly_map = anomaly_maps[i, 0] if len(anomaly_maps.shape) > 2 else anomaly_maps[i]
#                 axes[1].imshow(anomaly_map, cmap='jet', vmin=0, vmax=anomaly_maps.max())
#                 axes[1].set_title('Anomaly Map')
#                 axes[1].axis('off')
#
#                 # Ground Truth (如果有的话)
#                 if gts is not None:
#                     gt = gts[i].squeeze().cpu().numpy()
#                     axes[2].imshow(gt, cmap='gray')
#                     axes[2].set_title('Ground Truth')
#                     axes[2].axis('off')
#
#                 plt.tight_layout()
#                 save_path = os.path.join(result_path, f'sample_{cnt:03d}_label_{labels[i].item()}.png')
#                 plt.savefig(save_path, dpi=150, bbox_inches='tight')
#                 plt.close()
#
#                 cnt += 1
#                 if cnt >= 10:  # 限制可视化数量
#                     break
#             if cnt >= 10:
#                 break


# [VIS-DISABLED] loss_draw() 函数
# def loss_draw(loss_history, save_path=None):
#     """
#     绘制损失曲线。
#     - 横轴：epoch
#     - 纵轴：不同损失值
#     - 布局：动态网格
#     - 比例尺较大：调整为较大的画布和线宽
#     """
#     if not loss_history:
#         print("Warning: loss_history is empty, skipping plot generation")
#         return
#
#     try:
#         items = list(loss_history.items())
#         n = len(items)
#         if n == 0:
#             return
#
#         # 动态计算子图布局
#         ncols = 2
#         nrows = (n + ncols - 1) // ncols
#
#         fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6 * nrows))
#         if nrows * ncols == 1:
#             axes = [axes]
#         else:
#             axes = axes.ravel()
#
#         for idx, (name, values) in enumerate(items):
#             ax = axes[idx]
#             if values and len(values) > 0:
#                 epochs = range(1, len(values) + 1)
#                 ax.plot(epochs, values, marker='o', linewidth=3, markersize=5, color='blue')
#                 ax.set_xlabel('Epoch', fontsize=12)
#                 ax.set_ylabel(name, fontsize=12)
#                 ax.set_title(f'{name} vs Epoch', fontsize=14, fontweight='bold')
#                 ax.grid(True, linestyle='--', alpha=0.7)
#                 ax.tick_params(axis='both', which='major', labelsize=10)
#                 ax.margins(x=0.05, y=0.1)
#             else:
#                 ax.set_title(f"{name}\n(No data)", fontsize=14)
#                 ax.axis('off')
#
#         # 隐藏多余的子图
#         for idx in range(len(items), len(axes)):
#             axes[idx].axis('off')
#
#         plt.tight_layout(pad=3.0)
#
#         if save_path:
#             # 确保目录存在
#             save_dir = os.path.dirname(save_path)
#             if save_dir and not os.path.exists(save_dir):
#                 os.makedirs(save_dir, exist_ok=True)
#
#             # 保存为jpg格式
#             plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
#             print(f"Loss curve saved to: {save_path}")
#         else:
#             # 在无图形界面环境中，不显示图片，直接跳过
#             print("Warning: No save path provided, skipping plot display in headless environment")
#
#         plt.close()
#
#     except Exception as e:
#         print(f"Error generating loss plot: {e}")
#         plt.close()

def train(args):
    """
    CKAAD:整个模型的训练过程
    
    训练流程概述:
    1. 准备数据：正常样本和异常样本
    2. 特征提取：使用预训练模型提取多层级特征
    3. 自编码器训练：重建正常样本特征
    4. 判别器训练：区分正常特征、异常特征和重建特征
    5. 对抗训练：使重建特征更接近正常特征
    """
    # 1.设置日志的目录和文件名，并打印日志
    log_dir = os.path.join(args.log_dir, args.exp_name,
                           "lan{:.2f}_acn{}".format(args.labeled_anomaly_ratio,  args.labeled_anomaly_class_num),
                           args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger_filename = os.path.join(log_dir, 'n_{}_a_{}_s_{}'.format(args.normal, args.labeled_anomaly_class, args.seed) + '.txt')
    logger = get_logger(logger_filename)
    logger.info("log file: {}".format(logger_filename))
    logger.info("class: {}".format(args.normal))
    ckpt_dir = get_ckpt_dir(args)
    csv_path = os.path.join(ckpt_dir, 'eval_metrics.csv')
    logger.info("checkpoint dir: {}".format(ckpt_dir))
    logger.info("eval csv: {}".format(csv_path))
    
    print_args(logger, args)
    epochs = args.epochs
    batch_size = args.batch_size
    best_score = float('-inf')
    best_epoch = 0
    best_metrics = None
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {}".format(device))

    # 2.加载数据集，获取数据加载器
    dataset = OODDataSet(root='./data', dataset=args.dataset, image_size=args.img_size, category=args.normal,
                         labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                         labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                         labeled_anomaly_class=args.labeled_anomaly_class,
                         use_rrs=args.use_rrs,
                         rrs_anomaly_samples=args.rrs_anomaly_samples,
                         return_foreground_mask=args.use_pixel_anomaly)
    train_dataloader, valid_dataloader, anomaly_dataloader, test_dataloader, rrs_dataloader = dataset.get_data_loader(batch_size=batch_size)

    # 3.初始化模型
    # 3.1 初始化预训练特征提取器，冻结参数
    # 根据args.layer参数选择提取哪几层特征，例如[1,2,3]表示提取ResNet的第1、2、3层特征
    pfe = PretrainedFeatureExtractor(args.model, layers=args.layer, image_size=args.img_size).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)  # 冻结特征提取器参数
    pfe.eval()  # 设置为评估模式
    
    # 3.2 初始化AFS模块（可选）
    # AFS在PFE输出上进行通道选择，使后续所有模块使用精简后的通道数
    afs = None
    pfe_output_channels_base = pfe.output_channels  # 默认与PFE一致
    
    # 3.3 初始化轻量级合成异常生成器（可选，替代加载的真实异常图像）
    perlin_gen = None
    if args.use_synthetic_anomaly:
        if args.multi_scale_anomaly:
            perlin_gen = MultiScaleAnomalyGenerator(
                perturbation=args.anomaly_perturbation,
                noise_std=args.anomaly_noise_std,
                mix_noise=args.anomaly_mix_noise,
            )
        else:
            perlin_gen = PerlinAnomalyGenerator(
                anomaly_ratio=args.anomaly_ratio,
                perturbation=args.anomaly_perturbation,
                noise_std=args.anomaly_noise_std,
                mix_noise=args.anomaly_mix_noise,
            )
        logger.info("Synthetic anomaly generator initialized: type=%s, ratio=%.2f, std=%.3f, mix_noise=%d%s",
                     args.anomaly_perturbation, args.anomaly_ratio,
                     args.anomaly_noise_std, args.anomaly_mix_noise,
                     " (multi-scale)" if args.multi_scale_anomaly else "")
    
    if args.use_afs:
        # AFS输入通道 = PFE实际输出通道（含expansion）
        afs_in_channels = [c * pfe.expansion for c in pfe.output_channels]
        
        # 如果未指定select_planes，默认减半
        if args.afs_select_planes is None:
            afs_select_planes = [c // 2 for c in afs_in_channels]
        else:
            assert len(args.afs_select_planes) == len(afs_in_channels), \
                f"afs_select_planes长度 {len(args.afs_select_planes)} != " \
                f"特征层数 {len(afs_in_channels)}"
            afs_select_planes = args.afs_select_planes
        
        afs = AFS_Adapted(afs_in_channels, afs_select_planes).to(device)
        logger.info(f"AFS initialized: in_channels={afs_in_channels}, "
                    f"select_planes={afs_select_planes}")
        
        # AFS初始化需要PerlinAnomalyGenerator在特征空间合成异常
        # 如果用户未启用use_synthetic_anomaly，创建一个默认实例用于AFS初始化
        if not args.use_synthetic_anomaly or perlin_gen is None:
            init_perlin = PerlinAnomalyGenerator(
                anomaly_ratio=0.3, perturbation='noise', noise_std=0.15,
                mix_noise=args.anomaly_mix_noise
            )
            logger.info("AFS init: created default PerlinAnomalyGenerator for initialization")
        else:
            init_perlin = perlin_gen
        
        # 执行AFS通道索引初始化
        afs.init_idxs(pfe, init_perlin, train_dataloader,
                      args.afs_init_bsn, device)
        
        # 打印选中的通道索引
        for i in range(afs.num_layers):
            idx_list = afs.indexes[f"layer_{i}"].data.cpu().tolist()
            logger.info(f"AFS layer_{i}: selected {len(idx_list)}/"
                        f"{afs_in_channels[i]} channels, indices={idx_list}")
        
        # 下游模块使用AFS缩减后的基础通道数
        pfe_output_channels_base = [s // pfe.expansion for s in afs_select_planes]
        logger.info(f"AFS: downstream base channels changed from "
                    f"{pfe.output_channels} to {pfe_output_channels_base}")
    
    # 3.3 初始化编码器-解码器(自编码器)
    # 如果有AFS，输入通道数为AFS缩减后的基础通道数
    ae = ED(backbone=args.model, input_channels=pfe_output_channels_base,
            enable_enhancement=args.enable_enhancement).to(device)
    
    # 3.4 初始化判别器
    # input_sizes: 各层特征图的空间尺寸（AFS不改变空间尺寸）
    # input_channels: 各层特征图的基础通道数（AFS后可能减半）
    discriminator = Discriminator(
        input_sizes=pfe.output_sizes,
        input_channels=pfe_output_channels_base,
        expansion=pfe.expansion
    ).to(device)

    # 3.4 初始化像素级异常生成器（可选）
    pixel_gen = None
    if args.use_pixel_anomaly:
        pixel_gen = PixelAnomalyGenerator(
            dataset=args.dataset,
            class_name=args.normal,
            mode=args.pixel_anomaly_mode,
            patchguard_prob=args.pixel_patchguard_prob,
            cutpaste_prob=args.pixel_cutpaste_prob,
            cutout_prob=args.pixel_cutout_prob,
            max_attempts=args.pixel_max_attempts,
        )
        logger.info("Pixel anomaly generator initialized: mode=%s, pg=%.2f cp=%.2f co=%.2f",
                     args.pixel_anomaly_mode, args.pixel_patchguard_prob,
                     args.pixel_cutpaste_prob, args.pixel_cutout_prob)

    # 3.7 初始化异常合成控制器（可选）
    anomaly_controller = None
    if args.use_pixel_anomaly and args.use_synthetic_anomaly:
        anomaly_controller = UnifiedAnomalyController(
            args, args.epochs, dataset=args.dataset,
            class_name=args.normal, logger=logger,
        )
        anomaly_controller.set_pixel_generator(pixel_gen)
        anomaly_controller.set_perlin_generator(perlin_gen)
        logger.info("UnifiedAnomalyController initialized: strategy=%s, pixel=%.2f perlin=%.2f",
                     args.anomaly_strategy, args.pixel_anomaly_prob, args.perlin_anomaly_prob)

    # 3.6 初始化RRS模块（可选）
    rrs = None
    rrs_optimizer = None
    if args.use_rrs:
        # 如果有AFS，RRS使用AFS实际输出通道数（含expansion）
        if args.use_afs:
            rrs_layer_channels = afs_select_planes
            logger.info("RRS using AFS-reduced channels: {}".format(rrs_layer_channels))
        else:
            rrs_layer_channels = [c * pfe.expansion for c in pfe.output_channels]
        rrs_layer_strides = [args.img_size // s for s in pfe.output_sizes]
        rrs = RRS(
            layer_channels=rrs_layer_channels,
            layer_strides=rrs_layer_strides,
            modes=['max', 'mean'],
            mode_numbers=None,  # auto: min(total//2, 256) per mode
            num_residual_layers=2,
            stop_grad=args.rrs_stop_grad,
        ).to(device)
        rrs_optimizer = torch.optim.Adam(rrs.parameters(), lr=args.rrs_lr, betas=(0.5, 0.999))
        logger.info("RRS module initialized: channels={}, strides={}".format(rrs_layer_channels, rrs_layer_strides))
        logger.info("RRS mode_numbers={}, total_select={}".format(rrs.mode_numbers, rrs.total_select_number))

    # =================================================================
    # 【FeatureAdapter 缝合点 ③】-- 初始化与优化器
    # 取消下面的注释以启用 FeatureAdapter：
    #
    # if args.use_feature_adapter:
    #     # FeatureAdapter 输入通道 = AFS 实际输出通道（含 expansion）
    #     if args.use_afs:
    #         fa_in_channels = afs.select_planes_list  # AFS 缩减后的通道
    #         logger.info("FeatureAdapter using AFS-reduced channels: {}".format(fa_in_channels))
    #     else:
    #         fa_in_channels = [c * pfe.expansion for c in pfe.output_channels]
    #     feature_adapter = FeatureAdapter(fa_in_channels, n_layers=args.feature_adapter_layers).to(device)
    #     fa_optimizer = torch.optim.Adam(feature_adapter.parameters(), lr=args.lr, betas=(0.5, 0.999))
    #     logger.info("FeatureAdapter initialized: channels={}, layers={}".format(
    #         fa_in_channels, args.feature_adapter_layers))
    # else:
    #     feature_adapter = None
    #
    # 取消到上一行注释为止
    # =================================================================

    # 3.7 初始化优化器
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    # ----------------------------------------------------------------
    # 【FeatureAdapter 优化器续】-- 如有需要，在 ae_optimizer 后加入：
    # if args.use_feature_adapter:
    #     fa_optimizer = torch.optim.Adam(feature_adapter.parameters(), lr=args.lr, betas=(0.5, 0.999))
    #     logger.info("FeatureAdapter optimizer created (lr={})".format(args.lr))
    # ----------------------------------------------------------------
    
    # 设置权重系数和标签
    gamma = 0.5  # 控制重建特征损失的权重
    true_label = 0  # 正常样本的标签
    fake_label = 1  # 异常样本的标签

    # [VIS-DISABLED] 记录各类损失用于绘图
    # loss_history = {
    #     "dis_loss": [],
    #     "recon_loss": [],
    #     "adv_loss": [],
    #     "ae_loss": [],
    #     "seg_loss": [],
    # }
    
    # 4.开始训练循环
    # 准备 RRS 数据迭代器
    rrs_data_iter = None
    if rrs_dataloader is not None:
        rrs_data_iter = cycle(rrs_dataloader)

    for epoch in range(1, epochs+1):
        ae.train()
        discriminator.train()
        if rrs is not None:
            rrs.train()
        dis_loss_list = []
        recon_loss_list = []
        adv_loss_list = []
        ae_loss_list = []
        seg_loss_list = []
        
        # 使用zip和cycle将正常数据和异常数据配对
        # cycle确保异常数据可以循环使用，即使异常数据少于正常数据
        for normal, anomaly in tqdm.tqdm(zip(train_dataloader, cycle(anomaly_dataloader))):
            # 5.1 准备输入数据
            normal_img = normal[0].to(device)  # 正常图像: [batch_size, 3, img_size, img_size]

            # 提取 foreground_mask（如果可用）
            # 注意：当 labeled_anomaly_ratio>0 时，数据集被 AnomalyDataset 包装，
            # 返回的 4 个元素为 (img, anomaly_img, anomaly_gt, anomaly_label)，
            # 此时 normal[3] 不是 fg_mask，应跳过
            foreground_masks = None
            if (args.use_pixel_anomaly and len(normal) >= 4
                    and args.labeled_anomaly_ratio <= 0):
                foreground_masks = normal[3].to(device)

            # 5.2 特征提取（正常样本）：PFE → [AFS] → AE
            # PFE输出原始多尺度特征
            normal_raw = pfe(normal_img)

            if anomaly_controller is not None:
                # === 控制器模式：混合像素级 + 特征级异常合成 ===
                anomaly_features, anomaly_masks, anomaly_size = anomaly_controller(
                    normal_img, normal_raw, foreground_masks, pfe, epoch
                )

                # 经AFS通道筛选（如果启用）
                if afs is not None:
                    normal_inputs = afs(normal_raw)
                    anomaly_inputs = afs(anomaly_features) if anomaly_size > 0 else None
                else:
                    normal_inputs = normal_raw
                    anomaly_inputs = anomaly_features if anomaly_size > 0 else None

                # -------------------------------------------------------
                # 【FeatureAdapter 缝合点 ④-a】-- 合成异常模式：AFS → FeatureAdapter → AE
                # 取消下面注释以在 AFS 和 AE 之间插入 FeatureAdapter：
                # if feature_adapter is not None:
                #     normal_inputs = feature_adapter(normal_inputs)
                #     anomaly_inputs = feature_adapter(anomaly_inputs)
                # -------------------------------------------------------

                # AE重建
                normal_outputs = ae(normal_inputs)
                if anomaly_size > 0:
                    anomaly_outputs = ae(anomaly_inputs)
                    outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
                else:
                    outputs = normal_outputs
                anomaly_img = normal_img  # 占位，后续RRS使用自己的dataloader

            elif args.use_synthetic_anomaly and perlin_gen is not None:
                # === 合成异常模式：在PFE特征空间生成异常，再经AFS筛选 ===
                # 先由PerlinAnomalyGenerator在原始PFE特征上生成合成异常
                anomaly_raw, anomaly_masks = perlin_gen(normal_raw)
                anomaly_size = normal_img.size(0)

                # 再经AFS通道筛选（如果启用）
                if afs is not None:
                    normal_inputs = afs(normal_raw)
                    anomaly_inputs = afs(anomaly_raw)
                else:
                    normal_inputs = normal_raw
                    anomaly_inputs = anomaly_raw

                # -------------------------------------------------------
                # 【FeatureAdapter 缝合点 ④-a】-- 合成异常模式：AFS → FeatureAdapter → AE
                # 取消下面注释以在 AFS 和 AE 之间插入 FeatureAdapter：
                # if feature_adapter is not None:
                #     normal_inputs = feature_adapter(normal_inputs)
                #     anomaly_inputs = feature_adapter(anomaly_inputs)
                # -------------------------------------------------------

                # AE重建
                normal_outputs = ae(normal_inputs)
                anomaly_outputs = ae(anomaly_inputs)
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
                anomaly_img = normal_img  # 占位，后续RRS使用自己的dataloader

            elif args.use_pixel_anomaly and pixel_gen is not None:
                # === 纯像素级异常模式（无Perlin）：在像素空间生成异常，再经PFE+AFS ===
                anomaly_size = normal_img.size(0)
                anomaly_imgs, pixel_masks = pixel_gen(normal_img, foreground_masks)
                anomaly_raw = pfe(anomaly_imgs)

                if afs is not None:
                    normal_inputs = afs(normal_raw)
                    anomaly_inputs = afs(anomaly_raw)
                else:
                    normal_inputs = normal_raw
                    anomaly_inputs = anomaly_raw

                normal_outputs = ae(normal_inputs)
                anomaly_outputs = ae(anomaly_inputs)
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
                anomaly_img = normal_img
            else:
                # === 原始模式：从数据加载器加载真实异常图像 ===
                if anomaly is not None:
                    anomaly_img = anomaly[0].to(device)
                elif args.dataset in ['mvtec', 'visa', 'btad'] and len(normal) == 4:
                    anomaly_img = normal[1].to(device)
                else:
                    anomaly_img = normal_img[:0]  # 创建空张量

                anomaly_size = anomaly_img.size(0)
                
                # 正常特征经AFS（如果启用）
                if afs is not None:
                    normal_inputs = afs(normal_raw)
                else:
                    normal_inputs = normal_raw

                # -------------------------------------------------------
                # 【FeatureAdapter 缝合点 ④-b】-- 原始模式正常分支：AFS → FeatureAdapter → AE
                # 取消下面注释以在 AFS 和 AE 之间插入 FeatureAdapter：
                # if feature_adapter is not None:
                #     normal_inputs = feature_adapter(normal_inputs)
                # -------------------------------------------------------

                # 正常特征重建
                normal_outputs = ae(normal_inputs)

                if anomaly_size > 0:
                    # 异常特征也经AFS筛选
                    anomaly_raw = pfe(anomaly_img)
                    if afs is not None:
                        anomaly_inputs = afs(anomaly_raw)
                    else:
                        anomaly_inputs = anomaly_raw

                    # -------------------------------------------------------
                    # 【FeatureAdapter 缝合点 ④-c】-- 原始模式异常分支：AFS → FeatureAdapter → AE
                    # 取消下面注释以在 AFS 和 AE 之间插入 FeatureAdapter：
                    # if feature_adapter is not None:
                    #     anomaly_inputs = feature_adapter(anomaly_inputs)
                    # -------------------------------------------------------

                    anomaly_outputs = ae(anomaly_inputs)
                    outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
                else:
                    outputs = normal_outputs
                
            # 5.3 初始化损失值
            dis_loss = torch.tensor(0.0).to(device)
            adv_loss = torch.tensor(0.0).to(device)
            
            # 5.4 分离特征图的梯度，准备训练判别器
            # 分离正常样本特征的梯度
            normal_inputs_detach = [i.detach() for i in normal_inputs]  # 形状与normal_inputs相同，但不计算梯度
            
            # 如果有异常样本，分离异常样本特征的梯度
            if anomaly_size > 0:
                anomaly_inputs_detach = [i.detach() for i in anomaly_inputs]  # 形状与anomaly_inputs相同，但不计算梯度
                
            # 分离重建特征的梯度
            outputs_detach = [o.detach() for o in outputs]  # 形状与outputs相同，但不计算梯度
            
            # 5.5 训练判别器
            if anomaly_size > 0:
                # 判别器损失由三部分组成:
                # 1. 正常样本特征应被判为真(标签0)
                # 2. 异常样本特征应被判为假(标签1)，权重为(1-gamma)
                # 3. 重建特征应被判为假(标签1)，权重为gamma
                dis_loss = discriminator.calculate_loss(normal_inputs_detach, true_label) + \
                          (1 - gamma) * discriminator.calculate_loss(anomaly_inputs_detach, fake_label) + \
                          gamma * discriminator.calculate_loss(outputs_detach, fake_label)
                
                # 更新判别器参数
                discriminator_optimizer.zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                discriminator_optimizer.step()
                    
                # 5.6 计算对抗损失
                # 希望重建特征能够欺骗判别器，被判为真(标签0)
                adv_loss = discriminator.calculate_loss(outputs, true_label)
            
            # 5.7 计算重建损失和自编码器总损失
            # 重建损失使用余弦相似度，衡量正常样本特征和重建特征的相似程度
            recon_loss = loss_function(normal_inputs, normal_outputs,
                                      loss_type=args.recon_loss_type,
                                      alpha=args.loss_alpha,
                                      beta=args.loss_beta,
                                      gamma=args.loss_gamma)
            
            # 自编码器总损失 = 重建损失 + 对抗损失*权重
            ae_loss = recon_loss + args.adv_conf * adv_loss
            
            # 更新自编码器参数
            ae_optimizer.zero_grad()
            ae_loss.backward()
            ae_optimizer.step()

            # 5.9 RRS训练：用异常图像+GT mask训练RRS分割
            seg_loss = torch.tensor(0.0).to(device)
            if rrs is not None:
                # =============================================================
                # 分支 A：用真实异常图像的 GT mask 训练 RRS（原有，保留）
                # 数据来源：rrs_dataloader（测试集真实异常，rrs_anomaly_samples 张）
                # 使用真实 GT mask 提供强监督信号
                # =============================================================
                if rrs_data_iter is not None:
                    try:
                        rrs_batch = next(rrs_data_iter)
                    except StopIteration:
                        rrs_data_iter = cycle(rrs_dataloader)
                        rrs_batch = next(rrs_data_iter)

                    rrs_anomaly_img = rrs_batch[0].to(device)  # anomaly image
                    rrs_gt_mask = rrs_batch[1].to(device)       # GT mask: [B, 1, H, W]

                    # 异常图像通过 PFE → [AFS] → AE（detach，RRS独立训练不影响AE）
                    with torch.no_grad():
                        rrs_raw = pfe(rrs_anomaly_img)
                        if afs is not None:
                            rrs_inputs = afs(rrs_raw)
                        else:
                            rrs_inputs = rrs_raw

                        # -------------------------------------------------------
                        # 【FeatureAdapter 缝合点 ④-d】-- RRS 分支：AFS → FeatureAdapter → AE
                        # 取消下面注释以在 AFS 和 AE 之间插入 FeatureAdapter：
                        # if feature_adapter is not None:
                        #     rrs_inputs = feature_adapter(rrs_inputs)
                        # -------------------------------------------------------

                        rrs_outputs = ae(rrs_inputs)

                    # RRS前向（异常图像）
                    rrs_out = rrs(rrs_inputs, rrs_outputs, image=rrs_anomaly_img)

                    # SegmentCrossEntropyLoss（异常区域）
                    logit = rrs_out['logit']  # [B, 2, H, W]
                    bsz = logit.size(0)
                    logit_flat = logit.view(bsz, 2, -1)  # [B, 2, H*W]
                    gt_flat = rrs_gt_mask.view(bsz, -1).long()  # [B, H*W]
                    seg_loss_anomaly = torch.nn.functional.cross_entropy(logit_flat, gt_flat)
                    seg_loss = seg_loss + seg_loss_anomaly

                # =============================================================
                # 分支 B（新增）：用当前 batch 的合成异常训练 RRS
                # 数据来源：anomaly_controller / perlin_gen / pixel_gen 每步生成的合成异常
                # 利用合成异常的 mask（Perlin mask / 像素级 mask）作为监督信号
                # 梯度完全隔离，不影响 AE
                # =============================================================
                if anomaly_size > 0:
                    # 检查当前 batch 是否在合成异常模式下（anomaly_masks 可用）
                    in_synthetic_mode = (
                        anomaly_controller is not None
                        or args.use_synthetic_anomaly
                        or args.use_pixel_anomaly
                    )

                    if in_synthetic_mode:
                        # 合成异常的 features 已由前面的分支计算好：
                        #   anomaly_inputs：PFE → [AFS] → anomaly features
                        #   anomaly_outputs：AE 重建
                        # 合成异常的 masks（各特征尺度，取第 0 层最高分辨率）：
                        #   分支 controller:   anomaly_masks[0] (来自 anomaly_masks_list_list)
                        #   分支 perlin:       anomaly_masks[0] (来自 perlin_gen)
                        #   分支 pixel:        pixel_masks       (来自 pixel_gen)

                        if args.use_pixel_anomaly and not args.use_synthetic_anomaly and \
                           anomaly_controller is None:
                            # 纯像素异常模式：pixel_masks 在 [B,1,H_img,W_img] 空间
                            syn_gt_mask = pixel_masks
                        else:
                            # 控制器 / Perlin 模式：anomaly_masks[0] 在特征空间尺度
                            syn_gt_mask = anomaly_masks[0]  # [B, 1, H_feat, W_feat]

                        if syn_gt_mask.size(0) != anomaly_inputs[0].size(0):
                            raise RuntimeError(
                                "RRS synthetic mask/feature batch mismatch: "
                                f"mask={syn_gt_mask.size(0)}, feature={anomaly_inputs[0].size(0)}"
                            )

                        # detach 确保 RRS 独立训练，不回传到 AE / 前序特征图。
                        anomaly_inputs_det = [ai.detach() for ai in anomaly_inputs]
                        anomaly_outputs_det = [ao.detach() for ao in anomaly_outputs]
                        syn_rrs_out = rrs(anomaly_inputs_det, anomaly_outputs_det, image=normal_img)
                        syn_logit = syn_rrs_out['logit']  # [B, 2, H_img, W_img]

                        # 将 syn_gt_mask resize 到 logit 的空间尺寸
                        _, _, lh, lw = syn_logit.shape
                        syn_gt_resized = torch.nn.functional.interpolate(
                            syn_gt_mask.float(), size=(lh, lw), mode='nearest'
                        )
                        syn_bsz = syn_logit.size(0)
                        syn_logit_flat = syn_logit.view(syn_bsz, 2, -1)
                        syn_gt_flat = syn_gt_resized.view(syn_bsz, -1).long()
                        seg_loss_syn = torch.nn.functional.cross_entropy(
                            syn_logit_flat, syn_gt_flat
                        )
                        seg_loss = seg_loss + seg_loss_syn

                # =============================================================
                # 分支 C：正常图像的 RRS 损失（全 0 mask，增强 RRS 对正常样本的抑制能力）
                # =============================================================
                normal_inputs_det = [ni.detach() for ni in normal_inputs]
                normal_outputs_det = [no.detach() for no in normal_outputs]
                normal_rrs_out = rrs(normal_inputs_det, normal_outputs_det, image=normal_img)
                normal_logit = normal_rrs_out['logit']
                normal_bsz = normal_logit.size(0)
                normal_logit_flat = normal_logit.view(normal_bsz, 2, -1)
                normal_gt_flat = torch.zeros(normal_bsz, normal_logit_flat.size(-1),
                                             dtype=torch.long, device=device)
                seg_loss_normal = torch.nn.functional.cross_entropy(normal_logit_flat, normal_gt_flat)
                seg_loss = seg_loss + seg_loss_normal

                # 更新RRS参数（所有分支的损失合并后一起 backward）
                rrs_optimizer.zero_grad()
                seg_loss.backward()
                torch.nn.utils.clip_grad_norm_(rrs.parameters(), 1.0)
                rrs_optimizer.step()

            # 5.8 记录各项损失值
            dis_loss_list.append(dis_loss.item())
            ae_loss_list.append(ae_loss.item())
            recon_loss_list.append(recon_loss.item())
            adv_loss_list.append(adv_loss.item())
            seg_loss_list.append(seg_loss.item())

        # 6. 打印当前epoch的训练损失并记录到历史
        epoch_dis = np.mean(dis_loss_list)
        epoch_recon = np.mean(recon_loss_list)
        epoch_adv = np.mean(adv_loss_list)
        epoch_ae = np.mean(ae_loss_list)
        epoch_seg = np.mean(seg_loss_list)

        logger.info("epoch [{}/{}], dis_loss: {:.6f}, recon_loss:{:.6f}, adv_loss:{:.6f}, ae_loss: {:.6f}, seg_loss: {:.6f}".format(
            epoch, epochs, epoch_dis, epoch_recon, epoch_adv, epoch_ae, epoch_seg))

        # [VIS-DISABLED] 记录损失历史用于绘图
        # loss_history["dis_loss"].append(epoch_dis)
        # loss_history["recon_loss"].append(epoch_recon)
        # loss_history["adv_loss"].append(epoch_adv)
        # loss_history["ae_loss"].append(epoch_ae)
        # loss_history["seg_loss"].append(epoch_seg)

        # 7. 定期评估模型性能
        # ---------------------------------------------------------------
        # 【FeatureAdapter 缝合点 ④-e】-- 评估时传入 feature_adapter
        # 如果启用了 FeatureAdapter，需修改 evaluation 和 visualize_evaluation_anomaly_maps
        # 调用，增加 feature_adapter 参数（需同时修改 util/test.py 中的相应函数）：
        #   valid_metrics = evaluation(pfe, ae, valid_dataloader, device, args,
        #                              afs=afs, feature_adapter=feature_adapter)
        #   metrics = evaluation(pfe, ae, test_dataloader, device, args,
        #                        afs=afs, feature_adapter=feature_adapter)
        #   visualize_evaluation_anomaly_maps(..., afs=afs, feature_adapter=feature_adapter)
        # ---------------------------------------------------------------
        if (epoch) % args.eval_epoch == 0:
            if valid_dataloader is not None:
                valid_metrics = evaluation(pfe, ae, valid_dataloader, device, args, afs=afs, rrs=rrs)
                valid_info = get_res_str(valid_metrics)
                logger.info("Valid: {}".format(valid_info))

                # [VIS-DISABLED] 评估时可视化anomaly map
                # if args.eval_visualize and (epoch // args.eval_epoch) % args.eval_viz_freq == 0:
                #     visualize_evaluation_anomaly_maps(pfe, ae, valid_dataloader, args, device, epoch, "valid", afs=afs)

            metrics = evaluation(pfe, ae, test_dataloader, device, args, afs=afs, rrs=rrs)
            infostr = get_res_str(metrics)
            logger.info("Test: {}".format(infostr))

            current_score = get_best_score(metrics)
            append_eval_csv(csv_path, epoch, metrics, current_score)
            logger.info("Checkpoint score: {:.6f} (best: {:.6f} @ epoch {})".format(
                current_score, best_score, best_epoch))
            if args.save_best and current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_metrics = metrics
                best_path = os.path.join(ckpt_dir, 'best.pth')
                save_checkpoint(
                    best_path, epoch, args, metrics, best_score,
                    pfe, ae, discriminator, afs=afs, rrs=rrs,
                    ae_optimizer=ae_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    rrs_optimizer=rrs_optimizer,
                )
                logger.info("Saved best checkpoint: {}".format(best_path))

            # [VIS-DISABLED] 评估时可视化anomaly map
            # if args.eval_visualize and (epoch // args.eval_epoch) % args.eval_viz_freq == 0:
            #     visualize_evaluation_anomaly_maps(pfe, ae, test_dataloader, args, device, epoch, "test", afs=afs)

        # 8. 周期性更新 AFS 通道索引（新增）
        # 原因：渐进式异常策略下异常分布动态变化，AFS 的通道选择会随时间次优化
        # 频率：每 20 个 epoch 重新初始化一次
        # 注意：AFS 索引是非可训练参数 (requires_grad=False)，重新赋值不影响训练图
        # ---------------------------------------------------------------
        if args.use_afs and epoch % 20 == 0 and epoch < epochs:
            # 获取用于 AFS 初始化的 Perlin 生成器
            if anomaly_controller is not None and anomaly_controller.perlin_gen is not None:
                afs_perlin = anomaly_controller.perlin_gen
            elif perlin_gen is not None:
                afs_perlin = perlin_gen
            else:
                # 如果当前没有活跃的 perlin 生成器，创建一个默认的
                from model.perlin_anomaly import PerlinAnomalyGenerator
                afs_perlin = PerlinAnomalyGenerator(
                    anomaly_ratio=args.anomaly_ratio,
                    perturbation=args.anomaly_perturbation,
                    noise_std=args.anomaly_noise_std,
                    mix_noise=args.anomaly_mix_noise,
                ).to(device)
                logger.info("AFS re-init: created temporary PerlinAnomalyGenerator")

            logger.info("AFS re-initializing at epoch %d...", epoch)
            afs.init_idxs(pfe, afs_perlin, train_dataloader, args.afs_init_bsn, device)

            # 打印更新后的通道索引
            for i in range(afs.num_layers):
                idx_list = afs.indexes[f"layer_{i}"].data.cpu().tolist()
                logger.info(f"AFS layer_{i} (re-init @ep{epoch}): "
                            f"selected {len(idx_list)}/{afs.in_channels_list[i]} channels")
        # ---------------------------------------------------------------

    if args.save_best:
        last_path = os.path.join(ckpt_dir, 'last.pth')
        save_checkpoint(
            last_path, epochs, args, best_metrics, best_score,
            pfe, ae, discriminator, afs=afs, rrs=rrs,
            ae_optimizer=ae_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            rrs_optimizer=rrs_optimizer,
        )
        logger.info("Saved last checkpoint: {}".format(last_path))

    # [VIS-DISABLED] 训练结束后保存最终损失曲线
    # try:
    #     pic_dir = "./pic/"
    #     if not os.path.exists(pic_dir):
    #         os.makedirs(pic_dir, exist_ok=True)
    #
    #     loss_img_name = "loss_curve_final_n_{}_a_{}_s_{}.jpg".format(args.normal, args.labeled_anomaly_class, args.seed)
    #     loss_save_path = os.path.join(pic_dir, loss_img_name)
    #
    #     # 检查损失历史记录是否为空
    #     if any(loss_history.values()):
    #         loss_draw(loss_history, loss_save_path)
    #         logger.info("Final loss curve saved to: {}".format(loss_save_path))
    #     else:
    #         logger.warning("Loss history is empty, skipping plot generation")
    # except Exception as e:
    #     logger.error("Failed to generate final loss curve: {}".format(str(e)))

    # [VIS-DISABLED] 训练结束后进行anomaly map可视化
    # try:
    #     logger.info("Starting anomaly map visualization...")
    #
    #     # 设置数据变换（与训练时相同）
    #     if args.dataset in ['mvtec', 'visa', 'btad']:
    #         img_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         ])
    #         gt_transform = transforms.Compose([transforms.ToTensor()])
    #     else:
    #         img_transform = transforms.Compose([
    #             transforms.Resize(args.img_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #         ])
    #         gt_transform = transforms.Compose([transforms.ToTensor()])
    #
    #     # 创建测试数据集（只可视化前几个样本以节省时间）
    #     if args.dataset == 'mvtec':
    #         from dataset.mvtec import MVTecDataset
    #         viz_dataset = MVTecDataset(root='./data', category=args.normal, train=False,
    #                                  transform=img_transform, gt_target_transform=gt_transform,
    #                                  img_size=args.img_size)
    #         # 只可视化前5个样本（包括正常和异常样本）
    #         viz_indices = []
    #         normal_count = 0
    #         abnormal_count = 0
    #         for i, target in enumerate(viz_dataset.targets):
    #             if target == 0 and normal_count < 3:  # 正常样本
    #                 viz_indices.append(i)
    #                 normal_count += 1
    #             elif target > 0 and abnormal_count < 2:  # 异常样本
    #                 viz_indices.append(i)
    #                 abnormal_count += 1
    #             if len(viz_indices) >= 5:
    #                 break
    #
    #         viz_dataset.data = viz_dataset.data[viz_indices]
    #         viz_dataset.targets = viz_dataset.targets[viz_indices]
    #         viz_dataset.gt_paths = [viz_dataset.gt_paths[i] for i in viz_indices]
    #
    #     viz_dataloader = torch.utils.data.DataLoader(viz_dataset, batch_size=4, shuffle=False)
    #
    #     # 使用简化的matplotlib可视化（不依赖opencv）
    #     visualize_anomaly_maps_simple(pfe, ae, viz_dataloader, args, device, epochs, afs=afs)
    #
    #     viz_result_path = './results/{}_{}_final_epoch_{}'.format(args.dataset, args.normal, epochs)
    #     logger.info("Anomaly map visualization completed. Results saved to: {}".format(viz_result_path))
    #
    # except Exception as e:
    #     logger.error("Failed to generate anomaly map visualization: {}".format(str(e)))
    #     logger.error("This might be due to missing visualization dependencies")
    #     logger.info("You can manually implement visualization using the anomaly_map data from evaluation_pixel()")

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


if __name__ == '__main__':

    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)
   
