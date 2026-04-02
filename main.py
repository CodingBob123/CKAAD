# main.py
import torch
from contextlib import nullcontext
import numpy as np
import random
import os
from util.test import evaluation, visualize_anomaly_maps_simple, cal_anomaly_map, cal_energy_map
from util.visualize_comparison import (
    # visualize_recon_vs_energy,
    # visualize_multi_scale_energy,
    # compare_recon_energy_statistics,
    visualize_recon_energy_unified,
)
from model.model import PretrainedFeatureExtractor, ED, Discriminator
from util.checkpoint import (
    build_checkpoint_dir, build_eerm_checkpoint_dir, get_checkpoint_path, save_checkpoint,
    load_checkpoint, load_weights_only, check_config_compatibility,
    list_checkpoints
)
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端，避免WSL图形界面问题
import matplotlib.pyplot as plt
from torchvision import transforms
from model.error_reliability_modulation import ErrorReliabilityModulation

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
    
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet152'])
    
    parser.add_argument('--eval_epoch', type=int, default=1, help='every eval_epoch to eval')
    
    parser.add_argument('--layer', nargs='+', type=int, default=[2], help='choose pretrain resnet layer to reconstruct')
    
    parser.add_argument('--d_lr', type=float, default=1e-05, help='discriminator learning rate')
    
    parser.add_argument('--adv_conf', type=float, default=0.02, help='adversial loss conf')

    parser.add_argument('--topk', type=int, default=100, help='calculate topk values')

    parser.add_argument('--use_amp', action='store_true', help='enable mixed precision (AMP)')
    parser.add_argument('--compile', action='store_true', help='enable torch.compile for models if available')

    # Checkpoint 相关参数
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='checkpoint root directory; if None, defaults to ./checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                       help='save checkpoint every N epochs (0 means only save final and best)')
    parser.add_argument('--checkpoint_mode', type=str, default='best',
                       choices=['best', 'latest', 'final'],
                       help='checkpoint selection mode when loading: best/most recent metric, latest file, or final')
    parser.add_argument('--resume', action='store_true',
                       help='resume training from checkpoint (loads weights and optimizer state)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='explicit checkpoint path to load; if None, auto-searches by category config')
    parser.add_argument('--skip_training', action='store_true',
                       help='load checkpoint and skip training (for inference/evaluation only)')
    parser.add_argument('--list_checkpoints', action='store_true',
                       help='list available checkpoints for the current category config and exit')

    # 可视化相关参数
    # 系统A，原版重建误差图与原图对比
    parser.add_argument('--enable_epoch_viz',default=False, action='store_true', help='enable anomaly map visualization during training (every 8 epochs)')
    parser.add_argument('--viz_interval', type=int, default=0, help='interval for anomaly map visualization during training')
    parser.add_argument('--viz_samples_per_type', type=int, default=3, help='number of samples to visualize per anomaly type (including normal)')
    # 系统B，新版重建误差图与能量图情况对比
    parser.add_argument('--enable_recon_energy_viz', action='store_true', default=True,
                       help='enable unified reconstruction vs energy visualization')
    parser.add_argument('--viz_eval_interval', type=int, default=0,
                       help='visualization generation interval in epochs; '
                            '0 means only generate at the end of training; '
                            '>0 means generate every N epochs during training')

    # 新增：渐进式实验配置参数
    parser.add_argument('--enable_enhancement', nargs='*', type=lambda x: str(x).lower() in ('true', '1', 'yes', 't', 'y'),
                       default=[False, False, False],
                       help='enable enhancement for each branch [branch1, branch2, branch3]. Use --enable_enhancement True False True format')

    # 新增：Feature2边界增强融合权重控制
    parser.add_argument('--feature2_fusion_weight', type=float, default=0.5,
                       help='fusion weight for feature2 boundary enhancement (0.0=conservative only, 0.5=adaptive fusion, 1.0=aggressive only). '
                            'Recommended: 0.0 for zipper/tile, 0.5 for others')

    # 软门控融合参数
    parser.add_argument('--gate_k', type=float, default=0.45,
                       help='soft gate fusion: steepness of sigmoid curve (k parameter)')
    parser.add_argument('--gate_te', type=float, default=0.6,
                       help='soft gate fusion: energy map threshold (Te parameter), recommended 0.6 after per-sample minmax')
    parser.add_argument('--gate_sigma', type=float, default=0.0,
                       help='soft gate fusion: Gaussian smoothing sigma for energy map (0 means no smoothing)')
    parser.add_argument('--recon_norm_quantile_low', type=float, default=0.00,
                       help='soft gate fusion: lower quantile for reconstruction error normalization')
    parser.add_argument('--recon_norm_quantile_high', type=float, default=0.98,
                       help='soft gate fusion: upper quantile for reconstruction error normalization')
    parser.add_argument('--recon_compress', type=str, default='sqrt',
                       choices=['sqrt', 'log', 'none'],
                       help='soft gate fusion: high-tail compression method for reconstruction error')
    parser.add_argument('--no_fuse_output_norm', action='store_false', dest='fuse_output_norm',
                       help='disable min-max normalization of fused output (enabled by default)')
    # Recon-only 模式（Day 1 纯基线训练用）：跳过能量图融合
    parser.add_argument('--recon_only', action='store_true',
                       help='use only reconstruction error map (skip energy map fusion). '
                            'Used for Day 1 Recon baseline training.')
    # 能量差模式（V3-V6）：使用 D(input) - D(output) 作为能量图
    parser.add_argument('--energy_diff_mode', action='store_true',
                       help='use energy difference mode (D(input) - D(output)) for V3-V6. '
                            'Default is False (uses D(input) only).')

    # EERM 模块训练模式
    parser.add_argument('--eerm_mode', type=str, default='ckaad', choices=['ckaad', 'eerm'],
                       help='training mode: ckaad (train CKAAD only), eerm (freeze CKAAD and train EERM only)')
    parser.add_argument('--enable_eerm_training', action='store_true',
                       help='enable EERM modulation loss during training (only effective when eerm_mode=eerm)')
    parser.add_argument('--fusion_mode', type=str, default='soft_gate', choices=['soft_gate', 'eerm'],
                       help='fusion mode: soft_gate (original gating), eerm (reliability modulation)')
    parser.add_argument('--eerm_extra_channels', type=int, default=3,
                       help='number of extra channels for EERM module (for additional discrepancy maps)')
    parser.add_argument('--eerm_hidden_channels', type=int, default=32,
                       help='hidden channels for EERM fusion network')
    parser.add_argument('--eerm_lr', type=float, default=1e-4,
                       help='learning rate for EERM module')
    parser.add_argument('--lambda_good', type=float, default=1.0,
                       help='weight for EERM good-sample suppression loss')
    parser.add_argument('--lambda_keep', type=float, default=1.0,
                       help='weight for EERM peak-retention loss')
    parser.add_argument('--eerm_tau', type=float, default=0.7,
                       help='EERM peak-retention threshold tau')
    parser.add_argument('--eerm_k_ratio', type=float, default=0.01,
                       help='EERM top-k ratio for peak-retention loss')

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


def save_level_metrics(metrics, epoch, args, logger):
    """
    将各层级的评估指标保存到文本文件中。

    存储路径: ./metrics_per_epoch/{dataset}/{category}/n_{category}_a_{anomaly_class}_s_{seed}/
    文件名: epoch_{epoch:04d}_metrics.txt

    参数:
        metrics: dict，各层级指标字典
        epoch: int，当前 epoch
        args: 命令行参数
        logger: 日志记录器
    """
    import os

    # 构建目录路径
    metrics_root = './metrics_per_epoch'
    category_dir = os.path.join(
        metrics_root,
        args.dataset,
        args.normal,
        f"n_{args.normal}_a_{args.labeled_anomaly_class}_s_{args.seed}"
    )
    os.makedirs(category_dir, exist_ok=True)

    # 构建文件名
    filename = f"epoch_{epoch:04d}_metrics.txt"
    filepath = os.path.join(category_dir, filename)

    # 构建文本内容（保留原有格式）
    lines = []
    lines.append("=" * 40)
    lines.append(f"Epoch: {epoch:04d} | Category: {args.normal} | Dataset: {args.dataset}")
    lines.append("=" * 40)

    for key, value in metrics.items():
        lines.append(f"\n{key}:")
        if isinstance(value, dict):
            for metric_name, metric_val in value.items():
                lines.append(f"  {metric_name}: {metric_val}")
        else:
            lines.append(f"  {value}")

    lines.append("-" * 40)

    content = "\n".join(lines)

    # 写入文本文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Level metrics saved to: {filepath}")

    # ---------- 追加/保存 CSV 汇总，便于后续分析（每个类别一个 CSV） ----------
    try:
        # 扁平化 metrics 为字典：键示例 'Recon_Image-AUROC', 'Energy_Layer1_PRO', ...
        def flatten_metrics(mdict):
            flat = {}
            for k, v in mdict.items():
                if isinstance(v, dict):
                    for mk, mv in v.items():
                        flat[f"{k}_{mk}"] = mv
                else:
                    flat[str(k)] = v
            return flat

        flat = flatten_metrics(metrics)

        import csv

        csv_path = os.path.join(category_dir, 'metrics_summary.csv')
        write_header = not os.path.exists(csv_path)

        # 按固定列顺序：epoch, category, 然后按排序的 keys
        fieldnames = ['epoch', 'category'] + sorted(flat.keys())

        # 如果已有文件但列不一致，简单地追加缺失列为空（保持兼容性）
        if write_header:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # 读取已有 header（若有），并按其列写入，避免新旧列顺序冲突
        if not write_header:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                existing_header = next(reader)
        else:
            existing_header = fieldnames

        row = {col: '' for col in existing_header}
        row['epoch'] = f"{epoch:04d}"
        row['category'] = args.normal
        for k, v in flat.items():
            if k in row:
                row[k] = v

        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([row.get(col, '') for col in existing_header])

        logger.info(f"Metrics CSV updated: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write metrics CSV: {e}")

def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(b[item].shape[0], -1),
                                      b[item].view(b[item].shape[0], -1)))
    return loss

def loss_draw(loss_history, save_path=None):
    """
    绘制损失曲线。
    - 横轴：epoch
    - 纵轴：不同损失值
    - 布局：2x2网格，四个子图
    - 比例尺较大：调整为较大的画布和线宽
    """
    if not loss_history:
        print("Warning: loss_history is empty, skipping plot generation")
        return

    try:
        items = list(loss_history.items())
        if len(items) != 4:
            print(f"Warning: Expected 4 loss items, got {len(items)}")
            return

        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 更大的画布尺寸
        axes = axes.ravel()  # 扁平化axes数组

        for idx, (name, values) in enumerate(items):
            ax = axes[idx]
            if values and len(values) > 0:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, marker='o', linewidth=3, markersize=5, color='blue')
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel(name, fontsize=12)
                ax.set_title(f'{name} vs Epoch', fontsize=14, fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='both', which='major', labelsize=10)
                # 设置更大的边距
                ax.margins(x=0.05, y=0.1)
            else:
                ax.set_title(f"{name}\n(No data)", fontsize=14)
                ax.axis('off')

        plt.tight_layout(pad=3.0)

        if save_path:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            # 保存为jpg格式
            plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
            print(f"Loss curve saved to: {save_path}")
        else:
            # 在无图形界面环境中，不显示图片，直接跳过
            print("Warning: No save path provided, skipping plot display in headless environment")

        plt.close()

    except Exception as e:
        print(f"Error generating loss plot: {e}")
        plt.close()

def run_eval_and_viz(pfe, ae, discriminator, test_dataloader,
                     device, args, amp_ctx,
                     need_cached_maps, epochs, logger,
                     enable_stats=True,
                     recon_only=False,
                     energy_diff_mode=False,
                     eerm_module=None):
    """
    执行一次完整评估（metrics + 可选缓存 map），并生成重建误差/能量图可视化。

    参数:
        pfe, ae, discriminator: 模型
        test_dataloader: 测试数据加载器
        device: 设备
        args: 命令行参数
        amp_ctx: AMP 上下文（nullcontext 或 autocast）
        need_cached_maps: bool，是否返回并可视化中间 map
        epochs: 当前 epoch（用于可视化路径）
        logger: 日志记录器
        enable_stats: bool，训练结束后是否打印统计信息
        recon_only: bool，是否使用纯重建误差图（跳过能量图融合）
        energy_diff_mode: bool，是否使用能量差模式（V3-V6）

    返回:
        metrics: 评估指标字典
        recon_maps, energy_maps, final_maps, all_gts, anomaly_maps（仅当 need_cached_maps=True 时有值）
    """
    with amp_ctx():
        if need_cached_maps:
            if energy_diff_mode:
                metrics, recon_maps, energy_maps, energy_in_maps, energy_out_maps, final_maps, all_gts, anomaly_maps, weight_maps = evaluation(
                    pfe, ae, discriminator, test_dataloader, device, args, return_maps=True,
                    recon_only=recon_only, energy_diff_mode=energy_diff_mode, eerm_module=eerm_module, fusion_mode=args.fusion_mode)
            else:
                metrics, recon_maps, energy_maps, energy_in_maps, energy_out_maps, final_maps, all_gts, anomaly_maps, weight_maps = evaluation(
                    pfe, ae, discriminator, test_dataloader, device, args, return_maps=True,
                    recon_only=recon_only, energy_diff_mode=energy_diff_mode, eerm_module=eerm_module, fusion_mode=args.fusion_mode)
        else:
            metrics = evaluation(pfe, ae, discriminator, test_dataloader, device, args,
                              recon_only=recon_only, energy_diff_mode=energy_diff_mode, eerm_module=eerm_module, fusion_mode=args.fusion_mode)
            recon_maps, energy_maps, energy_in_maps, energy_out_maps, final_maps, all_gts, anomaly_maps, weight_maps = None, None, None, None, None, None, None, None

    infostr = get_res_str(metrics)
    logger.info("Test: {}".format(infostr))

    if need_cached_maps:
        try:
            if args.fusion_mode == 'eerm':
                # EERM 模式可视化
                cached_maps = {
                    'recon_maps': recon_maps,
                    'energy_maps': energy_maps,
                    'final_maps': final_maps,
                    'anomaly_maps': anomaly_maps,
                    'weight_maps': weight_maps,
                }
                from util.visualize_comparison import visualize_eerm_unified
                visualize_eerm_unified(
                    pfe, ae, discriminator, test_dataloader, args, device, epochs,
                    cached_maps=cached_maps,
                    enable_stats=enable_stats,
                )
                logger.info("EERM Visualization saved at epoch {}".format(epochs))
            elif energy_diff_mode:
                cached_maps = {
                    'recon_maps': recon_maps,
                    'energy_maps': energy_maps,
                    'energy_in_maps': energy_in_maps,
                    'energy_out_maps': energy_out_maps,
                    'final_maps': final_maps,
                    'anomaly_maps': anomaly_maps,
                    'weight_maps': weight_maps,
                }
                # 导入新的可视化函数
                from util.visualize_comparison import visualize_energy_diff_unified
                visualize_energy_diff_unified(
                    pfe, ae, discriminator, test_dataloader, args, device, epochs,
                    cached_maps=cached_maps,
                    enable_stats=enable_stats,
                )
                logger.info("Energy Diff viz saved at epoch {}".format(epochs))
            else:
                cached_maps = {
                    'recon_maps': recon_maps,
                    'energy_maps': energy_maps,
                    'final_maps': final_maps,
                    'anomaly_maps': anomaly_maps,
                    'weight_maps': weight_maps,
                }
                visualize_recon_energy_unified(
                    pfe, ae, discriminator, test_dataloader, args, device, epochs,
                    cached_maps=cached_maps,
                    enable_stats=enable_stats,
                )
                logger.info("Recon vs Energy viz saved at epoch {}".format(epochs))
        except Exception as e:
            logger.error("Failed to generate visualizations: {}".format(str(e)))

    return metrics, recon_maps, energy_maps, energy_in_maps, energy_out_maps, final_maps, all_gts, anomaly_maps, weight_maps


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
    log_dir = os.path.join(args.log_dir, "lan{:.2f}_acn{}".format(args.labeled_anomaly_ratio,  args.labeled_anomaly_class_num), args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger_filename = os.path.join(log_dir, 'n_{}_a_{}_s_{}'.format(args.normal, args.labeled_anomaly_class, args.seed) + '.txt')
    logger = get_logger(logger_filename)
    logger.info("log file: {}".format(logger_filename))
    logger.info("class: {}".format(args.normal))
    
    print_args(logger, args)
    epochs = args.epochs
    batch_size = args.batch_size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {}".format(device))

    # 2.0 自动构建 checkpoint 目录并处理特殊模式
    # EERM 模式使用独立目录
    if getattr(args, 'eerm_mode', 'ckaad') == 'eerm':
        ckpt_dir = build_eerm_checkpoint_dir(args)
        logger.info(f"EERM mode: using checkpoint dir: {ckpt_dir}")
    else:
        ckpt_dir = build_checkpoint_dir(args)

    # 2.0.1 列出已有 checkpoint 并退出
    if args.list_checkpoints:
        ckpts = list_checkpoints(ckpt_dir)
        if not ckpts:
            logger.info(f"No checkpoints found in: {ckpt_dir}")
        else:
            logger.info(f"Found {len(ckpts)} checkpoint(s) in: {ckpt_dir}")
            for ck in ckpts:
                logger.info(
                    f"  {ck['filename']}  |  epoch={ck['epoch']}  |  "
                    f"best_epoch={ck['best_epoch']}  |  best_{ck['metric_name']}={ck['best_metric']}  |  "
                    f"{ck['filesize_mb']:.1f} MB"
                )
        return  # 直接退出

    # 2.0.2 自动查找 checkpoint 路径（如果未显式指定）
    # EERM 模式：优先从 CKAAD 主干目录加载预训练权重
    if args.checkpoint_path is None:
        if getattr(args, 'eerm_mode', 'ckaad') == 'eerm':
            # EERM 模式：从 CKAAD 目录查找预训练模型
            ckaad_ckpt_dir = build_checkpoint_dir(args)
            found_path = get_checkpoint_path(ckaad_ckpt_dir, mode=args.checkpoint_mode)
            if found_path is not None:
                logger.info(f"EERM mode: auto-detected CKAAD checkpoint: {found_path}")
            else:
                logger.warning(f"EERM mode: no CKAAD checkpoint found in: {ckaad_ckpt_dir}")
        else:
            found_path = get_checkpoint_path(ckpt_dir, mode=args.checkpoint_mode)
            if found_path is not None:
                logger.info(f"Auto-detected checkpoint: {found_path}")
            else:
                logger.info(f"No checkpoint found in: {ckpt_dir}  (will train from scratch)")
        args.checkpoint_path = found_path

    # 2.加载数据集，获取数据加载器
    dataset = OODDataSet(root='./data', dataset=args.dataset, image_size=args.img_size, category=args.normal,
                         labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                         labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                         labeled_anomaly_class=args.labeled_anomaly_class)
    train_dataloader, valid_dataloader, anomaly_dataloader, test_dataloader = dataset.get_data_loader(batch_size=batch_size)

    # 3.初始化模型
    # 3.1 初始化预训练特征提取器，冻结参数
    # 根据args.layer参数选择提取哪几层特征，例如[1,2,3]表示提取ResNet的第1、2、3层特征
    pfe = PretrainedFeatureExtractor(args.model, layers=args.layer, image_size=args.img_size).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)  # 冻结特征提取器参数
    pfe.eval()  # 设置为评估模式
    
    # 3.2 初始化编码器-解码器(自编码器)
    # 输入通道数由预训练模型的输出通道数决定，例如对于ResNet50和layers=[1,2,3]，为[256,512,1024]
    # 传递渐进式实验配置参数
    ae = ED(backbone=args.model, input_channels=pfe.output_channels, enable_enhancement=args.enable_enhancement,
            feature2_fusion_weight=args.feature2_fusion_weight).to(device)
    
    # 3.3 初始化判别器
    # input_sizes: 各层特征图的空间尺寸，例如[64,32,16]
    # input_channels: 各层特征图的通道数，例如[64,128,256]
    # expansion: 通道扩展系数，ResNet18/34为1，ResNet50/101等为4
    discriminator = Discriminator(input_sizes=pfe.output_sizes, input_channels=pfe.output_channels, expansion=pfe.expansion).to(device)

    # 初始化误差分析头
    eerm_module = ErrorReliabilityModulation(
        extra_channels=args.eerm_extra_channels,
        hidden_channels=args.eerm_hidden_channels
    ).to(device)

    # 3.2.5 checkpoint 加载（优先级：显式路径 > 自动搜索 > 不加载）
    start_epoch = 1
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        ckpt_meta = load_checkpoint(
            ae, discriminator,
            ae_optimizer=None,
            discriminator_optimizer=None,
            checkpoint_path=args.checkpoint_path,
            device=device,
            strict=False,
        )
        saved_cfg = ckpt_meta.get("config", {})
        if not check_config_compatibility(saved_cfg, args, verbose=True):
            logger.warning("Model config mismatch — checkpoint weights may not align correctly.")
        start_epoch = ckpt_meta.get("epoch", 0) + 1
        logger.info(f"Loaded checkpoint from epoch {ckpt_meta.get('epoch', '?')}, training will resume from epoch {start_epoch}")
    elif args.skip_training:
        logger.error("--skip_training requires a valid --checkpoint_path. No checkpoint loaded.")
        return

    # 3.4 初始化优化器
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    eerm_optimizer = torch.optim.Adam(eerm_module.parameters(), lr=args.eerm_lr)

    # 根据训练模式设置模型冻结状态
    if args.eerm_mode == 'eerm':
        logger.info("EERM mode: freezing CKAAD backbone (ae and discriminator), training EERM module only")
        # 冻结 ae 和 discriminator
        for param in ae.parameters():
            param.requires_grad = False
        for param in discriminator.parameters():
            param.requires_grad = False
        ae.eval()
        discriminator.eval()
    else:
        logger.info("CKAAD mode: training CKAAD backbone (ae and discriminator)")

    # 设置权重系数和标签
    gamma = 0.5  # 控制重建特征损失的权重
    true_label = 0  # 正常样本的标签
    fake_label = 1  # 异常样本的标签

    # 如果设置了 --skip_training，跳过训练直接评估
    if args.skip_training:
        logger.info("--skip_training: running evaluation only without training.")
        ae.eval()
        discriminator.eval()
        eerm_module.eval()
        with amp_ctx():
            metrics = evaluation(pfe, ae, discriminator, test_dataloader, device, args,
                              recon_only=args.recon_only, eerm_module=eerm_module, fusion_mode=args.fusion_mode)
        infostr = get_res_str(metrics)
        logger.info("Test (from checkpoint): {}".format(infostr))
        return

    # AMP setup
    use_cuda_amp = args.use_amp and (device == 'cuda')
    if use_cuda_amp:
        from torch.cuda.amp import autocast, GradScaler
        scaler_ae = GradScaler()
        scaler_d = GradScaler()
        amp_ctx = autocast
        logger.info("AMP enabled (autocast + GradScaler)")
    else:
        scaler_ae = None
        scaler_d = None
        amp_ctx = nullcontext

    # 记录各类损失用于绘图
    loss_history = {
        "dis_loss": [],
        "recon_loss": [],
        "adv_loss": [],
        "ae_loss": [],
    }

    # 最佳指标跟踪（用于决定是否保存 best checkpoint）
    best_metric = -float("inf")
    best_epoch = start_epoch

    # 4.开始训练循环
    for epoch in range(start_epoch, epochs+1):
        ae.train()
        discriminator.train()
        dis_loss_list = []
        recon_loss_list = []
        adv_loss_list = []
        ae_loss_list = []
        
        # 使用zip和cycle将正常数据和异常数据配对
        # cycle确保异常数据可以循环使用，即使异常数据少于正常数据
        for normal, anomaly in tqdm.tqdm(zip(train_dataloader, cycle(anomaly_dataloader))):
            # 5.1 准备输入数据
            normal_img = normal[0].to(device)  # 正常图像: [batch_size, 3, img_size, img_size]
            
            # 处理异常图像，根据不同情况获取异常样本
            if anomaly is not None:
                anomaly_img = anomaly[0].to(device)  # 异常图像: [batch_size, 3, img_size, img_size]
            elif args.dataset in ['mvtec', 'visa', 'btad'] and len(normal) == 4:
                anomaly_img = normal[1].to(device)  # 某些数据集中，normal包含正常和异常样本
            else:
                anomaly_img = normal_img[:0]  # 创建空张量，表示没有异常样本
                
            anomaly_size = anomaly_img.size(0)  # 异常样本的数量
            
            # 5.2 特征提取和重建
            # 使用预训练特征提取器提取正常样本的多层级特征
            with amp_ctx():
                normal_inputs = pfe(normal_img)  # 列表，包含多个特征图: [
                                                #   [batch_size, 64*exp, H1, W1],
                                                #   [batch_size, 128*exp, H2, W2],
                                                #   [batch_size, 256*exp, H3, W3]
                                                # ]
                                                # 其中exp是扩展系数，H1>H2>H3, W1>W2>W3
                                                # 例如对于img_size=256，可能为[64,32,16]

                # 使用自编码器重建正常样本特征
                normal_outputs = ae(normal_inputs)  # 列表，包含多个重建特征图，形状与normal_inputs相同

            # 如果有异常样本，则提取和重建异常样本特征
            if anomaly_size > 0:
                with amp_ctx():
                    anomaly_inputs = pfe(anomaly_img)  # 列表，形状与normal_inputs相同
                    anomaly_outputs = ae(anomaly_inputs)  # 列表，形状与normal_outputs相同
                
                # 将正常样本重建特征和异常样本重建特征在批次维度上拼接
                # 对每个层级的特征分别拼接
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]  
                # outputs是列表，每个元素形状为: [batch_size*2, C, H, W]
            else:
                outputs = normal_outputs  # 如果没有异常样本，直接使用正常样本重建特征
                
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
                with amp_ctx():
                    dis_loss = discriminator.calculate_loss(normal_inputs_detach, true_label) + \
                              (1 - gamma) * discriminator.calculate_loss(anomaly_inputs_detach, fake_label) + \
                              gamma * discriminator.calculate_loss(outputs_detach, fake_label)
                discriminator_optimizer.zero_grad()
                if use_cuda_amp:
                    scaler_d.scale(dis_loss).backward()
                    # Unscale before clipping
                    scaler_d.unscale_(discriminator_optimizer)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                    scaler_d.step(discriminator_optimizer)
                    scaler_d.update()
                else:
                    dis_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
                    discriminator_optimizer.step()
                    
                # 5.6 计算对抗损失
                # 希望重建特征能够欺骗判别器，被判为真(标签0)
                with amp_ctx():
                    adv_loss = discriminator.calculate_loss(outputs, true_label)

            # 5.7 计算重建损失和自编码器总损失
            # 重建损失使用余弦相似度，衡量正常样本特征和重建特征的相似程度
            with amp_ctx():
                recon_loss = loss_function(normal_inputs, normal_outputs)
                ae_loss = recon_loss + args.adv_conf * adv_loss
            ae_optimizer.zero_grad()
            if use_cuda_amp:
                scaler_ae.scale(ae_loss).backward()
                scaler_ae.step(ae_optimizer)
                scaler_ae.update()
            else:
                ae_loss.backward()
                ae_optimizer.step()

            # 5.8 EERM 调制损失计算（仅在 eerm_mode 时启用）
            loss_eerm = torch.tensor(0.0).to(device)
            loss_eerm_list = []

            if args.eerm_mode == 'eerm' and args.enable_eerm_training:
                # EERM 训练：冻结主干，只训练 EERM 模块
                # 使用正常样本计算 L_good，使用异常样本计算 L_keep
                with amp_ctx():
                    # 计算当前批次的 base_error_map 和 refined_map
                    # 由于 ae.eval() 冻结，我们使用 eval 模式下的输出
                    ae.eval()
                    pfe.eval()
                    discriminator.eval()

                    with torch.no_grad():
                        # 重新计算特征（因为前面 ae 可能改变了状态）
                        normal_inputs_eval = pfe(normal_img)
                        normal_outputs_eval = ae(normal_inputs_eval)
                        recon_map_normal = cal_anomaly_map(normal_inputs_eval, normal_outputs_eval, normal_img.shape[-1], amap_mode='add')
                        energy_maps_normal = cal_energy_map(discriminator, normal_inputs_eval, normal_img.shape[-1])

                    # 对于正常样本：使用 L_good 损失
                    if anomaly_size > 0:
                        with torch.no_grad():
                            anomaly_inputs_eval = pfe(anomaly_img)
                            anomaly_outputs_eval = ae(anomaly_inputs_eval)
                            recon_map_anomaly = cal_anomaly_map(anomaly_inputs_eval, anomaly_outputs_eval, anomaly_img.shape[-1], amap_mode='add')
                            energy_maps_anomaly = cal_energy_map(discriminator, anomaly_inputs_eval, anomaly_img.shape[-1])

                        # 分别计算正常和异常样本的 refined map
                        from util.test import eerm_fuse
                        refined_normal, _ = eerm_fuse(
                            eerm_module=eerm_module,
                            recon_map=recon_map_normal,
                            energy_maps_t=energy_maps_normal,
                            inputs=normal_inputs_eval,
                            outputs=normal_outputs_eval,
                            device=device,
                            extra_channels=args.eerm_extra_channels,
                            recon_norm_quantile_low=args.recon_norm_quantile_low,
                            recon_norm_quantile_high=args.recon_norm_quantile_high,
                            recon_compress=args.recon_compress,
                            fuse_output_norm=False
                        )
                        refined_anomaly, _ = eerm_fuse(
                            eerm_module=eerm_module,
                            recon_map=recon_map_anomaly,
                            energy_maps_t=energy_maps_anomaly,
                            inputs=anomaly_inputs_eval,
                            outputs=anomaly_outputs_eval,
                            device=device,
                            extra_channels=args.eerm_extra_channels,
                            recon_norm_quantile_low=args.recon_norm_quantile_low,
                            recon_norm_quantile_high=args.recon_norm_quantile_high,
                            recon_compress=args.recon_compress,
                            fuse_output_norm=False
                        )

                        # 转换为 torch tensor
                        refined_normal_t = torch.from_numpy(refined_normal).float().to(device).unsqueeze(1)
                        refined_anomaly_t = torch.from_numpy(refined_anomaly).float().to(device).unsqueeze(1)
                        recon_normal_t = torch.from_numpy(recon_map_normal).float().to(device).unsqueeze(1)
                        recon_anomaly_t = torch.from_numpy(recon_map_anomaly).float().to(device).unsqueeze(1)

                        # 计算 L_good（正常样本）和 L_keep（异常样本）
                        loss_good = refined_normal_t.mean()
                        loss_keep = eerm_module.loss_keep(
                            refined_map=refined_anomaly_t,
                            base_error_map=recon_anomaly_t,
                            tau=args.eerm_tau,
                            k_ratio=args.eerm_k_ratio
                        )
                        loss_eerm = args.lambda_good * loss_good + args.lambda_keep * loss_keep

                        ae.train()
                        discriminator.train()
                    else:
                        # 只有正常样本时，只计算 L_good
                        with torch.no_grad():
                            refined_normal, _ = eerm_fuse(
                                eerm_module=eerm_module,
                                recon_map=recon_map_normal,
                                energy_maps_t=energy_maps_normal,
                                inputs=normal_inputs_eval,
                                outputs=normal_outputs_eval,
                                device=device,
                                extra_channels=args.eerm_extra_channels,
                                recon_norm_quantile_low=args.recon_norm_quantile_low,
                                recon_norm_quantile_high=args.recon_norm_quantile_high,
                                recon_compress=args.recon_compress,
                                fuse_output_norm=False
                            )
                        refined_normal_t = torch.from_numpy(refined_normal).float().to(device).unsqueeze(1)
                        loss_eerm = args.lambda_good * refined_normal_t.mean()

                        ae.train()
                        discriminator.train()

                # 反向传播 EERM 损失
                if loss_eerm > 0:
                    eerm_optimizer.zero_grad()
                    loss_eerm.backward()
                    eerm_optimizer.step()

                ae.train()
                discriminator.train()

            # 5.9 记录各项损失值
            dis_loss_list.append(dis_loss.item())
            ae_loss_list.append(ae_loss.item())
            recon_loss_list.append(recon_loss.item())
            adv_loss_list.append(adv_loss.item())
            if args.eerm_mode == 'eerm':
                loss_eerm_list.append(loss_eerm.item())

        # 6. 打印当前epoch的训练损失并记录到历史
        epoch_dis = np.mean(dis_loss_list)
        epoch_recon = np.mean(recon_loss_list)
        epoch_adv = np.mean(adv_loss_list)
        epoch_ae = np.mean(ae_loss_list)

        if args.eerm_mode == 'eerm' and len(loss_eerm_list) > 0:
            epoch_eerm = np.mean(loss_eerm_list)
            logger.info("epoch [{}/{}], dis_loss: {:.6f}, recon_loss:{:.6f}, adv_loss:{:.6f}, ae_loss: {:.6f}, eerm_loss: {:.6f}".format(
                epoch, epochs, epoch_dis, epoch_recon, epoch_adv, epoch_ae, epoch_eerm
            ))
            loss_history["eerm_loss"] = loss_history.get("eerm_loss", [])
            loss_history["eerm_loss"].append(epoch_eerm)
        else:
            logger.info("epoch [{}/{}], dis_loss: {:.6f}, recon_loss:{:.6f}, adv_loss:{:.6f}, ae_loss: {:.6f}".format(
                epoch, epochs, epoch_dis, epoch_recon, epoch_adv, epoch_ae
            ))

        # 记录损失历史用于绘图
        loss_history["dis_loss"].append(epoch_dis)
        loss_history["recon_loss"].append(epoch_recon)
        loss_history["adv_loss"].append(epoch_adv)
        loss_history["ae_loss"].append(epoch_ae)

        # 7. 定期评估模型性能
        if (epoch) % args.eval_epoch == 0:
            if valid_dataloader is not None:
                with amp_ctx():
                    valid_metrics = evaluation(pfe, ae, discriminator, valid_dataloader, device, args,
                                            recon_only=args.recon_only, eerm_module=eerm_module, fusion_mode=args.fusion_mode)
                valid_info = get_res_str(valid_metrics)
                logger.info("Valid: {}".format(valid_info))

            # 可视化在独立周期下触发（与 eval_epoch 解耦）
            need_viz = (args.enable_recon_energy_viz
                        and (args.viz_eval_interval == 0 or epoch % args.viz_eval_interval == 0))
            need_cached_maps = need_viz
            metrics, _, _, _, _, _, _, _, _ = run_eval_and_viz(
                pfe, ae, discriminator, test_dataloader,
                device, args, amp_ctx,
                need_cached_maps=need_cached_maps,
                epochs=epoch, logger=logger,
                enable_stats=False,
                recon_only=False,  # 强制计算能量图，用于层级指标评估
                energy_diff_mode=getattr(args, 'energy_diff_mode', False),
                eerm_module=eerm_module,
            )

            # 保存各层级指标到文件
            save_level_metrics(metrics, epoch, args, logger)

            # 可选：在定期评估时绘制异常热力图（独立计算，与 cached_maps 无关）
            if args.enable_epoch_viz and epoch % args.viz_interval == 0:
                try:
                    logger.info("Generating anomaly maps at epoch {}...".format(epoch))
                    visualize_anomaly_maps_simple(pfe, ae, test_dataloader, args, device, epoch)
                    viz_result_path = '/hy-tmp/results/{}_{}_epoch_{}'.format(args.dataset, args.normal, epoch)
                    logger.info("Anomaly maps saved to: {}".format(viz_result_path))
                except Exception as e:
                    logger.error("Failed to generate anomaly maps at epoch {}: {}".format(epoch, str(e)))

            # 7.1 checkpoint 保存
            # 保存 best checkpoint（基于 Image AUROC）
            current_metric = metrics.get("Image", {}).get("AUROC", 0.0)
            save_best = current_metric > best_metric
            if save_best:
                best_metric = current_metric
                best_epoch = epoch

            # 按间隔保存 regular checkpoint（每个 epoch_XXXX.pth）
            is_interval_save = (args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0)

            if save_best or is_interval_save:
                save_metrics = {
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "metric_name": "image_auc",
                }
                save_checkpoint(
                    ae=ae,
                    discriminator=discriminator,
                    ae_optimizer=ae_optimizer,
                    discriminator_optimizer=discriminator_optimizer,
                    epoch=epoch,
                    metrics=save_metrics,
                    args=args,
                    ckpt_dir=ckpt_dir,
                    save_best=save_best,
                )

    # 8. 训练结束后保存最终 checkpoint 和损失曲线
    try:
        # 保存 final checkpoint
        save_metrics = {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "metric_name": "image_auc",
        }
        save_checkpoint(
            ae=ae,
            discriminator=discriminator,
            ae_optimizer=ae_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epochs,
            metrics=save_metrics,
            args=args,
            ckpt_dir=ckpt_dir,
            is_final=True,
            save_best=(epochs == best_epoch and epochs % args.checkpoint_interval != 0),
        )

        pic_dir = "./pic/"
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir, exist_ok=True)

        loss_img_name = "loss_curve_final_n_{}_a_{}_s_{}.jpg".format(args.normal, args.labeled_anomaly_class, args.seed)
        loss_save_path = os.path.join(pic_dir, loss_img_name)

        # 检查损失历史记录是否为空
        if any(loss_history.values()):
            loss_draw(loss_history, loss_save_path)
            logger.info("Final loss curve saved to: {}".format(loss_save_path))
        else:
            logger.warning("Loss history is empty, skipping plot generation")
    except Exception as e:
        logger.error("Failed to generate final loss curve: {}".format(str(e)))

    # 9. 训练结束后进行统一可视化（使用测试数据集）
    try:
        logger.info("Starting post-training visualization...")

        # 设置数据变换（与训练时相同）
        if args.dataset in ['mvtec', 'visa', 'btad']:
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            gt_transform = transforms.Compose([transforms.ToTensor()])
        else:
            img_transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            gt_transform = transforms.Compose([transforms.ToTensor()])

        # 9.1 异常热力图可视化（使用独立采样的 dataloader）
        logger.info("Generating anomaly map visualization...")

        # 创建测试数据集（只可视化前几个样本以节省时间）
        if args.dataset == 'mvtec':
            from dataset.mvtec import MVTecDataset
            viz_dataset = MVTecDataset(root='./data', category=args.normal, train=False,
                                     transform=img_transform, gt_target_transform=gt_transform,
                                     img_size=args.img_size)
            # 加载实际的数据到内存中
            viz_dataset.load_data()

            # 方案A：类型平衡采样 - 为每个异常类型选择固定数量的样本
            samples_per_type = args.viz_samples_per_type  # 从命令行参数获取
            type_to_indices = {}

            # 按异常类型分组样本索引
            for i, target in enumerate(viz_dataset.targets):
                type_name = viz_dataset.types_set[target]
                if type_name not in type_to_indices:
                    type_to_indices[type_name] = []
                type_to_indices[type_name].append(i)

            # 为每个类型选择指定数量的样本
            viz_indices = []
            for type_name, indices in type_to_indices.items():
                # 选择前samples_per_type个样本，如果不够则选择全部
                selected_count = min(len(indices), samples_per_type)
                viz_indices.extend(indices[:selected_count])

                logger.info(f"Selected {selected_count} samples for type '{type_name}'")

            logger.info(f"Total selected {len(viz_indices)} samples for visualization")

            # 筛选数据
            viz_dataset.data = viz_dataset.data[viz_indices]
            viz_dataset.targets = viz_dataset.targets[viz_indices]
            viz_dataset.gt_targets = viz_dataset.gt_targets[viz_indices]

        viz_dataloader = torch.utils.data.DataLoader(viz_dataset, batch_size=4, shuffle=False)

        # 9.1 & 9.2 共用一次前向评估，消除重复计算
        logger.info("Running evaluation for visualizations (single forward pass)...")
        energy_diff_mode = getattr(args, 'energy_diff_mode', False)
        with amp_ctx():
            if energy_diff_mode:
                metrics, recon_maps, energy_maps, energy_in_maps, energy_out_maps, final_maps, all_gts, anomaly_maps = evaluation(
                    pfe, ae, discriminator, viz_dataloader, device, args,
                    return_maps=True, recon_only=args.recon_only, energy_diff_mode=energy_diff_mode)
            else:
                metrics, recon_maps, energy_maps, energy_in_maps, energy_out_maps, final_maps, all_gts, anomaly_maps = evaluation(
                    pfe, ae, discriminator, viz_dataloader, device, args,
                    return_maps=True, recon_only=args.recon_only, energy_diff_mode=energy_diff_mode)
        logger.info("Test: {}".format(get_res_str(metrics)))

        # 构建 cached_maps
        if energy_diff_mode:
            cached_maps = {
                'recon_maps': recon_maps,
                'energy_maps': energy_maps,
                'energy_in_maps': energy_in_maps,
                'energy_out_maps': energy_out_maps,
                'final_maps': final_maps,
                'anomaly_maps': anomaly_maps,
            }
        else:
            cached_maps = {
                'recon_maps': recon_maps,
                'energy_maps': energy_maps,
                'final_maps': final_maps,
                'anomaly_maps': anomaly_maps,
            }

        # 9.1 异常热力图可视化（使用预计算的 anomaly_maps，不再重复前向）
        logger.info("Generating anomaly map visualization...")
        visualize_anomaly_maps_simple(
            pfe, ae, viz_dataloader, args, device, epochs,
            anomaly_maps=anomaly_maps
        )
        viz_result_path = '/hy-tmp/results/{}_{}_final_epoch_{}'.format(args.dataset, args.normal, epochs)
        logger.info("Anomaly map visualization completed. Results saved to: {}".format(viz_result_path))

        # 9.2 重建误差 vs 能量图可视化（复用 cached_maps，不再重新评估）
        if args.enable_recon_energy_viz:
            try:
                logger.info("Generating recon vs energy visualizations (using cached maps)...")

                if energy_diff_mode:
                    from util.visualize_comparison import visualize_energy_diff_unified
                    visualize_energy_diff_unified(
                        pfe, ae, discriminator, viz_dataloader, args, device, epochs,
                        cached_maps=cached_maps,
                        enable_stats=False,
                    )
                    result_path = '/hy-tmp/results/{}_{}_energy_diff_final_epoch_{}'.format(
                        args.dataset, args.normal, epochs)
                else:
                    visualize_recon_energy_unified(
                        pfe, ae, discriminator, viz_dataloader, args, device, epochs,
                        cached_maps=cached_maps,
                        enable_stats=False,
                    )
                    result_path = '/hy-tmp/results/{}_{}_recon_vs_energy_final_epoch_{}'.format(
                        args.dataset, args.normal, epochs)

                logger.info("Energy visualization completed. Results saved to: {}".format(result_path))

            except Exception as e:
                logger.error("Failed to generate energy visualizations: {}".format(str(e)))

        logger.info("All post-training visualizations completed successfully!")

    except Exception as e:
        logger.error("Failed to generate post-training visualizations: {}".format(str(e)))
        logger.error("This might be due to missing visualization dependencies")

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


if __name__ == '__main__':

    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)