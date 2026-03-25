"""
inference_example.py
===================
Checkpoint 使用示例：演示如何加载已训练模型并执行推理/评估。

使用方法：
---------
1. 先完整训练一次模型（会自动保存 checkpoint）：
       python main.py --dataset mvtec --normal carpet --epochs 200

2. 使用 checkpoint 加载模型进行推理（跳过训练）：
       python inference_example.py --dataset mvtec --normal carpet --checkpoint_mode best

3. 列出当前配置已有的所有 checkpoint：
       python main.py --dataset mvtec --normal carpet --list_checkpoints

4. 从指定 checkpoint 恢复训练（从第 100 epoch 继续）：
       python main.py --dataset mvtec --normal carpet --resume --checkpoint_path ./checkpoints/mvtec/carpet/.../best.pth
"""

import torch
import numpy as np
import os
import sys
from argparse import ArgumentParser

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.checkpoint import load_trained_model, build_checkpoint_dir, list_checkpoints, get_checkpoint_path
from util.test import evaluation
from dataset.dataset import OODDataSet


def parse_args():
    parser = ArgumentParser(description="CKAAD Inference Example using Checkpoint")

    # 必需参数：必须与训练时一致
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--normal', type=str, default='carpet')
    parser.add_argument('--model', type=str, default='wide_resnet50_2')
    parser.add_argument('--layer', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--img_size', type=int, default=256)

    # 渐进式实验配置（需与训练时一致）
    parser.add_argument('--enable_enhancement', nargs='*',
                       type=lambda x: str(x).lower() in ('true', '1', 'yes', 't', 'y'),
                       default=[False, False, False])
    parser.add_argument('--feature2_fusion_weight', type=float, default=0.5)

    # Checkpoint 选择参数
    parser.add_argument('--checkpoint_mode', type=str, default='best',
                       choices=['best', 'latest', 'final'],
                       help='best: highest AUROC, latest: most recent file, final: _final.pth')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='explicit checkpoint path (overrides --checkpoint_mode)')

    # 其他参数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.0)
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=0)
    parser.add_argument('--labeled_anomaly_class', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--topk', type=int, default=100)

    # 软门控参数（需与训练时一致）
    parser.add_argument('--gate_k', type=float, default=0.45)
    parser.add_argument('--gate_te', type=float, default=0.6)
    parser.add_argument('--gate_sigma', type=float, default=0.0)
    parser.add_argument('--recon_norm_quantile_low', type=float, default=0.02)
    parser.add_argument('--recon_norm_quantile_high', type=float, default=0.98)
    parser.add_argument('--recon_compress', type=str, default='sqrt',
                        choices=['sqrt', 'log', 'none'])
    parser.add_argument('--no_fuse_output_norm', action='store_false', dest='fuse_output_norm',
                        default=True,
                        help='disable min-max normalization of fused output (enabled by default)')

    # 日志
    parser.add_argument('--log_dir', type=str, default='./log/')

    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Step 1: 构建 checkpoint 目录（用于自动搜索）
    ckpt_dir = build_checkpoint_dir(args)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Step 2: 列出当前配置已有的所有 checkpoint
    ckpts = list_checkpoints(ckpt_dir)
    if not ckpts:
        print(f"\n[Warning] No checkpoints found in: {ckpt_dir}")
        print("Please train the model first, then run this script.")
        return
    print(f"\nFound {len(ckpts)} checkpoint(s):")
    for ck in ckpts:
        marker = " <-- BEST" if "best.pth" in ck["filename"] else ""
        marker = " <-- FINAL" if "final.pth" in ck["filename"] else marker
        print(
            f"  [{ck['filename']}]"
            f"  epoch={ck['epoch']}  best_{ck['metric_name']}={ck['best_metric']}  {ck['filesize_mb']:.1f} MB"
            f"{marker}"
        )

    # Step 3: 加载已训练模型（仅加载权重，不加载 optimizer）
    print(f"\nLoading model from checkpoint (mode='{args.checkpoint_mode}')...")
    try:
        pfe, ae, discriminator, ckpt_meta = load_trained_model(
            args=args,
            device=device,
            mode=args.checkpoint_mode,
            ckpt_path=args.checkpoint_path,
        )
        print(f"Loaded from epoch={ckpt_meta.get('epoch', '?')}, "
              f"best_metric={ckpt_meta.get('best_metric', '?')}")

        # 打印保存时的配置信息
        saved_cfg = ckpt_meta.get("config", {})
        if saved_cfg:
            print("\nCheckpoint model config:")
            for k, v in saved_cfg.items():
                print(f"  {k}: {v}")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("\n>>> Suggestion: Run training first, e.g.:")
        print(f"    python main.py --dataset {args.dataset} --normal {args.normal} --epochs 200")
        return

    # Step 4: 加载测试数据
    print("\nLoading test data...")
    dataset = OODDataSet(
        root='./data',
        dataset=args.dataset,
        image_size=args.img_size,
        category=args.normal,
        labeled_anomaly_ratio=args.labeled_anomaly_ratio,
        labeled_anomaly_class_num=args.labeled_anomaly_class_num,
        labeled_anomaly_class=args.labeled_anomaly_class,
    )
    _, _, _, test_dataloader = dataset.get_data_loader(batch_size=args.batch_size)

    # Step 5: 评估
    print("\nRunning evaluation...")
    metrics = evaluation(pfe, ae, discriminator, test_dataloader, device, args)

    # Step 6: 打印结果
    print("\n" + "=" * 60)
    print("Evaluation Results (loaded from checkpoint)")
    print("=" * 60)
    for level, scores in metrics.items():
        print(f"\n  [{level}]")
        for metric_name, value in scores.items():
            print(f"    {metric_name}: {value:.6f}")
    print("\n" + "=" * 60)

    # Step 7: 推理单张图像示例（可选）
    print("\n[Optional] Single-image inference example:")
    try:
        single_batch = next(iter(test_dataloader))
        if args.dataset in ['mvtec', 'visa', 'btad']:
            imgs = single_batch[0].to(device)  # (B, 3, H, W)
        else:
            imgs = single_batch[0].to(device)
        print(f"  Batch shape: {imgs.shape}")

        with torch.no_grad():
            inputs = pfe(imgs)
            outputs = ae(inputs)
            print(f"  Input feature shapes:  {[inp.shape for inp in inputs]}")
            print(f"  Output feature shapes: {[out.shape for out in outputs]}")
        print("  Single-image inference: OK")

    except Exception as e:
        print(f"  Skipped: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
