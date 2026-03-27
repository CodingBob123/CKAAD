"""
test_energy_guidance.py
=====================
能量图引导与筛选创新点验证脚本（Day 2 评估）

使用方法：
---------
# 单类别评估（使用 Day1 训练的 checkpoint）：
python test_energy_guidance.py --dataset mvtec --normal capsule --checkpoint_mode latest

# 指定具体 checkpoint：
python test_energy_guidance.py --dataset mvtec --normal capsule --checkpoint_path /path/to/checkpoint.pth

# 列出可用 checkpoint：
python main.py --dataset mvtec --normal capsule --list_checkpoints
"""

import torch
import numpy as np
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.checkpoint import load_trained_model, build_checkpoint_dir, list_checkpoints
from util.energy_eval_utils import (
    run_all_versions,
    print_comparison_table,
    VERSIONS,
)


def parse_args():
    parser = ArgumentParser(
        description="能量图引导与筛选 - 多版本评估脚本（Day 2）"
    )

    # 必需参数
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--normal', type=str, default='capsule')
    parser.add_argument('--model', type=str, default='wide_resnet50_2')
    parser.add_argument('--layer', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--img_size', type=int, default=256)

    # 渐进式实验配置（需与训练时一致）
    parser.add_argument('--enable_enhancement', nargs='*',
                       type=lambda x: str(x).lower() in ('true', '1', 'yes', 't', 'y'),
                       default=[False, False, False])
    parser.add_argument('--feature2_fusion_weight', type=float, default=0.5)

    # Checkpoint 选择参数
    parser.add_argument('--checkpoint_mode', type=str, default='latest',
                       choices=['best', 'latest', 'final'],
                       help='best: highest AUROC, latest: most recent file, final: _final.pth')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='explicit checkpoint path (overrides --checkpoint_mode)')

    # 评估参数
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.05)
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=1)
    parser.add_argument('--labeled_anomaly_class', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--topk', type=int, default=100)

    # 软门控参数（用于 V2-V6）
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

    # 日志目录
    parser.add_argument('--log_dir', type=str, default='./log_day1/')

    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Step 1: 构建 checkpoint 目录
    ckpt_dir = build_checkpoint_dir(args)
    print(f"\nCheckpoint directory: {ckpt_dir}")

    # Step 2: 列出可用 checkpoint
    ckpts = list_checkpoints(ckpt_dir)
    if not ckpts:
        print(f"\n[Error] No checkpoints found in: {ckpt_dir}")
        print(">>> Suggestion: Run mvtec_day1.sh first to train Day 1 baseline.")
        return
    print(f"\nFound {len(ckpts)} checkpoint(s):")
    for ck in ckpts:
        marker = " <-- BEST" if "best.pth" in ck["filename"] else ""
        marker = " <-- FINAL" if "final.pth" in ck["filename"] else marker
        marker = " <-- LATEST" if args.checkpoint_mode == "latest" and ck == ckpts[-1] else marker
        print(
            f"  [{ck['filename']}]"
            f"  epoch={ck['epoch']}  best_{ck['metric_name']}={ck['best_metric']}  {ck['filesize_mb']:.1f} MB"
            f"{marker}"
        )

    # Step 3: 加载已训练模型
    print(f"\nLoading model from checkpoint (mode='{args.checkpoint_mode}')...")
    try:
        pfe, ae, discriminator, ckpt_meta = load_trained_model(
            args=args,
            device=device,
            mode=args.checkpoint_mode,
            ckpt_path=args.checkpoint_path,
        )
        print(f"Loaded: epoch={ckpt_meta.get('epoch', '?')}, "
              f"best_metric={ckpt_meta.get('best_metric', '?')}")

        saved_cfg = ckpt_meta.get("config", {})
        if saved_cfg:
            print("\nCheckpoint model config:")
            for k, v in saved_cfg.items():
                print(f"  {k}: {v}")
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        return

    # Step 4: 加载测试数据
    print("\nLoading test data...")
    from dataset.dataset import OODDataSet
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
    print(f"Test dataloader: {len(test_dataloader)} batches")

    # Step 5: 一次性跑全部 6 个版本
    print("\n" + "=" * 60)
    print("  能量图引导与筛选 - 多版本评估")
    print(f"  Dataset: {args.dataset}  |  Normal class: {args.normal}")
    print("=" * 60)
    print(f"\nRunning 6 versions...")
    print(f"  Gate params: k={args.gate_k}, Te={args.gate_te}, sigma={args.gate_sigma}")
    print(f"  Recon norm:  quantile=[{args.recon_norm_quantile_low}, {args.recon_norm_quantile_high}]")
    print(f"  Recon compress: {args.recon_compress}")
    print()

    results = run_all_versions(pfe, ae, discriminator, test_dataloader, device, args)

    # Step 6: 打印汇总表
    print_comparison_table(results)

    # Step 7: 打印结论
    version_order = [name for name, _ in VERSIONS]
    baseline = results.get('V1-Recon-only', {})
    baseline_pixel = baseline.get('Pixel', {}).get('AUROC', 0.0)
    baseline_pro = baseline.get('Pixel', {}).get('PRO', 0.0)
    baseline_image = baseline.get('Image', {}).get('AUROC', 0.0)

    print("\n" + "=" * 60)
    print("  结论分析（相对于 V1-Recon-only 基线）")
    print("=" * 60)
    print(f"  Baseline: Pixel-AUROC={baseline_pixel:.4f}  PRO={baseline_pro:.4f}  Image-AUROC={baseline_image:.4f}")
    print()

    improvements = []
    degradations = []
    for name in version_order[1:]:  # skip baseline
        m = results.get(name, {})
        pixel = m.get('Pixel', {}).get('AUROC', 0.0)
        pro = m.get('Pixel', {}).get('PRO', 0.0)
        image = m.get('Image', {}).get('AUROC', 0.0)
        delta_pixel = pixel - baseline_pixel
        delta_pro = pro - baseline_pro
        delta_image = image - baseline_image

        status = []
        if delta_pixel > 0: status.append(f"Pixel+{delta_pixel:.4f}")
        elif delta_pixel < 0: status.append(f"Pixel{delta_pixel:.4f}")
        if delta_pro > 0: status.append(f"PRO+{delta_pro:.4f}")
        elif delta_pro < 0: status.append(f"PRO{delta_pro:.4f}")
        if delta_image > 0: status.append(f"Image+{delta_image:.4f}")
        elif delta_image < 0: status.append(f"Image{delta_image:.4f}")

        print(f"  {name:<22}  delta: {', '.join(status)}")

        # 判断提升或下降
        has_improvement = (delta_pixel > 0 or delta_pro > 0 or delta_image > 0)
        has_degradation = (delta_pixel < -0.005 or delta_pro < -0.005 or delta_image < -0.005)
        if has_improvement and not has_degradation:
            improvements.append(name)
        elif has_degradation:
            degradations.append(name)

    print()
    if improvements:
        print(f"  [正信号] 有提升的版本: {', '.join(improvements)}")
    if degradations:
        print(f"  [负信号] 下降明显的版本: {', '.join(degradations)}")
    elif not improvements:
        print(f"  [信号弱] 无明显正信号，建议参考文档判断是否继续")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == '__main__':
    main()
