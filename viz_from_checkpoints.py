"""
viz_from_checkpoints.py
========================
从已保存的 checkpoint 重新生成重建误差图与能量图可视化。

原理（方案A）：
    checkpoint 中保存的是纯重建误差训练的 AE 和 Discriminator 权重，
    重新评估时去掉 --recon_only，强制计算 D(input) 能量图，
    然后对 recon_maps 和 energy_maps 做统一可视化。

使用方法：
---------
# 1) 单类别，使用 best checkpoint：
python viz_from_checkpoints.py --normals capsule --checkpoint_mode best

# 2) 多类别，使用 latest checkpoint（默认）：
python viz_from_checkpoints.py --normals capsule carpet grid --checkpoint_mode latest

# 3) 指定具体 checkpoint 路径：
python viz_from_checkpoints.py --normals capsule --checkpoint_path /path/to/epoch_0080.pth

# 4) 列出某类别的所有 checkpoint：
python main.py --dataset mvtec --normal capsule --list_checkpoints

依赖：
    训练时使用的脚本 mvtec_day1.sh 中的所有参数必须与本脚本保持一致，
    特别是 --recon_only 的移除（这是方案A的核心）。
"""

import torch
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.checkpoint import load_trained_model, build_checkpoint_dir, list_checkpoints
from util.test import evaluation
from util.visualize_comparison import visualize_recon_energy_unified
from dataset.dataset import OODDataSet


def parse_args():
    parser = ArgumentParser(
        description="从 checkpoint 重新生成重建误差 vs 能量图可视化"
    )

    # ---------- 数据集与模型（与 mvtec_day1.sh 保持一致） ----------
    parser.add_argument('--normals', nargs='+', type=str, required=True,
                       help='要可视化的 MvTec 类别列表，如 capsule carpet grid')
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--model', type=str, default='wide_resnet50_2')
    parser.add_argument('--layer', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)

    # ---------- 实验配置（与 mvtec_day1.sh 保持一致） ----------
    parser.add_argument('--seed', type=int, default=111,
                        help='训练时的 seed，默认 111')
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.05)
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=1)
    parser.add_argument('--labeled_anomaly_class', type=int, default=0)

    # ---------- 能量融合参数 ----------
    # 注意：不传 --recon_only，这样评估时会计算 D(input) 能量图（方案A核心）
    parser.add_argument('--enable_enhancement', nargs='*',
                       type=lambda x: str(x).lower() in ('true', '1', 'yes', 't', 'y'),
                       default=[False, False, False])
    parser.add_argument('--feature2_fusion_weight', type=float, default=0.5)

    # 软门控参数（与 main.py 默认值一致）
    parser.add_argument('--gate_k', type=float, default=0.45)
    parser.add_argument('--gate_te', type=float, default=0.6)
    parser.add_argument('--gate_sigma', type=float, default=0.0)
    parser.add_argument('--recon_norm_quantile_low', type=float, default=0.00)
    parser.add_argument('--recon_norm_quantile_high', type=float, default=0.98)
    parser.add_argument('--recon_compress', type=str, default='sqrt',
                        choices=['sqrt', 'log', 'none'])
    parser.add_argument('--no_fuse_output_norm', action='store_false', dest='fuse_output_norm',
                        default=True)

    # ---------- Checkpoint 选择 ----------
    parser.add_argument('--checkpoint_mode', type=str, default='latest',
                        choices=['best', 'latest', 'final'],
                        help='best: 最高 AUROC，latest: 最近保存，final: _final.pth')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='显式指定 checkpoint 路径（覆盖 --checkpoint_mode）')
    # checkpoint 存储路径需与训练时一致
    parser.add_argument('--checkpoint_dir', type=str, default='/hy-tmp/checkpoints',
                        help='checkpoint 根目录，需与训练时 --checkpoint_dir 一致')

    # ---------- 日志与输出 ----------
    parser.add_argument('--log_dir', type=str, default='./log_day1',
                        help='日志目录，需与训练时 --log_dir 一致')
    parser.add_argument('--viz_output_root', type=str, default='/hy-tmp/results',
                        help='可视化结果输出根目录')
    parser.add_argument('--topk', type=int, default=100)

    # ---------- 可视化选项 ----------
    parser.add_argument('--viz_samples_per_type', type=int, default=3,
                        help='每种类型（正常/异常）可视化样本数量')

    return parser.parse_args()


def run_viz_for_category(args, normal, device):
    """
    对单个类别执行：加载 checkpoint → 评估 → 可视化
    """
    print(f"\n{'=' * 60}")
    print(f"[viz_from_checkpoints] Processing category: {normal}")
    print('=' * 60)

    # ---------- 1. 构建 checkpoint 目录并列出已有 checkpoint ----------
    # 注意：这里用 checkpoint_dir 覆盖 args 的 log_dir 来构建 ckpt_dir
    original_log_dir = args.log_dir
    args.log_dir = args.checkpoint_dir   # build_checkpoint_dir 内部用 args.log_dir 拼接路径
    ckpt_dir = build_checkpoint_dir(args)
    args.log_dir = original_log_dir      # 恢复

    print(f"  Checkpoint directory: {ckpt_dir}")

    # 如果没有指定 checkpoint_path，尝试自动查找
    if args.checkpoint_path is None:
        ckpts = list_checkpoints(ckpt_dir)
        if not ckpts:
            print(f"  [SKIP] No checkpoints found in: {ckpt_dir}")
            return None
        print(f"  Found {len(ckpts)} checkpoint(s):")
        for ck in ckpts:
            marker = " <-- BEST" if "best.pth" in ck["filename"] else ""
            marker = " <-- FINAL" if "final.pth" in ck["filename"] else marker
            print(f"    [{ck['filename']}]  epoch={ck['epoch']}  "
                  f"best_{ck['metric_name']}={ck['best_metric']:.4f}  "
                  f"{ck['filesize_mb']:.1f} MB{marker}")

    # ---------- 2. 加载模型（仅权重，不加载 optimizer） ----------
    print(f"\n  Loading model from checkpoint (mode='{args.checkpoint_mode}')...")
    try:
        pfe, ae, discriminator, ckpt_meta = load_trained_model(
            args=args,
            device=device,
            mode=args.checkpoint_mode,
            ckpt_path=args.checkpoint_path,
            ckpt_dir=ckpt_dir,
        )
        print(f"  Loaded checkpoint: epoch={ckpt_meta.get('epoch', '?')}, "
              f"best_metric={ckpt_meta.get('best_metric', '?')}")

        # 打印保存时的配置信息
        saved_cfg = ckpt_meta.get("config", {})
        if saved_cfg:
            print("  Checkpoint model config:")
            for k, v in saved_cfg.items():
                print(f"    {k}: {v}")

    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return None

    # ---------- 3. 加载测试数据 ----------
    print("\n  Loading test dataset...")
    dataset = OODDataSet(
        root='./data',
        dataset=args.dataset,
        image_size=args.img_size,
        category=normal,
        labeled_anomaly_ratio=args.labeled_anomaly_ratio,
        labeled_anomaly_class_num=args.labeled_anomaly_class_num,
        labeled_anomaly_class=args.labeled_anomaly_class,
    )

    train_loader, val_loader, test_loader, _ = dataset.get_data_loader(
        batch_size=args.batch_size
    )
    print(f"  Test dataset loaded: {len(test_loader.dataset)} samples")

    # ---------- 4. 评估（关键：不使用 recon_only，获取 recon_maps + energy_maps） ----------
    print("\n  Running evaluation (with energy maps, recon_only=False)...")
    # evaluation 内部根据 args.recon_only 决定是否计算能量图
    # 因为本脚本没有传 --recon_only，args.recon_only 为 False，会走能量融合分支
    metrics, recon_maps, energy_maps, _, _, final_maps, _, anomaly_maps = evaluation(
        pfe, ae, discriminator, test_loader, device, args,
        return_maps=True,
        recon_only=False,        # 方案A核心：强制计算能量图
    )

    # ---------- 5. 打印评估指标 ----------
    print("\n  Evaluation metrics:")
    for level, scores in metrics.items():
        print(f"    [{level}]")
        for metric_name, value in scores.items():
            print(f"      {metric_name}: {value:.6f}")

    # ---------- 6. 统一可视化（重建误差 vs 能量图） ----------
    print("\n  Generating reconstruction vs energy visualization...")

    # 构造 cached_maps，与 run_eval_and_viz 保持一致
    cached_maps = {
        'recon_maps': recon_maps,
        'energy_maps': energy_maps,
        'final_maps': final_maps,
        'anomaly_maps': final_maps,
    }

    # 调用统一可视化函数
    viz_result_path = os.path.join(
        args.viz_output_root,
        f"{args.dataset}_{normal}_viz_from_ckpt"
    )
    os.makedirs(viz_result_path, exist_ok=True)

    # 注意：visualize_recon_energy_unified 输出路径由其内部决定，
    # 这里用 viz_samples_per_type 限制样本数
    visualize_recon_energy_unified(
        encoder=pfe,
        ed=ae,
        discriminator=discriminator,
        dataloader=test_loader,
        args=args,
        device=device,
        epochs=ckpt_meta.get('epoch', 0),
        cached_maps=cached_maps,
        enable_stats=True,
    )

    print(f"\n  Visualization completed for: {normal}")
    print(f"  Results saved to: {viz_result_path}")

    return {
        'normal': normal,
        'epoch': ckpt_meta.get('epoch', '?'),
        'metrics': metrics,
    }


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Visualization output root: {args.viz_output_root}")
    os.makedirs(args.viz_output_root, exist_ok=True)

    results = []
    for normal in args.normals:
        result = run_viz_for_category(args, normal, device)
        if result:
            results.append(result)

    # ---------- 汇总打印 ----------
    print("\n" + "=" * 70)
    print("Summary: Reconstruction vs Energy Visualization from Checkpoints")
    print("=" * 70)
    if not results:
        print("  No categories processed successfully.")
    else:
        print(f"  Processed {len(results)} / {len(args.normals)} category(ies)")
        print()
        print(f"  {'Category':<15} {'Epoch':<8} {'Image AUROC':<14} {'Pixel AUROC'}")
        print(f"  {'-'*15} {'-'*8} {'-'*14} {'-'*12}")
        for r in results:
            img_auc = r['metrics'].get('Image', {}).get('AUROC', 'N/A')
            pix_auc = r['metrics'].get('Pixel', {}).get('AUROC', 'N/A')
            img_str = f"{img_auc:.4f}" if isinstance(img_auc, float) else str(img_auc)
            pix_str = f"{pix_auc:.4f}" if isinstance(pix_auc, float) else str(pix_auc)
            print(f"  {r['normal']:<15} {str(r['epoch']):<8} {img_str:<14} {pix_str}")

    print("\nAll done.")


if __name__ == '__main__':
    main()
