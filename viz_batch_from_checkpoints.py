"""
viz_batch_from_checkpoints.py
==============================
批量从某个类别目录下的所有 checkpoint 重新生成重建误差图与能量图可视化。

使用场景：
    mvtec.sh 训练时每 20 轮保存一次 checkpoint，训练完成后可以用本脚本
    一次性生成所有 checkpoint 对应的可视化结果，便于对比不同训练阶段的
    recon vs energy 效果。

使用方法：
---------
# 1) 批量处理单个类别所有 checkpoint（默认 latest）：
python viz_batch_from_checkpoints.py --normals capsule

# 2) 只处理前 3 个 checkpoint（调试用）：
python viz_batch_from_checkpoints.py --normals capsule --max_checkpoints 3

# 3) 指定具体 checkpoint 目录：
python viz_batch_from_checkpoints.py --normals capsule \
    --checkpoint_dir /hy-tmp/checkpoints/mvtec/capsule/n_False_a_0_s_111

# 4) 列出某类别的所有 checkpoint 信息：
python main.py --dataset mvtec --normal capsule --list_checkpoints

依赖：
    训练时使用的脚本 mvtec_day1.sh 中的所有参数必须与本脚本保持一致。
    本脚本会遍历目录下所有 epoch_*.pth 文件，按 epoch 顺序逐一处理。
"""

import torch
import os
import sys
import glob
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from util.checkpoint import load_trained_model, build_checkpoint_dir, list_checkpoints
from util.test import evaluation, evaluation_per_layer
from util.visualize_comparison import visualize_recon_energy_unified
from dataset.dataset import OODDataSet


@dataclass
class CheckpointInfo:
    """单个 checkpoint 的元信息"""
    path: str
    filename: str
    epoch: int
    filesize_mb: float
    is_best: bool
    is_final: bool


def parse_args():
    parser = ArgumentParser(
        description="批量从 checkpoint 生成重建误差 vs 能量图可视化"
    )

    # ---------- 数据集与模型（与 mvtec_day1.sh / mvtec.sh 保持一致） ----------
    parser.add_argument('--normals', nargs='+', type=str, required=True,
                       help='要可视化的 MvTec 类别列表，如 capsule carpet')
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--model', type=str, default='wide_resnet50_2')
    parser.add_argument('--layer', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)

    # ---------- 实验配置（与 mvtec_day1.sh 保持一致） ----------
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.05)
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=1)
    parser.add_argument('--labeled_anomaly_class', type=int, default=0)
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

    # ---------- Checkpoint 目录 ----------
    # 需要与训练时 --checkpoint_dir 一致（这里是根目录，build_checkpoint_dir 会自动拼接完整路径）
    parser.add_argument('--checkpoint_dir', type=str, default='/hy-tmp/checkpoints',
                        help='checkpoint 根目录，需与训练时 --checkpoint_dir 一致')

    # ---------- 日志与输出 ----------
    parser.add_argument('--log_dir', type=str, default='./log_day1',
                        help='日志目录，需与训练时 --log_dir 一致')
    parser.add_argument('--viz_output_root', type=str, default='./viz_results/grid',
                        help='可视化结果输出根目录')
    parser.add_argument('--load_test_only', action='store_true', default=True,
                        help='只加载测试集（跳过 ./data/mvtec/xxx/train 目录）')

    # ---------- 批量处理选项 ----------
    parser.add_argument('--max_checkpoints', type=int, default=0,
                        help='最多处理多少个 checkpoint；0 表示处理全部')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='如果可视化结果已存在则跳过（根据输出目录判断）')
    parser.add_argument('--topk', type=int, default=100)

    return parser.parse_args()


def get_epoch_from_filename(filename: str) -> int:
    """从 checkpoint 文件名提取 epoch 编号，如 epoch_0080.pth -> 80"""
    import re
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def list_all_checkpoints(ckpt_dir: str) -> List[CheckpointInfo]:
    """
    列出目录下所有 checkpoint，按 epoch 排序。
    排除 best.pth 和 final.pth（它们是软链接或复制，无需重复处理）。
    """
    if not os.path.isdir(ckpt_dir):
        return []

    all_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    checkpoints = []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        # 跳过 best.pth / final.pth / symlink 指向的文件
        if fname in ('best.pth', 'final.pth'):
            continue
        if '_symlink' in fname:
            continue

        epoch = get_epoch_from_filename(fname)
        if epoch < 0:
            continue

        size_mb = os.path.getsize(fpath) / (1024 ** 2)
        is_best = os.path.islink(fpath) or 'best' in os.readlink(fpath).lower() if os.path.islink(fpath) else False
        is_final = 'final' in fname.lower()

        checkpoints.append(CheckpointInfo(
            path=fpath,
            filename=fname,
            epoch=epoch,
            filesize_mb=size_mb,
            is_best=is_best,
            is_final=is_final,
        ))

    # 按 epoch 排序
    checkpoints.sort(key=lambda x: x.epoch)
    return checkpoints


def check_viz_output_exists(args, normal: str, epoch: int) -> bool:
    """检查某 epoch 的可视化结果是否已存在（用于 --skip_existing）"""
    # visualize_recon_energy_unified 的输出路径由其内部决定
    # 这里简化为检查结果根目录下是否存在该类别的 epoch 子目录
    # （如果函数内部没有自动去重，这个检查可能不完全准确）
    search_pattern = os.path.join(args.viz_output_root, f"{args.dataset}_{normal}*epoch*{epoch}*")
    matches = glob.glob(search_pattern)
    return len(matches) > 0


def run_viz_for_single_checkpoint(
    args,
    normal: str,
    ckpt_info: CheckpointInfo,
    device: str,
) -> Optional[dict]:
    """
    对单个 checkpoint 执行：加载 → 评估 → 可视化
    返回结果 dict，失败返回 None
    """
    print(f"\n  [{ckpt_info.filename}]  epoch={ckpt_info.epoch}  {ckpt_info.filesize_mb:.1f} MB")

    # ---------- 1. 加载模型 ----------
    try:
        pfe, ae, discriminator, ckpt_meta = load_trained_model(
            args=args,
            device=device,
            mode='latest',          # 使用具体路径加载，mode 不影响
            ckpt_path=ckpt_info.path,
        )
    except Exception as e:
        print(f"    [SKIP] Failed to load checkpoint: {e}")
        return None

    # ---------- 2. 加载测试数据（只加载一次，放在外层循环更高效） ----------
    # 注意：这个放在外层循环中统一加载，放在这里仅作备用
    return {
        'normal': normal,
        'epoch': ckpt_info.epoch,
        'filename': ckpt_info.filename,
        'path': ckpt_info.path,
        'loaded': True,
    }


def run_batch_for_category(
    args,
    normal: str,
    device: str,
    test_loader,
) -> List[dict]:
    """
    对单个类别的所有 checkpoint 批量执行可视化。
    """
    print(f"\n{'=' * 70}")
    print(f"[viz_batch] Category: {normal}")
    print('=' * 70)

    # ---------- 构建 checkpoint 目录 ----------
    # build_checkpoint_dir 期望 args.normal（单数），但命令行参数是 --normals
    original_normal = getattr(args, 'normal', None)
    original_log_dir = args.log_dir
    args.normal = normal  # 设置 normal 属性
    args.log_dir = args.checkpoint_dir
    ckpt_dir = build_checkpoint_dir(args)
    args.log_dir = original_log_dir
    args.normal = original_normal

    print(f"  Checkpoint directory: {ckpt_dir}")

    if not os.path.isdir(ckpt_dir):
        print(f"  [SKIP] Directory does not exist: {ckpt_dir}")
        return []

    # ---------- 列出所有 checkpoint ----------
    checkpoints = list_all_checkpoints(ckpt_dir)

    if not checkpoints:
        print(f"  [SKIP] No epoch checkpoints found in: {ckpt_dir}")
        print(f"  (best.pth and final.pth are excluded from batch processing)")
        return []

    # ---------- 按 max_checkpoints 截断 ----------
    total = len(checkpoints)
    if args.max_checkpoints > 0 and args.max_checkpoints < total:
        checkpoints = checkpoints[:args.max_checkpoints]
        print(f"  Truncated to first {args.max_checkpoints} checkpoints (of {total} total)")

    print(f"  Found {total} checkpoint(s), processing {len(checkpoints)}...")
    print(f"  Epochs: {[ck.epoch for ck in checkpoints]}")

    # ---------- 批量处理每个 checkpoint ----------
    results = []
    failed = []

    for i, ckpt_info in enumerate(checkpoints):
        print(f"\n  [{i+1}/{len(checkpoints)}]", end="")

        # --skip_existing 检查
        if args.skip_existing and check_viz_output_exists(args, normal, ckpt_info.epoch):
            print(f"  [SKIP existing] epoch={ckpt_info.epoch}")
            results.append({
                'normal': normal,
                'epoch': ckpt_info.epoch,
                'filename': ckpt_info.filename,
                'status': 'skipped_existing',
            })
            continue

        try:
            # ---------- 加载模型 ----------
            pfe, ae, discriminator, ckpt_meta = load_trained_model(
                args=args,
                device=device,
                mode='latest',
                ckpt_path=ckpt_info.path,
            )

            # ---------- 评估（关键：不使用 recon_only，获取 recon_maps + energy_maps） ----------
            print(f"\n    Evaluating epoch={ckpt_info.epoch}...", end=" ", flush=True)
            metrics, recon_maps, energy_maps, _, _, final_maps, _, anomaly_maps = evaluation(
                pfe, ae, discriminator, test_loader, device, args,
                return_maps=True,
                recon_only=False,   # 方案A核心：强制计算能量图
            )
            print("done")

            # ---------- 打印指标 ----------
            img_auc = metrics.get('Image', {}).get('AUROC', None)
            pix_auc = metrics.get('Pixel', {}).get('AUROC', None)
            if img_auc is not None:
                pix_str = f"{pix_auc:.4f}" if pix_auc is not None else 'N/A'
                print(f"    Metrics: Image_AUROC={img_auc:.4f}, Pixel_AUROC={pix_str}")

            # ---------- 按层级评估能量图 ----------
            print(f"    Evaluating per-layer energy maps...", end=" ", flush=True)
            layer_metrics = evaluation_per_layer(
                pfe, ae, discriminator, test_loader, device, args
            )
            print("done")
            for layer_idx, m in layer_metrics.items():
                print(f"      Layer {layer_idx}: Image_AUROC={m['Image_AUROC']:.4f}, "
                      f"Pixel_AUROC={m['Pixel_AUROC']:.4f}, Pixel_PRO={m['Pixel_PRO']:.4f}")

            # ---------- 可视化 ----------
            print(f"    Generating visualization...", end=" ", flush=True)
            cached_maps = {
                'recon_maps': recon_maps,
                'energy_maps': energy_maps,
                'final_maps': final_maps,
                'anomaly_maps': anomaly_maps,
            }

            # 设置输出子目录：results/{dataset}_{normal}_epoch{epoch:04d}
            # 注意：visualize_recon_energy_unified 内部会创建目录
            # 这里通过修改一个临时属性来控制输出路径（如果有的话）
            # 如果没有这个机制，结果会保存在默认路径
            args._current_viz_epoch = ckpt_info.epoch

            visualize_recon_energy_unified(
                encoder=pfe,
                ed=ae,
                discriminator=discriminator,
                dataloader=test_loader,
                args=args,
                device=device,
                epochs=ckpt_info.epoch,
                cached_maps=cached_maps,
                enable_stats=False,   # 批量处理时关闭详细统计输出
                output_root=args.viz_output_root,  # 传递自定义输出根目录
            )

            print("done")

            results.append({
                'normal': normal,
                'epoch': ckpt_info.epoch,
                'filename': ckpt_info.filename,
                'status': 'success',
                'metrics': metrics,
            })

        except Exception as e:
            print(f"\n    [ERROR] epoch={ckpt_info.epoch}: {e}")
            import traceback
            traceback.print_exc()
            failed.append({
                'normal': normal,
                'epoch': ckpt_info.epoch,
                'filename': ckpt_info.filename,
                'status': 'failed',
                'error': str(e),
            })

    return results


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Visualization output root: {args.viz_output_root}")
    os.makedirs(args.viz_output_root, exist_ok=True)

    all_results = []
    all_failed = []

    for normal in args.normals:
        # ---------- 加载测试数据（每个类别只加载一次） ----------
        print(f"\nLoading test dataset for: {normal}")
        try:
            dataset = OODDataSet(
                root='./data',
                dataset=args.dataset,
                image_size=args.img_size,
                category=normal,
                labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                labeled_anomaly_class=args.labeled_anomaly_class,
                load_test_only=args.load_test_only,
            )
            # get_data_loader 返回顺序: (train, valid, anomaly, test)
            if args.load_test_only:
                test_loader = dataset.get_data_loader(batch_size=args.batch_size)
            else:
                _, _, _, test_loader = dataset.get_data_loader(batch_size=args.batch_size)
            print(f"  Test samples: {len(test_loader.dataset)}")
        except Exception as e:
            print(f"  [SKIP] Failed to load dataset: {e}")
            continue

        # ---------- 批量处理该类别的所有 checkpoint ----------
        results = run_batch_for_category(args, normal, device, test_loader)
        all_results.extend(results)

        # 统计失败
        failed = [r for r in results if r.get('status') == 'failed']
        all_failed.extend(failed)

    # ---------- 汇总打印 ----------
    print("\n" + "=" * 70)
    print("Batch Visualization Summary")
    print("=" * 70)

    success = [r for r in all_results if r.get('status') == 'success']
    skipped = [r for r in all_results if r.get('status') == 'skipped_existing']

    print(f"\n  Total processed: {len(all_results)}")
    print(f"    Success: {len(success)}")
    print(f"    Skipped (existing): {len(skipped)}")
    print(f"    Failed: {len(all_failed)}")

    if all_results:
        print(f"\n  {'Category':<15} {'Epoch':<8} {'Filename':<25} {'Status'}")
        print(f"  {'-'*15} {'-'*8} {'-'*25} {'-'*10}")
        for r in all_results:
            status = r.get('status', 'unknown')
            img_auc = ''
            if status == 'success' and 'metrics' in r:
                img = r['metrics'].get('Image', {}).get('AUROC')
                if img is not None:
                    img_auc = f"(AUROC={img:.4f})"
            print(f"  {r['normal']:<15} {str(r['epoch']):<8} {r['filename']:<25} {status} {img_auc}")

    if all_failed:
        print(f"\n  Failed checkpoints:")
        for f in all_failed:
            print(f"    [{f['normal']}] epoch={f['epoch']}: {f['error']}")

    print(f"\n  Output directory: {args.viz_output_root}")
    print("\nAll done.")


if __name__ == '__main__':
    main()
