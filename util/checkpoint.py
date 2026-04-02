"""
Checkpoint management utilities for CKAAD model.
Supports saving/loading ED and Discriminator weights, optimizer state,
and configuration metadata.
"""
import os
import torch
import glob
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_checkpoint_dir(args) -> str:
    """
    Build the checkpoint directory path based on training arguments.

    Args:
        args: argparse Namespace with dataset, normal, labeled_anomaly_class, seed,
              and optionally checkpoint_dir for custom root.

    Returns:
        Absolute path to the checkpoint directory, e.g.
        ./checkpoints/mvtec/carpet/n_0_a_1_s_0/
        or /hy-tmp/checkpoints/mvtec/carpet/n_0_a_1_s_0/ if --checkpoint_dir is set.
    """
    # 支持通过命令行参数自定义 checkpoint 根目录
    ckpt_root = getattr(args, "checkpoint_dir", None) or "checkpoints"
    ckpt_dir = os.path.join(
        ckpt_root,
        args.dataset,
        args.normal,
        f"n_{args.normal}_a_{args.labeled_anomaly_class}_s_{args.seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def build_eerm_checkpoint_dir(args) -> str:
    """
    Build the checkpoint directory path for EERM module.

    EERM checkpoints are saved separately from CKAAD checkpoints:
        /hy-tmp/checkpoints/eerm/{dataset}/{normal}/n_{normal}_a_{class}_s_{seed}/

    Args:
        args: argparse Namespace with dataset, normal, labeled_anomaly_class, seed,
              and optionally checkpoint_dir for custom root.

    Returns:
        Absolute path to the EERM checkpoint directory, e.g.
        /hy-tmp/checkpoints/eerm/mvtec/wood/n_wood_a_0_s_111/
    """
    # 获取 checkpoint 根目录
    ckpt_root = getattr(args, "checkpoint_dir", None) or "checkpoints"
    # EERM 专用子目录
    ckpt_dir = os.path.join(
        ckpt_root,
        "eerm",                              # EERM 独立子目录
        args.dataset,
        args.normal,
        f"n_{args.normal}_a_{args.labeled_anomaly_class}_s_{args.seed}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def get_checkpoint_path(
    ckpt_dir: str,
    mode: str = "best",
    metric: str = "image_auc",
    higher_is_better: bool = True,
) -> Optional[str]:
    """
    Find the best or latest checkpoint under ckpt_dir.

    Args:
        ckpt_dir: Directory containing checkpoint files.
        mode: "best" to return the checkpoint with the highest/lowest metric,
              "latest" to return the most recently saved one,
              "final" to return the final checkpoint.
        metric: Metric key to compare when mode="best".
        higher_is_better: If True, best = highest metric value.

    Returns:
        Absolute path to the selected checkpoint, or None if nothing found.
    """
    if not os.path.isdir(ckpt_dir):
        return None

    # Collect all .pth files
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")))
    if not all_ckpts:
        return None

    # "final" keyword
    if mode == "final":
        candidates = [c for c in all_ckpts if "_final.pth" in c]
        if candidates:
            return candidates[-1]
        return None

    # "latest" keyword
    if mode == "latest":
        return all_ckpts[-1]

    # "best" keyword — requires checkpoint metadata
    if mode == "best":
        best_path = None
        best_val = -float("inf") if higher_is_better else float("inf")
        for ckpt_path in all_ckpts:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                val = ckpt.get("best_metric", -float("inf"))
                if higher_is_better:
                    if val > best_val:
                        best_val = val
                        best_path = ckpt_path
                else:
                    if val < best_val:
                        best_val = val
                        best_path = ckpt_path
            except Exception:
                continue
        return best_path

    return all_ckpts[-1]


def save_checkpoint(
    ae: torch.nn.Module,
    discriminator: torch.nn.Module,
    ae_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    args: Any,
    ckpt_dir: Optional[str] = None,
    filename: Optional[str] = None,
    save_best: bool = False,
    is_final: bool = False,
) -> str:
    """
    Save training state to disk.

    Args:
        ae: The autoencoder (ED) model.
        discriminator: The discriminator model.
        ae_optimizer: Optimizer for ae.
        discriminator_optimizer: Optimizer for discriminator.
        epoch: Current epoch number (1-based).
        metrics: Dict of metric values, e.g. {"image_auc": 0.92, "pixel_auc": 0.85}.
        args: argparse Namespace with model config fields.
        ckpt_dir: Directory to save to (auto-built from args if None).
        filename: Explicit filename (overrides auto-generated name).
        save_best: If True, also save as "best.pth".
        is_final: If True, also save as "final.pth".

    Returns:
        Path to the saved checkpoint file.
    """
    if ckpt_dir is None:
        ckpt_dir = build_checkpoint_dir(args)

    if filename is None:
        filename = f"epoch_{epoch:04d}.pth"

    ckpt_path = os.path.join(ckpt_dir, filename)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Build checkpoint dict
    ckpt = {
        # Model weights
        "ae_state_dict": ae.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        # Optimizer states (for resume training)
        "ae_optimizer_state_dict": ae_optimizer.state_dict(),
        "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
        # Metadata
        "epoch": epoch,
        "best_epoch": metrics.get("best_epoch", epoch),
        "best_metric": metrics.get("best_metric", 0.0),
        "metric_name": metrics.get("metric_name", "image_auc"),
        "metrics": metrics,
        # Model config (for compatibility check)
        "config": {
            "backbone": args.model,
            "input_channels": args.layer,
            "enable_enhancement": args.enable_enhancement
                if hasattr(args, "enable_enhancement")
                else [False, False, False],
            "feature2_fusion_weight": args.feature2_fusion_weight
                if hasattr(args, "feature2_fusion_weight")
                else 0.5,
        },
        # Full args snapshot
        "args": vars(args) if hasattr(args, "__dict__") else {},
    }

    torch.save(ckpt, ckpt_path, pickle_protocol=4)
    logger.info(f"Checkpoint saved to: {ckpt_path}")

    if save_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        torch.save(ckpt, best_path, pickle_protocol=4)
        logger.info(f"Best checkpoint saved to: {best_path}")

    if is_final:
        final_path = os.path.join(ckpt_dir, "final.pth")
        torch.save(ckpt, final_path, pickle_protocol=4)
        logger.info(f"Final checkpoint saved to: {final_path}")

    return ckpt_path


def load_checkpoint(
    ae: torch.nn.Module,
    discriminator: torch.nn.Module,
    ae_optimizer: Optional[torch.optim.Optimizer],
    discriminator_optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: str = "cuda",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load training state from disk.

    Args:
        ae: The autoencoder (ED) model to load weights into.
        discriminator: The discriminator model to load weights into.
        ae_optimizer: Optimizer for ae (weights restored if not None).
        discriminator_optimizer: Optimizer for discriminator (weights restored if not None).
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to map tensors to.
        strict: Passed to load_state_dict (strict=True raises on key mismatch).

    Returns:
        Checkpoint metadata dict (keys: epoch, best_epoch, best_metric, config, args, etc.)

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        RuntimeError: If model architecture does not match checkpoint.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore model weights
    ae.load_state_dict(ckpt["ae_state_dict"], strict=strict)
    discriminator.load_state_dict(ckpt["discriminator_state_dict"], strict=strict)

    # Restore optimizer states
    if ae_optimizer is not None and "ae_optimizer_state_dict" in ckpt:
        ae_optimizer.load_state_dict(ckpt["ae_optimizer_state_dict"])
    if discriminator_optimizer is not None and "discriminator_optimizer_state_dict" in ckpt:
        discriminator_optimizer.load_state_dict(ckpt["discriminator_optimizer_state_dict"])

    # Log summary
    loaded_epoch = ckpt.get("epoch", "?")
    best_epoch = ckpt.get("best_epoch", "?")
    best_metric = ckpt.get("best_metric", "?")
    logger.info(
        f"Checkpoint loaded — epoch={loaded_epoch}, best_epoch={best_epoch}, "
        f"best_metric={best_metric}"
    )

    return ckpt


def load_weights_only(
    ae: torch.nn.Module,
    discriminator: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cuda",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load model weights only (no optimizer state), useful for inference.

    Args:
        ae: The autoencoder (ED) model to load weights into.
        discriminator: The discriminator model to load weights into.
        checkpoint_path: Path to the .pth checkpoint file.
        device: Device to map tensors to.
        strict: Passed to load_state_dict.

    Returns:
        Checkpoint metadata dict.
    """
    return load_checkpoint(
        ae, discriminator,
        ae_optimizer=None,
        discriminator_optimizer=None,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=strict,
    )


def check_config_compatibility(
    saved_config: Dict[str, Any],
    args: Any,
    verbose: bool = True,
) -> bool:
    """
    Check whether the saved checkpoint's model config is compatible
    with the current args.

    Args:
        saved_config: The "config" dict stored in the checkpoint.
        args: argparse Namespace with current model config.
        verbose: Print mismatch details if True.

    Returns:
        True if compatible, False otherwise.
    """
    checks = [
        ("backbone", args.model),
        ("input_channels", args.layer),
        ("enable_enhancement", args.enable_enhancement
         if hasattr(args, "enable_enhancement")
         else [False, False, False]),
        ("feature2_fusion_weight", args.feature2_fusion_weight
         if hasattr(args, "feature2_fusion_weight")
         else 0.5),
    ]

    all_ok = True
    for key, current_val in checks:
        saved_val = saved_config.get(key)
        if saved_val != current_val:
            if verbose:
                logger.warning(
                    f"Config mismatch: '{key}' — checkpoint has {saved_val}, "
                    f"current args have {current_val}"
                )
            all_ok = False

    return all_ok


def list_checkpoints(ckpt_dir: str) -> List[Dict[str, Any]]:
    """
    List all checkpoints in a directory with their metadata.

    Args:
        ckpt_dir: Directory containing checkpoint files.

    Returns:
        List of dicts with keys: path, epoch, best_epoch, best_metric, filesize_mb.
    """
    if not os.path.isdir(ckpt_dir):
        return []

    result = []
    for ckpt_path in sorted(glob.glob(os.path.join(ckpt_dir, "*.pth"))):
        item = {
            "path": ckpt_path,
            "filename": os.path.basename(ckpt_path),
            "filesize_mb": os.path.getsize(ckpt_path) / (1024**2),
        }
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            item["epoch"] = ckpt.get("epoch", None)
            item["best_epoch"] = ckpt.get("best_epoch", None)
            item["best_metric"] = ckpt.get("best_metric", None)
            item["metric_name"] = ckpt.get("metric_name", "image_auc")
        except Exception:
            item["epoch"] = None
            item["best_epoch"] = None
            item["best_metric"] = None
        result.append(item)
    return result


# ---------------------------------------------------------------------------
# High-level convenience API
# ---------------------------------------------------------------------------

def load_trained_model(
    args,
    device: str = "cuda",
    mode: str = "best",
    ckpt_dir: str = None,
    ckpt_path: str = None,
) -> tuple:
    """
    High-level API to initialize and load a trained CKAAD model.

    This is the recommended entry point for inference scripts or notebooks.

    Steps:
        1. Build PretrainedFeatureExtractor and freeze it.
        2. Build ED and Discriminator.
        3. Load weights from checkpoint (best / latest / final / explicit path).
        4. Set models to eval() mode.

    Args:
        args: argparse Namespace with all model config fields
              (model, layer, enable_enhancement, feature2_fusion_weight,
               dataset, normal, labeled_anomaly_class, seed,
               log_dir, labeled_anomaly_ratio, labeled_anomaly_class_num).
        device: torch device string, e.g. "cuda" or "cpu".
        mode: Which checkpoint to load when ckpt_path is None.
              Options: "best" (default, highest AUROC), "latest", "final".
        ckpt_dir: Explicit checkpoint directory. If None, auto-built from args.
        ckpt_path: Explicit checkpoint file path. If None, auto-searched.

    Returns:
        (pfe, ae, discriminator, ckpt_meta):
            - pfe: PretrainedFeatureExtractor (frozen, eval mode).
            - ae: ED autoencoder (eval mode, loaded from checkpoint).
            - discriminator: Discriminator (eval mode, loaded from checkpoint).
            - ckpt_meta: checkpoint metadata dict (epoch, best_metric, config, etc.).
            Returns (pfe, ae, discriminator, None) if no checkpoint found.

    Raises:
        FileNotFoundError: No checkpoint found and no path provided.
    """
    import torch.nn as nn
    from model.model import PretrainedFeatureExtractor, ED, Discriminator

    # Resolve checkpoint path
    if ckpt_path is None:
        if ckpt_dir is None:
            ckpt_dir = build_checkpoint_dir(args)
        ckpt_path = get_checkpoint_path(ckpt_dir, mode=mode)

    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint found. "
            f"Tried path: {ckpt_path or '(auto)'}, "
            f"dir: {ckpt_dir or '(auto)'}. "
            f"Train the model first, then retry."
        )

    # 1. Build pretrained feature extractor (frozen)
    pfe = PretrainedFeatureExtractor(
        backbone=args.model,
        layers=args.layer,
        image_size=args.img_size if hasattr(args, "img_size") else 256,
    ).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)
    pfe.eval()

    # 2. Build ED and Discriminator
    ae = ED(
        backbone=args.model,
        input_channels=pfe.output_channels,
        enable_enhancement=args.enable_enhancement
            if hasattr(args, "enable_enhancement")
            else [False, False, False],
        feature2_fusion_weight=args.feature2_fusion_weight
            if hasattr(args, "feature2_fusion_weight")
            else 0.5,
    ).to(device)

    discriminator = Discriminator(
        input_sizes=pfe.output_sizes,
        input_channels=pfe.output_channels,
        expansion=pfe.expansion,
    ).to(device)

    # 3. Load weights
    ckpt_meta = load_weights_only(
        ae=ae,
        discriminator=discriminator,
        checkpoint_path=ckpt_path,
        device=device,
        strict=False,
    )

    # 4. Set to eval mode
    ae.eval()
    discriminator.eval()

    logger.info(
        f"load_trained_model: loaded epoch={ckpt_meta.get('epoch','?')}, "
        f"best_metric={ckpt_meta.get('best_metric','?')} from {ckpt_path}"
    )

    return pfe, ae, discriminator, ckpt_meta
