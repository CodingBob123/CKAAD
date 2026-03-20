#!/usr/bin/env python3
"""
tools/make_checkpoint_demo.py

生成演示 checkpoint 文件（.pth），用于展示如何保存模型权重与 optimizer 状态。

用法示例：
  python tools/make_checkpoint_demo.py --save_dir ./checkpoints

"""
import os
import argparse
import torch
from model.model import PretrainedFeatureExtractor, ED, Discriminator


def convert_state_dict_dtype(state_dict, dtype=torch.float16):
    new = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            new[k] = v.to(dtype)
        elif isinstance(v, dict):
            # nested state (unlikely for model state_dict)
            new[k] = convert_state_dict_dtype(v, dtype)
        else:
            new[k] = v
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--include_pfe', action='store_true', help='also save pfe (pretrained feature extractor)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--no_optimizer', action='store_true', help='do not create/save optimizer state')
    parser.add_argument('--float16', action='store_true', help='save weights as float16 to reduce size')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = 'cpu'

    # instantiate models (pretrained=False to avoid downloads)
    pfe = PretrainedFeatureExtractor(backbone=args.backbone, pretrained=False, layers=[2], image_size=256).to(device)
    for p in pfe.parameters():
        p.requires_grad_(False)
    ae = ED(backbone=args.backbone, input_channels=pfe.output_channels, enable_enhancement=[False, False, False]).to(device)
    disc = Discriminator(input_sizes=pfe.output_sizes, input_channels=pfe.output_channels, expansion=pfe.expansion).to(device)

    # optionally create optimizers (to demo saving optimizer state)
    if not args.no_optimizer:
        ae_optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-5)
    else:
        ae_optimizer = None
        disc_optimizer = None

    # prepare checkpoint dict
    ckpt = {
        'epoch': 0,
        'ae_state_dict': ae.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'ae_optimizer': ae_optimizer.state_dict() if ae_optimizer is not None else None,
        'disc_optimizer': disc_optimizer.state_dict() if disc_optimizer is not None else None,
        'args': vars(args),
    }

    if args.include_pfe:
        ckpt['pfe_state_dict'] = pfe.state_dict()

    # optional float16 conversion (applied to saved copy)
    if args.float16:
        ckpt_to_save = {}
        for k, v in ckpt.items():
            if isinstance(v, dict) and 'weight' in ''.join(v.keys()):
                # heuristic: model state_dict
                ckpt_to_save[k] = convert_state_dict_dtype(v, dtype=torch.float16)
            else:
                try:
                    # try to convert any tensor values inside dicts
                    if isinstance(v, dict):
                        ckpt_to_save[k] = convert_state_dict_dtype(v, dtype=torch.float16)
                    else:
                        ckpt_to_save[k] = v
                except Exception:
                    ckpt_to_save[k] = v
    else:
        ckpt_to_save = ckpt

    out_path = os.path.join(args.save_dir, 'checkpoint_epoch_000.pth')
    best_path = os.path.join(args.save_dir, 'best.pth')
    torch.save(ckpt_to_save, out_path)
    torch.save(ckpt_to_save, best_path)

    # report sizes
    size_bytes = os.path.getsize(out_path)
    print(f'Saved checkpoint to: {out_path}  ({size_bytes/1024/1024:.2f} MB)')
    print(f'Also saved best -> {best_path}')


if __name__ == '__main__':
    main()
