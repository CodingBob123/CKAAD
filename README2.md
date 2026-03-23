# Checkpoint 功能说明

本文档介绍模型训练 checkpoint（断点保存与加载）功能的使用方法。

---

## 功能概述

训练完成后，模型权重与优化器状态会自动保存到本地，后续无需重新训练即可直接加载模型进行推理或继续训练。

**预期存储空间**：约 **120 MB**（以 wide_resnet50_2 + MVtec 数据集为例）。

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `util/checkpoint.py` | Checkpoint 工具模块，含 7 个公共函数 |
| `inference_example.py` | 独立推理脚本示例 |
| `test_checkpoint.sh` | 快速功能验证脚本 |
| `mvtec.sh` | 主训练脚本（已补全 checkpoint 参数） |

---

## 新增命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_interval N` | `10` | 每 N 个 epoch 保存一次 checkpoint；设为 `0` 则仅保存 final 和 best |
| `--checkpoint_mode MODE` | `best` | 自动加载时的选择策略：`best`（最高 AUROC）/ `latest`（最新文件）/ `final`（训练结束） |
| `--resume` | — | 从 checkpoint 恢复训练（加载权重 + optimizer 状态），需配合 `--checkpoint_path` 使用 |
| `--checkpoint_path PATH` | `None` | 显式指定 checkpoint 文件路径；不指定则按 `--checkpoint_mode` 在默认目录中自动搜索 |
| `--skip_training` | — | 加载 checkpoint 后跳过训练，直接执行评估；需配合 `--checkpoint_path` 使用 |
| `--list_checkpoints` | — | 列出当前配置已有的所有 checkpoint 并退出（不训练） |

---

## 使用方式

### 方式一：用测试脚本一键验证（推荐首次使用）

```bash
bash test_checkpoint.sh           # 完整测试：训练 3 epoch → 列出 checkpoint → skip_training → resume
bash test_checkpoint.sh list      # 仅列出已有 checkpoint，不训练
bash test_checkpoint.sh skip      # 仅测试 skip_training（需已有 best.pth）
bash test_checkpoint.sh resume    # 仅测试 resume（需已有 best.pth）
```

测试默认使用 `mvtec/carpet` + `batch_size=4` + `3 epochs`，适合快速验证功能逻辑，无需跑完整训练。

### 方式二：手动单步操作

#### Step 1：训练并自动保存 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset mvtec --normal carpet --seed 111 \
    --epochs 200 --model wide_resnet50_2 --layer 1 2 3 \
    --enable_enhancement False --batch_size 16 \
    --lr 5e-03 --d_lr 1e-04 --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --labeled_anomaly_ratio 0.0 --labeled_anomaly_class_num 0 --labeled_anomaly_class 0 \
    --log_dir ./log --eval_epoch 8 --use_amp \
    --checkpoint_interval 10 --checkpoint_mode best
```

训练过程中会自动保存：
- `best.pth` — 最佳 AUROC 对应的 checkpoint
- `final.pth` — 训练结束时的 checkpoint
- `epoch_XXXX.pth` — 每 N 个 epoch 的定期 checkpoint

#### Step 2：列出已有的 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset mvtec --normal carpet --seed 111 \
    --epochs 1 --model wide_resnet50_2 --layer 1 2 3 \
    --enable_enhancement False --batch_size 16 \
    --lr 5e-03 --d_lr 1e-04 --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --labeled_anomaly_ratio 0.0 --labeled_anomaly_class_num 0 --labeled_anomaly_class 0 \
    --log_dir ./log --eval_epoch 1 --use_amp \
    --checkpoint_interval 999 --checkpoint_mode best --list_checkpoints
```

输出示例：

```
Found 4 checkpoint(s) in: ./checkpoints/mvtec/carpet/n_carpet_a_0_s_111/
  [best.pth]       epoch=83  best_image_auc=0.9832  120.3 MB  <-- BEST
  [final.pth]      epoch=200 best_image_auc=0.9819  120.3 MB  <-- FINAL
  [epoch_0010.pth] epoch=10  best_image_auc=0.9741  120.3 MB
  [epoch_0020.pth] epoch=20  best_image_auc=0.9788  120.3 MB
```

#### Step 3：跳过训练，直接加载 checkpoint 评估

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset mvtec --normal carpet --seed 111 \
    --epochs 1 --model wide_resnet50_2 --layer 1 2 3 \
    --enable_enhancement False --batch_size 16 \
    --lr 5e-03 --d_lr 1e-04 --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --labeled_anomaly_ratio 0.0 --labeled_anomaly_class_num 0 --labeled_anomaly_class 0 \
    --log_dir ./log --eval_epoch 1 --use_amp \
    --checkpoint_interval 999 --checkpoint_mode best \
    --skip_training \
    --checkpoint_path ./checkpoints/mvtec/carpet/n_carpet_a_0_s_111/best.pth
```

#### Step 4：从 checkpoint 恢复训练（继续训练）

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset mvtec --normal carpet --seed 111 \
    --epochs 300 --model wide_resnet50_2 --layer 1 2 3 \
    --enable_enhancement False --batch_size 16 \
    --lr 5e-03 --d_lr 1e-04 --adv_conf 0.02 \
    --gate_k 10.0 --gate_te 0.5 --gate_sigma 0.0 \
    --labeled_anomaly_ratio 0.0 --labeled_anomaly_class_num 0 --labeled_anomaly_class 0 \
    --log_dir ./log --eval_epoch 8 --use_amp \
    --checkpoint_interval 10 --checkpoint_mode best \
    --resume \
    --checkpoint_path ./checkpoints/mvtec/carpet/n_carpet_a_0_s_111/best.pth
```

程序会自动从 best.pth 的 epoch（第 200 epoch）继续训练到第 300 epoch，并保留 optimizer 状态。

#### Step 5：使用独立推理脚本

```bash
python inference_example.py --dataset mvtec --normal carpet --checkpoint_mode best
```

脚本会：
1. 自动搜索当前配置下最佳的 checkpoint
2. 加载模型权重
3. 在测试集上运行评估
4. 输出结果

---

## Checkpoint 保存位置

```
./checkpoints/
└── {dataset}/
    └── {normal}/
        └── n_{normal}_a_{labeled_anomaly_class}_s_{seed}/
            ├── best.pth        # 最佳 AUROC checkpoint
            ├── final.pth       # 训练结束 checkpoint
            └── epoch_XXXX.pth  # 定期保存的 checkpoint
```

例如：
```
./checkpoints/mvtec/carpet/n_carpet_a_0_s_111/best.pth
```

---

## Checkpoint 保存内容

每个 `.pth` 文件包含：

| 字段 | 说明 |
|------|------|
| `ae_state_dict` | ED 自编码器权重 |
| `discriminator_state_dict` | 判别器权重 |
| `ae_optimizer_state_dict` | AE 优化器状态（用于恢复训练） |
| `discriminator_optimizer_state_dict` | 判别器优化器状态 |
| `epoch` | 当前 epoch 数 |
| `best_epoch` | 最佳指标的 epoch 数 |
| `best_metric` | 最佳 Image AUROC 值 |
| `config` | 模型架构配置（用于兼容性校验） |
| `args` | 完整超参快照 |

---

## `mvtec.sh` 中的 checkpoint 配置

在脚本顶部新增了配置区，按需取消注释即可启用：

```bash
# checkpoint 配置（新增功能）
checkpoint_interval=10          # 每 N 个 epoch 保存一次；0=仅保存 final 和 best
checkpoint_mode=best            # 自动加载策略：best / latest / final
# resume=false                  # 取消注释以从 checkpoint 恢复训练
# skip_training=false           # 取消注释以跳过训练直接评估
# checkpoint_path=              # 显式指定 checkpoint 路径
```

---

## 公共函数 API（`util/checkpoint.py`）

```python
from util.checkpoint import (
    build_checkpoint_dir,   # 根据 args 构建 checkpoint 目录路径
    get_checkpoint_path,   # 在目录中搜索 best/latest/final checkpoint
    save_checkpoint,       # 保存 checkpoint（权重 + optimizer + 元数据）
    load_checkpoint,       # 加载 checkpoint（权重 + optimizer）
    load_weights_only,     # 仅加载模型权重（推理用）
    check_config_compatibility,  # 校验 checkpoint 与当前 args 的兼容性
    list_checkpoints,      # 列出目录下所有 checkpoint 及其元数据
    load_trained_model,     # 高层 API：一行代码初始化并加载完整模型
)
```

### `load_trained_model` 快速使用

```python
from util.checkpoint import load_trained_model

pfe, ae, discriminator, ckpt_meta = load_trained_model(
    args=args,                    # argparse Namespace（必须包含所有模型配置字段）
    device="cuda",                # 设备
    mode="best",                  # best / latest / final
    ckpt_path=None,               # 显式路径（可选）
)

# 直接使用 ae 和 discriminator 进行推理
ae.eval()
discriminator.eval()
with torch.no_grad():
    outputs = ae(encoder(img))
```
