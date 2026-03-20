# Checkpoint 保存/恢复示例说明

此文档说明如何在本仓库中保存与加载训练 checkpoint，以及一个简单的演示脚本用于生成示例 `.pth` 文件。

关键点：

- checkpoint 通常包含模型权重 (`state_dict`)、优化器状态、当前 epoch、命令行参数等元数据。
- 默认 PyTorch 使用 float32（每个参数 4 字节）存储权重；Adam 优化器会为每个被优化参数维护两个缓冲（exp_avg, exp_avg_sq），因此 optimizer 状态通常比模型权重更大。

估算（来自仓库模型，实际可能略有差异）：

- resnet18 配置（常见默认）：AE + Discriminator 权重约 21 MB；若包含 `pfe`（预训练特征提取器）则约 66 MB；若连同两个 Adam 优化器状态一起保存，总体约 64 MB（不含 `pfe`）或 108 MB（含 `pfe`）。
- wide_resnet50_2 配置（上界参考）：总体可达数百 MB，取决于是否保存 `pfe` 与 optimizer 状态。

推荐实践：

- 若仅用于推理（eval only），只保存并加载模型权重，避免保存 optimizer/state，以节省磁盘空间。
- 如需暂停并恢复训练，请保存 optimizer 状态和 AMP scaler（若使用混合精度）。
- 若想进一步减小体积，可在保存时将权重转换为 float16（注意加载后需谨慎处理训练行为）。

示例：生成 checkpoint 的脚本为 `tools/make_checkpoint_demo.py`。

如何运行（示例）：

```bash
# 生成默认示例 checkpoint（resnet18，包含 ae+disc 权重与 optimizer 状态）
/home/bobbystone/miniconda3/envs/CKAAD/bin/python tools/make_checkpoint_demo.py --save_dir ./checkpoints

# 只保存模型权重（不含 optimizer、pfe）
/home/bobbystone/miniconda3/envs/CKAAD/bin/python tools/make_checkpoint_demo.py --save_dir ./checkpoints --no_optimizer

# 同时包含预训练提取器 pfe（会增大文件）
/home/bobbystone/miniconda3/envs/CKAAD/bin/python tools/make_checkpoint_demo.py --save_dir ./checkpoints --include_pfe
```

生成的文件示例：

- `./checkpoints/checkpoint_epoch_000.pth`
- `./checkpoints/best.pth`

如果需要，我可以把 README1.md 中的估算值写入 `script/README_训练指南.md` 的保存/恢复章节，或将演示脚本改为更接近你实际训练流程的保存格式。
