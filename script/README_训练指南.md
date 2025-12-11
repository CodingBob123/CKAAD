# CKAAD 训练脚本使用指南

## 📋 目录
- [模型配置验证](#模型配置验证)
- [快速测试](#快速测试)
- [完整训练](#完整训练)
- [张量形状说明](#张量形状说明)
- [常见问题](#常见问题)

---

## ✅ 模型配置验证

### 当前配置（已验证无误）

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **Backbone** | `wide_resnet50_2` | Wide ResNet-50 预训练模型 |
| **特征层** | `[1, 2, 3]` | 提取layer1、layer2、layer3的特征 |
| **图像尺寸** | `256×256` | 输入图像分辨率 |
| **批次大小** | `16` | 训练批次大小 |
| **Encoder** | `encoder_SENet.py` | 使用SE注意力机制 |

### 张量形状流程

```
输入图像: [B, 3, 256, 256]
    ↓
PretrainedFeatureExtractor
    ↓
特征图: [
    [B, 256, 64, 64],    # layer1
    [B, 512, 32, 32],    # layer2
    [B, 1024, 16, 16]    # layer3
]
    ↓
Encoder (SEAttention融合)
    ↓
编码特征: [B, 2048, 8, 8]
    ↓
Decoder
    ↓
重建特征: [
    [B, 256, 64, 64],
    [B, 512, 32, 32],
    [B, 1024, 16, 16]
]
    ↓
Loss Function (余弦相似度)
```

**✅ 所有张量形状完全对齐，无任何维度不匹配问题！**

---

## 🚀 快速测试

在开始完整训练前，建议先运行快速测试验证环境配置：

```bash
cd /home/bobbystone/CKAAD/script
./mvtec_test.sh
```

### 测试配置
- 类别: `bottle`
- 训练轮数: `2` （仅验证流程）
- 批次大小: `4` （加快测试速度）
- 预计时间: `2-5分钟`

### 测试通过标志
如果看到以下输出，说明模型配置正确：

```
✅ 模型测试成功！所有张量形状正确对齐！

📊 张量形状验证通过：
  ✅ PretrainedFeatureExtractor: 3个特征图
  ✅ Encoder: 特征对齐 + SEAttention融合
  ✅ Decoder: 3个重建特征图
  ✅ Loss Function: 余弦相似度损失
  ✅ Discriminator: 对抗训练
```

---

## 🎯 完整训练

### 训练所有MVTec类别

```bash
cd /home/bobbystone/CKAAD/script
./mvtec.sh
```

### 训练单个类别

编辑 `mvtec.sh`，修改第13行：

```bash
# 原始（训练所有类别）
categories=(
    'bottle' 'cable' 'capsule' 'carpet' 'grid'
    'hazelnut' 'leather' 'metal_nut' 'pill' 'screw'
    'tile' 'toothbrush' 'transistor' 'wood' 'zipper'
)

# 修改为只训练bottle
categories=('bottle')
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `mvtec` | 数据集名称 |
| `--model` | `wide_resnet50_2` | 预训练模型 |
| `--layer` | `1 2 3` | 提取的特征层 |
| `--img_size` | `256` | 图像尺寸 |
| `--batch_size` | `16` | 批次大小 |
| `--epochs` | `150-250` | 训练轮数（根据类别自动调整） |
| `--lr` | `5e-03` | 自编码器学习率 |
| `--d_lr` | `1e-04` | 判别器学习率 |
| `--adv_conf` | `0.02` | 对抗损失权重 |
| `--labeled_anomaly_ratio` | `0.05` | 标注异常样本比例 |
| `--seed` | `111` | 随机种子 |

### 不同类别的训练配置

脚本已针对不同类别优化：

#### 纹理类别（更多训练轮次）
- `carpet`, `grid`, `leather`, `tile`, `wood`
- 训练轮数: `200`
- 学习率: `8e-03`（carpet, grid）

#### 物体类别（标准配置）
- `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `toothbrush`, `zipper`
- 训练轮数: `150`
- 学习率: `5e-03`

#### 复杂类别（更多训练+小学习率）
- `screw`, `transistor`
- 训练轮数: `250`
- 学习率: `1e-03`
- 对抗权重: `0.03`

---

## 📊 张量形状说明

### 关键维度对应关系

#### 1. PretrainedFeatureExtractor 输出

| Layer | 输出形状 | 通道数 | 空间尺寸 |
|-------|---------|--------|---------|
| layer1 | `[B, 256, 64, 64]` | 256 | 64×64 |
| layer2 | `[B, 512, 32, 32]` | 512 | 32×32 |
| layer3 | `[B, 1024, 16, 16]` | 1024 | 16×16 |

#### 2. Encoder 特征对齐

```python
# 所有特征对齐到相同尺寸
conv_layers[0]: [B, 256, 64, 64]   → [B, 1024, 16, 16]
conv_layers[1]: [B, 512, 32, 32]   → [B, 1024, 16, 16]
conv_layers[2]: [B, 1024, 16, 16]  → [B, 1024, 16, 16]
```

#### 3. SEAttention 融合

```python
输入: 3个 [B, 1024, 16, 16]
处理: 加权融合（学习每个特征的重要性）
输出: [B, 1024, 16, 16]
```

#### 4. Encoder 最终输出

```python
encode_layer1: [B, 1024, 16, 16] → [B, 2048, 8, 8]
```

#### 5. Decoder 重建

```python
输入: [B, 2048, 8, 8]
输出: [
    [B, 256, 64, 64],    # 对应layer1
    [B, 512, 32, 32],    # 对应layer2
    [B, 1024, 16, 16]    # 对应layer3
]
```

#### 6. Loss 计算

```python
# 逐层计算余弦相似度损失
for i in range(3):
    loss += cosine_similarity(
        normal_inputs[i],   # 原始特征
        normal_outputs[i]   # 重建特征
    )
```

---

## ❓ 常见问题

### Q1: 为什么必须使用 `--layer 1 2 3`？

**A:** SEAttention 模块设计为融合3个特征图：

```python
def forward(self, x1, x2, x3):  # 需要3个输入
    x = x1 + x2 + x3
    ...
```

如果只提供1个或2个特征图会导致 `IndexError`。

---

### Q2: 可以使用其他图像尺寸吗？

**A:** 可以，但需要满足以下条件：

- 图像尺寸必须是32的倍数（因为有5次下采样：conv1 stride=2 + maxpool stride=2 + 3个layer）
- 推荐尺寸：`256`, `224`, `288`, `320`

修改方法：
```bash
--img_size 224  # 或其他32的倍数
```

---

### Q3: 显存不足怎么办？

**A:** 减小批次大小：

```bash
--batch_size 8   # 从16减到8
--batch_size 4   # 进一步减到4
```

或者使用较小的图像尺寸：
```bash
--img_size 224   # 从256减到224
```

---

### Q4: 如何切换到其他Encoder版本？

**A:** 修改 `model/model.py` 第4行：

```python
# 当前版本（SE注意力）
from model.encoder_SENet import Encoder

# 切换到坐标注意力前置版本
from model.encoder_CAfront import Encoder

# 切换到坐标注意力后置版本
from model.encoder_CAback import Encoder

# 切换到最新版本（CoordAtt + ECA）
from model.encoder import Encoder
```

**注意：** 切换到 `encoder.py` 时，`encode_layer1` 的输入通道会从 `1024` 变为 `3072`（3个分支拼接），但Decoder会自动适配，无需修改其他代码。

---

### Q5: 训练日志保存在哪里？

**A:** 日志保存路径：

```
./log/lan0.05_acn1/mvtec/n_{类别}_a_0_s_111.txt
```

例如：
```
./log/lan0.05_acn1/mvtec/n_bottle_a_0_s_111.txt
```

---

### Q6: 如何查看训练进度？

**A:** 实时查看日志：

```bash
tail -f ./log/lan0.05_acn1/mvtec/n_bottle_a_0_s_111.txt
```

---

## 📈 预期结果

### 训练输出示例

```
epoch [1/200], dis_loss: 2.345678, recon_loss:0.123456, adv_loss:0.012345, ae_loss: 0.135801
Valid: image_AUROC: 0.850000 pixel_AUROC: 0.920000 pixel_AUPRO: 0.880000
Test: image_AUROC: 0.870000 pixel_AUROC: 0.930000 pixel_AUPRO: 0.890000
```

### 性能指标说明

- **image_AUROC**: 图像级异常检测AUROC（越高越好，最大1.0）
- **pixel_AUROC**: 像素级异常定位AUROC（越高越好，最大1.0）
- **pixel_AUPRO**: 像素级异常定位AUPRO（越高越好，最大1.0）

### MVTec AD 基准性能

| 类别 | 预期 Image AUROC | 预期 Pixel AUROC |
|------|-----------------|-----------------|
| bottle | 0.95+ | 0.93+ |
| cable | 0.90+ | 0.92+ |
| capsule | 0.92+ | 0.94+ |
| carpet | 0.88+ | 0.95+ |
| grid | 0.90+ | 0.93+ |

---

## 🔧 故障排查

### 错误1: `IndexError: list index out of range`

**原因：** 使用的特征层数量不是3个

**解决：** 确保 `--layer 1 2 3`

---

### 错误2: `CUDA out of memory`

**原因：** 显存不足

**解决：** 
```bash
--batch_size 8   # 减小批次
--img_size 224   # 减小图像尺寸
```

---

### 错误3: 张量形状不匹配

**原因：** 可能使用了错误的Encoder版本

**解决：** 检查 `model/model.py` 第4行，确保导入正确的Encoder

---

## 📞 联系与支持

如有问题，请检查：
1. ✅ 是否使用 `--layer 1 2 3`
2. ✅ 是否使用 `wide_resnet50_2`
3. ✅ 图像尺寸是否是32的倍数
4. ✅ 数据集路径是否正确

---

**祝训练顺利！🎉**


