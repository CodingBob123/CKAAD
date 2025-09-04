# CKAAD增强判别器

本仓库包含CKAAD模型的增强判别器实现，融合了多头注意力机制、谱归一化和位置编码，以提高异常检测性能。

## 主要特点

- **多头注意力**：使判别器能够捕捉特征之间的空间关系
- **谱归一化**：通过约束判别器的Lipschitz常数来稳定训练
- **位置编码**：为判别器增加空间感知能力
- **同时支持图像级和补丁级判别器**：为不同的异常检测任务提供灵活的架构

## 实现细节

我们实现了两种类型的判别器：

1. **图像级判别器**：为整个图像输出单一的标量能量值
2. **补丁级判别器**：输出具有每个补丁能量值的分数图

两种判别器都可以通过以下方式增强：
- 多头注意力机制，用于更好的特征表示
- 谱归一化，用于训练稳定性
- 位置编码，用于空间感知

## 使用方法

要使用增强的判别器，请运行`main_enhanced.py`而不是原始的`main.py`：

```bash
python main_enhanced.py --discriminator_mode image --use_attention --use_spectral_norm --use_position_encoding
```

### 命令行参数

- `--discriminator_mode`：选择'image'（图像级）或'patch'（补丁级）判别器
- `--use_spectral_norm`：启用谱归一化（默认：True）
- `--use_attention`：启用多头注意力（默认：True）
- `--use_position_encoding`：启用位置编码（默认：True）
- `--margin`：补丁级判别器中铰链损失的边界值（默认：5.0）

## 示例

### 带注意力的图像级判别器

```bash
python main_enhanced.py --discriminator_mode image --use_attention --use_spectral_norm --use_position_encoding
```

### 带注意力的补丁级判别器

```bash
python main_enhanced.py --discriminator_mode patch --use_attention --use_spectral_norm --use_position_encoding --margin 3.0
```

### 不带注意力的图像级判别器

```bash
python main_enhanced.py --discriminator_mode image --use_spectral_norm --use_position_encoding
```

## 理论背景

增强的判别器与原始CKAAD理论框架保持兼容：

- 对于图像级判别器：实现了论文中的公式(8)和(9)
- 对于补丁级判别器：实现了论文中的公式(11)和(12)

这些修改仅增强了判别器的函数类，而不改变对抗博弈的公式，保留了原始方法的理论保证。

## 实现说明

- 图像级判别器使用金字塔结构处理多尺度特征
- 补丁级判别器使用1×1卷积独立处理每个尺度
- 两种判别器都通过Softplus激活确保非负能量输出
- 对两种判别器类型都应用梯度惩罚以强制Lipschitz约束
- 在处理前将位置编码添加到输入特征中 