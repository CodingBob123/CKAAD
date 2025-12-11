# Re-generate the comparison diagram with Chinese-friendly font settings.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# 配置中文字体显示
# 尝试多种可能的中文字体
plt.rcParams["font.sans-serif"] = [
    "SimHei",           # 黑体
    "Microsoft YaHei",  # 微软雅黑
    "WenQuanYi Zen Hei", # 文泉驿正黑
    "Noto Sans CJK SC", # 思源黑体
    "DejaVu Sans"       # 备用
]
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 8))

ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_flow(ax, y, steps, lane_title):
    box_w, box_h = 2.2, 1.0
    x = 0.7
    # Lane title on the left
    ax.text(0.1, y + box_h/2, lane_title, va='center', ha='left', fontsize=11, fontweight='bold')
    prev_center = None
    for s in steps:
        # Draw box
        rect = Rectangle((x, y), box_w, box_h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y + box_h/2, s, va='center', ha='center', fontsize=10, wrap=True)
        # Draw arrow from previous to current
        if prev_center is not None:
            arrow = FancyArrowPatch(prev_center, (x, y + box_h/2), arrowstyle='->', mutation_scale=12, linewidth=1.5)
            ax.add_patch(arrow)
        prev_center = (x + box_w, y + box_h/2)
        x += box_w + 0.8

# Define the four lanes
lanes = [
    (
        "SE 融合版（对齐后，用 SE 替代拼接）",
        ["x1, x2, x3", "conv_layers（对齐）", "SE 融合（三分支→单分支）", "encode_layer1"]
    ),
    (
        "CA 前置版（对齐前，每分支）",
        ["x1, x2, x3", "CoordAttention（逐分支）", "conv_layers（对齐）", "concat（拼接）", "encode_layer1"]
    ),
    (
        "CA 后置版（对齐后，每分支）",
        ["x1, x2, x3", "conv_layers（对齐）", "CoordAttention（逐分支）", "concat（拼接）", "encode_layer1"]
    ),
    (
        "CA + ECA 级联版（前置 CA + 拼接后 ECA）",
        ["x1, x2, x3", "CoordAttention（逐分支）", "conv_layers（对齐）", "concat（拼接）", "ECA（全局通道）", "encode_layer1"]
    ),
]

# Vertical positions for lanes (top to bottom)
ys = [7.5, 5.25, 3.0, 0.75]

for (title, steps), y in zip(lanes, ys):
    draw_flow(ax, y, steps, title)

# Title and legend-like notes
ax.text(0.1, 9.4, "Encoder 注意力插入位置对比（四种策略）", fontsize=15, fontweight='bold', ha='left', va='center')
ax.text(0.1, 8.8, "说明：每条横向流程从左到右表示数据流；方框为处理模块，箭头为处理顺序。", fontsize=10, ha='left', va='center')

# Save outputs
out_png = "./encoder_attention_compare_simhei.png"
out_pdf = "./encoder_attention_compare_simhei.pdf"
fig.tight_layout()
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
out_png, out_pdf
