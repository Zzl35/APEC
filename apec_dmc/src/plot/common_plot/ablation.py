import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


FONTSIZE = 20
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['font.family'] = 'Arial'
sns.set_theme(style='darkgrid')

# 示例数据
data = {
    "Group": ["Mujoco"] * 3 + ["DMControl"] * 3,
    "Category": ["APEC", "wo w-criterion", "fewer policies"] * 2,
    "Bar Value": [0.6371, 0.4941, 0.5119, 0.7972, 0.7690, 0.7044],
}

df = pd.DataFrame(data)

sns.set_theme(style='darkgrid')
# 设置图形大小
fig, ax = plt.subplots(figsize=(11, 8))  # 可以调整图形的宽度

# 自定义颜色
palette = {"APEC": "skyblue", "wo w-criterion": "darkorange", "fewer policies": "firebrick"}

# 绘制柱状图，使用 Group 和 Category 进行分组
sns.barplot(
    x="Group",
    y="Bar Value",
    hue="Category",
    data=df,
    ax=ax,
    palette=palette,
    edgecolor="black",
    dodge=True,
    width=0.4  # 调整柱子宽度
)

# 设置标题和标签
ax.set_title("Ablation Study Results", fontsize=FONTSIZE)
ax.set_xlabel("Environment", fontsize=FONTSIZE+1)
ax.set_ylabel("Reward Correlation", fontsize=FONTSIZE)
ax.tick_params(labelsize=FONTSIZE)

# 获取当前的图例句柄和标签
handles, labels = ax.get_legend_handles_labels()

# 自定义图例标签
custom_labels = data['Category']

# 更新图例内容
ax.legend(
    handles,
    custom_labels,  # 使用自定义的标签
    fontsize=FONTSIZE-1,
    loc="upper left",  # 图例位置
    bbox_to_anchor=(0.01, 0.97), 
    borderaxespad=0,
    ncol=2
)

# 添加网格线
ax.grid(axis="y", linestyle="--", alpha=0.6)

# 调整布局
plt.tight_layout()

# 保存和展示图形
plt.savefig(
    "/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/all_ablation.pdf",
    bbox_inches="tight",
)
# plt.show()
plt.close()
