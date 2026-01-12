import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_scripts.make_pref_dataset import PreferenceBuffer
import matplotlib.lines as mlines

COLORS = sns.color_palette() # default

FONTSIZE = 20
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['font.family'] = 'Arial'
sns.set_theme(style='darkgrid')


def load_params(path):
    load_dict = {}
    with open(path, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict

def calculate_stats(corrs):
    stats = {}
    for dataset, models in corrs.items():
        stats[dataset] = {}
        for model, values in models.items():
            mean = np.mean(values)
            std = np.std(values)
            stats[dataset][model] = f"{mean:.4f} ± {std:.4f}"

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    model_means = stats_df.map(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    
    # Add the mean row to the DataFrame
    stats_df.loc['Mean'] = mean_row.astype(str)
    return stats_df

def calculate_stats_mean(corrs):
    stats = {}
    for dataset, models in corrs.items():
        stats[dataset] = {}
        for model, values in models.items():
            stats[dataset][model] = np.mean(values)
        
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.loc['Mean'] = stats_df.mean().round(3)
    return stats_df


def concat_figures(figures, savepath, labels, how='row'):
    n = len(figures)

    # 获取原始的 figsize
    original_figsize = figures[0].get_size_inches()  
    width, height = original_figsize  
    
    # 创建新的大图
    if how == 'row':
        fig, axs = plt.subplots(1, n, figsize=(width * n, height))
    elif how == 'col':
        fig, axs = plt.subplots(n, 1, figsize=(width, height * n))
    else:
        assert False

    for ax, figure in zip(axs, figures):
        # 清除每个 Axes 对象的图例
        for ax_fig in figure.get_axes():
            ax_fig.legend_.remove() if ax_fig.get_legend() else None
        
        # 保存当前 figure 到临时文件
        temp_path = 'temp_image.png'
        figure.savefig(temp_path, dpi=300)  # 使用高 DPI 保存

        # 读取保存的图像
        image = plt.imread(temp_path)  # 使用 plt.imread 读取图像

        # 在新的轴上显示图像
        ax.imshow(image)
        ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    # handles = [mlines.Line2D([], [], color=COLORS[i], label=labels[i], linewidth=2) for i, label in enumerate(labels)]
    # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(handles), fontsize=FONTSIZE)
    # plt.subplots_adjust(bottom=0.2)

    plt.savefig(savepath, bbox_inches='tight', dpi=300)  # 设置 dpi
    plt.close()

    # 删除临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)