import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from data_scripts.make_pref_dataset import PreferenceBufferFeature
import pickle
import hydra
from plot_scripts.plot_utils import *
from plot_scripts.plot_configs import *

def set_model_hypeparams(method_name):    
    '''
        load_buffer_name: 用了谁的buffer
        load_model_name: reward model存在谁的log里面
    '''
    return model_hypeparams[method_name]

def get_trajs_from_buffer(path):
    with open(path, 'rb') as f:
        trajs_ = pickle.load(f)

    trajs = []
    for traj in trajs_:
        trajs.extend(traj[-1])
    
    states = np.concatenate([traj[0] for traj in trajs], axis=0)
    rewards = np.concatenate([traj[2] for traj in trajs], axis=0)
    return states, rewards

def get_states_from_states(states, size):
    states = states.reshape(-1, states.shape[-1])
    size = size if size > 0 else states.shape[0]
    random_indices = np.random.choice(states.shape[0], size=size, replace=False)
    return states[random_indices]

def plot_density_heatmap(axes, buffers, targets, dataset, methods, x_min, x_max, y_min, y_max, fontsize):
    for i, (states, method_name) in enumerate(zip(buffers, methods)):
        x, y = targets[dataset]
        x_data = states[:, x]
        y_data = states[:, y]
        cbar = axes[i].hexbin(
            x_data, y_data, gridsize=30, mincnt=1, cmap='jet',
            extent=(x_min, x_max, y_min, y_max)
        )
        # axes[i].set_title(method2label[method_name], fontsize=fontsize + 1)
        axes[i].tick_params(labelsize=fontsize)
    return cbar

def process_buffers_for_env(dataset, methods, targets, size=5000):
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf
    buffers = []

    for method_name in methods:
        _, load_buffer_name, _ = set_model_hypeparams(method_name)
        load_buffer_path = os.path.join(
            '/home/ubuntu/duxinghao/imitation_pref/buffer_qposvel',
            dataset, load_buffer_name, '0_0_1_0', '0', 'buffer.pkl'
        )
        states, _ = get_trajs_from_buffer(load_buffer_path)
        buffers.append(get_states_from_states(states, size=size))

        x, y = targets[dataset]
        x_max, x_min = max(x_max, states[:, x].max()), min(x_min, states[:, x].min())
        y_max, y_min = max(y_max, states[:, y].max()), min(y_min, states[:, y].min())

    return buffers, (x_min, x_max, y_min, y_max)

def plot_dmc_tasks(ax, buffers, methods, fontsize):
    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf
    datas = []

    for i, buffer in enumerate(buffers):
        x_data, y_data = [], []
        loader = torch.utils.data.DataLoader(buffer, batch_size=8, shuffle=False, num_workers=8)
        with tqdm(total=10000) as pbar:
            pbar.set_description(f'Plotting trajectories for {methods[i]}')
            for batch in loader:
                states1, states2, *_ = batch
                trajectories = torch.cat([states1, states2], dim=0)
                x_data.append(trajectories[:, :, 1].cpu().numpy()) # walker_walk: (1, 10)
                y_data.append(trajectories[:, :, 10].cpu().numpy()) 
                # break
                pbar.update(states1.shape[0]*2)

        x_data = np.concatenate(x_data, axis=0).flatten()
        y_data = np.concatenate(y_data, axis=0).flatten()
        x_max, x_min = max(x_max, x_data.max()), min(x_min, x_data.min())
        y_max, y_min = max(y_max, y_data.max()), min(y_min, y_data.min())
        datas.append((x_data, y_data))

    for i, (x_data, y_data) in enumerate(datas):
        cbar = ax[i].hexbin(
            x_data, y_data, gridsize=50, mincnt=1, cmap='jet',
            extent=(x_min, x_max, y_min, y_max)
        )
        ax[i].tick_params(labelsize=fontsize)
    return cbar

@hydra.main(config_path='/home/ubuntu/duxinghao/APEC/apec_dmc/src/cfgs', config_name='config_eval')
def main(cfg):
    FONTSIZE = 25
    sns.set_theme(style='darkgrid')
    savedir = '/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/speed_density'
    os.makedirs(savedir, exist_ok=True)

    fig = plt.figure(figsize=(8 * 2, 6 * 3))  # 2列环境，3行方法
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

    # MuJoCo task
    methods_mujoco = ['baseline_ssrr', 'baseline_drex', 'ablation_nonoise']
    dataset_mujoco = 'HopperFH-v0'
    # targets = {'HopperFH-v0': (1, 0), 'Walker2dFH-v0':(1, 9)}
    targets = {'HopperFH-v0':(0, 6)}

    ax_mujoco = [fig.add_subplot(gs[i, 0]) for i in range(len(methods_mujoco))]
    buffers_mujoco, bounds_mujoco = process_buffers_for_env(dataset_mujoco, methods_mujoco, targets)
    plot_density_heatmap(ax_mujoco, buffers_mujoco, targets, dataset_mujoco, methods_mujoco, *bounds_mujoco, FONTSIZE)

    # DMC tasks
    ax_dmc = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    buffers_dmc = [
        PreferenceBufferFeature(
            cfg.baseline_train_buffer_path.replace(cfg.baseline_type, method),
            cfg.baseline_train_idx_path.replace(cfg.baseline_type, method),
            500, device=torch.device(cfg.device)
        ) for method in ['ssrr', 'drex']
    ]
    buffers_dmc.append( PreferenceBufferFeature(
            cfg.train_buffer_path,
            cfg.train_idx_path,
            500, device=torch.device(cfg.device)
        ))
    plot_dmc_tasks(ax_dmc, buffers_dmc, ['SSRR', 'D-REX', 'APEC(ours)'], FONTSIZE)

    # Add colorbar for all subplots
    all_axes = ax_mujoco + ax_dmc
    cbar = fig.colorbar(ax_mujoco[0].collections[0], ax=all_axes, orientation='vertical', fraction=0.08, pad=0.04)
    cbar.set_label('Density', fontsize=FONTSIZE + 3, labelpad=-60)  # 减小 labelpad
    cbar_ticks = [cbar.vmin, cbar.vmax]  # 颜色条的最小值和最大值
    cbar.set_ticks(cbar_ticks)       # 设置位置
    cbar.set_ticklabels(['Low', 'High'])
    cbar.ax.tick_params(labelsize=FONTSIZE+1)

    # Set axis labels
    for ax in ax_mujoco + ax_dmc:
        ax.set_xlabel('X-axis Displacement', fontsize=FONTSIZE + 1)
    # for ax in ax_mujoco:
    #     ax.set_ylabel('Y-axis Velocity', fontsize=FONTSIZE + 1)

    # Add method and environment labels
    # 方法名标记在每行的最左侧
    method_labels = ['SSRR', 'D-REX', 'APEC(ours)']
    for i, ax in enumerate(ax_mujoco):
            ax.set_ylabel('X-axis Velocity', fontsize=FONTSIZE + 1, labelpad=0)
            ax.figure.text(ax.get_position().x0 - 0.07, ax.get_position().y0+0.1, method_labels[i], 
                   ha='center', va='center', fontsize=FONTSIZE + 1, rotation=90)

    # 环境名标记在每列的顶部
    ax_mujoco[0].set_title('Hopper-v2', fontsize=FONTSIZE + 1)
    ax_dmc[0].set_title(cfg.task_name, fontsize=FONTSIZE + 1)

    plt.tight_layout()
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(f'{savedir}/density.pdf')
    plt.close()

if __name__ == '__main__':
    main()
    # 运行时注意带参数：python plot/common_plot/speed_corr.py suite/dmc_task=walker_run seed=6
