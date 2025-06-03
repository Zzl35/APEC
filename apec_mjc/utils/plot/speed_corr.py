import sys
sys.path.append('.') # 放在config里失效

import gym
import envs

from datetime import datetime
# from parameter.Parameter import Parameter
# from sample import load_expert


from utils import system
from utils.plot.plot_utils import *
from utils.plot.plot_configs import *

targets={
    # 'AntFH-v0':(0, 15),
    'HalfCheetahFH-v0':(0, 9),
    'HopperFH-v0':(0, 6),
    # 'HumanoidFH-v0':(0, 24),
    # 'Walker2dFH-v0':(0, 9),
}

import argparse
import pickle 

def get_trajs_from_buffer(path):
    with open(path, 'rb') as f:
        trajs_ = pickle.load(f)

    trajs = []
    for traj in trajs_:
        trajs.extend(traj[-1])
    
    states = np.concatenate([traj[0] for traj in trajs], axis=0)
    # actions = np.concatenate([traj[1] for traj in trajs], axis=0)
    rewards = np.concatenate([traj[2] for traj in trajs], axis=0)
    # dones = np.concatenate([traj[3] for traj in trajs], axis=0)
    return states, rewards

def get_states_from_states(states, size):
    states = states.reshape(-1, states.shape[-1])
    # print(states.shape)
    size = size if size > 0 else states.shape[0]
    random_indices = np.random.choice(states.shape[0], size=size, replace=False)
    return states[random_indices]

if __name__ == '__main__':
    methods = ['baseline_ssrr', 'baseline_drex','ablation_nonoise']
    datasets = ['Hopper','HalfCheetah']
    seed = 0 # 只有seed=0重收集了buffer
    plt.rcParams['font.size'] = FONTSIZE
    fig = plt.figure(figsize=(8*len(methods), 6*len(datasets)))
    axeses = []
    for m, data_name in enumerate(datasets):
        gs = fig.add_gridspec(len(datasets), len(methods), width_ratios=[1] * len(methods), wspace=0.2)
        axes = [fig.add_subplot(gs[m,i]) for i in range(len(methods))]
        # fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        dataset = f'{data_name}FH-v0'
        env = gym.make(dataset)

        # 初始化最小最大值
        x_min = y_min = np.inf
        x_max = y_max = -np.inf
        buffers = []
        for i, method_name in enumerate(methods):
            algorithm, load_buffer_name, load_model_name = set_model_hypeparams(method_name)
            load_buffer_path = os.path.join('buffer_qposvel', dataset, load_buffer_name, '0_0_1_0', str(seed), 'buffer.pkl') 
            states, rewards = get_trajs_from_buffer(load_buffer_path)
            # print(states_1.shape), quit()
            buffers.append(get_states_from_states(states, size=5000))
            x, y = targets[dataset]
            x_data = states[:, x]
            y_data = states[:, y]

            # 更新坐标轴范围
            x_max, x_min = max(x_max, x_data.max()), min(x_min, x_data.min())
            y_max, y_min = max(y_max, y_data.max()), min(y_min, y_data.min())

        for i, method_name in enumerate(methods):
            states = buffers[i]
            x, y = targets[dataset]
            x_data = states[:, x]
            y_data = states[:, y]

            # # 计算二维直方图
            # hist, xedges, yedges = np.histogram2d(x_data, y_data, bins=50, range=[[x_min, x_max], [y_min, y_max]])
            # # 使用pcolormesh绘制热图
            # cbar = axes[i].pcolormesh(xedges, yedges, hist.T, shading='auto', vmin=1)

            cbar = axes[i].hexbin(x_data, y_data, gridsize=30, mincnt=1, cmap='jet', extent=(x_min, x_max, y_min, y_max))
            if m == 0:
                axes[i].set_title(method2label[method_name],fontsize=FONTSIZE+1)
            axes[i].tick_params(labelsize=FONTSIZE)
 
        # fig.suptitle(f'Trajectory Density Heatmap', fontsize=FONTSIZE+2)
        cbar = fig.colorbar(cbar, ax=axes, orientation='vertical', label='Density')
        cbar.ax.tick_params(labelsize=FONTSIZE)
        cbar.set_label('Density', fontsize=FONTSIZE+1)
        axeses.append(axes)

    for i in range(len(datasets)):
        axeses[i][0].set_ylabel('Velocity',fontsize=FONTSIZE+1)
    for i in range(len(methods)):
        axeses[-1][i].set_xlabel('Displacement',fontsize=FONTSIZE+1)
    fig.subplots_adjust(bottom=0.15, right=0.75)
    plt.tight_layout()

    # 保存图形
    plot_dir = f'eval/density'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f'{plot_dir}/all.pdf')
    plt.close()  
