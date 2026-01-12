import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import hydra
import sys
import torch
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_scripts.make_pref_dataset import PreferenceBufferFeature

FONTSIZE = 22
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['font.family'] = 'Arial'
sns.set_theme(style='darkgrid')



'''
state: env.physics.data.qpos(:9) + env.physics.data.qvel(9:)
features: orientations(:13) + torsor_height(14) + velocity(15:), velocity = [rootx, rootz, rooty, ....]
pixels: 3*84*84
'''
@hydra.main(config_path='../cfgs', config_name='config_eval')
def main(cfg):
    device=torch.device(cfg.device)
    savedir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/speed_density'
    
    drex_buffer = PreferenceBufferFeature(cfg.baseline_train_buffer_path.replace(cfg.baseline_type, 'drex'), cfg.baseline_train_idx_path.replace(cfg.baseline_type, 'drex'), 500, device=device)
    ssrr_buffer = PreferenceBufferFeature(cfg.baseline_train_buffer_path.replace(cfg.baseline_type, 'ssrr'), cfg.baseline_train_idx_path.replace(cfg.baseline_type, 'ssrr'), 500, device=device)
    prefgen_buffer = PreferenceBufferFeature(cfg.train_buffer_path, cfg.train_idx_path, 500, device=device)

    x_min = y_min = np.inf
    x_max = y_max = -np.inf

    plt.rcParams['font.size'] = FONTSIZE
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    methods = ['SSRR', 'D-REX', 'PrefGen(ours)']
    datas = []
    for i, buffer in enumerate([ssrr_buffer, drex_buffer, prefgen_buffer]):
        x_data, y_data = [], []
        loader = torch.utils.data.DataLoader(buffer, batch_size=8, shuffle=False, num_workers=16)
        with tqdm(total=10000) as pbar:
            pbar.set_description(f'plotting trajectories for {methods[i]}')
            for b, batch in enumerate(loader):
                states1, states2, _, _, rewards1, rewards2, *_ = batch
                # print(states1.shape)
                trajectories = torch.concat([states1, states2], dim=0)
                # returns = torch.concat([rewards1, rewards2], dim=0).sum(-1).flatten()
                x_data.append(trajectories[:, :, 0]) # position
                y_data.append(trajectories[:, :, 1]) # speed

                pbar.update(trajectories.shape[0])
                # if b > 1:
                #     break

        x_data = np.concatenate(x_data, axis=0).flatten()
        y_data = np.concatenate(y_data, axis=0).flatten()
        x_max, x_min = max(x_max, x_data.max()), min(x_min, x_data.min())
        y_max, y_min = max(y_max, y_data.max()), min(y_min, y_data.min())
        datas.append((x_data, y_data))

    for i in range(3):
        x_data, y_data = datas[i]
        cbar = axes[i].hexbin(x_data, y_data, gridsize=50, mincnt=1, cmap='jet', extent=(x_min, x_max, y_min, y_max))
        axes[i].set_title(methods[i],fontsize=FONTSIZE+1)
        axes[i].tick_params(labelsize=FONTSIZE)
        axes[i].set_ylabel('Velocity',fontsize=FONTSIZE+1)
        axes[i].set_xlabel('X-axis Displacement',fontsize=FONTSIZE+1)

    cbar = fig.colorbar(cbar, ax=fig.add_axes(), orientation='vertical', label='Density')
    cbar.ax.tick_params(labelsize=FONTSIZE)
    cbar.set_label('Density', fontsize=FONTSIZE+1)
    fig.subplots_adjust(bottom=0.15, right=0.95)
    # plt.tight_layout()

    # 保存图形
    os.makedirs(savedir, exist_ok=True)
    plt.savefig(f'{savedir}/all.pdf')
    plt.close()  


if __name__ == '__main__':
    main()

