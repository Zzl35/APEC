from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import os
import torch
import sys
sys.path.append('.')
device= 'cuda' if torch.cuda.is_available() else 'cpu'

import hydra
from data_scripts.make_pref_dataset import PreferenceBuffer

def get_consistency():
    methods = ['ssrr', 'ours']
    tasks = ['walker_walk', 'walker_run', 'cheetah_run']
    
    all_consistency = pd.DataFrame(columns=tasks, index=methods).applymap(lambda x: [])

    for j, env_name in enumerate(tasks):
        for i, method in enumerate(methods):
            if method in ['airl', 'drex', 'lerp', 'ssrr']:
                logdir = os.path.join('exp_local', f'train_{method}', 'dmc', env_name)
            else:
                logdir = os.path.join('exp_local', f'train_pbrl', 'dmc', env_name)
            curves = []
            if os.path.exists(logdir):
                for dirname in os.listdir(logdir):
                    tb_dir = os.path.join(logdir, dirname, 'tb')
                    if len(os.listdir(tb_dir)) > 1:
                        print(f'logfile@{tb_dir} is not unique, please check.')
                    tb_path = os.path.join(tb_dir, sorted(os.listdir(tb_dir))[-1])
                    ea = event_accumulator.EventAccumulator(tb_path)
                    ea.Reload()
                    if 'train/test_PrefLabelAcc' not in ea.Tags()['scalars']:
                        print(f'Tensorboard@{tb_dir} not contains train/test_PrefLabelAcc, skipped.')
                        continue
                    training_curve = ea.scalars.Items('train/test_PrefLabelAcc')
                    consist = np.array([dot.value for dot in training_curve])[-1]
                    all_consistency.loc[method, env_name].append(consist)
                print(f'{env_name}-{method}-done. acc: {consist}')
            else:
                print(f'Logfile@{logdir} not exists, skipped.')
                continue

    all_consistency = all_consistency.T
    all_consistency.to_csv(f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/consistency/consistency_data.csv')
    mean_var_df = all_consistency.applymap(lambda lst: f"{np.mean(lst):.4f} ± {np.var(lst):.4f}")
    model_means = mean_var_df.applymap(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    mean_var_df.loc['Mean'] = mean_row.astype(str)
    mean_var_df.to_csv(f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/consistency/consistency.csv')

# if __name__ == '__main__':
#     get_consistency()

from tqdm import tqdm
from plot.corr import get_airl_reward

def to_torch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch


@hydra.main(config_path='../cfgs', config_name='config_eval') 
def main(cfg):
    test_reward_buffer = PreferenceBuffer(cfg.test_buffer_path, cfg.test_idx_path, 500, device=device)
    loader = torch.utils.data.DataLoader(test_reward_buffer, batch_size=8, shuffle=False, num_workers=6)
    reward_model = get_airl_reward(cfg)
    accuracy = 0
    with tqdm(total=5000) as pbar:
        for batch in loader:
            batch = to_torch(batch, device)
            states1, states2, actions1, actions2, real_rewards1, real_rewards2, masks_1, masks_2, pref_label = batch
            bs, seq_len, c, h, w = states1.shape
            pred_rewards1 = reward_model(states1.reshape(-1, c, h, w), actions1.reshape(-1, actions1.shape[-1])).reshape(bs, -1)
            pred_rewards2 = reward_model(states2.reshape(-1, c, h, w), actions2.reshape(-1, actions2.shape[-1])).reshape(bs, -1)
            
            pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
            pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True) 
            real_returns1 = torch.sum(real_rewards1 * masks_1, dim=-1, keepdim=True)
            real_returns2 = torch.sum(real_rewards2 * masks_2, dim=-1, keepdim=True) 

            # Preference labels
            p_returns = torch.cat([pred_returns1, pred_returns2], dim=-1).detach().cpu().numpy()
            r_returns = torch.cat([real_returns1, real_returns2], dim=-1).detach().cpu().numpy()
            pred_label = np.argmax(p_returns, axis=1)
            real_label = np.argmax(r_returns, axis=1)
            accuracy += np.sum(pred_label == real_label)

            pbar.update(bs)
    
    accuracy /= 5000
    savedir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/consistency/{cfg.task_name}'
    os.makedirs(savedir, exist_ok=True)
    with open(f'{savedir}/{cfg.seed}.txt', 'w') as f:
        f.write(str(accuracy))


if __name__ == '__main__':
    main()
