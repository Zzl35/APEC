import torch
import pickle
import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_scripts.make_pref_dataset import PreferenceBufferFeature

# from plot_scripts.plot_utils import *
# from plot_scripts.plot_configs import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_torch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch


def compute_min_l1_distance(loader, test_states, device, batch_size=256):

    loader_states = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = to_torch(batch, device)
            states1, states2, *_ = batch
            _, _, state_dim = states1.shape
            trajectories = torch.cat([states1, states2], dim=0)
            loader_states.append(trajectories)
            if i == 50:
                break
        loader_states = torch.cat(loader_states, dim=0).reshape(-1, state_dim)  # Shape: (N_loader, D)
    random_indices = np.random.choice(loader_states.shape[0], size=10000, replace=False)
    loader_states = loader_states[random_indices]

    distance = torch.cdist(loader_states, test_states, p=1).min(dim=0).values.mean().item() / state_dim

    return distance


@hydra.main(config_path='/home/ubuntu/duxinghao/APEC/apec_dmc/src/cfgs', config_name='config_eval')
def main(cfg):
    env_names_dmc = ['cheetah_run','walker_run', 'walker_walk']
    methods = ['ssrr', 'drex', 'ours']
    seeds = [0, 1, 3, 4]
    distances = pd.DataFrame(index=env_names_dmc, columns=methods).applymap(lambda x: [])
    savedir = '/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/proj_distance'
    # dmc tasks
    with tqdm(total=len(env_names_dmc)*len(methods)*len(seeds)) as pbar:
        pbar.set_description('computing l1 distance')
        for i, env_name in enumerate(env_names_dmc):
            for seed in seeds:
                buffers = [
                    PreferenceBufferFeature(
                        cfg.baseline_train_buffer_path.replace(cfg.baseline_type, method).replace(cfg.task_name, env_name).replace(str(cfg.seed), str(seed)),
                        cfg.baseline_train_idx_path.replace(cfg.baseline_type, method).replace(cfg.task_name, env_name).replace(str(cfg.seed), str(seed)),
                        500, device=torch.device(cfg.device)
                    ) for method in ['ssrr', 'drex']
                ]
                buffers.append(PreferenceBufferFeature(
                        cfg.train_buffer_path.replace(cfg.task_name, env_name).replace(str(cfg.seed), str(seed)),
                        cfg.train_idx_path.replace(cfg.task_name, env_name).replace(str(cfg.seed), str(seed)),
                        500, device=torch.device(cfg.device)
                    ))
                test_buffer = PreferenceBufferFeature(
                        cfg.test_buffer_path.replace(cfg.task_name, env_name).replace(str(cfg.seed), str(seed)),
                        cfg.test_idx_path.replace(cfg.task_name, env_name).replace(str(cfg.seed), str(seed)),
                        500, device=torch.device(cfg.device)
                    )
                test_loader = torch.utils.data.DataLoader(test_buffer, batch_size=8, shuffle=False, num_workers=6)
                test_states = []
                with torch.no_grad():
                    for i, test_batch in enumerate(test_loader):
                        test_batch = to_torch(test_batch, device)
                        test_states1, test_states2, *_ = test_batch
                        test_trajectories = torch.cat([test_states1, test_states2], dim=0)
                        test_states.append(test_trajectories)
                        if i == 10:
                            break
                    test_states = torch.cat(test_states, dim=0).reshape(-1, test_states1.shape[-1])
                    random_indices = np.random.choice(test_states.shape[0], size=10000, replace=False)
                    test_states = test_states[random_indices]

                for i, buffer in enumerate(buffers):
                    loader = torch.utils.data.DataLoader(buffer, batch_size=8, shuffle=False, num_workers=6)
                    distances.loc[env_name, methods[i]].append(compute_min_l1_distance(loader, test_states, device))

                    distances.to_csv(f'{savedir}/projection_distance_data.csv') # 保存中间内容
                    pbar.update(1)

    mean_var_df = distances.applymap(lambda lst: f"{np.mean(lst):.4f} ± {np.var(lst):.4f}")
    model_means = mean_var_df.applymap(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    mean_var_df.loc['Mean'] = mean_row.astype(str)
    mean_var_df.to_csv(f'{savedir}/projection_distance.csv')


if __name__ == '__main__':
    main()

