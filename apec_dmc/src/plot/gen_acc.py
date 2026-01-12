import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
import sys
sys.path.append('.')

from data_scripts.make_pref_dataset import PreferenceBuffer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_torch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch

def compute_gen_acc(loader):
    accuracy = 0
    num_samples = 0
    with tqdm(total=5000) as pbar:
        pbar.set_description('computing accuracy')
        for batch in loader:
            batch = to_torch(batch, device)
            _, _, _, _, rewards1, rewards2, masks_1, masks_2, pref_label = batch
            real_returns1 = torch.sum(rewards1 * masks_1, dim=-1, keepdim=True).flatten()
            real_returns2 = torch.sum(rewards2 * masks_2, dim=-1, keepdim=True).flatten()
            gt_label = (real_returns1 < real_returns2).int()
            sample_label = torch.argmax(pref_label, dim=-1).squeeze()

            accuracy += (gt_label == sample_label).sum().item()
            num_samples += gt_label.shape[0]
            pbar.update(rewards1.shape[0])
    return accuracy / num_samples


def get_acc_df():
    load_dir = f'/infinite/common/buffer'
    methods = ['drex', 'ssrr', 'ours']
    tasks = ['walker_walk', 'walker_run', 'cheetah_run']
    
    all_accuracy = pd.DataFrame(columns=tasks, index=methods).applymap(lambda x: [])
    for task_name in tasks:
        for method in methods: 
            for seed in range(5):
                data_dir = f'{load_dir}/dmc/{task_name}/{seed}'
                if method in ['drex', 'ssrr']:
                    buffer_path = f'{data_dir}/{method}/suboptimal'
                    idx_path = f'{data_dir}/{method}/suboptimal_train.pkl'
                else: # our method
                    buffer_path = f'{data_dir}/suboptimal'
                    idx_path = f'{data_dir}/suboptimal_train.pkl'
                buffer = PreferenceBuffer(buffer_path, idx_path, 500, device=device)
                loader = torch.utils.data.DataLoader(buffer, batch_size=8, shuffle=False, num_workers=6)
                acc = compute_gen_acc(loader)
                if method in ['ssrr']:
                    acc = 1 - acc
                all_accuracy.loc[method, task_name].append(acc)
                print(f'{task_name}-{method}-{seed}-done. acc: {acc}')

    all_accuracy = all_accuracy.T
    all_accuracy.to_csv(f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/generate_accuracy/gen_acc_data.csv')
    mean_var_df = all_accuracy.applymap(lambda lst: f"{np.mean(lst):.4f} ± {np.var(lst):.4f}")
    model_means = mean_var_df.applymap(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    mean_var_df.loc['Mean'] = mean_row.astype(str)
    mean_var_df.to_csv(f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/generate_accuracy/gen_acc.csv')
                

if __name__ == '__main__':
    get_acc_df()