import pandas as pd
import numpy as np
import os
from plot.corr import plot_return_corr_from_df
from plot.reward_corr import plot_reward_corr_from_df

def get_return_stats_from_dict():
    load_dir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output_save/corr_data_return'
    methods = ['ssrr', 'airl', 'drex', 'lerp', 'ours']
    plot_labels = ['SSRR', 'AIRL', 'DREX', 'LERP', 'APEC (ours)']
    tasks = ['cheetah_run', 'walker_run', 'walker_walk']
    # all_train_corrs = pd.DataFrame(columns=tasks, index=methods).applymap(lambda x: [])
    all_test_corrs = pd.DataFrame(columns=tasks, index=plot_labels).applymap(lambda x: [])
    for task_name in tasks:
        for seed in [0, 1, 3, 4]:
            load_path = f'{load_dir}/{task_name}/{seed}'
            df_dict = {}
            for plot_label, method in zip(plot_labels, methods):
                load_data_path = f'{load_path}/{method}.csv'
                if os.path.exists(load_data_path):
                    df_dict[plot_label] = pd.read_csv(load_data_path, index_col=None)
                else:
                    print(f"File@{load_path}/{method}.csv not found, skipped.")
                    continue
            train_corrs, test_corrs = plot_return_corr_from_df(df_dict, savepath=f'{load_path}/return_corr.pdf')
            for plot_label, method in zip(plot_labels, methods):
                # all_train_corrs.loc[method, task_name].append(train_corrs[method])
                all_test_corrs.loc[plot_label, task_name].append(test_corrs[plot_label])
    all_test_corrs = all_test_corrs.T
    all_test_corrs.to_csv(f'{load_dir}/test_return_corrs_data.csv')
    mean_var_df = all_test_corrs.applymap(lambda lst: f"{np.mean(lst):.4f} ± {np.var(lst):.4f}")
    model_means = mean_var_df.applymap(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    mean_var_df.loc['Mean'] = mean_row.astype(str)
    mean_var_df.to_csv(f'{load_dir}/test_return_corrs.csv')

def get_reward_stats_from_dict():
    load_dir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output_save/corr_data_reward'
    methods = ['ssrr', 'airl', 'drex', 'lerp', 'ours']
    plot_labels = ['SSRR', 'AIRL', 'DREX', 'LERP', 'APEC (ours)']
    tasks = ['cheetah_run', 'walker_run', 'walker_walk']
    # all_train_corrs = pd.DataFrame(columns=tasks, index=methods).applymap(lambda x: [])
    all_test_corrs = pd.DataFrame(columns=tasks, index=plot_labels).applymap(lambda x: [])
    for task_name in tasks:
        for seed in [0, 1, 3, 4]:
            load_path = f'{load_dir}/{task_name}/{seed}'
            df_dict = {}
            for plot_label, method in zip(plot_labels, methods):
                load_data_path = f'{load_path}/{method}.csv'
                if os.path.exists(load_data_path):
                    df_dict[plot_label] = pd.read_csv(load_data_path, index_col=None)
                else:
                    print(f"File@{load_path}/{method}.csv not found, skipped.")
                    continue
            train_corrs, test_corrs = plot_reward_corr_from_df(df_dict, savepath=f'{load_path}/reward_corr.pdf')
            for plot_label, method in zip(plot_labels, methods):
                # all_train_corrs.loc[method, task_name].append(train_corrs[method])
                all_test_corrs.loc[plot_label, task_name].append(test_corrs[plot_label])
    all_test_corrs = all_test_corrs.T
    all_test_corrs.to_csv(f'{load_dir}/test_reward_corrs_data.csv')
    mean_var_df = all_test_corrs.applymap(lambda lst: f"{np.mean(lst):.4f} ± {np.var(lst):.4f}")
    model_means = mean_var_df.applymap(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    mean_var_df.loc['Mean'] = mean_row.astype(str)
    mean_var_df.to_csv(f'{load_dir}/test_reward_corrs.csv')

if __name__ == '__main__':
    get_return_stats_from_dict()
    get_reward_stats_from_dict()