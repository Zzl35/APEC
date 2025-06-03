import re

from utils.plot.plot_utils import *
from utils.plot.plot_configs import *


def eval_ablation(savefile=False, savepath=None):
    eval_items = ['test_PrefLabelAcc', 'test_ReturnCorr', 'test_RewardCorr', 'gen_accuracy']
    
    res_df = []
    methods = {"APEC":"ablation_nonoise"} 
    for expert_traj_num in ['0_0_1_0', '0_1_0_0', '0_0_10_0']:
        for env_n in ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'Walker2d']:
            method = 'ablation_nonoise'
            env_name = f'{env_n}FH-v0'
            logdir = os.path.join('logfile', 'train_pbrl', method, ENV2ENV_NAME[env_name], expert_traj_num)
            
            method_res_dict = {}
            for dirname in os.listdir(logdir):
                filepath = os.path.join(logdir, dirname, 'progress.csv')
                if os.path.getsize(filepath) > 0:
                    seed = int(re.search(r'seed_(\d+)', filepath).group(1))
                    progress_file = pd.read_csv(filepath)
                    res = progress_file.iloc[-1]
                    method_res_dict[f'{seed}'] = res

            methods_res = pd.DataFrame(method_res_dict).T[eval_items]
            method_res_val = methods_res.mean().values.squeeze()
            method_res_df = pd.DataFrame(data=method_res_val.reshape(1, -1), columns=eval_items)
            for key in method_res_df.keys():
                mean = methods_res[key].mean()
                std = methods_res[key].std()
                method_res_df[key] = f'{mean:.4f}+-{std:.4f}'
            method_res_df['Experiment'] = f'{env_name}-{expert_traj_num}'
            res_df.append(method_res_df)

    res_df = pd.concat(res_df)
    res_df.set_index('Experiment', inplace=True)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))

    if savefile:
        res_df.to_csv(savepath, index='Experiment')
    else:
        print(res_df)

if __name__ == '__main__':
    eval_ablation(savefile=True, savepath=f'eval/corr0326/data_quality_results.csv')