from tensorboard.backend.event_processing import event_accumulator


from plot_scripts.plot_utils import *
from plot_scripts.plot_configs import *

import pickle
import hydra
from data_scripts.make_pref_dataset import PreferenceBuffer

def get_consistency():
    methods = ['ssrr', 'drex', 'lerp', 'ours']
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

if __name__ == '__main__':
    get_consistency()

