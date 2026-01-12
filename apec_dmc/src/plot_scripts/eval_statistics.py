import re

from utils.plot.plot_utils import *
from utils.plot.plot_configs import *


def eval_pbrl(methods, env_name, parameter, savefile=False, savepath=None):
    eval_items = ['test_PrefLabelAcc', 'test_ReturnCorr', 'test_RewardCorr']
    
    res_df = []
    for i, key in enumerate(methods.keys()):
        method = methods[key]
        logdir = os.path.join('logfile', 'train_pbrl', method, ENV2ENV_NAME[env_name], parameter.expert_traj_nums)
        
        method_res_dict = {}
        for dirname in os.listdir(logdir):
            filepath = os.path.join(logdir, dirname, 'progress.csv')
            if os.path.getsize(filepath) > 0:
                seed = int(re.search(r'seed_(\d+)', filepath).group(1))
                progress_file = pd.read_csv(filepath)
                res = progress_file.iloc[-1]
                method_res_dict[f'{seed}'] = res

        method_res_val = pd.DataFrame(method_res_dict).T[eval_items].mean().values.squeeze()
        method_res_df = pd.DataFrame(data=method_res_val.reshape(1, -1), columns=eval_items)
        method_res_df['Method'] = method
        res_df.append(method_res_df)

    res_df = pd.concat(res_df)
    res_df.set_index('Method', inplace=True)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))

    if savefile:
        res_df.to_csv(savepath, index='Method')
    else:
        print(res_df)
