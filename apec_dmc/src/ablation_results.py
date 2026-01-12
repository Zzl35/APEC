import numpy as np
import pandas as pd
import os


root_dir = "/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/train_pbrl/dmc"
for file in sorted(os.listdir(root_dir)):
    task_name = '_'.join(file.split('_')[:2])
    if file.endswith('wo_distance'):
        algo_name = 'wo_wfilter'
    elif file.endswith('5ckpt'):
        algo_name = 'fewer_ckpt'
    elif file.endswith('500'):
        algo_name = 'wo_cropped'
    else:
        algo_name = 'PrefGen'
    
    reward_corr, return_corr, pref_acc = [], [], []
    for seed in os.listdir(os.path.join(root_dir, file)):
        result = pd.read_csv(os.path.join(root_dir, file, seed, 'train.csv'))
        reward_corr.append(result['test_RewardCorr'].values[-1])
        return_corr.append(result['test_ReturnCorr'].values[-1])
        pref_acc.append(result['test_PrefLabelAcc'].values[-1])
    reward_corr = np.array(reward_corr)
    return_corr = np.array(return_corr)
    pref_acc = np.array(pref_acc)
    print(f'------------------{task_name}-{algo_name}------------------')
    print(f'reward correlation: {reward_corr.mean()} {reward_corr.std()}')
    print(f'return correlation: {return_corr.mean()} {return_corr.std()}')
    print(f'preference accuracy: {pref_acc.mean()} {pref_acc.std()}')
    print('-----------------------------------------------------------')