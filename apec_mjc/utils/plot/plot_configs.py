import sys
sys.path.append('.')

import torch
import json
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.neighbors import NearestNeighbors

from utils.replay_buffer import PreferenceBuffer
from model.reward import PrefRewardModel, DrexRewardModel, SupervisedRewardModel, MLPReward, SSRRSigmoid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV2ENV_NAME = {"HopperFH-v0":"Hopper-v2", "HalfCheetahFH-v0":"HalfCheetah-v2", "Walker2dFH-v0":"Walker2d-v2",
                "AntFH-v0":"Ant-v2", "HumanoidFH-v0":"Humanoid-v2", 'dmc_quadruped_walk-v0':'quadruped_walk',
                'dmc_cheetah_run-v0':'cheetah_run', 'dmc_walker_walk-v0':'walker_walk'}

EPS=1e-5

method2label = {
    'ablation_nonoise': 'PrefGen (Ours)',
    # 'ablation_noepoch':'Ours (Last Iter)',
    'ablation_4epoch': 'PrefGen (with few checkpoints)',
    'ablation_40epoch': 'PrefGen (40 checkpoints)',
    'ablation_sample_nodistance': 'PrefGen (iteration guidance only)',
    'ablation_sample_noiter': 'PrefGen (OT distance guidance only)',
    'ablation_sample_relative': 'PrefGen (Relative distance guidance)',
    # 'pbrl': 'Ours (epsilon)', # we dont use that anymore
    # 'pbrl': 'Ours',
    'baseline_drex': 'D-REX',
    # 'baseline_ssrr_bc': 'SSRR(BC)',
    'baseline_ssrr': 'SSRR',
    'baseline_ssrr_save': 'SSRR',
    # 'baseline_ssrr': 'SSRR(IRL)',
    'baseline_lerp': 'LERP',
    'baseline_irl': 'AIRL',
    'ablation_noepoch':'D-REX(IRL)',
    # 'baseline_gail': 'D-REX(IRL)',
    'baseline_cail': 'CAIL',
}

model_hypeparams = {
    # method                algorithm   load_buffer_name    load_model_name
    'pbrl':[                'pbrl',     'pbrl',             'pbrl'],
    'supervised':[          'pbrl',     'supervised',       'supervised'],              # 预测轮数和噪声强度
    'ablation_noepoch':[    'pbrl',     'baseline_gail',    'ablation_noepoch'],        # 用irl最后一轮sample，使用baseline_gail的buffer即可
    'ablation_nonoise':[    'pbrl',     'ablation_nonoise', 'ablation_nonoise'],        # 用irl每轮的policy直接采样，不加噪声
    'ablation_4epoch':[     'pbrl',     'ablation_4epoch',  'ablation_4epoch'],         # 不加噪声，且均匀采样其中4轮（0，100，200，300）
    'ablation_40epoch':[    'pbrl',     'ablation_40epoch', 'ablation_40epoch'],        # 不加噪声，且均匀采样其中40轮（0，10，...，390）
    'ablation_sample_noiter':['pbrl',  'ablation_sample_noiter', 'ablation_sample_noiter'], 
    'ablation_sample_nodistance':['pbrl','ablation_sample_nodistance', 'ablation_sample_nodistance'], 
    'ablation_sample_nodistance_100ep':['pbrl','ablation_sample_nodistance_100ep', 'ablation_sample_nodistance_100ep'], 
    'ablation_sample_relative':['pbrl', 'ablation_sample_relative', 'ablation_sample_relative'],
    'ablation_segment500':[    'pbrl',     'ablation_nonoise', 'ablation_nonoise'],
    'ablation_segment100':[    'pbrl',     'ablation_nonoise', 'ablation_nonoise'],
    'ablation_sample_distance2':['pbrl',     'ablation_sample_distance2', 'ablation_sample_distance2'],
    'ablation_sample_distance05':['pbrl',     'ablation_sample_distance05', 'ablation_sample_distance05'],
    'ablation_metric_mse':[    'pbrl',     'ablation_metric_mse', 'ablation_metric_mse'], 

    'baseline_drex':[       'pbrl',     'baseline_drex',    'baseline_drex'],           # BC+epsilon
    'baseline_lerp':[       'pbrl',     'baseline_lerp',    'baseline_lerp'],           # 向drex的loss中加入noise level的影响，lerp buffer中的label是各自的noise
    'baseline_gail':[       'pbrl',     'baseline_gail',    'baseline_gail'],           # drex改为用irl的模型，即irl最后一轮+epsilon
    'baseline_ssrr':[       'pbrl',     'baseline_ssrr',    'baseline_ssrr'],
    # 'baseline_ssrr_save':[       'pbrl',     'baseline_ssrr',    'baseline_ssrr'],               # 自回归sigmoid，再用reward逼近sigmoid
    # 'baseline_ssrr_bc':[    'pbrl',     'baseline_lerp',    'baseline_ssrr_bc'], 
    # 'baseline_ssrr_irl':[   'pbrl',     'baseline_ssrr_irl','baseline_ssrr_irl'], 
    'baseline_irl':[        'irl',      'ablation_nonoise',  'irl'],            # 直接使用irl训练最后一轮的模型来eval
}

# COLORS = sns.color_palette("Set3") 
COLORS = sns.color_palette() # default

FONTSIZE = 20
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['font.family'] = 'Arial'
sns.set_theme(style='darkgrid')