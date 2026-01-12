import torch
import hydra
import sys
sys.path.append('.')
from reward.pref_reward import PrefRewardModel
from plot.plot_utils import *
from tqdm import tqdm

def to_torch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch

def reward_correlation(models: dict, samples: dict, savefig=True, savepath=None, alg='maxentirl', storefig=False):
    '''
    Show return/reward correlation with true results.
    rewards: {LABEL: model}
    samples: (states, actions, rewards) # test buffer, same for all
    '''
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(5*len(models), 5))
    axes = np.atleast_1d(axes).flatten()
    corrs = {}
    states, actions, rewards, masks = samples
    rewards = rewards[masks > 0].flatten()
    inputs = states
    inputs = inputs[masks > 0]
    real_rewards = rewards
    real_min, real_max = real_rewards.min(), real_rewards.max()
    batch_size = 64
    for i, key in enumerate(models.keys()):
        pred_rewards_list = []

        # 使用分批次处理来减少内存占用
        for start_idx in range(0, inputs.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, inputs.shape[0])
            batch_states = torch.tensor(states[masks>0][start_idx:end_idx], dtype=torch.float32).to(models[key].device)
            batch_actions = torch.tensor(actions[masks>0][start_idx:end_idx], dtype=torch.float32).to(models[key].device)
            
            seq_len, c, h, w = batch_states.shape
            # print(batch_inputs.shape)
            with torch.no_grad():
                batch_pred_rewards = models[key].r(batch_states.reshape(-1, c, h, w), batch_actions.reshape(-1, actions.shape[-1])).cpu().numpy()
                pred_rewards_list.append(batch_pred_rewards)

        pred_rewards = np.concatenate(pred_rewards_list).flatten()
        pred_min, pred_max = pred_rewards.min(), pred_rewards.max()

        indices = np.random.choice(inputs.shape[0], size=2000, replace=False)
        corr = np.corrcoef(pred_rewards, real_rewards)[0, 1]
        corrs[key] = corr

        axes[i].scatter(x=real_rewards[indices], y=pred_rewards[indices], label=key, marker='.', color=COLORS[1])
        axes[i].plot([real_min, real_max], [pred_min, pred_max], linestyle='--', color='black')
        if storefig:
            axes[i].set_title(f'{key}', fontsize=FONTSIZE+1)
        axes[i].text(0.02, 0.98, f'corr={corr:.2f}', transform=axes[i].transAxes, fontsize=FONTSIZE+1, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        axes[i].set_xlabel('Ground Truth Reward', fontsize=FONTSIZE+1)
        axes[i].set_ylabel('Predicted Reward', fontsize=FONTSIZE+1)
        axes[i].tick_params(labelsize=FONTSIZE)
    
    plt.tight_layout()
    # handles = []
    # handles.append(mlines.Line2D([], [], color=COLORS[1], label='Test Trajectories', linewidth=5))
    # fig.subplots_adjust(bottom=0.3)
    # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(handles), fontsize=FONTSIZE+1)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savefig:
        plt.savefig(savepath)
    # plt.close()

    return corrs, fig

def get_reward_corr_df(reward_model, loader, n_samples=None, device='cpu'):
    used_samples = 0
    total_pred, total_real = [], []
    # total_samples = n_samples if n_samples or n_samples < max_n_sample else max_n_sample
    with tqdm(total=n_samples) as pbar:
        for batch in loader:
            batch = to_torch(batch, device)
            states1, states2, actions1, actions2, rewards1, rewards2, masks_1, masks_2, pref_label = batch
            bs, seq_len, c, h, w = states1.shape
            pred_rewards1 = reward_model(states1.reshape(-1, c, h, w), actions1.reshape(-1, actions1.shape[-1])).reshape(bs, -1)
            pred_rewards2 = reward_model(states2.reshape(-1, c, h, w), actions2.reshape(-1, actions2.shape[-1])).reshape(bs, -1)

            pred_rewards = torch.cat([pred_rewards1[masks_1>0], pred_rewards2[masks_2>0]], dim=0).flatten().detach().cpu().numpy()
            real_rewards = torch.cat([rewards1[masks_1>0], rewards2[masks_2>0]], dim=0).flatten().detach().cpu().numpy()

            used_samples += bs
            total_pred.append(pred_rewards)
            total_real.append(real_rewards)

            pbar.update(bs)
            if n_samples and used_samples >= n_samples:
                break

    total_pred = np.concatenate(total_pred)
    total_real = np.concatenate(total_real)
    indices = np.random.choice(total_pred.shape[0], size=5000, replace=False)

    res_df = {'pred_reward':total_pred[indices], 'real_reward':total_real[indices]}
    return pd.DataFrame(res_df)

def plot_reward_corr_from_df(df_dict:dict, savepath=None):
    '''
    df_dict: {name: df}
    df columns: pred_return, real_return, type
    '''
    fig, axes = plt.subplots(nrows=1, ncols=len(df_dict), figsize=(5 * len(df_dict), 5))
    axes = np.atleast_1d(axes).flatten()
    train_corrs, test_corrs = {}, {}

    for i, key in enumerate(df_dict.keys()):
        df = df_dict[key]
        pred = df['pred_reward'].values
        real = df['real_reward'].values

        test_corr = np.corrcoef(pred, real)[0, 1]
        test_corrs[key] = test_corr

        real_min, real_max = real.min(), real.max()
        pred_min, pred_max = pred.min(), pred.max()

        axes[i].scatter(x=real, y=pred, marker='.', color=COLORS[0])
        axes[i].scatter(x=real, y=pred, marker='.', color=COLORS[1])
        
        axes[i].plot([real_min, real_max], [pred_min, pred_max], linestyle='--', color='black')
        axes[i].set_title(f'{key}', fontsize=FONTSIZE+1)
        axes[i].text(0.02, 0.98, f'corr={test_corr:.2f}', transform=axes[i].transAxes, fontsize=FONTSIZE+1, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        axes[i].set_xlabel('Ground Truth Reward', fontsize=FONTSIZE+1)
        axes[i].set_ylabel('Predicted Reward', fontsize=FONTSIZE+1)
        axes[i].tick_params(labelsize=FONTSIZE)
    
    plt.tight_layout()
    # handles = []
    # handles.append(mlines.Line2D([], [], color=COLORS[0], label='Synthetic Trajectories', linewidth=5))
    # handles.append(mlines.Line2D([], [], color=COLORS[1], label='Test Trajectories', linewidth=5))
    # fig.subplots_adjust(bottom=0.3)
    # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(handles), fontsize=FONTSIZE+1)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savepath:
        abspath = os.path.abspath(savepath)
        print(f'file saved at {abspath}')
        plt.savefig(savepath)
    plt.close()

    return train_corrs, test_corrs


from baselines.airl.eval_airl import load_dac, load_cfg_override, WorkspaceIL
def get_airl_reward(cfg):
    workspace = WorkspaceIL(cfg, work_dir='None')
    load_cfg = load_cfg_override(cfg)
    load_dac(workspace, load_cfg)
    return lambda x,y: torch.from_numpy(workspace.reward_agent.dac_rewarder(x,y)).to(cfg.device)

def get_baseline_reward(cfg):
    train_env = hydra.utils.call(cfg.suite.task_make_fn)
    # load reward model
    reward_model = PrefRewardModel(
                    obs_shape=train_env.observation_spec()[cfg.obs_type].shape,
                    action_dim=train_env.action_spec().shape,
                    use_action=cfg.use_action,
                    feature_dim=cfg.agent.feature_dim,
                    hidden_sizes=(512, 512),
                    device=cfg.device).to(torch.device(cfg.device))
    reward_path = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/train_{cfg.baseline_type}/dmc/{cfg.task_name}/{cfg.seed}/best_model/reward_model_last.pkl'
    reward_model.load_state_dict(torch.load(reward_path))
    return reward_model

def get_our_reward(cfg):
    train_env = hydra.utils.call(cfg.suite.task_make_fn)
    # load reward model
    reward_model = PrefRewardModel(
                    obs_shape=train_env.observation_spec()[cfg.obs_type].shape,
                    action_dim=train_env.action_spec().shape,
                    use_action=cfg.use_action,
                    feature_dim=cfg.agent.feature_dim,
                    hidden_sizes=(512, 512),
                    device=cfg.device).to(torch.device(cfg.device))
    reward_path = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/train_pbrl/dmc/{cfg.task_name}/{cfg.seed}/best_model/reward_model_last.pkl'
    reward_model.load_state_dict(torch.load(reward_path))
    return reward_model

def reproduce(seed):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path='../cfgs', config_name='config_eval')
def main(cfg):
    device=torch.device(cfg.device)
    savedir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/corr_data_reward/{cfg.task_name}/{cfg.seed}'
    reproduce(cfg.seed)

    method = cfg.baseline_type
    if method == 'airl':
        reward_model = get_airl_reward(cfg)
    elif method in ['drex', 'lerp', 'ssrr']:
        reward_model = get_baseline_reward(cfg)
    elif method == 'ours':
        reward_model = get_our_reward(cfg)

    test_reward_buffer = PreferenceBuffer(cfg.test_buffer_path, cfg.test_idx_path, 500, device=device)
    test_loader = torch.utils.data.DataLoader(test_reward_buffer, batch_size=8, shuffle=False, num_workers=8)

    # get dataframes
    savepath = f'{savedir}/{method}.csv'
    test_df = get_reward_corr_df(reward_model, test_loader, device=device, n_samples=500)

    os.makedirs(os.path.abspath(os.path.dirname(savepath)), exist_ok=True)
    test_df.to_csv(savepath, index=None)

    # get_stats_from_dict() # 画图以及计算数值保存csv

if __name__ == '__main__':
    main()
