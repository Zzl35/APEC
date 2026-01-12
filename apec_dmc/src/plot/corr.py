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

def get_return_corr_df(reward_model, loader, n_samples=None, device='cpu'):
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

            # Compute predicted returns
            pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
            pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True)
            pred_returns = torch.cat([pred_returns1, pred_returns2]).flatten().detach().cpu().numpy()

            # compute real returns
            real_returns1 = torch.sum(rewards1 * masks_1, dim=-1, keepdim=True)
            real_returns2 = torch.sum(rewards2 * masks_2, dim=-1, keepdim=True)
            real_returns = torch.cat([real_returns1, real_returns2]).flatten().detach().cpu().numpy()

            used_samples += bs
            total_pred.append(pred_returns)
            total_real.append(real_returns)

            pbar.update(bs)
            if n_samples and used_samples >= n_samples:
                break
    
    total_pred = np.concatenate(total_pred)
    total_real = np.concatenate(total_real)

    res_df = {'pred_return':total_pred, 'real_return':total_real}
    return pd.DataFrame(res_df)

def plot_return_corr_from_df(df_dict:dict, savepath=None):
    '''
    df_dict: {name: df}
    df columns: pred_return, real_return, type
    '''
    fig, axes = plt.subplots(nrows=1, ncols=len(df_dict), figsize=(5 * len(df_dict), 5))
    axes = np.atleast_1d(axes).flatten()
    train_corrs, test_corrs = {}, {}

    for i, key in enumerate(df_dict.keys()):
        df = df_dict[key]
        train_pred_returns = df[df['type']=='train']['pred_return'].values
        train_real_returns = df[df['type']=='train']['real_return'].values
        test_pred_returns = df[df['type']=='test']['pred_return'].values
        test_real_returns = df[df['type']=='test']['real_return'].values

        train_corr = np.corrcoef(train_pred_returns, train_real_returns)[0, 1]
        train_corrs[key] = train_corr
        test_corr = np.corrcoef(test_pred_returns, test_real_returns)[0, 1]
        test_corrs[key] = test_corr

        real_returns = np.concatenate([train_real_returns, test_real_returns])
        pred_returns = np.concatenate([train_pred_returns, test_pred_returns])
        real_min, real_max = real_returns.min(), real_returns.max()
        pred_min, pred_max = pred_returns.min(), pred_returns.max()

        train_indices = np.arange(len(train_real_returns))
        test_indices = np.arange(len(train_real_returns), len(real_returns))

        axes[i].scatter(x=real_returns[train_indices], y=pred_returns[train_indices], marker='.', color=COLORS[0])
        axes[i].scatter(x=real_returns[test_indices], y=pred_returns[test_indices], marker='.', color=COLORS[1])
        
        axes[i].plot([real_min, real_max], [pred_min, pred_max], linestyle='--', color='black')
        axes[i].set_title(f'{key}', fontsize=FONTSIZE+1)
        axes[i].text(0.02, 0.98, f'corr={test_corr:.2f}', transform=axes[i].transAxes, fontsize=FONTSIZE+1, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.5))
        axes[i].set_xlabel('Ground Truth Return', fontsize=FONTSIZE+1)
        axes[i].set_ylabel('Predicted Return', fontsize=FONTSIZE+1)
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

def get_stats_from_dict():
    load_dir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/corr_data'
    methods = ['airl', 'drex', 'lerp', 'ssrr']
    tasks = ['walker_walk', 'walker_run', 'cheetah_run']
    # all_train_corrs = pd.DataFrame(columns=tasks, index=methods).applymap(lambda x: [])
    all_test_corrs = pd.DataFrame(columns=tasks, index=methods).applymap(lambda x: [])
    for task_name in tasks:
        for seed in range(5):
            load_path = f'{load_dir}/{task_name}/{seed}'
            df_dict = {}
            for method in methods:
                df_dict[method] = pd.read_csv(f'{load_path}/{method}.csv', index_col=None)
            train_corrs, test_corrs = plot_return_corr_from_df(df_dict, savepath=f'{load_path}/return_corr.pdf')
            for method in methods:
                # all_train_corrs.loc[method, task_name].append(train_corrs[method])
                all_test_corrs.loc[method, task_name].append(test_corrs[method])
    all_test_corrs.to_csv(f'{load_dir}/test_return_corrs_data.csv')
    mean_var_df = all_test_corrs.applymap(lambda lst: f"{np.mean(lst):.4f} ± {np.var(lst):.4f}")
    mean_var_df.to_csv(f'{load_dir}/test_return_corrs.csv')

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
    savedir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/corr_data/{cfg.task_name}/{cfg.seed}'
    reproduce(cfg.seed)

    method = cfg.baseline_type
    # method = 'airl'
    if method == 'airl':
        reward_model = get_airl_reward(cfg)
    elif method in ['drex', 'lerp', 'ssrr']:
        reward_model = get_baseline_reward(cfg)
    elif method == 'ours':
        reward_model = get_our_reward(cfg)
    

    # load buffer 
    if method in ['drex', 'lerp', 'ssrr']:
        train_test_reward_buffer = PreferenceBuffer(cfg.baseline_train_buffer_path, cfg.baseline_train_idx_path, 500, device=device)
    else: # airl 用ours的buffer
        train_test_reward_buffer = PreferenceBuffer(cfg.train_buffer_path, cfg.train_idx_path, 500, device=device)

    train_loader = torch.utils.data.DataLoader(train_test_reward_buffer, batch_size=8, shuffle=False, num_workers=6)
    test_reward_buffer = PreferenceBuffer(cfg.test_buffer_path, cfg.test_idx_path, 500, device=device)
    test_loader = torch.utils.data.DataLoader(test_reward_buffer, batch_size=8, shuffle=False, num_workers=6)

    # get dataframes
    savepath = f'{savedir}/{method}.csv'
    train_df = get_return_corr_df(reward_model, train_loader, device=device, n_samples=500)
    train_df['type'] = 'train'
    test_df = get_return_corr_df(reward_model, test_loader, device=device, n_samples=500)
    test_df['type'] = 'test'
    df = pd.concat([train_df, test_df])
    os.makedirs(os.path.abspath(os.path.dirname(savepath)), exist_ok=True)
    df.to_csv(savepath, index=None)

    # get_stats_from_dict() # 画图以及计算数值保存csv

if __name__ == '__main__':
    main()
