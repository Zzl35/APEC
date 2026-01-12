from pbrl_utils.plot.plot_utils import *
from pbrl_utils.plot.plot_configs import *

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
    inputs = np.concatenate([states, actions], axis=-1) if alg in ['maxentirl_sa'] else states
    inputs = inputs[masks > 0]
    real_rewards = rewards
    real_min, real_max = real_rewards.min(), real_rewards.max()
    batch_size = 16
    for i, key in enumerate(models.keys()):
        pred_rewards_list = []

        # 使用分批次处理来减少内存占用
        for start_idx in range(0, inputs.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, inputs.shape[0])
            batch_inputs = torch.tensor(inputs[start_idx:end_idx], dtype=torch.float32).to(models[key].device)
            with torch.no_grad():
                batch_pred_rewards = models[key].r(batch_inputs).cpu().numpy()
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

def return_correlation(models: dict, samples: dict, savefig=True, savepath=None, alg='maxentirl', storefig=False):
    '''
    Show return/reward correlation with true results.
    rewards: {LABEL: model}
    samples: {LABEL: [train_buffer, test_buffer]}
    '''
    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize=(5 * len(models), 5))
    axes = np.atleast_1d(axes).flatten()
    train_corrs, test_corrs = {}, {}

    for i, key in enumerate(models.keys()):
        train_states, train_actions, train_rewards, train_masks = samples[key][0]
        test_states, test_actions, test_real_rewards, test_masks = samples[key][1]

        train_real_returns = np.sum(train_rewards * train_masks, axis=-1)
        train_inputs = np.concatenate([train_states, train_actions], axis=-1) if alg in ['maxentirl_sa'] else train_states
        
        test_real_returns = np.sum(test_real_rewards * test_masks, axis=-1)
        test_pred_inputs = np.concatenate([test_states, test_actions], axis=-1) if alg in ['maxentirl_sa'] else test_states

        train_pred_rewards_list = []
        test_pred_rewards_list = []
        batch_size = 128

        for start_idx in range(0, train_inputs.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, train_inputs.shape[0])
            batch_inputs = torch.tensor(train_inputs[start_idx:end_idx], dtype=torch.float32).to(models[key].device)
            with torch.no_grad():
                batch_pred_rewards = models[key].r(batch_inputs).cpu().numpy()
                train_pred_rewards_list.append(batch_pred_rewards)

        train_pred_rewards = np.concatenate(train_pred_rewards_list).squeeze()
        train_pred_returns = np.sum(train_pred_rewards * train_masks, axis=-1)

        for start_idx in range(0, test_pred_inputs.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, test_pred_inputs.shape[0])
            batch_inputs = torch.tensor(test_pred_inputs[start_idx:end_idx], dtype=torch.float32).to(models[key].device)
            with torch.no_grad():
                batch_pred_rewards = models[key].r(batch_inputs).cpu().numpy()
                test_pred_rewards_list.append(batch_pred_rewards)

        test_pred_rewards = np.concatenate(test_pred_rewards_list).squeeze()
        test_pred_returns = np.sum(test_pred_rewards * test_masks, axis=-1)

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
        if storefig:
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
    if savefig:
        plt.savefig(savepath)
    # plt.close()

    return train_corrs, test_corrs, fig

