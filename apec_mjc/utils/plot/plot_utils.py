from utils.plot.plot_configs import *

def get_states(preference_buffer: PreferenceBuffer, size=256):
    states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_labels = preference_buffer.get_all(device=device)
    states_1 = states_1.detach().cpu().numpy()
    states_2 = states_2.detach().cpu().numpy()
    masks_1 = masks_1.detach().cpu().numpy()
    masks_2 = masks_2.detach().cpu().numpy()
    batch_size, seq_len, state_dim = states_1.shape
    states = np.concatenate([states_1[masks_1>0], states_2[masks_2>0]], axis=0).reshape(-1, state_dim)
    sampled_states = states[np.random.choice(states.shape[0], size=size, replace=False)]
    return sampled_states

def get_trajs(preference_buffer: PreferenceBuffer, size=-1):
    states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, masks_1, masks_2, pref_labels = preference_buffer.get_all(device=device)
    states_1 = states_1.detach().cpu().numpy()
    states_2 = states_2.detach().cpu().numpy()
    actions_1 = actions_1.detach().cpu().numpy()
    actions_2 = actions_2.detach().cpu().numpy()
    rewards_1 = rewards_1.detach().cpu().numpy()
    rewards_2 = rewards_2.detach().cpu().numpy()
    masks_1 = masks_1.detach().cpu().numpy()
    masks_2 = masks_2.detach().cpu().numpy()
    states = np.concatenate([states_1, states_2], axis=0)
    actions = np.concatenate([actions_1,actions_2], axis=0)
    rewards = np.concatenate([rewards_1, rewards_2], axis=0)
    masks = np.concatenate([masks_1, masks_2], axis=0)

    size = size if size > 0 else states.shape[0]
    random_indices = np.random.choice(states.shape[0], size=size, replace=False)
    return states[random_indices], actions[random_indices], rewards[random_indices], masks[random_indices]

def load_params(path):
    load_dict = {}
    with open(path, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict

def calculate_stats(corrs):
    stats = {}
    for dataset, models in corrs.items():
        stats[dataset] = {}
        for model, values in models.items():
            mean = np.mean(values)
            std = np.std(values)
            stats[dataset][model] = f"{mean:.4f} ± {std:.4f}"

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    model_means = stats_df.map(lambda x: float(x.split(' ± ')[0]))
    mean_row = model_means.mean().round(4)
    
    # Add the mean row to the DataFrame
    stats_df.loc['Mean'] = mean_row.astype(str)
    return stats_df

def calculate_stats_mean(corrs):
    stats = {}
    for dataset, models in corrs.items():
        stats[dataset] = {}
        for model, values in models.items():
            stats[dataset][model] = np.mean(values)
        
    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.loc['Mean'] = stats_df.mean().round(3)
    return stats_df


def set_model_hypeparams(method_name):    
    '''
        load_buffer_name: 用了谁的buffer
        load_model_name: reward model存在谁的log里面
    '''
    return model_hypeparams[method_name]

def load_model(method_name, model_params, reward_indices, current_model_path):
    if method_name == 'pbrl' or method_name.startswith('ablation'):
        current_model = PrefRewardModel(input_dim=len(reward_indices), hidden_sizes=model_params['r_hidden_sizes'],
                                    hid_act=model_params['r_hid_act'], use_bn=model_params['r_use_bn'], residual=model_params['r_residual'],
                                    clamp_magnitude=model_params['r_clamp_magnitude'], device=device).to(device)
    elif method_name == 'supervised':
        current_model = SupervisedRewardModel(input_dim=len(reward_indices), hidden_sizes=model_params['r_hidden_sizes'],
                                            hid_act=model_params['r_hid_act'], use_bn=model_params['r_use_bn'], residual=model_params['r_residual'],
                                            clamp_magnitude=model_params['r_clamp_magnitude'], device=device).to(device)
    elif method_name in ['baseline_drex', 'baseline_lerp', 'baseline_gail', 'baseline_ssrr', 'baseline_ssrr_irl']:
        current_model = DrexRewardModel(input_dim=len(reward_indices), hidden_sizes=model_params['r_hidden_sizes'],
                                    hid_act=model_params['r_hid_act'], use_bn=model_params['r_use_bn'], residual=model_params['r_residual'],
                                    clamp_magnitude=model_params['r_clamp_magnitude'], device=device).to(device)
        # if method_name in ['baseline_ssrr', 'baseline_ssrr_irl']:
            # current_model.ssrr_sigmoid = SSRRSigmoid(device=device) # for load model correctly
    elif method_name in ['baseline_irl']:
        current_model = MLPReward(input_dim=len(reward_indices), hidden_sizes=model_params['r_hidden_sizes'],
                                hid_act=model_params['r_hid_act'], use_bn=model_params['r_use_bn'], residual=model_params['r_residual'],
                                clamp_magnitude=model_params['r_clamp_magnitude'], device=device).to(device)
    else:
        raise NotImplementedError
    
    current_model.load_state_dict(torch.load(current_model_path), strict=False) 
    return current_model 

def concat_figures(figures, savepath, labels, how='row'):
    n = len(figures)

    # 获取原始的 figsize
    original_figsize = figures[0].get_size_inches()  
    width, height = original_figsize  
    
    # 创建新的大图
    if how == 'row':
        fig, axs = plt.subplots(1, n, figsize=(width * n, height))
    elif how == 'col':
        fig, axs = plt.subplots(n, 1, figsize=(width, height * n))
    else:
        assert False

    for ax, figure in zip(axs, figures):
        # 清除每个 Axes 对象的图例
        for ax_fig in figure.get_axes():
            ax_fig.legend_.remove() if ax_fig.get_legend() else None
        
        # 保存当前 figure 到临时文件
        temp_path = 'temp_image.png'
        figure.savefig(temp_path, dpi=300)  # 使用高 DPI 保存

        # 读取保存的图像
        image = plt.imread(temp_path)  # 使用 plt.imread 读取图像

        # 在新的轴上显示图像
        ax.imshow(image)
        ax.axis('off')  # 关闭坐标轴

    plt.tight_layout()
    # handles = [mlines.Line2D([], [], color=COLORS[i], label=labels[i], linewidth=2) for i, label in enumerate(labels)]
    # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(handles), fontsize=FONTSIZE)
    # plt.subplots_adjust(bottom=0.2)

    plt.savefig(savepath, bbox_inches='tight', dpi=300)  # 设置 dpi
    plt.close()

    # 删除临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)