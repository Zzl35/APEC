from tensorboard.backend.event_processing import event_accumulator


from plot_scripts.plot_utils import *
from plot_scripts.plot_configs import *

import pickle

def smooth(x, window_len=11, window='hanning'):
    """平滑处理函数

    Args:
        x: 要平滑处理的数组
        window_len: 窗口长度，即平滑后的数组大小
        window: 窗口类型，如'hanning', 'hamming', 'bartlett', 'blackman'等

    Returns:
        平滑后的数组
    """
    if x.ndim != 1:
        raise ValueError("只支持一维数组的平滑处理")
    if x.size < window_len:
        raise ValueError("数组长度不能小于窗口长度")

    # 根据窗口类型生成窗口函数
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(f"不支持的窗口类型 {window}")
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # 平均窗口
        w = np.ones(window_len, 'd')
    else:
        w = eval(f'np.{window}(window_len)')

    # 求出平均值
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len//2):-(window_len//2)]

def resize_curve(y, new_length, history_window=20):
    # 滑动平均
    current_length = len(y)
    if current_length >= new_length:
        return y[:new_length]
    new_y = np.copy(y)
    while len(new_y) < new_length:
        next_value = np.mean(new_y[-history_window:])
        new_y = np.append(new_y, next_value)
    return new_y

import math
def create_subplots(num_plots):
    # 计算合适的行数和列数
    # cols = math.ceil(np.sqrt(num_plots))  # 列数：取平方根向上取整
    # rows = math.ceil(num_plots / cols)    # 行数：保证所有子图都能放下
    rows=2
    cols=math.ceil(num_plots/rows)

    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 8*rows))

    # 将axs展平成一维数组，便于循环操作
    axes = np.atleast_1d(axes).flatten()
    return fig, axes, cols,rows

class ExpertReplayBuffer:
    ### oracle rewards only for test
    def __init__(self, states=None, actions=None, rewards=None, masks=None, init_num_trajs=None, device=None):
        if states is None:
            self.states = self.actions = self.rewards = self.masks = None
        else:
            self.states = states
            self.actions = actions
            self.rewards = rewards
            self.masks = masks
        self.trajs_weight = None
        self.init_num_trajs = init_num_trajs
        self.device = device
    
    def store_batch(self, states, actions, rewards, masks):
        if self.states is None:
            self.states = states
            self.actions = actions
            self.rewards = rewards
            self.masks = masks
        else:
            self.states = np.concatenate([self.states, states], axis=0)
            self.actions = np.concatenate([self.actions, actions], axis=0)
            self.rewards = np.concatenate([self.rewards, rewards], axis=0)
            self.masks = np.concatenate([self.masks, masks], axis=0)

    def update_trajs_weight(self, pref_reward, beta, state_only=False, clip_max=None):
        self.beta = beta
        if not state_only:
            expert_trajs = np.concatenate([self.states, self.actions], axis=-1)
        else:
            expert_trajs = self.states
        with torch.no_grad():
            D_logits = pref_reward.predict_r_hat_torch(torch.FloatTensor(expert_trajs).to(self.device)).squeeze(-1)
        D_logits *= torch.FloatTensor(self.masks).to(self.device)
        self.fake_rewards = D_logits
        self.fake_returns = torch.sum(D_logits, dim=1)
        if clip_max is not None:
            self.fake_returns = torch.clamp(self.fake_returns, -float("inf"), clip_max)
        if beta > 0:
            self.trajs_weight = torch.exp(torch.clamp(self.fake_returns / beta, -float("inf"), 85))
        else:
            self.trajs_weight = torch.exp(self.fake_returns * 0)
        self.trajs_weight /= torch.sum(self.trajs_weight)

    def cail_update_trajs_weight(self, reward, beta, state_only=False, clip_max=None):
        self.beta = beta
        if not state_only:
            expert_trajs = np.concatenate([self.states, self.actions], axis=-1)
        else:
            expert_trajs = self.states
        with torch.no_grad():
            D_logits = reward.predict_r_hat_torch(torch.FloatTensor(expert_trajs).to(self.device)).squeeze(-1)
        D_logits *= torch.FloatTensor(self.masks).to(self.device)
        self.fake_rewards = D_logits
        self.fake_returns = torch.sum(D_logits, dim=1)
        if clip_max is not None:
            self.fake_returns = torch.clamp(self.fake_returns, -float("inf"), clip_max)
        if beta > 0:
            self.trajs_weight = torch.exp(torch.clamp(self.fake_returns / beta, -float("inf"), 85))
        else:
            self.trajs_weight = torch.exp(self.fake_returns * 0)
        self.trajs_weight /= torch.sum(self.trajs_weight)
    
    def sample_all_without_pref(self):
        return np.concatenate([self.states, self.actions], axis=-1), self.masks, None

    def sample_with_pref(self, num_trajs=None):
        if num_trajs is None:
            num_trajs = self.init_num_trajs
        if self.states.shape[0]<=num_trajs:
            return np.concatenate([self.states, self.actions], axis=-1), self.masks, self.trajs_weight
        else:
            index = sorted(range(self.trajs_weight.shape[0]), key=lambda k:self.trajs_weight[k])[-num_trajs:]
            return np.concatenate([self.states, self.actions], axis=-1)[index], self.masks[index], self.trajs_weight[index]

    def draw(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        oracle_returns_draw = np.sum(self.rewards * self.masks, axis=-1)
        fake_returns_draw = self.fake_returns.cpu().numpy()
        plt.scatter(oracle_returns_draw, fake_returns_draw)
        plt.xlabel("oracle returns")
        plt.ylabel("fake returns")
        plt.title("returns2fake returns")
        plt.savefig(f"{dir}/oracle_returns2fake_returns_{name}.png")
        plt.close()
        if self.beta>0:
            oracle_weights_draw = np.exp(np.clip(oracle_returns_draw / self.beta, -float("inf"), 85))
        else:
            oracle_weights_draw = np.exp(oracle_returns_draw * 0)
        oracle_weights_draw /= np.sum(oracle_weights_draw)
        fake_weights_draw = self.trajs_weight.cpu().numpy()
        plt.scatter(oracle_weights_draw, fake_weights_draw)
        plt.xlabel("oracle_weights")
        plt.ylabel("fake_weights")
        plt.title("oracle weights2fake weights")
        plt.savefig(f"{dir}/oracle_weights2fake_weights_{name}.png")
        plt.close()

    def draw_grid(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        oracle_returns_draw = np.sum(self.rewards * self.masks, axis=-1)
        fake_returns_draw = self.fake_returns.cpu().numpy()
        tar_norm = np.linalg.norm(self.states[:, 0, 2:], axis=-1)
        in_index = np.where(tar_norm<=0.75)[0]
        out_index = np.where(tar_norm>0.75)[0]
        plt.scatter(oracle_returns_draw[in_index], fake_returns_draw[in_index], label="In Circle")
        plt.scatter(oracle_returns_draw[out_index], fake_returns_draw[out_index], label="Out of Circle")
        plt.legend()
        plt.xlabel("oracle returns")
        plt.ylabel("fake returns")
        plt.title("returns2fake returns")
        plt.savefig(f"{dir}/oracle_returns2fake_returns_{name}.png")
        plt.close()
        if self.beta>0:
            oracle_weights_draw = np.exp(oracle_returns_draw / self.beta)
        else:
            oracle_weights_draw = np.exp(oracle_returns_draw * 0)
        oracle_weights_draw /= np.sum(oracle_weights_draw)
        fake_weights_draw = self.trajs_weight.cpu().numpy()
        plt.scatter(oracle_weights_draw, fake_weights_draw)
        plt.xlabel("oracle_weights")
        plt.ylabel("fake_weights")
        plt.title("oracle weights2fake weights")
        plt.savefig(f"{dir}/oracle_weights2fake_weights_{name}.png")
        plt.close()
        M_index = list(range(self.rewards.shape[0]//2))
        H_index = list(range(self.rewards.shape[0]//2, self.rewards.shape[0]))
        plt.scatter(oracle_returns_draw[M_index], fake_returns_draw[M_index], label="M")
        plt.scatter(oracle_returns_draw[H_index], fake_returns_draw[H_index], label="H")
        plt.legend()
        plt.xlabel("oracle returns")
        plt.ylabel("fake returns")
        plt.title("returns2fake returns")
        plt.savefig(f"{dir}/oracle_returns2fake_returns_level_{name}.png")
        plt.close()

    def draw_reward(self, dir, name):
        if not os.path.exists(dir):
            os.mkdir(dir)
        valid_index = np.where(self.masks.reshape((-1))==1)[0]
        oracle_rewards_draw = self.rewards.reshape((-1))[valid_index]
        fake_rewards_draw = self.fake_rewards.cpu().numpy().reshape((-1))[valid_index]
        plt.scatter(oracle_rewards_draw, fake_rewards_draw)
        plt.xlabel("oracle rewards")
        plt.ylabel("fake rewards")
        plt.title("rewards2fake rewards")
        plt.savefig(f"{dir}/oracle_rewards2fake_rewards_{name}.png")
        plt.close()

def load_expert(expert_traj_nums, env_name, device):
    ### load expert
    level2num = {"L":expert_traj_nums[0], "M":expert_traj_nums[1], "H":expert_traj_nums[2], "P":expert_traj_nums[3]}
    with open("/home/ubuntu/duxinghao/imitation_pref/expert_data/{}_expert.pkl".format(env_name), 'rb') as f:
        total_data = pickle.load(f)
    expert_data_dict = {}
    for level in ["L", "M", "H", "P"]:
        level_traj_num = level2num[level]
        if level_traj_num>0:
            for key in total_data[level].keys():
                if expert_data_dict.get(key) is None:
                    expert_data_dict[key] = total_data[level][key][:level_traj_num, ...]
                else:
                    expert_data_dict[key] = np.concatenate((expert_data_dict[key], total_data[level][key][:level_traj_num, ...]), axis=0)
    expert_rewards = expert_data_dict["rewards"]
    ###

    ### add expert data into expert replay buffer
    expert_replay_buffer = ExpertReplayBuffer(states=expert_data_dict['states'],
                                              actions=expert_data_dict['actions'],
                                              rewards=expert_data_dict['rewards'],
                                              masks=expert_data_dict['masks'],
                                              init_num_trajs=sum(expert_traj_nums),
                                              device=device)
    
    return expert_replay_buffer

def visual_reusability(methods_mjc, methods_dmc, savefig=True, savepath=None):
    FONTSIZE = 27
    env_names_mjc = ['AntFH-v0', 'HalfCheetahFH-v0', 'HopperFH-v0', 'HumanoidFH-v0', 'Walker2dFH-v0']
    env_names_dmc = ['cheetah_run', 'walker_run', 'walker_walk']

    # Mujoco tasks
    timesteps = 2400
    fig, axes, ncols, nrows = create_subplots(len(env_names_mjc + env_names_dmc))
    for j, env_name in enumerate(env_names_mjc):
        expert_buffer = load_expert([int(num) for num in '0_0_1_0'.split("_")], env_name, device)
        expert_return = np.sum(expert_buffer.rewards, axis=-1)[0]
        axes[j].plot(range(timesteps), [expert_return for _ in range(timesteps)], label='demos', linestyle='--', color=COLORS[0], linewidth=5)

        for i, key in enumerate(methods_mjc.keys()):
            method = methods_mjc[key]
            logdir = os.path.join('/home/ubuntu/duxinghao/imitation_pref/logfile', 'eval_reward', method, ENV2ENV_NAME[env_name])
            curves = []
            if os.path.exists(logdir):
                for dirname in os.listdir(logdir):
                    tb_dir = os.path.join(logdir, dirname, 'tbfile')
                    tb_path = os.path.join(tb_dir, os.listdir(tb_dir)[0])
                    ea = event_accumulator.EventAccumulator(tb_path)
                    ea.Reload()
                    if 'sac/test_eval' not in ea.Tags()['scalars']:
                        print(f'Tensorboard@{tb_dir} not contains sac/test_eval, skipped.')
                        continue
                    training_curve = ea.scalars.Items('sac/test_eval')
                    curve = np.array([dot.value for dot in training_curve])[:timesteps]
                    curve_resized = resize_curve(curve, new_length=timesteps)
                    curve_smoothed = gaussian_filter1d(curve_resized, sigma=5)
                    df = pd.DataFrame()
                    df['Training Iteration'] = np.arange(len(curve_smoothed))
                    df['Episode Return'] = curve_smoothed
                    curves.append(df)
            else:
                print(f'Logfile@{logdir} not exists, skipped.')
                continue
            curves = pd.concat(curves, axis=0, ignore_index=True)
            sns.lineplot(curves, x='Training Iteration', y='Episode Return', ax=axes[j], label=key, color=COLORS[i+1], errorbar='sd', err_kws={"alpha": 0.1}, legend=False, linewidth=4)
            print(f'{env_name}-{method}-done.')

    # dmc tasks
    timesteps = 100
    for j, env_name in enumerate(env_names_dmc):
        new_j = j+len(env_names_mjc)
        with open(f'../expert_demos/dmc/{env_name}/expert_demo_suboptimal.pkl', 'rb') as f:
            _, _, _, expert_reward = pickle.load(f)
        expert_reward_mean = np.mean(expert_reward[:10].sum(axis=1))

        axes[j+len(env_names_mjc)].plot(range(timesteps), [expert_reward_mean for _ in range(timesteps)], label='demos', linestyle='--', color=COLORS[0], linewidth=5)

        for i, key in enumerate(methods_dmc.keys()):
            method = methods_dmc[key]
            if method in ['airl', 'drex', 'lerp', 'ssrr']:
                logdir = os.path.join('exp_local', f'eval_baseline', 'dmc', env_name, method)
            else:
                logdir = os.path.join('exp_local', f'eval_reward', 'dmc', env_name)
            curves = []
            if os.path.exists(logdir):
                for dirname in os.listdir(logdir):
                    tb_dir = os.path.join(logdir, dirname, 'tb')
                    if len(os.listdir(tb_dir)) > 1:
                        print(f'logfile@{tb_dir} is not unique, please check.')
                    tb_path = os.path.join(tb_dir, sorted(os.listdir(tb_dir))[-1])
                    ea = event_accumulator.EventAccumulator(tb_path)
                    ea.Reload()
                    if 'eval/episode_reward_real' not in ea.Tags()['scalars']:
                        print(f'Tensorboard@{tb_dir} not contains eval/episode_reward_real, skipped.')
                        continue
                    training_curve = ea.scalars.Items('eval/episode_reward_real')
                    curve = np.array([dot.value for dot in training_curve])[:timesteps]
                    curve_resized = resize_curve(curve, new_length=timesteps)
                    curve_smoothed = gaussian_filter1d(curve_resized, sigma=0.5)
                    df = pd.DataFrame()
                    df['Training Iteration'] = np.arange(len(curve_smoothed))
                    df['Episode Return'] = curve_smoothed
                    curves.append(df)
            else:
                print(f'Logfile@{logdir} not exists, skipped.')
                continue
            curves = pd.concat(curves, axis=0, ignore_index=True)
            sns.lineplot(curves, x='Training Iteration', y='Episode Return', ax=axes[new_j], label=key, color=COLORS[i+1], errorbar='sd', err_kws={"alpha": 0.1}, legend=False, linewidth=4)
            print(f'{env_name}-{method}-done.')
    
    for i, env_name in enumerate(env_names_mjc+env_names_dmc):
        axes[i].grid(True)
        if env_name.endswith('FH-v0'):
            axes[i].set_title(env_name.replace('FH-v0', '-v2'), fontsize=FONTSIZE+3)
        else:
            axes[i].set_title(env_name, fontsize=FONTSIZE+1)

        if env_name in env_names_dmc:
            axes[i].set_xlabel('', fontsize=FONTSIZE+3)
            axes[i].set(xlabel=r"Training Frames ($\times 10^4$)", xticks=[0, 20, 40, 60, 80, 100], xticklabels=[0, 40, 80, 120, 160, 200])
        else:
            axes[i].set_xlabel('', fontsize=FONTSIZE+3)
            axes[i].set(xlabel=r"Training Steps ($\times 10^4$)", xticks=[0, 500, 1000, 1500, 2000, 2500], xticklabels=[0, 50, 100, 150, 200, 250])
        axes[i].tick_params(labelsize=FONTSIZE-5)
        if(i % ncols == 0):
            axes[i].set_ylabel("Episode Return", fontsize=FONTSIZE+3)
        else:
            axes[i].set_ylabel('', fontsize=FONTSIZE+1)

    plt.tight_layout()
    handles = [mlines.Line2D([], [], color=COLORS[i+1], label=method, linewidth=5) for i, method in enumerate(methods_mjc.keys())]
    handles.insert(0, mlines.Line2D([], [], color=COLORS[0], label='Demos.(Avg)', linewidth=5, linestyle='--'))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(handles), fontsize=FONTSIZE+3)
    plt.subplots_adjust(bottom=0.2, hspace=0.25, wspace=0.2)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savefig:
        plt.savefig(savepath)
    plt.close()


if __name__ == '__main__':
    methods_dmc = {
        'SSRR': 'ssrr',
        'AIRL': 'airl',
        'D-REX': 'drex',
        'LERP': 'lerp',
        'APEC(ours)': 'ours',
    }
    methods_mjc = {method2label[method]:method for method in ['baseline_ssrr', 'baseline_irl', 'baseline_drex', 'baseline_lerp', 'ablation_nonoise']}

    visual_reusability(methods_mjc, methods_dmc,
                       savepath='exp_local/plot_output0129/reusability.pdf')
    
