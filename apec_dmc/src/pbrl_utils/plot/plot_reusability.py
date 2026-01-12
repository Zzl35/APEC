from tensorboard.backend.event_processing import event_accumulator


from pbrl_utils.plot.plot_utils import *
from pbrl_utils.plot.plot_configs import *

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
    rows=1
    cols=math.ceil(num_plots/rows)

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))

    # 将axs展平成一维数组，便于循环操作
    axes = np.atleast_1d(axes).flatten()
    return fig, axes, cols,rows

def visual_reusability(methods, env_names, expert_returns, savefig=True, savepath=None):
    timesteps = 3000
    fig, axes, ncols, nrows = create_subplots(len(env_names))
    for j, env_name in enumerate(env_names):
        axes[j].plot(range(timesteps), [expert_returns[env_name] for _ in range(timesteps)], label='demos', linestyle='--', color=COLORS[-1])

        for i, key in enumerate(methods.keys()):
            method = methods[key]
            logdir = os.path.join('logfile', 'eval_reward', method, ENV2ENV_NAME[env_name])
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
            sns.lineplot(curves, x='Training Iteration', y='Episode Return', ax=axes[j], label=key, color=COLORS[i], errorbar='sd', err_kws={"alpha": 0.1}, legend=False, linewidth=3)
            
        axes[j].grid(True)
        axes[j].set_title(env_name, fontsize=FONTSIZE+1)
        axes[j].set_xlabel("Training Iterations", fontsize=FONTSIZE+1)
        axes[j].tick_params(labelsize=FONTSIZE)
        if(j * ncols == 0):
            axes[j].set_ylabel("Episode Return", fontsize=FONTSIZE+1)
        else:
            axes[j].set_ylabel('', fontsize=FONTSIZE+1)

    plt.tight_layout()
    handles = [mlines.Line2D([], [], color=COLORS[i], label=method, linewidth=5) for i, method in enumerate(methods)]
    handles.append(mlines.Line2D([], [], color=COLORS[-1], label='demos', linewidth=5))
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=len(handles), fontsize=FONTSIZE)
    plt.subplots_adjust(bottom=0.27)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savefig:
        plt.savefig(savepath)
    plt.close()

