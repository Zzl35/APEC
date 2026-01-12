from pbrl_utils.plot.plot_utils import *
from pbrl_utils.plot.plot_configs import *
from train_pbrl import try_evaluate as eval_reward_acc


def visual_accuracy(methods, env_names, savefig=True, savepath=None):
    total_accs = {env_name: {method: [] for method in methods} for env_name in env_names}
    target_key = 'tb/gen_accuracy'
    
    for j, env_name in enumerate(env_names):
        for i, key in enumerate(methods.keys()):
            method = methods[key]
            alg, load_buffer_name, load_model_name = set_model_hypeparams(method)
            logdir = os.path.join('logfile', 'train_pbrl', method, ENV2ENV_NAME[env_name], '0_0_1_0')
            if os.path.exists(logdir):
                for dirname in os.listdir(logdir):
                    tb_dir = os.path.join(logdir, dirname, 'tbfile')
                    tb_path = os.path.join(tb_dir, os.listdir(tb_dir)[0])
                    ea = event_accumulator.EventAccumulator(tb_path)
                    ea.Reload()
                    if target_key not in ea.Tags()['scalars']:
                        print(f'Tensorboard@{tb_dir} not contains {target_key}, skipped.')
                        continue
                    acc = ea.scalars.Items(target_key)[0].value
                    total_accs[env_name][key].append(1-acc if load_buffer_name in ['baseline_lerp'] else acc)
            else:
                print(f'Logfile@{logdir} not exists, skipped.')
                continue   

    acc_mean_df = calculate_stats_mean(total_accs)
    df_melted = pd.melt(acc_mean_df.reset_index(), id_vars=['index'], var_name='Method', value_name='Preference Accuracy')

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='index', y='Preference Accuracy', hue='Method', data=df_melted, dodge=True, width=0.5, palette=COLORS)
    plt.xticks(fontsize=FONTSIZE-1)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('', fontsize=FONTSIZE-1)
    plt.ylabel('Preference Accuracy', fontsize=FONTSIZE+1)
    
    handles = []
    for i, key in enumerate(methods.keys()):
        handles.append(mlines.Line2D([], [], color=COLORS[i], label=key, linewidth=5))
    
    plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(handles), fontsize=FONTSIZE+1)
    plt.tight_layout()

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savefig:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()

    return total_accs

# def compute_consistency(models: dict, samples: dict, savefig=True, savepath=None, alg='maxentirl'):
#     '''
#     Compute return/reward label correctness with our reward.
#     models: {LABEL: model}
#     buffers: {LABEL: [train_buffer, test_buffer]}
#     '''
#     train_cons, test_cons = {}, {}
#     for i, key in enumerate(models.keys()):
#         train_cons[key] = eval_reward_acc(alg, samples[key][0], models[key])['PrefLabelAcc']
#         test_cons[key] = eval_reward_acc(alg, samples[key][1], models[key])['PrefLabelAcc']

#     return train_cons, test_cons

def compute_consistency_batch(train_samples, batch_size, alg, model, device):
    accuracy = []
    
    for i in range(0, len(train_samples[0]), batch_size):
        batch_samples = tuple(torch.tensor(arr[i:i + batch_size], dtype=torch.float32).to(device) for arr in train_samples)
        accuracy.extend([eval_reward_acc(alg, batch_samples, model)['PrefLabelAcc']]*len(batch_samples))
    
    return np.mean(accuracy)

def compute_consistency(models: dict, samples: dict, batch_size=256, savefig=True, savepath=None, alg='maxentirl', device='cpu'):
    '''
    Compute return/reward label correctness with our reward.
    models: {LABEL: model}
    samples: {LABEL: [train_samples, test_samples]}
    batch_size: Number of samples to process in each batch
    '''
    train_cons, test_cons = {}, {}

    for key in models.keys():
        train_cons[key] = compute_consistency_batch(samples[key][0], batch_size, alg, models[key], device)
        test_cons[key] = compute_consistency_batch(samples[key][1], batch_size, alg, models[key], device)

    return train_cons, test_cons


def visual_consistency(total_cons, savefig=True, savepath=None):
    cons_mean_df = calculate_stats_mean(total_cons)
    df_melted = pd.melt(cons_mean_df.reset_index(), id_vars=['index'], var_name='Method', value_name='Preference Accuracy')

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='index', y='Preference Accuracy', hue='Method', data=df_melted, dodge=True, width=0.5, palette=COLORS)
    plt.xticks(fontsize=FONTSIZE-1)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('', fontsize=FONTSIZE+1)
    plt.ylabel('Preference Accuracy', fontsize=FONTSIZE+1)
    
    handles = []
    for i, key in enumerate(cons_mean_df.columns):
        handles.append(mlines.Line2D([], [], color=COLORS[i], label=key, linewidth=5))
    
    plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=len(handles), fontsize=FONTSIZE+1)
    plt.tight_layout()

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savefig:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()

    return total_cons