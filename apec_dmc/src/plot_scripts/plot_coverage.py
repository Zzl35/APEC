from sklearn.manifold import TSNE

from utils.plot.plot_utils import *
from utils.plot.plot_configs import *

def min_l1_distance(a, b):
    distances = np.abs(b[:, None, :] - a[None, :, :]).mean(axis=2)
    return distances.min(axis=1)

def compute_coverage(models: dict, samples: dict):
    '''
    Compute return/reward label correctness with our reward.
    models: {LABEL: model}
    samples: {LABEL: [train_samples, test_samples]}
    '''
    coverage = {}
    for key in models:
        train, test = samples[key]
        coverage[key] = min_l1_distance(train, test).mean()

    return coverage

def plot_coverage(total_diversity, savefig=True, savepath=None):
    cons_mean_df = calculate_stats_mean(total_diversity)
    df_melted = pd.melt(cons_mean_df.reset_index(), id_vars=['index'], var_name='Method', value_name='Preference Accuracy')

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='index', y='Preference Accuracy', hue='Method', data=df_melted, dodge=True, width=0.5, palette=COLORS)
    plt.xticks(fontsize=FONTSIZE-1)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('', fontsize=FONTSIZE+1)
    plt.ylabel('Min L1 distance', fontsize=FONTSIZE+1)
    
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

    return total_diversity 