from sklearn.manifold import TSNE

from pbrl_utils.plot.plot_utils import *
from pbrl_utils.plot.plot_configs import *

def tsne_visualization(data: dict, title=None, savefig=True, savepath=None, storefig=True):
    '''
    data : {LABEL_OF_SCATTER: trajs(sample_num, seq_len, n_dim)}
    '''
    tsne = TSNE(n_components=2)
    all_data, lengths = [], []
    
    for key in data.keys():
        all_data.append(data[key].reshape(-1, data[key].shape[-1]))
        lengths.append(len(all_data[-1]))
        
    all_data = np.concatenate(all_data)
    assert sum(lengths) == all_data.shape[0]
    
    all_data_tsne = tsne.fit_transform(all_data)
    all_min, all_max = all_data_tsne.min(0), all_data_tsne.max(0)
    all_data_tsne_norm = (all_data_tsne - all_min) / (all_max - all_min + EPS)  # normalize

    data_tsne = {}
    start_id = 0
    for i, key in enumerate(data.keys()):
        data_tsne[key] = all_data_tsne_norm[start_id: start_id + lengths[i]]
        start_id += lengths[i]

    labels = list(data.keys())
    num_labels = len(labels)
    fig, axes = plt.subplots(1, num_labels - 1, figsize=(5 * (num_labels - 1), 5))
    axes = np.atleast_1d(axes).flatten()

    first_data_xy = data_tsne[labels[0]]
    for i in range(1, num_labels):
        ax = axes[i - 1]
        current_data_xy = data_tsne[labels[i]]
        
        ax.scatter(first_data_xy[:, 0], first_data_xy[:, 1], s=10, label=labels[0], color=COLORS[0], alpha=0.5)
        ax.scatter(current_data_xy[:, 0], current_data_xy[:, 1], s=10, label=labels[i], color=COLORS[i], alpha=0.5)
        # ax.set_title(f'Comparison: {labels[0]} vs {labels[i]}')
        ax.tick_params(labelsize=FONTSIZE)

    if title is not None:
        plt.suptitle(title, fontsize=FONTSIZE+1)

    if not os.path.exists(os.path.abspath(os.path.dirname(savepath))):
        os.makedirs(os.path.abspath(os.path.dirname(savepath)))
    if savefig and (savepath is not None):
        plt.savefig(savepath)

    # plt.close()
    return fig


def plot_knn_consistency(total_diversity, savefig=True, savepath=None):
    cons_mean_df = calculate_stats_mean(total_diversity)
    # cons_mean_df = (cons_mean_df - cons_mean_df.min().min()) / (cons_mean_df.max().max() - cons_mean_df.min().min()) # global normalize
    df_melted = pd.melt(cons_mean_df.reset_index(), id_vars=['index'], var_name='Method', value_name='Preference Accuracy')

    plt.figure(figsize=(15, 6))
    ax = sns.barplot(x='index', y='Preference Accuracy', hue='Method', data=df_melted, dodge=True, width=0.5, palette=COLORS)
    plt.xticks(fontsize=FONTSIZE-1)
    plt.yticks(fontsize=FONTSIZE)
    plt.xlabel('', fontsize=FONTSIZE+1)
    plt.ylabel('KNN Entropy', fontsize=FONTSIZE+1)
    
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

def compute_knn_entropy(embeddings, k=1024):
    # print('embeddings:', embeddings.shape)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    # Skip the first column (distance to self, which is 0)
    distances = distances[:, 1:]
    entropy_estimates = np.log(distances + 1)
    # print(embeddings.shape, entropy_estimates.shape, distances.shape)
    diversity_result = entropy_estimates.mean()
    return diversity_result

def compute_diversity(models: dict, samples: dict):
    '''
    Compute return/reward label correctness with our reward.
    models: {LABEL: model}
    samples: {LABEL: [train_samples, test_samples]}
    batch_size: Number of samples to process in each batch
    '''
    diversity = {}
    for key in models:
        diversity[key] = compute_knn_entropy(samples[key])

    return diversity