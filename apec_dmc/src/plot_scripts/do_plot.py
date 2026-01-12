import sys
sys.path.append('.') # 放在config里失效

import gym
# import envs

from datetime import datetime
from parameter.Parameter import Parameter
from sample import load_expert


from utils import system
from utils.plot.plot_utils import *
from utils.plot.plot_configs import *
from utils.plot.plot_diversity import tsne_visualization, compute_diversity, plot_knn_consistency
from utils.plot.plot_correlation import return_correlation, reward_correlation
from utils.plot.plot_reusability import visual_reusability
from utils.plot.plot_accuracy import compute_consistency, visual_consistency, visual_accuracy
from utils.plot.plot_coverage import compute_coverage, plot_coverage


if __name__ == '__main__':
    parameter = Parameter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=5, suppress=True)
    pid=os.getpid()
    
    plot_dir = parameter.plot_dir if parameter.plot_dir is not None else datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    expert_traj_nums = [int(num) for num in parameter.expert_traj_nums.split("_")]
    expert_returns = {}
    total_reward_corrs = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}
    total_train_return_corrs = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}
    total_test_return_corrs = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}

    total_train_cons = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}
    total_test_cons = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}
    
    total_diversity = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}
    total_coverage = {dataset:{method2label[model]:[] for model in parameter.plot_methods} for dataset in parameter.plot_datasets}

    tsne_figs = {dataset:[False for seed in parameter.plot_seeds] for dataset in parameter.plot_datasets}

    reward_corr_figs = {dataset:[False for seed in parameter.plot_seeds] for dataset in parameter.plot_datasets}
    return_corr_figs = {dataset:[False for seed in parameter.plot_seeds] for dataset in parameter.plot_datasets}

    # 这些用作之后拼接
    if 'AntFH-v0' in parameter.plot_datasets:
        tsne_figs['AntFH-v0'][1] = True
        reward_corr_figs['AntFH-v0'][1] = True
        return_corr_figs['AntFH-v0'][1] = True

    if 'HalfCheetahFH-v0' in parameter.plot_datasets:
        tsne_figs['HalfCheetahFH-v0'][0] = True
        reward_corr_figs['HalfCheetahFH-v0'][1] = True
        return_corr_figs['HalfCheetahFH-v0'][1] = True

    if 'HopperFH-v0' in parameter.plot_datasets:
        tsne_figs['HopperFH-v0'][0] = True
        reward_corr_figs['HopperFH-v0'][1] = True
        return_corr_figs['HopperFH-v0'][1] = True

    if 'HumanoidFH-v0' in parameter.plot_datasets:
        tsne_figs['HumanoidFH-v0'][0] = True
        reward_corr_figs['HumanoidFH-v0'][0] = True
        return_corr_figs['HumanoidFH-v0'][0] = True

    if 'Walker2dFH-v0' in parameter.plot_datasets:
        tsne_figs['Walker2dFH-v0'][1] = True
        reward_corr_figs['Walker2dFH-v0'][2] = True
        return_corr_figs['Walker2dFH-v0'][2] = True

    require_buffer = True if parameter.plot_diversity or parameter.plot_corr or parameter.plot_consistency or parameter.plot_cover else False
    require_model = True if parameter.plot_corr or parameter.plot_consistency else False
    for dataset in parameter.plot_datasets:
        # common parameters
        alg_name = parameter.alg_name
        dataset = dataset
        env_fn = lambda : gym.make(dataset)
        gym_env = env_fn()
        state_size = gym_env.observation_space.shape[0]
        action_size = gym_env.action_space.shape[0]
        state_indices = list(range(state_size))
        action_indices = list(range(action_size))
        if alg_name in ['maxentirl_sa', 'gail']:
            reward_indices = list(range(state_size + action_size))
        elif alg_name in ['maxentirl']:
            reward_indices = list(range(state_size))

        expert_buffer = load_expert([int(num) for num in parameter.expert_traj_nums.split("_")], dataset, device)
        expert_return = np.sum(expert_buffer.rewards, axis=-1)[0]
        expert_returns[dataset] = expert_return
        expert_trajs = (expert_buffer.states, expert_buffer.actions, expert_buffer.rewards, expert_buffer.masks) 
        
        for seed in parameter.plot_seeds:
            seed = int(seed)
            system.reproduce(seed)    

            if require_buffer:
                test_replay_path = os.path.join('buffer', dataset, 'maxentirl_sa', '0_0_0_1', '1', 'test_pbrl_buffer.npz')
                test_buffer = PreferenceBuffer(memory_size=500)
                test_buffer.load(test_replay_path)
                test_trajs = get_trajs(test_buffer)
                test_states = get_states(test_buffer, size=4000)
                test_buffer_all = test_buffer.get_all_np()

            model_buffers = {}
            models = {}
            for method_name in parameter.plot_methods:
                print(f"Preparing-{dataset}-{seed}-{method_name}")
                algorithm, load_buffer_name, load_model_name = set_model_hypeparams(method_name)
                
                current_model = None
                model_exist = buffer_exist = False
                load_model_dir = os.path.join('logfile', f'train_{algorithm}', load_model_name, ENV2ENV_NAME[dataset], parameter.expert_traj_nums, f'{dataset}_maxentirl_use_pref_False_seed_{seed}_sac_epochs_5_sac_alpha_0.1_{parameter.expert_traj_nums}_last_n_samples-DEBUG-NEW')
                load_model_path = os.path.join(load_model_dir, 'best_model', 'reward_model_last.pkl')
                if os.path.exists(load_model_path) and require_model:
                    model_exist = True
                    model_params = load_params(os.path.join(load_model_dir, 'config', 'parameter.json'))
                    current_model = load_model(method_name, model_params, reward_indices, load_model_path)
                    models[method2label[method_name]]= current_model

                # if model exists, load buffer
                load_buffer_path = os.path.join('buffer', dataset, load_buffer_name, parameter.expert_traj_nums, str(seed), f'train_buffer_{load_buffer_name}.npz')     
                if os.path.exists(load_buffer_path) and require_buffer:
                    buffer_exist = True
                    current_buffer = PreferenceBuffer(memory_size=500)
                    current_buffer.load(load_buffer_path)
                    model_buffers[method2label[method_name]] = current_buffer

                if parameter.plot_corr and not (model_exist and buffer_exist): # Load sth error
                    print(f'Skipped {dataset}-{method_name}-{seed}:')
                    if not model_exist:
                        print(f'Model@{load_model_path} not found')
                    if not buffer_exist:
                        print(f'Buffer@{load_buffer_path} not found')
            
            if len(model_buffers) > 0 and parameter.plot_diversity:
                tsne_fig = tsne_visualization(
                    data={method_name:get_states(model_buffers[method_name], size=400 if method_name.startswith(method2label['ablation_nonoise']) else 200) for method_name in model_buffers.keys()}, 
                    title=dataset, 
                    savefig=True, 
                    savepath=f'eval/{plot_dir}/diversity/{dataset}/tsne_seed{seed}.pdf',
                    storefig=tsne_figs[dataset][seed])

                if tsne_figs[dataset][seed]:
                    tsne_figs[dataset][seed] = tsne_fig
                else:
                    plt.close(tsne_fig)

                diverse = compute_diversity(
                    models=[m for m in model_buffers.keys()],
                    samples={method_name:get_states(model_buffers[method_name], size=4000 if method_name.startswith(method2label['ablation_nonoise']) else 2000) for method_name in model_buffers.keys()},
                )
                # print(f'diverse: {diverse}')
                for m in diverse.keys():
                    total_diversity[dataset][m].append(diverse[m])
            else:
                if parameter.plot_diversity:
                    print('tsne visual error, not enough buffers')
            
            if len(model_buffers) > 0 and parameter.plot_cover:
                cover = compute_coverage(
                    models=[m for m in model_buffers.keys()],
                    samples={method_name:(get_states(model_buffers[method_name], size=4000 if method_name.startswith(method2label['ablation_nonoise']) else 2000), test_states) for method_name in model_buffers.keys()},
                )
                # print(f'diverse: {diverse}')
                for m in cover.keys():
                    total_coverage[dataset][m].append(cover[m])
            
            if len(models) > 0 and parameter.plot_corr:
                
                reward_corrs, reward_fig = reward_correlation(
                    models=models, 
                    samples=test_trajs, 
                    savefig=True, 
                    savepath=f'eval/{plot_dir}/reward_correlation/{dataset}/seed{seed}.pdf',
                    storefig=tsne_figs[dataset][seed]
                )

                train_return_corrs, test_return_corrs, return_fig = return_correlation(
                    models=models, 
                    samples={method_name:[get_trajs(model_buffers[method_name], size=1000), test_trajs] for method_name in model_buffers.keys()}, 
                    savefig=True, 
                    savepath=f'eval/{plot_dir}/return_correlation/{dataset}/seed{seed}.pdf',
                    storefig=tsne_figs[dataset][seed]
                )
                if reward_corr_figs[dataset][seed]:
                    reward_corr_figs[dataset][seed] = reward_fig
                else:
                    plt.close(reward_fig)
                if return_corr_figs[dataset][seed]:
                    return_corr_figs[dataset][seed] = return_fig
                else:
                    plt.close(return_fig)
                
                for m in reward_corrs.keys():
                    total_reward_corrs[dataset][m].append(reward_corrs[m])
                    total_train_return_corrs[dataset][m].append(train_return_corrs[m])
                    total_test_return_corrs[dataset][m].append(test_return_corrs[m])

            if parameter.plot_consistency:
                train_cons, test_cons = compute_consistency(
                    models=models,
                    samples={method_name:[model_buffers[method_name].get_all_np(), test_buffer_all] for method_name in model_buffers.keys()},
                    savefig=True, 
                    savepath=f'eval/{plot_dir}/reward_correlation/{dataset}/seed{seed}.pdf',
                    device=device
                )
                for m in train_cons.keys():
                    total_train_cons[dataset][m].append(train_cons[m])
                    total_test_cons[dataset][m].append(test_cons[m])

    if parameter.plot_corr:
        # 只有绘制了correlation才可以保存
        reward_df = pd.DataFrame.from_dict(total_reward_corrs, orient='index')
        test_return_df = pd.DataFrame.from_dict(total_test_return_corrs, orient='index')
        reward_df.to_csv(f'eval/{plot_dir}/reward_correlation/total_reward_correlation_data.csv')
        test_return_df.to_csv(f'eval/{plot_dir}/return_correlation/total_test_return_correlation_data.csv')
        reward_stats_df = calculate_stats(total_reward_corrs) 
        test_return_stats_df = calculate_stats(total_test_return_corrs)
        reward_stats_df.to_csv(f'eval/{plot_dir}/reward_correlation/total_reward_correlation.csv')  
        test_return_stats_df.to_csv(f'eval/{plot_dir}/return_correlation/total_test_return_correlation.csv')    
  
            
    if parameter.plot_reuse: # load every log despite plot_seeds
        visual_reusability(
            methods={method2label[method]:method for method in parameter.plot_methods}, 
            env_names=parameter.plot_datasets,
            expert_returns=expert_returns,
            savefig=True,
            savepath=f'eval/{plot_dir}/total_reusability.pdf')
    
    if parameter.plot_acc:
        visual_accuracy(
                methods={method2label[method]:method for method in parameter.plot_methods}, 
                env_names=parameter.plot_datasets,
                savefig=True,
                savepath=f'eval/{plot_dir}/accuracy/total_generate_accuracy.pdf')
    
    if parameter.plot_consistency: # load every log despite plot_seeds
        test_cons_df =  pd.DataFrame.from_dict(test_cons)
        test_cons_df.to_csv(f'eval/{plot_dir}/accuracy/total_test_consistency_data.csv') 
        calculate_stats(test_cons_df).to_csv(f'eval/{plot_dir}/accuracy/total_test_consistency.csv') 
        test_cons = visual_consistency(
                total_cons=total_test_cons,
                savefig=True,
                savepath=f'eval/{plot_dir}/accuracy/total_test_consistency.pdf')
        test_cons_df =  pd.DataFrame.from_dict(test_cons)
        test_cons_df.to_csv(f'eval/{plot_dir}/accuracy/total_test_consistency_data.csv') 
        calculate_stats(test_cons_df).to_csv(f'eval/{plot_dir}/accuracy/total_test_consistency.csv') 

    if parameter.plot_diversity:
        diverse_df = pd.DataFrame.from_dict(total_diversity, orient='index')
        diverse_df.to_csv(f'eval/{plot_dir}/diversity/total_knn_diversity.csv')
        plot_knn_consistency(
            total_diversity,
            savepath=f'eval/{plot_dir}/diversity/knn_diversity.pdf'
        )
        # # 自己选一点好的图拼在一起
        concat_figures(figures=[tsne_figs[dataset][seed] for dataset, seeds in tsne_figs.items() for seed, value in enumerate(seeds) if value],
                    savepath=f'eval/{plot_dir}/diversity/total_tsne.pdf',
                    labels=[method2label[m] for m in parameter.plot_methods],
                    how='row',)
    if parameter.plot_cover:
        plot_coverage(
            total_coverage,
            savepath=f'eval/{plot_dir}/coverage/l1_coverage.pdf'
        )
        coverage_df = pd.DataFrame.from_dict(total_coverage, orient='index')
        coverage_df.to_csv(f'eval/{plot_dir}/coverage/total_coverage.csv')

    if parameter.plot_corr:
        concat_figures(figures=[reward_corr_figs[dataset][seed] for dataset, seeds in reward_corr_figs.items() for seed, value in enumerate(seeds) if value],
                       savepath=f'eval/{plot_dir}/reward_correlation/total_reward_correlation.pdf',
                       labels=[method2label[m] for m in parameter.plot_methods],
                       how='col',)
        concat_figures(figures=[return_corr_figs[dataset][seed] for dataset, seeds in return_corr_figs.items() for seed, value in enumerate(seeds) if value],
                       savepath=f'eval/{plot_dir}/return_correlation/total_return_correlation.pdf',
                       labels=[method2label[m] for m in parameter.plot_methods],
                       how='col',)

