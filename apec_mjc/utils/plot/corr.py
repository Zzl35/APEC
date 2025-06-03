from tqdm import tqdm
from utils.plot.plot_utils import *
from utils.plot.plot_configs import *
from parameter.Parameter import Parameter
import gym
import envs


def get_return_corr_df(reward_model, buffer, n_samples=None, device='cpu'):
    batch_size = 64
    all_pred_returns, all_real_returns = [], []
    saved_samples = 0
    for _ in range(0, n_samples, batch_size):
        b = min(abs(n_samples - saved_samples), batch_size)
        states1, states2, actions1, actions2, real_rewards1, real_rewards2, masks_1, masks_2, pref_label = buffer.get_samples(b, device)
        inputs1 = torch.cat([states1, actions1], dim=-1) if alg_name in ['maxentirl_sa'] else states1
        inputs2 = torch.cat([states2, actions2], dim=-1) if alg_name in ['maxentirl_sa'] else states2
        pred_rewards1 = reward_model.r(inputs1).squeeze()
        pred_rewards2 = reward_model.r(inputs2).squeeze()

        pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
        pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True)
        real_returns1 = torch.sum(real_rewards1.squeeze() * masks_1, dim=-1, keepdim=True)
        real_returns2 = torch.sum(real_rewards2.squeeze() * masks_2, dim=-1, keepdim=True)
        
        masks = torch.cat([masks_1, masks_2], dim=0).detach().cpu().numpy().reshape(-1)
        real_rewards = torch.cat([real_rewards1, real_rewards2], dim=0).detach().cpu().numpy().reshape(-1)
        pred_rewards = torch.cat([pred_rewards1, pred_rewards2], dim=0).detach().cpu().numpy().reshape(-1)
        real_rewards = real_rewards[masks > 0]
        pred_rewards = pred_rewards[masks > 0]
        real_returns = torch.cat([real_returns1, real_returns2], dim=0).detach().cpu().numpy().reshape(-1)
        pred_returns = torch.cat([pred_returns1, pred_returns2], dim=0).detach().cpu().numpy().reshape(-1)

        saved_samples += pred_returns.shape[0]
        all_pred_returns.append(pred_returns)
        all_real_returns.append(real_returns)

    pred_returns = np.concatenate(all_pred_returns)
    real_returns = np.concatenate(all_real_returns)
    res_df = {'pred_return':pred_returns, 'real_return':real_returns}
    return pd.DataFrame(res_df)

if __name__ == '__main__':
    parameter = Parameter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=5, suppress=True)
    pid=os.getpid()
    # methods = ['baseline_drex', 'baseline_ssrr','baseline_irl', 'baseline_lerp', 'ablation_nonoise']
    types = ['0_0_1_0', '0_1_0_0', '0_0_10_0']
    datasets = ['Ant','HalfCheetah', 'Hopper', 'Humanoid', 'Walker2d']
    datasets = [f'{d}FH-v0' for d in datasets]

    for seed in range(3):
        for dataset in datasets:
            for expert_traj_type in types:
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

                method_name='ablation_nonoise'
                algorithm, load_buffer_name, load_model_name = set_model_hypeparams(method_name)
                load_model_dir = os.path.join('logfile', f'train_{algorithm}', load_model_name, ENV2ENV_NAME[dataset], expert_traj_type, f'{dataset}_maxentirl_use_pref_False_seed_{seed}_sac_epochs_5_sac_alpha_0.1_{expert_traj_type}_last_n_samples-DEBUG-NEW')
                load_model_path = os.path.join(load_model_dir, 'best_model', 'reward_model_last.pkl')
                model_params = load_params(os.path.join(load_model_dir, 'config', 'parameter.json'))
                reward_model = load_model(method_name, model_params, reward_indices, load_model_path)

                savepath = f'/home/ubuntu/duxinghao/imitation_pref/eval/corr_tmp/{dataset}/{seed}/{method_name}_{expert_traj_type}.csv'
                test_replay_path = os.path.join('buffer', dataset, 'maxentirl_sa', '0_0_0_1', '1', 'test_pbrl_buffer.npz')
                test_buffer = PreferenceBuffer(memory_size=500)
                test_buffer.load(test_replay_path)

                load_buffer_path = os.path.join('buffer', dataset, load_buffer_name, expert_traj_type, '0', f'train_buffer_{load_buffer_name}.npz')  
                train_buffer = PreferenceBuffer(memory_size=500)
                train_buffer.load(load_buffer_path)

                train_df = get_return_corr_df(reward_model, train_buffer, device=device, n_samples=500)
                train_df['type']='train'
                test_df = get_return_corr_df(reward_model, test_buffer, device=device, n_samples=500)
                test_df['type'] = 'test'
                df = pd.concat([train_df, test_df])
                os.makedirs(os.path.abspath(os.path.dirname(savepath)), exist_ok=True)
                df.to_csv(savepath, index=None)
                print(f'file saved at {savepath}')