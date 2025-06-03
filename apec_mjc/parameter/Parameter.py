from smart_logger.parameter.ParameterTemplate2 import ParameterTemplate
import argparse
import smart_logger

class Parameter(ParameterTemplate):
    def __init__(self, config_path=None, debug=False):
        super(Parameter, self).__init__(config_path, debug)

    def parser_init(self):
        parser = argparse.ArgumentParser(description=smart_logger.experiment_config.EXPERIMENT_TARGET)

        self.alg_name = 'maxentirl'
        parser.add_argument('--alg_name', type=str, default=self.alg_name,
                            help="algorithm name")

        self.information = "None"
        parser.add_argument('--information', '-i', type=str, default=self.information,
                            help="information")

        self.seed = 0
        parser.add_argument('--seed', type=int, default=self.seed,
                            help="seed")

        ## env
        # 'HopperFH-v0', 'HalfCheetahFH-v0', 'Walker2dFH-v0', 'AntFH-v0', 'HumanoidFH-v0'
        self.env_name = 'HalfCheetahFH-v0'
        parser.add_argument('--env_name', type=str, default=self.env_name,
                            help="environment name")
        
        self.env_T = 1000
        parser.add_argument('--env_T', type=int, default=self.env_T,
                            help="the horizon of environment")

        ## sac
        self.sac_epochs = 5
        parser.add_argument('--sac_epochs', type=int, default=self.sac_epochs,
                            help="total epoch of sac")

        self.sac_explore_episodes = 10
        parser.add_argument('--sac_explore_episodes', type=int, default=self.sac_explore_episodes,
                            help="episode of sac exploration")

        self.sac_batch_size = 256
        parser.add_argument('--sac_batch_size', type=int, default=self.sac_batch_size,
                            help="batch size of sac")

        self.sac_lr = 1e-3
        parser.add_argument('--sac_lr', type=float, default=self.sac_lr,
                            help="lr for policy and Q of sac")

        self.sac_alpha = 0.1
        parser.add_argument('--sac_alpha', type=float, default=self.sac_alpha,
                            help="alpha for sac")
        
        self.auto_alpha = False
        parser.add_argument('--auto_alpha', action='store_true',
                            help='whether to learn alpha in sac')

        self.sac_buffer_size = int(1e6)
        parser.add_argument('--sac_buffer_size', type=int, default=self.sac_buffer_size,
                            help="buffer size of sac")

        self.sac_reinitialize = False
        parser.add_argument('--sac_reinitialize', action='store_true',
                            help='whether to reinitialize in sac')

        self.sac_k = 1
        parser.add_argument('--sac_k', type=int, default=self.sac_k,
                            help="k of sac")
        
        self.sac_log_step_interval = 5000
        parser.add_argument('--sac_log_step_interval', type=int, default=self.sac_log_step_interval,
                            help="log step interval of sac")

        self.sac_update_every = 1
        parser.add_argument('--sac_update_every', type=int, default=self.sac_update_every,
                            help="how many steps between two sac updates")

        self.ac_hidden_sizes = [256, 256]
        parser.add_argument('--ac_hidden_sizes', nargs='+', type=int, default=self.ac_hidden_sizes,
                            help="architecture of the hidden layers of actor critic")

        ## test
        self.num_test_episodes = 10
        parser.add_argument('--num_test_episodes', type=int, default=self.num_test_episodes,
                            help="num of episodes for test")

        ## expert
        self.expert_samples_episode = 64
        parser.add_argument('--expert_samples_episode', type=int, default=self.expert_samples_episode,
                            help="num of episodes for sample expert")

        ## irl
        self.irl_n_iters = int(400)
        parser.add_argument('--irl_n_iters', type=int, default=self.irl_n_iters,
                            help="num iters of irl")
        
        self.training_n_trajs = 10
        parser.add_argument('--training_n_trajs', type=int, default=self.training_n_trajs,
                            help="num of training trajs in irl")
        
        self.expert_traj_nums = "0_0_10_0"
        parser.add_argument('--expert_traj_nums', type=str, default=self.expert_traj_nums,
                            help="nums of expert traj by level (L, M, H, P) in irl")
        
        self.irl_eval_episodes = 10
        parser.add_argument('--irl_eval_episodes', type=int, default=self.irl_eval_episodes,
                            help="irl eval episodes")
        
        ## irl reward
        self.r_use_bn = False
        parser.add_argument('--r_use_bn', action='store_true',
                            help='whether to use bn in reward net')

        self.r_residual = False
        parser.add_argument('--r_residual', action='store_true',
                            help='whether to use residual in reward net')

        self.r_hid_act = 'relu'
        parser.add_argument('--r_hid_act', type=str, default=self.r_hid_act,
                            help="hidden activation in reward net")

        self.r_hidden_sizes = [512, 512]
        parser.add_argument('--r_hidden_sizes', nargs='+', type=int, default=self.r_hidden_sizes,
                            help="architecture of the hidden layers of reward")

        self.r_clamp_magnitude = 10
        parser.add_argument('--r_clamp_magnitude', type=int, default=self.r_clamp_magnitude,
                            help="clamp magnitude for reward")
        
        self.r_lr = 1e-4
        parser.add_argument('--r_lr', type=float, default=self.r_lr,
                            help="lr for reward")
        
        self.r_weight_decay = 1e-3
        parser.add_argument('--r_weight_decay', type=float, default=self.r_weight_decay,
                            help="weight decay for reward")
        
        self.r_gradient_step = 1
        parser.add_argument('--r_gradient_step', type=int, default=self.r_gradient_step,
                            help="gradient step for reward")

        self.r_momentum = 0.9
        parser.add_argument('--r_momentum', type=float, default=self.r_momentum,
                            help="momentum for reward")

        ## pref reward
        self.use_pref = False
        parser.add_argument('--use_pref', action='store_true',
                            help='whether to use preference reward in IRL')

        self.pref_size_segment = 1000
        parser.add_argument('--pref_size_segment', type=int, default=self.pref_size_segment,
                            help="size segment of preference reward")

        self.expert_pref_size = 50
        parser.add_argument('--expert_pref_size', type=int, default=self.expert_pref_size, metavar='N',
                            help="control the preference size of expert")

        self.pref_beta = 0.1
        parser.add_argument('--pref_beta', type=float, default=self.pref_beta,
                            help="beta when compute expert weight")

        self.debug = False
        parser.add_argument('--debug', action='store_true',
                            help='debug mode')

        self.debug = False
        parser.add_argument('--train_disc_with_buffer', action='store_true',
                            help='train discriminator with replay buffer')

        self.expand_expert = False
        parser.add_argument('--expand_expert', action='store_true',
                            help='whether to expand expert dataset when training')

        self.expand_expert_interval = 100
        parser.add_argument('--expand_expert_interval', type=int, default=self.expand_expert_interval,
                            help="interval to expand expert dataset")
        
        self.expand_pref_once = 1
        parser.add_argument('--expand_pref_once', type=int, default=self.expand_pref_once,
                            help="preference added when expanding expert dataset")

        self.expand_expert_once = 1
        parser.add_argument('--expand_expert_once', type=int, default=self.expand_expert_once,
                            help="expert trajs added when expanding expert dataset")

        self.pref_model_step = 2
        parser.add_argument('--pref_model_step', type=int, default=self.pref_model_step,
                            help="train step for preference reward model")

        self.disc_agent_sample_mode = "last_n_samples"
        parser.add_argument('--disc_agent_sample_mode', type=str, default=self.disc_agent_sample_mode,
                            help="agent sample mode for discriminator")

        self.use_best_traj = False
        parser.add_argument('--use_best_traj', action='store_true',
                            help='whether to use the best trajectory')

        self.expand_expert_warm_up = 0
        parser.add_argument('--expand_expert_warm_up', type=int, default=self.expand_expert_warm_up,
                            help="expand expert warm up iteration")

        self.max_pref_size = int(1e6) # default inf
        parser.add_argument('--max_pref_size', type=int, default=self.max_pref_size,
                            help="max size of human preferences")

        self.teacher_eps_mistake = 0
        parser.add_argument('--teacher_eps_mistake', type=float, default=self.teacher_eps_mistake,
                            help="teacher eps mistake")

        self.teacher_beta = -1
        parser.add_argument('--teacher_beta', type=float, default=self.teacher_beta,
                            help="teacher beta")

        self.teacher_gamma = 1
        parser.add_argument('--teacher_gamma', type=float, default=self.teacher_gamma,
                            help="teacher gamma")

        self.teacher_eps_skip = 0
        parser.add_argument('--teacher_eps_skip', type=float, default=self.teacher_eps_skip,
                            help="teacher eps skip")

        self.teacher_eps_equal = 0
        parser.add_argument('--teacher_eps_equal', type=float, default=self.teacher_eps_equal,
                            help="teacher eps equal")
        
        self.pbrl_n_iters = 100
        parser.add_argument('--pbrl_n_iters', type=int, default=self.pbrl_n_iters,
                            help="num iters of pbrl")

        # epsilon sampling
        self.env_type = 'mujoco'
        parser.add_argument('--env_type', type=str, default=self.env_type)

        self.task_type = None
        parser.add_argument('--task_type', type=str, default=self.task_type,
                            help="task name for eval") 
        
        self.collect_dataset = False
        parser.add_argument('--collect_dataset', action='store_true', default=self.collect_dataset,
                            help='whether to reconstruction epsilon sampling trajectories')
        
        self.sample_schedule = 'uniform'
        parser.add_argument('--sample_schedule', type=str, default=self.sample_schedule)

        self.sample_mode = 'train'
        parser.add_argument('--sample_mode', type=str, default=self.sample_mode)       

        # plot
        self.plot_dir = None
        parser.add_argument('--plot_dir', type=str, default=self.plot_dir)

        self.plot_diversity = False
        parser.add_argument('--plot_diversity', action='store_true', default=self.plot_diversity)

        self.plot_consistency = False
        parser.add_argument('--plot_consistency', action='store_true', default=self.plot_consistency)

        self.plot_corr = False
        parser.add_argument('--plot_corr', action='store_true', default=self.plot_corr)

        self.plot_reuse = False
        parser.add_argument('--plot_reuse', action='store_true', default=self.plot_reuse)     

        self.plot_acc = False
        parser.add_argument('--plot_acc', action='store_true', default=self.plot_acc) 

        self.plot_cover = False
        parser.add_argument('--plot_cover', action='store_true', default=self.plot_cover) 
        
        self.plot_methods = []
        parser.add_argument('--plot_methods', nargs='+', default=self.plot_methods)

        self.plot_datasets = []
        parser.add_argument('--plot_datasets', nargs='+', default=self.plot_datasets)

        self.plot_seeds = []
        parser.add_argument('--plot_seeds', nargs='+', default=self.plot_seeds)

        self.segment_len = 1000
        parser.add_argument('--segment_len', type=int, default=self.segment_len)

        self.fix_seed = False
        parser.add_argument('--fix_seed', action='store_true')

        self.fix_task = False
        parser.add_argument('--fix_task', action='store_true')

        self.distance_metric = 'ot'
        parser.add_argument('--distance_metric', type=str, default=self.distance_metric)
        
        return parser


if __name__ == '__main__':
    def main():
        parameter = Parameter()
        print(parameter)
    main()