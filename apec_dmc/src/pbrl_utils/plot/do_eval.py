from parameter.Parameter import Parameter

from pbrl_utils import system
from pbrl_utils.plot.plot_utils import *
from pbrl_utils.plot.plot_configs import *
from pbrl_utils.plot.eval_statistics import eval_pbrl


if __name__ == '__main__':
    parameter = Parameter()
    algorithm = 'pbrl'

    # common parameters
    alg_name = parameter.alg_name
    env_name = parameter.env_name
    seed = parameter.seed
    expert_traj_nums = [int(num) for num in parameter.expert_traj_nums.split("_")]

    # system: device, threads, seed, pid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()

    save_dir = f'eval/{env_name}/{parameter.task_type}'

    methods = {}
    for method in [parameter.task_type] + parameter.plot_baselines:
        methods[method] = method
    eval_pbrl(            
        methods=methods, 
        env_name=parameter.env_name,
        parameter=parameter,
        savefile=True,
        savepath=f'{save_dir}/pbrl_statistic.csv')


