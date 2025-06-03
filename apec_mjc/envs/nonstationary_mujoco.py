# Fixed Horizon wrapper of mujoco environments
import gym
import numpy as np

# RAND_PARAMS = ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction', 'gravity', 'density',
#                    'wind', 'geom_friction_1_dim', 'dof_damping_1_dim']
# RAND_PARAMS_EXTENDED = RAND_PARAMS + ['geom_size']

class NonstationaryMujoco(gym.Env):
    def __init__(self, env_name, T=1000, r=None, obs_mean=None, obs_std=None, seed=1, state_only_reward=True,
                 rand_params=['None'], log_scale_limit=1.5, n_tasks=20, action_dependent=False):
        self.env = gym.make(env_name)
        self.T = T
        self.r = r
        assert (obs_mean is None and obs_std is None) or (obs_mean is not None and obs_std is not None)
        self.obs_mean, self.obs_std = obs_mean, obs_std
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state_only_reward = state_only_reward
        self.seed(seed)
        
        self.rand_params = rand_params
        self.log_scale_limit = log_scale_limit
        self.n_tasks = n_tasks
        self.init_parameter()
        self.min_param, self.max_param = self.get_minmax_parameter(log_scale_limit)
        
        self.tasks = self.sample_tasks(n_tasks)
        self.cur_ind = None
        
    def seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def reset(self):
        self.t = 0
        self.terminated = False
        self.terminal_state = None

        self.cur_ind = np.random.randint(0, 20)
        self.set_task(self.tasks[self.cur_ind])
        
        self.obs = self.env.reset()
        self.obs = self.normalize_obs(self.obs)
        return self.obs.copy()
    
    def step(self, action):
        self.t += 1

        if self.terminated:
            return self.terminal_state, 0, self.t == self.T, True
        else:
            prev_obs = self.obs.copy()
            self.obs, r, done, info = self.env.step(action)
            self.obs = self.normalize_obs(self.obs)
            
            if self.r is not None:  # from irl model
                if self.state_only_reward:
                    r = self.r(prev_obs)
                else:
                    sa = np.concatenate((prev_obs, action), axis=0)
                    r = self.r(sa)
            
            if done:
                self.terminated = True
                self.terminal_state = self.obs
            
            return self.obs.copy(), r, done, done
    
    def normalize_obs(self, obs):
        if self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs

    def get_minmax_parameter(self, log_scale_limit):
        min_param = {}
        max_param = {}
        bound = lambda x, y: np.array(1.5) ** (np.ones(shape=x) * ((-1 if y == 'low' else 1) * log_scale_limit))
        if 'body_mass' in self.rand_params:
            min_multiplyers = bound(self.env.model.body_mass.shape, 'low')
            max_multiplyers = bound(self.env.model.body_mass.shape, 'high')
            min_param['body_mass'] = self.init_params['body_mass'] * min_multiplyers
            max_param['body_mass'] = self.init_params['body_mass'] * max_multiplyers

        # body_inertia
        if 'body_inertia' in self.rand_params:
            min_multiplyers = bound(self.env.model.body_inertia.shape, 'low')
            max_multiplyers = bound(self.env.model.body_inertia.shape, 'high')
            min_param['body_inertia'] = self.init_params['body_inertia'] * min_multiplyers
            max_param['body_inertia'] = self.init_params['body_inertia'] * max_multiplyers

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            min_multiplyers = bound(self.env.model.dof_damping.shape, 'low')
            max_multiplyers = bound(self.env.model.dof_damping.shape, 'high')
            min_param['dof_damping'] = self.init_params['dof_damping'] * min_multiplyers
            max_param['dof_damping'] = self.init_params['dof_damping'] * max_multiplyers

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            min_multiplyers = bound(self.env.model.geom_friction.shape, 'low')
            max_multiplyers = bound(self.env.model.geom_friction.shape, 'high')
            min_param['geom_friction'] = self.init_params['geom_friction'] * min_multiplyers
            max_param['geom_friction'] = self.init_params['geom_friction'] * max_multiplyers

        if 'geom_friction_1_dim' in self.rand_params:
            min_multiplyers = bound((1,), 'low')
            max_multiplyers = bound((1,), 'high')
            min_param['geom_friction_1_dim'] = np.array([min_multiplyers])
            max_param['geom_friction_1_dim'] = np.array([max_multiplyers])

        if 'dof_damping_1_dim' in self.rand_params:
            min_multiplyers = bound((1,), 'low')
            max_multiplyers = bound((1,), 'high')
            min_param['dof_damping_1_dim'] = np.array([min_multiplyers])
            max_param['dof_damping_1_dim'] = np.array([max_multiplyers])

        if 'gravity' in self.rand_params:
            min_multiplyers = bound(self.env.model.opt.gravity.shape, 'low')
            max_multiplyers = bound(self.env.model.opt.gravity.shape, 'high')
            min_param['gravity'] = self.init_params['gravity'] * min_multiplyers
            max_param['gravity'] = self.init_params['gravity'] * max_multiplyers

            if 'gravity_angle' in self.rand_params:
                min_param['gravity'][:2] = min_param['gravity'][2]
                max_param['gravity'][:2] = max_param['gravity'][2]

        if 'wind' in self.rand_params:
            min_param['wind'] = np.array([-log_scale_limit, -log_scale_limit])
            max_param['wind'] = np.array([log_scale_limit, log_scale_limit])

        if 'density' in self.rand_params:
            min_multiplyers = bound((1,), 'low')
            max_multiplyers = bound((1,), 'high')
            min_param['density'] = self.init_params['density'] * min_multiplyers
            max_param['density'] = self.init_params['density'] * max_multiplyers

        for key in min_param:
            min_it = min_param[key]
            max_it = max_param[key]
            min_real = np.min([min_it, max_it], 0)
            max_real = np.max([max_it, min_it], 0)
            min_param[key] = min_real
            max_param[key] = max_real
        return min_param, max_param
    
    
    def set_task(self, task):
        for param, param_val in task.items():
            if param == 'gravity_angle':
                continue
            if param == 'gravity':
                param_variable = getattr(self.env.model.opt, param)
            elif param == 'density':
                self.env.model.opt.density = float(param_val[0])
                continue
            elif param == 'wind':
                param_variable = getattr(self.env.model.opt, param)
                param_variable[:2] = param_val
                continue
            elif param == 'geom_friction_1_dim':
                param_variable = getattr(self.env.model, 'geom_friction')
                param_variable[:] = self.init_params[param][:] * param_val
                continue
            elif param == 'dof_damping_1_dim':
                param_variable = getattr(self.env.model, 'dof_damping')
                param_variable[:] = self.init_params[param][:] * param_val
                continue
            else:
                param_variable = getattr(self.env.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            param_variable[:] = param_val
    
    def sample_tasks(self, n_tasks, dig_range=None, linspace=False):
        """
        Generates randomized parameter sets for the mujoco env
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        current_task_count_ = 0
        param_sets = []
        if dig_range is None:
            if linspace:
                def uniform_function(low_, up_, size):
                    res = [0] * np.prod(size)
                    interval = (up_ - low_) / (n_tasks - 1)
                    for i in range(len(res)):
                        res[i] = interval * current_task_count_ + low_
                    res = np.array(res).reshape(size)
                    return res
                uniform = uniform_function
            else:
                uniform = lambda low_,up_,size: np.random.uniform(low_, up_, size=size)
        else:
            dig_range = np.abs(dig_range)
            def uniform_function(low_, up_, size):
                res = [0] * np.prod(size)
                for i in range(len(res)):
                    if linspace:
                        if current_task_count_ >= n_tasks // 2:
                            interval = (up_ - dig_range) / (n_tasks // 2)
                            # res[i] = interval * (current_task_count_ - n_tasks // 2 + 1) + dig_range
                            res[i] = interval * (current_task_count_ - n_tasks // 2 ) + dig_range
                        else:
                            interval = (-dig_range - low_) / (n_tasks // 2)
                            # res[i] = interval * (n_tasks // 2 - current_task_count_ - 1) + low_
                            res[i] = interval * (n_tasks // 2 - current_task_count_) + low_
                    else:
                        while True:
                            rand = np.random.uniform(low_, up_)
                            if rand > dig_range or rand < -dig_range:
                                res[i] = rand
                                break
                res = np.array(res).reshape(size)
                return res
            uniform = uniform_function
        bound = lambda x: np.array(1.5) ** uniform(-self.log_scale_limit, self.log_scale_limit,  x)
        bound_uniform = lambda x: uniform(-self.log_scale_limit, self.log_scale_limit,  x)
        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts
            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = bound(self.env.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = bound(self.env.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = bound(self.env.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = bound(self.env.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            if 'geom_friction_1_dim' in self.rand_params:
                geom_friction_1_dim_multipliers = bound((1,))
                new_params['geom_friction_1_dim'] = geom_friction_1_dim_multipliers

            if 'dof_damping_1_dim' in self.rand_params:
                dof_damping_1_dim_multipliers = bound((1,))
                new_params['dof_damping_1_dim'] = dof_damping_1_dim_multipliers

            if 'gravity' in self.rand_params:
                gravity_mutipliers = bound(self.env.model.opt.gravity.shape)
                new_params['gravity'] = np.multiply(self.init_params['gravity'], gravity_mutipliers)

                if 'gravity_angle' in self.rand_params:
                    min_angle = - self.log_scale_limit * np.array([1, 1]) / 8
                    max_angle = self.log_scale_limit * np.array([1, 1]) / 8
                    angle = np.random.uniform(min_angle, max_angle)
                    new_params['gravity'][0] = new_params['gravity'][2] * np.sin(angle[0]) * np.sin(angle[1])
                    new_params['gravity'][1] = new_params['gravity'][2] * np.sin(angle[0]) * np.cos(angle[1])
                    new_params['gravity'][2] *= np.cos(angle[0])

            if 'wind' in self.rand_params:
                new_params['wind'] = bound_uniform((2, ))

            if 'density' in self.rand_params:
                density_mutipliers = bound((1,))
                new_params['density'] = np.multiply(self.init_params['density'], density_mutipliers)
            param_sets.append(new_params)
            current_task_count_ += 1
        return param_sets

    def init_parameter(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.env.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.env.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = np.array(self.env.model.dof_damping).copy()

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = np.array(self.env.model.geom_friction).copy()

        if 'geom_friction_1_dim' in self.rand_params:
            self.init_params['geom_friction_1_dim'] = np.array(self.env.model.geom_friction).copy()

        if 'dof_damping_1_dim' in self.rand_params:
            self.init_params['dof_damping_1_dim'] = np.array(self.env.model.dof_damping).copy()

        if 'gravity' in self.rand_params:
            self.init_params['gravity'] = self.env.model.opt.gravity

        if 'wind' in self.rand_params:
            self.init_params['wind'] = self.env.model.opt.wind[:2]

        if 'density' in self.rand_params:
            self.init_params['density'] = np.array([self.env.model.opt.density])