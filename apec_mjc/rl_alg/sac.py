from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import sys
import rl_alg.sac_agent as core
from utils.replay_buffer import ReplayBuffer
from f_div import irl_bc_loss

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class SAC:

    def __init__(self, env_fn, replay_buffer, obj=None, k=1, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, add_time=False,
            polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
            update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
            log_step_interval=None, reward_indices=None,
            save_freq=1, device=torch.device("cpu"), automatic_alpha_tuning=True, reinitialize=True):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act`` 
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of 
                observations as inputs, and ``q1`` and ``q2`` should accept a batch 
                of observations and a batch of actions as inputs. When called, 
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current 
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
                you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
                for the agent and the environment in each epoch.

            epochs (int): Number of epochs to run and train agent.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target 
                networks. Target networks are updated towards main networks 
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow 
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.

            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long 
                you wait between updates, the ratio of env steps to gradient steps 
                is locked to 1.

            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.

            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.

        """

        self.obj = obj
        self.env, self.test_env = env_fn(), env_fn()
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape[0]
        self.max_ep_len=max_ep_len
        self.start_steps=start_steps
        self.batch_size=batch_size
        self.gamma=gamma
        
        self.polyak=polyak
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.env.action_space.high[0]
        self.steps_per_epoch = steps_per_epoch
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_episodes = num_test_episodes
        self.epochs = epochs
        # Create actor-critic module and target networks
        # self.ac = actor_critic(self.env.observation_space, self.env.action_space, k, add_time=add_time, device=device, **ac_kwargs)
        # self.ac_targ = deepcopy(self.ac)
        self.init_actor(actor=actor_critic(self.env.observation_space, self.env.action_space, k, add_time=add_time, device=device, **ac_kwargs), lr=lr)

        # Experience buffer
        self.replay_buffer = replay_buffer

        self.device = device

        self.automatic_alpha_tuning = automatic_alpha_tuning
        if self.automatic_alpha_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        self.true_state_dim = self.env.observation_space.shape[0]

        if log_step_interval is None:
            log_step_interval = steps_per_epoch
        self.log_step_interval = log_step_interval
        self.reinitialize = reinitialize

        self.obs_indices = list(range(self.obs_dim[0]))
        self.act_indices = list(range(self.act_dim))
        self.reward_function = None
        self.reward_indices = reward_indices

        self.test_fn = self.test_agent

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self,data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2[:, :self.true_state_dim])

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self,data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o[:, :self.true_state_dim])
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        return loss_pi, logp_pi


    # Set up model saving

    def update(self,data):
        # called by adv IRL
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True 

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return np.array([loss_q.item(), loss_pi.item(), log_pi.detach().cpu().mean().item()])
    
    
    def update_with_expert(self,data,iter):
        # called by adv IRL
        
        # bc loss
        loss_e = irl_bc_loss(self.expert_buffer, self.ac.act_batch_with_grad, self.device) * self.expert_schedule[iter]
        self.e_optimizer.zero_grad()
        loss_e.backward()
        self.e_optimizer.step()

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, log_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True 

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return np.array([loss_q.item(), loss_pi.item(), log_pi.detach().cpu().mean().item()])

    def get_action(self, o, deterministic=False, get_logprob=False):
        if len(o.shape) < 2:
            o = o[None, :]
        return self.ac.act(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
                    deterministic, get_logprob=get_logprob)

    def get_action_batch(self, o, deterministic=False):
        if len(o.shape) < 2:
            o = o[None, :]
        return self.ac.act_batch(torch.as_tensor(o[:, :self.true_state_dim], dtype=torch.float32).to(self.device), 
                    deterministic)

    def init_actor(self, actor, lr):
        self.ac = actor
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
            
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Count variables (protip: try to get a feel for how different size networks behave!)
        self.var_counts = tuple(count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

    def reset(self):
        pass

    def test_agent(self):
        # NOTE: drawback, didn't use task reward!

        avg_ep_return  = 0.
        for j in range(self.num_test_episodes):
            o = self.test_env.reset()
            obs = np.zeros((self.max_ep_len, o.shape[0]))
            next_obs = np.zeros((self.max_ep_len, o.shape[0]))
            actions = np.zeros((self.max_ep_len, self.act_dim))
            for t in range(self.max_ep_len):
                # Take deterministic actions at test time?
                a = self.get_action(o, True)
                next_o, _, _, _ = self.test_env.step(a)
                obs[t] = o.copy()
                next_obs[t] = next_o.copy()
                actions[t] = a.copy()
                o = next_o
            obs = torch.FloatTensor(obs).to(self.device)[:, self.obs_indices]
            next_obs = torch.FloatTensor(next_obs).to(self.device)[:, self.obs_indices]
            actions = torch.FloatTensor(actions).to(self.device)[:, self.act_indices]
            if self.obj in ['fkl', 'rkl', 'js', 'emd', 'maxentirl']:
                avg_ep_return += self.reward_function(next_obs).sum() # (T, d) -> (T)
            elif self.obj in ['maxentirl_sa']:
                avg_ep_return += self.reward_function(torch.concat([obs, actions], dim=-1)).sum()
        return avg_ep_return/self.num_test_episodes

    def test_agent_ori_env(self, deterministic=True):
        # for expert evaluation
        if hasattr(self.test_env, 'eval'):
            self.test_env.eval()

        rets = []
        for _ in range(self.num_test_episodes):
            ret = 0
            o = self.test_env.reset()
            for t in range(self.max_ep_len):
                a = self.get_action(o, True)
                o, r, done, _ = self.test_env.step(a)
                ret += r
                if done:
                    break
            rets.append(ret)      
        return np.mean(rets)
    
    def test_agent_batch(self):
        # for vectorize goal grid
        if hasattr(self.test_env, 'eval'):
            self.test_env.eval()

        o, ep_ret = self.test_env.reset(self.num_test_episodes), np.zeros((self.num_test_episodes))
        log_pi = np.zeros(self.num_test_episodes)
        for t in range(self.max_ep_len-1):
            # Take stochastic action!
            a, log_pi_ = self.get_action_batch(o)
            o, r, _, _ = self.test_env.step(a)
            # print(t, o, r, a)
            ep_ret += r
            log_pi += log_pi_

        return ep_ret.mean(), log_pi.mean()

    def reset_train(self):
        self.o, self.ep_len = self.env.reset(), 0

    # Learns from single trajectories rather than batch
    def learn_mujoco(self, print_out=False, save_path=None, logger=None, replay_buffer_expert=None, buffer_save_path=None, learn_grid=False):
        # only called by SMM-IRL
        # Prepare for interaction with environment
        total_steps = int(self.steps_per_epoch * self.epochs)
        start_time = time.time()
        local_time = time.time()
        best_eval = -np.inf
        # o, ep_len = self.env.reset(), 0

        # print(f"+ Training SAC for IRL agent: Total steps {total_steps:d}")
        # Main loop: collect experience in env and update/log each epoch
        test_rets = []
        alphas = []
        loss_qs = []
        loss_pis = []
        log_pis = []
        test_time_steps = []

        
        for t in range(total_steps):
            
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            if self.replay_buffer.size > self.start_steps:
                a = self.get_action(self.o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)

            self.ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            # important, assume all trajecotires are synchronized.
            # HACK:
            # For expert, done = True is episode terminates early
            # done = False if episode terminates at end of time horizon
            d = False if self.ep_len==self.max_ep_len else d

            # print(r,d)
            # Store experience to replay buffer
            self.replay_buffer.store(self.o, a, r, o2, d)


            # Super critical, easy to overlook step: make sure to update 
            # most recent observation!
            self.o = o2

            # End of trajectory handling
            # implictly assume all trajectories are synchronized
            if d or self.ep_len==self.max_ep_len:
                self.replay_buffer.add_traj_begin_point()
                self.o, self.ep_len = self.env.reset(), 0

            # Update handling
            log_pi = loss_q = loss_pi = 0
            if self.reinitialize: # default True
                # NOTE: assert training expert policy
                if t >= self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        loss_q, loss_pi, log_pi = self.update(data=batch)
                        # loss_q, loss_pi, log_pi = self.update_with_expert(data=batch, iter=t)
            else:
                # NOTE: assert training agent policy
                if self.replay_buffer.size>self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        obs = batch['obs'][:, self.obs_indices]
                        actions = batch['act'][:, self.act_indices]
                        next_obs = batch['obs2'][:, self.obs_indices]
                        if self.obj in ['maxentirl']:
                            batch['rew'] = torch.FloatTensor(self.reward_function(next_obs)).to(self.device)
                        elif self.obj in ['maxentirl_sa']:
                            batch['rew'] = torch.FloatTensor(self.reward_function(torch.concat([obs, actions], dim=-1))).to(self.device)
                        elif self.obj in ['gail_so']:
                            batch['rew'] = torch.FloatTensor(self.reward_function(obs)).to(self.device)
                        elif self.obj in ['gail']:
                            batch['rew'] = torch.FloatTensor(self.reward_function(torch.concat([obs, actions], dim=-1))).to(self.device)
                        else:
                            assert False
                        loss_q, loss_pi, log_pi = self.update(data=batch)
                        # loss_q, loss_pi, log_pi = self.update_with_expert(data=batch, iter=t)

            # End of epoch handling
            if t % self.log_step_interval == 0:
                # Test the performance of the deterministic version of the agent.
                test_epret = self.test_fn()
                test_epret_reward = self.test_agent()
                
                if print_out:
                    print(f"SAC Training | Evaluation: {test_epret:.3f} Timestep: {t+1:d} Elapsed {time.time() - local_time:.0f}s")
                if save_path is not None:
                    if test_epret>best_eval:
                        best_eval=test_epret
                        torch.save(self.ac.state_dict(), save_path.replace('last.pt', 'best.pt')) # save last
                    torch.save(self.ac.state_dict(), save_path) # save last
                alphas.append(self.alpha.item() if self.automatic_alpha_tuning else self.alpha)
                test_rets.append(test_epret)
                loss_qs.append(loss_q)
                loss_pis.append(loss_pi)
                log_pis.append(log_pi)
                test_time_steps.append(t)
                local_time = time.time()

                if logger is not None:
                    res = {
                        'timestep': int(t),
                        'test_eval': test_epret,
                        'test_eval_reward': test_epret_reward,
                        'alpha': self.alpha.item() if self.automatic_alpha_tuning else self.alpha,
                        'loss_q': loss_q,
                        'loss_pi': loss_pi,
                        'log_pi': log_pi
                    }
                    logger.add_tabular_data(tb_prefix='sac', **res)
                    logger.dump_tabular()

            
            ### sample expert buffer
            # if not learn_grid:
            #     if replay_buffer_expert is not None and (t + 1) % self.steps_per_epoch==0:
            #         self.save_expert_traj(replay_buffer_expert)
            #     if replay_buffer_expert is not None and (t + 1) % (500 * self.steps_per_epoch)==0:
            #         assert buffer_save_path is not None
            #         replay_buffer_expert.save(buffer_save_path)
            # else:
            #     if replay_buffer_expert is not None and (t + 1) == total_steps:
            #         for _ in range(500):
            #             self.save_expert_traj(replay_buffer_expert)
            #         replay_buffer_expert.save(buffer_save_path)
            if replay_buffer_expert is not None and (t + 1) % self.steps_per_epoch==0:
                self.save_expert_traj(replay_buffer_expert)
            if replay_buffer_expert is not None and (t + 1) % (500 * self.steps_per_epoch)==0:
                assert buffer_save_path is not None
                replay_buffer_expert.save(buffer_save_path)
            ###

        # print(f"- SAC Training End: time {time.time() - start_time:.0f}s")
        return [test_rets, alphas, loss_qs, loss_pis, log_pis, test_time_steps]

    def save_expert_traj(self, replay_buffer_expert):
        assert not hasattr(self.test_env, 'eval')
        obs_episode = np.zeros((1, self.max_ep_len, self.obs_dim[0]))
        act_episode = np.zeros((1, self.max_ep_len, self.act_dim))
        next_obs_episode = np.zeros((1, self.max_ep_len, self.obs_dim[0]))
        reward_episode = np.zeros((1, self.max_ep_len))
        done_episode = np.ones((1, self.max_ep_len))
        mask_episode = np.zeros((1, self.max_ep_len))

        valid = True
        done = False
        obs = self.test_env.reset()
        for t in range(self.max_ep_len):
            if valid:
                act = self.get_action(obs, True)
                next_obs, reward, done, _ = self.test_env.step(act)
                obs_episode[0, t, :] = obs
                act_episode[0, t, :] = act
                next_obs_episode[0, t, :] = next_obs
                reward_episode[0, t] = reward
                done_episode[0, t] = done
                mask_episode[0, t] = 1
                obs = next_obs
            if done:
                break
        replay_buffer_expert.add(obs_episode, act_episode, next_obs_episode, reward_episode, done_episode, mask_episode)     
        return



    @property
    def networks(self):
        return [self.ac.pi, self.ac.q1, self.ac.q2]