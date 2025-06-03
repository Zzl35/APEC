import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import itertools

EPS=1e-6

def gradient_penalty(net, real_data, generated_data):
    batch_size = real_data.size()[0]

    # Calculate interpolationsubsampling_rate=20
    alpha = torch.rand(batch_size, 1).requires_grad_()
    alpha = alpha.expand_as(real_data).to(real_data)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data

    # Calculate probability of interpolated examples
    prob_interpolated = net(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(real_data),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Return gradient penalty
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def maxentirl_sa_loss(div: str, agent_samples, expert_samples, reward_func, device, max_T=None, gp_coef=10):
    ''' NOTE: only for maxentirl (FKL in trajectory): E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['maxentirl_sa']
    sA, aA = agent_samples
    saA = np.concatenate((sA, aA), axis=-1)
    if not max_T:
        T = 1000
    else:
        T = max_T
    _, d = saA.shape

    saE, maskE, expert_trajs_weight = expert_samples
    maskE = torch.FloatTensor(maskE).to(device)

    sA_vec = torch.FloatTensor(saA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(saE).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).squeeze(dim=-1) # E_p[r(tau)]
    t2 = torch.sum(t2 * maskE, dim=-1) / torch.sum(maskE, dim=-1)

    if expert_trajs_weight is not None:
        ### normalize weight
        expert_trajs_weight = expert_trajs_weight / torch.sum(expert_trajs_weight) * expert_trajs_weight.shape[0]
        ###
        t2 = t2 * expert_trajs_weight

    # surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    # gp = gradient_penalty(reward_func, sE_vec, sA_vec) * gp_coef
    surrogate_objective = t1.mean() - t2.mean()
    
    return T * surrogate_objective # same scale

def maxentirl_loss(div: str, agent_samples, expert_samples, reward_func, device, max_T=None):
    ''' NOTE: only for maxentirl (FKL in trajectory): E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['maxentirl']
    sA, aA = agent_samples
    # saA = np.concatenate((sA, aA), axis=-1)
    if not max_T:
        T = 1000
    else:
        T = max_T
    _, d = sA.shape

    saE, maskE, expert_trajs_weight = expert_samples
    sE = saE[..., :d]
    maskE = torch.FloatTensor(maskE).to(device)

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(sE).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).squeeze(dim=-1) # E_p[r(tau)]
    t2 = torch.sum(t2 * maskE, dim=-1) / torch.sum(maskE, dim=-1)

    if expert_trajs_weight is not None:
        ### normalize weight
        expert_trajs_weight = expert_trajs_weight / torch.sum(expert_trajs_weight) * expert_trajs_weight.shape[0]
        ###
        t2 = t2 * expert_trajs_weight

    surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    return T * surrogate_objective # same scale

def airl_loss(div: str, agent_samples, expert_samples, reward_func, device, max_T=None, policy=None):
    '''
    Implements the loss function with importance sampling adjustment as shown in Equation 6.
    '''
    assert div in ['maxentirl']
    sA, aA, maskA, log_pi_mu = agent_samples
    if not max_T:
        T = 1000
    else:
        T = max_T
    _, d = sA.shape

    saE, maskE, expert_trajs_weight = expert_samples
    sE, aE = saE[..., :d], saE[..., d:]
    maskE = torch.FloatTensor(maskE).to(device)

    sA_vec = sA.reshape(-1, d)
    aA_vec = aA.reshape(-1, aA.shape[-1])
    sE_vec = torch.tensor(sE, dtype=torch.float32, device=device)
    aE_vec = torch.tensor(aE, dtype=torch.float32, device=device)

    # Compute D(s,a) for both agent and expert samples
    D_agent = torch.sigmoid(reward_func.r(sA_vec)).squeeze(dim=-1)
    D_expert = torch.sigmoid(reward_func.r(sE_vec)).squeeze(dim=-1)
    
    # Compute importance weights
    with torch.no_grad():
        log_pi = policy.log_prob(sA_vec, aA_vec)
    
    importance_weights = log_pi / log_pi_mu # noisy airl
    # importance_weights = 1 # airl original

    # Compute loss terms
    expert_lprob = torch.log(D_expert)
    # Using logsumexp for numerical stability on expert loss

    agent_lprob = torch.log(1 - D_agent).reshape(1, -1)

    # ea_lprob = torch.logsumexp(torch.concat([expert_lprob, agent_lprob], dim=-1).squeeze(), dim=0) # error :padding 
    # Using logsumexp for numerical stability on agent loss
    expert_loss = torch.sum(expert_lprob * maskE, dim=-1) / torch.sum(maskE, dim=-1) # E(expert)
    agent_loss = torch.sum(importance_weights * agent_lprob * maskA, dim=-1)  / torch.sum(maskA, dim=-1) # pi_a / pi_mu * E(agent)

    # Combine losses
    surrogate_objective = - agent_loss - expert_loss

    return T * surrogate_objective


def gail_so_loss(div: str, agent_samples, expert_samples, reward_func, device, max_T=None):
    ''' NOTE: only for maxentirl (FKL in trajectory): E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['gail_so']
    sA, aA = agent_samples
    # saA = np.concatenate((sA, aA), axis=-1)
    if not max_T:
        T = 1000
    else:
        T = max_T
    _, d = sA.shape

    saE, maskE, expert_trajs_weight = expert_samples
    sE = saE[..., :d]
    maskE = torch.FloatTensor(maskE).to(device)

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(sE).to(device)

    agent_logits = reward_func.r(sA_vec)[0]
    likelihood_agent = (-agent_logits - torch.nn.functional.softplus(-agent_logits)).mean()

    expert_logits = reward_func.r(sE_vec)[0]
    likelihood_expert = (-torch.nn.functional.softplus(-expert_logits)).squeeze(dim=-1)
    likelihood_expert = torch.sum(likelihood_expert * maskE, dim=-1) / torch.sum(maskE, dim=-1)
    if expert_trajs_weight is not None:
        ### normalize weight
        expert_trajs_weight = expert_trajs_weight / torch.sum(expert_trajs_weight) * expert_trajs_weight.shape[0]
        ###
        likelihood_expert = likelihood_expert * expert_trajs_weight

    surrogate_objective = - (likelihood_agent + likelihood_expert.mean()) # gradient ascent
    return T * surrogate_objective # same scale


def gail_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for maxentirl (FKL in trajectory): E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['gail']
    sA, aA = agent_samples
    saA = np.concatenate((sA, aA), axis=-1)
    T = 1000
    _, d = saA.shape

    saE, maskE, expert_trajs_weight = expert_samples
    maskE = torch.FloatTensor(maskE).to(device)

    sA_vec = torch.FloatTensor(saA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(saE).to(device)

    agent_logits = reward_func.r(sA_vec)[0]
    likelihood_agent = (-agent_logits - torch.nn.functional.softplus(-agent_logits)).mean()

    expert_logits = reward_func.r(sE_vec)[0]
    likelihood_expert = (-torch.nn.functional.softplus(-expert_logits)).squeeze(dim=-1)
    likelihood_expert = torch.sum(likelihood_expert * maskE, dim=-1) / torch.sum(maskE, dim=-1)
    if expert_trajs_weight is not None:
        ### normalize weight
        expert_trajs_weight = expert_trajs_weight / torch.sum(expert_trajs_weight) * expert_trajs_weight.shape[0]
        ###
        likelihood_expert = likelihood_expert * expert_trajs_weight

    surrogate_objective = - (likelihood_agent + likelihood_expert.mean()) # gradient ascent
    return T * surrogate_objective # same scale
    
def bt_loss(alg, sample, reward_func, segment_len=1000):
    states1, states2, actions1, actions2, rewards1, rewards2, masks_1, masks_2, pref_label = sample
    inputs1 = torch.cat([states1, actions1], dim=-1) if alg in ['maxentirl_sa'] else states1
    inputs2 = torch.cat([states2, actions2], dim=-1) if alg in ['maxentirl_sa'] else states2
    
    bs, seq_len, in_dim = inputs1.shape
    if seq_len > segment_len:
        start_id = np.random.randint(0, seq_len-segment_len)
        inputs1 = inputs1[:, start_id:start_id+segment_len, :]
        inputs2 = inputs2[:, start_id:start_id+segment_len, :]
        masks_1 = masks_1[:, start_id:start_id+segment_len]
        masks_2 = masks_2[:, start_id:start_id+segment_len]
        
    rewards1 = reward_func(inputs1).squeeze()
    rewards2 = reward_func(inputs2).squeeze()
    
    returns1 = torch.sum(rewards1 * masks_1, dim=-1, keepdim=True)
    returns2 = torch.sum(rewards2 * masks_2, dim=-1, keepdim=True)
    
    # Compute probabilities using Bradley-Terry model
    prob = torch.sigmoid(returns2 - returns1)
    loss = F.binary_cross_entropy(prob, torch.argmax(pref_label.squeeze(), dim=-1, keepdim=True).float())
    return loss

def baseline_loss(alg, sample, reward_func, type='baseline_drex'):
    states1, states2, actions1, actions2, rewards1, rewards2, masks_1, masks_2, pref_label = sample
    inputs1 = torch.cat([states1, actions1], dim=-1) if alg in ['maxentirl_sa'] else states1
    inputs2 = torch.cat([states2, actions2], dim=-1) if alg in ['maxentirl_sa'] else states2
    pred_rewards1 = reward_func(inputs1).squeeze()
    pred_rewards2 = reward_func(inputs2).squeeze()
    
    pred_returns1 = torch.sum(pred_rewards1 * masks_1, dim=-1, keepdim=True)
    pred_returns2 = torch.sum(pred_rewards2 * masks_2, dim=-1, keepdim=True)
    
    if type in ['baseline_drex', 'baseline_gail']:
        # Compute probabilities using Bradley-Terry model
        prob = torch.sigmoid(pred_returns2 - pred_returns1)
        loss = F.binary_cross_entropy(prob, torch.argmax(pref_label.squeeze(), dim=-1, keepdim=True).float())
        
    elif type in ['baseline_lerp']:
        # lerp loss: update on drex loss
        pref_label = pref_label.reshape(-1, 2)
        level1, level2 = pref_label[:, 0].reshape(-1, 1), pref_label[:, 1].reshape(-1, 1)
        lerp_label = torch.eye(2, dtype=torch.int).to(pref_label.device)[torch.argmin(torch.stack((level1, level2), dim=1), dim=1)]
        prob = torch.sigmoid((pred_returns2 - pred_returns1) / torch.abs(level1 - level2))
        loss = F.binary_cross_entropy(prob, torch.argmax(lerp_label.squeeze(), dim=-1, keepdim=True).float())

    elif type in ['baseline_ssrr_irl', 'baseline_ssrr']:
        # ssrr loss: regression not preferenced
        pred_returns = torch.cat([pred_returns1, pred_returns2])
        noise_levels = torch.cat((pref_label.squeeze()[:, 0], pref_label.squeeze()[:, 1]))
        sigmoid_output = reward_func.ssrr_sigmoid(noise_levels)
        loss = F.mse_loss(pred_returns.flatten(), sigmoid_output)

    return loss

def irl_bc_loss(expert_replay_buffer, actor, device):
    expert_states = torch.from_numpy(expert_replay_buffer.states).to(device).float()
    expert_actions = torch.from_numpy(expert_replay_buffer.actions).to(device).float()
    expert_rewards = torch.from_numpy(expert_replay_buffer.rewards).to(device).float()
    expert_masks = torch.from_numpy(expert_replay_buffer.masks).to(device).float()
    predict_actions = actor(expert_states.reshape(-1, expert_states.shape[-1]))
    loss = ((predict_actions - expert_actions).sum(axis=-1) * expert_masks).square().mean()
    return loss

def cail_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for maxentirl (FKL in trajectory): E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    sA, aA = agent_samples
    saA = np.concatenate((sA, aA), axis=-1)
    T = 1000
    _, d = sA.shape

    saE, maskE, expert_conf= expert_samples
    sE, aE = saE[..., :d], saE[..., d:]
    maskE = torch.FloatTensor(maskE).to(device)

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(sE).to(device)

    agent_logits = reward_func.r(sA_vec).squeeze(dim=-1)
    likelihood_agent = (-agent_logits - torch.nn.functional.softplus(-agent_logits)).mean()

    expert_logits = reward_func.r(sE_vec).squeeze(dim=-1)
    likelihood_expert = (-torch.nn.functional.softplus(-expert_logits)).squeeze(dim=-1)
    if expert_conf is not None:
        expert_weights = (expert_conf / torch.mean(expert_conf)).clamp(0, 2)
        likelihood_expert = torch.sum(expert_weights * likelihood_expert * maskE, dim=-1) / torch.sum(maskE, dim=-1)
        print(expert_weights)
    else:
        likelihood_expert = torch.sum(likelihood_expert * maskE, dim=-1) / torch.sum(maskE, dim=-1)
    surrogate_objective = - (likelihood_agent + likelihood_expert.mean()) # gradient ascent
    return T * surrogate_objective # same scale

def ranking_loss(truth: list, approx: torch.Tensor, device) -> torch.Tensor:
    """
    Calculate the total ranking loss of two list of rewards

    Parameters
    ----------
    truth: list
        ground truth rewards of trajectories
    approx: torch.Tensor
        learned rewards of trajectories

    Returns
    -------
    loss: torch.Tensor
        ranking loss
    """
    margin = 1e-5  # factor to make the loss Lipschitz-smooth

    loss_func = nn.MarginRankingLoss().to(device)
    loss = torch.Tensor([0]).to(device)

    # loop over all the combinations of the rewards
    for c in itertools.combinations(range(approx.shape[0]), 2):
        if truth[c[0]] > truth[c[1]]:
            if torch.abs(abs(approx[c[0]] - approx[c[1]])) < margin:
                loss += (1 / (4 * margin)) * (torch.abs(approx[c[0]] - approx[c[1]]) - margin) ** 2
            else:
                y = torch.Tensor([1]).to(device)
                loss += loss_func(approx[c[0]].unsqueeze(0), approx[c[1]].unsqueeze(0), y)
        else:
            if torch.abs(abs(approx[c[0]] - approx[c[1]])) < margin:
                loss += (1 / (4 * margin)) * (torch.abs(approx[c[0]] - approx[c[1]]) - margin) ** 2
            else:
                y = torch.Tensor([-1]).to(device)
                loss += loss_func(approx[c[0]].unsqueeze(0), approx[c[1]].unsqueeze(0), y)
    return loss