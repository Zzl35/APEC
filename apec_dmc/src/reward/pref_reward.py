import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.encoder import Encoder


def soft_clamp(
    x : torch.Tensor,
    _min=None,
    _max=None,
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class PrefRewardModel(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_dim,
        feature_dim,
        hidden_sizes=(256, 256),
        hid_act='tanh',
        use_bn=False,
        residual=False,
        clamp_magnitude=2.0,
        device=torch.device('cpu'),
        use_action=False,
        **kwargs
    ):
        super().__init__()

        if hid_act == 'relu':
            hid_act_class = nn.ReLU
        elif hid_act == 'tanh':
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()
        
        self.encoder = Encoder(obs_shape=obs_shape)  # 用于 pixel embedding
        self.trunk = nn.Sequential(nn.Linear(self.encoder.repr_dim, feature_dim),
                            nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.norm = nn.LayerNorm(feature_dim)

        self.clamp_magnitude = clamp_magnitude
        self.device = device
        self.residual = residual
        
        self._min_r = nn.Parameter(torch.zeros(1))
        self._max_r = nn.Parameter(torch.ones(1))
        if use_action:
            feature_dim += action_dim[0]


        self.first_fc = nn.Linear(feature_dim, hidden_sizes[0])
        self.blocks_list = nn.ModuleList()
        self.use_action = use_action
  
        for i in range(len(hidden_sizes) - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if use_bn: 
                block.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.apply(weight_init)

    def forward(self, obs, action=None):
        x = self.encoder(obs)
        x = self.trunk(x)

        if self.use_action and action is not None:
            x = torch.cat([x, action], dim=-1)

        x = self.first_fc(x)

        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        
        output = self.last_fc(x)
        return output

    @torch.no_grad()
    def r(self, obs, action=None):
        reward = self.forward(obs, action)
        return reward
    
    def gradient_penalty(self, states1, states2, action1=None, action2=None, lambda_gp=10):
        """
        Computes the gradient penalty for the model, applied only to the main layers (after trunk).
        
        Args:
            real_obs: The real observations tensor.
            fake_obs: The fake observations tensor (generated or perturbed).
            action: The action tensor (optional).
            lambda_gp: The weight for the gradient penalty.

        Returns:
            The gradient penalty value.
        """

        # Forward pass through encoder and trunk
        with torch.no_grad():
            states1 = self.encoder(states1)
            states2 = self.encoder(states2)
            states1 = self.trunk(states1)
            states2 = self.trunk(states2)

        # Make sure we do not compute gradients with respect to the inputs
        states1.requires_grad_(True)
        states2.requires_grad_(True)

        # Concatenate action if necessary
        if self.use_action and action1 is not None:
            states1 = torch.cat([states1, action1], dim=-1)
            states2 = torch.cat([states2, action2], dim=-1)

        # Pass through first_fc
        states1 = self.first_fc(states1)
        states2 = self.first_fc(states2)

        # Interpolate between real and fake observations
        epsilon = torch.rand(states1.size(0), 1).to(self.device)
        x = epsilon * states1 + (1 - epsilon) * states2
        x.requires_grad_(True)

        # Forward pass through the remaining blocks
        for block in self.blocks_list:
            if self.residual:
                x = x + block(x)
            else:
                x = block(x)
        # Calculate the reward for the interpolated observation
        r = self.last_fc(x)

        # Compute the gradients of the reward with respect to the interpolated observation
        gradients = torch.autograd.grad(
            outputs=r,
            inputs=x,
            grad_outputs=torch.ones_like(r),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Compute the L2 norm of the gradients
        gradients = gradients.view(gradients.size(0), -1)
        grad_norm = gradients.norm(2, dim=1)  # L2 norm

        # Compute the gradient penalty
        grad_penalty = ((grad_norm - 1) ** 2).mean()

        # Multiply by lambda_gp to control the penalty strength
        return lambda_gp * grad_penalty