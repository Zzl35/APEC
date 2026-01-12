import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


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


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()
        # Keep track of evaluation time so that total time only includes train time
        self._eval_start_time = 0
        self._eval_time = 0
        self._eval_flag = False

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time - self._eval_time
        return elapsed_time, total_time

    def eval(self):
        if not self._eval_flag:
            self._eval_flag = True
            self._eval_start_time = time.time()
        else:
            self._eval_time += time.time() - self._eval_start_time
            self._eval_flag = False
            self._eval_start_time = 0

    def total_time(self):
        return time.time() - self._start_time - self._eval_time


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)



def difference_aug(x):
    """
    计算输入帧之间的差异，并将差异作为新的通道加入到输入中。
    输入 x 应该是一个形状为 (batch_size, num_frames, channels, height, width) 的五维张量。
    """
    n, c, h, w = x.size()
    f = c // 3
    x = x.view(n, f, 3, h, w)
    
    # 计算帧之间的差分
    diffs = []
    for i in range(1, f):
        diff = x[:, i, :, :, :] - x[:, i-1, :, :, :]
        diffs.append(diff)
    
    # 将差分作为新的通道拼接到原始图像上
    diffs = torch.stack(diffs, dim=1)  # 形状: (n, f-1, c, h, w)
    diffs = diffs.view(n, -1, h, w)
    
    return diffs


class MotionBlurAug(nn.Module):
    def __init__(self, kernel_size=5, max_angle=30, max_blur=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_angle = max_angle
        self.max_blur = max_blur

    def forward(self, x):
        """
        对输入图像应用运动模糊。
        输入 x 应该是形状为 (batch_size, channels, height, width) 的四维张量。
        """
        n, c, h, w = x.size()
        
        # 随机选择模糊方向和强度
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        blur_strength = np.random.uniform(0, self.max_blur)
        
        # 创建运动模糊核
        kernel = self.create_motion_blur_kernel(angle, blur_strength)
        kernel = torch.tensor(kernel, device=x.device).float()

        # 扩展kernel到 (c, c, k, k) 的形状，以便可以应用于多个通道
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
        kernel = kernel.expand(c, c, self.kernel_size, self.kernel_size)  # (c, c, k, k)
        
        # 对输入图像应用卷积
        x_blurred = F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=c)

        return x_blurred

    def create_motion_blur_kernel(self, angle, blur_strength):
        """
        创建运动模糊的卷积核。
        :param angle: 模糊的方向角度
        :param blur_strength: 模糊强度
        :return: 模糊卷积核
        """
        size = self.kernel_size
        kernel = np.zeros((size, size), dtype=np.float32)
        
        # 计算模糊方向的增量
        angle_rad = np.deg2rad(angle)
        x_offset = np.cos(angle_rad) * blur_strength
        y_offset = np.sin(angle_rad) * blur_strength

        for i in range(size):
            for j in range(size):
                dx = j - size // 2
                dy = i - size // 2
                dist = np.abs(dx * x_offset + dy * y_offset)
                kernel[i, j] = max(0, 1 - dist)

        kernel = kernel / kernel.sum()  # 归一化，使得模糊核的总和为 1
        return kernel
    

class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var,
            batch_count)

    @property
    def std(self):
        return torch.sqrt(self.var)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var,
                                       batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
