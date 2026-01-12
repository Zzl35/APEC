import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import hydra
import sys
import torch
sys.path.append('.')
from plot.plot_utils import *
import numpy as np
import matplotlib.pyplot as plt

def downsample_trajectory(trajectory, target_seq_len):
    seq_len = trajectory.shape[0]
    indices = np.round(np.linspace(0, seq_len - 1, target_seq_len)).astype(int)  # 生成下采样的帧索引
    return trajectory[indices]

def plot_trajectory_as_long_image(loader, h, w, target_seq_len, savepath=None):
    plt.figure(figsize=(w * target_seq_len / 70, h / 70))  # 控制图像尺寸
    plt.axis('off')  # 关闭坐标轴
    with tqdm(total=10000) as pbar:
        pbar.set_description('plotting trajectories')
        for b, batch in enumerate(loader):
            # states1, states2, *_ = batch
            states1, states2, _, _, rewards1, rewards2, *_ = batch
            trajectories = torch.concat([states1, states2], dim=0)
            returns = torch.concat([rewards1, rewards2], dim=0).sum(-1).flatten()

            bs, seq_len, c, h, w = trajectories.shape
            for i in range(bs):
                if returns[i] < 100 :
                    continue
                trajectory = downsample_trajectory(trajectories[i].detach().cpu().numpy(), target_seq_len)  # target_seq_len * 9 * h * w
                frames = []
                for t in range(target_seq_len):
                    frame_rgb = [trajectory[t][3*i:3*(i+1)].transpose(1,2,0) for i in range(3)][-1] # 取最后一帧
                    
                    frames.append(frame_rgb)
                concatenated_image = np.concatenate(frames, axis=1) # 最终图像形状为 (h, w * target_seq_len, 3)

                plt.imshow(concatenated_image.astype(np.uint8), aspect='auto', alpha=0.1)  # 显示拼接后的长图

            pbar.update(bs)
            # print(f"figure {b} done.")

    # 确保保存目录存在
    if savepath:
        os.makedirs(os.path.abspath(os.path.dirname(savepath)), exist_ok=True)
        plt.savefig(savepath, dpi=150)
    plt.close()

def visual_long_image(loader, n_samples, n_steps, savepath):
    used_samples = 0
    os.makedirs(savepath, exist_ok=True)  # 确保保存目录存在
    plot_trajectory_as_long_image(loader, 84, 84, n_steps, savepath=f"{savepath}/samples.png")

def reproduce(seed):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@hydra.main(config_path='../cfgs', config_name='config_eval')
def main(cfg):
    device=torch.device(cfg.device)
    savedir = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/plot_output/{cfg.task_name}/{cfg.seed}/longimages2'
    env = hydra.utils.call(cfg.suite.task_make_fn)
    
    reproduce(cfg.seed)

    # load buffer 
    train_test_reward_buffer = PreferenceBuffer(cfg.baseline_train_buffer_path, cfg.baseline_train_idx_path, 500, device=device)
    # train_test_reward_buffer = PreferenceBuffer(cfg.train_buffer_path, cfg.train_idx_path, 500, device=device)
    train_loader = torch.utils.data.DataLoader(train_test_reward_buffer, batch_size=8, shuffle=False, num_workers=3)
    # test_reward_buffer = PreferenceBuffer(cfg.test_buffer_path, cfg.test_idx_path, 500, device=device)
    # test_loader = torch.utils.data.DataLoader(test_reward_buffer, batch_size=8, shuffle=False, num_workers=6)
    visual_long_image(train_loader, n_samples=10, n_steps=10, savepath=savedir)

if __name__ == '__main__':
    main()