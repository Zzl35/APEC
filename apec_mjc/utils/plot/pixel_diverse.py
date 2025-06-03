import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import pickle

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


if __name__ == '__main__':
    method = 'baseline_drex'
    dataset = 'AntFH-v0'
    target_seq_len = 10
    w = h = 500
    plt.figure(figsize=(w * target_seq_len / 70, h / 70))  # 控制图像尺寸
    plt.axis('off')  # 关闭坐标轴
    load_path = f'/home/ubuntu/duxinghao/imitation_pref/buffer_pixel/{dataset}/{method}/0_0_1_0/0'
    with tqdm(total=10000) as pbar:
        pbar.set_description('plotting trajectories')
        for b, idx in enumerate(os.listdir(load_path)):
            cur_dir = f'{load_path}/{idx}'
            for file_name in enumerate(cur_dir):
                with open(f'{cur_dir}/{file_name}', 'rb') as f:
                    traj = pickle.load(f)
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
