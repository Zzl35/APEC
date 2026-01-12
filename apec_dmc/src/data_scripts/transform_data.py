import pandas as pd
import numpy as np
import pickle
import os
import argparse


def transform_traj(file_path, save_dir):
    with open(file_path, 'rb') as f:
        trajs_ = pickle.load(f)
    for i, traj in enumerate(trajs_):
        os.makedirs(f'{save_dir}/{i}', exist_ok=True)
        for j, traj_ in enumerate(traj[1]):
            with open(f'{save_dir}/{i}/{j}.pkl', 'wb') as f:
                pickle.dump(traj_, f)
    # for traj in trajs_:
    #     np.savez()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='walker_walk')
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    load_path = f'/infinite/common/buffer/dmc/{args.task}/{args.seed}/buffer/suboptimal/buffer_suboptimal.pkl'
    save_dir = f'/infinite/common/buffer/dmc/{args.task}/{args.seed}/suboptimal'
    os.makedirs(save_dir, exist_ok=True)

    transform_traj(load_path, save_dir)
