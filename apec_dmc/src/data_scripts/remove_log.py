import os
import shutil

if __name__ == '__main__':
    for env_name in [ 'walker_walk']:
        for method_name in ['drex']:
            for seed in range(5):
                dirpath = f'/home/ubuntu/duxinghao/APEC/apec_dmc/src/exp_local/eval_baseline/dmc/{env_name}/{method_name}/{seed}'
                snapshots = f'{dirpath}/snapshots'
                buffer = f'{dirpath}/buffer'

                for file in [snapshots, buffer]:
                    if os.path.exists(file):
                        print(f'remove {file}')
                        shutil.rmtree(file)