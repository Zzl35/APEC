# README for APEC on DMControl tasks

## Environment 
```bash
conda env create -f conda_env.yml
conda activate apec_dmc
```

## Project Structure
```
apec_dmc/
├── src/
│   ├── train_scripts/     # Training related scripts
│   │   ├── train.py       # Main training script
│   │   ├── train_reward.py # Reward model training
│   │   └── eval_reward.py  # Reward model evaluation
│   ├── data_scripts/      # Data processing scripts
│   │   ├── get_suboptimal_dataset.py    # Get suboptimal expert data
│   │   ├── collect_dataset.py           # Collect trajectory datasets
│   │   └── make_pref_dataset.py         # Create preference datasets
│   ├── cfgs/              # Configuration files
│   └── run_pipeline.sh    # Complete pipeline script
├── conda_env.yml          # Conda environment specification
└── README.md
```

## APEC Workflow

### Prerequisites
1. First, update the `root_dir` variable in `src/cfgs/config.yaml` to match your local path.

### Step-by-step Instructions

#### 1. Train Optimal Expert
```bash
cd src
python train_scripts/train.py suite/dmc_task=[task] buffer_type=optimal seed=[seed] num_demos=[num_demos]
```
Example: `python train_scripts/train.py suite/dmc_task=walker_run buffer_type=optimal seed=0 num_demos=1`

#### 2. Get Suboptimal Expert Data
```bash
python data_scripts/get_suboptimal_dataset.py suite/dmc_task=[task] 
```
This generates suboptimal expert demonstrations based on the trained optimal expert.

#### 3. Train Suboptimal Expert (Optional)
```bash
python train_scripts/train.py suite/dmc_task=[task] buffer_type=suboptimal seed=[seed] num_demos=[num_demos]
```

#### 4. Collect Trajectory Datasets
```bash
python data_scripts/collect_dataset.py suite/dmc_task=[task] buffer_type=[optimal/suboptimal] seed=[seed]
```
This generates `buffer_[type].pkl` files containing trajectory data.

#### 5. Create Preference Datasets
```bash
python data_scripts/make_pref_dataset.py suite/dmc_task=[task] buffer_type=[type] seed=[seed]
```
This creates training and testing preference datasets for reward learning.

#### 6. Train Reward Model
```bash
python train_scripts/train_reward.py suite/dmc_task=[task] buffer_type=[type] seed=[seed]
```

#### 7. Evaluate Reward Model
```bash
python train_scripts/eval_reward.py suite/dmc_task=[task] buffer_type=[type] seed=[seed]
```

### Available DMC Tasks
Common tasks include:
- `walker_run`
- `walker_walk` 
- `cheetah_run`

### Notes
- All scripts should be run from the `src/` directory
- Adjust `CUDA_VISIBLE_DEVICES` according to your GPU setup
- Check configuration files in `cfgs/` for task-specific settings
- Results and logs will be saved in the `exp_local/` directory 






