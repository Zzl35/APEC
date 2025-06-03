# README for APEC on MujoCo tasks
Workspace directory: APEC/apec_mjc

## Environment
```bash
conda env create -f conda_env.yml
conda activate apec_mjc
```

## APEC Workflow

### Step 1: Train IRL to obtain policy checkpoints
```bash
python train_irl.py --env_name [env_name]FH-v0 --expert_traj_nums [expert_traj_nums] --seed [seed] --alg_name maxentirl --task_type irl
```
- `[env_name]`: MujoCo task name (Ant, HalfCheetah, Hopper, Humanoid, Walker2d)
- `[expert_traj_nums]`: Expert trajectory numbers in format "L_M_H_P" (e.g., "0_0_1_0" for 1 high-quality trajectory)
- `[seed]`: Random seed for reproducibility

### Step 2: Sample trajectories from policy checkpoints and collect trajectory pairs
```bash
python sample.py --expert_traj_nums [expert_traj_nums] --env_name [env_name]FH-v0 --task_type [task_type] --seed [seed] --collect_dataset
```
- `[task_type]`: Task type for APEC, should be 'ablation_nonoise'
- Add `--collect_dataset` flag to enable trajectory collection

### Step 3: Train preference-based reward model
```bash
python train_pbrl.py --env_name [env_name]FH-v0 --expert_traj_nums [expert_traj_nums] --seed [seed] --alg_name maxentirl --auto_alpha --task_type [task_type] --segment_len [segment_len]
```
- `[segment_len]`: Segment length for preference learning (default: 1000, use -1 for full trajectory)
- `--auto_alpha`: Enable automatic temperature tuning for SAC

### Step 4: Evaluate the learned reward model
```bash
python eval_reward.py --env_name [env_name]FH-v0 --expert_traj_nums [expert_traj_nums] --seed [seed] --alg_name maxentirl --task_type [task_type]
```

## Example Usage
For HalfCheetah with 1 high-quality expert trajectory:
```bash
# Step 1: Train IRL
python train_irl.py --env_name HalfCheetahFH-v0 --expert_traj_nums 0_0_1_0 --seed 0 --alg_name maxentirl --task_type irl

# Step 2: Sample trajectories
python sample.py --expert_traj_nums 0_0_1_0 --env_name HalfCheetahFH-v0 --task_type ablation_nonoise --seed 0 --collect_dataset

# Step 3: Train reward model
python train_pbrl.py --env_name HalfCheetahFH-v0 --expert_traj_nums 0_0_1_0 --seed 0 --alg_name maxentirl --auto_alpha --task_type ablation_nonoise --segment_len 1000

# Step 4: Evaluate reward
python eval_reward.py --env_name HalfCheetahFH-v0 --expert_traj_nums 0_0_1_0 --seed 0 --alg_name maxentirl --task_type ablation_nonoise
```

## Expert Data Format
Expert trajectories should be stored in `./expert_data/{env_name}_expert.pkl` with the following structure:
- Keys: "L", "M", "H", "P" (Low, Medium, High, Perfect quality levels)
- Each level contains: "states", "actions", "rewards", "masks"
