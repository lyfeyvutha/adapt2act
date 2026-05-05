#!/bin/bash
#SBATCH --job-name=train-idm
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

# Train one inverse dynamics checkpoint on a processed dataset (train.npz / val.npz).
# Run after build_idm_dataset. Edit DATASET_DIR to the folder that contains train.npz.

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

cd "${SLURM_SUBMIT_DIR}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

# --- edit: directory must contain train.npz and val.npz ---
DATASET_DIR=/cs/data/people/cvutha/idm_data/processed/mw_equal_50_50
EXPERIMENT_NAME=default
TRAIN_SEED=0
NUM_STEPS=50000
# ---

python src/train_idm.py \
  --dataset-dir "$DATASET_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --seed "$TRAIN_SEED" \
  --num-steps "$NUM_STEPS" \
  --batch-size 256 \
  --val-batch-size 256 \
  --eval-every 1000 \
  --save-every 5000 \
  --lr 1e-4 \
  --device cuda \
  --output-root /cs/data/people/cvutha/checkpoints/idm
