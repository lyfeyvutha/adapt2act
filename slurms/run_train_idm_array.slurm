#!/bin/bash
#SBATCH --job-name=train-idm-array
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2002
#SBATCH --gres=gpu:1
#SBATCH --time=148:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

# Cartesian product: DATASETS x SEEDS. Adjust arrays, NUM_SEEDS, and #SBATCH --array=0-$((TOTAL-1)).

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

cd "${SLURM_SUBMIT_DIR}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

IDM_PROCESSED_ROOT=/cs/data/people/cvutha/idm_data/processed
DATASETS=(mw_all_available mw_equal_50_50 mw_ratio_70_30_total200)
SEEDS=(0 1 2)
NUM_STEPS=50000
EXPERIMENT_NAME=default

NUM_DATASETS=${#DATASETS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_DATASETS * NUM_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Array index out of range: $SLURM_ARRAY_TASK_ID >= $TOTAL"
  exit 1
fi

d=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
s=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
DATASET_NAME="${DATASETS[$d]}"
TRAIN_SEED="${SEEDS[$s]}"
DATASET_DIR="${IDM_PROCESSED_ROOT}/${DATASET_NAME}"

echo "Training IDM dataset=$DATASET_NAME seed=$TRAIN_SEED dir=$DATASET_DIR"

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
