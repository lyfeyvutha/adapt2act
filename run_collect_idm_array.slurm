#!/bin/bash
#SBATCH --job-name=chen-collect-idm-array
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-5
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

# Grid definition (edit these)
SEEDS=(0 1 2)
TASKS=("metaworld-door-close" "metaworld-drawer-open")
SOURCES=("expert")   # e.g. ("expert" "random")

# Flatten grid index -> (seed, task, source)
NUM_SEEDS=${#SEEDS[@]}
NUM_TASKS=${#TASKS[@]}
NUM_SOURCES=${#SOURCES[@]}
TOTAL=$((NUM_SEEDS * NUM_TASKS * NUM_SOURCES))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Array index out of range: $SLURM_ARRAY_TASK_ID >= $TOTAL"
  exit 1
fi

idx=$SLURM_ARRAY_TASK_ID
source_idx=$(( idx / (NUM_SEEDS * NUM_TASKS) ))
rem=$(( idx % (NUM_SEEDS * NUM_TASKS) ))
task_idx=$(( rem / NUM_SEEDS ))
seed_idx=$(( rem % NUM_SEEDS ))

SOURCE=${SOURCES[$source_idx]}
TASK=${TASKS[$task_idx]}
SEED=${SEEDS[$seed_idx]}

echo "Running SOURCE=$SOURCE TASK=$TASK SEED=$SEED"

python src/collect_idm_data.py \
  --tasks "$TASK" \
  --policy-source "$SOURCE" \
  --goal-mode hidden \
  --num-episodes 100 \
  --seed "$SEED" \
  --output-dir idm_data/raw