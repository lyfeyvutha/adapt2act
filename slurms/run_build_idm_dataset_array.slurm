#!/bin/bash
#SBATCH --job-name=build-idm-dataset-array
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu1907
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

# One array task = one processed dataset (different mixing strategy).
# Set --array=0-(N-1) to match the number of cases below.

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

cd "${SLURM_SUBMIT_DIR}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

IDM_RAW_ROOT=/cs/data/people/cvutha/idm_data/raw
IDM_PROCESSED_ROOT=/cs/data/people/cvutha/idm_data/processed
TASKS="metaworld-door-close,metaworld-drawer-open"
SOURCES=expert
BUILD_SEED=0

case "$SLURM_ARRAY_TASK_ID" in
  0)
    DATASET_NAME=mw_all_available
    EXTRA=( )
    ;;
  1)
    DATASET_NAME=mw_equal_50_50
    EXTRA=( --counts-per-task "metaworld-door-close:50,metaworld-drawer-open:50" )
    ;;
  2)
    DATASET_NAME=mw_ratio_70_30_total200
    EXTRA=( --mix-ratios "metaworld-door-close:0.7,metaworld-drawer-open:0.3" --total-trajectories 200 )
    ;;
  *)
    echo "Array index $SLURM_ARRAY_TASK_ID has no case; extend the case statement or shrink --array."
    exit 1
    ;;
esac

python src/build_idm_dataset.py \
  --input-root "$IDM_RAW_ROOT" \
  --output-root "$IDM_PROCESSED_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --tasks "$TASKS" \
  --sources "$SOURCES" \
  --seed "$BUILD_SEED" \
  --success-filter all \
  --val-ratio 0.1 \
  "${EXTRA[@]}"
