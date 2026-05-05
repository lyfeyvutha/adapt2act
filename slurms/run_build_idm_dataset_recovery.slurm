#!/bin/bash
#SBATCH --job-name=build-idm-recovery
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2003
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

# Recovery build for Milestone 2 after mw_all_available was killed.
# This script only builds memory-safe dataset variants from collected data.

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

# Quick preflight so empty/mis-mounted roots fail fast in logs.
for t in metaworld-door-close metaworld-drawer-open; do
  if ! compgen -G "${IDM_RAW_ROOT}/${t}/${SOURCES}/seed*/episodes/*.npz" > /dev/null; then
    echo "No raw episodes found for task=${t} under ${IDM_RAW_ROOT}/${t}/${SOURCES}/seed*/episodes"
    exit 2
  fi
done

case "$SLURM_ARRAY_TASK_ID" in
  0)
    # Replacement for mw_all_available: equal sample but capped to avoid OOM.
    DATASET_NAME=mw_equal_100_100
    EXTRA=( --counts-per-task "metaworld-door-close:100,metaworld-drawer-open:100" )
    ;;
  1)
    # Larger mixed dataset than 50/50, still bounded for stable build memory.
    DATASET_NAME=mw_ratio_70_30_total300
    EXTRA=( --mix-ratios "metaworld-door-close:0.7,metaworld-drawer-open:0.3" --total-trajectories 300 )
    ;;
  *)
    echo "Array index $SLURM_ARRAY_TASK_ID has no case; shrink --array or add a case."
    exit 1
    ;;
esac

echo "Building DATASET_NAME=${DATASET_NAME}"
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
