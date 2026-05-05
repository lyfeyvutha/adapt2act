#!/bin/bash
#SBATCH --job-name=build-idm-dataset
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2201
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

# Build one processed IDM dataset from collected raw episodes.
# Edit variables below, then: sbatch run_build_idm_dataset.slurm

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

cd "${SLURM_SUBMIT_DIR}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

# --- edit ---
IDM_RAW_ROOT=/cs/data/people/cvutha/idm_data/raw
IDM_PROCESSED_ROOT=/cs/data/people/cvutha/idm_data/processed
DATASET_NAME=mw_equal_50_50
TASKS="metaworld-door-close,metaworld-drawer-open"
SOURCES=expert
BUILD_SEED=0
# BUILD_MODE: all_available | counts | ratios
BUILD_MODE=counts
COUNTS_PER_TASK="metaworld-door-close:50,metaworld-drawer-open:50"
MIX_RATIOS="metaworld-door-close:0.7,metaworld-drawer-open:0.3"
TOTAL_TRAJECTORIES=200
# ---

BASE_ARGS=(
  --input-root "$IDM_RAW_ROOT"
  --output-root "$IDM_PROCESSED_ROOT"
  --dataset-name "$DATASET_NAME"
  --tasks "$TASKS"
  --sources "$SOURCES"
  --seed "$BUILD_SEED"
  --success-filter all
  --val-ratio 0.1
)

case "$BUILD_MODE" in
  all_available)
    python src/build_idm_dataset.py "${BASE_ARGS[@]}"
    ;;
  counts)
    python src/build_idm_dataset.py "${BASE_ARGS[@]}" --counts-per-task "$COUNTS_PER_TASK"
    ;;
  ratios)
    python src/build_idm_dataset.py "${BASE_ARGS[@]}" \
      --mix-ratios "$MIX_RATIOS" \
      --total-trajectories "$TOTAL_TRAJECTORIES"
    ;;
  *)
    echo "Unknown BUILD_MODE=$BUILD_MODE (use all_available, counts, or ratios)"
    exit 1
    ;;
esac
