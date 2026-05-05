#!/bin/bash
#SBATCH --job-name=train-idm-missing
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2002
#SBATCH --gres=gpu:1
#SBATCH --time=148:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

# Finish IDM training jobs needed for the MetaWorld IDM data-mixing milestone.
# Covers datasets that exist but do not currently have a clean full checkpoint set.
#
# Cases:
#   0-1: mw_ratio_70_30_total200 seeds 0,1  (previous jobs were killed)
#   2-4: mw_equal_100_100 seeds 0,1,2
#   5-7: mw_ratio_70_30_total300 seeds 0,1,2

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

cd "${SLURM_SUBMIT_DIR}"

mkdir -p runs/ours/logs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

IDM_PROCESSED_ROOT=/cs/data/people/cvutha/idm_data/processed
IDM_CKPT_ROOT=/cs/data/people/cvutha/checkpoints/idm
EXPERIMENT_NAME=default
NUM_STEPS=50000

case "$SLURM_ARRAY_TASK_ID" in
  0)
    DATASET_NAME=mw_ratio_70_30_total200
    TRAIN_SEED=0
    ;;
  1)
    DATASET_NAME=mw_ratio_70_30_total200
    TRAIN_SEED=1
    ;;
  2)
    DATASET_NAME=mw_equal_100_100
    TRAIN_SEED=0
    ;;
  3)
    DATASET_NAME=mw_equal_100_100
    TRAIN_SEED=1
    ;;
  4)
    DATASET_NAME=mw_equal_100_100
    TRAIN_SEED=2
    ;;
  5)
    DATASET_NAME=mw_ratio_70_30_total300
    TRAIN_SEED=0
    ;;
  6)
    DATASET_NAME=mw_ratio_70_30_total300
    TRAIN_SEED=1
    ;;
  7)
    DATASET_NAME=mw_ratio_70_30_total300
    TRAIN_SEED=2
    ;;
  *)
    echo "Array index $SLURM_ARRAY_TASK_ID has no configured case."
    exit 1
    ;;
esac

DATASET_DIR="${IDM_PROCESSED_ROOT}/${DATASET_NAME}"
FINAL_CKPT="${IDM_CKPT_ROOT}/${EXPERIMENT_NAME}/idm_${DATASET_NAME}_seed${TRAIN_SEED}.ckpt"
METRICS_JSON="${IDM_CKPT_ROOT}/${EXPERIMENT_NAME}/idm_${DATASET_NAME}_seed${TRAIN_SEED}_metrics.json"
mkdir -p "${IDM_CKPT_ROOT}/${EXPERIMENT_NAME}"

if [ ! -f "${DATASET_DIR}/train.npz" ] || [ ! -f "${DATASET_DIR}/val.npz" ]; then
  echo "Missing train.npz/val.npz in ${DATASET_DIR}. Build this dataset first."
  exit 2
fi

if [ -f "$FINAL_CKPT" ] && [ -f "$METRICS_JSON" ]; then
  echo "Skipping completed run: dataset=${DATASET_NAME} seed=${TRAIN_SEED}"
  echo "Found ${FINAL_CKPT} and ${METRICS_JSON}"
  exit 0
fi

echo "Training IDM dataset=${DATASET_NAME} seed=${TRAIN_SEED} dir=${DATASET_DIR}"
echo "Saving checkpoints under ${IDM_CKPT_ROOT}/${EXPERIMENT_NAME}"
python src/train_idm.py \
  --dataset-dir "$DATASET_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --seed "$TRAIN_SEED" \
  --num-steps "$NUM_STEPS" \
  --batch-size 4 \
  --val-batch-size 4 \
  --eval-every 1000 \
  --save-every 5000 \
  --lr 1e-4 \
  --num-workers 2 \
  --device cuda \
  --output-root "$IDM_CKPT_ROOT"
