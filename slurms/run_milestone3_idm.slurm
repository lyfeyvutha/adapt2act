#!/bin/bash
#SBATCH --job-name=m3-idm
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --time=148:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-23
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

# Milestone 3: extract frozen visual representations, then train IDMs on
# precomputed latent pairs. Cartesian product:
#   DATASETS x EMBEDDINGS x SEEDS
#
# Default below is:
#   2 datasets x 4 representations x 3 seeds = 24 array jobs.
#
# Submit:
#   sbatch run_milestone3_idm.slurm

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

cd "${SLURM_SUBMIT_DIR}"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

mkdir -p runs/ours/logs

# --- edit ---
IDM_PROCESSED_ROOT=/cs/data/people/cvutha/idm_data/processed
IDM_FEATURE_ROOT=/cs/data/people/cvutha/idm_data/features
IDM_CKPT_ROOT=/cs/data/people/cvutha/checkpoints/idm

DATASETS=(mw_equal_50_50 mw_ratio_70_30_total200)
EMBEDDINGS=(vc-1 dino-v3 siglip vgg-t)
SEEDS=(0 1 2)

NUM_STEPS=50000
BATCH_SIZE=256
VAL_BATCH_SIZE=256
EXPERIMENT_PREFIX=milestone3

# Set to 1 after upgrading transformers if the current env cannot load DINOv3.
SKIP_DINOV3_IF_UNSUPPORTED=1
# ---

NUM_DATASETS=${#DATASETS[@]}
NUM_EMBEDDINGS=${#EMBEDDINGS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_DATASETS * NUM_EMBEDDINGS * NUM_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Array index out of range: $SLURM_ARRAY_TASK_ID >= $TOTAL"
  exit 1
fi

idx="$SLURM_ARRAY_TASK_ID"
d=$((idx / (NUM_EMBEDDINGS * NUM_SEEDS)))
rem=$((idx % (NUM_EMBEDDINGS * NUM_SEEDS)))
e=$((rem / NUM_SEEDS))
s=$((rem % NUM_SEEDS))

DATASET_NAME="${DATASETS[$d]}"
EMBEDDING_NAME="${EMBEDDINGS[$e]}"
TRAIN_SEED="${SEEDS[$s]}"

DATASET_DIR="${IDM_PROCESSED_ROOT}/${DATASET_NAME}"
FEATURE_DIR="${IDM_FEATURE_ROOT}/${DATASET_NAME}_${EMBEDDING_NAME}"
EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_${EMBEDDING_NAME}"

if [ ! -f "${DATASET_DIR}/train.npz" ] || [ ! -f "${DATASET_DIR}/val.npz" ]; then
  echo "Missing train.npz/val.npz in ${DATASET_DIR}. Build the processed IDM dataset first."
  exit 2
fi

if [ "${EMBEDDING_NAME}" = "dino-v3" ] && [ "${SKIP_DINOV3_IF_UNSUPPORTED}" = "1" ]; then
  if ! python - <<'PY'
from transformers import AutoConfig
AutoConfig.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m", local_files_only=True, trust_remote_code=True)
PY
  then
    echo "Skipping dino-v3: this transformers install cannot load facebook/dinov3-vitb16-pretrain-lvd1689m."
    echo "Upgrade transformers, set SKIP_DINOV3_IF_UNSUPPORTED=0, then resubmit the dino-v3 jobs."
    exit 0
  fi
fi

if [ ! -f "${FEATURE_DIR}/train.npz" ] || [ ! -f "${FEATURE_DIR}/val.npz" ]; then
  echo "Extracting features dataset=${DATASET_NAME} embedding=${EMBEDDING_NAME} -> ${FEATURE_DIR}"
  python src/feature_extractor.py \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$FEATURE_DIR" \
    --embeddings "$EMBEDDING_NAME" \
    --split train,val \
    --batch-size 128 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --device cuda
else
  echo "Using existing features: ${FEATURE_DIR}"
fi

echo "Training milestone 3 IDM dataset=${DATASET_NAME} embedding=${EMBEDDING_NAME} seed=${TRAIN_SEED}"
python src/train_idm.py \
  --dataset-dir "$FEATURE_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --seed "$TRAIN_SEED" \
  --num-steps "$NUM_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --eval-every 1000 \
  --save-every 5000 \
  --lr 1e-4 \
  --num-workers "$SLURM_CPUS_PER_TASK" \
  --device cuda \
  --output-root "$IDM_CKPT_ROOT" \
  --embeddings "$EMBEDDING_NAME" \
  --input-mode features
