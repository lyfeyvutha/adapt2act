#!/bin/bash
#SBATCH --job-name=final-proj
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --gres=gpu:1
#SBATCH --time=5689:56:89
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-5
#SBATCH --output=runs/ours/logs/%x_%A_%a.out
#SBATCH --error=runs/ours/logs/%x_%A_%a.err

# Evaluate milestone 3 IDMs. Cartesian product:
#   DATASETS x EMBEDDINGS
#
# Each array task evaluates all train seeds for one dataset/embedding group
# across EVAL_SEEDS. Default: 3 IDM ckpts x 3 eval seeds = 9 rollout runs.
#
# Submit after the corresponding milestone3 training jobs have produced *_best.ckpt:
#   sbatch run_eval_milestone3_idm.slurm

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

REPO_BASE=/home/cvutha/adapt2act
PEOPLE_BASE=/cs/data/people/cvutha
EVAL_OUT_DIR="$PEOPLE_BASE/idm_eval"
mkdir -p "$PEOPLE_BASE" "$EVAL_OUT_DIR" "$REPO_BASE/runs/ours/logs"

# visual_planning.py expects cfgs under cwd.
if [ ! -e "$PEOPLE_BASE/cfgs" ]; then
  ln -s "$REPO_BASE/cfgs" "$PEOPLE_BASE/cfgs"
fi

cd "$PEOPLE_BASE"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MKL_SERVICE_FORCE_INTEL=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# --- edit ---
DATASETS=(mw_equal_50_50 mw_ratio_70_30_total200)
EMBEDDINGS=(vc-1 siglip vgg-t)

EVAL_TASK=metaworld-door-close
TEXT_PROMPT="a robot arm closing a door"
EVAL_SEEDS="0,1,2"
GUIDANCE_SCALE=2.5

# Match the current baseline eval setting.
USE_PROBADAP=1
PRIOR_STRENGTH=0.5
INVERTED=1
# ---

NUM_DATASETS=${#DATASETS[@]}
NUM_EMBEDDINGS=${#EMBEDDINGS[@]}
TOTAL=$((NUM_DATASETS * NUM_EMBEDDINGS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
  echo "Array index out of range: $SLURM_ARRAY_TASK_ID >= $TOTAL"
  exit 1
fi

d=$((SLURM_ARRAY_TASK_ID / NUM_EMBEDDINGS))
e=$((SLURM_ARRAY_TASK_ID % NUM_EMBEDDINGS))
DATASET_NAME="${DATASETS[$d]}"
EMBEDDING_NAME="${EMBEDDINGS[$e]}"

CKPT_GLOB="checkpoints/idm/milestone3_${EMBEDDING_NAME}/idm_${DATASET_NAME}_${EMBEDDING_NAME}_*_best.ckpt"
EVAL_RUN_LOG_DIR="$EVAL_OUT_DIR/milestone3_${DATASET_NAME}_${EMBEDDING_NAME}_run_logs"
OUT_CSV="$EVAL_OUT_DIR/milestone3_${DATASET_NAME}_${EMBEDDING_NAME}_results.csv"
OUT_JSON="$EVAL_OUT_DIR/milestone3_${DATASET_NAME}_${EMBEDDING_NAME}_summary.json"
mkdir -p "$EVAL_RUN_LOG_DIR"

MATCH_COUNT=$(find checkpoints/idm/milestone3_"${EMBEDDING_NAME}" -maxdepth 1 -name "idm_${DATASET_NAME}_${EMBEDDING_NAME}_*_best.ckpt" 2>/dev/null | wc -l)
if [ "$MATCH_COUNT" -eq 0 ]; then
  echo "No milestone3 checkpoints matched yet: $CKPT_GLOB"
  echo "This usually means the corresponding training array tasks are still running."
  exit 0
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Evaluating milestone3 dataset=${DATASET_NAME} embedding=${EMBEDDING_NAME}"
echo "Matched checkpoints: $MATCH_COUNT"
echo "Results:"
echo "  $OUT_CSV"
echo "  $OUT_JSON"
echo "  $EVAL_RUN_LOG_DIR"

python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this Slurm allocation; aborting before evaluation.")
PY

PROB_ARGS=( )
if [ "${USE_PROBADAP:-0}" = 1 ]; then
  PROB_ARGS=( --plan-with-probadap --prior-strength "$PRIOR_STRENGTH" )
  if [ "${INVERTED:-0}" = 1 ]; then
    PROB_ARGS+=( --inverted-probadap )
  fi
fi

python "$REPO_BASE/src/eval_idm_checkpoints.py" \
  --ckpt-glob "$CKPT_GLOB" \
  --task "$EVAL_TASK" \
  --text-prompt "$TEXT_PROMPT" \
  --seeds "$EVAL_SEEDS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --planner-script "$REPO_BASE/src/visual_planning.py" \
  --output-csv "$OUT_CSV" \
  --output-json "$OUT_JSON" \
  --log-dir "$EVAL_RUN_LOG_DIR" \
  --fail-on-error \
  "${PROB_ARGS[@]}"
