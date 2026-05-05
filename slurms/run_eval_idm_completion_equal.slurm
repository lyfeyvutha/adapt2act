#!/bin/bash
#SBATCH --job-name=eval-idm-equal
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --reservation=cvutha_res
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

# Re-evaluate the equal-mix IDM checkpoints.
# Writes separate completion files so parallel eval jobs do not overwrite each other.

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

REPO_BASE=/home/cvutha/adapt2act
PEOPLE_BASE=/cs/data/people/cvutha
EVAL_OUT_DIR="$PEOPLE_BASE/idm_eval"
EVAL_RUN_LOG_DIR="$EVAL_OUT_DIR/completion_equal_run_logs"
mkdir -p "$PEOPLE_BASE" "$EVAL_OUT_DIR" "$EVAL_RUN_LOG_DIR" "$REPO_BASE/runs/ours/logs"

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

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("CUDA is not available on this Slurm allocation; aborting before evaluation.")
PY

CKPT_GLOB="checkpoints/idm/default/idm_mw_equal_*_best.ckpt"
EVAL_TASK=metaworld-door-close
TEXT_PROMPT="a robot arm closing a door"
EVAL_SEEDS="0,1,2"
GUIDANCE_SCALE=2.5

# Match the current comparison setting used by run_eval_idm_checkpoints.slurm.
USE_PROBADAP=1
PRIOR_STRENGTH=0.5
INVERTED=1

PROB_ARGS=( )
if [ "${USE_PROBADAP:-0}" = 1 ]; then
  PROB_ARGS=( --plan-with-probadap --prior-strength "$PRIOR_STRENGTH" )
  if [ "${INVERTED:-0}" = 1 ]; then
    PROB_ARGS+=( --inverted-probadap )
  fi
fi

OUT_CSV="$EVAL_OUT_DIR/idm_eval_completion_equal_results.csv"
OUT_JSON="$EVAL_OUT_DIR/idm_eval_completion_equal_summary.json"

echo "Saving evaluation results to:"
echo "  $OUT_CSV"
echo "  $OUT_JSON"
echo "  $EVAL_RUN_LOG_DIR"

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
