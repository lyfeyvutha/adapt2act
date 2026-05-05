#!/bin/bash
#SBATCH --job-name=eval-idm-ckpts
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --reservation=cvutha_res
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

# Compare trained IDMs on MetaWorld by running visual_planning.py per checkpoint x seed.
# Heavy job: one full planner rollout per (ckpt, seed). Adjust time partition as needed.

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

REPO_BASE=/home/cvutha/adapt2act
PEOPLE_BASE=/cs/data/people/cvutha
mkdir -p "$PEOPLE_BASE"

# Make configs available from PEOPLE_BASE so visual_planning.py can load cfgs via Path().cwd()/cfgs,
# and so its heavy rollout artifacts get written under PEOPLE_BASE/logs/...
if [ ! -e "$PEOPLE_BASE/cfgs" ]; then
  ln -s "$REPO_BASE/cfgs" "$PEOPLE_BASE/cfgs"
fi

cd "$PEOPLE_BASE"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

# --- edit ---
# Glob is relative to cwd (repo root); matches train_idm output layout.
CKPT_GLOB="checkpoints/idm/default/idm_*_best.ckpt"
EVAL_TASK=metaworld-door-close
TEXT_PROMPT="a robot arm closing a door"
EVAL_SEEDS="0,1,2"
GUIDANCE_SCALE=2.5
# Match inverse ProbAdapt slurm: set USE_PROBADAP=1 and INVERTED=1 for that setting
USE_PROBADAP=1
PRIOR_STRENGTH=0.5
INVERTED=1
# ---

PROB_ARGS=( )
if [ "${USE_PROBADAP:-0}" = 1 ]; then
  PROB_ARGS=( --plan-with-probadap --prior-strength "$PRIOR_STRENGTH" )
  if [ "${INVERTED:-0}" = 1 ]; then
    PROB_ARGS+=( --inverted-probadap )
  fi
fi

OUT_CSV="$REPO_BASE/runs/ours/idm_eval_results.csv"
OUT_JSON="$REPO_BASE/runs/ours/idm_eval_summary.json"
OUT_LOG_DIR="$REPO_BASE/runs/ours/idm_eval_run_logs"
mkdir -p "$OUT_LOG_DIR"

python "$REPO_BASE/src/eval_idm_checkpoints.py" \
  --ckpt-glob "$CKPT_GLOB" \
  --task "$EVAL_TASK" \
  --text-prompt "$TEXT_PROMPT" \
  --seeds "$EVAL_SEEDS" \
  --guidance-scale "$GUIDANCE_SCALE" \
  --planner-script "$REPO_BASE/src/visual_planning.py" \
  --output-csv "$OUT_CSV" \
  --output-json "$OUT_JSON" \
  --log-dir "$OUT_LOG_DIR" \
  "${PROB_ARGS[@]}"
