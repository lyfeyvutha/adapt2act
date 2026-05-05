#!/bin/bash
#SBATCH --job-name=daddy
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --reservation=cvutha_res
#SBATCH --gres=gpu:1
#SBATCH --time=5689:56:89
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

REPO_BASE=/home/cvutha/adapt2act
EVAL_OUT_DIR=/cs/data/people/cvutha/idm_eval
mkdir -p "$EVAL_OUT_DIR" "$REPO_BASE/runs/ours/logs"

python "$REPO_BASE/src/merge_idm_eval_completion.py" \
  --input-jsons "$EVAL_OUT_DIR/idm_eval_completion_equal_summary.json,$EVAL_OUT_DIR/idm_eval_completion_ratio_summary.json" \
  --input-csvs "$EVAL_OUT_DIR/idm_eval_completion_equal_results.csv,$EVAL_OUT_DIR/idm_eval_completion_ratio_results.csv" \
  --output-csv "$EVAL_OUT_DIR/idm_eval_completion_results.csv" \
  --output-json "$EVAL_OUT_DIR/idm_eval_completion_summary.json" \
  --output-md "$EVAL_OUT_DIR/milestone2_report.md" \
  --require-complete
