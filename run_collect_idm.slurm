#!/bin/bash
#SBATCH --job-name=chen-collect-idm-mw
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

[[ -f ~/.bashrc ]] && source ~/.bashrc
source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

python src/collect_idm_data.py \
  --tasks "metaworld-door-close,metaworld-drawer-open" \
  --policy-source expert \
  --goal-mode hidden \
  --num-episodes 100 \
  --seed 0 \
  --output-dir idm_data/raw