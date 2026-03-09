#!/bin/bash
#SBATCH --job-name=george-vanilla-animdiff
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --gres=gpu:1
#SBATCH --time=1148:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=runs/ours/logs/%x_%j.out
#SBATCH --error=runs/ours/logs/%x_%j.err

[[ -f ~/.bashrc ]] && source ~/.bashrc

source /home/cvutha/opt/miniconda3/etc/profile.d/conda.sh
conda activate adapt2act
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia

# Vanilla AnimateDiff
'''
/home/cvutha/opt/miniconda3/envs/adapt2act/bin/python src/visual_planning.py task="metaworld-door-close" \
  text_prompt="a robot arm closing a door" \
  seed=0 guidance_scale=7.5 \
  plan_with_probadap=False plan_with_dreambooth=False plan_with_finetuned=False
'''
# Direct Finetuning
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=8 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=True