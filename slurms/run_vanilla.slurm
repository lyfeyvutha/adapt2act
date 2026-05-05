#!/bin/bash
#SBATCH --job-name=vanilla-animdiff
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2201
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
REPO_BASE=/home/cvutha/adapt2act
PEOPLE_BASE=/cs/data/people/cvutha
mkdir -p "$PEOPLE_BASE"
if [ ! -e "$PEOPLE_BASE/cfgs" ]; then
  ln -s "$REPO_BASE/cfgs" "$PEOPLE_BASE/cfgs"
fi
cd "$PEOPLE_BASE"

INV_CKPT="$PEOPLE_BASE/checkpoints/inv_dyn.ckpt"

PYTHON=/home/cvutha/opt/miniconda3/envs/adapt2act/bin/python
$PYTHON "$REPO_BASE/src/visual_planning.py" task="metaworld-door-close" \
  inv_ckpt_path="$INV_CKPT" \
  text_prompt="a robot arm closing a door" \
  seed=0 \
  guidance_scale=8 \
  plan_with_probadap=False \
  plan_with_dreambooth=False \
  plan_with_finetuned=True