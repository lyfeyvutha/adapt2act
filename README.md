# Solving New Tasks by Adapting Internet Video Knowledge
## Quick Start
### Setup Conda Environment
```sh
conda create -n adapt2act python=3.9
conda activate adapt2act

pip install pip==21.0 wheel==0.38.0 setuptools==65.5.0  # specified gym version requires these tools to be old
pip install -r requirements.txt
```

### Install Visual-Cortex
```sh
git clone https://github.com/facebookresearch/eai-vc.git
cd eai-vc

pip install -e ./vc_models
```

### Customize Configurations for DeepMind Control Environments
Please follow the same instruction in [TADPoLe](https://github.com/brown-palm/tadpole?tab=readme-ov-file#customize-configurations-for-dog-and-humanoid-environments) to customize configurations for Dog and Humanoid environments.

### Checkpoints
Please put `checkpoints/` under the `adapt2act/` folder, and the directory should have the following structure:
```
adapt2act/
└── checkpoints/
    ├── animatediff_finetuned/
    │   ├── {domain}_finetuned.ckpt
    │   └── ...
    ├── in_domain/
    │   ├── {domain}/
    │   ├── {domain}_suboptimal/
    │   └── ...
    ├── dreambooth/
    │   ├── {domain}_lora/
    │   └── ...
    └── inv_dyn.ckpt
```
We currently support three domains: Metaworld `mw`, Humanoid `humanoid` and Dog `dog`. The model checkpoints can be downloaded [here](https://drive.google.com/file/d/1bDoQq_z605cX6czWGKVRLnSgXjR050A8/view?usp=drive_link).

## Policy Supervision
> [!TIP]
> To enable wandb logging, enter your wandb entity in `cfgs/default.yaml` and add `use_wandb=True` to the commands below
### AnimateDiff
Vanilla AnimateDiff
```shell
python src/vidtadpole_train.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=False \
    use_finetuned=False
```

Direct Finetuning
```shell
python src/vidtadpole_train.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=False \
    use_finetuned=True
```

Subject Customization
```shell
python src/vidtadpole_train.py task="metaworld-door-close" \
    text_prompt="a [D] robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=True \
    use_finetuned=False
```

Probabilistic Adaptation
```shell
python src/vidtadpole_train_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.1 \
    inverted_probadap=False
```


Inverse Probabilistic Adaptation
```shell
python src/vidtadpole_train_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.1 \
    inverted_probadap=True
```

### AnimateLCM
Vanilla AnimateLCM
```shell
python src/vidtadpole_train_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=False \
    use_finetuned=False
```

Probabilistic Adaptation
```shell
python src/vidtadpole_train_lcm_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.1 \
    inverted_probadap=False
```

Inverse Probabilistic Adaptation
```shell
python src/vidtadpole_train_lcm_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.2 \
    inverted_probadap=True
```

### In-Domain-Only
```shell
python src/vidtadpole_train_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0 \
    inverted_probadap=False
```

## Visual Planning

### AnimateDiff

Vanilla AnimateDiff
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=7.5 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=False
```

Direct Finetuning
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=8 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=True
```

Subject Customization
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a [D] robot arm closing a door" \
    seed=0 \
    guidance_scale=7.5 \
    plan_with_probadap=False \
    plan_with_dreambooth=True \
    plan_with_finetuned=False
```


Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.1 \
    inverted_probadap=False
```

Inverse Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.5 \
    inverted_probadap=True
```

### AnimateLCM

Vanilla AnimateLCM
```shell
python src/visual_planning_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=False
```


Probabilistic Adaptation
```shell
python src/visual_planning_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.1 \
    inverted_probadap=False
```

Inverse Probabilistic Adaptation
```shell
python src/visual_planning_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.2 \
    inverted_probadap=True
```

### In-Domain-Only
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0 \
    inverted_probadap=False
```

## Visual Planning with Suboptimal Data

In-Domain-Only
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0 \
    inverted_probadap=False \
    use_suboptimal=True
```

Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.1 \
    inverted_probadap=False \
    use_suboptimal=True
```

Inverse Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.5 \
    inverted_probadap=True \
    use_suboptimal=True
```
## IDM Training
We provide the implementation of Inverse Dynamics Model in `src/utils.py`. For inverse dynamics training, we provide the hyperparameters in the Appendix of the paper and related code snippets below for reference.

Dataset Structure:
```python
class IDMDataset(Dataset):
    def __init__(self):
        self.frames = []
        self.actions = [] # Padding actions with an additional dummy None element to match the length of frames

    def __getitem__(self, index):
        if self.actions[index] is None:  # dummy action
            frames = torch.cat([self.frames[index - 1], self.frames[index]])
            action = self.actions[index - 1]
        else:
            frames = torch.cat([self.frames[index], self.frames[index + 1]])
            action = self.actions[index]

        return (frames, action)

    def __len__(self):
        return len(self.frames) - 1 if len(self.frames) > 0 else 0 # prevent index + 1 out of range
```

Training Loop:
```python
device = torch.device('cuda')
idm_loader = DataLoader(idm_dataset, batch_size=idm_batch_size, shuffle=True)
for step in tqdm(range(num_training_steps)):
    obs, actions = next(iter(idm_loader))
    obs = obs.contiguous()
    loss = idm_model.calculate_loss(obs.to(device), actions.to(device))
    idm_optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(idm_model.parameters(), 1.0)
    idm_optimizer.step()
```

## IDM Data Collection and Comparison Pipeline (MetaWorld)

This repository now includes a reproducible IDM pipeline:
1. collect trajectories,
2. build mixed IDM datasets,
3. train multiple IDM checkpoints,
4. evaluate checkpoints with `src/visual_planning.py`,
5. aggregate mean/std success metrics across seeds.

### 1) Collect IDM trajectories

Use `expert`, `random`, or `agent_ckpt` policy sources.

```shell
python src/collect_idm_data.py \
    --tasks metaworld-door-close,metaworld-drawer-open \
    --policy-source expert \
    --goal-mode hidden \
    --num-episodes 100 \
    --seed 0 \
    --output-dir idm_data/raw
```

Output layout:
```text
idm_data/raw/<task>/<source>/seed<seed>/episodes/episode_00000.npz
```

### 2) Build mixed datasets

Example: fixed counts per task.

```shell
python src/build_idm_dataset.py \
    --input-root idm_data/raw \
    --dataset-name mw_mix_close_drawer_50_50 \
    --tasks metaworld-door-close,metaworld-drawer-open \
    --counts-per-task metaworld-door-close:50,metaworld-drawer-open:50 \
    --success-filter all \
    --seed 0
```

Alternative: weighted ratios + total trajectories.

```shell
python src/build_idm_dataset.py \
    --input-root idm_data/raw \
    --dataset-name mw_mix_ratio \
    --tasks metaworld-door-close,metaworld-drawer-open \
    --mix-ratios metaworld-door-close:0.75,metaworld-drawer-open:0.25 \
    --total-trajectories 200 \
    --success-filter success_only \
    --seed 0
```

Outputs:
```text
idm_data/processed/<dataset_name>/train.npz
idm_data/processed/<dataset_name>/val.npz
idm_data/processed/<dataset_name>/manifest.json
```

### 3) Train IDM checkpoints

```shell
python src/train_idm.py \
    --dataset-dir idm_data/processed/mw_mix_close_drawer_50_50 \
    --experiment-name mix_ablation \
    --seed 0 \
    --num-steps 50000 \
    --batch-size 256
```

Checkpoint format is planner-compatible and includes `inv_model`:
```text
checkpoints/idm/<experiment_name>/idm_<dataset_name>_seed<seed>.ckpt
```

### 4) Evaluate multiple IDM checkpoints with visual planning

`src/visual_planning.py` now prints:
```text
[RESULT] avg_episode_reward=... avg_success_rate=...
```

Run sweep:

```shell
python src/eval_idm_checkpoints.py \
    --ckpt-glob "checkpoints/idm/mix_ablation/*.ckpt" \
    --task metaworld-door-close \
    --text-prompt "a robot arm closing a door" \
    --seeds 0,1,2 \
    --plan-with-probadap \
    --prior-strength 0.1 \
    --output-csv idm_eval/results.csv \
    --output-json idm_eval/summary.json
```

### 5) Recommended ablation grid

- trajectories per task: `25, 50, 100, 200`
- mix ratios (target:aux): `100:0`, `75:25`, `50:50`
- quality filter: `all`, `success_only`
- seeds: `0, 1, 2`

Config templates are provided in:
- `cfgs/idm/collect_default.yaml`
- `cfgs/idm/mix_default.yaml`
- `cfgs/idm/train_default.yaml`

## Citation
If you find this repository helpful, please consider citing our work:
```bibtex
@inproceedings{luo2024solving,
  title={Solving New Tasks by Adapting Internet Video Knowledge},
  author={Luo, Calvin and Zeng, Zilai and Du, Yilun and Sun, Chen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgement
This repo contains code adapted from [flowdiffusion](https://github.com/flow-diffusion/AVDC), [TDMPC](https://github.com/nicklashansen/tdmpc) and [TADPoLe](https://github.com/brown-palm/tadpole). We thank the authors and contributors for open-sourcing their code.
