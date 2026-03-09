import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from algorithm.tdmpc import TDMPC
from env import MetaWorldWrapper
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect IDM training trajectories in MetaWorld.")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task list, e.g. metaworld-door-close,metaworld-drawer-open")
    parser.add_argument("--num-episodes", type=int, default=100, help="Episodes to collect per task")
    parser.add_argument("--policy-source", type=str, default="expert", choices=["expert", "random", "agent_ckpt"])
    parser.add_argument("--agent-ckpt-path", type=str, default="", help="Path to TD-MPC checkpoint for agent_ckpt mode")
    parser.add_argument("--goal-mode", type=str, default="hidden", choices=["hidden", "observable"], help="MetaWorld goal visibility for collection")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=200)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="idm_data/raw")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing episode files")
    return parser.parse_args()


def _task_to_policy_class_name(task: str) -> str:
    # metaworld-door-close -> SawyerDoorCloseV2Policy
    _, task_name = task.split("-", 1)
    camel = "".join(part.capitalize() for part in task_name.split("-"))
    return f"Sawyer{camel}V2Policy"


def _load_expert_policy(task: str):
    import metaworld.policies as mw_policies

    class_name = _task_to_policy_class_name(task)
    if not hasattr(mw_policies, class_name):
        raise ValueError(f"No MetaWorld policy found for task {task}. Expected class: {class_name}")
    policy_cls = getattr(mw_policies, class_name)
    return policy_cls()


def _build_meta_env(task: str, seed: int, action_repeat: int, episode_length: int, goal_mode: str):
    _, task_name = task.split("-", 1)
    env_key = task_name.replace("_", "-")
    if goal_mode == "observable":
        env_ctor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{env_key}-v2-goal-observable"]
    else:
        env_ctor = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[f"{env_key}-v2-goal-hidden"]
    env = env_ctor(seed=seed)
    env = MetaWorldWrapper(env, action_repeat=action_repeat)
    env = TimeLimit(env, max_episode_steps=episode_length)
    return env


def _build_agent_cfg(task: str, seed: int, env) -> OmegaConf:
    cfg_root = Path(__file__).resolve().parent.parent / "cfgs"
    base = OmegaConf.load(cfg_root / "default.yaml")
    domain_cfg = OmegaConf.load(cfg_root / "tasks" / "metaworld.yaml")
    base.merge_with(domain_cfg)
    base.task = task
    base.seed = seed
    base.modality = "state"
    base.device = "cuda"
    base.obs_shape = tuple(int(x) for x in env.observation_space.shape)
    base.action_shape = tuple(int(x) for x in env.action_space.shape)
    base.action_dim = env.action_space.shape[0]
    base.task_title = base.task.replace("-", " ").title()
    return base


def _render_frame(env, render_size: int, camera_id: int) -> np.ndarray:
    return env.render(height=render_size, width=render_size, camera_id=camera_id).copy()


def _episode_output_path(base_dir: Path, episode_idx: int) -> Path:
    return base_dir / f"episode_{episode_idx:05d}.npz"


def _collect_single_task(
    task: str,
    args: argparse.Namespace,
    summary: Dict,
):
    env = _build_meta_env(task, args.seed, args.action_repeat, args.episode_length, args.goal_mode)
    expert_policy = _load_expert_policy(task) if args.policy_source == "expert" else None

    agent = None
    if args.policy_source == "agent_ckpt":
        if not args.agent_ckpt_path:
            raise ValueError("--agent-ckpt-path is required when --policy-source=agent_ckpt")
        if not torch.cuda.is_available():
            raise RuntimeError("agent_ckpt mode requires CUDA for TDMPC")
        cfg = _build_agent_cfg(task, args.seed, env)
        agent = TDMPC(cfg)
        agent.load(args.agent_ckpt_path)

    task_out = (
        Path(args.output_dir)
        / task
        / args.policy_source
        / f"seed{args.seed}"
        / "episodes"
    )
    task_out.mkdir(parents=True, exist_ok=True)

    collected = 0
    success_count = 0
    for ep in range(args.num_episodes):
        ep_fp = _episode_output_path(task_out, ep)
        if ep_fp.exists() and not args.overwrite:
            continue

        obs = env.reset()
        frames: List[np.ndarray] = [_render_frame(env, args.render_size, args.camera_id)]
        actions: List[np.ndarray] = []
        rewards: List[float] = []
        success = 0.0
        done = False
        t = 0
        while not done:
            if args.policy_source == "expert":
                action = expert_policy.get_action(obs)
            elif args.policy_source == "agent_ckpt":
                with torch.no_grad():
                    action = agent.plan(obs, eval_mode=True, step=t, t0=(t == 0)).cpu().numpy()
            else:
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)
            frames.append(_render_frame(env, args.render_size, args.camera_id))
            actions.append(np.asarray(action, dtype=np.float32))
            rewards.append(float(reward))
            success = max(success, float(info.get("success", 0.0)))
            t += 1

        np.savez_compressed(
            ep_fp,
            frames=np.asarray(frames, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            success=np.float32(success),
            task=np.asarray(task),
            policy_source=np.asarray(args.policy_source),
            seed=np.int32(args.seed),
        )
        collected += 1
        success_count += int(success > 0.0)

    env.close()
    summary["tasks"][task] = {
        "requested_episodes": args.num_episodes,
        "written_episodes": collected,
        "success_episodes": success_count,
        "success_rate_written": (success_count / collected) if collected > 0 else 0.0,
        "output_dir": str(task_out),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if not tasks:
        raise ValueError("No tasks provided in --tasks")

    summary = {
        "seed": args.seed,
        "policy_source": args.policy_source,
        "goal_mode": args.goal_mode,
        "tasks": {},
    }
    for task in tasks:
        if not task.startswith("metaworld-"):
            raise ValueError(f"Task must start with 'metaworld-': {task}")
        _collect_single_task(task, args, summary)

    summary_dir = Path(args.output_dir) / "_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_fp = summary_dir / f"collect_{args.policy_source}_seed{args.seed}.json"
    with open(summary_fp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[collect_idm_data] Wrote summary: {summary_fp}")


if __name__ == "__main__":
    main()
