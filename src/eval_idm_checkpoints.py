import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


RESULT_RE = re.compile(
    r"\[RESULT\]\s+avg_episode_reward=(?P<reward>-?\d+(?:\.\d+)?)\s+avg_success_rate=(?P<success>-?\d+(?:\.\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple IDM checkpoints using visual_planning.py")
    parser.add_argument("--ckpt-glob", type=str, required=True, help="Glob for IDM checkpoints")
    parser.add_argument("--task", type=str, default="metaworld-door-close")
    parser.add_argument("--text-prompt", type=str, default="a robot arm closing a door")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output-csv", type=str, default="idm_eval_results.csv")
    parser.add_argument("--output-json", type=str, default="idm_eval_summary.json")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--planner-script", type=str, default="src/visual_planning.py")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--plan-with-probadap", action="store_true")
    parser.add_argument("--prior-strength", type=float, default=0.1)
    parser.add_argument("--inverted-probadap", action="store_true")
    parser.add_argument("--extra-args", type=str, default="", help="Extra OmegaConf CLI args passed through as-is")
    return parser.parse_args()


def _parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _run_single(args: argparse.Namespace, ckpt_path: Path, seed: int) -> Dict:
    cmd = [
        args.python_bin,
        args.planner_script,
        f"task={args.task}",
        f"text_prompt={args.text_prompt}",
        f"seed={seed}",
        f"inv_ckpt_path={ckpt_path}",
        f"guidance_scale={args.guidance_scale}",
        f"plan_with_probadap={str(args.plan_with_probadap)}",
        "plan_with_dreambooth=False",
        "plan_with_finetuned=False",
        f"prior_strength={args.prior_strength}",
        f"inverted_probadap={str(args.inverted_probadap)}",
        "use_wandb=False",
    ]
    if args.extra_args.strip():
        cmd.extend([x.strip() for x in args.extra_args.split(" ") if x.strip()])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    match = RESULT_RE.search(merged)

    reward = float("nan")
    success = float("nan")
    if match:
        reward = float(match.group("reward"))
        success = float(match.group("success"))

    return {
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "return_code": proc.returncode,
        "avg_episode_reward": reward,
        "avg_success_rate": success,
        "parsed_result": bool(match),
    }


def main():
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    ckpts = sorted(Path().glob(args.ckpt_glob))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched: {args.ckpt_glob}")

    rows: List[Dict] = []
    for ckpt in ckpts:
        for seed in seeds:
            row = _run_single(args, ckpt, seed)
            rows.append(row)
            print(
                f"[eval_idm_checkpoints] ckpt={ckpt.name} seed={seed} "
                f"success={row['avg_success_rate']} reward={row['avg_episode_reward']} rc={row['return_code']}"
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "seed",
                "return_code",
                "avg_episode_reward",
                "avg_success_rate",
                "parsed_result",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate metrics per checkpoint over all seeds.
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        grouped.setdefault(r["checkpoint"], {"success": [], "reward": [], "return_codes": []})
        grouped[r["checkpoint"]]["return_codes"].append(r["return_code"])
        if np.isfinite(r["avg_success_rate"]):
            grouped[r["checkpoint"]]["success"].append(float(r["avg_success_rate"]))
        if np.isfinite(r["avg_episode_reward"]):
            grouped[r["checkpoint"]]["reward"].append(float(r["avg_episode_reward"]))

    summary = {"rows": rows, "aggregate": {}}
    for ckpt, values in grouped.items():
        succ = values["success"]
        rew = values["reward"]
        summary["aggregate"][ckpt] = {
            "num_runs": len(values["return_codes"]),
            "num_parsed": len(succ),
            "all_return_codes": values["return_codes"],
            "avg_success_rate_mean": float(np.mean(succ)) if succ else float("nan"),
            "avg_success_rate_std": float(np.std(succ)) if succ else float("nan"),
            "avg_episode_reward_mean": float(np.mean(rew)) if rew else float("nan"),
            "avg_episode_reward_std": float(np.std(rew)) if rew else float("nan"),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval_idm_checkpoints] Wrote: {output_csv}")
    print(f"[eval_idm_checkpoints] Wrote: {output_json}")


if __name__ == "__main__":
    main()
