import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge split IDM completion evaluations and write a Milestone 2 report.")
    parser.add_argument("--input-jsons", type=str, required=True, help="Comma-separated eval summary JSON files")
    parser.add_argument("--input-csvs", type=str, required=True, help="Comma-separated eval result CSV files")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="/cs/data/people/cvutha/idm_data")
    parser.add_argument("--ckpt-root", type=str, default="/cs/data/people/cvutha/checkpoints/idm/default")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def _split_csv(raw: str) -> List[Path]:
    return [Path(x.strip()) for x in raw.split(",") if x.strip()]


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_rows(paths: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out = dict(row)
                out["seed"] = int(out["seed"])
                out["return_code"] = int(out["return_code"])
                out["avg_episode_reward"] = float(out["avg_episode_reward"])
                out["avg_success_rate"] = float(out["avg_success_rate"])
                out["parsed_result"] = str(out["parsed_result"]).lower() == "true"
                rows.append(out)
    return rows


def _aggregate(rows: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        grouped.setdefault(row["checkpoint"], {"success": [], "reward": [], "return_codes": []})
        grouped[row["checkpoint"]]["return_codes"].append(row["return_code"])
        success = row["avg_success_rate"]
        reward = row["avg_episode_reward"]
        if math.isfinite(success):
            grouped[row["checkpoint"]]["success"].append(success)
        if math.isfinite(reward):
            grouped[row["checkpoint"]]["reward"].append(reward)

    out = {}
    for ckpt, values in grouped.items():
        succ = values["success"]
        rew = values["reward"]
        out[ckpt] = {
            "num_runs": len(values["return_codes"]),
            "num_parsed": len(succ),
            "all_return_codes": values["return_codes"],
            "avg_success_rate_mean": float(np.mean(succ)) if succ else float("nan"),
            "avg_success_rate_std": float(np.std(succ)) if succ else float("nan"),
            "avg_episode_reward_mean": float(np.mean(rew)) if rew else float("nan"),
            "avg_episode_reward_std": float(np.std(rew)) if rew else float("nan"),
        }
    return out


def _dataset_from_ckpt(checkpoint: str) -> str:
    name = Path(checkpoint).name
    match = re.match(r"idm_(?P<dataset>.+)_seed\d+_best\.ckpt$", name)
    return match.group("dataset") if match else name


def _training_seed_from_ckpt(checkpoint: str) -> str:
    name = Path(checkpoint).name
    match = re.search(r"_seed(?P<seed>\d+)_best\.ckpt$", name)
    return match.group("seed") if match else "?"


def _load_dataset_manifests(data_root: Path) -> Dict[str, Dict]:
    manifests = {}
    for manifest in sorted((data_root / "processed").glob("*/manifest.json")):
        manifests[manifest.parent.name] = _load_json(manifest)
    return manifests


def _load_train_metrics(ckpt_root: Path) -> Dict[str, Dict]:
    metrics = {}
    for fp in sorted(ckpt_root.glob("idm_*_metrics.json")):
        metrics[fp.stem.removesuffix("_metrics")] = _load_json(fp)
    return metrics


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint",
        "seed",
        "return_code",
        "avg_episode_reward",
        "avg_success_rate",
        "parsed_result",
        "stdout_log",
        "stderr_log",
        "command_log",
        "stdout_tail",
        "stderr_tail",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict], aggregate: Dict, manifests: Dict, metrics: Dict) -> None:
    complete = all(r["return_code"] == 0 and r["parsed_result"] for r in rows)
    lines = [
        "# Milestone 2 IDM Data-Mixing Evaluation",
        "",
        f"Status: {'complete' if complete else 'incomplete'}",
        f"Rollout runs: {sum(1 for r in rows if r['return_code'] == 0 and r['parsed_result'])}/{len(rows)} parsed successfully",
        "",
        "## Processed Datasets",
        "",
        "| dataset | selected episodes | train pairs | val pairs | mixing |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, manifest in sorted(manifests.items()):
        if dataset == "mw_all_available":
            continue
        selected = manifest.get("selected_episodes_per_task", {})
        mix = manifest.get("mix_ratios") or manifest.get("counts_per_task", {})
        lines.append(
            f"| {dataset} | {sum(selected.values())} | {manifest.get('num_train_pairs', 0)} | "
            f"{manifest.get('num_val_pairs', 0)} | `{json.dumps(mix, sort_keys=True)}` |"
        )

    lines.extend([
        "",
        "## MetaWorld Door-Close Evaluation",
        "",
        "| dataset | train seed | parsed | success mean | reward mean | best val loss |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for ckpt, stats in sorted(aggregate.items(), key=lambda item: (_dataset_from_ckpt(item[0]), _training_seed_from_ckpt(item[0]))):
        dataset = _dataset_from_ckpt(ckpt)
        train_seed = _training_seed_from_ckpt(ckpt)
        key = f"idm_{dataset}_seed{train_seed}"
        best_val = metrics.get(key, {}).get("best_val_loss", float("nan"))
        lines.append(
            f"| {dataset} | {train_seed} | {stats['num_parsed']}/{stats['num_runs']} | "
            f"{stats['avg_success_rate_mean']:.6f} | {stats['avg_episode_reward_mean']:.6f} | {best_val:.8f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_jsons = _split_csv(args.input_jsons)
    input_csvs = _split_csv(args.input_csvs)
    for path in input_jsons + input_csvs:
        if not path.exists():
            raise FileNotFoundError(path)

    rows = _load_csv_rows(input_csvs)
    rows.sort(key=lambda r: (r["checkpoint"], r["seed"]))
    aggregate = _aggregate(rows)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "aggregate": aggregate, "inputs": [str(p) for p in input_jsons]}, f, indent=2)

    _write_csv(Path(args.output_csv), rows)
    _write_markdown(
        Path(args.output_md),
        rows,
        aggregate,
        _load_dataset_manifests(Path(args.data_root)),
        _load_train_metrics(Path(args.ckpt_root)),
    )

    print(f"[merge_idm_eval_completion] Wrote: {args.output_csv}")
    print(f"[merge_idm_eval_completion] Wrote: {args.output_json}")
    print(f"[merge_idm_eval_completion] Wrote: {args.output_md}")

    failed = [r for r in rows if r["return_code"] != 0 or not r["parsed_result"]]
    if args.require_complete and failed:
        print(
            f"[merge_idm_eval_completion] {len(failed)} / {len(rows)} rows failed or were unparsed.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
