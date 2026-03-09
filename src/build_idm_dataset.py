import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed IDM datasets from collected episode files.")
    parser.add_argument("--input-root", type=str, default="idm_data/raw")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-root", type=str, default="idm_data/processed")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated tasks")
    parser.add_argument("--sources", type=str, default="expert,random,agent_ckpt", help="Comma-separated allowed policy sources")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--success-filter", type=str, default="all", choices=["all", "success_only"])
    parser.add_argument("--counts-per-task", type=str, default="", help="Explicit counts: taskA:100,taskB:50")
    parser.add_argument("--mix-ratios", type=str, default="", help="Ratio map: taskA:0.7,taskB:0.3 (requires --total-trajectories)")
    parser.add_argument("--total-trajectories", type=int, default=0)
    parser.add_argument("--default-trajectories-per-task", type=int, default=0, help="Used when no explicit counts/ratios")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def _parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_kv_map(raw: str) -> Dict[str, float]:
    if not raw:
        return {}
    out: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split(":")
        out[key.strip()] = float(val.strip())
    return out


def _discover_episodes(
    input_root: Path,
    task: str,
    sources: Sequence[str],
    success_filter: str,
) -> List[Path]:
    out: List[Path] = []
    for source in sources:
        source_dir = input_root / task / source
        if not source_dir.exists():
            continue
        for ep_file in sorted(source_dir.glob("seed*/episodes/*.npz")):
            if success_filter == "success_only":
                with np.load(ep_file, allow_pickle=False) as d:
                    if float(d["success"]) <= 0.0:
                        continue
            out.append(ep_file)
    return out


def _resolve_task_counts(
    tasks: List[str],
    available: Dict[str, int],
    explicit_counts: Dict[str, float],
    mix_ratios: Dict[str, float],
    total_trajectories: int,
    default_per_task: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if explicit_counts:
        for task in tasks:
            counts[task] = int(explicit_counts.get(task, 0))
        return counts

    if mix_ratios:
        if total_trajectories <= 0:
            raise ValueError("--total-trajectories must be > 0 when --mix-ratios is provided")
        ratio_sum = sum(mix_ratios.get(task, 0.0) for task in tasks)
        if ratio_sum <= 0:
            raise ValueError("mix ratios sum to 0 for selected tasks")
        for task in tasks:
            counts[task] = int(round(total_trajectories * (mix_ratios.get(task, 0.0) / ratio_sum)))
        # Ensure exact total by distributing rounding residue.
        residue = total_trajectories - sum(counts.values())
        i = 0
        while residue != 0 and tasks:
            t = tasks[i % len(tasks)]
            if residue > 0:
                counts[t] += 1
                residue -= 1
            elif counts[t] > 0:
                counts[t] -= 1
                residue += 1
            i += 1
        return counts

    if default_per_task > 0:
        for task in tasks:
            counts[task] = int(default_per_task)
    else:
        for task in tasks:
            counts[task] = int(available.get(task, 0))
    return counts


def _sample_episodes(task_to_files: Dict[str, List[Path]], task_counts: Dict[str, int], rng: np.random.Generator):
    selected: Dict[str, List[Path]] = {}
    for task, files in task_to_files.items():
        if not files:
            selected[task] = []
            continue
        want = task_counts.get(task, 0)
        if want <= 0:
            selected[task] = []
            continue
        if want >= len(files):
            selected[task] = list(files)
            continue
        idxs = rng.choice(len(files), size=want, replace=False)
        selected[task] = [files[i] for i in sorted(idxs)]
    return selected


def _split_train_val(files: List[Path], val_ratio: float, rng: np.random.Generator) -> Tuple[List[Path], List[Path]]:
    if not files:
        return [], []
    order = np.arange(len(files))
    rng.shuffle(order)
    n_val = int(round(len(files) * val_ratio))
    n_val = min(max(n_val, 1 if len(files) > 1 else 0), len(files))
    val_set = set(order[:n_val].tolist())
    train_files, val_files = [], []
    for i, fp in enumerate(files):
        if i in val_set:
            val_files.append(fp)
        else:
            train_files.append(fp)
    return train_files, val_files


def _episodes_to_arrays(files: List[Path]) -> Dict[str, np.ndarray]:
    frame_t = []
    frame_tp1 = []
    actions = []
    task_ids = []
    task_to_id: Dict[str, int] = {}

    for fp in files:
        with np.load(fp, allow_pickle=True) as data:
            frames = data["frames"]  # [T+1, H, W, 3]
            act = data["actions"]  # [T, A]
            task = str(data["task"].item() if np.asarray(data["task"]).shape == () else data["task"])
            if task not in task_to_id:
                task_to_id[task] = len(task_to_id)
            tid = task_to_id[task]
            if len(frames) != len(act) + 1:
                raise ValueError(f"Invalid episode lengths in {fp}: frames={len(frames)} actions={len(act)}")
            frame_t.append(frames[:-1])
            frame_tp1.append(frames[1:])
            actions.append(act)
            task_ids.append(np.full((len(act),), tid, dtype=np.int32))

    if not actions:
        return {
            "frame_t": np.zeros((0, 0, 0, 0), dtype=np.uint8),
            "frame_tp1": np.zeros((0, 0, 0, 0), dtype=np.uint8),
            "actions": np.zeros((0, 0), dtype=np.float32),
            "task_ids": np.zeros((0,), dtype=np.int32),
            "task_to_id": np.asarray([], dtype=object),
        }

    return {
        "frame_t": np.concatenate(frame_t, axis=0).astype(np.uint8),
        "frame_tp1": np.concatenate(frame_tp1, axis=0).astype(np.uint8),
        "actions": np.concatenate(actions, axis=0).astype(np.float32),
        "task_ids": np.concatenate(task_ids, axis=0).astype(np.int32),
        "task_to_id": np.asarray(sorted(task_to_id.items(), key=lambda x: x[1]), dtype=object),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_root = Path(args.input_root)
    output_dir = Path(args.output_root) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _parse_csv_list(args.tasks)
    if not tasks:
        raise ValueError("No tasks provided")
    sources = _parse_csv_list(args.sources)
    explicit_counts = _parse_kv_map(args.counts_per_task)
    mix_ratios = _parse_kv_map(args.mix_ratios)

    task_to_files: Dict[str, List[Path]] = {}
    available: Dict[str, int] = {}
    for task in tasks:
        files = _discover_episodes(input_root, task, sources, args.success_filter)
        task_to_files[task] = files
        available[task] = len(files)

    task_counts = _resolve_task_counts(
        tasks=tasks,
        available=available,
        explicit_counts=explicit_counts,
        mix_ratios=mix_ratios,
        total_trajectories=args.total_trajectories,
        default_per_task=args.default_trajectories_per_task,
    )
    selected_per_task = _sample_episodes(task_to_files, task_counts, rng)
    all_selected = [fp for task in tasks for fp in selected_per_task[task]]

    train_eps, val_eps = _split_train_val(all_selected, args.val_ratio, rng)
    train_arrays = _episodes_to_arrays(train_eps)
    val_arrays = _episodes_to_arrays(val_eps)

    np.savez_compressed(
        output_dir / "train.npz",
        frame_t=train_arrays["frame_t"],
        frame_tp1=train_arrays["frame_tp1"],
        actions=train_arrays["actions"],
        task_ids=train_arrays["task_ids"],
        task_to_id=train_arrays["task_to_id"],
    )
    np.savez_compressed(
        output_dir / "val.npz",
        frame_t=val_arrays["frame_t"],
        frame_tp1=val_arrays["frame_tp1"],
        actions=val_arrays["actions"],
        task_ids=val_arrays["task_ids"],
        task_to_id=val_arrays["task_to_id"],
    )

    manifest = {
        "dataset_name": args.dataset_name,
        "input_root": str(input_root),
        "seed": args.seed,
        "tasks": tasks,
        "sources": sources,
        "success_filter": args.success_filter,
        "available_episodes_per_task": available,
        "selected_episodes_per_task": {k: len(v) for k, v in selected_per_task.items()},
        "selected_episode_files": [str(x) for x in all_selected],
        "train_episode_files": [str(x) for x in train_eps],
        "val_episode_files": [str(x) for x in val_eps],
        "counts_per_task": task_counts,
        "mix_ratios": mix_ratios,
        "total_trajectories": args.total_trajectories,
        "default_trajectories_per_task": args.default_trajectories_per_task,
        "num_train_pairs": int(train_arrays["actions"].shape[0]),
        "num_val_pairs": int(val_arrays["actions"].shape[0]),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[build_idm_dataset] Wrote dataset at: {output_dir}")
    print(f"[build_idm_dataset] Train pairs: {manifest['num_train_pairs']} | Val pairs: {manifest['num_val_pairs']}")


if __name__ == "__main__":
    main()
