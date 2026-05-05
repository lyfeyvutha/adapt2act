import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from feature_extractor import InvDynamics, parse_embeddings


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=False)
        self.uses_features = "features_t" in data and "features_tp1" in data
        if self.uses_features:
            self.features_t = data["features_t"].astype(np.float32)  # [N, D]
            self.features_tp1 = data["features_tp1"].astype(np.float32)  # [N, D]
            self.feature_dim = int(self.features_t.shape[-1])
        else:
            self.frame_t = data["frame_t"]  # [N, H, W, 3], uint8
            self.frame_tp1 = data["frame_tp1"]  # [N, H, W, 3], uint8
            self.feature_dim = None
        self.actions = data["actions"].astype(np.float32)  # [N, A]
        first_len = len(self.features_t) if self.uses_features else len(self.frame_t)
        if first_len != len(self.actions):
            raise ValueError(f"Mismatched data lengths in {npz_path}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        if self.uses_features:
            f0 = torch.from_numpy(self.features_t[idx])
            f1 = torch.from_numpy(self.features_tp1[idx])
            return torch.stack([f0, f1], dim=0), torch.from_numpy(self.actions[idx])

        # Convert to [2, 3, H, W] in [0, 1] range.
        f0 = torch.from_numpy(self.frame_t[idx]).permute(2, 0, 1).float() / 255.0
        f1 = torch.from_numpy(self.frame_tp1[idx]).permute(2, 0, 1).float() / 255.0
        obs = torch.stack([f0, f1], dim=0)
        act = torch.from_numpy(self.actions[idx])
        return obs, act


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inverse dynamics model on processed IDM pairs.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing train.npz and val.npz")
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="/cs/data/people/cvutha/checkpoints/idm")
    parser.add_argument("--embeddings", type=str, default="vc-1", help="Comma-separated: vc-1,dino-v3,dino-v2,siglip,vgg-t")
    parser.add_argument(
        "--input-mode",
        type=str,
        default="auto",
        choices=["auto", "images", "features"],
        help="Use raw image pairs, precomputed feature pairs, or infer from train.npz.",
    )
    return parser.parse_args()


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model: InvDynamics, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for obs, action in loader:
        obs = obs.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        loss = model.calculate_test_loss(obs, action)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def save_ckpt(path: Path, model: InvDynamics, optimizer: torch.optim.Optimizer, step: int, metrics: Dict, args: argparse.Namespace):
    state_dict_fn = getattr(model, "trainable_state_dict", None)
    payload = {
        "inv_model": state_dict_fn() if state_dict_fn is not None else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "args": vars(args),
        "idm_metadata": {
            "embeddings": getattr(model, "embeddings", ["vc-1"]),
            "feature_dim": getattr(model, "embd_size", None),
            "action_dim": getattr(model, "action_dim", None),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / "train.npz"
    val_path = dataset_dir / "val.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Expected train/val files under {dataset_dir}")

    train_ds = PairDataset(train_path)
    val_ds = PairDataset(val_path)
    if len(train_ds) == 0:
        raise ValueError("Training set is empty")
    if train_ds.uses_features != val_ds.uses_features:
        raise ValueError("Train and val splits must both use images or both use precomputed features")

    inferred_mode = "features" if train_ds.uses_features else "images"
    input_mode = inferred_mode if args.input_mode == "auto" else args.input_mode
    if input_mode != inferred_mode:
        raise ValueError(f"--input-mode={args.input_mode} does not match dataset format ({inferred_mode})")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0 if args.num_workers == 0 else max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    train_iter = cycle(train_loader)

    embeddings = parse_embeddings(args.embeddings)
    model = InvDynamics(
        action_dim=train_ds.actions.shape[-1],
        embeddings=embeddings,
        feature_dim=train_ds.feature_dim,
        build_encoders=(input_mode == "images"),
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    dataset_name = dataset_dir.name
    run_name = f"idm_{dataset_name}_seed{args.seed}"
    out_dir = Path(args.output_root) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_idm] output_dir={out_dir}")
    print(f"[train_idm] input_mode={input_mode} embeddings={embeddings} feature_dim={model.embd_size}")

    best_val = float("inf")
    history = []
    model.train()
    for step in range(1, args.num_steps + 1):
        obs, action = next(train_iter)
        obs = obs.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)

        loss = model.calculate_loss(obs, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            val_loss = evaluate(model, val_loader, device)
            entry = {"step": step, "train_loss": float(loss.item()), "val_loss": float(val_loss)}
            history.append(entry)
            print(f"[train_idm] step={step} train_loss={entry['train_loss']:.6f} val_loss={entry['val_loss']:.6f}")
            model.train()

            if val_loss < best_val:
                best_val = val_loss
                best_fp = out_dir / f"{run_name}_best.ckpt"
                save_ckpt(best_fp, model, optimizer, step, {"best_val_loss": best_val, "history_tail": history[-10:]}, args)
                print(f"[train_idm] Saved best checkpoint: {best_fp}")

        if step % args.save_every == 0:
            step_fp = out_dir / f"{run_name}_step{step}.ckpt"
            save_ckpt(step_fp, model, optimizer, step, {"best_val_loss": best_val}, args)

    final_fp = out_dir / f"{run_name}.ckpt"
    save_ckpt(final_fp, model, optimizer, args.num_steps, {"best_val_loss": best_val, "history_tail": history[-20:]}, args)
    print(f"[train_idm] Saved final checkpoint: {final_fp}")

    with open(out_dir / f"{run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_loss": best_val, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
