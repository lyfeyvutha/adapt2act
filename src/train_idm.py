import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils import InvDynamics, set_seed


class PairDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=False)
        self.frame_t = data["frame_t"]  # [N, H, W, 3], uint8
        self.frame_tp1 = data["frame_tp1"]  # [N, H, W, 3], uint8
        self.actions = data["actions"].astype(np.float32)  # [N, A]
        if len(self.frame_t) != len(self.actions):
            raise ValueError(f"Mismatched data lengths in {npz_path}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
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
    parser.add_argument("--output-root", type=str, default="checkpoints/idm")
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
    payload = {
        "inv_model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "args": vars(args),
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
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    train_iter = cycle(train_loader)

    model = InvDynamics(action_dim=train_ds.actions.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_name = dataset_dir.name
    run_name = f"idm_{dataset_name}_seed{args.seed}"
    out_dir = Path(args.output_root) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

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
