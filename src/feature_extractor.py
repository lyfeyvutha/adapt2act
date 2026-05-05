import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_IDS = {
    "dino-v3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dino-v2": "facebook/dinov2-base",
    "siglip": "google/siglip-base-patch16-224",
    "vgg-t": "vgg16_bn",
}


def parse_embeddings(raw: str | Sequence[str]) -> List[str]:
    if isinstance(raw, str):
        values = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        values = [str(x).strip() for x in raw if str(x).strip()]
    if not values:
        raise ValueError("At least one embedding name is required")
    return values


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _pool_hf_output(outputs):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    hidden = outputs.last_hidden_state
    return hidden[:, 0, :] if hidden.ndim == 3 else hidden


class VC1Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        from vc_models.models.vit import model_utils

        self.model, self.embd_size, self.transform, self.model_info = model_utils.load_model(
            model_utils.VC1_BASE_NAME
        )
        _freeze(self.model)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.transform(x))


class HFVisionWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        try:
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        except ValueError as exc:
            message = str(exc)
            if "dinov3_vit" in message:
                raise RuntimeError(
                    "DINOv3 checkpoints require a newer transformers build than this environment provides. "
                    "Upgrade transformers, then rerun the extractor."
                ) from exc
            raise

        vision_config = getattr(config, "vision_config", None)
        if isinstance(vision_config, dict):
            vision_hidden_size = vision_config.get("hidden_size", 0)
        else:
            vision_hidden_size = getattr(vision_config, "hidden_size", 0)
        self.embd_size = int(
            getattr(config, "hidden_size", 0)
            or vision_hidden_size
            or getattr(config, "projection_dim", 0)
        )
        if self.embd_size <= 0:
            raise ValueError(f"Could not infer embedding size for {model_name}")
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.Normalize(mean=list(image_mean), std=list(image_std)),
            ]
        )
        _freeze(self.model)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self.transform(x)
        if hasattr(self.model, "get_image_features"):
            return self.model.get_image_features(pixel_values=pixel_values)
        return _pool_hf_output(self.model(pixel_values=pixel_values))


class TimmVisionWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        image_size: int = 224,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.embd_size = int(getattr(self.model, "num_features", 0))
        if self.embd_size <= 0:
            raise ValueError(f"Could not infer embedding size for timm model {model_name}")
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size), antialias=True),
                T.Normalize(mean=list(image_mean), std=list(image_std)),
            ]
        )
        _freeze(self.model)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.transform(x))


def build_encoder(
    embedding: str,
    model_overrides: Optional[Dict[str, str]] = None,
    local_files_only: bool = False,
    timm_pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    name = embedding.strip().lower()
    model_overrides = model_overrides or {}

    if name == "vc-1":
        wrapper = VC1Wrapper()
    elif name in {"dino-v3", "dino-v2"}:
        wrapper = HFVisionWrapper(
            model_overrides.get(name, DEFAULT_MODEL_IDS[name]),
            image_mean=(0.485, 0.456, 0.406),
            image_std=(0.229, 0.224, 0.225),
            local_files_only=local_files_only,
        )
    elif name == "siglip":
        wrapper = HFVisionWrapper(
            model_overrides.get(name, DEFAULT_MODEL_IDS[name]),
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            local_files_only=local_files_only,
        )
    elif name in {"vgg-t", "vgg", "vgg16"}:
        wrapper = TimmVisionWrapper(
            model_overrides.get("vgg-t", DEFAULT_MODEL_IDS["vgg-t"]),
            pretrained=timm_pretrained,
        )
    else:
        raise ValueError(f"Unknown embedding '{embedding}'. Try vc-1,dino-v3,dino-v2,siglip,vgg-t")
    return wrapper, int(wrapper.embd_size)


class FrozenFeatureStack(nn.Module):
    def __init__(
        self,
        embeddings: Sequence[str] = ("vc-1",),
        model_overrides: Optional[Dict[str, str]] = None,
        local_files_only: bool = False,
        timm_pretrained: bool = True,
    ):
        super().__init__()
        self.embeddings = parse_embeddings(embeddings)
        self.visual_models = nn.ModuleList()
        self.embedding_dims: List[int] = []
        for embedding in self.embeddings:
            model, embd_size = build_encoder(
                embedding,
                model_overrides=model_overrides,
                local_files_only=local_files_only,
                timm_pretrained=timm_pretrained,
            )
            self.visual_models.append(model)
            self.embedding_dims.append(embd_size)
        self.embd_size = int(sum(self.embedding_dims))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = [model(images) for model in self.visual_models]
        return torch.cat(features, dim=1) if len(features) > 1 else features[0]


class InvDynamics(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        embeddings: Sequence[str] = ("vc-1",),
        feature_dim: Optional[int] = None,
        build_encoders: bool = True,
        model_overrides: Optional[Dict[str, str]] = None,
        local_files_only: bool = False,
        timm_pretrained: bool = True,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.embeddings = parse_embeddings(embeddings)
        self.features: Optional[FrozenFeatureStack] = None

        if feature_dim is None:
            if not build_encoders:
                raise ValueError("feature_dim is required when build_encoders=False")
            self.features = FrozenFeatureStack(
                self.embeddings,
                model_overrides=model_overrides,
                local_files_only=local_files_only,
                timm_pretrained=timm_pretrained,
            )
            self.embd_size = self.features.embd_size
        else:
            self.embd_size = int(feature_dim)
            if build_encoders:
                self.features = FrozenFeatureStack(
                    self.embeddings,
                    model_overrides=model_overrides,
                    local_files_only=local_files_only,
                    timm_pretrained=timm_pretrained,
                )
                if self.features.embd_size != self.embd_size:
                    raise ValueError(
                        f"feature_dim={self.embd_size} does not match encoder dim={self.features.embd_size}"
                    )

        self.inv_model = nn.Linear(2 * self.embd_size, self.action_dim)

    def encode_pair(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 3:
            b, n, d = obs.shape
            if n != 2 or d != self.embd_size:
                raise ValueError(f"Expected latent obs [B,2,{self.embd_size}], got {tuple(obs.shape)}")
            return torch.reshape(obs, [b, 2 * d])

        if obs.ndim != 5:
            raise ValueError(f"Expected image obs [B,2,C,H,W] or latent obs [B,2,D], got {tuple(obs.shape)}")
        if self.features is None:
            raise ValueError("This IDM was constructed without encoders and cannot consume images")

        b, n, c, h, w = obs.shape
        obs = torch.reshape(obs, [b * n, c, h, w])
        with torch.no_grad():
            embed = self.features(obs)
        return torch.reshape(embed, [b, n * self.embd_size])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.inv_model(self.encode_pair(obs))

    def calculate_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(obs), action)

    @torch.no_grad()
    def calculate_test_loss(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(obs), action)

    def trainable_state_dict(self):
        return {"inv_model.weight": self.inv_model.weight.detach().cpu(), "inv_model.bias": self.inv_model.bias.detach().cpu()}


class PairImageDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=False)
        self.frame_t = data["frame_t"]
        self.frame_tp1 = data["frame_tp1"]
        self.actions = data["actions"].astype(np.float32)
        self.task_ids = data["task_ids"] if "task_ids" in data else None
        if len(self.frame_t) != len(self.actions):
            raise ValueError(f"Mismatched data lengths in {npz_path}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        f0 = torch.from_numpy(self.frame_t[idx]).permute(2, 0, 1).float() / 255.0
        f1 = torch.from_numpy(self.frame_tp1[idx]).permute(2, 0, 1).float() / 255.0
        return torch.stack([f0, f1], dim=0), torch.from_numpy(self.actions[idx])


@torch.no_grad()
def extract_split(
    encoder: FrozenFeatureStack,
    npz_path: Path,
    output_path: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    dataset = PairImageDataset(npz_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    features_t = []
    features_tp1 = []
    actions = []
    encoder.eval()
    for obs, action in loader:
        obs = obs.to(device, non_blocking=True)
        b, n, c, h, w = obs.shape
        feats = encoder(torch.reshape(obs, [b * n, c, h, w]))
        feats = torch.reshape(feats, [b, n, encoder.embd_size]).cpu().numpy().astype(np.float32)
        features_t.append(feats[:, 0])
        features_tp1.append(feats[:, 1])
        actions.append(action.numpy().astype(np.float32))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=True) as data:
        extra = {}
        for key in ("task_ids", "task_to_id"):
            if key in data:
                extra[key] = data[key]
    np.savez_compressed(
        output_path,
        features_t=np.concatenate(features_t, axis=0),
        features_tp1=np.concatenate(features_tp1, axis=0),
        actions=np.concatenate(actions, axis=0),
        **extra,
    )


def _parse_model_overrides(raw: Iterable[str]) -> Dict[str, str]:
    out = {}
    for item in raw:
        if not item:
            continue
        key, value = item.split("=", 1)
        out[key.strip().lower()] = value.strip()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen visual representations for IDM training.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing train.npz and val.npz")
    parser.add_argument("--output-dir", type=str, default="", help="Defaults to <dataset-dir>_features_<embeddings>")
    parser.add_argument("--embeddings", type=str, default="vc-1", help="Comma-separated: vc-1,dino-v3,dino-v2,siglip,vgg-t")
    parser.add_argument("--split", type=str, default="train,val", help="Comma-separated splits to extract")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-timm-pretrained",
        action="store_true",
        help="Instantiate timm encoders without downloading pretrained weights; intended only for offline smoke tests.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Override backing model, e.g. --model dino-v3=facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    embeddings = parse_embeddings(args.embeddings)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir.with_name(
        f"{dataset_dir.name}_features_{'-'.join(embeddings)}"
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    encoder = FrozenFeatureStack(
        embeddings,
        model_overrides=_parse_model_overrides(args.model),
        local_files_only=args.local_files_only,
        timm_pretrained=not args.no_timm_pretrained,
    ).to(device)

    splits = [x.strip() for x in args.split.split(",") if x.strip()]
    for split in splits:
        src = dataset_dir / f"{split}.npz"
        if not src.exists():
            raise FileNotFoundError(f"Missing split file: {src}")
        dst = output_dir / f"{split}.npz"
        print(f"[feature_extractor] extracting split={split} embeddings={embeddings} src={src}")
        extract_split(encoder, src, dst, args.batch_size, args.num_workers, device)
        print(f"[feature_extractor] wrote {dst}")

    metadata = {
        "source_dataset_dir": str(dataset_dir),
        "embeddings": embeddings,
        "embedding_dims": encoder.embedding_dims,
        "feature_dim": encoder.embd_size,
        "splits": splits,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[feature_extractor] feature_dim={encoder.embd_size} metadata={output_dir / 'feature_metadata.json'}")


if __name__ == "__main__":
    main()
