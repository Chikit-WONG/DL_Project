from __future__ import annotations

import os
from pathlib import Path


DEFAULT_MODEL_ROOT = Path("/hpc2hdd/home/ckwong627/workdir/models")
DEFAULT_OPEN_CLIP_DIRNAME = "CLIP-ViT-H-14-laion2B-s32B-b79K"
OPEN_CLIP_WEIGHT_FILES = (
    "open_clip_pytorch_model.bin",
    "open_clip_model.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
)


def resolve_model_root(model_root: str | Path | None = None) -> Path:
    if model_root is not None:
        return Path(model_root)
    return Path(os.environ.get("EEG_IMAGE_DECODE_MODEL_ROOT", DEFAULT_MODEL_ROOT))


def resolve_open_clip_checkpoint(model_root: str | Path | None = None) -> Path:
    explicit_checkpoint = os.environ.get("EEG_IMAGE_DECODE_CLIP_CHECKPOINT")
    if explicit_checkpoint:
        checkpoint_path = Path(explicit_checkpoint)
        if checkpoint_path.is_file():
            return checkpoint_path
        raise FileNotFoundError(
            f"EEG_IMAGE_DECODE_CLIP_CHECKPOINT points to a missing file: {checkpoint_path}"
        )

    root = resolve_model_root(model_root)
    clip_dir = root / DEFAULT_OPEN_CLIP_DIRNAME
    for filename in OPEN_CLIP_WEIGHT_FILES:
        checkpoint_path = clip_dir / filename
        if checkpoint_path.is_file():
            return checkpoint_path

    raise FileNotFoundError(
        "Could not find a local OpenCLIP checkpoint. "
        f"Checked under {clip_dir} for: {', '.join(OPEN_CLIP_WEIGHT_FILES)}"
    )


def create_open_clip_vit_h_14(*, device: str, precision: str = "fp32", model_root: str | Path | None = None):
    import open_clip

    checkpoint_path = resolve_open_clip_checkpoint(model_root)
    return open_clip.create_model_and_transforms(
        "ViT-H-14",
        pretrained=str(checkpoint_path),
        precision=precision,
        device=device,
    )


def get_feature_cache_path(
    *,
    feature_root: str | Path | None,
    model_type: str,
    split: str,
) -> Path:
    cache_root = Path(feature_root) if feature_root is not None else Path.cwd()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{model_type}_features_{split}.pt"
