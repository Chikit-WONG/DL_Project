from __future__ import annotations

from pathlib import Path

import torch

from safe_bpmgd.data.leakage_guard import LeakageGuard
from safe_bpmgd.utils.io import ensure_dir


def build_train_memory_bank(cache_dir: str | Path, out_path: str | Path, guard: LeakageGuard | None = None) -> Path:
    cache_dir = Path(cache_dir)
    out_path = Path(out_path)
    clip = torch.load(cache_dir / "clip_rn50.pt", map_location="cpu", weights_only=False)
    bank = {
        "image_ids": clip["image_ids"],
        "image_paths": clip["image_paths"],
        "source_indices": clip["source_indices"],
        "clip": clip["features"],
        "source": "train-only",
    }
    optional = {
        "multiblur": "multiblur_clip.pt",
        "evnet": "evnet_struct.pt",
        "edge": "edge_clip.pt",
        "depth": "depth_clip.pt",
        "vae": "vae_latents.pt",
    }
    for key, name in optional.items():
        path = cache_dir / name
        if path.exists():
            bank[key] = torch.load(path, map_location="cpu", weights_only=False)["features"]
    if guard is not None:
        guard.assert_train_memory_bank_paths(bank["image_paths"])
    ensure_dir(out_path.parent)
    torch.save(bank, out_path)
    return out_path
