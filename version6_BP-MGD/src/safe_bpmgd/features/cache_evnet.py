from __future__ import annotations

from pathlib import Path

import torch

from safe_bpmgd.features.cache_vae import cache_compact_vae_like_latents


def cache_evnet_fallback(records, out_path: str | Path, struct_dim: int = 256) -> Path:
    path = cache_compact_vae_like_latents(records, out_path, image_size=48, latent_dim=struct_dim)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["note"] = "EVNet fallback structural descriptor; replace with frozen EVNet activations when integrated"
    torch.save(payload, path)
    return Path(path)
