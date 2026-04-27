from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from safe_bpmgd.utils.io import ensure_dir


def cache_compact_vae_like_latents(records, out_path: str | Path, image_size: int = 64, latent_dim: int = 512) -> Path:
    """Dependency-light low-level latent fallback.

    It is not a replacement for SDXL VAE latents, but keeps the training interface
    stable when diffusers/VAE jobs are not available yet.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    vectors = []
    for rec in records:
        with Image.open(rec.image_path) as image:
            flat = transform(image.convert("RGB")).flatten()
            if flat.numel() >= latent_dim:
                vec = F.adaptive_avg_pool1d(flat[None, None, :], latent_dim).flatten()
            else:
                vec = F.pad(flat, (0, latent_dim - flat.numel()))
            vectors.append(vec.float())
    torch.save(
        {
            "source_indices": [int(rec.source_index) for rec in records],
            "image_ids": [rec.image_id for rec in records],
            "image_paths": [rec.image_path for rec in records],
            "features": torch.stack(vectors),
            "note": "compact image latent fallback; replace with SDXL VAE cache when available",
        },
        out_path,
    )
    return out_path
