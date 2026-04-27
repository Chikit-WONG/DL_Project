from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from safe_bpmgd.utils.io import ensure_dir


def cache_edge_depth_maps(records, out_dir: str | Path, image_size: int = 256) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    edge_tensors = []
    depth_tensors = []
    for rec in records:
        with Image.open(rec.image_path) as image:
            gray = ImageOps.grayscale(image.convert("RGB").resize((image_size, image_size), Image.BILINEAR))
            edge = gray.filter(ImageFilter.FIND_EDGES)
            depth = _pseudo_depth(gray)
            edge_tensors.append(_to_tensor(edge))
            depth_tensors.append(_to_tensor(depth))
    payload = {
        "source_indices": [int(rec.source_index) for rec in records],
        "image_ids": [rec.image_id for rec in records],
        "image_paths": [rec.image_path for rec in records],
    }
    edge_path = out_dir / "edge_maps.pt"
    depth_path = out_dir / "depth_maps.pt"
    torch.save({**payload, "features": torch.stack(edge_tensors)}, edge_path)
    torch.save({**payload, "features": torch.stack(depth_tensors)}, depth_path)
    return {"edge_maps": edge_path, "depth_maps": depth_path}


def _to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _pseudo_depth(gray: Image.Image) -> Image.Image:
    arr = np.asarray(gray, dtype=np.float32)
    yy = np.linspace(1.0, 0.6, arr.shape[0], dtype=np.float32)[:, None]
    depth = 0.55 * arr + 0.45 * arr.mean() * yy
    depth = np.clip(depth, 0, 255).astype(np.uint8)
    return Image.fromarray(depth)
