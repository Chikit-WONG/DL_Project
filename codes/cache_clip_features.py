"""One-time CLIP feature extraction.

Walks ``training_images/`` and ``test_images/``, runs each image through
IP-Adapter's bundled CLIP-ViT-H-14 image encoder, and saves a single
``{image_id: tensor[1024]}`` dict per split to ``clip_cache/``.

Run via ``sbatch slurm_scripts/run_cache_clip.sh``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

# Allow ``python codes/cache_clip_features.py`` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG  # noqa: E402
from utils import build_image_id_to_path  # noqa: E402


def cache_split(split: str, image_root: Path, encoder, processor, device, batch_size: int = 32):
    id2path = build_image_id_to_path(image_root)
    image_ids = sorted(id2path.keys())
    print(f"[{split}] {len(image_ids)} images under {image_root}")

    out: dict[str, torch.Tensor] = {}
    for i in tqdm(range(0, len(image_ids), batch_size), desc=f"encoding {split}"):
        batch_ids = image_ids[i : i + batch_size]
        pils = [Image.open(id2path[ids]).convert("RGB") for ids in batch_ids]
        inputs = processor(images=pils, return_tensors="pt").to(device)
        with torch.no_grad():
            embeds = encoder(**inputs).image_embeds  # [B, 1024]
        embeds = embeds.float().cpu()
        for j, ids in enumerate(batch_ids):
            out[ids] = embeds[j].clone()
    return out


def main():
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading CLIP image encoder from {cfg.image_encoder_path}")
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(cfg.image_encoder_path)
    ).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(str(cfg.image_processor_path))

    train_cache_path = cfg.cache_dir / "clip_train_features.pt"
    test_cache_path = cfg.cache_dir / "clip_test_features.pt"

    train_features = cache_split(
        "train", cfg.train_image_dir, encoder, processor, device, batch_size=64
    )
    torch.save(train_features, train_cache_path)
    print(f"saved {len(train_features)} train features -> {train_cache_path}")

    test_features = cache_split(
        "test", cfg.test_image_dir, encoder, processor, device, batch_size=64
    )
    torch.save(test_features, test_cache_path)
    print(f"saved {len(test_features)} test features  -> {test_cache_path}")

    # Sanity print: first feature shape and norm
    sample_id = next(iter(test_features))
    print(f"sample {sample_id}: shape={tuple(test_features[sample_id].shape)} "
          f"norm={test_features[sample_id].norm().item():.4f}")


if __name__ == "__main__":
    main()
