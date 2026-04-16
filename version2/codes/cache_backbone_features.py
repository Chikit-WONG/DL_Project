from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

from config import DEFAULT_CONFIG
from data import load_split_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_clip_encoder(model_dir: Path, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model = CLIPVisionModelWithProjection.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()
    return processor, model


def load_rn50_encoder(device: torch.device):
    try:
        import clip  # type: ignore

        model, preprocess = clip.load("RN50", device=device)
        model.eval()
        return model, preprocess
    except Exception:
        return None, None


def image_to_tensor(image: Image.Image, size: int = 512) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )(image)


@torch.no_grad()
def encode_images(processor, model, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    return outputs.image_embeds.detach().cpu().float()


@torch.no_grad()
def encode_rn50(model, preprocess, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    batch = torch.stack([preprocess(image) for image in images], dim=0).to(device)
    outputs = model.encode_image(batch)
    return outputs.detach().cpu().float()


@torch.no_grad()
def encode_vae(vae: AutoencoderKL, images: list[Image.Image], device: torch.device) -> torch.Tensor:
    batch = torch.stack([image_to_tensor(image) for image in images], dim=0).to(device)
    batch = batch * 2.0 - 1.0
    posterior = vae.encode(batch).latent_dist
    latents = posterior.mean * vae.config.scaling_factor
    return latents.detach().cpu().float()


def cache_split(cfg, split: str, batch_size: int, limit: int | None = None) -> Path:
    cfg.ensure_dirs()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    records = load_split_records(cfg, split=split, limit=limit)
    processor_h14, model_h14 = load_clip_encoder(cfg.h14_model_dir, device)
    processor_b32, model_b32 = load_clip_encoder(cfg.b32_model_dir, device)
    rn50_model, rn50_preprocess = load_rn50_encoder(device)
    vae = AutoencoderKL.from_pretrained(str(cfg.sd15_dir), subfolder="vae").to(device)
    vae.eval()

    image_ids: list[str] = []
    image_paths: list[str] = []
    labels: list[int] = []
    label_names: list[str] = []
    subject_ids: list[int] = []
    h14_chunks: list[torch.Tensor] = []
    b32_chunks: list[torch.Tensor] = []
    rn50_chunks: list[torch.Tensor] = []
    vae_chunks: list[torch.Tensor] = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        images = [Image.open(record.image_path).convert("RGB") for record in batch_records]
        h14 = encode_images(processor_h14, model_h14, images, device)
        b32 = encode_images(processor_b32, model_b32, images, device)
        rn50 = encode_rn50(rn50_model, rn50_preprocess, images, device) if rn50_model is not None else b32.clone()
        vae_latent = encode_vae(vae, images, device)
        h14_chunks.append(F.normalize(h14, dim=-1))
        b32_chunks.append(F.normalize(b32, dim=-1))
        rn50_chunks.append(F.normalize(rn50, dim=-1))
        vae_chunks.append(vae_latent)
        image_ids.extend(record.image_id for record in batch_records)
        image_paths.extend(str(record.image_path) for record in batch_records)
        labels.extend(record.label for record in batch_records)
        label_names.extend(record.label_name for record in batch_records)
        subject_ids.extend(record.subject_id for record in batch_records)
        for image in images:
            image.close()

    payload = {
        "split": split,
        "image_ids": image_ids,
        "image_paths": image_paths,
        "labels": torch.tensor(labels, dtype=torch.long),
        "label_names": label_names,
        "subject_ids": torch.tensor(subject_ids, dtype=torch.long),
        "features": {
            "h14": torch.cat(h14_chunks, dim=0),
            "b32": torch.cat(b32_chunks, dim=0),
            "rn50": torch.cat(rn50_chunks, dim=0),
            "vae_latent": torch.cat(vae_chunks, dim=0),
        },
    }
    out_path = cfg.cache_dir / f"backbone_cache_{split}.pt"
    torch.save(payload, out_path)
    meta_path = cfg.cache_dir / f"backbone_cache_{split}.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split,
                "count": len(image_ids),
                "h14_shape": list(payload["features"]["h14"].shape),
                "b32_shape": list(payload["features"]["b32"].shape),
                "rn50_shape": list(payload["features"]["rn50"].shape),
                "vae_latent_shape": list(payload["features"]["vae_latent"].shape),
            },
            handle,
            indent=2,
        )
    return out_path


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        out_path = cache_split(cfg, split=split, batch_size=args.batch_size, limit=args.limit)
        print(f"Saved cache: {out_path}")


if __name__ == "__main__":
    main()
