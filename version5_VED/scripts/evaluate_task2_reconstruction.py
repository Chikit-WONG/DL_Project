from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import torch
from torchvision import transforms


DEFAULT_CLIP_CANDIDATES = [
    (
        "ViT-H-14",
        "/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    ),
    (
        "ViT-L-14",
        "/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin",
    ),
    (
        "RN50",
        "/hpc2hdd/home/ckwong627/workdir/models/CLIP-RN50-openai/open_clip_pytorch_model.bin",
    ),
]


def resolve_clip_weights(requested_model_name: str | None, requested_path: str | None) -> tuple[str, str]:
    if requested_model_name and requested_path and Path(requested_path).exists():
        return requested_model_name, requested_path
    for model_name, path in DEFAULT_CLIP_CANDIDATES:
        if Path(path).exists():
            return model_name, path
    raise FileNotFoundError("No local OpenCLIP checkpoint found for reconstruction evaluation.")


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )(image.convert("RGB"))
    return tensor


def load_paired_images(real_root: Path, fake_root: Path, image_size: int):
    real_by_name = {path.name: path for path in real_root.rglob("*") if path.is_file()}
    fake_paths = sorted(path for path in fake_root.rglob("*") if path.is_file())
    matched_names = [path.name for path in fake_paths if path.name in real_by_name]
    if not matched_names:
        raise FileNotFoundError(f"No matching images found between {real_root} and {fake_root}")
    real_images = torch.stack([load_image_tensor(real_by_name[name], image_size) for name in matched_names])
    fake_images = torch.stack([load_image_tensor(next(path for path in fake_paths if path.name == name), image_size) for name in matched_names])
    return real_images, fake_images, matched_names


def ssim_metric(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    resize = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_gray = rgb2gray(resize(real_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(resize(fake_images).permute((0, 2, 3, 1)).cpu())
    scores = []
    for real, recon in zip(img_gray, recon_gray):
        scores.append(
            structural_similarity(
                recon,
                real,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )
    return float(np.mean(scores))


@torch.no_grad()
def clip_two_way_metric(real_images, fake_images, device, clip_model_name, clip_pretrained):
    model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained, device=device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    pred = model.encode_image(torch.stack([preprocess(img) for img in fake_images], dim=0).to(device))
    real = model.encode_image(torch.stack([preprocess(img) for img in real_images], dim=0).to(device))
    pred = pred.float().flatten(1).cpu().numpy()
    real = real.float().flatten(1).cpu().numpy()
    correlations = np.corrcoef(real, pred)
    correlations = correlations[: len(real_images), len(real_images):]
    congruent = np.diag(correlations)
    success = correlations < congruent
    success_cnt = np.sum(success, axis=0)
    return float(np.mean(success_cnt) / (len(real_images) - 1))


def main():
    parser = argparse.ArgumentParser(description="Evaluate task2 reconstructions with course-style SSIM + CLIP.")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--clip-model-name", type=str, default=None)
    parser.add_argument("--clip-pretrained", type=str, default=None)
    args = parser.parse_args()

    clip_model_name, clip_pretrained = resolve_clip_weights(args.clip_model_name, args.clip_pretrained)
    real_images, fake_images, matched_names = load_paired_images(args.real_root, args.fake_root, args.image_size)
    device = torch.device(args.device)
    payload = {
        "eval_ssim": ssim_metric(real_images, fake_images),
        "eval_clip": clip_two_way_metric(real_images, fake_images, device, clip_model_name, clip_pretrained),
        "matched_images": len(matched_names),
        "clip_model_name": clip_model_name,
        "clip_pretrained": clip_pretrained,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
