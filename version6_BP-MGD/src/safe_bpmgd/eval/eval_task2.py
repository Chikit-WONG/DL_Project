from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms


def evaluate_reconstruction(real_root: str | Path, fake_root: str | Path, output_path: str | Path, image_size: int = 256) -> dict:
    real_root = Path(real_root)
    fake_root = Path(fake_root)
    real_by_name = {path.name: path for path in real_root.rglob("*") if path.is_file()}
    fake_paths = sorted(path for path in fake_root.rglob("*") if path.is_file())
    names = [path.name for path in fake_paths if path.name in real_by_name]
    if not names:
        raise FileNotFoundError(f"No matched images between {real_root} and {fake_root}")
    real = torch.stack([_load_tensor(real_by_name[name], image_size) for name in names])
    fake = torch.stack([_load_tensor(next(path for path in fake_paths if path.name == name), image_size) for name in names])
    payload = {
        "eval_ssim": ssim_metric(real, fake),
        "eval_clip": clip_metric_if_available(real, fake),
        "matched_images": len(names),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def ssim_metric(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    resize = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    real_gray = rgb2gray(resize(real_images).permute(0, 2, 3, 1).numpy())
    fake_gray = rgb2gray(resize(fake_images).permute(0, 2, 3, 1).numpy())
    scores = [
        structural_similarity(fake, real, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
        for real, fake in zip(real_gray, fake_gray)
    ]
    return float(np.mean(scores))


def clip_metric_if_available(real_images: torch.Tensor, fake_images: torch.Tensor) -> float | None:
    try:
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained="/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
            device=device,
        )
        model.eval().requires_grad_(False)
        preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
            ]
        )
        with torch.no_grad():
            pred = model.encode_image(torch.stack([preprocess(img) for img in fake_images]).to(device)).float().cpu().numpy()
            real = model.encode_image(torch.stack([preprocess(img) for img in real_images]).to(device)).float().cpu().numpy()
        corr = np.corrcoef(real, pred)[: len(real), len(real):]
        success = corr < np.diag(corr)
        return float(np.sum(success, axis=0).mean() / (len(real) - 1))
    except Exception:
        return None


def _load_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])(image.convert("RGB"))
