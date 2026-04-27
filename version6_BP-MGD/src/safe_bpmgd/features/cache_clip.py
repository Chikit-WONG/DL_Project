from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms

from safe_bpmgd.utils.io import ensure_dir


BLUR_LEVELS = ("raw", "blur_sigma_1", "blur_sigma_2", "blur_sigma_4", "blur_sigma_8", "foveated_blur")


def load_openclip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval().requires_grad_(False)
    return model, preprocess


def encode_image_paths(
    image_paths: list[str],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
    image_size: int = 224,
) -> torch.Tensor:
    model, preprocess = load_openclip(model_name, pretrained, device)
    dataset = ImagePathDataset(image_paths, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    chunks = []
    with torch.no_grad():
        for batch in loader:
            images = batch.to(device)
            feats = model.encode_image(images)
            chunks.append(F.normalize(feats.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def encode_multiblur_paths(
    image_paths: list[str],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
    image_size: int = 224,
) -> torch.Tensor:
    model, preprocess = load_openclip(model_name, pretrained, device)
    transform = preprocess
    all_levels = []
    with torch.no_grad():
        for level in BLUR_LEVELS:
            dataset = BlurImagePathDataset(image_paths, transform, level)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            chunks = []
            for batch in loader:
                feats = model.encode_image(batch.to(device))
                chunks.append(F.normalize(feats.float(), dim=-1).cpu())
            all_levels.append(torch.cat(chunks, dim=0))
    return torch.stack(all_levels, dim=1)


def save_feature_payload(path: str | Path, records, tensor: torch.Tensor, **metadata) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "source_indices": [int(rec.source_index) for rec in records],
            "image_ids": [rec.image_id for rec in records],
            "image_paths": [rec.image_path for rec in records],
            "features": tensor.cpu(),
            **metadata,
        },
        path,
    )


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list[str], preprocess) -> None:
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as image:
            return self.preprocess(image.convert("RGB"))


class BlurImagePathDataset(ImagePathDataset):
    def __init__(self, image_paths: list[str], preprocess, blur_level: str) -> None:
        super().__init__(image_paths, preprocess)
        self.blur_level = blur_level

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.image_paths[idx]) as image:
            image = image.convert("RGB")
            image = apply_blur_level(image, self.blur_level)
            return self.preprocess(image)


def apply_blur_level(image: Image.Image, level: str) -> Image.Image:
    if level == "raw":
        return image
    if level == "foveated_blur":
        blurred = image.filter(ImageFilter.GaussianBlur(radius=6))
        w, h = image.size
        box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
        blurred.paste(image.crop(box), box)
        return blurred
    sigma = float(level.rsplit("_", 1)[-1])
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))
