from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
from PIL import Image
import scipy as sp
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import torch
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
    resnet50,
)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm


DEFAULT_CLIP_CANDIDATES: list = []  # pass --clip-pretrained explicitly


def resolve_clip_weights(clip_model_name: str, clip_pretrained: str | None) -> tuple[str, str]:
    if clip_pretrained is not None:
        requested_path = Path(clip_pretrained)
        if requested_path.exists():
            return clip_model_name, str(requested_path)

    for candidate_model_name, candidate_path in DEFAULT_CLIP_CANDIDATES:
        if Path(candidate_path).exists():
            return candidate_model_name, candidate_path

    raise FileNotFoundError(
        "No CLIP checkpoint found. Pass --clip-pretrained /path/to/open_clip_pytorch_model.bin"
    )


def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )(image.convert("RGB"))
    return tensor


def load_paired_images(real_root: Path, fake_root: Path, image_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    real_by_name = {path.name: path for path in real_root.rglob("*") if path.is_file()}
    fake_paths = sorted(path for path in fake_root.rglob("*") if path.is_file())
    matched_names = [path.name for path in fake_paths if path.name in real_by_name]
    if not matched_names:
        raise FileNotFoundError(f"No matching reconstructions found between {real_root} and {fake_root}")

    real_images = torch.stack([load_image_tensor(real_by_name[name], image_size) for name in matched_names])
    fake_images = torch.stack([load_image_tensor(fake_root / name, image_size) if (fake_root / name).exists() else load_image_tensor(next(path for path in fake_paths if path.name == name), image_size) for name in matched_names])
    return real_images, fake_images, matched_names


@torch.no_grad()
def two_way_identification(
    all_brain_recons: torch.Tensor,
    all_images: torch.Tensor,
    model,
    preprocess,
    feature_layer: str | None = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    correlations = np.corrcoef(reals, preds)
    correlations = correlations[: len(all_images), len(all_images) :]
    congruents = np.diag(correlations)
    success = correlations < congruents
    success_cnt = np.sum(success, axis=0)
    return float(np.mean(success_cnt) / (len(all_images) - 1))


def pixcorr(all_images: torch.Tensor, all_brain_recons: torch.Tensor) -> float:
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    reals = preprocess(all_images).reshape(len(all_images), -1).cpu()
    fakes = preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()
    correlations = [np.corrcoef(reals[i], fakes[i])[0][1] for i in tqdm(range(len(reals)), desc="PixCorr")]
    score = float(np.nanmean(correlations))
    return 0.0 if np.isnan(score) else score


def ssim(all_images: torch.Tensor, all_brain_recons: torch.Tensor) -> float:
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu())
    scores = []
    for real, recon in tqdm(zip(img_gray, recon_gray), total=len(all_images), desc="SSIM"):
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


def alexnet_metric(all_images: torch.Tensor, all_brain_recons: torch.Tensor, device: torch.device) -> tuple[float, float]:
    model = create_feature_extractor(alexnet(weights=AlexNet_Weights.IMAGENET1K_V1), return_nodes=["features.4", "features.11"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return (
        two_way_identification(all_brain_recons, all_images, model, preprocess, "features.4", device=device),
        two_way_identification(all_brain_recons, all_images, model, preprocess, "features.11", device=device),
    )


def inception_metric(all_images: torch.Tensor, all_brain_recons: torch.Tensor, device: torch.device) -> float:
    model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return two_way_identification(all_brain_recons, all_images, model, preprocess, "avgpool", device=device)


def clip_metric(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
) -> float:
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
    return two_way_identification(all_brain_recons, all_images, model.encode_image, preprocess, None, device=device)


def effnet_metric(all_images: torch.Tensor, all_brain_recons: torch.Tensor, device: torch.device) -> float:
    model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gt = model(preprocess(all_images).to(device))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fake = model(preprocess(all_brain_recons).to(device))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return float(np.mean([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]))


def swav_metric(all_images: torch.Tensor, all_brain_recons: torch.Tensor, device: torch.device) -> float:
    swav = resnet50(weights=None)
    state_dict = torch.hub.load_state_dict_from_url(
        url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
        map_location="cpu",
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    swav.load_state_dict(state_dict, strict=False)
    model = create_feature_extractor(swav, return_nodes=["avgpool"]).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gt = model(preprocess(all_images).to(device))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fake = model(preprocess(all_brain_recons).to(device))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return float(np.mean([sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]))


def eval_images(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
) -> dict[str, float]:
    real_images = real_images.to(device).float()
    fake_images = fake_images.to(device).float()
    alex2, alex5 = alexnet_metric(real_images, fake_images, device=device)
    return {
        "eval_pixcorr": pixcorr(real_images, fake_images),
        "eval_ssim": ssim(real_images, fake_images),
        "eval_alex2": alex2,
        "eval_alex5": alex5,
        "eval_inception": inception_metric(real_images, fake_images, device=device),
        "eval_clip": clip_metric(real_images, fake_images, device=device, clip_model_name=clip_model_name, clip_pretrained=clip_pretrained),
        "eval_effnet": effnet_metric(real_images, fake_images, device=device),
        "eval_swav": swav_metric(real_images, fake_images, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed images with the course metrics.")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--clip-model-name", type=str, default="ViT-H-14")
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=None,
        help="Path to open_clip_pytorch_model.bin for the CLIP model used in scoring",
    )
    args = parser.parse_args()

    resolved_clip_model_name, resolved_clip_pretrained = resolve_clip_weights(
        args.clip_model_name,
        args.clip_pretrained,
    )
    real_images, fake_images, matched_names = load_paired_images(args.real_root, args.fake_root, args.image_size)
    metrics = eval_images(
        real_images=real_images,
        fake_images=fake_images,
        device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
        clip_model_name=resolved_clip_model_name,
        clip_pretrained=resolved_clip_pretrained,
    )
    metrics["matched_images"] = len(matched_names)
    metrics["clip_model_name"] = resolved_clip_model_name
    metrics["clip_pretrained"] = resolved_clip_pretrained
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
