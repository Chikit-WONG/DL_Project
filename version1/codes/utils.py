"""Utilities used across the project.

The retrieval/reconstruction evaluation functions in this module are
copied verbatim from ``sample_codes/eeg_project_sample_code.ipynb`` so the
TA's official scoring is reproduced exactly. Only ``build_image_id_to_path``
and ``setup_logger`` are project-specific helpers.
"""
from __future__ import annotations

import datetime
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

# Optional: openai/CLIP is required only when computing the eval_clip metric.
try:
    import clip  # type: ignore
except Exception:  # pragma: no cover - clip is installed via pip
    clip = None  # noqa: N816


# ---------------------------------------------------------------------------
# Reproducibility (verbatim from sample_code cell 11)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Retrieval metrics (verbatim from sample_code cell 11)
# ---------------------------------------------------------------------------
def compute_retrieval_metrics(logits: torch.Tensor) -> Dict[str, float]:
    """Top-1 / Top-5 retrieval accuracy from a square similarity matrix."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("Expected a square similarity matrix of shape [N, N].")

    n = logits.shape[0]
    targets = torch.arange(n)

    top1_pred = logits.argmax(dim=1)
    top1_acc = (top1_pred == targets).float().mean().item()

    top5_idx = logits.topk(k=5, dim=1).indices
    top5_acc = (top5_idx == targets[:, None]).any(dim=1).float().mean().item()

    return {"top1_acc": top1_acc, "top5_acc": top5_acc}


def summarize_metrics_over_seeds(metric_list):
    keys = metric_list[0].keys()
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = np.array([m[key] for m in metric_list], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


# ---------------------------------------------------------------------------
# Image-level evaluation helpers (verbatim from sample_code cell 15)
# ---------------------------------------------------------------------------
@torch.no_grad()
def two_way_identification(
    all_brain_recons,
    all_images,
    model,
    preprocess,
    feature_layer=None,
    return_avg=True,
    device: torch.device = torch.device("cpu"),
):
    preds = model(
        torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device)
    )
    reals = model(
        torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device)
    )
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[: len(all_images), len(all_images) :]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images) - 1)
        return perf
    return success_cnt, len(all_images) - 1


def pixcorr(all_images, all_brain_recons):
    preprocess = transforms.Compose(
        [transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)]
    )
    all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
    all_brain_recons_flattened = (
        preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()
    )

    corrsum = 0
    n = min(len(all_images_flattened), len(all_brain_recons_flattened))
    for i in tqdm(range(n)):
        corrsum += np.corrcoef(
            all_images_flattened[i], all_brain_recons_flattened[i]
        )[0][1]
    return corrsum / n


def ssim(all_images, all_brain_recons):
    preprocess = transforms.Compose(
        [transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)]
    )
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu())

    ssim_score = []
    for im, rec in tqdm(zip(img_gray, recon_gray), total=len(all_images)):
        ssim_score.append(
            structural_similarity(
                rec,
                im,
                multichannel=True,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )
        )
    return np.mean(ssim_score)


def alexnet_metric(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    alex_weights = AlexNet_Weights.IMAGENET1K_V1
    alex_model = create_feature_extractor(
        alexnet(weights=alex_weights), return_nodes=["features.4", "features.11"]
    ).to(device)
    alex_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    a2 = np.mean(
        two_way_identification(
            all_brain_recons.float(), all_images, alex_model, preprocess, "features.4", device=device
        )
    )
    a5 = np.mean(
        two_way_identification(
            all_brain_recons.float(), all_images, alex_model, preprocess, "features.11", device=device
        )
    )
    return a2, a5


def inception_metric(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    weights = Inception_V3_Weights.DEFAULT
    inception_model = create_feature_extractor(
        inception_v3(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    inception_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return np.mean(
        two_way_identification(
            all_brain_recons, all_images, inception_model, preprocess, "avgpool", device=device
        )
    )


def clip_metric(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    if clip is None:
        raise RuntimeError("openai/CLIP is not installed; pip install git+https://github.com/openai/CLIP.git")
    clip_model, _ = clip.load("ViT-L/14", device=device)
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return np.mean(
        two_way_identification(
            all_brain_recons, all_images, clip_model.encode_image, preprocess, None, device=device
        )
    )


def effnet_metric(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    weights = EfficientNet_B1_Weights.DEFAULT
    eff_model = create_feature_extractor(
        efficientnet_b1(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    eff_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gt = eff_model(preprocess(all_images))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fake = eff_model(preprocess(all_brain_recons))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()


def swav_metric(all_images, all_brain_recons, device: torch.device = torch.device("cpu")):
    swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    swav_model = create_feature_extractor(swav_model, return_nodes=["avgpool"]).to(device)
    swav_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    gt = swav_model(preprocess(all_images))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fake = swav_model(preprocess(all_brain_recons))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return np.array(
        [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
    ).mean()


def eval_images(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device = torch.device("cpu"),
):
    """Official TA evaluation. Returns a dict of all reconstruction metrics."""
    real_images = real_images.to(device).float()
    fake_images = fake_images.to(device).float()

    pixcorrs = pixcorr(real_images, fake_images)
    ssims = ssim(real_images, fake_images)
    alex2, alex5 = alexnet_metric(real_images, fake_images, device=device)
    inceptions = inception_metric(real_images, fake_images, device=device)
    clips = clip_metric(real_images, fake_images, device=device)
    effnets = effnet_metric(real_images, fake_images, device=device)
    swavs = swav_metric(real_images, fake_images, device=device)

    return {
        "eval_pixcorr": float(pixcorrs),
        "eval_ssim": float(ssims),
        "eval_alex2": float(alex2),
        "eval_alex5": float(alex5),
        "eval_inception": float(inceptions),
        "eval_clip": float(clips),
        "eval_effnet": float(effnets),
        "eval_swav": float(swavs),
    }


# ---------------------------------------------------------------------------
# Project-specific helpers
# ---------------------------------------------------------------------------
def build_image_id_to_path(image_root: Path) -> Dict[str, str]:
    """Walk training_images/ or test_images/ and map ``image_id`` -> file path.

    ``image_id`` is the file stem (e.g. ``aardvark_01b``) which matches what
    ``load_eeg_dataset`` exposes in the ``image_id`` column.
    """
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(image_root)

    mapping: Dict[str, str] = {}
    for category_dir in sorted(image_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                mapping[img_file.stem] = str(img_file)
    return mapping


def load_real_test_images(image_id_order: List[str], image_root: Path, size: int = 256) -> torch.Tensor:
    """Load real test images in the given image_id order, resized to ``size``."""
    id2path = build_image_id_to_path(image_root)
    tx = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0, 1] float
        ]
    )
    out = []
    for img_id in image_id_order:
        img = Image.open(id2path[img_id]).convert("RGB")
        out.append(tx(img))
    return torch.stack(out, dim=0)


def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
