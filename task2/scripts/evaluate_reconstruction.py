"""
Reconstruction evaluator aligned 1:1 with the course's TA notebook
(eeg_project_sample_code.ipynb).

Key differences from the previous version of this file:
  - CLIP is now OpenAI's `clip.load("ViT-L/14")`, NOT open_clip ViT-H-14.
    This is what the course's eval_images() uses; the two backbones produce
    visibly different 2-way identification scores, so reporting numbers from
    open_clip ViT-H-14 would not match the TA's grading run.
  - SSIM keeps the rgb2gray + gaussian_weights + sigma=1.5 setup. The TA's
    `multichannel=True` argument is redundant once the image is converted
    to grayscale (skimage ignores / deprecates it), so behavior matches.
  - SwAV uses torch.hub `facebookresearch/swav:main` resnet50, matching the
    TA notebook. The previous local-file fallback is removed for parity.
  - The CLI is unchanged so the rest of the pipeline (run_full_experiment.sh)
    keeps working. The --clip-model-name / --clip-pretrained flags are now
    accepted but ignored, because the TA evaluator pins the CLIP backbone.

Usage (unchanged):
  python scripts/evaluate_reconstruction.py \
      --real-root /path/to/test_images \
      --fake-root /path/to/generated_image/ssim_all30 \
      --output    /path/to/generated_image/ssim_all30/reconstruction_metrics.json \
      --device cuda --image-size 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy as sp
import torch
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


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )(image.convert("RGB"))
    return tensor


def load_paired_images(
    real_root: Path,
    fake_root: Path,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    real_by_name = {path.name: path for path in real_root.rglob("*") if path.is_file()}
    fake_paths = sorted(path for path in fake_root.rglob("*") if path.is_file())
    matched_names = [path.name for path in fake_paths if path.name in real_by_name]
    if not matched_names:
        raise FileNotFoundError(
            f"No matching reconstructions found between {real_root} and {fake_root}"
        )

    real_images = torch.stack(
        [load_image_tensor(real_by_name[name], image_size) for name in matched_names]
    )
    fake_images = torch.stack(
        [
            load_image_tensor(fake_root / name, image_size)
            if (fake_root / name).exists()
            else load_image_tensor(
                next(path for path in fake_paths if path.name == name), image_size
            )
            for name in matched_names
        ]
    )
    return real_images, fake_images, matched_names


# ---------------------------------------------------------------------------
# Core 2-way identification (copied verbatim from the TA notebook in spirit;
# only minor formatting / typing changes).
# ---------------------------------------------------------------------------

@torch.no_grad()
def two_way_identification(
    all_brain_recons: torch.Tensor,
    all_images: torch.Tensor,
    model,
    preprocess,
    feature_layer: str | None = None,
    device: torch.device = torch.device("cpu"),
) -> float:
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

    correlations = np.corrcoef(reals, preds)
    correlations = correlations[: len(all_images), len(all_images):]
    congruents = np.diag(correlations)
    success = correlations < congruents
    success_cnt = np.sum(success, axis=0)
    return float(np.mean(success_cnt) / (len(all_images) - 1))


# ---------------------------------------------------------------------------
# Per-metric implementations (matched against the TA notebook).
# ---------------------------------------------------------------------------

def pixcorr(all_images: torch.Tensor, all_brain_recons: torch.Tensor) -> float:
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    reals = preprocess(all_images).reshape(len(all_images), -1).cpu()
    fakes = preprocess(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()
    correlations = [
        np.corrcoef(reals[i], fakes[i])[0][1]
        for i in tqdm(range(len(reals)), desc="PixCorr")
    ]
    score = float(np.nanmean(correlations))
    return 0.0 if np.isnan(score) else score


def ssim(all_images: torch.Tensor, all_brain_recons: torch.Tensor) -> float:
    """SSIM matched to the TA notebook.

    The TA passes `multichannel=True` but then converts to grayscale via
    rgb2gray (1 channel), so the flag has no real effect. We reproduce the
    same behavior here without passing `multichannel=True`, to avoid the
    deprecation warning in newer skimage versions.
    """
    preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    img_gray = rgb2gray(preprocess(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0, 2, 3, 1)).cpu())
    scores = []
    for real, recon in tqdm(
        zip(img_gray, recon_gray), total=len(all_images), desc="SSIM"
    ):
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


def alexnet_metric(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        return_nodes=["features.4", "features.11"],
    ).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return (
        two_way_identification(
            all_brain_recons, all_images, model, preprocess, "features.4", device=device
        ),
        two_way_identification(
            all_brain_recons, all_images, model, preprocess, "features.11", device=device
        ),
    )


def inception_metric(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    model = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT),
        return_nodes=["avgpool"],
    ).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return two_way_identification(
        all_brain_recons, all_images, model, preprocess, "avgpool", device=device
    )


def clip_metric(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    """CLIP 2-way identification with OpenAI's ViT-L/14, matching the TA eval.

    This requires the OpenAI CLIP package (NOT open_clip):
        pip install git+https://github.com/openai/CLIP.git

    If the import fails we surface a clear error message so the user knows
    exactly what to install. We deliberately don't fall back to open_clip,
    because mixing backbones makes the score incomparable with the grading run.
    """
    try:
        import clip  # openai/CLIP
    except ImportError as e:
        raise ImportError(
            "OpenAI CLIP is required for the course-aligned evaluator. "
            "Install it with:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "Do NOT confuse this with the `open_clip_torch` package."
        ) from e

    clip_model, _ = clip.load("ViT-L/14", device=device)
    clip_model.eval().requires_grad_(False)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return two_way_identification(
        all_brain_recons,
        all_images,
        clip_model.encode_image,
        preprocess,
        None,
        device=device,
    )


# Optional local weight paths to avoid hitting the HuggingFace hub on the
# cluster. The first path that exists is used; otherwise open_clip falls back
# to its remote pretrained id "laion2b_s32b_b79k".
_VITH14_LOCAL_WEIGHTS = [
    "/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    "/hpc2hdd/home/ckwong627/workdir/models/CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors",
    "/hpc2hdd/home/dsaa2012_042/project/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin",
    "/hpc2hdd/home/dsaa2012_042/project/models/CLIP-ViT-H-14-laion2B-s32B-b79K/model.safetensors",
]


def _resolve_vith14_pretrained() -> str:
    """Pick a local ViT-H-14 weight file if available, else the remote id."""
    from pathlib import Path as _Path
    for candidate in _VITH14_LOCAL_WEIGHTS:
        try:
            if _Path(candidate).exists():
                return candidate
        except OSError:
            continue
    return "laion2b_s32b_b79k"


def clip_metric_vith14(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    """Secondary CLIP score: open_clip ViT-H-14 (LAION-2B) 2-way identification.

    This is NOT used for course grading. Its sole purpose is to make our
    numbers directly comparable to the CogCapPro report (and the upstream
    CognitionCapturer paper), which both evaluate CLIP with ViT-H-14.

    We use open_clip (not openai/CLIP), which is the package that hosts
    ViT-H-14. If open_clip or the weights are unavailable, the caller (see
    eval_images) catches the exception and records None so the rest of the
    pipeline keeps working.
    """
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "open_clip is required for the ViT-H-14 secondary CLIP score. "
            "Install with: pip install open_clip_torch"
        ) from e

    pretrained = _resolve_vith14_pretrained()
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained=pretrained, device=device
    )
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
    return two_way_identification(
        all_brain_recons,
        all_images,
        model.encode_image,
        preprocess,
        None,
        device=device,
    )


def effnet_metric(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    model = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
        return_nodes=["avgpool"],
    ).to(device)
    model.eval().requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    gt = (
        model(preprocess(all_images).to(device))["avgpool"]
        .reshape(len(all_images), -1)
        .cpu()
        .numpy()
    )
    fake = (
        model(preprocess(all_brain_recons).to(device))["avgpool"]
        .reshape(len(all_brain_recons), -1)
        .cpu()
        .numpy()
    )
    return float(
        np.mean(
            [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
        )
    )


def swav_metric(
    all_images: torch.Tensor,
    all_brain_recons: torch.Tensor,
    device: torch.device,
) -> float:
    """SwAV via torch.hub, matching the TA notebook.

    The TA does `torch.hub.load("facebookresearch/swav:main", "resnet50")`.
    First call will download the weights to ~/.cache/torch/hub. If that hub
    fetch fails (offline cluster), fall back to the FAIR pretrained URL.
    """
    try:
        swav = torch.hub.load("facebookresearch/swav:main", "resnet50")
    except Exception:
        # Fallback path used in the original evaluator.
        from torchvision.models import resnet50
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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    gt = (
        model(preprocess(all_images).to(device))["avgpool"]
        .reshape(len(all_images), -1)
        .cpu()
        .numpy()
    )
    fake = (
        model(preprocess(all_brain_recons).to(device))["avgpool"]
        .reshape(len(all_brain_recons), -1)
        .cpu()
        .numpy()
    )
    return float(
        np.mean(
            [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
        )
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def eval_images(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> dict[str, float | None]:
    real_images = real_images.to(device).float()
    fake_images = fake_images.to(device).float()
    alex2, alex5 = alexnet_metric(real_images, fake_images, device=device)

    # Secondary CLIP score for comparison with the original CogCapPro report
    # (which uses ViT-H-14, LAION-2B). We isolate this from the main metric
    # set: if open_clip or the ViT-H-14 weights are unavailable, we record
    # None and keep going rather than tossing out the whole eval run after
    # potentially hours of generation.
    try:
        clip_vith14 = clip_metric_vith14(real_images, fake_images, device=device)
    except Exception as e:
        print(f"[evaluate_reconstruction] WARN: clip_metric_vith14 failed: {e}")
        clip_vith14 = None

    return {
        "eval_pixcorr": pixcorr(real_images, fake_images),
        "eval_ssim": ssim(real_images, fake_images),
        "eval_alex2": alex2,
        "eval_alex5": alex5,
        "eval_inception": inception_metric(real_images, fake_images, device=device),
        "eval_clip": clip_metric(real_images, fake_images, device=device),
        "eval_clip_vith14": clip_vith14,
        "eval_effnet": effnet_metric(real_images, fake_images, device=device),
        "eval_swav": swav_metric(real_images, fake_images, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstructed images with the course's TA metrics."
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    # The two flags below are accepted for backward compatibility with the
    # existing run_full_experiment.sh and grid scripts, but ignored: this
    # evaluator pins CLIP to OpenAI's ViT-L/14 to match the TA notebook.
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default=None,
        help="(deprecated, ignored) CLIP backbone is pinned to ViT-L/14 for course parity.",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=None,
        help="(deprecated, ignored) CLIP backbone is pinned to ViT-L/14 for course parity.",
    )
    args = parser.parse_args()

    if args.clip_model_name or args.clip_pretrained:
        print(
            "[evaluate_reconstruction] Note: --clip-model-name / --clip-pretrained "
            "are ignored. The course evaluator pins CLIP to OpenAI ViT-L/14."
        )

    real_images, fake_images, matched_names = load_paired_images(
        args.real_root, args.fake_root, args.image_size
    )
    metrics = eval_images(
        real_images=real_images,
        fake_images=fake_images,
        device=torch.device(args.device if torch.cuda.is_available() else "cpu"),
    )
    metrics["matched_images"] = len(matched_names)
    metrics["clip_model_name"] = "ViT-L/14 (openai)"
    metrics["clip_vith14_model_name"] = "ViT-H-14 (open_clip, LAION-2B)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
