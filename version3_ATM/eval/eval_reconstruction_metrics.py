"""
Official reconstruction evaluation for the course project.

Metrics (from Project1.pdf + sample code):
  - SSIM   (structural similarity)
  - CLIP Score (two-way identification using CLIP ViT-L/14)
  - Additionally: PixCorr, AlexNet2, AlexNet5, Inception, EffNet, SwAV

Uses the official eval_images() implementation from the sample notebook.

Usage:
  python eval/eval_reconstruction_metrics.py \
    --tensors_path ./outputs/reconstructions/<run>/recon_tensors.pt \
    --output_csv   ./outputs/reconstruction_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import scipy as sp
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from tqdm import tqdm
import clip


# ──────────────────────────────────────────────────────────────────────────────
# Official evaluation functions (from sample notebook eeg_project_sample_code_done.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess,
                            feature_layer=None, return_avg=True, device=torch.device("cpu")):
    preds = model(torch.stack([preprocess(r) for r in all_brain_recons]).to(device))
    reals = model(torch.stack([preprocess(i) for i in all_images]).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()
    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)
    success = r < congruents
    success_cnt = np.sum(success, 0)
    if return_avg:
        return np.mean(success_cnt) / (len(all_images) - 1)
    return success_cnt, len(all_images) - 1


def pixcorr(all_images, all_brain_recons):
    pre = transforms.Compose([transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)])
    imgs_flat = pre(all_images).reshape(len(all_images), -1).cpu()
    recon_flat = pre(all_brain_recons).reshape(len(all_brain_recons), -1).cpu()
    n = min(len(imgs_flat), len(recon_flat))
    corrsum = sum(np.corrcoef(imgs_flat[i].numpy(), recon_flat[i].numpy())[0][1] for i in range(n))
    return corrsum / n


def ssim_metric(all_images, all_brain_recons):
    pre = transforms.Compose([transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)])
    img_gray = rgb2gray(pre(all_images).permute(0, 2, 3, 1).cpu().numpy())
    recon_gray = rgb2gray(pre(all_brain_recons).permute(0, 2, 3, 1).cpu().numpy())
    scores = []
    for im, rec in zip(img_gray, recon_gray):
        scores.append(structural_similarity(
            rec, im, multichannel=True,
            gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, data_range=1.0))
    return float(np.mean(scores))


def alexnet_metric(all_images, all_brain_recons, device):
    from torchvision.models import alexnet, AlexNet_Weights
    alex = create_feature_extractor(
        alexnet(weights=AlexNet_Weights.IMAGENET1K_V1),
        return_nodes=["features.4", "features.11"]).to(device)
    alex.eval().requires_grad_(False)
    pre = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    alex2 = two_way_identification(all_brain_recons, all_images, alex, pre, "features.4", device=device)
    alex5 = two_way_identification(all_brain_recons, all_images, alex, pre, "features.11", device=device)
    return float(np.mean(alex2)), float(np.mean(alex5))


def inception_metric(all_images, all_brain_recons, device):
    inc = create_feature_extractor(
        inception_v3(weights=Inception_V3_Weights.DEFAULT),
        return_nodes=["avgpool"]).to(device)
    inc.eval().requires_grad_(False)
    pre = transforms.Compose([
        transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return float(np.mean(two_way_identification(all_brain_recons, all_images, inc, pre, "avgpool", device=device)))


def clip_metric(all_images, all_brain_recons, device):
    clip_model, _ = clip.load("ViT-L/14", device=device)
    pre = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711])])
    return float(np.mean(two_way_identification(
        all_brain_recons, all_images, clip_model.encode_image, pre, None, device=device)))


def effnet_metric(all_images, all_brain_recons, device):
    eff = create_feature_extractor(
        efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
        return_nodes=["avgpool"]).to(device)
    eff.eval().requires_grad_(False)
    pre = transforms.Compose([
        transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    gt = eff(pre(all_images))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fk = eff(pre(all_brain_recons))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return float(np.mean([sp.spatial.distance.correlation(gt[i], fk[i]) for i in range(len(gt))]))


def swav_metric(all_images, all_brain_recons, device):
    swav = create_feature_extractor(
        torch.hub.load("facebookresearch/swav:main", "resnet50"),
        return_nodes=["avgpool"]).to(device)
    swav.eval().requires_grad_(False)
    pre = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    gt = swav(pre(all_images))["avgpool"].reshape(len(all_images), -1).cpu().numpy()
    fk = swav(pre(all_brain_recons))["avgpool"].reshape(len(all_brain_recons), -1).cpu().numpy()
    return float(np.mean([sp.spatial.distance.correlation(gt[i], fk[i]) for i in range(len(gt))]))


def eval_images(real_images: torch.Tensor, fake_images: torch.Tensor,
                device: torch.device = torch.device("cpu")):
    """Official evaluation function from sample notebook."""
    real_images = real_images.to(device).float()
    fake_images = fake_images.to(device).float()

    print("  Computing PixCorr …")
    pc = pixcorr(real_images, fake_images)
    print("  Computing SSIM …")
    ss = ssim_metric(real_images, fake_images)
    print("  Computing AlexNet …")
    a2, a5 = alexnet_metric(real_images, fake_images, device)
    print("  Computing Inception …")
    inc = inception_metric(real_images, fake_images, device)
    print("  Computing CLIP …")
    cl = clip_metric(real_images, fake_images, device)
    print("  Computing EffNet …")
    en = effnet_metric(real_images, fake_images, device)
    print("  Computing SwAV …")
    sw = swav_metric(real_images, fake_images, device)

    return {
        "eval_pixcorr": pc,
        "eval_ssim": ss,
        "eval_alex2": a2,
        "eval_alex5": a5,
        "eval_inception": inc,
        "eval_clip": cl,
        "eval_effnet": en,
        "eval_swav": sw,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device)
    tensors_path = Path(args.tensors_path)

    print(f"Loading tensors from {tensors_path} …")
    saved = torch.load(tensors_path, map_location="cpu")
    real_images = saved["real"].float()   # [200, 3, 256, 256]  in [0,1]
    fake_images = saved["fake"].float()  # [200, 3, 256, 256]  in [0,1]
    print(f"  real: {real_images.shape}, fake: {fake_images.shape}")

    # Run over args.num_seeds seeds (sample different subsets, if desired)
    # For deterministic generation (one set of 200 images), all seeds give the same result.
    seed_results = []
    for seed in range(args.num_seeds):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        metrics = eval_images(real_images, fake_images, device)
        metrics["seed"] = seed
        seed_results.append(metrics)
        print(f"  SSIM  : {metrics['eval_ssim']:.4f}")
        print(f"  CLIP  : {metrics['eval_clip']:.4f}")
        print(f"  PixCorr: {metrics['eval_pixcorr']:.4f}")

    # Summarize
    metric_keys = [k for k in seed_results[0] if k != "seed"]
    print(f"\n{'='*60}")
    print(f"Reconstruction Evaluation over {args.num_seeds} seeds")
    print(f"{'='*60}")
    summaries = {}
    for key in metric_keys:
        vals = np.array([r[key] for r in seed_results])
        mean, std = vals.mean(), vals.std(ddof=1)
        summaries[key] = (mean, std)
        print(f"  {key:<20s}: {mean:.4f} ± {std:.4f}")

    # Save CSV
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["seed"] + metric_keys
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in seed_results:
                writer.writerow(r)
            mean_row = {"seed": "mean", **{k: summaries[k][0] for k in metric_keys}}
            std_row = {"seed": "std", **{k: summaries[k][1] for k in metric_keys}}
            writer.writerow(mean_row)
            writer.writerow(std_row)
        print(f"\nResults saved to {out}")

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors_path", required=True,
                        help="Path to recon_tensors.pt (output of generate_reconstructions.py)")
    parser.add_argument("--output_csv",
                        default="./outputs/reconstruction_eval.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds (1 is sufficient for deterministic generation)")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
