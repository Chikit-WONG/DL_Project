"""End-to-end evaluation: 200-way retrieval + official reconstruction metrics.

Retrieval is deterministic from a single trained checkpoint, so we report
``mean=top1, std=0`` over 1 seed (running multi-seed retrieval would
require re-training the encoder, which we leave for the score-optimization
phase).  Reconstruction loops over the saved 10-seed image stack and
computes the official ``eval_images`` metrics per seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, Config  # noqa: E402
from data import EEGImageDataset, collate_eeg_batch, load_eeg_dataset  # noqa: E402
from model import UnifiedModel  # noqa: E402
from utils import (  # noqa: E402
    compute_retrieval_metrics,
    eval_images,
    load_real_test_images,
    set_seed,
    setup_logger,
    summarize_metrics_over_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="Trained EEG model checkpoint")
    p.add_argument("--recon_tag", type=str, default=None,
                   help="Tag of the recon_images_<tag>.pt to evaluate (skip recon if absent)")
    p.add_argument("--retrieval_only", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate_retrieval(model: UnifiedModel, cfg: Config, device) -> dict:
    test_ds = load_eeg_dataset(
        data_directory=cfg.data_dir,
        split="test",
        avg_trials=cfg.avg_trials,
        eeg_channel_jsonl=cfg.eeg_channels_jsonl,
    )
    clip_features = torch.load(cfg.cache_dir / "clip_test_features.pt", weights_only=False)
    test_set = EEGImageDataset(test_ds, clip_features, augmentation=None)
    loader = DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_eeg_batch,
    )

    eeg_embs, clip_embs = [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        clip = batch["clip"].to(device)
        emb = model.encode(eeg)
        eeg_embs.append(F.normalize(emb, dim=-1).cpu())
        clip_embs.append(F.normalize(clip, dim=-1).cpu())
    eeg_all = torch.cat(eeg_embs, dim=0)
    clip_all = torch.cat(clip_embs, dim=0)
    logits = eeg_all @ clip_all.T
    return compute_retrieval_metrics(logits)


def evaluate_reconstruction(cfg: Config, recon_tag: str, device) -> dict:
    # Try single merged file first; fall back to per-seed files.
    merged_path = cfg.output_dir / f"recon_images_{recon_tag}.pt"
    if merged_path.exists():
        bundle = torch.load(merged_path, weights_only=False)
        images: torch.Tensor = bundle["images"]
        image_ids = bundle["image_ids"]
        seeds = bundle["seeds"]
        print(f"loaded {merged_path}: shape {tuple(images.shape)}, {len(seeds)} seeds")
    else:
        # Load individual per-seed files and stack them.
        per_seed_files = sorted(
            (cfg.output_dir / f"recon_images_{recon_tag}_s{s:02d}.pt" for s in range(10)),
            key=lambda p: p.name
        )
        available = [p for p in per_seed_files if p.exists()]
        if not available:
            raise FileNotFoundError(
                f"No recon files found for tag '{recon_tag}'. "
                f"Expected {merged_path} or per-seed files like {per_seed_files[0]}."
            )
        print(f"found {len(available)} per-seed files")
        parts, seeds = [], []
        image_ids = None
        for p in available:
            b = torch.load(p, weights_only=False)
            parts.append(b["images"])   # [1, N, 3, H, W]
            seeds.extend(b["seeds"])
            if image_ids is None:
                image_ids = b["image_ids"]
        images = torch.cat(parts, dim=0)   # [n_seeds, N, 3, H, W]
        print(f"stacked: shape {tuple(images.shape)}")

    real = load_real_test_images(image_ids, cfg.test_image_dir, size=cfg.recon_eval_size)
    print(f"loaded {len(real)} real test images, shape {tuple(real.shape)}")

    per_seed_metrics = []
    for s_idx, seed in enumerate(seeds):
        print(f"--- evaluating seed {seed} ---")
        fake = images[s_idx]
        m = eval_images(real_images=real, fake_images=fake, device=device)
        m["seed"] = int(seed)
        per_seed_metrics.append(m)
        print(f"seed {seed}: {m}")

    summary = summarize_metrics_over_seeds(per_seed_metrics)
    return {"per_seed": per_seed_metrics, "summary": summary}


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log_path = cfg.output_dir / f"evaluate_{Path(args.ckpt).stem}.log"
    logger = setup_logger("evaluate", log_path)
    logger.info(f"args: {vars(args)}")

    set_seed(0)
    model = UnifiedModel(cfg, alpha=1.0, beta=1.0, learnable_loss_weights=False).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    logger.info(f"loaded checkpoint {args.ckpt}")

    retrieval_metrics = evaluate_retrieval(model, cfg, device)
    logger.info(f"RETRIEVAL: top1={retrieval_metrics['top1_acc']:.4f} "
                f"top5={retrieval_metrics['top5_acc']:.4f}")

    full_results = {"retrieval": retrieval_metrics}

    if not args.retrieval_only and args.recon_tag is not None:
        recon_results = evaluate_reconstruction(cfg, args.recon_tag, device)
        full_results["reconstruction"] = recon_results
        logger.info("RECONSTRUCTION SUMMARY:")
        for k, v in recon_results["summary"].items():
            logger.info(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    out_path = cfg.output_dir / f"metrics_{Path(args.ckpt).stem}.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
