"""Training script for the unified EEG-to-CLIP model.

Supports phase-1 / phase-2 training, optional learnable loss weights, and
running architecture-B baselines as the special cases ``--alpha 1 --beta 0``
or ``--alpha 0 --beta 1`` — no second codebase needed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, Config  # noqa: E402
from data import EEGAugmentation, EEGImageDataset, collate_eeg_batch, load_eeg_dataset  # noqa: E402
from model import UnifiedModel, count_parameters  # noqa: E402
from utils import compute_retrieval_metrics, set_seed, setup_logger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.add_argument("--alpha", type=float, default=None,
                   help="InfoNCE loss weight (overrides phase default)")
    p.add_argument("--beta", type=float, default=None,
                   help="MSE loss weight (overrides phase default)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override phase epoch count")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--learnable_weights", action="store_true",
                   help="Make alpha/beta nn.Parameters")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from (model only)")
    p.add_argument("--tag", type=str, default=None,
                   help="Filename suffix to disambiguate experiments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--light_aug", action="store_true",
                   help="Use the lighter augmentation set (jitter + noise only)")
    p.add_argument("--num_eeg_timesteps", type=int, default=None)
    return p.parse_args()


def make_dataloader(cfg: Config, batch_size: int, augment: bool, light_aug: bool):
    train_ds = load_eeg_dataset(
        data_directory=cfg.data_dir,
        split="train",
        avg_trials=cfg.avg_trials,
        eeg_channel_jsonl=cfg.eeg_channels_jsonl,
    )
    test_ds = load_eeg_dataset(
        data_directory=cfg.data_dir,
        split="test",
        avg_trials=cfg.avg_trials,
        eeg_channel_jsonl=cfg.eeg_channels_jsonl,
    )

    train_features = torch.load(cfg.cache_dir / "clip_train_features.pt", weights_only=False)
    test_features = torch.load(cfg.cache_dir / "clip_test_features.pt", weights_only=False)

    aug = EEGAugmentation(
        jitter=cfg.aug_jitter,
        channel_dropout=cfg.aug_channel_dropout,
        noise_std=cfg.aug_noise_std,
        time_mask_steps=cfg.aug_time_mask_steps,
        amp_scale_range=cfg.aug_amp_scale_range,
        apply_prob=cfg.aug_apply_prob,
        light=light_aug,
    ) if augment else None

    train_set = EEGImageDataset(train_ds, train_features, augmentation=aug)
    test_set = EEGImageDataset(test_ds, test_features, augmentation=None)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_eeg_batch,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_eeg_batch,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, test_loader, train_set, test_set


@torch.no_grad()
def evaluate_retrieval(model, test_loader, device):
    model.eval()
    eeg_embs, clip_embs = [], []
    for batch in test_loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        clip = batch["clip"].to(device, non_blocking=True)
        e = model.encode(eeg)
        eeg_embs.append(F.normalize(e, dim=-1).cpu())
        clip_embs.append(F.normalize(clip, dim=-1).cpu())
    eeg_all = torch.cat(eeg_embs, dim=0)
    clip_all = torch.cat(clip_embs, dim=0)
    logits = eeg_all @ clip_all.T
    return compute_retrieval_metrics(logits)


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    if args.num_eeg_timesteps is not None:
        cfg.num_eeg_timesteps = args.num_eeg_timesteps

    set_seed(args.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Phase defaults
    if args.phase == 1:
        epochs = args.epochs or cfg.phase1_epochs
        batch_size = args.batch_size or cfg.phase1_batch_size
        lr = args.lr or cfg.phase1_lr
        wd = cfg.phase1_weight_decay
        alpha = args.alpha if args.alpha is not None else cfg.phase1_alpha
        beta = args.beta if args.beta is not None else cfg.phase1_beta
        light_aug = args.light_aug
    else:
        epochs = args.epochs or cfg.phase2_epochs
        batch_size = args.batch_size or cfg.phase2_batch_size
        lr = args.lr or cfg.phase2_lr
        wd = cfg.phase2_weight_decay
        alpha = args.alpha if args.alpha is not None else cfg.phase2_alpha
        beta = args.beta if args.beta is not None else cfg.phase2_beta
        light_aug = True if not args.light_aug else args.light_aug
        # Phase 2 always uses lighter aug unless overridden by config
        if args.light_aug is False:
            light_aug = True

    tag = args.tag or f"phase{args.phase}_a{alpha}_b{beta}"
    if args.learnable_weights:
        tag += "_learnable"

    log_path = cfg.output_dir / f"train_{tag}.log"
    logger = setup_logger(f"train_{tag}", log_path)
    logger.info(f"args: {vars(args)}")
    logger.info(f"phase={args.phase} epochs={epochs} bs={batch_size} lr={lr} "
                f"alpha={alpha} beta={beta} learnable={args.learnable_weights} "
                f"light_aug={light_aug}")

    # Data — try the configured num_eeg_timesteps; if mismatch, recreate model with the actual T
    train_loader, test_loader, train_set, test_set = make_dataloader(
        cfg, batch_size=batch_size, augment=True, light_aug=light_aug
    )
    sample_eeg = train_set[0]["eeg"]
    actual_T = sample_eeg.shape[-1]
    if actual_T != cfg.num_eeg_timesteps:
        logger.info(f"updating num_eeg_timesteps {cfg.num_eeg_timesteps} -> {actual_T}")
        cfg.num_eeg_timesteps = actual_T

    logger.info(f"train samples: {len(train_set)}  test samples: {len(test_set)}")
    logger.info(f"EEG shape: [C={sample_eeg.shape[0]}, T={actual_T}]")
    logger.info(f"CLIP target dim: {train_set[0]['clip'].shape[0]}")

    model = UnifiedModel(
        cfg,
        alpha=alpha,
        beta=beta,
        learnable_loss_weights=args.learnable_weights,
    ).to(device)
    logger.info(f"model parameters: {count_parameters(model):,}")

    if args.resume is not None:
        ckpt_path = Path(args.resume)
        logger.info(f"resuming from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Allow loose matching so changing alpha/beta does not break loading.
        model.load_state_dict(state["model"], strict=False)
        # Re-apply requested alpha/beta even if checkpoint had different ones.
        model.set_weights(alpha, beta)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_top1 = -1.0
    best_path = cfg.ckpt_dir / f"{tag}_best.pt"
    final_path = cfg.ckpt_dir / f"{tag}_final.pt"

    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"loss": 0.0, "ret": 0.0, "rec": 0.0, "n": 0}
        for batch in train_loader:
            eeg = batch["eeg"].to(device, non_blocking=True)
            clip = batch["clip"].to(device, non_blocking=True)
            emb = model.encode(eeg)
            loss, l_ret, l_rec, a_used, b_used = model.compute_loss(emb, clip)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = eeg.size(0)
            running["loss"] += loss.item() * bs
            running["ret"] += l_ret.item() * bs
            running["rec"] += l_rec.item() * bs
            running["n"] += bs

        scheduler.step()
        ret_metrics = evaluate_retrieval(model, test_loader, device)
        a_used, b_used = model.get_weights()
        msg = (
            f"epoch {epoch:03d}/{epochs} "
            f"loss={running['loss']/running['n']:.4f} "
            f"ret={running['ret']/running['n']:.4f} "
            f"rec={running['rec']/running['n']:.4f} "
            f"alpha={a_used.item():.3f} beta={b_used.item():.3f} "
            f"top1={ret_metrics['top1_acc']:.4f} top5={ret_metrics['top5_acc']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        logger.info(msg)

        history.append({
            "epoch": epoch,
            "loss": running["loss"] / running["n"],
            "l_ret": running["ret"] / running["n"],
            "l_rec": running["rec"] / running["n"],
            "alpha": float(a_used.item()),
            "beta": float(b_used.item()),
            "top1": ret_metrics["top1_acc"],
            "top5": ret_metrics["top5_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if ret_metrics["top1_acc"] > best_top1:
            best_top1 = ret_metrics["top1_acc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": ret_metrics},
                best_path,
            )

    torch.save(
        {"model": model.state_dict(), "epoch": epochs, "metrics": ret_metrics},
        final_path,
    )

    elapsed = time.time() - t_start
    logger.info(f"done in {elapsed/60:.1f} min   best top1={best_top1:.4f}")
    logger.info(f"best ckpt: {best_path}")
    logger.info(f"final ckpt: {final_path}")

    history_path = cfg.output_dir / f"history_{tag}.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "history": history, "best_top1": best_top1}, f, indent=2)


if __name__ == "__main__":
    main()
