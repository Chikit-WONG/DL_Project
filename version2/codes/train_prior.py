from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from config import DEFAULT_CONFIG
from data import build_dataloader
from model import EEGEncoderV2, PriorUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_ckpt", required=True, type=str)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit_train", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs if args.epochs is not None else cfg.prior_epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.prior_batch_size

    encoder_state = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder = EEGEncoderV2(cfg).to(device)
    encoder.load_state_dict(encoder_state["model"], strict=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    train_loader = build_dataloader(cfg, "train", batch_size=batch_size, shuffle=True, limit=args.limit_train)
    cache = torch.load(cfg.cache_dir / "backbone_cache_train.pt", map_location="cpu", weights_only=False)
    prior = PriorUNet(embed_dim=cfg.semantic_dim, hidden_dim=cfg.prior_hidden_dim).to(device)
    optimizer = AdamW(prior.parameters(), lr=cfg.prior_lr, weight_decay=cfg.weight_decay)

    alphas = torch.linspace(1e-4, 0.02, cfg.prior_timesteps, device=device)
    alpha_bars = torch.cumprod(1.0 - alphas, dim=0)
    history = []
    best_cos = -1.0

    for epoch in range(epochs):
        prior.train()
        losses = []
        cosines = []
        pbar = tqdm(train_loader, desc=f"prior {epoch + 1}/{epochs}")
        for batch in pbar:
            idx = batch["index"]
            eeg = batch["eeg"].to(device)
            subject_ids = batch["subject_id"].to(device)
            target = cache["features"]["h14"][idx].to(device)
            with torch.no_grad():
                cond = encoder(eeg, subject_ids=subject_ids)["semantic"]

            t = torch.randint(0, cfg.prior_timesteps, (eeg.size(0),), device=device)
            noise = torch.randn_like(target)
            alpha_bar = alpha_bars[t].unsqueeze(1)
            noisy_target = alpha_bar.sqrt() * target + (1.0 - alpha_bar).sqrt() * noise

            drop_mask = (torch.rand(cond.size(0), device=device) < 0.1).float().unsqueeze(1)
            cond_input = cond * (1.0 - drop_mask)
            pred_noise = prior(noisy_target, t, cond_input)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_clean = (noisy_target - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-6)
                cosines.append(cosine_mean(pred_clean, target))
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", cos=f"{cosines[-1]:.4f}")

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "cosine": float(np.mean(cosines)) if cosines else 0.0,
        }
        history.append(row)
        print(json.dumps(row))

        if row["cosine"] >= best_cos:
            best_cos = row["cosine"]
            torch.save(
                {
                    "model": prior.state_dict(),
                    "tag": args.tag,
                    "epoch": epoch,
                    "best_cosine": best_cos,
                },
                cfg.ckpt_dir / f"{args.tag}_best.pt",
            )

    torch.save(
        {"model": prior.state_dict(), "tag": args.tag, "epoch": epochs - 1, "best_cosine": best_cos},
        cfg.ckpt_dir / f"{args.tag}_last.pt",
    )
    with (cfg.result_dir / f"history_{args.tag}.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best_cosine": best_cos}, handle, indent=2)


if __name__ == "__main__":
    main()
