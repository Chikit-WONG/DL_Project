#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from safe_bpmgd.data.dataset import load_train_dataset
from safe_bpmgd.encoders.eeg_cogcap import SafeBPMGDEEGModel
from safe_bpmgd.prior.prior_diffusion import MLPPriorMapper, prior_mapper_loss
from safe_bpmgd.utils.config import load_config
from safe_bpmgd.utils.io import ensure_dir, write_json
from safe_bpmgd.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_bpmgd.yaml")
    parser.add_argument("--run-name", default="prior_dev")
    parser.add_argument("--mode", choices=["dev", "full_train"], default="dev")
    parser.add_argument("--encoder-ckpt", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    cache_dir = Path(cfg.paths.feature_cache) / ("final_train" if args.mode == "full_train" else "train")
    clip_payload = torch.load(cache_dir / "clip_rn50.pt", map_location="cpu", weights_only=False)
    clip_feats = clip_payload["features"]
    index = {int(src): i for i, src in enumerate(clip_payload["source_indices"])}

    encoder = SafeBPMGDEEGModel(cfg).to(device)
    state = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(state["model"], strict=False)
    encoder.eval().requires_grad_(False)
    prior = MLPPriorMapper(dim=int(cfg.model.semantic_dim)).to(device)
    opt = torch.optim.AdamW(prior.parameters(), lr=float(cfg.prior.lr), weight_decay=float(cfg.optimizer.weight_decay))
    dataset = load_train_dataset(cfg)
    if args.train_limit is not None:
        dataset = Subset(dataset, list(range(min(args.train_limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=args.batch_size or int(cfg.optimizer.batch_size), shuffle=True)
    history = []
    for epoch in range(args.epochs):
        total = 0.0
        steps = 0
        for batch in loader:
            eeg = batch["eeg"].to(device)
            rows = [index[int(src)] for src in batch["source_index"].tolist()]
            target = clip_feats[rows].to(device, non_blocking=True)
            with torch.no_grad():
                z_sem = encoder(eeg)["z_sem"]
            pred = prior(z_sem)
            loss = prior_mapper_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            steps += 1
            if args.max_steps_per_epoch is not None and steps >= args.max_steps_per_epoch:
                break
        row = {"epoch": epoch, "prior_loss": total / max(steps, 1)}
        history.append(row)
        print(row)
    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoints) / args.run_name)
    torch.save({"prior": prior.state_dict(), "cfg": dict(cfg), "history": history}, ckpt_dir / "prior_mapper.pt")
    write_json(history, Path(cfg.paths.outputs) / args.run_name / "prior_history.json")


if __name__ == "__main__":
    main()
