#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from safe_bpmgd.data.dataset import load_train_dataset
from safe_bpmgd.data.splits import make_train_val_indices, save_split_indices
from safe_bpmgd.encoders.eeg_cogcap import SafeBPMGDEEGModel
from safe_bpmgd.losses.contrastive import clip_contrastive_loss, cosine_loss
from safe_bpmgd.losses.multiblur_loss import multiblur_alignment_loss
from safe_bpmgd.losses.structure_loss import smooth_l1_cosine
from safe_bpmgd.utils.config import load_config, save_config, to_plain_data
from safe_bpmgd.utils.io import ensure_dir, write_json
from safe_bpmgd.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_bpmgd.yaml")
    parser.add_argument("--run-name", default="encoder_dev")
    parser.add_argument("--mode", choices=["dev", "full_train"], default="dev")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    run_dir = ensure_dir(Path(cfg.paths.outputs) / args.run_name)
    ckpt_dir = ensure_dir(Path(cfg.paths.checkpoints) / args.run_name)
    save_config(cfg, run_dir / "selected_config.yaml")

    dataset = load_train_dataset(cfg)
    if args.mode == "dev":
        if args.train_limit is not None or args.val_limit is not None:
            val_take = args.val_limit or max(1, int(round(float(cfg.data.val_fraction) * (args.train_limit or len(dataset)))))
            train_take = args.train_limit or max(1, len(dataset) - val_take)
            pool_size = min(len(dataset), train_take + val_take)
            train_local, val_local = make_train_val_indices(pool_size, val_take / max(pool_size, 1), int(cfg.seed))
            train_idx = train_local[:train_take]
            val_idx = val_local[:val_take]
        else:
            train_idx, val_idx = make_train_val_indices(len(dataset), float(cfg.data.val_fraction), int(cfg.seed))
    else:
        if args.train_limit is not None:
            train_idx = list(range(min(args.train_limit, len(dataset))))
        else:
            train_idx = list(range(len(dataset)))
        val_idx = []
    save_split_indices(train_idx, val_idx, run_dir / "val_indices.json")
    batch_size = args.batch_size or int(cfg.optimizer.batch_size)
    train_subset = Subset(dataset, train_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=len(train_subset) >= batch_size)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False) if val_idx else None

    cache_dir = Path(cfg.paths.feature_cache) / ("final_train" if args.mode == "full_train" else "train")
    targets = FeatureTargets(cache_dir, device)
    model = SafeBPMGDEEGModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.optimizer.lr), weight_decay=float(cfg.optimizer.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.optimizer.mixed_precision) and device.type == "cuda")
    max_epochs = args.max_epochs or int(cfg.optimizer.epochs)

    best_val = -1e9
    history = []
    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            eeg = batch["eeg"].to(device)
            source = batch["source_index"].tolist()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(eeg)
                clip_target = targets.lookup("clip", source)
                loss = float(cfg.loss.lambda_semantic) * clip_contrastive_loss(out["z_sem"], clip_target)
                if targets.has("multiblur"):
                    loss = loss + float(cfg.loss.lambda_blur) * multiblur_alignment_loss(
                        out["z_sem"], out["z_blur_logits"], targets.lookup("multiblur", source)
                    )
                if targets.has("evnet"):
                    loss = loss + float(cfg.loss.lambda_struct) * smooth_l1_cosine(out["z_struct"], targets.lookup("evnet", source))
                if targets.has("vae"):
                    loss = loss + float(cfg.loss.lambda_vae) * F.smooth_l1_loss(out["z_vae"], targets.lookup("vae", source))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu())
            steps += 1
            if args.max_steps_per_epoch is not None and steps >= args.max_steps_per_epoch:
                break
        metrics = {"epoch": epoch, "train_loss": total / max(steps, 1)}
        if val_loader is not None:
            metrics.update(evaluate(model, val_loader, targets, device))
        history.append(metrics)
        score = metrics.get("val_semantic_cosine", -metrics["train_loss"])
        torch.save({"model": model.state_dict(), "cfg": to_plain_data(cfg), "metrics": metrics}, ckpt_dir / "encoder_last.pt")
        if score > best_val:
            best_val = score
            torch.save({"model": model.state_dict(), "cfg": to_plain_data(cfg), "metrics": metrics}, ckpt_dir / "encoder_best_val.pt")
        print(metrics)
    final_name = "encoder_final_fulltrain.pt" if args.mode == "full_train" else "encoder_best_val.pt"
    if args.mode == "full_train":
        torch.save({"model": model.state_dict(), "cfg": to_plain_data(cfg), "metrics": history[-1]}, ckpt_dir / final_name)
    write_json(history, run_dir / "encoder_history.json")


@torch.no_grad()
def evaluate(model, loader, targets, device) -> dict:
    model.eval()
    cosines = []
    top1 = 0
    top5 = 0
    total = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        source = batch["source_index"].tolist()
        out = model(eeg)
        target = F.normalize(targets.lookup("clip", source), dim=-1)
        pred = F.normalize(out["z_sem"], dim=-1)
        cosines.extend(F.cosine_similarity(pred, target, dim=-1).detach().cpu().tolist())
        sim = pred @ target.T
        k = min(5, sim.shape[1])
        inds = sim.topk(k, dim=-1).indices
        labels = torch.arange(sim.shape[0], device=device)[:, None]
        top1 += (inds[:, :1] == labels).sum().item()
        top5 += (inds == labels).any(dim=1).sum().item()
        total += sim.shape[0]
    return {"val_semantic_cosine": sum(cosines) / max(len(cosines), 1), "val_top1_inbatch": top1 / max(total, 1), "val_top5_inbatch": top5 / max(total, 1)}


class FeatureTargets:
    def __init__(self, cache_dir: Path, device) -> None:
        self.payloads = {}
        self.index = {}
        self.device = device
        names = {"clip": "clip_rn50.pt", "multiblur": "multiblur_clip.pt", "evnet": "evnet_struct.pt", "vae": "vae_latents.pt"}
        for key, name in names.items():
            path = cache_dir / name
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.payloads[key] = payload["features"]
                self.index[key] = {int(src): i for i, src in enumerate(payload["source_indices"])}
        if "clip" not in self.payloads:
            raise FileNotFoundError(f"Missing CLIP cache in {cache_dir}; run scripts/01_cache_train_features.py first")

    def has(self, key: str) -> bool:
        return key in self.payloads

    def lookup(self, key: str, source_indices: list[int]) -> torch.Tensor:
        rows = [self.index[key][int(src)] for src in source_indices]
        return self.payloads[key][rows].to(self.device, non_blocking=True)


if __name__ == "__main__":
    main()
