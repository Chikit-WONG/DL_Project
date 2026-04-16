from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from config import DEFAULT_CONFIG
from data import build_dataloader
from model import EEGEncoderV2, hard_negative_infonce, info_nce_loss, supervised_contrastive_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["warmup", "multitarget", "finetune"], required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(cfg, split: str) -> dict:
    path = cfg.cache_dir / f"backbone_cache_{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache file: {path}. Run cache_backbone_features.py before training."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


def ensure_alignment(loader, cache_payload: dict) -> None:
    dataset_ids = [sample["image_id"] for sample in loader.dataset]
    cache_ids = list(cache_payload["image_ids"])
    if dataset_ids[: min(16, len(dataset_ids))] != cache_ids[: min(16, len(cache_ids))]:
        raise RuntimeError("Dataset order does not match cache order; cannot align features safely.")


def compute_retrieval_metrics(semantic: torch.Tensor, candidate_h14: torch.Tensor) -> dict[str, float]:
    semantic = F.normalize(semantic, dim=-1)
    candidate_h14 = F.normalize(candidate_h14, dim=-1)
    sims = semantic @ candidate_h14.t()
    targets = torch.arange(sims.size(0), device=sims.device)
    top1 = (sims.argmax(dim=1) == targets).float().mean().item()
    topk = min(5, sims.size(1))
    top5 = (sims.topk(topk, dim=1).indices == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1": top1, "top5": top5}


def align_embedding_dim(embedding: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = embedding.size(-1)
    if current_dim == target_dim:
        return embedding
    if current_dim > target_dim:
        return embedding[..., :target_dim]
    return F.pad(embedding, (0, target_dim - current_dim))


def run_eval(cfg, model: EEGEncoderV2, test_loader, test_cache: dict, device: torch.device) -> dict[str, float]:
    model.eval()
    sems: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device)
            subject_ids = batch["subject_id"].to(device)
            outputs = model(eeg, subject_ids=subject_ids)
            sems.append(outputs["semantic"].detach().cpu())
    semantic = torch.cat(sems, dim=0).to(device)
    candidate_h14 = test_cache["features"]["h14"][: semantic.size(0)].to(device)
    return compute_retrieval_metrics(semantic, candidate_h14)


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    stage_cfg = cfg.stage_params(args.stage)
    epochs = int(args.epochs if args.epochs is not None else stage_cfg["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else stage_cfg["batch_size"])

    train_loader = build_dataloader(cfg, "train", batch_size=batch_size, shuffle=True, limit=args.limit_train)
    test_loader = build_dataloader(cfg, "test", batch_size=batch_size, shuffle=False, limit=args.limit_test)
    train_cache = load_cache(cfg, "train")
    test_cache = load_cache(cfg, "test")
    ensure_alignment(train_loader, train_cache)
    ensure_alignment(test_loader, test_cache)

    model = EEGEncoderV2(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=float(stage_cfg["lr"]), weight_decay=cfg.weight_decay)
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    best_top1 = -1.0
    best_path = cfg.ckpt_dir / f"{args.tag}_best.pt"

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        resume_stage = state.get("stage")
        if resume_stage == args.stage and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state.get("epoch", -1)) + 1
            best_top1 = float(state.get("best_top1", -1.0))

    train_h14 = train_cache["features"]["h14"]
    train_b32 = train_cache["features"]["b32"]
    train_rn50 = train_cache["features"]["rn50"]
    train_struct = train_cache["features"]["vae_latent"].flatten(1)
    log_path = cfg.log_dir / f"train_encoder_{args.tag}.log"

    with log_path.open("a", encoding="utf-8") as log_handle:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"{args.tag} epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                indices = batch["index"]
                eeg = batch["eeg"].to(device)
                labels = batch["label"].to(device)
                subject_ids = batch["subject_id"].to(device)

                target_h14 = train_h14[indices].to(device)
                target_b32 = train_b32[indices].to(device)
                target_rn50 = train_rn50[indices].to(device)
                target_struct = train_struct[indices].to(device)

                outputs = model(eeg, subject_ids=subject_ids)
                semantic = outputs["semantic"]
                structural = outputs["structural"]
                semantic_b32 = align_embedding_dim(semantic, target_b32.size(-1))
                semantic_rn50 = align_embedding_dim(semantic, target_rn50.size(-1))

                loss = torch.tensor(0.0, device=device)
                if stage_cfg["w_h14_nce"]:
                    loss = loss + float(stage_cfg["w_h14_nce"]) * info_nce_loss(semantic, target_h14)
                if stage_cfg["w_h14_mse"]:
                    loss = loss + float(stage_cfg["w_h14_mse"]) * F.mse_loss(semantic, target_h14)
                if stage_cfg["w_b32_nce"]:
                    loss = loss + float(stage_cfg["w_b32_nce"]) * info_nce_loss(semantic_b32, target_b32)
                if stage_cfg["w_rn50_nce"]:
                    loss = loss + float(stage_cfg["w_rn50_nce"]) * info_nce_loss(semantic_rn50, target_rn50)
                if stage_cfg["w_struct"]:
                    loss = loss + float(stage_cfg["w_struct"]) * F.smooth_l1_loss(structural, target_struct)
                if stage_cfg["w_hard"]:
                    loss = loss + float(stage_cfg["w_hard"]) * hard_negative_infonce(semantic, target_h14)
                if stage_cfg["w_supcon"]:
                    loss = loss + float(stage_cfg["w_supcon"]) * supervised_contrastive_loss(semantic, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            metrics = run_eval(cfg, model, test_loader, test_cache, device)
            mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            row = {
                "epoch": epoch,
                "stage": args.stage,
                "loss": mean_loss,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
            }
            history.append(row)
            line = json.dumps(row, ensure_ascii=False)
            print(line)
            log_handle.write(line + "\n")
            log_handle.flush()

            if metrics["top1"] >= best_top1:
                best_top1 = metrics["top1"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "stage": args.stage,
                        "tag": args.tag,
                        "best_top1": best_top1,
                        "metrics": metrics,
                    },
                    best_path,
                )

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "stage": args.stage,
                    "tag": args.tag,
                    "best_top1": best_top1,
                    "metrics": metrics,
                },
                cfg.ckpt_dir / f"{args.tag}_last.pt",
            )

    with (cfg.result_dir / f"history_{args.tag}.json").open("w", encoding="utf-8") as handle:
        json.dump({"history": history, "best_top1": best_top1, "tag": args.tag}, handle, indent=2)


if __name__ == "__main__":
    main()
