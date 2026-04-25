import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader

REPO_DIR = Path(__file__).resolve().parents[1]
import sys

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import models
from main_eeg_course import ClipLoss, create_logger, device, get_dataset, set_seed
from scripts.task2_common import (
    BLUR_LEVELS,
    DEFAULT_CLIP_WEIGHTS,
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROMPT_TEMPLATE,
    aggregate_retrieval,
    build_text_prototype_cache,
    class_name_from_key,
    ensure_dir,
    load_feature_bank,
    path_hash,
)


def class_indices_from_keys(keys, class_to_idx):
    return torch.tensor([class_to_idx[class_name_from_key(key)] for key in keys], dtype=torch.long)


@torch.no_grad()
def evaluate_prompt_retrieval(model, train_bank, dataloader, class_to_idx, topk):
    model.eval()
    total = 0
    top1 = 0
    top5 = 0
    train_embeds = train_bank["embeddings"].to(device)
    train_classes = train_bank["class_names"]

    for batch in dataloader:
        eeg = batch["eeg"].to(device)
        eeg_embed = F.normalize(model(eeg).float(), dim=-1)
        similarities = eeg_embed @ train_embeds.T
        for sample_idx in range(similarities.shape[0]):
            total += 1
            agg = aggregate_retrieval(similarities[sample_idx], train_bank["keys"], train_classes, topk)
            ranked_classes = [item["class_name"] for item in agg["ranked_classes"]]
            gt_class = class_name_from_key(batch["x_key"][sample_idx])
            if ranked_classes and ranked_classes[0] == gt_class:
                top1 += 1
            if gt_class in ranked_classes[:5]:
                top5 += 1

    if total == 0:
        return 0.0, 0.0
    return top1 / total, top5 / total


def train_one_seed(args, logger, seed, prototype_payload):
    set_seed(seed)
    train_dataset, val_dataset, test_dataset = get_dataset(
        str(args.data_path),
        str(args.feature_path),
        1,
        args.select_chs,
        args.use_filter,
        args.low_freq,
        args.high_freq,
        [0, 250],
    )
    logger.info(
        "Task2 seed=%s dataset sizes: train=%s val=%s test=%s",
        seed,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=min(len(val_dataset), args.eval_batch_size), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=min(len(test_dataset), args.eval_batch_size), shuffle=False)

    model = models.__dict__[args.net_name](len(args.select_chs), 1024, 250).to(device)
    state_dict = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = ClipLoss()
    text_prototypes = prototype_payload["embeddings"].to(device)
    class_to_idx = prototype_payload["class_to_idx"]
    scale = float(args.class_logit_scale)

    train_bank_cache = ensure_dir(args.save_path / "cache") / f"train_bank_seed{seed}.pt"
    train_feature_bank = load_feature_bank(args.feature_path, "train")
    from scripts.task2_common import build_adapted_image_bank

    train_bank = build_adapted_image_bank(
        model, train_feature_bank, train_bank_cache, batch_size=args.bank_batch_size, force=True
    )

    best_state = None
    best_score = -1.0
    best_metrics = {}
    history = []

    for epoch in range(args.epoch):
        model.train()
        loss_sum = 0.0
        image_loss_sum = 0.0
        class_loss_sum = 0.0
        steps = 0

        for batch in tqdm.tqdm(train_loader, desc=f"Task2 seed={seed} epoch={epoch}"):
            optimizer.zero_grad()
            eeg = batch["eeg"].to(device)
            img_list = torch.cat([batch[key][:, None].to(device) for key in BLUR_LEVELS], dim=1)

            eeg_features = model(eeg)
            eeg_norm = F.normalize(eeg_features.float(), dim=-1)
            img_features = F.normalize(model.get_image_feature(img_list).float(), dim=-1)

            eeg_loss, img_loss = criterion(eeg_norm, img_features, 1.0)
            image_loss = (eeg_loss.mean() + img_loss.mean()) / 2.0

            class_targets = class_indices_from_keys(batch["x_key"], class_to_idx).to(device)
            class_logits = scale * (eeg_norm @ text_prototypes.T)
            class_loss = F.cross_entropy(class_logits, class_targets)

            loss = args.image_loss_weight * image_loss + args.class_loss_weight * class_loss
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            image_loss_sum += float(image_loss.detach().cpu())
            class_loss_sum += float(class_loss.detach().cpu())
            steps += 1

        train_bank = build_adapted_image_bank(
            model, train_feature_bank, train_bank_cache, batch_size=args.bank_batch_size, force=True
        )
        val_prompt_top1, val_prompt_top5 = evaluate_prompt_retrieval(
            model, train_bank, val_loader, class_to_idx, args.prompt_topk
        )
        val_retrieval_top1, _, val_retrieval_top5 = get_retrieval_metrics(model, val_loader)
        test_retrieval_top1, _, test_retrieval_top5 = get_retrieval_metrics(model, test_loader)

        mean_loss = loss_sum / max(steps, 1)
        row = {
            "seed": seed,
            "epoch": epoch,
            "loss": mean_loss,
            "image_loss": image_loss_sum / max(steps, 1),
            "class_loss": class_loss_sum / max(steps, 1),
            "val_prompt_top1": val_prompt_top1,
            "val_prompt_top5": val_prompt_top5,
            "val_retrieval_top1": val_retrieval_top1,
            "val_retrieval_top5": val_retrieval_top5,
            "test_retrieval_top1": test_retrieval_top1,
            "test_retrieval_top5": test_retrieval_top5,
        }
        history.append(row)

        logger.info(
            "Task2 seed=%s epoch=%s loss=%.4f image_loss=%.4f class_loss=%.4f "
            "VAL prompt_top1=%.4f prompt_top5=%.4f retrieval_top1=%.4f retrieval_top5=%.4f "
            "TEST retrieval_top1=%.4f retrieval_top5=%.4f",
            seed,
            epoch,
            row["loss"],
            row["image_loss"],
            row["class_loss"],
            val_prompt_top1,
            val_prompt_top5,
            val_retrieval_top1,
            val_retrieval_top5,
            test_retrieval_top1,
            test_retrieval_top5,
        )

        if val_prompt_top1 > best_score:
            best_score = val_prompt_top1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = copy.deepcopy(row)
            ckpt_path = args.save_path / f"{args.net_name}_seed{seed}_task2_best.pth"
            bank_path = args.save_path / f"train_bank_seed{seed}_task2_best.pt"
            torch.save(best_state, ckpt_path)
            torch.save(train_bank, bank_path)
            best_metrics["best_ckpt"] = str(ckpt_path)
            best_metrics["best_train_bank"] = str(bank_path)

    pd.DataFrame(history).to_csv(args.save_path / f"task2_seed{seed}_history.csv", index=False)
    return best_metrics


@torch.no_grad()
def get_retrieval_metrics(model, dataloader):
    total = top1 = top3 = top5 = 0
    model.eval()
    for batch in dataloader:
        eeg = batch["eeg"].to(device)
        img_list = torch.cat([batch[key][:, None].to(device) for key in BLUR_LEVELS], dim=1)
        eeg_features = F.normalize(model(eeg).float(), dim=-1)
        img_features = F.normalize(model.get_image_feature(img_list).float(), dim=-1)
        sim = eeg_features @ img_features.T
        _, indices = sim.topk(5, dim=-1)
        labels = torch.arange(eeg.shape[0], device=indices.device)[:, None]
        top1 += (indices[:, :1] == labels).sum().item()
        top3 += (indices[:, :3] == labels).any(dim=1).sum().item()
        top5 += (indices[:, :5] == labels).any(dim=1).sum().item()
        total += eeg.shape[0]
    if total == 0:
        return 0.0, 0.0, 0.0
    return top1 / total, top3 / total, top5 / total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name", type=str, default="Brain_Visual_Encoder_EEG")
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--bank_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--first_seed", type=int, default=21)
    parser.add_argument("--class_logit_scale", type=float, default=10.0)
    parser.add_argument("--image_loss_weight", type=float, default=0.7)
    parser.add_argument("--class_loss_weight", type=float, default=0.3)
    parser.add_argument("--prompt_topk", type=int, default=20)
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--feature_path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--clip_checkpoint", type=Path, default=DEFAULT_CLIP_WEIGHTS)
    parser.add_argument("--init_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "semantic_finetune")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_filter", action="store_true")
    parser.add_argument("--low_freq", type=float, default=0.1)
    parser.add_argument("--high_freq", type=float, default=50.0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.select_chs = [
        "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
        "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
        "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
        "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
        "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
        "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
        "O1", "Oz", "O2",
    ]
    run_name = args.run_name or (time.strftime("%Y-%m-%d-%H-%M") + f"_{path_hash(args.init_ckpt, args.prompt_template)}")
    args.save_path = ensure_dir(args.output_dir / run_name)
    logger = create_logger(str(args.save_path))
    logger.info("Task2 semantic fine-tuning output: %s", args.save_path)
    logger.info("Task2 args: %s", vars(args))

    train_feature_bank = load_feature_bank(args.feature_path, "train")
    prototype_cache = args.save_path / "class_text_prototypes.pt"
    prototype_payload = build_text_prototype_cache(
        train_feature_bank,
        args.clip_checkpoint,
        prototype_cache,
        args.prompt_template,
        device,
    )

    with (args.save_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "net_name": args.net_name,
                "epoch": args.epoch,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "lr": args.lr,
                "n_seeds": args.n_seeds,
                "first_seed": args.first_seed,
                "class_logit_scale": args.class_logit_scale,
                "image_loss_weight": args.image_loss_weight,
                "class_loss_weight": args.class_loss_weight,
                "prompt_topk": args.prompt_topk,
                "prompt_template": args.prompt_template,
                "data_path": str(args.data_path),
                "feature_path": str(args.feature_path),
                "clip_checkpoint": str(args.clip_checkpoint),
                "init_ckpt": str(args.init_ckpt),
                "prototype_cache": str(prototype_cache),
            },
            handle,
            indent=2,
        )

    all_best = []
    for seed in range(args.first_seed, args.first_seed + args.n_seeds):
        logger.info("Start Task2 fine-tuning seed=%s", seed)
        metrics = train_one_seed(args, logger, seed, prototype_payload)
        all_best.append(metrics)

    summary = {
        "val_prompt_top1_mean": float(np.mean([row["val_prompt_top1"] for row in all_best])),
        "val_prompt_top1_std": float(np.std([row["val_prompt_top1"] for row in all_best])),
        "val_prompt_top5_mean": float(np.mean([row["val_prompt_top5"] for row in all_best])),
        "val_prompt_top5_std": float(np.std([row["val_prompt_top5"] for row in all_best])),
        "test_retrieval_top1_mean": float(np.mean([row["test_retrieval_top1"] for row in all_best])),
        "test_retrieval_top1_std": float(np.std([row["test_retrieval_top1"] for row in all_best])),
        "test_retrieval_top5_mean": float(np.mean([row["test_retrieval_top5"] for row in all_best])),
        "test_retrieval_top5_std": float(np.std([row["test_retrieval_top5"] for row in all_best])),
    }
    pd.DataFrame(all_best).to_csv(args.save_path / "task2_best_metrics.csv", index=False)
    with (args.save_path / "task2_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Task2 summary: %s", summary)


if __name__ == "__main__":
    main()
