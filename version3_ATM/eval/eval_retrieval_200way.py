"""
Official 200-way retrieval evaluation for the course project.

Evaluation protocol (from Project1.pdf + sample code):
  - Test set: 200 EEG samples (avg_trials=True), one per test class
  - Candidate set: 200 CLIP image features (one per test image)
  - For each test EEG, compute similarity to all 200 candidates
  - Report Top-1 and Top-5 accuracy
  - Run over 10 seeds and report mean ± std

Usage:
  python eval/eval_retrieval_200way.py \
    --checkpoint ./models/contrast/ATMS/sub-01/<run>/40.pth \
    --output_csv  ./outputs/retrieval_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.data_bridge import load_pt_data
from models.clip_bridge import create_open_clip_vit_h_14, get_feature_cache_path
from models.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from models.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.subject_layers.Embed import DataEmbedding
from models.loss import ClipLoss
from einops.layers.torch import Rearrange


# ──────────────────────────────────────────────────────────────────────────────
# ATMS model (same architecture as ATMS_retrieval.py)
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    task_name = 'classification'; seq_len = 250; pred_len = 250
    output_attention = False; d_model = 250; embed = 'timeF'; freq = 'h'
    dropout = 0.25; factor = 1; n_heads = 4; e_layers = 1; d_ff = 256
    activation = 'gelu'; enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, cfg, num_subjects=2):
        super().__init__()
        self.enc_embedding = DataEmbedding(
            cfg.seq_len, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout,
            joint_train=False, num_subjects=num_subjects)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(FullAttention(False, cfg.factor,
                    attention_dropout=cfg.dropout,
                    output_attention=cfg.output_attention),
                    cfg.d_model, cfg.n_heads),
                cfg.d_model, cfg.d_ff,
                dropout=cfg.dropout, activation=cfg.activation)
             for _ in range(cfg.e_layers)],
            norm_layer=nn.LayerNorm(cfg.d_model))

    def forward(self, x, x_mark, subject_ids=None):
        enc_out, _ = self.encoder(self.enc_embedding(x, x_mark, subject_ids))
        return enc_out[:, :63, :]


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)), nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (63, 1)), nn.BatchNorm2d(40), nn.ELU(),
            nn.Dropout(0.5))
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'))

    def forward(self, x):
        return self.projection(self.tsconv(x.unsqueeze(1)))


class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x, **kw): return x + self.fn(x, **kw)


class FlattenHead(nn.Sequential):
    def forward(self, x): return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(proj_dim, proj_dim),
                                      nn.Dropout(drop_proj))),
            nn.LayerNorm(proj_dim))


class ATMS(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Config()
        self.encoder = iTransformer(cfg, num_subjects=10)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(cfg.d_model, 250) for _ in range(2)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        return self.proj_eeg(self.enc_eeg(x))


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_retrieval_metrics(logits: torch.Tensor):
    """
    Parameters
    ----------
    logits : [N, N] similarity matrix (EEG × image).
             Row i should match column i (identity correspondence).
    """
    n = logits.shape[0]
    targets = torch.arange(n)
    top1_acc = (logits.argmax(dim=1) == targets).float().mean().item()
    top5_idx = logits.topk(k=5, dim=1).indices
    top5_acc = (top5_idx == targets[:, None]).any(dim=1).float().mean().item()
    return {"top1_acc": top1_acc, "top5_acc": top5_acc}


def _extract_sub_id(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else 1


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device)
    data_path = Path(args.data_path)

    # ── 1. Load test EEG data (avg_trials=True required) ─────────────────────
    loaded = load_pt_data(
        data_path=data_path,
        split="test",
        avg_trials=True,
        image_dir=data_path / "test_images",
    )
    eeg_data = loaded["eeg"].float()         # [200, 63, 250]
    labels = loaded["labels"]               # [200]  values in {0,...,199}
    test_images = loaded["images"]          # list of 200 image paths
    sample_img_idx = loaded["sample_image_indices"]  # [200]
    print(f"Test EEG: {eeg_data.shape}")

    # ── 2. Compute CLIP image features for all 200 test images ───────────────
    clip_cache = Path(args.clip_cache_dir) / "ViT-H-14_features_test.pt"
    if clip_cache.exists():
        print(f"Loading cached test CLIP features from {clip_cache}")
        test_img_features = torch.load(clip_cache, map_location="cpu")["img_features"]
    else:
        print("Computing CLIP ViT-H-14 test image features …")
        vlmodel, preprocess_train, _ = create_open_clip_vit_h_14(
            device=str(device), precision='fp32')
        vlmodel.eval()

        from PIL import Image as PILImage
        img_paths_ordered = [test_images[i] for i in sample_img_idx]
        feats = []
        bs = 20
        with torch.no_grad():
            for i in range(0, len(img_paths_ordered), bs):
                batch = img_paths_ordered[i:i + bs]
                imgs = torch.stack([
                    preprocess_train(PILImage.open(p).convert("RGB"))
                    for p in batch
                ]).to(device)
                f = vlmodel.encode_image(imgs)
                f = F.normalize(f.float(), dim=-1)
                feats.append(f.cpu())
        test_img_features = torch.cat(feats, dim=0)  # [200, 1024]
        clip_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"img_features": test_img_features}, clip_cache)
        print(f"Cached to {clip_cache}")

    # The 200-way candidate set: indexed by label (0–199)
    # sample_img_idx maps each EEG sample to an image index in test_images
    # Reorder so that img_features[label] corresponds to the correct image for each EEG
    n_test = len(labels)
    candidate_features = torch.zeros(n_test, test_img_features.shape[1])
    for i in range(n_test):
        candidate_features[labels[i]] = test_img_features[i]
    candidate_features = F.normalize(candidate_features, dim=-1)
    print(f"Candidate image features: {candidate_features.shape}")

    # ── 3. Load ATMS encoder ──────────────────────────────────────────────────
    model = ATMS().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Checkpoint: {args.checkpoint}")

    sub_id = _extract_sub_id(args.subject)

    # ── 4. Run inference: get EEG embeddings ─────────────────────────────────
    all_eeg_embeds = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(eeg_data), bs):
            batch = eeg_data[i:i + bs].to(device)
            n = batch.size(0)
            sids = torch.full((n,), sub_id, dtype=torch.long, device=device)
            emb = model(batch, sids).float()
            emb = F.normalize(emb, dim=-1)
            all_eeg_embeds.append(emb.cpu())
    eeg_embeds = torch.cat(all_eeg_embeds, dim=0)  # [200, 1024]

    # ── 5. 200-way retrieval over 10 seeds ───────────────────────────────────
    # For the full 200-way protocol, the similarity matrix is [N, N].
    # The correct image for sample i is at column labels[i].
    # We build the logit matrix using the logit_scale from the model.
    logit_scale = model.logit_scale.exp().item()
    print(f"logit_scale = {logit_scale:.4f}")

    # EEG embeddings: [200, 1024], rows indexed by sample
    # Candidate features: [200, 1024], columns indexed by class label (0..199)
    # logits[i, j] = logit_scale * eeg_embeds[i] @ candidate_features[j]
    logits = logit_scale * (eeg_embeds @ candidate_features.T)  # [200, 200]

    # Ground truth: logits[i] should be maximum at column labels[i]
    # Re-index so that logits[i, labels[i]] is the correct match
    # The current logits[i, j] = sim(EEG_i, image_j)
    # targets[i] = labels[i] (the correct image column)
    targets = labels  # [200], values in {0,...,199}

    # Top-1: argmax across 200 columns
    top1_pred = logits.argmax(dim=1)
    top1_acc = (top1_pred == targets).float().mean().item()

    # Top-5
    top5_idx = logits.topk(k=5, dim=1).indices  # [200, 5]
    top5_acc = (top5_idx == targets[:, None]).any(dim=1).float().mean().item()

    print(f"\n{'='*50}")
    print(f"200-way Retrieval Results")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"{'='*50}")

    # ── 6. Report over 10 seeds (randomized 200-way, matches paper protocol) ──
    # The randomized version samples k-1 distractors + correct class per query.
    # For k=200 with 200 test classes, all 200 classes are always selected,
    # so the result is identical across seeds. We run it anyway for completeness.
    import random

    seed_results = []
    for seed in range(10):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        all_labels_set = set(range(n_test))
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        for i in range(n_test):
            label = labels[i].item()
            possible = list(all_labels_set - {label})
            selected = random.sample(possible, 199) + [label]  # 200 classes
            sel_feats = candidate_features[selected]            # [200, 1024]
            sim = logit_scale * (eeg_embeds[i] @ sel_feats.T)  # [200]
            pred_idx = torch.argmax(sim).item()
            pred_label = selected[pred_idx]
            if pred_label == label:
                correct_top1 += 1
            top5_sel = torch.topk(sim, 5).indices.tolist()
            if label in [selected[j] for j in top5_sel]:
                correct_top5 += 1
            total += 1
        seed_results.append({
            "seed": seed,
            "top1_acc": correct_top1 / total,
            "top5_acc": correct_top5 / total,
        })
        print(f"seed={seed:02d} | Top-1={seed_results[-1]['top1_acc']:.4f} | "
              f"Top-5={seed_results[-1]['top5_acc']:.4f}")

    top1_vals = np.array([r["top1_acc"] for r in seed_results])
    top5_vals = np.array([r["top5_acc"] for r in seed_results])
    print(f"\nOver 10 seeds:")
    print(f"  Top-1: {top1_vals.mean():.4f} ± {top1_vals.std(ddof=1):.4f}")
    print(f"  Top-5: {top5_vals.mean():.4f} ± {top5_vals.std(ddof=1):.4f}")

    # ── 7. Save results ───────────────────────────────────────────────────────
    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "top1_acc", "top5_acc"])
            writer.writeheader()
            writer.writerows(seed_results)
            writer.writerow({
                "seed": "mean",
                "top1_acc": top1_vals.mean(),
                "top5_acc": top5_vals.mean(),
            })
            writer.writerow({
                "seed": "std",
                "top1_acc": top1_vals.std(ddof=1),
                "top5_acc": top5_vals.std(ddof=1),
            })
        print(f"Results saved to {out_path}")

    return {
        "top1_mean": top1_vals.mean(),
        "top1_std": top1_vals.std(ddof=1),
        "top5_mean": top5_vals.mean(),
        "top5_std": top5_vals.std(ddof=1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained ATMS retrieval checkpoint (.pth)")
    parser.add_argument("--data_path",
                        default="/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning"
                                "/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--clip_cache_dir",
                        default="/hpc2hdd/home/ckwong627/workdir/models",
                        help="Directory where CLIP feature caches are stored")
    parser.add_argument("--output_csv",
                        default="./outputs/retrieval_eval.csv")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
