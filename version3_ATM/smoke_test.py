"""
CPU-only smoke test. Validates imports, model instantiation, and data loading.
Run from the version3/ directory:
    python smoke_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

DATA_PATH = Path("/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning"
                 "/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data")

ERRORS: list[str] = []


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK] {label}")
        return result
    except Exception as e:
        msg = f"  [FAIL] {label}: {e}"
        print(msg)
        ERRORS.append(msg)
        return None


print("=" * 60)
print("SMOKE TEST — version3 (CPU only)")
print("=" * 60)

# ── 1. Core imports ───────────────────────────────────────────
print("\n[1] Core imports")
check("torch", lambda: __import__("torch"))
check("numpy", lambda: __import__("numpy"))
check("einops", lambda: __import__("einops"))
check("PIL", lambda: __import__("PIL"))

# ── 2. Project module imports ─────────────────────────────────
print("\n[2] Project module imports")
check("models.data_bridge", lambda: __import__("models.data_bridge", fromlist=["load_pt_data"]))
check("models.loss", lambda: __import__("models.loss", fromlist=["ClipLoss"]))
check("models.subject_layers.Transformer_EncDec",
      lambda: __import__("models.subject_layers.Transformer_EncDec", fromlist=["Encoder", "EncoderLayer"]))
check("models.subject_layers.SelfAttention_Family",
      lambda: __import__("models.subject_layers.SelfAttention_Family", fromlist=["FullAttention", "AttentionLayer"]))
check("models.subject_layers.Embed",
      lambda: __import__("models.subject_layers.Embed", fromlist=["DataEmbedding"]))

# ── 3. ATMS model instantiation + forward pass (CPU) ─────────
print("\n[3] ATMS model (CPU forward pass)")
import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange
from models.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from models.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.subject_layers.Embed import DataEmbedding
from models.loss import ClipLoss


class _Config:
    task_name = 'classification'; seq_len = 250; pred_len = 250
    output_attention = False; d_model = 250; embed = 'timeF'; freq = 'h'
    dropout = 0.25; factor = 1; n_heads = 4; e_layers = 1; d_ff = 256
    activation = 'gelu'; enc_in = 63


class _iTransformer(nn.Module):
    def __init__(self, cfg, num_subjects=10):
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


class _PatchEmbedding(nn.Module):
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


class _ATMS(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = _Config()
        self.encoder = _iTransformer(cfg, num_subjects=10)
        self.enc_eeg = nn.Sequential(_PatchEmbedding(40),
                                     type('FH', (nn.Sequential,), {
                                         'forward': lambda self, x: x.contiguous().view(x.size(0), -1)})())
        self.proj_eeg = nn.Sequential(
            nn.Linear(1440, 1024),
            nn.LayerNorm(1024))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        return self.proj_eeg(self.enc_eeg(x))


def _test_atms():
    model = _ATMS().eval()
    x = torch.randn(2, 63, 250)
    sids = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = model(x, sids)
    assert out.shape == (2, 1024), f"Expected (2,1024), got {out.shape}"
    return out.shape


check("ATMS instantiation + forward (2,63,250) → (2,1024)", _test_atms)

# ── 4. Data path check ────────────────────────────────────────
print("\n[4] Data paths")
check("DATA_PATH exists",
      lambda: DATA_PATH if DATA_PATH.exists() else (_ for _ in ()).throw(FileNotFoundError(DATA_PATH)))
check("train.pt exists", lambda: (DATA_PATH / "train.pt") if (DATA_PATH / "train.pt").exists()
      else (_ for _ in ()).throw(FileNotFoundError(DATA_PATH / "train.pt")))
check("test.pt exists", lambda: (DATA_PATH / "test.pt") if (DATA_PATH / "test.pt").exists()
      else (_ for _ in ()).throw(FileNotFoundError(DATA_PATH / "test.pt")))
check("test_images/ exists", lambda: (DATA_PATH / "test_images") if (DATA_PATH / "test_images").exists()
      else (_ for _ in ()).throw(FileNotFoundError(DATA_PATH / "test_images")))

# ── 5. data_bridge: load test split ──────────────────────────
print("\n[5] data_bridge — load test.pt")
def _test_data_bridge():
    from models.data_bridge import load_pt_data
    loaded = load_pt_data(
        data_path=DATA_PATH,
        split="test",
        avg_trials=True,
        image_dir=DATA_PATH / "test_images",
    )
    eeg = loaded["eeg"]
    assert eeg.ndim == 3 and eeg.shape[1:] == (63, 250), f"Unexpected EEG shape: {eeg.shape}"
    assert len(loaded["sample_image_indices"]) == eeg.shape[0]
    return f"EEG shape={tuple(eeg.shape)}, n_images={len(loaded['images'])}"

check("load_pt_data(split='test', avg_trials=True)", _test_data_bridge)

# ── 6. ClipLoss ───────────────────────────────────────────────
print("\n[6] ClipLoss")
def _test_clip_loss():
    from models.loss import ClipLoss
    loss_fn = ClipLoss()
    a = torch.randn(4, 1024)
    b = torch.randn(4, 1024)
    loss = loss_fn(a, b, torch.tensor(1.0))
    assert loss.ndim == 0
    return f"loss={loss.item():.4f}"

check("ClipLoss forward", _test_clip_loss)

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
if ERRORS:
    print(f"SMOKE TEST FAILED — {len(ERRORS)} error(s):")
    for e in ERRORS:
        print(e)
    sys.exit(1)
else:
    print("SMOKE TEST PASSED — all checks OK")
print("=" * 60)
