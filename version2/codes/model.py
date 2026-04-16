from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_electrode_positions(jsonl_path: Path, num_channels: int) -> torch.Tensor:
    coords = torch.zeros(num_channels, 3, dtype=torch.float32)
    if not jsonl_path.exists():
        return coords
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            xyz = None
            for key_triplet in (("x", "y", "z"), ("X", "Y", "Z")):
                if all(key in payload for key in key_triplet):
                    xyz = [float(payload[key]) for key in key_triplet]
                    break
            if xyz is None and "coord" in payload and len(payload["coord"]) >= 3:
                xyz = [float(v) for v in payload["coord"][:3]]
            if xyz is not None:
                rows.append(xyz)
    if not rows:
        return coords
    rows = rows[:num_channels]
    coords[: len(rows)] = torch.tensor(rows, dtype=torch.float32)
    return coords


class ElectrodePositionEncoding(nn.Module):
    def __init__(self, jsonl_path: Path, num_channels: int, embed_dim: int) -> None:
        super().__init__()
        coords = load_electrode_positions(jsonl_path, num_channels)
        self.register_buffer("coords", coords)
        self.proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self.proj(self.coords).unsqueeze(0)
        return x + pos


class RegionAwareGating(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        init = torch.ones(num_channels, dtype=torch.float32)
        occipital_start = max(num_channels - 17, 0)
        init[occipital_start:] = 2.0
        self.logits = nn.Parameter(init.log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.exp(self.logits).view(1, -1, 1)
        return x * gates


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class DualPathConv(nn.Module):
    def __init__(self, in_channels: int, spatial_hidden: int, temporal_hidden: int) -> None:
        super().__init__()
        half_hidden = temporal_hidden // 2
        self.path_a = nn.Sequential(
            nn.Conv1d(in_channels, spatial_hidden, kernel_size=1),
            nn.BatchNorm1d(spatial_hidden),
            nn.GELU(),
            nn.Conv1d(spatial_hidden, half_hidden, kernel_size=15, padding=7, stride=1),
            nn.BatchNorm1d(half_hidden),
            nn.GELU(),
        )
        self.path_b = nn.Sequential(
            nn.Conv1d(in_channels, half_hidden, kernel_size=9, padding=4, stride=1),
            nn.BatchNorm1d(half_hidden),
            nn.GELU(),
            nn.Conv1d(half_hidden, half_hidden, kernel_size=1),
            nn.BatchNorm1d(half_hidden),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.Conv1d(temporal_hidden, temporal_hidden, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(temporal_hidden),
            nn.GELU(),
            SqueezeExcite1D(temporal_hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([self.path_a(x), self.path_b(x)], dim=1))


class SubjectAdapter(nn.Module):
    def __init__(self, max_subjects: int, model_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_subjects, model_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        subject_token = self.embedding(subject_ids).unsqueeze(1)
        return tokens + subject_token


class EEGEncoderV2(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.position = ElectrodePositionEncoding(
            jsonl_path=cfg.eeg_channels_jsonl,
            num_channels=cfg.num_eeg_channels,
            embed_dim=cfg.num_eeg_timesteps,
        )
        self.gating = RegionAwareGating(cfg.num_eeg_channels)
        self.dual_path = DualPathConv(
            in_channels=cfg.num_eeg_channels,
            spatial_hidden=cfg.spatial_hidden,
            temporal_hidden=cfg.temporal_hidden,
        )
        self.proj = nn.Conv1d(cfg.temporal_hidden, cfg.model_dim, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)
        self.subject_adapter = SubjectAdapter(cfg.max_subjects, cfg.model_dim)
        self.norm = nn.LayerNorm(cfg.model_dim)
        self.semantic_head = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim, cfg.semantic_dim),
        )
        self.structural_head = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.model_dim, cfg.structural_dim),
        )

    def forward(self, eeg: torch.Tensor, subject_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = eeg.float()
        x = self.position(x)
        x = self.gating(x)
        x = self.dual_path(x)
        x = self.proj(x)
        tokens = x.transpose(1, 2)
        if subject_ids is None:
            subject_ids = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        tokens = self.subject_adapter(tokens, subject_ids)
        tokens = self.transformer(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        semantic = self.semantic_head(pooled)
        structural = self.structural_head(pooled)
        return {
            "tokens": tokens,
            "pooled": pooled,
            "semantic": semantic,
            "structural": structural,
        }


class PriorUNet(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if emb.shape[1] < dim:
            emb = F.pad(emb, (0, dim - emb.shape[1]))
        return emb

    def forward(self, noisy_target: torch.Tensor, timesteps: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(self.timestep_embedding(timesteps, noisy_target.shape[1]))
        c_emb = self.cond(cond)
        return self.net(torch.cat([noisy_target, t_emb, c_emb], dim=1))

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: int = 25,
        guidance_scale: float = 1.5,
    ) -> torch.Tensor:
        device = cond.device
        x = torch.randn_like(cond)
        null_cond = torch.zeros_like(cond)
        for step in reversed(range(num_steps)):
            t = torch.full((cond.size(0),), step, device=device, dtype=torch.long)
            pred_cond = self.forward(x, t, cond)
            pred_null = self.forward(x, t, null_cond)
            pred = pred_null + guidance_scale * (pred_cond - pred_null)
            alpha = 1.0 - (step + 1) / (num_steps + 1)
            x = alpha * x + (1.0 - alpha) * pred
        return x


def info_nce_loss(query: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = query @ target.t() / temperature
    labels = torch.arange(query.size(0), device=query.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def hard_negative_infonce(query: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = query @ target.t() / temperature
    labels = torch.arange(query.size(0), device=query.device)
    eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    masked = logits.masked_fill(eye, float("-inf"))
    hard_neg = masked.max(dim=1).values
    pos = logits[torch.arange(logits.size(0), device=logits.device), labels]
    margin_logits = torch.stack([pos, hard_neg], dim=1)
    margin_labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
    return F.cross_entropy(margin_logits, margin_labels)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    logits = features @ features.t() / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(mask.size(0), device=mask.device, dtype=torch.bool)
    positives = mask & ~eye
    negatives = ~mask
    exp_logits = torch.exp(logits) * (~eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob * positives).sum(dim=1) / pos_count
    return loss.mean()
