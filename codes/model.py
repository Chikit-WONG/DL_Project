"""EEG encoder + UnifiedModel.

The unified model carries both losses (InfoNCE + MSE) so that
Architecture B (independent retrieval / reconstruction encoders) can be
trained as the special case alpha=1, beta=0 or alpha=0, beta=1, while
Architecture A is alpha>0, beta>0. Loss weights can also be made
``learnable`` via ``nn.Parameter`` in log-space, giving the homoscedastic
auto-balance variant.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ---------------------------------------------------------------------------
# EEG Encoder: spatial Conv -> temporal Conv stack -> Transformer -> MLP head
# ---------------------------------------------------------------------------
class EEGEncoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c_spatial = cfg.encoder_spatial_dim
        t1, t2, t3 = cfg.encoder_temporal_channels
        kt = cfg.encoder_temporal_kernel
        st = cfg.encoder_temporal_stride
        pt = kt // 2
        drop = cfg.encoder_dropout

        # Spatial mixing across the C=63 EEG channels
        self.spatial = nn.Sequential(
            nn.Conv1d(cfg.num_eeg_channels, c_spatial, kernel_size=1),
            nn.BatchNorm1d(c_spatial),
            nn.GELU(),
            nn.Conv1d(c_spatial, c_spatial, kernel_size=1),
            nn.BatchNorm1d(c_spatial),
            nn.GELU(),
        )

        # Temporal feature extraction
        self.temporal = nn.Sequential(
            nn.Conv1d(c_spatial, t1, kernel_size=kt, stride=st, padding=pt),
            nn.BatchNorm1d(t1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(t1, t2, kernel_size=kt, stride=st, padding=pt),
            nn.BatchNorm1d(t2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(t2, t3, kernel_size=kt, stride=st, padding=pt),
            nn.BatchNorm1d(t3),
            nn.GELU(),
            nn.Dropout(drop),
        )

        d_model = t3
        # Compute the time-dim length after the temporal stack so we can
        # create the positional embedding eagerly (otherwise the optimizer
        # would never see it).
        T = cfg.num_eeg_timesteps
        for _ in range(3):
            T = (T + 2 * pt - kt) // st + 1
        self._pos_len = T
        self.pos_embedding = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.encoder_transformer_heads,
            dim_feedforward=cfg.encoder_transformer_ffn,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.encoder_n_transformer_layers
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, cfg.encoder_mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(cfg.encoder_mlp_hidden, cfg.embed_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Args: eeg [B, C, T]. Returns: raw embedding [B, embed_dim] (unnormalized)."""
        x = self.spatial(eeg)         # [B, c_spatial, T]
        x = self.temporal(x)          # [B, d_model, T']
        x = x.transpose(1, 2)         # [B, T', d_model]
        if x.shape[1] != self._pos_len:
            raise RuntimeError(
                f"Encoder time-dim mismatch: expected {self._pos_len}, got {x.shape[1]}. "
                "Update Config.num_eeg_timesteps to match the dataset."
            )
        x = x + self.pos_embedding
        x = self.transformer(x)       # [B, T', d_model]
        x = x.mean(dim=1)             # global average pool over time -> [B, d_model]
        x = self.head(x)              # [B, embed_dim]
        return x


# ---------------------------------------------------------------------------
# Unified model: encoder + dual loss (InfoNCE + MSE) with optional learnable weights
# ---------------------------------------------------------------------------
class UnifiedModel(nn.Module):
    def __init__(
        self,
        cfg: Config,
        alpha: float = 1.0,
        beta: float = 0.5,
        learnable_loss_weights: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.eeg_encoder = EEGEncoder(cfg)

        # Learnable temperature in CLIP convention: logit_scale = log(1/τ)
        if cfg.learnable_logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(cfg.init_logit_scale))
        else:
            self.register_buffer("logit_scale", torch.tensor(cfg.init_logit_scale))

        self.learnable_loss_weights = learnable_loss_weights
        if learnable_loss_weights:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(max(alpha, 1e-6))))
            self.log_beta = nn.Parameter(torch.tensor(math.log(max(beta, 1e-6))))
        else:
            self.register_buffer("alpha_buf", torch.tensor(float(alpha)))
            self.register_buffer("beta_buf", torch.tensor(float(beta)))

    # ----- weight helpers -----
    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.learnable_loss_weights:
            return torch.exp(self.log_alpha), torch.exp(self.log_beta)
        return self.alpha_buf, self.beta_buf

    def set_weights(self, alpha: float, beta: float) -> None:
        """Override loss weights at runtime (e.g. switching from phase 1 to phase 2)."""
        if self.learnable_loss_weights:
            with torch.no_grad():
                self.log_alpha.fill_(math.log(max(alpha, 1e-6)))
                self.log_beta.fill_(math.log(max(beta, 1e-6)))
        else:
            self.alpha_buf.fill_(float(alpha))
            self.beta_buf.fill_(float(beta))

    # ----- core forward / loss -----
    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.eeg_encoder(eeg)

    def compute_loss(self, eeg_emb: torch.Tensor, clip_emb: torch.Tensor):
        """Joint loss = alpha * InfoNCE + beta * MSE.

        - InfoNCE uses L2-normalized embeddings (CLIP standard).
        - MSE operates on the raw (unnormalized) outputs so the encoder is
          encouraged to match real CLIP magnitudes — important for IP-Adapter.
        """
        alpha, beta = self.get_weights()

        eeg_norm = F.normalize(eeg_emb, dim=-1)
        clip_norm = F.normalize(clip_emb, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (eeg_norm @ clip_norm.T)             # [B, B]
        labels = torch.arange(eeg_emb.size(0), device=eeg_emb.device)
        l_ret = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

        l_rec = F.mse_loss(eeg_emb, clip_emb)

        total = alpha * l_ret + beta * l_rec
        return total, l_ret.detach(), l_rec.detach(), alpha.detach(), beta.detach()

    @torch.no_grad()
    def encode_for_retrieval(self, eeg: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode(eeg), dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
