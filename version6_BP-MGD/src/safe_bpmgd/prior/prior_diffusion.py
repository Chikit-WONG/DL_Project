from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MLPPriorMapper(nn.Module):
    def __init__(self, dim: int = 1024, hidden_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_sem: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z_sem.float()), dim=-1)


def prior_mapper_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return F.mse_loss(pred, target) + (1.0 - F.cosine_similarity(pred, target, dim=-1).mean())
