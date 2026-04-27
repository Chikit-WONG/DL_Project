from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from safe_bpmgd.encoders.projection import MLPProjector


class MultiModalHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        semantic_dim: int = 1024,
        struct_dim: int = 256,
        edge_dim: int = 512,
        depth_dim: int = 512,
        vae_dim: int = 512,
        num_blur_levels: int = 6,
    ) -> None:
        super().__init__()
        self.semantic_head = MLPProjector(hidden_dim, semantic_dim)
        self.blur_head = nn.Linear(hidden_dim, num_blur_levels)
        self.struct_head = MLPProjector(hidden_dim, struct_dim)
        self.edge_head = MLPProjector(hidden_dim, edge_dim)
        self.depth_head = MLPProjector(hidden_dim, depth_dim)
        self.vae_head = MLPProjector(hidden_dim, vae_dim)
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        z_sem_raw = self.semantic_head(h)
        return {
            "h": h,
            "z_sem_raw": z_sem_raw,
            "z_sem": F.normalize(z_sem_raw.float(), dim=-1),
            "z_blur_logits": self.blur_head(h),
            "z_struct": F.normalize(self.struct_head(h).float(), dim=-1),
            "z_edge": F.normalize(self.edge_head(h).float(), dim=-1),
            "z_depth": F.normalize(self.depth_head(h).float(), dim=-1),
            "z_vae": self.vae_head(h),
            "uncertainty": self.uncertainty_head(h).squeeze(-1),
        }
