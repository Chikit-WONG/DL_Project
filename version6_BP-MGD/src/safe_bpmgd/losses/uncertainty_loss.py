from __future__ import annotations

import torch


def uncertainty_weighted(base_loss: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    u = uncertainty.float().clamp(min=-5.0, max=5.0).mean()
    return torch.exp(-u) * base_loss + u
