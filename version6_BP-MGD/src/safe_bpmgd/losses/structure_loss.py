from __future__ import annotations

import torch
import torch.nn.functional as F

from safe_bpmgd.losses.contrastive import cosine_loss


def smooth_l1_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred.float(), target.float()) + cosine_loss(pred, target)
