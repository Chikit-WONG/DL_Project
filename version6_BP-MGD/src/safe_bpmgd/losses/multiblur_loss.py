from __future__ import annotations

import torch
import torch.nn.functional as F

from safe_bpmgd.losses.contrastive import cosine_loss


def multiblur_alignment_loss(z_sem: torch.Tensor, blur_logits: torch.Tensor, multiblur_targets: torch.Tensor) -> torch.Tensor:
    weights = F.softmax(blur_logits.float(), dim=-1)
    target = (weights.unsqueeze(-1) * multiblur_targets.float()).sum(dim=1)
    return cosine_loss(z_sem, F.normalize(target, dim=-1))
