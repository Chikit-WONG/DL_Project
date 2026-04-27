from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(eeg_embed: torch.Tensor, image_embed: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    eeg = F.normalize(eeg_embed.float(), dim=-1)
    image = F.normalize(image_embed.float(), dim=-1)
    logits = eeg @ image.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1).mean()
