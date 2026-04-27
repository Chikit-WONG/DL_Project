from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_WEIGHTS = {"clip": 0.40, "evnet": 0.25, "vae": 0.15, "edge": 0.10, "depth": 0.10}


def retrieve_topk(z_sem: torch.Tensor, bank: dict, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    bank_clip = F.normalize(bank["clip"].float(), dim=-1)
    query = F.normalize(z_sem.float(), dim=-1)
    sims = query @ bank_clip.T
    return torch.topk(sims, k=min(top_k, bank_clip.shape[0]), dim=-1)


def score_candidates(candidate_features: list[dict], predicted: dict, weights: dict | None = None) -> list[dict]:
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for idx, feats in enumerate(candidate_features):
        score = 0.0
        terms = {}
        for key, weight in weights.items():
            if key in feats and key in predicted:
                term = F.cosine_similarity(feats[key].float().flatten()[None], predicted[key].float().flatten()[None]).item()
                terms[key] = term
                score += weight * term
        rows.append({"candidate_index": idx, "score": float(score), "terms": terms})
    return sorted(rows, key=lambda item: item["score"], reverse=True)


def prototype_candidate_features(bank: dict, indices: list[int]) -> list[dict]:
    rows = []
    key_map = {"clip": "clip", "evnet": "evnet", "vae": "vae", "edge": "edge", "depth": "depth"}
    for index in indices:
        row = {}
        for source, target in key_map.items():
            if source in bank:
                row[target] = bank[source][index]
        rows.append(row)
    return rows
