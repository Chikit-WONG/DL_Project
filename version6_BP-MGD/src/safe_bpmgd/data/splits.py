from __future__ import annotations

from pathlib import Path

import numpy as np

from safe_bpmgd.utils.io import write_json


def make_train_val_indices(n_items: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_items)
    rng.shuffle(indices)
    n_val = max(1, int(round(n_items * val_fraction))) if val_fraction > 0 else 0
    val = sorted(indices[:n_val].tolist())
    train = sorted(indices[n_val:].tolist())
    return train, val


def save_split_indices(train_indices: list[int], val_indices: list[int], path: str | Path) -> None:
    write_json({"train_indices": train_indices, "val_indices": val_indices}, path)
