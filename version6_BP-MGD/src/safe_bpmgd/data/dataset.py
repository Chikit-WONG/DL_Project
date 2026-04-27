from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class EEGRecord:
    source_index: int
    image_id: str
    image_path: str
    eeg: torch.Tensor | None = None
    label: int | None = None
    label_name: str = ""


class EEGImageDataset(Dataset):
    def __init__(self, records: Sequence[EEGRecord], eeg_array=None, avg_trials: bool = True, flat_trials: bool = False) -> None:
        self.records = list(records)
        self.eeg_array = eeg_array
        self.avg_trials = avg_trials
        self.flat_trials = flat_trials
        self.repeats = int(eeg_array.shape[1]) if eeg_array is not None and getattr(eeg_array, "ndim", 0) == 4 else 1

    def __len__(self) -> int:
        if self.flat_trials and self.repeats > 1:
            return len(self.records) * self.repeats
        return len(self.records)

    def __getitem__(self, idx: int):
        rec_idx = idx // self.repeats if self.flat_trials and self.repeats > 1 else idx
        trial_idx = idx % self.repeats if self.flat_trials and self.repeats > 1 else None
        rec = self.records[rec_idx]
        eeg = rec.eeg if rec.eeg is not None else self._read_eeg(rec.source_index, trial_idx)
        return {
            "source_index": rec.source_index,
            "image_id": rec.image_id,
            "image_path": rec.image_path,
            "eeg": eeg,
            "label": -1 if rec.label is None else rec.label,
            "label_name": rec.label_name,
        }

    def _read_eeg(self, source_index: int, trial_idx: int | None) -> torch.Tensor:
        eeg = self.eeg_array[source_index]
        if torch.is_tensor(eeg):
            eeg = eeg.float()
            if eeg.ndim == 3:
                eeg = eeg.mean(dim=0) if self.avg_trials or trial_idx is None else eeg[trial_idx]
        else:
            eeg = torch.as_tensor(eeg, dtype=torch.float32)
            if eeg.ndim == 3:
                eeg = eeg.mean(dim=0) if self.avg_trials or trial_idx is None else eeg[trial_idx]
        if tuple(eeg.shape) != (63, 250):
            raise ValueError(f"Expected EEG shape [63,250], got {tuple(eeg.shape)} at source_index={source_index}")
        return eeg


def load_train_dataset(config, avg_trials: bool | None = None) -> EEGImageDataset:
    avg = _cfg(config, "data.avg_trials_train", True) if avg_trials is None else avg_trials
    records, eeg = load_split_records(config, "train", avg_trials=avg, require_images=True)
    return EEGImageDataset(records, eeg, avg_trials=avg, flat_trials=not avg)


def load_val_dataset(config, indices: Sequence[int]) -> Subset:
    dataset = load_train_dataset(config, avg_trials=_cfg(config, "data.avg_trials_train", True))
    return Subset(dataset, list(indices))


def load_test_eeg(config, avg_trials: bool = True, include_image_paths: bool = False) -> EEGImageDataset:
    records, eeg = load_split_records(config, "test", avg_trials=avg_trials, require_images=include_image_paths)
    return EEGImageDataset(records, eeg, avg_trials=avg_trials, flat_trials=not avg_trials)


def load_split_records(config, split: str, avg_trials: bool, require_images: bool) -> tuple[list[EEGRecord], torch.Tensor]:
    data_root = Path(_cfg(config, "paths.data_root"))
    split_path = data_root / f"{split}.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    try:
        payload = torch.load(split_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        payload = torch.load(split_path, map_location="cpu", weights_only=False)
    eeg = payload["eeg"]
    img = np.asarray(payload.get("img", [""] * len(eeg)), dtype=object)
    labels = np.asarray(payload.get("label", [-1] * len(eeg)), dtype=object)
    texts = np.asarray(payload.get("text", [""] * len(eeg)), dtype=object)

    img, labels, texts = _apply_metadata_trial_policy(eeg, img, labels, texts, avg_trials)
    records: list[EEGRecord] = []
    for idx in range(min(len(eeg), len(img))):
        image_path = _resolve_image_path(data_root, str(img[idx]))
        if require_images and image_path and not Path(image_path).exists():
            continue
        records.append(
            EEGRecord(
                source_index=idx,
                image_id=Path(str(img[idx])).stem if str(img[idx]) else f"{split}_{idx:05d}",
                image_path=image_path,
                label=None if str(labels[idx]) == "" else int(labels[idx]),
                label_name=str(texts[idx]),
            )
        )
    return records, eeg


def _apply_metadata_trial_policy(eeg, img, labels, texts, avg_trials: bool):
    if getattr(eeg, "ndim", 0) == 4:
        if avg_trials:
            img = _first_trial(img)
            labels = _first_trial(labels)
            texts = _first_trial(texts)
        else:
            n, r = eeg.shape[:2]
            img = _repeat_or_flatten(img, n, r)
            labels = _repeat_or_flatten(labels, n, r)
            texts = _repeat_or_flatten(texts, n, r)
    elif getattr(eeg, "ndim", 0) != 3:
        raise ValueError(f"Unexpected EEG tensor shape: {eeg.shape}")
    return np.asarray(img).reshape(-1), np.asarray(labels).reshape(-1), np.asarray(texts).reshape(-1)


def _first_trial(values):
    values = np.asarray(values, dtype=object)
    return values[:, 0] if values.ndim >= 2 else values


def _repeat_or_flatten(values, n: int, repeats: int):
    values = np.asarray(values, dtype=object)
    if values.ndim >= 2 and values.shape[1] == repeats:
        return values.reshape(n * repeats)
    return np.repeat(values.reshape(n), repeats)


def _resolve_image_path(data_root: Path, rel_path: str) -> str:
    if not rel_path:
        return ""
    path = data_root / rel_path
    if path.exists():
        return str(path)
    replacements = [
        rel_path.replace("train_images/", "training_images/", 1),
        rel_path.replace("test_images/", "testing_images/", 1),
    ]
    for candidate in replacements:
        path = data_root / candidate
        if path.exists():
            return str(path)
    return str(data_root / rel_path)


def _cfg(config, dotted: str, default=None):
    current = config
    for part in dotted.split("."):
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current
