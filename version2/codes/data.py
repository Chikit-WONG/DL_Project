from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EEGSample:
    source_index: int
    image_id: str
    image_path: Path
    eeg: torch.Tensor
    label: int
    label_name: str
    subject_id: int


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _load_pickle_like(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=False)
    raise ValueError(f"Unsupported array file: {path}")


def _coerce_array(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, list):
        return np.asarray(obj)
    raise TypeError(f"Cannot coerce object of type {type(obj)} to ndarray")


def _unwrap_to_eeg_array(obj: Any) -> np.ndarray:
    if isinstance(obj, dict):
        preferred = [
            "eeg",
            "eegs",
            "data",
            "x",
            "xtrain",
            "xtest",
            "train_data",
            "test_data",
            "signals",
        ]
        normalized = {_normalize_key(k): v for k, v in obj.items()}
        for key in preferred:
            if key in normalized:
                return _coerce_array(normalized[key])
        for value in obj.values():
            try:
                arr = _coerce_array(value)
            except TypeError:
                continue
            if arr.ndim >= 3:
                return arr
        raise KeyError("No EEG-like array found in dictionary payload")
    return _coerce_array(obj)


def _move_axis(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    return np.moveaxis(arr, src, dst)


def _normalize_eeg_shape(arr: np.ndarray, num_channels: int = 63) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim < 3:
        raise ValueError(f"EEG array must have >=3 dims, got {arr.shape}")

    if arr.ndim == 4:
        channel_axis_candidates = [i for i, size in enumerate(arr.shape) if size == num_channels]
        if not channel_axis_candidates:
            raise ValueError(f"Could not find channel axis in shape {arr.shape}")
        channel_axis = channel_axis_candidates[0]
        arr = _move_axis(arr, channel_axis, -2)

        trial_axis = None
        for axis, size in enumerate(arr.shape[:-2]):
            if size in {4, 10, 20, 40, 80}:
                trial_axis = axis
                break
        if trial_axis is None:
            trial_axis = 1 if arr.shape[1] <= 128 else 0
        if trial_axis != 1:
            arr = _move_axis(arr, trial_axis, 1)
        if arr.shape[0] < arr.shape[1]:
            return arr.astype(np.float32)
        return arr.astype(np.float32)

    channel_axis_candidates = [i for i, size in enumerate(arr.shape) if size == num_channels]
    if not channel_axis_candidates:
        raise ValueError(f"Could not find channel axis in shape {arr.shape}")
    channel_axis = channel_axis_candidates[0]
    arr = _move_axis(arr, channel_axis, -2)
    if arr.shape[-1] < arr.shape[-2]:
        arr = _move_axis(arr, -1, 0)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected EEG shape after normalization: {arr.shape}")
    return arr.astype(np.float32)


def _read_subject_ids(path: Path | None, length: int, max_subjects: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros(length, dtype=np.int64)
    payload = _load_pickle_like(path)
    if isinstance(payload, dict):
        normalized = {_normalize_key(k): v for k, v in payload.items()}
        for key in ("subject", "subjects", "subjectid", "subjectids"):
            if key in normalized:
                values = np.asarray(normalized[key]).reshape(-1)
                return np.mod(values.astype(np.int64), max_subjects)
    values = np.asarray(payload).reshape(-1)
    return np.mod(values.astype(np.int64), max_subjects)


def _split_pt_path(data_dir: Path, split: str) -> Path:
    path = data_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Expected split file not found: {path}")
    return path


def load_split_records(cfg, split: str, limit: int | None = None) -> list[EEGSample]:
    split_path = _split_pt_path(cfg.data_dir, split)
    loaded = torch.load(str(split_path), map_location="cpu", weights_only=False)
    eeg_array = np.asarray(loaded["eeg"], dtype=np.float32)
    image_entries = np.asarray(loaded["img"])
    label_entries = np.asarray(loaded["label"])
    text_entries = np.asarray(loaded["text"])
    session_entries = np.asarray(loaded["session"])

    if eeg_array.ndim == 4 and cfg.avg_trials:
        eeg_array = eeg_array.mean(axis=1)
    elif eeg_array.ndim == 4:
        eeg_array = eeg_array[:, 0]
    elif eeg_array.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {eeg_array.shape} in {split_path}")

    if cfg.avg_trials:
        if image_entries.ndim == 2:
            image_entries = image_entries[:, 0]
        if label_entries.ndim == 2:
            label_entries = label_entries[:, 0]
        if text_entries.ndim == 2:
            text_entries = text_entries[:, 0]
        if session_entries.ndim == 2:
            session_entries = session_entries[:, 0]
    else:
        image_entries = image_entries.reshape(-1)
        label_entries = label_entries.reshape(-1)
        text_entries = text_entries.reshape(-1)
        session_entries = session_entries.reshape(-1)

    image_entries = image_entries.reshape(-1)
    label_entries = label_entries.reshape(-1)
    text_entries = text_entries.reshape(-1)
    session_entries = session_entries.reshape(-1)

    n = min(
        eeg_array.shape[0],
        len(image_entries),
        len(label_entries),
        len(text_entries),
        len(session_entries),
    )
    if limit is not None:
        n = min(n, limit)

    records: list[EEGSample] = []
    missing_count = 0
    for idx in range(n):
        rel_image_path = str(image_entries[idx])
        image_path = cfg.data_dir / rel_image_path
        if not image_path.exists() and rel_image_path.startswith("train_images/"):
            image_path = cfg.data_dir / rel_image_path.replace("train_images/", "training_images/", 1)
        if not image_path.exists():
            missing_count += 1
            continue
        label = int(label_entries[idx])
        label_name = str(text_entries[idx])
        image_id = image_path.stem
        eeg_tensor = torch.from_numpy(eeg_array[idx]).float()
        if eeg_tensor.ndim != 2:
            eeg_tensor = eeg_tensor.view(cfg.num_eeg_channels, -1)
        records.append(
            EEGSample(
                source_index=idx,
                image_id=image_id,
                image_path=image_path,
                eeg=eeg_tensor,
                label=label,
                label_name=label_name,
                subject_id=int(float(session_entries[idx])) % cfg.max_subjects,
            )
        )
    if missing_count > 0:
        print(f"[load_split_records] split={split} skipped {missing_count} missing images")
    return records


class EEGImageDataset(Dataset):
    def __init__(self, cfg, split: str, limit: int | None = None) -> None:
        self.cfg = cfg
        self.split = split
        self.records = load_split_records(cfg, split=split, limit=limit)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.records[index]
        return {
            "index": index,
            "source_index": sample.source_index,
            "image_id": sample.image_id,
            "image_path": str(sample.image_path),
            "eeg": sample.eeg.clone(),
            "label": sample.label,
            "label_name": sample.label_name,
            "subject_id": sample.subject_id,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "source_index": torch.tensor([item["source_index"] for item in batch], dtype=torch.long),
        "image_id": [item["image_id"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "eeg": torch.stack([item["eeg"] for item in batch], dim=0),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "label_name": [item["label_name"] for item in batch],
        "subject_id": torch.tensor([item["subject_id"] for item in batch], dtype=torch.long),
    }


def build_dataloader(
    cfg,
    split: str,
    batch_size: int,
    shuffle: bool,
    limit: int | None = None,
) -> DataLoader:
    dataset = EEGImageDataset(cfg=cfg, split=split, limit=limit)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )


def load_image_rgb(path: str | Path, size: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size))
    return image


def describe_data_layout(cfg) -> dict[str, Any]:
    train_records = load_split_records(cfg, "train", limit=8)
    test_records = load_split_records(cfg, "test", limit=8)
    return {
        "data_dir": str(cfg.data_dir),
        "train_count_preview": len(train_records),
        "test_count_preview": len(test_records),
        "train_image_example": train_records[0].image_path.as_posix() if train_records else "",
        "test_image_example": test_records[0].image_path.as_posix() if test_records else "",
        "eeg_shape_preview": list(train_records[0].eeg.shape) if train_records else [],
    }
