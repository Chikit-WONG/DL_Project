from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch


def detect_data_format(data_path: str | Path) -> str:
    root = Path(data_path)
    if root.joinpath("train.pt").exists() and root.joinpath("test.pt").exists():
        return "pt_single_subject"
    return "paper_subject_dirs"


def _folder_description(folder_name: str) -> str:
    if "_" in folder_name:
        return folder_name.split("_", 1)[1]
    return folder_name


def build_image_catalog(image_dir: str | Path):
    image_dir = Path(image_dir)
    folders = sorted(d for d in image_dir.iterdir() if d.is_dir())

    texts: List[str] = []
    image_paths: List[str] = []
    stem_to_index: dict[str, int] = {}
    class_indices: List[int] = []
    class_name_to_image_indices: dict[str, List[int]] = {}

    class_idx = 0
    for folder in folders:
        class_name = _folder_description(folder.name)
        folder_image_paths = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not folder_image_paths:
            continue

        texts.append(f"This picture is {class_name}")
        class_name_to_image_indices[class_name] = []
        for image_path in folder_image_paths:
            stem = image_path.stem
            stem_to_index[stem] = len(image_paths)
            image_paths.append(str(image_path))
            class_indices.append(class_idx)
            class_name_to_image_indices[class_name].append(len(image_paths) - 1)
        class_idx += 1

    return texts, image_paths, stem_to_index, class_indices, class_name_to_image_indices


def _stem_class_name(stem: str) -> str:
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def _normalize_image_entries(raw_entries, num_trials: int | None, avg_trials: bool) -> List[str]:
    entries = np.asarray(raw_entries, dtype=object)

    if num_trials is not None:
        if entries.ndim >= 2:
            entries = entries[:, 0] if avg_trials else entries.reshape(-1)
        else:
            entries = entries.reshape(-1) if avg_trials else np.repeat(entries.reshape(-1), num_trials)
    else:
        entries = entries.reshape(-1)

    normalized = []
    for entry in entries.tolist():
        normalized.append(Path(str(entry)).stem)
    return normalized


def _filter_catalog(
    texts: Sequence[str],
    image_paths: Sequence[str],
    class_indices: Sequence[int],
    classes: Sequence[int] | None,
    pictures: Sequence[int] | None,
):
    if classes is None:
        return list(texts), list(image_paths), {idx: idx for idx in range(len(image_paths))}, {
            cls_idx: cls_idx for cls_idx in range(len(texts))
        }

    selected_classes = list(classes)
    label_map = {old_cls: new_cls for new_cls, old_cls in enumerate(selected_classes)}
    selected_texts = [texts[idx] for idx in selected_classes]

    class_to_image_indices: dict[int, List[int]] = {cls: [] for cls in selected_classes}
    for image_idx, class_idx in enumerate(class_indices):
        if class_idx in class_to_image_indices:
            class_to_image_indices[class_idx].append(image_idx)

    old_to_new_image_idx: dict[int, int] = {}
    selected_images: List[str] = []
    if pictures is None:
        for class_idx in selected_classes:
            for old_image_idx in class_to_image_indices[class_idx]:
                old_to_new_image_idx[old_image_idx] = len(selected_images)
                selected_images.append(image_paths[old_image_idx])
    else:
        if len(pictures) != len(selected_classes):
            raise ValueError("pictures must have the same length as classes")
        for class_idx, picture_idx in zip(selected_classes, pictures):
            image_candidates = class_to_image_indices[class_idx]
            if picture_idx >= len(image_candidates):
                raise IndexError(f"picture index {picture_idx} is out of range for class {class_idx}")
            old_image_idx = image_candidates[picture_idx]
            old_to_new_image_idx[old_image_idx] = len(selected_images)
            selected_images.append(image_paths[old_image_idx])

    return selected_texts, selected_images, old_to_new_image_idx, label_map


def load_pt_data(
    *,
    data_path: str | Path,
    split: str,
    avg_trials: bool,
    image_dir: str | Path,
    classes: Sequence[int] | None = None,
    pictures: Sequence[int] | None = None,
):
    pt_path = Path(data_path).joinpath(f"{split}.pt")
    loaded = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    eeg = torch.as_tensor(loaded["eeg"]).float()
    num_trials = eeg.shape[1] if eeg.ndim == 4 else None
    if eeg.ndim == 4:
        eeg = eeg.mean(dim=1) if avg_trials else eeg.reshape(-1, *eeg.shape[2:])
    elif eeg.ndim != 3:
        raise ValueError(f"Unexpected EEG shape {tuple(eeg.shape)} in {pt_path}")

    raw_image_entries = loaded.get("img")
    if raw_image_entries is None:
        raw_image_entries = loaded.get("image_id")
    if raw_image_entries is None:
        raise KeyError(f"{pt_path} must contain 'img' or 'image_id'")

    image_stems = _normalize_image_entries(raw_image_entries, num_trials=num_trials, avg_trials=avg_trials)
    texts, image_paths, stem_to_index, class_indices, class_name_to_image_indices = build_image_catalog(image_dir)
    texts, image_paths, old_to_new_image_idx, label_map = _filter_catalog(
        texts, image_paths, class_indices, classes, pictures
    )

    sample_image_indices: List[int] = []
    labels: List[int] = []
    fallback_count = 0
    skipped_missing_class_count = 0
    for stem in image_stems:
        old_image_index = stem_to_index.get(stem)
        if old_image_index is None:
            class_name = _stem_class_name(stem)
            fallback_candidates = class_name_to_image_indices.get(class_name)
            if not fallback_candidates:
                skipped_missing_class_count += 1
                continue
            old_image_index = fallback_candidates[0]
            fallback_count += 1
        if old_image_index not in old_to_new_image_idx:
            continue
        old_class_index = class_indices[old_image_index]
        sample_image_indices.append(old_to_new_image_idx[old_image_index])
        labels.append(label_map[old_class_index])

    if len(sample_image_indices) != eeg.shape[0]:
        selected_mask = []
        for stem in image_stems:
            old_image_index = stem_to_index.get(stem)
            if old_image_index is None:
                fallback_candidates = class_name_to_image_indices.get(_stem_class_name(stem), [])
                old_image_index = fallback_candidates[0] if fallback_candidates else None
            selected_mask.append(old_image_index in old_to_new_image_idx if old_image_index is not None else False)
        eeg = eeg[torch.tensor(selected_mask, dtype=torch.bool)]

    if len(sample_image_indices) != eeg.shape[0]:
        raise ValueError(
            f"EEG/image mismatch for {pt_path}: {eeg.shape[0]} EEG samples vs {len(sample_image_indices)} image ids"
        )

    if fallback_count:
        print(f"[load_pt_data] used {fallback_count} class-level image fallbacks for {pt_path.name}")
    if skipped_missing_class_count:
        print(f"[load_pt_data] skipped {skipped_missing_class_count} samples with no available class images for {pt_path.name}")

    return {
        "eeg": eeg,
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
        "images": image_paths,
        "sample_image_indices": sample_image_indices,
        "sample_text_indices": labels,
        "times": loaded.get("times"),
        "channel_names": loaded.get("ch_names"),
    }
