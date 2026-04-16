"""EEG dataset loading + augmentation.

The ``load_eeg_dataset`` function is copied verbatim from the TA's
``sample_codes/eeg_project_sample_code.ipynb`` (cell 4). Everything below
it is project-specific glue: pairing each EEG sample with a cached CLIP
embedding, and applying the 5-way augmentation described in the plan.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import datasets
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# load_eeg_dataset (verbatim from sample notebook cell 4)
# ---------------------------------------------------------------------------
def _selected_channel_indices_from_jsonl(
    selected_channels: Union[str, Sequence[str]],
    eeg_channel_jsonl: Union[str, Path],
) -> List[int]:
    if isinstance(selected_channels, str):
        selected_channels = [selected_channels]
    selected_channels = list(selected_channels)

    channel_names: List[str] = []
    with open(eeg_channel_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            name = item.get("name") or item.get("channel_name") or item.get("label")
            if name is None:
                raise KeyError(
                    "Each JSONL record must contain one of: 'name', 'channel_name', or 'label'."
                )
            channel_names.append(str(name))

    name_to_index = {name: idx for idx, name in enumerate(channel_names)}
    missing = [ch for ch in selected_channels if ch not in name_to_index]
    if missing:
        raise ValueError(f"Unknown EEG channels: {missing}")
    return [name_to_index[ch] for ch in selected_channels]


def load_eeg_dataset(
    *,
    data_directory: Union[str, Path],
    split: Literal["train", "test"],
    avg_trials: bool = True,
    selected_channels: Optional[Union[str, Sequence[str]]] = None,
    eeg_channel_jsonl: Union[str, Path] = "image-eeg-data/EEG_CHANNELS.jsonl",
) -> datasets.Dataset:
    """Build a Hugging Face dataset for the released EEG data."""
    pt_path = Path(data_directory).joinpath(f"{split}.pt")
    loaded = torch.load(str(pt_path), weights_only=False)

    x = torch.as_tensor(loaded["eeg"])  # [N, TRIAL, C, T] or [N, C, T]
    if x.ndim == 4:
        if avg_trials:
            x = x.mean(dim=1)
        else:
            x = x.reshape(-1, *x.shape[2:])
    elif x.ndim != 3:
        raise ValueError(f"Unexpected EEG shape: {tuple(x.shape)} in {pt_path}")

    if selected_channels is not None:
        sel_idx = _selected_channel_indices_from_jsonl(selected_channels, eeg_channel_jsonl)
        x = x[:, sel_idx, :]

    imgs = np.array(loaded["img"])
    if avg_trials:
        if imgs.ndim == 2:
            imgs = imgs[:, 0]
        imgs = imgs.reshape(-1)[: x.shape[0]]
    else:
        imgs = imgs.reshape(-1)

    image_ids = [Path(p).stem for p in imgs.tolist()]
    if len(image_ids) != x.shape[0]:
        raise ValueError(
            f"EEG/image mismatch: {x.shape[0]} vs {len(image_ids)} for {pt_path}"
        )

    x_np = x.float().cpu().numpy()
    C, T = x_np.shape[1], x_np.shape[2]

    features = datasets.Features(
        {
            "eeg": datasets.Array2D(shape=(C, T), dtype="float32"),
            "image_id": datasets.Value("string"),
        }
    )
    return datasets.Dataset.from_dict(
        {"eeg": list(x_np), "image_id": image_ids},
        features=features,
    )


# ---------------------------------------------------------------------------
# Augmentation (5 independent transforms, each applied with prob p)
# ---------------------------------------------------------------------------
class EEGAugmentation:
    """Composable, channel-aware EEG augmentation.

    Each EEG sample has shape [C, T]. All transforms preserve that shape.
    """

    def __init__(
        self,
        jitter: int = 5,
        channel_dropout: float = 0.10,
        noise_std: float = 0.02,
        time_mask_steps: int = 20,
        amp_scale_range=(0.8, 1.2),
        apply_prob: float = 0.5,
        light: bool = False,
    ):
        self.jitter = jitter
        self.channel_dropout = channel_dropout
        self.noise_std = noise_std
        self.time_mask_steps = time_mask_steps
        self.amp_scale_range = amp_scale_range
        self.apply_prob = apply_prob
        # In phase 2 we drop the heavier transforms (channel dropout, time mask, amp scale).
        self.light = light

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"EEG must be [C, T], got {tuple(x.shape)}")
        x = x.clone()
        C, T = x.shape

        if random.random() < self.apply_prob and self.jitter > 0:
            shift = random.randint(-self.jitter, self.jitter)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
                if shift > 0:
                    x[..., :shift] = 0.0
                else:
                    x[..., shift:] = 0.0

        if random.random() < self.apply_prob and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        if not self.light:
            if random.random() < self.apply_prob and self.channel_dropout > 0:
                mask = torch.rand(C) > self.channel_dropout
                x = x * mask.unsqueeze(-1).to(x.dtype)

            if random.random() < self.apply_prob and self.time_mask_steps > 0:
                mask_len = random.randint(1, self.time_mask_steps)
                start = random.randint(0, max(0, T - mask_len))
                x[..., start : start + mask_len] = 0.0

            if random.random() < self.apply_prob and self.amp_scale_range:
                lo, hi = self.amp_scale_range
                scale = random.uniform(lo, hi)
                x = x * scale

        return x


# ---------------------------------------------------------------------------
# Dataset wrapper: pair EEG with a cached CLIP image embedding
# ---------------------------------------------------------------------------
class EEGImageDataset(Dataset):
    """Wraps a HF dataset and joins each sample with its cached CLIP feature.

    The cached features are produced once by ``cache_clip_features.py`` and
    stored as ``{image_id: tensor[1024]}``.
    """

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        clip_features: Dict[str, torch.Tensor],
        augmentation: Optional[EEGAugmentation] = None,
    ):
        self.clip_features = clip_features
        self.augmentation = augmentation

        # Filter out EEG samples whose image is not on disk / not cached.
        # The course dataset provides ~7968 training images while train.pt
        # may reference more; we silently drop the extras.
        hf = hf_dataset.with_format(None)  # numpy format for fast column access
        all_ids: List[str] = hf["image_id"]
        valid_indices = [i for i, img_id in enumerate(all_ids) if img_id in clip_features]
        n_dropped = len(all_ids) - len(valid_indices)
        if n_dropped > 0:
            print(f"[EEGImageDataset] dropped {n_dropped}/{len(all_ids)} samples "
                  f"(image not cached); keeping {len(valid_indices)}")
        self.hf_dataset = hf_dataset.select(valid_indices).with_format("torch")

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        row = self.hf_dataset[idx]
        eeg = row["eeg"].float()  # [C, T]
        if self.augmentation is not None:
            eeg = self.augmentation(eeg)
        image_id = row["image_id"]
        clip_target = self.clip_features[image_id].float()
        return {"eeg": eeg, "clip": clip_target, "image_id": image_id}


def collate_eeg_batch(batch):
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0),
        "clip": torch.stack([b["clip"] for b in batch], dim=0),
        "image_id": [b["image_id"] for b in batch],
    }
