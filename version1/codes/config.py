"""Single source of truth for project hyperparameters and paths.

Everything else (data loading, model, training, evaluation) reads from
``DEFAULT_CONFIG``.  Override fields via ``Config(**overrides)`` from CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Project root: resolved relative to this file (codes/config.py -> DL_Project/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = Path("/hpc2hdd/home/ckwong627/workdir/models")


@dataclass
class Config:
    # ---------------- paths ----------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "image-eeg-data"
    train_image_dir: Path = PROJECT_ROOT / "image-eeg-data" / "training_images"
    test_image_dir: Path = PROJECT_ROOT / "image-eeg-data" / "test_images"
    eeg_channels_jsonl: Path = PROJECT_ROOT / "image-eeg-data" / "EEG_CHANNELS.jsonl"

    cache_dir: Path = PROJECT_ROOT / "clip_cache"
    ckpt_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # IP-Adapter ships its own CLIP-ViT-H-14 image encoder (1024-d projection).
    # Using it directly removes the need for any projector.
    image_encoder_path: Path = MODELS_ROOT / "IP-Adapter" / "models" / "image_encoder"
    # The IP-Adapter folder is missing preprocessor_config.json, so for the
    # image processor we point to the LAION ViT-H-14 release (same architecture,
    # standard CLIP normalization).
    image_processor_path: Path = MODELS_ROOT / "CLIP-ViT-H-14-laion2B-s32B-b79K"
    sd_model_path: Path = MODELS_ROOT / "stable-diffusion-v1-5"
    ip_adapter_root: Path = MODELS_ROOT / "IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight: str = "ip-adapter_sd15.bin"

    # ---------------- data ----------------
    avg_trials: bool = True
    num_eeg_channels: int = 63
    num_eeg_timesteps: int = 250  # confirmed at runtime; THINGS-EEG default
    num_workers: int = 2

    # ---------------- model ----------------
    embed_dim: int = 1024  # CLIP ViT-H-14 projection dim
    encoder_spatial_dim: int = 128
    encoder_temporal_channels: Tuple[int, int, int] = (192, 256, 320)
    encoder_temporal_kernel: int = 15
    encoder_temporal_stride: int = 2
    encoder_n_transformer_layers: int = 3
    encoder_transformer_heads: int = 8
    encoder_transformer_ffn: int = 640
    encoder_dropout: float = 0.1
    encoder_mlp_hidden: int = 640

    # ---------------- training ----------------
    seed: int = 0
    device: str = "cuda"

    # Phase 1 (coarse)
    phase1_epochs: int = 50
    phase1_batch_size: int = 128
    phase1_lr: float = 3e-4
    phase1_weight_decay: float = 0.05
    phase1_alpha: float = 1.0
    phase1_beta: float = 0.5

    # Phase 2 (fine-tune)
    phase2_epochs: int = 100
    phase2_batch_size: int = 64
    phase2_lr: float = 5e-5
    phase2_weight_decay: float = 0.05
    phase2_alpha: float = 0.5
    phase2_beta: float = 1.0

    # Loss
    init_logit_scale: float = 2.6593  # log(1/0.07), CLIP convention
    learnable_logit_scale: bool = True
    learnable_loss_weights: bool = False  # set True for the auto-balance variant
    init_log_alpha: float = 0.0  # exp(0) = 1
    init_log_beta: float = 0.0

    # ---------------- augmentation ----------------
    aug_jitter: int = 5            # max ± shift
    aug_channel_dropout: float = 0.10
    aug_noise_std: float = 0.02
    aug_time_mask_steps: int = 20
    aug_amp_scale_range: Tuple[float, float] = (0.8, 1.2)
    aug_apply_prob: float = 0.5    # per-aug independent probability

    # Phase 2 uses lighter augmentation; we toggle here so train.py stays simple.
    light_augmentation: bool = False

    # ---------------- reconstruction / evaluation ----------------
    num_test_samples: int = 200
    recon_image_size: int = 512
    recon_eval_size: int = 256
    recon_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    ip_adapter_scale: float = 0.7
    sd_guidance_scale: float = 7.5
    sd_num_inference_steps: int = 20

    def ensure_dirs(self) -> None:
        for d in (self.cache_dir, self.ckpt_dir, self.output_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()
