from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_DATA_ROOT = Path(
    "/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/"
    "ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data"
)
MODELS_ROOT = Path("/hpc2hdd/home/ckwong627/workdir/models")
VERSION1_ROOT = PROJECT_ROOT.parent / "version1"


@dataclass
class Config:
    project_root: Path = PROJECT_ROOT
    version1_root: Path = VERSION1_ROOT
    data_dir: Path = SHARED_DATA_ROOT
    train_image_dir: Path = SHARED_DATA_ROOT / "training_images"
    test_image_dir: Path = SHARED_DATA_ROOT / "test_images"
    eeg_channels_jsonl: Path = SHARED_DATA_ROOT / "EEG_CHANNELS.jsonl"

    code_dir: Path = PROJECT_ROOT / "codes"
    cache_dir: Path = PROJECT_ROOT / "cache"
    ckpt_dir: Path = PROJECT_ROOT / "checkpoints"
    result_dir: Path = PROJECT_ROOT / "results"
    log_dir: Path = PROJECT_ROOT / "logs"
    plan_dir: Path = PROJECT_ROOT / "plan"
    slurm_dir: Path = PROJECT_ROOT / "slurm_scripts"

    seed: int = 0
    device: str = "cuda"
    num_workers: int = 1
    avg_trials: bool = True
    max_subjects: int = 16
    num_eeg_channels: int = 63
    num_eeg_timesteps: int = 250
    semantic_dim: int = 1024
    structural_dim: int = 4 * 64 * 64

    spatial_hidden: int = 128
    temporal_hidden: int = 256
    model_dim: int = 384
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ffn_dim: int = 768
    dropout: float = 0.1

    warmup_epochs: int = 20
    multitarget_epochs: int = 20
    finetune_epochs: int = 20
    warmup_batch_size: int = 96
    multitarget_batch_size: int = 64
    finetune_batch_size: int = 64
    warmup_lr: float = 3e-4
    multitarget_lr: float = 2e-4
    finetune_lr: float = 1e-4
    weight_decay: float = 0.02
    label_smoothing: float = 0.0
    grad_clip_norm: float = 1.0

    prior_epochs: int = 40
    prior_batch_size: int = 128
    prior_lr: float = 2e-4
    prior_timesteps: int = 100
    prior_hidden_dim: int = 2048

    h14_model_dir: Path = MODELS_ROOT / "CLIP-ViT-H-14-laion2B-s32B-b79K"
    b32_model_dir: Path = MODELS_ROOT / "CLIP-ViT-B-32-laion2B-s34B-b79K"
    sd15_dir: Path = MODELS_ROOT / "stable-diffusion-v1-5"
    ip_adapter_root: Path = MODELS_ROOT / "IP-Adapter"
    ip_adapter_sd15_subfolder: str = "models"
    ip_adapter_sd15_weight: str = "ip-adapter_sd15.bin"
    ip_adapter_sdxl_subfolder: str = "sdxl_models"
    ip_adapter_sdxl_weight: str = "ip-adapter_sdxl_vit-h.safetensors"
    sdxl_image_encoder_dir: Path = MODELS_ROOT / "IP-Adapter" / "sdxl_models" / "image_encoder"
    sdxl_turbo_dir: Path = MODELS_ROOT / "sdxl-turbo"

    recon_height: int = 512
    recon_width: int = 512
    recon_eval_size: int = 256
    recon_num_inference_steps: int = 4
    recon_guidance_scale: float = 0.0
    recon_ip_adapter_scale: float = 0.8
    recon_img2img_strength: float = 0.5
    recon_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    literature_rows: list[dict[str, str]] = field(
        default_factory=lambda: [
            {
                "model": "Version1 Joint Baseline",
                "source": "local",
                "top1": "13.5%",
                "top5": "36.5%",
                "ssim": "0.276",
                "clip": "0.708",
            },
            {
                "model": "Version1 Retrieval-only",
                "source": "local",
                "top1": "14.5%",
                "top5": "34.5%",
                "ssim": "0.198",
                "clip": "0.658",
            },
            {
                "model": "Version1 Reconstruction-only",
                "source": "local",
                "top1": "9.0%",
                "top5": "24.0%",
                "ssim": "0.275",
                "clip": "0.753",
            },
            {
                "model": "Consensus Target",
                "source": "version2 plan target",
                "top1": ">=20%",
                "top5": ">=48%",
                "ssim": ">=0.310",
                "clip": ">=0.760",
            },
        ]
    )

    def ensure_dirs(self) -> None:
        for directory in (
            self.code_dir,
            self.cache_dir,
            self.ckpt_dir,
            self.result_dir,
            self.log_dir,
            self.plan_dir,
            self.slurm_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def stage_params(self, stage: str) -> dict[str, float | int]:
        stage = stage.lower()
        if stage == "warmup":
            return {
                "epochs": self.warmup_epochs,
                "batch_size": self.warmup_batch_size,
                "lr": self.warmup_lr,
                "w_h14_nce": 1.0,
                "w_h14_mse": 0.5,
                "w_b32_nce": 0.0,
                "w_rn50_nce": 0.0,
                "w_struct": 0.0,
                "w_hard": 0.0,
                "w_supcon": 0.0,
            }
        if stage == "multitarget":
            return {
                "epochs": self.multitarget_epochs,
                "batch_size": self.multitarget_batch_size,
                "lr": self.multitarget_lr,
                "w_h14_nce": 1.0,
                "w_h14_mse": 0.5,
                "w_b32_nce": 0.3,
                "w_rn50_nce": 0.3,
                "w_struct": 0.3,
                "w_hard": 0.0,
                "w_supcon": 0.0,
            }
        if stage == "finetune":
            return {
                "epochs": self.finetune_epochs,
                "batch_size": self.finetune_batch_size,
                "lr": self.finetune_lr,
                "w_h14_nce": 1.0,
                "w_h14_mse": 0.5,
                "w_b32_nce": 0.3,
                "w_rn50_nce": 0.3,
                "w_struct": 0.3,
                "w_hard": 0.1,
                "w_supcon": 0.2,
            }
        raise ValueError(f"Unsupported stage: {stage}")


DEFAULT_CONFIG = Config()
