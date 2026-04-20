"""
Generate reconstructed images from EEG test data using a trained ATMS encoder
and IP-Adapter with Stable Diffusion v1.5.

Pipeline:
  EEG (test, avg_trials=True) → ATMS encoder → 1024-dim CLIP ViT-H-14 embedding
  → ImageProjection (from ip-adapter_sd15.bin) → 4 cross-attention tokens [4, 768]
  → SD v1.5 UNet (IP-Adapter weights) → generated image (512×512)

Usage (after ATMS_reconstruction.py training finishes):
  python Generation/generate_reconstructions.py \
    --checkpoint ./models/contrast/ATMS/sub-01/<run>/40.pth \
    --output_dir  ./outputs/reconstructions/<run>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from diffusers import StableDiffusionPipeline
from diffusers.models.embeddings import ImageProjection

from models.data_bridge import load_pt_data
from models.subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from models.subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.subject_layers.Embed import DataEmbedding
from models.loss import ClipLoss
from einops.layers.torch import Rearrange


# ──────────────────────────────────────────────────────────────────────────────
# ATMS model (identical to ATMS_reconstruction.py)
# ──────────────────────────────────────────────────────────────────────────────

class Config:
    task_name = 'classification'; seq_len = 250; pred_len = 250
    output_attention = False; d_model = 250; embed = 'timeF'; freq = 'h'
    dropout = 0.25; factor = 1; n_heads = 4; e_layers = 1; d_ff = 256
    activation = 'gelu'; enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, cfg, num_subjects=2):
        super().__init__()
        self.enc_embedding = DataEmbedding(
            cfg.seq_len, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout,
            joint_train=False, num_subjects=num_subjects)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(FullAttention(False, cfg.factor,
                    attention_dropout=cfg.dropout,
                    output_attention=cfg.output_attention),
                    cfg.d_model, cfg.n_heads),
                cfg.d_model, cfg.d_ff,
                dropout=cfg.dropout, activation=cfg.activation)
             for _ in range(cfg.e_layers)],
            norm_layer=nn.LayerNorm(cfg.d_model))

    def forward(self, x, x_mark, subject_ids=None):
        enc_out, _ = self.encoder(self.enc_embedding(x, x_mark, subject_ids))
        return enc_out[:, :63, :]


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)), nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.Conv2d(40, 40, (63, 1)), nn.BatchNorm2d(40), nn.ELU(),
            nn.Dropout(0.5))
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'))

    def forward(self, x):
        return self.projection(self.tsconv(x.unsqueeze(1)))


class ResidualAdd(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x, **kw): return x + self.fn(x, **kw)


class FlattenHead(nn.Sequential):
    def forward(self, x): return x.contiguous().view(x.size(0), -1)


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(nn.GELU(), nn.Linear(proj_dim, proj_dim),
                                      nn.Dropout(drop_proj))),
            nn.LayerNorm(proj_dim))


class ATMS(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Config()
        self.encoder = iTransformer(cfg, num_subjects=10)
        self.subject_wise_linear = nn.ModuleList(
            [nn.Linear(cfg.d_model, 250) for _ in range(2)])
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        return self.proj_eeg(self.enc_eeg(x))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _extract_sub_id(s: str) -> int:
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else 1


def generate(args):
    device = torch.device(args.device)

    # ── 1. Load test EEG ─────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    loaded = load_pt_data(
        data_path=data_path,
        split="test",
        avg_trials=True,         # required by course
        image_dir=data_path / "test_images",
    )
    eeg_data = loaded["eeg"].float()             # [200, 63, 250]
    gt_paths = [loaded["images"][i] for i in loaded["sample_image_indices"]]
    print(f"Test EEG: {eeg_data.shape}, {len(gt_paths)} GT images")

    # ── 2. Run ATMS encoder ───────────────────────────────────────────────────
    model = ATMS().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    sub_id = _extract_sub_id(args.subject)
    all_embeds = []
    bs = args.encode_batch_size
    with torch.no_grad():
        for i in range(0, len(eeg_data), bs):
            batch = eeg_data[i:i + bs].to(device)
            n = batch.size(0)
            sids = torch.full((n,), sub_id, dtype=torch.long, device=device)
            emb = model(batch, sids).float()
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embeds.append(emb.cpu())
    eeg_embeds = torch.cat(all_embeds, dim=0)    # [200, 1024]
    print(f"EEG embeddings: {eeg_embeds.shape}")

    # Free ATMS model from GPU before loading SD to avoid memory/context conflicts
    del model
    torch.cuda.empty_cache()

    # ── 3. Load SD v1.5 + IP-Adapter ─────────────────────────────────────────
    print("Loading Stable Diffusion v1.5 …")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    print("Loading IP-Adapter into UNet …")
    # image_encoder_folder=None: we bypass the CLIP image encoder entirely;
    # we provide pre-computed embeddings via ip_adapter_image_embeds.
    pipe.load_ip_adapter(
        args.ip_adapter_dir,
        subfolder="",
        weight_name=args.ip_adapter_weight,
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    # Get IP-Adapter projection dtype (the UNet's ImageProjection does 1024→4×768 internally)
    image_proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    ip_dtype = next(image_proj.parameters()).dtype
    print(f"ImageProjection: {image_proj}")

    # ── 4. Prepare raw EEG embeddings for IP-Adapter ──────────────────────────
    # diffusers 0.37.1 check_inputs requires 3D tensors in ip_adapter_image_embeds.
    # Pass [2, 1, 1024]: Linear(1024→3072) → reshape(-1,4,768) → [2, 4, 768].
    neg_embed = torch.zeros(1, 1, 1024, dtype=ip_dtype, device=device)  # [1, 1, 1024]
    eeg_embeds_ip = eeg_embeds.unsqueeze(1).to(device=device, dtype=ip_dtype)  # [200, 1, 1024]

    # ── 5. Generate images ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    gt_dir = output_dir / "ground_truth"
    gen_dir = output_dir / "generated"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    gt_tensors, gen_tensors = [], []

    # Save GT images
    for idx, p in enumerate(gt_paths):
        img = Image.open(p).convert("RGB").resize((256, 256), Image.LANCZOS)
        img.save(gt_dir / f"{idx:04d}.png")
        gt_tensors.append(to_tensor(img))

    print(f"Generating {len(eeg_embeds)} images …")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for idx in range(len(eeg_embeds)):
        # [neg, pos] → [2, 1, 1024]; ImageProjection: [2,1,1024]→[2,1,3072]→[2,4,768]
        ip_embeds_i = torch.cat(
            [neg_embed, eeg_embeds_ip[idx:idx+1]], dim=0
        )                                       # [2, 1, 1024]

        with torch.no_grad():
            result = pipe(
                prompt="",
                negative_prompt="low quality, blurry, distorted",
                ip_adapter_image_embeds=[ip_embeds_i],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=512, width=512,
            )
        gen_img = result.images[0].resize((256, 256), Image.LANCZOS)
        gen_img.save(gen_dir / f"{idx:04d}.png")
        gen_tensors.append(to_tensor(gen_img))

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  {idx + 1}/{len(eeg_embeds)} done")

    # ── 6. Save tensors for eval_reconstruction_metrics.py ───────────────────
    gt_tensor = torch.stack(gt_tensors)    # [200, 3, 256, 256]
    gen_tensor = torch.stack(gen_tensors)  # [200, 3, 256, 256]
    torch.save({"real": gt_tensor, "fake": gen_tensor},
               output_dir / "recon_tensors.pt")

    print(f"\nDone.")
    print(f"  Ground truth : {gt_dir}")
    print(f"  Generated    : {gen_dir}")
    print(f"  Tensors      : {output_dir / 'recon_tensors.pt'}")


def main():
    BASE = Path(ROOT_DIR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to ATMS reconstruction checkpoint (.pth)")
    parser.add_argument("--output_dir",
                        default=str(BASE / "outputs" / "reconstructions"),
                        help="Directory for output images and tensors")
    parser.add_argument("--data_path",
                        default="/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning"
                                "/ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--sd_model_path",
                        default="/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5")
    parser.add_argument("--ip_adapter_dir",
                        default="/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter/models",
                        help="Directory containing ip-adapter_sd15.bin")
    parser.add_argument("--ip_adapter_weight",
                        default="ip-adapter_sd15.bin",
                        help="IP-Adapter weight filename (uses ViT-H-14 embeddings)")
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
