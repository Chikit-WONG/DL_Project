from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import DEFAULT_CONFIG
from data import build_dataloader
from model import EEGEncoderV2, PriorUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_ckpt", required=True, type=str)
    parser.add_argument("--prior_ckpt", required=False, type=str, default=None)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONFIG.recon_seeds))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_encoder(cfg, ckpt_path: str, device: torch.device) -> EEGEncoderV2:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EEGEncoderV2(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def load_prior(cfg, ckpt_path: str | None, device: torch.device) -> PriorUNet | None:
    if not ckpt_path:
        return None
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PriorUNet(embed_dim=cfg.semantic_dim, hidden_dim=cfg.prior_hidden_dim).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def load_pipeline(cfg, device: torch.device):
    if cfg.sdxl_turbo_dir.exists():
        try:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                str(cfg.sdxl_turbo_dir),
                torch_dtype=torch.float16,
                variant="fp16",
            ).to(device)
        except Exception:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                str(cfg.sdxl_turbo_dir),
                torch_dtype=torch.float16,
            ).to(device)
        pipe.load_ip_adapter(
            str(cfg.ip_adapter_root),
            subfolder=cfg.ip_adapter_sdxl_subfolder,
            weight_name=cfg.ip_adapter_sdxl_weight,
        )
        pipe.set_ip_adapter_scale(cfg.recon_ip_adapter_scale)
        family = "sdxl_turbo"
        return pipe, family

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        str(cfg.sd15_dir),
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.load_ip_adapter(
        str(cfg.ip_adapter_root),
        subfolder=cfg.ip_adapter_sd15_subfolder,
        weight_name=cfg.ip_adapter_sd15_weight,
    )
    pipe.set_ip_adapter_scale(cfg.recon_ip_adapter_scale)
    family = "sd15_fallback"
    return pipe, family


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    return transforms.ToPILImage()(tensor)


def pil_to_eval_tensor(image: Image.Image, size: int) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
    )(image)


@torch.no_grad()
def decode_blurry_images(vae: AutoencoderKL, latent_flat: torch.Tensor) -> torch.Tensor:
    latents = latent_flat.view(latent_flat.size(0), 4, 64, 64)
    latents = latents / vae.config.scaling_factor
    decoded = vae.decode(latents).sample
    decoded = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
    return decoded


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    encoder = load_encoder(cfg, args.encoder_ckpt, device)
    prior = load_prior(cfg, args.prior_ckpt, device)
    pipe, pipeline_family = load_pipeline(cfg, device)
    vae = AutoencoderKL.from_pretrained(str(cfg.sd15_dir), subfolder="vae").to(device)
    vae.eval()

    test_loader = build_dataloader(cfg, "test", batch_size=1, shuffle=False, limit=args.limit)
    all_seed_images = torch.empty(
        (len(args.seeds), len(test_loader.dataset), 3, cfg.recon_eval_size, cfg.recon_eval_size),
        dtype=torch.float32,
    )
    image_ids = []

    for seed_idx, seed in enumerate(args.seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        for i, batch in enumerate(tqdm(test_loader, desc=f"seed {seed}")):
            eeg = batch["eeg"].to(device)
            subject_ids = batch["subject_id"].to(device)
            outputs = encoder(eeg, subject_ids=subject_ids)
            semantic = outputs["semantic"]
            if prior is not None:
                semantic = prior.sample(semantic, num_steps=15, guidance_scale=1.25)
            structural = outputs["structural"]
            blurry = decode_blurry_images(vae, structural)[0]
            blurry_pil = tensor_to_pil(blurry)

            cond = semantic.to(device=device, dtype=torch.float16).unsqueeze(1)
            uncond = torch.zeros_like(cond)
            ip_embeds = [torch.cat([uncond, cond], dim=0)]

            result = pipe(
                prompt="",
                negative_prompt="",
                image=blurry_pil,
                ip_adapter_image_embeds=ip_embeds,
                strength=cfg.recon_img2img_strength,
                guidance_scale=cfg.recon_guidance_scale,
                num_inference_steps=cfg.recon_num_inference_steps,
                generator=generator,
            )
            generated = result.images[0]
            all_seed_images[seed_idx, i] = pil_to_eval_tensor(generated, cfg.recon_eval_size)
            if seed_idx == 0:
                image_ids.append(batch["image_id"][0])

    out_path = cfg.result_dir / f"recon_images_{args.tag}.pt"
    torch.save(
        {
            "images": all_seed_images,
            "image_ids": image_ids,
            "seeds": args.seeds,
            "pipeline_family": pipeline_family,
            "encoder_ckpt": args.encoder_ckpt,
            "prior_ckpt": args.prior_ckpt,
        },
        out_path,
    )
    with (cfg.result_dir / f"recon_meta_{args.tag}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "tag": args.tag,
                "seeds": args.seeds,
                "pipeline_family": pipeline_family,
                "count": len(image_ids),
            },
            handle,
            indent=2,
        )
    print(f"Saved reconstructions to {out_path}")


if __name__ == "__main__":
    main()
