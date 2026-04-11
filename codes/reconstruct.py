"""Image reconstruction from EEG via IP-Adapter + Stable Diffusion v1.5.

For each test EEG sample we encode it with the trained EEG encoder, treat
the resulting vector as a CLIP-image embedding, and pass it to IP-Adapter
through ``StableDiffusionPipeline.__call__(ip_adapter_image_embeds=...)``.

Generations for all 10 seeds are saved as a single ``.pt`` tensor of shape
``[num_seeds, 200, 3, 256, 256]`` so ``evaluate.py`` can run the official
``eval_images`` function directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DEFAULT_CONFIG, Config  # noqa: E402
from data import load_eeg_dataset  # noqa: E402
from model import UnifiedModel  # noqa: E402
from utils import set_seed, setup_logger  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--tag", type=str, default="recon")
    p.add_argument("--ip_scale", type=float, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None,
                   help="Limit number of test samples (for quick debug)")
    return p.parse_args()


def load_model(cfg: Config, ckpt_path: Path, device):
    model = UnifiedModel(cfg, alpha=1.0, beta=1.0, learnable_loss_weights=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def load_pipeline(cfg: Config, device, ip_scale: float):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        str(cfg.sd_model_path),
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.load_ip_adapter(
        str(cfg.ip_adapter_root),
        subfolder=cfg.ip_adapter_subfolder,
        weight_name=cfg.ip_adapter_weight,
    )
    pipe.set_ip_adapter_scale(ip_scale)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def encode_test_eegs(model, cfg: Config, device, num_samples: int | None = None):
    test_ds = load_eeg_dataset(
        data_directory=cfg.data_dir,
        split="test",
        avg_trials=cfg.avg_trials,
        eeg_channel_jsonl=cfg.eeg_channels_jsonl,
    ).with_format("torch")

    if num_samples is not None:
        test_ds = test_ds.select(range(num_samples))

    embs = []
    image_ids = []
    for i in range(len(test_ds)):
        row = test_ds[i]
        eeg = row["eeg"].float().unsqueeze(0).to(device)
        emb = model.encode(eeg)  # [1, 1024], unnormalized
        embs.append(emb.squeeze(0).cpu())
        image_ids.append(row["image_id"])
    embs_tensor = torch.stack(embs, dim=0)  # [N, 1024]
    return embs_tensor, image_ids


def to_image_tensor(pil_img: Image.Image, size: int) -> torch.Tensor:
    """Resize a PIL image to ``size`` and convert to a [3, size, size] tensor in [0, 1]."""
    tx = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    return tx(pil_img)


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log_path = cfg.output_dir / f"reconstruct_{args.tag}.log"
    logger = setup_logger(f"reconstruct_{args.tag}", log_path)
    logger.info(f"args: {vars(args)}")

    ip_scale = args.ip_scale if args.ip_scale is not None else cfg.ip_adapter_scale
    guidance = args.guidance_scale if args.guidance_scale is not None else cfg.sd_guidance_scale
    steps = args.num_inference_steps if args.num_inference_steps is not None else cfg.sd_num_inference_steps
    logger.info(f"ip_scale={ip_scale} guidance={guidance} steps={steps}")

    model = load_model(cfg, Path(args.ckpt), device)
    embs, image_ids = encode_test_eegs(model, cfg, device, args.num_samples)
    n_test = embs.shape[0]
    logger.info(f"encoded {n_test} test EEGs -> embeds shape {tuple(embs.shape)}")

    # Save the encoder outputs alongside generated images so eval can match by index
    torch.save(
        {"embeds": embs, "image_ids": image_ids},
        cfg.output_dir / f"test_eeg_embeds_{args.tag}.pt",
    )

    pipe = load_pipeline(cfg, device, ip_scale)
    logger.info("loaded SD + IP-Adapter pipeline")

    # IP-Adapter expects a list of [batch, num_image_embeds, embed_dim] tensors
    # in fp16. We pass classifier-free guidance pairs (uncond + cond) ourselves.
    embs_fp16 = embs.to(device, dtype=torch.float16)
    zero_emb = torch.zeros_like(embs_fp16)

    all_seeds_images = torch.empty(
        (len(args.seeds), n_test, 3, cfg.recon_eval_size, cfg.recon_eval_size),
        dtype=torch.float32,
    )

    for s_idx, seed in enumerate(args.seeds):
        logger.info(f"--- seed {seed} ---")
        set_seed(seed)
        for i in tqdm(range(n_test), desc=f"seed {seed}"):
            cond = embs_fp16[i : i + 1].unsqueeze(1)         # [1, 1, 1024]
            uncond = zero_emb[i : i + 1].unsqueeze(1)        # [1, 1, 1024]
            ip_embeds = [torch.cat([uncond, cond], dim=0)]   # batch=2 (CFG)

            generator = torch.Generator(device=device).manual_seed(int(seed) * 1000 + i)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = pipe(
                    prompt="",
                    negative_prompt="",
                    ip_adapter_image_embeds=ip_embeds,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    height=cfg.recon_image_size,
                    width=cfg.recon_image_size,
                    generator=generator,
                )
            img: Image.Image = result.images[0]
            all_seeds_images[s_idx, i] = to_image_tensor(img, cfg.recon_eval_size)

    # Save per-seed files when running a single seed so multiple jobs can run
    # in parallel without overwriting each other.  evaluate.py knows to merge them.
    if len(args.seeds) == 1:
        seed_str = f"s{args.seeds[0]:02d}"
        out_path = cfg.output_dir / f"recon_images_{args.tag}_{seed_str}.pt"
        torch.save(
            {
                "images": all_seeds_images[0:1],   # [1, 200, 3, 256, 256]
                "image_ids": image_ids,
                "seeds": list(args.seeds),
                "ip_scale": ip_scale,
                "guidance": guidance,
                "steps": steps,
            },
            out_path,
        )
    else:
        out_path = cfg.output_dir / f"recon_images_{args.tag}.pt"
        torch.save(
            {
                "images": all_seeds_images,
                "image_ids": image_ids,
                "seeds": list(args.seeds),
                "ip_scale": ip_scale,
                "guidance": guidance,
                "steps": steps,
            },
            out_path,
        )
    logger.info(f"saved reconstructions -> {out_path}")


if __name__ == "__main__":
    main()
