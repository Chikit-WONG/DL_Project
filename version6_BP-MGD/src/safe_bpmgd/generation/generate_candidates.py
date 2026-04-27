from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance

from safe_bpmgd.utils.io import ensure_dir


NEUTRAL_PROMPT = "a natural image, high quality, realistic"


def generate_prototype_candidates(
    prototype_paths: list[str],
    out_dir: str | Path,
    sample_name: str,
    num_candidates: int,
    image_size: int = 256,
) -> list[str]:
    """Leakage-safe fallback: use train-only prototypes as weak reconstructed candidates."""
    out_dir = ensure_dir(out_dir)
    paths = []
    for idx in range(num_candidates):
        source = prototype_paths[idx % len(prototype_paths)]
        with Image.open(source) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
            factor = 0.96 + 0.02 * (idx % 5)
            image = ImageEnhance.Color(image).enhance(factor)
            out_path = out_dir / f"{sample_name}_cand{idx:02d}.png"
            image.save(out_path)
            paths.append(str(out_path))
    return paths


def generate_sdxl_ipadapter_candidates(
    pipe,
    prototype_images,
    out_dir,
    sample_name,
    cfg,
    device: str,
    num_candidates: int | None = None,
) -> list[str]:
    out_dir = ensure_dir(out_dir)
    prompt = cfg["generation"].get("prompt", NEUTRAL_PROMPT)
    seeds = cfg["generation"].get("seed_list", [42])
    num_candidates = int(num_candidates or cfg["generation"].get("num_candidates_per_eeg", 16))
    height = int(cfg["generation"].get("height", 512))
    width = int(cfg["generation"].get("width", 512))
    steps = int(cfg["generation"].get("num_inference_steps", 4))
    guidance = float(cfg["generation"].get("guidance_scale", 0.0))
    paths = []
    import torch

    for idx in range(num_candidates):
        image = prototype_images[idx % len(prototype_images)]
        generator = torch.Generator(device=device).manual_seed(int(seeds[idx % len(seeds)]))
        result = pipe(
            prompt=prompt,
            ip_adapter_image=image,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=generator,
        )
        out_path = out_dir / f"{sample_name}_cand{idx:02d}.png"
        result.images[0].resize((256, 256), Image.BILINEAR).save(out_path)
        paths.append(str(out_path))
    return paths
