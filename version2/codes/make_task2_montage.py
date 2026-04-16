from __future__ import annotations

import argparse

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from config import DEFAULT_CONFIG
from data import load_split_records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=20)
    return parser.parse_args()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.clamp(0.0, 1.0))


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    payload = torch.load(cfg.result_dir / f"recon_images_{args.tag}.pt", map_location="cpu", weights_only=False)
    generated = payload["images"][args.seed_index]
    image_ids = payload["image_ids"][: args.num_samples]
    generated = generated[: args.num_samples]
    records = load_split_records(cfg, "test")
    path_map = {record.image_id: record.image_path for record in records}

    size = cfg.recon_eval_size
    canvas = Image.new("RGB", (size * args.num_samples, size * 2 + 28), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), f"Top: ground truth | Bottom: generated | tag={args.tag}", fill="black")

    for idx, image_id in enumerate(image_ids):
        gt = Image.open(path_map[image_id]).convert("RGB").resize((size, size))
        pred = tensor_to_pil(generated[idx]).resize((size, size))
        canvas.paste(gt, (idx * size, 28))
        canvas.paste(pred, (idx * size, 28 + size))
        gt.close()

    out_path = cfg.result_dir / f"task2_montage_{args.tag}_s{args.seed_index:02d}.png"
    canvas.save(out_path)
    print(f"Saved montage to {out_path}")


if __name__ == "__main__":
    main()
