#!/usr/bin/env python3
"""Create grey-background composite images for ablation experiments.

Each source image is:
  1. Cropped to square, resized to 640×640
  2. Centred on a 1280×1280 grey canvas (RGB 158, 162, 166)
  3. Overlaid with a semi-transparent red fixation point (redpoint.png)

The output directory mirrors the input's sub-directory structure
(train_images/, test_images/, etc.), so process_image_course.py can be
pointed at it directly via --image_set_dir.

Usage:
    python task1/scripts/make_greybg_images.py \
        --input  <Image_set_dir>    \
        --output <output_dir>       \
        [--redpoint <redpoint.png>]

Defaults:
    --input    DL_Project/image-eeg-data/converted_for_cogcappro/ThingsEEG/Image_set_Resize
    --output   DL_Project/task1/output/Image_set_greybg
    --redpoint same directory as this script / redpoint.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
TASK1_DIR = SCRIPT_DIR.parent
DL_PROJECT = TASK1_DIR.parent

DEFAULT_INPUT = (
    DL_PROJECT / "image-eeg-data" / "converted_for_cogcappro"
    / "ThingsEEG" / "Image_set_Resize"
)
DEFAULT_OUTPUT = TASK1_DIR / "output" / "Image_set_greybg"
DEFAULT_REDPOINT = SCRIPT_DIR / "redpoint.png"

BG_SIZE = (1280, 1280)
BG_COLOR = (158, 162, 166)
IMG_SIZE = (640, 640)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def make_composite(src_path: Path, redpoint: Image.Image, dest_path: Path) -> None:
    bg = Image.new("RGB", BG_SIZE, BG_COLOR)

    img = Image.open(src_path).convert("RGB")
    side = min(img.width, img.height)
    img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                    (img.width + side) // 2, (img.height + side) // 2))
    img = img.resize(IMG_SIZE, Image.LANCZOS)

    cx, cy = BG_SIZE[0] // 2, BG_SIZE[1] // 2
    bg.paste(img, (cx - IMG_SIZE[0] // 2, cy - IMG_SIZE[1] // 2))
    bg.paste(redpoint, (cx - redpoint.width // 2, cy - redpoint.height // 2), mask=redpoint)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    bg.save(dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",    default=str(DEFAULT_INPUT),
                        help="Source Image_set directory (contains train_images/, test_images/)")
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT),
                        help="Destination directory (same structure as input)")
    parser.add_argument("--redpoint", default=str(DEFAULT_REDPOINT),
                        help="Path to redpoint.png fixation-marker image")
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")
    if not Path(args.redpoint).exists():
        raise FileNotFoundError(f"redpoint.png not found: {args.redpoint}")

    redpoint = Image.open(args.redpoint).convert("RGBA")
    print(f"Source : {src_root}")
    print(f"Output : {dst_root}")
    print(f"Redpoint: {args.redpoint}")

    all_images = sorted(
        p for p in src_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Found {len(all_images)} images to process")

    skipped = done = 0
    for i, src in enumerate(all_images, 1):
        rel = src.relative_to(src_root)
        dst = (dst_root / rel).with_suffix(".png")

        if dst.exists():
            skipped += 1
            continue

        make_composite(src, redpoint, dst)
        done += 1
        if i % 500 == 0:
            print(f"  [{i}/{len(all_images)}] processed={done} skipped={skipped}")

    print(f"\nDone. processed={done}, skipped={skipped}, total={len(all_images)}")
    print(f"Output: {dst_root}")


if __name__ == "__main__":
    main()
