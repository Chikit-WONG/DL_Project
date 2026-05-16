#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create comparison montages for version4_CCP outputs.")
    parser.add_argument("--gt-root", type=Path, required=True, help="Ground-truth test_images root.")
    parser.add_argument("--all-dir", type=Path, required=True, help="Generated all directory.")
    parser.add_argument("--all-before-dir", type=Path, required=True, help="Generated all_before directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory for montages.")
    parser.add_argument("--sheet-columns", type=int, default=5, help="Columns in overview sheets.")
    return parser.parse_args()


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def find_gt_image(gt_root: Path, image_name: str) -> Path:
    matches = list(gt_root.glob(f"*/{image_name}"))
    if not matches:
        raise FileNotFoundError(f"Ground truth not found for {image_name} under {gt_root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple ground-truth matches for {image_name}: {matches}")
    return matches[0]


def labeled_tile(image_path: Path, label: str, font: ImageFont.ImageFont, banner_h: int = 32) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    tile = Image.new("RGB", (image.width, image.height + banner_h), "white")
    tile.paste(image, (0, banner_h))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, image.width, banner_h), fill=(245, 245, 245))
    draw.text((10, 7), label, fill="black", font=font)
    return tile


def concat_row(image_paths: list[Path], labels: list[str], font: ImageFont.ImageFont) -> Image.Image:
    tiles = [labeled_tile(path, label, font) for path, label in zip(image_paths, labels)]
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    return canvas


def save_overview_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int,
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    tile_w, tile_h = images[0].size
    title_h = 40
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * tile_w, title_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, title_h), fill=(245, 245, 245))
    draw.text((10, 10), title, fill="black", font=font)

    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        canvas.paste(image, (col * tile_w, title_h + row * tile_h))
    canvas.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    font = load_font()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    per_image_dirs = {
        "gt_plus_all": output_root / "gt_plus_all",
        "gt_plus_all_before": output_root / "gt_plus_all_before",
        "gt_plus_all_before_all": output_root / "gt_plus_all_before_all",
    }
    for path in per_image_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        path for path in args.all_dir.glob("*.jpg") if path.name != "reconstruction_metrics.json"
    )
    if not all_images:
        raise RuntimeError(f"No generated images found in {args.all_dir}")

    generated_gt_all: list[Path] = []
    generated_gt_all_before: list[Path] = []
    generated_gt_all_before_all: list[Path] = []

    for all_path in all_images:
        name = all_path.name
        gt_path = find_gt_image(args.gt_root, name)
        all_before_path = args.all_before_dir / name
        if not all_before_path.exists():
            raise FileNotFoundError(f"Missing all_before image for {name}: {all_before_path}")

        gt_all = concat_row([gt_path, all_path], ["ground truth", "all"], font)
        gt_all_path = per_image_dirs["gt_plus_all"] / name
        gt_all.save(gt_all_path, quality=95)
        generated_gt_all.append(gt_all_path)

        gt_all_before = concat_row([gt_path, all_before_path], ["ground truth", "all_before"], font)
        gt_all_before_path = per_image_dirs["gt_plus_all_before"] / name
        gt_all_before.save(gt_all_before_path, quality=95)
        generated_gt_all_before.append(gt_all_before_path)

        gt_all_before_all = concat_row(
            [gt_path, all_before_path, all_path],
            ["ground truth", "all_before", "all"],
            font,
        )
        gt_all_before_all_path = per_image_dirs["gt_plus_all_before_all"] / name
        gt_all_before_all.save(gt_all_before_all_path, quality=95)
        generated_gt_all_before_all.append(gt_all_before_all_path)

    save_overview_sheet(
        generated_gt_all,
        output_root / "gt_plus_all_sheet.jpg",
        args.sheet_columns,
        "ground truth + all",
        font,
    )
    save_overview_sheet(
        generated_gt_all_before,
        output_root / "gt_plus_all_before_sheet.jpg",
        args.sheet_columns,
        "ground truth + all_before",
        font,
    )
    save_overview_sheet(
        generated_gt_all_before_all,
        output_root / "gt_plus_all_before_all_sheet.jpg",
        args.sheet_columns,
        "ground truth + all_before + all",
        font,
    )


if __name__ == "__main__":
    main()
