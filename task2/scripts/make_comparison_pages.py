#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paginated comparison pages from montage tiles.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of per-image comparison montages.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for paginated pages.")
    parser.add_argument("--title", type=str, required=True, help="Page title suffix.")
    parser.add_argument("--columns", type=int, default=5, help="Tiles per row.")
    parser.add_argument("--rows", type=int, default=4, help="Rows per page.")
    parser.add_argument("--tile-width", type=int, default=250, help="Rendered tile width.")
    parser.add_argument("--label-height", type=int, default=26, help="Filename label height.")
    parser.add_argument("--header-height", type=int, default=34, help="Page header height.")
    parser.add_argument("--padding", type=int, default=4, help="Padding between tiles.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit_image(image: Image.Image, target_width: int) -> Image.Image:
    ratio = target_width / image.width
    target_height = max(1, int(round(image.height * ratio)))
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def render_tile(
    image_path: Path,
    tile_width: int,
    label_height: int,
    font: ImageFont.ImageFont,
    bg_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image = fit_image(image, tile_width)
    tile = Image.new("RGB", (image.width, image.height + label_height), bg_color)
    tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    label = image_path.stem
    draw.text((4, image.height + 4), label, fill=text_color, font=font)
    return tile


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tile_font = load_font(15)
    header_font = load_font(22)
    bg_color = (36, 36, 36)
    text_color = (235, 235, 235)

    image_paths = sorted(path for path in args.input_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No .jpg files found in {args.input_dir}")

    tiles = [
        render_tile(path, args.tile_width, args.label_height, tile_font, bg_color, text_color)
        for path in image_paths
    ]

    per_page = args.columns * args.rows
    total_pages = math.ceil(len(tiles) / per_page)
    tile_height = max(tile.height for tile in tiles)
    page_width = args.padding + args.columns * (args.tile_width + args.padding)
    page_height = args.header_height + args.padding + args.rows * (tile_height + args.padding)

    for page_idx in range(total_pages):
        page = Image.new("RGB", (page_width, page_height), bg_color)
        draw = ImageDraw.Draw(page)
        page_number = page_idx + 1
        header = f"Page {page_number} — {args.title}"
        draw.text((8, 6), header, fill=text_color, font=header_font)

        start = page_idx * per_page
        end = min(start + per_page, len(tiles))
        page_tiles = tiles[start:end]
        for idx, tile in enumerate(page_tiles):
            row = idx // args.columns
            col = idx % args.columns
            x = args.padding + col * (args.tile_width + args.padding)
            y = args.header_height + args.padding + row * (tile_height + args.padding)
            page.paste(tile, (x, y))

        page.save(args.output_dir / f"comparison_page_{page_number:02d}.jpg", quality=92)


if __name__ == "__main__":
    main()
