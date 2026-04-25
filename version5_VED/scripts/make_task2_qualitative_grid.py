import argparse
import math
from pathlib import Path

from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Create a simple qualitative grid for task2 outputs.")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--fake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--padding", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    fake_paths = sorted(path for path in args.fake_root.iterdir() if path.is_file())[: args.count]
    if not fake_paths:
        raise FileNotFoundError(f"No generated images found under {args.fake_root}")

    pairs = []
    for fake_path in fake_paths:
        real_path = args.real_root / fake_path.name
        if not real_path.exists():
            continue
        real_img = Image.open(real_path).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)
        fake_img = Image.open(fake_path).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)
        tile = Image.new("RGB", (args.image_size * 2, args.image_size), "white")
        tile.paste(real_img, (0, 0))
        tile.paste(fake_img, (args.image_size, 0))
        pairs.append(ImageOps.expand(tile, border=1, fill="black"))

    columns = max(1, args.columns)
    rows = math.ceil(len(pairs) / columns)
    tile_w, tile_h = pairs[0].size
    canvas = Image.new(
        "RGB",
        (
            columns * tile_w + (columns + 1) * args.padding,
            rows * tile_h + (rows + 1) * args.padding,
        ),
        "white",
    )

    for idx, tile in enumerate(pairs):
        row = idx // columns
        col = idx % columns
        x = args.padding + col * (tile_w + args.padding)
        y = args.padding + row * (tile_h + args.padding)
        canvas.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved qualitative grid to {args.output}")


if __name__ == "__main__":
    main()
