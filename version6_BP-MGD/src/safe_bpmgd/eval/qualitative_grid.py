from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def make_grid(real_root: str | Path, fake_root: str | Path, output_path: str | Path, max_items: int = 12, cell: int = 256) -> Path:
    real_root = Path(real_root)
    fake_root = Path(fake_root)
    real_by_name = {path.name: path for path in real_root.rglob("*") if path.is_file()}
    fake_paths = [path for path in sorted(fake_root.rglob("*")) if path.is_file() and path.name in real_by_name][:max_items]
    canvas = Image.new("RGB", (cell * 2, cell * len(fake_paths)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, fake_path in enumerate(fake_paths):
        y = row * cell
        with Image.open(real_by_name[fake_path.name]) as real, Image.open(fake_path) as fake:
            canvas.paste(real.convert("RGB").resize((cell, cell)), (0, y))
            canvas.paste(fake.convert("RGB").resize((cell, cell)), (cell, y))
            draw.text((4, y + 4), "GT", fill=(255, 0, 0))
            draw.text((cell + 4, y + 4), "Recon", fill=(255, 0, 0))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
