from __future__ import annotations

from PIL import Image, ImageFilter, ImageOps


def make_edge_condition(image: Image.Image) -> Image.Image:
    return ImageOps.grayscale(image.convert("RGB")).filter(ImageFilter.FIND_EDGES).convert("RGB")


def make_depth_condition_fallback(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image.convert("RGB"))
    return ImageOps.colorize(gray, black="black", white="white")
