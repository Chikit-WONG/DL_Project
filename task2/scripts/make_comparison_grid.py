"""
Create 3-way comparison grids: Ground Truth | all_before | all.

Outputs:
  comparison/grid_all200.png          — full 200-triplet overview grid
  comparison/grid_page01.png …        — page grids (20 triplets each, 4 cols)
  comparison/single/banana_09s.png …  — one image per test item (3 panels)
Usage:
    python scripts/make_comparison_grid.py
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── paths ────────────────────────────────────────────────────────────────────
BASE = Path(
    "/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/"
    "ChiKitWONG/Assignments/Project/DL_Project/references/repository/"
    "CognitionCapturerPro/runs/full_v2/"
    "intra-subject_cogcappro_EEGProjectLayer_multimodal_cogcap_list_ViT-H-14/"
    "sub-01_seed0/generated_image"
)
REAL_ROOT = Path(
    "/hpc2hdd/home/ckwong627/workdir/Class/DSAA2012-Deep_Learning/"
    "ChiKitWONG/Assignments/Project/DL_Project/image-eeg-data/"
    "converted_for_cogcappro/ThingsEEG/Image_set_Resize/test_images"
)
BEFORE_ROOT = BASE / "all_before"
AFTER_ROOT  = BASE / "all"
OUT_DIR     = BASE / "comparison"
SINGLE_DIR  = OUT_DIR / "single"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_DIR.mkdir(parents=True, exist_ok=True)

# ── build real-image index ───────────────────────────────────────────────────
real_index: dict[str, Path] = {}
for subdir in REAL_ROOT.iterdir():
    if subdir.is_dir():
        for f in subdir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                real_index[f.name] = f

# ── collect triplets ─────────────────────────────────────────────────────────
triplets: list[tuple[Path, Path, Path, str]] = []  # (real, before, after, stem)
for before_p in sorted(BEFORE_ROOT.glob("*.jpg")):
    name = before_p.name
    after_p = AFTER_ROOT / name
    if name in real_index and after_p.exists():
        triplets.append((real_index[name], before_p, after_p, name.replace(".jpg", "")))
    else:
        print(f"  [warn] missing match for {name}")

print(f"Found {len(triplets)} matched triplets")

# ── helpers ──────────────────────────────────────────────────────────────────
THUMB = 128
GAP   = 4
BG    = (30, 30, 30)
FG    = (220, 220, 220)
HEADER_H = 22
LABEL_H  = 14

try:
    font     = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    font_col = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
except Exception:
    font = font_hdr = font_col = ImageFont.load_default()

# column header colours
COL_COLORS = [(100, 210, 100), (100, 160, 255), (255, 160, 80)]
COL_LABELS = ["Ground Truth", "all_before (EEG direct)", "all (post-align)"]

def load_thumb(path: Path, size: int = THUMB) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    out = Image.new("RGB", (size, size), (55, 55, 55))
    out.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return out

TRIPLET_W = THUMB * 3 + GAP * 2   # width of one triplet cell
TRIPLET_H = THUMB + LABEL_H + 2   # height of one triplet cell (thumb + name label)

# ── 1. individual single images ──────────────────────────────────────────────
# Labels placed BELOW thumbnails, each broken into two short lines so they
# fit within their 128-px column without overlapping the neighbour.
SINGLE_COL_LABELS = [
    ("Ground Truth", ""),
    ("all_before", "(EEG direct)"),
    ("all", "(post-align)"),
]
SINGLE_LABEL_H = 28  # two label lines × 14px
COL_HDR_H = 14       # column-header row height used in page grids
SINGLE_W = TRIPLET_W + 8
SINGLE_H = HEADER_H + THUMB + SINGLE_LABEL_H

for real_p, before_p, after_p, stem in triplets:
    img = Image.new("RGB", (SINGLE_W, SINGLE_H), BG)
    d = ImageDraw.Draw(img)

    d.text((4, 3), stem[:40], font=font_hdr, fill=FG)

    thumbs = [load_thumb(real_p), load_thumb(before_p), load_thumb(after_p)]
    for ci, (thumb, col_color, (line1, line2)) in enumerate(
        zip(thumbs, COL_COLORS, SINGLE_COL_LABELS)
    ):
        x = 4 + ci * (THUMB + GAP)
        img.paste(thumb, (x, HEADER_H))
        d.text((x, HEADER_H + THUMB + 2),  line1, font=font_col, fill=col_color)
        if line2:
            d.text((x, HEADER_H + THUMB + 14), line2, font=font,     fill=col_color)

    img.save(SINGLE_DIR / f"{stem}.png", optimize=True)

print(f"Saved {len(triplets)} single images → {SINGLE_DIR}")

# ── 2. page grids ────────────────────────────────────────────────────────────
PAGE_SIZE = 20
PAGE_COLS = 4
ROW_GAP   = 6

def make_grid(chunk, title: str, cols: int) -> Image.Image:
    rows = math.ceil(len(chunk) / cols)
    w = cols * (TRIPLET_W + GAP) - GAP
    h = HEADER_H + COL_HDR_H + rows * (TRIPLET_H + ROW_GAP) - ROW_GAP
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((4, 3), title, font=font_hdr, fill=FG)

    # column-header bar
    for ci, (lbl, col) in enumerate(zip(COL_LABELS, COL_COLORS)):
        d.text((4 + ci * (THUMB + GAP), HEADER_H), lbl, font=font_col, fill=col)

    y_base = HEADER_H + COL_HDR_H
    for idx, (real_p, before_p, after_p, stem) in enumerate(chunk):
        col = idx % cols
        row = idx // cols
        x0 = col * (TRIPLET_W + GAP)
        y0 = y_base + row * (TRIPLET_H + ROW_GAP)

        for ci, path in enumerate([real_p, before_p, after_p]):
            img.paste(load_thumb(path), (x0 + ci * (THUMB + GAP), y0))

        d.text((x0, y0 + THUMB + 2), stem[:22], font=font, fill=FG)
    return img

for page_idx in range(math.ceil(len(triplets) / PAGE_SIZE)):
    chunk = triplets[page_idx * PAGE_SIZE : (page_idx + 1) * PAGE_SIZE]
    title = f"Page {page_idx+1}  ·  Ground Truth | all_before | all"
    page = make_grid(chunk, title, PAGE_COLS)
    path = OUT_DIR / f"grid_page{page_idx+1:02d}.png"
    page.save(path, optimize=True)
    print(f"  page {page_idx+1:2d} → {path.name}")

# ── 3. full overview grid (all 200, smaller thumbnails) ──────────────────────
OV_THUMB = 72
OV_COLS  = 10
OV_GAP   = 3
OV_TRIP_W = OV_THUMB * 3 + OV_GAP * 2
OV_TRIP_H = OV_THUMB + 12
OV_ROWS   = math.ceil(len(triplets) / OV_COLS)
ov_w = OV_COLS * (OV_TRIP_W + OV_GAP) - OV_GAP
ov_h = HEADER_H + OV_ROWS * (OV_TRIP_H + OV_GAP) - OV_GAP

overview = Image.new("RGB", (ov_w, ov_h), BG)
od = ImageDraw.Draw(overview)
od.text((4, 3), "All 200 · Ground Truth | all_before | all", font=font_hdr, fill=FG)

for idx, (real_p, before_p, after_p, stem) in enumerate(triplets):
    col = idx % OV_COLS
    row = idx // OV_COLS
    x0 = col * (OV_TRIP_W + OV_GAP)
    y0 = HEADER_H + row * (OV_TRIP_H + OV_GAP)
    for ci, path in enumerate([real_p, before_p, after_p]):
        overview.paste(load_thumb(path, OV_THUMB), (x0 + ci * (OV_THUMB + OV_GAP), y0))
    od.text((x0, y0 + OV_THUMB + 1), stem[:14], font=font, fill=FG)

ov_path = OUT_DIR / "grid_all200.png"
overview.save(ov_path, optimize=True)
print(f"Saved overview → {ov_path}")
print("Done.")
