from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageOps


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    if not src.exists():
        return
    dst.symlink_to(src)


def _safe_alias(root: Path, src_name: str, alias_name: str) -> None:
    src = root / src_name
    alias = root / alias_name
    if not src.exists() or alias.exists() or alias.is_symlink():
        return
    try:
        alias.symlink_to(src.name)
    except FileExistsError:
        # Another process may create the alias concurrently.
        return


def _iter_images(split_root: Path):
    for path in sorted(split_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def _make_depth_image(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as img:
        depth = ImageOps.autocontrast(ImageOps.grayscale(img.convert("RGB"))).convert("RGB")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        depth.save(dst_path, quality=95)


def _make_edge_image(src_path: Path, dst_path: Path) -> None:
    image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {src_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), cv2.cvtColor(edge_rgb, cv2.COLOR_RGB2BGR))


def _prepare_split(source_root: Path, depth_root: Path, edge_root: Path, split_name: str) -> dict[str, int]:
    image_count = 0
    split_root = source_root / split_name
    if not split_root.exists() and split_name == "training_images":
        split_root = source_root / "train_images"
    if not split_root.exists():
        return {"images": 0}
    for src_path in _iter_images(split_root):
        relative_path = src_path.relative_to(source_root)
        depth_path = depth_root / relative_path
        edge_path = edge_root / relative_path
        if not depth_path.exists():
            _make_depth_image(src_path, depth_path)
        if not edge_path.exists():
            _make_edge_image(src_path, edge_path)
        image_count += 1
    return {"images": image_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the course EEG dataset for CogCapPro.")
    parser.add_argument("--course-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", type=str, default="sub-01")
    args = parser.parse_args()

    course_data_root = args.course_data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    things_eeg_root = output_root / "ThingsEEG"
    preprocessed_root = things_eeg_root / "Preprocessed_data_250Hz_whiten" / args.subject
    image_root = things_eeg_root / "Image_set_Resize"
    depth_root = things_eeg_root / "Image_depth_set_Resize"
    edge_root = things_eeg_root / "Image_edge_set_Resize"

    preprocessed_root.mkdir(parents=True, exist_ok=True)
    _safe_symlink(course_data_root / "train.pt", preprocessed_root / "train.pt")
    _safe_symlink(course_data_root / "test.pt", preprocessed_root / "test.pt")
    _safe_symlink(course_data_root / "training_images", image_root / "training_images")
    _safe_symlink(course_data_root / "train_images", image_root / "train_images")
    _safe_symlink(course_data_root / "test_images", image_root / "test_images")
    _safe_alias(image_root, "training_images", "train_images")
    _safe_alias(image_root, "train_images", "training_images")

    summary = {
        "subject": args.subject,
        "course_data_root": str(course_data_root),
        "output_root": str(output_root),
        "splits": {},
    }
    for split_name in ("training_images", "test_images"):
        summary["splits"][split_name] = _prepare_split(image_root, depth_root, edge_root, split_name)

    _safe_alias(depth_root, "training_images", "train_images")
    _safe_alias(depth_root, "train_images", "training_images")
    _safe_alias(edge_root, "training_images", "train_images")
    _safe_alias(edge_root, "train_images", "training_images")

    summary_path = output_root / "prepare_course_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
