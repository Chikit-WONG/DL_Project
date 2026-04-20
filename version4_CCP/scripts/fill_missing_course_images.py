from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
from PIL import Image, ImageOps
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def _make_placeholder_image(dst_path: Path, size: int = 512) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=(127, 127, 127)).save(dst_path, quality=95)


def _load_required_relpaths(pt_path: Path) -> list[str]:
    payload = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=False)
    img_array = payload["img"]
    refs = sorted({str(item) for row in img_array for item in row})
    return refs


def _first_existing_image(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            return path
    return None


def _resolve_target(course_root: Path, rel_path: str) -> Path:
    normalized = rel_path.replace("train_images/", "training_images/", 1)
    return course_root / normalized


def _fill_one(
    *,
    rel_path: str,
    course_root: Path,
    adapted_root: Path,
    fallback_image: Path | None,
) -> dict[str, str]:
    target_image = _resolve_target(course_root, rel_path)
    target_depth = adapted_root / "ThingsEEG" / "Image_depth_set_Resize" / target_image.relative_to(course_root)
    target_edge = adapted_root / "ThingsEEG" / "Image_edge_set_Resize" / target_image.relative_to(course_root)
    if target_image.exists() and target_depth.exists() and target_edge.exists():
        return {"status": "exists", "rel_path": rel_path}

    class_dir = target_image.parent
    source_image = _first_existing_image(class_dir)
    if source_image is not None:
        source_kind = "same_class_copy"
    elif fallback_image is not None:
        source_image = fallback_image
        source_kind = "global_fallback_copy"
    else:
        source_kind = "generated_gray_placeholder"

    if not target_image.exists():
        target_image.parent.mkdir(parents=True, exist_ok=True)
        if source_kind == "generated_gray_placeholder":
            _make_placeholder_image(target_image)
        else:
            shutil.copy2(source_image, target_image)

    if not target_depth.exists():
        if source_kind != "generated_gray_placeholder":
            source_depth = adapted_root / "ThingsEEG" / "Image_depth_set_Resize" / source_image.relative_to(course_root)
            if source_depth.exists():
                target_depth.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_depth, target_depth)
            else:
                _make_depth_image(target_image, target_depth)
        else:
            _make_depth_image(target_image, target_depth)

    if not target_edge.exists():
        if source_kind != "generated_gray_placeholder":
            source_edge = adapted_root / "ThingsEEG" / "Image_edge_set_Resize" / source_image.relative_to(course_root)
            if source_edge.exists():
                target_edge.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_edge, target_edge)
            else:
                _make_edge_image(target_image, target_edge)
        else:
            _make_edge_image(target_image, target_edge)

    return {
        "status": "filled",
        "rel_path": rel_path,
        "image_path": str(target_image),
        "source_kind": source_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing course image files referenced by train/test pt files.")
    parser.add_argument("--course-data-root", type=Path, required=True)
    parser.add_argument("--adapted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-list-file", type=Path, default=None)
    args = parser.parse_args()

    course_root = args.course_data_root.expanduser().resolve()
    adapted_root = args.adapted_root.expanduser().resolve()

    if args.missing_list_file is not None:
        raw_items = args.missing_list_file.read_text(encoding="utf-8").splitlines()
        refs = sorted({item.strip().strip("'[]\",") for item in raw_items if item.strip()})
    else:
        refs = []
        refs.extend(_load_required_relpaths(course_root / "train.pt"))
        refs.extend(_load_required_relpaths(course_root / "test.pt"))
        refs = sorted(set(refs))

    fallback_image = _first_existing_image(course_root / "training_images" / "00001_aardvark")
    summary = {
        "course_data_root": str(course_root),
        "adapted_root": str(adapted_root),
        "required_refs": len(refs),
        "filled": [],
        "already_present": 0,
    }

    for rel_path in refs:
        result = _fill_one(
            rel_path=rel_path,
            course_root=course_root,
            adapted_root=adapted_root,
            fallback_image=fallback_image,
        )
        if result["status"] == "exists":
            summary["already_present"] += 1
        else:
            summary["filled"].append(result)

    summary["filled_count"] = len(summary["filled"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
