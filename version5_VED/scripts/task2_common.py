import hashlib
import json
import os
import re
from pathlib import Path

import open_clip
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_DIR / "data" / "things-eeg"
DEFAULT_FEATURE_PATH = REPO_DIR / "output" / "Image_feature"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "output" / "task2"
DEFAULT_CLIP_WEIGHTS = REPO_DIR / "data" / "weights" / "open_clip_pytorch_model.bin"
DEFAULT_PROMPT_TEMPLATE = "a realistic photo of a {class_name}"
DEFAULT_NEGATIVE_PROMPT = (
    "deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"
)
DEFAULT_SD15_PATH = Path("/hpc2hdd/home/ckwong627/workdir/models/stable-diffusion-v1-5")
DEFAULT_IP_ADAPTER_ROOT = Path("/hpc2hdd/home/ckwong627/workdir/models/IP-Adapter")
DEFAULT_IP_ADAPTER_SUBFOLDER = "models"
DEFAULT_IP_ADAPTER_WEIGHT = "ip-adapter_sd15.bin"
BLUR_LEVELS = ["l_1", "l_3", "l_9", "l_15", "l_21", "l_27", "l_33", "l_39", "l_45", "l_51", "l_57", "l_63"]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_hash(*parts):
    digest = hashlib.sha1("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:12]


def normalize_class_dir_name(name):
    return re.sub(r"^\d+_", "", name)


def class_name_from_key(x_key):
    parts = Path(x_key).parts
    if len(parts) < 2:
        return Path(x_key).stem
    return normalize_class_dir_name(parts[1])


def prompt_from_class(class_name, template=DEFAULT_PROMPT_TEMPLATE):
    return template.format(class_name=class_name.replace("_", " "))


def resolve_image_path(data_path, x_key):
    return Path(data_path) / "Image_set" / x_key


def load_openclip_rn50(checkpoint_path, device):
    model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained=str(checkpoint_path), device=device)
    model.eval()
    return model


def load_feature_bank(feature_path, split):
    file_name = f"MultiBlur_RN50_{split}.pt"
    return torch.load(Path(feature_path) / file_name, weights_only=False)


def sorted_feature_keys(feature_bank):
    return sorted(str(key).replace("\\", "/") for key in feature_bank["1"].keys())


def build_text_prototype_cache(feature_bank, checkpoint_path, cache_path, prompt_template, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    class_names = sorted({class_name_from_key(key) for key in sorted_feature_keys(feature_bank)})
    model = load_openclip_rn50(checkpoint_path, device)
    prompts = [prompt_from_class(class_name, prompt_template) for class_name in class_names]

    with torch.no_grad():
        tokens = open_clip.tokenize(prompts).to(device)
        embeddings = model.encode_text(tokens)
        embeddings = F.normalize(embeddings.float(), dim=-1).cpu()

    payload = {
        "class_names": class_names,
        "prompt_template": prompt_template,
        "checkpoint_path": str(checkpoint_path),
        "embeddings": embeddings,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
    }
    ensure_dir(cache_path.parent)
    torch.save(payload, cache_path)
    return payload


def build_adapted_image_bank(model, feature_bank, cache_path, batch_size=512, force=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    device = next(model.parameters()).device
    keys = sorted_feature_keys(feature_bank)
    class_names = [class_name_from_key(key) for key in keys]
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start:start + batch_size]
            img_list = torch.cat(
                [
                    torch.stack([feature_bank[level.split("_")[1]][key].float() for key in batch_keys], dim=0)[:, None, :]
                    for level in BLUR_LEVELS
                ],
                dim=1,
            ).to(device)
            batch_embed = model.get_image_feature(img_list)
            batch_embed = F.normalize(batch_embed.float(), dim=-1).cpu()
            embeddings.append(batch_embed)

    payload = {
        "keys": keys,
        "class_names": class_names,
        "embeddings": torch.cat(embeddings, dim=0),
    }
    ensure_dir(cache_path.parent)
    torch.save(payload, cache_path)
    return payload


def aggregate_retrieval(similarities, keys, class_names, topk):
    topk = min(topk, similarities.numel())
    top_values, top_indices = torch.topk(similarities, k=topk, largest=True)
    class_scores = {}
    best_per_class = {}
    top_items = []

    for score, index in zip(top_values.tolist(), top_indices.tolist()):
        class_name = class_names[index]
        key = keys[index]
        class_scores[class_name] = class_scores.get(class_name, 0.0) + float(score)
        current_best = best_per_class.get(class_name)
        if current_best is None or score > current_best["score"]:
            best_per_class[class_name] = {"score": float(score), "key": key, "index": index}
        top_items.append({"rank": len(top_items) + 1, "score": float(score), "key": key, "class_name": class_name})

    ranked_classes = sorted(
        (
            {
                "class_name": class_name,
                "score": score,
                "best_key": best_per_class[class_name]["key"],
                "best_score": best_per_class[class_name]["score"],
                "best_index": best_per_class[class_name]["index"],
            }
            for class_name, score in class_scores.items()
        ),
        key=lambda item: (item["score"], item["best_score"]),
        reverse=True,
    )

    selected = ranked_classes[0]
    return {
        "selected_class_name": selected["class_name"],
        "selected_reference_key": selected["best_key"],
        "selected_reference_index": selected["best_index"],
        "selected_class_score": selected["score"],
        "ranked_classes": ranked_classes,
        "top_items": top_items,
    }


def save_json(payload, path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_reference_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )


def load_reference_image(image_path, image_size):
    with Image.open(image_path) as image:
        pil = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    return pil
