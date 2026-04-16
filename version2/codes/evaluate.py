from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision import transforms
from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

from config import DEFAULT_CONFIG
from data import build_dataloader, load_split_records
from model import EEGEncoderV2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--compare_v1", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_encoder_ckpt(cfg, tag: str, user_ckpt: str | None) -> Path:
    if user_ckpt:
        return Path(user_ckpt)
    preferred = cfg.ckpt_dir / f"{tag}_best.pt"
    if preferred.exists():
        return preferred
    fallback = cfg.ckpt_dir / f"{tag}_last.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve encoder checkpoint for tag={tag}")


def load_encoder(cfg, ckpt_path: Path, device: torch.device) -> EEGEncoderV2:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = EEGEncoderV2(cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


def compute_retrieval(cfg, model: EEGEncoderV2, device: torch.device, limit: int | None) -> dict[str, float]:
    test_loader = build_dataloader(cfg, "test", batch_size=64, shuffle=False, limit=limit)
    cache = torch.load(cfg.cache_dir / "backbone_cache_test.pt", map_location="cpu", weights_only=False)
    semantic_list = []
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device)
            subject_ids = batch["subject_id"].to(device)
            outputs = model(eeg, subject_ids=subject_ids)
            semantic_list.append(outputs["semantic"].detach().cpu())
    semantic = F.normalize(torch.cat(semantic_list, dim=0), dim=-1)
    candidates = F.normalize(cache["features"]["h14"][: semantic.size(0)], dim=-1)
    sims = semantic @ candidates.t()
    targets = torch.arange(sims.size(0))
    top1 = (sims.argmax(dim=1).cpu() == targets).float().mean().item()
    topk = min(5, sims.size(1))
    top5 = (
        sims.topk(topk, dim=1).indices.cpu() == targets.unsqueeze(1)
    ).any(dim=1).float().mean().item()
    return {"top1_acc": top1, "top5_acc": top5}


def load_clip_encoder(cfg, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(str(cfg.h14_model_dir))
    model = CLIPVisionModelWithProjection.from_pretrained(str(cfg.h14_model_dir)).to(device)
    model.eval()
    return processor, model


def load_generated_images(cfg, tag: str) -> tuple[torch.Tensor, list[str], list[int]]:
    payload = torch.load(cfg.result_dir / f"recon_images_{tag}.pt", map_location="cpu", weights_only=False)
    return payload["images"].float(), list(payload["image_ids"]), list(payload.get("seeds", []))


def load_gt_images(cfg, image_ids: list[str]) -> torch.Tensor:
    records = load_split_records(cfg, "test")
    path_map = {record.image_id: record.image_path for record in records}
    tensorizer = transforms.Compose(
        [transforms.Resize((cfg.recon_eval_size, cfg.recon_eval_size)), transforms.ToTensor()]
    )
    tensors = []
    for image_id in image_ids:
        image = Image.open(path_map[image_id]).convert("RGB")
        tensors.append(tensorizer(image))
        image.close()
    return torch.stack(tensors, dim=0)


def pixcorr_score(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().numpy()
    b = b.flatten().numpy()
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def clip_similarity(processor, model, images: torch.Tensor, target_features: torch.Tensor, device: torch.device) -> float:
    pil_images = [transforms.ToPILImage()(img) for img in images]
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)
    pred = F.normalize(outputs.image_embeds.detach().cpu(), dim=-1)
    target = F.normalize(target_features.cpu(), dim=-1)
    return F.cosine_similarity(pred, target, dim=-1).mean().item()


def compute_reconstruction(cfg, tag: str, limit: int | None = None) -> dict[str, object]:
    generated, image_ids, seeds = load_generated_images(cfg, tag)
    if limit is not None:
        generated = generated[:, :limit]
        image_ids = image_ids[:limit]
    gt = load_gt_images(cfg, image_ids)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    processor, model = load_clip_encoder(cfg, device)
    test_cache = torch.load(cfg.cache_dir / "backbone_cache_test.pt", map_location="cpu", weights_only=False)
    target_features = test_cache["features"]["h14"][: len(image_ids)]

    per_seed = []
    for seed_idx in range(generated.size(0)):
        gen = generated[seed_idx]
        pixcorr = float(np.mean([pixcorr_score(gen[i], gt[i]) for i in range(gen.size(0))]))
        ssim_values = []
        for i in range(gen.size(0)):
            x = gen[i].permute(1, 2, 0).numpy()
            y = gt[i].permute(1, 2, 0).numpy()
            ssim_values.append(
                structural_similarity(x, y, data_range=1.0, channel_axis=2)
            )
        clip_score = clip_similarity(processor, model, gen, target_features, device)
        per_seed.append(
            {
                "seed": seed_idx,
                "seed_value": seeds[seed_idx] if seed_idx < len(seeds) else seed_idx,
                "eval_pixcorr": pixcorr,
                "eval_ssim": float(np.mean(ssim_values)),
                "eval_clip": clip_score,
            }
        )

    summary = {}
    for key in ("eval_pixcorr", "eval_ssim", "eval_clip"):
        values = [row[key] for row in per_seed]
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return {"per_seed": per_seed, "summary": summary}


def load_version1_baseline(cfg) -> dict | None:
    path = cfg.version1_root / "outputs" / "metrics_phase2_main_best.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    encoder_ckpt = resolve_encoder_ckpt(cfg, args.tag, args.encoder_ckpt)
    encoder = load_encoder(cfg, encoder_ckpt, device)

    metrics = {
        "retrieval": compute_retrieval(cfg, encoder, device, limit=args.limit),
    }
    recon_path = cfg.result_dir / f"recon_images_{args.tag}.pt"
    if recon_path.exists():
        metrics["reconstruction"] = compute_reconstruction(cfg, args.tag, limit=args.limit)
    if args.compare_v1:
        baseline = load_version1_baseline(cfg)
        if baseline is not None:
            metrics["comparison_to_version1"] = baseline

    out_path = cfg.result_dir / f"metrics_{args.tag}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
