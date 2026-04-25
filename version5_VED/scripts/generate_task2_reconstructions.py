import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

REPO_DIR = Path(__file__).resolve().parents[1]
import sys

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import models
from diffusers import StableDiffusionPipeline
from main_eeg_course import device, get_dataset, set_seed
from scripts.task2_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEATURE_PATH,
    DEFAULT_IP_ADAPTER_ROOT,
    DEFAULT_IP_ADAPTER_SUBFOLDER,
    DEFAULT_IP_ADAPTER_WEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_SD15_PATH,
    aggregate_retrieval,
    build_adapted_image_bank,
    ensure_dir,
    load_feature_bank,
    load_reference_image,
    path_hash,
    prompt_from_class,
    resolve_image_path,
    save_json,
)


def build_pipeline(args):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        str(args.ip_adapter_root),
        subfolder=f"{args.ip_adapter_subfolder}/image_encoder",
        torch_dtype=torch.float16,
    ).to(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.sd_model_path),
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.load_ip_adapter(
        str(args.ip_adapter_root),
        subfolder=args.ip_adapter_subfolder,
        weight_name=args.ip_adapter_weight,
        torch_dtype=torch.float16,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_model(args):
    model = models.__dict__[args.net_name](63, 1024, 250).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train_bank", type=Path, default=None)
    parser.add_argument("--net_name", type=str, default="Brain_Visual_Encoder_EEG")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--feature_path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "reconstructions")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.8)
    parser.add_argument("--sd_model_path", type=Path, default=DEFAULT_SD15_PATH)
    parser.add_argument("--ip_adapter_root", type=Path, default=DEFAULT_IP_ADAPTER_ROOT)
    parser.add_argument("--ip_adapter_subfolder", type=str, default=DEFAULT_IP_ADAPTER_SUBFOLDER)
    parser.add_argument("--ip_adapter_weight", type=str, default=DEFAULT_IP_ADAPTER_WEIGHT)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.generation_seed)
    run_name = args.run_name or f"{args.checkpoint.stem}_genseed{args.generation_seed}_{path_hash(args.prompt_template, args.top_k)}"
    run_dir = ensure_dir(args.output_dir / run_name)
    generated_dir = ensure_dir(run_dir / "generated")
    gt_dir = ensure_dir(run_dir / "ground_truth")
    meta_dir = ensure_dir(run_dir / "metadata")

    model = load_model(args)
    train_feature_bank = load_feature_bank(args.feature_path, "train")
    if args.train_bank is not None and args.train_bank.exists():
        train_bank = torch.load(args.train_bank, map_location="cpu", weights_only=False)
    else:
        train_bank = build_adapted_image_bank(
            model,
            train_feature_bank,
            run_dir / "train_bank_runtime.pt",
            batch_size=512,
            force=True,
        )

    _, _, test_dataset = get_dataset(str(args.data_path), str(args.feature_path), 1, [
        "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
        "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
        "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
        "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
        "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
        "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
        "O1", "Oz", "O2",
    ], False, 0.1, 50.0, [0, 250])

    pipe = build_pipeline(args)
    retrieval_meta = []
    train_embeddings = train_bank["embeddings"].to(device)
    generator = torch.Generator(device=device).manual_seed(args.generation_seed)

    with torch.no_grad():
        for index in tqdm(range(len(test_dataset)), desc="Task2 reconstruction"):
            sample = test_dataset[index]
            eeg = sample["eeg"].unsqueeze(0).to(device)
            eeg_embed = F.normalize(model(eeg).float(), dim=-1)[0]
            similarities = eeg_embed @ train_embeddings.T
            agg = aggregate_retrieval(similarities, train_bank["keys"], train_bank["class_names"], args.top_k)
            prompt = prompt_from_class(agg["selected_class_name"], args.prompt_template)
            reference_path = resolve_image_path(args.data_path, agg["selected_reference_key"])
            gt_path = resolve_image_path(args.data_path, sample["x_key"])
            reference_image = load_reference_image(reference_path, max(args.height, args.width))

            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                ip_adapter_image=reference_image,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                generator=generator,
            )
            generated = result.images[0].resize((args.eval_size, args.eval_size), Image.BILINEAR)
            gt_image = Image.open(gt_path).convert("RGB").resize((args.eval_size, args.eval_size), Image.BILINEAR)
            output_name = Path(sample["x_key"]).name
            generated.save(generated_dir / output_name)
            gt_image.save(gt_dir / output_name)

            record = {
                "sample_index": index,
                "ground_truth_key": sample["x_key"],
                "ground_truth_path": str(gt_path),
                "output_name": output_name,
                "prompt": prompt,
                "selected_class_name": agg["selected_class_name"],
                "selected_reference_key": agg["selected_reference_key"],
                "selected_reference_path": str(reference_path),
                "selected_class_score": agg["selected_class_score"],
                "ranked_classes": agg["ranked_classes"][:5],
                "top_items": agg["top_items"][:10],
            }
            retrieval_meta.append(record)

    config = {
        "checkpoint": str(args.checkpoint),
        "train_bank": str(args.train_bank) if args.train_bank else None,
        "prompt_template": args.prompt_template,
        "negative_prompt": args.negative_prompt,
        "top_k": args.top_k,
        "generation_seed": args.generation_seed,
        "sd_model_path": str(args.sd_model_path),
        "ip_adapter_root": str(args.ip_adapter_root),
        "ip_adapter_weight": args.ip_adapter_weight,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "ip_adapter_scale": args.ip_adapter_scale,
    }
    save_json(config, meta_dir / "run_config.json")
    save_json(retrieval_meta, meta_dir / "retrieval_metadata.json")
    print(f"Saved reconstructions to {run_dir}")


if __name__ == "__main__":
    main()
