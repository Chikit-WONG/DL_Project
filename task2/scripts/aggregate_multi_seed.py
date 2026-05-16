#!/usr/bin/env python
"""Aggregate reconstruction metrics across multiple seeds.

For each (variant, stage) cell, collects the metric across all seeds and
reports mean ± std, matching the report's `mean ± std, N seeds` format.

This is run after the per-seed grid finishes; it does not re-evaluate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


METRIC_KEYS = [
    "eval_pixcorr",
    "eval_ssim",
    "eval_alex2",
    "eval_alex5",
    "eval_inception",
    "eval_clip",
    "eval_clip_vith14",  # secondary CLIP, ViT-H-14, for CogCapPro paper comparison
    "eval_effnet",
    "eval_swav",
]

METRIC_DIRECTION = {
    "eval_pixcorr": +1,
    "eval_ssim": +1,
    "eval_alex2": +1,
    "eval_alex5": +1,
    "eval_inception": +1,
    "eval_clip": +1,
    "eval_clip_vith14": +1,
    "eval_effnet": -1,
    "eval_swav": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--modality-mode", type=str, default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def variant_run_tag(variant: str) -> str:
    return "grid_base" if variant == "base" else f"grid_{variant}"


def metrics_path_for(args: argparse.Namespace, variant: str, seed: int, stage_suffix: str) -> Path:
    return (
        args.runs_root
        / variant_run_tag(variant)
        / args.exp_name
        / f"{args.subject}_seed{seed}"
        / "generated_image"
        / f"{args.modality_mode}{stage_suffix}"
        / "reconstruction_metrics.json"
    )


def load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    out: dict[str, float] = {}
    for k in METRIC_KEYS:
        v = payload.get(k)
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        for stage_suffix, stage_label in [("", "after"), ("_before", "before")]:
            per_metric: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
            seeds_found: list[int] = []
            seeds_missing: list[int] = []
            for seed in args.seeds:
                m = load_metrics(metrics_path_for(args, variant, seed, stage_suffix))
                if m is None:
                    seeds_missing.append(seed)
                    continue
                seeds_found.append(seed)
                for k in METRIC_KEYS:
                    if k in m:
                        per_metric[k].append(m[k])
            stats = {k: summarize(v) for k, v in per_metric.items()}
            rows.append(
                {
                    "variant": variant,
                    "stage": stage_label,
                    "seeds_found": seeds_found,
                    "seeds_missing": seeds_missing,
                    "stats": stats,
                }
            )
    return {
        "variants": args.variants,
        "seeds": args.seeds,
        "modality_mode": args.modality_mode,
        "rows": rows,
    }


def render_markdown(data: dict[str, Any]) -> str:
    rows = data["rows"]
    seeds_str = ", ".join(str(s) for s in data["seeds"])
    lines = [
        f"# Multi-seed summary ({data['modality_mode']}, seeds: {seeds_str})",
        "",
        "Values are `mean ± std` over the seeds where the metrics file was found.",
        "`CLIP↑` is OpenAI ViT-L/14 (course standard). `CLIP-H↑` is open_clip ViT-H-14 LAION-2B (paper-comparable).",
        "EffNet and SwAV are `1 - pearson_r`; **lower is better**.",
        "",
        "| variant | stage | n | PixCorr↑ | SSIM↑ | Alex2↑ | Alex5↑ | Incept↑ | CLIP↑ | CLIP-H↑ | EffNet↓ | SwAV↓ |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cell(metric_stats: dict[str, float]) -> str:
        if metric_stats["n"] == 0:
            return "—"
        if metric_stats["n"] == 1:
            return f"{metric_stats['mean']:.3f}"
        return f"{metric_stats['mean']:.3f} ± {metric_stats['std']:.3f}"

    after_rows = [r for r in rows if r["stage"] == "after"]
    before_rows = {r["variant"]: r for r in rows if r["stage"] == "before"}

    # Sort after-rows by SSIM mean descending, base pinned first.
    def sort_key(r: dict[str, Any]) -> tuple[int, float]:
        is_base = 0 if r["variant"] == "base" else 1
        ssim_mean = r["stats"]["eval_ssim"]["mean"]
        return (is_base, -ssim_mean if ssim_mean == ssim_mean else 1e9)

    after_rows.sort(key=sort_key)

    for r in after_rows:
        s = r["stats"]
        line = (
            f"| `{r['variant']}` | {r['stage']} | {s['eval_ssim']['n']} "
            f"| {cell(s['eval_pixcorr'])} | {cell(s['eval_ssim'])} "
            f"| {cell(s['eval_alex2'])} | {cell(s['eval_alex5'])} "
            f"| {cell(s['eval_inception'])} | {cell(s['eval_clip'])} "
            f"| {cell(s['eval_clip_vith14'])} "
            f"| {cell(s['eval_effnet'])} | {cell(s['eval_swav'])} |"
        )
        lines.append(line)

        # Show paired before-row right under each variant for readability.
        br = before_rows.get(r["variant"])
        if br is not None and br["stats"]["eval_ssim"]["n"] > 0:
            bs = br["stats"]
            bline = (
                f"| `{br['variant']}` | {br['stage']} | {bs['eval_ssim']['n']} "
                f"| {cell(bs['eval_pixcorr'])} | {cell(bs['eval_ssim'])} "
                f"| {cell(bs['eval_alex2'])} | {cell(bs['eval_alex5'])} "
                f"| {cell(bs['eval_inception'])} | {cell(bs['eval_clip'])} "
                f"| {cell(bs['eval_clip_vith14'])} "
                f"| {cell(bs['eval_effnet'])} | {cell(bs['eval_swav'])} |"
            )
            lines.append(bline)

    # Missing-data report.
    missing_lines = []
    for r in rows:
        if r["seeds_missing"]:
            missing_lines.append(
                f"- `{r['variant']}` / {r['stage']}: missing seeds "
                f"{r['seeds_missing']}"
            )
    if missing_lines:
        lines.append("")
        lines.append("## Missing metrics")
        lines.extend(missing_lines)

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(data), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown}")


if __name__ == "__main__":
    main()
