#!/usr/bin/env python
"""Aggregate reconstruction_metrics.json across grid variants into one table.

Two outputs:
  - JSON: per-variant raw metrics for downstream tooling.
  - Markdown: a sortable table you can paste into a report.

We emit BOTH the "after-align" (the directory named MODALITY_MODE) and
"before-align" (MODALITY_MODE_before) folders for each variant, because the
two are useful to compare independently. By default we look at the
ssim_all30 modality mode; pass --modality-mode to change it.

Convention notes:
  - eval_pixcorr / eval_ssim / eval_alex* / eval_inception / eval_clip:
    higher is better.
  - eval_effnet / eval_swav: these are 1 - pearson_r (scipy correlation
    distance), so LOWER is better. We mark them with a downward arrow in
    the markdown and invert their sign when computing the composite score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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

# Direction: +1 means higher is better, -1 means lower is better.
METRIC_DIRECTION = {
    "eval_pixcorr": +1,
    "eval_ssim": +1,
    "eval_alex2": +1,
    "eval_alex5": +1,
    "eval_inception": +1,
    "eval_clip": +1,
    "eval_clip_vith14": +1,
    "eval_effnet": -1,  # 1 - corr, lower = closer
    "eval_swav": -1,    # 1 - corr, lower = closer
}

# Weights for the composite "user-priority" score. The user explicitly wants
# CLIP, SSIM, and SwAV to improve, so those get extra weight. The secondary
# CLIP (ViT-H-14) is for reference only and gets 0 weight in the composite so
# it doesn't double-count CLIP.
COMPOSITE_WEIGHTS = {
    "eval_clip": 2.0,
    "eval_ssim": 2.0,
    "eval_swav": 2.0,
    "eval_pixcorr": 1.0,
    "eval_alex2": 1.0,
    "eval_alex5": 1.0,
    "eval_inception": 1.0,
    "eval_clip_vith14": 0.0,
    "eval_effnet": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--modality-mode", type=str, default="ssim_all30")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def variant_run_tag(variant: str) -> str:
    # The base variant lives at grid_base; the rest at grid_<name>.
    if variant == "base":
        return "grid_base"
    return f"grid_{variant}"


def load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    # Skip None values (e.g. eval_clip_vith14 when open_clip / weights are
    # unavailable). Skip non-numeric values too, defensively.
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


def collect(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for variant in args.variants:
        run_tag = variant_run_tag(variant)
        subject_run = (
            args.runs_root
            / run_tag
            / args.exp_name
            / f"{args.subject}_seed{args.seed}"
        )
        for suffix, label in [
            ("", "after"),
            ("_before", "before"),
        ]:
            metrics_path = (
                subject_run
                / "generated_image"
                / f"{args.modality_mode}{suffix}"
                / "reconstruction_metrics.json"
            )
            metrics = load_metrics(metrics_path)
            rows.append(
                {
                    "variant": variant,
                    "stage": label,
                    "run_tag": run_tag,
                    "metrics_path": str(metrics_path),
                    "found": metrics is not None,
                    "metrics": metrics or {},
                }
            )
    return {"variants": args.variants, "modality_mode": args.modality_mode, "rows": rows}


def composite_score(metrics: dict[str, float], baseline: dict[str, float] | None) -> float | None:
    """A simple weighted z-improvement score relative to `baseline`.

    For each metric we compute the signed improvement (positive = better)
    relative to baseline, normalized by the baseline magnitude (with a floor
    to avoid blow-up), then weight. This is just for ranking inside the
    grid; the per-metric numbers are still the source of truth.
    """
    if baseline is None or not metrics:
        return None
    total = 0.0
    weight_sum = 0.0
    for key in METRIC_KEYS:
        if key not in metrics or key not in baseline:
            continue
        direction = METRIC_DIRECTION[key]
        weight = COMPOSITE_WEIGHTS.get(key, 1.0)
        denom = max(abs(baseline[key]), 1e-3)
        delta = (metrics[key] - baseline[key]) / denom
        total += direction * delta * weight
        weight_sum += weight
    if weight_sum == 0.0:
        return None
    return total / weight_sum


def render_markdown(data: dict[str, Any]) -> str:
    rows = data["rows"]

    # Baseline = base variant, after stage. Used as anchor for composite.
    baseline_metrics: dict[str, float] | None = None
    for row in rows:
        if row["variant"] == "base" and row["stage"] == "after" and row["found"]:
            baseline_metrics = row["metrics"]
            break

    # Pre-compute composite scores so we can rank.
    for row in rows:
        row["composite"] = composite_score(row["metrics"], baseline_metrics) if row["found"] else None

    header_cells = ["variant", "stage", "PixCorr↑", "SSIM↑", "Alex2↑", "Alex5↑", "Incept↑", "CLIP↑", "CLIP-H↑", "EffNet↓", "SwAV↓", "Δ-score"]
    sep_cells = ["---"] * len(header_cells)

    lines = [
        f"# Grid summary ({data['modality_mode']})",
        "",
        "`CLIP↑` uses OpenAI ViT-L/14 (course-standard, used for grading).",
        "`CLIP-H↑` uses open_clip ViT-H-14 LAION-2B (for direct comparison with the CogCapPro paper).",
        "EffNet and SwAV are `1 - pearson_r` (scipy correlation distance); **lower is better**.",
        "`Δ-score` is a weighted composite of per-metric improvement over the `base / after` row,",
        "with CLIP / SSIM / SwAV weighted 2x. Use it only as a quick ranking heuristic; trust the columns.",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]

    # Sort rows: keep base first, then after-rows sorted by composite desc,
    # with before-rows of each variant immediately following.
    after_rows = [r for r in rows if r["stage"] == "after"]
    before_by_variant = {r["variant"]: r for r in rows if r["stage"] == "before"}

    def sort_key(r: dict[str, Any]) -> tuple[int, float]:
        is_base = 0 if r["variant"] == "base" else 1
        score = r["composite"] if r["composite"] is not None else -1e9
        return (is_base, -score)

    after_rows.sort(key=sort_key)

    def fmt_metric(value: float | None, direction: int) -> str:
        if value is None:
            return "—"
        return f"{value:.3f}"

    def fmt_delta(value: float | None) -> str:
        if value is None:
            return "—"
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.3f}"

    def row_line(r: dict[str, Any]) -> str:
        m = r["metrics"]
        cells = [
            f"`{r['variant']}`",
            r["stage"],
            fmt_metric(m.get("eval_pixcorr"), +1),
            fmt_metric(m.get("eval_ssim"), +1),
            fmt_metric(m.get("eval_alex2"), +1),
            fmt_metric(m.get("eval_alex5"), +1),
            fmt_metric(m.get("eval_inception"), +1),
            fmt_metric(m.get("eval_clip"), +1),
            fmt_metric(m.get("eval_clip_vith14"), +1),
            fmt_metric(m.get("eval_effnet"), -1),
            fmt_metric(m.get("eval_swav"), -1),
            fmt_delta(r["composite"]),
        ]
        return "| " + " | ".join(cells) + " |"

    for r in after_rows:
        lines.append(row_line(r))
        before = before_by_variant.get(r["variant"])
        if before and before["found"]:
            lines.append(row_line(before))

    lines.append("")
    lines.append("## Missing runs")
    missing = [r for r in rows if not r["found"]]
    if not missing:
        lines.append("(none)")
    else:
        for r in missing:
            lines.append(f"- `{r['variant']}` / {r['stage']}: `{r['metrics_path']}` not found")

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
