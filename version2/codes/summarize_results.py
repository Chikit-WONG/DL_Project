from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import DEFAULT_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, type=str)
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "TBD"
    return f"{value * 100:.2f}%"


def fmt_num(value: float | None) -> str:
    if value is None:
        return "TBD"
    return f"{value:.4f}"


def build_markdown(cfg, tag: str, lang: str) -> str:
    metrics = load_json(cfg.result_dir / f"metrics_{tag}.json")
    v1 = load_json(cfg.version1_root / "outputs" / "metrics_phase2_main_best.json")
    title = "Version 2 Result Summary" if lang == "en" else "Version 2 结果汇总"
    metrics = metrics or {}
    v1 = v1 or {}

    v2_ret = metrics.get("retrieval", {})
    v2_rec = metrics.get("reconstruction", {}).get("summary", {})
    v1_ret = v1.get("retrieval", {})
    v1_rec = v1.get("reconstruction", {}).get("summary", {})
    has_v2_metrics = bool(v2_ret) and bool(v2_rec)

    lines = [f"# {title}", ""]
    if lang == "en":
        lines.extend(
            [
                f"Tag: `{tag}`",
                "",
                "## Core Comparison",
                "",
                "| Model | Top-1 | Top-5 | SSIM | CLIP |",
                "|---|---:|---:|---:|---:|",
                f"| Version2 `{tag}` | {fmt_pct(v2_ret.get('top1_acc'))} | {fmt_pct(v2_ret.get('top5_acc'))} | {fmt_num(v2_rec.get('eval_ssim', {}).get('mean'))} | {fmt_num(v2_rec.get('eval_clip', {}).get('mean'))} |",
                f"| Version1 Joint | {fmt_pct(v1_ret.get('top1_acc'))} | {fmt_pct(v1_ret.get('top5_acc'))} | {fmt_num(v1_rec.get('eval_ssim', {}).get('mean'))} | {fmt_num(v1_rec.get('eval_clip', {}).get('mean'))} |",
                "",
                "## Reference Rows",
                "",
                "| Reference | Source | Top-1 | Top-5 | SSIM | CLIP |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in cfg.literature_rows:
            lines.append(
                f"| {row['model']} | {row['source']} | {row['top1']} | {row['top5']} | {row['ssim']} | {row['clip']} |"
            )
        lines.extend(
            [
                "",
                "## Notes",
                "",
                (
                    "- Version2 metrics in this table come from the completed `evaluate.py` run."
                    if has_v2_metrics
                    else "- Version2 values remain `TBD` until `evaluate.py` is executed."
                ),
                "- Literature rows are intentionally conservative: local baselines plus consensus targets only.",
            ]
        )
    else:
        lines.extend(
            [
                f"标签：`{tag}`",
                "",
                "## 核心对比",
                "",
                "| 模型 | Top-1 | Top-5 | SSIM | CLIP |",
                "|---|---:|---:|---:|---:|",
                f"| Version2 `{tag}` | {fmt_pct(v2_ret.get('top1_acc'))} | {fmt_pct(v2_ret.get('top5_acc'))} | {fmt_num(v2_rec.get('eval_ssim', {}).get('mean'))} | {fmt_num(v2_rec.get('eval_clip', {}).get('mean'))} |",
                f"| Version1 Joint | {fmt_pct(v1_ret.get('top1_acc'))} | {fmt_pct(v1_ret.get('top5_acc'))} | {fmt_num(v1_rec.get('eval_ssim', {}).get('mean'))} | {fmt_num(v1_rec.get('eval_clip', {}).get('mean'))} |",
                "",
                "## 参考行",
                "",
                "| 参考方法 | 来源 | Top-1 | Top-5 | SSIM | CLIP |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in cfg.literature_rows:
            lines.append(
                f"| {row['model']} | {row['source']} | {row['top1']} | {row['top5']} | {row['ssim']} | {row['clip']} |"
            )
        lines.extend(
            [
                "",
                "## 说明",
                "",
                (
                    "- 当前表格中的 Version2 数值来自已经完成的 `evaluate.py` 评估。"
                    if has_v2_metrics
                    else "- 在运行 `evaluate.py` 之前，Version2 的数值会显示为 `TBD`。"
                ),
                "- 参考对比表目前只放本地基线与统一计划目标，避免凭空编造论文数值。",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    cfg = DEFAULT_CONFIG
    cfg.ensure_dirs()
    zh = build_markdown(cfg, args.tag, "zh")
    en = build_markdown(cfg, args.tag, "en")
    (cfg.result_dir / "results_summary_zh.md").write_text(zh, encoding="utf-8")
    (cfg.result_dir / "results_summary_en.md").write_text(en, encoding="utf-8")
    print("Saved bilingual result summaries")


if __name__ == "__main__":
    main()
