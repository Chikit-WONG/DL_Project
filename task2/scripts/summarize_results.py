from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_subject_runs(runs_root: Path) -> list[Path]:
    return sorted(path for path in runs_root.rglob("sub-*_seed*") if path.is_dir())


def _flatten_metrics(prefix: str, payload) -> dict[str, float]:
    if not payload:
        return {}
    if isinstance(payload, list):
        if not payload:
            return {}
        payload = payload[0]
    if not isinstance(payload, dict):
        return {}
    flat = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            flat[f"{prefix}{key}"] = float(value)
    return flat


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1}
    return {"mean": mean(values), "std": pstdev(values), "n": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize retrieval and reconstruction metrics across subject runs.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subject_runs = _collect_subject_runs(args.runs_root)
    per_run = []
    aggregate: dict[str, list[float]] = {}

    for run_dir in subject_runs:
        lightning_logs = sorted(run_dir.glob("lightning_logs/version_*"))
        retrieval_payload = None
        for version_dir in reversed(lightning_logs):
            candidate = _load_json(version_dir / "test_results.json")
            if candidate:
                retrieval_payload = candidate
                break

        reconstruction_payload = _load_json(run_dir / "generated_image" / "all" / "reconstruction_metrics.json")
        run_metrics = {}
        run_metrics.update(_flatten_metrics("retrieval_", retrieval_payload))
        run_metrics.update(_flatten_metrics("reconstruction_", reconstruction_payload))
        if not run_metrics:
            continue

        per_run.append({"run_dir": str(run_dir), "metrics": run_metrics})
        for key, value in run_metrics.items():
            aggregate.setdefault(key, []).append(value)

    summary = {
        "runs": per_run,
        "aggregate": {key: _summarize(values) for key, values in sorted(aggregate.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
