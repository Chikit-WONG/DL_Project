from __future__ import annotations

from pathlib import Path


def assert_outputs_exist(paths: list[str | Path]) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing expected outputs: " + ", ".join(missing))
