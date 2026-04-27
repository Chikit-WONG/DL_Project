from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Dictionary with attribute access for nested config values."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[key] = value
        return value

    def copy(self) -> "Config":
        return Config(deepcopy(dict(self)))


def to_plain_data(value: Any) -> Any:
    if isinstance(value, Config):
        return {key: to_plain_data(val) for key, val in value.items()}
    if isinstance(value, dict):
        return {key: to_plain_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg = Config(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        set_by_dotted_key(cfg, key, parse_scalar(value))
    return cfg


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    current = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(cfg), handle, sort_keys=False, allow_unicode=True)
