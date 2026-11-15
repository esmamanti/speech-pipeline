"""Load settings from configs/*.yaml and resolve project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

# src/speech_pipeline/config.py -> the project root is two levels above src/
ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read a YAML config file (default: configs/default.yaml)."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(cfg: dict, key: str) -> Path:
    """Absolute path of a folder listed under `paths:` in the config."""
    return ROOT / cfg["paths"][key]


def relative(path: str | Path) -> str:
    """Path shown relative to the project root, so printed output never
    reveals where the project lives on this machine."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name