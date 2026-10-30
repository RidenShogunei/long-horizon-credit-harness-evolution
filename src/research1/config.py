"""Load pilot.yaml and related locks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_pilot(path: Path | None = None) -> dict[str, Any]:
    p = path or (REPO_ROOT / "configs/research1/pilot.yaml")
    cfg = load_yaml(p)
    cfg["_config_path"] = str(p.resolve())
    cfg["_config_hash"] = config_hash({k: v for k, v in cfg.items() if not str(k).startswith("_")})
    return cfg


def resolve_path(cfg: dict[str, Any], key: str, default: str | None = None) -> Path:
    raw = cfg.get(key, default)
    if raw is None:
        raise KeyError(key)
    p = Path(str(raw))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def train_task_names(dataset_tb: Path) -> list[str]:
    """Read train split from manifest.json or train_120.txt — never hand-fill."""
    manifest = dataset_tb / "manifest.json"
    train_list = dataset_tb / "train_120.txt"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        rows = payload.get("tasks") or []
        return [row["name"] for row in rows if row.get("split") == "train"]
    if train_list.is_file():
        return [ln.strip() for ln in train_list.read_text().splitlines() if ln.strip()]
    raise FileNotFoundError(f"no manifest.json or train_120.txt under {dataset_tb}")
