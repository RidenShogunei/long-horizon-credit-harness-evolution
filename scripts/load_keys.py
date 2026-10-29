"""Load API keys from Hermes without printing them."""

from __future__ import annotations

import os
from pathlib import Path

SECRETS = Path("/home/jinxu/.hermes/.env")


def load_keys(wanted: set[str] | None = None) -> None:
    wanted = wanted or {"DEEPSEEK_API_KEY"}
    if not SECRETS.is_file():
        raise FileNotFoundError(f"missing secrets file: {SECRETS}")
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in wanted and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")
