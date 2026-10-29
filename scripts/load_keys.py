"""Load API keys from Hermes without printing them."""

from __future__ import annotations

import os
from pathlib import Path

SECRETS = Path("/home/jinxu/.hermes/.env")

DEFAULT_WANTED = {
    "DEEPSEEK_API_KEY",
    "TONGJI_AGI_API_KEY",
    "TONGJI_AGI_BASE_URL",
    "TONGJI_AGI_DEFAULT_MODEL",
    "TONGJI_AGI_WEEKLY_TOKEN_LIMIT",
    "LLM_PROVIDER",
}


def load_keys(wanted: set[str] | None = None) -> None:
    wanted = wanted or DEFAULT_WANTED
    if not SECRETS.is_file():
        raise FileNotFoundError(f"missing secrets file: {SECRETS}")
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in wanted and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")
