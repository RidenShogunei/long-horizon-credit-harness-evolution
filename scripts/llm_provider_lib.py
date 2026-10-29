"""Multi-provider LLM routing with weekly quota ledger and failover."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDERS_PATH = REPO / "configs" / "providers.v0.json"
DEFAULT_SECRETS = Path("/home/jinxu/.hermes/.env")
DEFAULT_STATE = Path("/home/jinxu/.hermes/llm_provider_state.json")


def iso_week_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    y, w, _ = now.isocalendar()
    return f"{y}-W{w:02d}"


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    path = path or DEFAULT_SECRETS
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k] = v.strip().strip('"').strip("'")
    return out


def load_provider_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_PROVIDERS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _state_path(cfg: dict[str, Any]) -> Path:
    return Path(os.environ.get("LLM_PROVIDER_STATE_FILE") or cfg.get("state_file") or DEFAULT_STATE)


def read_state(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_provider_config()
    path = _state_path(cfg)
    if not path.is_file():
        return {
            "mode": cfg.get("default_mode", "auto"),
            "pinned_provider": None,
            "weeks": {},
            "updated_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any], cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_provider_config()
    path = _state_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


@dataclass
class ResolvedProvider:
    provider_id: str
    mode: str
    api_base: str
    api_key_env: str
    api_key: str
    harbor_model: str
    api_model_id: str
    api_model_name: str
    weekly_token_limit: int | None
    weekly_tokens_used: int
    reason: str

    def harbor_exports(self) -> dict[str, str]:
        return {
            "LLM_PROVIDER_ID": self.provider_id,
            "LLM_PROVIDER_MODE": self.mode,
            "API_BASE": self.api_base,
            "HARBOR_EVO_API_KEY": self.api_key,
            "OPENAI_API_KEY": self.api_key,
            "MODEL": self.harbor_model,
            "HARBOR_MODEL": self.harbor_model,
        }


def _week_bucket(state: dict[str, Any], week: str | None = None) -> dict[str, Any]:
    week = week or iso_week_id()
    weeks = state.setdefault("weeks", {})
    bucket = weeks.setdefault(week, {"providers": {}, "exhausted": {}, "notes": []})
    bucket.setdefault("providers", {})
    bucket.setdefault("exhausted", {})
    return bucket


def weekly_tokens_used(state: dict[str, Any], provider_id: str, week: str | None = None) -> int:
    bucket = _week_bucket(state, week)
    entry = bucket["providers"].get(provider_id) or {}
    return int(entry.get("tokens") or 0)


def weekly_limit_for(cfg: dict[str, Any], provider_id: str, env: dict[str, str] | None = None) -> int | None:
    env = env or {**load_dotenv(), **os.environ}
    p = cfg["providers"][provider_id]
    if p.get("weekly_token_limit_env"):
        raw = env.get(p["weekly_token_limit_env"])
        if raw not in (None, ""):
            return int(raw)
    if p.get("weekly_token_limit") is not None:
        return int(p["weekly_token_limit"])
    if p.get("weekly_token_limit_default") is not None:
        return int(p["weekly_token_limit_default"])
    return None


def is_exhausted(state: dict[str, Any], provider_id: str, week: str | None = None) -> bool:
    bucket = _week_bucket(state, week)
    return bool(bucket.get("exhausted", {}).get(provider_id))


def mark_exhausted(
    provider_id: str,
    *,
    reason: str,
    cfg: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_provider_config()
    state = state or read_state(cfg)
    week = iso_week_id()
    bucket = _week_bucket(state, week)
    bucket["exhausted"][provider_id] = {
        "at": datetime.now(timezone.utc).isoformat(),
        "reason": reason[:500],
    }
    write_state(state, cfg)
    return state


def record_usage(
    provider_id: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    meta: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_provider_config()
    state = read_state(cfg)
    week = iso_week_id()
    bucket = _week_bucket(state, week)
    entry = bucket["providers"].setdefault(
        provider_id, {"tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    )
    total = total_tokens if total_tokens is not None else int(prompt_tokens) + int(completion_tokens)
    entry["tokens"] = int(entry.get("tokens") or 0) + int(total)
    entry["prompt_tokens"] = int(entry.get("prompt_tokens") or 0) + int(prompt_tokens)
    entry["completion_tokens"] = int(entry.get("completion_tokens") or 0) + int(completion_tokens)
    entry["calls"] = int(entry.get("calls") or 0) + 1
    if meta:
        entry.setdefault("last_meta", meta)
    limit = weekly_limit_for(cfg, provider_id)
    if limit is not None and entry["tokens"] >= limit:
        bucket["exhausted"][provider_id] = {
            "at": datetime.now(timezone.utc).isoformat(),
            "reason": f"weekly_token_limit reached: used={entry['tokens']} limit={limit}",
        }
    write_state(state, cfg)
    return state


def set_mode(mode: str, pinned_provider: str | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_provider_config()
    state = read_state(cfg)
    if mode in cfg["providers"]:
        state["mode"] = mode
        state["pinned_provider"] = mode
    elif mode == "auto":
        state["mode"] = "auto"
        state["pinned_provider"] = pinned_provider
    else:
        raise ValueError(f"unknown mode/provider: {mode}")
    write_state(state, cfg)
    return state


def clear_exhausted(provider_id: str | None = None, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_provider_config()
    state = read_state(cfg)
    bucket = _week_bucket(state)
    if provider_id:
        bucket.get("exhausted", {}).pop(provider_id, None)
    else:
        bucket["exhausted"] = {}
    write_state(state, cfg)
    return state


def looks_like_quota_error(text: str, provider_cfg: dict[str, Any]) -> bool:
    low = (text or "").lower()
    for s in provider_cfg.get("exhaust_error_substrings") or []:
        if s.lower() in low:
            return True
    return bool(re.search(r"\b429\b", low))


def _resolve_one(
    provider_id: str,
    *,
    cfg: dict[str, Any],
    state: dict[str, Any],
    env: dict[str, str],
    mode: str,
    reason: str,
) -> ResolvedProvider | None:
    p = cfg["providers"][provider_id]
    key_env = p["api_key_env"]
    key = env.get(key_env) or os.environ.get(key_env)
    if not key:
        return None
    used = weekly_tokens_used(state, provider_id)
    limit = weekly_limit_for(cfg, provider_id, env)
    if is_exhausted(state, provider_id):
        return None
    if limit is not None and used >= limit:
        return None
    return ResolvedProvider(
        provider_id=provider_id,
        mode=mode,
        api_base=p["api_base"],
        api_key_env=key_env,
        api_key=key,
        harbor_model=p["harbor_model"],
        api_model_id=p["api_model_id"],
        api_model_name=p["api_model_name"],
        weekly_token_limit=limit,
        weekly_tokens_used=used,
        reason=reason,
    )


def resolve_provider(
    *,
    mode: str | None = None,
    cfg: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
    prefer: str | None = None,
) -> ResolvedProvider:
    cfg = cfg or load_provider_config()
    state = state or read_state(cfg)
    env = {**load_dotenv(Path(cfg.get("secrets_file") or DEFAULT_SECRETS)), **dict(os.environ)}
    mode = mode or os.environ.get("LLM_PROVIDER") or state.get("mode") or cfg.get("default_mode") or "auto"

    if mode in cfg["providers"]:
        got = _resolve_one(mode, cfg=cfg, state=state, env=env, mode=mode, reason=f"forced mode={mode}")
        if not got:
            raise RuntimeError(f"provider {mode} unavailable (missing key, exhausted, or over weekly limit)")
        return got

    if mode != "auto":
        raise ValueError(f"unknown LLM_PROVIDER mode: {mode}")

    order = list(cfg.get("selection_order_auto") or [])
    if prefer and prefer in order:
        order = [prefer] + [x for x in order if x != prefer]
    pinned = state.get("pinned_provider")
    if pinned and pinned in order:
        order = [pinned] + [x for x in order if x != pinned]

    errors: list[str] = []
    for pid in order:
        got = _resolve_one(pid, cfg=cfg, state=state, env=env, mode="auto", reason=f"auto selected {pid}")
        if got:
            return got
        errors.append(pid)
    raise RuntimeError(f"no provider available; tried={errors}")


def status_dict() -> dict[str, Any]:
    cfg = load_provider_config()
    state = read_state(cfg)
    week = iso_week_id()
    providers = {}
    for pid, p in cfg["providers"].items():
        providers[pid] = {
            "label": p.get("label"),
            "api_base": p.get("api_base"),
            "api_key_env": p.get("api_key_env"),
            "key_present": bool(load_dotenv().get(p["api_key_env"]) or os.environ.get(p["api_key_env"])),
            "weekly_token_limit": weekly_limit_for(cfg, pid),
            "weekly_tokens_used": weekly_tokens_used(state, pid, week),
            "exhausted": is_exhausted(state, pid, week),
            "harbor_model": p.get("harbor_model"),
        }
    active = None
    active_error = None
    try:
        r = resolve_provider(cfg=cfg, state=state)
        active = {
            "provider_id": r.provider_id,
            "mode": r.mode,
            "api_base": r.api_base,
            "harbor_model": r.harbor_model,
            "reason": r.reason,
            "weekly_tokens_used": r.weekly_tokens_used,
            "weekly_token_limit": r.weekly_token_limit,
        }
    except Exception as e:  # noqa: BLE001
        active_error = str(e)
    return {
        "week": week,
        "mode": state.get("mode"),
        "pinned_provider": state.get("pinned_provider"),
        "active": active,
        "active_error": active_error,
        "providers": providers,
        "state_file": str(_state_path(cfg)),
    }
