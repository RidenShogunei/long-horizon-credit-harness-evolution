"""Trial evaluation helpers and failure-layer classification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from research1.contracts import FailureLayer, RunStatus, TaskRun

_INFRA_PATTERNS = [
    r"HTTP.?429",
    r"HTTP.?502",
    r"HTTP.?503",
    r"HTTP.?504",
    r"connection\s+(refused|reset|error|timed?\s*out)",
    r"docker",
    r"unauthorized",
    r"authentication",
    r"no space left",
    r"proxy",
]
_SCORER_PATTERNS = [
    r"uvx?:?\s*(not found|No such file)",
    r"failed to download.*uv",
    r"test\.sh",
    r"verifier",
    r"pytest",
]
_HARNESS_PATTERNS = [
    r"module.*import",
    r"composer",
    r"active_bundle",
    r"AttributeError",
    r"ImportError",
]
_MODEL_PATTERNS = [
    r"rate.?limit",
    r"context.?length",
    r"max.?tokens",
    r"model.*error",
]


def classify_failure(text: str, reward: float | None) -> FailureLayer:
    if reward is not None and reward > 0:
        return "model"  # scored success path; layer unused
    blob = text.lower()
    for pat in _INFRA_PATTERNS:
        if re.search(pat, blob, re.I):
            return "infrastructure"
    for pat in _SCORER_PATTERNS:
        if re.search(pat, blob, re.I):
            return "scorer"
    for pat in _HARNESS_PATTERNS:
        if re.search(pat, blob, re.I):
            return "harness"
    for pat in _MODEL_PATTERNS:
        if re.search(pat, blob, re.I):
            return "model"
    if reward is None:
        return "unknown"
    # reward present including 0 with no signal → treat as model/task outcome
    return "model"


def _reward_from_obj(data: dict[str, Any]) -> float | None:
    for key in ("reward", "score", "official_reward"):
        if key in data and data[key] is not None:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                pass
    vr = data.get("verifier_result") or data.get("verifier") or {}
    if isinstance(vr, dict):
        for key in ("reward", "score"):
            if key in vr and vr[key] is not None:
                try:
                    return float(vr[key])
                except (TypeError, ValueError):
                    pass
        nested = vr.get("rewards")
        if isinstance(nested, dict) and nested.get("reward") is not None:
            try:
                return float(nested["reward"])
            except (TypeError, ValueError):
                pass
    rewards = data.get("rewards")
    if isinstance(rewards, dict) and rewards.get("reward") is not None:
        try:
            return float(rewards["reward"])
        except (TypeError, ValueError):
            pass
    return None


def find_result_json(output_dir: Path) -> Path | None:
    candidates = list(output_dir.rglob("result.json"))
    if not candidates:
        return None

    def score_path(p: Path) -> tuple[int, float]:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return (0, p.stat().st_mtime)
        has_reward = 1 if _reward_from_obj(data) is not None else 0
        return (has_reward, p.stat().st_mtime)

    candidates.sort(key=score_path, reverse=True)
    return candidates[0]


def parse_reward(result_path: Path) -> float | None:
    try:
        data = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _reward_from_obj(data)


def collect_log_text(output_dir: Path, limit_chars: int = 200_000) -> str:
    parts: list[str] = []
    for name in ("console.log", "agent.log", "verifier.log", "exception.txt"):
        for p in output_dir.rglob(name):
            try:
                parts.append(p.read_text(errors="replace")[:50_000])
            except OSError:
                pass
    # also scan result.json message fields
    rj = find_result_json(output_dir)
    if rj:
        parts.append(rj.read_text(errors="replace")[:20_000])
    blob = "\n".join(parts)
    return blob[:limit_chars]


def summarize_trial(
    *,
    run_id: str,
    version_id: str,
    task_name: str,
    trial_index: int,
    output_dir: Path,
    active_bundle: dict[str, str] | None = None,
    model: str = "",
    temperature: float | None = None,
    provider_id: str = "",
) -> TaskRun:
    result = find_result_json(output_dir)
    reward: float | None = None
    if result:
        reward = parse_reward(result)
    logs = collect_log_text(output_dir)
    layer = classify_failure(logs, reward)
    if reward is None and not result:
        status: RunStatus = "INCOMPLETE"
        layer = "infrastructure"
    elif reward is None:
        status = "INCOMPLETE"
        # keep layer from classifier; unknown stays unknown (not coerced to 0)
    else:
        status = "PASS"
    return TaskRun(
        run_id=run_id,
        version_id=version_id,
        task_name=task_name,
        trial_index=trial_index,
        reward=reward,
        failure_layer=layer if reward is None or reward == 0 else "model",
        status=status,
        output_dir=str(output_dir.resolve()),
        active_bundle=dict(active_bundle or {}),
        model=model,
        temperature=temperature,
        provider_id=provider_id,
    )


def aggregate_task_means(runs: list[TaskRun], *, require_reward: bool = True) -> dict[str, float]:
    """Mean reward per task; skips null rewards (unknown ≠ 0)."""
    buckets: dict[str, list[float]] = {}
    for r in runs:
        if r.reward is None:
            if require_reward:
                continue
            continue
        buckets.setdefault(r.task_name, []).append(float(r.reward))
    return {t: sum(vs) / len(vs) for t, vs in buckets.items() if vs}


def should_retry(run: TaskRun) -> bool:
    """Infra-only retry rule from protocol."""
    if run.reward is not None:
        return False
    return run.failure_layer in ("infrastructure", "scorer", "container")
