"""Trial evaluation helpers and failure-layer classification."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research1.contracts import FailureLayer, RunStatus, TaskRun

_INFRA_PATTERNS = [
    r"HTTP(?:\s*Error)?[\s:_]*(?:401|403|429|502|503|504)\b",
    r"connection\s+(refused|reset|error|timed?\s*out)",
    r"\b(?:APIConnectionError|ConnectionError|ConnectTimeout|ProxyError|AuthenticationError)\b",
    r"\b(?:DockerException|DockerError)\b|cannot connect to (?:the )?docker daemon|error response from daemon",
    r"\bunauthorized\b|authentication\s+(?:failed|failure|error|denied)",
    r"no space left on device",
    r"proxy\s+(?:connection\s+)?(?:failed|failure|error|refused)",
]
_SCORER_PATTERNS = [
    r"uvx?:?\s*(not found|No such file)",
    r"failed to download.*uv",
    r"\b(?:Verifier\w*(?:Error|Exception)|RewardFileNotFoundError)\b",
    r"(?:test\.sh|pytest).*?(?:not found|no such file|failed to (?:start|run))",
    r"verifier\s+(?:failed|failure|error|timed out)",
]
_HARNESS_PATTERNS = [
    r"\b(?:ModuleNotFoundError|AttributeError|ImportError)\b",
    r"composer\s+(?:failed|failure|error)",
    r"active_bundle.*?(?:mismatch|not found|missing|invalid)",
]
_MODEL_PATTERNS = [
    r"\b(?:RateLimitError|ContextWindowExceededError)\b",
    r"(?:context.?length|max.?tokens).*?(?:exceeded|too (?:long|many))",
    r"model\s+(?:request\s+)?(?:failed|error)",
]


def classify_failure(text: str, reward: float | None) -> FailureLayer:
    if reward is not None and reward > 0:
        return "model"  # scored success path; layer unused
    blob = text.lower()
    if reward is not None and re.search(r"AgentTimeoutError|agent execution timed out", blob, re.I):
        # Harbor still runs the verifier after the agent exhausts its task budget.
        return "model"
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
    # Only exception fields are error evidence. Config values such as docker,
    # proxy URLs and verifier names are ordinary runtime settings.
    rj = find_result_json(output_dir)
    if rj:
        try:
            data = json.loads(rj.read_text(errors="replace"))
        except (OSError, json.JSONDecodeError):
            data = {}
        exception = data.get("exception_info") or {}
        if isinstance(exception, dict):
            for key in ("exception_type", "exception_message"):
                value = exception.get(key)
                if isinstance(value, str):
                    parts.append(value[:20_000])
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
    """Diagnostic means; skips nulls. Not sufficient for a complete return label."""
    buckets: dict[str, list[float]] = {}
    for r in runs:
        if r.reward is None:
            if require_reward:
                continue
            continue
        buckets.setdefault(r.task_name, []).append(float(r.reward))
    return {t: sum(vs) / len(vs) for t, vs in buckets.items() if vs}


@dataclass
class TaskEvaluation:
    scores: dict[str, float]
    trial_counts: dict[str, int]
    expected_tasks: list[str]
    required_repetitions: int
    is_complete: bool
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_task_evaluation(
    runs: list[TaskRun], *, expected_tasks: list[str], required_repetitions: int,
) -> TaskEvaluation:
    """Fixed task set and exactly R distinct scored trials, without dropping nulls.

    Partial means are diagnostic only; is_complete remains false when any planned
    trial is missing, unscored, duplicated, extra, or belongs to another version.
    """
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise ValueError("expected_tasks must be nonempty and unique")
    if isinstance(required_repetitions, bool) or not isinstance(required_repetitions, int) or required_repetitions < 1:
        raise ValueError("required_repetitions must be a positive integer")
    buckets: dict[str, dict[int, float]] = {task: {} for task in expected_tasks}
    seen: set[tuple[str, int]] = set()
    issues: list[str] = []
    if len({run.version_id for run in runs}) > 1:
        issues.append("mixed version IDs in one evaluation")
    for run in runs:
        if run.task_name not in buckets or run.trial_index not in range(1, required_repetitions + 1):
            issues.append(f"unexpected task/trial {run.task_name}/{run.trial_index}")
            continue
        key = (run.task_name, run.trial_index)
        if key in seen:
            issues.append(f"duplicate task/trial {key}")
            continue
        seen.add(key)
        reward = run.reward
        if run.status != "PASS" or isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
            issues.append(f"unscored task/trial {key}")
            continue
        buckets[run.task_name][run.trial_index] = float(reward)
    counts = {task: len(values) for task, values in buckets.items()}
    for task, count in counts.items():
        if count != required_repetitions:
            issues.append(f"{task}: scored trials {count}, fixed R={required_repetitions}")
    scores = {task: math.fsum(values.values()) / len(values) for task, values in buckets.items() if values}
    return TaskEvaluation(scores, counts, list(expected_tasks), required_repetitions, not issues, issues)


def should_retry(run: TaskRun) -> bool:
    """Infra-only retry rule from protocol."""
    if run.reward is not None:
        return False
    return run.failure_layer in ("infrastructure", "scorer", "container")
