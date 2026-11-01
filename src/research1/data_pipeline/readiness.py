"""Historical resource compatibility, independent of whether a task scored 0 or 1.

This module reads execution metadata only. It never calls a model, provisions a
container, reads task tests/solutions, or substitutes a favorable result. Image
availability is supplied separately by the campaign's environment inventory.
"""
from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_SOLVER = {"model": "openai/DeepSeek-V4.1-Flash", "temperature": 0.7,
                  "max_turns": 200, "provider_id": "tongji_agi",
                  "api_base": "https://agi.tongji.edu.cn/v1"}


def _load(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _time(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return None


def _elapsed(row: dict) -> float | None:
    explicit = _number(row.get("duration_seconds"))
    if explicit is not None and explicit >= 0:
        return explicit
    start, end = _time(row.get("started_at")), _time(row.get("finished_at"))
    if start is not None and end is not None and end >= start:
        return end - start
    # Agent-only time omits provisioning and verification. Record it below,
    # but do not use it to certify a whole-task operational wall limit.
    return None


def _condition(row: dict, concrete: dict | None = None) -> dict:
    spec = row.get("evaluation_spec") or {}
    budget = row.get("evaluation_conditions") or {}
    config = (concrete or row).get("config") or {}
    agent = config.get("agent") or {}
    kwargs = agent.get("kwargs") or {}
    return {
        "model": agent.get("model_name", spec.get("model", row.get("model"))),
        "temperature": kwargs.get("temperature", spec.get("temperature", row.get("temperature"))),
        "max_turns": kwargs.get("max_turns", spec.get("max_turns", budget.get("max_turns"))),
        "provider_id": spec.get("provider_id", row.get("provider_id")),
        "api_base": kwargs.get("api_base", spec.get("api_base")),
    }


def _matches(actual: dict, expected: dict) -> bool:
    if any(actual.get(key) != expected.get(key) for key in ("model", "temperature", "max_turns")):
        return False
    # A concrete endpoint or recorded provider must establish the frozen
    # service; absence of both is unknown rather than a compatibility pass.
    provider, endpoint = actual.get("provider_id"), actual.get("api_base")
    if provider is not None and provider != expected.get("provider_id"):
        return False
    if endpoint is not None and str(endpoint).rstrip("/") != str(expected.get("api_base")).rstrip("/"):
        return False
    return provider is not None or endpoint is not None


def _scored(row: dict, *, concrete: bool = False) -> bool:
    if concrete:
        reward = ((row.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        return bool(row.get("finished_at") and _number(reward) is not None)
    return row.get("status") in {"PASS", "FAIL"} and _number(row.get("reward")) is not None


def _task_name(row: dict) -> str | None:
    return row.get("task_name") or (row.get("evaluation_spec") or {}).get("task")


def _bound_trial(row: dict, path: Path, task: str) -> bool:
    config = row.get("config") or {}
    task_path = (config.get("task") or {}).get("path")
    return bool(row.get("task_name") == task and row.get("trial_name") == path.parent.name
                and isinstance(task_path, str) and Path(task_path).name == task
                and isinstance(config.get("agent"), dict))


def _concrete_for(wrapper: dict, root: Path) -> tuple[Path | None, dict | None, int]:
    """Prefer an explicitly bound result; otherwise require one matching trial."""
    task = _task_name(wrapper)
    exact = wrapper.get("trial_result_path")
    if exact:
        path = Path(exact)
        if not path.is_absolute():
            path = root / path
        result = _load(path)
        if result and _bound_trial(result, path, task):
            return path.resolve(), result, 1
        return None, None, 0
    output = wrapper.get("output_dir")
    if not isinstance(output, str):
        return None, None, 0
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    if not output_path.is_dir():
        return None, None, 0
    matches = []
    for path in output_path.rglob("result.json"):
        result = _load(path)
        if result and _bound_trial(result, path, task):
            matches.append((path.resolve(), result))
    return (*matches[0], 1) if len(matches) == 1 else (None, None, len(matches))


def _observation(path: Path, row: dict, expected: dict, wall_seconds: int,
                 concrete_path: Path | None = None, concrete: dict | None = None,
                 matching_trials: int = 0, *, raw_trial: bool = False) -> dict:
    raw = concrete or row
    spec = row.get("evaluation_spec") or {}
    budget = row.get("evaluation_conditions") or {}
    condition = _condition(row, concrete)
    elapsed = _elapsed(raw)
    if _elapsed(row) is not None:
        elapsed = _elapsed(row)
    scored = _scored(raw, concrete=True) if concrete is not None or raw_trial else _scored(row)
    limit = _number(spec.get("wall_seconds", row.get("wall_seconds", budget.get("wall_seconds"))))
    limit_hit = bool(row.get("wall_limit_reached") or row.get("stopped_for_budget")
                     or row.get("stop_reason") == "operational_wall_limit"
                     or budget.get("timed_out_by_batch_budget"))
    timeout_here = (not scored and limit_hit and
                    ((limit is not None and limit >= wall_seconds)
                     or (limit is None and elapsed is not None and elapsed >= wall_seconds)))
    compatible = _matches(condition, expected)
    completion_time = _time(raw.get("finished_at")) or _time(row.get("finished_at"))
    if completion_time is None:
        try:
            completion_time = path.stat().st_mtime
        except OSError:
            completion_time = 0.0
    execution = raw.get("agent_execution") or {}
    return {
        "source": str(path.resolve()), "concrete_result": str(concrete_path) if concrete_path else None,
        "matching_concrete_trials": matching_trials, "condition_matches": compatible,
        "scored": scored, "elapsed_seconds": elapsed,
        "agent_execution_seconds": _elapsed(execution), "completion_time": completion_time,
        "operational_limit_seconds": limit, "operational_limit_reached": limit_hit,
        "budget_incompatible": compatible and timeout_here,
        "completed_within_limit": bool(compatible and scored and elapsed is not None
                                       and elapsed <= wall_seconds and matching_trials <= 1),
        "transient_infrastructure": bool(not scored and not limit_hit and row.get("failure_layer") in {"infrastructure", "container", "scorer"}),
        "ambiguous_result_binding": matching_trials > 1,
    }


def build_readiness(root: Path, dataset: Path, tasks: list[str], wall_seconds: int = 900,
                    *, solver: dict | None = None) -> dict[str, dict]:
    """Freeze observed runtime compatibility without exposing score values.

    Unknown tasks remain explicitly unknown. A completed zero is exactly as
    eligible as a completed one. Both successful and unsuccessful durations are
    retained; no latest/favorable result is selected to erase a slow attempt.
    """
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, int) or wall_seconds <= 0:
        raise ValueError("wall_seconds must be a positive integer")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Task names must be unique")
    root, dataset = Path(root), Path(dataset)
    expected = dict(DEFAULT_SOLVER, **(solver or {}))
    observations = {task: [] for task in tasks}
    seen_trials = {}
    for path in sorted((root / "runs").rglob("task_run.json")):
        row = _load(path)
        if not row or _task_name(row) not in observations:
            continue
        concrete_path, concrete, count = _concrete_for(row, root)
        observation = _observation(path, row, expected, wall_seconds, concrete_path, concrete, count)
        observation["source_records"] = [str(path.resolve())]
        if concrete_path in seen_trials:
            original = seen_trials[concrete_path]
            original["source_records"].append(str(path.resolve()))
            original["operational_limit_reached"] |= observation["operational_limit_reached"]
            original["budget_incompatible"] |= observation["budget_incompatible"]
            continue
        observations[_task_name(row)].append(observation)
        if concrete_path:
            seen_trials[concrete_path] = observation
    # Some official runs do not have our TaskRun wrapper. Only concrete trials
    # with exact task/trial identity are considered, never aggregate results.
    for path in sorted((root / "runs").rglob("result.json")):
        if path.resolve() in seen_trials:
            continue
        row = _load(path)
        if (not row or row.get("task_name") not in observations
                or not _bound_trial(row, path, row["task_name"])):
            continue
        observations[row["task_name"]].append(_observation(path, row, expected, wall_seconds,
            path.resolve(), row, 1, raw_trial=True))
    result = {}
    for task, history in observations.items():
        fits = [row for row in history if row["completed_within_limit"]]
        limits = [row for row in history if row["budget_incompatible"]]
        transient = [row for row in history if row["condition_matches"] and row["transient_infrastructure"]]
        environment = dataset / "tasks" / task / "environment"
        available = environment.is_dir() and (environment / "Dockerfile").is_file()
        later_completion = bool(fits and limits and max(r["completion_time"] for r in fits) > max(r["completion_time"] for r in limits))
        if not available:
            status = "env_unavailable"
        elif limits and not later_completion:
            status = "known_budget_incompatible"
        elif limits and fits:
            status = "mixed_latency"
        elif fits:
            status = "completed_compatible"
        elif transient:
            status = "transient_infra"
        else:
            status = "unknown"
        result[task] = {
            "status": status, "wall_seconds": wall_seconds,
            "expected_solver": expected, "observation_count": len(history),
            "compatible_completion_count": len(fits), "budget_limit_count": len(limits),
            "transient_infrastructure_count": len(transient), "later_compatible_completion": later_completion,
            "noise_eligible": status == "completed_compatible",
            "initialization_eligible": status not in {"known_budget_incompatible", "mixed_latency", "env_unavailable"},
            "runtime_certified": status == "completed_compatible", "observations": history,
            "selection_basis": "execution configuration and completion time only; valid score magnitude is not used",
        }
    return result
