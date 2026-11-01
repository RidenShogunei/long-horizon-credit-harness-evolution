"""Deterministic, bounded selection of contemporaneous agent-visible evidence.

Only explicitly supplied trajectory files are read. Top-level evaluator data,
metrics, reasoning_content, hidden tests and other branches are never selected.
Omitted text is represented by exact source offsets, not silently concatenated.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SELECTOR_VERSION = "agent-visible-signals-v2"
VIEW_REVISION = 2
_COMMAND_REF = re.compile(r"\bCommand\s+(\d+)\s+(?:should|must|had|has|failed|timed|was|is\b)", re.I)
_SIGNALS = (
    (3, "parser_feedback", re.compile(r"Previous response had (?:warnings|parsing errors)|\bWARNINGS?:|\bPARS(?:E|ING)[ _-]?ERROR", re.I)),
    (2, "timeout", re.compile(r"\btimed out\b|\btimeout(?:error|\s+after)?\b", re.I)),
    (2, "runtime_error", re.compile(r"Traceback \(most recent call last\)|(?m:^\s*(?:error|exception|permission denied|command not found|no such file)\b)", re.I)),
    (1, "runtime_warning", re.compile(r"\b(?:\w*Warning|\w*Error|\w*Exception):", re.I)),
)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _visible(step: dict[str, Any], index: int) -> dict[str, Any] | None:
    if step.get("source") not in {"user", "agent", "assistant", "tool"}:
        return None
    obs = step.get("observation")
    contents = []
    if isinstance(obs, dict):
        if isinstance(obs.get("content"), str):
            contents.append(obs["content"])
        for result in obs.get("results", []):
            if isinstance(result, dict) and isinstance(result.get("content"), str):
                contents.append(result["content"])
    calls = []
    for call in step.get("tool_calls") or []:
        if isinstance(call, dict):
            calls.append({key: call[key] for key in ("tool_call_id", "function_name", "arguments") if key in call})
    return {"source_index": index, "step_id": step.get("step_id", index), "source": step["source"],
            "message": step.get("message") if isinstance(step.get("message"), str) else "",
            "tool_calls": calls, "observation": "\n".join(contents)}


def _action_views(calls: list[dict[str, Any]], budget: int, referenced: set[int]) -> list[dict[str, Any]]:
    """Retain every call's position and critical semantics outside body truncation.

    Each long keystroke body gets its own offset-bearing prefix/suffix slices.
    A middle empty wait can therefore never vanish behind two large commands.
    """
    per_call = max(0, budget // max(1, len(calls)))
    result = []
    for position, call in enumerate(calls, 1):
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        args = args if isinstance(args, dict) else {}
        keystrokes = args.get("keystrokes")
        has_text = isinstance(keystrokes, str)
        has_duration = "duration" in args or "duration_sec" in args
        if position in referenced and (not has_text or not has_duration):
            raise ValueError(f"feedback references Command {position} without recoverable keystrokes/duration")
        kept: dict[str, Any] = {}
        for key in ("duration", "duration_sec"):
            if key in args:
                kept[key] = args[key]
        kept["keystrokes_metadata"] = {
            "present": "keystrokes" in args, "is_string": has_text,
            "source_chars": len(keystrokes) if has_text else None,
            "is_empty": keystrokes == "" if has_text else None,
            "ends_with_newline": keystrokes.endswith("\n") if has_text else None,
            # Boundaries remain exact even when the body excerpt budget is zero.
            "first_char": keystrokes[:1] if has_text else None,
            "last_char": keystrokes[-1:] if has_text else None,
        }
        if has_text:
            kept["keystrokes_excerpt"] = _segments(keystrokes, per_call)
            if len(keystrokes) <= per_call or keystrokes == "":
                kept["keystrokes"] = keystrokes
        other = {key: value for key, value in args.items() if key not in {"keystrokes", "duration", "duration_sec"}}
        if other:
            kept["other_arguments_excerpt"] = _segments(_dump(other), max(0, per_call // 3))
        result.append({"position": position, "tool_call_id": call.get("tool_call_id"),
                       "function_name": call.get("function_name"), "arguments": kept,
                       "referenced_by_feedback": position in referenced})
    return result


def _segments(text: str, budget: int, centers: list[int] | None = None) -> dict[str, Any]:
    """Return exact slices with offsets; no lost text is disguised as adjacent."""
    if len(text) <= budget:
        spans = [(0, len(text))] if text else []
    elif budget <= 0:
        spans = []
    else:
        centers = list(dict.fromkeys(centers or []))
        # Priority feedback survives even when it lies in the middle of a long
        # terminal observation. Prefix/suffix context uses the remaining budget.
        anchors = centers[:3] if centers else [0, len(text) - 1]
        width = max(1, budget // len(anchors))
        spans = []
        for center in anchors:
            start = max(0, min(center - width // 3, len(text) - width))
            spans.append((start, min(len(text), start + width)))
        spans.sort()
        merged = []
        for start, end in spans:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        spans = merged
    return {"source_chars": len(text), "truncated": sum(b - a for a, b in spans) < len(text),
            "segments": [{"start": a, "end": b, "text": text[a:b]} for a, b in spans]}


def _render(step: dict[str, Any], reasons: list[str], cap: int) -> dict[str, Any]:
    matches = [m.start() for _, _, pattern in _SIGNALS for m in pattern.finditer(step["observation"])]
    body_budget = max(0, cap - 650)
    referenced = {int(match.group(1)) for match in _COMMAND_REF.finditer(step["observation"])}
    if any(position < 1 or position > len(step["tool_calls"]) for position in referenced):
        raise ValueError("feedback references an unavailable Command N; action context cannot be safely summarized")
    while True:
        weights = {"message": 0.15, "tool_calls": 0.35, "observation": 0.50}
        active_weight = sum(weight for key, weight in weights.items() if step[key]) or 1
        record = {key: step[key] for key in ("source_index", "step_id", "source")}
        record["selected_for"] = reasons
        record["visible_step_sha256"] = hashlib.sha256(_dump(step).encode()).hexdigest()
        record["referenced_command_positions"] = sorted(referenced)
        for key, weight in weights.items():
            field_budget = int(body_budget * weight / active_weight)
            if key == "tool_calls":
                record["tool_calls"] = _action_views(step[key], field_budget, referenced)
            else:
                record[key + "_excerpt"] = _segments(step[key], field_budget, matches if key == "observation" else None)
        if len(_dump(record)) <= cap:
            return record
        if body_budget == 0:
            # The caller validates a cap large enough for structural metadata.
            raise ValueError("per-step limit cannot preserve every call's critical arguments; increase the evidence budget")
        body_budget = max(0, int(body_budget * 0.75) - 1)


def select_trajectory_evidence(
    trajectory: Path, *, max_total_chars: int = 24000, max_step_chars: int = 3600,
    max_steps: int = 12, context_radius: int = 1, head_steps: int = 1,
    tail_steps: int = 1, decision_step_id: Any | None = None,
) -> list[dict[str, Any]]:
    """Return collection-compatible evidence; reading never scans sibling paths.

    decision_step_id optionally restricts the view to an earlier decision. The
    complete file is hashed for provenance, while only prefix-visible messages
    can appear in the selected evidence. Rewards should be attached separately
    from the authorized parent TaskRun, never taken from final_metrics here.
    """
    trajectory = Path(trajectory).resolve()
    if max_total_chars < 3000 or max_step_chars < 800 or max_steps < 3:
        raise ValueError("limits must permit signal, action and boundary context")
    if not 0 <= context_radius <= 3 or head_steps < 1 or tail_steps < 1:
        raise ValueError("invalid context settings")
    raw = trajectory.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    source_steps = data.get("steps")
    if not isinstance(source_steps, list):
        raise ValueError("trajectory has no steps list")
    visible = []
    cutoff_found = decision_step_id is None
    for index, step in enumerate(source_steps):
        if not isinstance(step, dict):
            continue
        item = _visible(step, index)
        if item:
            visible.append(item)
        if decision_step_id is not None and step.get("step_id") == decision_step_id:
            cutoff_found = True
            break
    if not cutoff_found:
        raise ValueError("decision_step_id does not occur in this trajectory")
    if not visible:
        raise ValueError("trajectory has no supported agent-visible messages")
    signals = []
    for i, step in enumerate(visible):
        hits = [(priority, kind) for priority, kind, pattern in _SIGNALS if pattern.search(step["observation"])]
        if hits:
            signals.append((max(p for p, _ in hits), i, sorted({kind for _, kind in hits})))
    signals.sort(key=lambda row: (-row[0], row[1]))
    reasons: dict[int, list[str]] = {}

    def add(index: int, reason: str) -> None:
        if index in reasons:
            if reason not in reasons[index]:
                reasons[index].append(reason)
        elif len(reasons) < max_steps:
            reasons[index] = [reason]

    # Reserve boundaries, then prioritize signals and their adjacent actions.
    for i in list(range(min(head_steps, len(visible)))) + list(range(max(0, len(visible) - tail_steps), len(visible))):
        add(i, "boundary_context")
    for _, i, kinds in signals:
        add(i, "signal:" + ",".join(kinds))
        for delta in range(1, context_radius + 1):
            for neighbor in (i - delta, i + delta):
                if 0 <= neighbor < len(visible):
                    add(neighbor, f"adjacent_to_step:{visible[i]['step_id']}")
    # Fill remaining budget with deterministic chronological context, without
    # treating absent selected signals as proof that the whole run was clean.
    for i in range(len(visible)):
        add(i, "additional_context")
    base = {"selector_version": SELECTOR_VERSION, "view_revision": VIEW_REVISION,
            "origin_id": "trajectory_" + digest, "trajectory_sha256": digest,
            "decision_step_id": decision_step_id, "visible_steps_at_cutoff": len(visible),
            "selected_step_ids": [visible[i]["step_id"] for i in sorted(reasons)],
            "omitted_step_ids": [s["step_id"] for i, s in enumerate(visible) if i not in reasons],
            "signal_step_ids": [visible[i]["step_id"] for _, i, _ in signals],
            "omitted_signal_step_ids": [visible[i]["step_id"] for _, i, _ in signals if i not in reasons],
            "diagnosis_scope": "selected agent-visible message excerpts only; absence in this bounded view does not establish absence in the full trajectory"}
    base_chars = len(_dump(base)) + 20
    cap = min(max_step_chars, (max_total_chars - base_chars) // len(reasons) - 2)
    if cap < 800:
        raise ValueError("total limit too small for selected context; increase budget or lower max_steps")
    while True:
        payload = dict(base, steps=[_render(visible[i], reasons[i], cap) for i in sorted(reasons)])
        text = _dump(payload)
        if len(text) <= max_total_chars:
            break
        cap -= max(1, (len(text) - max_total_chars) // len(reasons) + 1)
    return [{"id": "trajectory_view_" + hashlib.sha256(text.encode()).hexdigest()[:24],
             "text": text, "path": str(trajectory), "source_sha256": digest,
             "view_limited": True, "selector_version": SELECTOR_VERSION,
             "origin_id": "trajectory_" + digest, "view_revision": VIEW_REVISION}]
