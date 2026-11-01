"""Generic bounded implementation of a supplied plan as exact local edits.

This adapter supplies no diagnosis or repair recipe. Executable verification
and admission remain the responsibility of the collector's frozen contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research1.contracts import write_json


POLICY_ID = "exact_replacements_v1"


def apply_replacements(source: str, edits: list[dict[str, str]]) -> str:
    """Apply 1-4 unambiguous nonoverlapping replacements to the frozen source."""
    if not isinstance(edits, list) or not 1 <= len(edits) <= 4:
        raise ValueError("Expected 1-4 exact replacements")
    spans = []
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("Each replacement must be an object")
        old, new = edit.get("old"), edit.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str) or source.count(old) != 1:
            raise ValueError("Each old block must match the frozen source exactly once")
        start = source.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Overlapping replacements are rejected")
    for start, end, new in reversed(spans):
        source = source[:start] + new + source[end:]
    compile(source, "fixed_editor_replacement.py", "exec")
    return source


def compact_editor(plan: dict[str, Any], state: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Implement a tools plan without modifying its teacher input or output."""
    from research1.distill import _apply_edit, _tongji_chat

    if plan.get("target_module") != "tools" or plan.get("locations") != ["tools/baseline.py"]:
        return {"status": "pending", "reason": "fixed editor currently supports tools/baseline.py only"}
    snapshot = Path(state["modules_root"]) / "tools/baseline.py"
    source = snapshot.read_text(encoding="utf-8")
    bundle = state.get("checks") or {}
    scope = {key: bundle.get(key, []) for key in (
        "allowed_targets", "allowed_changed_functions", "allowed_changed_globals", "allowed_imports"
    )}
    prompt = (
        "Implement only the supplied modification plan in the frozen Python file. "
        'Return ONLY JSON {"edits":[{"old":"exact existing block","new":"replacement block"}]}. '
        "Use 1-4 exact nonoverlapping replacements. Each old block must occur exactly once "
        "in the frozen source. Do not return the complete file or invent a new diagnosis. "
        "Preserve all code and runtime behaviors unrelated to the supplied plan, including "
        "the supplied preservation requirements. Use only the frozen scope's approved "
        "function, global and import names. Do not add markers or hardcode task answers.\n"
        "Frozen scope: " + json.dumps(scope, ensure_ascii=False, sort_keys=True)
        + "\nSupplied plan: " + json.dumps(plan, ensure_ascii=False, sort_keys=True)
        + "\nFrozen source:\n" + source
    )
    write_json(directory / "editor_input.json", {
        "prompt": prompt, "plan": plan, "policy": POLICY_ID,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "temperature": 0.0, "max_tokens": 6000,
        "json_output": True, "reasoning_effort": "none",
    })
    raw, metadata = _tongji_chat(
        prompt, max_tokens=6000, role="collection_fixed_editor",
        json_output=True, reasoning_effort="none",
    )
    write_json(directory / "editor_output.json", {"raw": raw, "metadata": metadata})
    if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop":
        raise ValueError("Editor requires complete final content; reasoning or truncated outputs are not code")
    signature = {key: metadata.get(key) for key in ("provider", "model", "temperature")}
    policy_path = directory.parent / "fixed_editor_policy.json"
    if policy_path.exists() and json.loads(policy_path.read_text(encoding="utf-8")) != signature:
        return {"status": "pending", "reason": "editor model/provider/temperature changed within the frozen batch",
                "metadata": metadata}
    if metadata.get("temperature") != 0.0:
        return {"status": "pending", "reason": "fixed editor temperature differs from the frozen adapter policy",
                "metadata": metadata}
    expected_model = state.get("view", {}).get("config", {}).get("editor_model")
    if expected_model and metadata.get("model") != expected_model:
        return {"status": "pending", "reason": "editor model differs from the frozen policy view",
                "metadata": metadata}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Editor JSON must be an object with an edits list")
    replacement = apply_replacements(source, value.get("edits"))
    write_json(policy_path, signature)
    record = directory / "plan_record.json"
    write_json(record, {"plan": plan})
    old_plan = {"target": "tools/baseline.py", "intent": plan["intended_change"]
                + "\nPreserve: " + json.dumps(plan.get("preserve_behaviors", []), ensure_ascii=False),
                "trigger": None}
    result = _apply_edit(run_dir=directory, record_path=record, plan=old_plan,
                         snapshot=snapshot, replacement=replacement)
    if result.get("status") == "applied":
        result["modules_root"] = str(directory / "versions" / result["child_version_id"] / "modules")
    result.update(editor_metadata=metadata, editor_policy_id=POLICY_ID)
    return result
