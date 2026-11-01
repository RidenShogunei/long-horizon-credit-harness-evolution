#!/usr/bin/env python3
"""Child probe: Task.interpolate_inputs JSON-in-braces behavior (crewai-1791)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from task_probe_bootstrap import bootstrap_task_imports


def run_case(task_path: Path, mode: str) -> dict:
    bootstrap_task_imports()
    spec = importlib.util.spec_from_file_location("crewai.task", task_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crewai.task"] = mod
    spec.loader.exec_module(mod)
    Task = mod.Task
    if mode == "json_braces":
        eo = 'Summary for {user}: {"count": 1}'
        inputs = {"user": "Ada"}
        description = "Do work"
    elif mode == "missing_key":
        eo = "Hello {user}"
        description = "Need {missing} value"
        inputs = {"user": "Bob"}
    else:
        eo = "Hello {user}"
        description = "Do work"
        inputs = {"user": "Bob"}
    task = Task.model_construct(
        description=description,
        expected_output=eo,
    )
    out = {"error": None, "expected_output": None, "description": None}
    try:
        if hasattr(task, "interpolate_inputs_and_add_conversation_history"):
            task.interpolate_inputs_and_add_conversation_history(inputs)
        else:
            task.interpolate_inputs(inputs)
        out["expected_output"] = task.expected_output
        out["description"] = task.description
    except Exception as exc:  # noqa: BLE001
        out["error"] = type(exc).__name__
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: probe task_path mode", file=sys.stderr)
        sys.exit(2)
    task_path = Path(sys.argv[1])
    mode = sys.argv[2]
    print(json.dumps(run_case(task_path, mode)))


if __name__ == "__main__":
    main()
