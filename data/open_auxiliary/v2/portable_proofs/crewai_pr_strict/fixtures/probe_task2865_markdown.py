#!/usr/bin/env python3
"""Child probe: Task.prompt markdown flag (crewai-2865)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from task_probe_bootstrap import bootstrap_task_imports


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: probe task_path markdown_flag", file=sys.stderr)
        sys.exit(2)
    task_path = Path(sys.argv[1])
    markdown = sys.argv[2].lower() == "true"
    bootstrap_task_imports()
    spec = importlib.util.spec_from_file_location("crewai.task", task_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crewai.task"] = mod
    spec.loader.exec_module(mod)
    Task = mod.Task
    fields = {}
    if hasattr(Task, "model_fields") and "markdown" in Task.model_fields:
        fields["markdown"] = markdown
    task = Task.model_construct(
        description="Write a report",
        expected_output="A short report",
        **fields,
    )
    text = task.prompt()
    print(
        json.dumps(
            {
                "has_markdown_instruction": "Markdown syntax" in text,
                "len": len(text),
                "prompt": text,
            }
        )
    )


if __name__ == "__main__":
    main()
