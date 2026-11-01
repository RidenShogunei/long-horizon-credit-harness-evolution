#!/usr/bin/env python3
"""Child probe: pinned Task._save_file with bootstrap stubs."""
from __future__ import annotations

import importlib.util
import sys
import types
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

# Allow import of fixture_paths from same fixtures dir
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_paths import assert_under_workspace, fixture_path, ensure_workspace_root  # noqa: E402


def ensure(name: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            if i < len(parts):
                mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[sub] = mod
            if i > 1:
                setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    return sys.modules[name]


class OutputFormat(str, Enum):
    RAW = "raw"
    JSON = "json"
    PYDANTIC = "pydantic"


class TaskOutput(BaseModel):
    raw: str = ""


class BaseAgent(BaseModel):
    role: str = "agent"


class BaseTool(BaseModel):
    name: str = "tool"


class SecurityConfig(BaseModel):
    pass


class Fingerprint(BaseModel):
    pass


class GuardrailResult(BaseModel):
    pass


class Converter:
    pass


class Printer:
    def print(self, *args, **kwargs):
        pass


class I18N:
    def slice(self, key: str) -> str:
        return key


class _NotSpecified:
    pass


NOT_SPECIFIED = _NotSpecified()


def process_config(*args, **kwargs):
    return {}


def process_guardrail(*args, **kwargs):
    return GuardrailResult()


def convert_to_model(*args, **kwargs):
    return None


def interpolate_only(input_string: str, inputs: dict) -> str:
    return input_string


class crewai_event_bus:
    @staticmethod
    def emit(*args, **kwargs):
        pass


def bootstrap() -> None:
    for mod_name in list(sys.modules):
        if mod_name == "crewai" or mod_name.startswith("crewai."):
            del sys.modules[mod_name]
    ensure("crewai.agents.agent_builder.base_agent").BaseAgent = BaseAgent
    sec = ensure("crewai.security")
    sec.Fingerprint = Fingerprint
    sec.SecurityConfig = SecurityConfig
    tasks = ensure("crewai.tasks")
    of = ensure("crewai.tasks.output_format")
    of.OutputFormat = OutputFormat
    to = ensure("crewai.tasks.task_output")
    to.TaskOutput = TaskOutput
    tools = ensure("crewai.tools.base_tool")
    tools.BaseTool = BaseTool
    cfg = ensure("crewai.utilities.config")
    cfg.process_config = process_config
    const = ensure("crewai.utilities.constants")
    const.NOT_SPECIFIED = NOT_SPECIFIED
    const._NotSpecified = _NotSpecified
    gr = ensure("crewai.utilities.guardrail")
    gr.process_guardrail = process_guardrail
    gr.GuardrailResult = GuardrailResult
    conv = ensure("crewai.utilities.converter")
    conv.Converter = Converter
    conv.convert_to_model = convert_to_model
    ev = ensure("crewai.utilities.events")

    class TaskCompletedEvent:
        pass

    class TaskFailedEvent:
        pass

    class TaskStartedEvent:
        pass

    ev.TaskCompletedEvent = TaskCompletedEvent
    ev.TaskFailedEvent = TaskFailedEvent
    ev.TaskStartedEvent = TaskStartedEvent
    evb = ensure("crewai.utilities.events.crewai_event_bus")
    evb.crewai_event_bus = crewai_event_bus
    i18n = ensure("crewai.utilities.i18n")
    i18n.I18N = I18N
    pr = ensure("crewai.utilities.printer")
    pr.Printer = Printer
    su = ensure("crewai.utilities.string_utils")
    su.interpolate_only = interpolate_only


def main() -> None:
    if len(sys.argv) != 5:
        print("usage: probe task_path case_tag rel_output create_directory|unset", file=sys.stderr)
        sys.exit(2)
    task_path = Path(sys.argv[1])
    case_tag = sys.argv[2]
    rel_out = sys.argv[3]
    mode = sys.argv[4].lower()
    create_directory = mode == "true"
    unset = mode == "unset"
    if unset:
        target = None
    else:
        target = fixture_path(case_tag, rel_out, mkdir=False)
        ensure_workspace_root()
    bootstrap()
    spec = importlib.util.spec_from_file_location("crewai.task", task_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crewai.task"] = mod
    spec.loader.exec_module(mod)
    Task = mod.Task
    if unset:
        task = Task.model_construct(output_file=None)
    elif hasattr(Task, "model_fields") and "create_directory" in Task.model_fields:
        task = Task.model_construct(
            output_file=str(target),
            create_directory=create_directory,
        )
    else:
        task = Task.model_construct(output_file=str(target))
    try:
        Task._save_file(task, "hello")
        if unset:
            print("UNEXPECTED_OK")
            sys.exit(0)
        assert target is not None
        print("OK", target.read_text(encoding="utf-8"))
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(type(exc).__name__, str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
