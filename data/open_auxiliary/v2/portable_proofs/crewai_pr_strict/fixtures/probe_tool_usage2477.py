#!/usr/bin/env python3
"""Child probe: ToolUsage.on_tool_use_finished emits output from tool result (crewai-2477)."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path

from pydantic import BaseModel


def ensure(name: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[sub] = mod
            if i > 1:
                setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    return sys.modules[name]


CAPTURED: list[dict] = []


class _Bus:
    @staticmethod
    def emit(_src, event):
        CAPTURED.append(
            {
                "type": getattr(event, "type", None),
                "output": getattr(event, "output", None),
                "tool_name": getattr(event, "tool_name", None),
            }
        )


def load_tool_usage(tool_usage_path: Path, events_path: Path):
    for name in list(sys.modules):
        if name == "crewai" or name.startswith("crewai."):
            del sys.modules[name]
    CAPTURED.clear()

    base = ensure("crewai.utilities.events.base_events")

    class CrewEvent(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    base.CrewEvent = CrewEvent
    ev_spec = importlib.util.spec_from_file_location(
        "crewai.utilities.events.tool_usage_events", events_path
    )
    assert ev_spec and ev_spec.loader
    ev = importlib.util.module_from_spec(ev_spec)
    sys.modules["crewai.utilities.events.tool_usage_events"] = ev
    ev_spec.loader.exec_module(ev)

    th = ensure("crewai.agents.tools_handler")

    class ToolsHandler:
        pass

    th.ToolsHandler = ToolsHandler
    task_mod = ensure("crewai.task")

    class Task:
        delegations = 0

        def increment_tools_errors(self):
            pass

    task_mod.Task = Task
    tel = ensure("crewai.telemetry.telemetry")

    class Telemetry:
        pass

    tel.Telemetry = Telemetry
    tel_pkg = ensure("crewai.telemetry")
    tel_pkg.Telemetry = Telemetry
    tools = ensure("crewai.tools")

    class BaseTool:
        pass

    tools.BaseTool = BaseTool
    st = ensure("crewai.tools.structured_tool")

    class CrewStructuredTool:
        pass

    st.CrewStructuredTool = CrewStructuredTool
    tc = ensure("crewai.tools.tool_calling")

    class ToolCalling:
        def __init__(self, tool_name: str, arguments: dict):
            self.tool_name = tool_name
            self.arguments = arguments

    class InstructorToolCalling(ToolCalling):
        pass

    tc.ToolCalling = ToolCalling
    tc.InstructorToolCalling = InstructorToolCalling
    i18n = ensure("crewai.utilities.i18n")

    class I18N:
        pass

    i18n.I18N = I18N
    ensure("crewai.utilities").I18N = I18N
    conv = ensure("crewai.utilities.converter")

    class Converter:
        pass

    class ConverterError(Exception):
        pass

    conv.Converter = Converter
    conv.ConverterError = ConverterError
    pr = ensure("crewai.utilities.printer")

    class Printer:
        def print(self, *a, **k):
            pass

    pr.Printer = Printer
    util = ensure("crewai.utilities")
    util.I18N = I18N
    util.Converter = Converter
    util.ConverterError = ConverterError
    util.Printer = Printer
    eb = ensure("crewai.utilities.events.crewai_event_bus")
    eb.crewai_event_bus = _Bus()

    import json as _json

    json5 = types.ModuleType("json5")
    json5.loads = _json.loads
    sys.modules["json5"] = json5
    jr = types.ModuleType("json_repair")
    jr.repair_json = lambda s: s
    sys.modules["json_repair"] = jr

    spec = importlib.util.spec_from_file_location("crewai.tools.tool_usage", tool_usage_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crewai.tools.tool_usage"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_probe(tool_usage_path: Path, events_path: Path) -> dict:
    mod = load_tool_usage(tool_usage_path, events_path)
    ToolUsage = mod.ToolUsage

    class Agent:
        key = "agent-key"
        role = "Researcher"
        _original_role = None
        verbose = False
        i18n = type("I18N", (), {})()

    class SearchTool:
        name = "search_tool"

    tool = SearchTool()
    calling = mod.ToolCalling("search_tool", {"q": "x"})
    task = mod.Task()
    handler = mod.ToolsHandler()
    tu = ToolUsage(
        tools_handler=handler,
        tools=[],
        original_tools=[],
        tools_description="",
        tools_names="",
        task=task,
        function_calling_llm=None,
        agent=Agent(),
        action=None,
    )
    started = time.time()
    tool_result = {"answer": "42"}
    sig = mod.ToolUsage.on_tool_use_finished
    try:
        if "result" in sig.__code__.co_varnames:
            sig(tu, tool, calling, False, started, tool_result)
        else:
            sig(tu, tool, calling, False, started)
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__, "events": list(CAPTURED)}

    finished = [e for e in CAPTURED if e.get("type") == "tool_usage_finished"]
    return {
        "error": None,
        "finished_count": len(finished),
        "output": finished[-1]["output"] if finished else None,
        "events": list(CAPTURED),
    }


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: probe tool_usage_path events_path", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(run_probe(Path(sys.argv[1]), Path(sys.argv[2]))))


if __name__ == "__main__":
    main()
