"""Shared import stubs for loading pinned crewai.task modules in child probes."""
from __future__ import annotations

import sys
import types
from enum import Enum

from pydantic import BaseModel


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


class I18N(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    def slice(self, key: str) -> str:
        return f"<{key}>"


class _NotSpecified:
    pass


NOT_SPECIFIED = _NotSpecified()


class Span:
    pass


class Telemetry:
    def task_started(self, *args, **kwargs):
        pass

    def task_completed(self, *args, **kwargs):
        pass


def process_config(values, cls=None):
    return values


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


def bootstrap_task_imports() -> None:
    for mod_name in list(sys.modules):
        if mod_name == "crewai" or mod_name.startswith("crewai."):
            del sys.modules[mod_name]
    ot = ensure("opentelemetry.trace")
    ot.Span = Span
    ensure("crewai.agents.agent_builder.base_agent").BaseAgent = BaseAgent
    sec = ensure("crewai.security")
    sec.Fingerprint = Fingerprint
    sec.SecurityConfig = SecurityConfig
    gr_mod = ensure("crewai.tasks.guardrail_result")
    gr_mod.GuardrailResult = GuardrailResult
    of = ensure("crewai.tasks.output_format")
    of.OutputFormat = OutputFormat
    to = ensure("crewai.tasks.task_output")
    to.TaskOutput = TaskOutput
    tel = ensure("crewai.telemetry.telemetry")
    tel.Telemetry = Telemetry
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
    ev.crewai_event_bus = crewai_event_bus

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
