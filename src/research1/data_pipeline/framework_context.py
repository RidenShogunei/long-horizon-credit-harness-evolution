"""Source-bound, read-only framework context for the five-module planner.

The builder reads only a fixed source allowlist. It never imports the runtime,
walks a task/output directory, serializes the protocol, or reads environment
variables. Source excerpts are interfaces and call sites, not editing targets.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CONTEXT_VERSION = "research1.framework-context.v1"
_PREFIX = "src/harbor/agents/"
_PROTOCOLS = _PREFIX + "terminus_2_modular/protocols.py"
_KERNEL = _PREFIX + "terminus_2_modular/kernel/orchestration.py"
_AGENT = _PREFIX + "terminus_2_modular/agent.py"
_STATIC = _PREFIX + "terminus_2_modular/composer/static.py"
_TMUX = _PREFIX + "terminus_2/tmux_session.py"
_PARSER = {name: _PREFIX + f"terminus_2/terminus_{name}_plain_parser.py" for name in ("json", "xml")}
_TEMPLATE = {name: _PREFIX + f"terminus_2/templates/terminus-{name}-plain.txt" for name in ("json", "xml")}
_ALLOWLIST = frozenset((_PROTOCOLS, _KERNEL, _AGENT, _STATIC, _TMUX, *_PARSER.values(), *_TEMPLATE.values()))
_MODULES = ("agent_loop", "observation", "tools", "context_mgmt", "verification")
_PROTOCOL_SYMBOLS = (
    "RuntimeState", "SharedResources", "FailureReporter", "StatsWriter", "Logger",
    "TrajectoryRecorder", "KernelServices", "ModuleCtx", "ObsResult", "ObsState",
    "ToolCall", "ToolResult", "LLMResponseParseResult", "AgentLoopState", "AgentLoopResult",
    "Observation", "CompressResult", "ContextMgmt", "ToolSet", "VerificationLoop", "AgentLoop",
    "ModuleSpec", "ModuleBundle",
)
_DEFAULT_NAMES = ("parser_name", "enable_summarize", "proactive_summarization_threshold",
                  "tmux_pane_width", "tmux_pane_height", "interleaved_thinking")
_MAX_SOURCE_BYTES = 250_000
_INTERPRETATION_LIMITS = [
    "Source describes available runtime fields and call paths; it does not expose their live environment values.",
    "The original instruction is available to modules through ctx.state.instruction. Mutable chat, observation and completion state follow the displayed Protocols.",
    "The displayed agent_loop code determines which observations and continuation prompts reach the solver. A verification return reason is not automatically solver feedback.",
    "A tool transport success, completion signal or a printed test claim is not proof of task correctness.",
    "Source and task observations may justify a testable process hypothesis. Expected benefit remains unproven until a saved implementation and applicable checks are evaluated.",
    "These excerpts are bounded and omit other runtime implementations. An omitted field or interface is not proof it does not exist.",
]
_DEFAULT_CAVEAT = "Constructor defaults are source facts; a composed module's parameters can override them."


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read(root: Path, relative: str) -> tuple[str, str]:
    if relative not in _ALLOWLIST:
        raise ValueError("Framework source is outside the fixed allowlist")
    root = Path(root).resolve(strict=True)
    cursor = root
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Framework context cannot follow symlinked source")
    target = cursor.resolve(strict=True)
    if root not in target.parents or not target.is_file():
        raise ValueError("Framework source escapes the runtime root")
    if target.stat().st_size > _MAX_SOURCE_BYTES:
        raise ValueError("Framework source exceeds its bounded source size")
    raw = target.read_bytes()
    return raw.decode("utf-8"), _sha(raw)


def _symbol(tree: ast.AST, name: str) -> ast.AST:
    nodes = [node for node in getattr(tree, "body", ()) if getattr(node, "name", None) == name]
    if len(nodes) != 1:
        raise ValueError("Required framework symbol is missing or ambiguous: " + name)
    return nodes[0]


def _segment(path: str, source: str, source_hash: str, symbol: str, node: ast.AST | None = None) -> dict:
    if node is None:
        start, end = 1, len(source.splitlines())
    else:
        start, end = node.lineno, node.end_lineno
        decorators = getattr(node, "decorator_list", ())
        if decorators:
            start = min(start, *(decorator.lineno for decorator in decorators))
    text = "\n".join(source.splitlines()[start - 1:end])
    return {"path": path, "symbol": symbol, "start_line": start, "end_line": end,
            "source_sha256": source_hash, "excerpt_sha256": _sha(text.encode("utf-8")),
            "access": "read_only", "text": text}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return prefix + "." + node.attr if prefix else node.attr
    return ""


def _assigned_names(node: ast.AST) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return {item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)}


def _constructor_defaults(tree: ast.AST) -> dict:
    cls = _symbol(tree, "Terminus2Modular")
    ctor = _symbol(cls, "__init__")
    args = ctor.args.args
    defaults = dict(zip((arg.arg for arg in args[-len(ctor.args.defaults):]), ctor.args.defaults))
    selected = {}
    for name in _DEFAULT_NAMES:
        if name not in defaults:
            raise ValueError("Required constructor default is absent: " + name)
        try:
            value = ast.literal_eval(defaults[name])
        except (ValueError, TypeError) as exc:
            raise ValueError("Framework default must be a literal: " + name) from exc
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError("Unsupported framework constructor default")
        selected[name] = value
    return selected


def _configuration(protocol: dict, defaults: dict, evaluation_spec: dict | None) -> dict:
    """Project a tiny typed view; unlisted config/identity/credential fields vanish."""
    runtime = protocol.get("runtime", {})
    solver = protocol.get("policies", {}).get("solver", {})
    values = {"max_turns": runtime.get("max_turns"), "wall_seconds": runtime.get("wall_seconds")}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Invalid frozen runtime limit: " + name)
    configuration = {"limits": values, "constructor_defaults": defaults,
        "default_caveat": _DEFAULT_CAVEAT,
        "binding": "declared_frozen_protocol"}
    if evaluation_spec is not None:
        if evaluation_spec.get("role") != "proposal":
            raise ValueError("Only a finished decision-time proposal execution can supply runtime settings")
        for name, value in values.items():
            if evaluation_spec.get(name) != value:
                raise ValueError("Proposal runtime setting differs from the frozen protocol: " + name)
        if not isinstance(evaluation_spec.get("composer"), dict) or evaluation_spec["composer"] != {"name": "static", "scope": "all"}:
            raise ValueError("Framework context currently supports the frozen static composer")
        temperature = evaluation_spec.get("temperature")
        effort = evaluation_spec.get("reasoning_effort")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError("Invalid proposal sampling temperature")
        if effort is not None and effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Unrecognized proposal reasoning setting")
        if temperature != solver.get("temperature") or effort != solver.get("reasoning_effort"):
            raise ValueError("Proposal sampling settings differ from the frozen solver policy")
        model_info = evaluation_spec.get("model_info", {})
        token_limits = {}
        for name in ("max_input_tokens", "max_output_tokens"):
            value = model_info.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Invalid proposal token limit")
            token_limits[name] = value
        configuration.update(binding="proposal_execution_spec", composer={"name": "static", "scope": "all"},
                             sampling={"temperature": temperature, "reasoning_effort": effort}, token_limits=token_limits)
    return configuration


def build_framework_context(upstream_root: Path, protocol: dict | None = None, *, runtime_limits: dict | None = None,
                            evaluation_spec: dict | None = None) -> dict:
    """Freeze actual runtime interfaces without exposing tasks or execution artifacts.

    Call before teacher/diagnostic calls, include the resulting object in the
    immutable state and protocol provenance, and render it unchanged for training.
    Source hashes must be covered by the campaign's existing runtime-source hash.
    A caller passing an execution spec must separately establish that execution
    finished before this decision and belongs to its exact parent implementation.
    """
    if protocol is not None and runtime_limits is not None:
        raise ValueError("Pass a frozen protocol or runtime_limits, not both")
    if protocol is None:
        protocol = {"runtime": runtime_limits or {}}
    sources = {}
    for path in (_PROTOCOLS, _KERNEL, _AGENT, _STATIC, _TMUX):
        sources[path] = _read(upstream_root, path)
    defaults = _constructor_defaults(ast.parse(sources[_AGENT][0]))
    parser = defaults["parser_name"]
    if parser not in _PARSER:
        raise ValueError("Unsupported active parser default")
    for path in (_PARSER[parser], _TEMPLATE[parser]):
        sources[path] = _read(upstream_root, path)
    fragments = []

    def add(path: str, symbol: str, node: ast.AST | None = None) -> None:
        source, source_hash = sources[path]
        fragments.append(_segment(path, source, source_hash, symbol, node))

    protocol_tree = ast.parse(sources[_PROTOCOLS][0])
    for name in _PROTOCOL_SYMBOLS:
        add(_PROTOCOLS, name, _symbol(protocol_tree, name))

    kernel_tree = ast.parse(sources[_KERNEL][0])
    run = _symbol(kernel_tree, "run_task")
    wanted_assignments = {"state", "ctx", "agent_loop_spec", "tools_spec", "context_mgmt_spec",
                          "agent_loop", "observation", "context_mgmt", "tools", "verification"}
    found = set()
    for statement in run.body:
        names = _assigned_names(statement) & wanted_assignments
        if names:
            found.update(names)
            add(_KERNEL, "run_task." + ",".join(sorted(names)), statement)
    if found != wanted_assignments:
        raise ValueError("Runtime module construction call sites changed")
    required_calls = {"tools.setup", "observation.capture", "tools.format_initial_prompt", "agent_loop.run", "tools.teardown"}
    found_calls = set()
    for statement in ast.walk(run):
        if not isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        calls = {_call_name(node.func) for node in ast.walk(statement) if isinstance(node, ast.Call)} & required_calls
        if calls:
            found_calls.update(calls)
            add(_KERNEL, "run_task." + ",".join(sorted(calls)), statement)
    if found_calls != required_calls:
        raise ValueError("Runtime lifecycle call sites changed")

    static_tree = ast.parse(sources[_STATIC][0])
    assignments = [node for node in static_tree.body if "DEFAULT_BUNDLE" in _assigned_names(node)]
    if len(assignments) != 1:
        raise ValueError("Default module bundle is missing or ambiguous")
    add(_STATIC, "DEFAULT_BUNDLE", assignments[0])
    add(_STATIC, "StaticComposer.choose", _symbol(_symbol(static_tree, "StaticComposer"), "choose"))

    # Active tool grammar and execution/observation semantics are external to the
    # five editable baseline files. The bounded source remains read-only.
    add(_PARSER[parser], "active_parser_source")
    add(_TEMPLATE[parser], "active_response_template")
    tmux = _symbol(ast.parse(sources[_TMUX][0]), "TmuxSession")
    for name in ("send_keys", "capture_pane", "get_incremental_output"):
        add(_TMUX, "TmuxSession." + name, _symbol(tmux, name))

    context = {"schema": CONTEXT_VERSION,
        "editing_scope": {"editable_files": [module + "/baseline.py" for module in _MODULES],
            "single_module_per_plan": True, "framework_fragments": "read_only",
            "rule": "Framework source clarifies runtime contracts; it does not expand editable locations or authorize environment, solver, task or evaluator changes."},
        "source_file_hashes": {path: value[1] for path, value in sorted(sources.items())},
        "source_fragments": fragments,
        "runtime_configuration": _configuration(protocol, defaults, evaluation_spec),
        "interpretation_limits": list(_INTERPRETATION_LIMITS)}
    context["context_id"] = "framework_" + _sha(_canonical(context).encode("utf-8"))
    return context


def validate_framework_context(context: dict) -> None:
    """Reject mutation before rendering a previously frozen context."""
    expected_keys = {"schema", "editing_scope", "source_file_hashes", "source_fragments", "runtime_configuration", "interpretation_limits", "context_id"}
    if not isinstance(context, dict) or context.get("schema") != CONTEXT_VERSION or set(context) != expected_keys:
        raise ValueError("Unsupported framework context")
    expected = "framework_" + _sha(_canonical({key: value for key, value in context.items() if key != "context_id"}).encode("utf-8"))
    if context.get("context_id") != expected:
        raise ValueError("Frozen framework context changed")
    if any(path not in _ALLOWLIST for path in context.get("source_file_hashes", {})):
        raise ValueError("Framework source is outside the fixed allowlist")
    expected_scope = {"editable_files": [module + "/baseline.py" for module in _MODULES],
        "single_module_per_plan": True, "framework_fragments": "read_only",
        "rule": "Framework source clarifies runtime contracts; it does not expand editable locations or authorize environment, solver, task or evaluator changes."}
    if context["editing_scope"] != expected_scope:
        raise ValueError("Framework editing boundary changed")
    if context["interpretation_limits"] != _INTERPRETATION_LIMITS:
        raise ValueError("Framework interpretation boundaries changed")
    expected_config = {"limits", "constructor_defaults", "default_caveat", "binding"}
    config = context["runtime_configuration"]
    if config.get("binding") == "proposal_execution_spec":
        expected_config |= {"composer", "sampling", "token_limits"}
    if (config.get("binding") not in {"declared_frozen_protocol", "proposal_execution_spec"}
            or set(config) != expected_config or set(config.get("constructor_defaults", {})) != set(_DEFAULT_NAMES)
            or config.get("default_caveat") != _DEFAULT_CAVEAT):
        raise ValueError("Framework runtime settings contain unlisted values")
    if set(config["limits"]) != {"max_turns", "wall_seconds"}:
        raise ValueError("Framework runtime limits contain unlisted values")
    for value in config["limits"].values():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Invalid framework runtime limit")
    defaults = config["constructor_defaults"]
    parser = defaults["parser_name"]
    if parser not in _PARSER:
        raise ValueError("Invalid framework parser default")
    for name in ("enable_summarize", "interleaved_thinking"):
        if not isinstance(defaults[name], bool):
            raise ValueError("Invalid framework boolean default")
    for name in ("proactive_summarization_threshold", "tmux_pane_width", "tmux_pane_height"):
        if isinstance(defaults[name], bool) or not isinstance(defaults[name], int) or defaults[name] <= 0:
            raise ValueError("Invalid framework numeric default")
    expected_sources = {_PROTOCOLS, _KERNEL, _AGENT, _STATIC, _TMUX, _PARSER[parser], _TEMPLATE[parser]}
    if set(context["source_file_hashes"]) != expected_sources or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in context["source_file_hashes"].values()):
        raise ValueError("Framework source hashes are incomplete or malformed")
    if config["binding"] == "proposal_execution_spec":
        if config["composer"] != {"name": "static", "scope": "all"} or set(config["sampling"]) != {"temperature", "reasoning_effort"}:
            raise ValueError("Framework runtime settings contain unlisted values")
        temperature, effort = config["sampling"]["temperature"], config["sampling"]["reasoning_effort"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2 or effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("Invalid framework sampling settings")
        if set(config["token_limits"]) != {"max_input_tokens", "max_output_tokens"} or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in config["token_limits"].values()):
            raise ValueError("Invalid framework token limits")
    if not isinstance(context["source_fragments"], list) or not 1 <= len(context["source_fragments"]) <= 70:
        raise ValueError("Framework excerpts must be bounded")
    for fragment in context.get("source_fragments", []):
        if set(fragment) != {"path", "symbol", "start_line", "end_line", "source_sha256", "excerpt_sha256", "access", "text"}:
            raise ValueError("Framework excerpt has unknown fields")
        if fragment.get("path") not in context["source_file_hashes"] or fragment.get("source_sha256") != context["source_file_hashes"][fragment["path"]]:
            raise ValueError("Framework excerpt source binding changed")
        if (not isinstance(fragment["symbol"], str) or not fragment["symbol"] or len(fragment["symbol"]) > 128
                or isinstance(fragment["start_line"], bool) or not isinstance(fragment["start_line"], int)
                or isinstance(fragment["end_line"], bool) or not isinstance(fragment["end_line"], int)
                or not 0 < fragment["start_line"] <= fragment["end_line"]
                or not isinstance(fragment["text"], str) or not fragment["text"]
                or len(fragment["text"].encode("utf-8")) > _MAX_SOURCE_BYTES
                or not isinstance(fragment["excerpt_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", fragment["excerpt_sha256"]) is None):
            raise ValueError("Malformed framework excerpt")
        if fragment.get("access") != "read_only" or fragment.get("excerpt_sha256") != _sha(fragment["text"].encode("utf-8")):
            raise ValueError("Framework excerpt content changed")
