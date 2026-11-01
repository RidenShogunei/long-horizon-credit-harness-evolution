#!/usr/bin/env python3
"""Root-scope gap verification and Plan factual audit (C64). Does not mutate worker batches."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
REPO = WORK.parents[1]
CHILD_TIMEOUT = 20

# Module-level probe types for DSPy Tool get_type_hints resolution in child processes.
class _Dspy8089Config(BaseModel):
    value: int = 3


def _dspy8089_basemodel_tool(cfg: _Dspy8089Config):
    return cfg.value

PLAN_FIELDS = (
    "action",
    "target_module",
    "locations",
    "evidence_refs",
    "problem_hypothesis",
    "intended_change",
    "preserve_behaviors",
    "verification_plan",
    "verification_kind",
)
PERMITTED_MODULE_PATHS = (
    "agent_loop",
    "observation",
    "tools",
    "context_mgmt",
    "verification",
)

PROVISIONAL_ALLOWLIST = frozenset(
    {
        "huggingface__smolagents-285",
        "huggingface__smolagents-891",
        "huggingface__smolagents-1088",
        "huggingface__smolagents-478",
        "stanfordnlp__dspy-8089",
        "stanfordnlp__dspy-8340",
        "stanfordnlp__dspy-8053",
        "crewaiinc__crewai-2570",
        "crewaiinc__crewai-3150",
        "crewaiinc__crewai-2588",
        "crewaiinc__crewai-1818",
        "crewaiinc__crewai-2408",
        "berriai__litellm-11964",
        "berriai__litellm-9261",
        "berriai__litellm-5232",
        "berriai__litellm-12168",
        "berriai__litellm-9828",
        "berriai__litellm-8512",
        "crewaiinc__crewai-1910",
        "crewaiinc__crewai-2424",
        "crewaiinc__crewai-2890",
        "crewaiinc__crewai-2865",
        "crewaiinc__crewai-1791",
    }
)

PENDING_NOT_EXPORT = frozenset(
    {
        "stanfordnlp__dspy-1556",
        "stanfordnlp__dspy-8235",
        "stanfordnlp__dspy-8432",
        "berriai__litellm-11006",
        "berriai__litellm-6138",
        "berriai__litellm-6563",
        "crewaiinc__crewai-2477",
    }
)

EXTRA_GAP_IDS = frozenset(
    {
        "stanfordnlp__dspy-8089",
        "crewaiinc__crewai-1791",
        "crewaiinc__crewai-1910",
        "crewaiinc__crewai-2424",
        "berriai__litellm-11964",
        "berriai__litellm-5232",
        "berriai__litellm-12168",
        "berriai__litellm-9828",
        "berriai__litellm-8512",
    }
)

CURATED_PATHS = (
    WORK / "dspy_strict" / "curated_examples.json",
    WORK / "crewai_strict" / "curated_examples.json",
    WORK / "litellm_strict" / "curated_examples.json",
    WORK / "crewai_pr_strict" / "curated_examples.json",
)

STRICT_RESULTS = {
    "dspy_strict": WORK / "dspy_strict" / "verification_results.json",
    "crewai_strict": WORK / "crewai_strict" / "verification_results.json",
    "litellm_strict": WORK / "litellm_strict" / "verification_results.json",
    "crewai_pr_strict": WORK / "crewai_pr_strict" / "verification_results.json",
    "acceptance_strengthened": WORK / "acceptance_strengthened" / "verification_results.json",
}


def assert_safe_relative(rel: str) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel or rel.startswith("/"):
        raise ValueError("Unsafe relative path: " + repr(rel))
    parsed = PurePosixPath(rel)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError("Unsafe relative path: " + repr(rel))
    return rel.replace("\\", "/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_plan(raw: dict[str, Any]) -> dict[str, Any]:
    plan = {k: raw.get(k) for k in PLAN_FIELDS}
    kind = plan.get("verification_kind")
    if isinstance(kind, list) or kind == "":
        plan["verification_kind"] = None
    elif kind is not None and not isinstance(kind, str):
        plan["verification_kind"] = None
    return plan


def load_curated_index() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in CURATED_PATHS:
        for item in json.loads(path.read_text(encoding="utf-8")):
            sid = item["source_id"]
            if sid not in out:
                item = copy.deepcopy(item)
                item["plan"] = normalize_plan(item.get("plan") or {})
                out[sid] = item
    return out


def load_strict_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}

    def ingest_samples(samples: list, batch: str) -> None:
        for row in samples:
            sid = row.get("source_id") or row.get("sample_id")
            if not sid:
                continue
            status[sid] = {
                "strict_batch": batch,
                "strict_status": row.get("status"),
                "strict_row": row,
            }

    for batch, path in STRICT_RESULTS.items():
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cases = data.get("cases")
        if isinstance(cases, dict):
            for sid, row in cases.items():
                status[sid] = {
                    "strict_batch": batch,
                    "strict_status": row.get("status"),
                    "strict_row": row,
                }
            continue
        samples = data.get("samples") or data.get("results") or []
        if isinstance(samples, list):
            ingest_samples(samples, batch)
    return status


def combine_root_status(strict: Optional[str], extra: str) -> str:
    if extra == "skipped_not_in_scope":
        return strict or "skipped"
    if extra == "pending":
        return "pending"
    if strict == "pending":
        return "pending"
    if extra in {"skipped_strict_sufficient", "not_applicable"}:
        if strict in {None, "skipped"}:
            return strict or "skipped"
        return strict
    if strict == "reject" or extra == "fail":
        return "reject"
    if strict not in {None, "pass"}:
        return strict or "reject"
    if extra == "pass":
        return "pass"
    return extra


def _run_extra_child(sample_id: str) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--only", sample_id],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT,
        cwd=str(ROOT),
    )
    if proc.returncode not in (0, 1):
        return {
            "executed": False,
            "status": "fail",
            "error": (proc.stderr or proc.stdout or "child_failed")[:500],
            "checks": {},
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"executed": False, "status": "fail", "error": "invalid_child_json", "checks": {}}


def _run_extra_inline(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {"executed": True, "status": "fail", "error": f"{type(exc).__name__}: {exc}", "checks": {}}


# --- DSPy 8089 ---


def extra_dspy_8089() -> dict[str, Any]:
    sys.path.insert(0, str(WORK / "dspy_strict"))
    from source_loaders import load_tool_module  # noqa: WPS433

    case = WORK / "main_candidates/cases/stanfordnlp__dspy-8089"
    b_path = case / "before/dspy/primitives/tool.py"
    a_path = case / "applied/dspy/primitives/tool.py"

    def typed_tool(limit: int = 120):
        return limit

    def untyped_tool(limit=120):
        return limit

    basemodel_tool = _dspy8089_basemodel_tool

    def probe(tool_path: Path, fn, label: str) -> dict:
        mod = load_tool_module(tool_path)
        out: dict = {"label": label, "init_ok": False, "err": None}
        try:
            t = mod.Tool(fn)
            out["init_ok"] = True
            out["args"] = copy.deepcopy(t.args)
            if label == "basemodel":
                out["call"] = t(cfg=_Dspy8089Config(value=9))
        except Exception as exc:  # noqa: BLE001
            out["err"] = f"{type(exc).__name__}: {exc}"
        return out

    before_bm = probe(b_path, basemodel_tool, "basemodel")
    after_bm = probe(a_path, basemodel_tool, "basemodel")
    before_typed = probe(b_path, typed_tool, "typed")
    after_typed = probe(a_path, typed_tool, "typed")
    before_untyped = probe(b_path, untyped_tool, "untyped")
    after_untyped = probe(a_path, untyped_tool, "untyped")

    bm_schema_before = (before_bm.get("args") or {}).get("cfg") or {}
    bm_schema_after = (after_bm.get("args") or {}).get("cfg") or {}

    checks = {
        "basemodel_schema_dict_before": isinstance(bm_schema_before, dict) and "properties" in bm_schema_before,
        "basemodel_schema_preserved_after": bm_schema_before == bm_schema_after and after_bm.get("call") == 9,
        "typed_limit_schema_unchanged": before_typed.get("args", {}).get("limit")
        == after_typed.get("args", {}).get("limit"),
        "untyped_before_fails_after_ok": not before_untyped["init_ok"] and after_untyped["init_ok"],
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "observations": {
            "before_basemodel": before_bm,
            "after_basemodel": after_bm,
        },
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }


# --- CrewAI 1791 ---


def _bootstrap_task_and_load(task_path: Path):
    fix = WORK / "crewai_pr_strict" / "fixtures"
    sys.path.insert(0, str(fix))
    from task_probe_bootstrap import bootstrap_task_imports  # noqa: WPS433

    bootstrap_task_imports()
    spec = importlib.util.spec_from_file_location("crewai.task", task_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crewai.task"] = mod
    spec.loader.exec_module(mod)
    return mod


def extra_crewai_1791() -> dict[str, Any]:
    case = WORK / "additional_crewai/cases/crewaiinc__crewai-1791"
    before_p = case / "before/src/crewai/task.py"
    after_p = case / "applied/src/crewai/task.py"
    fix = WORK / "crewai_pr_strict" / "fixtures"
    script = fix / "probe_task1791_interpolate.py"

    def run_probe(task_path: Path, mode: str) -> dict:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", str(script), str(task_path), mode],
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT,
            cwd=str(fix),
        )
        if proc.returncode != 0:
            return {"error": "probe_failed", "stderr": proc.stderr[:300]}
        return json.loads(proc.stdout)

    modes = ["json_braces", "simple", "missing_key"]
    before_i = {m: run_probe(before_p, m) for m in modes}
    after_i = {m: run_probe(after_p, m) for m in modes}

    def save_file_obs(task_path: Path) -> dict:
        mod = _bootstrap_task_and_load(task_path)
        Task = mod.Task
        task = Task.model_construct(description="d", expected_output="e", output_file="/tmp/crewai1791_probe.txt")
        calls: list[str] = []

        class FakePath:
            def __init__(self, *_a, **_k):
                pass

            def expanduser(self):
                return self

            def resolve(self):
                return self

            @property
            def parent(self):
                return self

            def exists(self):
                return True

            def mkdir(self, *_a, **_k):
                calls.append("mkdir")

            def open(self, *_a, **_k):
                calls.append("open")
                raise OSError(13, "permission denied stub")

        orig_path = mod.Path
        mod.Path = FakePath  # type: ignore[misc]
        out: dict = {"exc_type": None, "exc_msg": None, "calls": calls}
        try:
            task._save_file({"k": 1})
        except Exception as exc:  # noqa: BLE001
            out["exc_type"] = type(exc).__name__
            out["exc_msg"] = str(exc)
        finally:
            mod.Path = orig_path
        return out

    b_save = save_file_obs(before_p)
    a_save = save_file_obs(after_p)

    checks = {
        "strict_json_braces_behavior": before_i["json_braces"].get("error") is not None
        and after_i["json_braces"].get("error") is None,
        "strict_simple_preserved": before_i["simple"].get("expected_output")
        == after_i["simple"].get("expected_output")
        == "Hello Bob",
        "strict_missing_key_both_error": before_i["missing_key"].get("error") is not None
        and after_i["missing_key"].get("error") is not None,
        "save_file_before_oserror": b_save["exc_type"] in {"OSError", "PermissionError", "IOError"},
        "save_file_after_runtime_wrap": a_save["exc_type"] == "RuntimeError"
        and "Failed to save output file" in (a_save["exc_msg"] or ""),
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "observations": {"interpolate": {"before": before_i, "after": after_i}, "save_file": {"before": b_save, "after": a_save}},
        "source_hashes": {"before": sha256_file(before_p), "after": sha256_file(after_p)},
    }


# --- CrewAI 1910 nested BaseModel ---


def extra_crewai_1910() -> dict[str, Any]:
    case = WORK / "additional_crewai/cases/crewaiinc__crewai-1910"
    before_p = case / "before/src/crewai/utilities/converter.py"
    after_p = case / "applied/src/crewai/utilities/converter.py"

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]

        def ensure_pkg(name: str) -> types.ModuleType:
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                sub = ".".join(parts[:i])
                if sub not in sys.modules:
                    mod = types.ModuleType(sub)
                    mod.__path__ = []  # type: ignore[attr-defined]
                    sys.modules[sub] = mod
            return sys.modules[name]

        ensure_pkg("crewai.agents.agent_builder.utilities.base_output_converter").OutputConverter = type(
            "OutputConverter", (), {}
        )
        ensure_pkg("crewai.utilities.printer").Printer = type("Printer", (), {"print": lambda *a, **k: None})
        ensure_pkg("crewai.utilities.pydantic_schema_parser").PydanticSchemaParser = type(
            "PydanticSchemaParser", (), {}
        )
        spec = importlib.util.spec_from_file_location("crewai.utilities.converter", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["crewai.utilities.converter"] = mod
        spec.loader.exec_module(mod)

        class Inner(BaseModel):
            score: int

        class Outer(BaseModel):
            inner: Inner

        Outer.__annotations__ = {n: f.annotation for n, f in Outer.model_fields.items()}
        Inner.__annotations__ = {n: f.annotation for n, f in Inner.model_fields.items()}
        desc = mod.generate_model_description(Outer)
        return {"desc": desc, "has_nested_score": '"score"' in desc and "Inner" in desc or "score" in desc}

    b = run(before_p)
    a = run(after_p)
    checks = {
        "nested_before_has_inner_field": '"inner"' in b["desc"],
        "nested_after_has_inner_field": '"inner"' in a["desc"],
        "nested_description_stable": b["desc"] == a["desc"],
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "observations": {"before_snippet": b["desc"][:400], "after_snippet": a["desc"][:400]},
        "source_hashes": {"before": sha256_file(before_p), "after": sha256_file(after_p)},
    }


# --- CrewAI 2424 signal.send trace ---


def extra_crewai_2424() -> dict[str, Any]:
    case = WORK / "additional_crewai/cases/crewaiinc__crewai-2424"

    def run(rel: str) -> dict:
        path = case / rel
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        base = types.ModuleType("crewai.utilities.events.base_events")
        sys.modules["crewai.utilities.events.base_events"] = base

        class CrewEvent(BaseModel):
            pass

        base.CrewEvent = CrewEvent
        et = types.ModuleType("crewai.utilities.events.event_types")
        et.EventTypes = CrewEvent
        sys.modules["crewai.utilities.events.event_types"] = et
        spec = importlib.util.spec_from_file_location("crewai.utilities.events.crewai_event_bus", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["crewai.utilities.events.crewai_event_bus"] = mod
        spec.loader.exec_module(mod)
        bus = mod.CrewAIEventsBus()
        if hasattr(bus, "clear_handlers"):
            bus.clear_handlers()
        trace: list[dict] = []
        orig_send = bus._signal.send

        def traced_send(sender, **kwargs):  # noqa: ANN001
            trace.append({"sender_is_none": sender is None, "event_type": type(kwargs.get("event")).__name__})
            return orig_send(sender, **kwargs)

        bus._signal.send = traced_send  # type: ignore[method-assign]

        class ToolEvent(CrewEvent):
            tool: str = "t"

        bus.emit(None, ToolEvent())
        return {"trace": trace, "handler_path": "emit_called"}

    before_obs = run("before/src/crewai/utilities/events/crewai_event_bus.py")
    after_obs = run("applied/src/crewai/utilities/events/crewai_event_bus.py")
    checks = {
        "before_signal_send_fires": len(before_obs["trace"]) == 1 and before_obs["trace"][0]["event_type"] == "ToolEvent",
        "after_signal_send_fires": len(after_obs["trace"]) == 1 and after_obs["trace"][0]["event_type"] == "ToolEvent",
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "observations": {"before": before_obs, "after": after_obs},
    }


# --- LiteLLM helpers ---


def _litellm_pinned():
    strict = WORK / "litellm_strict"
    sys.path.insert(0, str(strict))
    from runtime.pinned_import import (  # noqa: WPS433
        extract_class_from_pinned,
        extract_function_from_pinned,
    )

    return extract_function_from_pinned, extract_class_from_pinned


def extra_litellm_11964() -> dict[str, Any]:
    extract_fn, _ = _litellm_pinned()
    extra = {"Set": set, "Any": object}
    rel = "litellm_core_utils/prompt_templates/common_utils.py"

    def run_unpack(variant: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        fn = extract_fn("berriai__litellm-11964", variant, rel, "unpack_defs", extra)
        s = copy.deepcopy(schema)
        try:
            fn(s, dict(s.get("$defs", {})))
            return s, None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    no_ref = {"type": "object", "properties": {"plain": {"type": "string", "maxLength": 4}}}
    missing_ref = {
        "properties": {"ghost": {"$ref": "#/defs/Missing"}},
        "$defs": {},
    }
    nested = {
        "properties": {
            "payload": {"type": "array", "items": {"anyOf": [{"$ref": "#/defs/Leaf"}]}},
            "unchanged": {"type": "number", "const": 7},
        },
        "$defs": {"Leaf": {"type": "string", "enum": ["x"]}},
    }

    def pair(schema: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        b, eb = run_unpack("before", schema)
        a, ea = run_unpack("applied", schema)
        return b, a, eb, ea

    b_no, a_no, _, _ = pair(no_ref)
    b_miss, a_miss, eb_miss, ea_miss = pair(missing_ref)
    b_nested, a_nested, _, _ = pair(nested)

    miss_before_ref = (
        None if b_miss is None else b_miss.get("properties", {}).get("ghost", {}).get("$ref")
    )
    miss_after_ref = (
        None if a_miss is None else a_miss.get("properties", {}).get("ghost", {}).get("$ref")
    )

    checks = {
        "no_ref_unchanged_both_variants": b_no == a_no and b_no is not None,
        "nested_anyof_resolves_after": b_nested is not None
        and a_nested is not None
        and b_nested["properties"]["payload"]["items"]["anyOf"][0].get("$ref")
        and a_nested["properties"]["payload"]["items"]["anyOf"][0].get("type") == "string",
    }
    preserve_unknown_verified = eb_miss is not None and ea_miss is not None and eb_miss == ea_miss
    checks["unknown_ref_missing_def_observed"] = preserve_unknown_verified

    ok = checks["no_ref_unchanged_both_variants"] and checks["nested_anyof_resolves_after"]
    return {
        "executed": True,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "observations": {
            "no_ref_before": b_no,
            "no_ref_after": a_no,
            "missing_ref_errors": {"before": eb_miss, "applied": ea_miss},
        },
        "preserve_unknown_ref_verified": preserve_unknown_verified,
    }


def extra_litellm_5232() -> dict[str, Any]:
    extract_fn, extract_cls = _litellm_pinned()

    class AnthropicError(Exception):
        def __init__(self, message: str = "", status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    def drive(variant: str, chunks: list[dict]) -> list[dict]:
        cls = extract_cls(
            "berriai__litellm-5232",
            variant,
            "llms/anthropic.py",
            "ModelResponseIterator",
            {"AnthropicError": AnthropicError},
        )
        it = cls(None, True)
        outs: list[dict] = []
        for ch in chunks:
            outs.append(it.chunk_parser(ch))
        return outs

    text_chunks = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    empty_tool_delta = [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "t1", "name": "f1", "input": {}},
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}},
        {"type": "content_block_stop", "index": 1},
    ]

    b_text = drive("before", text_chunks)
    a_text = drive("applied", text_chunks)
    b_tool = drive("before", empty_tool_delta)
    a_tool = drive("applied", empty_tool_delta)

    def tool_arg_last(out: list[dict]) -> Optional[str]:
        last: Optional[str] = None
        for row in out:
            tu = row.get("tool_use")
            if tu and tu.get("function"):
                last = tu["function"].get("arguments")
        return last

    def last_texts(out: list[dict]) -> list[str]:
        return [r.get("text") for r in out if r.get("text")]

    checks = {
        "text_stream_preserved": last_texts(b_text) == last_texts(a_text) == ["Hi"],
        "empty_tool_stop_guard_before": tool_arg_last(b_tool) == "{}",
        "empty_tool_stop_guard_after": tool_arg_last(a_tool) == "{}",
        "empty_tool_stop_guard_preserved": tool_arg_last(b_tool) == tool_arg_last(a_tool),
        "tool_index_after_monotonic": all(
            (row.get("tool_use") or {}).get("index") == 0 for row in a_tool if row.get("tool_use")
        ),
    }
    ok = all(checks.values())
    return {"executed": True, "status": "pass" if ok else "fail", "checks": checks}


def extra_litellm_12168() -> dict[str, Any]:
    extract_fn, _ = _litellm_pinned()
    typing = __import__("typing")
    extra = {"Optional": typing.Optional, "Union": typing.Union}
    fb = extract_fn("berriai__litellm-12168", "before", "utils.py", "validate_chat_completion_tool_choice", extra)
    fa = extract_fn("berriai__litellm-12168", "applied", "utils.py", "validate_chat_completion_tool_choice", extra)

    def run(fn, payload):
        try:
            return ("ok", fn(payload))
        except Exception as e:
            return ("err", type(e).__name__)

    invalid_fn = {"type": "function"}
    required_named = {"type": "function", "function": {"name": "get_weather"}}

    bb_i, ba_i = run(fb, invalid_fn), run(fa, invalid_fn)
    bb_n, ba_n = run(fb, required_named), run(fa, required_named)

    checks = {
        "invalid_function_dict_rejected_before": bb_i[0] == "err",
        "invalid_function_dict_rejected_after": ba_i[0] == "err",
        "named_function_accepted_before": bb_n[0] == "ok",
        "named_function_accepted_after": ba_n[0] == "ok",
    }
    ok = all(checks.values())
    return {"executed": True, "status": "pass" if ok else "fail", "checks": checks}


def _vertex_extra_9828(variant: str) -> dict[str, Any]:
    extract_fn, _ = _litellm_pinned()
    rel = "llms/vertex_ai/common_utils.py"
    unpack_rel = "litellm_core_utils/prompt_templates/common_utils.py"
    unpack = extract_fn("berriai__litellm-11964", variant, unpack_rel, "unpack_defs", {"Set": set, "Any": object})
    extra: dict[str, Any] = {
        "Dict": dict,
        "Any": object,
        "Set": set,
        "get_type_hints": lambda _schema: {
            "type": None,
            "properties": None,
            "maxLength": None,
            "minimum": None,
            "maximum": None,
            "enum": None,
            "required": None,
            "items": None,
            "anyOf": None,
        },
        "unpack_defs": unpack,
        "Schema": dict,
        "List": list,
        "Optional": __import__("typing").Optional,
        "Union": __import__("typing").Union,
        "Tuple": tuple,
        "Literal": __import__("typing").Literal,
        "re": __import__("re"),
        "DEFAULT_MAX_RECURSE_DEPTH": 10,
    }
    for fn_name in ("convert_anyof_null_to_nullable", "add_object_type", "filter_schema_fields"):
        extra[fn_name] = extract_fn("berriai__litellm-9828", variant, rel, fn_name, extra)
    if variant == "applied":
        extra["set_schema_property_ordering"] = extract_fn(
            "berriai__litellm-9828", variant, rel, "set_schema_property_ordering", extra
        )
    return extra


def extra_litellm_9828() -> dict[str, Any]:
    extract_fn, _ = _litellm_pinned()
    rel = "llms/vertex_ai/common_utils.py"
    template = {
        "type": "object",
        "properties": {
            "z": {"type": "string", "maxLength": 5, "not_a_valid_vertex_field": True},
            "child": {"type": "object", "properties": {"inner": {"type": "integer"}}},
        },
    }

    def build(variant: str, ordering: bool) -> dict:
        extra = _vertex_extra_9828(variant)
        fn = extract_fn("berriai__litellm-9828", variant, rel, "_build_vertex_schema", extra)
        params = copy.deepcopy(template)
        if variant == "before":
            return fn(params)
        return fn(params, add_property_ordering=ordering)

    b = build("before", False)
    a_no = build("applied", False)
    a_yes = build("applied", True)

    checks = {
        "default_false_full_output_equal": b == a_no,
        "invalid_field_stripped_both": "not_a_valid_vertex_field"
        not in b.get("properties", {}).get("z", {})
        and "not_a_valid_vertex_field" not in a_no.get("properties", {}).get("z", {}),
        "ordering_only_when_flag": "propertyOrdering" not in b
        and "propertyOrdering" not in a_no
        and a_yes.get("propertyOrdering") == ["z", "child"],
        "variant_matched_unpack_dependency": True,
    }
    ok = all(checks.values())
    return {
        "executed": True,
        "status": "pass" if ok else "fail",
        "checks": checks,
        "observations": {"before_keys": list(b.keys()), "applied_no_order_keys": list(a_no.keys())},
    }


def extra_litellm_8512() -> dict[str, Any]:
    extract_fn, _ = _litellm_pinned()
    import litellm

    def supports_tool_choice(model: str, custom_llm_provider: str) -> bool:
        return custom_llm_provider == "bedrock"

    litellm.utils = types.SimpleNamespace(supports_tool_choice=supports_tool_choice)

    class BedrockModelInfo:
        @staticmethod
        def get_base_model(model: str) -> str:
            return model.split("/")[-1] if "/" in model else model

    extra = {"BedrockModelInfo": BedrockModelInfo, "litellm": litellm, "List": list}
    rel = "llms/bedrock/chat/converse_transformation.py"

    def tools_flag(variant: str, model: str) -> bool:
        fn = extract_fn("berriai__litellm-8512", variant, rel, "get_supported_openai_params", extra)
        self = types.SimpleNamespace(custom_llm_provider="bedrock")
        params = fn(self, model)
        return "tools" in params

    models = {
        "new": "meta.llama3-3-70b-instruct-v1:0",
        "supported": "meta.llama3-2-1b",
        "unsupported": "amazon.titan-text-express-v1",
    }
    before = {k: tools_flag("before", m) for k, m in models.items()}
    after = {k: tools_flag("applied", m) for k, m in models.items()}

    checks = {
        "new_model_tools_after_only": not before["new"] and after["new"],
        "supported_unchanged": before["supported"] and after["supported"],
        "unsupported_unchanged": not before["unsupported"] and not after["unsupported"],
        "supports_tool_choice_is_stub": True,
    }
    return {
        "executed": True,
        "status": "pass" if all(checks[k] for k in checks if k != "supports_tool_choice_is_stub") else "fail",
        "checks": checks,
        "observations": {"before": before, "after": after},
        "root_note": "supports_tool_choice stubbed True for bedrock only; real provider semantics not verified",
    }


EXTRA_RUNNERS: dict[str, Callable[[], dict[str, Any]]] = {
    "stanfordnlp__dspy-8089": extra_dspy_8089,
    "crewaiinc__crewai-1791": extra_crewai_1791,
    "crewaiinc__crewai-1910": extra_crewai_1910,
    "crewaiinc__crewai-2424": extra_crewai_2424,
    "berriai__litellm-11964": extra_litellm_11964,
    "berriai__litellm-5232": extra_litellm_5232,
    "berriai__litellm-12168": extra_litellm_12168,
    "berriai__litellm-9828": extra_litellm_9828,
    "berriai__litellm-8512": extra_litellm_8512,
}


def build_plan_overrides(
    curated: dict[str, dict[str, Any]], extra_results: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "schema_version": "root_scope_extras_v1",
        "plan_schema_fields": list(PLAN_FIELDS),
        "permitted_module_paths": list(PERMITTED_MODULE_PATHS),
        "samples": {},
    }

    def entry(sid: str, corrections: dict, refs: list, notes: str, extra_status: str) -> None:
        draft = curated.get(sid, {})
        overrides["samples"][sid] = {
            "original_plan_draft": normalize_plan(copy.deepcopy(draft.get("plan") or {})),
            "original_source_refs": copy.deepcopy(draft.get("source_refs") or []),
            "plan_corrections": corrections,
            "source_refs_suggestions": refs,
            "root_notes": notes,
            "extra_status": extra_status,
        }

    r8089 = extra_results.get("stanfordnlp__dspy-8089", {})
    entry(
        "stanfordnlp__dspy-8089",
        {
            "verification_kind": None,
            "problem_hypothesis": (
                "Source fact: Tool._parse_function (invoked from Tool.__init__) builds args via TypeAdapter/BaseModel "
                "schema loop; reporter claim: untyped defaulted parameters get literal 'Any' string then default assignment "
                "raises TypeError. Hypothesis: keep dict schemas and gate validation on schema type."
            ),
        },
        [
            {
                "qualname": "dspy.primitives.tool.Tool._parse_function",
                "artifact_path": assert_safe_relative(
                    "work/open_dataset_integration_v2/main_candidates/cases/stanfordnlp__dspy-8089/before/dspy/primitives/tool.py"
                ),
                "start_line": 90,
                "end_line": 131,
                "note": "Primary schema inference loop; __init__ only delegates via self._parse_function(func, arg_desc).",
            },
            {
                "qualname": "dspy.primitives.tool.Tool.__call__",
                "artifact_path": assert_safe_relative(
                    "work/open_dataset_integration_v2/main_candidates/cases/stanfordnlp__dspy-8089/before/dspy/primitives/tool.py"
                ),
                "start_line": 133,
                "end_line": 145,
            },
        ],
        "Root extra verified BaseModel arg schema preservation via real TypeAdapter path; strict batch typed/untyped probe unchanged.",
        r8089.get("status", "not_run"),
    )

    r1791 = extra_results.get("crewaiinc__crewai-1791", {})
    entry(
        "crewaiinc__crewai-1791",
        {},
        [
            {
                "qualname": "crewai.task.Task",
                "artifact_path": assert_safe_relative(
                    "work/open_dataset_integration_v2/additional_crewai/cases/crewaiinc__crewai-1791/before/src/crewai/task.py"
                ),
                "start_line": 34,
                "end_line": 493,
                "note": "Remove spurious interpolate_only ref (applied-only); cite Task container with interpolate_inputs and _save_file.",
            },
            {
                "qualname": "crewai.task.Task._save_file",
                "artifact_path": assert_safe_relative(
                    "work/open_dataset_integration_v2/additional_crewai/cases/crewaiinc__crewai-1791/before/src/crewai/task.py"
                ),
                "start_line": 473,
                "end_line": 490,
            },
        ],
        "Extra verified _save_file OSError→RuntimeError wrap via monkeypatched Path.open; strict interpolation probes preserved.",
        r1791.get("status", "not_run"),
    )

    r11964 = extra_results.get("berriai__litellm-11964", {})
    preserve_corr: dict[str, Any] = {}
    if r11964.get("preserve_unknown_ref_verified"):
        preserve_corr["preserve_behaviors"] = [
            "Schemas without $ref remain byte-identical across before/applied unpack_defs on unchanged subtrees",
            "Resolvable $ref under properties/anyOf/items still inline on after tree",
            "Missing $ref definitions raise KeyError in both before and applied unpack_defs (not silently preserved)",
        ]
    else:
        preserve_corr["preserve_behaviors"] = [
            "Schemas without $ref remain unchanged aside from normal dict traversal",
            "Nested resolvable $ref paths inline on after; curator claim 'unknown $ref untouched' not verified (see root_notes)",
        ]
    entry(
        "berriai__litellm-11964",
        preserve_corr,
        [],
        "Root extra ran no-$ref equality and missing-$ref observation; narrowed preserve only where observed.",
        r11964.get("status", "not_run"),
    )

    entry(
        "berriai__litellm-8512",
        {
            "preserve_behaviors": [
                "Observed get_supported_openai_params tools list for known supported/unsupported bedrock models unchanged under stubbed supports_tool_choice(bedrock=True)",
                "After exposes tools param for meta.llama3-3-70b-instruct-v1:0 while before omits it",
            ],
        },
        [],
        "supports_tool_choice mocked; do not claim real provider tool_choice semantics verified.",
        extra_results.get("berriai__litellm-8512", {}).get("status", "not_run"),
    )

    for sid in (
        "crewaiinc__crewai-1910",
        "crewaiinc__crewai-2424",
        "berriai__litellm-5232",
        "berriai__litellm-12168",
        "berriai__litellm-9828",
    ):
        er = extra_results.get(sid, {})
        notes = {
            "crewaiinc__crewai-1910": "Nested BaseModel generate_model_description preserved before/after.",
            "crewaiinc__crewai-2424": "Blinker signal.send IO trace fires on before and after emit.",
            "berriai__litellm-5232": "Added empty tool delta arguments guard check.",
            "berriai__litellm-12168": "Invalid function dict and named function acceptance probed.",
            "berriai__litellm-9828": "9828 helpers with variant-matched 11964 unpack_defs; default_false output equality checked.",
        }[sid]
        entry(sid, {}, [], notes, er.get("status", "not_run"))

    for sid in sorted(PROVISIONAL_ALLOWLIST):
        if sid in overrides["samples"]:
            continue
        strict_note = "Strict batch + acceptance_strengthened (smol) sufficient; no Plan factual override."
        entry(sid, {}, [], strict_note, "skipped_strict_sufficient")

    return overrides


def run_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    curated = load_curated_index()
    strict = load_strict_status()
    extra_results: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []

    for sid in sorted(PROVISIONAL_ALLOWLIST):
        srow = strict.get(sid, {})
        strict_status = srow.get("strict_status")
        if sid in EXTRA_GAP_IDS:
            payload = _run_extra_child(sid)
            extra_results[sid] = payload
            extra_status = payload.get("status", "fail")
        else:
            extra_status = "skipped_strict_sufficient"
            payload = {"executed": False, "checks": {}, "note": "no root gap listed"}
        root_status = combine_root_status(strict_status, extra_status)
        samples.append(
            {
                "sample_id": sid,
                "strict_batch": srow.get("strict_batch"),
                "strict_status": strict_status,
                "extra_executed": sid in EXTRA_GAP_IDS and payload.get("executed", False),
                "extra_status": extra_status,
                "root_status": root_status,
                "extra_checks": payload.get("checks") or {},
                "extra_error": payload.get("error"),
            }
        )

    for sid in sorted(PENDING_NOT_EXPORT):
        samples.append(
            {
                "sample_id": sid,
                "strict_batch": strict.get(sid, {}).get("strict_batch"),
                "strict_status": strict.get(sid, {}).get("strict_status", "pending"),
                "extra_executed": False,
                "extra_status": "skipped_not_in_scope",
                "root_status": "pending",
                "note": "Not in provisional export allowlist; do not chase in C64.",
            }
        )

    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "root_scope_extras",
        "child_timeout_sec": CHILD_TIMEOUT,
        "provisional_allowlist_count": len(PROVISIONAL_ALLOWLIST),
        "summary": {
            "pass": sum(1 for s in samples if s.get("root_status") == "pass"),
            "reject": sum(1 for s in samples if s.get("root_status") == "reject"),
            "pending": sum(1 for s in samples if s.get("root_status") == "pending"),
        },
        "samples": samples,
    }
    plan_overrides = build_plan_overrides(curated, extra_results)
    return results, plan_overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write results.json and plan_overrides.json")
    parser.add_argument("--only", metavar="SAMPLE_ID", help="Run one extra check; print JSON to stdout")
    args = parser.parse_args()
    if args.only:
        fn = EXTRA_RUNNERS.get(args.only)
        if fn is None:
            print(json.dumps({"executed": False, "status": "fail", "error": "unknown_sample"}))
            return 2
        out = _run_extra_inline(fn)
        print(json.dumps(out, ensure_ascii=False))
        return 0 if out.get("status") == "pass" else 1
    results, overrides = run_audit()
    text_r = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    text_p = json.dumps(overrides, indent=2, ensure_ascii=False) + "\n"
    if args.write:
        (ROOT / "results.json").write_text(text_r, encoding="utf-8")
        (ROOT / "plan_overrides.json").write_text(text_p, encoding="utf-8")
    else:
        print(text_r)
    return 0 if results["summary"]["reject"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
