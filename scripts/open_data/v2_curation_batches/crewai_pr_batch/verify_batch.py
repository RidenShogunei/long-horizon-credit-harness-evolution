#!/usr/bin/env python3
"""Bounded before/after behavioral checks for crewai_pr_batch accepted rows."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

BATCH = Path(__file__).resolve().parent
WORK = BATCH.parent
CASES = WORK / "additional_crewai/cases"
FIXTURES = BATCH / "fixtures"
CHILD_TIMEOUT_SEC = 20

ACCEPTED = [
    "crewaiinc__crewai-1910",
    "crewaiinc__crewai-2424",
    "crewaiinc__crewai-2890",
    "crewaiinc__crewai-2865",
    "crewaiinc__crewai-1791",
    "crewaiinc__crewai-2604",
    "crewaiinc__crewai-2477",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def case_paths(iid: str, rel: str) -> tuple[Path, Path]:
    before = CASES / iid / "before" / rel
    after = CASES / iid / "applied" / rel
    return before, after


def ensure_pkg(name: str) -> types.ModuleType:
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        sub = ".".join(parts[:i])
        if sub not in sys.modules:
            mod = types.ModuleType(sub)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[sub] = mod
            if i > 1:
                parent = sys.modules[".".join(parts[: i - 1])]
                setattr(parent, parts[i - 1], mod)
    return sys.modules[name]


def load_module_from_path(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_child(script: Path, args: List[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SEC,
        cwd=str(FIXTURES),
        env=os.environ.copy(),
    )
    out = proc.stdout.strip()
    parsed: Any = None
    if out:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = out
    return {
        "returncode": proc.returncode,
        "stdout": out,
        "parsed": parsed,
        "stderr": proc.stderr.strip()[:500],
    }


def verify_1910() -> dict:
    rel = "src/crewai/utilities/converter.py"
    before_p, after_p = case_paths("crewaiinc__crewai-1910", rel)

    class Pep604Model(BaseModel):
        opt_pep604: int | None

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        ensure_pkg("crewai.agents.agent_builder.utilities.base_output_converter")

        class OutputConverter:
            pass

        sys.modules[
            "crewai.agents.agent_builder.utilities.base_output_converter"
        ].OutputConverter = OutputConverter
        pr = ensure_pkg("crewai.utilities.printer")

        class Printer:
            def print(self, *a, **k):
                pass

        pr.Printer = Printer
        psp = ensure_pkg("crewai.utilities.pydantic_schema_parser")

        class PydanticSchemaParser:
            pass

        psp.PydanticSchemaParser = PydanticSchemaParser
        mod = load_module_from_path("crewai.utilities.converter", path)
        err = None
        try:
            desc = mod.generate_model_description(Pep604Model)
        except Exception as exc:  # noqa: BLE001
            err = type(exc).__name__
            desc = ""
        return {
            "has_pep604_optional": '"opt_pep604": Optional[int]' in desc,
            "error": err,
            "snippet": desc[:240],
        }

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior_before = (
        before_obs["has_pep604_optional"] is False and before_obs.get("error") is not None
    )
    behavior_after = (
        after_obs.get("error") is None
        and "opt_pep604" in after_obs.get("snippet", "")
        and (
            "Optional[int]" in after_obs.get("snippet", "")
            or "int | None" in after_obs.get("snippet", "")
        )
    )
    preservation = after_obs.get("error") is None
    ok = behavior_before and behavior_after and preservation
    return {
        "source_id": "crewaiinc__crewai-1910",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_pep604_not_optional_in_description": behavior_before,
            "after_pep604_described_without_crash": behavior_after,
        },
    }


def verify_2424() -> dict:
    rel = "src/crewai/utilities/events/crewai_event_bus.py"
    before_p, after_p = case_paths("crewaiinc__crewai-2424", rel)

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        base = ensure_pkg("crewai.utilities.events.base_events")

        class CrewEvent(BaseModel):
            pass

        base.CrewEvent = CrewEvent
        et = ensure_pkg("crewai.utilities.events.event_types")
        et.EventTypes = CrewEvent
        mod = load_module_from_path("crewai.utilities.events.crewai_event_bus", path)
        bus = mod.CrewAIEventsBus()
        if hasattr(bus, "clear_handlers"):
            bus.clear_handlers()
        else:
            bus._handlers.clear()

        class BaseToolEvent(CrewEvent):
            pass

        class ToolFinishedEvent(BaseToolEvent):
            tool: str = "t"

        hits: list[str] = []

        @bus.on(BaseToolEvent)
        def on_base(_src, _ev):  # noqa: ANN001
            hits.append("base")

        @bus.on(ToolFinishedEvent)
        def on_concrete(_src, _ev):  # noqa: ANN001
            hits.append("concrete")

        bus.emit(None, ToolFinishedEvent())
        return {"hits": hits}

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior_before = before_obs["hits"] == ["concrete"]
    behavior_after = after_obs["hits"].count("base") == 1 and after_obs["hits"].count(
        "concrete"
    ) == 1
    preservation = "concrete" in before_obs["hits"] and "concrete" in after_obs["hits"]
    ok = behavior_before and behavior_after and preservation
    return {
        "source_id": "crewaiinc__crewai-2424",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_exact_type_handler_only": behavior_before,
            "after_base_handler_also_fires": behavior_after,
            "concrete_handler_preserved": preservation,
        },
    }


def verify_2890() -> dict:
    rel = "src/crewai/project/crew_base.py"
    before_p, after_p = case_paths("crewaiinc__crewai-2890", rel)

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        mod = load_module_from_path("crewai.project.crew_base", path)
        CrewBase = mod.CrewBase

        @CrewBase
        class ResearchCrew:
            agents = []

        return {
            "has_crew_name": hasattr(ResearchCrew, "_crew_name"),
            "crew_name": getattr(ResearchCrew, "_crew_name", None),
        }

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior_before = before_obs["has_crew_name"] is False
    behavior_after = after_obs["crew_name"] == "ResearchCrew"
    ok = behavior_before and behavior_after
    return {
        "source_id": "crewaiinc__crewai-2890",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_missing_crew_name": behavior_before,
            "after_crew_name_matches_original_class": behavior_after,
        },
    }


def verify_2865() -> dict:
    rel = "src/crewai/task.py"
    before_p, after_p = case_paths("crewaiinc__crewai-2865", rel)
    script = FIXTURES / "probe_task2865_markdown.py"
    before_obs = run_child(script, [str(before_p), "true"])
    after_obs = run_child(script, [str(after_p), "true"])
    before_plain = run_child(script, [str(before_p), "false"])
    after_plain = run_child(script, [str(after_p), "false"])

    def parsed(child: dict) -> dict:
        p = child.get("parsed")
        return p if isinstance(p, dict) else {}

    b_md = parsed(before_obs)
    a_md = parsed(after_obs)
    behavior_before = b_md.get("has_markdown_instruction") is False
    behavior_after = a_md.get("has_markdown_instruction") is True
    preservation = (
        parsed(before_plain).get("has_markdown_instruction") is False
        and parsed(after_plain).get("has_markdown_instruction") is False
        and parsed(before_plain).get("len", 0) > 0
    )
    ok = (
        behavior_before
        and behavior_after
        and preservation
        and before_obs["returncode"] == 0
        and after_obs["returncode"] == 0
    )
    return {
        "source_id": "crewaiinc__crewai-2865",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
            "fixture_script_sha256": sha256_file(script),
        },
        "before_observed": {"markdown_true": b_md, "markdown_false": parsed(before_plain)},
        "after_observed": {"markdown_true": a_md, "markdown_false": parsed(after_plain)},
        "checks": {
            "before_no_markdown_block": behavior_before,
            "after_includes_markdown_block": behavior_after,
            "markdown_false_unchanged_both": preservation,
        },
    }


def verify_1791() -> dict:
    rel = "src/crewai/task.py"
    before_p, after_p = case_paths("crewaiinc__crewai-1791", rel)
    script = FIXTURES / "probe_task1791_interpolate.py"
    before_json = run_child(script, [str(before_p), "json_braces"])
    after_json = run_child(script, [str(after_p), "json_braces"])
    before_simple = run_child(script, [str(before_p), "simple"])
    after_simple = run_child(script, [str(after_p), "simple"])

    def parsed(child: dict) -> dict:
        p = child.get("parsed")
        return p if isinstance(p, dict) else {}

    bj = parsed(before_json)
    aj = parsed(after_json)
    behavior_before = bj.get("error") is not None
    behavior_after = aj.get("error") is None and "Ada" in str(aj.get("expected_output"))
    preservation = (
        parsed(before_simple).get("expected_output") == "Hello Bob"
        and parsed(after_simple).get("expected_output") == "Hello Bob"
    )
    ok = behavior_before and behavior_after and preservation
    return {
        "source_id": "crewaiinc__crewai-1791",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
            "fixture_script_sha256": sha256_file(script),
        },
        "before_observed": {"json_braces": bj, "simple": parsed(before_simple)},
        "after_observed": {"json_braces": aj, "simple": parsed(after_simple)},
        "checks": {
            "before_format_breaks_on_json_braces": behavior_before,
            "after_interpolate_preserves_json_and_user": behavior_after,
            "simple_placeholder_still_works": preservation,
        },
    }


def verify_2604() -> dict:
    rel = "src/crewai/memory/storage/mem0_storage.py"
    before_p, after_p = case_paths("crewaiinc__crewai-2604", rel)

    class Memory:
        def search(self, **params):
            return {"results": [{"score": 0.9, "id": "a"}, {"score": 0.1, "id": "b"}]}

        def add(self, value, **params):
            return None

    class MemoryClient:
        def search(self, **params):
            return {"results": [{"score": 0.9, "id": "a"}]}

        def add(self, value, **params):
            return None

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai.") or name == "mem0":
                del sys.modules[name]
        mem0 = types.ModuleType("mem0")
        mem0.Memory = Memory
        mem0.MemoryClient = MemoryClient
        sys.modules["mem0"] = mem0
        iface = ensure_pkg("crewai.memory.storage.interface")

        class Storage:
            pass

        iface.Storage = Storage
        mod = load_module_from_path("crewai.memory.storage.mem0_storage", path)
        store = mod.Mem0Storage(type="short_term", crew=None, config={"user_id": "u1"})
        store.memory = Memory()
        return store.search("query", limit=5, score_threshold=0.35)

    def safe_run(path: Path) -> dict:
        try:
            results = run(path)
            return {"count": len(results), "ids": [r.get("id") for r in results]}
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__}

    before_obs = safe_run(before_p)
    after_obs = safe_run(after_p)
    behavior_before = "error" in before_obs or before_obs.get("count", 0) == 0
    behavior_after = after_obs.get("count") == 1 and after_obs.get("ids") == ["a"]
    ok = behavior_before and behavior_after
    return {
        "source_id": "crewaiinc__crewai-2604",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_search_on_v11_dict_shape_fails_or_empty": behavior_before,
            "after_filters_results_list": behavior_after,
        },
    }


def verify_2477() -> dict:
    rel_ev = "src/crewai/utilities/events/tool_usage_events.py"
    before_ev, after_ev = case_paths("crewaiinc__crewai-2477", rel_ev)

    def load_events(path: Path):
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        base = ensure_pkg("crewai.utilities.events.base_events")

        class CrewEvent(BaseModel):
            pass

        base.CrewEvent = CrewEvent
        return load_module_from_path("crewai.utilities.events.tool_usage_events", path)

    def probe(path: Path) -> dict:
        mod = load_events(path)
        Finished = mod.ToolUsageFinishedEvent
        Started = mod.ToolUsageStartedEvent
        now = datetime.now()
        base_fields = {
            "agent_key": "k",
            "agent_role": "r",
            "tool_name": "search",
            "tool_args": {},
            "tool_class": "T",
            "started_at": now,
            "finished_at": now,
        }
        out: dict[str, Any] = {"missing_output_error": None, "started_ok": False}
        try:
            Finished(**base_fields)
        except Exception as exc:  # noqa: BLE001
            out["missing_output_error"] = type(exc).__name__
        try:
            Started(
                agent_key="k",
                agent_role="r",
                tool_name="search",
                tool_args={},
                tool_class="T",
            )
            out["started_ok"] = True
        except Exception:
            out["started_ok"] = False
        try:
            Finished(**base_fields, output="done")
            out["with_output_ok"] = True
        except Exception:
            out["with_output_ok"] = False
        return out

    before_obs = probe(before_ev)
    after_obs = probe(after_ev)
    behavior_before = before_obs.get("missing_output_error") is None
    behavior_after = (
        after_obs.get("missing_output_error") is not None
        and after_obs.get("with_output_ok") is True
    )
    preservation = before_obs.get("started_ok") and after_obs.get("started_ok")
    ok = behavior_before and behavior_after and preservation
    return {
        "source_id": "crewaiinc__crewai-2477",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_events_sha256": sha256_file(before_ev),
            "after_events_sha256": sha256_file(after_ev),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_finished_event_without_output_valid": behavior_before,
            "after_output_required_and_populated": behavior_after,
            "started_event_still_valid": preservation,
        },
    }


VERIFY_FNS: Dict[str, Callable[[], dict]] = {
    "crewaiinc__crewai-1910": verify_1910,
    "crewaiinc__crewai-2424": verify_2424,
    "crewaiinc__crewai-2890": verify_2890,
    "crewaiinc__crewai-2865": verify_2865,
    "crewaiinc__crewai-1791": verify_1791,
    "crewaiinc__crewai-2604": verify_2604,
    "crewaiinc__crewai-2477": verify_2477,
}


def main() -> None:
    results: List[dict] = []
    for iid in ACCEPTED:
        fn = VERIFY_FNS[iid]
        try:
            results.append(fn())
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "source_id": iid,
                    "executed": False,
                    "status": "reject",
                    "error": "child_timeout",
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "source_id": iid,
                    "executed": False,
                    "status": "reject",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "batch": "crewai_pr_batch",
        "cases_root": "work/open_dataset_integration_v2/additional_crewai/cases",
        "dataset_compilation_license": "CC-BY-4.0",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "child_timeout_sec": CHILD_TIMEOUT_SEC,
        "results": results,
        "summary": {
            "pass": sum(1 for r in results if r.get("status") == "pass"),
            "reject": sum(1 for r in results if r.get("status") == "reject"),
            "executed": sum(1 for r in results if r.get("executed")),
        },
    }
    out_path = BATCH / "verification_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))

    curated_path = BATCH / "curated_examples.json"
    if curated_path.exists():
        rows = json.loads(curated_path.read_text(encoding="utf-8"))
        status_by_id = {r["source_id"]: r["status"] for r in results}
        for row in rows:
            st = status_by_id.get(row["source_id"])
            if st == "pass":
                row["verification_status"] = "bounded_local_execution_pass"
            elif st == "reject":
                row["verification_status"] = "bounded_local_execution_reject"
            else:
                row["verification_status"] = "bounded_local_execution_pending"
        curated_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
