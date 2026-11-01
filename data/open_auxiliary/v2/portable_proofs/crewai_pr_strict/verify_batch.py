#!/usr/bin/env python3
"""Strict before/after behavioral + preservation checks for crewai_pr_batch accepted rows."""
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
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

STRICT = Path(__file__).resolve().parent
WORK = STRICT.parent
CASES = WORK / "additional_crewai/cases"
BATCH = WORK / "crewai_pr_batch"
FIXTURES = STRICT / "fixtures"
CHILD_TIMEOUT_SEC = 20

VALID_MODULES = frozenset(
    {"tools", "agent_loop", "context_mgmt", "observation", "verification"}
)

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
    return (
        CASES / iid / "before" / rel,
        CASES / iid / "applied" / rel,
    )


def sample_status(
    behavior: Dict[str, bool], preservation: Dict[str, bool], pending: List[str]
) -> str:
    if pending:
        return "pending"
    if not all(behavior.values()):
        return "reject"
    if not all(preservation.values()):
        return "reject"
    return "pass"


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
    sid = "crewaiinc__crewai-1910"
    before_p, after_p = case_paths(sid, "src/crewai/utilities/converter.py")

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

        class ClassicModel(BaseModel):
            opt: Optional[int]
            items: List[str]
            mix: Union[str, int]

        class Pep604Model(BaseModel):
            opt_pep604: int | None

        for probe_model in (ClassicModel, Pep604Model):
            probe_model.__annotations__ = {
                name: field.annotation for name, field in probe_model.model_fields.items()
            }

        out: dict[str, Any] = {"classic": "", "pep604_error": None, "pep604_snippet": ""}
        try:
            out["classic"] = mod.generate_model_description(ClassicModel)
        except Exception as exc:  # noqa: BLE001
            out["classic_error"] = type(exc).__name__
        try:
            out["pep604_snippet"] = mod.generate_model_description(Pep604Model)
        except Exception as exc:  # noqa: BLE001
            out["pep604_error"] = type(exc).__name__
        return out

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior = {
        "before_pep604_raises": before_obs.get("pep604_error") is not None,
        "after_pep604_described": after_obs.get("pep604_error") is None
        and "opt_pep604" in after_obs.get("pep604_snippet", ""),
    }
    preservation = {
        "classic_optional_list_in_both": '"opt": Optional[int]' in before_obs.get("classic", "")
        and '"items": List[str]' in before_obs.get("classic", "")
        and '"opt": Optional[int]' in after_obs.get("classic", "")
        and '"items": List[str]' in after_obs.get("classic", ""),
        "classic_still_valid_before": before_obs.get("classic_error") is None,
        "before_union_and_after_union_detail": "Union" in before_obs.get("classic", "")
        and "Union" in after_obs.get("classic", ""),
    }
    pending: List[str] = []
    status = sample_status(behavior, preservation, pending)
    return _result(sid, status, behavior, preservation, pending, before_p, after_p, before_obs, after_obs)


def verify_2424() -> dict:
    sid = "crewaiinc__crewai-2424"
    before_p, after_p = case_paths(sid, "src/crewai/utilities/events/crewai_event_bus.py")

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

        class UnrelatedEvent(CrewEvent):
            tag: str = "u"

        hits: list[str] = []

        @bus.on(BaseToolEvent)
        def on_base(_src, _ev):  # noqa: ANN001
            hits.append("base")

        @bus.on(ToolFinishedEvent)
        def on_concrete(_src, _ev):  # noqa: ANN001
            hits.append("concrete")

        handler_count = len(getattr(bus, "_handlers", {}))
        bus.emit(None, ToolFinishedEvent())
        unrelated_hits: list[str] = []

        @bus.on(UnrelatedEvent)
        def on_unrelated(_src, _ev):  # noqa: ANN001
            unrelated_hits.append("unrelated")

        bus.emit(None, UnrelatedEvent())
        return {
            "hits": hits,
            "handler_count": handler_count,
            "unrelated_hits": unrelated_hits,
        }

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior = {
        "before_exact_type_only": before_obs["hits"] == ["concrete"],
        "after_base_and_concrete": after_obs["hits"].count("base") == 1
        and after_obs["hits"].count("concrete") == 1,
    }
    preservation = {
        "concrete_still_fires_both": "concrete" in before_obs["hits"]
        and "concrete" in after_obs["hits"],
        "unrelated_single_handler_no_dupes": after_obs["unrelated_hits"] == ["unrelated"]
        and after_obs["handler_count"] == 2,
    }
    status = sample_status(behavior, preservation, [])
    return _result(sid, status, behavior, preservation, [], before_p, after_p, before_obs, after_obs)


def verify_2890() -> dict:
    sid = "crewaiinc__crewai-2890"
    before_p, after_p = case_paths(sid, "src/crewai/project/crew_base.py")

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        yaml_mod = types.ModuleType("yaml")
        yaml_mod.safe_load = lambda f: {}
        sys.modules["yaml"] = yaml_mod
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda: None
        sys.modules["dotenv"] = dotenv
        mod = load_module_from_path("crewai.project.crew_base", path)
        CrewBase = mod.CrewBase

        @CrewBase
        class ResearchCrew:
            agents = []

        inst = ResearchCrew()
        return {
            "has_crew_name": hasattr(ResearchCrew, "_crew_name"),
            "crew_name": getattr(ResearchCrew, "_crew_name", None),
            "wrapped_name": ResearchCrew.__name__,
            "is_crew_class": getattr(ResearchCrew, "is_crew_class", False),
            "agents_config_loaded": isinstance(getattr(inst, "agents_config", None), dict),
            "tasks_config_loaded": isinstance(getattr(inst, "tasks_config", None), dict),
        }

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior = {
        "before_no_crew_name": before_obs["has_crew_name"] is False,
        "after_crew_name_original_class": after_obs.get("crew_name") == "ResearchCrew",
    }
    preservation = {
        "decoration_instantiation_both": before_obs["is_crew_class"]
        and after_obs["is_crew_class"]
        and before_obs["agents_config_loaded"]
        and after_obs["agents_config_loaded"]
        and before_obs["tasks_config_loaded"]
        and after_obs["tasks_config_loaded"],
        "wrapped_name_contains_research": "ResearchCrew" in after_obs.get("wrapped_name", ""),
    }
    status = sample_status(behavior, preservation, [])
    return _result(sid, status, behavior, preservation, [], before_p, after_p, before_obs, after_obs)


def verify_2865() -> dict:
    sid = "crewaiinc__crewai-2865"
    before_p, after_p = case_paths(sid, "src/crewai/task.py")
    script = FIXTURES / "probe_task2865_markdown.py"
    before_md = run_child(script, [str(before_p), "true"])
    after_md = run_child(script, [str(after_p), "true"])
    before_plain = run_child(script, [str(before_p), "false"])
    after_plain = run_child(script, [str(after_p), "false"])

    def parsed(child: dict) -> dict:
        p = child.get("parsed")
        return p if isinstance(p, dict) else {}

    b_md, a_md = parsed(before_md), parsed(after_md)
    b_pl, a_pl = parsed(before_plain), parsed(after_plain)
    behavior = {
        "before_no_markdown_block": b_md.get("has_markdown_instruction") is False,
        "after_markdown_block": a_md.get("has_markdown_instruction") is True,
    }
    preservation = {
        "markdown_false_prompt_identical_before_after": b_pl.get("prompt") == a_pl.get("prompt")
        and b_pl.get("prompt") is not None
        and b_pl.get("has_markdown_instruction") is False,
    }
    pending: List[str] = []
    if before_plain["returncode"] != 0 or after_plain["returncode"] != 0:
        pending.append("markdown_false_child_failed")
    status = sample_status(behavior, preservation, pending)
    obs = {
        "before": {"markdown_true": b_md, "markdown_false": b_pl},
        "after": {"markdown_true": a_md, "markdown_false": a_pl},
    }
    return _result(
        sid,
        status,
        behavior,
        preservation,
        pending,
        before_p,
        after_p,
        obs["before"],
        obs["after"],
        extra_hashes={"fixture_script_sha256": sha256_file(script)},
    )


def verify_1791() -> dict:
    sid = "crewaiinc__crewai-1791"
    before_p, after_p = case_paths(sid, "src/crewai/task.py")
    script = FIXTURES / "probe_task1791_interpolate.py"
    modes = ["json_braces", "simple", "missing_key"]
    before: dict[str, dict] = {}
    after: dict[str, dict] = {}

    def parsed(child: dict) -> dict:
        p = child.get("parsed")
        return p if isinstance(p, dict) else {}

    for mode in modes:
        before[mode] = parsed(run_child(script, [str(before_p), mode]))
        after[mode] = parsed(run_child(script, [str(after_p), mode]))

    behavior = {
        "before_json_braces_error": before["json_braces"].get("error") is not None,
        "after_json_braces_ok": after["json_braces"].get("error") is None
        and "Ada" in str(after["json_braces"].get("expected_output")),
    }
    preservation = {
        "simple_placeholder_unchanged": before["simple"].get("expected_output")
        == after["simple"].get("expected_output")
        == "Hello Bob",
        "missing_key_still_errors_both": before["missing_key"].get("error") is not None
        and after["missing_key"].get("error") is not None,
    }
    status = sample_status(behavior, preservation, [])
    return _result(
        sid,
        status,
        behavior,
        preservation,
        [],
        before_p,
        after_p,
        before,
        after,
        extra_hashes={"fixture_script_sha256": sha256_file(script)},
    )


def verify_2604() -> dict:
    sid = "crewaiinc__crewai-2604"
    before_p, after_p = case_paths(sid, "src/crewai/memory/storage/mem0_storage.py")

    class VersionedMemoryBackend:
        """Returns v1 list or v1.1 dict based on requested output_format."""

        last_search_params: dict | None = None

        def __init__(self, *args, **kwargs):
            self.last_params: dict | None = None
            pass

        def search(self, **params):
            self.last_params = dict(params)
            VersionedMemoryBackend.last_search_params = self.last_params
            fmt = params.get("output_format")
            rows = [
                {"score": 0.9, "id": "a"},
                {"score": 0.1, "id": "b"},
                {"score": 0.2, "id": "c"},
            ]
            if fmt == "v1.1":
                return {"results": rows[: int(params.get("limit", 3))]}
            return rows[: int(params.get("limit", 3))]

        def add(self, value, **params):
            return None

    def run(path: Path, *, force_dict_bug: bool = False) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai.") or name == "mem0":
                del sys.modules[name]

        class Memory(VersionedMemoryBackend):
            @classmethod
            def from_config(cls, _config):
                return cls()

        class MemoryClient(VersionedMemoryBackend):
            def __init__(self, api_key=None, org_id=None, project_id=None, **kwargs):
                super().__init__()

        class BugMemory(VersionedMemoryBackend):
            @classmethod
            def from_config(cls, _config):
                return cls()

            def search(self, **params):
                VersionedMemoryBackend.last_search_params = dict(params)
                return {"results": [{"score": 0.9, "id": "a"}, {"score": 0.1, "id": "b"}]}

        if force_dict_bug:
            mem0 = types.ModuleType("mem0")
            mem0.Memory = BugMemory
            mem0.MemoryClient = MemoryClient
        else:
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
        mem_instance = BugMemory() if force_dict_bug else Memory()
        store.memory = mem_instance
        try:
            results = store.search("query", limit=2, score_threshold=0.35)
            params = VersionedMemoryBackend.last_search_params or {}
            return {
                "count": len(results),
                "ids": [r.get("id") for r in results],
                "params": params,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__, "params": VersionedMemoryBackend.last_search_params}

    before_v1 = run(before_p, force_dict_bug=False)
    before_bug = run(before_p, force_dict_bug=True)

    def run_after(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai.") or name == "mem0":
                del sys.modules[name]

        class Memory(VersionedMemoryBackend):
            @classmethod
            def from_config(cls, _config):
                return cls()

            def search(self, **params):
                VersionedMemoryBackend.last_search_params = dict(params)
                rows = [
                    {"score": 0.9, "id": "a"},
                    {"score": 0.1, "id": "b"},
                    {"score": 0.2, "id": "c"},
                ]
                lim = int(params.get("limit", 3))
                return {"results": rows[:lim]}

        class MemoryClient(VersionedMemoryBackend):
            def __init__(self, api_key=None, org_id=None, project_id=None, **kwargs):
                super().__init__()

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
        try:
            results = store.search("query", limit=2, score_threshold=0.35)
            params = VersionedMemoryBackend.last_search_params or {}
            return {
                "count": len(results),
                "ids": [r.get("id") for r in results],
                "params": params,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__, "params": VersionedMemoryBackend.last_search_params}

    after_obs = run_after(after_p)
    after_client_p, _ = case_paths(sid, "src/crewai/memory/storage/mem0_storage.py")

    def run_client(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai.") or name == "mem0":
                del sys.modules[name]
        mem0 = types.ModuleType("mem0")

        class Mem(VersionedMemoryBackend):
            @classmethod
            def from_config(cls, _config):
                return cls()

        class ClientMemory(VersionedMemoryBackend):
            def __init__(self, api_key=None, org_id=None, project_id=None, **kwargs):
                super().__init__()

        mem0.Memory = Mem
        mem0.MemoryClient = ClientMemory
        sys.modules["mem0"] = mem0
        iface = ensure_pkg("crewai.memory.storage.interface")

        class Storage:
            pass

        iface.Storage = Storage
        mod = load_module_from_path("crewai.memory.storage.mem0_storage", path)
        os.environ["MEM0_API_KEY"] = "test-key"
        store = mod.Mem0Storage(type="short_term", crew=None, config={"user_id": "u1", "api_key": "k"})
        try:
            results = store.search("query", limit=1, score_threshold=0.35)
            params = getattr(store.memory, "last_params", None) or {}
            return {
                "count": len(results),
                "ids": [r.get("id") for r in results],
                "params": params,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__}

    after_client = run_client(after_client_p)

    behavior = {
        "before_dict_response_breaks": before_bug.get("error") is not None,
        "after_v11_dict_filters": after_obs.get("error") is None
        and after_obs.get("ids") == ["a"]
        and after_obs.get("count") == 1,
        "after_client_requests_v11": after_client.get("error") is None
        and after_client.get("count") == 1
        and after_client.get("params", {}).get("metadata") is not None,
    }
    preservation = {
        "before_v1_list_ids_and_limit": before_v1.get("error") is None
        and before_v1.get("ids") == ["a"]
        and before_v1.get("count") == 1
        and before_v1.get("params", {}).get("limit") == 2,
        "after_local_memory_dict_parse_path": after_obs.get("error") is None
        and after_obs.get("params", {}).get("output_format") is None,
    }
    pending: List[str] = []
    status = sample_status(behavior, preservation, pending)
    return _result(
        sid,
        status,
        behavior,
        preservation,
        pending,
        before_p,
        after_p,
        {"before_v1": before_v1, "before_dict_bug": before_bug},
        {"after_memory": after_obs, "after_client": after_client},
    )


def verify_2477() -> dict:
    sid = "crewaiinc__crewai-2477"
    rel_tu = "src/crewai/tools/tool_usage.py"
    rel_ev = "src/crewai/utilities/events/tool_usage_events.py"
    before_tu, after_tu = case_paths(sid, rel_tu)
    before_ev, after_ev = case_paths(sid, rel_ev)
    script = FIXTURES / "probe_tool_usage2477.py"

    def run_pair(tu: Path, ev: Path) -> dict:
        if not ev.is_file():
            return {"error": "missing_events_module", "path": str(ev)}
        child = run_child(script, [str(tu), str(ev)])
        p = child.get("parsed")
        return p if isinstance(p, dict) else {"parse_error": child.get("stderr")}

    before_obs = run_pair(before_tu, before_ev)
    after_obs = run_pair(after_tu, after_ev)

    behavior = {
        "before_finished_without_output_field": before_obs.get("error") is None
        and before_obs.get("output") is None
        and before_obs.get("finished_count", 0) == 1,
        "after_output_from_tool_result": after_obs.get("error") is None
        and after_obs.get("output") == {"answer": "42"},
    }
    preservation = {
        "finished_event_still_emitted_both": before_obs.get("finished_count") == 1
        and after_obs.get("finished_count") == 1,
    }
    pending: List[str] = []
    if before_obs.get("parse_error") or after_obs.get("parse_error"):
        pending.append("tool_usage_probe_failed")
    status = sample_status(behavior, preservation, pending)
    return _result(
        sid,
        status,
        behavior,
        preservation,
        pending,
        before_tu,
        after_tu,
        before_obs,
        after_obs,
        extra_hashes={
            "before_events_sha256": sha256_file(before_ev),
            "after_events_sha256": sha256_file(after_ev),
            "fixture_script_sha256": sha256_file(script),
        },
    )


def _result(
    sid: str,
    status: str,
    behavior: Dict[str, bool],
    preservation: Dict[str, bool],
    pending: List[str],
    before_p: Path,
    after_p: Path,
    before_obs: Any,
    after_obs: Any,
    extra_hashes: dict | None = None,
) -> dict:
    hashes = {
        "before_sha256": sha256_file(before_p),
        "after_sha256": sha256_file(after_p),
    }
    if extra_hashes:
        hashes.update(extra_hashes)
    return {
        "source_id": sid,
        "executed": True,
        "status": status,
        "behavior": behavior,
        "preservation": preservation,
        "pending": pending,
        "source_hashes": hashes,
        "before_observed": before_obs,
        "after_observed": after_obs,
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


def sync_curated(rows: list, results: list[dict]) -> None:
    status_by_id = {r["source_id"]: r["status"] for r in results}
    module_fix = {
        "crewaiinc__crewai-1910": ("context_mgmt", "context_mgmt/converter.py"),
        "crewaiinc__crewai-2424": ("observation", "observation/crewai_event_bus.py"),
        "crewaiinc__crewai-2477": ("observation", "observation/tool_usage.py"),
        "crewaiinc__crewai-2604": ("context_mgmt", "context_mgmt/mem0_storage.py"),
        "crewaiinc__crewai-2890": ("agent_loop", "agent_loop/crew_base.py"),
    }
    for row in rows:
        st = status_by_id.get(row["source_id"], "pending")
        if st == "pass":
            row["verification_status"] = "strict_local_execution_pass"
        elif st == "reject":
            row["verification_status"] = "strict_local_execution_reject"
        else:
            row["verification_status"] = "strict_local_execution_pending"
        if row["source_id"] in module_fix:
            harness, visible = module_fix[row["source_id"]]
            row.setdefault("source_mapping", {})["harness_module"] = harness
            row["source_mapping"]["visible_path"] = visible
        state = row.get("state") or {}
        state["permitted_modules"] = sorted(VALID_MODULES)
        plan = row.get("plan") or {}
        hm = row.get("source_mapping", {}).get("harness_module")
        if hm in VALID_MODULES:
            plan["target_module"] = hm
        row["plan"] = plan
        row["state"] = state


def main() -> None:
    results: List[dict] = []
    for iid in ACCEPTED:
        try:
            results.append(VERIFY_FNS[iid]())
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
        "batch": "crewai_pr_strict",
        "preserved_batch": "crewai_pr_batch",
        "cases_root": "work/open_dataset_integration_v2/additional_crewai/cases",
        "dataset_compilation_license": "CC-BY-4.0",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "child_timeout_sec": CHILD_TIMEOUT_SEC,
        "valid_permitted_modules": sorted(VALID_MODULES),
        "results": results,
        "summary": {
            "pass": sum(1 for r in results if r.get("status") == "pass"),
            "reject": sum(1 for r in results if r.get("status") == "reject"),
            "pending": sum(1 for r in results if r.get("status") == "pending"),
            "executed": sum(1 for r in results if r.get("executed")),
        },
    }
    out_path = STRICT / "verification_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))

    curated_path = STRICT / "curated_examples.json"
    preserved = STRICT / "curated_examples_crewai_pr_batch_preserved.json"
    if preserved.exists():
        rows = json.loads(preserved.read_text(encoding="utf-8"))
        sync_curated(rows, results)
        curated_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
