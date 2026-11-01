#!/usr/bin/env python3
"""Bounded before/after behavioral checks for crewai_batch accepted rows."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

BATCH = Path(__file__).resolve().parent
WORK = BATCH.parent
CASES = WORK / "main_candidates/cases"
FIXTURES = BATCH / "fixtures"
CHILD_TIMEOUT_SEC = 20


async def _async_probe_coro():
    return {"rows": [1, 2]}


ACCEPTED = [
    "crewaiinc__crewai-2570",
    "crewaiinc__crewai-3150",
    "crewaiinc__crewai-2588",
    "crewaiinc__crewai-1818",
    "crewaiinc__crewai-2408",
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


def load_base_tool_module(base_tool_path: Path):
    for name in list(sys.modules):
        if name == "crewai" or name.startswith("crewai."):
            del sys.modules[name]
    ensure_pkg("crewai")
    ensure_pkg("crewai.tools")
    st = ensure_pkg("crewai.tools.structured_tool")

    class CrewStructuredTool:  # noqa: N801
        pass

    st.CrewStructuredTool = CrewStructuredTool
    spec = importlib.util.spec_from_file_location("crewai.tools.base_tool", base_tool_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crewai.tools.base_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


def verify_2570() -> dict:
    before_p, after_p = case_paths("crewaiinc__crewai-2570", "src/crewai/tools/base_tool.py")

    def run_probe(path: Path) -> dict:
        mod = load_base_tool_module(path)
        BaseTool = mod.BaseTool

        class AsyncTool(BaseTool):
            name: str = "async_probe"
            description: str = "probe"

            def _run(self, *args, **kwargs):
                return _async_probe_coro()

        class SyncTool(BaseTool):
            name: str = "sync_probe"
            description: str = "probe"

            def _run(self, *args, **kwargs):
                return 7

        raw_async = AsyncTool().run()
        if asyncio.iscoroutine(raw_async):
            async_is_coro = True
            async_value = None
            raw_async.close()
        else:
            async_is_coro = False
            async_value = raw_async
        sync_out = SyncTool().run()
        return {
            "async_is_coroutine": async_is_coro,
            "async_value": async_value,
            "sync_value": sync_out,
        }

    before_obs = run_probe(before_p)
    after_obs = run_probe(after_p)
    behavior_before = before_obs["async_is_coroutine"] is True
    behavior_after = (
        before_obs["async_is_coroutine"] is True
        and after_obs["async_is_coroutine"] is False
        and after_obs["async_value"] == {"rows": [1, 2]}
    )
    preservation = before_obs["sync_value"] == 7 and after_obs["sync_value"] == 7
    ok = behavior_before and behavior_after and preservation
    return {
        "source_id": "crewaiinc__crewai-2570",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_async_returns_coroutine": behavior_before,
            "after_async_resolves_dict": behavior_after,
            "sync_tool_still_returns_7": preservation,
        },
    }


def verify_2588() -> dict:
    before_p, after_p = case_paths(
        "crewaiinc__crewai-2588", "src/crewai/memory/storage/mem0_storage.py"
    )
    captured: dict[str, Any] = {"from_config_arg": None, "client_used": False}

    class Memory:
        def __init__(self):
            pass

        @classmethod
        def from_config(cls, cfg):
            captured["from_config_arg"] = cfg
            return cls()

    class MemoryClient:
        def __init__(self, **kwargs):
            captured["client_used"] = True

    def run(path: Path) -> dict:
        captured["from_config_arg"] = None
        captured["client_used"] = False
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
        spec = importlib.util.spec_from_file_location(
            "crewai.memory.storage.mem0_storage", path
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        local_cfg = {"vector_store": {"provider": "qdrant", "config": {"host": "localhost"}}}
        mod.Mem0Storage(
            type="short_term",
            crew=None,
            config={"local_mem0_config": local_cfg, "user_id": "u1"},
        )
        return {
            "from_config_arg": captured["from_config_arg"],
            "client_used": captured["client_used"],
        }

    before_obs = run(before_p)
    after_obs = run(after_p)
    local_cfg = {"vector_store": {"provider": "qdrant", "config": {"host": "localhost"}}}
    behavior_before = before_obs["from_config_arg"] == {
        "local_mem0_config": local_cfg,
        "user_id": "u1",
    } and before_obs["client_used"] is False
    behavior_after = after_obs["from_config_arg"] == local_cfg and after_obs["client_used"] is False
    ok = behavior_before and behavior_after
    return {
        "source_id": "crewaiinc__crewai-2588",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_passes_full_config_dict": behavior_before,
            "after_passes_local_mem0_config_only": behavior_after,
        },
    }


def verify_2408() -> dict:
    before_p, after_p = case_paths("crewaiinc__crewai-2408", "src/crewai/utilities/prompts.py")

    from pydantic import BaseModel, ConfigDict

    class AgentStub(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        goal: str = "g"
        role: str = "r"
        backstory: str = "b"

    class I18N(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        def slice(self, key: str) -> str:
            return f"<{key}>"

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        util = ensure_pkg("crewai.utilities")
        util.I18N = I18N
        spec = importlib.util.spec_from_file_location("crewai.utilities.prompts", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        prompts = mod.Prompts(agent=AgentStub(), system_template="SYS {{ .System }}", prompt_template=None)
        out: dict = {"error": None, "prompt_len": 0}
        try:
            text = prompts._build_prompt(
                ["role_playing", "no_tools", "task"],
                prompts.system_template,
                prompts.prompt_template,
                None,
            )
            out["prompt_len"] = len(text)
        except Exception as exc:  # noqa: BLE001
            out["error"] = type(exc).__name__
        return out

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior_before = before_obs.get("error") == "AttributeError"
    behavior_after = after_obs.get("error") is None and after_obs.get("prompt_len", 0) > 0
    ok = behavior_before and behavior_after
    return {
        "source_id": "crewaiinc__crewai-2408",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_system_only_attribute_error": behavior_before,
            "after_system_only_fallback_prompt": behavior_after,
        },
    }


def verify_1818() -> dict:
    before_p, after_p = case_paths(
        "crewaiinc__crewai-1818", "src/crewai/utilities/planning_handler.py"
    )

    class Knowledge:
        def __init__(self, content: str):
            self.content = content

    class AgentStub:
        role = "Researcher"
        goal = "Find facts"
        tools = []

        def __init__(self):
            self.knowledge_sources = [Knowledge("secret-fact-42")]

    class TaskStub:
        description = "Do research"
        expected_output = "Report"
        tools = []

        def __init__(self):
            self.agent = AgentStub()

    def run(path: Path) -> dict:
        for name in list(sys.modules):
            if name == "crewai" or name.startswith("crewai."):
                del sys.modules[name]
        agent_mod = ensure_pkg("crewai.agent")
        task_mod = ensure_pkg("crewai.task")
        agent_mod.Agent = AgentStub
        task_mod.Task = TaskStub
        spec = importlib.util.spec_from_file_location(
            "crewai.utilities.planning_handler", path
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        planner = mod.CrewPlanner(tasks=[TaskStub()])
        summary = planner._create_tasks_summary()
        return {"has_knowledge": "secret-fact-42" in summary, "len": len(summary)}

    before_obs = run(before_p)
    after_obs = run(after_p)
    behavior_before = before_obs["has_knowledge"] is False
    behavior_after = after_obs["has_knowledge"] is True
    preservation = before_obs["len"] > 0 and after_obs["len"] > 0
    ok = behavior_before and behavior_after and preservation
    return {
        "source_id": "crewaiinc__crewai-1818",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
        },
        "before_observed": before_obs,
        "after_observed": after_obs,
        "checks": {
            "before_omits_knowledge": behavior_before,
            "after_includes_knowledge": behavior_after,
            "summary_nonempty_both": preservation,
        },
    }


def verify_3150() -> dict:
    """Run isolated child with task bootstrap fixture (heavy Task deps)."""
    script = FIXTURES / "probe_task3150_save.py"
    before_p, after_p = case_paths("crewaiinc__crewai-3150", "src/crewai/task.py")
    tmp_base = FIXTURES / "tmp_out_3150"
    if tmp_base.exists():
        shutil.rmtree(tmp_base)
    tmp_base.mkdir(parents=True)

    def run_child(task_path: Path, create_directory: bool) -> dict:
        out_dir = tmp_base / ("after" if "applied" in str(task_path) else "before")
        out_dir.mkdir(parents=True, exist_ok=True)
        nested = out_dir / "nested" / "deep"
        target = nested / "out.txt"
        env = os.environ.copy()
        cmd = [
            sys.executable,
            str(script),
            str(task_path),
            str(target),
            "true" if create_directory else "false",
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_SEC,
            cwd=str(FIXTURES),
            env=env,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()[:500],
            "file_exists": target.is_file(),
        }

    before_fail = run_child(before_p, False)
    before_ok = run_child(before_p, True)
    after_fail = run_child(after_p, False)
    after_ok = run_child(after_p, True)

    # Before: create_directory flag absent -> always mkdir -> False flag still creates (no raise)
    # After: create_directory False should fail (nonzero) and no file; True creates file
    behavior_before = (
        before_fail["returncode"] == 0
        and before_fail["file_exists"] is True
        and before_ok["file_exists"] is True
    )
    behavior_after = (
        after_fail["returncode"] != 0
        and after_fail["file_exists"] is False
        and after_ok["returncode"] == 0
        and after_ok["file_exists"] is True
    )
    ok = behavior_before and behavior_after
    return {
        "source_id": "crewaiinc__crewai-3150",
        "executed": True,
        "status": "pass" if ok else "reject",
        "source_hashes": {
            "before_sha256": sha256_file(before_p),
            "after_sha256": sha256_file(after_p),
            "fixture_script_sha256": sha256_file(script),
        },
        "before_observed": {"mkdir_always": before_fail, "mkdir_always_ok": before_ok},
        "after_observed": {"create_directory_false": after_fail, "create_directory_true": after_ok},
        "checks": {
            "before_always_creates_even_when_flag_false": behavior_before,
            "after_false_raises_true_creates": behavior_after,
        },
    }


VERIFY_FNS: Dict[str, Callable[[], dict]] = {
    "crewaiinc__crewai-2570": verify_2570,
    "crewaiinc__crewai-3150": verify_3150,
    "crewaiinc__crewai-2588": verify_2588,
    "crewaiinc__crewai-1818": verify_1818,
    "crewaiinc__crewai-2408": verify_2408,
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
        "batch": "crewai_batch",
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
    # Update curated verification_status honestly
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
