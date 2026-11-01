#!/usr/bin/env python3
"""Strict crewai acceptance: separate behavior vs preservation; honest pass/reject/pending."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

BATCH = Path(__file__).resolve().parent
WORK = BATCH.parent
CASES = WORK / "main_candidates/cases"
FIXTURES = BATCH / "fixtures"
PROBE3150 = FIXTURES / "probe_task3150_save.py"
CHILD_TIMEOUT_SEC = 20

sys.path.insert(0, str(FIXTURES))
from fixture_paths import ensure_workspace_root, fixture_path, safe_rmtree  # noqa: E402

ACCEPTED = [
    "crewaiinc__crewai-2570",
    "crewaiinc__crewai-3150",
    "crewaiinc__crewai-2588",
    "crewaiinc__crewai-1818",
    "crewaiinc__crewai-2408",
]


async def _async_probe_coro():
    return {"rows": [1, 2]}


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


def sample_status(behavior: Dict[str, bool], preservation: Dict[str, bool], pending: List[str]) -> str:
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


@contextmanager
def dummy_mem0_api_key():
    old = os.environ.get("MEM0_API_KEY")
    os.environ["MEM0_API_KEY"] = "dummy-test-key-not-real"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MEM0_API_KEY", None)
        else:
            os.environ["MEM0_API_KEY"] = old


def load_base_tool_module(base_tool_path: Path):
    for name in list(sys.modules):
        if name == "crewai" or name.startswith("crewai."):
            del sys.modules[name]
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
    sid = "crewaiinc__crewai-2570"
    before_p, after_p = case_paths(sid, "src/crewai/tools/base_tool.py")

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
        if hasattr(raw_async, "close"):
            async_is_coro = True
            async_value = None
            raw_async.close()
        else:
            async_is_coro = False
            async_value = raw_async

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sync_out = SyncTool().run()
        return {
            "async_is_coroutine": async_is_coro,
            "async_value": async_value,
            "sync_value": sync_out,
            "stdout": buf.getvalue(),
        }

    before_obs = run_probe(before_p)
    after_obs = run_probe(after_p)
    behavior = {
        "before_async_returns_coroutine_object": before_obs["async_is_coroutine"] is True,
        "after_async_resolves_to_dict": (
            after_obs["async_is_coroutine"] is False and after_obs["async_value"] == {"rows": [1, 2]}
        ),
    }
    preservation = {
        "sync_tool_returns_7_before_after": (
            before_obs["sync_value"] == 7 and after_obs["sync_value"] == 7
        ),
        "using_tool_print_line_unchanged": (
            "Using Tool: sync_probe" in before_obs["stdout"]
            and "Using Tool: sync_probe" in after_obs["stdout"]
        ),
    }
    pending: List[str] = []
    limitations = [
        "asyncio.run inside already-running event loop not exercised; do not claim all async contexts.",
    ]
    status = sample_status(behavior, preservation, pending)
    return {
        "source_id": sid,
        "executed": True,
        "status": status,
        "behavior": behavior,
        "preservation": preservation,
        "pending": pending,
        "limitations": limitations,
        "source_hashes": {"before": sha256_file(before_p), "after": sha256_file(after_p)},
        "observations": {"before": before_obs, "after": after_obs},
    }


def run_3150_child(task_path: Path, case_tag: str, rel_out: str, mode: str) -> dict:
    cmd = [sys.executable, str(PROBE3150), str(task_path), case_tag, rel_out, mode]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SEC,
        cwd=str(FIXTURES),
    )
    target = fixture_path(case_tag, rel_out) if mode != "unset" else None
    content = None
    if target and target.is_file():
        content = target.read_text(encoding="utf-8")
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[:400],
        "file_exists": bool(target and target.is_file()),
        "content": content,
        "target": str(target) if target else None,
    }


def verify_3150() -> dict:
    sid = "crewaiinc__crewai-3150"
    before_p, after_p = case_paths(sid, "src/crewai/task.py")
    root = ensure_workspace_root()
    tag = sid.replace("__", "_")
    tag_root = root / tag
    if tag_root.exists():
        safe_rmtree(tag_root)

    before_false = run_3150_child(before_p, tag, "before/missing_parent/out.txt", "false")
    after_false = run_3150_child(after_p, tag, "after/missing_parent/out.txt", "false")
    after_true = run_3150_child(after_p, tag, "after/missing_parent_ok/out.txt", "true")
    parent = fixture_path(tag, "after/existing_parent/deep", mkdir=True)
    parent.mkdir(parents=True, exist_ok=True)
    after_existing = run_3150_child(
        after_p, tag, "after/existing_parent/deep/out.txt", "false"
    )
    before_unset = run_3150_child(before_p, tag, "noop.txt", "unset")
    after_unset = run_3150_child(after_p, tag, "noop.txt", "unset")

    behavior = {
        "after_missing_parent_create_directory_false_fails": (
            after_false["returncode"] != 0
            and not after_false["file_exists"]
            and "RuntimeError" in after_false["stdout"]
        ),
        "after_missing_parent_create_directory_true_writes": (
            after_true["returncode"] == 0
            and after_true["file_exists"]
            and after_true["content"] == "hello"
        ),
        "before_missing_parent_always_creates_even_false_flag": (
            before_false["returncode"] == 0 and before_false["content"] == "hello"
        ),
    }
    preservation = {
        "output_file_unset_valueerror_before": (
            before_unset["returncode"] != 0 and "ValueError" in before_unset["stdout"]
        ),
        "output_file_unset_valueerror_after": (
            after_unset["returncode"] != 0 and "ValueError" in after_unset["stdout"]
        ),
        "after_existing_parent_false_still_writes": (
            after_existing["returncode"] == 0
            and after_existing["file_exists"]
            and after_existing["content"] == "hello"
        ),
        "default_true_nested_content_hello": after_true["content"] == "hello",
    }
    pending: List[str] = []
    status = sample_status(behavior, preservation, pending)
    return {
        "source_id": sid,
        "executed": True,
        "status": status,
        "behavior": behavior,
        "preservation": preservation,
        "pending": pending,
        "source_hashes": {
            "before": sha256_file(before_p),
            "after": sha256_file(after_p),
            "probe": sha256_file(PROBE3150),
            "workspace_root": sha256_file(root / ".keep") if (root / ".keep").exists() else None,
        },
        "fixture_workspace_root": str(root),
        "observations": {
            "before_false": before_false,
            "after_false": after_false,
            "after_true": after_true,
            "after_existing_parent": after_existing,
            "before_unset": before_unset,
            "after_unset": after_unset,
        },
    }


def verify_2588() -> dict:
    sid = "crewaiinc__crewai-2588"
    before_p, after_p = case_paths(sid, "src/crewai/memory/storage/mem0_storage.py")
    captured: dict[str, Any] = {
        "from_config_arg": None,
        "memory_default": False,
        "client_kwargs": None,
    }

    class Memory:
        def __init__(self):
            captured["memory_default"] = True

        @classmethod
        def from_config(cls, cfg):
            captured["from_config_arg"] = cfg
            return cls()

    class MemoryClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

    def load_run(path: Path, config: dict, use_env_key: bool = False) -> dict:
        captured["from_config_arg"] = None
        captured["memory_default"] = False
        captured["client_kwargs"] = None
        for name in list(sys.modules):
            if name in ("mem0",) or name.startswith("crewai."):
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
        if use_env_key:
            with dummy_mem0_api_key():
                mod.Mem0Storage(type="short_term", crew=None, config=config)
        else:
            old = os.environ.pop("MEM0_API_KEY", None)
            try:
                mod.Mem0Storage(type="short_term", crew=None, config=config)
            finally:
                if old is not None:
                    os.environ["MEM0_API_KEY"] = old
        return dict(captured)

    local_cfg = {"vector_store": {"provider": "qdrant", "config": {"host": "localhost"}}}
    base_cfg = {"local_mem0_config": local_cfg, "user_id": "u1"}
    before_local = load_run(before_p, base_cfg)
    after_local = load_run(after_p, base_cfg)
    before_default = load_run(before_p, {"user_id": "u1"})
    after_default = load_run(after_p, {"user_id": "u1"})
    api_cfg = {"api_key": "cfg-key", "org_id": "org1", "project_id": "proj1", "user_id": "u1"}
    before_client = load_run(before_p, api_cfg)
    after_client = load_run(after_p, api_cfg)
    env_client_before = load_run(before_p, {"user_id": "u1"}, use_env_key=True)
    env_client_after = load_run(after_p, {"user_id": "u1"}, use_env_key=True)

    behavior = {
        "before_local_passes_full_config_to_from_config": before_local["from_config_arg"]
        == base_cfg,
        "after_local_passes_local_mem0_config_only": after_local["from_config_arg"] == local_cfg,
    }
    preservation = {
        "before_default_uses_Memory_constructor": before_default["memory_default"] is True
        and before_default["from_config_arg"] is None,
        "after_default_uses_Memory_constructor": after_default["memory_default"] is True
        and after_default["from_config_arg"] is None,
        "before_api_config_MemoryClient_kwargs": before_client["client_kwargs"]
        == {"api_key": "cfg-key", "org_id": "org1", "project_id": "proj1"},
        "after_api_config_MemoryClient_kwargs": after_client["client_kwargs"]
        == {"api_key": "cfg-key", "org_id": "org1", "project_id": "proj1"},
        "before_env_api_key_MemoryClient": env_client_before["client_kwargs"]
        == {"api_key": "dummy-test-key-not-real"},
        "after_env_api_key_MemoryClient": env_client_after["client_kwargs"]
        == {"api_key": "dummy-test-key-not-real"},
    }
    pending: List[str] = []
    status = sample_status(behavior, preservation, pending)
    return {
        "source_id": sid,
        "executed": True,
        "status": status,
        "behavior": behavior,
        "preservation": preservation,
        "pending": pending,
        "source_hashes": {"before": sha256_file(before_p), "after": sha256_file(after_p)},
        "observations": {
            "before_local": before_local,
            "after_local": after_local,
            "before_default": before_default,
            "after_default": after_default,
            "before_client": before_client,
            "after_client": after_client,
        },
    }


def _1818_summary(path: Path, with_knowledge: bool) -> str:
    class Knowledge:
        def __init__(self, content: str):
            self.content = content

    class AgentStub:
        role = "Researcher"
        goal = "Find facts"
        tools = []

        def __init__(self, with_k: bool):
            self.knowledge_sources = [Knowledge("secret-fact-42")] if with_k else []

    class TaskStub:
        description = "Do research"
        expected_output = "Report"
        tools = ["toolA"]

        def __init__(self, with_k: bool):
            self.agent = AgentStub(with_k)

    for name in list(sys.modules):
        if name == "crewai" or name.startswith("crewai."):
            del sys.modules[name]
    ensure_pkg("crewai.agent").Agent = AgentStub
    ensure_pkg("crewai.task").Task = TaskStub
    spec = importlib.util.spec_from_file_location(
        "crewai.utilities.planning_handler", path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    planner = mod.CrewPlanner(tasks=[TaskStub(with_knowledge)])
    return planner._create_tasks_summary()


def _1818_fields(summary: str) -> dict:
    return {
        "has_knowledge": "agent_knowledge" in summary,
        "has_description": "Do research" in summary and "task_description" in summary,
        "has_expected": "Report" in summary and "task_expected_output" in summary,
        "has_role": "Researcher" in summary,
        "has_goal": "Find facts" in summary,
        "has_task_tools": "toolA" in summary or "task_tools" in summary,
    }


def verify_1818() -> dict:
    sid = "crewaiinc__crewai-1818"
    before_p, after_p = case_paths(sid, "src/crewai/utilities/planning_handler.py")
    b_k = _1818_summary(before_p, True)
    a_k = _1818_summary(after_p, True)
    a_no = _1818_summary(after_p, False)
    b_no = _1818_summary(before_p, False)
    behavior = {
        "before_with_knowledge_omits_agent_knowledge_key": not _1818_fields(b_k)["has_knowledge"],
        "after_with_knowledge_includes_content": "secret-fact-42" in a_k,
    }
    f_b_no = _1818_fields(b_no)
    f_a_no = _1818_fields(a_no)
    preservation = {
        "without_knowledge_no_agent_knowledge_after": not f_a_no["has_knowledge"],
        "without_knowledge_core_fields_after": all(
            f_a_no[k]
            for k in ("has_description", "has_expected", "has_role", "has_goal", "has_task_tools")
        ),
        "without_knowledge_core_fields_before": all(
            f_b_no[k]
            for k in ("has_description", "has_expected", "has_role", "has_goal", "has_task_tools")
        ),
    }
    pending: List[str] = []
    status = sample_status(behavior, preservation, pending)
    return {
        "source_id": sid,
        "executed": True,
        "status": status,
        "behavior": behavior,
        "preservation": preservation,
        "pending": pending,
        "source_hashes": {"before": sha256_file(before_p), "after": sha256_file(after_p)},
        "observations": {
            "before_with_knowledge_fields": _1818_fields(b_k),
            "after_with_knowledge_fields": _1818_fields(a_k),
            "after_without_knowledge_fields": f_a_no,
        },
    }


def _load_prompts(path: Path):
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

    for name in list(sys.modules):
        if name == "crewai" or name.startswith("crewai."):
            del sys.modules[name]
    ensure_pkg("crewai.utilities").I18N = I18N
    spec = importlib.util.spec_from_file_location("crewai.utilities.prompts", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Prompts, AgentStub, I18N


def verify_2408() -> dict:
    sid = "crewaiinc__crewai-2408"
    before_p, after_p = case_paths(sid, "src/crewai/utilities/prompts.py")
    PromptsB, AgentB, _ = _load_prompts(before_p)
    PromptsA, AgentA, _ = _load_prompts(after_p)
    components = ["role_playing", "no_tools", "task"]
    agent_b = AgentB()
    agent_a = AgentA()

    def default_render(Prompts, agent):
        p = Prompts(agent=agent)
        return p._build_prompt(components, None, None, None)

    def dual_render(Prompts, agent):
        p = Prompts(agent=agent)
        return p._build_prompt(
            components,
            "SYS {{ .System }}",
            "PROMPT {{ .Prompt }}",
            "RESP {{ .Response }}",
        )

    default_before = default_render(PromptsB, agent_b)
    default_after = default_render(PromptsA, agent_a)
    dual_before = dual_render(PromptsB, agent_b)
    dual_after = dual_render(PromptsA, agent_a)

    system_only_before: dict = {"error": None, "text": None}
    try:
        p = PromptsB(agent=agent_b, system_template="ONLY {{ .System }}", prompt_template=None)
        system_only_before["text"] = p._build_prompt(
            components, p.system_template, p.prompt_template, None
        )
    except Exception as exc:  # noqa: BLE001
        system_only_before["error"] = type(exc).__name__

    p_after = PromptsA(agent=agent_a, system_template="ONLY {{ .System }}", prompt_template=None)
    system_only_after = p_after._build_prompt(
        components, p_after.system_template, p_after.prompt_template, None
    )

    behavior = {
        "before_system_only_raises": system_only_before["error"] == "AttributeError",
        "after_system_only_matches_default_path": system_only_after == default_after,
    }
    preservation = {
        "default_template_output_unchanged": default_before == default_after,
        "dual_template_output_unchanged": dual_before == dual_after,
    }
    pending: List[str] = []
    status = sample_status(behavior, preservation, pending)
    return {
        "source_id": sid,
        "executed": True,
        "status": status,
        "behavior": behavior,
        "preservation": preservation,
        "pending": pending,
        "source_hashes": {"before": sha256_file(before_p), "after": sha256_file(after_p)},
        "observations": {
            "default_before": default_before,
            "default_after": default_after,
            "dual_before": dual_before,
            "dual_after": dual_after,
            "system_only_before": system_only_before,
            "system_only_after": system_only_after,
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
    ensure_workspace_root()
    (FIXTURES / "workspace_root" / ".keep").write_text("", encoding="utf-8")
    samples: List[dict] = []
    for iid in ACCEPTED:
        try:
            samples.append(VERIFY_FNS[iid]())
        except subprocess.TimeoutExpired:
            samples.append(
                {
                    "source_id": iid,
                    "executed": False,
                    "status": "reject",
                    "behavior": {},
                    "preservation": {},
                    "pending": ["child_timeout"],
                    "error": "child_timeout",
                }
            )
        except Exception as exc:  # noqa: BLE001
            samples.append(
                {
                    "source_id": iid,
                    "executed": False,
                    "status": "reject",
                    "behavior": {},
                    "preservation": {},
                    "pending": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    payload = {
        "batch": "crewai_strict",
        "prior_immutable_batch": "work/open_dataset_integration_v2/crewai_batch",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "child_timeout_sec": CHILD_TIMEOUT_SEC,
        "samples": samples,
        "summary": {
            "pass": sum(1 for s in samples if s.get("status") == "pass"),
            "reject": sum(1 for s in samples if s.get("status") == "reject"),
            "pending": sum(1 for s in samples if s.get("status") == "pending"),
        },
    }
    (BATCH / "verification_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    curated_path = BATCH / "curated_examples.json"
    if curated_path.exists():
        rows = json.loads(curated_path.read_text(encoding="utf-8"))
        by_id = {s["source_id"]: s for s in samples}
        for row in rows:
            st = by_id.get(row["source_id"], {}).get("status", "pending")
            row["verification_status"] = f"strict_bounded_{st}"
            row["verification_fixture_path"] = (
                f"work/open_dataset_integration_v2/crewai_strict/verification_results.json"
            )
        curated_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
