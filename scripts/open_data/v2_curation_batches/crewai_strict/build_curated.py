#!/usr/bin/env python3
"""Build crewai_strict curated_examples.json (correct 9-field modules + AST qualnames)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ast_utils import class_through_methods_excerpt, module_qual_prefix

BATCH = Path(__file__).resolve().parent
WORK = BATCH.parent
REPO_ROOT = WORK.parent.parent
CASES = WORK / "main_candidates/cases"
RAW = WORK / "main_candidates/raw_records.jsonl"
LEGACY = WORK / "crewai_batch"

PERMITTED = ["tools", "agent_loop", "context_mgmt"]
POLICY = "proposal_single_plan_v6"
PARTITION = "AGENT_INFRASTRUCTURE_AUXILIARY"

ACCEPTED = [
    "crewaiinc__crewai-2570",
    "crewaiinc__crewai-3150",
    "crewaiinc__crewai-2588",
    "crewaiinc__crewai-1818",
    "crewaiinc__crewai-2408",
]


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/")


def load_raw() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec["instance_id"] in ACCEPTED:
                out[rec["instance_id"]] = rec
    return out


def build_row(
    iid: str,
    rec: dict,
    *,
    harness_module: str,
    visible_rel: str,
    orig_path: str,
    before_path: Path,
    class_name: str,
    method_names: list[str],
    plan: dict,
    notes: str,
    dependency_refs: list[dict] | None = None,
) -> dict:
    excerpt, start, end, qual_spans = class_through_methods_excerpt(
        before_path, orig_path, class_name, method_names
    )
    visible_path = f"{harness_module}/{visible_rel.split('/')[-1]}"
    source_refs = []
    for qualname, s, e in qual_spans:
        source_refs.append(
            {
                "qualname": qualname,
                "original_path": orig_path,
                "artifact_path": rel(before_path),
                "sha256": sha256_path(before_path),
                "start_line": s,
                "end_line": e,
            }
        )
    issue = iid.split("-")[-1]
    row = {
        "source_id": iid,
        "split": "train",
        "verification_status": "strict_pending",
        "curation_notes": notes,
        "lineage": {
            "prior_batch": "work/open_dataset_integration_v2/crewai_batch",
            "prior_batch_immutable": True,
        },
        "provenance": {
            "partition": PARTITION,
            "original_teacher_plan": False,
            "formal": False,
        },
        "source_mapping": {
            "dataset": "nebius/SWE-rebench-V2",
            "repo": rec["repo"],
            "base_commit": rec["base_commit"],
            "original_source_path": orig_path,
            "visible_path": visible_path,
            "harness_module": harness_module,
            "module_path_prefix": f"{harness_module}/",
        },
        "source_refs": source_refs,
        "state": {
            "state_id": f"REBENCH-CREWAI-STRICT-{issue}",
            "root_lineage_id": f"crewAIInc/crewAI@{rec['base_commit'][:12]}",
            "split": "train",
            "plan_policy_version": POLICY,
            "permitted_modules": PERMITTED,
            "module_code": {visible_path: excerpt},
            "history": [],
            "evidence": [
                {
                    "id": "ev_reported_issue",
                    "text": "Published reporter issue (SWE-rebench-V2): "
                    + rec.get("problem_statement", "").replace("\n", " ")[:1500],
                },
                {
                    "id": "ev_exact_source",
                    "text": (
                        f"Exact before excerpt {orig_path} lines {start}-{end} "
                        f"sha256={sha256_path(before_path)} (module qual prefix "
                        f"{module_qual_prefix(orig_path)})."
                    ),
                },
            ],
            "verification_catalog": [],
        },
        "plan": plan,
    }
    if dependency_refs:
        row["dependency_source_refs"] = dependency_refs
    return row


def main() -> None:
    raw = load_raw()
    rows: list[dict] = []

    # 2570
    iid = "crewaiinc__crewai-2570"
    rec = raw[iid]
    bp = CASES / iid / "before" / "src/crewai/tools/base_tool.py"
    plan = {
        "action": "edit",
        "target_module": "tools",
        "locations": ["tools/base_tool.py"],
        "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
        "problem_hypothesis": (
            "Current-code: crewai.tools.base_tool.BaseTool.run returns self._run(...) directly. "
            "Reporter: async tool _run returns coroutine object. Hypothesis: asyncio.iscoroutine "
            "then asyncio.run before return (nested active event loop not verified here)."
        ),
        "intended_change": (
            "In BaseTool.run, assign result = self._run(...); if asyncio.iscoroutine(result) "
            "return asyncio.run(result); else return result."
        ),
        "preserve_behaviors": [
            "Sync _run return values pass through unchanged",
            "Using Tool: {name} print line still emitted before _run",
        ],
        "verification_plan": [
            "Async _run: before returns coroutine object; after returns resolved dict",
            "Sync _run: int 7 before and after",
            "Stdout preserves Using Tool print prefix",
        ],
        "verification_kind": None,
    }
    rows.append(
        build_row(
            iid,
            rec,
            harness_module="tools",
            visible_rel="base_tool.py",
            orig_path="src/crewai/tools/base_tool.py",
            before_path=bp,
            class_name="BaseTool",
            method_names=["run", "_run"],
            plan=plan,
            notes="Strict acceptance; tools module alias; async.run scope limitation documented in verify.",
        )
    )

    # 3150
    iid = "crewaiinc__crewai-3150"
    rec = raw[iid]
    bp = CASES / iid / "before" / "src/crewai/task.py"
    plan = {
        "action": "edit",
        "target_module": "agent_loop",
        "locations": ["agent_loop/task.py"],
        "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
        "problem_hypothesis": (
            "Before: crewai.task.Task._save_file always mkdir(parents=True) when parent missing. "
            "Reporter wants opt-out via new Task.create_directory field (not in before state)."
        ),
        "intended_change": (
            "Add Task.create_directory default True on Task model; gate mkdir in _save_file; "
            "when create_directory is False and parent missing raise RuntimeError; when parent "
            "exists still write file."
        ),
        "preserve_behaviors": [
            "output_file unset raises ValueError in _save_file (both versions)",
            "Default create_directory=True still creates nested dirs and writes content",
            "create_directory=False with existing parent directory still writes content",
        ],
        "verification_plan": [
            "Missing parent + create_directory=False: after rejects; before still creates",
            "Missing parent + default True: nested file created with content hello",
            "Existing parent + create_directory=False: after still writes",
            "output_file None: ValueError both versions",
        ],
        "verification_kind": None,
    }
    rows.append(
        build_row(
            iid,
            rec,
            harness_module="agent_loop",
            visible_rel="task.py",
            orig_path="src/crewai/task.py",
            before_path=bp,
            class_name="Task",
            method_names=["_save_file"],
            plan=plan,
            notes="Before state cites Task container + _save_file only; create_directory is plan-only field.",
        )
    )

    # 2588
    iid = "crewaiinc__crewai-2588"
    rec = raw[iid]
    bp = CASES / iid / "before" / "src/crewai/memory/storage/mem0_storage.py"
    plan = {
        "action": "edit",
        "target_module": "context_mgmt",
        "locations": ["context_mgmt/mem0_storage.py"],
        "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
        "problem_hypothesis": (
            "Before: Mem0Storage.__init__ calls Memory.from_config(config) where config is API "
            "metadata dict, not mem0_local_config. Hypothesis: pass mem0_local_config."
        ),
        "intended_change": (
            "In local branch use Memory.from_config(mem0_local_config) instead of Memory.from_config(config)."
        ),
        "preserve_behaviors": [
            "No API key and empty local config: Memory() default constructor",
            "MEM0_API_KEY set: MemoryClient constructed with same org/project kwargs as before",
        ],
        "verification_plan": [
            "Local mem0_local_config forwarded to fake Memory.from_config",
            "Default Memory() path unchanged",
            "Dummy MEM0_API_KEY env: MemoryClient receives api_key/org_id/project_id",
        ],
        "verification_kind": None,
    }
    rows.append(
        build_row(
            iid,
            rec,
            harness_module="context_mgmt",
            visible_rel="mem0_storage.py",
            orig_path="src/crewai/memory/storage/mem0_storage.py",
            before_path=bp,
            class_name="Mem0Storage",
            method_names=["__init__"],
            plan=plan,
            notes="Mem0 external I/O faked at import boundary only.",
        )
    )

    # 1818
    iid = "crewaiinc__crewai-1818"
    rec = raw[iid]
    bp = CASES / iid / "before" / "src/crewai/utilities/planning_handler.py"
    applied_ph = CASES / iid / "applied" / "src/crewai/utilities/planning_handler.py"
    dep_refs = [
        {
            "qualname": "crewai.agent.Agent.knowledge_sources",
            "original_path": "src/crewai/agent.py",
            "note": (
                "Field not present in pinned case tree; runtime read documented in applied "
                "CrewPlanner._get_agent_knowledge at same PR lineage."
            ),
            "access_site_qualname": "crewai.utilities.planning_handler.CrewPlanner._get_agent_knowledge",
            "access_site_artifact_path": rel(applied_ph),
            "access_site_sha256": sha256_path(applied_ph),
            "access_site_start_line": 75,
            "access_site_end_line": 90,
        }
    ]
    plan = {
        "action": "edit",
        "target_module": "agent_loop",
        "locations": ["agent_loop/planning_handler.py"],
        "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
        "problem_hypothesis": (
            "Before: CrewPlanner._create_tasks_summary omits agent knowledge. Reporter: "
            "knowledge_sources ignored during planning."
        ),
        "intended_change": (
            "Add _get_agent_knowledge(task) reading task.agent.knowledge_sources[].content; "
            "append agent_knowledge to summary when non-empty."
        ),
        "preserve_behaviors": [
            "Tasks without knowledge_sources omit agent_knowledge key",
            "task_description, task_expected_output, agent role/goal remain present",
        ],
        "verification_plan": [
            "With knowledge: after summary contains content; before does not",
            "Without knowledge: after summary lacks agent_knowledge; core fields present",
        ],
        "verification_kind": None,
    }
    rows.append(
        build_row(
            iid,
            rec,
            harness_module="agent_loop",
            visible_rel="planning_handler.py",
            orig_path="src/crewai/utilities/planning_handler.py",
            before_path=bp,
            class_name="CrewPlanner",
            method_names=["_create_tasks_summary"],
            plan=plan,
            notes="CrewPlanner orchestration under agent_loop; Agent API via dependency_source_refs.",
            dependency_refs=dep_refs,
        )
    )

    # 2408
    iid = "crewaiinc__crewai-2408"
    rec = raw[iid]
    bp = CASES / iid / "before" / "src/crewai/utilities/prompts.py"
    plan = {
        "action": "edit",
        "target_module": "context_mgmt",
        "locations": ["context_mgmt/prompts.py"],
        "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
        "problem_hypothesis": (
            "Before: Prompts._build_prompt uses custom branch when either template set; "
            "system-only leaves prompt_template None causing AttributeError."
        ),
        "intended_change": (
            "Require both templates for custom branch; fallback to i18n slices if either missing; "
            "optional response_template guarded."
        ),
        "preserve_behaviors": [
            "No custom templates: rendered prompt identical before/after",
            "Both custom templates provided: rendered prompt identical before/after",
        ],
        "verification_plan": [
            "Default path string equality before vs after",
            "Dual-template path string equality before vs after",
            "System-only: before AttributeError; after equals default-path string",
        ],
        "verification_kind": None,
    }
    rows.append(
        build_row(
            iid,
            rec,
            harness_module="context_mgmt",
            visible_rel="prompts.py",
            orig_path="src/crewai/utilities/prompts.py",
            before_path=bp,
            class_name="Prompts",
            method_names=["_build_prompt"],
            plan=plan,
            notes="Prompt assembly under context_mgmt; docs patch excluded.",
        )
    )

    (BATCH / "curated_examples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    meta = {
        "batch": "crewai_strict",
        "prior_immutable_batch": rel(LEGACY),
        "permitted_modules": PERMITTED,
        "plan_policy_version": POLICY,
        "accepted_count": len(rows),
    }
    (BATCH / "batch_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} strict curated rows")


if __name__ == "__main__":
    main()
