#!/usr/bin/env python3
"""Assemble crewai_batch curated_examples.json and curation_decisions.json from pinned cases."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Callable

BATCH = Path(__file__).resolve().parent
WORK = BATCH.parent
REPO_ROOT = WORK.parent.parent
CASES = WORK / "main_candidates/cases"
RAW = WORK / "main_candidates/raw_records.jsonl"
OUT = BATCH

PERMITTED = ["tools", "agent_loop", "memory", "planning", "prompts"]
POLICY = "proposal_single_plan_v6"
VERIFY_STATUS = "pending_bounded_verification"
PARTITION = "AGENT_INFRASTRUCTURE_AUXILIARY"

INSTANCES = [
    "crewaiinc__crewai-2570",
    "crewaiinc__crewai-3150",
    "crewaiinc__crewai-2588",
    "crewaiinc__crewai-1818",
    "crewaiinc__crewai-2408",
    "crewaiinc__crewai-2495",
    "crewaiinc__crewai-2552",
]


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def extract_class_methods(
    path: Path, class_name: str, method_names: list[str]
) -> tuple[str, int, int]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            ends: list[int] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in method_names:
                    ends.append(item.end_lineno or item.lineno)
            if not ends:
                raise SystemExit(f"methods {method_names} not in {class_name} @ {path}")
            start = node.lineno
            end = max(ends)
            excerpt = "\n".join(lines[start - 1 : end]) + "\n"
            return excerpt, start, end
    raise SystemExit(f"class {class_name} not found in {path}")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/")


SPECS: dict[str, dict] = {
    "crewaiinc__crewai-2570": {
        "decision": "accept",
        "module": "tools",
        "visible": "tools/base_tool.py",
        "orig": "src/crewai/tools/base_tool.py",
        "extract": lambda b: extract_class_methods(
            b / "src/crewai/tools/base_tool.py", "BaseTool", ["run", "_run"]
        ),
        "qualnames": ["crewai.tools.base_tool.BaseTool.run"],
        "hypothesis": (
            "Current-code: BaseTool.run returns self._run(...) directly. "
            "Reporter: async Snowflake _run yields coroutine object to caller. "
            "Hypothesis: detect coroutine with asyncio.iscoroutine and asyncio.run before return."
        ),
        "intended": (
            "In BaseTool.run, store _run result; if asyncio.iscoroutine(result), "
            "return asyncio.run(result); else return result."
        ),
        "preserve": [
            "Sync _run return values pass through unchanged",
            "Tool still prints Using Tool name line before executing _run",
        ],
        "vplan": [
            "Async _run tool returns resolved value not coroutine via BaseTool.run",
            "Sync lambda tool still returns int literal",
        ],
        "notes": "Scoped to base_tool.py; original patch single-file.",
    },
    "crewaiinc__crewai-3150": {
        "decision": "accept",
        "module": "agent_loop",
        "visible": "agent_loop/task.py",
        "orig": "src/crewai/task.py",
        "extract": lambda b: extract_class_methods(
            b / "src/crewai/task.py", "Task", ["_save_file"]
        ),
        "qualnames": ["crewai.task.Task._save_file", "crewai.task.Task.create_directory"],
        "hypothesis": (
            "Before: _save_file always mkdir(parents=True) when parent missing. "
            "Reporter wants opt-out. Hypothesis: create_directory field gates mkdir; "
            "False raises when parent absent."
        ),
        "intended": (
            "Add create_directory field default True; in _save_file mkdir only when "
            "create_directory and parent missing; elif not create_directory and missing "
            "parent raise RuntimeError."
        ),
        "preserve": [
            "create_directory=True still creates nested dirs and writes content",
            "output_file unset still raises ValueError",
        ],
        "vplan": [
            "Fixture nested path with create_directory=False raises before write",
            "Same path create_directory=True creates file",
        ],
        "notes": "Mechanism in Task._save_file only; field declared on Task model.",
    },
    "crewaiinc__crewai-2588": {
        "decision": "accept",
        "module": "memory",
        "visible": "memory/mem0_storage.py",
        "orig": "src/crewai/memory/storage/mem0_storage.py",
        "extract": lambda b: extract_class_methods(
            b / "src/crewai/memory/storage/mem0_storage.py", "Mem0Storage", ["__init__"]
        ),
        "qualnames": ["crewai.memory.storage.mem0_storage.Mem0Storage.__init__"],
        "hypothesis": (
            "Current-code bug: Memory.from_config(config) passes wrong dict "
            "(api metadata not local_mem0_config). Reporter: local mem0 config ignored. "
            "Hypothesis: pass mem0_local_config into Memory.from_config."
        ),
        "intended": (
            "Replace Memory.from_config(config) with Memory.from_config(mem0_local_config) "
            "in local branch."
        ),
        "preserve": [
            "When MEM0_API_KEY set, MemoryClient path unchanged",
            "Empty local config still uses Memory() default",
        ],
        "vplan": [
            "Fake mem0.Memory.from_config records argument; before gets config dict without local key; after gets local_mem0_config",
        ],
        "notes": "Mem0 faked at import boundary only.",
    },
    "crewaiinc__crewai-1818": {
        "decision": "accept",
        "module": "planning",
        "visible": "planning/planning_handler.py",
        "orig": "src/crewai/utilities/planning_handler.py",
        "extract": lambda b: extract_class_methods(
            b / "src/crewai/utilities/planning_handler.py",
            "CrewPlanner",
            ["_get_agent_knowledge", "_create_tasks_summary"],
        ),
        "qualnames": ["crewai.utilities.planning_handler.CrewPlanner._create_tasks_summary"],
        "hypothesis": (
            "Before: _create_tasks_summary omits agent knowledge from planning text. "
            "Reporter: knowledge_sources ignored. Hypothesis: _get_agent_knowledge plus "
            "inject agent_knowledge into summary fixes omission."
        ),
        "intended": (
            "Add _get_agent_knowledge reading task.agent.knowledge_sources content; "
            "include agent_knowledge line in _create_tasks_summary when present."
        ),
        "preserve": [
            "Tasks without knowledge_sources omit agent_knowledge line",
            "Core task description fields still present",
        ],
        "vplan": [
            "Stub task/agent with knowledge_sources; before summary lacks knowledge string; after includes it",
        ],
        "notes": "No LLM execution; summary builder only.",
    },
    "crewaiinc__crewai-2408": {
        "decision": "accept",
        "module": "prompts",
        "visible": "prompts/prompts.py",
        "orig": "src/crewai/utilities/prompts.py",
        "extract": lambda b: extract_class_methods(
            b / "src/crewai/utilities/prompts.py", "Prompts", ["_build_prompt"]
        ),
        "qualnames": ["crewai.utilities.prompts.Prompts._build_prompt"],
        "hypothesis": (
            "Before: custom branch triggers when either template set; system-only passes "
            "None prompt_template leading to AttributeError on replace. Hypothesis: require "
            "both templates OR fall back when either missing; guard response_template."
        ),
        "intended": (
            "Use if not system_template or not prompt_template for default fallback; only use "
            "custom concatenation when both provided; handle missing response_template."
        ),
        "preserve": [
            "Both templates absent still uses i18n slice concatenation",
            "Both templates provided still substitutes role goal backstory",
        ],
        "vplan": [
            "Agent stub with only system_template: before raises AttributeError; after returns non-empty prompt string",
        ],
        "notes": "Docs portion of patch excluded from plan scope.",
    },
    "crewaiinc__crewai-2495": {
        "decision": "reject",
        "module": None,
        "visible": None,
        "orig": "src/crewai/crew.py",
        "extract": None,
        "qualnames": [],
        "hypothesis": "",
        "intended": "",
        "preserve": [],
        "vplan": [],
        "notes": "Multi-file mem0 user_memory/docs plumbing.",
        "reject_reason": (
            "Cross-module memory_config/user_memory initialization plus docs; not separable "
            "single-module harness check without reimplementing Crew construction."
        ),
    },
    "crewaiinc__crewai-2552": {
        "decision": "reject",
        "module": None,
        "visible": None,
        "orig": "docs/concepts/llms.mdx",
        "extract": None,
        "qualnames": [],
        "hypothesis": "",
        "intended": "",
        "preserve": [],
        "vplan": [],
        "notes": "HF_TOKEN env rename and provider list constants/docs.",
        "reject_reason": (
            "Docs and CLI constants churn only; excluded as config/docs-only per task guidance."
        ),
    },
}


def main() -> None:
    raw_by_id: dict[str, dict] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["instance_id"] in INSTANCES:
            raw_by_id[rec["instance_id"]] = rec

    rows: list[dict] = []
    decisions: dict[str, dict] = {}
    per_case_dir = OUT / "curation_decisions"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    for iid in INSTANCES:
        spec = SPECS[iid]
        rec = raw_by_id[iid]
        before_dir = CASES / iid / "before"
        decision_doc: dict = {
            "source_id": iid,
            "repo": rec["repo"],
            "base_commit": rec["base_commit"],
            "decision": spec["decision"],
            "original_teacher_plan": False,
            "formal": False,
            "partition": PARTITION,
            "curation_notes": spec["notes"],
            "reconstruction_path": rel(CASES / iid / "reconstruction.json"),
            "original_patch_path": rel(CASES / iid / "original.patch"),
            "problem_statement_excerpt": rec.get("problem_statement", "")[:800],
        }
        if spec["decision"] == "reject":
            decision_doc["reject_reason"] = spec["reject_reason"]
            decision_doc["plan"] = {
                "action": "keep",
                "target_module": None,
                "locations": [],
                "evidence_refs": ["ev_reported_issue"],
                "problem_hypothesis": spec["reject_reason"],
                "intended_change": "No curated edit plan; case excluded from train examples.",
                "preserve_behaviors": [],
                "verification_plan": [],
                "verification_kind": None,
            }
        else:
            extract_fn: Callable[[Path], tuple[str, int, int]] = spec["extract"]
            excerpt, s, e = extract_fn(before_dir)
            src_path = before_dir / spec["orig"]
            plan = {
                "action": "edit",
                "target_module": spec["module"],
                "locations": [spec["visible"]],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": spec["hypothesis"],
                "intended_change": spec["intended"],
                "preserve_behaviors": spec["preserve"],
                "verification_plan": spec["vplan"],
                "verification_kind": None,
            }
            decision_doc["plan"] = plan
            issue_num = iid.split("-")[-1]
            row = {
                "source_id": iid,
                "split": "train",
                "verification_status": VERIFY_STATUS,
                "curation_notes": spec["notes"],
                "provenance": {
                    "partition": PARTITION,
                    "original_teacher_plan": False,
                    "formal": False,
                },
                "source_mapping": {
                    "dataset": "nebius/SWE-rebench-V2",
                    "repo": rec["repo"],
                    "base_commit": rec["base_commit"],
                    "original_source_path": spec["orig"],
                    "visible_path": spec["visible"],
                    "harness_module": spec["module"],
                },
                "source_refs": [
                    {
                        "qualname": q,
                        "original_path": spec["orig"],
                        "artifact_path": rel(src_path),
                        "sha256": sha256_path(src_path),
                        "start_line": s,
                        "end_line": e,
                    }
                    for q in spec["qualnames"]
                ],
                "state": {
                    "state_id": f"REBENCH-CREWAI-{issue_num}",
                    "root_lineage_id": f"crewAIInc/crewAI@{rec['base_commit'][:12]}",
                    "split": "train",
                    "plan_policy_version": POLICY,
                    "permitted_modules": PERMITTED,
                    "module_code": {spec["visible"]: excerpt},
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
                                f"Exact before excerpt {spec['orig']} lines {s}-{e} "
                                f"sha256={sha256_path(src_path)}:\n{excerpt}"
                            ),
                        },
                    ],
                    "verification_catalog": [],
                },
                "plan": plan,
            }
            rows.append(row)

        (per_case_dir / f"{iid}.json").write_text(
            json.dumps(decision_doc, indent=2), encoding="utf-8"
        )
        decisions[iid] = decision_doc

    (OUT / "curated_examples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (OUT / "curation_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (OUT / "batch_meta.json").write_text(
        json.dumps(
            {
                "batch": "crewai_batch",
                "repo": "crewAIInc/crewAI",
                "permitted_modules": PERMITTED,
                "plan_policy_version": POLICY,
                "accepted_count": len(rows),
                "total_cases": len(INSTANCES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} accepted rows, {len(decisions)} decisions")


if __name__ == "__main__":
    main()
