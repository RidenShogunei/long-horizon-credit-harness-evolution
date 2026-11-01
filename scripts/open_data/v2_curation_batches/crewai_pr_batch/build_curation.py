#!/usr/bin/env python3
"""Build curated_examples.json and curation_decisions.json for crewai_pr_batch."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BATCH = Path(__file__).resolve().parent
WORK = BATCH.parent
CASES = WORK / "additional_crewai/cases"
RAW = WORK / "additional_crewai/selected_raw_records.jsonl"
RECON = CASES

MODULE_ALIAS = {
    "prompts": "prompts",
    "agent_loop": "agent_loop",
    "planning": "planning",
    "memory": "memory",
    "tools": "tools",
}

ACCEPT_IDS = [
    "crewaiinc__crewai-1910",
    "crewaiinc__crewai-2424",
    "crewaiinc__crewai-2890",
    "crewaiinc__crewai-2865",
    "crewaiinc__crewai-1791",
    "crewaiinc__crewai-2604",
    "crewaiinc__crewai-2477",
]

CURATED_META: Dict[str, Dict[str, Any]] = {
    "crewaiinc__crewai-1910": {
        "target_module": "prompts",
        "visible_path": "prompts/converter.py",
        "original_path": "src/crewai/utilities/converter.py",
        "qualnames": ["crewai.utilities.converter.generate_model_description"],
        "curation_notes": "Scoped to generate_model_description PEP604/union handling in converter.py.",
        "plan": {
            "action": "edit",
            "target_module": "prompts",
            "locations": ["prompts/converter.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: describe_field only treats typing.Union optional forms and calls __name__ on all types. Reporter: PEP604 `int | None` and some annotations crash. Hypothesis: treat pipe-union args like Optional and guard types lacking __name__.",
            "intended_change": "Extend union detection for PEP604 args; optional branch handles single and multi non-None members; fall back to str(field_type) when __name__ absent.",
            "preserve_behaviors": [
                "typing.Optional fields still render as Optional[inner]",
                "Nested BaseModel fields still expand recursively",
            ],
            "verification_plan": [
                "Model with `int | None` field: before generate_model_description raises; after returns description containing opt field",
            ],
            "verification_kind": None,
        },
    },
    "crewaiinc__crewai-2424": {
        "target_module": "agent_loop",
        "visible_path": "agent_loop/crewai_event_bus.py",
        "original_path": "src/crewai/utilities/events/crewai_event_bus.py",
        "qualnames": ["crewai.utilities.events.crewai_event_bus.CrewAIEventsBus.emit"],
        "curation_notes": "emit() handler dispatch only in crewai_event_bus.py.",
        "plan": {
            "action": "edit",
            "target_module": "agent_loop",
            "locations": ["agent_loop/crewai_event_bus.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: emit registers handlers by exact event type only. Reporter: base-type listeners miss subclass events. Hypothesis: iterate registered types and invoke handlers when isinstance(event, registered_type).",
            "intended_change": "Replace exact type lookup with isinstance-based dispatch loop over handler map.",
            "preserve_behaviors": [
                "Handlers registered for concrete event type still run",
                "Blinker signal send still occurs after handler loop",
            ],
            "verification_plan": [
                "Subclass event: before only concrete handler; after both base and concrete handlers fire",
            ],
            "verification_kind": None,
        },
    },
    "crewaiinc__crewai-2890": {
        "target_module": "planning",
        "visible_path": "planning/crew_base.py",
        "original_path": "src/crewai/project/crew_base.py",
        "qualnames": ["crewai.project.crew_base.CrewBase"],
        "curation_notes": "CrewBase decorator sets _crew_name on wrapped class.",
        "plan": {
            "action": "edit",
            "target_module": "planning",
            "locations": ["planning/crew_base.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: CrewBase wrapper renames class but stores no logical crew identifier. Reporter: logging/debug needs original class name. Hypothesis: assign WrappedClass._crew_name = cls.__name__.",
            "intended_change": "After wrapper naming, set _crew_name to decorated class __name__.",
            "preserve_behaviors": [
                "Decorator still returns wrapped crew class with is_crew_class flag",
                "Wrapped __name__ still includes original class name in parentheses",
            ],
            "verification_plan": [
                "Before: decorated class lacks _crew_name; after: _crew_name equals original class name",
            ],
            "verification_kind": None,
        },
    },
    "crewaiinc__crewai-2865": {
        "target_module": "agent_loop",
        "visible_path": "agent_loop/task.py",
        "original_path": "src/crewai/task.py",
        "qualnames": ["crewai.task.Task.prompt"],
        "curation_notes": "Task.markdown field and prompt() instruction block only.",
        "plan": {
            "action": "edit",
            "target_module": "agent_loop",
            "locations": ["agent_loop/task.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: Task.prompt never adds Markdown formatting guidance. Reporter: users need optional Markdown output contract. Hypothesis: markdown bool field gates appended instruction block in prompt().",
            "intended_change": "Add markdown Optional field default False; when True append fixed Markdown syntax instruction slice to prompt parts.",
            "preserve_behaviors": [
                "markdown=False prompt matches prior description+expected_output structure",
                "i18n expected_output slice still included",
            ],
            "verification_plan": [
                "markdown=True: after prompt contains Markdown syntax block; before does not",
            ],
            "verification_kind": None,
        },
    },
    "crewaiinc__crewai-1791": {
        "target_module": "agent_loop",
        "visible_path": "agent_loop/task.py",
        "original_path": "src/crewai/task.py",
        "qualnames": [
            "crewai.task.Task.interpolate_only",
            "crewai.task.Task.interpolate_inputs",
        ],
        "curation_notes": "Safe interpolation for expected_output JSON braces; _save_file IOError wrap.",
        "plan": {
            "action": "edit",
            "target_module": "agent_loop",
            "locations": ["agent_loop/task.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: expected_output uses str.format and breaks on JSON braces. Reporter: templates with JSON snippets raise KeyError. Hypothesis: interpolate_only escapes braces except explicit placeholders; wrap file IO errors as RuntimeError.",
            "intended_change": "Route expected_output through interpolate_only; add try/except around _save_file raising RuntimeError on OSError/IOError.",
            "preserve_behaviors": [
                "Simple `{user}` placeholders still interpolate on before and after",
                "description interpolation behavior unchanged",
            ],
            "verification_plan": [
                "JSON braces + {user}: before KeyError; after preserves JSON and user",
            ],
            "verification_kind": None,
        },
    },
    "crewaiinc__crewai-2604": {
        "target_module": "memory",
        "visible_path": "memory/mem0_storage.py",
        "original_path": "src/crewai/memory/storage/mem0_storage.py",
        "qualnames": ["crewai.memory.storage.mem0_storage.Mem0Storage.search"],
        "curation_notes": "Mem0 API v1.1 response shape and add() params; pyproject pin auxiliary.",
        "plan": {
            "action": "edit",
            "target_module": "memory",
            "locations": ["memory/mem0_storage.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: search iterates dict return from mem0 v1.1; add always merges output_format. Reporter: mem0ai API break. Hypothesis: read results['results'] list; gate output_format to MemoryClient.add only.",
            "intended_change": "search filters results['results']; Memory.add path drops metadata/output_format params; client add sets output_format separately.",
            "preserve_behaviors": [
                "score_threshold filtering still applies to scored entries",
                "user_id still injected into search params when configured",
            ],
            "verification_plan": [
                "Mock v1.1 search dict: before TypeError/empty; after returns filtered ids",
            ],
            "verification_kind": None,
        },
    },
    "crewaiinc__crewai-2477": {
        "target_module": "tools",
        "visible_path": "tools/tool_usage_events.py",
        "original_path": "src/crewai/utilities/events/tool_usage_events.py",
        "qualnames": ["crewai.utilities.events.tool_usage_events.ToolUsageFinishedEvent"],
        "curation_notes": "ToolUsageFinishedEvent.output field; emit payload wired in tool_usage.py (same PR).",
        "plan": {
            "action": "edit",
            "target_module": "tools",
            "locations": ["tools/tool_usage_events.py"],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": "Current-code: finished event schema lacks tool result payload. Reporter: observers cannot read tool output. Hypothesis: add required output field on ToolUsageFinishedEvent populated by tool_usage emit path.",
            "intended_change": "Add output: Any on ToolUsageFinishedEvent; tool_usage includes result in event_data.",
            "preserve_behaviors": [
                "ToolUsageStartedEvent construction unchanged",
                "Finished event still carries timing and cache flags",
            ],
            "verification_plan": [
                "Before: Finished event validates without output; after: output required and accepted when provided",
            ],
            "verification_kind": None,
        },
    },
}

REJECT: Dict[str, str] = {
    "crewaiinc__crewai-1009": "Inseparable task.py + converter.py refactor for JSON control-character handling; cannot scope single module.",
    "crewaiinc__crewai-2428": "Inseparable multi-module placeholder interpolation (string_utils, formatter, task, base_agent).",
    "crewaiinc__crewai-1382": "Inseparable multi-module observable tool event plumbing (tool_usage + new event modules).",
    "crewaiinc__crewai-2370": "Inseparable multi-module security/fingerprint feature spanning crew, agent, task, docs.",
}

PENDING: Dict[str, str] = {
    "crewaiinc__crewai-2637": "Knowledge limits/threshold span knowledge.py, config, crew/agent wiring; bounded local verify not qualified without chromadb stack.",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        out[o["instance_id"]] = o
    return out


def find_span(path: Path, symbol: str) -> Tuple[int, int]:
    """Return 1-based inclusive line span for a top-level class or def name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == symbol:
            end = node.end_lineno or node.lineno
            return node.lineno, end
    raise ValueError(f"symbol {symbol} not found in {path}")


def extract_module_fragment(path: Path, qualname: str) -> Tuple[str, int, int]:
    parts = qualname.split(".")
    name = parts[-1]
    if len(parts) >= 2 and parts[-2][0].isupper():
        start, end = find_span(path, parts[-2])
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        chunk = "\n".join(lines[start - 1 : end]) + "\n"
        return chunk, start, end
    start, end = find_span(path, name)
    lines = path.read_text(encoding="utf-8").splitlines()
    chunk = "\n".join(lines[start - 1 : end]) + "\n"
    return chunk, start, end


def build_source_refs(iid: str, meta: dict) -> List[dict]:
    rel = meta["original_path"]
    before = CASES / iid / "before" / rel
    refs = []
    for qn in meta["qualnames"]:
        sym = qn.split(".")[-1]
        parent = qn.split(".")[-2]
        if sym[0].isupper() or parent[0].isupper():
            lookup = sym if sym[0].isupper() else parent
        else:
            lookup = sym
        start, end = find_span(before, lookup)
        refs.append(
            {
                "qualname": qn,
                "original_path": rel,
                "artifact_path": f"work/open_dataset_integration_v2/additional_crewai/cases/{iid}/before/{rel}",
                "sha256": sha256_file(before),
                "start_line": start,
                "end_line": end,
            }
        )
    return refs


def build_curated_row(iid: str, raw: dict, meta: dict) -> dict:
    rel = meta["original_path"]
    before = CASES / iid / "before" / rel
    fragment, _, _ = extract_module_fragment(before, meta["qualnames"][0])
    recon = json.loads((CASES / iid / "reconstruction.json").read_text(encoding="utf-8"))
    base = recon["base_commit"]
    ps = raw.get("problem_statement") or raw.get("pr_description") or ""
    excerpt = ps[:1200]
    sha = sha256_file(before)
    start_line = build_source_refs(iid, meta)[0]["start_line"]
    end_line = build_source_refs(iid, meta)[-1]["end_line"]
    return {
        "source_id": iid,
        "split": "train",
        "verification_status": "bounded_local_execution_pending",
        "source_license": raw.get("license"),
        "dataset_compilation_license": "CC-BY-4.0",
        "curation_notes": meta["curation_notes"],
        "provenance": {
            "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
            "original_teacher_plan": False,
            "formal": False,
        },
        "source_mapping": {
            "dataset": "nebius/SWE-rebench-V2",
            "repo": raw["repo"],
            "base_commit": base,
            "original_source_path": rel,
            "visible_path": meta["visible_path"],
            "harness_module": meta["target_module"],
        },
        "source_refs": build_source_refs(iid, meta),
        "state": {
            "state_id": f"REBENCH-CREWAI-{iid.split('-')[-1]}",
            "root_lineage_id": f"{raw['repo']}@{base[:12]}",
            "split": "train",
            "plan_policy_version": "proposal_single_plan_v6",
            "permitted_modules": [
                "tools",
                "agent_loop",
                "memory",
                "planning",
                "prompts",
            ],
            "module_code": {meta["visible_path"]: fragment},
            "history": [],
            "evidence": [
                {
                    "id": "ev_reported_issue",
                    "text": f"Published reporter issue (SWE-rebench-V2): {excerpt}",
                },
                {
                    "id": "ev_exact_source",
                    "text": f"Exact before excerpt {rel} lines {start_line}-{end_line} sha256={sha}:\n{fragment}",
                },
            ],
            "verification_catalog": [],
        },
        "plan": meta["plan"],
    }


def reject_plan(reason: str) -> dict:
    return {
        "action": "keep",
        "target_module": None,
        "locations": [],
        "evidence_refs": ["ev_reported_issue"],
        "problem_hypothesis": reason,
        "intended_change": "No curated edit plan; case excluded from train examples.",
        "preserve_behaviors": [],
        "verification_plan": [],
        "verification_kind": None,
    }


def main() -> None:
    raw_by_id = load_raw()
    curated: List[dict] = []
    decisions: Dict[str, dict] = {}

    for iid in sorted(raw_by_id.keys()):
        raw = raw_by_id[iid]
        recon_path = f"work/open_dataset_integration_v2/additional_crewai/cases/{iid}/reconstruction.json"
        patch_path = f"work/open_dataset_integration_v2/additional_crewai/cases/{iid}/original.patch"
        ps = raw.get("problem_statement") or ""
        if iid in ACCEPT_IDS:
            meta = CURATED_META[iid]
            row = build_curated_row(iid, raw, meta)
            curated.append(row)
            decisions[iid] = {
                "source_id": iid,
                "repo": raw["repo"],
                "base_commit": raw["base_commit"],
                "decision": "accept",
                "source_license": raw.get("license"),
                "dataset_compilation_license": "CC-BY-4.0",
                "original_teacher_plan": False,
                "formal": False,
                "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
                "curation_notes": meta["curation_notes"],
                "reconstruction_path": recon_path,
                "original_patch_path": patch_path,
                "problem_statement_excerpt": ps[:800],
                "plan": meta["plan"],
            }
        elif iid in REJECT:
            decisions[iid] = {
                "source_id": iid,
                "repo": raw["repo"],
                "base_commit": raw["base_commit"],
                "decision": "reject",
                "source_license": raw.get("license"),
                "dataset_compilation_license": "CC-BY-4.0",
                "original_teacher_plan": False,
                "formal": False,
                "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
                "curation_notes": REJECT[iid],
                "reconstruction_path": recon_path,
                "original_patch_path": patch_path,
                "problem_statement_excerpt": ps[:800],
                "reject_reason": REJECT[iid],
                "plan": reject_plan(REJECT[iid]),
            }
        elif iid in PENDING:
            decisions[iid] = {
                "source_id": iid,
                "repo": raw["repo"],
                "base_commit": raw["base_commit"],
                "decision": "pending",
                "source_license": raw.get("license"),
                "dataset_compilation_license": "CC-BY-4.0",
                "original_teacher_plan": False,
                "formal": False,
                "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
                "curation_notes": PENDING[iid],
                "reconstruction_path": recon_path,
                "original_patch_path": patch_path,
                "problem_statement_excerpt": ps[:800],
                "pending_reason": PENDING[iid],
                "plan": reject_plan(PENDING[iid]),
            }

    (BATCH / "curated_examples.json").write_text(
        json.dumps(curated, indent=2), encoding="utf-8"
    )
    (BATCH / "curation_decisions.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    print(f"curated={len(curated)} decisions={len(decisions)}")


if __name__ == "__main__":
    main()
