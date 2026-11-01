#!/usr/bin/env python3
"""One-shot assembler: reads frozen rebench sources and writes curated_examples.json (root executes)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REBENCH = ROOT / "work" / "open_dataset_integration_v1" / "rebench"
OUT = Path(__file__).resolve().parent / "curated_examples.json"
POLICY = "proposal_single_plan_v6"
VERIFY = "dataset_published_test_evidence_and_local_patch_reconstruction_only"


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def slice_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end]) + "\n"


def extract_function(path: Path, name: str) -> tuple[str, int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = path.read_text(encoding="utf-8").splitlines()
            start = node.lineno
            end = node.end_lineno or start
            return "\n".join(lines[start - 1 : end]) + "\n", start, end
    raise SystemExit(f"function {name} not found in {path}")


def row(
    source_id: str,
    split: str,
    base_commit: str,
    orig_path: str,
    visible_path: str,
    module: str,
    source_file: Path,
    excerpt: str,
    start: int,
    end: int,
    problem_excerpt: str,
    plan: dict,
    notes: str,
) -> dict:
    return {
        "source_id": source_id,
        "split": split,
        "verification_status": VERIFY,
        "curation_notes": notes,
        "provenance": {
            "partition": "AUXILIARY_REPORTED_ISSUE_PLANNING",
            "original_teacher_plan": False,
            "formal": False,
        },
        "source_mapping": {
            "dataset": "nebius/SWE-rebench",
            "repo": "huggingface/smolagents",
            "base_commit": base_commit,
            "original_source_path": orig_path,
            "visible_path": visible_path,
            "harness_module": module,
        },
        "source_refs": [
            {
                "artifact_path": str(source_file.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_path(source_file),
                "start_line": start,
                "end_line": end,
            }
        ],
        "state": {
            "state_id": f"REBENCH-SMAG-{source_id.split('-')[-1]}",
            "root_lineage_id": f"huggingface/smolagents@{base_commit[:12]}",
            "split": split,
            "plan_policy_version": POLICY,
            "permitted_modules": [module],
            "module_code": {visible_path: excerpt},
            "history": [],
            "evidence": [
                {
                    "id": "ev_reported_issue",
                    "text": (
                        "Reporter claim (not locally reproduced here): "
                        + problem_excerpt.replace("\n", " ")[:1200]
                    ),
                },
                {
                    "id": "ev_exact_source",
                    "text": (
                        f"Exact before excerpt {orig_path} lines {start}-{end} "
                        f"sha256={sha256_path(source_file)}:\n{excerpt}"
                    ),
                },
            ],
            "verification_catalog": [],
        },
        "plan": plan,
    }


def main() -> None:
    rows = []

    p285 = REBENCH / "sources/huggingface__smolagents-285/src/smolagents/local_python_executor.py"
    ex285, s285, e285 = extract_function(p285, "evaluate_augassign")
    rows.append(
        row(
            "huggingface__smolagents-285",
            "train",
            "bd08d6485daad3a8eba8d22ca0c185c6653acf2c",
            "src/smolagents/local_python_executor.py",
            "tools/local_python_executor.py",
            "tools",
            p285,
            ex285,
            s285,
            e285,
            "local interpreter implements y += x as y = y + x, bypassing __iadd__ on objects like PuLP LpProblem",
            {
                "action": "edit",
                "target_module": "tools",
                "locations": ["tools/local_python_executor.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": "Current-code: evaluate_augassign always builds updated_value with binops like current_value + value_to_add. Reporter claim: inplace operators should invoke __i*__ on the target object; hypothesis: routing augassign through inplace dunder calls (with AttributeError fallback) would match CPython for objects like PuLP LpProblem.",
                "intended_change": "In evaluate_augassign, for each aug operator attempt current_value.__i*__(value_to_add) before falling back to binary operators, preserving set_value assignment of the result.",
                "preserve_behaviors": [
                    "List += still rejects adding non-list values",
                    "Augassign still updates state via set_value on the original target",
                    "Unsupported aug operators still raise InterpreterError",
                ],
                "verification_plan": [
                    "Construct local interpreter snippet using a stub class implementing only __iadd__ and confirm y += x mutates without calling __add__",
                    "Re-run list += path to ensure list concatenation behavior unchanged",
                ],
                "verification_kind": None,
            },
            "Gold patch (postdecision) edits only evaluate_augassign; patch_apply_check ok.",
        )
    )

    p891 = REBENCH / "sources/huggingface__smolagents-891/src/smolagents/local_python_executor.py"
    ex891, s891, e891 = extract_function(p891, "evaluate_ast")
    rows.append(
        row(
            "huggingface__smolagents-891",
            "train",
            "0329674dcb515e2f0130e6fb22b4f2cd5652038f",
            "src/smolagents/local_python_executor.py",
            "tools/local_python_executor.py",
            "tools",
            p891,
            ex891,
            s891,
            e891,
            "MAX_OPERATIONS guard works for nested loops in main code but not when inner loop is inside a user function",
            {
                "action": "edit",
                "target_module": "tools",
                "locations": ["tools/local_python_executor.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": "Current-code: evaluate_ast stores _operations_count as an int on state; nested function calls reuse the same counter but reporter claim says inner loops in functions bypass the cap. Hypothesis: storing the counter in a mutable container shared across evaluate_ast frames prevents reassignment from resetting the count.",
                "intended_change": "Represent _operations_count as a mutable mapping (e.g., dict with counter key) incremented in evaluate_ast and initialized in evaluate_python_code so nested calls share one limit.",
                "preserve_behaviors": [
                    "Top-level for-loop still raises InterpreterError once MAX_OPERATIONS exceeded",
                    "evaluate_python_code still resets counter at start of each execution",
                    "Other evaluate_ast dispatch branches unchanged",
                ],
                "verification_plan": [
                    "Execute nested-loop example with inner loop inside def foo and confirm InterpreterError at MAX_OPERATIONS",
                    "Confirm flat double loop still triggers the same error",
                ],
                "verification_kind": None,
            },
            "Gold patch changes _operations_count to dict counter in evaluate_ast and evaluate_python_code; patch_apply_check ok.",
        )
    )

    p1088 = REBENCH / "sources/huggingface__smolagents-1088/src/smolagents/_function_type_hints_utils.py"
    ex1088 = slice_lines(p1088, 221, 267)
    rows.append(
        row(
            "huggingface__smolagents-1088",
            "train",
            "3240fa878ac288a217d62fcf516241ac54ab4184",
            "src/smolagents/_function_type_hints_utils.py",
            "tools/_function_type_hints_utils.py",
            "tools",
            p1088,
            ex1088,
            221,
            267,
            "DocstringParsingException when Args use typed forms like name (str): desc or next arg also typed",
            {
                "action": "edit",
                "target_module": "tools",
                "locations": ["tools/_function_type_hints_utils.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": "Current-code: args_split_re stops at next arg only when line matches word+: without allowing parenthesized types; Returns regex includes type text in capture. Reporter claim: typed Google docstrings fail parsing. Hypothesis: relax args_split_re lookahead and strip return type prefix in returns_re fixes schema extraction.",
                "intended_change": "Update args_split_re to treat typed argument headers as boundaries and adjust returns_re to capture description after optional type prefix.",
                "preserve_behaviors": [
                    "Untyped Google Args blocks still populate args_dict",
                    "Functions without Returns still parse description-only",
                    "Missing arg docstrings still surface DocstringParsingException downstream",
                ],
                "verification_plan": [
                    "Parse sample docstring with book_title (str): description and ensure args_dict contains book_title",
                    "Parse multi-arg typed docstring where next arg includes (str) in header",
                ],
                "verification_kind": None,
            },
            "Gold patch edits args_split_re and returns_re only in _function_type_hints_utils.py; patch_apply_check ok.",
        )
    )

    p877 = REBENCH / "sources/huggingface__smolagents-877/src/smolagents/local_python_executor.py"
    ex877, s877, e877 = extract_function(p877, "get_safe_module")
    rows.append(
        row(
            "huggingface__smolagents-877",
            "train",
            "ff7ba5117fa6329af1f7b21e6c174320c729bfb4",
            "src/smolagents/local_python_executor.py",
            "tools/local_python_executor.py",
            "tools",
            p877,
            ex877,
            s877,
            e877,
            "FastAPI wrapper sees thousands of INFO logs Skipping dangerous attribute when importing pandas",
            {
                "action": "edit",
                "target_module": "tools",
                "locations": ["tools/local_python_executor.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": "Current-code: get_safe_module loops dir(raw_module) and logs logger.info for each attribute skipped by DANGEROUS_PATTERNS substring match on module path segments (reporter cites pandas.io.formats paths matching io). Hypothesis: removing the per-attribute skip/log loop during safe module copying eliminates INFO spam during authorized imports.",
                "intended_change": "Stop iterating attributes solely to log and skip dangerous-pattern matches at INFO during get_safe_module attribute copy (gold removes that block).",
                "preserve_behaviors": [
                    "ImportError/AttributeError during getattr still logged and skipped",
                    "Recursive get_safe_module on nested ModuleType attributes preserved",
                    "Unauthorized imports still raise InterpreterError in import_modules",
                ],
                "verification_plan": [
                    "Import pandas under authorized_imports and compare count of INFO lines containing Skipping dangerous attribute before vs after",
                    "Ensure authorized import of safe modules still succeeds",
                ],
                "verification_kind": None,
            },
            "Gold patch removes dangerous-attribute INFO skip loop in get_safe_module; patch_apply_check ok.",
        )
    )

    p226 = REBENCH / "sources/huggingface__smolagents-226/src/smolagents/agents.py"
    tree = ast.parse(p226.read_text(encoding="utf-8"))
    ex226, s226, e226 = "", 0, 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CodeAgent":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    lines = p226.read_text(encoding="utf-8").splitlines()
                    ex226 = "\n".join(lines[item.lineno - 1 : item.end_lineno]) + "\n"
                    s226, e226 = item.lineno, item.end_lineno
                    break
            break
    rows.append(
        row(
            "huggingface__smolagents-226",
            "train",
            "a4ec1e5be3fb553b45af5f38f134f7be26d3a274",
            "src/smolagents/agents.py",
            "agent_loop/agents.py",
            "agent_loop",
            p226,
            ex226,
            s226,
            e226,
            "CodeAgent system prompt missing authorized_imports after run calls initialize_system_prompt again",
            {
                "action": "edit",
                "target_module": "agent_loop",
                "locations": ["agent_loop/agents.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": "Current-code: CodeAgent.__init__ replaces {{authorized_imports}} before super().__init__, but MultiStepAgent.run calls initialize_system_prompt() which rebuilds prompt from template without reapplying authorized_imports substitution. Reporter claim: run-time prompt lacks import list.",
                "intended_change": "Override CodeAgent.initialize_system_prompt to call super then replace {{authorized_imports}} with str(self.authorized_imports) or wildcard message, deferring substitution from __init__.",
                "preserve_behaviors": [
                    "Missing {{authorized_imports}} tag still raises AgentError at construction",
                    "super().__init__ still registers tools and managed agents templates",
                    "authorized_imports list still merges BASE_BUILTIN_MODULES with additional_authorized_imports",
                ],
                "verification_plan": [
                    "Instantiate CodeAgent with restricted additional_authorized_imports, call run reset path, inspect system prompt text for authorized module list",
                    "Confirm initialize_system_prompt still includes tool descriptions after substitution",
                ],
                "verification_kind": None,
            },
            "Gold patch moves authorized_imports replacement into CodeAgent.initialize_system_prompt; patch_apply_check ok despite other files in merge.",
        )
    )

    p478 = REBENCH / "sources/huggingface__smolagents-478/src/smolagents/default_tools.py"
    tree = ast.parse(p478.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SpeechToTextTool":
            lines = p478.read_text(encoding="utf-8").splitlines()
            ex478 = "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"
            s478, e478 = node.lineno, node.end_lineno
            break
    rows.append(
        row(
            "huggingface__smolagents-478",
            "train",
            "39908439d29d3004ec59313be2fd89f43f06b485",
            "src/smolagents/default_tools.py",
            "tools/default_tools.py",
            "tools",
            p478,
            ex478,
            s478,
            e478,
            "SpeechToTextTool() fails TypeError object.__new__(): not enough arguments",
            {
                "action": "edit",
                "target_module": "tools",
                "locations": ["tools/default_tools.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": "Current-code: SpeechToTextTool.__new__ calls return super().__new__() without forwarding cls/args to PipelineTool.__new__. Reporter stack trace matches TypeError on object.__new__.",
                "intended_change": "Define __new__(cls, *args, **kwargs), set Whisper processor/model classes on cls, and return super().__new__(cls, *args, **kwargs).",
                "preserve_behaviors": [
                    "encode/decode/forward methods unchanged",
                    "Tool metadata name transcriber and audio input schema unchanged",
                    "Other default tools instantiation unchanged",
                ],
                "verification_plan": [
                    "Instantiate SpeechToTextTool() without TypeError in isolated import",
                    "Smoke-call encode/decode with mocked processor/model classes",
                ],
                "verification_kind": None,
            },
            "Gold patch fixes SpeechToTextTool.__new__ signature; patch_apply_check ok.",
        )
    )

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
