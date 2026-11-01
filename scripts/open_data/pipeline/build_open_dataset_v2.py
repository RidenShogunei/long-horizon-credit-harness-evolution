#!/usr/bin/env python3
"""Deterministic portable v2 open auxiliary dataset exporter (no training/inference)."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
V1 = WORK / "open_dataset_integration_v1" / "release_v1"
V2 = WORK / "open_dataset_integration_v2"
OUT = V2 / "release_v2"
SERVER_SRC = WORK / "server_fix" / "src"
PROMPT_PROFILES = WORK / "cursor_team_workspaces/C39_prompt_controls/approved_v2/prompt_profiles.json"
REBENCH_V1_REVISION = "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"
REBENCH_V2_DEFAULT_REVISION = "10483de0f50fe5da545942705a76c6150171af7f"
DATASET_ID = "research1-open-auxiliary-integrated-v2"
ROOT_DECISIONS_PATH = V2 / "root_final_decisions.json"

# Exporter-controlled extraction (curator JSON preserved separately under curator_drafts/).
SAMPLE_QUALNAMES: dict[str, list[str]] = {
    "stanfordnlp__dspy-8089": ["dspy.primitives.tool.Tool"],
    "stanfordnlp__dspy-8340": ["dspy.adapters.chat_adapter.ChatAdapter"],
    "stanfordnlp__dspy-8053": ["dspy.propose.grounded_proposer.GroundedProposer.propose_instructions_for_program"],
    "crewaiinc__crewai-2570": ["crewai.tools.base_tool.BaseTool"],
    "crewaiinc__crewai-3150": ["crewai.task.Task"],
    "crewaiinc__crewai-2588": ["crewai.memory.storage.mem0_storage.Mem0Storage"],
    "crewaiinc__crewai-1818": ["crewai.utilities.planning_handler.CrewPlanner"],
    "crewaiinc__crewai-2408": ["crewai.utilities.prompts.Prompts"],
    "berriai__litellm-11964": ["unpack_defs"],
    "berriai__litellm-9261": ["ollama_pt"],
    "berriai__litellm-5232": ["ModelResponseIterator"],
    "berriai__litellm-12168": ["validate_chat_completion_tool_choice"],
    "berriai__litellm-9828": [
        "_build_vertex_schema",
        "filter_schema_fields",
        "convert_anyof_null_to_nullable",
        "add_object_type",
    ],
    "berriai__litellm-8512": ["AmazonConverseConfig.get_supported_openai_params"],
    "crewaiinc__crewai-1791": ["crewai.task.Task"],
    "crewaiinc__crewai-1910": ["generate_model_description"],
    "crewaiinc__crewai-2424": ["crewai.utilities.events.crewai_event_bus.CrewAIEventsBus"],
    "crewaiinc__crewai-2890": ["crewai.project.crew_base.CrewBase"],
    "crewaiinc__crewai-2865": ["crewai.task.Task"],
}

SAMPLE_MODULE_ROLE: dict[str, dict[str, Any]] = {
    "berriai__litellm-9261": {
        "target_module": "context_mgmt",
        "permitted_modules": ["context_mgmt"],
        "visible_path": "context_mgmt/prompt_templates/factory.py",
        "curator_alias": "agent_runtime/prompt_templates/factory.py",
    },
    "berriai__litellm-5232": {
        "target_module": "observation",
        "permitted_modules": ["observation"],
        "visible_path": "observation/providers/anthropic_stream.py",
        "curator_alias": "agent_runtime/providers/anthropic_stream.py",
    },
    "berriai__litellm-9828": {
        "target_module": "tools",
        "permitted_modules": ["tools"],
        "visible_paths": [
            "tools/vertex/common_utils.py",
            "tools/vertex/vertex_gemini_config.py",
        ],
        "curator_alias": "agent_runtime/vertex/common_utils.py",
    },
    "crewaiinc__crewai-2424": {
        "target_module": "observation",
        "permitted_modules": ["observation"],
        "visible_path": "observation/crewai_event_bus.py",
        "curator_alias": "agent_loop/crewai_event_bus.py",
    },
    "crewaiinc__crewai-2890": {
        "target_module": "agent_loop",
        "permitted_modules": ["agent_loop"],
        "visible_path": "agent_loop/crew_base.py",
        "curator_alias": "planning/crew_base.py",
    },
}

SAMPLE_MULTI_FILE: dict[str, list[dict[str, Any]]] = {
    "berriai__litellm-9828": [
        {
            "artifact_path": "work/open_dataset_integration_v2/main_candidates/cases/berriai__litellm-9828/before/litellm/llms/vertex_ai/common_utils.py",
            "qualnames": [
                "_build_vertex_schema",
                "filter_schema_fields",
                "convert_anyof_null_to_nullable",
                "add_object_type",
            ],
            "visible_path": "tools/vertex/common_utils.py",
        },
        {
            "artifact_path": "work/open_dataset_integration_v2/main_candidates/cases/berriai__litellm-9828/before/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py",
            "qualnames": [
                "VertexGeminiConfig.apply_response_schema_transformation",
                "VertexGeminiConfig._map_response_schema",
            ],
            "visible_path": "tools/vertex/vertex_gemini_config.py",
        },
    ],
}

PLAN_PATCHES: dict[str, dict[str, Any]] = {
    "stanfordnlp__dspy-8089": {
        "problem_hypothesis": (
            "Source fact: Tool._parse_function (invoked from Tool.__init__) assigns args[k] from "
            "TypeAdapter(v).json_schema() or the literal string 'Any', then writes args[k]['default']. "
            "Reporter claim: untyped parameters with defaults raise TypeError because 'Any' is a str. "
            "Hypothesis: avoid literal 'Any' schema strings and gate __call__ validation on schema type."
        ),
        "intended_change": (
            "In _parse_function stop coercing falsy json_schema() to the literal 'Any' string; in __call__ "
            "run jsonschema validate only when args[k] is a dict and args[k].get('type') is not None and "
            "not equal to 'Any'."
        ),
    },
    "berriai__litellm-9828": {
        "locations": ["tools/vertex/common_utils.py", "tools/vertex/vertex_gemini_config.py"],
        "target_module": "tools",
        "intended_change": (
            "Thread add_property_ordering through _build_vertex_schema and ensure VertexGeminiConfig "
            "response-schema mapping passes ordering when building Vertex response_schema payloads."
        ),
    },
}

ALLOWLIST = frozenset(
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

PENDING_ROOT = frozenset(
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

REJECT_EXPORT = frozenset({"huggingface__smolagents-226", "crewaiinc__crewai-2604"})

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

# Curator-visible aliases mapped to Harness module names (semantic only; not TerminalBench paths).
MODULE_ALIAS = {"agent_runtime": "tools"}
POLICY_MODULES = frozenset({"tools", "agent_loop", "context_mgmt", "observation", "verification"})

CURATED_BATCHES = (
    ("dspy_strict", V2 / "dspy_strict" / "curated_examples.json"),
    ("crewai_strict", V2 / "crewai_strict" / "curated_examples.json"),
    ("litellm_strict", V2 / "litellm_strict" / "curated_examples.json"),
    ("crewai_pr_strict", V2 / "crewai_pr_strict" / "curated_examples.json"),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def assert_safe_relative(rel: str) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel or rel.startswith("/"):
        raise ValueError("Unsafe relative path: " + repr(rel))
    parsed = PurePosixPath(rel)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError("Unsafe relative path: " + repr(rel))
    return rel.replace("\\", "/")


def repo_key(repo: str) -> str:
    return repo.strip().lower().replace("_", "")


def split_for_repo(repo: str) -> str:
    if repo_key(repo) == "stanfordnlp/dspy":
        return "validation"
    return "train"


def usage_label(source_id: str, partition: str | None) -> str:
    if partition == "AGENT_INFRASTRUCTURE_AUXILIARY" or source_id == "stanfordnlp__dspy-8053":
        return "AGENT_INFRASTRUCTURE_AUXILIARY"
    return "AUXILIARY_REPORTED_ISSUE_PLANNING"


def resolve_module_name(name: str) -> str:
    return MODULE_ALIAS.get(name, name)


def remap_module_path(path: str) -> str:
    module, _, rest = path.partition("/")
    resolved = resolve_module_name(module)
    return f"{resolved}/{rest}" if rest else resolved


def normalize_harness_modules(state: dict[str, Any], plan: dict[str, Any]) -> None:
    state["permitted_modules"] = [
        resolve_module_name(m) for m in state.get("permitted_modules") or [] if resolve_module_name(m) in POLICY_MODULES
    ]
    if not state["permitted_modules"]:
        target = resolve_module_name(plan.get("target_module") or "")
        if target in POLICY_MODULES:
            state["permitted_modules"] = [target]
    if any(m not in POLICY_MODULES for m in state["permitted_modules"]):
        raise ValueError("Invalid permitted_modules after alias normalization")
    new_code: dict[str, str] = {}
    for path, code in (state.get("module_code") or {}).items():
        new_code[remap_module_path(path)] = code
    state["module_code"] = new_code
    if plan.get("target_module") is not None:
        plan["target_module"] = resolve_module_name(plan["target_module"])
    plan["locations"] = [remap_module_path(p) for p in plan.get("locations") or []]


def normalize_plan(raw: dict[str, Any]) -> dict[str, Any]:
    plan = {k: raw.get(k) for k in PLAN_FIELDS}
    kind = plan.get("verification_kind")
    if isinstance(kind, list) or kind == "":
        plan["verification_kind"] = None
    elif kind is not None and not isinstance(kind, str):
        plan["verification_kind"] = None
    if plan["action"] not in {"edit", "keep"}:
        raise ValueError("Invalid plan action")
    if plan["action"] == "edit" and plan.get("target_module") is None:
        raise ValueError("Edit plan missing target_module")
    return plan


def find_node_by_suffix(tree: ast.Module, qualname: str) -> ast.AST:
    parts = qualname.split(".")
    last_error: Exception | None = None
    for start in range(len(parts)):
        suffix = parts[start:]

        def walk(pool: list[ast.stmt], index: int) -> ast.AST | None:
            if index >= len(suffix):
                return None
            name = suffix[index]
            for node in pool:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                    if index == len(suffix) - 1:
                        return node
                    body = getattr(node, "body", None)
                    if isinstance(body, list):
                        found = walk(body, index + 1)
                        if found is not None:
                            return found
            return None

        node = walk(tree.body, 0)
        if node is not None:
            return node
        last_error = ValueError(f"No AST match for {qualname!r} (tried suffix {suffix!r})")
    raise last_error or ValueError(f"No AST match for {qualname!r}")


def resolve_ast_node(tree: ast.Module, qualname: str) -> ast.AST:
    try:
        return find_node_by_suffix(tree, qualname)
    except ValueError:
        parts = qualname.split(".")
        if len(parts) >= 2:
            parent = ".".join(parts[:-1])
            try:
                return find_node_by_suffix(tree, parent)
            except ValueError:
                pass
        raise


def extract_declaration(source_text: str, node: ast.AST) -> tuple[str, int, int]:
    lines = source_text.splitlines()
    if getattr(node, "decorator_list", None):
        start = min(d.lineno for d in node.decorator_list)
    else:
        start = node.lineno
    end = node.end_lineno or node.lineno
    segment = "\n".join(lines[start - 1 : end])
    if not segment.endswith("\n"):
        segment += "\n"
    return segment, start, end


def load_root_final_decisions() -> dict[str, Any]:
    if not ROOT_DECISIONS_PATH.is_file():
        return {"plan_overrides": {}, "exclude_ids": [], "source_scopes": {}}
    data = json.loads(ROOT_DECISIONS_PATH.read_text(encoding="utf-8"))
    data.setdefault("plan_overrides", {})
    data.setdefault("exclude_ids", [])
    data.setdefault("source_scopes", {})
    return data


def raw_dataset_meta(raw: dict[str, Any]) -> tuple[str, str | None, str | None]:
    src = raw.get("_source") or {}
    if isinstance(src, dict):
        dataset = src.get("dataset") or "nebius/SWE-rebench-V2"
        revision = src.get("revision")
        compilation = src.get("license_compilation")
        return dataset, revision, compilation
    return "nebius/SWE-rebench", None, None


def dedupe_declarations(pieces: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    pieces.sort(key=lambda p: (p[1] - p[0], -p[0]), reverse=True)
    kept: list[tuple[int, int, str, str]] = []
    for start, end, segment, qual in pieces:
        if any(start >= ks and end <= ke for ks, ke, _, _ in kept):
            continue
        kept.append((start, end, segment, qual))
    kept.sort(key=lambda p: p[0])
    return kept


def extract_qualnames_from_artifact(
    artifact: Path, qualnames: list[str], expected_sha: str | None, sid: str
) -> tuple[list[tuple[int, int, str, str]], list[dict[str, Any]]]:
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing artifact {artifact} for {sid}")
    actual_sha = sha256_file(artifact)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"SHA mismatch {sid} {artifact}: {actual_sha} != {expected_sha}")
    text = artifact.read_text(encoding="utf-8")
    tree = ast.parse(text)
    pieces: list[tuple[int, int, str, str]] = []
    prov: list[dict[str, Any]] = []
    for qual in qualnames:
        node = resolve_ast_node(tree, qual)
        segment, start, end = extract_declaration(text, node)
        pieces.append((start, end, segment, qual))
        prov.append(
            {
                "qualname": qual,
                "artifact_path": assert_safe_relative(str(artifact.relative_to(ROOT)).replace("\\", "/")),
                "file_sha256": actual_sha,
                "start_line": start,
                "end_line": end,
                "excerpt_sha256": sha256_text(segment),
                "bundle_source_path": None,
            }
        )
    merged = dedupe_declarations(pieces)
    return merged, prov


def rebuild_module_code(draft: dict[str, Any], root: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    sid = draft["source_id"]
    mapping = draft["source_mapping"]
    provenance_refs: list[dict[str, Any]] = []
    module_code: dict[str, str] = {}

    if sid in SAMPLE_MULTI_FILE:
        for spec in SAMPLE_MULTI_FILE[sid]:
            artifact = root / assert_safe_relative(spec["artifact_path"])
            ref_sha = None
            for ref in draft.get("source_refs") or []:
                if ref.get("artifact_path") == spec["artifact_path"]:
                    ref_sha = ref.get("sha256") or ref.get("file_sha256")
                    break
            merged, prov = extract_qualnames_from_artifact(
                artifact, spec["qualnames"], ref_sha, sid
            )
            vis = spec["visible_path"]
            module_code[vis] = "\n\n".join(seg.rstrip("\n") for _, _, seg, _ in merged) + "\n"
            provenance_refs.extend(prov)
        return module_code, provenance_refs

    qualnames = SAMPLE_QUALNAMES.get(sid)
    refs = draft.get("source_refs") or []
    if not refs:
        raise ValueError(f"Missing source_refs for {sid}")
    if qualnames:
        artifact = root / assert_safe_relative(refs[0]["artifact_path"])
        expected_sha = refs[0].get("sha256") or refs[0].get("file_sha256")
        merged, prov = extract_qualnames_from_artifact(artifact, qualnames, expected_sha, sid)
        role = SAMPLE_MODULE_ROLE.get(sid)
        if role and role.get("visible_path"):
            vis = role["visible_path"]
        else:
            vis = remap_module_path(mapping["visible_path"])
        module_code[vis] = "\n\n".join(seg.rstrip("\n") for _, _, seg, _ in merged) + "\n"
        provenance_refs.extend(prov)
        return module_code, provenance_refs

    vis_primary = remap_module_path(mapping["visible_path"])
    plan_locations = [remap_module_path(p) for p in draft["plan"].get("locations") or []]
    by_artifact: dict[str, list[tuple[int, int, str, str]]] = {}
    for ref in refs:
        qual = ref.get("qualname")
        if not qual:
            raise ValueError(f"Missing qualname in source_refs for {sid}")
        artifact = root / assert_safe_relative(ref["artifact_path"])
        expected_sha = ref.get("sha256") or ref.get("file_sha256")
        merged, _ = extract_qualnames_from_artifact(artifact, [qual], expected_sha, sid)
        rel = assert_safe_relative(str(artifact.relative_to(root)).replace("\\", "/"))
        by_artifact.setdefault(rel, []).extend(merged)
        provenance_refs.extend(
            extract_qualnames_from_artifact(artifact, [qual], expected_sha, sid)[1]
        )

    for rel, chunks in by_artifact.items():
        deduped = dedupe_declarations(chunks)
        if len(by_artifact) == 1:
            key = vis_primary
        else:
            basename = Path(rel).name
            candidates = [loc for loc in plan_locations if loc.endswith("/" + basename) or loc == basename]
            key = candidates[0] if len(candidates) == 1 else vis_primary
        module_code[key] = "\n\n".join(seg.rstrip("\n") for _, _, seg, _ in deduped) + "\n"
    return module_code, provenance_refs


def load_raw_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for rel in (
        "open_dataset_integration_v2/main_candidates/raw_records.jsonl",
        "open_dataset_integration_v2/additional_crewai/raw_records.jsonl",
    ):
        path = WORK / rel
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            iid = row["instance_id"]
            index[iid] = row
    return index


def load_curated_drafts() -> dict[str, dict[str, Any]]:
    drafts: dict[str, dict[str, Any]] = {}
    for batch_name, path in CURATED_BATCHES:
        for item in json.loads(path.read_text(encoding="utf-8")):
            sid = item["source_id"]
            item["_batch"] = batch_name
            drafts[sid] = item
    return drafts


def load_v1_smolagents_records() -> dict[str, dict[str, Any]]:
    path = V1 / "evolver_aux.train.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["sample_id"]] = row
    return out


def copy_file_stable(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_tree_files_stable(src_dir: Path, dest_dir: Path, *, skip_suffixes: frozenset[str] = frozenset({".pyc"})) -> None:
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
        for name in filenames:
            if any(name.endswith(s) for s in skip_suffixes):
                continue
            src = Path(dirpath) / name
            rel = src.relative_to(src_dir)
            copy_file_stable(src, dest_dir / rel)


def count_v2_reconstructed_cases() -> int:
    total = 0
    for root in (V2 / "main_candidates" / "cases", V2 / "additional_crewai" / "cases"):
        if not root.is_dir():
            continue
        for case in root.iterdir():
            if case.is_dir() and (case / "original.patch").is_file():
                total += 1
    return total


def bundle_v1_smol_portable(sid: str, portable: Path) -> None:
    src_tree = WORK / "open_dataset_integration_v1" / "rebench" / "sources" / sid
    if src_tree.is_dir():
        copy_tree_files_stable(src_tree, portable / sid / "before")
    raw_path = V1 / "swe_rebench.raw.jsonl"
    patch_text = None
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("instance_id") == sid or row.get("source_row_id") == sid:
            patch_text = row.get("patch")
            break
    if patch_text:
        (portable / sid / "original.patch").parent.mkdir(parents=True, exist_ok=True)
        (portable / sid / "original.patch").write_text(patch_text, encoding="utf-8")
    if sid == "huggingface__smolagents-478":
        dep = WORK / "open_dataset_integration_v1" / "rebench" / "dependencies" / sid / "src" / "smolagents" / "tools.py"
        if dep.is_file():
            copy_file_stable(dep, portable / sid / "read_only_deps" / "smolagents" / "tools.py")
    v1_prov = json.loads((V1 / "provenance.json").read_text(encoding="utf-8"))
    for item in v1_prov:
        if item.get("sample_id") == sid:
            (portable / sid / "v1_provenance_refs.json").write_text(
                json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            break


def case_dir_for(source_id: str) -> Path | None:
    main = V2 / "main_candidates" / "cases" / source_id
    if main.is_dir():
        return main
    extra = V2 / "additional_crewai" / "cases" / source_id
    if extra.is_dir():
        return extra
    return None


def align_plan_locations(plan: dict[str, Any], module_code: dict[str, str]) -> None:
    if plan.get("action") != "edit":
        return
    keys = list(module_code.keys())
    locs = plan.get("locations") or []
    if not keys:
        raise ValueError("Edit plan without module_code")
    if len(keys) == 1 and (not locs or locs[0] not in module_code):
        plan["locations"] = [keys[0]]
        return
    by_name = {Path(k).name: k for k in keys}
    fixed: list[str] = []
    for loc in locs:
        if loc in module_code:
            fixed.append(loc)
        elif Path(loc).name in by_name:
            fixed.append(by_name[Path(loc).name])
        else:
            raise ValueError(f"Plan location {loc!r} not in module_code keys {keys}")
    plan["locations"] = fixed


def apply_plan_patches(sid: str, plan: dict[str, Any], root_decisions: dict[str, Any]) -> None:
    for key, value in PLAN_PATCHES.get(sid, {}).items():
        plan[key] = value
    overrides = root_decisions.get("plan_overrides", {}).get(sid) or {}
    for key, value in overrides.items():
        if key in PLAN_FIELDS:
            plan[key] = value


def apply_module_role(sid: str, draft: dict[str, Any], plan: dict[str, Any], state: dict[str, Any]) -> str | None:
    role = SAMPLE_MODULE_ROLE.get(sid)
    if not role:
        return draft["source_mapping"].get("visible_path")
    plan["target_module"] = role["target_module"]
    plan["locations"] = role.get("visible_paths") or [role["visible_path"]]
    state["permitted_modules"] = list(role["permitted_modules"])
    return role.get("curator_alias")


def build_record_from_curated(
    draft: dict[str, Any],
    raw: dict[str, Any],
    profile_system: str,
    validate_plan,
    render_messages,
    root_decisions: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sid = draft["source_id"]
    state = copy.deepcopy(draft["state"])
    plan = normalize_plan(draft["plan"])
    apply_plan_patches(sid, plan, root_decisions)
    repo = draft["source_mapping"]["repo"]
    split = split_for_repo(repo)
    state["split"] = split
    module_code, prov_refs = rebuild_module_code(draft, ROOT)
    state["module_code"] = module_code
    curator_alias = apply_module_role(sid, draft, plan, state)
    align_plan_locations(plan, module_code)
    normalize_harness_modules(state, plan)
    ps = raw.get("problem_statement") or ""
    if not ps.strip():
        raise ValueError(f"Empty problem_statement for {sid}")
    state["evidence"] = [
        {
            "id": "ev_reported_issue",
            "text": (
                "Published reporter issue text (reporter assertions and examples, not a locally executed trajectory):\n"
                + ps
            ),
        },
        {
            "id": "ev_exact_source",
            "text": (
                "Current source excerpts in module_code are complete selected AST declarations. "
                "Unshown code is not evidence of global absence. Read-only dependencies below are "
                "current-source context. Source provenance is stored outside policy messages."
            ),
        },
    ]
    for dep in draft.get("dependency_source_refs") or []:
        note = dep.get("note") or "Read-only interface context from related pinned source; not a repair recommendation."
        state["evidence"].append(
            {
                "id": "ev_readonly_interface_context",
                "text": (
                    "Read-only framework interface context (not an observed task failure): "
                    + note
                ),
            }
        )
        if "ev_readonly_interface_context" not in plan["evidence_refs"]:
            plan["evidence_refs"] = list(plan["evidence_refs"]) + ["ev_readonly_interface_context"]
    validate_plan(plan, state)
    messages = render_messages(state, plan)
    messages[0]["content"] = profile_system
    partition = (draft.get("provenance") or {}).get("partition")
    dataset_name, dataset_revision, license_compilation = raw_dataset_meta(raw)
    if sid == "stanfordnlp__dspy-8053":
        partition = "AGENT_INFRASTRUCTURE_AUXILIARY"
    record = {
        "sample_id": sid,
        "source_dataset": dataset_name,
        "source_revision": dataset_revision or REBENCH_V2_DEFAULT_REVISION,
        "source_repo": repo,
        "source_base_commit": draft["source_mapping"]["base_commit"],
        "source_group": f"external/{repo_key(repo).split('/')[0]}/{repo.split('/')[-1]}",
        "split": split,
        "usage": usage_label(sid, partition),
        "original_teacher_plan": False,
        "formal_research1_sft": False,
        "messages": messages,
    }
    scope = root_decisions.get("source_scopes", {}).get(sid)
    prov = {
        "sample_id": sid,
        "source_license": raw.get("license"),
        "source_revision": dataset_revision,
        "source_dataset": dataset_name,
        "source_base_commit": raw.get("base_commit") or draft["source_mapping"]["base_commit"],
        "patch_sha256": sha256_text(raw.get("patch") or ""),
        "source_refs": prov_refs,
        "original_record_sha256": sha256_text(canonical_json(raw)),
        "original_patch_reconstructed": True,
        "local_behavior_execution_verified": None,
        "root_behavior_verified": False,
        "root_review_status": "awaiting_final_admission",
        "published_test_metadata_kept_outside_input": True,
        "license_compilation": license_compilation or draft.get("dataset_compilation_license"),
        "curator_batch": draft.get("_batch"),
        "curator_visible_alias": curator_alias,
        "root_source_scope": scope,
    }
    return record, prov


def build_acceptance_ledger(
    raw_index: dict[str, dict[str, Any]],
    curated: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    decision_map: dict[str, str] = {}
    for path in (
        V2 / "dspy_strict" / "curation_decisions.json",
        V2 / "litellm_strict" / "curation_decisions.json",
    ):
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("decisions") if isinstance(data, dict) else data
        for item in items:
            sid = item.get("source_row_id") or item.get("source_id")
            if not sid:
                continue
            disp = item.get("disposition") or item.get("decision")
            if disp == "include" or disp == "accept":
                decision_map[sid] = "curator_accept"
            elif disp == "exclude" or disp == "reject":
                decision_map[sid] = "curator_reject"

    pr_dec = json.loads((V2 / "crewai_pr_strict" / "curation_decisions.json").read_text(encoding="utf-8"))
    for sid, item in pr_dec.items():
        if item.get("decision") == "accept":
            decision_map[sid] = "curator_accept"
        elif item.get("decision") == "reject":
            decision_map[sid] = "curator_reject"

    crew_dec = json.loads((V2 / "crewai_strict" / "curation_decisions.json").read_text(encoding="utf-8"))
    if isinstance(crew_dec, dict):
        for sid, item in crew_dec.items():
            if isinstance(item, dict) and item.get("decision") == "accept":
                decision_map[sid] = "curator_accept"
            elif isinstance(item, dict) and item.get("decision") == "reject":
                decision_map[sid] = "curator_reject"

    ledger: list[dict[str, Any]] = []
    for iid, raw in sorted(raw_index.items()):
        if iid in REJECT_EXPORT:
            status = "reject"
        elif iid in PENDING_ROOT:
            status = "pending_root_review"
        elif iid in ALLOWLIST:
            status = "provisional_pass_awaiting_final_admission"
        elif decision_map.get(iid) == "curator_reject":
            status = "reject"
        elif decision_map.get(iid) == "curator_accept":
            status = "curator_accept_not_in_allowlist"
        elif iid in curated:
            status = "curator_draft_unlisted"
        else:
            status = "unprocessed"
        ledger.append(
            {
                "instance_id": iid,
                "repo": raw.get("repo"),
                "base_commit": raw.get("base_commit"),
                "patch_sha256": sha256_text(raw.get("patch") or ""),
                "export_status": status,
                "in_provisional_allowlist": iid in ALLOWLIST,
            }
        )
    for sid in sorted(REJECT_EXPORT):
        if sid not in raw_index and sid.startswith("huggingface__"):
            ledger.append(
                {
                    "instance_id": sid,
                    "repo": "huggingface/smolagents",
                    "base_commit": None,
                    "patch_sha256": None,
                    "export_status": "reject_revoked_v1",
                    "in_provisional_allowlist": False,
                }
            )
    return ledger


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root_decisions = load_root_final_decisions()

    sys.path.insert(0, str(SERVER_SRC))
    from research1.data_pipeline.policy import render_messages, validate_plan  # noqa: E402

    profile = json.loads(PROMPT_PROFILES.read_text(encoding="utf-8"))["P2"]
    raw_index = load_raw_index()
    curated = load_curated_drafts()
    v1_rows = load_v1_smolagents_records()
    excluded = set(root_decisions.get("exclude_ids") or [])

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []

    smol_v1_ids = [x for x in ALLOWLIST if x.startswith("huggingface__smolagents-")]
    for sid in smol_v1_ids:
        if sid in excluded:
            continue
        if sid not in v1_rows:
            raise SystemExit(f"Missing v1 train row for {sid}")
        rec = copy.deepcopy(v1_rows[sid])
        rec["source_revision"] = REBENCH_V1_REVISION
        rec["source_dataset"] = "nebius/SWE-rebench"
        if rec["split"] != "train":
            raise SystemExit(f"Unexpected split for {sid}")
        train_rows.append(rec)
        provenance.append(
            {
                "sample_id": sid,
                "note": "Reused v1 release messages/module_code (226 excluded)",
                "root_behavior_verified": False,
                "root_review_status": "awaiting_final_admission",
            }
        )
        reviews.append({"source_id": sid, "admitted": True, "batch": "v1_release_reuse"})

    v2_ids = sorted((ALLOWLIST - set(smol_v1_ids)) - excluded)
    for sid in v2_ids:
        if sid not in curated:
            raise SystemExit(f"Allowlisted sample missing curated draft: {sid}")
        if sid not in raw_index:
            raise SystemExit(f"Allowlisted sample missing raw record: {sid}")
        draft = curated[sid]
        if draft["plan"].get("action") != "edit":
            raise SystemExit(f"Non-edit plan in allowlist: {sid}")
        record, prov = build_record_from_curated(
            draft, raw_index[sid], profile, validate_plan, render_messages, root_decisions
        )
        if record["split"] == "validation":
            val_rows.append(record)
        else:
            train_rows.append(record)
        provenance.append(prov)
        reviews.append({"source_id": sid, "admitted": True, "batch": draft.get("_batch")})

    all_ids = {r["sample_id"] for r in train_rows + val_rows}
    if len(all_ids) != 23:
        raise SystemExit(f"Expected 23 exported samples, got {len(all_ids)}")
    if len(all_ids) != len(train_rows) + len(val_rows):
        raise SystemExit("Duplicate sample_id across splits")
    repos_split: dict[str, set[str]] = {}
    for row in train_rows + val_rows:
        repos_split.setdefault(repo_key(row["source_repo"]), set()).add(row["split"])
    for repo, splits in repos_split.items():
        if len(splits) > 1:
            raise SystemExit(f"Cross-split repo violation: {repo} -> {splits}")

    edit_train = [r for r in train_rows if json.loads(r["messages"][-1]["content"])["action"] == "edit"]
    edit_val = [r for r in val_rows if json.loads(r["messages"][-1]["content"])["action"] == "edit"]
    if len(edit_train) + len(edit_val) != len(all_ids):
        raise SystemExit("Keep rows present or non-edit assistant labels")
    if any(json.loads(r["messages"][-1]["content"])["action"] == "keep" for r in train_rows + val_rows):
        raise SystemExit("Keep action in exported set")

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    write_jsonl(OUT / "evolver_aux.train.jsonl", train_rows)
    write_jsonl(OUT / "evolver_aux.validation.jsonl", val_rows)
    write_jsonl(OUT / "evolver_aux.train.edit.jsonl", edit_train)
    write_jsonl(OUT / "evolver_aux.validation.edit.jsonl", edit_val)

    write_jsonl(OUT / "evolver_aux.train.keep.jsonl", [])
    write_jsonl(OUT / "evolver_aux.validation.keep.jsonl", [])

    ledger = build_acceptance_ledger(raw_index, curated)
    (OUT / "acceptance_ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    revocation = {
        "revoked_samples": [
            {
                "sample_id": "huggingface__smolagents-226",
                "reason": "Wildcard logger regression; v1 admission revoked for v2 export",
                "reversible": True,
            },
            {
                "sample_id": "crewaiinc__crewai-2604",
                "reason": "Root reject; not in provisional allowlist",
                "reversible": True,
            },
        ]
    }
    (OUT / "revocation_manifest.json").write_text(json.dumps(revocation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    drafts_dir = OUT / "curator_drafts"
    (drafts_dir / "accepted").mkdir(parents=True, exist_ok=True)
    (drafts_dir / "rejected").mkdir(parents=True, exist_ok=True)
    for sid in sorted(ALLOWLIST):
        if sid.startswith("huggingface__"):
            src = V1 / "curated_examples.original_cursor.json"
            payload = {"source_id": sid, "note": "v1 bundle draft; see original_cursor aggregate"}
        else:
            payload = curated[sid]
        (drafts_dir / "accepted" / f"{sid}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if "huggingface__smolagents-226" in v1_rows:
        (drafts_dir / "rejected" / "huggingface__smolagents-226.json").write_text(
            json.dumps(v1_rows.get("huggingface__smolagents-226") or {"sample_id": "huggingface__smolagents-226"}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    if "crewaiinc__crewai-2604" in curated:
        (drafts_dir / "rejected" / "crewaiinc__crewai-2604.json").write_text(
            json.dumps(curated["crewaiinc__crewai-2604"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    portable = OUT / "portable_sources"
    for sid in sorted(ALLOWLIST):
        if sid.startswith("huggingface__"):
            bundle_v1_smol_portable(sid, portable)
            excerpt = V1 / "sources" / sid
            if excerpt.is_dir():
                copy_tree_files_stable(excerpt, portable / sid / "release_v1_excerpt")
            continue
        case = case_dir_for(sid)
        if case is None:
            raise SystemExit(f"Missing case dir for {sid}")
        for sub in ("before", "applied", "original.patch"):
            src = case / sub
            if src.is_file():
                copy_file_stable(src, portable / sid / sub)
            elif src.is_dir():
                copy_tree_files_stable(src, portable / sid / sub)

    proofs = OUT / "portable_proofs"
    for batch in ("dspy_strict", "crewai_strict", "litellm_strict", "crewai_pr_strict"):
        batch_dir = V2 / batch
        script = batch_dir / "verify_batch.py"
        if script.is_file():
            copy_file_stable(script, proofs / batch / "verify_batch.py")
        fixtures = batch_dir / "fixtures"
        if fixtures.is_dir():
            copy_tree_files_stable(fixtures, proofs / batch / "fixtures")
    if (V1 / "outcomes" / "huggingface__smolagents-226.json").is_file():
        copy_file_stable(
            V1 / "outcomes" / "huggingface__smolagents-226.json",
            proofs / "revoked" / "huggingface__smolagents-226_outcome.json",
        )

    v1_raw_src = V1 / "swe_rebench.raw.jsonl"
    v2_main_raw = WORK / "open_dataset_integration_v2" / "main_candidates" / "raw_records.jsonl"
    v2_pr_raw = WORK / "open_dataset_integration_v2" / "additional_crewai" / "raw_records.jsonl"
    corpus_dir = OUT / "source_record_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    copy_file_stable(v1_raw_src, corpus_dir / "v1_swe_rebench.raw.jsonl")
    copy_file_stable(v2_main_raw, corpus_dir / "v2_main_candidates.raw.jsonl")
    copy_file_stable(v2_pr_raw, corpus_dir / "v2_additional_crewai.raw.jsonl")
    combined = corpus_dir / "combined_source_records_96.jsonl"
    combined.write_text(
        v1_raw_src.read_text(encoding="utf-8")
        + v2_main_raw.read_text(encoding="utf-8")
        + v2_pr_raw.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Portable raw rows for admitted samples only
    raw_out = OUT / "source_records"
    raw_out.mkdir(parents=True, exist_ok=True)
    for sid in sorted(ALLOWLIST):
        if sid in raw_index:
            (raw_out / f"{sid}.json").write_text(
                json.dumps(raw_index[sid], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "curation_review.json").write_text(json.dumps(reviews, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = """# research1 open auxiliary dataset v2

English: Integrated 23 auxiliary Plan demonstrations (provisional allowlist). Root bounded probes reported pass; formal admission and export verification flags remain false. Not formal research1 SFT; no gold patches in messages. C64 scope extras are separate from this bundle.

中文：23 条辅助 Plan（临时 allowlist）。根审 bounded probe 已通过，但 formal admission 与 export 验证标记仍为 false。非正式 research1 SFT；messages 不含 gold patch。C64 范围扩展与本包独立。

Counts: train={train_n}, validation={val_n}, edit-only, keep=0 (empty keep jsonl files present).
Source record corpus: 96 indexed rows (v1 14 + v2 82), not training-positive counts.
""".format(
        train_n=len(train_rows), val_n=len(val_rows)
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    source_cards = {
        "nebius/SWE-rebench": {
            "url": "https://huggingface.co/datasets/nebius/SWE-rebench",
            "revision_v1_bundle": REBENCH_V1_REVISION,
            "license_compilation": "CC-BY-4.0",
            "code_license_note": "Per-row license field in source JSONL (nullable); do not infer beyond recorded values.",
        },
        "nebius/SWE-rebench-V2": {
            "url": "https://huggingface.co/datasets/nebius/SWE-rebench-V2",
            "revision_default_in_v2_rows": REBENCH_V2_DEFAULT_REVISION,
            "license_compilation": "CC-BY-4.0",
        },
    }
    (OUT / "source_cards.json").write_text(json.dumps(source_cards, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    card = {
        "dataset_id": DATASET_ID,
        "english_summary": "Auxiliary reported-issue and agent-infrastructure planning corpus; 23 edit plans.",
        "chinese_summary": "辅助性已报告问题与 agent 基础设施规划语料，23 条 edit Plan。",
        "formal_training_ready": False,
        "terminal_bench_transfer_verified": False,
        "root_behavior_verified": False,
        "root_review_status": "awaiting_final_admission",
    }
    (OUT / "DATASET_CARD.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    file_hashes = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rel = assert_safe_relative(str(path.relative_to(OUT)).replace("\\", "/"))
            file_hashes[rel] = sha256_file(path)

    v1_raw_n = sum(1 for l in v1_raw_src.read_text(encoding="utf-8").splitlines() if l.strip())
    v2_recon = count_v2_reconstructed_cases()
    manifest = {
        "dataset_id": DATASET_ID,
        "exporter_script": "work/build_open_dataset_v2.py",
        "provisional_allowlist_count": len(ALLOWLIST),
        "exported_counts": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "total": len(all_ids),
            "edit": len(all_ids),
            "keep": 0,
        },
        "source_record_catalog": {
            "v1_swe_rebench_raw_rows": v1_raw_n,
            "v2_main_candidates_raw_rows": 41,
            "v2_additional_crewai_raw_rows": 41,
            "combined_indexed_rows": v1_raw_n + len(raw_index),
            "note": "96 = 14 v1 + 82 v2 published rows; broad raw corpus is not SFT-positive count",
        },
        "reconstruction_counts": {
            "v1_original_patch_reconstructed_ok": 13,
            "v1_original_patch_total_selected": 14,
            "v2_cases_with_original_patch": v2_recon,
            "v2_published_raw_rows": len(raw_index),
            "combined_reconstructed_ok_approx": 13 + 53,
            "note": "53/82 v2 strict reconstructions per curation scope; not equal to curator draft file count",
        },
        "admission_counts": {
            "provisional_export_rows": len(all_ids),
            "pending_root_review": len(PENDING_ROOT),
            "revoked_not_exported": len(REJECT_EXPORT),
        },
        "readiness": {
            "generic_schema_passed": True,
            "behavior_verified_locally_for_all_export_rows": False,
            "root_behavior_verified": False,
            "root_review_status": "awaiting_final_admission",
            "formal_training_ready": False,
            "terminal_bench_transfer_verified": False,
            "collection_complete": False,
        },
        "files": file_hashes,
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    train_path = OUT / "evolver_aux.train.jsonl"
    val_path = OUT / "evolver_aux.validation.jsonl"
    summary = {
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "train": len(train_rows),
        "validation": len(val_rows),
        "total": len(all_ids),
        "allowlist": len(ALLOWLIST),
        "ledger_rows": len(ledger),
        "raw_index_v2": len(raw_index),
        "source_record_corpus_rows": v1_raw_n + len(raw_index),
        "revoked": list(REJECT_EXPORT),
        "pending_root": sorted(PENDING_ROOT),
        "train_jsonl_bytes": train_path.stat().st_size if train_path.is_file() else 0,
        "validation_jsonl_bytes": val_path.stat().st_size if val_path.is_file() else 0,
        "root_behavior_verified": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
