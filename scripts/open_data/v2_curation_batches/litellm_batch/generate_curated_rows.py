"""Offline generator for litellm_batch curated_examples.json and curation_decisions.json."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
MC = REPO_ROOT / "work/open_dataset_integration_v2/main_candidates"
BATCH = REPO_ROOT / "work/open_dataset_integration_v2/litellm_batch"
MODULE = "agent_runtime"

ACCEPTED: dict[str, dict[str, Any]] = {
    "berriai__litellm-11964": {
        "rel": "litellm/litellm_core_utils/prompt_templates/common_utils.py",
        "visible": "agent_runtime/prompt_templates/common_utils.py",
        "qualname": "unpack_defs",
        "target_module": MODULE,
        "problem": "Reporter claim: unpack_defs only inlines $ref under properties/anyOf/items and misses nested anyOf/items $ref paths used for structured tool schemas.",
        "change": "Replace unpack_defs with a depth-first walker that merges $defs/definitions and resolves every $ref in-place across dict/list subtrees.",
        "preserve": [
            "Schemas without $ref remain unchanged aside from normal dict traversal",
            "Unknown $ref names are left untouched rather than raising",
        ],
        "verify": [
            "Nested anyOf $ref resolves to inlined subschema on after tree",
            "Property $ref under properties still inlines as before",
        ],
        "notes": "Single-file JSON-schema $ref inlining for agent tool/response_format plumbing.",
    },
    "berriai__litellm-9261": {
        "rel": "litellm/litellm_core_utils/prompt_templates/factory.py",
        "visible": "agent_runtime/prompt_templates/factory.py",
        "qualname": "ollama_pt",
        "target_module": MODULE,
        "problem": "Reporter claim: ollama_pt prompt builder skips consecutive system-role messages after user blocks, so system instructions never reach the Ollama prompt template.",
        "change": "After user content merge, merge consecutive system messages into a ### System block before assistant merging.",
        "preserve": [
            "User and assistant merge loops still advance msg_i monotonically",
            "BadMessageError still raised when msg_i stalls",
        ],
        "verify": [
            "Two consecutive system messages after user produce one ### System section on after",
            "Before omits system section for the same message list",
        ],
        "notes": "Chat-role formatting for local/agent runtimes using Ollama prompt template.",
    },
    "berriai__litellm-5232": {
        "rel": "litellm/llms/anthropic.py",
        "visible": "agent_runtime/providers/anthropic_stream.py",
        "qualname": "ModelResponseIterator",
        "target_module": MODULE,
        "problem": "Reporter claim: streaming Anthropic tool-call deltas reuse provider content_block index so parallel tool calls get wrong OpenAI-style tool index.",
        "change": "Track self.tool_index on content_block_start for tool_use and emit that stable index in streamed tool_calls chunks.",
        "preserve": [
            "Non-tool streaming chunks still map through existing delta handler",
            "empty tool-call arg guard behavior unchanged",
        ],
        "verify": [
            "After iterator assigns monotonic tool_index starting at 0 on tool_use start",
            "Before leaves default -1 and passes provider index through",
        ],
        "notes": "Streaming tool index mapping for agent function-calling interop.",
    },
    "berriai__litellm-12168": {
        "rel": "litellm/utils.py",
        "visible": "agent_runtime/utils.py",
        "qualname": "validate_chat_completion_tool_choice",
        "target_module": MODULE,
        "problem": "Reporter claim: validate_chat_completion_tool_choice rejects Cursor-style dict tool_choice {\"type\":\"auto\"} lacking function field.",
        "change": "Accept dict tool_choice when type is auto/none/required and no function key; keep strict OpenAI function dict validation otherwise.",
        "preserve": [
            "String tool_choice values still pass through unchanged",
            "Invalid function dicts still raise with OpenAI-spec error",
        ],
        "verify": [
            "Before raises on {\"type\":\"auto\"}",
            "After returns same dict unchanged",
        ],
        "notes": "Scoped to utils.py validation helper; published patch also contains docs-only Slack links excluded from plan module.",
        "scope_patch": "scope_patches/berriai__litellm-12168_utils_only.patch",
    },
    "berriai__litellm-9828": {
        "rel": "litellm/llms/vertex_ai/common_utils.py",
        "visible": "agent_runtime/vertex/common_utils.py",
        "qualname": "_build_vertex_schema",
        "target_module": MODULE,
        "problem": "Reporter claim: Vertex structured-output schema builder drops JSON key order so propertyOrdering never reaches Gemini/Vertex requests.",
        "change": "Thread add_property_ordering through _build_vertex_schema and set_schema_property_ordering when building tool/response schemas.",
        "preserve": [
            "Existing unpack_defs flattening of $defs still runs before field filtering",
            "Invalid schema fields still stripped via Schema TypedDict hints",
        ],
        "verify": [
            "After with add_property_ordering=True adds propertyOrdering list matching property declaration order",
            "Before leaves parameters without propertyOrdering",
        ],
        "notes": "JSON schema key ordering for structured agent outputs on Vertex.",
    },
    "berriai__litellm-11006": {
        "rel": "litellm/litellm_core_utils/streaming_handler.py",
        "visible": "agent_runtime/streaming_handler.py",
        "qualname": "CustomStreamWrapper.set_model_id",
        "target_module": MODULE,
        "problem": "Reporter claim: first streaming chunk may carry empty id so response_id stays None and later chunks with valid OpenRouter ids are ignored.",
        "change": "Prefer first non-empty string id from any chunk while still backfilling when response_id unset.",
        "preserve": [
            "model_response.id still mirrors stored response_id when set",
            "Empty/whitespace ids do not overwrite an existing valid response_id",
        ],
        "verify": [
            "After adopts id from second chunk when first chunk id blank",
            "Before keeps None after blank-first sequence",
        ],
        "notes": "Streamed response id preservation for agent observability correlation.",
    },
    "berriai__litellm-6138": {
        "rel": "litellm/llms/AzureOpenAI/azure.py",
        "visible": "agent_runtime/providers/azure_openai.py",
        "qualname": "AzureChatCompletion.completion",
        "target_module": MODULE,
        "problem": "Reporter claim: Azure chat path pops max_retries from optional_params so caller cannot override retry count forwarded to the SDK.",
        "change": "Read max_retries with get (not pop) and list max_retries in supported openai params; reinsert into optional_params when provided.",
        "preserve": [
            "Default max_retries remains 2 when key absent",
            "Other optional_params popping behavior unchanged",
        ],
        "verify": [
            "After leaves max_retries key in optional_params when set to 5",
            "Before removes max_retries from optional_params after read",
        ],
        "notes": "Explicit max_retries forwarding for Azure-hosted agent backends.",
    },
    "berriai__litellm-8512": {
        "rel": "litellm/llms/bedrock/chat/converse_transformation.py",
        "visible": "agent_runtime/providers/bedrock_converse.py",
        "qualname": "AmazonConverseConfig.get_supported_openai_params",
        "target_module": MODULE,
        "problem": "Reporter claim: Bedrock Converse tool-calling support list omits meta.llama3-3 models so tools param dropped for Llama 3.3 agents.",
        "change": "Treat base_model.startswith('meta.llama3-3') like other tool-capable Bedrock model families.",
        "preserve": [
            "Non-tool models still omit tools from supported params",
            "tool_choice gating via supports_tool_choice unchanged",
        ],
        "verify": [
            "After includes tools for meta.llama3-3-70b-instruct-v1:0",
            "Before omits tools for same model id",
        ],
        "notes": "Tool-call capability detection for Bedrock Llama 3.3 agent models.",
    },
    "berriai__litellm-6563": {
        "rel": "litellm/integrations/langfuse/langfuse.py",
        "visible": "agent_runtime/observability/langfuse.py",
        "qualname": "LangFuseLogger._prepare_metadata",
        "target_module": MODULE,
        "problem": "Reporter claim: Langfuse logging deepcopy on metadata fails when values include non-copyable objects like threading.RLock.",
        "change": "Add _prepare_metadata using MutableMapping/Sequence checks with per-key deepcopy fallback instead of whole-dict deepcopy.",
        "preserve": [
            "Plain dict/str metadata still round-trips via deepcopy",
            "BaseModel values still serialized with model_dump",
        ],
        "verify": [
            "After returns serializable dict when nested dict contains threading.RLock",
            "Before deepcopy raises TypeError on same structure",
        ],
        "notes": "Observation/metadata serialization for agent tracing integrations.",
    },
}

EXCLUDE: dict[str, str] = {
    "berriai__litellm-4760": "exclude_price_constant_update",
    "berriai__litellm-10697": "exclude_billing_db_migration_unrelated",
    "berriai__litellm-10959": "exclude_guardrail_registry_broad_feature",
    "berriai__litellm-12551": "exclude_new_provider_docs_feature",
    "berriai__litellm-6265": "exclude_mixed_proxy_docs_and_broad_multi_file",
    "berriai__litellm-9419": "exclude_responses_api_broad_multi_module",
    "berriai__litellm-5025": "exclude_stream_chunk_builder_broad",
    "berriai__litellm-3470": "exclude_sync_wrapper_embedding_not_bounded_plan",
    "berriai__litellm-4716": "exclude_token_counting_feature_broad",
    "berriai__litellm-1200": "exclude_signature_refactor_not_agent_runtime",
    "berriai__litellm-12003": "exclude_nim_provider_broad_multi_file",
    "berriai__litellm-10633": "pending_bounded_verify_deferred_proxy_flag",
    "berriai__litellm-8131": "pending_top_k_bedrock_needs_heavy_imports",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_symbol(source: str, qualname: str) -> tuple[int, int, str]:
    parts = qualname.split(".")
    mod = ast.parse(source)
    lines = source.splitlines(True)
    if len(parts) == 1:
        for node in mod.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                return node.lineno, node.end_lineno or node.lineno, "".join(
                    lines[node.lineno - 1 : node.end_lineno]
                )
    if len(parts) == 2:
        cls_name, meth = parts
        for node in mod.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for n in node.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == meth:
                        return n.lineno, n.end_lineno or n.lineno, "".join(
                            lines[n.lineno - 1 : n.end_lineno]
                        )
    raise ValueError(f"symbol not found: {qualname}")


def load_rebench_meta(source_id: str) -> dict[str, Any]:
    raw = MC / "raw_records.jsonl"
    for line in raw.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["instance_id"] == source_id:
            return row
    raise KeyError(source_id)


def build_row(source_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    recon = json.loads((MC / "cases" / source_id / "reconstruction.json").read_text(encoding="utf-8"))
    rel = spec["rel"]
    before_path = REPO_ROOT / recon["source_files"][rel]["before"]
    before_sha = recon["source_files"][rel]["before_sha256"]
    base_commit = recon["base_commit"]
    raw = load_rebench_meta(source_id)
    ps = raw.get("problem_statement", "").strip()
    first_line = ps.split("\n", 1)[0] if ps else source_id

    src = before_path.read_text(encoding="utf-8")
    qualname = spec["qualname"]
    if qualname == "AmazonConverseConfig.get_supported_openai_params":
        start, end, code = extract_symbol(src, "AmazonConverseConfig.get_supported_openai_params")
    elif qualname == "CustomStreamWrapper.set_model_id":
        start, end, code = extract_symbol(src, "CustomStreamWrapper.set_model_id")
    elif qualname == "LangFuseLogger._prepare_metadata":
        start, end, code = extract_symbol(src, "LangFuseLogger._log_langfuse_v2")
        qualname = "LangFuseLogger._log_langfuse_v2 (pre-patch metadata copy path)"
    elif qualname == "AzureChatCompletion.completion":
        lines = src.splitlines(True)
        start, end = 580, 610
        code = "".join(lines[start - 1 : end])
    elif qualname == "ModelResponseIterator":
        mod = ast.parse(src)
        for node in mod.body:
            if isinstance(node, ast.ClassDef) and node.name == "ModelResponseIterator":
                lines = src.splitlines(True)
                start, end = node.lineno, min(node.end_lineno or node.lineno + 80, node.lineno + 80)
                code = "".join(lines[start - 1 : end])
                break
        else:
            raise ValueError("ModelResponseIterator missing")
    else:
        sym = qualname.split(".")[-1]
        start, end, code = extract_symbol(src, sym)

    artifact = str(before_path.relative_to(REPO_ROOT)).replace("\\", "/")
    ev_issue = {
        "id": "ev_reported_issue",
        "text": f"Reporter claim (not executed trajectory): {first_line}",
    }
    ev_src = {
        "id": "ev_exact_source",
        "text": (
            f"Exact before excerpt {rel} qualname={qualname} lines {start}-{end} "
            f"sha256={before_sha}:\n{code}"
        ),
    }

    row = {
        "source_id": source_id,
        "split": "train",
        "verification_status": "local_bounded_behavior_verification_via_verify_batch",
        "curation_notes": spec["notes"],
        "provenance": {
            "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
            "original_teacher_plan": False,
            "formal": False,
        },
        "source_mapping": {
            "dataset": "nebius/SWE-rebench",
            "repo": "BerriAI/litellm",
            "base_commit": base_commit,
            "original_source_path": rel,
            "visible_path": spec["visible"],
            "harness_module": MODULE,
        },
        "source_refs": [
            {
                "artifact_path": artifact,
                "sha256": before_sha,
                "qualname": qualname,
                "start_line": start,
                "end_line": end,
            }
        ],
        "state": {
            "state_id": f"REBENCH-LITELLM-{source_id.split('-')[-1]}",
            "root_lineage_id": f"BerriAI/litellm@{base_commit[:12]}",
            "split": "train",
            "plan_policy_version": "proposal_single_plan_v6",
            "permitted_modules": [MODULE],
            "module_code": {spec["visible"]: code},
            "history": [],
            "evidence": [ev_issue, ev_src],
            "verification_catalog": [],
        },
        "plan": {
            "action": "edit",
            "target_module": spec["target_module"],
            "locations": [spec["visible"]],
            "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
            "problem_hypothesis": spec["problem"],
            "intended_change": spec["change"],
            "preserve_behaviors": spec["preserve"],
            "verification_plan": spec["verify"],
            "verification_kind": None,
        },
    }
    if spec.get("scope_patch"):
        row["scope_patch"] = spec["scope_patch"]
    return row


def all_litellm_ids() -> list[str]:
    recs = json.loads((MC / "reconstruction_results.json").read_text(encoding="utf-8"))
    return sorted(r["source_id"] for r in recs if r["source_id"].startswith("berriai__litellm"))


def main() -> None:
    BATCH.mkdir(parents=True, exist_ok=True)
    (BATCH / "scope_patches").mkdir(exist_ok=True)

    # utils-only subset patch for 12168
    patch_src = (MC / "cases/berriai__litellm-12168/original.patch").read_text(encoding="utf-8")
    utils_chunk = []
    keep = False
    for line in patch_src.splitlines(keepends=True):
        if line.startswith("diff --git a/litellm/utils.py"):
            keep = True
        elif line.startswith("diff --git") and keep:
            break
        if keep:
            utils_chunk.append(line)
    (BATCH / "scope_patches/berriai__litellm-12168_utils_only.patch").write_text(
        "".join(utils_chunk), encoding="utf-8"
    )

    curated = []
    for sid, spec in ACCEPTED.items():
        curated.append(build_row(sid, spec))

    decisions = []
    for sid in all_litellm_ids():
        if sid in ACCEPTED:
            decisions.append(
                {
                    "source_id": sid,
                    "decision": "accept",
                    "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
                    "reason": ACCEPTED[sid]["notes"],
                }
            )
        elif sid in EXCLUDE:
            decisions.append(
                {
                    "source_id": sid,
                    "decision": "reject",
                    "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
                    "reason": EXCLUDE[sid],
                }
            )
        else:
            decisions.append(
                {
                    "source_id": sid,
                    "decision": "reject",
                    "partition": "AGENT_INFRASTRUCTURE_AUXILIARY",
                    "reason": "unclassified_reject",
                }
            )

    (BATCH / "curated_examples.json").write_text(
        json.dumps(curated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (BATCH / "curation_decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(curated)} curated rows and {len(decisions)} decisions")


if __name__ == "__main__":
    main()
