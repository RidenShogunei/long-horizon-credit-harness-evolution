#!/usr/bin/env python3
"""Write curated_examples.json and curation_decisions.json for dspy_batch (C55)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CASES = WORK / "main_candidates/cases"
RAW = WORK / "main_candidates/raw_records.jsonl"


def load_issues() -> dict[str, str]:
    out = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        out[o["instance_id"]] = o.get("problem_statement", "")
    return out


def excerpt(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end])


def base_row(
    source_id: str,
    repo: str,
    base_commit: str,
    harness: str,
    visible_path: str,
    original_path: str,
    sha256: str,
    start: int,
    end: int,
    state_id: str,
    issue: str,
    module_code_key: str,
    module_code: str,
    plan: dict,
    verification_status: str,
    fixture_rel: str,
):
    return {
        "source_id": source_id,
        "split": "train",
        "verification_status": verification_status,
        "verification_fixture_path": fixture_rel,
        "curation_notes": "Reconstructed before/applied per-case sources; bounded verify_batch.py executed.",
        "provenance": {
            "partition": "AUXILIARY_REPORTED_ISSUE_PLANNING",
            "original_teacher_plan": False,
            "formal": False,
        },
        "source_mapping": {
            "dataset": "nebius/SWE-rebench",
            "repo": repo,
            "base_commit": base_commit,
            "original_source_path": original_path,
            "visible_path": visible_path,
            "harness_module": harness,
        },
        "source_refs": [
            {
                "qualname": plan.get("_qualname", ""),
                "artifact_path": f"work/open_dataset_integration_v2/main_candidates/cases/{source_id}/before/{original_path}",
                "sha256": sha256,
                "start_line": start,
                "end_line": end,
            }
        ],
        "state": {
            "state_id": state_id,
            "root_lineage_id": f"{repo}@{base_commit[:12]}",
            "split": "train",
            "plan_policy_version": "proposal_single_plan_v6",
            "permitted_modules": [harness],
            "module_code": {module_code_key: module_code},
            "history": [],
            "evidence": [
                {
                    "id": "ev_reported_issue",
                    "text": f"Reported issue (original task text, not locally reproduced end-to-end): {issue[:1200]}",
                },
                {
                    "id": "ev_exact_source",
                    "text": (
                        f"Source fact: before snapshot {original_path} lines {start}-{end} sha256={sha256} "
                        f"(see source_refs; full tree under main_candidates/cases/{source_id}/before)."
                    ),
                },
            ],
            "verification_catalog": [],
        },
        "plan": {k: v for k, v in plan.items() if not k.startswith("_")},
    }


def main() -> None:
    issues = load_issues()
    rows = []

    c8089 = CASES / "stanfordnlp__dspy-8089"
    rows.append(
        base_row(
            "stanfordnlp__dspy-8089",
            "stanfordnlp/dspy",
            "96f1e95712cd4cd5bc30eb0cfc1eb0ea9a2e8512",
            "tools",
            "tools/tool.py",
            "dspy/primitives/tool.py",
            "2247475bc1d6cfbbac4267328d0843f9f0a20b1ace86a7bca715eb7b24f7a683",
            110,
            144,
            "REBENCH-DSPY-8089",
            issues["stanfordnlp__dspy-8089"],
            "tools/tool.py",
            excerpt(c8089 / "before/dspy/primitives/tool.py", 110, 144),
            {
                "_qualname": "dspy.primitives.tool.Tool.__init__",
                "action": "edit",
                "target_module": "tools",
                "locations": ["tools/tool.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": (
                    "Source fact: Tool.__init__ sets args[k] to TypeAdapter(v).json_schema() or the string 'Any', "
                    "then assigns args[k]['default']. Reporter claim: untyped defaulted parameters raise TypeError "
                    "because 'Any' is a str. Hypothesis: keep dict schemas and gate validation on schema['type']."
                ),
                "intended_change": (
                    "Stop coercing missing hints to the literal 'Any' string; omit json_schema() falsy fallback; "
                    "in __call__ validate only when args[k] is a dict with type not equal to 'Any'."
                ),
                "preserve_behaviors": [
                    "Tools with explicit typed parameters still expose JSON schema dicts",
                    "Tool invocation still forwards kwargs to the wrapped callable",
                    "Pydantic BaseModel-typed parameters still use resolved pydantic schemas",
                ],
                "verification_plan": [
                    "Construct Tool around def f(limit=120) without annotations; before raises on init, after succeeds",
                    "Call after Tool with explicit limit and confirm return value",
                ],
                "verification_kind": ["behavioral_before_after", "preservation_after_only"],
            },
            "local_bounded_verification_pass",
            "work/open_dataset_integration_v2/dspy_batch/verification_fixtures/stanfordnlp__dspy-8089.json",
        )
    )

    c8340 = CASES / "stanfordnlp__dspy-8340"
    rows.append(
        base_row(
            "stanfordnlp__dspy-8340",
            "stanfordnlp/dspy",
            "e4ec979a3d76f3375637f3a06bb92d406a36a31e",
            "context_mgmt",
            "context_mgmt/chat_adapter.py",
            "dspy/adapters/chat_adapter.py",
            "5c1a8121fb9f388956c67f61f74df80b2b2a0f67a65b8772d468763e0babba8f",
            138,
            151,
            "REBENCH-DSPY-8340",
            issues["stanfordnlp__dspy-8340"],
            "context_mgmt/chat_adapter.py",
            excerpt(c8340 / "before/dspy/adapters/chat_adapter.py", 138, 151),
            {
                "_qualname": "dspy.adapters.chat_adapter.ChatAdapter.format_assistant_message_content",
                "action": "edit",
                "target_module": "context_mgmt",
                "locations": ["context_mgmt/chat_adapter.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": (
                    "Reporter claim: History/adapter-constructed assistant messages omit the "
                    "[[ ## completed ## ]] marker required by ChatAdapter field instructions. "
                    "Source fact: format_assistant_message_content returned only format_field_with_value output."
                ),
                "intended_change": (
                    "Append '\\n\\n[[ ## completed ## ]]\\n' to assistant_message_content before returning."
                ),
                "preserve_behaviors": [
                    "Output field headers/content from format_field_with_value remain unchanged",
                    "parse() and other ChatAdapter methods untouched in scoped plan",
                ],
                "verification_plan": [
                    "Call format_assistant_message_content on before vs after ChatAdapter with stubbed body",
                    "Confirm after includes completed marker; before lacks it while body preserved",
                ],
                "verification_kind": ["behavioral_before_after"],
            },
            "local_bounded_verification_pass",
            "work/open_dataset_integration_v2/dspy_batch/verification_fixtures/stanfordnlp__dspy-8340.json",
        )
    )

    c8235 = CASES / "stanfordnlp__dspy-8235"
    rows.append(
        base_row(
            "stanfordnlp__dspy-8235",
            "stanfordnlp/dspy",
            "ed3736e95a2b0f5c9760e7dc21d5efee067e0de4",
            "observation",
            "observation/streaming_listener.py",
            "dspy/streaming/streaming_listener.py",
            json.loads((c8235 / "reconstruction.json").read_text())["source_files"][
                "dspy/streaming/streaming_listener.py"
            ]["before_sha256"],
            36,
            90,
            "REBENCH-DSPY-8235",
            issues["stanfordnlp__dspy-8235"],
            "observation/streaming_listener.py",
            excerpt(c8235 / "before/dspy/streaming/streaming_listener.py", 36, 90),
            {
                "_qualname": "dspy.streaming.streaming_listener.StreamListener.receive",
                "action": "edit",
                "target_module": "observation",
                "locations": ["observation/streaming_listener.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": (
                    "Reporter claim: Google/Gemini streaming emits JSON chunks like "
                    "'{\"answer\": \"text\"}' so legacy start pattern '{\"field\":\"' never matches. "
                    "Hypothesis: relax JSON start/end identifiers and buffering logic."
                ),
                "intended_change": (
                    "Use '\"{field}\":' start token and updated end regex; improve cache-hit detection "
                    "and partial identifier buffering via _buffered_message_end_with_start_identifier."
                ),
                "preserve_behaviors": [
                    "ChatAdapter streaming identifiers unchanged",
                    "Listener still ignores empty delta content safely",
                ],
                "verification_plan": [
                    "Feed single-chunk spaced JSON to StreamListener with JSONAdapter settings",
                    "Before fails to start stream; after marks stream_start/stream_end",
                ],
                "verification_kind": ["behavioral_before_after"],
            },
            "local_bounded_verification_pass",
            "work/open_dataset_integration_v2/dspy_batch/verification_fixtures/stanfordnlp__dspy-8235.json",
        )
    )

    c8432 = CASES / "stanfordnlp__dspy-8432"
    rows.append(
        base_row(
            "stanfordnlp__dspy-8432",
            "stanfordnlp/dspy",
            "94299c7edbbe39cbb086cff892e3902e27cb2f1d",
            "agent_loop",
            "agent_loop/streaming_messages.py",
            "dspy/streaming/messages.py",
            json.loads((c8432 / "reconstruction.json").read_text())["source_files"]["dspy/streaming/messages.py"][
                "before_sha256"
            ],
            23,
            53,
            "REBENCH-DSPY-8432",
            issues["stanfordnlp__dspy-8432"],
            "agent_loop/streaming_messages.py",
            excerpt(c8432 / "before/dspy/streaming/messages.py", 23, 53),
            {
                "_qualname": "dspy.streaming.messages.sync_send_to_stream",
                "action": "edit",
                "target_module": "agent_loop",
                "locations": ["agent_loop/streaming_messages.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": (
                    "Reporter claim: status streaming intermittently fails when callers already run inside "
                    "an event loop. Source fact: sync_send_to_stream used get_event_loop/is_running branching."
                ),
                "intended_change": (
                    "Detect running loop via asyncio.get_running_loop(); offload send to a thread-local loop "
                    "or use asyncer.syncify when no loop is running."
                ),
                "preserve_behaviors": [
                    "Synchronous callers without a running loop still deliver messages",
                    "StatusMessageProvider API unchanged",
                ],
                "verification_plan": [
                    "Invoke sync_send_to_stream from plain sync context before/after",
                    "Confirm after source uses get_running_loop and still sends message",
                ],
                "verification_kind": ["behavioral_before_after", "preservation_after_only"],
            },
            "local_bounded_verification_pass",
            "work/open_dataset_integration_v2/dspy_batch/verification_fixtures/stanfordnlp__dspy-8432.json",
        )
    )

    c1556 = CASES / "stanfordnlp__dspy-1556"
    rows.append(
        base_row(
            "stanfordnlp__dspy-1556",
            "stanfordnlp/dspy",
            "92d25a7066227d18f852eb49a397b46c11f0dc1a",
            "context_mgmt",
            "context_mgmt/predict.py",
            "dspy/predict/predict.py",
            "cf4228bc94f2f57a0652c063f59dfc820a756d044dde3607cd9300ab6ce5c91d",
            34,
            72,
            "REBENCH-DSPY-1556",
            issues["stanfordnlp__dspy-1556"],
            "context_mgmt/predict.py",
            excerpt(c1556 / "before/dspy/predict/predict.py", 34, 72),
            {
                "_qualname": "dspy.predict.predict.Predict.dump_state",
                "action": "edit",
                "target_module": "context_mgmt",
                "locations": [
                    "context_mgmt/predict.py",
                    "context_mgmt/signature.py",
                    "context_mgmt/module.py",
                ],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": (
                    "Reporter claim: dump_state/load_state only persisted signature_instructions/prefix, "
                    "not full signature objects for Predict modules. Hypothesis: nest signature.dump_state()."
                ),
                "intended_change": (
                    "Replace legacy instruction/prefix keys with state['signature']=self.signature.dump_state() "
                    "and extended_signature handling; add legacy loader gate."
                ),
                "preserve_behaviors": [
                    "demos/train/lm still serialized",
                    "BaseModel demo fields still converted to model_dump_json in dump_state",
                ],
                "verification_plan": [
                    "Instantiate Predict with stub signature; compare dump_state keys before vs after",
                    "Ensure after includes signature blob while demos/train/lm preserved",
                ],
                "verification_kind": ["behavioral_before_after"],
            },
            "local_bounded_verification_pass",
            "work/open_dataset_integration_v2/dspy_batch/verification_fixtures/stanfordnlp__dspy-1556.json",
        )
    )

    c8053 = CASES / "stanfordnlp__dspy-8053"
    rows.append(
        base_row(
            "stanfordnlp__dspy-8053",
            "stanfordnlp/dspy",
            "dd954e3cf2fbb119530ca2e6885b784bad9bace9",
            "agent_loop",
            "agent_loop/grounded_proposer.py",
            "dspy/propose/grounded_proposer.py",
            json.loads((c8053 / "reconstruction.json").read_text())["source_files"]["dspy/propose/grounded_proposer.py"][
                "before_sha256"
            ],
            321,
            378,
            "REBENCH-DSPY-8053",
            issues["stanfordnlp__dspy-8053"],
            "agent_loop/grounded_proposer.py",
            excerpt(c8053 / "before/dspy/propose/grounded_proposer.py", 321, 378),
            {
                "_qualname": "dspy.propose.grounded_proposer.GroundedProposer.propose_instructions_for_program",
                "action": "edit",
                "target_module": "agent_loop",
                "locations": ["agent_loop/grounded_proposer.py"],
                "evidence_refs": ["ev_reported_issue", "ev_exact_source"],
                "problem_hypothesis": (
                    "Reporter claim: MIPROv2 bootstrap can yield demo_candidates=None leading to "
                    "TypeError subscripting None inside propose_instructions_for_program. "
                    "Source fact: falsy demo_candidates branch still iterated len(demo_candidates[0])."
                ),
                "intended_change": (
                    "When demo_candidates is falsy, set num_demos=N instead of indexing demo_candidates[0]; "
                    "loop range(num_demos)."
                ),
                "preserve_behaviors": [
                    "When demo_candidates provided, still iterate max(len(demo_candidates[0]),1) demos",
                    "propose_instruction_for_predictor still invoked per predictor/demo index",
                ],
                "verification_plan": [
                    "Call propose_instructions_for_program with demo_candidates=None and N=2 using stubbed predictor hook",
                    "Before raises TypeError; after performs two stub calls",
                ],
                "verification_kind": ["behavioral_before_after"],
            },
            "local_bounded_verification_pass",
            "work/open_dataset_integration_v2/dspy_batch/verification_fixtures/stanfordnlp__dspy-8053.json",
        )
    )

    (ROOT / "curated_examples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    decisions = {
        "policy_note": "Curator dispositions for 11 stanfordnlp/dspy + 1 mem0ai/mem0 rows owned by dspy_batch (C55).",
        "lineage": {
            "dataset": "nebius/SWE-rebench",
            "raw_records_path": "work/open_dataset_integration_v2/main_candidates/raw_records.jsonl",
        },
        "decisions": [
            {
                "source_row_id": "stanfordnlp__dspy-8089",
                "disposition": "include",
                "reason": "Single-file Tool schema/default fix matches TypeError report; bounded verification pass.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8340",
                "disposition": "include",
                "reason": "ChatAdapter assistant completion marker fix matches History formatting report; verification pass on scoped method.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8235",
                "disposition": "include",
                "reason": "StreamListener JSON chunk tokenization fix matches Google streaming report; verification pass on receive().",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8432",
                "disposition": "include",
                "reason": "sync_send_to_stream loop handling matches status streaming report; verification pass with dependency stub asyncer only.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-1556",
                "disposition": "include",
                "reason": "Predict dump_state signature persistence matches save/load report; verification pass on dump_state keys (multi-file patch noted in plan).",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8053",
                "disposition": "include",
                "reason": "GroundedProposer None demo_candidates guard matches MIPROv2 TypeError report; verification pass via stubbed propose_instruction_for_predictor.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8445",
                "disposition": "exclude",
                "reason": "Patch spans docs plus json_adapter and lm error handling; not a single bounded harness-module mechanism.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-1651",
                "disposition": "exclude",
                "reason": "TypedPredictor signature property fix is real but pinned tree lacks importable dsp stack for honest execution; pending bounded run.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8146",
                "disposition": "exclude",
                "reason": "Usage tracking change touches lm, predict, tool, streaming modules together; inseparable for single-module plan.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-8339",
                "disposition": "exclude",
                "reason": "Overlaps dspy-8340 completion-marker fix plus FastAPI/Bedrock usage-tracker churn across many files.",
            },
            {
                "source_row_id": "stanfordnlp__dspy-1853",
                "disposition": "exclude",
                "reason": "Refactors adapters/utils extraction across chat/json/react/signature/requirements; not separable typed-input JSON serialization check.",
            },
            {
                "source_row_id": "mem0ai__mem0-2774",
                "disposition": "exclude",
                "reason": "Primary fix is except-block indentation in AsyncMemory helpers but pinned snapshot lacks importable mem0 dependency graph for bounded execution.",
            },
        ],
    }
    (ROOT / "curation_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    print(json.dumps({"curated_rows": len(rows), "decisions": len(decisions["decisions"])}))


if __name__ == "__main__":
    main()
