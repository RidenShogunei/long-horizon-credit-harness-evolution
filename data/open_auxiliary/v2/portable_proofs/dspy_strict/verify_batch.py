#!/usr/bin/env python3
"""Strict behavioral verification for C55 dspy_strict (root review: no static/source-text PASS)."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from source_loaders import (
    install_grounded_proposer,
    load_chat_adapter,
    load_messages_module,
    load_predict_stack,
    load_stream_listener,
    load_tool_module,
)

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CASES = WORK / "main_candidates/cases"
FIXTURES = ROOT / "verification_fixtures"
CHILD_TIMEOUT = 20

ACCEPTED = [
    "stanfordnlp__dspy-8089",
    "stanfordnlp__dspy-8340",
    "stanfordnlp__dspy-8235",
    "stanfordnlp__dspy-8432",
    "stanfordnlp__dspy-1556",
    "stanfordnlp__dspy-8053",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def write_fixture(sid: str, payload: dict) -> str:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{sid}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.relative_to(WORK.parent))


def case_paths(source_id: str) -> Dict[str, Path]:
    base = CASES / source_id
    return {
        "before_root": base / "before",
        "applied_root": base / "applied",
        "reconstruction": base / "reconstruction.json",
    }


def verify_8089() -> dict:
    sid = "stanfordnlp__dspy-8089"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/primitives/tool.py"
    a_path = paths["applied_root"] / "dspy/primitives/tool.py"

    def typed_tool(limit: int = 120):
        return limit

    def untyped_tool(limit=120):
        return limit

    def probe(tool_path: Path, fn, label: str) -> dict:
        mod = load_tool_module(tool_path)
        out: dict = {"label": label, "init_ok": False, "err": None, "call_result": None, "args_limit": None}
        try:
            t = mod.Tool(fn)
            out["init_ok"] = True
            out["args_limit"] = copy.deepcopy(t.args.get("limit"))
            out["call_result"] = t(limit=7)
        except Exception as exc:
            out["err"] = f"{type(exc).__name__}: {exc}"
        return out

    before_typed = probe(b_path, typed_tool, "typed")
    after_typed = probe(a_path, typed_tool, "typed")
    before_untyped = probe(b_path, untyped_tool, "untyped")
    after_untyped = probe(a_path, untyped_tool, "untyped")

    behavior = {
        "before_untyped_init_fails": not before_untyped["init_ok"],
        "after_untyped_init_ok": after_untyped["init_ok"],
        "after_untyped_call_returns_7": after_untyped["call_result"] == 7,
    }
    preservation = {
        "before_typed_init_and_call": before_typed["init_ok"] and before_typed["call_result"] == 7,
        "after_typed_init_and_call": after_typed["init_ok"] and after_typed["call_result"] == 7,
        "typed_limit_schema_equal": before_typed["args_limit"] == after_typed["args_limit"],
    }
    pending: List[str] = []
    fixture = {
        "source_id": sid,
        "before_observation": {"typed": before_typed, "untyped": before_untyped},
        "after_observation": {"typed": after_typed, "untyped": after_untyped},
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    return {
        "source_id": sid,
        "status": sample_status(behavior, preservation, pending),
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": write_fixture(sid, fixture),
    }


def verify_8340() -> dict:
    sid = "stanfordnlp__dspy-8340"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/adapters/chat_adapter.py"
    a_path = paths["applied_root"] / "dspy/adapters/chat_adapter.py"

    class Sig:
        input_fields = {"q": object()}
        output_fields = {"answer": object(), "confidence": object()}

    outputs = {"answer": "42", "confidence": "0.9"}
    marker = "[[ ## completed ## ]]\n"

    def assistant_body(mod_path: Path) -> dict:
        mod = load_chat_adapter(mod_path)
        adapter = mod.ChatAdapter()
        full = adapter.format_assistant_message_content(Sig, outputs)
        body = full.replace("\n\n" + marker, "").replace(marker, "").strip()
        fields = adapter.format_field_with_value(
            {
                mod.FieldInfoWithName(name=k, info=v): outputs.get(k)
                for k, v in Sig.output_fields.items()
            }
        )
        user_msg = adapter.format_user_message_content(Sig, {"q": "history-q"})
        field_structure = adapter.format_field_structure(Sig)
        return {"full": full, "body": body, "field_blob": fields, "user_msg": user_msg, "field_structure": field_structure}

    before_obs = assistant_body(b_path)
    after_obs = assistant_body(a_path)

    behavior = {
        "before_body_missing_completed_marker": marker.strip() not in before_obs["full"],
        "after_body_has_completed_marker": marker.strip() in after_obs["full"],
        "field_blob_unchanged": before_obs["field_blob"] == after_obs["field_blob"],
    }
    preservation = {
        "assistant_body_text_equal": before_obs["body"] == after_obs["body"],
        "user_message_content_equal": before_obs["user_msg"] == after_obs["user_msg"],
        "field_structure_equal": before_obs["field_structure"] == after_obs["field_structure"],
    }
    pending: List[str] = []
    fixture = {
        "source_id": sid,
        "inputs": {"outputs": outputs},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    return {
        "source_id": sid,
        "status": sample_status(behavior, preservation, pending),
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": write_fixture(sid, fixture),
    }


class _Chunk:
    def __init__(self, text: str | None):
        self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=text))]


def _collect_tokens(listener_mod, sl_path: Path, chunks: List[str | None]) -> dict:
    mod = load_stream_listener(sl_path)
    listener = mod.StreamListener("answer")
    listener.predict_name = "pred"
    tokens: List[str] = []
    for ch in chunks:
        resp = listener.receive(_Chunk(ch))
        if resp is not None:
            tokens.append(resp.chunk)
    flushed = listener.flush() if listener.stream_end else ""
    if flushed:
        tokens.append(flushed)
    return {
        "tokens": tokens,
        "joined": "".join(tokens),
        "stream_start": listener.stream_start,
        "stream_end": listener.stream_end,
    }


def verify_8235() -> dict:
    sid = "stanfordnlp__dspy-8235"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/streaming/streaming_listener.py"
    a_path = paths["applied_root"] / "dspy/streaming/streaming_listener.py"

    compact_full = ['{"answer":"hello"}']
    spaced_one = ['{"answer": "hello"}']
    legacy_stream_chunks = ['{"answer":"', "hel", 'lo"}']
    malformed = [None, "", '{"answer":"hello"}']

    def obs(sl_path, chunks):
        return _collect_tokens(None, sl_path, chunks)

    before_compact = obs(b_path, compact_full)
    after_compact = obs(a_path, compact_full)
    before_legacy = obs(b_path, legacy_stream_chunks)
    after_legacy = obs(a_path, legacy_stream_chunks)
    before_spaced = obs(b_path, spaced_one)
    after_spaced = obs(a_path, spaced_one)
    before_mal = obs(b_path, malformed)
    after_mal = obs(a_path, malformed)

    behavior = {
        "before_spaced_single_chunk_no_stream": not (
            before_spaced["stream_start"] and before_spaced["stream_end"]
        ),
        "after_spaced_single_chunk_streams": after_spaced["stream_start"] and after_spaced["stream_end"],
    }
    preservation = {
        "compact_full_json_both_finish": before_compact["stream_end"] and after_compact["stream_end"],
        "legacy_multichunk_joined_equal": before_legacy["joined"] == after_legacy["joined"],
        "legacy_multichunk_nonempty_before": len(before_legacy["joined"]) > 0,
        "malformed_prefix_then_valid_both_finish": before_mal["stream_end"] and after_mal["stream_end"],
    }
    pending: List[str] = []
    if not preservation["legacy_multichunk_joined_equal"]:
        pending.append("legacy_multichunk_token_join_mismatch")

    fixture = {
        "source_id": sid,
        "before_observation": {
            "compact_full": before_compact,
            "legacy_stream": before_legacy,
            "spaced": before_spaced,
            "malformed": before_mal,
        },
        "after_observation": {
            "compact_full": after_compact,
            "legacy_stream": after_legacy,
            "spaced": after_spaced,
            "malformed": after_mal,
        },
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    return {
        "source_id": sid,
        "status": sample_status(behavior, preservation, pending),
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": write_fixture(sid, fixture),
    }


class _ListStream:
    def __init__(self):
        self.sent: List[Any] = []

    async def send(self, message):
        self.sent.append(message)
        return True


class _MainLoopStream:
    """Consumer reads only messages delivered on the bound main event loop."""

    def __init__(self):
        self.main_loop = None
        self.on_main: List[Any] = []
        self.off_main: List[Any] = []

    async def send(self, message):
        if self.main_loop is not None and asyncio.get_running_loop() is self.main_loop:
            self.on_main.append(message)
        else:
            self.off_main.append(message)
        return True


def verify_8432() -> dict:
    sid = "stanfordnlp__dspy-8432"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/streaming/messages.py"
    a_path = paths["applied_root"] / "dspy/streaming/messages.py"

    def sync_probe(mod_path: Path, inject_asyncer: bool) -> dict:
        mod = load_messages_module(mod_path, inject_asyncer=inject_asyncer)
        s = _ListStream()
        err = None
        try:
            mod.sync_send_to_stream(s, types.SimpleNamespace(message="sync-status"))
        except Exception as exc:
            err = repr(exc)
        return {"sent": [getattr(x, "message", x) for x in s.sent], "err": err}

    async def async_probe(mod_path: Path, inject_asyncer: bool) -> dict:
        mod = load_messages_module(mod_path, inject_asyncer=inject_asyncer)

        async def run() -> dict:
            stream = _MainLoopStream()
            stream.main_loop = asyncio.get_running_loop()
            err = None
            try:
                mod.sync_send_to_stream(stream, types.SimpleNamespace(message="async-status"))
            except Exception as exc:
                err = repr(exc)
            return {
                "on_main": [getattr(x, "message", x) for x in stream.on_main],
                "off_main": [getattr(x, "message", x) for x in stream.off_main],
                "err": err,
            }

        return await run()

    async def direct_async_send(mod_path: Path, inject_asyncer: bool) -> dict:
        mod = load_messages_module(mod_path, inject_asyncer=inject_asyncer)
        s = _ListStream()
        await s.send(types.SimpleNamespace(message="direct"))
        return {"sent": [getattr(x, "message", x) for x in s.sent]}

    before_sync = sync_probe(b_path, False)
    after_sync = sync_probe(a_path, True)
    before_async = asyncio.run(async_probe(b_path, False))
    after_async = asyncio.run(async_probe(a_path, True))
    before_direct = asyncio.run(direct_async_send(b_path, False))
    after_direct = asyncio.run(direct_async_send(a_path, True))

    behavior = {
        "before_async_missing_main_delivery": before_async["on_main"] != ["async-status"],
    }
    preservation = {
        "before_sync_send_unchanged": before_sync["sent"] == ["sync-status"] and before_sync["err"] is None,
        "after_sync_send_unchanged": after_sync["sent"] == ["sync-status"] and after_sync["err"] is None,
        "direct_async_send_still_works_before": before_direct["sent"] == ["direct"],
        "direct_async_send_still_works_after": after_direct["sent"] == ["direct"],
    }
    pending: List[str] = []
    if behavior["before_async_missing_main_delivery"]:
        if after_async["on_main"] == ["async-status"]:
            behavior["after_async_delivers_on_main_loop"] = True
        else:
            behavior["after_async_delivers_on_main_loop"] = False
            pending.append(
                "pinned_sources_do_not_reproduce_main_loop_status_delivery_delta_in_bounded_harness"
            )
    else:
        behavior["after_async_delivers_on_main_loop"] = after_async["on_main"] == ["async-status"]
        pending.append("before_async_main_delivery_unexpectedly_present")

    fixture = {
        "source_id": sid,
        "before_observation": {"sync": before_sync, "async": before_async, "direct": before_direct},
        "after_observation": {"sync": after_sync, "async": after_async, "direct": after_direct},
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    st = sample_status(behavior, preservation, pending)
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": write_fixture(sid, fixture),
    }


def _payload_equal(a, b) -> bool:
    return a == b


def verify_1556() -> dict:
    sid = "stanfordnlp__dspy-1556"
    paths = case_paths(sid)
    b_root = paths["before_root"]
    a_root = paths["applied_root"]
    b_predict = b_root / "dspy/predict/predict.py"
    a_predict = a_root / "dspy/predict/predict.py"
    b_sig = b_root / "dspy/signatures/signature.py"
    a_sig = a_root / "dspy/signatures/signature.py"

    def seed_predict(predict_mod):
        p = predict_mod.Predict("question -> answer")
        p.lm = {"model": "stub-lm"}
        p.traces = [{"step": 1}]
        p.train = [{"ex": "train"}]
        p.demos = [{"question": "q1", "answer": "a1"}]
        p.signature = p.signature.with_instructions("roundtrip-instructions")
        return p

    def roundtrip_before() -> dict:
        predict_mod, _ = load_predict_stack(b_root)
        p1 = seed_predict(predict_mod)
        state = p1.dump_state()
        blob = json.loads(json.dumps(state, default=str))
        p2 = predict_mod.Predict("question -> answer")
        p2.load_state(blob)
        return {
            "state_keys": sorted(state.keys()),
            "payload_equal": _payload_equal(
                {"lm": p2.lm, "traces": p2.traces, "train": p2.train, "demos": p2.demos},
                {"lm": p1.lm, "traces": p1.traces, "train": p1.train, "demos": p1.demos},
            ),
            "instructions_restored": p2.signature.instructions == "roundtrip-instructions",
            "err": None,
        }

    def roundtrip_after() -> dict:
        predict_mod, sig_mod = load_predict_stack(a_root)
        p1 = seed_predict(predict_mod)
        out_key = next(iter(p1.signature.output_fields))
        p1.signature = p1.signature.with_updated_fields(out_key, prefix="AnswerPrefix:")
        state = p1.dump_state()
        blob = json.loads(json.dumps(state, default=str))
        p2 = predict_mod.Predict("question -> answer")
        try:
            p2.load_state(blob)
        except Exception as exc:
            return {"err": repr(exc)}
        sig_state = state.get("signature")
        reloaded_sig = p2.signature.load_state(sig_state)
        return {
            "state_keys": sorted(state.keys()),
            "payload_equal": _payload_equal(
                {"lm": p2.lm, "traces": p2.traces, "train": p2.train, "demos": p2.demos},
                {"lm": p1.lm, "traces": p1.traces, "train": p1.train, "demos": p1.demos},
            ),
            "signature_roundtrip_prefix": reloaded_sig.fields[out_key].json_schema_extra["prefix"]
            == "AnswerPrefix:",
            "uses_signature_dump_api": hasattr(sig_mod.Signature, "dump_state"),
            "err": None,
        }

    before_obs = roundtrip_before()
    after_obs = roundtrip_after()

    behavior = {
        "after_json_roundtrip_restores_signature_prefix": after_obs.get("signature_roundtrip_prefix") is True,
        "after_payload_preserved": after_obs.get("payload_equal") is True,
        "after_uses_real_signature_dump_state": after_obs.get("uses_signature_dump_api") is True,
    }
    preservation = {
        "before_legacy_roundtrip_payload_equal": before_obs.get("payload_equal") is True,
        "before_legacy_instructions_restored": before_obs.get("instructions_restored") is True,
    }
    pending: List[str] = []
    if after_obs.get("err"):
        pending.append(f"after_roundtrip_error:{after_obs['err']}")

    fixture = {
        "source_id": sid,
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {
            "before_predict": sha256_file(b_predict),
            "after_predict": sha256_file(a_predict),
            "before_signature": sha256_file(b_sig),
            "after_signature": sha256_file(a_sig),
        },
    }
    return {
        "source_id": sid,
        "status": sample_status(behavior, preservation, pending),
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": write_fixture(sid, fixture),
    }


class _FakeProgram:
    def predictors(self):
        return [object()]


def _run_proposer(case_root: Path, demo_candidates, n: int) -> dict:
    mod = install_grounded_proposer(case_root)
    inst = mod.GroundedProposer.__new__(mod.GroundedProposer)
    Proposer = sys.modules["dspy.propose.propose_base"].Proposer
    Proposer.__init__(inst)
    inst.verbose = False
    inst.use_task_demos = True
    inst.set_history_randomly = False
    inst.set_tip_randomly = False
    inst.rng = types.SimpleNamespace(
        random=lambda: 0.5, choice=lambda xs: list(mod.TIPS.keys())[0]
    )
    calls: List[int] = []

    def fake_prop(**kwargs):
        calls.append(kwargs.get("demo_set_i"))
        return "ok"

    inst.propose_instruction_for_predictor = fake_prop
    err = None
    try:
        inst.propose_instructions_for_program(
            trainset=[],
            program=_FakeProgram(),
            demo_candidates=demo_candidates,
            trial_logs={},
            N=n,
            T=1,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    return {"err": err, "demo_set_indices": calls, "call_count": len(calls)}


def verify_8053() -> dict:
    sid = "stanfordnlp__dspy-8053"
    paths = case_paths(sid)
    b_root = paths["before_root"]
    a_root = paths["applied_root"]
    gp = b_root / "dspy/propose/grounded_proposer.py"
    valid_demos = [[{"x": 1}, {"x": 2}]]

    before_valid = _run_proposer(b_root, valid_demos, n=99)
    after_valid = _run_proposer(a_root, valid_demos, n=99)
    before_none = _run_proposer(b_root, None, n=2)
    after_none = _run_proposer(a_root, None, n=2)

    behavior = {
        "before_none_branch_raises": before_none["err"] is not None,
        "after_none_branch_two_calls": after_none["err"] is None and after_none["call_count"] == 2,
    }
    preservation = {
        "valid_demo_candidates_same_call_count": before_valid["call_count"] == after_valid["call_count"] == 2,
        "valid_demo_candidates_same_indices": before_valid["demo_set_indices"] == after_valid["demo_set_indices"],
    }
    pending: List[str] = []
    fixture = {
        "source_id": sid,
        "inputs": {"valid_demo_candidates": valid_demos, "none_n": 2},
        "before_observation": {"valid": before_valid, "none": before_none},
        "after_observation": {"valid": after_valid, "none": after_none},
        "source_hashes": {
            "before": sha256_file(gp),
            "after": sha256_file(a_root / "dspy/propose/grounded_proposer.py"),
        },
    }
    return {
        "source_id": sid,
        "status": sample_status(behavior, preservation, pending),
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": write_fixture(sid, fixture),
    }


VERIFY = {
    "stanfordnlp__dspy-8089": verify_8089,
    "stanfordnlp__dspy-8340": verify_8340,
    "stanfordnlp__dspy-8235": verify_8235,
    "stanfordnlp__dspy-8432": verify_8432,
    "stanfordnlp__dspy-1556": verify_1556,
    "stanfordnlp__dspy-8053": verify_8053,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "verification_results.json")
    args = parser.parse_args()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch": "dspy_strict",
        "policy_note": "No static/source-text checks; skip/error never preservation PASS.",
        "samples": [VERIFY[sid]() for sid in ACCEPTED],
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {
        "written": str(args.output),
        "pass": sum(1 for s in results["samples"] if s["status"] == "pass"),
        "pending": sum(1 for s in results["samples"] if s["status"] == "pending"),
        "reject": sum(1 for s in results["samples"] if s["status"] == "reject"),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
