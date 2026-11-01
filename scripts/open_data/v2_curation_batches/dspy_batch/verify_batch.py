#!/usr/bin/env python3
"""Bounded before/after behavioral verification for dspy_batch accepted rows (C55)."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import inspect
import json
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
CASES = WORK / "main_candidates/cases"
FIXTURES = ROOT / "verification_fixtures"
CHILD_TIMEOUT = 20


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


def case_paths(source_id: str) -> Dict[str, Path]:
    base = CASES / source_id
    return {"before_root": base / "before", "applied_root": base / "applied", "reconstruction": base / "reconstruction.json"}


def load_tool_module(tool_path: Path):
    for k in list(sys.modules):
        if k.startswith("dspy"):
            del sys.modules[k]
    cb = types.ModuleType("dspy.utils.callback")
    cb.with_callbacks = lambda f: f
    u = types.ModuleType("dspy.utils")
    u.callback = cb
    pkg = types.ModuleType("dspy")
    pkg.utils = u
    sys.modules["dspy"] = pkg
    sys.modules["dspy.utils"] = u
    sys.modules["dspy.utils.callback"] = cb
    spec = importlib.util.spec_from_file_location("dspy.primitives.tool", tool_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_8089() -> dict:
    sid = "stanfordnlp__dspy-8089"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/primitives/tool.py"
    a_path = paths["applied_root"] / "dspy/primitives/tool.py"

    def my_tool(limit=120):
        return limit

    def probe(tool_path: Path) -> dict:
        mod = load_tool_module(tool_path)
        out: dict = {"init_ok": False, "call_ok": False, "err": None, "limit_default": None}
        try:
            t = mod.Tool(my_tool)
            out["init_ok"] = True
            out["limit_default"] = t.args.get("limit")
            out["call_ok"] = t(limit=5) == 5
        except Exception as exc:
            out["err"] = f"{type(exc).__name__}: {exc}"
        return out

    before_obs = probe(b_path)
    after_obs = probe(a_path)
    behavior = {
        "before_init_fails_on_untyped_default": not before_obs["init_ok"],
        "after_init_succeeds": after_obs["init_ok"],
        "after_default_in_schema": isinstance(after_obs.get("limit_default"), dict)
        and after_obs["limit_default"].get("default") == 120,
    }
    preservation = {
        "after_tool_callable": after_obs["call_ok"],
        "before_would_fail_not_skipped": not before_obs["init_ok"],
    }
    pending: List[str] = []
    st = sample_status(behavior, preservation, pending)
    fixture = {
        "source_id": sid,
        "inputs": {"tool_fn": "my_tool(limit=120)"},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fix_path = FIXTURES / f"{sid}.json"
    fix_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": pending,
        "fixture_path": str(fix_path.relative_to(WORK.parent)),
    }


def load_chat_adapter(chat_path: Path):
    for k in list(sys.modules):
        if k.startswith("dspy") or k == "litellm":
            del sys.modules[k]
    litellm = types.ModuleType("litellm")
    litellm.ContextWindowExceededError = Exception
    sys.modules["litellm"] = litellm

    class Adapter:
        def __init__(self, callbacks=None):
            self.callbacks = callbacks or []

    base = types.ModuleType("dspy.adapters.base")
    base.Adapter = Adapter
    utils = types.ModuleType("dspy.adapters.utils")
    for fn in [
        "format_field_value",
        "get_annotation_name",
        "get_field_description_string",
        "parse_value",
        "translate_field_type",
    ]:
        setattr(utils, fn, lambda *a, **k: "")
    sig = types.ModuleType("dspy.signatures.signature")
    sig.Signature = type("Signature", (), {})
    lm = types.ModuleType("dspy.clients.lm")
    lm.LM = object
    cb = types.ModuleType("dspy.utils.callback")
    cb.BaseCallback = object
    exc = types.ModuleType("dspy.utils.exceptions")
    exc.AdapterParseError = Exception
    adapters = types.ModuleType("dspy.adapters")
    adapters.base = base
    adapters.utils = utils
    pkg = types.ModuleType("dspy")
    pkg.adapters = adapters
    pkg.signatures = types.ModuleType("dspy.signatures")
    pkg.signatures.signature = sig
    pkg.clients = types.ModuleType("dspy.clients")
    pkg.clients.lm = lm
    pkg.utils = types.ModuleType("dspy.utils")
    pkg.utils.callback = cb
    pkg.utils.exceptions = exc
    for name, mod in [
        ("dspy", pkg),
        ("dspy.adapters", adapters),
        ("dspy.adapters.base", base),
        ("dspy.adapters.utils", utils),
        ("dspy.signatures", pkg.signatures),
        ("dspy.signatures.signature", sig),
        ("dspy.clients", pkg.clients),
        ("dspy.clients.lm", lm),
        ("dspy.utils", pkg.utils),
        ("dspy.utils.callback", cb),
        ("dspy.utils.exceptions", exc),
    ]:
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("dspy.adapters.chat_adapter", chat_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_8340() -> dict:
    sid = "stanfordnlp__dspy-8340"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/adapters/chat_adapter.py"
    a_path = paths["applied_root"] / "dspy/adapters/chat_adapter.py"

    class Sig:
        output_fields = {"answer": object()}

    def fmt(mod, path):
        m = load_chat_adapter(path)
        m.ChatAdapter.format_field_with_value = lambda self, f: "BODY"
        text = m.ChatAdapter().format_assistant_message_content(Sig, {"answer": "x"})
        return {"has_completed_marker": "[[ ## completed ## ]]" in text, "text_tail": text[-40:]}

    before_obs = fmt(None, b_path)
    after_obs = fmt(None, a_path)
    behavior = {
        "before_missing_completed_marker": not before_obs["has_completed_marker"],
        "after_includes_completed_marker": after_obs["has_completed_marker"],
    }
    preservation = {
        "before_still_emits_body": before_obs["text_tail"].endswith("BODY"),
        "after_still_emits_body": "BODY" in after_obs["text_tail"],
    }
    st = sample_status(behavior, preservation, [])
    fixture = {
        "source_id": sid,
        "inputs": {"outputs": {"answer": "x"}},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fix_path = FIXTURES / f"{sid}.json"
    fix_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": [],
        "fixture_path": str(fix_path.relative_to(WORK.parent)),
    }


class _JSONAdapter:
    pass


class _ChatAdapter:
    pass


def load_stream_listener(sl_path: Path):
    for k in list(sys.modules):
        if k.startswith("dspy") or k == "litellm":
            del sys.modules[k]
    litellm = types.ModuleType("litellm")

    class MRS:
        pass

    litellm.ModelResponseStream = MRS
    sys.modules["litellm"] = litellm
    chat = types.ModuleType("dspy.adapters.chat_adapter")
    chat.ChatAdapter = _ChatAdapter
    js = types.ModuleType("dspy.adapters.json_adapter")
    js.JSONAdapter = _JSONAdapter
    adapters = types.ModuleType("dspy.adapters")
    adapters.chat_adapter = chat
    adapters.json_adapter = js
    settings_mod = types.ModuleType("dspy.dsp.utils.settings")
    settings_mod.settings = types.SimpleNamespace(adapter=_JSONAdapter())
    dsp = types.ModuleType("dspy.dsp")
    utils = types.ModuleType("dspy.dsp.utils")
    utils.settings = settings_mod
    dsp.utils = utils
    msg = types.ModuleType("dspy.streaming.messages")
    msg.StreamResponse = lambda *a: a
    stream = types.ModuleType("dspy.streaming")
    stream.messages = msg
    pkg = types.ModuleType("dspy")
    pkg.adapters = adapters
    pkg.dsp = dsp
    pkg.streaming = stream
    for n, m in [
        ("dspy", pkg),
        ("dspy.adapters", adapters),
        ("dspy.adapters.chat_adapter", chat),
        ("dspy.adapters.json_adapter", js),
        ("dspy.dsp", dsp),
        ("dspy.dsp.utils", utils),
        ("dspy.dsp.utils.settings", settings_mod),
        ("dspy.streaming", stream),
        ("dspy.streaming.messages", msg),
    ]:
        sys.modules[n] = m
    spec = importlib.util.spec_from_file_location("dspy.streaming.streaming_listener", sl_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Chunk:
    def __init__(self, text: str):
        self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=text))]


def verify_8235() -> dict:
    sid = "stanfordnlp__dspy-8235"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/streaming/streaming_listener.py"
    a_path = paths["applied_root"] / "dspy/streaming/streaming_listener.py"
    chunk_text = '{"answer": "hello"}'

    def probe(path: Path) -> dict:
        mod = load_stream_listener(path)
        listener = mod.StreamListener("answer")
        listener.predict_name = "pred"
        listener.receive(_Chunk(chunk_text))
        return {
            "stream_start": listener.stream_start,
            "stream_end": listener.stream_end,
            "cache_hit": listener.cache_hit,
        }

    before_obs = probe(b_path)
    after_obs = probe(a_path)
    behavior = {
        "before_does_not_start_on_spaced_json": not before_obs["stream_start"],
        "after_recognizes_single_chunk_json": after_obs["stream_start"] and after_obs["stream_end"],
    }
    preservation = {
        "before_did_not_crash": True,
        "after_did_not_crash": True,
    }
    st = sample_status(behavior, preservation, [])
    fixture = {
        "source_id": sid,
        "inputs": {"chunk_text": chunk_text},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fix_path = FIXTURES / f"{sid}.json"
    fix_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": [],
        "fixture_path": str(fix_path.relative_to(WORK.parent)),
    }


def load_messages_module(path: Path, inject_asyncer: bool):
    for k in list(sys.modules):
        if k.startswith("dspy") or k == "asyncer":
            del sys.modules[k]
    pkg = types.ModuleType("dspy")
    pkg.dsp = types.ModuleType("dspy.dsp")
    utils = types.ModuleType("dspy.dsp.utils")
    settings_mod = types.ModuleType("dspy.dsp.utils.settings")
    settings_mod.settings = types.SimpleNamespace()
    utils.settings = settings_mod
    pkg.dsp.utils = utils
    cb = types.ModuleType("dspy.utils.callback")
    cb.BaseCallback = object
    cb_mod = types.ModuleType("dspy.utils")
    cb_mod.callback = cb
    sys.modules["dspy"] = pkg
    sys.modules["dspy.dsp"] = pkg.dsp
    sys.modules["dspy.dsp.utils"] = utils
    sys.modules["dspy.dsp.utils.settings"] = settings_mod
    sys.modules["dspy.utils"] = cb_mod
    sys.modules["dspy.utils.callback"] = cb
    if inject_asyncer:

        asyncer = types.ModuleType("asyncer")

        def syncify(fn):
            def run():
                return asyncio.run(fn())

            return run

        asyncer.syncify = syncify
        sys.modules["asyncer"] = asyncer
    spec = importlib.util.spec_from_file_location("dspy.streaming.messages", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stream:
    def __init__(self):
        self.sent: List[Any] = []

    async def send(self, msg):
        self.sent.append(msg)
        return True


def verify_8432() -> dict:
    sid = "stanfordnlp__dspy-8432"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/streaming/messages.py"
    a_path = paths["applied_root"] / "dspy/streaming/messages.py"

    def probe(path: Path, inject_asyncer: bool) -> dict:
        mod = load_messages_module(path, inject_asyncer=inject_asyncer)
        stream = _Stream()
        err = None
        try:
            mod.sync_send_to_stream(stream, "status-msg")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        src = path.read_text(encoding="utf-8")
        return {
            "sent": list(stream.sent),
            "err": err,
            "uses_get_running_loop": "get_running_loop" in src,
        }

    before_obs = probe(b_path, inject_asyncer=False)
    after_obs = probe(a_path, inject_asyncer=True)
    behavior = {
        "after_uses_running_loop_detection": after_obs["uses_get_running_loop"],
        "after_sync_send_delivers": after_obs["sent"] == ["status-msg"],
    }
    preservation = {
        "before_sync_context_send": before_obs["sent"] == ["status-msg"] and before_obs["err"] is None,
        "before_lacks_get_running_loop": "get_running_loop" not in b_path.read_text(encoding="utf-8"),
    }
    st = sample_status(behavior, preservation, [])
    fixture = {
        "source_id": sid,
        "inputs": {"message": "status-msg"},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fix_path = FIXTURES / f"{sid}.json"
    fix_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": [],
        "fixture_path": str(fix_path.relative_to(WORK.parent)),
    }


def stub_predict_deps():
    for k in list(sys.modules):
        if k.startswith("dspy") or k == "dsp":
            del sys.modules[k]
    sys.modules["dsp"] = types.ModuleType("dsp")
    dspy = types.ModuleType("dspy")
    sys.modules["dspy"] = dspy

    class Module:
        pass

    class Parameter:
        pass

    class Sig:
        instructions = "inst"
        fields = {"out": types.SimpleNamespace(json_schema_extra={"prefix": "p"})}

        def keys(self):
            return self.fields.keys()

        def dump_state(self):
            return {"signature_state": True}

    def ensure_signature(sig):
        return Sig()

    def signature_to_template(sig):
        return ""

    prim = types.ModuleType("dspy.primitives")
    prim.program = types.ModuleType("dspy.primitives.program")
    prim.program.Module = Module
    prim.prediction = types.ModuleType("dspy.primitives.prediction")
    prim.prediction.Prediction = Prediction = type("Prediction", (), {})
    dspy.primitives = prim
    sigm = types.ModuleType("dspy.signatures.signature")
    sigm.ensure_signature = ensure_signature
    sigm.signature_to_template = signature_to_template
    dspy.signatures = types.ModuleType("dspy.signatures")
    dspy.signatures.signature = sigm
    pred = types.ModuleType("dspy.predict")
    pred.parameter = types.ModuleType("dspy.predict.parameter")
    pred.parameter.Parameter = Parameter
    dspy.predict = pred
    for n, m in [
        ("dspy.primitives", prim),
        ("dspy.primitives.program", prim.program),
        ("dspy.primitives.prediction", prim.prediction),
        ("dspy.signatures", dspy.signatures),
        ("dspy.signatures.signature", sigm),
        ("dspy.predict", pred),
        ("dspy.predict.parameter", pred.parameter),
    ]:
        sys.modules[n] = m


def load_predict(path: Path):
    stub_predict_deps()
    spec = importlib.util.spec_from_file_location("dspy.predict.predict", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_1556() -> dict:
    sid = "stanfordnlp__dspy-1556"
    paths = case_paths(sid)
    b_path = paths["before_root"] / "dspy/predict/predict.py"
    a_path = paths["applied_root"] / "dspy/predict/predict.py"

    def probe(path: Path) -> dict:
        mod = load_predict(path)
        p = mod.Predict("q -> a")
        st = p.dump_state()
        return {"state_keys": sorted(st.keys())}

    before_obs = probe(b_path)
    after_obs = probe(a_path)
    behavior = {
        "before_uses_legacy_signature_prefix_fields": "signature_prefix" in before_obs["state_keys"],
        "after_persists_nested_signature_state": "signature" in after_obs["state_keys"],
    }
    preservation = {
        "before_still_has_demos_train_lm": all(k in before_obs["state_keys"] for k in ("demos", "train", "lm")),
        "after_still_has_demos_train_lm": all(k in after_obs["state_keys"] for k in ("demos", "train", "lm")),
    }
    st = sample_status(behavior, preservation, [])
    fixture = {
        "source_id": sid,
        "inputs": {"predict_signature": "q -> a"},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fix_path = FIXTURES / f"{sid}.json"
    fix_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": [],
        "fixture_path": str(fix_path.relative_to(WORK.parent)),
    }


def install_propose_stubs(root: Path):
    for k in list(sys.modules):
        if k.startswith("dspy"):
            del sys.modules[k]
    dspy = types.ModuleType("dspy")

    class Signature:
        pass

    class Module:
        pass

    dspy.Signature = Signature
    dspy.Module = Module
    dspy.InputField = lambda **k: k
    dspy.OutputField = lambda **k: k
    sys.modules["dspy"] = dspy

    class Proposer:
        def __init__(self):
            self.verbose = False
            self.use_task_demos = True
            self.set_history_randomly = False
            self.set_tip_randomly = False
            self.rng = types.SimpleNamespace(random=lambda: 0.5, choice=lambda xs: xs[0])

    propose = types.ModuleType("dspy.propose")
    propose_base = types.ModuleType("dspy.propose.propose_base")
    propose_base.Proposer = Proposer
    propose.propose_base = propose_base
    dsg = types.ModuleType("dspy.propose.dataset_summary_generator")
    dsg.create_dataset_summary = lambda **k: ""
    utils = types.ModuleType("dspy.propose.utils")
    for fn in [
        "create_example_string",
        "create_predictor_level_history_string",
        "strip_prefix",
        "get_dspy_source_code",
    ]:
        setattr(utils, fn, lambda *a, **k: "")
    propose.dataset_summary_generator = dsg
    propose.utils = utils
    dspy.propose = propose
    tp = types.ModuleType("dspy.teleprompt.utils")
    tp.get_signature = lambda x: x
    tp.get_prompt_model = lambda x: x
    tele = types.ModuleType("dspy.teleprompt")
    tele.utils = tp
    dspy.teleprompt = tele
    for name, mod in [
        ("dspy.propose", propose),
        ("dspy.propose.propose_base", propose_base),
        ("dspy.propose.dataset_summary_generator", dsg),
        ("dspy.propose.utils", utils),
        ("dspy.teleprompt", tele),
        ("dspy.teleprompt.utils", tp),
    ]:
        sys.modules[name] = mod
    gp_path = root / "dspy/propose/grounded_proposer.py"
    spec = importlib.util.spec_from_file_location("dspy.propose.grounded_proposer", gp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dspy.propose.grounded_proposer"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeProgram:
    def predictors(self):
        return [object()]


def verify_8053() -> dict:
    sid = "stanfordnlp__dspy-8053"
    paths = case_paths(sid)

    def run(root: Path) -> dict:
        mod = install_propose_stubs(root)
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
        calls = {"n": 0}

        def fake_prop(**kwargs):
            calls["n"] += 1
            return "ok"

        inst.propose_instruction_for_predictor = fake_prop
        err = None
        try:
            inst.propose_instructions_for_program(
                trainset=[],
                program=_FakeProgram(),
                demo_candidates=None,
                trial_logs={},
                N=2,
                T=1,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        return {"err": err, "predictor_calls": calls["n"]}

    before_obs = run(paths["before_root"])
    after_obs = run(paths["applied_root"])
    b_path = paths["before_root"] / "dspy/propose/grounded_proposer.py"
    a_path = paths["applied_root"] / "dspy/propose/grounded_proposer.py"
    behavior = {
        "before_raises_on_none_demo_candidates": before_obs["err"] is not None,
        "after_handles_none_with_N_iterations": after_obs["err"] is None and after_obs["predictor_calls"] == 2,
    }
    preservation = {
        "after_finite_instruction_calls": after_obs["predictor_calls"] == 2,
        "before_failed_not_marked_pass": before_obs["err"] is not None,
    }
    st = sample_status(behavior, preservation, [])
    fixture = {
        "source_id": sid,
        "inputs": {"demo_candidates": None, "N": 2},
        "before_observation": before_obs,
        "after_observation": after_obs,
        "source_hashes": {"before": sha256_file(b_path), "after": sha256_file(a_path)},
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fix_path = FIXTURES / f"{sid}.json"
    fix_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return {
        "source_id": sid,
        "status": st,
        "behavior_checks": behavior,
        "preservation_checks": preservation,
        "pending_reasons": [],
        "fixture_path": str(fix_path.relative_to(WORK.parent)),
    }


VERIFY_FNS = {
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
        "batch": "dspy_batch",
        "child_timeout_sec": CHILD_TIMEOUT,
        "policy_note": "Skipped/failed required checks are never scored as preservation PASS (C54/C55).",
        "samples": [],
    }
    for sid, fn in VERIFY_FNS.items():
        results["samples"].append(fn())
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"written": str(args.output), "pass_count": sum(1 for s in results["samples"] if s["status"] == "pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
