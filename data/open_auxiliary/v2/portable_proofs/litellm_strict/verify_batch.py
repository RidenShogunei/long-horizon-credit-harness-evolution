"""Strict before/after behavioral verification using pinned LiteLLM source callables."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

STRICT_ROOT = Path(__file__).resolve().parent
REPO = STRICT_ROOT.parents[2]
CASES = REPO / "work/open_dataset_integration_v2/main_candidates/cases"
BATCH = REPO / "work/open_dataset_integration_v2/litellm_batch"

sys.path.insert(0, str(STRICT_ROOT))
from runtime.pinned_import import (  # noqa: E402
    extract_class_from_pinned,
    extract_function_from_pinned,
)

CHILD_TIMEOUT = 20


@dataclass
class Obs:
    name: str
    status: str  # pass | fail | skip
    inputs: dict[str, Any]
    before: Any
    after: Any
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["inputs_sha256"] = hashlib.sha256(
            json.dumps(self.inputs, sort_keys=True, default=str).encode()
        ).hexdigest()
        return d


def _run5232_stream() -> list[Obs]:
    case = "berriai__litellm-5232"

    class AnthropicError(Exception):
        def __init__(self, message: str = "", status_code: int = 500):
            super().__init__(message)
            self.status_code = status_code

    def drive(variant: str) -> dict[str, Any]:
        cls = extract_class_from_pinned(
            case,
            variant,
            "llms/anthropic.py",
            "ModelResponseIterator",
            {"AnthropicError": AnthropicError},
        )
        it = cls(None, True)
        chunks = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "f1",
                    "input": {},
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "content_block_start",
                "index": 3,
                "content_block": {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "f2",
                    "input": {},
                },
            },
            {"type": "content_block_stop", "index": 3},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        ]
        texts: list[str] = []
        tool_indices: list[int] = []
        finish: str | None = None
        for ch in chunks:
            out = it.chunk_parser(ch)
            if out.get("text"):
                texts.append(out["text"])
            tu = out.get("tool_use")
            if tu is not None and tu.get("index") is not None:
                tool_indices.append(int(tu["index"]))
            if out.get("is_finished"):
                finish = out.get("finish_reason")
        return {"texts": texts, "tool_indices": tool_indices, "finish_reason": finish}

    before = drive("before")
    after = drive("applied")
    inputs = {"chunk_sequence": "text@0,tool@1,tool@3,message_delta"}
    checks = [
        Obs(
            "tool_index_remap",
            "pass"
            if before["tool_indices"] == [1, 3] and after["tool_indices"] == [0, 1]
            else "fail",
            inputs,
            before["tool_indices"],
            after["tool_indices"],
        ),
        Obs(
            "text_and_stop_preserved",
            "pass"
            if before["texts"] == after["texts"] == ["Hi"]
            and before["finish_reason"] == after["finish_reason"]
            else "fail",
            inputs,
            {"texts": before["texts"], "finish": before["finish_reason"]},
            {"texts": after["texts"], "finish": after["finish_reason"]},
        ),
    ]
    return checks


def _run8512_tools() -> list[Obs]:
    case = "berriai__litellm-8512"
    import litellm

    def supports_tool_choice(model: str, custom_llm_provider: str) -> bool:
        return custom_llm_provider == "bedrock"

    litellm.utils = types.SimpleNamespace(supports_tool_choice=supports_tool_choice)

    class BedrockModelInfo:
        @staticmethod
        def get_base_model(model: str) -> str:
            return model.split("/")[-1] if "/" in model else model

    extra = {
        "BedrockModelInfo": BedrockModelInfo,
        "litellm": litellm,
        "List": list,
    }
    models = {
        "new_target": "meta.llama3-3-70b-instruct-v1:0",
        "known_supported": "meta.llama3-2-1b",
        "known_unsupported": "amazon.titan-text-express-v1",
    }

    def tools_for(variant: str, model: str) -> bool:
        fn = extract_function_from_pinned(
            case,
            variant,
            "llms/bedrock/chat/converse_transformation.py",
            "get_supported_openai_params",
            extra,
        )
        self = types.SimpleNamespace(custom_llm_provider="bedrock")
        return "tools" in fn(self, model)

    before = {k: tools_for("before", m) for k, m in models.items()}
    after = {k: tools_for("applied", m) for k, m in models.items()}
    return [
        Obs(
            "llama33_tools_support",
            "pass" if not before["new_target"] and after["new_target"] else "fail",
            models,
            before["new_target"],
            after["new_target"],
        ),
        Obs(
            "supported_and_unsupported_unchanged",
            "pass"
            if before["known_supported"]
            and after["known_supported"]
            and not before["known_unsupported"]
            and not after["known_unsupported"]
            else "fail",
            models,
            before,
            after,
        ),
    ]


def _vertex_extra(case: str, variant: str) -> dict[str, Any]:
    rel = "llms/vertex_ai/common_utils.py"
    unpack = extract_function_from_pinned(
        "berriai__litellm-11964",
        "applied",
        "litellm_core_utils/prompt_templates/common_utils.py",
        "unpack_defs",
        {"Set": set, "Any": object},
    )
    extra: dict[str, Any] = {
        "Dict": dict,
        "Any": object,
        "Set": set,
        "get_type_hints": lambda _schema: {
            "type": None,
            "properties": None,
            "maxLength": None,
            "minimum": None,
            "maximum": None,
            "enum": None,
            "required": None,
            "items": None,
            "anyOf": None,
        },
        "unpack_defs": unpack,
        "Schema": dict,
        "List": list,
        "Optional": __import__("typing").Optional,
        "Union": __import__("typing").Union,
        "Tuple": tuple,
        "Literal": __import__("typing").Literal,
        "re": __import__("re"),
        "DEFAULT_MAX_RECURSE_DEPTH": 10,
    }
    for fn_name in (
        "convert_anyof_null_to_nullable",
        "add_object_type",
        "filter_schema_fields",
    ):
        extra[fn_name] = extract_function_from_pinned(case, variant, rel, fn_name, extra)
    if variant == "applied":
        extra["set_schema_property_ordering"] = extract_function_from_pinned(
            case, variant, rel, "set_schema_property_ordering", extra
        )
    return extra


def _run9828_schema_order() -> list[Obs]:
    case = "berriai__litellm-9828"

    nested_order = {
        "type": "object",
        "properties": {
            "z": {"type": "string"},
            "child": {
                "type": "object",
                "properties": {"inner": {"type": "integer"}},
            },
        },
    }
    nested_constraints = {
        "type": "object",
        "properties": {
            "z": {"type": "string", "maxLength": 5},
            "nested": {
                "type": "object",
                "properties": {"inner": {"type": "integer", "minimum": 2}},
            },
        },
        "required": ["z"],
    }

    def build(variant: str, ordering: bool, template: dict[str, Any]) -> dict[str, Any]:
        extra = _vertex_extra(case, variant)
        fn = extract_function_from_pinned(
            case, variant, "llms/vertex_ai/common_utils.py", "_build_vertex_schema", extra
        )
        params = copy.deepcopy(template)
        if variant == "before":
            return fn(params)
        return fn(params, add_property_ordering=ordering)

    before_out = build("before", False, nested_order)
    after_no = build("applied", False, nested_order)
    after_yes = build("applied", True, nested_order)
    before_constraints = build("before", False, nested_constraints)
    after_constraints = build("applied", True, nested_constraints)

    return [
        Obs(
            "property_ordering_when_enabled",
            "pass"
            if after_yes.get("propertyOrdering") == ["z", "child"]
            and "propertyOrdering" not in before_out
            else "fail",
            {"nested": nested_order, "flag": True},
            before_out.get("propertyOrdering"),
            after_yes.get("propertyOrdering"),
        ),
        Obs(
            "constraints_preserved",
            "pass"
            if before_constraints["properties"]["z"].get("maxLength") == 5
            and after_constraints["properties"]["z"].get("maxLength") == 5
            and after_constraints["properties"]["nested"]["properties"]["inner"].get("minimum") == 2
            else "fail",
            nested_constraints,
            before_constraints["properties"],
            after_constraints["properties"],
        ),
        Obs(
            "no_property_ordering_without_flag",
            "pass"
            if "propertyOrdering" not in after_no and "propertyOrdering" not in before_out
            else "fail",
            {"template": nested_order},
            before_out.get("propertyOrdering"),
            after_no.get("propertyOrdering"),
        ),
    ]


def _run11964_unpack() -> list[Obs]:
    extra = {"Set": set, "Any": object}

    def run_unpack(variant: str, schema: dict[str, Any]) -> dict[str, Any]:
        fn = extract_function_from_pinned(
            "berriai__litellm-11964",
            variant,
            "litellm_core_utils/prompt_templates/common_utils.py",
            "unpack_defs",
            extra,
        )
        s = copy.deepcopy(schema)
        fn(s, dict(s.get("$defs", {})))
        return s

    schema = {
        "properties": {
            "payload": {
                "type": "array",
                "items": {"anyOf": [{"$ref": "#/defs/Leaf"}]},
            },
            "unchanged": {"type": "number", "const": 7},
        },
        "$defs": {
            "Leaf": {"type": "string", "enum": ["x"]},
        },
    }
    before = run_unpack("before", schema)
    after = run_unpack("applied", schema)
    b_leaf = (
        before["properties"]["payload"]["items"]["anyOf"][0].get("$ref")
        or before["properties"]["payload"]["items"]["anyOf"][0].get("type")
    )
    a_leaf = after["properties"]["payload"]["items"]["anyOf"][0].get("type")
    return [
        Obs(
            "nested_anyof_items_ref",
            "pass" if isinstance(b_leaf, str) and b_leaf.startswith("#") and a_leaf == "string" else "fail",
            schema,
            before["properties"]["payload"]["items"],
            after["properties"]["payload"]["items"],
        ),
        Obs(
            "unrelated_property_preserved",
            "pass"
            if before["properties"]["unchanged"] == after["properties"]["unchanged"]
            else "fail",
            schema,
            before["properties"]["unchanged"],
            after["properties"]["unchanged"],
        ),
    ]


def _run9261_ollama() -> list[Obs]:
    case = "berriai__litellm-9261"

    class BadRequestError(Exception):
        def __init__(self, message: str = "", model: str = "", llm_provider: str = ""):
            super().__init__(message)

    extras = {
        "json": json,
        "BAD_MESSAGE_ERROR_STR": "bad message",
        "litellm": types.SimpleNamespace(BadRequestError=BadRequestError),
    }

    def prompt(variant: str, messages: list[dict[str, Any]]) -> str | None:
        fn = extract_function_from_pinned(
            case,
            variant,
            "litellm_core_utils/prompt_templates/factory.py",
            "ollama_pt",
            extras,
        )
        try:
            out = fn("llama3", messages)
            if isinstance(out, dict):
                return out["prompt"]
            return str(out)
        except BadRequestError:
            return None

    ua = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    sys_end = ua + [{"role": "system", "content": "tail"}]
    sys_begin = [{"role": "system", "content": "head"}] + ua
    multi = [{"role": "user", "content": "u"}] + [
        {"role": "system", "content": "s1"},
        {"role": "system", "content": "s2"},
    ]

    b_ua, a_ua = prompt("before", ua), prompt("applied", ua)
    b_begin, a_begin = prompt("before", sys_begin), prompt("applied", sys_begin)
    b_end, a_end = prompt("before", sys_end), prompt("applied", sys_end)
    b_multi, a_multi = prompt("before", multi), prompt("applied", multi)

    return [
        Obs(
            "user_assistant_format_unchanged",
            "pass" if b_ua == a_ua and "### User:" in (a_ua or "") and "### Assistant:" in (a_ua or "") else "fail",
            {"messages": ua},
            b_ua,
            a_ua,
        ),
        Obs(
            "system_at_beginning",
            "pass"
            if a_begin and "### System:" in a_begin and "head" in a_begin
            else "fail",
            {"messages": sys_begin},
            b_begin,
            a_begin,
        ),
        Obs(
            "system_after_user_multiple",
            "pass"
            if a_multi and "### System:" in a_multi and "s1" in a_multi and "s2" in a_multi
            else "fail",
            {"messages": multi},
            b_multi,
            a_multi,
            detail="before may stall without system merge; after must merge consecutive system roles",
        ),
        Obs(
            "system_at_end_not_false_preservation",
            "pass"
            if b_end is None and a_end and "### System:" in a_end
            else "fail",
            {"messages": sys_end},
            b_end,
            a_end,
            detail="before BadRequestError on trailing system is failure mode, not preserved behavior",
        ),
    ]


def _run12168_tool_choice() -> list[Obs]:
    case = "berriai__litellm-12168"
    typing = __import__("typing")
    extra = {"Optional": typing.Optional, "Union": typing.Union}

    def load(variant: str):
        return extract_function_from_pinned(
            case, variant, "utils.py", "validate_chat_completion_tool_choice", extra
        )

    fb, fa = load("before"), load("applied")
    cases = {
        "dict_auto": {"type": "auto"},
        "dict_none": {"type": "none"},
        "string_auto": "auto",
        "named_function": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
    }

    def run(fn, payload):
        try:
            return ("ok", fn(payload))
        except Exception as e:
            return ("err", str(e))

    obs: list[Obs] = []
    for name, payload in cases.items():
        bb, ba = run(fb, payload), run(fa, payload)
        if name == "dict_auto":
            st = "pass" if bb[0] == "err" and ba[0] == "ok" and ba[1] == payload else "fail"
        elif name == "dict_none":
            st = "pass" if bb[0] == "err" and ba[0] == "ok" and ba[1] == payload else "fail"
        elif name in ("string_auto", "named_function"):
            st = "pass" if bb[0] == "ok" and ba[0] == "ok" and ba[1] == payload else "fail"
        else:
            st = "fail"
        obs.append(Obs(name, st, {"tool_choice": payload}, bb, ba))
    return obs


def _pending(case_id: str, reason: str) -> dict[str, Any]:
    return {
        "source_id": case_id,
        "status": "pending",
        "reason": reason,
        "checks": [],
    }


def _run_pending_cases() -> dict[str, Any]:
    return {
        "berriai__litellm-6138": _pending(
            "berriai__litellm-6138",
            "AzureChatCompletion.completion path requires full azure.py import graph; not isolated in pinned tree",
        ),
        "berriai__litellm-11006": _pending(
            "berriai__litellm-11006",
            "CustomStreamWrapper.set_model_id requires streaming_handler module bootstrap beyond bounded stub budget",
        ),
        "berriai__litellm-6563": _pending(
            "berriai__litellm-6563",
            "LangFuseLogger._prepare_metadata recursive self binding needs full integration module load",
        ),
    }


RUNNERS: dict[str, Callable[[], list[Obs]]] = {
    "berriai__litellm-5232": _run5232_stream,
    "berriai__litellm-8512": _run8512_tools,
    "berriai__litellm-9828": _run9828_schema_order,
    "berriai__litellm-11964": _run11964_unpack,
    "berriai__litellm-9261": _run9261_ollama,
    "berriai__litellm-12168": _run12168_tool_choice,
}


def main() -> int:
    curated_path = STRICT_ROOT / "curated_examples.json"
    if not curated_path.is_file():
        src = json.loads((BATCH / "curated_examples.json").read_text(encoding="utf-8"))
        curated_path.write_text(json.dumps(src, indent=2) + "\n", encoding="utf-8")

    curated = json.loads(curated_path.read_text(encoding="utf-8"))
    accepted = [row["source_id"] for row in curated]

    results: dict[str, Any] = {
        "generated_by": "litellm_strict/verify_batch.py",
        "method": "pinned_source_callable_before_after_execution",
        "repo_root": str(REPO),
        "cases": {},
        "summary": {},
    }

    failures = 0
    pending = _run_pending_cases()
    for sid in accepted:
        if sid in pending:
            results["cases"][sid] = pending[sid]
            continue
        runner = RUNNERS.get(sid)
        if not runner:
            results["cases"][sid] = _pending(sid, "no_strict_runner_implemented")
            continue
        checks: list[Obs] = []
        case_status = "pass"
        try:
            checks = runner()
            for c in checks:
                if c.status != "pass":
                    case_status = "fail"
                    failures += 1
        except Exception as e:
            case_status = "error"
            failures += 1
            checks = [Obs("exception", "fail", {}, None, None, str(e))]
        results["cases"][sid] = {
            "status": case_status,
            "checks": [c.to_json() for c in checks],
        }

    results["summary"] = {
        "accepted_in_curated": len(accepted),
        "strict_pass_cases": sum(
            1 for sid in accepted if results["cases"].get(sid, {}).get("status") == "pass"
        ),
        "strict_pending_cases": sum(
            1 for sid in accepted if results["cases"].get(sid, {}).get("status") == "pending"
        ),
        "failed_checks": failures,
        "overall": "pass" if failures == 0 else "fail",
    }

    out = STRICT_ROOT / "verification_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Update verification_status on curated rows
    for row in curated:
        sid = row["source_id"]
        st = results["cases"].get(sid, {}).get("status", "missing")
        if st == "pass":
            row["verification_status"] = "strict_pinned_before_after_behavior_pass"
        elif st == "pending":
            row["verification_status"] = "strict_pending_isolated_import"
        else:
            row["verification_status"] = f"strict_behavior_{st}"
    curated_path.write_text(json.dumps(curated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(results["summary"], indent=2))
    return 0 if results["summary"]["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
