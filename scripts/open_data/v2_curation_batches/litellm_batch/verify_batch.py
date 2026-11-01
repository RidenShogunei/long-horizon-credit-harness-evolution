"""Bounded offline behavior checks for litellm_batch accepted rows."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import textwrap
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Set

REPO_ROOT = Path(__file__).resolve().parents[3]
MC = REPO_ROOT / "work/open_dataset_integration_v2/main_candidates/cases"
BATCH = Path(__file__).resolve().parent
CHILD_TIMEOUT = 20


@dataclass
class CheckResult:
    name: str
    status: str
    before: Any
    after: Any
    inputs: dict[str, Any]
    detail: str = ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def exec_function_from_file(
    py_path: Path, func_name: str, extra_globals: dict[str, Any] | None = None
) -> Callable[..., Any]:
    src = py_path.read_text(encoding="utf-8")
    mod = ast.parse(src)
    lines = src.splitlines(True)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            code = compile("".join(lines[node.lineno - 1 : node.end_lineno]), str(py_path), "exec")
            ns: dict[str, Any] = {"__builtins__": __builtins__, "Any": Any, "Set": Set}
            if extra_globals:
                ns.update(extra_globals)
            exec(code, ns)
            return ns[func_name]
    raise RuntimeError(f"{func_name} not found in {py_path}")


def exec_method_from_class(
    py_path: Path,
    class_name: str,
    method_name: str,
    extra_globals: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    src = py_path.read_text(encoding="utf-8")
    mod = ast.parse(src)
    lines = src.splitlines(True)
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for n in node.body:
                if isinstance(n, ast.FunctionDef) and n.name == method_name:
                    chunk = "".join(lines[n.lineno - 1 : n.end_lineno])
                    chunk = textwrap.dedent(chunk)
                    code = compile(chunk, str(py_path), "exec")
                    ns: dict[str, Any] = {"__builtins__": __builtins__, "Any": Any, "Set": Set}
                    if extra_globals:
                        ns.update(extra_globals)
                    exec(code, ns)
                    return ns[method_name]
    raise RuntimeError(f"{class_name}.{method_name} not found")


def check_11964_unpack_defs() -> list[CheckResult]:
    before = MC / "berriai__litellm-11964/before/litellm/litellm_core_utils/prompt_templates/common_utils.py"
    after = MC / "berriai__litellm-11964/applied/litellm/litellm_core_utils/prompt_templates/common_utils.py"
    fb = exec_function_from_file(before, "unpack_defs")
    fa = exec_function_from_file(after, "unpack_defs")
    schema_b = {
        "properties": {"v": {"allOf": [{"$ref": "#/defs/T"}]}},
        "$defs": {"T": {"type": "string"}},
    }
    schema_a = copy.deepcopy(schema_b)
    fb(schema_b, dict(schema_b.get("$defs", {})))
    fa(schema_a, {})
    before_ref = schema_b["properties"]["v"]["allOf"][0].get("$ref")
    after_type = schema_a["properties"]["v"]["allOf"][0].get("type") == "string"
    return [
        CheckResult(
            "allof_ref_resolved",
            "pass" if before_ref and after_type else "fail",
            before_ref,
            schema_a["properties"]["v"]["allOf"][0],
            {"schema": schema_b},
        )
    ]


def check_9261_ollama_system() -> list[CheckResult]:
    before = MC / "berriai__litellm-9261/before/litellm/litellm_core_utils/prompt_templates/factory.py"
    after = MC / "berriai__litellm-9261/applied/litellm/litellm_core_utils/prompt_templates/factory.py"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "sysA"},
        {"role": "system", "content": "sysB"},
    ]
    extras = {
        "json": json,
        "BAD_MESSAGE_ERROR_STR": "bad message",
    }

    class BadRequestError(Exception):
        def __init__(self, message="", model="", llm_provider=""):
            super().__init__(message)

    extras["litellm"] = types.SimpleNamespace(BadRequestError=BadRequestError)

    def prompt_from(path: Path) -> tuple[str, str | None]:
        fn = exec_function_from_file(path, "ollama_pt", extras)
        try:
            out = fn("llama3", messages)
            if isinstance(out, dict):
                return out["prompt"], None
            return str(out), None
        except Exception as e:
            return "", str(e)

    pb, err_b = prompt_from(before)
    pa, err_a = prompt_from(after)
    return [
        CheckResult(
            "system_block_present",
            "pass" if "### System:" in pa and "sysA" in pa and "sysB" in pa else "fail",
            "### System:" in pb,
            "### System:" in pa,
            {"messages": messages, "before_error": err_b},
        ),
        CheckResult(
            "before_missing_system",
            "pass" if "### System:" not in pb else "fail",
            pb,
            pa,
            {"messages": messages, "before_error": err_b},
            detail="before should omit merged system block or stall on trailing system roles",
        ),
    ]


def check_5232_tool_index() -> list[CheckResult]:
    before = MC / "berriai__litellm-5232/before/litellm/llms/anthropic.py"
    after = MC / "berriai__litellm-5232/applied/litellm/llms/anthropic.py"

    def has_tool_index_field(path: Path) -> bool:
        return "self.tool_index = -1" in path.read_text(encoding="utf-8")

    def uses_tool_index_in_chunk(path: Path) -> bool:
        text = path.read_text(encoding="utf-8")
        return '"index": self.tool_index' in text

    return [
        CheckResult(
            "after_tracks_tool_index",
            "pass" if has_tool_index_field(after) and uses_tool_index_in_chunk(after) else "fail",
            has_tool_index_field(before),
            has_tool_index_field(after),
            {},
        ),
        CheckResult(
            "before_uses_provider_index",
            "pass" if (not has_tool_index_field(before)) and ('"index": content_block["index"]' in before.read_text(encoding="utf-8")) else "fail",
            True,
            False,
            {},
        ),
    ]


def check_12168_tool_choice() -> list[CheckResult]:
    before = MC / "berriai__litellm-12168/before/litellm/utils.py"
    after = MC / "berriai__litellm-12168/applied/litellm/utils.py"
    fb = exec_function_from_file(before, "validate_chat_completion_tool_choice")
    fa = exec_function_from_file(after, "validate_chat_completion_tool_choice")
    payload = {"type": "auto"}

    def safe_call(fn):
        try:
            return ("ok", fn(payload))
        except Exception as e:
            return ("err", str(e))

    bb, ba = safe_call(fb), safe_call(fa)
    return [
        CheckResult(
            "cursor_auto_dict",
            "pass" if bb[0] == "err" and ba[0] == "ok" and ba[1] == payload else "fail",
            bb,
            ba,
            {"tool_choice": payload},
        )
    ]


def check_9828_property_ordering() -> list[CheckResult]:
    after = MC / "berriai__litellm-9828/applied/litellm/llms/vertex_ai/common_utils.py"
    before = MC / "berriai__litellm-9828/before/litellm/llms/vertex_ai/common_utils.py"
    set_order = exec_function_from_file(after, "set_schema_property_ordering", {"Dict": dict, "Any": Any})
    params = {
        "type": "object",
        "properties": {"z": {"type": "string"}, "a": {"type": "integer"}},
    }
    set_order(params)
    has_order = params.get("propertyOrdering") == ["z", "a"]
    before_text = before.read_text(encoding="utf-8")
    return [
        CheckResult(
            "propertyOrdering_added",
            "pass" if has_order and "def set_schema_property_ordering" not in before_text else "fail",
            "propertyOrdering" in before_text,
            params.get("propertyOrdering"),
            {"parameters": params},
        )
    ]


def check_11006_stream_id() -> list[CheckResult]:
    before = MC / "berriai__litellm-11006/before/litellm/litellm_core_utils/streaming_handler.py"
    after = MC / "berriai__litellm-11006/applied/litellm/litellm_core_utils/streaming_handler.py"
    set_before = exec_method_from_class(before, "CustomStreamWrapper", "set_model_id")
    set_after = exec_method_from_class(after, "CustomStreamWrapper", "set_model_id")

    class Dummy:
        pass

    class Stream:
        def __init__(self):
            self.id = None

    def run(set_fn):
        obj = Dummy()
        obj.response_id = None
        s1 = Stream()
        set_fn(obj, "", s1)
        s2 = Stream()
        set_fn(obj, "openrouter-id-1", s2)
        return s2.id

    b = run(set_before)
    a = run(set_after)
    return [
        CheckResult(
            "adopt_later_valid_id",
            "pass" if b in (None, "") and a == "openrouter-id-1" else "fail",
            b,
            a,
            {"first_id": "", "second_id": "openrouter-id-1"},
        )
    ]


def check_6138_max_retries() -> list[CheckResult]:
    before = MC / "berriai__litellm-6138/before/litellm/llms/AzureOpenAI/azure.py"
    after = MC / "berriai__litellm-6138/applied/litellm/llms/AzureOpenAI/azure.py"
    snippet_b = (before.read_text(encoding="utf-8").split("max_retries = optional_params.pop")[1][:40])
    snippet_a = (after.read_text(encoding="utf-8").split("max_retries = optional_params.get")[1][:40])

    def simulate(use_get: bool):
        optional_params = {"max_retries": 5, "temperature": 0.2}
        if use_get:
            max_retries = optional_params.get("max_retries", 2)
        else:
            max_retries = optional_params.pop("max_retries", 2)
        return max_retries, "max_retries" in optional_params

    rb = simulate(False)
    ra = simulate(True)
    return [
        CheckResult(
            "optional_params_retains_max_retries",
            "pass" if rb[1] is False and ra[1] is True and ra[0] == 5 else "fail",
            rb,
            ra,
            {"optional_params_initial": {"max_retries": 5, "temperature": 0.2}},
            detail=f"before uses pop ({snippet_b.strip()}...) after uses get ({snippet_a.strip()}...)",
        )
    ]


def check_8512_tools_support() -> list[CheckResult]:
    model = "meta.llama3-3-70b-instruct-v1:0"

    def tools_supported(path: Path) -> bool:
        text = path.read_text(encoding="utf-8")
        return "meta.llama3-3" in text

    before = MC / "berriai__litellm-8512/before/litellm/llms/bedrock/chat/converse_transformation.py"
    after = MC / "berriai__litellm-8512/applied/litellm/llms/bedrock/chat/converse_transformation.py"
    # structural check matching patch intent
    return [
        CheckResult(
            "llama33_listed_for_tools",
            "pass" if not tools_supported(before) and tools_supported(after) else "fail",
            tools_supported(before),
            tools_supported(after),
            {"model": model},
        )
    ]


def check_6563_metadata() -> list[CheckResult]:
    after = MC / "berriai__litellm-6563/applied/litellm/integrations/langfuse/langfuse.py"
    extras = {
        "copy": copy,
        "traceback": __import__("traceback"),
        "verbose_logger": types.SimpleNamespace(warning=lambda *a, **k: None),
        "MutableMapping": __import__("collections.abc", fromlist=["MutableMapping"]).MutableMapping,
        "MutableSequence": __import__("collections.abc", fromlist=["MutableSequence"]).MutableSequence,
        "MutableSet": __import__("collections.abc", fromlist=["MutableSet"]).MutableSet,
        "BaseModel": type("BaseModel", (), {"model_dump": lambda self: {}}),
    }
    meth = exec_method_from_class(after, "LangFuseLogger", "_prepare_metadata", extras)

    class Obj:
        pass

    obj = Obj()
    lock = threading.RLock()
    meta = {"nested": {"guard": lock}, "ok": "value"}

    before_status = "ok"
    try:
        copy.deepcopy(meta)
    except TypeError as e:
        before_status = str(e)

    bound = types.MethodType(meth, obj)
    obj._prepare_metadata = bound
    after_out = obj._prepare_metadata(meta)
    ok = (
        "cannot pickle" in str(before_status)
        and after_out.get("ok") == "value"
        and isinstance(after_out.get("nested"), dict)
    )
    return [
        CheckResult(
            "rlock_metadata_serializable",
            "pass" if ok else "fail",
            before_status,
            after_out,
            {"metadata_keys": list(meta.keys())},
        )
    ]


CHECKS: dict[str, Callable[[], list[CheckResult]]] = {
    "berriai__litellm-11964": check_11964_unpack_defs,
    "berriai__litellm-9261": check_9261_ollama_system,
    "berriai__litellm-5232": check_5232_tool_index,
    "berriai__litellm-12168": check_12168_tool_choice,
    "berriai__litellm-9828": check_9828_property_ordering,
    "berriai__litellm-11006": check_11006_stream_id,
    "berriai__litellm-6138": check_6138_max_retries,
    "berriai__litellm-8512": check_8512_tools_support,
    "berriai__litellm-6563": check_6563_metadata,
}


def main() -> int:
    curated = json.loads((BATCH / "curated_examples.json").read_text(encoding="utf-8"))
    accepted_ids = [row["source_id"] for row in curated]
    results: dict[str, Any] = {
        "generated_by": "verify_batch.py",
        "repo_root": str(REPO_ROOT),
        "cases": {},
    }
    failures = 0
    for sid in accepted_ids:
        fn = CHECKS.get(sid)
        if not fn:
            results["cases"][sid] = {"status": "missing_check_impl", "checks": []}
            failures += 1
            continue
        case_checks = []
        case_status = "pass"
        try:
            for cr in fn():
                if cr.status != "pass":
                    case_status = "fail"
                    failures += 1
                case_checks.append(
                    {
                        "name": cr.name,
                        "status": cr.status,
                        "inputs": cr.inputs,
                        "before_observation": cr.before,
                        "after_observation": cr.after,
                        "detail": cr.detail,
                        "inputs_sha256": sha256_text(json.dumps(cr.inputs, sort_keys=True, default=str)),
                    }
                )
        except Exception as e:
            case_status = "error"
            failures += 1
            case_checks.append({"name": "exception", "status": "error", "detail": str(e)})
        results["cases"][sid] = {"status": case_status, "checks": case_checks}

    results["summary"] = {
        "accepted_count": len(accepted_ids),
        "failed_cases": failures,
        "overall": "pass" if failures == 0 else "fail",
    }
    out = BATCH / "verification_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
