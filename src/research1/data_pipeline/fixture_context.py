"""Read-only applicability context from an ORIGINAL frozen probe source file.

This adds test visibility, not tests, scope, attribution, a patch or a PASS.
It neither imports the snapshot as executable code nor runs any probe/model/task.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from types import CodeType

SCHEMA = "research1.original-prefrozen-fixture-source.v1"
SYMBOLS = ("frozen_registry", "_scope", "probe_module", "_probe_in_process")
RELATIVE_SOURCE = "src/research1/data_pipeline/modules.py"
STRICT_RELATIVE_SOURCE = "src/research1/data_pipeline/modules_v3.py"
LIMITATIONS = [
    "Read-only ORIGINAL preteacher fixture source; no test, threshold, scope, parent probe result or candidate plan is changed.",
    "The source shows exactly how the unchanged synthetic goal and preservation checks are evaluated. Existing IDs or a matching mechanism name alone do not establish applicability.",
    "Synthetic completion checks establish bounded prompt/context plumbing for their shown fixtures; they do not establish solver compliance, complete task correctness, future score gain or long-horizon utility.",
    "Determine whether the candidate's described implementation and own verification requirements fit these exact checks. An unimplemented plan need not already supply a child patch or passing child result.",
    "Do not infer that the plan fits merely because fixture source is supplied. Missing scope, infeasible runtime dataflow, contradictory bounds or incomplete behavioral coverage must remain pending.",
    "Do not rewrite the plan, invent another hook, relax a bound, add a check or demand prior proof of an explicitly hypothetical future benefit.",
    "This context belongs only to independent applicability review. It must not enter teacher generation, student input or raw plan answers.",
]


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path, *, max_bytes=250_000):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Fixture source paths must be absolute")
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValueError("Fixture source cannot follow symlinks")
        cursor = cursor.parent
    if not path.is_file():
        raise ValueError("Fixture source must be an existing regular file")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError("Fixture source exceeds its read-only bound")
    return raw


def _constant(value):
    if isinstance(value, CodeType):
        return {"code": _code_identity(value)}
    if isinstance(value, tuple):
        return {"tuple": [_constant(item) for item in value]}
    if isinstance(value, frozenset):
        return {"frozenset": sorted((_constant(item) for item in value), key=_canonical)}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    return {"type": type(value).__name__, "literal": repr(value)}


def _code_identity(code):
    """Executable fingerprint excluding filenames and source-position tables."""
    return {"bytecode": code.co_code.hex(), "constants": [_constant(item) for item in code.co_consts],
        "names": code.co_names, "varnames": code.co_varnames, "freevars": code.co_freevars,
        "cellvars": code.co_cellvars, "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount, "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals, "stacksize": code.co_stacksize, "flags": code.co_flags,
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex()}


def _build(source_path, executing_module):
    path = Path(source_path)
    raw = _read_regular(path)
    snapshot_roots = [parent for parent in path.parents if parent.name == "source_snapshot"]
    if len(snapshot_roots) != 1 or path.relative_to(snapshot_roots[0]).as_posix() not in {RELATIVE_SOURCE,STRICT_RELATIVE_SOURCE}:
        raise ValueError("Fixture source must be the original campaign source_snapshot modules.py")
    relative_source=path.relative_to(snapshot_roots[0]).as_posix()
    from research1.data_pipeline.module_adapters import LEGACY_VERSION, STRICT_VERSION
    strict=relative_source==STRICT_RELATIVE_SOURCE
    expected_version=STRICT_VERSION if strict else LEGACY_VERSION
    if getattr(executing_module,'REGISTRY_VERSION',None)!=expected_version:
        raise ValueError('Fixture source and executing registry adapter differ')
    protocol_path = snapshot_roots[0].parent / "protocol.json"
    protocol_raw = _read_regular(protocol_path, max_bytes=2_000_000)
    protocol = json.loads(protocol_raw)
    protocol_body = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if protocol.get("protocol_hash") != _sha(_canonical(protocol_body).encode("utf-8")):
        raise ValueError("Original source protocol content binding changed")
    source_sha = _sha(raw)
    if protocol.get("source_sha256", {}).get(relative_source) != source_sha:
        raise ValueError("Fixture source is not the original protocol-frozen modules.py bytes")
    if strict and protocol.get('verification_registry_version') not in {STRICT_VERSION,'v3'}:
        raise ValueError('New fixture source requires its explicitly frozen registry version')
    executing_path = Path(executing_module.__file__).resolve()
    if _sha(_read_regular(executing_path)) != source_sha:
        raise ValueError("Currently executed probe module file differs from the original snapshot")
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    compiled = compile(source, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    compiled_functions = {item.co_name: item for item in compiled.co_consts if isinstance(item, CodeType)}
    fragments = []
    for name in SYMBOLS:
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(nodes) != 1 or name not in compiled_functions:
            raise ValueError("Original fixture function missing or ambiguous: " + name)
        node = nodes[0]
        start = min([node.lineno] + [item.lineno for item in node.decorator_list])
        text = "\n".join(source.splitlines()[start - 1:node.end_lineno])
        loaded = getattr(executing_module, name, None)
        if not callable(loaded) or not hasattr(loaded, "__code__"):
            raise ValueError("Executing fixture function is not an inspectable Python function")
        expected_code = _sha(_canonical(_code_identity(compiled_functions[name])).encode("utf-8"))
        actual_code = _sha(_canonical(_code_identity(loaded.__code__)).encode("utf-8"))
        if expected_code != actual_code:
            raise ValueError("Loaded executing fixture code differs from original snapshot: " + name)
        fragments.append({"symbol": name, "start_line": start, "end_line": node.end_lineno,
            "sha256": _sha(text.encode("utf-8")),
            "ast_sha256": _sha(ast.dump(node, include_attributes=False).encode("utf-8")),
            "executing_code_sha256": actual_code, "text": text, "access": "read_only"})
    if protocol.get("registry_sha256") != _sha(_canonical(executing_module.frozen_registry()).encode("utf-8")):
        raise ValueError("Current unchanged registry differs from the original frozen registry")
    probe = next(fragment for fragment in fragments if fragment["symbol"] == "_probe_in_process")
    context = {"schema": SCHEMA, "path": str(path), "sha256": source_sha,
        "function_sha256": probe["sha256"], "text": "\n\n".join(fragment["text"] for fragment in fragments),
        "fragments": fragments, "executing_module_path": str(executing_path),
        "original_protocol_path": str(protocol_path), "original_protocol_sha256": _sha(protocol_raw),
        "original_protocol_hash": protocol["protocol_hash"], "registry_sha256": protocol.get("registry_sha256"),
        "read_only": True, "limitations": list(LIMITATIONS)}
    if strict:
        context['verification_registry_version']=STRICT_VERSION
    context["context_id"] = "fixtures_" + _sha(_canonical(context).encode("utf-8"))
    return context


def freeze_probe_contract(source_path, *, executing_module=None):
    if executing_module is None:
        from research1.data_pipeline.module_adapters import get_adapter, STRICT_VERSION
        executing_module=get_adapter(STRICT_VERSION if Path(source_path).name=='modules_v3.py' else None)
    return _build(Path(source_path), executing_module)


def verify_probe_contract(frozen, *, executing_module=None):
    """Re-read original source/protocol and verify current loaded functions."""
    if not isinstance(frozen, dict) or frozen.get("schema") != SCHEMA:
        raise ValueError("Unsupported applicability fixture source context")
    rebuilt = freeze_probe_contract(Path(frozen["path"]), executing_module=executing_module)
    if rebuilt != frozen:
        raise ValueError("Frozen applicability fixture source context changed")
    return rebuilt


def build_fixture_review_context(frozen, *, executing_module=None):
    return verify_probe_contract(frozen, executing_module=executing_module)
