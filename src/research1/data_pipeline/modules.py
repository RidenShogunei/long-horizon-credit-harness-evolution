"""Five-module implementation adapters and pre-frozen executable contracts.

These are narrow mechanism contracts, not a universal semantic verifier. A
failing synthetic target fixture is insufficient attribution: selection also
requires proposal-visible evidence (and independent diagnostic review for new
mechanisms). Unknown or unsupported plans remain pending.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any, Callable

MODULES = ("agent_loop", "observation", "tools", "context_mgmt", "verification")
REGISTRY_VERSION = "research1.module-contracts.v2"
EDITOR_POLICY = "five-module-exact-replacements.v1"
PLAN_BINDING_POLICY = "proposal_single_plan_v6"
PLAN_BINDING_MODE = "independent_plan_to_prefrozen_checks.v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError("modules_root is not a directory")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("module snapshots cannot contain symlinks")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not hashes:
        raise ValueError("empty module snapshot")
    return hashes


def code_hash(root: Path) -> str:
    """Content identity of the entire module tree, not just the edited file."""
    return _digest(tree_hashes(Path(root)))


def frozen_registry() -> dict[str, Any]:
    """Versioned narrow goals plus mandatory preservation cases for all modules."""
    packs = {
        "false_concat_preservation": {
            "module": "tools", "mechanism_id": "false_concat_preservation",
            "goal_checks": ["empty_wait_mid_batch", "mixed_empty_false_absent"],
            "preserve_checks": ["nonempty_missing_newline", "well_formed_commands", "mixed_real_present", "nonempty_space", "nonempty_tab", "parser_fields_preserved"],
            "functions": ["BaselineTools.parse_llm_response", "_suppress_empty_keystroke_newline_warning", "_warning_refers_to_empty_keystrokes"],
            "globals": ["NEWLINE_WARNING_SUPPRESSED_EMPTY_KEYSTROKES", "_NEWLINE_WARNING_RE"],
            "signal": "empty_wait_newline_warning", "diagnostic_review_required": False,
            "coverage": "Only empty-wait newline warnings; full command/error/completion preservation, mixed real warnings and whitespace inputs.",
        },
        "verification_error_gate": {
            "module": "verification", "mechanism_id": "verification_error_gate",
            "goal_checks": ["failed_tool_retracts_completion"],
            "preserve_checks": ["first_signal_waits", "second_signal_confirms", "reset_signal_waits", "custom_required_count"],
            "functions": ["BaselineVerification.should_terminate", "_research1_completion_error"],
            "globals": [], "signal": "completion_after_tool_error", "diagnostic_review_required": True,
            "coverage": "Gate completed states containing an explicit failed last tool result, preserving normal two-signal and configurable confirmation.",
        },
        "observation_error_anchor": {
            "module": "observation", "mechanism_id": "observation_error_anchor",
            "goal_checks": ["middle_error_visible"],
            "preserve_checks": ["short_output_exact", "utf8_valid", "tail_retained", "head_retained", "bounded_capture", "no_session_quiet", "capture_error_quiet", "incremental_capture"],
            "functions": ["_smart_truncate", "_research1_error_anchors"],
            "globals": [], "signal": "truncated_error_context", "diagnostic_review_required": True,
            "coverage": "Preserve a single middle error anchor under bounded truncation; keep short bytes, UTF8 boundaries, head/tail and incremental capture behavior.",
        },
        "context_recent_history": {
            "module": "context_mgmt", "mechanism_id": "context_recent_history",
            "goal_checks": ["recent_exchange_retained"],
            "preserve_checks": ["system_retained", "budget_reached", "whole_exchanges", "chain_reset", "disabled_noop", "roomy_noop", "instructionless_noop", "context_failure_noop"],
            "functions": ["BaselineContextMgmt._unwind_messages_to_free_tokens", "_research1_unwind_history"],
            "globals": [], "signal": "recent_context_lost", "diagnostic_review_required": True,
            "coverage": "Unwind oldest complete exchanges while retaining system and recent exchange; preserve no-compression/estimation-failure paths.",
        },
        "agent_loop_error_retraction": {
            "module": "agent_loop", "mechanism_id": "agent_loop_error_retraction",
            "goal_checks": ["tool_error_resets_completion"],
            "preserve_checks": ["two_signal_confirmation", "false_resets_confirmation", "tool_results_ordered", "all_tool_feedback_visible", "real_warning_visible", "parse_feedback", "session_dead_stops", "session_error_continues"],
            "functions": ["BaselineAgentLoop._build_observation_text", "_research1_completion_error"],
            "globals": [], "signal": "completion_not_retracted_after_error", "diagnostic_review_required": True,
            "coverage": "Failed tool execution retracts completion, preserving normal confirmation, command order, feedback, parse retry and dead-session handling.",
        },
        "completion_requirement_recheck": {
            "module": "agent_loop", "mechanism_id": "completion_requirement_recheck",
            "goal_checks": ["completion_requirements_rechecked"],
            "preserve_checks": ["noncomplete_prompt_exact", "completion_observation_preserved", "completion_prompt_bounded", "instructionless_completion_safe", "instruction_source_unchanged", "two_signal_confirmation", "false_resets_confirmation", "tool_results_ordered", "all_tool_feedback_visible", "real_warning_visible", "parse_feedback", "session_dead_stops", "session_error_continues"],
            "functions": ["BaselineAgentLoop._next_prompt", "BaselineAgentLoop._completion_confirmation", "_research1_completion_recheck"],
            "globals": [], "signal": "explicit_requirement_coverage_gap", "diagnostic_review_required": True,
            "coverage": "First-completion prompt must carry the current original task instruction with an explicit request to compare behavioral/functional checks against requirements; long instructions must add at most 8192 characters. This verifies dynamic runtime prompt/context plumbing only, not solver compliance, task correctness or score benefit. Preserve ordinary continuation, terminal feedback, two-signal control, command order and error/session handling. A concrete visible missed requirement is independently attributed before this probe; neither a score nor this catalog entry supplies that evidence.",
        },
    }
    for pack in packs.values():
        pack["id"] = pack["mechanism_id"] + ".v1"
        pack["allowed_paths"] = [pack["module"] + "/baseline.py"]
        pack["allowed_imports"] = ["re", "logging", "typing", "collections", "dataclasses"]
        pack["max_files"] = 5
        pack["max_changed_lines"] = 200
    return {"version": REGISTRY_VERSION, "packs": packs}


def planning_interfaces(permitted_modules=MODULES, *, include_scope=False) -> list[dict[str, Any]]:
    """Describe frozen checks without asserting that any applies to this parent.

    Generation may use an unknown mechanism. Execution/admission still goes
    through the independent attribution and content-bound selected contract.
    """
    permitted = set(permitted_modules)
    rows=[]
    for _,pack in sorted(frozen_registry()['packs'].items()):
        if pack['module'] not in permitted:continue
        row={key:pack[key] for key in ('module','mechanism_id','coverage','goal_checks','preserve_checks')}
        if include_scope:
            row['scope']={key:pack[key] for key in ('allowed_paths','functions','globals','allowed_imports','max_files','max_changed_lines')}
        rows.append(row)
    return rows


def proposal_catalog(state: dict[str, Any], contract: dict[str, Any]) -> list[dict[str, str]]:
    if state.get('plan_policy_version') in {'proposal_single_plan_v5', PLAN_BINDING_POLICY}:
        rows = planning_interfaces(state.get('permitted_modules', MODULES))
    else:
        rows = contract.get('selected', [])
    return [{'mechanism_id': row['mechanism_id'], 'module': row['module']} for row in rows]


def _empty_warning(evidence: dict[str, Any]) -> bool:
    """Match referenced Command N to its exact empty call in the same step."""
    import re
    try:
        data = json.loads(evidence.get("text", ""))
    except (TypeError, ValueError):
        return False
    for step in data.get("steps", []):
        observation = step.get("observation", "")
        if not observation and isinstance(step.get("observation_excerpt"), dict):
            observation = "\n".join(s.get("text", "") for s in step["observation_excerpt"].get("segments", []))
        if not isinstance(observation, str):
            continue
        for number in re.findall(r"Command (\d+) should end with newline", observation):
            calls = step.get("tool_calls") or []
            index = int(number) - 1
            if not 0 <= index < len(calls):
                continue
            arguments = calls[index].get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    continue
            if isinstance(arguments, dict) and arguments.get("keystrokes") == "" and any(k in arguments for k in ("duration", "duration_sec")):
                return True
    return False


def build_contract(state: dict[str, Any]) -> dict[str, Any]:
    """Freeze registry, evidence attribution and module hashes before proposals.

    New-mechanism diagnostics must be independently reviewed, reference visible
    evidence, supply an exact witness substring and reproduce a target fixture
    on this parent. Caller assertions alone never make a contract applicable.
    """
    from research1.data_pipeline.policy import _witness_sources
    registry = frozen_registry()
    evidence = {e["id"]: e for e in state.get("evidence", [])}
    permitted = set(state.get("permitted_modules", MODULES))
    selected = []
    # New-protocol trials freeze every available target probe before seeing a
    # teacher plan. This is check availability, never attribution or admission.
    parent_probes, preprobed = {}, []
    if state.get('plan_policy_version') == PLAN_BINDING_POLICY:
        for mechanism, pack in sorted(registry['packs'].items()):
            if pack['module'] not in permitted:
                continue
            try:
                probe = probe_module(Path(state['modules_root']), mechanism, python=state.get('probe_python'))
                checks = probe['checks']
                reproduced = all(checks.get(name) is False for name in pack['goal_checks'])
                preserved = all(checks.get(name) is True for name in pack['preserve_checks'])
                row = {'mechanism_id':mechanism, 'module':pack['module'], 'parent_probe':probe,
                       'target_reproduced':reproduced, 'preservation_baseline_passed':preserved}
                parent_probes[mechanism] = dict(row, status='available')
                if reproduced and preserved:
                    preprobed.append(row)
            except Exception as exc:
                parent_probes[mechanism] = {'mechanism_id':mechanism, 'module':pack['module'],
                    'status':'unavailable', 'reason':type(exc).__name__ + ': ' + str(exc)}
    diagnostics = []
    for evidence_id, row in evidence.items():
        if "tools" in permitted and _empty_warning(row):
            diagnostics.append({"mechanism_id": "false_concat_preservation", "module": "tools", "evidence_refs": [evidence_id], "signal": "empty_wait_newline_warning", "automatic_exact_call_match": True})
    diagnostics.extend(copy.deepcopy(state.get("diagnostics", [])))
    reasons = []
    for diagnostic in diagnostics:
        mechanism = diagnostic.get("mechanism_id")
        pack = registry["packs"].get(mechanism)
        if not pack or pack["module"] not in permitted or diagnostic.get("module") != pack["module"]:
            reasons.append("unknown or disallowed diagnostic mechanism")
            continue
        refs = diagnostic.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(ref not in evidence for ref in refs) or diagnostic.get("signal") != pack["signal"]:
            reasons.append(mechanism + ": diagnostic has no matching visible evidence")
            continue
        if pack["diagnostic_review_required"]:
            review = diagnostic.get("review") or {}
            witness = diagnostic.get("witness")
            if (review.get("status") != "PASS" or not review.get("reviewer_id") or not isinstance(witness, str) or not witness
                    or not any(witness in segment['text'] for ref in refs
                               for segment in _witness_sources(evidence[ref].get('text', '')))):
                reasons.append(mechanism + ": independent attribution and exact visible witness required")
                continue
        elif not any(_empty_warning(evidence[ref]) for ref in refs):
            reasons.append(mechanism + ": empty warned call is not reproduced in visible evidence")
            continue
        try:
            if state.get('plan_policy_version') == PLAN_BINDING_POLICY:
                frozen = parent_probes.get(mechanism, {})
                if frozen.get('status') != 'available':
                    raise ValueError('preteacher parent probe unavailable')
                probe = frozen['parent_probe']
            else:
                probe = probe_module(Path(state["modules_root"]), mechanism, python=state.get("probe_python"))
            checks = probe["checks"]
            reproduced = all(checks.get(name) is False for name in pack["goal_checks"])
            preserved = all(checks.get(name) is True for name in pack["preserve_checks"])
        except Exception as exc:
            reasons.append(mechanism + ": parent probe unavailable: " + str(exc))
            continue
        if not reproduced or not preserved:
            reasons.append(mechanism + ": parent target not reproduced or preservation baseline incomplete")
            continue
        selected.append({"mechanism_id": mechanism, "module": pack["module"], "evidence_refs": refs, "parent_probe": probe})
    result = {"registry": registry, "selected": selected, "pending_reasons": reasons,
              "parent_hashes": tree_hashes(Path(state["modules_root"])),
              "evidence_hashes": {key: _digest(value) for key, value in evidence.items()}}
    if state.get('plan_policy_version') == PLAN_BINDING_POLICY:
        result.update(binding_mode=PLAN_BINDING_MODE, parent_probes=parent_probes, preprobed=preprobed)
    result["checks_id"] = "checks_" + _digest(result)[:24]
    return result


def _pack(plan: dict[str, Any], state: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any] | None:
    # The contract is content-bound, not just a mutable caller-supplied ID.
    if contract.get("checks_id") != "checks_" + _digest({k:v for k,v in contract.items() if k != "checks_id"})[:24]:
        return None
    if contract.get("registry") != frozen_registry():
        return None
    if contract.get("evidence_hashes") != {e["id"]:_digest(e) for e in state.get("evidence", [])}:
        return None
    kind = plan.get("verification_kind")
    selected = any(item.get("mechanism_id") == kind and item.get("module") == plan.get("target_module") for item in contract.get("selected", []))
    if state.get('plan_policy_version') == PLAN_BINDING_POLICY:
        from research1.data_pipeline.applicability import validate_binding, validate_factual_review
        if contract.get('binding_mode') != PLAN_BINDING_MODE:
            return None
        try:
            if tree_hashes(Path(state['modules_root'])) != contract.get('parent_hashes'):
                return None
            if any((Path(state['modules_root'])/path).read_text(encoding='utf-8') != text
                   for path,text in state.get('module_code', {}).items()):
                return None
        except (OSError, ValueError, TypeError):
            return None
        if selected:
            if not validate_factual_review(state.get('plan_factual_review'), state, plan):
                return None
        elif not validate_binding(state.get('plan_contract_binding'), state, plan, contract):
            return None
    elif not selected:
        return None
    pack = contract.get("registry", {}).get("packs", {}).get(kind)
    if not pack or plan.get("target_module") not in state.get("permitted_modules", MODULES):
        return None
    if not plan.get("locations") or any(path not in pack["allowed_paths"] or path not in state.get("module_code", {}) for path in plan["locations"]):
        return None
    return pack


def _symbols(source: str) -> dict[str, str]:
    from research1.collection import _symbols as symbols
    return symbols(source)


def _scope(parent: Path, child: Path, plan: dict[str, Any], contract: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
    before = tree_hashes(parent)
    after = tree_hashes(child)
    if before != contract["parent_hashes"]:
        return {"status": "FAIL", "reason": "frozen parent changed"}
    if set(before) != set(after):
        return {"status": "FAIL", "reason": "added or removed module files"}
    changed = [rel for rel in before if before[rel] != after[rel]]
    if not changed or len(changed) > pack["max_files"] or any(rel not in plan["locations"] for rel in changed):
        return {"status": "FAIL", "reason": "changed file scope is empty or exceeds plan"}
    allowed = {"function:" + name for name in pack["functions"]}
    allowed |= {"global:" + name for name in pack["globals"]}
    allowed |= {"import:" + name for name in pack["allowed_imports"]}
    line_count = 0
    all_symbols = []
    for rel in changed:
        old, new = (parent / rel).read_text(encoding="utf-8"), (child / rel).read_text(encoding="utf-8")
        first, second = _symbols(old), _symbols(new)
        symbols = [key for key in first.keys() | second.keys() if first.get(key) != second.get(key)]
        if any(key not in allowed for key in symbols):
            return {"status": "FAIL", "reason": "changed executable symbols exceed frozen mechanism scope", "changed_symbols": symbols}
        if not any(key.startswith("function:") for key in symbols):
            return {"status": "FAIL", "reason": "no functional change; marker or constant-only patch rejected"}
        all_symbols.extend(symbols)
        line_count += sum(line.startswith(("+ ", "- ")) for line in difflib.ndiff(old.splitlines(), new.splitlines()))
    if line_count > pack["max_changed_lines"]:
        return {"status": "FAIL", "reason": "change exceeds frozen line budget", "changed_lines": line_count}
    return {"status": "PASS", "changed_files": changed, "changed_symbols": all_symbols, "changed_lines": line_count}


def implement(plan: dict[str, Any], state: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Generate a bounded implementation once; resumes validate stored artifacts."""
    from research1.distill import _tongji_chat
    from research1.fixed_editor import apply_replacements
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = state.get("contract") or state.get("checks") or {}
    if plan.get("action") != "edit":
        return {"status": "pending", "reason": "keep is a decision without an implementation"}
    pack = _pack(plan, state, contract)
    if pack is None:
        return {"status": "pending", "reason": "no applicable pre-frozen executable mechanism contract"}
    parent = Path(state["modules_root"])
    if tree_hashes(parent) != contract["parent_hashes"]:
        raise ValueError("frozen parent changed before implementation")
    editor_policy = 'five-module-exact-replacements.v2' if state.get('plan_policy_version')==PLAN_BINDING_POLICY else EDITOR_POLICY
    request_id = _digest({"plan": plan, "parent_hashes": contract["parent_hashes"], "checks_id": contract["checks_id"], "policy": editor_policy})
    cached = directory / "implementation.json"
    if cached.exists():
        result = json.loads(cached.read_text(encoding="utf-8"))
        if result.get("request_id") != request_id:
            raise ValueError("saved implementation belongs to another frozen request")
        if result.get("status") == "applied":
            if code_hash(Path(result["modules_root"])) != result["code_hash"] or hashlib.sha256(Path(result["patch_path"]).read_bytes()).hexdigest() != result["patch_sha256"]:
                raise ValueError("saved implementation or patch has changed")
            _validate_editor_metadata(result.get("editor_metadata", {}), state)
        return result
    sources = {rel: (parent / rel).read_text(encoding="utf-8") for rel in plan["locations"]}
    scope = {key: pack[key] for key in ("allowed_paths", "functions", "globals", "allowed_imports", "max_files", "max_changed_lines")}
    prompt = ('Implement only the supplied plan; do not invent a diagnosis or task answer. Return final JSON only: '
              '{"files":[{"path":"relative.py","edits":[{"old":"exact unique original block","new":"replacement block"}]}]}. '
              'Each file uses 1-4 exact, nonoverlapping replacements. Preserve all unrelated runtime behavior. '
              'Use only listed frozen symbols and imports; no markers, task answers, extra files or scope expansion.\n'
              + 'Scope: ' + _json(scope) + '\nPlan: ' + _json(plan) + '\nSources: ' + _json(sources))
    if state.get('plan_policy_version')==PLAN_BINDING_POLICY:
        prompt += ('\nFrozen behavior coverage: '+_json({key:pack[key] for key in ('coverage','goal_checks','preserve_checks')})
            +'\nRead-only runtime interfaces: '+_json(state.get('framework_context',{}))
            +'\nImplement the complete stated scope and preservation requirements. A total added-text size bound includes headings, '
             'requests and quoted content together; capping only the quoted content does not establish that total bound. '
             'Runtime source is for understanding reachable values only and does not expand editable files or symbols.')
    input_record = {"request_id": request_id, "prompt": prompt, "plan": plan, "policy": editor_policy}
    input_path = directory / "editor_input.json"
    if input_path.exists() and json.loads(input_path.read_text(encoding="utf-8")) != input_record:
        raise ValueError("frozen implementer input changed")
    _write(input_path, input_record)
    output = directory / "editor_output.json"
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("request_id") != request_id:
            raise ValueError("saved editor output belongs to another request")
        raw, metadata = saved["raw"], saved["metadata"]
    else:
        transport=({'allow_http_fallback':False} if state.get('allow_teacher_http_fallback') is False else {})
        raw, metadata = _tongji_chat(prompt, max_tokens=8000, role="data_frozen_implementer", json_output=True, reasoning_effort="none",**transport)
        _write(output, {"request_id": request_id, "raw": raw, "metadata": metadata})
    _validate_editor_metadata(metadata, state)
    signature = {key: metadata.get(key) for key in ("provider", "model", "temperature")}
    lock = directory.parent / "fixed_implementer_policy.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != signature:
        raise ValueError("implementer provider/model/temperature changed within frozen branch")
    _write(lock, signature)
    value = json.loads(raw)
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, list) or not 1 <= len(files) <= pack["max_files"]:
        raise ValueError("expected bounded files array")
    replacements = {}
    for row in files:
        rel = row.get("path")
        if not isinstance(rel, str) or PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts or rel not in sources or rel in replacements:
            raise ValueError("file outside frozen visible plan or repeated file")
        replacements[rel] = apply_replacements(sources[rel], row.get("edits"))
    child = directory / "child/modules"
    partial = []
    recovery = 0
    # Preserve interrupted trees, rebuild from the immutable parent and saved
    # exact replacement bytes. Recovery never requests another implementation.
    while child.exists():
        partial.append(str(child))
        recovery += 1
        child = directory / ("child-recovery-" + str(recovery)) / "modules"
    shutil.copytree(parent, child, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel, replacement in replacements.items():
        (child / rel).write_text(replacement, encoding="utf-8")
    scope_result = _scope(parent, child, plan, contract, pack)
    patch = ''.join(''.join(difflib.unified_diff(sources[rel].splitlines(keepends=True), replacements[rel].splitlines(keepends=True), fromfile='a/'+rel, tofile='b/'+rel)) for rel in replacements)
    patch_path = directory / "implementation.diff"
    patch_path.write_text(patch, encoding="utf-8")
    result = {"status": "applied" if scope_result["status"] == "PASS" else "rejected", "request_id": request_id,
              "modules_root": str(child.resolve()), "code_hash": code_hash(child), "parent_code_hash": code_hash(parent),
              "patch_path": str(patch_path.resolve()), "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
              "scope": scope_result, "editor_metadata": metadata, "editor_policy_id": editor_policy,
              "reason": scope_result.get("reason", ""), "checks_id": contract["checks_id"],
              "preserved_partial_trees": partial, "child_version_id": "data_" + code_hash(child)[:24]}
    _write(cached, result)
    return result


def _validate_editor_metadata(metadata: dict[str, Any], state: dict[str, Any]) -> None:
    if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop" or metadata.get("temperature") != 0:
        raise ValueError("implementation requires complete final JSON at frozen temperature zero")
    expected = state.get("editor_model") or state.get("policy", {}).get("editor_model")
    if expected and metadata.get("model") != expected:
        raise ValueError("implementer model differs from frozen policy")
    expected_provider = state.get("editor_provider") or state.get("policy", {}).get("editor_provider")
    if expected_provider and metadata.get("provider") != expected_provider:
        raise ValueError("implementer provider differs from frozen policy")


def verify(plan: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any], contract: dict[str, Any], *, probe_runner: Callable | None = None) -> dict[str, Any]:
    """Verify only selected mechanisms; scope+compile never imply coverage."""
    base = {"checks_id": contract.get("checks_id"), "mechanism_id": plan.get("verification_kind"), "executable": True}
    pack = _pack(plan, state, contract)
    if pack is None:
        return dict(base, status="pending", covers_plan=False, reason="no evidence-grounded pre-frozen contract for this mechanism")
    if attempt.get("status") != "applied":
        return dict(base, status="pending", covers_plan=False, reason="no applied implementation")
    try:
        _validate_editor_metadata(attempt.get("editor_metadata", {}), state)
        parent, child = Path(state["modules_root"]), Path(attempt["modules_root"])
        if (attempt.get("checks_id") != contract.get("checks_id")
                or attempt.get("parent_code_hash") != code_hash(parent)
                or attempt.get("code_hash") != code_hash(child)):
            return dict(base, status="FAIL", covers_plan=False, reason="implementation identity differs from frozen contract/artifacts")
        patch_path = Path(attempt["patch_path"])
        if hashlib.sha256(patch_path.read_bytes()).hexdigest() != attempt.get("patch_sha256"):
            return dict(base, status="FAIL", covers_plan=False, reason="stored patch changed")
        scope = _scope(parent, child, plan, contract, pack)
        if scope["status"] != "PASS":
            return dict(base, status="FAIL", covers_plan=True, reason=scope["reason"], scope=scope)
        runner = probe_runner or probe_module
        before = runner(parent, pack["mechanism_id"])
        after = runner(child, pack["mechanism_id"])
        parent_failed = all(before["checks"].get(name) is False for name in pack["goal_checks"])
        parent_preserved = all(before["checks"].get(name) is True for name in pack["preserve_checks"])
        if not parent_failed or not parent_preserved:
            return dict(base, status="pending", covers_plan=False, reason="parent no longer reproduces frozen target/preservation cases", parent_probe=before)
        names = pack["goal_checks"] + pack["preserve_checks"]
        passed = all(after["checks"].get(name) is True for name in names)
        return dict(base, status="PASS" if passed else "FAIL", covers_plan=True, parent_probe=before, child_probe=after,
                    scope=scope, reason="target and preservation cases passed" if passed else "target or preservation fixture failed")
    except Exception as exc:
        return dict(base, status="pending", covers_plan=False, reason="executable verification unavailable: " + str(exc))


def probe_module(root: Path, mechanism_id: str, *, python: str | None = None) -> dict[str, Any]:
    """Run fixtures in a fresh process with no solver, task or scorer context."""
    interpreter = python or str(Path('/home/jinxu/arms_cache/modularrsi/ModularRSI/.venv/bin/python'))
    if not Path(interpreter).is_file():
        interpreter = sys.executable
    request = {"root": str(Path(root).resolve()), "mechanism_id": mechanism_id, "research_src": str(Path(__file__).resolve().parents[2])}
    script = "import sys,json;req=json.load(sys.stdin);sys.path.insert(0,req['research_src']);from research1.data_pipeline.modules import _probe_in_process;print(json.dumps(_probe_in_process(req['root'],req['mechanism_id'])))"
    proc = subprocess.run([interpreter, '-I', '-c', script], input=json.dumps(request), capture_output=True, text=True,
                          timeout=120, cwd=str(Path(root).resolve()))
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout).splitlines()[-1][:500])
    lines = [line for line in proc.stdout.splitlines() if line.startswith('{')]
    if len(lines) != 1:
        raise ValueError("ambiguous fixture process output")
    result = json.loads(lines[0])
    if result.get('mechanism_id') != mechanism_id or not isinstance(result.get('checks'), dict):
        raise ValueError("fixture process returned invalid mechanism/checks")
    return result


def _probe_in_process(root: str, mechanism_id: str) -> dict[str, Any]:
    """Private deterministic fixture runner; never calls an LLM or a task."""
    import asyncio
    import importlib.util
    import types
    packs = frozen_registry()['packs']
    if mechanism_id not in packs:
        raise ValueError('unregistered fixture')
    if mechanism_id == 'false_concat_preservation':
        from research1.plan_init import frozen_false_concat_checks, probe_checks
        rows = frozen_false_concat_checks()
        results = probe_checks(Path(root), rows)
        warning = lambda name: str(results[name].get('warning') or '')
        checks = {c['id']: ((c['needle'] in warning(c['id'])) == (c['after'] == 'present')) for c in rows}
        checks['mixed_empty_false_absent'] = 'Command 1 should end with newline' not in warning('mixed_empty_and_real')
        checks['mixed_real_present'] = 'Command 2 should end with newline' in warning('mixed_empty_and_real')
        checks['parser_fields_preserved'] = all(
            r.get('commands') == [[c['keystrokes'], float(min(c['duration'], 60))] for c in row['payload']['commands']]
            and not r.get('error') and r.get('is_task_complete') is False
            for row in rows for r in [results[row['id']]])
        checks.pop('mixed_empty_and_real', None)
        return {'mechanism_id': mechanism_id, 'checks': checks, 'results': results}

    class Record:
        def __init__(self, **kwargs):
            self.consecutive_complete_signals = 0
            self.text = ''
            self.prev_terminal = ''
            self.success = True
            self.output = ''
            self.error = ''
            self.handoff_prompt = None
            self.subagent_refs = None
            self.summarization_occurred = False
            self.__dict__.update(kwargs)

    # Stub only Harbor data protocols; execution requires neither Harbor models
    # nor task/scorer imports. Each process loads one module tree in isolation.
    for name in ('harbor', 'harbor.agents', 'harbor.agents.terminus_2_modular', 'harbor.llms', 'harbor.models', 'harbor.utils'):
        module = types.ModuleType(name); module.__path__ = []; sys.modules[name] = module
    for name in ('harbor.agents.terminus_2_modular.protocols', 'harbor.llms.base', 'harbor.llms.chat', 'harbor.models.trajectories', 'harbor.utils.trajectory_utils'):
        module = types.ModuleType(name)
        module.__getattr__ = lambda key: Record
        sys.modules[name] = module
    for key in ('ContextLengthExceededError', 'OutputLengthExceededError'):
        setattr(sys.modules['harbor.llms.base'], key, type(key, (Exception,), {}))
    sys.modules['harbor.utils.trajectory_utils'].format_trajectory_json = lambda x: json.dumps(x)
    if 'tenacity' not in sys.modules:
        try:
            import tenacity
        except ImportError:
            class RetryCondition:
                def __and__(self, other): return self
            tenacity = types.ModuleType('tenacity')
            tenacity.retry = lambda **kw: lambda f: f
            for key in ('retry_if_exception_type', 'retry_if_not_exception_type', 'stop_after_attempt'):
                setattr(tenacity, key, lambda *a, **k: RetryCondition())
            sys.modules['tenacity'] = tenacity
    module_name = packs[mechanism_id]['module']
    path = Path(root) / module_name / 'baseline.py'
    spec = importlib.util.spec_from_file_location('research1_probe_target', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    log = types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    ctx = Record(shared=Record(tmux_session=None), services=Record(logger=log))
    checks = {}

    async def run():
        if module_name == 'verification':
            verifier = module.BaselineVerification()
            async def result(count, failed=False):
                state = Record(consecutive_complete_signals=count, last_tool_result=Record(success=not failed, error='execution failed' if failed else ''))
                return (await verifier.should_terminate(state, ctx))[0]
            checks['first_signal_waits'] = await result(1) is False
            checks['second_signal_confirms'] = await result(2) is True
            checks['reset_signal_waits'] = await result(0) is False
            checks['failed_tool_retracts_completion'] = await result(2, True) is False
            custom = module.BaselineVerification(required_consecutive=3)
            checks['custom_required_count'] = (await custom.should_terminate(Record(consecutive_complete_signals=2), ctx))[0] is False and (await custom.should_terminate(Record(consecutive_complete_signals=3), ctx))[0] is True
        elif module_name == 'observation':
            truncate = module._smart_truncate
            checks['short_output_exact'] = truncate('plain\nERROR: visible\n雪', 128) == 'plain\nERROR: visible\n雪'
            raw = 'HEAD\n' + 'x'*1000 + '\nERROR: middle-anchor-17\n' + 'y'*1000 + '\nTAIL'
            text = truncate(raw, 128)
            checks['middle_error_visible'] = 'ERROR: middle-anchor-17' in text
            checks['tail_retained'] = text.endswith('TAIL')
            checks['head_retained'] = text.startswith('HEAD')
            checks['bounded_capture'] = len(text.encode()) <= 128 + 256 and len(text) < len(raw)
            unicode_text = truncate('首'*100+'末'*100, 65)
            checks['utf8_valid'] = '\ufffd' not in unicode_text and unicode_text.encode().decode() == unicode_text
            obs = module.BaselineObservation(max_bytes=128)
            prev = Record(prev_terminal='previous')
            value, returned = await obs.capture(prev, ctx)
            checks['no_session_quiet'] = value.text == '' and returned is prev
            class Session:
                async def get_incremental_output(self): return 'new-only'
            ctx.shared.tmux_session = Session()
            value, returned = await obs.capture(prev, ctx)
            checks['incremental_capture'] = value.text == 'new-only' and returned.prev_terminal == 'new-only'
            class Broken:
                async def get_incremental_output(self): raise RuntimeError('capture failed')
            ctx.shared.tmux_session = Broken()
            value, returned = await obs.capture(prev, ctx)
            checks['capture_error_quiet'] = value.text == ''
        elif module_name == 'context_mgmt':
            class Model:
                def count_message_tokens(self, messages): return sum(len(m['content']) for m in messages)
                def get_model_context_limit(self): return 80
            class Chat:
                def __init__(self):
                    self._model = Model(); self.resets = 0
                    self._messages = [{'role':'system','content':'SYS'}, {'role':'user','content':'OLD-U'*4}, {'role':'assistant','content':'OLD-A'*4}, {'role':'user','content':'RECENT-U'}, {'role':'assistant','content':'RECENT-A'}]
                @property
                def messages(self): return self._messages
                def reset_response_chain(self): self.resets += 1
            manager = module.BaselineContextMgmt()
            chat = Chat(); manager._unwind_messages_to_free_tokens(chat, target_free_tokens=45)
            checks['recent_exchange_retained'] = len(chat.messages) >= 3 and chat.messages[-2:] == [{'role':'user','content':'RECENT-U'}, {'role':'assistant','content':'RECENT-A'}]
            checks['system_retained'] = chat.messages[0] == {'role':'system','content':'SYS'}
            checks['budget_reached'] = 80 - chat._model.count_message_tokens(chat.messages) >= 45
            checks['whole_exchanges'] = [m['role'] for m in chat.messages] in (['system','user','assistant'], ['system'])
            checks['chain_reset'] = chat.resets >= 1
            disabled = module.BaselineContextMgmt(enable_summarize=False)
            chat = Chat(); original = copy.deepcopy(chat.messages)
            value = await disabled.maybe_compress(chat, 'task', ctx)
            checks['disabled_noop'] = value.chat is chat and chat.messages == original and not value.summarization_occurred
            roomy = module.BaselineContextMgmt(proactive_summarization_threshold=1)
            value = await roomy.maybe_compress(chat, 'task', ctx)
            checks['roomy_noop'] = value.chat is chat and chat.messages == original and not value.summarization_occurred
            value = await manager.maybe_compress(chat, '', ctx)
            checks['instructionless_noop'] = value.chat is chat and chat.messages == original
            class BrokenModel(Model):
                def count_message_tokens(self, messages): raise RuntimeError('estimator failed')
            chat._model = BrokenModel(); value = await manager.maybe_compress(chat, 'task', ctx)
            checks['context_failure_noop'] = value.chat is chat and chat.messages == original
        elif module_name == 'agent_loop':
            loop = module.BaselineAgentLoop()
            state = Record(consecutive_complete_signals=0)
            pending = False
            parse = Record(is_task_complete=True, warning='', error='')
            text, pending = loop._build_observation_text(parse=parse, obs_result=Record(text='terminal'), tool_results=[], state=state, pending_completion=pending)
            first = state.consecutive_complete_signals == 1 and pending and 'terminal' in text
            text, pending = loop._build_observation_text(parse=parse, obs_result=Record(text='terminal'), tool_results=[], state=state, pending_completion=pending)
            checks['two_signal_confirmation'] = first and state.consecutive_complete_signals == 2 and text == 'terminal'
            parse.is_task_complete = False
            loop._build_observation_text(parse=parse, obs_result=Record(text=''), tool_results=[], state=state, pending_completion=pending)
            checks['false_resets_confirmation'] = state.consecutive_complete_signals == 0
            parse.warning = 'real parser warning'
            text, _ = loop._build_observation_text(parse=parse, obs_result=Record(text='terminal'), tool_results=[Record(output='first'), Record(success=False, error='failed command')], state=state, pending_completion=False)
            checks['all_tool_feedback_visible'] = all(s in text for s in ('first', 'failed command', 'terminal'))
            checks['real_warning_visible'] = 'real parser warning' in text
            calls=[]
            class Tools:
                _parser_name='JSON'
                async def execute(self, call, ctx):
                    calls.append(call); return Record(output=str(call), success=call != 2, error='second failed' if call==2 else '')
            results,last = await loop._execute_commands(Record(commands=[1,2,3]), Tools(), ctx, iteration=0)
            checks['tool_results_ordered'] = calls == [1,2,3] and len(results)==3 and last is results[-1]
            prompt = loop._build_parse_error_prompt(Record(error='bad JSON',warning='warning retained'), Tools())
            checks['parse_feedback'] = 'bad JSON' in prompt and 'warning retained' in prompt and 'JSON' in prompt
            class Session:
                async def is_session_alive(self): return False
            ctx.shared.tmux_session = Session(); dead = await loop._check_session_alive(ctx)
            checks['session_dead_stops'] = dead is not None and dead.failure_tag == 'session_dead'
            class Broken:
                async def is_session_alive(self): raise RuntimeError('transient')
            ctx.shared.tmux_session=Broken()
            checks['session_error_continues'] = await loop._check_session_alive(ctx) is None
            parse.is_task_complete=True;parse.warning=''
            state.consecutive_complete_signals=1
            text,pending=loop._build_observation_text(parse=parse,obs_result=Record(text='terminal'),tool_results=[Record(success=False,error='execution failed')],state=state,pending_completion=True)
            checks['tool_error_resets_completion'] = state.consecutive_complete_signals == 0 and pending is False and 'execution failed' in text
            if mechanism_id == 'completion_requirement_recheck':
                import re
                # Drive the actual parent/child functions through the same
                # completion-observation -> next-prompt interface as run().
                # The two unrelated tasks prevent a fixed instruction or a
                # task-specific answer from satisfying dynamic context flow.
                tasks = (
                    'REQ-C17: render rows in ascending order; retain Unicode identifiers and handle an empty collection.',
                    'REQ-K42: retain duplicate entries and reject an invalid date.',
                )
                terminal = 'EXECUTED-OBSERVATION-C83\noutput remains visible'
                class PromptTools:
                    def format_continuation_prompt(self, observation_text, context):
                        return 'CONTINUE<' + observation_text + '>:' + context.state.task_id
                tools = PromptTools()
                probe_ctx = Record(state=Record(instruction='', task_id='runtime-context'), shared=ctx.shared, services=ctx.services)
                source_unchanged, preserved, targeted, normal = [], [], [], []
                lengths = []
                for instruction in tasks:
                    probe_ctx.state.instruction = instruction
                    parse = Record(is_task_complete=True, warning='', error='')
                    completion_state = Record(consecutive_complete_signals=0)
                    text, pending = loop._build_observation_text(parse=parse, obs_result=Record(text=terminal), tool_results=[], state=completion_state, pending_completion=False)
                    prompt = loop._next_prompt(text, parse, tools, probe_ctx)
                    if not isinstance(prompt, str):
                        raise ValueError('Completion prompt must remain a string')
                    # Cues are checked outside the supplied instruction: simply
                    # copying a task containing the word "behavioral" is not a
                    # requirement-review request from the Harness.
                    cue = prompt.replace(instruction, '').lower()
                    request = (re.search(r'\brequirements?\b', cue) is not None
                               and re.search(r'\b(?:behavioral|behavioural|functional)\b', cue) is not None
                               and re.search(r'\b(?:checks?|tests?|coverage|verify|verification)\b', cue) is not None)
                    targeted.append(instruction in prompt and request and all(other not in prompt for other in tasks if other != instruction))
                    preserved.append(text in prompt and terminal in prompt)
                    source_unchanged.append(probe_ctx.state.instruction == instruction and completion_state.consecutive_complete_signals == 1 and pending is True)
                    parse.is_task_complete = False
                    ordinary = loop._next_prompt('ordinary terminal output', parse, tools, probe_ctx)
                    normal.append(ordinary == 'CONTINUE<ordinary terminal output>:runtime-context')
                    lengths.append(len(prompt) - len(text))
                checks['completion_requirements_rechecked'] = all(targeted)
                checks['noncomplete_prompt_exact'] = all(normal)
                checks['completion_observation_preserved'] = all(preserved)
                # A task instruction is input data. The loop must neither
                # mutate it nor silently alter the completion counter to make
                # its new prompt fit the existing verification cadence.
                checks['instruction_source_unchanged'] = all(source_unchanged)
                long_instruction = 'LONG-REQ-F58:' + ('identifier-\u96ea;' * 12000)
                probe_ctx.state.instruction = long_instruction
                parse.is_task_complete = True
                long_state = Record(consecutive_complete_signals=0)
                text, _ = loop._build_observation_text(parse=parse, obs_result=Record(text=terminal), tool_results=[], state=long_state, pending_completion=False)
                long_prompt = loop._next_prompt(text, parse, tools, probe_ctx)
                checks['completion_prompt_bounded'] = (isinstance(long_prompt, str) and text in long_prompt
                    and 0 <= len(long_prompt) - len(text) <= 8192 and all(0 <= length <= 8192 for length in lengths)
                    and probe_ctx.state.instruction == long_instruction)
                quiet = []
                for instruction in ('', None):
                    probe_ctx.state.instruction = instruction
                    try:
                        result = loop._next_prompt(text, parse, tools, probe_ctx)
                        quiet.append(isinstance(result, str) and text in result and len(result)-len(text) <= 8192
                                     and probe_ctx.state.instruction is instruction)
                    except (AttributeError, TypeError):
                        quiet.append(False)
                checks['instructionless_completion_safe'] = all(quiet)
                names = packs[mechanism_id]['goal_checks'] + packs[mechanism_id]['preserve_checks']
                checks_subset = {name: checks[name] for name in names}
                checks.clear(); checks.update(checks_subset)
    asyncio.run(run())
    result = {'mechanism_id': mechanism_id, 'checks': checks}
    if mechanism_id == 'completion_requirement_recheck':
        result['scope'] = 'Deterministic runtime completion prompt/context plumbing and preservation only; not solver compliance, task correctness or score benefit.'
    return result
