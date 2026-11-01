"""Read-only, byte-bound admission proof for the v6 binding protocol.

Sealing is an explicit controller operation at first positive admission. Export
only validates that seal; it never repairs data, reseals changed bytes, executes
probes, requests a model, or launches an evaluation.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

from research1.data_pipeline.policy import canonical, digest, parse_candidates, public_view, render_messages

SCHEMA = "research1.positive-admission-artifacts.v1"
PLAN_VERSION = "proposal_single_plan_v6"
BINDING_MODE = "independent_plan_to_prefrozen_checks.v1"


def artifact_gate_applies(record: dict) -> bool:
    state = record.get("state") or {}
    return bool(state.get("plan_policy_version") == PLAN_VERSION
                and (state.get("contract") or {}).get("binding_mode") == BINDING_MODE)


def _read(path_value) -> tuple[Path, dict, str]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("Required positive-admission artifact path missing")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Positive-admission artifacts must be absolute regular files")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Positive-admission JSON artifact is not an object")
    return path, value, hashlib.sha256(raw).hexdigest()


def _read_source_artifact(path_value) -> tuple[Path,str]:
    """Source manifests also bind Python fixtures; JSON originals stay typed."""
    if not isinstance(path_value,str) or not path_value:
        raise ValueError('Original source artifact path is missing')
    path=Path(path_value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Original source artifacts must be absolute regular files')
    cursor=path
    while cursor!=cursor.parent:
        if cursor.is_symlink():
            raise ValueError('Original source artifacts cannot follow symlinked paths')
        cursor=cursor.parent
    raw=path.read_bytes()
    if path.suffix.lower()=='.json' and not isinstance(json.loads(raw),dict):
        raise ValueError('Original JSON source artifact must remain an object')
    return path,hashlib.sha256(raw).hexdigest()


def _identity(record: dict) -> str:
    keys = ("sample_id", "protocol_id", "source_kind", "state_id", "root_lineage_id",
            "state", "plan", "qualification", "implementation", "teacher_path",
            "teacher_record_path", "teacher_prompt_sha256")
    return digest({key: record.get(key) for key in keys})


def _teacher(record: dict, files: dict) -> None:
    from research1.data_pipeline.comparison import _collection_prompt
    state, plan = record["state"], record["plan"]
    paths = {value for value in (record.get("teacher_record_path"), record.get("teacher_path"),
                                state.get("teacher_record_path")) if value}
    if len(paths) != 1:
        raise ValueError("Teacher record path is missing or inconsistent")
    path, saved, sha = _read(next(iter(paths)))
    prompt = _collection_prompt(state)
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if (saved.get("prompt") != prompt or saved.get("prompt_sha256") != prompt_sha
            or record.get("teacher_prompt_sha256") != prompt_sha
            or saved.get("policy_id") != PLAN_VERSION
            or saved.get("visible_view_sha256") != digest(public_view(state))):
        raise ValueError("Original teacher prompt/view binding changed")
    # Rendering here also validates the current deployment single-plan schema.
    if render_messages(state, plan)[-1] != {"role": "assistant", "content": canonical(plan)}:
        raise ValueError("Teacher plan differs from the deployment answer")
    effort = state.get("teacher_reasoning_effort", "none")
    settings = {"reasoning_effort": effort, "max_tokens": state.get('teacher_max_tokens',8000), "temperature": 0,
                "allow_http_fallback": state.get("allow_teacher_http_fallback") is not False}
    if effort not in {"none", "low"} or saved.get("request_settings") != settings:
        raise ValueError("Teacher request settings changed")
    meta = saved.get("metadata") or {}
    expected = {"model": state.get("teacher_model"), "provider": state.get("teacher_provider"),
                "temperature": 0, "raw_field": "content", "finish_reason": "stop", "reasoning_effort": effort}
    if not expected["model"] or not expected["provider"] or any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Teacher final metadata differs from the frozen policy")
    if plan not in parse_candidates(saved.get("raw"), state, state.get("max_candidates", 2)):
        raise ValueError("Original complete teacher reply does not contain this plan")
    files["teacher"] = {"path": str(path), "sha256": sha}


def _result(result: dict, state: dict, expected_hash: str, label: str, files: dict) -> None:
    from research1.data_pipeline.execution import bind_trial
    from research1.data_pipeline.initialization import _check_result_artifacts
    tasks = state.get("proposal_task_ids", [])
    if not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("Positive artifact needs exact visible proposal task identities")
    if not isinstance(result, dict) or result.get("status") != "PASS" or result.get("failure_layer") not in {"model", "harness"}:
        raise ValueError("Positive admission requires an actual valid scored proposal execution")
    spec = result.get("evaluation_spec")
    if not isinstance(spec, dict) or spec.get("task") not in tasks or not result.get("output_dir"):
        raise ValueError("Positive execution lacks its original physical evaluation binding")
    _check_result_artifacts(result, expected_task=spec["task"], expected_code_hash=expected_hash)
    # bind_trial is a read-only config/result binder, not a task execution.
    rebound = bind_trial(Path(result["output_dir"]), spec)
    for key in ("status", "reward", "failure_layer", "task_name", "model", "provider_id", "temperature",
                "trial_name", "trial_result_path", "trajectory", "trajectory_sha256", "result_sha256"):
        if result.get(key) != rebound.get(key):
            raise ValueError("Cached score/result differs from the original physical trial: " + key)
    for key, sha_key in (("trial_result_path", "result_sha256"), ("trajectory", "trajectory_sha256")):
        path, _, sha = _read(result.get(key))
        if result.get(sha_key) != sha:
            raise ValueError("Execution artifact bytes differ from their committed identity")
        files[label + "_" + key] = {"path": str(path), "sha256": sha}
    trial = Path(result["trial_result_path"]).parent
    for suffix, path in (("trial_config", trial / "config.json"), ("job_config", trial.parent / "config.json")):
        path, _, sha = _read(str(path))
        files[label + "_" + suffix] = {"path": str(path), "sha256": sha}


def _implementation(state: dict, plan: dict, attempt: dict, contract: dict, files: dict) -> None:
    from research1.data_pipeline.initialization import _check_attempt
    from research1.data_pipeline.module_adapters import get_adapter
    adapter=get_adapter(state)
    _scope,_validate_editor_metadata=adapter._scope,adapter._validate_editor_metadata
    from research1.fixed_editor import apply_replacements
    if attempt.get("status") != "applied" or attempt.get("code_hash") == state.get("code_hash"):
        raise ValueError("Positive edit requires an applied functional child")
    _check_attempt(state, attempt)
    if attempt.get("checks_id") != contract.get("checks_id"):
        raise ValueError("Implementation refers to different frozen checks")
    patch = Path(attempt["patch_path"])
    if not patch.is_absolute() or patch.is_symlink() or not patch.is_file():
        raise ValueError("Saved edit patch is not a regular absolute artifact")
    files["patch"] = {"path": str(patch), "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
    artifacts = {}
    for label, name in (("editor_input", "editor_input.json"), ("editor_output", "editor_output.json"), ("saved_implementation", "implementation.json")):
        path, value, sha = _read(str(patch.parent / name))
        artifacts[label] = value
        files[label] = {"path": str(path), "sha256": sha}
    if artifacts["saved_implementation"] != attempt:
        raise ValueError("Implementation record differs from the original saved result")
    if (artifacts["editor_input"].get("plan") != plan
            or artifacts["editor_input"].get("request_id") != attempt.get("request_id")
            or artifacts["editor_output"].get("request_id") != attempt.get("request_id")
            or artifacts["editor_output"].get("metadata") != attempt.get("editor_metadata")):
        raise ValueError("Original editor records are not bound to this implementation and plan")
    _validate_editor_metadata(attempt.get("editor_metadata", {}), state)
    pack = contract["registry"]["packs"][plan["verification_kind"]]
    parent, child = Path(state["modules_root"]), Path(attempt["modules_root"])
    scope = _scope(parent, child, plan, contract, pack)
    if scope.get("status") != "PASS" or attempt.get("scope") != scope:
        raise ValueError("Actual child exceeds the frozen plan/symbol scope")
    value = json.loads(artifacts["editor_output"]["raw"])
    rows = value.get("files") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows or len(rows) > pack["max_files"]:
        raise ValueError("Original editor reply does not describe bounded replacements")
    diffs, seen = [], set()
    for row in rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if relative not in plan["locations"] or relative in seen:
            raise ValueError("Original editor replacement path exceeds the plan")
        seen.add(relative)
        old = (parent / relative).read_text(encoding="utf-8")
        new = apply_replacements(old, row.get("edits"))
        if new != (child / relative).read_text(encoding="utf-8"):
            raise ValueError("Actual child differs from the original saved editor replacements")
        diffs.append("".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile="a/" + relative, tofile="b/" + relative)))
    if patch.read_text(encoding="utf-8") != "".join(diffs):
        raise ValueError("Saved patch does not reproduce the exact original replacement diff")


def _materials(record: dict) -> dict:
    from research1.data_pipeline.applicability import validate_binding, validate_factual_review
    from research1.data_pipeline.initialization import _check_attempt
    state, plan, q = record["state"], record["plan"], record["qualification"]
    if q.get("status") != "verified":
        raise ValueError("Artifacts can only seal a fully verified positive admission")
    from research1.data_pipeline.module_adapters import get_adapter,STRICT_VERSION
    adapter=get_adapter(state)
    if adapter.REGISTRY_VERSION==STRICT_VERSION:
        contract=state.get('contract') or {}
        if (contract.get('registry')!=adapter.frozen_registry()
                or contract.get('checks_id')!='checks_'+adapter._digest({key:value for key,value in contract.items() if key!='checks_id'})[:24]):
            raise ValueError('New positive admission refers to a different frozen verification registry')
    files = {}
    if state.get('predecision_probe_io_policy') is not None:
        from research1.data_pipeline.predecision_probe_io import SCHEMA,validate_probe_io_state
        if state['predecision_probe_io_policy'] != SCHEMA:
            raise ValueError('Positive admission has an unknown predecision probe evidence policy')
        proofs=validate_probe_io_state(state)
        if (not isinstance(proofs,dict) or proofs.get('schema')!=SCHEMA or proofs.get('verified') is not True
                or not isinstance(proofs.get('artifacts'),dict)
                or set(proofs['artifacts'])!={'raw_contract','receipt'}):
            raise ValueError('Predecision probe proof must bind its original raw and receipt artifacts')
        for label,artifact in proofs['artifacts'].items():
            path,_,sha=_read(artifact.get('path'))
            if sha!=artifact.get('sha256'):
                raise ValueError('Original predecision API probe artifact changed')
            files['predecision_probe_'+label]={'path':str(path),'sha256':sha}
    source_artifacts={**state.get('fact_source_artifacts',{}),**{'teacher_'+k:v for k,v in state.get('teacher_source_artifacts',{}).items()}}
    for label,artifact in source_artifacts.items():
        path,sha=_read_source_artifact(artifact.get('path'))
        if sha!=artifact.get('sha256'):
            raise ValueError('Original fact-selector source artifact changed')
        files['fact_source_'+label]={'path':str(path),'sha256':sha}
    _teacher(record, files)
    review = q.get("factual_review")
    if not validate_factual_review(review, state, plan):
        raise ValueError("Original separately saved factual review no longer establishes PASS")
    raw_path = Path(review["review_path"])
    for label, path in (("factual_raw", raw_path), ("factual_result", raw_path.parent / "factual_review_result.json")):
        path, _, sha = _read(str(path)); files[label] = {"path": str(path), "sha256": sha}
    _check_attempt(state, record.get("implementation") or {})
    _result(q.get("proposal_task_result"), state, state["code_hash"], "parent", files)
    if plan["action"] == "keep":
        attempt = record.get("implementation") or {}
        if attempt.get("status") != "kept" or attempt.get("code_hash") != state["code_hash"]:
            raise ValueError("Positive keep must retain its exact parent code")
        from research1.data_pipeline.independent_plan_audit import gate_applies as independent_gate, validate_independent_audit, seal_file_bindings
        if independent_gate(state=state):
            validate_independent_audit(record)
            proof = q.get("independent_plan_audit") or {}
            files.update(seal_file_bindings(proof))
        return files
    contract = state["contract"]
    if review != state.get("plan_factual_review"):
        raise ValueError("Admission factual review differs from the independently bound plan review")
    binding = state.get("plan_contract_binding")
    if not validate_binding(binding, state, plan, contract):
        raise ValueError("Original applicability records no longer bind this plan to frozen checks")
    for kind in ("input", "output", "started"):
        path, _, sha = _read(binding.get("applicability_" + kind + "_path"))
        files["applicability_" + kind] = {"path": str(path), "sha256": sha}
    path, _, sha = _read(str(Path(binding["applicability_input_path"]).parent / "applicability_result.json"))
    files["applicability_result"] = {"path": str(path), "sha256": sha}
    behavior = q.get("behavior") or {}
    kind = plan["verification_kind"]
    if (behavior.get("status") != "PASS" or behavior.get("covers_plan") is not True or behavior.get("executable") is not True
            or behavior.get("checks_id") != contract["checks_id"] or behavior.get("mechanism_id") != kind):
        raise ValueError("Positive edit lacks its exact executable behavior/check binding")
    pack = contract["registry"]["packs"][kind]
    before, after = behavior.get("parent_probe") or {}, behavior.get("child_probe") or {}
    if before.get("mechanism_id") != kind or after.get("mechanism_id") != kind:
        raise ValueError("Saved behavior probes belong to another mechanism")
    if (not all(before.get("checks", {}).get(name) is False for name in pack["goal_checks"])
            or not all(before.get("checks", {}).get(name) is True for name in pack["preserve_checks"])
            or not all(after.get("checks", {}).get(name) is True for name in pack["goal_checks"] + pack["preserve_checks"])):
        raise ValueError("Every existing target and preservation check must have its expected parent/child value")
    frozen_parent = next(row["parent_probe"] for row in contract["preprobed"] if row["mechanism_id"] == kind)
    if before != frozen_parent:
        raise ValueError("Saved parent behavior differs from the exact preteacher reproduction")
    attempt = record.get("implementation") or {}
    _implementation(state, plan, attempt, contract, files)
    if behavior.get("scope") != attempt.get("scope"):
        raise ValueError("Saved behavior scope differs from the actual verified implementation scope")
    if ((q.get("implementation_task_result") or {}).get("evaluation_spec", {}).get("task")
            != q["proposal_task_result"]["evaluation_spec"]["task"]):
        raise ValueError("Parent and child qualification executions belong to different proposal tasks")
    _result(q.get("implementation_task_result"), state, attempt["code_hash"], "child", files)
    from research1.data_pipeline.independent_plan_audit import gate_applies as independent_gate, validate_independent_audit, seal_file_bindings
    if independent_gate(state=state):
        validate_independent_audit(record)
        proof = q.get("independent_plan_audit") or {}
        files.update(seal_file_bindings(proof))
    return files


def seal_positive_artifacts(record: dict) -> dict:
    """Controller-only first-admission operation. Never invoke to fix a seal."""
    if not artifact_gate_applies(record):
        return {}
    if record.get("admission_artifacts"):
        valid, reason = validate_positive_artifacts(record)
        if not valid:
            raise ValueError(reason)
        return record["admission_artifacts"]
    try:
        files = _materials(record)
    except (ValueError, TypeError, KeyError, OSError, StopIteration, AttributeError, IndexError) as exc:
        raise ValueError("Cannot seal invalid positive-admission artifacts") from exc
    seal = {"schema": SCHEMA, "record_sha256": _identity(record), "files": files}
    seal["manifest_sha256"] = digest(seal)
    return seal


def validate_positive_artifacts(record: dict) -> tuple[bool, str]:
    """Export-time pure validation; missing/changed proofs exclude the sample."""
    if not artifact_gate_applies(record):
        return True, "historical_artifact_gate_unchanged"
    try:
        seal = record.get("admission_artifacts")
        if not isinstance(seal, dict) or seal.get("schema") != SCHEMA:
            raise ValueError("New-protocol positive admission lacks its original artifact seal")
        if seal.get("manifest_sha256") != digest({key: value for key, value in seal.items() if key != "manifest_sha256"}):
            raise ValueError("Positive-admission artifact manifest changed")
        if seal.get("record_sha256") != _identity(record):
            raise ValueError("Positive-admission plan/state/qualification changed after sealing")
        if seal.get("files") != _materials(record):
            raise ValueError("Original positive-admission artifact bytes changed after sealing")
        return True, "verified_v6_positive_artifact_binding"
    except (ValueError, TypeError, KeyError, OSError, StopIteration, AttributeError, IndexError):
        return False, "v6_positive_admission_artifact_invalid"
