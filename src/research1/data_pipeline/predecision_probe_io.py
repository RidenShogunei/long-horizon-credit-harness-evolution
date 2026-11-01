"""Pre-teacher publication of deterministic parent Harness API probe I/O.

Projects concrete local parser probe outputs already frozen in the contract.
Never re-executes probes, models, tasks, or scorers.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from research1.data_pipeline.module_adapters import get_adapter
from research1.data_pipeline.policy import _atomic_json

SCHEMA = "research1.predecision-harness-api-io.v1"
SOURCE_KIND = "predecision_harness_api_probe"
RAW_CONTRACT_NAME = "predecision_harness_api_io_raw_contract.json"
RECEIPT_NAME = "predecision_harness_api_io_receipt.json"
EVIDENCE_ID_PREFIX = "predecision.harness_api.io."
EXECUTION_NOTE = (
    "Local Harness API execution on frozen parent modules; not an actual original task step, "
    "not hidden test evidence, and not a task-score cause or utility claim."
)
_FALSE_CONCAT_MECHANISM = "false_concat_preservation"
_FALSE_CONCAT_API = "BaselineTools.parse_llm_response"
_PUBLIC_ROOT_KEYS = frozenset(
    {
        "schema",
        "source_kind",
        "execution_note",
        "parent_code_hash",
        "registry_version",
        "adapter_source_sha256",
        "invoked_api",
        "observations",
    }
)
_PUBLIC_OBSERVATION_KEYS = frozenset(
    {
        "case",
        "fixture_defined_input",
        "input_provenance",
        "parsed_commands",
        "warning",
        "error",
        "is_task_complete",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "checks",
        "goal_checks",
        "preserve_checks",
        "target_reproduced",
        "preservation_baseline_passed",
        "selected",
        "mechanism_id",
        "module",
        "local_functions",
        "functions",
        "needle",
        "after",
        "role",
        "goal",
        "preserve",
        "target",
        "verification_kind",
        "intended_change",
        "repair",
        "parent_probe",
        "preprobed",
        "evidence_refs",
        "witness",
        "case_id",
    }
)
_FIXTURE_INPUT_KEYS = frozenset({"analysis", "plan", "commands", "task_complete"})


def _walk_forbidden_keys(value: Any, *, fixture_subtree: bool = False) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError("projected evidence leaked forbidden probe metadata")
            if fixture_subtree:
                if key not in _FIXTURE_INPUT_KEYS and not isinstance(item, (dict, list, str, int, float, bool, type(None))):
                    raise ValueError("projected evidence includes undocumented fixture field")
            _walk_forbidden_keys(item, fixture_subtree=fixture_subtree or key == "fixture_defined_input")
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden_keys(item, fixture_subtree=fixture_subtree)


def _assert_public_body(body: dict[str, Any]) -> None:
    if set(body.keys()) != _PUBLIC_ROOT_KEYS:
        raise ValueError("projected evidence includes undocumented field")
    observations = body.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a non-empty list")
    for row in observations:
        if not isinstance(row, dict) or set(row.keys()) != _PUBLIC_OBSERVATION_KEYS:
            raise ValueError("projected observation includes undocumented field")
        if row["input_provenance"] == "fixture_definition":
            fixture = row["fixture_defined_input"]
            if not isinstance(fixture, dict) or set(fixture.keys()) != _FIXTURE_INPUT_KEYS:
                raise ValueError("fixture_defined_input shape differs from frozen definition")
        elif row["input_provenance"] == "not_available":
            if row["fixture_defined_input"] is not None:
                raise ValueError("missing fixture input must be null")
        else:
            raise ValueError("unknown input_provenance")
    _walk_forbidden_keys(body)


def _reject_symlink_ancestors(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError("predecision probe path cannot use symlinks")
        if current.parent == current:
            return
        current = current.parent


def _require_absolute_directory(directory: Path) -> Path:
    directory = Path(directory)
    if not directory.is_absolute():
        raise ValueError("predecision probe directory must be absolute")
    _reject_symlink_ancestors(directory)
    resolved = directory.resolve()
    _reject_symlink_ancestors(resolved)
    return resolved


def _read_file(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("predecision probe artifact must be an absolute regular file")
    _reject_symlink_ancestors(path.parent)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("predecision probe artifact is not an object")
    return value, hashlib.sha256(raw).hexdigest()


def _adapter_digest(adapter, value: Any) -> str:
    return adapter._digest(value)  # noqa: SLF001 — matches build_contract identity


def _checks_id(adapter, contract: dict[str, Any]) -> str:
    payload = {key: value for key, value in contract.items() if key != "checks_id"}
    return "checks_" + _adapter_digest(adapter, payload)[:24]


def _evidence_digest(adapter, item: dict[str, Any]) -> str:
    return _adapter_digest(adapter, item)


def _parent_code_hash(adapter, state: dict[str, Any]) -> str:
    return adapter.code_hash(Path(state["modules_root"]))


def _adapter_source_sha256(adapter) -> str:
    return hashlib.sha256(Path(adapter.__file__).read_bytes()).hexdigest()


def _fixture_input_source() -> dict[str, str]:
    import research1.plan_init as plan_init

    path = Path(plan_init.__file__).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _evidence_hash_mapping(adapter, evidence: list[dict[str, Any]]) -> dict[str, str]:
    return {item["id"]: _evidence_digest(adapter, item) for item in evidence}


def _validate_original_contract(
    state: dict[str, Any],
    contract: dict[str, Any],
    adapter,
    *,
    evidence: list[dict[str, Any]] | None = None,
) -> None:
    if contract.get("checks_id") != _checks_id(adapter, contract):
        raise ValueError("contract checks_id is inconsistent")
    if contract.get("registry") != adapter.frozen_registry():
        raise ValueError("contract registry differs from adapter")
    rows = evidence if evidence is not None else state.get("evidence") or []
    if _evidence_hash_mapping(adapter, rows) != (contract.get("evidence_hashes") or {}):
        raise ValueError("contract evidence_hashes differ from current state evidence")
    try:
        live_hashes = adapter.tree_hashes(Path(state["modules_root"]))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError("parent module tree unavailable") from exc
    if contract.get("parent_hashes") != live_hashes:
        raise ValueError("contract parent_hashes differ from current modules_root")
    module_code = state.get("module_code") or {}
    if any(
        not isinstance(rel, str)
        or not isinstance(text, str)
        or (Path(state["modules_root"]) / rel).read_text(encoding="utf-8") != text
        for rel, text in module_code.items()
    ):
        raise ValueError("module_code differs from modules_root bytes")


def _validate_protocol_hash(state: dict[str, Any], protocol_hash: str, receipt: dict[str, Any] | None = None) -> None:
    decision = state.get("decision_protocol_hash")
    if decision is not None and protocol_hash != decision:
        raise ValueError("protocol_hash differs from decision_protocol_hash")
    if receipt is not None and decision is not None and receipt.get("protocol_hash") != decision:
        raise ValueError("receipt protocol_hash differs from decision_protocol_hash")


def _ordered_result_keys(results: dict[str, Any]) -> list[str]:
    from research1.plan_init import frozen_false_concat_checks

    known = [row["id"] for row in frozen_false_concat_checks()]
    ordered = [key for key in known if key in results]
    ordered.extend(sorted(key for key in results if key not in known))
    return ordered


def _fixture_payloads() -> dict[str, Any]:
    from research1.plan_init import frozen_false_concat_checks

    return {row["id"]: copy.deepcopy(row["payload"]) for row in frozen_false_concat_checks()}


def _project_observations(
    results: dict[str, Any], *, attach_fixture_inputs: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    fixtures = _fixture_payloads() if attach_fixture_inputs else {}
    observations: list[dict[str, Any]] = []
    case_map: list[dict[str, str]] = []
    for index, source_key in enumerate(_ordered_result_keys(results)):
        row = results[source_key]
        if not isinstance(row, dict):
            raise ValueError("probe result row is not an object")
        if "commands" not in row:
            raise ValueError("probe result row lacks parser commands")
        public_case = f"case_{index}"
        if attach_fixture_inputs and source_key in fixtures:
            fixture_input = fixtures[source_key]
            provenance = "fixture_definition"
        else:
            fixture_input = None
            provenance = "not_available"
        observations.append(
            {
                "case": public_case,
                "fixture_defined_input": fixture_input,
                "input_provenance": provenance,
                "parsed_commands": row["commands"],
                "warning": row.get("warning"),
                "error": row.get("error"),
                "is_task_complete": row.get("is_task_complete"),
            }
        )
        case_map.append({"public_case": public_case, "results_key": source_key})
    return observations, case_map


def _neutral_evidence_id(public_body: dict[str, Any]) -> str:
    canonical = json.dumps(public_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return EVIDENCE_ID_PREFIX + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


def _project_evidence_items(
    contract: dict[str, Any], adapter, state: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    registry = contract.get("registry") or {}
    packs = registry.get("packs") or {}
    parent_probes = contract.get("parent_probes") or {}
    items: list[dict[str, Any]] = []
    metadata_only: list[str] = []
    traces: list[dict[str, Any]] = []
    existing_ids = {item.get("id") for item in state.get("evidence", []) if isinstance(item, dict)}
    for mechanism_id in sorted(packs):
        frozen = parent_probes.get(mechanism_id) or {}
        probe = frozen.get("parent_probe") if isinstance(frozen, dict) else None
        results = probe.get("results") if isinstance(probe, dict) else None
        if not isinstance(results, dict) or not results:
            metadata_only.append(mechanism_id)
            continue
        invoked_api = _FALSE_CONCAT_API if mechanism_id == _FALSE_CONCAT_MECHANISM else None
        observations, case_map = _project_observations(
            results, attach_fixture_inputs=mechanism_id == _FALSE_CONCAT_MECHANISM
        )
        body = {
            "schema": SCHEMA,
            "source_kind": SOURCE_KIND,
            "execution_note": EXECUTION_NOTE,
            "parent_code_hash": _parent_code_hash(adapter, state),
            "registry_version": registry.get("version"),
            "adapter_source_sha256": _adapter_source_sha256(adapter),
            "invoked_api": invoked_api,
            "observations": observations,
        }
        _assert_public_body(body)
        evidence_id = _neutral_evidence_id(body)
        if evidence_id in existing_ids:
            raise ValueError("predecision evidence id collision")
        existing_ids.add(evidence_id)
        items.append({"id": evidence_id, "text": json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")), "partition": "proposal"})
        traces.append(
            {
                "evidence_id": evidence_id,
                "producer_mechanism_id": mechanism_id,
                "observation_case_map": case_map,
            }
        )
    return items, metadata_only, traces


def _finalize_contract(adapter, original_contract: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    final = copy.deepcopy(original_contract)
    final["evidence_hashes"] = _evidence_hash_mapping(adapter, evidence)
    final["checks_id"] = _checks_id(adapter, final)
    return final


def _build_receipt(
    *,
    protocol_hash: str,
    original_contract: dict[str, Any],
    final_contract: dict[str, Any],
    adapter,
    state: dict[str, Any],
    original_evidence_order: list[str],
    added_ids: list[str],
    metadata_only: list[str],
    io_projection_trace: list[dict[str, Any]],
    fixture_input_source: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "protocol_hash": protocol_hash,
        "original_checks_id": original_contract.get("checks_id"),
        "final_checks_id": final_contract.get("checks_id"),
        "parent_code_hash": _parent_code_hash(adapter, state),
        "adapter_source_sha256": _adapter_source_sha256(adapter),
        "registry_version": (original_contract.get("registry") or {}).get("version"),
        "original_evidence_order": original_evidence_order,
        "evidence_ids_added": added_ids,
        "mechanisms_metadata_only": metadata_only,
        "fixture_input_source": fixture_input_source,
        "io_projection_trace": io_projection_trace,
    }


def _artifact_paths(directory: Path) -> tuple[Path, Path]:
    directory = _require_absolute_directory(directory)
    return directory / RAW_CONTRACT_NAME, directory / RECEIPT_NAME


def publish_parent_probe_io(
    state: dict[str, Any],
    contract: dict[str, Any],
    directory: Path,
    *,
    protocol_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(protocol_hash, str) or not protocol_hash:
        raise ValueError("protocol_hash is required")
    directory = _require_absolute_directory(directory)
    _validate_protocol_hash(state, protocol_hash)
    adapter = get_adapter(state)
    original_contract = copy.deepcopy(contract)
    raw_path, receipt_path = _artifact_paths(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if state.get("predecision_probe_io_policy") == SCHEMA:
        finished = copy.deepcopy(state)
        validate_probe_io_state(finished)
        saved_contract, saved_contract_sha = _read_file(raw_path)
        if saved_contract != original_contract:
            raise ValueError("original contract changed since predecision publication")
        recorded = finished.get("predecision_probe_artifacts") or {}
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != (recorded.get("raw_contract") or {}).get("sha256"):
            raise ValueError("raw contract artifact bytes changed")
        if hashlib.sha256(receipt_path.read_bytes()).hexdigest() != (recorded.get("receipt") or {}).get("sha256"):
            raise ValueError("predecision receipt artifact bytes changed")
        receipt, receipt_sha = _read_file(receipt_path)
        if receipt.get("protocol_hash") != protocol_hash:
            raise ValueError("protocol_hash differs from saved receipt")
        _validate_protocol_hash(finished, protocol_hash, receipt)
        if receipt_sha != recorded.get("receipt", {}).get("sha256"):
            raise ValueError("predecision receipt hash mismatch")
        if saved_contract_sha != recorded.get("raw_contract", {}).get("sha256"):
            raise ValueError("raw contract hash mismatch")
        return finished, copy.deepcopy(finished["contract"])

    projected_preview, _, _ = _project_evidence_items(original_contract, adapter, state)
    existing_ids = {item.get("id") for item in state.get("evidence", []) if isinstance(item, dict)}
    for item in projected_preview:
        if item["id"] in existing_ids:
            raise ValueError("predecision evidence id collision")

    _validate_original_contract(state, original_contract, adapter)

    if raw_path.exists() or receipt_path.exists():
        raise ValueError("predecision probe artifacts already exist")

    original_evidence = copy.deepcopy(state.get("evidence", []))
    original_evidence_order = [item["id"] for item in original_evidence]
    projected, metadata_only, io_trace = _project_evidence_items(original_contract, adapter, state)
    new_evidence = original_evidence + projected
    if len(new_evidence) != len(original_evidence) + len(projected):
        raise ValueError("projected evidence count mismatch")
    final_contract = _finalize_contract(adapter, original_contract, new_evidence)
    fixture_source = _fixture_input_source()
    receipt = _build_receipt(
        protocol_hash=protocol_hash,
        original_contract=original_contract,
        final_contract=final_contract,
        adapter=adapter,
        state=state,
        original_evidence_order=original_evidence_order,
        added_ids=[item["id"] for item in projected],
        metadata_only=metadata_only,
        io_projection_trace=io_trace,
        fixture_input_source=fixture_source,
    )

    _atomic_json(raw_path, original_contract)
    _atomic_json(receipt_path, receipt)
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()

    new_state = copy.deepcopy(state)
    new_state["evidence"] = new_evidence
    new_state["contract"] = final_contract
    new_state["predecision_probe_io_policy"] = SCHEMA
    new_state["predecision_probe_artifacts"] = {
        "raw_contract": {"path": str(raw_path), "sha256": raw_sha},
        "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
    }
    validate_probe_io_state(new_state)
    return new_state, copy.deepcopy(final_contract)


def validate_probe_io_state(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("predecision_probe_io_policy") != SCHEMA:
        raise ValueError("state lacks predecision probe publication")
    adapter = get_adapter(state)
    artifacts = state.get("predecision_probe_artifacts") or {}
    raw_meta = artifacts.get("raw_contract") or {}
    receipt_meta = artifacts.get("receipt") or {}
    raw_path = Path(raw_meta.get("path") or "")
    receipt_path = Path(receipt_meta.get("path") or "")
    saved_contract, raw_sha = _read_file(raw_path)
    receipt, receipt_sha = _read_file(receipt_path)
    if raw_sha != raw_meta.get("sha256") or receipt_sha != receipt_meta.get("sha256"):
        raise ValueError("predecision artifact sha256 mismatch")
    if saved_contract.get("checks_id") != receipt.get("original_checks_id"):
        raise ValueError("raw contract checks_id differs from receipt")
    _validate_protocol_hash(state, receipt.get("protocol_hash") or "", receipt)
    original_order = receipt.get("original_evidence_order")
    if not isinstance(original_order, list) or not all(isinstance(item, str) and item for item in original_order):
        raise ValueError("receipt original_evidence_order invalid")
    evidence = state.get("evidence") or []
    added_ids = list(receipt.get("evidence_ids_added") or [])
    if len(evidence) != len(original_order) + len(added_ids):
        raise ValueError("final evidence length differs from original plus projected")
    if [item.get("id") for item in evidence[: len(original_order)]] != original_order:
        raise ValueError("original evidence order or ids changed")
    if [item.get("id") for item in evidence[len(original_order) :]] != added_ids:
        raise ValueError("projected evidence order differs from receipt")
    binding_state = dict(state, evidence=evidence[: len(original_order)])
    _validate_original_contract(binding_state, saved_contract, adapter, evidence=evidence[: len(original_order)])
    projected, metadata_only, io_trace = _project_evidence_items(saved_contract, adapter, binding_state)
    if receipt.get("io_projection_trace") != io_trace:
        raise ValueError("receipt io projection trace differs from reprojection")
    for index, item in enumerate(projected):
        if evidence[len(original_order) + index] != item:
            raise ValueError("projected evidence differs from deterministic reprojection")
    if receipt.get("evidence_ids_added") != [item["id"] for item in projected]:
        raise ValueError("receipt evidence list differs from projection")
    if receipt.get("mechanisms_metadata_only") != metadata_only:
        raise ValueError("receipt metadata-only mechanisms differ")
    fixture_source = _fixture_input_source()
    recorded_source = receipt.get("fixture_input_source") or {}
    if recorded_source != fixture_source:
        raise ValueError("fixture input definition source changed")
    if receipt.get("parent_code_hash") != _parent_code_hash(adapter, state):
        raise ValueError("receipt parent_code_hash differs from current parent")
    if receipt.get("adapter_source_sha256") != _adapter_source_sha256(adapter):
        raise ValueError("receipt adapter source hash differs")
    if receipt.get("registry_version") != (saved_contract.get("registry") or {}).get("version"):
        raise ValueError("receipt registry version differs")
    final_contract = _finalize_contract(adapter, saved_contract, evidence)
    if state.get("contract") != final_contract:
        raise ValueError("final contract binding differs from original-plus-projected evidence")
    if receipt.get("final_checks_id") != final_contract.get("checks_id"):
        raise ValueError("receipt final_checks_id differs from bound contract")
    for key in ("parent_probes", "preprobed", "selected", "pending_reasons", "parent_hashes", "registry", "binding_mode"):
        if saved_contract.get(key) != state.get("contract", {}).get(key):
            raise ValueError("contract probe or scope content changed during publication")
    proof = {
        "schema": SCHEMA,
        "verified": True,
        "artifacts": copy.deepcopy(artifacts),
        "original_checks_id": saved_contract.get("checks_id"),
        "final_checks_id": final_contract.get("checks_id"),
        "parent_code_hash": receipt.get("parent_code_hash"),
        "adapter_source_sha256": receipt.get("adapter_source_sha256"),
        "registry_version": receipt.get("registry_version"),
        "evidence_ids_added": list(added_ids),
        "mechanisms_metadata_only": list(receipt.get("mechanisms_metadata_only") or []),
        "original_evidence_order": list(original_order),
        "fixture_input_source": copy.deepcopy(recorded_source),
    }
    return proof
