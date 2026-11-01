"""Independent blinded curator gate before implementation; no models or probes here."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from research1.data_pipeline.policy import digest, public_view, validate_plan

POLICY_ID = 'research1.independent-plan-audit.v1'
REQUEST_SCHEMA = 'research1.independent-plan-audit-request.v1'
REPORT_SCHEMA = 'research1.independent-plan-audit-report.v1'
RECEIPT_SCHEMA = 'research1.independent-plan-audit-coordinator-receipt.v1'
REPORT_NAME = 'external_auditor_report.json'
RECEIPT_NAME = 'coordinator_receipt.json'
REQUEST_NAME = 'audit_request.json'
TERMINAL_DECISIONS = frozenset({'APPROVE', 'REJECT'})
ALLOWED_AUDITOR_KINDS = frozenset({'cursor_cli_blinded_plan_audit', 'independent_plan_curator'})
_POST_REVIEW_STATE_FIELDS = frozenset({
    'plan_factual_review', 'plan_contract_binding', 'independent_plan_audit_protocol_ref',
})
_HASH_EXCLUDED = frozenset({'request_sha256', 'request_record_sha256', '_request_path'})


def gate_applies(*, state: dict | None = None, protocol: dict | None = None) -> bool:
    if protocol is not None:
        return protocol.get('independent_plan_audit_policy') == POLICY_ID
    if state is not None:
        return state.get('independent_plan_audit_policy') == POLICY_ID
    return False


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, json_object: bool = False, raw_bytes: bool = False) -> tuple[Path, bytes, str]:
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Independent audit artifact path must be absolute')
    if path.is_symlink() or not path.is_file():
        raise ValueError('Independent audit binding requires an absolute regular file')
    raw = path.read_bytes()
    if json_object:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Bound JSON artifact must remain an object')
    elif not raw_bytes and path.suffix.lower() == '.json':
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError('Bound JSON artifact must remain an object')
    return path, raw, _sha_bytes(raw)


def _bind(path_value: str | None, *, json_object: bool = False, optional: bool = False) -> dict[str, str] | None:
    if path_value is None or path_value == '':
        if optional:
            return None
        raise ValueError('Required independent audit source path missing')
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError('Independent audit artifact path must be absolute')
    if not path.is_file():
        if optional:
            return None
        raise ValueError('Independent audit binding requires an absolute regular file')
    if path.is_symlink():
        raise ValueError('Independent audit binding requires an absolute regular file')
    encoding = 'json_object' if json_object else 'raw'
    _, _, sha = _read_regular(path, json_object=json_object)
    return {'path': str(path), 'sha256': sha, 'encoding': encoding}


def _bind_bytes(path_value: str | None, *, optional: bool = False) -> dict[str, str] | None:
    if path_value is None or path_value == '':
        if optional:
            return None
        raise ValueError('Required independent audit source path missing')
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError('Independent audit artifact path must be absolute')
    if not path.is_file():
        if optional:
            return None
        raise ValueError('Independent audit fixture source must be an absolute regular file')
    if path.is_symlink():
        raise ValueError('Independent audit fixture source must be an absolute regular file')
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': _sha_bytes(raw), 'encoding': 'bytes'}


def _verify_bound(meta: dict[str, str] | None, *, json_object: bool = False, raw_bytes: bool = False) -> None:
    if meta is None:
        return
    path, raw, sha = _read_regular(Path(meta['path']), json_object=json_object, raw_bytes=raw_bytes)
    if meta.get('sha256') != sha:
        raise ValueError('Independent audit bound artifact bytes changed')
    expected = meta.get('encoding')
    if expected == 'json_object' and not json_object:
        raise ValueError('Independent audit binding expects a JSON object artifact')
    if expected == 'bytes' and not raw_bytes:
        raise ValueError('Independent audit binding expects raw source bytes')
    if expected == 'raw' and json_object:
        raise ValueError('Independent audit binding expects opaque raw bytes')


def _contract_blind(contract: dict) -> dict:
    keys = ('checks_id', 'registry', 'preprobed', 'binding_mode', 'evidence_hashes')
    return {key: copy.deepcopy(contract[key]) for key in keys if key in contract}


def _blind_input(state: dict, plan: dict, *, contract: dict, fixture_source_path: str | None) -> dict:
    blind: dict[str, Any] = {'plan': copy.deepcopy(plan), 'public_view': public_view(state)}
    blind['preteacher_contract'] = _contract_blind(contract)
    fixture = state.get('applicability_fixture_source')
    if isinstance(fixture, dict):
        blind['applicability_fixture_context'] = copy.deepcopy(fixture)
    if fixture_source_path:
        path, _, sha = _read_regular(Path(fixture_source_path), raw_bytes=True)
        if blind.get('applicability_fixture_context', {}).get('sha256') not in {None, sha}:
            raise ValueError('Fixture source bytes disagree with frozen applicability context')
        blind['fixture_source_bytes_sha256'] = sha
    return blind


def _factual_bindings(review: dict, review_dir: Path) -> dict[str, dict[str, str]]:
    review_path = review.get('review_path') or str(review_dir / 'factual_review.json')
    catalog_path = review.get('witness_catalog_path') or str(review_dir / 'factual_witness_catalog.json')
    input_path = review_dir / 'factual_review_input.json'
    started_path = review_dir / 'factual_review_started.json'
    result_path = Path(review_path).parent / 'factual_review_result.json'
    return {
        'factual_review_input': _bind(str(input_path), json_object=True),
        'factual_review_started': _bind(str(started_path), json_object=True),
        'factual_raw': _bind(review_path, json_object=False),
        'factual_result': _bind(str(result_path), json_object=True),
        'factual_witness_catalog': _bind(catalog_path, json_object=True),
    }


def _applicability_bindings(binding: dict | None) -> dict[str, dict[str, str] | None]:
    if not isinstance(binding, dict):
        return {}
    out: dict[str, dict[str, str] | None] = {}
    for kind in ('input', 'output', 'started'):
        key = 'applicability_' + kind + '_path'
        if binding.get(key):
            encoding = 'json_object' if kind in {'input', 'started', 'result'} else 'raw'
            out['applicability_' + kind] = _bind(binding[key], json_object=(encoding == 'json_object'))
    if binding.get('applicability_input_path'):
        result = str(Path(binding['applicability_input_path']).parent / 'applicability_result.json')
        out['applicability_result'] = _bind(result, json_object=True)
    return out


def _probe_io_bindings(state: dict) -> dict[str, dict[str, str]]:
    proofs = state.get('predecision_probe_artifacts') or {}
    files: dict[str, dict[str, str]] = {}
    for label in ('raw_contract', 'receipt'):
        meta = proofs.get(label) or {}
        bound = _bind(meta.get('path'), json_object=True)
        if bound and meta.get('sha256') and bound['sha256'] != meta['sha256']:
            raise ValueError('Predecision probe artifact changed since publication')
        if bound:
            files['predecision_probe_' + label] = bound
    return files


def _candidate_binding(state: dict, plan: dict, *, teacher_path: str, contract_raw: bytes, contract: dict) -> dict[str, str]:
    teacher_meta = _bind(teacher_path, json_object=True)
    if teacher_meta is None:
        raise ValueError('Teacher record binding missing')
    return {
        'state_id': state['state_id'],
        'arm_id': state.get('teacher_arm_id') or '',
        'plan_digest': digest(plan),
        'public_view_digest': digest(public_view(state)),
        'teacher_record_sha256': teacher_meta['sha256'],
        'preteacher_contract_sha256': _sha_bytes(contract_raw),
        'checks_id': contract.get('checks_id', ''),
    }


def _request_body(state: dict, plan: dict, factual_review: dict, app_binding: dict | None, *,
                  protocol_ref: dict, contract_path: str, teacher_path: str, state_path: str,
                  review_dir: Path, campaign: Path) -> dict[str, Any]:
    contract_raw = Path(contract_path).read_bytes()
    contract = json.loads(contract_raw)
    if not isinstance(contract, dict):
        raise ValueError('Preteacher contract must be a JSON object')
    fixture_ref = state.get('applicability_fixture_source') or {}
    fixture_path = fixture_ref.get('path') if isinstance(fixture_ref, dict) else None
    if not fixture_path and protocol_ref.get('fixture_source_ref', {}).get('path'):
        fixture_path = protocol_ref['fixture_source_ref']['path']
    body: dict[str, Any] = {
        'schema': REQUEST_SCHEMA,
        'policy': POLICY_ID,
        'candidate_binding': _candidate_binding(state, plan, teacher_path=teacher_path,
            contract_raw=contract_raw, contract=contract),
        'protocol_ref': copy.deepcopy(protocol_ref),
        'source_bindings': {
            'state_json': _bind(state_path, json_object=True),
            'protocol_json': _bind(protocol_ref['path'], json_object=True),
            'teacher_json': _bind(teacher_path, json_object=True),
            'preteacher_contract_json': {'path': str(Path(contract_path).resolve()),
                'sha256': _sha_bytes(contract_raw), 'encoding': 'json_object'},
        },
        'review_bindings': _factual_bindings(factual_review, review_dir),
        'applicability_bindings': _applicability_bindings(app_binding),
        'probe_io_bindings': _probe_io_bindings(state),
        'campaign_root': str(campaign.resolve()),
    }
    if fixture_path:
        body['source_bindings']['applicability_fixture_source'] = _bind_bytes(fixture_path)
    facts_dir = campaign / 'facts' / 'edittrial-parent'
    for label, name, json_object in (('fact_input', 'fact_input.json', True),
            ('fact_output', 'fact_output.json', False)):
        fact_path = facts_dir / name
        if fact_path.is_file():
            body['source_bindings'][label] = _bind(str(fact_path), json_object=json_object, optional=False)
    blind = _blind_input(state, plan, contract=contract, fixture_source_path=fixture_path)
    body['blind_reading_input'] = blind
    body['blind_reading_input_digest'] = digest(blind)
    return body


def _request_hash(body: dict) -> str:
    payload = {key: value for key, value in body.items() if key not in _HASH_EXCLUDED}
    return digest(payload)


def _load_json_binding(meta: dict[str, str]) -> dict:
    _, raw, _ = _read_regular(Path(meta['path']), json_object=True)
    return json.loads(raw)


def _assert_frozen_state_matches_record(frozen_state: dict, record_state: dict) -> None:
    for key, value in frozen_state.items():
        if key in _POST_REVIEW_STATE_FIELDS:
            continue
        if record_state.get(key) != value:
            raise ValueError('Current candidate state diverged from the frozen independent-audit decision state')


def _verify_factual_saved_proof(state: dict, plan: dict, review: dict) -> None:
    if review.get('status') != 'PASS':
        raise ValueError('Independent audit requires an original saved factual PASS')
    from research1.data_pipeline.applicability import validate_factual_review
    if not validate_factual_review(review, state, plan):
        raise ValueError('Original factual review no longer validates against saved physical proofs')


def _verify_applicability_saved_proof(state: dict, plan: dict, binding: dict | None, contract: dict) -> None:
    if plan.get('action') == 'keep':
        return
    if not isinstance(binding, dict):
        raise ValueError('Edit candidate lacks applicability binding artifacts')
    from research1.data_pipeline.applicability import validate_binding
    if not validate_binding(binding, state, plan, contract):
        raise ValueError('Original applicability binding no longer validates against saved physical proofs')


def validate_frozen_request(request: dict, *, record: dict | None = None, plan: dict | None = None,
        state: dict | None = None, factual_review: dict | None = None, app_binding: dict | None = None,
        contract: dict | None = None, request_path: str | None = None) -> dict:
    """Pure read validation of immutable request bytes and every bound source."""
    if request.get('schema') != REQUEST_SCHEMA or request.get('policy') != POLICY_ID:
        raise ValueError('Independent audit request is not the frozen v1 policy')
    if request.get('request_sha256') != _request_hash(request):
        raise ValueError('Independent audit request canonical identity is inconsistent')
    path_value = request_path or request.get('_request_path')
    if not path_value:
        raise ValueError('Independent audit requires its original physical request')
    _, raw, record_sha = _read_regular(Path(path_value), json_object=True)
    if request.get('request_record_sha256') != record_sha:
        raise ValueError('Independent audit request record bytes changed')
    stored_body = {key: value for key, value in request.items()
                   if key not in {'request_record_sha256', '_request_path'}}
    expected_bytes = (json.dumps(stored_body, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    if raw != expected_bytes:
        raise ValueError('Independent audit request file bytes differ from frozen contents')
    binding = request.get('candidate_binding') or {}
    for group in ('source_bindings', 'review_bindings', 'applicability_bindings', 'probe_io_bindings'):
        entries = request.get(group) or {}
        if not isinstance(entries, dict):
            raise ValueError('Independent audit binding group must be an object')
        for label, meta in entries.items():
            if meta is None:
                continue
            json_object = meta.get('encoding') == 'json_object'
            raw_bytes = meta.get('encoding') == 'bytes'
            _verify_bound(meta, json_object=json_object, raw_bytes=raw_bytes)
    frozen_state = _load_json_binding(request['source_bindings']['state_json'])
    frozen_plan = copy.deepcopy((request.get('blind_reading_input') or {}).get('plan'))
    if not isinstance(frozen_plan, dict):
        raise ValueError('Independent audit blind plan binding missing')
    required_sources = {'state_json', 'protocol_json', 'teacher_json',
                        'preteacher_contract_json', 'applicability_fixture_source'}
    required_reviews = {'factual_review_input', 'factual_review_started', 'factual_raw',
                        'factual_result', 'factual_witness_catalog'}
    if (not required_sources.issubset(request.get('source_bindings', {}))
            or set(request.get('review_bindings', {})) != required_reviews
            or set(request.get('probe_io_bindings', {})) != {'predecision_probe_raw_contract', 'predecision_probe_receipt'}):
        raise ValueError('Independent audit source manifest is incomplete')
    required_app = {'applicability_input', 'applicability_output', 'applicability_started', 'applicability_result'}
    if set(request.get('applicability_bindings', {})) != (required_app if frozen_plan.get('action') == 'edit' else set()):
        raise ValueError('Independent audit applicability manifest is incomplete')
    contract_path = request['source_bindings']['preteacher_contract_json']['path']
    contract_raw = Path(contract_path).read_bytes()
    if request['source_bindings']['preteacher_contract_json']['sha256'] != _sha_bytes(contract_raw):
        raise ValueError('Preteacher contract bytes changed')
    if contract is None:
        contract = json.loads(contract_raw)
    if contract != frozen_state.get('contract'):
        raise ValueError('Independent audit contract differs from the original decision state')
    expected_candidate = _candidate_binding(frozen_state, frozen_plan,
        teacher_path=request['source_bindings']['teacher_json']['path'],
        contract_raw=contract_raw, contract=contract)
    expected_blind = _blind_input(frozen_state, frozen_plan, contract=contract,
        fixture_source_path=request['source_bindings']['applicability_fixture_source']['path'])
    if (binding != expected_candidate or request.get('blind_reading_input') != expected_blind
            or request.get('blind_reading_input_digest') != digest(expected_blind)):
        raise ValueError('Independent audit candidate binding is internally inconsistent')
    review_dir = Path(request['review_bindings']['factual_raw']['path']).parent
    if factual_review is None:
        factual_review = _load_json_binding(request['review_bindings']['factual_result'])
    _verify_factual_saved_proof(frozen_state, validate_plan(frozen_plan, frozen_state), factual_review)
    if app_binding is None and record is not None:
        app_binding = (record.get('qualification') or {}).get('applicability')
    if app_binding is None and frozen_plan.get('action') == 'edit':
        app_binding = _load_json_binding(request['applicability_bindings']['applicability_result'])
    _verify_applicability_saved_proof(frozen_state, frozen_plan, app_binding, contract)
    if record is not None:
        rec_plan = plan if plan is not None else record.get('plan')
        rec_state = state if state is not None else record.get('state')
        if not isinstance(rec_plan, dict) or not isinstance(rec_state, dict):
            raise ValueError('Independent audit record lacks plan/state identity')
        if (binding.get('plan_digest') != digest(rec_plan)
                or binding.get('public_view_digest') != digest(public_view(rec_state))
                or binding.get('state_id') != rec_state.get('state_id')
                or binding.get('arm_id') != rec_state.get('teacher_arm_id')):
            raise ValueError('Independent audit artifacts belong to another candidate')
        teacher_sha = request['source_bindings']['teacher_json']['sha256']
        if binding.get('teacher_record_sha256') != teacher_sha:
            raise ValueError('Independent audit teacher identity mismatch')
        if str(rec_state.get('teacher_record_path', '')).strip():
            live_teacher = _bind(rec_state['teacher_record_path'], json_object=True)
            if live_teacher is None or live_teacher['sha256'] != teacher_sha:
                raise ValueError('Current teacher record differs from the bound original')
        _assert_frozen_state_matches_record(frozen_state, rec_state)
    return {'verified': True, 'review_dir': str(review_dir.resolve()), 'state': frozen_state, 'plan': frozen_plan,
            'contract': contract, 'factual_review': factual_review}


def freeze_audit_request(state: dict, plan: dict, factual_review: dict, app_binding: dict | None,
        directory: Path, *, protocol_ref: dict, campaign: Path | None = None,
        contract_path: str | None = None) -> dict:
    """Create-once immutable audit request; never overwrites differing bytes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    campaign = Path(campaign or directory.parent.parent)
    arm = state.get('teacher_arm_id') or directory.name
    review_dir = campaign / 'reviews' / 'edittrial' / arm
    contract_path = contract_path or str(campaign / 'edittrial_preteacher_contract.json')
    state_path = str(campaign / 'states' / state['state_id'] / 'state.json')
    teacher_path = state['teacher_record_path']
    body = _request_body(state, plan, factual_review, app_binding, protocol_ref=protocol_ref,
        contract_path=contract_path, teacher_path=teacher_path, state_path=state_path,
        review_dir=review_dir, campaign=campaign)
    body['request_sha256'] = _request_hash(body)
    path = directory / REQUEST_NAME
    if path.is_file():
        _, raw, _ = _read_regular(path.resolve(), json_object=True)
        expected = (json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
        if raw != expected:
            raise ValueError('Independent audit request already frozen with different bytes')
    else:
        raw = (json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
        # Binary exclusive creation preserves the same bytes on Windows/Linux.
        with path.open('xb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    record_sha = _sha_bytes(path.read_bytes())
    # The file's SHA is metadata outside the file it hashes.
    body['request_record_sha256'] = record_sha
    body['_request_path'] = str(path.resolve())
    return body


def _read_json_ref(ref: dict, *, schema: str) -> tuple[dict, str]:
    path, raw, sha = _read_regular(Path(ref['path']), json_object=True)
    value = json.loads(raw)
    if value.get('schema') != schema:
        raise ValueError('Independent audit artifact schema mismatch')
    if ref.get('sha256') and ref['sha256'] != sha:
        raise ValueError('Independent audit artifact bytes changed')
    return value, sha


def _validate_provenance(report: dict, receipt: dict, *, report_sha: str, auditor_report_ref: dict) -> tuple[str, str]:
    auditor = report.get('auditor') or {}
    if not isinstance(auditor, dict) or auditor.get('kind') not in ALLOWED_AUDITOR_KINDS:
        raise ValueError('Independent audit report lacks allowed auditor provenance')
    task_id = auditor.get('task_id')
    model = auditor.get('model')
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError('Independent audit report lacks auditor task id')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('Independent audit report lacks auditor model')
    provenance = receipt.get('auditor_task_provenance') or {}
    prov_path = provenance.get('path')
    if not prov_path:
        raise ValueError('Coordinator receipt lacks auditor task provenance path')
    prov_file, prov_raw, prov_sha = _read_regular(Path(prov_path), json_object=True)
    if provenance.get('sha256') != prov_sha:
        raise ValueError('Auditor task provenance bytes changed')
    provenance_doc = json.loads(prov_raw)
    if provenance_doc.get('task_id') != task_id or provenance_doc.get('model') != model:
        raise ValueError('Auditor task provenance does not match report auditor identity')
    if str(provenance_doc.get('status', '')).upper() != 'DONE':
        raise ValueError('Auditor task provenance does not record terminal DONE status')
    if type(provenance_doc.get('exit_code')) is not int or provenance_doc['exit_code'] != 0:
        raise ValueError('Auditor task provenance must record exit_code 0')
    if receipt.get('auditor_task_id') != task_id:
        raise ValueError('Coordinator receipt task id differs from report auditor')
    if receipt.get('auditor_model') != model:
        raise ValueError('Coordinator receipt model differs from report auditor')
    report_path = receipt.get('report_path')
    if not report_path or str(Path(report_path).resolve()) != str(Path(auditor_report_ref['path']).resolve()):
        raise ValueError('Coordinator receipt report path does not match physical report')
    if receipt.get('report_sha256') != report_sha:
        raise ValueError('Coordinator receipt report hash mismatch')
    return str(prov_file.resolve()), prov_sha


def read_independent_audit(request: dict, *, auditor_report_ref: dict, coordinator_receipt_ref: dict,
        record: dict | None = None) -> dict:
    """Validate external report + coordinator receipt against a frozen request."""
    request_path = request.get('_request_path')
    validate_frozen_request(request, record=record, request_path=request_path)
    expected_hash = request.get('request_sha256')
    report, report_sha = _read_json_ref(auditor_report_ref, schema=REPORT_SCHEMA)
    receipt, receipt_sha = _read_json_ref(coordinator_receipt_ref, schema=RECEIPT_SCHEMA)
    if report.get('request_sha256') != expected_hash or receipt.get('request_sha256') != expected_hash:
        raise ValueError('Independent audit report or receipt binds a different request')
    if report.get('decision') != receipt.get('decision'):
        raise ValueError('Independent audit report and coordinator receipt decisions differ')
    decision = report.get('decision')
    if decision not in TERMINAL_DECISIONS | {'PENDING'}:
        raise ValueError('Independent audit decision is not recognized')
    if not isinstance(report.get('findings'), str) or not report.get('findings', '').strip():
        raise ValueError('Independent audit report lacks concrete findings')
    if not isinstance(report.get('reason'), str) or not report['reason'].strip():
        raise ValueError('Independent audit report lacks reason')
    prov_path, prov_sha = _validate_provenance(report, receipt, report_sha=report_sha,
        auditor_report_ref=auditor_report_ref)
    coordinator = receipt.get('coordinator') or {}
    if not isinstance(coordinator.get('identity'), str) or not coordinator['identity'].strip():
        raise ValueError('Coordinator receipt lacks coordinator identity')
    if not isinstance(receipt.get('reason'), str) or not receipt['reason'].strip():
        raise ValueError('Coordinator receipt lacks reason')
    auditor = report.get('auditor') or {}
    proof = {
        'schema': 'research1.independent-plan-audit-proof.v1',
        'policy': POLICY_ID,
        'request_sha256': expected_hash,
        'request_record_sha256': request.get('request_record_sha256'),
        'decision': decision,
        'status': decision if decision in TERMINAL_DECISIONS else 'PENDING',
        'request_path': str(Path(request_path).resolve()) if request_path else None,
        'report_path': str(Path(auditor_report_ref['path']).resolve()),
        'report_sha256': report_sha,
        'coordinator_receipt_path': str(Path(coordinator_receipt_ref['path']).resolve()),
        'coordinator_receipt_sha256': receipt_sha,
        'auditor_task_provenance_path': prov_path,
        'auditor_task_provenance_sha256': prov_sha,
        'auditor_kind': auditor.get('kind'),
        'auditor_task_id': auditor.get('task_id'),
        'auditor_model': auditor.get('model'),
        'findings': report.get('findings'),
        'reason': report.get('reason'),
        'coordinator_identity': coordinator.get('identity'),
    }
    proof['proof_sha256'] = digest({key: value for key, value in proof.items() if key != 'proof_sha256'})
    return proof


def validate_independent_audit(record: dict) -> dict:
    """Pure read validation for seal/export; rereads bound bytes and candidate identity."""
    if not gate_applies(state=record.get('state') or {}):
        return {'schema': 'research1.independent-plan-audit-proof.v1', 'verified': True,
                'reason': 'independent_plan_audit_not_applicable'}
    proof = (record.get('qualification') or {}).get('independent_plan_audit')
    if not isinstance(proof, dict) or proof.get('policy') != POLICY_ID:
        raise ValueError('Verified positive lacks independent plan audit proof')
    if proof.get('decision') != 'APPROVE':
        raise ValueError('Positive admission requires independent APPROVE audit')
    request_path = proof.get('request_path')
    if not request_path:
        raise ValueError('Independent audit proof lacks request path')
    _, raw, request_record_sha = _read_regular(Path(request_path), json_object=True)
    if proof.get('request_record_sha256') != request_record_sha:
        raise ValueError('Independent audit request record bytes changed')
    request = json.loads(raw)
    request['_request_path'] = str(Path(request_path).resolve())
    request['request_record_sha256'] = request_record_sha
    rebound = read_independent_audit(request,
        auditor_report_ref={'path': proof['report_path'], 'sha256': proof.get('report_sha256')},
        coordinator_receipt_ref={'path': proof['coordinator_receipt_path'], 'sha256': proof.get('coordinator_receipt_sha256')},
        record=record)
    for key in ('decision', 'request_sha256', 'request_record_sha256', 'report_sha256',
                'coordinator_receipt_sha256', 'auditor_task_provenance_sha256', 'auditor_task_id', 'auditor_model'):
        if proof.get(key) != rebound.get(key):
            raise ValueError('Independent audit proof no longer matches physical artifacts')
    return {'schema': 'research1.independent-plan-audit-proof.v1', 'verified': True,
            'request_sha256': proof['request_sha256'], 'request_record_sha256': proof.get('request_record_sha256'),
            'proof_sha256': proof.get('proof_sha256')}


def seal_file_bindings(proof: dict) -> dict[str, dict[str, str]]:
    files = {}
    for label, path_key, sha_key in (
            ('independent_audit_request', 'request_path', 'request_record_sha256'),
            ('independent_audit_report', 'report_path', 'report_sha256'),
            ('independent_audit_coordinator_receipt', 'coordinator_receipt_path', 'coordinator_receipt_sha256'),
            ('independent_audit_task_provenance', 'auditor_task_provenance_path', 'auditor_task_provenance_sha256')):
        path_value = proof.get(path_key)
        if not path_value:
            continue
        json_object = label != 'independent_audit_task_provenance'
        _, _, sha = _read_regular(Path(path_value), json_object=json_object)
        if proof.get(sha_key) and sha != proof[sha_key]:
            raise ValueError('Independent audit sealed artifact changed')
        files[label] = {'path': str(Path(path_value).resolve()), 'sha256': sha}
    return files
