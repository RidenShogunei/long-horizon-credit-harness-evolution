"""Independent plan-to-check binding using saved factual witness IDs, not re-quoted text."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from research1.data_pipeline.applicability import (
    BINDING_MODE,
    _candidate,
    _factual_proof,
    _metadata,
    _parse as _legacy_parse,
    _read,
)
from research1.data_pipeline.policy import canonical, digest, public_view, validate_plan, _atomic_json, probe_source_strength

POLICY_ID = 'proposal-plan-applicability.v3.fact_ids'
_RAW_KEYS = frozenset({'status', 'reason', 'mechanism_id', 'witness_ids', 'code_refs', 'covered_checks'})

_INSTRUCTIONS = (
    'Independently assess this single original plan against ONLY the frozen decision-time proposal view, '
    'current visible code/runtime interfaces, the separately saved factual review witness catalog and existing '
    'preteacher checks. Select witness_ids only from factual_witness_catalog.items; do not re-quote proposal '
    'text or invent witnesses. Check whether the proposed implementation is feasible within its own explicit '
    'scope, including how every needed runtime value reaches the named function. Do not silently add another '
    'hook, hidden state, arguments or source access. Check that these exact existing goal and preservation '
    'fixtures cover the whole proposed behavior, not merely a related mechanism name. Do not create, alter or '
    'substitute tests, broaden scope, rewrite the plan, infer a defect from task feedback alone, or use future '
    'implementation/results. Observed process limitations may support a bounded evidence-grounded improvement '
    'even when baseline behavior works as designed; proven causal responsibility for the original failed task, '
    'solver benefit and future task-score improvement are not required before implementation. A saved factual PASS '
    'or any prior applicability PASS does not alone establish applicability or utility. Unknown, ungrounded, infeasible, partially covered or uncertain plans remain '
    'pending. Definition-derived local API inputs are not captured inputs, task success or score evidence. '
    'Return final JSON only with exactly these fields: '
    '{"status":"PASS|pending|FAIL","reason":"...","mechanism_id":"...",'
    '"witness_ids":["existing factual_witness_catalog id"],'
    '"code_refs":[{"path":"visible module or read-only framework source path","text":"exact source substring"}],'
    '"covered_checks":["every existing goal and preservation check id"]}. '
    'For PASS choose nonempty distinct catalog ids, cite exact current source in code_refs, and explain why '
    'the unchanged listed checks cover the plan.\n'
)


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _require_policy(state: dict) -> None:
    if state.get('applicability_policy') != POLICY_ID:
        raise ValueError('Fact-ids applicability requires applicability_policy ' + POLICY_ID)


def _factual_witness_catalog(review: dict) -> dict:
    witness_ids = review.get('witness_ids')
    witnesses = review.get('witnesses')
    if (not isinstance(witness_ids, list) or not isinstance(witnesses, list)
            or not witness_ids or len(witness_ids) != len(witnesses)):
        raise ValueError('Factual PASS lacks a witness id catalog')
    if len(set(witness_ids)) != len(witness_ids):
        raise ValueError('Factual witness ids must be unique')
    items = []
    for identifier, witness in zip(witness_ids, witnesses):
        if not isinstance(identifier, str) or not isinstance(witness, dict):
            raise ValueError('Invalid factual witness catalog entry')
        if not isinstance(witness.get('evidence_ref'), str) or not isinstance(witness.get('text'), str):
            raise ValueError('Factual catalog witness lacks evidence_ref or text')
        row = {'id': identifier, 'evidence_ref': witness['evidence_ref'], 'text': witness['text']}
        for key in ('kind', 'validation', 'locations', 'serialized_evidence_sha256'):
            if key in witness:
                row[key] = witness[key]
        items.append(row)
    return {'review_policy': review.get('review_policy'), 'witness_ids': list(witness_ids), 'items': items}


def _materialize_raw(value: dict, catalog: dict) -> dict:
    lookup = {row['id']: row for row in catalog['items']}
    witness_ids = value['witness_ids']
    if (not isinstance(witness_ids, list) or not witness_ids
            or len(set(witness_ids)) != len(witness_ids)):
        raise ValueError('Applicability PASS requires distinct factual witness ids')
    for identifier in witness_ids:
        if not isinstance(identifier, str) or identifier not in lookup:
            raise ValueError('Unknown factual witness id')
    refs = list(dict.fromkeys(lookup[identifier]['evidence_ref'] for identifier in witness_ids))
    witnesses = [{'evidence_ref': lookup[identifier]['evidence_ref'], 'text': lookup[identifier]['text']}
                 for identifier in witness_ids]
    return {'status': 'PASS', 'reason': value['reason'], 'mechanism_id': value['mechanism_id'],
            'evidence_refs': refs, 'witnesses': witnesses, 'code_refs': value['code_refs'],
            'covered_checks': value['covered_checks']}


def _parse(saved: dict, state: dict, plan: dict, contract: dict, catalog: dict, catalog_hash: str) -> dict:
    _metadata(saved.get('metadata'), state, 'applicability')
    value = json.loads(saved['raw'])
    if (not isinstance(value, dict) or set(value) != _RAW_KEYS
            or value.get('status') not in {'PASS', 'pending', 'FAIL'}
            or not isinstance(value.get('reason'), str) or not value['reason'].strip()):
        raise ValueError('Applicability reply differs from the frozen fact-ids review shape')
    if value['status'] != 'PASS':
        return {'status': 'pending', 'reason': value['reason'], 'raw_status': value['status']}
    materialized = _materialize_raw(value, catalog)
    parsed = _legacy_parse(dict(saved, raw=json.dumps(materialized)), state, plan, contract)
    if parsed.get('status') != 'PASS':
        raise ValueError('Selected factual witness ids failed legacy applicability validation')
    parsed['factual_witness_ids'] = list(value['witness_ids'])
    parsed['factual_witness_catalog_sha256'] = catalog_hash
    return parsed


def _request(state: dict, plan: dict, review: dict, contract: dict) -> tuple[dict, dict, dict, str]:
    _require_policy(state)
    pack, parent = _candidate(state, plan, contract)
    proof = _factual_proof(review, state, plan)
    catalog = _factual_witness_catalog(review)
    catalog_hash = digest(catalog)
    request = {'policy': POLICY_ID, 'parent_view': public_view(state), 'plan': plan,
        'factual_review_sha256': proof['factual_review_sha256'],
        'factual_raw_record_sha256': proof['factual_raw_record_sha256'],
        'factual_result_record_sha256': proof['factual_result_record_sha256'],
        'factual_witness_catalog': catalog, 'factual_witness_catalog_sha256': catalog_hash,
        'frozen_checks': {'checks_id': contract['checks_id'], 'registry_version': contract['registry']['version'],
            'mechanism_id': pack['mechanism_id'], 'module': pack['module'],
            'scope': {key: pack[key] for key in ('allowed_paths', 'functions', 'globals', 'allowed_imports',
                'max_files', 'max_changed_lines')},
            'goal_checks': pack['goal_checks'], 'preserve_checks': pack['preserve_checks'],
            'coverage': pack['coverage'], 'parent_probe': parent['parent_probe']}}
    instructions = _INSTRUCTIONS
    fixture = state.get('applicability_fixture_source')
    if fixture is not None:
        from research1.data_pipeline.fixture_context import build_fixture_review_context
        request['original_fixture_source'] = build_fixture_review_context(fixture)
        instructions += (
            'The separately supplied original_fixture_source is the exact original preteacher test and scope code, '
            'verified against the executing fixtures. Use it to assess coverage of the original plan. '
            'Do not require the unimplemented plan to already contain child test results; this stage binds checks '
            'before implementation. Explain fixture coverage in reason and covered_checks. code_refs accepts only '
            'paths from parent_view.module_code or parent_view.framework_context.source_fragments; never put a '
            'fixture snapshot path or fixture-only quotation in code_refs.\n'
        )
    prompt = instructions + probe_source_strength(request['parent_view']) + canonical(request)
    input_record = {**request, 'request_id': digest(request), 'prompt': prompt,
        'prompt_sha256': _sha_text(prompt), 'plan_sha256': digest(plan),
        'visible_view_sha256': digest(request['parent_view']), 'checks_id': contract['checks_id']}
    return input_record, proof, catalog, catalog_hash


def review_applicability(state: dict, plan: dict, factual_review: dict, directory: Path) -> dict:
    """One persisted call. Started-without-output is unknown and never resent."""
    from research1.distill import _tongji_chat

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    contract = state.get('contract') or state.get('checks') or {}
    pending = {'schema': BINDING_MODE, 'policy': POLICY_ID, 'status': 'pending',
        'plan_sha256': digest(plan), 'checks_id': contract.get('checks_id')}
    try:
        plan = validate_plan(plan, state)
        request, proof, catalog, catalog_hash = _request(state, plan, factual_review, contract)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        if (directory / 'applicability_input.json').exists():
            raise ValueError('Frozen applicability preconditions changed: ' + str(exc)) from exc
        result = dict(pending, reason=type(exc).__name__ + ': ' + str(exc))
        _atomic_json(directory / 'applicability_result.json', result)
        return result
    input_path = directory / 'applicability_input.json'
    if input_path.exists():
        existing, _ = _read(input_path)
        if existing != request:
            raise ValueError('Immutable applicability input changed')
    else:
        _atomic_json(input_path, request)
    started_path, output_path = directory / 'applicability_started.json', directory / 'applicability_output.json'
    if output_path.exists():
        saved, _ = _read(output_path)
        if saved.get('request_id') != request['request_id']:
            raise ValueError('Saved applicability response belongs to another request')
    elif started_path.exists():
        started, _ = _read(started_path)
        if started.get('request_id') != request['request_id']:
            raise ValueError('Saved applicability start belongs to another request')
        result = dict(pending, reason='CALL_OUTCOME_UNKNOWN: a started applicability request has no saved reply')
        _atomic_json(directory / 'applicability_result.json', result)
        return result
    else:
        started = {'request_id': request['request_id'], 'prompt_sha256': request['prompt_sha256'],
                   'started_unix': time.time(), 'automatic_resend': False}
        try:
            with started_path.open('x', encoding='utf-8') as handle:
                handle.write(json.dumps(started, ensure_ascii=False, indent=2) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            return dict(pending, reason='CALL_OUTCOME_UNKNOWN: applicability request was claimed by another writer')
        try:
            raw, metadata = _tongji_chat(request['prompt'], max_tokens=6000,
                role='data_plan_applicability_fact_ids', json_output=True, reasoning_effort='none',
                allow_http_fallback=False)
            saved = {'policy': POLICY_ID, 'request_id': request['request_id'],
                'prompt_sha256': request['prompt_sha256'], 'raw': raw, 'metadata': metadata}
            _atomic_json(output_path, saved)
        except Exception as exc:
            _atomic_json(directory / 'applicability_error.json', {'request_id': request['request_id'],
                'error_type': type(exc).__name__, 'reason': str(exc)[:1000], 'automatic_resend': False})
            result = dict(pending, reason='Applicability request failed or outcome unknown; no automatic resend')
            _atomic_json(directory / 'applicability_result.json', result)
            return result
    try:
        parsed = _parse(saved, state, plan, contract, catalog, catalog_hash)
        _, input_hash = _read(input_path)
        _, output_hash = _read(output_path)
        _, started_hash = _read(started_path)
        result = dict(pending, **parsed, request_id=request['request_id'],
            mechanism_id=plan['verification_kind'], target_module=plan['target_module'],
            visible_view_sha256=request['visible_view_sha256'], factual_review=factual_review,
            **proof, applicability_input_path=str(input_path), applicability_input_record_sha256=input_hash,
            applicability_output_path=str(output_path), applicability_raw_record_sha256=output_hash,
            applicability_started_path=str(started_path), applicability_started_record_sha256=started_hash)
        result['binding_id'] = 'plan_binding_' + digest(result)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        result = dict(pending, reason=type(exc).__name__ + ': ' + str(exc))
    _atomic_json(directory / 'applicability_result.json', result)
    return result


def validate_binding(binding: Any, state: dict, plan: dict, contract: dict) -> bool:
    """Re-read real saved records; an asserted or stale PASS cannot bind tests."""
    try:
        if (not isinstance(binding, dict) or binding.get('status') != 'PASS'
                or binding.get('schema') != BINDING_MODE or binding.get('policy') != POLICY_ID
                or binding.get('binding_id') != 'plan_binding_' + digest({key: value for key, value in binding.items() if key != 'binding_id'})):
            return False
        plan = validate_plan(plan, state)
        request, proof, catalog, catalog_hash = _request(state, plan, binding['factual_review'], contract)
        for key, value in proof.items():
            if binding.get(key) != value:
                return False
        if (binding.get('checks_id') != contract['checks_id'] or binding.get('plan_sha256') != digest(plan)
                or binding.get('visible_view_sha256') != request['visible_view_sha256']
                or binding.get('request_id') != request['request_id']
                or binding.get('mechanism_id') != plan['verification_kind']
                or binding.get('target_module') != plan['target_module']
                or binding.get('factual_witness_catalog_sha256') != catalog_hash):
            return False
        paths = {}
        for kind, hash_key in [('input', 'applicability_input_record_sha256'),
                              ('output', 'applicability_raw_record_sha256'),
                              ('started', 'applicability_started_record_sha256')]:
            path = Path(binding['applicability_' + kind + '_path'])
            record, actual_hash = _read(path)
            if binding.get(hash_key) != actual_hash:
                return False
            paths[kind] = (path, record)
        if len({path.parent for path, _ in paths.values()}) != 1:
            return False
        if paths['input'][1] != request:
            return False
        for kind in ('output', 'started'):
            record = paths[kind][1]
            if (record.get('request_id') != request['request_id']
                    or record.get('prompt_sha256') != request['prompt_sha256']):
                return False
        if paths['output'][1].get('policy') != POLICY_ID or paths['started'][1].get('automatic_resend') is not False:
            return False
        parsed = _parse(paths['output'][1], state, plan, contract, catalog, catalog_hash)
        if parsed['status'] != 'PASS' or any(binding.get(key) != value for key, value in parsed.items()):
            return False
        result_path = paths['input'][0].parent / 'applicability_result.json'
        result, _ = _read(result_path)
        return result == binding
    except (ValueError, TypeError, KeyError, OSError):
        return False
