"""Independent binding of a plan to checks frozen before its generation.

This reviewer cannot define tests, change scope, implement code, observe child
outcomes, or establish future utility. Its PASS remains a model judgment. A
binding additionally requires the separately saved factual PASS, exact original
witnesses/code references, the unchanged parent and a pre-reproduced target.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from research1.data_pipeline.policy import canonical, digest, public_view, validate_plan, _atomic_json, _witness_sources, probe_source_strength

APPLICABILITY_POLICY = 'proposal-plan-applicability.v1'
BINDING_MODE = 'independent_plan_to_prefrozen_checks.v1'
PLAN_POLICY = 'proposal_single_plan_v6'
FIXTURE_POLICY = 'proposal-plan-applicability.v2.fixtures'
FACT_IDS_POLICY = 'proposal-plan-applicability.v3.fact_ids'

def _policy(state):
    return FIXTURE_POLICY if state.get('applicability_policy')==FIXTURE_POLICY else APPLICABILITY_POLICY

_INSTRUCTIONS = (
    'Independently assess this single original plan against ONLY the frozen decision-time proposal view, '
    'current visible code/runtime interfaces, independently saved factual review and existing preteacher checks. '
    'Check whether the proposed implementation is feasible within its own explicit scope, including how every '
    'needed runtime value reaches the named function. Do not silently add another hook, hidden state, arguments '
    'or source access. Check that these exact existing goal and preservation fixtures cover the whole proposed '
    'behavior, not merely a related mechanism name. Do not create, alter or substitute tests, broaden scope, '
    'rewrite the plan, infer a defect from task feedback alone, or use future implementation/results. '
    'Observed process limitations may support a falsifiable improvement even when baseline works as designed; '
    'proven causal responsibility, solver benefit or task-score improvement are not required in advance. '
    'Separate observed facts/current-code facts from unproven expected effects. An existing synthetic failure '
    'and a factual PASS do not alone establish applicability. Unknown, ungrounded, infeasible, partially covered '
    'or uncertain plans remain pending. Return final JSON only with exactly these fields: '
    '{"status":"PASS|pending|FAIL","reason":"...","mechanism_id":"...",'
    '"evidence_refs":["original evidence id"],"witnesses":[{"evidence_ref":"id","text":"exact visible substring"}],'
    '"code_refs":[{"path":"visible module or read-only framework source path","text":"exact source substring"}],'
    '"covered_checks":["every existing goal and preservation check id"]}. '
    'For PASS cite exact proposal witnesses and current source proving the claimed interfaces/dataflow and '
    'explain why the unchanged listed checks cover the plan. Quote original decoded visible characters. '
    'A witness is not independently proven execution success, command freshness or causal attribution.\n'
)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_text(text: str) -> str:
    return _sha_bytes(text.encode('utf-8'))


def _read(path: Path) -> tuple[dict, str]:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Saved review record must be an existing absolute regular file')
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Saved review record is not an object')
    return value, _sha_bytes(raw)


def _metadata(meta: dict, state: dict, role: str) -> None:
    if (not isinstance(meta, dict) or meta.get('raw_field') != 'content'
            or meta.get('finish_reason') != 'stop' or meta.get('temperature') != 0):
        raise ValueError('Review requires complete final JSON at frozen temperature zero')
    for key in ('model', 'provider'):
        expected = state.get(role + '_' + key) or state.get('review_' + key) or state.get('teacher_' + key)
        if expected and meta.get(key) != expected:
            raise ValueError('Review provider/model differs from frozen policy')


def _factual_proof(review: dict, state: dict, plan: dict) -> dict:
    if state.get('replay_factual_from_source') is True:
        if review != state.get('original_factual_review'):
            raise ValueError('Saved-plan factual replay must use the exact original review object')
        from research1.data_pipeline.teacher_replay import validate_replayed_factual_state
        from research1.data_pipeline.engine import ArtifactChangedError
        try:
            proof=validate_replayed_factual_state(state,plan)
        except ArtifactChangedError as exc:
            raise ValueError('Original factual source was revoked or its bound artifacts changed: '+str(exc)) from exc
        if not isinstance(proof,dict) or proof.get('factual_review_sha256')!=digest(review):
            raise ValueError('Replayed factual source did not establish the original physical review proof')
        return proof
    if isinstance(review,dict) and review.get('review_policy') in {'proposal-factual-review.v7.selectors','proposal-factual-review.v8.probe_io'}:
        from research1.data_pipeline.selector_review import verify_saved_selected_review
        return verify_saved_selected_review(review,state,plan)
    if isinstance(review,dict) and review.get('review_policy')=='proposal-factual-review.v6.typed':
        from research1.data_pipeline.typed_review import verify_saved_typed_review
        return verify_saved_typed_review(review,state,plan)
    from research1.data_pipeline.policy import FACTUAL_REVIEW_POLICY, FACTUAL_WITNESS_VALIDATOR, _review_witnesses
    if (not isinstance(review, dict) or review.get('status') != 'PASS' or review.get('independent') is not True
            or review.get('review_type') != 'model_fact_review'
            or review.get('review_policy') != FACTUAL_REVIEW_POLICY
            or review.get('witness_validator') != FACTUAL_WITNESS_VALIDATOR):
        raise ValueError('A separately saved factual PASS is required')
    view = public_view(state)
    if review.get('plan_sha256') != digest(plan) or review.get('visible_view_sha256') != digest(view):
        raise ValueError('Factual review belongs to another plan/view')
    raw_path = Path(review.get('review_path', ''))
    saved, raw_hash = _read(raw_path)
    result_path = raw_path.parent / 'factual_review_result.json'
    result, result_hash = _read(result_path)
    if result != review:
        raise ValueError('Factual result differs from the real saved result')
    prompt = saved.get('prompt')
    if not isinstance(prompt, str) or saved.get('prompt_sha256') != _sha_text(prompt):
        raise ValueError('Factual prompt hash changed')
    suffix = 'Frozen visible view:\n' + canonical(view) + '\nCandidate:\n' + canonical(plan)
    if (not prompt.startswith('Factual review policy: ' + FACTUAL_REVIEW_POLICY + '. ')
            or not prompt.endswith(suffix) or saved.get('review_policy') != FACTUAL_REVIEW_POLICY
            or saved.get('visible_view_sha256') != digest(view) or saved.get('plan_sha256') != digest(plan)):
        raise ValueError('Factual request is not bound to this original input')
    meta = saved.get('metadata')
    _metadata(meta, state, 'review')
    value = json.loads(saved['raw'])
    if (not isinstance(value, dict) or value.get('status') != 'PASS' or not isinstance(value.get('reason'), str)
            or not value['reason'].strip() or value.get('reason') != review.get('reason')
            or value.get('evidence_refs') != review.get('evidence_refs')):
        raise ValueError('Original factual reply did not support the saved PASS')
    refs = value.get('evidence_refs')
    visible_refs = {row['id'] for row in view['evidence']}
    if not isinstance(refs, list) or not refs or any(ref not in visible_refs for ref in refs):
        raise ValueError('Factual references are not original visible evidence')
    witnesses = _review_witnesses(value.get('witnesses'), state, plan['action'])
    if witnesses != review.get('witnesses') or any(row['evidence_ref'] not in refs for row in witnesses):
        raise ValueError('Factual witnesses changed or lack exact support')
    expected_reviewer = 'factual_review_' + digest({'prompt':saved['prompt_sha256'],
        'model':meta.get('model'), 'provider':meta.get('provider')})[:24]
    if review.get('reviewer_id') != expected_reviewer:
        raise ValueError('Factual reviewer identity changed')
    return {'factual_review_sha256':digest(review), 'factual_raw_path':str(raw_path),
            'factual_raw_record_sha256':raw_hash, 'factual_result_path':str(result_path),
            'factual_result_record_sha256':result_hash}


def validate_factual_review(review: Any, state: dict, plan: dict) -> bool:
    try:
        _factual_proof(review, state, validate_plan(plan, state))
        return True
    except (ValueError, TypeError, KeyError, OSError):
        return False


def _candidate(state: dict, plan: dict, contract: dict) -> tuple[dict, dict]:
    from research1.data_pipeline.module_adapters import get_adapter
    adapter=get_adapter(state)
    frozen_registry,tree_hashes,_digest=adapter.frozen_registry,adapter.tree_hashes,adapter._digest
    if state.get('plan_policy_version') != PLAN_POLICY or contract.get('binding_mode') != BINDING_MODE:
        raise ValueError('Post-plan binding requires the new explicitly frozen protocol')
    if contract.get('checks_id') != 'checks_' + _digest({key:value for key,value in contract.items() if key != 'checks_id'})[:24]:
        raise ValueError('Preteacher contract hash changed')
    if contract.get('registry') != frozen_registry():
        raise ValueError('Existing check registry changed')
    if tree_hashes(Path(state['modules_root'])) != contract.get('parent_hashes'):
        raise ValueError('Frozen parent source changed')
    if any((Path(state['modules_root'])/path).read_text(encoding='utf-8') != text
           for path,text in state['module_code'].items()):
        raise ValueError('Visible source differs from the real frozen parent')
    if contract.get('evidence_hashes') != {row['id']:_digest(row) for row in state.get('evidence', [])}:
        raise ValueError('Frozen parent evidence changed')
    kind = plan.get('verification_kind')
    pack = contract['registry']['packs'].get(kind)
    if (plan.get('action') != 'edit' or not pack or pack['module'] != plan.get('target_module')
            or pack['module'] not in state.get('permitted_modules', ())
            or not plan.get('locations') or any(path not in pack['allowed_paths'] for path in plan['locations'])):
        raise ValueError('Unknown mechanism or plan outside frozen scope')
    rows = [row for row in contract.get('preprobed', []) if row.get('mechanism_id') == kind and row.get('module') == pack['module']]
    if len(rows) != 1:
        raise ValueError('No unique preteacher reproduced target')
    row = rows[0]
    checks = row.get('parent_probe', {}).get('checks', {})
    recorded = contract.get('parent_probes', {}).get(kind, {})
    if (recorded.get('status') != 'available' or recorded.get('parent_probe') != row.get('parent_probe')
            or row.get('target_reproduced') is not True or row.get('preservation_baseline_passed') is not True
            or not all(checks.get(name) is False for name in pack['goal_checks'])
            or not all(checks.get(name) is True for name in pack['preserve_checks'])):
        raise ValueError('Parent target/preservation evidence is incomplete')
    return pack, row


def _request(state: dict, plan: dict, review: dict, contract: dict) -> tuple[dict, dict]:
    pack, parent = _candidate(state, plan, contract)
    proof = _factual_proof(review, state, plan)
    policy=_policy(state)
    request = {'policy':policy, 'parent_view':public_view(state), 'plan':plan,
        'factual_review_sha256':proof['factual_review_sha256'],
        'frozen_checks':{'checks_id':contract['checks_id'], 'registry_version':contract['registry']['version'],
            'mechanism_id':pack['mechanism_id'], 'module':pack['module'],
            'scope':{key:pack[key] for key in ('allowed_paths','functions','globals','allowed_imports','max_files','max_changed_lines')},
            'goal_checks':pack['goal_checks'], 'preserve_checks':pack['preserve_checks'],
            'coverage':pack['coverage'], 'parent_probe':parent['parent_probe']}}
    instructions=_INSTRUCTIONS
    if policy==FIXTURE_POLICY:
        from research1.data_pipeline.fixture_context import build_fixture_review_context
        request['original_fixture_source']=build_fixture_review_context(state['applicability_fixture_source'])
        instructions+=('The separately supplied original_fixture_source is the exact original preteacher test and scope code, '
            'verified against the executing fixtures. Use it to assess coverage of the original plan. '
            'Do not require the unimplemented plan to already contain child test results; this stage binds checks before implementation. '
            'The plan need not predict the fixture implementation, but its requested behavior and verification must fit these unchanged checks. '
            'Explain fixture coverage in reason and covered_checks. code_refs accepts only paths from parent_view.module_code '
            'or parent_view.framework_context.source_fragments; never put a fixture snapshot path or fixture-only quotation in code_refs. '
            'This adds visibility only and does not relax coverage or require expected score benefit to be proven.\n')
    prompt = instructions + probe_source_strength(request['parent_view']) + canonical(request)
    record = {'policy':policy, 'request_id':digest(request), 'prompt':prompt,
        'prompt_sha256':_sha_text(prompt), 'plan_sha256':digest(plan),
        'visible_view_sha256':digest(request['parent_view']), 'checks_id':contract['checks_id'],
        'factual_review_sha256':proof['factual_review_sha256']}
    return record, proof


def _parse(saved: dict, state: dict, plan: dict, contract: dict) -> dict:
    _metadata(saved.get('metadata'), state, 'applicability')
    value = json.loads(saved['raw'])
    keys = {'status','reason','mechanism_id','evidence_refs','witnesses','code_refs','covered_checks'}
    if (not isinstance(value, dict) or set(value) != keys or value.get('status') not in {'PASS','pending','FAIL'}
            or not isinstance(value.get('reason'), str) or not value['reason'].strip()):
        raise ValueError('Applicability reply differs from the frozen review shape')
    if value['status'] != 'PASS':
        return {'status':'pending', 'reason':value['reason'], 'raw_status':value['status']}
    pack, _ = _candidate(state, plan, contract)
    if value['mechanism_id'] != plan['verification_kind']:
        raise ValueError('Applicability reply names a different mechanism')
    refs = value['evidence_refs']
    sources = {row['id']:row['text'] for row in public_view(state)['evidence']}
    if (not isinstance(refs, list) or not refs or len(set(refs)) != len(refs)
            or any(not isinstance(ref, str) or ref not in sources for ref in refs)):
        raise ValueError('Applicability needs original proposal references')
    witnesses = value['witnesses']
    if not isinstance(witnesses, list) or not witnesses:
        raise ValueError('Applicability PASS requires exact proposal witnesses')
    matched = []
    for witness in witnesses:
        if (not isinstance(witness, dict) or set(witness) != {'evidence_ref','text'}
                or witness['evidence_ref'] not in refs or not isinstance(witness['text'], str)
                or not witness['text'].strip()):
            raise ValueError('Invalid applicability witness')
        locations = [dict(source, start=source['text'].find(witness['text']),
                          end=source['text'].find(witness['text'])+len(witness['text']))
                     for source in _witness_sources(sources[witness['evidence_ref']]) if witness['text'] in source['text']]
        if not locations:
            raise ValueError('Applicability witness is absent from original decoded evidence')
        matched.append(dict(witness, serialized_evidence_sha256=_sha_text(sources[witness['evidence_ref']]),
                            locations=[{key:value for key,value in row.items() if key != 'text'} for row in locations]))
    code_refs = value['code_refs']
    if not isinstance(code_refs, list) or not code_refs:
        raise ValueError('Applicability PASS requires source/dataflow references')
    code = {path:[text] for path,text in state['module_code'].items()}
    for fragment in state.get('framework_context', {}).get('source_fragments', []):
        code.setdefault(fragment['path'], []).append(fragment['text'])
    validated_code = []
    for reference in code_refs:
        if (not isinstance(reference, dict) or set(reference) != {'path','text'}
                or reference.get('path') not in code or not isinstance(reference.get('text'), str)
                or not reference['text'].strip()):
            raise ValueError('Invalid source reference')
        matching = [text for text in code[reference['path']] if reference['text'] in text]
        if not matching:
            raise ValueError('Applicability source quote is absent from visible code')
        validated_code.append(dict(reference, matching_source_sha256=[_sha_text(text) for text in matching]))
    if not any(row['path'] in plan['locations'] for row in code_refs):
        raise ValueError('Applicability lacks a reference to the planned editing location')
    names = pack['goal_checks'] + pack['preserve_checks']
    if (not isinstance(value['covered_checks'], list) or len(value['covered_checks']) != len(names)
            or any(not isinstance(name, str) for name in value['covered_checks'])
            or set(value['covered_checks']) != set(names)):
        raise ValueError('Applicability must cover every unchanged goal and preservation check')
    return {'status':'PASS', 'reason':value['reason'], 'raw_status':'PASS', 'witnesses':matched,
            'code_refs':validated_code, 'covered_checks':list(value['covered_checks'])}


def review_applicability(state: dict, plan: dict, factual_review: dict, directory: Path) -> dict:
    """One persisted call. Started-without-output is unknown and never resent."""
    if state.get('applicability_policy')==FACT_IDS_POLICY:
        from research1.data_pipeline.applicability_v3 import review_applicability as review_fact_ids
        return review_fact_ids(state,plan,factual_review,directory)
    from research1.distill import _tongji_chat
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    contract = state.get('contract') or state.get('checks') or {}
    pending = {'schema':BINDING_MODE, 'policy':_policy(state), 'status':'pending',
        'plan_sha256':digest(plan), 'checks_id':contract.get('checks_id')}
    try:
        plan = validate_plan(plan, state)
        request, proof = _request(state, plan, factual_review, contract)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        if (directory/'applicability_input.json').exists():
            raise ValueError('Frozen applicability preconditions changed: ' + str(exc)) from exc
        result = dict(pending, reason=type(exc).__name__ + ': ' + str(exc))
        _atomic_json(directory/'applicability_result.json', result)
        return result
    input_path = directory/'applicability_input.json'
    if input_path.exists():
        existing, _ = _read(input_path)
        if existing != request:
            raise ValueError('Immutable applicability input changed')
    else:
        _atomic_json(input_path, request)
    started_path, output_path = directory/'applicability_started.json', directory/'applicability_output.json'
    if output_path.exists():
        saved, _ = _read(output_path)
        if saved.get('request_id') != request['request_id']:
            raise ValueError('Saved applicability response belongs to another request')
    elif started_path.exists():
        started, _ = _read(started_path)
        if started.get('request_id') != request['request_id']:
            raise ValueError('Saved applicability start belongs to another request')
        result = dict(pending, reason='CALL_OUTCOME_UNKNOWN: a started applicability request has no saved reply')
        _atomic_json(directory/'applicability_result.json', result)
        return result
    else:
        started = {'request_id':request['request_id'], 'prompt_sha256':request['prompt_sha256'],
                   'started_unix':time.time(), 'automatic_resend':False}
        try:
            with started_path.open('x', encoding='utf-8') as handle:
                handle.write(json.dumps(started, ensure_ascii=False, indent=2)+'\n')
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            return dict(pending, reason='CALL_OUTCOME_UNKNOWN: applicability request was claimed by another writer')
        try:
            raw, metadata = _tongji_chat(request['prompt'], max_tokens=6000, role='data_plan_applicability',
                json_output=True, reasoning_effort='none', allow_http_fallback=False)
            saved = {'policy':_policy(state), 'request_id':request['request_id'],
                'prompt_sha256':request['prompt_sha256'], 'raw':raw, 'metadata':metadata}
            _atomic_json(output_path, saved)
        except Exception as exc:
            _atomic_json(directory/'applicability_error.json', {'request_id':request['request_id'],
                'error_type':type(exc).__name__, 'reason':str(exc)[:1000], 'automatic_resend':False})
            result = dict(pending, reason='Applicability request failed or outcome unknown; no automatic resend')
            _atomic_json(directory/'applicability_result.json', result)
            return result
    try:
        parsed = _parse(saved, state, plan, contract)
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
    _atomic_json(directory/'applicability_result.json', result)
    return result


def validate_binding(binding: Any, state: dict, plan: dict, contract: dict) -> bool:
    """Re-read real saved records; an asserted or stale PASS cannot bind tests."""
    if (state.get('applicability_policy')==FACT_IDS_POLICY
            or (isinstance(binding,dict) and (binding.get('policy')==FACT_IDS_POLICY or binding.get('policy_id')==FACT_IDS_POLICY))):
        from research1.data_pipeline.applicability_v3 import validate_binding as validate_fact_ids
        return validate_fact_ids(binding,state,plan,contract)
    try:
        if (not isinstance(binding, dict) or binding.get('status') != 'PASS'
                or binding.get('schema') != BINDING_MODE or binding.get('policy') != _policy(state)
                or binding.get('binding_id') != 'plan_binding_' + digest({key:value for key,value in binding.items() if key != 'binding_id'})):
            return False
        plan = validate_plan(plan, state)
        request, proof = _request(state, plan, binding['factual_review'], contract)
        for key,value in proof.items():
            if binding.get(key) != value:
                return False
        if (binding.get('checks_id') != contract['checks_id'] or binding.get('plan_sha256') != digest(plan)
                or binding.get('visible_view_sha256') != request['visible_view_sha256']
                or binding.get('request_id') != request['request_id'] or binding.get('mechanism_id') != plan['verification_kind']
                or binding.get('target_module') != plan['target_module']):
            return False
        paths = {}
        for kind,hash_key in [('input','applicability_input_record_sha256'),('output','applicability_raw_record_sha256'),
                              ('started','applicability_started_record_sha256')]:
            path = Path(binding['applicability_'+kind+'_path'])
            record, actual_hash = _read(path)
            if binding.get(hash_key) != actual_hash:
                return False
            paths[kind] = (path,record)
        if len({path.parent for path,_ in paths.values()}) != 1:
            return False
        if paths['input'][1] != request:
            return False
        for kind in ('output','started'):
            record = paths[kind][1]
            if record.get('request_id') != request['request_id'] or record.get('prompt_sha256') != request['prompt_sha256']:
                return False
        if paths['output'][1].get('policy') != _policy(state) or paths['started'][1].get('automatic_resend') is not False:
            return False
        parsed = _parse(paths['output'][1], state, plan, contract)
        if parsed['status'] != 'PASS' or any(binding.get(key) != value for key,value in parsed.items()):
            return False
        result_path = paths['input'][0].parent/'applicability_result.json'
        result, _ = _read(result_path)
        return result == binding
    except (ValueError, TypeError, KeyError, OSError):
        return False
