"""Independent, bounded proposal-to-SFT collection without noise or MC gates.

An unscored task remains in the fixed queue's results. It does not become zero,
and does not prevent the next preregistered parent from producing a plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from research1.data_pipeline.engine import ArtifactChangedError, _cached, _public_state, _usable
from research1.data_pipeline.execution import evaluate, tree_hash
from research1.data_pipeline.ledger import BudgetExceeded, Ledger
from research1.data_pipeline.policy import validate_plan
from research1.data_pipeline.util import atomic_json, digest


class _StopInitialization(RuntimeError):
    def __init__(self, status: str, reason: str, result: dict | None = None):
        super().__init__(reason)
        self.status, self.result = status, result


def _add_addressable_feedback(state: dict) -> None:
    """Give new-interface prior metadata its own context-only evidence source."""
    if state.get('plan_policy_version') in {None, 'proposal_single_plan_v2', 'proposal_single_plan_v3'}:
        return
    from research1.data_pipeline.policy import public_view, canonical
    outcomes = public_view(state).get('prior_proposal_outcomes', [])
    for outcome in outcomes:
        reference = outcome['proposal_task_id'] + ':proposal_feedback_' + outcome['execution_id']
        text = canonical(dict(outcome, kind='recorded_proposal_feedback',
            interpretation='Recorded classification only; not a causal diagnosis or execution observation.'))
        item = {'id': reference, 'text': text, 'partition': 'proposal',
            'proposal_task_id': outcome['proposal_task_id'], 'source_type': 'recorded_proposal_feedback'}
        existing = next((e for e in state['evidence'] if e['id'] == reference), None)
        if existing is not None and existing != item:
            raise ArtifactChangedError('Addressable proposal feedback changed')
        if existing is None:
            state['evidence'].append(item)


def _review_valid(review: dict, state: dict) -> bool:
    refs = review.get('evidence_refs')
    return bool(review.get('status') == 'PASS' and review.get('independent') is True
        and isinstance(review.get('reviewer_id'), str) and review['reviewer_id'].strip()
        and isinstance(review.get('reason'), str) and review['reason'].strip()
        and isinstance(refs, list) and refs
        and all(isinstance(ref, str) and ref in {e['id'] for e in state['evidence']} for ref in refs))


def _check_result_artifacts(result: dict, *, expected_task=None, expected_code_hash=None) -> None:
    """Stored successful results remain tied to their original bytes and tree."""
    for key, hash_key in [('trial_result_path', 'result_sha256'), ('trajectory', 'trajectory_sha256')]:
        if result.get(hash_key):
            path = Path(result.get(key, ''))
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != result[hash_key]:
                raise ArtifactChangedError('Committed initialization execution artifact changed: ' + key)
    spec = result.get('evaluation_spec')
    if isinstance(spec, dict):
        if spec.get('role') != 'proposal' or (expected_task and spec.get('task') != expected_task):
            raise ArtifactChangedError('Initialization result is not the expected proposal task')
        if expected_code_hash and spec.get('code_hash') != expected_code_hash:
            raise ArtifactChangedError('Initialization result belongs to another implementation')
        for path_key, hash_key in [('modules_root', 'code_hash'), ('task_path', 'task_hash')]:
            if spec.get(path_key) and spec.get(hash_key) and tree_hash(Path(spec[path_key])) != spec[hash_key]:
                raise ArtifactChangedError('Committed initialization execution tree changed: ' + path_key)


def _raise_policy_failure(exc: Exception) -> None:
    if re.search(r'HTTP\s*(401|403)|authentication|unauthorized|invalid.api.key|differs from.*(?:frozen|policy)|changed.*policy|provider.*mismatch', str(exc), re.I):
        raise _StopInitialization('BLOCKED_CONFIGURATION', str(exc)) from exc


def _check_attempt(state: dict, attempt: dict) -> None:
    if tree_hash(Path(state['modules_root'])) != state['code_hash']:
        raise ArtifactChangedError('Frozen initialization parent changed')
    if attempt.get('status') != 'applied':
        return
    patch = Path(attempt.get('patch_path', ''))
    if (attempt.get('parent_code_hash') != state['code_hash']
        or tree_hash(Path(attempt['modules_root'])) != attempt.get('code_hash')
        or not patch.is_file()
        or hashlib.sha256(patch.read_bytes()).hexdigest() != attempt.get('patch_sha256')):
        raise ArtifactChangedError('Stored initialization implementation or patch changed')


def run_initialization(campaign: Path, *, evaluator=None, proposer=None, implementer=None,
                       verifier=None, reviewer=None, diagnostician=None) -> dict:
    """Collect independent parents, persist plans first, then qualify each plan.

    Evaluator/proposer/implementer/verifier callbacks match the MC engine.
    Injected reviewers receive (plan, state); the production review adapter
    receives only the original state and plan, never child outcomes.
    """
    from research1.data_pipeline.protocol import load_protocol
    from research1.data_pipeline.module_adapters import get_adapter
    from research1.data_pipeline.policy import collect_candidates, review_plan
    from research1.data_pipeline.diagnostics import review_diagnostics
    from research1.data_pipeline.sft import export_dataset
    from research1.data_pipeline.source_parents import validate_parent_sources, source_provenance

    campaign = Path(campaign).resolve()
    protocol = load_protocol(campaign, verify_source=evaluator is None)
    module_adapter = get_adapter(protocol)
    if protocol.get('execution_mode') != 'initialization':
        raise ValueError('Independent initialization requires its own frozen execution_mode')
    config = protocol.get('initialization') or {}
    limit, count = config.get('max_parents'), config.get('max_candidates', 2)
    tasks = config.get('task_order')
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('Initialization max_parents must be positive')
    if count not in (1, 2) or isinstance(count, bool):
        raise ValueError('Initialization max_candidates must be one or two')
    if not isinstance(tasks, list) or not tasks or any(not isinstance(t, str) or not t for t in tasks) or len(tasks) != len(set(tasks)):
        raise ValueError('Initialization requires a unique frozen task_order')
    tasks = tasks[:limit]
    sets = protocol['task_sets']
    if not set(tasks).issubset(sets['proposal']):
        raise ValueError('Initialization can execute only the proposal partition')
    if set(sets['proposal']) & (set(sets['reward']) | set(sets['dev'])):
        raise ValueError('Initialization proposal tasks overlap reward/dev')
    train, validation = set(sets['proposal_train']), set(sets['proposal_validation'])
    if train & validation or train | validation != set(sets['proposal']):
        raise ValueError('Initialization proposal training and validation folds are invalid')

    ledger = Ledger(campaign)
    evaluate_task = evaluator or evaluate
    propose = proposer or collect_candidates
    implement = implementer or module_adapter.implement
    verify = verifier or module_adapter.verify
    baseline = Path(protocol['modules_root'])
    source_rows = []
    status = {'schema': 'research1.initialization.v1', 'protocol_id': protocol['protocol_id'],
        'campaign': str(campaign), 'status': 'RUNNING', 'phase': 'initialization',
        'calibration_complete': False, 'data_collection_complete': False,
        'training_started': False, 'task_order': tasks, 'started_unix': time.time()}

    def save(phase=None):
        if phase:
            status['phase'] = phase
        all_samples = ledger.records('samples')
        samples = [r for r in all_samples if not r.get('revoked')]
        parents = ledger.records('init_parents')
        status['counts'] = {'parents_processed': sum(r.get('status') in {'DONE', 'UNSCORED', 'TEACHER_ERROR', 'EVIDENCE_PENDING'} for r in parents),
            'raw_plans': len(all_samples), 'revoked': len(all_samples) - len(samples),
            'qualified_edit': sum(r['plan']['action'] == 'edit' and r.get('qualification', {}).get('status') == 'verified' for r in samples),
            'qualified_keep': sum(r['plan']['action'] == 'keep' and r.get('qualification', {}).get('status') == 'verified' for r in samples),
            'pending': sum(r.get('qualification', {}).get('status') == 'pending' for r in samples),
            'rejected': sum(r.get('qualification', {}).get('status') == 'rejected' for r in samples),
            'failed_parents': sum(r.get('status') in {'UNSCORED', 'TEACHER_ERROR', 'EVIDENCE_PENDING'} for r in parents)}
        status['counts'].update(unique_parent_roots=len({r['root_lineage_id'] for r in parents if r.get('root_lineage_id')}),
            reused_parent_executions=len({r['parent_result']['evaluation_id'] for r in parents if r.get('parent_source')}),
            new_independent_parents=0 if protocol.get('source_parents') else len({r['root_lineage_id'] for r in parents if r.get('root_lineage_id')}))
        status['ledger'] = ledger.status()
        atomic_json(campaign / 'status.json', status)

    def export(*, stop_on_invalid=False):
        path = campaign / 'student_profile.json'
        profile = json.loads(path.read_text()) if path.is_file() else protocol.get('sft', {}).get('target_profile')
        manifest = export_dataset(campaign, ledger.records('samples'), target_profile=profile)
        status['sft'] = {key: manifest.get(key) for key in ('dataset_revision', 'counts', 'training_ready', 'readiness_reasons')}
        source_validation = manifest.get('source_parent_validation')
        if source_validation and source_validation['status'] != 'PASS':
            status.update(status='BLOCKED_ARTIFACT_CHANGED', reason=source_validation['reason'])
            if stop_on_invalid:
                raise ArtifactChangedError(source_validation['reason'])

    def execute(code, task, scope):
        if protocol.get('modules_hash') and tree_hash(baseline) != protocol['modules_hash']:
            raise ArtifactChangedError('Frozen initialization H0 changed')
        result = evaluate_task(campaign, protocol, ledger, code, task, 'proposal', scope, 1)
        if not isinstance(result, dict):
            raise ValueError('Evaluator returned no structured TaskRun')
        ledger.add_event('initialization_execution', {'task': task, 'scope': scope, 'role': 'proposal',
            'evaluation_id': result.get('evaluation_id'), 'status': result.get('status'),
            'wall_limit_reached': result.get('wall_limit_reached', False)})
        if result.get('in_progress'):
            raise _StopInitialization('WAITING_EXISTING_JOB', 'Initialization job is already running; resume the same reserved job', result)
        if result.get('fatal'):
            raise _StopInitialization('BLOCKED_CONFIGURATION', result.get('error') or 'Fatal initialization execution failure', result)
        if _usable(result):
            _check_result_artifacts(result, expected_task=task, expected_code_hash=tree_hash(code))
        return result

    def make_state(task, index, parent, source_row=None):
        if evaluator is not None and isinstance(parent.get('evidence'), list):
            evidence = copy.deepcopy(parent['evidence'])
        else:
            from research1.evidence import select_trajectory_evidence
            if not parent.get('trajectory'):
                raise ValueError('Scored initialization parent has no recoverable proposal trajectory')
            evidence = select_trajectory_evidence(Path(parent['trajectory']), **protocol['evidence_budget'])
        for item in evidence:
            # Evaluation values stay outside the model-facing evidence.
            item.pop('task_result', None)
            item['id'] = task + ':' + item['id']
            item['proposal_task_id'] = task
        lineage = (source_row['root_lineage_id'] if source_row else 'lineage_' + digest({
            'campaign': str(campaign), 'task': task, 'parent_index': index,
            'protocol_hash': protocol.get('protocol_hash'), 'evaluation_id': parent.get('evaluation_id')})[:24])
        split = source_row['split'] if source_row else 'train' if task in train else 'validation'
        state = _public_state(baseline, evidence, [], lineage, split, [task], protocol)
        if source_row:
            # This metadata stays outside public_view. Only the verified,
            # finished original proposal observation is visible to the model.
            state['parent_execution_provenance'] = source_provenance(source_row, protocol)
            state['decision_protocol_hash'] = protocol['protocol_hash']
            state['origin_id'] = source_row['source_state_id']
        parent_spec = parent.get('evaluation_spec')
        if (isinstance(parent_spec, dict) and parent_spec.get('role') == 'proposal'
            and parent_spec.get('task') == task and parent_spec.get('code_hash') == state['code_hash']
            and (Path(parent_spec.get('modules_root', '')).resolve() == baseline.resolve()
                 or (source_row is not None and source_row['parent_result'] == parent))
            and parent.get('result_sha256') and parent.get('evaluation_id')):
            state['prior_proposal_outcomes'] = [{'proposal_task_id': task,
                'execution_id': parent['evaluation_id'],
                'outcome': 'PASS' if parent['reward'] == 1 else 'FAIL' if parent['reward'] == 0 else 'UNSCORED',
                'failure_layer': parent['failure_layer'], 'parent_code_hash': state['code_hash'],
                'source_sha256': parent['result_sha256'], 'finished_before_decision': True,
                'source_partition': 'proposal'}]
            if protocol.get('framework_context') is not None:
                from research1.data_pipeline.framework_context import build_framework_context
                state['framework_context'] = build_framework_context(Path(protocol['upstream_root']),
                    protocol, evaluation_spec=parent_spec)
        _add_addressable_feedback(state)
        for field, role in [('review', 'factual_reviewer'), ('diagnostic', 'diagnostician')]:
            policy = protocol['policies'].get(role, protocol['policies'].get('teacher', {}))
            for key in ('model', 'provider'):
                if policy.get(key):
                    state[field + '_' + key] = policy[key]
        state['max_candidates'] = count
        base_id = 'state_' + digest({k: v for k, v in state.items() if k not in {'state_id', 'modules_root'}})[:24]
        state['state_id'] = base_id
        def diagnose():
            callback = diagnostician or review_diagnostics
            return callback(copy.deepcopy(state), campaign / 'diagnostics' / base_id)
        try:
            state['diagnostics'] = _cached(ledger, 'init_diagnostics', base_id, diagnose)
        except (ValueError, RuntimeError, OSError) as exc:
            _raise_policy_failure(exc)
            # Attribution is a qualification gate, not a prerequisite for
            # retaining an independently generated teacher plan.
            state['diagnostics'] = []
            state['diagnostic_pending_reason'] = type(exc).__name__ + ': ' + str(exc)
            ledger.put_record('init_diagnostics', base_id, {'_operation_key': base_id, 'value': [],
                'pending_reason': state['diagnostic_pending_reason']})
            ledger.add_event('initialization_diagnostic_pending', {'state_id': base_id,
                'reason': state['diagnostic_pending_reason']})
        contract = _cached(ledger, 'init_contracts', base_id + ':' + digest(state['diagnostics']),
            lambda: module_adapter.build_contract(state))
        state['contract'] = contract
        state['verification_catalog'] = module_adapter.proposal_catalog(state, contract)
        state['sampling_stratum'] = ('harness_signal' if contract.get('selected') else
            'valid_failure' if parent['reward'] == 0 else 'success_control')
        state['state_id'] = 'state_' + digest({'base_id': base_id, 'checks_id': contract.get('checks_id', digest(contract)),
            'verification_catalog': state['verification_catalog']})[:24]
        atomic_json(campaign / 'states' / state['state_id'] / 'state.json', state)
        ledger.put_record('states', state['state_id'], state)
        return state

    def sample_record(state, plan, scope, parent, *, attempt=None, behavior=None, review=None, child=None):
        attempt, behavior = attempt or {}, behavior or {'status': 'pending', 'covers_plan': False, 'executable': False}
        review = review or {'status': 'pending_review', 'reason': 'Factual review not completed', 'evidence_refs': []}
        implementation_usable = plan['action'] == 'keep' or (child is not None and _usable(child))
        checked = behavior.get('status') == 'PASS' and behavior.get('covers_plan') is True and behavior.get('executable') is True
        if plan['action'] == 'edit':
            checked = checked and bool(plan.get('verification_kind')) and behavior.get('mechanism_id') == plan['verification_kind']
        qualified = _review_valid(review, state) and checked and implementation_usable and _usable(parent)
        sample_id = 'sample_' + digest({'state': state['state_id'], 'plan': plan, 'scope': scope})[:24]
        record = {'sample_id': sample_id, 'candidate_id': scope, 'state': copy.deepcopy(state), 'plan': copy.deepcopy(plan),
            'qualification': {'status': 'verified' if qualified else 'pending', 'factual_review': review,
                'behavior': behavior, 'proposal_task_result': parent,
                'implementation_task_result': child if plan['action'] == 'edit' else None,
                'task_result': (child or {'status': 'INCOMPLETE', 'reward': None, 'failure_layer': 'unknown'}) if plan['action'] == 'edit' else parent},
            'implementation': attempt, 'source_kind': 'fresh_teacher_plan', 'protocol_id': protocol['protocol_id'],
            'root_lineage_id': state['root_lineage_id'], 'state_id': state['state_id'],
            'mechanism_id': behavior.get('mechanism_id', plan.get('verification_kind')),
            'teacher_path': state.get('teacher_record_path'), 'teacher_record_path': state.get('teacher_record_path'),
            'source_refs': [state['state_id'], scope] + [r['evaluation_id'] for r in (parent, child or {}) if r.get('evaluation_id')],
            'revoked': False}
        if state.get('parent_execution_provenance'):
            record['parent_execution_provenance'] = copy.deepcopy(state['parent_execution_provenance'])
            record['source_kind'] = 'fresh_teacher_plan_from_reused_parent'
            record['source_refs'] += [state['parent_execution_provenance']['source_snapshot_sha256'],
                state['parent_execution_provenance']['original_evaluation_id']]
        if behavior.get('status') == 'FAIL' or review.get('status') == 'FAIL':
            record['qualification']['status'] = 'rejected'
        teacher_path = Path(state.get('teacher_record_path', ''))
        if teacher_path.is_file():
            record['teacher_prompt_sha256'] = json.loads(teacher_path.read_text()).get('prompt_sha256')
        previous = dict(ledger.record_items('samples')).get(sample_id)
        if previous and previous.get('revoked'):
            record.update(revoked=True, revocation_reason=previous.get('revocation_reason'))
        if qualified and state.get('contract',{}).get('binding_mode')=='independent_plan_to_prefrozen_checks.v1':
            from research1.data_pipeline.admission_artifacts import seal_positive_artifacts
            if previous and previous.get('admission_artifacts'):
                record['admission_artifacts']=previous['admission_artifacts']
            record['admission_artifacts']=seal_positive_artifacts(record)
        ledger.put_record('samples', sample_id, record)
        return sample_id

    def process_parent(index, task, parent_scope, source_row=None):
        checkpoint = dict(ledger.record_items('init_parents')).get(parent_scope)
        if source_row and checkpoint is not None and (checkpoint.get('parent_result') != source_row['parent_result']
            or checkpoint.get('parent_source') != source_provenance(source_row, protocol)):
            raise ArtifactChangedError('Replanned decision checkpoint changed its original execution provenance')
        if source_row:
            saved = {'_operation_key': parent_scope, 'value': source_row['parent_result']}
            stored = dict(ledger.record_items('init_parent_executions')).get(parent_scope)
            reuse = dict(ledger.record_items('init_parent_source_reuses')).get(parent_scope)
            if ((stored is not None and stored != saved)
                or (reuse is not None and reuse != source_provenance(source_row, protocol))):
                raise ArtifactChangedError('Stored source-parent cache/provenance changed')
        if checkpoint is not None and checkpoint.get('status') in {'UNSCORED', 'TEACHER_ERROR', 'EVIDENCE_PENDING'}:
            return checkpoint
        if checkpoint is not None and checkpoint.get('status') == 'DONE':
            _check_result_artifacts(checkpoint['parent_result'], expected_task=task, expected_code_hash=tree_hash(baseline))
            for record in ledger.records('samples'):
                if record.get('root_lineage_id') == checkpoint.get('root_lineage_id'):
                    _check_attempt(record['state'], record.get('implementation') or {})
                    child = record.get('qualification', {}).get('implementation_task_result')
                    if child and _usable(child):
                        _check_result_artifacts(child, expected_task=task, expected_code_hash=record['implementation'].get('code_hash'))
            return checkpoint
        if source_row:
            # No evaluate(), job reservation, new EvaluationSpec or fabricated
            # current-protocol score. Keep the old result exactly byte-for-byte.
            parent = copy.deepcopy(source_row['parent_result'])
            existing = dict(ledger.record_items('init_parent_executions')).get(parent_scope)
            saved = {'_operation_key': parent_scope, 'value': parent}
            if existing is not None and existing != saved:
                raise ArtifactChangedError('Cached reused parent differs from the original source execution')
            ledger.put_record('init_parent_executions', parent_scope, saved)
            ledger.put_record('init_parent_source_reuses', parent_scope, source_provenance(source_row, protocol))
        else:
            parent = _cached(ledger, 'init_parent_executions', parent_scope,
                lambda: execute(baseline, task, parent_scope))
        checkpoint = {'scope': parent_scope, 'task': task, 'status': 'RUNNING', 'parent_result': parent}
        if source_row:
            checkpoint['parent_source'] = source_provenance(source_row, protocol)
        ledger.put_record('init_parents', parent_scope, checkpoint)
        if not _usable(parent):
            checkpoint.update(status='UNSCORED', reason='Parent has no valid official proposal score; next frozen parent remains eligible')
            ledger.put_record('init_parents', parent_scope, checkpoint)
            return checkpoint
        _check_result_artifacts(parent, expected_task=task, expected_code_hash=tree_hash(baseline))
        try:
            state = make_state(task, index, parent, source_row)
        except (ValueError, OSError) as exc:
            _raise_policy_failure(exc)
            checkpoint.update(status='EVIDENCE_PENDING', reason=str(exc))
            ledger.put_record('init_parents', parent_scope, checkpoint)
            return checkpoint
        checkpoint.update(root_lineage_id=state['root_lineage_id'], state_id=state['state_id'])
        ledger.put_record('init_parents', parent_scope, checkpoint)
        directory = campaign / 'decisions' / digest(parent_scope)
        directory.mkdir(parents=True, exist_ok=True)
        def propose_all():
            candidates = propose(copy.deepcopy(state), directory)
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= count:
                raise ValueError('Teacher must return one or two honest plans')
            result, seen = [], set()
            for candidate in candidates:
                candidate = validate_plan(candidate, state)
                signature = digest(candidate)
                if signature not in seen:
                    seen.add(signature);result.append(candidate)
            return result
        try:
            plans = _cached(ledger, 'init_proposals', parent_scope, propose_all)
        except (ValueError, RuntimeError) as exc:
            _raise_policy_failure(exc)
            checkpoint.update(status='TEACHER_ERROR', reason=type(exc).__name__ + ': ' + str(exc))
            ledger.put_record('init_parents', parent_scope, checkpoint)
            return checkpoint
        # The whole candidate set and pending samples exist before any child result.
        atomic_json(directory / 'all_candidates.json', plans)
        state['teacher_record_path'] = str(directory / 'teacher.json')
        scopes = [parent_scope + ':candidate:' + str(i) for i in range(len(plans))]
        for plan, scope in zip(plans, scopes):
            sample_record(state, plan, scope, parent)
        checkpoint.update(raw_plans=len(plans), status='PLANS_SAVED')
        ledger.put_record('init_parents', parent_scope, checkpoint)
        save('teacher_plans_saved')
        for plan, scope in zip(plans, scopes):
            def independent_review():
                try:
                    if reviewer is not None:
                        return reviewer(copy.deepcopy(plan), copy.deepcopy(state))
                    return review_plan(copy.deepcopy(state), copy.deepcopy(plan), campaign / 'reviews' / digest(scope))
                except (ValueError, RuntimeError, OSError) as exc:
                    _raise_policy_failure(exc)
                    return {'status': 'pending_review', 'reason': type(exc).__name__ + ': ' + str(exc),
                        'evidence_refs': [], 'review_type': 'model_fact_review'}
            review = _cached(ledger, 'init_factual_reviews', scope, independent_review)
            if not isinstance(review, dict):
                review = {'status': 'pending_review', 'reason': 'Reviewer returned no structured review', 'evidence_refs': []}
            if review.get('status') != 'PASS':
                _raise_policy_failure(ValueError(str(review.get('reason', ''))))
            if (plan['action']=='edit' and
                state.get('contract',{}).get('binding_mode')=='independent_plan_to_prefrozen_checks.v1'):
                from research1.data_pipeline.applicability import review_applicability
                state['plan_factual_review']=review
                state['plan_contract_binding']=_cached(ledger,'init_plan_applicability',scope,
                    lambda:review_applicability(copy.deepcopy(state),copy.deepcopy(plan),copy.deepcopy(review),
                        campaign/'applicability'/digest(scope)))
            child_result = None
            if plan['action'] == 'keep':
                _check_attempt(state, {'status': 'kept'})
                attempt = {'status': 'kept', 'modules_root': state['modules_root'], 'code_hash': state['code_hash']}
                behavior = {'status': 'PASS', 'covers_plan': True, 'executable': True,
                    'mechanism_id': 'explicit_keep', 'reason': 'Unchanged code; factual eligibility reviewed independently'}
            elif review.get('status') == 'FAIL':
                attempt = {'status': 'not_attempted', 'reason': 'Teacher factual review failed'}
                behavior = {'status': 'pending', 'covers_plan': False, 'executable': False, 'reason': attempt['reason']}
            else:
                implementation_dir = campaign / 'implementations' / digest(scope)
                implementation_dir.mkdir(parents=True, exist_ok=True)
                attempt = _cached(ledger, 'init_implementations', scope,
                    lambda: implement(copy.deepcopy(plan), copy.deepcopy(state), implementation_dir))
                _check_attempt(state, attempt)
                if attempt.get('status') == 'applied':
                    behavior = _cached(ledger, 'init_verifications', scope,
                        lambda: verify(copy.deepcopy(plan), copy.deepcopy(state), copy.deepcopy(attempt), state['contract']))
                    accepted = (behavior.get('status') == 'PASS' and behavior.get('covers_plan') is True
                        and behavior.get('executable') is True and behavior.get('checks_id') == state['contract'].get('checks_id')
                        and behavior.get('mechanism_id') == plan.get('verification_kind'))
                    if accepted and attempt['code_hash'] == state['code_hash']:
                        behavior = {'status': 'FAIL', 'covers_plan': False, 'executable': True, 'reason': 'Edit produced no functional code change'}
                        accepted = False
                    if accepted:
                        child_result = _cached(ledger, 'init_child_executions', scope,
                            lambda: execute(Path(attempt['modules_root']), task, 'init:child:' + scope))
                        if _usable(child_result):
                            _check_result_artifacts(child_result, expected_task=task, expected_code_hash=attempt['code_hash'])
                        else:
                            checkpoint['unscored_children'] = checkpoint.get('unscored_children', 0) + 1
                else:
                    behavior = {'status': 'pending', 'covers_plan': False, 'executable': False,
                        'reason': attempt.get('reason', 'No applied implementation with an applicable contract')}
            sample_record(state, plan, scope, parent, attempt=attempt, behavior=behavior, review=review, child=child_result)
            save('qualifying_plan')
        checkpoint['status'] = 'DONE'
        ledger.put_record('init_parents', parent_scope, checkpoint)
        return checkpoint

    with ledger.controller():
        try:
            save()
            # Recheck the complete source set on every run, even when all new
            # decisions were DONE, so upstream revocation invalidates exports.
            source_rows = validate_parent_sources(campaign, protocol)
            for index, task in enumerate(tasks):
                if source_rows:
                    validate_parent_sources(campaign, protocol)
                status['current_task'] = task
                save('parent_proposal')
                process_parent(index, task, 'init:parent:' + str(index), source_rows[index] if source_rows else None)
                export(stop_on_invalid=True);save('parent_finished')
            parents = ledger.records('init_parents')
            partial = any(r.get('status') != 'DONE' or r.get('unscored_children') for r in parents)
            status.update(status='INIT_BATCH_PARTIAL' if partial else 'INIT_BATCH_COMPLETE', phase='finished',
                finished_unix=time.time())
        except _StopInitialization as exc:
            status.update(status=exc.status, reason=str(exc), blocked_result=exc.result)
        except ArtifactChangedError as exc:
            for lineage in {sample['root_lineage_id'] for sample in ledger.records('samples')}:
                ledger.revoke(lineage, str(exc))
            status.update(status='BLOCKED_ARTIFACT_CHANGED', reason=str(exc))
        except BudgetExceeded as exc:
            status.update(status='BUDGET_EXHAUSTED', reason=str(exc))
        except Exception as exc:
            reason = type(exc).__name__ + ': ' + str(exc)
            status.update(status='BLOCKED_RESOURCE' if 'disk reserve' in str(exc).lower() else 'INCOMPLETE', reason=reason)
            ledger.add_event('initialization_error', {'type': type(exc).__name__, 'reason': str(exc)})
        export();save()
    return status
