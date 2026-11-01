"""Frozen-policy calibration: real decisions, shared first edits and MC paths.

Decision nodes are not code versions. A real keep or rejected implementation
advances the decision index while retaining the last accepted code and scores.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

from research1.data_pipeline.execution import evaluate, tree_hash
from research1.data_pipeline.ledger import BudgetExceeded, Ledger
from research1.data_pipeline.util import digest


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _usable(result: dict) -> bool:
    value = result.get('reward')
    return bool(result.get('status') == 'PASS' and result.get('failure_layer') not in
                {'infrastructure', 'container', 'scorer', 'unknown'}
                and isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value))


class IncompleteEvaluation(RuntimeError):
    def __init__(self, result: dict):
        super().__init__(result.get('error') or 'A required task did not have a valid official score')
        self.result = result


class InsufficientEvidence(RuntimeError):
    pass


class ArtifactChangedError(ValueError):
    pass


def _cached(ledger: Ledger, kind: str, key: str, function):
    existing = {row['_operation_key']: row for row in ledger.records(kind)}
    if key in existing:
        return copy.deepcopy(existing[key]['value'])
    value = function()
    ledger.put_record(kind, key, {'_operation_key': key, 'value': value})
    return value


def _public_state(modules: Path, evidence: list, history: list, lineage: str, split: str,
                  proposal_tasks: list, protocol: dict) -> dict:
    permitted = ['agent_loop', 'observation', 'tools', 'context_mgmt', 'verification']
    code = {}
    for module in permitted:
        path = modules / module / 'baseline.py'
        if path.is_file():
            code[f'{module}/baseline.py'] = path.read_text(encoding='utf-8')
    state = {'root_lineage_id': lineage, 'split': split, 'modules_root': str(modules.resolve()),
             'module_code': code, 'evidence': evidence, 'history': history,
             'proposal_task_ids': list(proposal_tasks), 'permitted_modules': permitted,
               'plan_policy_version': protocol.get('sft',{}).get('plan_policy_version','proposal_single_plan_v2'),
             'protocol_id': protocol['protocol_id'],
             'policy_id': digest(protocol['policies']), 'code_hash': tree_hash(modules)}
    from research1.data_pipeline.module_adapters import get_adapter
    adapter = get_adapter(protocol)
    if protocol.get('verification_registry_version') is not None:
        state['verification_registry_version'] = adapter.REGISTRY_VERSION
    if protocol.get('framework_context') is not None:
        state['framework_context'] = copy.deepcopy(protocol['framework_context'])
    if protocol.get('sft',{}).get('fact_index_projection'):
        state['fact_index_projection']=protocol['sft']['fact_index_projection']
    if state['plan_policy_version'] in {'proposal_single_plan_v5','proposal_single_plan_v6'}:
        state['verification_interfaces'] = adapter.planning_interfaces(permitted,
            include_scope=state['plan_policy_version']=='proposal_single_plan_v6')
    teacher=protocol['policies'].get('teacher',{})
    implementer=protocol['policies'].get('implementer',{})
    for role,policy in [('teacher',teacher),('editor',implementer),('review',teacher),('diagnostic',teacher)]:
        if policy.get('model'):state[role+'_model']=policy['model']
        if policy.get('provider'):state[role+'_provider']=policy['provider']
    review_policy=protocol.get('policies',{}).get('factual_reviewer',{}).get('version')
    if review_policy in {'proposal-factual-review.v6.typed','proposal-factual-review.v7.selectors','proposal-factual-review.v8.probe_io'}:
        state['factual_review_policy']=review_policy
    state['state_id'] = 'state_' + digest({k: v for k, v in state.items() if k != 'modules_root'})[:24]
    return state


def _plan_review(plan: dict, state: dict, reviewer) -> dict:
    if reviewer is None:
        return {'status': 'pending', 'reason': 'Independent factual review has not been supplied',
                'evidence_refs': [], 'reviewer_id': None}
    review = reviewer(copy.deepcopy(plan), copy.deepcopy(state))
    if not isinstance(review, dict):
        return {'status': 'pending', 'reason': 'Reviewer returned no structured factual review', 'evidence_refs': []}
    return review


def _history(node_id: str, plan: dict, attempt: dict, verification: dict) -> dict:
    return {'decision_id': node_id, 'action': plan['action'], 'plan': copy.deepcopy(plan),
            'implementation_status': attempt.get('status'),
            'verification_feedback': {k: verification[k] for k in ('status', 'reason', 'mechanism_id') if k in verification},
            'evidence_refs': list(plan.get('evidence_refs', []))}


def _return_components(nodes: list, tasks: list, weight: float, penalty: float) -> dict:
    for node in nodes:
        if set(node['scores']) != set(tasks) or any(not math.isfinite(node['scores'][t]) for t in tasks):
            raise ValueError('Incomplete fixed-task score table')
    gain = sum(weight * (nodes[-1]['scores'][t] - nodes[0]['scores'][t]) for t in tasks)
    regression = sum(weight * max(0., before['scores'][t] - after['scores'][t])
                     for before, after in zip(nodes, nodes[1:]) for t in tasks)
    return {'terminal_gain': gain, 'cumulative_regression': regression,
            'u_lambda': gain - penalty * regression, 'lambda_value': penalty}


def run_campaign(campaign: Path, *, evaluator=None, proposer=None, implementer=None,
                 verifier=None, reviewer=None, diagnostician=None) -> dict:
    """Resume the same preregistered calibration, never a favorable replacement.

    The evaluator receives the same arguments as execution.evaluate. Injected
    tests may return ``evidence`` directly; production evidence is selected only
    from the concrete proposal-role trajectory returned by the exact binder.
    """
    from research1.data_pipeline.protocol import load_protocol
    from research1.data_pipeline.module_adapters import get_adapter
    from research1.data_pipeline.policy import collect_candidates
    protocol = load_protocol(Path(campaign), verify_source=evaluator is None)
    module_adapter = get_adapter(protocol)
    campaign = Path(campaign)
    ledger = Ledger(campaign)
    evaluate_task = evaluator or evaluate
    propose = proposer or collect_candidates
    implement = implementer or module_adapter.implement
    verify = verifier or module_adapter.verify
    modules = Path(protocol['modules_root'])
    sampling = protocol['sampling']
    K, M, L, R = sampling['K'], sampling['M'], sampling['L_after'], sampling['R']
    if (K, M, L) != (2, 2, 2) or R < 1:
        raise ValueError('This bounded calibration requires K=2, M=2, L_after=2 and positive R')
    task_sets = protocol['task_sets']
    split=protocol.get('collection_split','train')
    if split not in {'train','validation'}:raise ValueError('Unknown collection split')
    proposal_pool = task_sets['proposal_validation' if split=='validation' else 'proposal_train']
    reward_tasks = task_sets['reward'][:sampling['reward_n']]
    dev_tasks = task_sets['dev'][:sampling['dev_n']]
    if not reward_tasks or len(proposal_pool) < sampling['proposal_per_decision']:
        raise ValueError('Frozen task partitions are too small')
    for first, second in [('proposal', 'reward'), ('proposal', 'dev'), ('reward', 'dev')]:
        if set(task_sets[first]) & set(task_sets[second]):
            raise ValueError('Task roles overlap')
    condition = digest({'protocol': protocol.get('protocol_hash', digest(protocol)), 'tasks': reward_tasks,
                        'R': R, 'M': M, 'L': L, 'lambda': sampling['lambda']})
    lineage = 'lineage_' + digest({'campaign': str(campaign.resolve()), 'condition': condition})[:24]
    status = {'schema': 'research1.calibration.v1', 'protocol_id': protocol['protocol_id'],
              'campaign': str(campaign.resolve()), 'status': 'RUNNING', 'phase': 'noise',
              'calibration_complete': False, 'data_collection_complete': False,
              'formal_scale_locked': protocol.get('formal_scale') is not None,
              'started_unix': time.time(), 'sampling': sampling}

    def save(phase=None):
        if phase:
            status['phase'] = phase
        status['ledger'] = ledger.status()
        _write(campaign / 'status.json', status)

    def task_result(code: Path, task: str, role: str, scope: str, repeat=1):
        result = evaluate_task(campaign, protocol, ledger, code, task, role, scope, repeat)
        ledger.add_event('task_result_ref', {'scope': scope, 'task': task, 'role': role,
                         'repeat': repeat, 'evaluation_id': result.get('evaluation_id'),
                         'result_path': result.get('trial_result_path'), 'status': result.get('status')})
        save(role + ':' + scope)
        if not _usable(result):
            raise IncompleteEvaluation(result)
        return result

    def score_table(code: Path, tasks: list, role: str, scope: str):
        table = {}
        results = []
        for task in tasks:
            rewards = []
            for rep in range(1, R+1):
                result = task_result(code, task, role, scope, rep)
                results.append(result)
                rewards.append(result['reward'])
            table[task] = sum(rewards) / R
        return table, results

    def support_state(code: Path, seed: int, scope: str, history: list):
        rng = random.Random(seed)
        tasks = rng.sample(proposal_pool, sampling['proposal_per_decision'])
        manifest = {'scope': scope, 'local_sampling_seed': seed, 'proposal_tasks': tasks,
                    'solver_seed_applied': False, 'fresh_scope': scope}
        draw = _cached(ledger, 'draws', scope, lambda: manifest)
        if draw != manifest:
            raise ValueError('Registered support draw changed')
        evidence = []
        results = []
        for task in tasks:
            result = task_result(code, task, 'proposal', scope)
            results.append(result)
            if evaluator is not None and 'evidence' in result:
                selected = copy.deepcopy(result['evidence'])
            else:
                from research1.evidence import select_trajectory_evidence
                trajectory = result.get('trajectory')
                if not trajectory:
                    raise IncompleteEvaluation(dict(result, reward=None, status='INCOMPLETE',
                                                    error='Proposal result has no recoverable trajectory'))
                selected = select_trajectory_evidence(Path(trajectory), **protocol['evidence_budget'])
            for row in selected:
                # Parent proposal scores may be metadata but never decision input.
                row.pop('task_result', None)
                row['id'] = task + ':' + row['id']
                row['proposal_task_id'] = task
            evidence.extend(selected)
        state = _public_state(code, evidence, history, lineage, split, tasks, protocol)
        base_id = state['state_id']
        def parent_diagnosis():
            if diagnostician is not None:
                return diagnostician(copy.deepcopy(state), campaign / 'diagnostics' / base_id)
            from research1.data_pipeline.diagnostics import review_diagnostics
            return review_diagnostics(copy.deepcopy(state), campaign / 'diagnostics' / base_id)
        state['diagnostics'] = _cached(ledger, 'diagnostics', base_id, parent_diagnosis)
        contract = _cached(ledger, 'contracts', base_id + ':' + digest(state['diagnostics']),
                           lambda: module_adapter.build_contract(state))
        state['contract'] = contract
        state['sampling_stratum'] = ('harness_signal' if contract.get('selected') else
                                     'valid_failure' if any(r['reward']==0 for r in results) else 'success_control')
        state['verification_catalog'] = module_adapter.proposal_catalog(state, contract)
        state['state_id'] = 'state_' + digest({'base_id': base_id, 'checks_id': contract.get('checks_id', digest(contract)),
                                             'verification_catalog': state['verification_catalog']})[:24]
        _write(campaign / 'states' / state['state_id'] / 'state.json', state)
        ledger.put_record('states', state['state_id'], state)
        return state, results

    def candidates(state: dict, scope: str):
        directory = campaign / 'decisions' / digest(scope)
        directory.mkdir(parents=True, exist_ok=True)
        _write(directory / 'decision_scope.json', {'scope': scope})
        plans = _cached(ledger, 'proposals', scope, lambda: propose(copy.deepcopy(state), directory))
        if not isinstance(plans, list) or not 1 <= len(plans) <= K:
            raise ValueError('Planner must emit one or two independently justified candidates')
        _write(directory / 'all_candidates.json', plans)
        state['teacher_record_path'] = str(directory / 'teacher.json')
        return plans

    def factual_review(plan, state, scope):
        if state.get('contract',{}).get('binding_mode')=='independent_plan_to_prefrozen_checks.v1':
            scope='plan:'+digest({'state_id':state['state_id'],'plan':plan})
        def independent_review():
            if reviewer is not None:
                return _plan_review(plan,state,reviewer)
            from research1.data_pipeline.policy import review_plan
            return review_plan(copy.deepcopy(state),copy.deepcopy(plan),campaign/'reviews'/digest(scope))
        return _cached(ledger,'factual_reviews',scope,independent_review)

    def bound_state(plan,state,scope):
        if (plan['action']!='edit' or
            state.get('contract',{}).get('binding_mode')!='independent_plan_to_prefrozen_checks.v1'):
            return state
        from research1.data_pipeline.applicability import review_applicability
        state=copy.deepcopy(state)
        review=factual_review(plan,state,scope)
        key='plan:'+digest({'state_id':state['state_id'],'plan':plan})
        state['plan_factual_review']=review
        state['plan_contract_binding']=_cached(ledger,'plan_applicability',key,
            lambda:review_applicability(copy.deepcopy(state),copy.deepcopy(plan),copy.deepcopy(review),
                campaign/'applicability'/digest(key)))
        return state

    def realize(plan: dict, state: dict, scope: str):
        state=bound_state(plan,state,scope)
        directory = campaign / 'decisions' / digest(scope)
        directory.mkdir(parents=True, exist_ok=True)
        if tree_hash(Path(state['modules_root'])) != state['code_hash']:
            raise ArtifactChangedError('Frozen decision parent changed')
        if plan['action'] == 'keep':
            attempt = {'status': 'kept', 'modules_root': state['modules_root'], 'code_hash': state['code_hash']}
            verification = {'status': 'PASS', 'covers_plan': True, 'mechanism_id': 'explicit_keep',
                            'executable': True,
                            'reason': 'No code change; factual eligibility is reviewed separately'}
        else:
            attempt = _cached(ledger, 'implementations', scope,
                              lambda: implement(copy.deepcopy(plan), copy.deepcopy(state), directory))
            if attempt.get('status') == 'applied':
                patch_path=Path(attempt.get('patch_path',''))
                if (attempt.get('parent_code_hash') != state['code_hash']
                        or tree_hash(Path(attempt['modules_root'])) != attempt.get('code_hash')
                        or not patch_path.is_file()
                        or hashlib.sha256(patch_path.read_bytes()).hexdigest() != attempt.get('patch_sha256')):
                    raise ArtifactChangedError('Stored first/continuation implementation or patch changed')
                verification = _cached(ledger, 'verifications', scope,
                    lambda: verify(copy.deepcopy(plan), copy.deepcopy(state), copy.deepcopy(attempt), state['contract']))
            else:
                verification = {'status': 'pending', 'covers_plan': False,
                                'reason': attempt.get('reason', 'Implementation did not apply')}
        accepted = bool(verification.get('status') == 'PASS' and verification.get('covers_plan') is True
                        and verification.get('executable') is True
                        and (plan['action']=='keep' or verification.get('checks_id')==state['contract'].get('checks_id')))
        child = Path(attempt['modules_root']) if accepted else Path(state['modules_root'])
        code_hash = tree_hash(child)
        functional = plan['action'] == 'edit' and accepted and code_hash != state['code_hash']
        if plan['action'] == 'edit' and accepted and not functional:
            accepted = False
            verification = {'status': 'FAIL', 'covers_plan': False, 'reason': 'Edit has no code change'}
        ledger.add_event('implementation_outcome', {'scope': scope, 'action': plan['action'],
                         'accepted': accepted, 'functional_change': functional,
                         'attempt': attempt, 'verification': verification})
        ledger.put_record('transitions', scope, {'decision_id': scope, 'action': plan['action'],
            'outcome_class': 'apply-failure' if attempt.get('status') in {'error','apply-failure','rejected'} else None,
            'measurement_status': 'unmeasured', 'implementation_status': attempt.get('status'),
            'behavior_status': verification.get('status'), 'task_reward': None,
            'causal_claim': False, 'task_reward_imputed': False})
        return child, attempt, verification, accepted, functional

    def measured_transition(scope, plan, attempt, accepted, parent_scores, child_scores):
        drops = {t:max(0.,parent_scores[t]-child_scores[t]) for t in reward_tasks}
        gain = sum(child_scores[t]-parent_scores[t] for t in reward_tasks)/len(reward_tasks)
        if attempt.get('status') in {'error','apply-failure','rejected'}:
            outcome = 'apply-failure'
        elif not accepted:
            outcome = None
        else:
            outcome = 'regression' if any(drops.values()) else 'success' if gain>0 else 'no-effect'
        ledger.put_record('transitions', scope, {'decision_id':scope,'action':plan['action'],
            'outcome_class':outcome,'measurement_status':'observed' if accepted else 'unchanged_state_after_rejection',
            'implementation_status':attempt.get('status'),'parent_scores':parent_scores,
            'child_scores':child_scores,'per_task_drops':drops,'immediate_gain':gain,
            'R':R,'descriptive_only':True,'causal_claim':False,'task_reward_imputed':False})

    def store_sample(state, plan, scope, attempt, verification, support_results, implementation_results=None):
        state=bound_state(plan,state,scope)
        review = factual_review(plan,state,scope)
        valid_refs = {e['id'] for e in state['evidence']}
        refs = review.get('evidence_refs')
        factual = (review.get('status') == 'PASS' and review.get('independent') is True
                   and isinstance(review.get('reason'),str) and bool(review['reason'].strip())
                   and isinstance(review.get('reviewer_id'),str) and bool(review['reviewer_id'].strip())
                   and isinstance(refs, list) and bool(refs) and set(refs).issubset(valid_refs))
        behavior = (verification.get('status') == 'PASS' and verification.get('covers_plan') is True
                    and verification.get('executable') is True)
        execution = implementation_results if plan['action'] == 'edit' else support_results
        qualified = factual and behavior and bool(execution) and all(_usable(r) for r in execution)
        sample_id = 'sample_' + digest({'state': state['state_id'], 'plan': plan, 'scope': scope})[:24]
        record = {'sample_id': sample_id, 'candidate_id': scope, 'state': state, 'plan': plan,
                  'qualification': {'status': 'verified' if qualified else 'pending',
                                    'factual_review': review, 'behavior': verification,
                                    'proposal_task_result': support_results[0],
                                    'implementation_task_result': execution[0] if execution and plan['action']=='edit' else None,
                                    'task_result': execution[0] if execution else {'status':'INCOMPLETE','reward':None,'failure_layer':'unknown'}},
                  'source_kind': 'fresh_teacher_plan', 'protocol_id': protocol['protocol_id'],
                  'teacher_record_path': state.get('teacher_record_path'),
                  'implementation_code_hash': attempt.get('code_hash'),
                  'implementation': attempt,
                  'mechanism_id': verification.get('mechanism_id', plan.get('verification_kind')),
                  'root_lineage_id': lineage, 'state_id': state['state_id'],
                  'source_refs': [state['state_id'], scope] + [r['evaluation_id'] for r in
                                  support_results + (implementation_results or []) if r.get('evaluation_id')],
                  'revoked': False}
        if verification.get('status') == 'FAIL' or review.get('status') == 'FAIL':
            record['qualification']['status'] = 'rejected'
        if record['qualification']['status']=='verified' and state.get('contract',{}).get('binding_mode')=='independent_plan_to_prefrozen_checks.v1':
            from research1.data_pipeline.admission_artifacts import seal_positive_artifacts
            from research1.data_pipeline.comparison import _collection_prompt
            record['teacher_prompt_sha256']=hashlib.sha256(_collection_prompt(state).encode()).hexdigest()
            previous=dict(ledger.record_items('samples')).get(sample_id)
            if previous and previous.get('admission_artifacts'):
                record['admission_artifacts']=previous['admission_artifacts']
            record['admission_artifacts']=seal_positive_artifacts(record)
        ledger.put_record('samples', sample_id, record)
        return sample_id

    with ledger.controller():
        try:
            save()
            for task in task_sets['debug'] if protocol.get('noise_enabled',True) else []:
                for repeat in range(1, 4):
                    result = task_result(modules, task, 'noise', 'noise', repeat)
                    ledger.put_record('noise', task + ':' + str(repeat), {'task': task, 'repeat': repeat,
                                      'reward': result['reward'], 'result': result})
            root_seed = int(digest({'seed': protocol['seed'], 'scope': 'root'})[:16], 16)
            state, initial_support = support_state(modules, root_seed, 'root', [])
            plans = candidates(state, 'root')
            manifest = {'root_state_id': state['state_id'], 'root_lineage_id': lineage,
                        'condition_id': condition, 'candidates': []}
            for index, plan in enumerate(plans):
                candidate = 'candidate_' + digest({'state': state['state_id'], 'index': index, 'plan': plan})[:24]
                paths = [{'path_id': candidate + f':path:{i}',
                          'sampling_seed': int(digest({'master': protocol['seed'], 'candidate': candidate, 'path': i})[:16], 16)}
                         for i in range(M)]
                manifest['candidates'].append({'candidate_id': candidate, 'plan': plan, 'paths': paths})
            registered = _cached(ledger, 'manifests', 'mc', lambda: manifest)
            if registered != manifest:
                raise ValueError('MC candidate/path registration changed')
            _write(campaign / 'mc_manifest.json', manifest)
            root_scores, root_trials = score_table(modules, reward_tasks, 'reward', 'root')
            root_node = {'decision_id': 'root', 'action': 'root', 'code_hash': tree_hash(modules),
                         'modules_root': str(modules), 'scores': root_scores,
                         'evaluation_refs': [r.get('evaluation_id') for r in root_trials],
                         'trial_counts': {t:R for t in reward_tasks}}
            starts = []
            for item in manifest['candidates']:
                cid, plan = item['candidate_id'], item['plan']
                child, attempt, verification, accepted, functional = realize(plan, state, cid + ':first')
                sample = store_sample(state, plan, cid, attempt, verification, initial_support)
                starts.append((item, child, attempt, verification, accepted, functional, sample))
            all_paths = []
            value_records = []
            functional_first = sum(row[5] for row in starts)
            if not functional_first:
                for item, child, attempt, checked, accepted, functional, sample in starts:
                    ledger.put_record('values', item['candidate_id'] + ':' + condition,
                        {'candidate_id':item['candidate_id'],'candidate_ids':[item['candidate_id']],
                         'condition_id':condition,'status':'INCOMPLETE','is_complete':False,
                         'reason':'No real accepted first edit; registered paths were not sampled',
                         'mean_u_lambda':None,'source_refs':[sample,state['state_id']],'revoked':False})
                status.update(functional_first_edits=0,complete_candidates=0,complete_paths=0)
                raise InsufficientEvidence('No real accepted functional first edit; keep-only calibration is not modification acceptance')
            realized=[tree_hash(row[1]) for row in starts if row[4]]
            if len(realized)==K and len(set(realized))<K:
                for item, child, attempt, checked, accepted, functional, sample in starts:
                    ledger.put_record('values',item['candidate_id']+':'+condition,
                        {'candidate_id':item['candidate_id'],'candidate_ids':[item['candidate_id']],
                         'condition_id':condition,'status':'INCOMPLETE','is_complete':False,
                         'mean_u_lambda':None,'reason':'Distinct plans produced identical accepted implementations',
                         'source_refs':[sample,state['state_id']],'revoked':False})
                status.update(functional_first_edits=functional_first,complete_candidates=0,complete_paths=0)
                raise InsufficientEvidence('Two candidate texts realized the same action/code; no replacement sampled')
            for item, child, attempt, verification, accepted, functional, sample in starts:
                cid = item['candidate_id']
                value_key = cid + ':' + condition
                if not accepted:
                    ledger.put_record('values', value_key, {'candidate_id': cid, 'candidate_ids': [cid],
                        'condition_id': condition, 'status': 'INCOMPLETE', 'is_complete': False,
                        'reason': 'First implementation was not accepted', 'mean_u_lambda': None,
                        'source_refs': [sample, state['state_id']], 'revoked': False})
                    continue
                if functional:
                    first_scores, first_trials = score_table(child, reward_tasks, 'reward', cid + ':first')
                else:
                    first_scores, first_trials = root_scores, root_trials
                measured_transition(cid + ':first', item['plan'], attempt, accepted, root_scores, first_scores)
                store_sample(state, item['plan'], cid, attempt, verification, initial_support, first_trials)
                first_node = {'decision_id': cid + ':first', 'action': item['plan']['action'],
                              'code_hash': tree_hash(child), 'modules_root': str(child), 'scores': first_scores,
                              'evaluation_refs': [r.get('evaluation_id') for r in first_trials],
                              'trial_counts': {t:R for t in reward_tasks},
                              'evaluation_reused': not functional}
                expected_hash = first_node['code_hash']
                paths = []
                for path_manifest in item['paths']:
                    pid = path_manifest['path_id']
                    nodes = [copy.deepcopy(root_node), copy.deepcopy(first_node)]
                    current = child
                    history = [_history(cid + ':first', item['plan'], attempt, verification)]
                    refs = [sample, state['state_id']]
                    for decision in range(L):
                        if tree_hash(current) != nodes[-1]['code_hash']:
                            raise ArtifactChangedError('Continuation state changed after its recorded evaluation')
                        scope = pid + ':decision:' + str(decision)
                        seed = int(digest({'path_seed': path_manifest['sampling_seed'], 'decision': decision})[:16], 16)
                        next_state, support = support_state(current, seed, scope, history)
                        next_plans = candidates(next_state, scope)
                        # The continuation policy executes its first proposal; it
                        # cannot inspect downstream reward to select a proposal.
                        next_plan = next_plans[0]
                        new_code, next_attempt, checked, good, changed = realize(next_plan, next_state, scope)
                        if changed:
                            scores, trials = score_table(new_code, reward_tasks, 'reward', scope)
                        else:
                            scores, trials = nodes[-1]['scores'], []
                        measured_transition(scope,next_plan,next_attempt,good,nodes[-1]['scores'],scores)
                        child_sample = store_sample(next_state, next_plan, scope, next_attempt, checked, support, trials)
                        refs.append(child_sample)
                        current = new_code
                        node = {'decision_id': scope, 'action': next_plan['action'],
                                'implementation_status': next_attempt.get('status'), 'accepted': good,
                                'functional_change': changed, 'code_hash': tree_hash(current),
                                'modules_root': str(current), 'scores': scores, 'evaluation_reused': not changed,
                                'evaluation_refs': [r.get('evaluation_id') for r in trials] if changed else nodes[-1]['evaluation_refs'],
                                'trial_counts': {t:R for t in reward_tasks}}
                        nodes.append(node)
                        history = history + [_history(scope, next_plan, next_attempt, checked)]
                    if len(nodes) != L + 2 or nodes[1]['code_hash'] != expected_hash:
                        raise ValueError('Path horizon or shared first implementation changed')
                    components = _return_components(nodes, reward_tasks, 1 / len(reward_tasks), sampling['lambda'])
                    path = {'schema': 'research1.mc_path.v2', 'path_id': pid, 'candidate_id': cid,
                            'first_implementation_hash': expected_hash, 'sampling_seed': path_manifest['sampling_seed'],
                            'rng_mode': 'independent_support_draws_and_fresh_solver_requests',
                            'solver_seed_applied': False, 'condition_id': condition, 'L_after': L,
                            'nodes': nodes, 'is_complete': True,
                            'source_refs': refs + [r for node in nodes for r in node['evaluation_refs'] if r],
                            'reward_tasks': reward_tasks, 'required_repetitions': R, **components}
                    _write(campaign / 'paths' / (digest(pid) + '.json'), path)
                    ledger.put_record('paths', pid, path)
                    paths.append(path)
                    all_paths.append(path)
                if len(paths) != M or len({p['sampling_seed'] for p in paths}) != M:
                    raise ValueError('Incomplete or duplicated independent continuation paths')
                record = {'schema': 'research1.mc_value.v2', 'candidate_id': cid, 'candidate_ids': [cid],
                          'condition_id': condition, 'root_lineage_id': lineage, 'split': split,
                          'policy_id': digest(protocol['policies']),
                          'status': 'COMPLETE', 'is_complete': True, 'first_implementation_hash': expected_hash,
                          'M': M, 'L_after': L, 'R': R, 'reward_tasks': reward_tasks,
                          'task_weights': {t:1/len(reward_tasks) for t in reward_tasks}, 'lambda_value': sampling['lambda'],
                          'paths': [p['path_id'] for p in paths], 'source_refs': list(dict.fromkeys(
                              [sample, state['state_id']] + [r for p in paths for r in p['source_refs']])),
                          'revoked': False}
                record['reward_asset_binding_hash']=digest([r for r in ledger.records('task_assets') if r['task'] in reward_tasks])
                for field in ('terminal_gain', 'cumulative_regression', 'u_lambda'):
                    record['mean_' + field] = statistics.mean(p[field] for p in paths)
                    record['std_' + field] = statistics.stdev(p[field] for p in paths)
                ledger.put_record('values', value_key, record)
                value_records.append(record)
            if len(value_records) == K:
                comparison = {'schema': 'research1.candidate_comparison.v1',
                              'candidate_ids': [v['candidate_id'] for v in value_records],
                              'condition_id': condition, 'source_refs': [v['candidate_id'] + ':' + condition for v in value_records],
                              'observed_delta_u': value_records[0]['mean_u_lambda'] - value_records[1]['mean_u_lambda'],
                              'preference': 'undetermined', 'reason': 'No calibrated preference threshold is frozen',
                              'revoked': False}
                ledger.put_record('comparisons', condition, comparison)
            score_table(modules, dev_tasks, 'dev', 'dev:root')
            for path in all_paths:
                score_table(Path(path['nodes'][-1]['modules_root']), dev_tasks, 'dev', 'dev:' + path['path_id'])
            full = len(value_records) == K and len(all_paths) == K * M
            status.update(status='NEEDS_SCALE_LOCK' if full and functional_first else 'INSUFFICIENT_EVIDENCE',
                          phase='finished', calibration_complete=bool(full and functional_first),
                          functional_first_edits=functional_first, complete_candidates=len(value_records),
                          complete_paths=len(all_paths), independent_candidate_plans=len(plans),
                          new_samples=len(ledger.records('samples')), finished_unix=time.time())
            if not functional_first:
                status['reason'] = 'No real accepted functional first edit; keep-only calibration is not modification acceptance'
            elif not full:
                status['reason'] = 'The preregistered two-candidate continuation group is incomplete'
        except ArtifactChangedError as exc:
            ledger.revoke(lineage,str(exc))
            status.update(status='BLOCKED_ARTIFACT_CHANGED',reason=str(exc))
        except InsufficientEvidence as exc:
            status.update(status='INSUFFICIENT_EVIDENCE',phase='finished',reason=str(exc))
        except IncompleteEvaluation as exc:
            result = exc.result
            status.update(status=('WAITING_EXISTING_JOB' if result.get('in_progress') else
                                  'BLOCKED_CONFIGURATION' if result.get('fatal') else 'INCOMPLETE'),
                          reason=str(exc), blocked_result=result)
        except BudgetExceeded as exc:
            status.update(status='BUDGET_EXHAUSTED', reason=str(exc))
        except Exception as exc:
            status.update(status='INCOMPLETE', reason=f'{type(exc).__name__}: {exc}')
            ledger.add_event('controller_error', {'type': type(exc).__name__, 'reason': str(exc)})
        save()
    return status
