"""Explicit, deterministic revalidation of a complete frozen factual-review corpus.

No prompts are rebuilt and no models/evaluations are called. The old critic,
decision, evidence and plan stay immutable; only a linked admission revision is
derived from the original critic's exact response using current local checks.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from types import ModuleType

from research1.data_pipeline import policy
from research1.data_pipeline.engine import ArtifactChangedError
from research1.data_pipeline.initialization import _check_attempt, _check_result_artifacts
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.protocol import load_protocol
from research1.data_pipeline.sft import _qualification, _valid_task
from research1.data_pipeline.util import atomic_json, canonical, digest, tree_hash

SCHEMA = 'research1.factual-admission-recheck.v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ArtifactChangedError('Immutable factual recheck artifact changed: ' + str(path))
    else:
        atomic_json(path, value)


def validator_fingerprint():
    """Cover imported validator helpers, including the observation-level adapter."""
    modules = {policy.__name__: policy, __name__: sys.modules[__name__]}
    for value in vars(policy).values():
        name = getattr(value, '__module__', '')
        module = value if isinstance(value, ModuleType) else sys.modules.get(name)
        if module and 'observation_validator' in module.__name__:
            modules[module.__name__] = module
    for name in ('research1.data_pipeline.sft', 'research1.data_pipeline.initialization',
                 'research1.data_pipeline.source_parents', 'research1.data_pipeline.execution', 'research1.evidence'):
        if name in sys.modules:
            modules[name] = sys.modules[name]
    files = {name: {'file_name': Path(inspect.getsourcefile(module)).name,
                    'sha256': _sha(inspect.getsourcefile(module))}
             for name, module in modules.items()}
    # The wrapper can import its validator locally on first use. Fingerprint
    # that dependency before the first validation too, so the first and cached
    # run never acquire different identities merely due to import timing.
    for name in ('observation_validator', 'research1.data_pipeline.observation_validator'):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin and Path(spec.origin).is_file():
            files[name] = {'file_name': Path(spec.origin).name, 'sha256': _sha(spec.origin)}
    body = {'schema': SCHEMA, 'witness_validator': policy.FACTUAL_WITNESS_VALIDATOR,
            'source_files': files}
    return dict(body, fingerprint_sha256=digest(body))


def _rows(ledger):
    with ledger.connection() as conn:
        return {row['key']: {'revision': row['revision'], 'payload': json.loads(row['payload']),
                    'payload_sha256': hashlib.sha256(row['payload'].encode()).hexdigest()}
                for row in conn.execute("SELECT key,revision,payload FROM records WHERE kind='samples' ORDER BY key")}


def _file_binding(path):
    if not isinstance(path, str) or not path:
        return {'path': None, 'exists': False, 'sha256_observed_at_recheck': None}
    path = Path(path).resolve()
    return {'path': str(path), 'exists': path.is_file(),
            'sha256_observed_at_recheck': _sha(path) if path.is_file() else None}


def _freeze_corpus(campaign, protocol, ledger):
    saved = dict(ledger.record_items('factual_recheck_corpora')).get('fixed')
    if saved:
        path = Path(saved['path'])
        if not path.is_file() or _sha(path) != saved['file_sha256']:
            raise ArtifactChangedError('Frozen factual recheck corpus changed')
        corpus = json.loads(path.read_text(encoding='utf-8'))
        if corpus['corpus_id'] != saved['corpus_id']:
            raise ArtifactChangedError('Factual recheck corpus identity changed')
    else:
        rows = _rows(ledger)
        if not rows:
            raise ValueError('No frozen sample records to recheck')
        status = json.loads((Path(campaign) / 'status.json').read_text(encoding='utf-8'))
        if status.get('status') not in {'INIT_BATCH_COMPLETE', 'INIT_BATCH_PARTIAL', 'INTERRUPTED_SUPERSEDED'}:
            raise ValueError('Factual recheck requires a finished/stopped immutable initialization batch')
        with ledger.connection() as conn:
            if conn.execute("SELECT 1 FROM jobs WHERE status IN ('READY','RUNNING') LIMIT 1").fetchone():
                raise ValueError('Finish the fixed collection batch before factual recheck')
        items = []
        for key, row in rows.items():
            record = row['payload']; review = record.get('qualification', {}).get('factual_review') or {}
            raw = _file_binding(review.get('review_path'))
            result_path = str(Path(raw['path']).with_name('factual_review_result.json')) if raw['path'] else None
            items.append(dict(row, sample_id=key, original_qualification_sha256=digest(record.get('qualification')),
                raw_review=raw, original_review_result=_file_binding(result_path),
                teacher_record=_file_binding(record.get('teacher_record_path', record.get('teacher_path')))))
        body = {'schema': SCHEMA, 'decision_protocol_hash': protocol['protocol_hash'], 'items': items,
            'selection': 'all_current_samples_in_ledger_key_order_once',
            'hash_observation_time': 'first_explicit_recheck_not_original_model_call'}
        corpus = dict(body, corpus_id='factual_corpus_' + digest(body))
        path = Path(campaign) / 'admission_rechecks/corpora' / (corpus['corpus_id'] + '.json')
        _write_once(path, corpus)
        ledger.put_record('factual_recheck_corpora', 'fixed', {'corpus_id': corpus['corpus_id'],
            'path': str(path), 'file_sha256': _sha(path), 'sample_ids': list(rows)})
    if corpus['decision_protocol_hash'] != protocol['protocol_hash']:
        raise ArtifactChangedError('Factual recheck may not replace the decision protocol')
    current = _rows(ledger)
    if set(current) != {item['sample_id'] for item in corpus['items']}:
        raise ArtifactChangedError('Factual recheck requires the complete original fixed sample set')
    for item in corpus['items']:
        # Verify the committed historical payload, not a newly favorable current q.
        with ledger.connection() as conn:
            original = conn.execute("SELECT payload FROM record_history WHERE kind='samples' AND key=? AND revision=?",
                (item['sample_id'], item['revision'])).fetchone()
        if not original or hashlib.sha256(original[0].encode()).hexdigest() != item['payload_sha256']:
            raise ArtifactChangedError('Original sample admission history changed')
        for field in ('raw_review', 'original_review_result', 'teacher_record'):
            binding = item[field]
            if _file_binding(binding['path']) != binding:
                raise ArtifactChangedError('Frozen original raw record changed: ' + field)
        now = current[item['sample_id']]['payload']; old = item['payload']
        for field in ('state', 'plan', 'protocol_id', 'root_lineage_id', 'candidate_id', 'implementation', 'source_kind'):
            if now.get(field) != old.get(field):
                raise ArtifactChangedError('Recheck cannot change original decision facts: ' + field)
        derived = now.get('factual_recheck')
        if derived:
            key = derived.get('run_id', '') + ':' + item['sample_id']
            binding = dict(ledger.record_items('factual_recheck_outcomes')).get(key)
            if not binding or not Path(binding['path']).is_file() or _sha(binding['path']) != binding['sha256']:
                raise ArtifactChangedError('Current admission is not linked to its immutable deterministic outcome')
            expected = json.loads(Path(binding['path']).read_text(encoding='utf-8'))['proposed_record']
        else:
            expected = old
        clean = lambda value: {k: v for k, v in value.items() if k not in {'revoked', 'revocation_reason'}}
        if clean(now) != clean(expected):
            raise ArtifactChangedError('Current admission changed outside its linked recheck/revocation history')
    return corpus


def _old_review_identity(item):
    """Use the stored old prompt bytes, never today's review prompt template."""
    record = item['payload']; state, plan = record['state'], record['plan']
    old = record['qualification']['factual_review']; binding = item['raw_review']
    if not binding['exists']:
        raise ValueError('Original raw factual review is missing')
    saved = json.loads(Path(binding['path']).read_text(encoding='utf-8'))
    prompt, raw, metadata = saved['prompt'], saved['raw'], saved['metadata']
    if not isinstance(prompt, str) or not isinstance(raw, str) or not isinstance(metadata, dict):
        raise ValueError('Original critic prompt/response/metadata has invalid types')
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if saved.get('prompt_sha256') != prompt_sha:
        raise ValueError('Stored original prompt SHA does not match its bytes')
    if (saved.get('review_type') != 'model_fact_review' or old.get('review_type') != 'model_fact_review'
        or old.get('independent') is not True or saved.get('review_policy') != old.get('review_policy')
        or not prompt.startswith('Factual review policy: ' + saved['review_policy'] + '. ')):
        raise ValueError('Original independent review policy identity is incomplete')
    if (metadata.get('raw_field') != 'content' or metadata.get('finish_reason') != 'stop'
        or metadata.get('temperature') != 0):
        raise ValueError('Original critic response was incomplete or outside its frozen policy')
    expected_model = state.get('review_model', state.get('teacher_model'))
    expected_provider = state.get('review_provider', state.get('teacher_provider'))
    if (not expected_model or not expected_provider or metadata.get('model') != expected_model
        or metadata.get('provider') != expected_provider):
        raise ValueError('Original critic model/provider cannot be authenticated against the frozen state')
    reviewer_id = 'factual_review_' + digest({'prompt': prompt_sha, 'model': metadata['model'], 'provider': metadata['provider']})[:24]
    if old.get('reviewer_id') != reviewer_id:
        raise ValueError('Original reviewer id does not bind this saved prompt/model/provider')
    # Parse only the two exact JSON payloads embedded in the saved old request.
    prefix, candidate = prompt.rsplit('\nCandidate:\n', 1)
    _, view_text = prefix.rsplit('\nFrozen visible view:\n', 1)
    old_view, old_plan = json.loads(view_text), json.loads(candidate)
    view = policy.public_view(state)
    if (old_plan != plan or old_view != view or saved.get('plan_sha256') != digest(plan)
        or old.get('plan_sha256') != digest(plan) or saved.get('visible_view_sha256') != digest(view)
        or old.get('visible_view_sha256') != digest(view)):
        raise ValueError('Original prompt, qualification, plan or visible-view hashes disagree')
    expected_text = {row['id']: hashlib.sha256(row['text'].encode()).hexdigest() for row in view['evidence']}
    if old.get('evidence_text_sha256') != expected_text:
        raise ValueError('Original qualification does not bind the exact frozen evidence text')
    if saved.get('raw_sha256') is not None and saved['raw_sha256'] != hashlib.sha256(raw.encode()).hexdigest():
        raise ValueError('Original response SHA does not match its bytes')
    if old.get('raw_review_file_sha256') is not None and old['raw_review_file_sha256'] != binding['sha256_observed_at_recheck']:
        raise ValueError('Historically stored raw-file SHA does not match')
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('status') not in {'PASS', 'pending_review', 'FAIL'}:
        raise ValueError('Original critic has no complete structured verdict')
    return saved, value


def _bound_result(result, *, task, code_hash, protocol, ledger, source_rows=(), parent=False):
    if not _valid_task(result):
        raise ValueError('Existing actual proposal execution is not a valid scored task')
    spec = result.get('evaluation_spec') or {}
    if (spec.get('role') != 'proposal' or spec.get('task') != task or spec.get('code_hash') != code_hash
        or result.get('evaluation_id') != digest(spec) or not result.get('result_sha256')
        or not result.get('trajectory_sha256') or not result.get('job_key')):
        raise ValueError('Existing actual execution identity is incomplete or belongs to another task/code')
    _check_result_artifacts(result, expected_task=task, expected_code_hash=code_hash)
    if not spec.get('task_hash') or spec['task_hash'] != protocol.get('task_asset_hashes', {}).get(task):
        raise ValueError('Existing actual execution task/scorer assets do not match')
    if parent and source_rows:
        if not any(row['parent_result'] == result for row in source_rows):
            raise ValueError('Reused parent result differs from the verified source execution')
    else:
        job = ledger.get_job(result['job_key'])
        if (spec.get('protocol_hash') != protocol['protocol_hash'] or not job or job['status'] != 'COMPLETE'
            or job['spec'] != spec or job['result'] != result):
            raise ValueError('Existing execution is not the exact immutable completed current-protocol job')


def _existing_admission_checks(record, protocol, ledger, source_rows):
    state, plan, q = record['state'], record['plan'], record['qualification']
    policy.validate_plan(plan, state)
    tasks = state.get('proposal_task_ids')
    if not isinstance(tasks, list) or len(tasks) != 1 or tasks[0] not in protocol['task_sets']['proposal']:
        raise ValueError('Single-plan admission requires its original proposal task')
    task = tasks[0]
    expected_split = 'train' if task in protocol['task_sets']['proposal_train'] else 'validation'
    if state.get('split') != expected_split or record.get('root_lineage_id') != state.get('root_lineage_id'):
        raise ValueError('Original root/split binding is invalid')
    if tree_hash(Path(state['modules_root'])) != state.get('code_hash'):
        raise ValueError('Original parent code changed')
    _bound_result(q.get('proposal_task_result'), task=task, code_hash=state['code_hash'],
        protocol=protocol, ledger=ledger, source_rows=source_rows, parent=True)
    # Authenticate the frozen view against its actual original trajectory and
    # frozen selector, rather than trusting hand-written view hashes alone.
    import research1.evidence as selector
    selector_rel = 'src/research1/evidence.py'
    expected_selector = protocol.get('source_sha256', {}).get(selector_rel)
    if (not expected_selector or _sha(selector.__file__) != expected_selector
        or _sha(Path(ledger.campaign) / 'source_snapshot' / selector_rel) != expected_selector):
        raise ValueError('Frozen decision-time evidence selector cannot be authenticated')
    evidence = selector.select_trajectory_evidence(Path(q['proposal_task_result']['trajectory']), **protocol['evidence_budget'])
    for item in evidence:
        item.pop('task_result', None); item['id'] = task + ':' + item['id']; item['proposal_task_id'] = task
    if evidence != state['evidence']:
        raise ValueError('Frozen supplied evidence is not the original bounded proposal trajectory view')
    teacher = Path(record.get('teacher_record_path', record.get('teacher_path', '')))
    if not teacher.is_file():
        raise ValueError('Original teacher raw record is missing')
    saved_teacher = json.loads(teacher.read_text(encoding='utf-8'))
    teacher_prompt = saved_teacher.get('prompt')
    if not isinstance(teacher_prompt, str):
        raise ValueError('Original teacher prompt is missing')
    teacher_sha = hashlib.sha256(teacher_prompt.encode()).hexdigest()
    teacher_metadata = saved_teacher.get('metadata') or {}
    if (record.get('teacher_prompt_sha256') != teacher_sha or saved_teacher.get('prompt_sha256') != teacher_sha
        or saved_teacher.get('visible_view_sha256') != digest(policy.public_view(state))
        or saved_teacher.get('policy_id') != state.get('plan_policy_version')
        or teacher_metadata.get('raw_field') != 'content' or teacher_metadata.get('finish_reason') != 'stop'
        or teacher_metadata.get('temperature') != 0 or teacher_metadata.get('model') != state.get('teacher_model')
        or teacher_metadata.get('provider') != state.get('teacher_provider')
        or plan not in policy.parse_candidates(saved_teacher['raw'], state, max_candidates=state.get('max_candidates', 2))):
        raise ValueError('Original teacher identity, view, complete response or exact plan is not bound')
    behavior, implementation = q.get('behavior') or {}, record.get('implementation') or {}
    if not (behavior.get('status') == 'PASS' and behavior.get('covers_plan') is True and behavior.get('executable') is True):
        raise ValueError('Original executable behavior/preservation checks are not valid')
    _check_attempt(state, implementation)
    if plan['action'] == 'keep':
        if (implementation.get('status') != 'kept' or implementation.get('code_hash') != state['code_hash']
            or behavior.get('mechanism_id') != 'explicit_keep' or q.get('task_result') != q['proposal_task_result']):
            raise ValueError('KEEP lacks its original unchanged-code and proposal binding')
    else:
        if (implementation.get('status') != 'applied' or not state.get('contract', {}).get('checks_id')
            or behavior.get('checks_id') != state['contract']['checks_id']
            or behavior.get('mechanism_id') != plan.get('verification_kind')
            or record.get('mechanism_id') != plan.get('verification_kind')
            or implementation.get('code_hash') == state.get('code_hash')):
            raise ValueError('EDIT lacks original frozen mechanism and implementation checks')
        _bound_result(q.get('implementation_task_result'), task=task, code_hash=implementation['code_hash'],
            protocol=protocol, ledger=ledger)
    valid, reason = _qualification(record)
    if not valid:
        raise ValueError('Existing SFT qualification gate: ' + reason)


def _derive(item, fingerprint, run_id, protocol, ledger, source_rows):
    original = item['payload']; old = original.get('qualification', {}).get('factual_review') or {}
    review = copy.deepcopy(old)
    review.update(status='pending_review', reason='Deterministic recheck has not validated the original critic',
        witness_validator=policy.FACTUAL_WITNESS_VALIDATOR, review_type='deterministic_factual_recheck',
        original_review_type=old.get('review_type'), new_independent_review=False,
        original_review_path=item['raw_review']['path'], recheck_run_id=run_id,
        raw_review_file_sha256_observed_at_recheck=item['raw_review']['sha256_observed_at_recheck'])
    review.pop('witnesses', None)
    try:
        saved, value = _old_review_identity(item)
        review['original_critic_status'] = value['status']
        if value['status'] != 'PASS':
            raise ValueError('Original critic did not PASS; its verdict cannot be upgraded by deterministic checking')
        refs, reason = value.get('evidence_refs'), value.get('reason')
        ids = {row['id'] for row in policy.public_view(original['state'])['evidence']}
        if not isinstance(reason, str) or not reason.strip() or not isinstance(refs, list) or not refs or any(ref not in ids for ref in refs):
            raise ValueError('Original critic reason/evidence references are invalid')
        witnesses = policy._review_witnesses(value.get('witnesses'), original['state'], original['plan']['action'])
        if any(witness['evidence_ref'] not in refs for witness in witnesses):
            raise ValueError('Original witness references are not cited by the original critic')
        claims = original['plan']['problem_hypothesis'] + ' ' + original['plan']['intended_change']
        if re.search(r'\b(?:globally\s+defect[- ]free|defect[- ]free\s+(?:everywhere|globally)|no\s+(?:defects?|warnings?|errors?)\s+(?:exist\s+)?(?:anywhere|everywhere|in\s+all\s+tasks))\b', claims, re.I):
            raise ValueError('Original blanket global absence claim is unsupported')
        review.update(status='PASS', reason=reason, evidence_refs=refs, witnesses=witnesses,
            original_prompt_sha256=saved['prompt_sha256'],
            raw_response_sha256_observed_at_recheck=hashlib.sha256(saved['raw'].encode()).hexdigest())
    except (ValueError, TypeError, KeyError, OSError) as exc:
        review['reason'] = type(exc).__name__ + ': ' + str(exc)
    proposed = copy.deepcopy(original)
    q = proposed['qualification']; q['factual_review'] = review; q['status'] = 'verified'
    eligible, reason = False, review['reason']
    if review['status'] == 'PASS':
        try:
            _existing_admission_checks(proposed, protocol, ledger, source_rows)
            eligible, reason = True, 'Original critic PASS revalidated; existing behavior and task bindings verified'
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reason = type(exc).__name__ + ': ' + str(exc)
    q['status'] = 'verified' if eligible else ('rejected' if original['qualification'].get('status') == 'rejected' else 'pending')
    derivation = {'schema': SCHEMA, 'run_id': run_id, 'corpus_id': run_id.split(':')[0],
        'original_sample_revision': item['revision'], 'original_payload_sha256': item['payload_sha256'],
        'original_qualification_sha256': item['original_qualification_sha256'],
        'original_raw_file_sha256_observed_at_recheck': item['raw_review']['sha256_observed_at_recheck'],
        'validator_fingerprint_sha256': fingerprint['fingerprint_sha256'],
        'new_independent_review': False, 'reason': reason}
    proposed['factual_recheck'] = derivation
    proposed['source_refs'] = list(dict.fromkeys(proposed.get('source_refs', []) + [run_id, derivation['corpus_id']]))
    return {'sample_id': item['sample_id'], 'derived_review': review, 'admission_status': q['status'],
        'proposed_record': proposed, 'derivation': derivation}


def recheck_factual(campaign: Path, *, ledger=None) -> dict:
    """Caller holds its controller lock. Runs once per frozen corpus/fingerprint."""
    from research1.data_pipeline.source_parents import audit_sources_for_export, validate_parent_sources
    campaign = Path(campaign).resolve(); ledger = ledger or Ledger(campaign)
    protocol = load_protocol(campaign)
    guard = audit_sources_for_export(campaign, protocol, ledger)
    if guard and guard['status'] != 'PASS':
        return {'status': 'BLOCKED_ARTIFACT_CHANGED', 'reason': guard['reason']}
    try:
        corpus = _freeze_corpus(campaign, protocol, ledger)
        fingerprint = validator_fingerprint()
        run_id = corpus['corpus_id'] + ':' + fingerprint['fingerprint_sha256']
        directory = campaign / 'admission_rechecks/runs' / digest(run_id)[:24]
        _write_once(directory / 'validator.json', fingerprint)
        prior = dict(ledger.record_items('factual_recheck_runs')).get(run_id)
        if prior and prior.get('status') == 'COMPLETE':
            for binding in prior['outcome_files']:
                if _sha(binding['path']) != binding['sha256']:
                    raise ArtifactChangedError('Completed deterministic recheck output changed')
            _write_once(directory / 'manifest.json', dict(prior['summary'],
                validator_fingerprint=fingerprint, outcome_files=prior['outcome_files']))
            active = ledger.records('samples')
            source_rows = validate_parent_sources(campaign, protocol)
            for record in active:
                if not record.get('revoked') and record.get('qualification', {}).get('status') == 'verified':
                    try:
                        _existing_admission_checks(record, protocol, ledger, source_rows)
                    except (ValueError, KeyError, TypeError, OSError) as exc:
                        raise ArtifactChangedError('Committed admission artifacts changed: ' + str(exc)) from exc
            return dict(prior['summary'], cached=True,
                active_qualified=sum(not r.get('revoked') and r.get('qualification', {}).get('status') == 'verified' for r in active),
                revoked=sum(bool(r.get('revoked')) for r in active))
        source_rows = validate_parent_sources(campaign, protocol)
        outcomes, bindings = [], []
        for item in corpus['items']:
            path = directory / 'outcomes' / (digest(item['sample_id'])[:24] + '.json')
            outcome_key = run_id + ':' + item['sample_id']
            committed = dict(ledger.record_items('factual_recheck_outcomes')).get(outcome_key)
            if committed:
                if str(path) != committed['path'] or not path.is_file() or _sha(path) != committed['sha256']:
                    raise ArtifactChangedError('Interrupted deterministic outcome bytes changed')
                outcome = json.loads(path.read_text(encoding='utf-8'))
                if (outcome.get('sample_id') != item['sample_id'] or outcome.get('derivation', {}).get('run_id') != run_id
                    or outcome['derivation'].get('original_payload_sha256') != item['payload_sha256']):
                    raise ArtifactChangedError('Interrupted recheck output has different frozen inputs')
            else:
                outcome = _derive(item, fingerprint, run_id, protocol, ledger, source_rows)
                _write_once(path, outcome)
                ledger.put_record('factual_recheck_outcomes', outcome_key, {'path': str(path), 'sha256': _sha(path)})
            outcomes.append(outcome); bindings.append({'path': str(path), 'sha256': _sha(path)})
        # Nothing is admitted until the whole fixed corpus has a saved outcome.
        guard = audit_sources_for_export(campaign, protocol, ledger)
        if guard and guard['status'] != 'PASS':
            return {'status': 'BLOCKED_ARTIFACT_CHANGED', 'reason': guard['reason']}
        for outcome in outcomes:
            if outcome['admission_status'] == 'verified':
                try:
                    _existing_admission_checks(outcome['proposed_record'], protocol, ledger, source_rows)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    raise ArtifactChangedError('Saved deterministic outcome artifacts changed: ' + str(exc)) from exc
        now = _rows(ledger)
        with ledger.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            try:
                for outcome in outcomes:
                    key = outcome['sample_id']; proposed = copy.deepcopy(outcome['proposed_record'])
                    current = now[key]['payload']
                    if current.get('revoked'):
                        proposed.update(revoked=True, revocation_reason=current.get('revocation_reason'))
                    encoded = canonical(proposed)
                    old = conn.execute("SELECT revision,payload FROM records WHERE kind='samples' AND key=?", (key,)).fetchone()
                    if old['payload'] == encoded:
                        continue
                    revision = old['revision'] + 1
                    conn.execute('INSERT INTO record_history VALUES(?,?,?,?,?)', ('samples', key, revision, encoded, time.time()))
                    conn.execute('UPDATE records SET revision=?,payload=? WHERE kind=? AND key=?', (revision, encoded, 'samples', key))
                event = {'run_id': run_id, 'corpus_id': corpus['corpus_id'], 'outcome_files': bindings,
                    'original_revisions': {item['sample_id']: item['revision'] for item in corpus['items']},
                    'validator_fingerprint': fingerprint, 'new_independent_reviews': 0}
                conn.execute('INSERT OR IGNORE INTO events VALUES(?,?,?,?)', (run_id, 'factual_admission_recheck', canonical(event), time.time()))
                conn.commit()
            except BaseException:
                conn.rollback(); raise
        summary = {'status': 'FACTUAL_RECHECK_COMPLETE', 'run_id': run_id, 'corpus_id': corpus['corpus_id'],
            'samples_rechecked': len(outcomes), 'critic_pass_revalidated': sum(o['derived_review']['status'] == 'PASS' for o in outcomes),
            'eligible': sum(o['admission_status'] == 'verified' for o in outcomes),
            'new_model_calls': 0, 'new_evaluations': 0, 'new_independent_reviews': 0, 'cached': False}
        ledger.put_record('factual_recheck_runs', run_id, {'status': 'COMPLETE', 'summary': summary, 'outcome_files': bindings})
        _write_once(directory / 'manifest.json', dict(summary, validator_fingerprint=fingerprint, outcome_files=bindings))
        return summary
    except ArtifactChangedError as exc:
        frozen = dict(ledger.record_items('factual_recheck_corpora')).get('fixed')
        if frozen:
            ledger.revoke(frozen['corpus_id'], str(exc))
        return {'status': 'BLOCKED_ARTIFACT_CHANGED', 'reason': str(exc)}


def audit_recheck_exports(campaign, ledger):
    """Check committed derivation artifacts, without performing another recheck."""
    derived = [row for row in ledger.records('samples') if row.get('factual_recheck')]
    if not derived:
        return None
    frozen = dict(ledger.record_items('factual_recheck_corpora')).get('fixed')
    try:
        if not frozen:
            raise ArtifactChangedError('Derived factual admission has no explicit frozen corpus')
        protocol = load_protocol(campaign)
        corpus = _freeze_corpus(Path(campaign), protocol, ledger)
        from research1.data_pipeline.source_parents import validate_parent_sources
        source_rows = validate_parent_sources(campaign, protocol)
        for record in derived:
            if not record.get('revoked') and record.get('qualification', {}).get('status') == 'verified':
                _existing_admission_checks(record, protocol, ledger, source_rows)
        return {'status': 'PASS', 'corpus_id': corpus['corpus_id'], 'new_model_calls': 0}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        for row in derived:
            ledger.revoke(row['sample_id'], str(exc))
        return {'status': 'FAIL', 'reason': str(exc), 'new_model_calls': 0}
