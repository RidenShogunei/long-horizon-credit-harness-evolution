"""Explicit, read-only reuse of a complete frozen set of proposal executions.

This is provenance transfer, not a result cache. Original EvaluationSpecs and
job identities retain their original protocol; a new decision has a new policy
and never claims that the old solver execution ran under it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sqlite3

from research1.config import load_pilot
from research1.data_pipeline.engine import ArtifactChangedError, _usable
from research1.data_pipeline.execution import _image_identity
from research1.data_pipeline.util import atomic_json, digest, tree_hash

SCHEMA = 'research1.initialization.source_parents.v1'
TERMINAL = {'INIT_BATCH_COMPLETE', 'INIT_BATCH_PARTIAL', 'INTERRUPTED_SUPERSEDED',
            'GROUNDED_EDIT_TRIAL_COMPLETE','GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT'}
EXECUTION_SOURCES = ('src/research1/data_pipeline/execution.py', 'src/research1/adapters.py',
                     'src/research1/config.py')


def _sha(path: Path) -> str:
    if not path.is_file():
        raise ArtifactChangedError('Source-parent file is missing: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_ledger(campaign: Path) -> dict:
    """Do not construct Ledger: even its constructor writes schema pragmas."""
    path = campaign / 'ledger.sqlite3'
    if not path.is_file():
        raise ValueError('Source parent campaign has no committed ledger')
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('BEGIN')
        records = {(row['kind'], row['key']): json.loads(row['payload']) for row in
                   conn.execute('SELECT kind,key,payload FROM records')}
        jobs = {row['key']: dict(row, spec=json.loads(row['spec']),
                result=json.loads(row['result']) if row['result'] else None) for row in
                conn.execute('SELECT key,status,spec,spec_hash,result,kind FROM jobs')}
        events = [json.loads(row[0]) for row in conn.execute("SELECT payload FROM events WHERE kind='revocation'")]
        return {'records': records, 'jobs': jobs, 'revocations': events}
    finally:
        conn.close()


def preview_handover_budget(previous: Path, successor: Path) -> dict:
    """Validate compatibility before the existing handover mutates its predecessor."""
    from research1.data_pipeline.protocol import load_protocol
    previous, successor = Path(previous).resolve(), Path(successor).resolve()
    if previous == successor:
        raise ValueError('Budget predecessor must be a different campaign')
    protocol = load_protocol(previous)
    status = json.loads((previous / 'status.json').read_text(encoding='utf-8'))
    if status.get('status') not in TERMINAL | {'INCOMPLETE', 'INSUFFICIENT_EVIDENCE'}:
        raise ValueError('Budget predecessor must be stopped before replanning preparation')
    rows = _read_ledger(previous)
    if any(job['status'] in {'READY', 'RUNNING'} for job in rows['jobs'].values()):
        raise ValueError('Budget predecessor still has unfinished reserved jobs')
    prior = protocol.get('budget_lineage') or {}
    authorized = prior.get('authorized', protocol['budget'])
    carried = prior.get('carried_spend', {'regular': 0, 'retries': 0, 'total': 0})
    own = {kind: sum(job['kind'] == kind for job in rows['jobs'].values()) for kind in ('regular', 'retry')}
    spent = {'regular': carried['regular'] + own['regular'], 'retries': carried['retries'] + own['retry']}
    spent['total'] = spent['regular'] + spent['retries']
    remaining = {key: authorized[key] - spent[key] for key in ('regular', 'retries', 'total')}
    if remaining['regular'] < 1 or remaining['total'] < 1 or remaining['retries'] < 0:
        raise ValueError('No authorized evaluation budget remains')
    result = {'authorized': authorized, 'carried_spend': spent, 'remaining': remaining,
        'previous_campaign': str(previous), 'previous_protocol_hash': protocol['protocol_hash'],
        'successor_campaign': str(successor)}
    prior_handover = rows['records'].get(('budget_handover', 'successor'))
    if prior_handover is not None and prior_handover != result:
        raise ValueError('Budget predecessor already transferred authorization to another successor')
    return result


def execution_context(protocol: dict) -> dict:
    """Fields relevant to a solver run; teacher/review/contract revisions may differ."""
    return {key: copy.deepcopy(protocol.get(key)) for key in ('config_hash', 'modules_hash',
        'module_files', 'dataset_manifest_sha256', 'runtime_source_hash', 'runtime')} | {
        'solver': copy.deepcopy(protocol.get('policies', {}).get('solver')),
        'execution_sources': {key: protocol.get('source_sha256', {}).get(key) for key in EXECUTION_SOURCES}}


def _compatible(source: dict, target: dict) -> dict:
    old, new = execution_context(source), execution_context(target)
    if old != new:
        differing = [key for key in old if old[key] != new[key]]
        raise ValueError('Source parent execution conditions differ: ' + ', '.join(differing))
    if any(value is None for value in old.values()) or any(value is None for value in old['execution_sources'].values()):
        raise ValueError('Source parent execution compatibility is incomplete')
    if source['task_sets'] != target['task_sets']:
        raise ValueError('Replanning must retain original task roles and root train/validation groups')
    for protocol in (source, target):
        if tree_hash(Path(protocol['modules_root'])) != protocol['modules_hash']:
            raise ArtifactChangedError('Source/target baseline differs from frozen full content hash')
        if tree_hash(Path(protocol['upstream_root']) / 'src/harbor') != protocol['runtime_source_hash']:
            raise ArtifactChangedError('Actual Harbor runtime differs from the compatible frozen hash')
        if _sha(Path(protocol['dataset_path']) / 'manifest.json') != protocol['dataset_manifest_sha256']:
            raise ArtifactChangedError('Actual dataset manifest differs from frozen source-parent binding')
        if load_pilot(Path(protocol['config_path']))['_config_hash'] != protocol['config_hash']:
            raise ArtifactChangedError('Actual solver configuration changed since the parent execution')
    return {'source_execution_context_sha256': digest(old), 'target_execution_context_sha256': digest(new),
        'source_modules_root': source['modules_root'], 'target_modules_root': target['modules_root'],
        'code_content_sha256': source['modules_hash'], 'path_relocation_allowed': True,
        'original_execution_protocol_hash': source['protocol_hash']}


def _check_revocations(rows: dict, row: dict) -> None:
    invalid = {row['root_lineage_id'], row['source_state_id'], row['source_scope'],
        row['parent_result']['evaluation_id'], row['parent_result']['job_key'], row['parent_result']['result_sha256']}
    # A revoked old plan does not revoke the independent execution it used.
    if any(event.get('reference') in invalid for event in rows['revocations']):
        raise ArtifactChangedError('Source parent execution/root was explicitly revoked')
    if row['source_checkpoint'].get('revoked'):
        raise ArtifactChangedError('Source parent checkpoint was revoked')


def _validate_row(source: dict, target: dict, rows: dict, row: dict) -> None:
    result = row['parent_result']; spec = result.get('evaluation_spec') or {}
    task = row['task']; solver = source['policies']['solver']; runtime = source['runtime']
    expected = {'protocol_hash': source['protocol_hash'], 'role': 'proposal', 'task': task,
        'scope': row['source_scope'], 'repeat': 1, 'code_hash': source['modules_hash'],
        'modules_root': source['modules_root'], 'model': solver['model'], 'provider_id': solver['provider'],
        'temperature': solver['temperature'], 'reasoning_effort': solver.get('reasoning_effort'),
        'max_turns': runtime['max_turns'], 'wall_seconds': runtime['wall_seconds']}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ArtifactChangedError('Saved parent EvaluationSpec is not the original compatible proposal execution')
    if not _usable(result) or result.get('fatal') or result.get('role') != 'proposal':
        raise ValueError('Only committed, valid scored proposal parents can be replanned')
    if result.get('evaluation_id') != digest(spec):
        raise ArtifactChangedError('Original parent execution identity does not match its EvaluationSpec')
    job = rows['jobs'].get(result.get('job_key'))
    if (not job or job['status'] != 'COMPLETE' or job['spec'] != spec or job['spec_hash'] != digest(spec)
        or job['result'] != result):
        raise ArtifactChangedError('Source parent is not backed by the exact immutable completed job')
    if not result.get('result_sha256') or not result.get('trajectory_sha256'):
        raise ValueError('Source parent needs complete official result and proposal trajectory byte hashes')
    if _sha(Path(result.get('trial_result_path', ''))) != result['result_sha256']:
        raise ArtifactChangedError('Original parent official result bytes changed')
    if _sha(Path(result.get('trajectory', ''))) != result['trajectory_sha256']:
        raise ArtifactChangedError('Original parent trajectory bytes changed')
    task_hash = source.get('task_asset_hashes', {}).get(task)
    if not task_hash or target.get('task_asset_hashes', {}).get(task) != task_hash or spec.get('task_hash') != task_hash:
        raise ArtifactChangedError('Source parent task/scorer assets are incompatible')
    for task_path in (Path(spec.get('task_path', '')), Path(source['dataset_path']) / 'tasks' / task,
                      Path(target['dataset_path']) / 'tasks' / task):
        if not task_path.is_dir() or tree_hash(task_path) != task_hash:
            raise ArtifactChangedError('Actual full task/scorer tree changed')
    final_assets = result.get('bound_assets') or {}
    image = final_assets.get('image') or {}
    if final_assets.get('task') != task or final_assets.get('task_hash') != task_hash:
        raise ArtifactChangedError('Source parent lacks its final environment/scorer asset binding')
    current = _image_identity(Path(target['dataset_path']) / 'tasks' / task)
    known = image.get('image_ids') or {}
    if (not any(known.values()) or image.get('dockerfile_sha256') != current.get('dockerfile_sha256')
        or any(value is not None and current.get('image_ids', {}).get(name) != value for name, value in known.items())):
        raise ArtifactChangedError('Source parent Docker environment changed or cannot be verified')
    expected_split = 'train' if task in target['task_sets']['proposal_train'] else 'validation'
    if task not in target['task_sets']['proposal'] or row['split'] != expected_split:
        raise ArtifactChangedError('Source parent moved across a proposal role or train/validation root group')
    if row['source_checkpoint'].get('parent_result') != result:
        raise ArtifactChangedError('Source checkpoint and immutable execution differ')
    if rows['records'].get(('init_parents', row['source_scope'])) != row['source_checkpoint']:
        raise ArtifactChangedError('Frozen source parent checkpoint changed')
    state = rows['records'].get(('states', row['source_state_id']))
    if not state or digest(state) != row['source_state_sha256']:
        raise ArtifactChangedError('Frozen source root state changed')
    if state.get('root_lineage_id') != row['root_lineage_id'] or state.get('split') != row['split']:
        raise ArtifactChangedError('Source root identity or split changed')
    _check_revocations(rows, row)


def freeze_parent_sources(source_campaign: Path, target_protocol: dict) -> dict:
    """Select every scored parent in source queue order, once, without score ranking."""
    from research1.data_pipeline.protocol import load_protocol
    source_campaign = Path(source_campaign).resolve()
    source = load_protocol(source_campaign)
    if source.get('execution_mode') != 'initialization' or source.get('source_parents'):
        raise ValueError('Replan from the original initialization execution campaign, not another replan')
    source_status = json.loads((source_campaign / 'status.json').read_text(encoding='utf-8'))
    if source_status.get('status') not in TERMINAL:
        raise ValueError('Source parent campaign must be finished/stopped before freezing reuse')
    compatibility = _compatible(source, target_protocol)
    rows = _read_ledger(source_campaign)
    if any(job['status'] in {'READY', 'RUNNING'} for job in rows['jobs'].values()):
        raise ValueError('Source parent campaign still has a reserved/running execution')
    selected, excluded = [], []
    config = source.get('initialization') or {}
    queue = config.get('task_order', [])[:config.get('max_parents', 0)]
    if not queue or len(set(queue)) != len(queue):
        raise ValueError('Source parent queue is not a valid fixed order')
    for index, task in enumerate(queue):
        scope = 'init:parent:' + str(index)
        checkpoint = rows['records'].get(('init_parents', scope))
        parent = (checkpoint or {}).get('parent_result')
        if not parent or not _usable(parent):
            excluded.append({'source_scope': scope, 'task': task, 'reason': 'No valid scored committed parent'})
            continue
        state_id = checkpoint.get('state_id')
        state = rows['records'].get(('states', state_id))
        if not state or not checkpoint.get('root_lineage_id') or checkpoint['root_lineage_id'] != state.get('root_lineage_id'):
            raise ValueError('Scored source parent has no frozen original root state; cannot fabricate grouping')
        row = {'source_index': index, 'source_scope': scope, 'task': task,
            'root_lineage_id': checkpoint['root_lineage_id'], 'split': state.get('split'),
            'source_state_id': state_id, 'source_state_sha256': digest(state),
            'source_checkpoint': copy.deepcopy(checkpoint), 'parent_result': copy.deepcopy(parent)}
        _validate_row(source, target_protocol, rows, row)
        selected.append(row)
    if not selected or len({row['root_lineage_id'] for row in selected}) != len(selected):
        raise ValueError('Source parent set must contain unique original root lineages')
    snapshots = source.get('source_sha256') or {}
    for rel, expected in snapshots.items():
        if _sha(source_campaign / 'source_snapshot' / rel) != expected:
            raise ArtifactChangedError('Original controller source snapshot changed: ' + rel)
    old_artifacts = {path.relative_to(source_campaign).as_posix(): _sha(path)
        for directory in ('decisions', 'reviews', 'diagnostics', 'states')
        for path in sorted((source_campaign / directory).rglob('*')) if path.is_file()}
    result = {'schema': SCHEMA, 'source_campaign': str(source_campaign),
        'source_protocol_hash': source['protocol_hash'],
        'source_protocol_file_sha256': _sha(source_campaign / 'protocol.json'),
        'source_controller_sha256': copy.deepcopy(snapshots), 'original_model_artifact_sha256': old_artifacts,
        'compatibility': compatibility, 'rows': selected, 'excluded_unscored': excluded,
        'selection': 'all_scored_parents_in_original_queue_order_without_score_ranking',
        'source_independent_parents': len(selected), 'new_independent_parent_increment': 0}
    selection = target_protocol.get('diagnostic_source_selection')
    if selection is not None:
        if (not isinstance(selection,dict) or set(selection)!={'tasks','rationale','frozen_before_candidates'}
            or selection['frozen_before_candidates'] is not True
            or not isinstance(selection['rationale'],str) or len(selection['rationale'].strip())<30
            or not isinstance(selection['tasks'],list) or not selection['tasks']
            or any(not isinstance(task,str) for task in selection['tasks'])
            or len(set(selection['tasks']))!=len(selection['tasks'])
            or not set(selection['tasks']) <= {row['task'] for row in selected}
            or target_protocol.get('trial_driver')!='grounded_edit_trial.v1'):
            raise ValueError('Diagnostic source selection must be explicit, justified and frozen before candidates')
        # All original rows above were validated, and all model artifacts remain
        # byte-bound. This narrows a declared mechanism diagnosis, not an MC
        # path replacement or a new independent parent.
        result['all_scored_source_parents']=[{'task':row['task'],'root_lineage_id':row['root_lineage_id'],
            'source_scope':row['source_scope'],'original_evaluation_id':row['parent_result']['evaluation_id']}
            for row in selected]
        by_task={row['task']:row for row in selected}
        result['rows']=[by_task[task] for task in selection['tasks']]
        result['diagnostic_source_selection']=copy.deepcopy(selection)
        result['selection']='preregistered_diagnostic_subset_all_original_evidence_preserved'
    return result


def write_parent_snapshot(campaign: Path, snapshot: dict) -> dict:
    path = Path(campaign) / 'source_parents.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != snapshot:
            raise ArtifactChangedError('A frozen source-parent selection cannot be replaced or extended')
    else:
        atomic_json(path, snapshot)
    return {'schema': SCHEMA, 'snapshot_sha256': digest(snapshot),
        'source_campaign': snapshot['source_campaign'], 'source_protocol_hash': snapshot['source_protocol_hash'],
        'count': len(snapshot['rows']), 'new_independent_parent_increment': 0}


def validate_parent_sources(campaign: Path, protocol: dict) -> list[dict]:
    """Revalidate source bytes, provenance, grouping and revocations on every resume."""
    frozen = protocol.get('source_parents')
    if not frozen:
        return []
    path = Path(campaign) / 'source_parents.json'
    if not path.is_file():
        raise ArtifactChangedError('Frozen source-parent snapshot is missing')
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    if snapshot.get('schema') != SCHEMA or digest(snapshot) != frozen.get('snapshot_sha256'):
        raise ArtifactChangedError('Frozen source-parent snapshot changed')
    if snapshot.get('source_campaign') != frozen.get('source_campaign') or snapshot.get('source_protocol_hash') != frozen.get('source_protocol_hash'):
        raise ArtifactChangedError('Source-parent campaign provenance changed')
    try:
        current = freeze_parent_sources(Path(snapshot['source_campaign']), protocol)
    except ArtifactChangedError:
        raise
    except (ValueError, OSError, KeyError) as exc:
        raise ArtifactChangedError('Frozen source-parent provenance is no longer valid: ' + str(exc)) from exc
    if current != snapshot:
        raise ArtifactChangedError('The complete preregistered source-parent snapshot changed')
    order = [row['task'] for row in snapshot['rows']]
    config = protocol.get('initialization') or {}
    if config.get('task_order') != order or config.get('max_parents') != len(order) or frozen.get('count') != len(order):
        raise ArtifactChangedError('Replanning must visit the complete fixed source-parent set exactly once')
    return copy.deepcopy(snapshot['rows'])


def source_provenance(row: dict, protocol: dict) -> dict:
    """Metadata only; never splice this result blob into teacher/student input."""
    return {'kind': 'reused_original_proposal_execution', 'source_campaign': protocol['source_parents']['source_campaign'],
        'source_protocol_hash': protocol['source_parents']['source_protocol_hash'],
        'source_snapshot_sha256': protocol['source_parents']['snapshot_sha256'],
        'original_evaluation_id': row['parent_result']['evaluation_id'],
        'original_evaluation_spec_sha256': digest(row['parent_result']['evaluation_spec']),
        'original_result_sha256': row['parent_result']['result_sha256'],
        'original_trajectory_sha256': row['parent_result']['trajectory_sha256'],
        'root_lineage_id': row['root_lineage_id'], 'split': row['split'],
        'decision_protocol_hash': protocol['protocol_hash'], 'new_independent_parent_increment': 0}


def audit_sources_for_export(campaign: Path, protocol: dict, ledger) -> dict | None:
    """Call under the campaign controller lock before every publication path."""
    if not protocol.get('source_parents'):
        return None
    try:
        rows = validate_parent_sources(campaign, protocol)
    except ArtifactChangedError as exc:
        # Source integrity invalidates its derived qualification. Old raw
        # decisions/reviews are preserved as ledger history; no relabeling.
        reason = str(exc)
        for lineage in {sample['root_lineage_id'] for sample in ledger.records('samples')}:
            ledger.revoke(lineage, reason)
        ledger.add_event('source_parent_export_blocked', {'reason': reason,
            'snapshot_sha256': protocol['source_parents']['snapshot_sha256']})
        path = Path(campaign) / 'status.json'
        status = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
        status.update(status='BLOCKED_ARTIFACT_CHANGED', reason=reason)
        atomic_json(path, status)
        return {'status': 'FAIL', 'reason': reason,
            'snapshot_sha256': protocol['source_parents']['snapshot_sha256']}
    return {'status': 'PASS', 'snapshot_sha256': protocol['source_parents']['snapshot_sha256'],
        'reused_parent_executions': len(rows), 'new_independent_parent_increment': 0}
