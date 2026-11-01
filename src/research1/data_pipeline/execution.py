"""Bounded Harbor jobs with exact trial binding and transactional retry budgets.

Only proposal-role results may be rendered as policy evidence. Raw results and
scorer assets are retained for auditing but never rendered by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any

from research1.data_pipeline.util import digest


class BindingError(ValueError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def tree_hash(root: Path) -> str:
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file()
             and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if not files:
        raise ValueError('Cannot bind an empty source tree: ' + str(root))
    return digest(files)


def _finite_reward(result: dict) -> float | None:
    value = (result.get('verifier_result') or {}).get('rewards', {}).get('reward')
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        return None
    return float(value)


def bind_trial(output: Path, spec: dict) -> dict:
    """Bind one job and one concrete trial; aggregate job rewards are ignored."""
    jobs = [p for p in output.iterdir() if p.is_dir() and (p / 'config.json').is_file()]
    if len(jobs) != 1:
        raise BindingError('Expected exactly one Harbor job directory')
    job = jobs[0]
    job_config = json.loads((job / 'config.json').read_text(encoding='utf-8'))
    tasks = job_config.get('tasks') or []
    if len(tasks) != 1 or Path(tasks[0].get('path', '')).resolve() != Path(spec['task_path']).resolve():
        raise BindingError('Job task does not match the frozen EvaluationSpec')
    if Path(job_config.get('jobs_dir', '')).resolve() != output.resolve():
        raise BindingError('Job output directory does not match its reservation')
    trials = [p for p in job.iterdir() if p.is_dir() and (p / 'config.json').is_file()]
    if len(trials) != 1:
        raise BindingError('Expected exactly one task trial; ambiguous results cannot be selected')
    trial = trials[0]
    config = json.loads((trial / 'config.json').read_text(encoding='utf-8'))
    agent = config.get('agent') or {}
    kwargs = agent.get('kwargs') or {}
    if (Path((config.get('task') or {}).get('path', '')).resolve() != Path(spec['task_path']).resolve()
            or config.get('trial_name') != trial.name
            or Path(config.get('trials_dir', '')).resolve() != job.resolve()
            or Path(kwargs.get('modules_root', '')).resolve() != Path(spec['modules_root']).resolve()
            or agent.get('model_name') != spec['model']
            or kwargs.get('temperature') != spec['temperature']
            or kwargs.get('max_turns') != spec['max_turns']
            or kwargs.get('composer_name') != 'static'
            or kwargs.get('composer_scope') != 'all'
            or ('reasoning_effort' in spec and kwargs.get('reasoning_effort') != spec['reasoning_effort'])
            or ('model_info' in spec and kwargs.get('model_info') != spec['model_info'])):
        raise BindingError('Trial/config/model/modules/runtime binding mismatch')
    path = trial / 'result.json'
    if not path.is_file():
        raise BindingError('Concrete trial has no result.json')
    result = json.loads(path.read_text(encoding='utf-8'))
    if (result.get('task_name') != spec['task'] or result.get('trial_name') != trial.name
            or result.get('config') != config or not result.get('finished_at')):
        raise BindingError('Concrete result does not belong to the expected completed trial')
    if tree_hash(Path(spec['modules_root'])) != spec['code_hash']:
        raise BindingError('Version tree changed while the task was running')
    if tree_hash(Path(spec['task_path'])) != spec['task_hash']:
        raise BindingError('Task/scorer tree changed while the task was running')
    from research1.evaluation import classify_failure
    exception = result.get('exception_info') or {}
    exception_text = ' '.join(str(exception.get(k, '')) for k in ('exception_type', 'exception_message'))
    reward = _finite_reward(result)
    layer = classify_failure(exception_text, reward)
    if reward is None and layer == 'unknown':
        layer = 'infrastructure'
    usage = result.get('agent_result') or {}
    bound={'status': 'PASS' if reward is not None else 'INCOMPLETE', 'reward': reward,
            'failure_layer': layer, 'task_name': spec['task'], 'model': spec['model'],
            'provider_id': spec['provider_id'], 'temperature': spec['temperature'],
            'trial_name': trial.name, 'trial_result_path': str(path), 'output_dir': str(output),
            'trajectory': str(trial / 'agent/trajectory.json') if (trial / 'agent/trajectory.json').is_file() else None,
            'trajectory_sha256': hashlib.sha256((trial / 'agent/trajectory.json').read_bytes()).hexdigest() if (trial / 'agent/trajectory.json').is_file() else None,
            'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'exception_type': exception.get('exception_type'),
            'fatal': bool(re.search(r'Authentication|HTTP\s*(401|403)|unauthorized|invalid.api.key', exception_text, re.I)),
            'usage': {k: usage.get(k) for k in ('n_input_tokens', 'n_cache_tokens', 'n_output_tokens', 'cost_usd')}}
    trajectory=trial/'agent/trajectory.json'
    if trajectory.exists():
        data=json.loads(trajectory.read_text(encoding='utf-8'))
        steps=[s for s in data.get('steps',[]) if s.get('source')=='agent']
        characters=sum(len(s.get('reasoning_content') or '') for s in steps)
        bound['solver_reasoning_observation']={'agent_steps':len(steps),'reasoning_characters':characters,
            'requested_effort':spec.get('reasoning_effort'),'content_disclosed':False}
        if spec.get('reasoning_effort')=='none' and characters:
            bound.update(status='INCOMPLETE',fatal=True,retryable=False,policy_valid=False,
                         error='Solver emitted reasoning despite the frozen nonthinking request; do not relabel or retry another task')
    return bound


def _birth_token(pid: int) -> str | None:
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _owned_process(marker: dict) -> bool:
    pid = marker.get('pid')
    if not isinstance(pid, int) or _birth_token(pid) != marker.get('birth_token'):
        return False
    try:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        return b'harbor.cli.main' in args and str(marker['output_dir']).encode() in args
    except OSError:
        return False


def cleanup_owned(output: Path, marker: dict | None = None) -> dict:
    """Stop only a verified process group and the exact trial container names."""
    stopped = []
    if marker and _owned_process(marker):
        pid = marker['pid']
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGTERM)
            stopped.append(pid)
    names = set()
    for config_path in output.glob('*/*/config.json'):
        try:
            config = json.loads(config_path.read_text())
            if config.get('trial_name') == config_path.parent.name:
                names.add(config['trial_name'].lower() + '-main-1')
        except (OSError, ValueError, KeyError):
            pass
    containers = []
    try:
        listing = subprocess.run(['docker', 'ps', '--format', '{{.ID}} {{.Names}}'],
                                 capture_output=True, text=True, timeout=30, check=True)
        for line in listing.stdout.splitlines():
            cid, name = line.split()
            if name in names:
                proc = subprocess.run(['docker', 'stop', '--time', '5', cid],
                                      capture_output=True, text=True, timeout=30)
                containers.append({'id': cid, 'name': name, 'returncode': proc.returncode})
    except (OSError, subprocess.SubprocessError):
        pass
    return {'process_groups': stopped, 'containers': containers}


def _image_identity(task_path: Path) -> dict:
    dockerfile = task_path / 'environment/Dockerfile'
    text = dockerfile.read_text() if dockerfile.exists() else ''
    matches = re.findall(r'^FROM\s+(?:--platform=\S+\s+)?(\S+)', text, re.M | re.I)
    ids = {}
    for image in ['hb__' + task_path.name, *matches]:
        try:
            proc = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                                  capture_output=True, text=True, timeout=30)
            ids[image] = proc.stdout.strip() if proc.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            ids[image] = None
    return {'dockerfile_sha256': hashlib.sha256(text.encode()).hexdigest(), 'image_ids': ids}


def _bind_assets(ledger: Any, protocol: dict, task: str, task_hash: str, image: dict) -> dict:
    """One task/scorer/image identity across every version and path in a campaign."""
    expected=protocol.get('task_asset_hashes',{}).get(task)
    if expected and expected!=task_hash:
        raise BindingError('Task/scorer assets differ from protocol preparation: '+task)
    existing={row['task']:row for row in ledger.records('task_assets')}.get(task)
    identity={'task':task,'task_hash':task_hash,'image':image}
    if existing:
        if existing['task_hash']!=task_hash:
            raise BindingError('Task/scorer assets changed between scopes: '+task)
        known=existing['image'].get('image_ids',{})
        actual=image.get('image_ids',{})
        if existing['image'].get('dockerfile_sha256')!=image.get('dockerfile_sha256'):
            raise BindingError('Environment definition changed between scopes: '+task)
        if any(value is not None and actual.get(name)!=value for name,value in known.items()):
            raise BindingError('Bound Docker image changed or disappeared: '+task)
        # A first build/pull may resolve previously unknown IDs exactly once.
        identity['image']=dict(image,image_ids={**known,**actual})
    ledger.put_record('task_assets',task,identity)
    return identity


def _command(protocol: dict, spec: dict, output: Path, provider: Any) -> tuple[list, dict, Path]:
    from research1.adapters import harbor_python, proxy_url, merge_no_proxy
    from research1.config import load_pilot, resolve_path
    cfg = load_pilot(Path(protocol['config_path']))
    if protocol.get('config_hash') and cfg['_config_hash'] != protocol['config_hash']:
        raise ValueError('Execution configuration changed after protocol freeze')
    upstream = resolve_path(cfg, 'modularrsi_root')
    if protocol.get('upstream_root') and upstream.resolve()!=Path(protocol['upstream_root']).resolve():
        raise ValueError('Harbor root differs from the frozen protocol')
    py = harbor_python(upstream)
    model_info = json.dumps(spec['model_info'],sort_keys=True,separators=(',',':'))
    cmd = [str(py), '-m', 'harbor.cli.main', 'run', '--yes', '-a', 'terminus-2-modular',
           '-m', spec['model'], '-e', 'docker', '-n', '1', '-k', '1', '-o', str(output),
           '-p', spec['task_path']]
    for key, value in [('modules_root', spec['modules_root']), ('api_base', provider.api_base),
                       ('max_turns', spec['max_turns']), ('model_info', model_info),
                       ('temperature', spec['temperature']),
                       ('composer_scope', 'all'), ('composer_name', 'static')]:
        cmd.extend(['--ak', f'{key}={value}'])
    if spec.get('reasoning_effort') is not None:
        cmd.extend(['--ak','reasoning_effort='+str(spec['reasoning_effort'])])
    px = proxy_url()
    no_proxy = merge_no_proxy(os.environ.get('NO_PROXY') or os.environ.get('no_proxy'))
    for flag in ('--ve', '--ae'):
        for key, value in [('HTTP_PROXY', px), ('HTTPS_PROXY', px), ('http_proxy', px),
                           ('https_proxy', px), ('NO_PROXY', no_proxy), ('no_proxy', no_proxy)]:
            cmd.extend([flag, f'{key}={value}'])
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', NO_PROXY=no_proxy, no_proxy=no_proxy)
    return cmd, env, upstream


def _incomplete(spec: dict, output: Path, reason: str, *, fatal: bool = False,
                in_progress: bool = False) -> dict:
    return {'status': 'INCOMPLETE', 'reward': None, 'failure_layer': 'infrastructure',
            'task_name': spec['task'], 'output_dir': str(output), 'error': reason,
            'fatal': fatal, 'in_progress': in_progress}


def evaluate(campaign: Path, protocol: dict, ledger: Any, modules: Path, task: str,
             role: str, scope: str, repeat: int = 1) -> dict:
    """Run or resume a single R slot with at most two infrastructure retries."""
    from research1.adapters import apply_provider_to_environ, with_llm_provider_env
    from research1.config import load_pilot, resolve_path
    modules = Path(modules).resolve()
    cfg = load_pilot(Path(protocol['config_path']))
    if protocol.get('config_hash') and cfg['_config_hash'] != protocol['config_hash']:
        raise ValueError('Execution configuration changed after protocol freeze')
    task_path = (resolve_path(cfg, 'dataset_tb') / 'tasks' / task).resolve()
    if protocol.get('dataset_path') and resolve_path(cfg,'dataset_tb').resolve()!=Path(protocol['dataset_path']).resolve():
        raise ValueError('Dataset root differs from the frozen protocol')
    if protocol.get('upstream_root') and resolve_path(cfg,'modularrsi_root').resolve()!=Path(protocol['upstream_root']).resolve():
        raise ValueError('Harbor root differs from the frozen protocol')
    if protocol.get('runtime_source_hash') and tree_hash(resolve_path(cfg,'modularrsi_root')/'src/harbor')!=protocol['runtime_source_hash']:
        raise ValueError('Harbor runtime source differs from the frozen protocol')
    try:
        assets=_bind_assets(ledger,protocol,task,tree_hash(task_path),_image_identity(task_path))
    except BindingError as exc:
        for binding in ledger.records('evaluation_specs'):
            if binding['spec'].get('task')==task:
                ledger.revoke(digest(binding['spec']),str(exc))
        raise
    provider = with_llm_provider_env('tongji_agi')
    apply_provider_to_environ(provider)
    runtime = protocol['runtime']
    solver = protocol.get('policies', {}).get('solver', {})
    model = solver.get('model', provider.model)
    if model != provider.model:
        raise ValueError('Resolved solver model differs from the frozen policy')
    if solver.get('provider') and solver['provider']!=provider.provider_id:
        raise ValueError('Resolved solver provider differs from the frozen policy')
    if solver.get('api_base') and solver['api_base'].rstrip('/')!=provider.api_base.rstrip('/'):
        raise ValueError('Resolved solver endpoint differs from the frozen policy')
    spec = {'schema': 'research1.evaluation.v2', 'protocol_hash': protocol.get('protocol_hash', digest(protocol)),
            'modules_root': str(modules), 'code_hash': tree_hash(modules),
            'task': task, 'task_path': str(task_path), 'task_hash': assets['task_hash'],
            'image': assets['image'], 'model': model, 'provider_id': provider.provider_id,
            'temperature': solver.get('temperature', .7), 'max_turns': runtime['max_turns'],
            'model_info': {'max_input_tokens':176000,'max_output_tokens':16000,
                           'input_cost_per_token':0.0,'output_cost_per_token':0.0},
            'composer': {'name':'static','scope':'all'},
            'wall_seconds': runtime['wall_seconds'], 'role': role, 'scope': scope, 'repeat': repeat,
            'provider_seed_applied': False, 'solver_randomness': 'independent_provider_request'}
    if solver.get('reasoning_effort') is not None:spec['reasoning_effort']=solver['reasoning_effort']
    # A declared slot keeps the original asset binding across crash/resume,
    # including a task image that was built during the first attempt.
    request = digest({k: spec[k] for k in spec if k != 'image'})
    prior_specs = {row['request_id']: row['spec'] for row in ledger.records('evaluation_specs')}
    if request in prior_specs:
        spec = prior_specs[request]
    else:
        ledger.put_record('evaluation_specs', request, {'request_id': request, 'spec': spec})
    logical = digest(spec)
    final_result = None
    for number in range(3):
        key = logical + ':' + str(number)
        output = Path(campaign) / 'executions' / logical / f'attempt_{number}'
        existing = ledger.get_job(key)
        if existing and existing['status'] in {'COMPLETE', 'INCOMPLETE'}:
            result = existing['result']
            if existing['status']=='COMPLETE':
                try:
                    raw=Path(result['trial_result_path'])
                    if hashlib.sha256(raw.read_bytes()).hexdigest()!=result['result_sha256']:
                        raise BindingError('Committed concrete result changed')
                    if result.get('trajectory') and hashlib.sha256(Path(result['trajectory']).read_bytes()).hexdigest()!=result.get('trajectory_sha256'):
                        raise BindingError('Committed proposal trajectory changed')
                    bind_trial(output,spec)
                except (OSError,ValueError,KeyError) as exc:
                    ledger.revoke(logical,str(exc))
                    return _incomplete(spec,output,'Committed result artifacts are no longer valid: '+str(exc),fatal=True)
        else:
            if existing is None:
                if shutil.disk_usage(campaign).free < runtime['disk_floor_gb'] * 1024**3:
                    raise RuntimeError('Disk reserve below the frozen floor')
                if number == 0 and role in protocol.get('stage_budget', {}):
                    from research1.data_pipeline.ledger import BudgetExceeded
                    with ledger.connection() as conn:
                        used = sum(json.loads(row['spec']).get('role') == role for row in
                                   conn.execute("SELECT spec FROM jobs WHERE kind='regular'"))
                    if used >= protocol['stage_budget'][role]:
                        raise BudgetExceeded('Frozen stage budget exhausted: ' + role)
                ledger.reserve_job(key, spec, 'regular' if number == 0 else 'retry')
                existing = ledger.get_job(key)
            output.mkdir(parents=True, exist_ok=True)
            _write(output / 'evaluation_spec.json', spec)
            marker_path = output / 'launch.json'
            marker = json.loads(marker_path.read_text()) if marker_path.is_file() else None
            wall_limit_reached = bool(marker and marker.get('wall_limit_reached'))
            if existing['status'] == 'RUNNING':
                if marker and _owned_process(marker):
                    if time.time() - marker['started_unix'] < spec['wall_seconds']:
                        return _incomplete(spec, output, 'Existing Harbor job is still running', in_progress=True)
                    wall_limit_reached = True
                    marker.update(wall_limit_reached=True,stop_reason='operational_wall_limit')
                    _write(marker_path,marker)
                    cleanup_owned(output, marker)
                try:
                    result = bind_trial(output, spec)
                except (BindingError, OSError, ValueError) as exc:
                    # A controller killed between Popen and launch-marker fsync
                    # cannot prove whether an untracked child is still alive.
                    result = _incomplete(spec, output, 'Interrupted task: ' + str(exc), fatal=marker is None)
            else:
                ledger.start_job(key, str(os.getpid()))
                cmd, env, cwd = _command(protocol, spec, output, provider)
                started = time.time()
                with (output / 'harbor.stdout').open('w') as stdout, (output / 'harbor.stderr').open('w') as stderr:
                    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                            stdout=stdout, stderr=stderr, start_new_session=True)
                    marker = {'pid': proc.pid, 'birth_token': _birth_token(proc.pid),
                              'output_dir': str(output), 'started_unix': started}
                    _write(marker_path, marker)
                    try:
                        returncode = proc.wait(timeout=spec['wall_seconds'])
                    except subprocess.TimeoutExpired:
                        # This is the frozen operational budget, not a transient
                        # infrastructure failure eligible for another full run.
                        wall_limit_reached = True
                        marker.update(wall_limit_reached=True,stop_reason='operational_wall_limit')
                        _write(marker_path,marker)
                        cleanup = cleanup_owned(output, marker)
                        try:
                            returncode = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if _owned_process(marker):
                                os.killpg(proc.pid, signal.SIGKILL)
                            returncode = proc.wait(timeout=10)
                        _write(output / 'cleanup.json', cleanup)
                try:
                    result = bind_trial(output, spec)
                except (BindingError, OSError, ValueError) as exc:
                    result = _incomplete(spec, output, str(exc))
                    logs = '\n'.join(p.read_text(errors='replace')[-20000:]
                                     for p in (output / 'harbor.stdout', output / 'harbor.stderr') if p.exists())
                    result['fatal'] = bool(re.search(r'Authentication|HTTP\s*(401|403)|unauthorized|invalid.api.key', logs, re.I))
                result.update(process_returncode=returncode, duration_seconds=time.time()-started)
            if result.get('reward') is not None:
                try:
                    bound_assets=_bind_assets(ledger,protocol,task,tree_hash(task_path),_image_identity(task_path))
                    result['bound_assets']=bound_assets
                except BindingError as exc:
                    result=_incomplete(spec,output,str(exc),fatal=True)
            result.update(job_key=key, evaluation_id=logical, evaluation_spec=spec,
                          attempt_index=number, role=role, scope=scope, repeat=repeat)
            result['wall_limit_reached']=wall_limit_reached
            if wall_limit_reached:
                result.update(retryable=False,stopped_for_budget=True,stop_reason='operational_wall_limit')
            _write(output / 'task_run.json', result)
            if result.get('reward') is not None:
                ledger.complete_job(key, result)
            else:
                ledger.fail_job(key, result)
        final_result = result
        if (result.get('reward') is not None or result.get('fatal') or result.get('in_progress')
                or result.get('retryable') is False or result.get('wall_limit_reached')
                or result.get('failure_layer') not in {'infrastructure', 'container', 'scorer'}
                or re.search(r'AgentTimeout|OutputLengthExceeded|ContextWindowExceeded', result.get('exception_type') or '', re.I)):
            return result
    return final_result
