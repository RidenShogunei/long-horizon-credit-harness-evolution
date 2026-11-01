"""Realize an archived teacher plan without rewriting its original policy input.

This is a historical-plan initialization acceptance run, not a new teacher
diagnosis or a long-horizon experiment. All attempts and outcomes are retained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from research1.collection import (_default_verifier, append_unique,
    freeze_parent, set_admission, valid_task_result)
from research1.config import REPO_ROOT, load_pilot, resolve_path
from research1.contracts import write_json
from research1.evidence import select_trajectory_evidence
from research1.plan_init import frozen_false_concat_checks


def apply_replacements(source: str, edits: list[dict]) -> str:
    """Apply exact nonoverlapping edits against the frozen source, never eval."""
    if not isinstance(edits,list) or not 1 <= len(edits) <= 4:
        raise ValueError('Expected 1-4 exact replacements')
    spans=[]
    for edit in edits:
        old,new=edit.get('old'),edit.get('new')
        if not isinstance(old,str) or not old or not isinstance(new,str) or source.count(old)!=1:
            raise ValueError('Each old block must match the frozen source exactly once')
        start=source.index(old); spans.append((start,start+len(old),new))
    spans.sort()
    if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
        raise ValueError('Overlapping replacements are rejected')
    for start,end,new in reversed(spans): source=source[:start]+new+source[end:]
    compile(source,'fixed_editor_replacement.py','exec')
    return source


def compact_editor(plan: dict,state: dict,directory: Path) -> dict:
    from research1.distill import _apply_edit,_tongji_chat
    snapshot=Path(state['modules_root'])/'tools/baseline.py'
    source=snapshot.read_text()
    scope={k:state['checks'].get(k,[]) for k in ('allowed_changed_functions','allowed_changed_globals','allowed_imports')}
    prompt=('Implement the supplied plan as exact local replacements in the frozen Python file. '
            'Return ONLY JSON {"edits":[{"old":"exact existing block","new":"replacement block"}]}. '
            'Use 1-4 nonoverlapping replacements. Do not return the complete file. Preserve all unrelated code, commands, '
            'durations, errors, completion flags, and non-target diagnostics. No markers or task-answer hardcoding. '
            'IMPORTANT: parser warning may be ONE string containing multiple newline-separated warnings. '
            'Filter each individual warning line independently. Never discard the entire string because its first '
            'matching warning refers to an empty command. A mixed batch must retain every real warning. '
            'Empty keystrokes means exactly the empty string. A space or tab sends input; never strip '
            'whitespace when deciding whether to suppress a warning. Preserve space/tab warnings. '
            'Use only the approved helper/global/import names.\nScope: '+json.dumps(scope)+
            '\nPlan: '+json.dumps(plan)+'\nFrozen source:\n'+source)
    write_json(directory/'editor_input.json',{'prompt':prompt,'policy':'exact_replacements_v1','temperature':0})
    raw,meta=_tongji_chat(prompt,max_tokens=6000,role='acceptance_compact_fixed_editor',
                         json_output=True,reasoning_effort='none')
    write_json(directory/'editor_output.json',{'raw':raw,'metadata':meta})
    if meta.get('temperature')!=0.0: raise ValueError('Editor temperature changed')
    if meta.get('raw_field')!='content' or meta.get('finish_reason')!='stop':
        raise ValueError('Editor requires complete final content; reasoning or truncated outputs are not code')
    text=raw.strip()
    if text.startswith('```'): text='\n'.join(text.splitlines()[1:-1])
    replacement=apply_replacements(source,json.loads(text)['edits'])
    record=directory/'plan_record.json'; write_json(record,{'plan':plan})
    old_plan={'target':'tools/baseline.py','intent':plan['intended_change'],'trigger':None}
    result=_apply_edit(run_dir=directory,record_path=record,plan=old_plan,snapshot=snapshot,replacement=replacement)
    if result.get('status')=='applied': result['modules_root']=str(directory/'versions'/result['child_version_id']/'modules')
    result['editor_metadata']=meta
    return result


def normalize_plan(plan: dict, evidence_id: str) -> dict:
    if plan.get('action') != 'edit' or plan.get('target') != 'tools/baseline.py' or not plan.get('intent'):
        raise ValueError('Only a saved tools edit plan is supported by this acceptance entry')
    return {'action':'edit','target_module':'tools','locations':['tools/baseline.py'],
            'evidence_refs':[evidence_id], 'problem_hypothesis':plan['intent'],
            'intended_change':plan['intent'], 'preserve_behaviors':[plan['intent']],
            'verification_plan':['pre-frozen parser behavior and preservation contract'],
            'verification_kind':'false_concat_preservation'}


def run_acceptance(source: Path, *, root: Path = REPO_ROOT, config_path: Path | None = None,
                   max_attempts: int = 2, wall_seconds: int = 900) -> dict:
    if max_attempts not in (1,2) or not 60 <= wall_seconds <= 3600:
        raise ValueError('Acceptance must be bounded to 1-2 edits and 60-3600 seconds')
    source = source.resolve()
    original = json.loads((source / 'decision.json').read_text())['decision_row']
    prompt = (source / 'teacher_prompt.txt').read_text()
    raw = (source / 'teacher_raw.txt').read_text()
    parsed = json.loads((source / 'plan.json').read_text())
    modules = Path(original['modules_root'])
    if not modules.is_absolute(): modules = root / modules
    trajectory = Path(original['trajectory'])
    if not trajectory.is_absolute(): trajectory = root / trajectory
    config_path = config_path or root / 'configs/research1/pilot.yaml'
    cfg = load_pilot(config_path)
    evidence = select_trajectory_evidence(trajectory)
    bundle = {'id':'false_concat_full_preservation_acceptance_v2','kind':'parser_contract',
              'mechanism_id':'false_concat_preservation','coverage_statement':'Remove only false empty-wait newline warnings, preserve mixed real warnings, commands, errors and completion.',
              'allowed_targets':['tools/baseline.py'],
              'allowed_changed_functions':['BaselineTools.parse_llm_response','_suppress_empty_keystroke_newline_warning','_warning_refers_to_empty_keystrokes'],
              'allowed_changed_globals':['NEWLINE_WARNING_SUPPRESSED_EMPTY_KEYSTROKES','_NEWLINE_WARNING_RE'],
              'allowed_imports':['re','logging','typing'], 'checks':frozen_false_concat_checks()}
    state = freeze_parent(root=root,modules_root=modules,task_name=original['task_name'],evidence=evidence,
        public_config={'config_path':str(config_path),'config_hash':cfg['_config_hash'],
                       'visible_files':['tools/baseline.py'],'max_code_chars':100000,
                       'evidence_selector_id':'archived-teacher-input-recovery',
                       'verification_contract_id':bundle['id'],
                       'teacher_input_sha256':hashlib.sha256(prompt.encode()).hexdigest()},checks=bundle)
    plan = normalize_plan(parsed,evidence[0]['id'])
    run_id = time.strftime('acceptance_%Y%m%d_%H%M%S',time.gmtime())
    run = root / 'runs' / run_id
    run.mkdir(parents=True)
    write_json(run / 'archived_teacher.json',{'source':str(source),'prompt':prompt,'raw':raw,'parsed_plan':parsed,
        'note':'Exact original teacher input/output retained. Full parent snapshot is recovery/verification context, not a rewritten teacher input.'})
    write_json(run / 'protocol.json',{'max_attempts':max_attempts,'wall_seconds':wall_seconds,
        'max_turns':200,'R':1,'new_mechanisms':0,'checks':bundle,'origin_id':state['origin_id'],
        'no_new_teacher_calls':True,'long_horizon_labels':False})
    status = {'run_id':run_id,'status':'RUNNING','phase':'fixed_editor','attempts':[],
              'source_run':str(source),'new_mechanisms':0,'long_horizon_label':None}
    write_json(run / 'status.json',status)
    selected = None
    for i in range(1,max_attempts+1):
        directory = run / f'attempt_{i}'
        directory.mkdir()
        try:
            editor_plan=dict(plan)
            if status['attempts']:
                previous=status['attempts'][-1]['verification']
                feedback=[{'id':c['id'],'ok':c.get('ok'),'before_warning':(c.get('parent_result') or {}).get('warning'),
                           'after_warning':(c.get('child_result') or {}).get('warning')}
                          for c in previous.get('behavior',{}).get('checks',[]) if not c.get('ok')]
                editor_plan['intended_change'] += '\nEditor-only validation feedback (not teacher input): '+json.dumps(feedback)
            attempt = compact_editor(editor_plan,state,directory)
            verification = _default_verifier(plan,state,attempt) if attempt.get('status')=='applied' else {'status':'FAIL','reason':attempt.get('reject_reason','edit not applied')}
        except Exception as exc:
            attempt,verification = {'status':'error','reason':str(exc)}, {'status':'FAIL','reason':type(exc).__name__}
        attempt.update(attempt_id=f'{run_id}:{i}',source_run=str(source))
        write_json(directory / 'attempt.json',attempt)
        write_json(directory / 'verification.json',verification)
        append_unique(root / 'datasets/research1/raw/attempts.jsonl',attempt,'attempt_id')
        status['attempts'].append({'attempt':attempt,'verification':verification})
        write_json(run / 'status.json',status)
        if verification.get('status')=='PASS' and verification.get('covers_plan'):
            selected=(directory,attempt)
            break
    sample_id='archived_repair_'+hashlib.sha256((str(source)+state['origin_id']).encode()).hexdigest()[:24]
    task = {'status':'INCOMPLETE','reward':None,'failure_layer':'infrastructure','note':'no behavior-valid edit'}
    if selected:
        directory,attempt=selected
        status['phase']='real_child_task'
        write_json(run / 'status.json',status)
        from research1.adapters import apply_provider_to_environ, run_harbor_task, with_llm_provider_env
        from research1.evaluation import summarize_trial
        from research1.versions import BASELINE_BUNDLE
        provider=with_llm_provider_env('tongji_agi')
        apply_provider_to_environ(provider)
        output=run / 'real_child_task'
        output.mkdir(parents=True,exist_ok=True)
        timed_out=False
        execution_error=None
        returncode=None
        try:
            proc=run_harbor_task(modularrsi_root=resolve_path(cfg,'modularrsi_root'),modules_root=Path(attempt['modules_root']).resolve(),
                task_name=original['task_name'],output_dir=output,model=provider.model,api_base=provider.api_base,
                temperature=float(cfg.get('p2b_solver_temperature',.7)),dataset_path=resolve_path(cfg,'dataset_tb'),
                max_turns=200,timeout_sec=wall_seconds,locked_module_type='tools',composer_name='static')
            (output / 'harbor.stdout').write_text(proc.stdout or '')
            (output / 'harbor.stderr').write_text(proc.stderr or '')
            returncode=proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out=True
            execution_error={'type':type(exc).__name__,'message':'Real task exceeded the frozen wall limit'}
            for name,value in (('harbor.stdout',exc.stdout),('harbor.stderr',exc.stderr)):
                if isinstance(value,bytes): value=value.decode('utf-8',errors='replace')
                (output / name).write_text(value or '')
        except Exception as exc:
            execution_error={'type':type(exc).__name__,'message':str(exc)}
        if execution_error:
            write_json(output / 'execution_error.json',execution_error)
        task=summarize_trial(run_id=run_id,version_id=attempt['child_version_id'],task_name=original['task_name'],trial_index=1,
            output_dir=output,active_bundle=BASELINE_BUNDLE,model=provider.model,temperature=float(cfg.get('p2b_solver_temperature',.7)),provider_id=provider.provider_id).to_dict()
        task.update(task_run_id=run_id+':child:1',wall_seconds=wall_seconds,wall_limit_reached=timed_out,
                    process_returncode=returncode,execution_error=execution_error)
        write_json(output / 'task_run.json',task)
        append_unique(root / 'datasets/research1/raw/task_runs.jsonl',task,'task_run_id')
    admitted=selected is not None and valid_task_result(task)
    row={'sample_id':sample_id,'run_id':run_id,'state_id':state['state_id'],'origin_id':state['origin_id'],
         'prompt':prompt,'completion':json.dumps(parsed,ensure_ascii=False),'task_name':original['task_name'],
         'use':'directed_behavior_repair_init','source_kind':'archived_teacher_plan_reimplementation',
         'long_horizon_label':None,'source_run':str(source)}
    set_admission(root,sample_id,'verified' if admitted else 'pending',
                  'Archived plan realized by fixed editor, full preservation checks and valid real benchmark execution' if admitted else 'Acceptance incomplete; all artifacts retained',
                  sft_row=row,evidence={'run':str(run),'task_run':task,'verification':str(selected[0] / 'verification.json') if selected else None})
    status.update(status='PASS' if admitted else 'INCOMPLETE',phase='finished',task_run=task,exported_sft=admitted,sample_id=sample_id)
    write_json(run / 'status.json',status)
    write_json(root / 'reports' / (run_id+'.json'),status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--max-attempts',type=int,default=2)
    parser.add_argument('--wall-seconds',type=int,default=900)
    args=parser.parse_args()
    print(json.dumps(run_acceptance(args.source_run,config_path=args.config,max_attempts=args.max_attempts,wall_seconds=args.wall_seconds),indent=2))
