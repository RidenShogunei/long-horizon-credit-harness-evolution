"""Prepare a frozen, credential-free research data campaign without model calls."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from research1.config import load_pilot,resolve_path,train_task_names
from research1.data_pipeline import PROTOCOL_ID
from research1.data_pipeline.util import atomic_json,digest,tree_hash

MODULES=('agent_loop','observation','tools','context_mgmt','verification')
V2_REGISTRY='research1.module-contracts.v2'
V3_REGISTRY='research1.module-contracts.v3'


def _validate_registry_options(version, diagnostic_selection, *, teacher_replay_from=None,
        fact_replay_from=None, fact_replay_state=None, teacher_max_tokens=8000,
        admission_review_policy=None, applicability_fixtures=False, publish_probe_io=False,
        replay_factual_from_source=False, applicability_review_policy=None, plan_precision_policy=None,
        independent_plan_audit_policy=None):
    if version not in {None,V2_REGISTRY,V3_REGISTRY}:
        raise ValueError('Unknown verification registry version')
    fresh_v3=version==V3_REGISTRY and bool(diagnostic_selection) and teacher_replay_from is None
    if not isinstance(replay_factual_from_source,bool):
        raise ValueError('Saved factual replay must be an explicit boolean opt-in')
    saved_v3=(version==V3_REGISTRY and bool(diagnostic_selection)
        and teacher_replay_from is not None and replay_factual_from_source)
    if replay_factual_from_source and not saved_v3:
        raise ValueError('Saved factual replay requires an original same-v3 diagnostic teacher source')
    if saved_v3 and (admission_review_policy!='proposal-factual-review.v8.probe_io'
            or applicability_review_policy!='proposal-plan-applicability.v3.fact_ids'
            or not applicability_fixtures or publish_probe_io):
        raise ValueError('Same-v3 replay requires original factual8, independent applicability3, original fixtures and no publication')
    fresh_app3=(fresh_v3 and publish_probe_io and applicability_fixtures
        and admission_review_policy=='proposal-factual-review.v8.probe_io'
        and applicability_review_policy=='proposal-plan-applicability.v3.fact_ids')
    if applicability_review_policy is not None and not (saved_v3 or fresh_app3):
        raise ValueError('Applicability policy selection requires explicit same-v3 replay or fresh published-v3 factual8 fixtures')
    if plan_precision_policy is not None:
        if plan_precision_policy!='proposal-plan-precision.v1.field_audit' or not fresh_app3 or replay_factual_from_source:
            raise ValueError('Plan precision audit requires fresh published-v3 factual8, original fixtures and applicability3; saved plans cannot be upgraded')
    if independent_plan_audit_policy is not None:
        if independent_plan_audit_policy!='research1.independent-plan-audit.v1' or not fresh_app3 or replay_factual_from_source:
            raise ValueError('Independent plan audit requires fresh published-v3 factual8, original fixtures, applicability3 and plan precision; saved plans cannot be upgraded')
        if plan_precision_policy!='proposal-plan-precision.v1.field_audit':
            raise ValueError('Independent plan audit requires explicit plan precision field audit on the same fresh profile')
    if not isinstance(publish_probe_io,bool):
        raise ValueError('Parent probe publication must be an explicit boolean opt-in')
    if publish_probe_io and not fresh_v3:
        raise ValueError('Parent probe publication requires a fresh fixed-source v3 diagnostic trial')
    if admission_review_policy=='proposal-factual-review.v8.probe_io' and not (publish_probe_io and fresh_v3 or saved_v3):
        raise ValueError('Probe-I/O factual review v8 requires published parent I/O in a fresh fixed-source v3 diagnostic trial')
    if version==V3_REGISTRY:
        if not (fresh_v3 or saved_v3):
            raise ValueError('Verification registry v3 requires a fresh fixed-source diagnostic trial; old teacher plans cannot be upgraded')
        if fact_replay_from is not None or fact_replay_state is not None:
            raise ValueError('Fresh registry v3 diagnosis requires a new bounded fact selection, not quote replay')
        if teacher_max_tokens!=16000:
            raise ValueError('Fresh registry v3 diagnosis requires the frozen 16000-token teacher setting')
    if (admission_review_policy is not None or applicability_fixtures) and teacher_replay_from is None and not fresh_v3:
        raise ValueError('Reviewer options require saved original teacher plans or a fresh fixed-source v3 diagnostic trial')
    return V3_REGISTRY if fresh_v3 or saved_v3 else V2_REGISTRY

def exposed_tasks(root:Path,tasks:list[str])->set[str]:
    """Conservative exposure inventory; never inspect task tests or solutions."""
    exposed=set();runs=root/'runs'
    for path in runs.rglob('task_run.json'):
        try:
            row=json.loads(path.read_text(encoding='utf-8'));name=row.get('task_name')
            if name in tasks:exposed.add(name)
        except (ValueError,OSError):continue
    for path in runs.rglob('trajectory.json'):
        parts=set(path.parts)
        for task in tasks:
            if task in parts or any(p.startswith(task[:30]+'__') for p in parts):exposed.add(task)
    return exposed

def partition_tasks(tasks:list[str],exposed:set[str],*,seed:int,ready:dict[str,dict]|None=None)->dict:
    if len(set(tasks))!=len(tasks):raise ValueError('Duplicate task names in manifest train split')
    ready=ready or {}
    def order(name):
        info=ready.get(name,{})
        return (not info.get('cached',False),info.get('heavy_build',False),hashlib.sha256(f'{seed}:{name}'.encode()).hexdigest())
    fresh=sorted(set(tasks)-exposed,key=order)
    if len(fresh)<72:raise ValueError('Need 72 unexposed train tasks for proposal/reward/dev; do not substitute pool or previously exposed tasks')
    proposal,reward,dev=fresh[:24],fresh[24:48],fresh[48:72]
    debug_candidates=sorted(set(tasks)-set(proposal)-set(reward)-set(dev),key=order)
    if len(debug_candidates)<8:raise ValueError('Need eight disjoint debug tasks')
    debug=debug_candidates[:8]
    result={'debug':debug,'proposal':proposal,'reward':reward,'dev':dev,
            'proposal_train':proposal[:19],'proposal_validation':proposal[19:],
            'reserve':sorted(set(tasks)-set(debug)-set(proposal)-set(reward)-set(dev))}
    validate_partitions(result,tasks,exposed)
    return result

def validate_partitions(sets:dict,tasks:list[str],exposed:set[str]|None=None):
    used=set()
    for role,size in [('debug',8),('proposal',24),('reward',24),('dev',24)]:
        names=sets.get(role,[])
        if len(names)!=size or len(set(names))!=size or not set(names)<=set(tasks):raise ValueError('Invalid '+role+' task partition')
        if used&set(names):raise ValueError('Overlapping task roles')
        used.update(names)
    if set(sets.get('proposal_train',[]))&set(sets.get('proposal_validation',[])):raise ValueError('Proposal split overlap')
    if set(sets.get('proposal_train',[]))|set(sets.get('proposal_validation',[]))!=set(sets['proposal']):raise ValueError('Proposal folds incomplete')
    if exposed and (set(sets['proposal'])|set(sets['reward'])|set(sets['dev']))&exposed:raise ValueError('Historical exposure in new formal task partition')

def image_inventory(dataset:Path,tasks:list[str],cached:set[str])->dict:
    inventory={}
    for task in tasks:
        environment=dataset/'tasks'/task/'environment'
        dockerfile=environment/'Dockerfile'
        text=dockerfile.read_text(encoding='utf-8') if dockerfile.is_file() else ''
        match=re.search(r'^FROM\s+(?:--[^\s]+\s+)?([^\s]+)',text,re.M|re.I)
        base=match.group(1) if match else None
        inventory[task]={'base_image':base,'cached':base in cached,
            'full_image_cached':('hb__'+task+':latest') in cached or ('hb__'+task) in cached,
            'heavy_build':bool(re.search(r'build-essential|imagemagick|ffmpeg|git clone',text,re.I)),
            'dockerfile_sha256':hashlib.sha256(text.encode()).hexdigest()}
    return inventory

def _file_digests(root:Path)->dict:
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}

def handover_budget(previous:Path,successor:Path)->dict:
    """One successor inherits unspent authorization; a restart never resets caps."""
    from research1.data_pipeline.ledger import Ledger
    previous,successor=Path(previous).resolve(),Path(successor).resolve()
    if previous==successor:raise ValueError('Supersession requires a new immutable campaign')
    old=load_protocol(previous)
    status=json.loads((previous/'status.json').read_text())
    if status.get('status') not in {'INTERRUPTED_SUPERSEDED','INCOMPLETE','INSUFFICIENT_EVIDENCE','INIT_BATCH_COMPLETE','INIT_BATCH_PARTIAL',
                                  'GROUNDED_EDIT_TRIAL_COMPLETE','GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT'}:
        raise ValueError('Only a terminal stopped campaign can hand over budget')
    ledger=Ledger(previous)
    with ledger.controller():
        with ledger.connection() as conn:
            if conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('READY','RUNNING')").fetchone()[0]:
                raise ValueError('Previous campaign has unfinished reserved/running jobs')
        prior=old.get('budget_lineage') or {}
        authorized=prior.get('authorized',old['budget'])
        carried=prior.get('carried_spend',{'regular':0,'retries':0,'total':0})
        own=ledger.status()['budget']['used']
        spent={'regular':carried['regular']+own.get('regular',0),
               'retries':carried['retries']+own.get('retry',0)}
        spent['total']=spent['regular']+spent['retries']
        remaining={key:authorized[key]-spent[key] for key in ('regular','retries','total')}
        if remaining['regular']<1 or remaining['total']<1 or remaining['retries']<0:raise ValueError('No authorized budget remains')
        result={'authorized':authorized,'carried_spend':spent,'remaining':remaining,
                'previous_campaign':str(previous),'previous_protocol_hash':old['protocol_hash'],'successor_campaign':str(successor)}
        existing=dict(ledger.record_items('budget_handover')).get('successor')
        if existing is not None and existing!=result:raise ValueError('Previous budget already handed to a different successor')
        ledger.put_record('budget_handover','successor',result)
        if status.get('status')!='INTERRUPTED_SUPERSEDED':
            atomic_json(previous/'status.before_budget_handover.json',status)
            status.update(status='INTERRUPTED_SUPERSEDED',superseded_from_status=status.get('status'),superseded_by=str(successor))
            atomic_json(previous/'status.json',status)
    return result

def initialization_queue(sets:dict,inventory:dict,readiness:dict,*,seed:int,max_parents:int=6,
                         excluded_tasks:set[str]|None=None,task_order:list[str]|None=None)->list[str]:
    if isinstance(max_parents,bool) or not isinstance(max_parents,int) or not 1<=max_parents<=24:raise ValueError('Initialization max_parents must be 1..24')
    excluded_tasks=set(excluded_tasks or ())
    def order(task):
        history=readiness.get(task,{})
        status=history.get('status',history.get('classification','unknown'))
        known=status in {'completed_compatible','compatible'}
        image=inventory.get(task,{})
        return (not known,not image.get('full_image_cached',False),not image.get('cached',False),
                image.get('heavy_build',False),hashlib.sha256(f'{seed}:init:{task}'.encode()).hexdigest())
    incompatible={'known_budget_incompatible','budget_incompatible','env_unavailable'}
    def available(fold):
        eligible=[]
        for task in sets[fold]:
            if task in excluded_tasks:continue
            history=readiness.get(task,{})
            status=history.get('status',history.get('classification','unknown'))
            image=inventory.get(task,{})
            if status in incompatible or history.get('initialization_eligible') is False:continue
            if not (status in {'completed_compatible','compatible'} or image.get('full_image_cached')
                    or (image.get('cached') and not image.get('heavy_build'))):continue
            eligible.append(task)
        return sorted(eligible,key=order)
    pools={'train':available('proposal_train'),'validation':available('proposal_validation')}
    if task_order is not None:
        if (not isinstance(task_order,list) or not task_order or len(task_order)>max_parents
                or any(not isinstance(task,str) for task in task_order) or len(set(task_order))!=len(task_order)):
            raise ValueError('Explicit parent queue must be unique and fit max_parents')
        if not set(task_order)<=set(pools['train']+pools['validation']):
            raise ValueError('Explicit parent queue contains exposed, unavailable or non-proposal tasks')
        return list(task_order)
    result=[]
    for index in range(max_parents):
        fold='validation' if (index+1)%5==0 else 'train'
        if pools[fold]:result.append(pools[fold].pop(0))
    return result

def runtime_debug_partition(sets:dict,tasks:list[str],readiness:dict,*,seed:int)->dict:
    excluded=set(sets['proposal'])|set(sets['reward'])|set(sets['dev'])
    candidates=[t for t in tasks if t not in excluded and readiness.get(t,{}).get('noise_eligible') is True]
    candidates.sort(key=lambda t:hashlib.sha256(f'{seed}:noise:{t}'.encode()).hexdigest())
    if len(candidates)<8:
        raise ValueError('Noise calibration needs eight runtime-certified debug tasks; collect bounded initialization/profiling first instead of retrying known over-budget tasks')
    result=json.loads(json.dumps(sets));result['debug']=candidates[:8]
    result['reserve']=sorted(set(tasks)-excluded-set(result['debug']))
    validate_partitions(result,tasks)
    return result

def prepare(campaign:Path,config_path:Path,*,root:Path|None=None,seed:int=20261101,
            cached_images:set[str]|None=None,supersedes:Path|None=None,mode:str='calibration',max_parents:int=6,
            parent_tasks:list[str]|None=None,replan_from:Path|None=None,
            diagnostic_tasks:list[str]|None=None,diagnostic_rationale:str|None=None,
            fact_replay_from:Path|None=None,fact_replay_state:Path|None=None,
            compact_fact_index:bool=False,teacher_max_tokens:int=8000,submitted_code_index:bool=False,
            teacher_replay_from:Path|None=None,admission_review_policy:str|None=None,
            applicability_fixtures:bool=False,verification_registry_version:str|None=None,
            publish_probe_io:bool=False,replay_factual_from_source:bool=False,
            applicability_review_policy:str|None=None,plan_precision_policy:str|None=None,
            independent_plan_audit_policy:str|None=None)->dict:
    from research1.config import REPO_ROOT
    root=Path(root or REPO_ROOT).resolve();campaign=Path(campaign).resolve()
    if mode not in {'calibration','initialization'}:raise ValueError('Unknown execution mode')
    if parent_tasks is not None and mode!='initialization':raise ValueError('Explicit parent queue requires initialization mode')
    diagnostic_selection=None
    fact_source=None
    teacher_source=None
    fixture_source=None
    if admission_review_policy not in {None,'proposal-factual-review.v6.typed','proposal-factual-review.v7.selectors','proposal-factual-review.v8.probe_io'}:
        raise ValueError('Unknown saved-plan admission reviewer')
    if applicability_review_policy not in {None,'proposal-plan-applicability.v3.fact_ids'}:
        raise ValueError('Unknown saved-plan applicability reviewer')
    if teacher_max_tokens not in {8000,16000} or isinstance(teacher_max_tokens,bool):
        raise ValueError('Teacher output limit must be one of the bounded supported settings')
    if diagnostic_tasks is not None or diagnostic_rationale is not None:
        if (not replan_from or mode!='initialization' or not diagnostic_tasks or not diagnostic_rationale
            or len(diagnostic_tasks)!=1 or len(set(diagnostic_tasks))!=1
            or len(diagnostic_rationale.strip())<30):
            raise ValueError('Grounded edit trial requires exactly one predeclared source task, a concrete rationale and a source initialization campaign')
        diagnostic_selection={'tasks':list(diagnostic_tasks),'rationale':diagnostic_rationale,'frozen_before_candidates':True}
    registry_version=_validate_registry_options(verification_registry_version,diagnostic_selection,
        teacher_replay_from=teacher_replay_from,fact_replay_from=fact_replay_from,
        fact_replay_state=fact_replay_state,teacher_max_tokens=teacher_max_tokens,
        admission_review_policy=admission_review_policy,applicability_fixtures=applicability_fixtures,
        publish_probe_io=publish_probe_io,replay_factual_from_source=replay_factual_from_source,
        applicability_review_policy=applicability_review_policy,plan_precision_policy=plan_precision_policy,
        independent_plan_audit_policy=independent_plan_audit_policy)
    if fact_replay_from is not None or fact_replay_state is not None:
        if not diagnostic_selection or not fact_replay_from or not fact_replay_state:
            raise ValueError('Quote replay requires the explicitly frozen diagnostic trial and old state')
        directory=Path(fact_replay_from).resolve();old_state=Path(fact_replay_state).resolve()
        artifacts={}
        for label,path in [('input',directory/'fact_input.json'),('output',directory/'fact_output.json'),('state',old_state)]:
            if not path.is_file() or path.is_symlink() or not path.is_relative_to((root/'runs').resolve()):
                raise ValueError('Fact replay sources must be existing regular run artifacts')
            artifacts[label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        fact_source={'kind':'saved_quote_replay','directory':str(directory),'old_state_path':str(old_state),'artifacts':artifacts}
    if (compact_fact_index or teacher_max_tokens!=8000 or submitted_code_index) and not diagnostic_selection:
        raise ValueError('Indexed diagnosis settings require a predeclared diagnostic trial')
    if teacher_replay_from is not None:
        if not diagnostic_selection:raise ValueError('Saved-plan admission needs a declared diagnostic source')
        source_campaign=Path(teacher_replay_from).resolve()
        if not source_campaign.is_relative_to((root/'runs').resolve()):raise ValueError('Saved teacher source must be a project run')
        source_doc=load_protocol(source_campaign)
        if source_doc.get('verification_registry_version',V2_REGISTRY)!=registry_version:
            raise ValueError('Saved teacher replay cannot change its original verification registry version')
        source_status=json.loads((source_campaign/'status.json').read_text())
        if source_status.get('status')=='INTERRUPTED_SUPERSEDED':
            before_path=source_campaign/'status.before_budget_handover.json'
            if not before_path.is_file() or before_path.is_symlink():
                raise ValueError('Superseded teacher source needs its original terminal status record')
            before=json.loads(before_path.read_text())
            expected=dict(before,status='INTERRUPTED_SUPERSEDED',superseded_from_status=before.get('status'),
                superseded_by=source_status.get('superseded_by'))
            if source_status!=expected or not source_status.get('superseded_by'):
                raise ValueError('Original teacher terminal status changed after budget handover')
            source_status=before
        if source_status.get('status') not in {'GROUNDED_EDIT_TRIAL_COMPLETE','GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT'}:
            raise ValueError('Original teacher trial must finish before saved-plan admission')
        if source_doc.get('grounded_edit_trial',{}).get('teacher_source'):
            raise ValueError('Reuse the original teacher generation trial, not another admission replay')
        arms={};artifacts={}
        for arm in ('standard','reasoned'):
            teacher_path=source_campaign/'decisions'/'edittrial'/arm/'teacher.json'
            state_paths=[]
            for path in (source_campaign/'states').glob('*/state.json'):
                if json.loads(path.read_text()).get('teacher_arm_id')==arm:state_paths.append(path)
            if len(state_paths)!=1 or not teacher_path.is_file():raise ValueError('Saved teacher arm is missing or ambiguous')
            state_path=state_paths[0]
            arms[arm]={'teacher_path':str(teacher_path),'state_path':str(state_path)}
            for label,path in ((arm+'_teacher',teacher_path),(arm+'_state',state_path)):
                artifacts[label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        contract_path=source_campaign/'edittrial_preteacher_contract.json'
        if not contract_path.is_file():raise ValueError('Original preteacher contract is missing')
        artifacts['contract']={'path':str(contract_path),'sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest()}
        teacher_source={'campaign':str(source_campaign),'arms':arms,'contract_path':str(contract_path),'artifacts':artifacts}
        if replay_factual_from_source:
            from research1.data_pipeline.teacher_replay import freeze_factual_replay_source
            teacher_source=freeze_factual_replay_source(teacher_source)
            originals=[json.loads(Path(value['state_path']).read_bytes()) for value in arms.values()]
            fixture_source=originals[0].get('applicability_fixture_source')
            if not fixture_source or any(item.get('applicability_fixture_source')!=fixture_source for item in originals):
                raise ValueError('Original two-arm fixture source is missing or differs')
        elif applicability_fixtures:
            from research1.data_pipeline.fixture_context import freeze_probe_contract
            fixture_source=freeze_probe_contract(source_campaign/'source_snapshot/src/research1/data_pipeline/modules.py')
    if replan_from is not None:
        if mode!='initialization' or supersedes is None:
            raise ValueError('Parent replanning requires initialization mode and the latest stopped budget predecessor via --supersedes')
        if parent_tasks is not None:raise ValueError('Replanning uses every scored source parent; a task subset is not permitted')
        if Path(replan_from).resolve()==campaign:raise ValueError('Replanning requires a new immutable campaign')
    if (campaign/'protocol.json').exists():
        existing=load_protocol(campaign)
        if existing['seed']!=seed or existing['config_path']!=str(config_path.resolve()):raise ValueError('Prepared campaign conditions differ; create a new campaign')
        if (existing.get('budget_lineage') or {}).get('previous_campaign')!=(str(Path(supersedes).resolve()) if supersedes else None):
            raise ValueError('Prepared budget lineage differs')
        if existing.get('execution_mode','calibration')!=mode:raise ValueError('Prepared execution mode differs')
        if parent_tasks is not None and existing.get('initialization',{}).get('task_order')!=parent_tasks:
            raise ValueError('Prepared parent queue differs; create a new immutable campaign')
        expected_source=str(Path(replan_from).resolve()) if replan_from else None
        if (existing.get('source_parents') or {}).get('source_campaign')!=expected_source:
            raise ValueError('Prepared source-parent campaign differs')
        if existing.get('diagnostic_source_selection')!=diagnostic_selection:
            raise ValueError('Prepared diagnostic source selection differs')
        if existing.get('verification_registry_version',V2_REGISTRY)!=registry_version:
            raise ValueError('Prepared verification registry version differs')
        trial=existing.get('grounded_edit_trial',{})
        if admission_review_policy is not None and existing['policies']['factual_reviewer']['version']!=admission_review_policy:
            raise ValueError('Prepared admission reviewer differs')
        if applicability_review_policy is not None and existing['policies']['applicability_reviewer']['version']!=applicability_review_policy:
            raise ValueError('Prepared applicability reviewer differs')
        if (existing.get('plan_precision_policy')!=plan_precision_policy or
                existing.get('policies',{}).get('factual_reviewer',{}).get('precision_policy')!=plan_precision_policy):
            raise ValueError('Prepared plan precision policy differs')
        if existing.get('independent_plan_audit_policy')!=independent_plan_audit_policy:
            raise ValueError('Prepared independent plan audit policy differs')
        if existing.get('policies',{}).get('applicability_reviewer',{}).get('fixture_source')!=fixture_source:
            raise ValueError('Prepared applicability fixture visibility differs')
        fixture_ref=existing.get('policies',{}).get('applicability_reviewer',{}).get('fixture_source_ref')
        if registry_version==V3_REGISTRY and applicability_fixtures and not replay_factual_from_source:
            relative='src/research1/data_pipeline/modules_v3.py'
            expected_ref={'path':str(campaign/'source_snapshot'/relative),
                'sha256':existing.get('source_sha256',{}).get(relative)}
            reference_path=Path(expected_ref['path'])
            if (not expected_ref['sha256'] or fixture_ref!=expected_ref or not reference_path.is_file()
                    or hashlib.sha256(reference_path.read_bytes()).hexdigest()!=expected_ref['sha256']):
                raise ValueError('Prepared fresh applicability fixture reference differs')
        elif fixture_ref is not None:
            raise ValueError('Prepared fresh applicability fixture visibility differs')
        if (trial.get('fact_source')!=fact_source or trial.get('teacher_max_tokens',8000)!=teacher_max_tokens or
            (existing.get('sft',{}).get('fact_index_projection')=='compact_v1')!=compact_fact_index or
            trial.get('submitted_code_index',False)!=submitted_code_index or trial.get('teacher_source')!=teacher_source or
            trial.get('publish_probe_io',False)!=publish_probe_io or
            trial.get('replay_factual_from_source',False)!=replay_factual_from_source):
            raise ValueError('Prepared diagnostic input or token budget differs')
        if replan_from:
            from research1.data_pipeline.source_parents import validate_parent_sources
            validate_parent_sources(campaign,existing)
        if replay_factual_from_source:
            from research1.data_pipeline.teacher_replay import load_teacher_replay
            rows=validate_parent_sources(campaign,existing)
            load_teacher_replay(existing,rows[0])
        return existing
    inherited=None
    cfg=load_pilot(config_path);dataset=resolve_path(cfg,'dataset_tb')
    runtime_source=resolve_path(cfg,'modularrsi_root')/'src/harbor'
    if not runtime_source.is_dir():raise ValueError('Harbor runtime source directory missing')
    tasks=train_task_names(dataset)
    if cached_images is None:
        if shutil.disk_usage(root).free<15*1024**3:raise RuntimeError('Disk reserve below 15GB')
        proc=subprocess.run(['docker','images','--format','{{.Repository}}:{{.Tag}}'],capture_output=True,text=True,timeout=30,check=True)
        cached_images=set(proc.stdout.splitlines())
    inventory=image_inventory(dataset,tasks,cached_images)
    exposed=exposed_tasks(root,tasks)
    if supersedes:
        sets=json.loads(json.dumps(load_protocol(Path(supersedes))['task_sets']))
        validate_partitions(sets,tasks)
    else:sets=partition_tasks(tasks,exposed,seed=seed,ready=inventory)
    from research1.data_pipeline.readiness import build_readiness
    readiness=build_readiness(root,dataset,tasks,wall_seconds=900)
    if mode=='calibration':sets=runtime_debug_partition(sets,tasks,readiness,seed=seed)
    # Supersession preserves role/fold assignment, but an observed proposal is
    # not a fresh independent parent and must not be replayed as one.
    init_excluded=(exposed & set(sets['proposal'])) if supersedes else set()
    # Source selection is frozen only after full compatibility checks below.
    init_order=([] if replan_from else initialization_queue(sets,inventory,readiness,seed=seed,max_parents=max_parents,
                                    excluded_tasks=init_excluded,task_order=parent_tasks)) if mode=='initialization' else []
    if mode=='initialization' and not init_order and not replan_from:raise ValueError('No budget-compatible proposal tasks available')
    selected=set().union(*(set(sets[role]) for role in ('debug','proposal','reward','dev')))
    task_asset_hashes={}
    for task in sorted(selected):
        task_root=dataset/'tasks'/task
        if not task_root.is_dir():raise ValueError('Manifest task directory missing: '+task)
        task_asset_hashes[task]=tree_hash(task_root)
    snapshot=json.loads((root/'configs/h0.snapshot.v0.json').read_text())
    source=Path(snapshot['modules_root'])
    for rel,expected in snapshot['file_sha256'].items():
        path=source/rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('H0 source does not match frozen snapshot: '+rel)
    from research1.data_pipeline.framework_context import build_framework_context
    framework_context = build_framework_context(resolve_path(cfg,'modularrsi_root'),
        runtime_limits={'max_turns':200,'wall_seconds':900})
    if replan_from:
        from research1.data_pipeline.source_parents import preview_handover_budget
        inherited=preview_handover_budget(supersedes,campaign)
    else:inherited=handover_budget(supersedes,campaign) if supersedes else None
    campaign.mkdir(parents=True,exist_ok=True)
    dest=campaign/'baseline/modules'
    if dest.exists():raise ValueError('Unfinished preparation directory: inspect it before choosing a new campaign')
    shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    code_files={}
    for area in ('src/research1','scripts'):
        for p in (root/area).rglob('*.py'):
            if '__pycache__' in p.parts:continue
            rel=p.relative_to(root).as_posix();target=campaign/'source_snapshot'/rel
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            code_files[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    from research1.data_pipeline.module_adapters import get_adapter
    from research1.data_pipeline.policy import FACTUAL_WITNESS_VALIDATOR, FACTUAL_REVIEW_POLICY, POLICY_ID
    from research1.data_pipeline.diagnostics import DIAGNOSTIC_POLICY
    registry=get_adapter(registry_version).frozen_registry()
    atomic_json(campaign/'contracts.json',registry)
    manifest=dataset/'manifest.json'
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol={'protocol_id':PROTOCOL_ID,'schema_version':'research1.data.campaign.v1',
        'campaign_id':campaign.name,'created_utc':now,'seed':seed,'repo_root':str(root),
        'config_path':str(config_path.resolve()),'config_hash':cfg['_config_hash'],
        'modules_root':str(dest),'modules_hash':tree_hash(dest),'module_files':_file_digests(dest),
        'dataset_path':str(dataset),'dataset_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'upstream_root':str(resolve_path(cfg,'modularrsi_root')),
        'runtime_source_hash':tree_hash(runtime_source),
        'framework_context':framework_context,
        'planning_flow':{'catalog_mode':'all_frozen_interfaces_not_parent_diagnoses',
            'unknown_mechanism':'preserve_raw_pending_executable_verification',
            'admission':'independent_facts_and_preteacher_selected_behavior_contract_required',
            'comparison_required_before_new_sampling':True},
        'task_sets':sets,'task_environment_inventory':inventory,'historical_exposure':sorted(exposed),
        'runtime_readiness':readiness,'execution_mode':mode,
        'initialization':{'max_parents':max_parents,'task_order':init_order,'max_candidates':2,
                          'queue_selection':'explicit_predeclared' if parent_tasks is not None else 'seeded_runtime_readiness',
                          'excluded_observed_parents':sorted(init_excluded),
                          'stop_policy':'record task-local incompletes and continue the fixed queue; stop on global config, auth, artifact or resource failures'},
        'task_asset_hashes':task_asset_hashes,
        'policies':{'solver':{'provider':'tongji_agi','model':'openai/DeepSeek-V4.1-Flash','temperature':.7,'api_base':'https://agi.tongji.edu.cn/v1','reasoning_effort':'none'},
                    'teacher':{'provider':'tongji_agi','model':'DeepSeek-V4.1-Flash','temperature':0,'format':'json_final','version':POLICY_ID},
                    'implementer':{'provider':'tongji_agi','model':'DeepSeek-V4.1-Flash','temperature':0,'version':'data-implementer-v1'},
                    'factual_reviewer':{'provider':'tongji_agi','model':'DeepSeek-V4.1-Flash','temperature':0,'version':'proposal-factual-review.v6.typed' if POLICY_ID=='proposal_single_plan_v6' else FACTUAL_REVIEW_POLICY,'witness_validator':FACTUAL_WITNESS_VALIDATOR},
                    'diagnostician':{'provider':'tongji_agi','model':'DeepSeek-V4.1-Flash','temperature':0,'version':DIAGNOSTIC_POLICY},
                    'continuation':{'selection':'first_saved_candidate_without_reward_access',
                                    'support_sampling':'independent_local_rng_without_replacement_within_decision',
                                    'solver_requests':'fresh_per_path_decision','teacher_temperature':0},
                    'replay':{'mode':'saved_patch_only'}},
        'budget':inherited['remaining'] if inherited else {'regular':96,'retries':32,'total':128},
        'budget_lineage':inherited,
        'stage_budget':({'noise':0,'reward':0,'proposal':min(inherited['remaining']['regular'] if inherited else 96,3*len(init_order)),'dev':0}
                        if mode=='initialization' else {'noise':24,'reward':44,'proposal':18,'dev':10}),
        'runtime':{'max_turns':200,'wall_seconds':900,'disk_floor_gb':15,'infra_extra_retries':2},
        'sampling':{'K':2,'M':2,'L_after':2,'R':1,'reward_n':4,'proposal_per_decision':2,'dev_n':2,'lambda':.5},
        'evidence_budget':{'max_total_chars':96000,'max_step_chars':16000,'max_steps':12},
        'sft':{'interface':'single_plan_v1','plan_policy_version':POLICY_ID,'keep_max_fraction':.2,'training_objective':'basic_evolver_initialization','target_profile':None},
        'permitted_modules':list(MODULES),'registry_sha256':digest(registry),'source_sha256':code_files,
        'formal_scale':None,'training_started':False,'noise_enabled':mode=='calibration','collection_split':'train',
        'completion_rule':('Independent initialization batch; does not certify noise/MC calibration or complete formal quotas'
                           if mode=='initialization' else 'Calibration requires a functional edit and a complete K2 M2 L_after2 group; then NEEDS_SCALE_LOCK, never DATA_COLLECTION_COMPLETE')}
    if registry_version==V3_REGISTRY:
        protocol['verification_registry_version']=V3_REGISTRY
    if replan_from:
        from research1.data_pipeline.source_parents import freeze_parent_sources,write_parent_snapshot
        if diagnostic_selection:
            protocol.update(trial_driver='grounded_edit_trial.v1',diagnostic_source_selection=diagnostic_selection,
                grounded_edit_trial={'arms':[{'id':'standard','reasoning_effort':'none'},
                    {'id':'reasoned','reasoning_effort':'low'}],'stop_after_qualified_edit':1})
            protocol['policies']['fact_indexer']={'provider':'tongji_agi','model':'DeepSeek-V4.1-Flash',
                'temperature':0,'reasoning_effort':'none','version':'proposal-fact-index.v1'}
            protocol['policies']['applicability_reviewer']={'provider':'tongji_agi','model':'DeepSeek-V4.1-Flash',
                'temperature':0,'reasoning_effort':'none','version':'proposal-plan-applicability.v1'}
            if admission_review_policy is not None:
                protocol['policies']['factual_reviewer']['version']=admission_review_policy
            if plan_precision_policy is not None:
                protocol['plan_precision_policy']=plan_precision_policy
                protocol['policies']['factual_reviewer']['precision_policy']=plan_precision_policy
            if independent_plan_audit_policy is not None:
                protocol['independent_plan_audit_policy']=independent_plan_audit_policy
                protocol['policies']['independent_plan_audit']={
                    'version':independent_plan_audit_policy,
                    'protocol_ref':{'path':str(campaign/'protocol.json')}}
            if fixture_source is not None:
                protocol['policies']['applicability_reviewer'].update(
                    version=applicability_review_policy or 'proposal-plan-applicability.v2.fixtures',fixture_source=fixture_source)
            elif registry_version==V3_REGISTRY and applicability_fixtures:
                relative='src/research1/data_pipeline/modules_v3.py'
                if not code_files.get(relative):
                    raise ValueError('Fresh applicability fixture source must be frozen in the new campaign snapshot')
                protocol['policies']['applicability_reviewer'].update(
                    version=applicability_review_policy or 'proposal-plan-applicability.v2.fixtures',
                    fixture_source_ref={'path':str(campaign/'source_snapshot'/relative),
                        'sha256':code_files[relative]})
            protocol['planning_flow']['admission']='independent_plan_to_prefrozen_checks.v1'
            protocol['policies']['implementer']['version']=('five-module-exact-replacements.v3'
                if registry_version==V3_REGISTRY else 'five-module-exact-replacements.v2')
            protocol['grounded_edit_trial']['teacher_max_tokens']=teacher_max_tokens
            if replay_factual_from_source:
                protocol['grounded_edit_trial']['replay_factual_from_source']=True
            if publish_probe_io:
                protocol['grounded_edit_trial']['publish_probe_io']=True
            if submitted_code_index:
                protocol['grounded_edit_trial']['submitted_code_index']=True
            if fact_source:
                protocol['grounded_edit_trial']['fact_source']=fact_source
                protocol['policies']['fact_indexer']['version']='proposal-fact-index.validated-partial-replay.v2'
            if compact_fact_index:
                protocol['sft']['fact_index_projection']='compact_v1'
            if teacher_source:
                protocol['grounded_edit_trial']['teacher_source']=teacher_source
        snapshot=freeze_parent_sources(Path(replan_from),protocol)
        init_order=[row['task'] for row in snapshot['rows']]
        protocol['initialization'].update(max_parents=len(init_order),task_order=init_order,
            queue_selection='all_frozen_source_parents',excluded_observed_parents=[],
            parent_execution_mode='reuse_original_proposal_execution')
        protocol['source_parents']=write_parent_snapshot(campaign,snapshot)
        protocol['planning_experiment']={'kind':'fixed-source-policy-revision',
            'source_protocol_hash':snapshot['source_protocol_hash'],'complete_source_queue':list(init_order),
            'teacher_instruction_version':POLICY_ID,'diagnostic_version':DIAGNOSTIC_POLICY,
            'passes':1,'action_conditioned_retry':False,'replace_unfavorable_parent':False,
            'stop_rule':'If no qualified functional edit is found, report insufficient evidence and use new preregistered parents; do not keep changing prompts to obtain edit.'}
        if diagnostic_selection:
            protocol['planning_experiment'].update(kind='grounded-edit-capability-diagnosis',
                source_selection_rationale=diagnostic_rationale,
                stop_rule='Exactly two preregistered teacher arms over the same source view and automatic fact index; preserve all outputs. Qualify at most the first valid edit in arm order; no task or MC path replacement.',
                reference_patch_visible_to_teacher=False,expert_fact_card_visible_to_teacher=False,
                changes_after_candidates_forbidden=True)
        protocol['stage_budget']['proposal']=min(inherited['remaining']['regular'],2*len(init_order))
        if teacher_source:
            from research1.data_pipeline.teacher_replay import load_teacher_replay
            load_teacher_replay(dict(protocol,protocol_hash='preparation-preflight'),snapshot['rows'][0])
            protocol['planning_experiment'].update(kind='saved-teacher-plan-typed-admission',
                teacher_generation_calls=0,original_teacher_campaign=teacher_source['campaign'],
                contract_reused_before_original_teacher=True,
                stop_rule='Preserve the original two plans and preteacher contract; rerun only independent typed factual/applicability admission, implementation and required child execution, in original arm order. Do not generate or rewrite plans.')
        # Source and solver/asset compatibility are already checked before any
        # predecessor authorization is changed. No source score is a new job.
        committed=handover_budget(supersedes,campaign)
        if committed!=inherited:raise ValueError('Budget predecessor changed during source-parent preparation')
    protocol['protocol_hash']=digest(protocol)
    atomic_json(campaign/'protocol.json',protocol)
    atomic_json(campaign/'runtime_readiness.json',readiness)
    atomic_json(campaign/'exposure_inventory.json',{'historical_tasks':sorted(exposed),'sets':sets,
        'selection':'Preserve prior role partitions on supersession; runtime-compatibility and provisioning readiness before seeded task order; numeric task score never ranks tasks'})
    legacy=root/'datasets/research1/sft/verified_plans.jsonl'
    legacy_rows=[]
    if legacy.exists():
        for line in legacy.read_text().splitlines():
            if line.strip():
                row=json.loads(line);legacy_rows.append({'sample_id':row.get('sample_id'),'source_run':row.get('source_run',row.get('run_id')),'formal_eligible':False,'reason':'Separate historical protocol; no automatic promotion to new formal SFT'})
    atomic_json(campaign/'legacy_inventory.json',legacy_rows)
    from research1.data_pipeline.legacy import quarantine_directed
    atomic_json(campaign/'legacy_migration.json',quarantine_directed(root))
    atomic_json(campaign/'status.json',{'status':'PREPARED','phase':'prepared','protocol_hash':protocol['protocol_hash'],'model_calls':0,'task_evaluations':0,'training_started':False})
    from research1.data_pipeline.ledger import Ledger
    Ledger(campaign)
    return protocol

def load_protocol(campaign:Path,*,verify_source:bool=False)->dict:
    doc=json.loads((Path(campaign)/'protocol.json').read_text(encoding='utf-8'))
    expected=doc.get('protocol_hash');body={k:v for k,v in doc.items() if k!='protocol_hash'}
    if expected!=digest(body):raise ValueError('Frozen campaign protocol changed')
    if tree_hash(Path(doc['modules_root']))!=doc['modules_hash']:raise ValueError('Frozen H0 source changed')
    registry=json.loads((Path(campaign)/'contracts.json').read_text(encoding='utf-8'))
    if digest(registry)!=doc['registry_sha256']:raise ValueError('Frozen behavior registry changed')
    if doc.get('verification_registry_version') is not None:
        from research1.data_pipeline.module_adapters import get_adapter
        if registry!=get_adapter(doc).frozen_registry():
            raise ValueError('Frozen behavior registry differs from its explicit version')
    if verify_source:
        for artifact in doc.get('grounded_edit_trial',{}).get('fact_source',{}).get('artifacts',{}).values():
            path=Path(artifact['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
                raise ValueError('Frozen fact-source replay artifact changed')
        root=Path(doc['repo_root'])
        cfg=load_pilot(Path(doc['config_path']))
        if cfg['_config_hash']!=doc['config_hash']:raise ValueError('Execution config changed since preparation')
        if str(resolve_path(cfg,'dataset_tb'))!=doc['dataset_path'] or str(resolve_path(cfg,'modularrsi_root'))!=doc['upstream_root']:
            raise ValueError('Execution paths differ from frozen protocol')
        manifest=Path(doc['dataset_path'])/'manifest.json'
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=doc['dataset_manifest_sha256']:
            raise ValueError('Dataset manifest changed since preparation')
        if doc.get('runtime_source_hash') and tree_hash(Path(doc['upstream_root'])/'src/harbor')!=doc['runtime_source_hash']:
            raise ValueError('Harbor runtime source changed since preparation')
        for rel,expected in doc['source_sha256'].items():
            if not (root/rel).is_file() or hashlib.sha256((root/rel).read_bytes()).hexdigest()!=expected:raise ValueError('Controller source changed since preparation: '+rel)
    return doc
