"""One public entrypoint for research data collection and canonical SFT."""
from __future__ import annotations
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from research1.config import REPO_ROOT
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.protocol import prepare,load_protocol
from research1.data_pipeline.util import atomic_json

def _export(campaign,profile=None):
    from research1.data_pipeline.sft import export_dataset
    ledger=Ledger(campaign)
    protocol=load_protocol(campaign)
    if protocol.get('grounded_edit_trial',{}).get('replay_factual_from_source'):
        from research1.data_pipeline.teacher_replay import validate_replayed_factual_state
        from research1.data_pipeline.engine import ArtifactChangedError
        for row in ledger.records('samples'):
            if not row.get('revoked'):
                try:
                    validate_replayed_factual_state(row['state'],row['plan'])
                except (ArtifactChangedError,ValueError,KeyError,TypeError,OSError) as exc:
                    ledger.revoke(row['sample_id'],'Original saved-plan source no longer validates: '+str(exc))
    saved_profile=campaign/'student_profile.json'
    profile=profile or (json.loads(saved_profile.read_text()) if saved_profile.exists() else protocol.get('sft',{}).get('target_profile'))
    result=export_dataset(campaign,ledger.records('samples'),target_profile=profile)
    out=campaign/'exports';out.mkdir(exist_ok=True)
    for kind in ('values','comparisons'):
        rows=[r for r in ledger.records(kind) if not r.get('revoked')]
        if kind=='values':rows=[r for r in rows if r.get('is_complete') is True]
        atomic_json(out/(kind+'.json'),rows)
    source_validation=result.get('source_parent_validation')
    if source_validation and source_validation.get('status')!='PASS':
        result=dict(result,status='BLOCKED_ARTIFACT_CHANGED',reason=source_validation['reason'])
    recheck_validation=result.get('factual_recheck_validation')
    if recheck_validation and recheck_validation.get('status')!='PASS':
        result=dict(result,status='BLOCKED_ARTIFACT_CHANGED',reason=recheck_validation['reason'])
    status_path=Path(campaign)/'status.json'
    if status_path.is_file():
        status=json.loads(status_path.read_text())
        status['sft']={key:result.get(key) for key in ('dataset_revision','counts','training_ready','readiness_reasons')}
        counts=status.get('counts')
        if isinstance(counts,dict):
            # Export may revoke invalid derived records, and explicit revoke
            # leaves original qualification fields intact for audit. Summarize
            # only active qualifications while retaining every raw plan.
            all_samples=ledger.records('samples')
            active=[row for row in all_samples if not row.get('revoked')]
            sample_counts={'raw_plans':len(all_samples),'revoked':len(all_samples)-len(active),
                'qualified_edit':sum(row.get('plan',{}).get('action')=='edit' and row.get('qualification',{}).get('status')=='verified' for row in active),
                'qualified_keep':sum(row.get('plan',{}).get('action')=='keep' and row.get('qualification',{}).get('status')=='verified' for row in active),
                'pending':sum(row.get('qualification',{}).get('status')=='pending' for row in active),
                'rejected':sum(row.get('qualification',{}).get('status')=='rejected' for row in active)}
            counts.update({key:value for key,value in sample_counts.items() if key in counts or key=='revoked'})
        atomic_json(status_path,status)
    return result

def command(args):
    action=args.data_action
    campaign=Path(args.campaign).resolve() if args.campaign else REPO_ROOT/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('data_%Y%m%d_%H%M%S')
    if action=='prepare':
        if bool(args.from_calibration)!=bool(args.scale_config):raise ValueError('--from-calibration and --scale-config must be supplied together')
        if args.from_calibration:
            from research1.data_pipeline.scale import prepare_formal
            protocol=prepare_formal(campaign,Path(args.from_calibration),json.loads(Path(args.scale_config).read_text()))
        else:protocol=prepare(campaign,Path(args.config),seed=args.seed,supersedes=Path(args.supersedes) if args.supersedes else None,
                              mode=args.mode,max_parents=args.parents,parent_tasks=args.parent_task,
                              replan_from=Path(args.replan_from) if args.replan_from else None,
                              diagnostic_tasks=getattr(args,'diagnostic_source_task',None),
                              diagnostic_rationale=getattr(args,'diagnostic_rationale',None),
                              fact_replay_from=Path(args.fact_replay_from) if getattr(args,'fact_replay_from',None) else None,
                              fact_replay_state=Path(args.fact_replay_state) if getattr(args,'fact_replay_state',None) else None,
                              compact_fact_index=getattr(args,'compact_fact_index',False),
                              teacher_max_tokens=getattr(args,'teacher_max_tokens',8000),
                              submitted_code_index=getattr(args,'submitted_code_index',False),
                              teacher_replay_from=Path(args.teacher_replay_from) if getattr(args,'teacher_replay_from',None) else None,
                              admission_review_policy=getattr(args,'admission_review_policy',None),
                              applicability_fixtures=getattr(args,'applicability_fixtures',False),
                              verification_registry_version=getattr(args,'verification_registry_version',None),
                              publish_probe_io=getattr(args,'publish_probe_io',False),
                              replay_factual_from_source=getattr(args,'replay_factual_from_source',False),
                              applicability_review_policy=getattr(args,'applicability_review_policy',None),
                              plan_precision_policy=getattr(args,'plan_precision_policy',None),
                              independent_plan_audit_policy=getattr(args,'independent_plan_audit_policy',None))
        result={'status':'PREPARED','campaign':str(campaign),'protocol_hash':protocol['protocol_hash'],
                'task_partitions':{k:len(v) for k,v in protocol['task_sets'].items()},'budget':protocol['budget'],'model_calls':0}
    else:
        protocol=load_protocol(campaign,verify_source=action in {'run','resume'})
        ledger=Ledger(campaign)
        if action=='status':
            status=json.loads((campaign/'status.json').read_text())
            result={'campaign':str(campaign),'status':status,'ledger':ledger.status()}
            launch=campaign/'launch.json'
            if launch.exists():result['launch']=json.loads(launch.read_text())
        elif action in {'run','resume'}:
            if protocol.get('formal_group_of'):raise ValueError('Formal child groups must be driven by their parent campaign to enforce the global budget')
            if args.background:
                with ledger.controller('launcher'):
                    launch_path=campaign/'launch.json'
                    if launch_path.exists():
                        old=json.loads(launch_path.read_text());pid=old.get('pid')
                        if pid and _pid_alive(pid):raise RuntimeError('Campaign controller is already running')
                    with (campaign/'controller.log').open('a') as log:
                        env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
                        process=subprocess.Popen([sys.executable,'-m','research1','data',action,'--campaign',str(campaign)],
                            cwd=protocol['repo_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                            start_new_session=True)
                    result={'status':'STARTING','campaign':str(campaign),'pid':process.pid,'action':action,
                            'protocol_hash':protocol['protocol_hash'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    atomic_json(launch_path,result)
            else:
                if protocol.get('formal_scale') is not None:
                    from research1.data_pipeline.scale import run_formal
                    result=run_formal(campaign)
                elif protocol.get('trial_driver')=='grounded_edit_trial.v1':
                    from research1.data_pipeline.edit_trial import run_grounded_edit_trial
                    result=run_grounded_edit_trial(campaign)
                elif protocol.get('execution_mode')=='initialization':
                    from research1.data_pipeline.initialization import run_initialization
                    result=run_initialization(campaign)
                else:
                    from research1.data_pipeline.engine import run_campaign
                    result=run_campaign(campaign)
                if result.get('status') not in {'RUNNING','PREPARED'}:
                    with ledger.controller():result['exports']=_export(campaign)
                from research1.data_pipeline.report import build_report
                build_report(campaign)
                result['cost_report']=str(campaign/'report.json')
        elif action=='export':
            profile=json.loads(Path(args.target_profile).read_text()) if args.target_profile else None
            if profile is not None and (not isinstance(profile,dict) or set(profile)-{'model_id','tokenizer_path','max_context_tokens'}):
                raise ValueError('Student profile accepts only model_id, local tokenizer_path and max_context_tokens')
            with ledger.controller():
                recheck=None
                if getattr(args,'recheck_factual',False):
                    from research1.data_pipeline.admission_recheck import recheck_factual
                    recheck=recheck_factual(campaign,ledger=ledger)
                result=_export(campaign,profile)
                if recheck is not None:
                    result=dict(result,factual_recheck=recheck)
                    if recheck.get('status')=='BLOCKED_ARTIFACT_CHANGED':
                        result.update(status='BLOCKED_ARTIFACT_CHANGED',reason=recheck['reason'])
                if profile is not None:
                    atomic_json(campaign/'student_profile.json',profile)
                    ledger.add_event('student_render_profile',{'profile':profile,'dataset_revision':result.get('dataset_revision'),'training_ready':result.get('training_ready')})
        elif action=='revoke':
            with ledger.controller():
                changed=ledger.revoke(args.reference,args.reason)
                result={'status':'REVOKED','changed':changed,'exports':_export(campaign)}
        else:raise ValueError(action)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    status=result.get('status')
    if isinstance(status,dict):status=status.get('status')
    if status in {'FAILED','FAIL','ERROR'}:return 1
    if status in {'INCOMPLETE','PARTIAL','INIT_BATCH_PARTIAL','BUDGET_EXHAUSTED','INSUFFICIENT_EVIDENCE','NEEDS_SCALE_LOCK','NEEDS_REVIEW','WAITING_EXISTING_JOB','BLOCKED_CONFIGURATION','BLOCKED_ARTIFACT_CHANGED','EXHAUSTED_EVIDENCE','NEEDS_STUDENT_PROFILE','GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT'}:return 2
    return 0

def _pid_alive(pid):
    if os.name!='nt':
        stat=Path('/proc')/str(pid)/'stat'
        if not stat.exists():return False
        try:return stat.read_text().split(') ',1)[1][0]!='Z'
        except (OSError,IndexError):return False
    try:os.kill(pid,0);return True
    except OSError:return False

def add_parser(subparsers):
    parser=subparsers.add_parser('data',help='Research-one collection, canonical SFT and real MC; separate from foundation diagnostics')
    actions=parser.add_subparsers(dest='data_action',required=True)
    for name in ('prepare','run','status','resume','export','revoke'):
        sub=actions.add_parser(name)
        sub.add_argument('--campaign',required=name!='prepare')
        if name=='prepare':
            sub.add_argument('--config',default=str(REPO_ROOT/'configs/research1/pilot.yaml'))
            sub.add_argument('--seed',type=int,default=20261101)
            sub.add_argument('--from-calibration',help='Completed functional calibration campaign; requires explicit scale config')
            sub.add_argument('--scale-config',help='User-locked formal quotas, sampling and total budget JSON')
            sub.add_argument('--supersedes',help='Explicitly stopped calibration; inherit remaining authorization without resetting budget')
            sub.add_argument('--replan-from',help='Reuse all valid scored parents from an original finished initialization campaign; requires --mode initialization and the latest --supersedes budget predecessor')
            sub.add_argument('--diagnostic-source-task',action='append',help='One explicitly predeclared source parent for grounded edit capability diagnosis; preserves original root and all original outcomes')
            sub.add_argument('--diagnostic-rationale',help='Concrete observed-evidence rationale for the single diagnostic source, frozen before new candidates')
            sub.add_argument('--fact-replay-from',help='Existing saved quote-selector directory; reuse its original reply without another model request')
            sub.add_argument('--fact-replay-state',help='Original immutable state used by the saved quote selector')
            sub.add_argument('--compact-fact-index',action='store_true',help='Project source-verified navigation without verbose audit fields; full records remain saved')
            sub.add_argument('--teacher-max-tokens',type=int,choices=(8000,16000),default=8000)
            sub.add_argument('--submitted-code-index',action='store_true',help='Deterministically navigate literal submitted Python code; no execution or diagnosis')
            sub.add_argument('--teacher-replay-from',help='Finished original two-arm trial; preserve its teacher plans and preteacher contract and rerun admission only')
            sub.add_argument('--replay-factual-from-source',action='store_true',help='Same-v3 original-plan replay only: preserve both original factual8 results and all physical source proofs; no generation or fact review')
            sub.add_argument('--applicability-review-policy',choices=['proposal-plan-applicability.v3.fact_ids'],help='Independent applicability policy for fresh published-v3 diagnosis or explicit same-v3 saved-factual admission')
            sub.add_argument('--plan-precision-policy',choices=['proposal-plan-precision.v1.field_audit'],help='For fresh published registry-v3 diagnosis, audit every plan narrative field in the existing factual review')
            sub.add_argument('--independent-plan-audit-policy',choices=['research1.independent-plan-audit.v1'],help='For the same fresh published-v3 profile, require external blinded independent plan audit before any editor call')
            sub.add_argument('--admission-review-policy',choices=('proposal-factual-review.v6.typed','proposal-factual-review.v7.selectors','proposal-factual-review.v8.probe_io'))
            sub.add_argument('--applicability-fixtures',action='store_true',help='Show the frozen preteacher fixture source only to the admission reviewer, for saved-plan replay or fresh registry-v3 diagnosis')
            sub.add_argument('--verification-registry-version',choices=('research1.module-contracts.v2','research1.module-contracts.v3'),help='Opt in to registry v3 only for a fresh fixed-source diagnostic trial; historical v2 plans keep their original contract')
            sub.add_argument('--publish-probe-io',action='store_true',help='For fresh registry-v3 diagnosis, show the already executed parent local-API probe input/output before planning; never rerun or expose verdicts')
            sub.add_argument('--mode',choices=('calibration','initialization'),default='calibration',help='Initialization generates plans independently of noise/MC calibration')
            sub.add_argument('--parents',type=int,default=6,help='Bounded initialization parent queue length')
            sub.add_argument('--parent-task',action='append',help='Exact predeclared initialization proposal task order; repeat per task, preserving frozen folds and readiness guards')
        if name in {'run','resume'}:sub.add_argument('--background',action='store_true')
        if name=='export':
            sub.add_argument('--target-profile',help='JSON local tokenizer/chat-template/context profile; no profile means not training-ready')
            sub.add_argument('--recheck-factual',action='store_true',help='Explicit local deterministic revalidation of every frozen original critic response; never requests a new model review')
        if name=='revoke':
            sub.add_argument('--reference',required=True);sub.add_argument('--reason',required=True)
        sub.set_defaults(func=command)

