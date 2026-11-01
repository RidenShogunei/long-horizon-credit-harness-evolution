"""Describe observed calibration costs without inventing billing or readiness."""
from __future__ import annotations
import datetime
import json
from pathlib import Path
import statistics
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.util import atomic_json

def build_report(campaign:Path)->dict:
    campaign=Path(campaign)
    protocol=json.loads((campaign/'protocol.json').read_text())
    state=json.loads((campaign/'status.json').read_text())
    ledger=Ledger(campaign)
    with ledger.connection() as conn:
        rows=[dict(r) for r in conn.execute('SELECT kind,status,spec,result FROM jobs')]
    for child_db in (campaign/'groups').glob('group_*/ledger.sqlite3'):
        with Ledger(child_db.parent).connection() as conn:
            rows.extend(dict(r) for r in conn.execute('SELECT kind,status,spec,result FROM jobs'))
    tasks=[]
    for row in rows:
        result=json.loads(row['result']) if row['result'] else None
        spec=json.loads(row['spec'])
        tasks.append({'kind':row['kind'],'status':row['status'],'role':spec.get('role'),
                      'duration_seconds':(result or {}).get('duration_seconds'),
                      'reward':(result or {}).get('reward'),'usage':(result or {}).get('usage',{}),
                      'failure_layer':(result or {}).get('failure_layer')})
    calls=[]
    for name in ('teacher.json','editor_output.json','diagnostic_output.json','factual_review.json',
                 'fact_output.json','applicability_output.json'):
        for path in sorted(campaign.rglob(name)):
            try:
                row=json.loads(path.read_text());meta=row.get('metadata') or {}
                if 'raw' not in row or not meta:continue
                reused_quote=(name=='fact_output.json' and protocol.get('grounded_edit_trial',{}).get('fact_source',{}).get('kind')=='saved_quote_replay')
                calls.append({'artifact':str(path.relative_to(campaign)),'kind':name,'model':meta.get('model'),
                              'provider':meta.get('provider'),'usage':meta.get('usage',{}),
                              'finish_reason':meta.get('finish_reason'),'raw_field':meta.get('raw_field'),
                              'reused_quote_reply':reused_quote})
            except (ValueError,OSError):continue
    durations=[r['duration_seconds'] for r in tasks if isinstance(r['duration_seconds'],(int,float))]
    total_tokens=sum(int(c['usage'].get('total_tokens') or 0) for c in calls if not c.get('reused_quote_reply'))
    samples=ledger.records('samples')
    values=[r for r in ledger.records('values') if r.get('is_complete') and not r.get('revoked')]
    export_manifest=None
    current=campaign/'exports/sft/CURRENT.json'
    if current.exists():export_manifest=json.loads(Path(json.loads(current.read_text())['manifest']).read_text())
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    noise=ledger.records('noise')
    bytask={}
    for row in noise:bytask.setdefault(row['task'],[]).append(row['reward'])
    flips={task:len(set(scores))>1 for task,scores in bytask.items() if len(scores)==3}
    result={'campaign':str(campaign),'updated_china_time':now,'status':state['status'],
        'calibration_complete':state.get('calibration_complete',False),
        'data_collection_complete':state.get('data_collection_complete',False),
        'reason':state.get('reason'),'budget':state.get('aggregate_job_budget',ledger.status()['budget']),
        'authorization_lineage':protocol.get('budget_lineage'),
        'counts':{'raw_plan_samples':len(samples),
            'verified_plan_records':sum(r.get('qualification',{}).get('status')=='verified' and not r.get('revoked') for r in samples),
            'functional_first_edits':(sum(r['plan']['action']=='edit' and
                r.get('qualification',{}).get('status')=='verified' and not r.get('revoked') for r in samples)
                if protocol.get('trial_driver')=='grounded_edit_trial.v1' else state.get('functional_first_edits',0)),
            'complete_mc_candidates':len(values),'complete_mc_paths':state.get('complete_paths',0)},
        'sft':export_manifest,'noise':{'completed_three_repeat_tasks':len(flips),'tasks_with_flip':sum(flips.values()),
            'per_task_scores':bytask,'interpretation':'descriptive small-sample probe, not evidence of a causal edit effect'},
        'cost':{'task_jobs_reserved':len(tasks),'jobs_with_observed_duration':len(durations),
            'task_duration_median_seconds':statistics.median(durations) if durations else None,
            'task_duration_total_seconds':sum(durations),
            'recorded_planning_review_implementation_calls':sum(not c.get('reused_quote_reply') for c in calls),'recorded_model_total_tokens':total_tokens,
            'billing_amount':None,'billing_note':'Custom provider model-info contains zero pricing; monetary cost is unknown, not free'},
        'formal_one_group_task_upper_bound_R3_reward4_dev2':180,
        'formal_scale_locked':protocol.get('formal_scale') is not None,'training_started':False,
        'task_jobs':tasks,'model_calls':calls}
    atomic_json(campaign/'report.json',result)
    txt=(f"研究一数据采集实施与校准状态\n更新时间：{now}\n\n"
         f"状态：{result['status']}\n原因：{result['reason'] or '见阶段状态'}\n"
         f"校准通过：{result['calibration_complete']}；研究数据采集完成：{result['data_collection_complete']}。\n"
         f"任务预算：{result['budget']}\n样本与标签：{result['counts']}\n"
         f"SFT训练就绪：{bool(export_manifest and export_manifest.get('training_ready'))}。\n\n"
         "正式SFT采用单计划接口，修改与保留分别统计；未来评分、补丁和MC标签不进入输入。"
         "旧定向修复属于辅助记录。尚未指定并验证学生模型模板时，不标记训练就绪。\n"
         "本阶段未训练模型。金额无法从当前代理零价格元数据可靠推算，报告token与时间，不将其写成免费。\n"
         f"服务器证据目录：{campaign}\n")
    (campaign/'report.txt').write_text(txt,encoding='utf-8')
    return result
