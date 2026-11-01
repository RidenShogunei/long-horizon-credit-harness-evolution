"""Root admission, evidence isolation, complete ledger and portable release sealing."""
import collections,copy,datetime,hashlib,json,pathlib,shutil,sys
W=pathlib.Path(__file__).resolve().parent;R=W.parent;B=W/'open_dataset_integration_v2';D=B/'release_v2';V1=W/'open_dataset_integration_v1/release_v1'
sys.path.insert(0,str(W/'server_fix/src'))
from research1.data_pipeline.policy import validate_plan,public_view
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,v):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def jsonl(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
def copy_file(src,dst):
    assert src.is_file(),src
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

raw=list(map(json.loads,(D/'source_record_corpus/combined_source_records_96.jsonl').read_text(encoding='utf8').splitlines()))
raw_by_id={r['instance_id']:r for r in raw};assert len(raw)==len(raw_by_id)==96
rows=[json.loads(s) for split in ['train','validation'] for s in (D/f'evolver_aux.{split}.jsonl').read_text(encoding='utf8').splitlines()]
assert len(rows)==23 and len({r['sample_id'] for r in rows})==23
old_prov={r['sample_id']:r for r in read(V1/'provenance.json')}
new_prov={r['sample_id']:r for r in read(D/'provenance.json')}
rerun=read(B/'root_rerun/summary.json');assert rerun['counts']=={'pass':23,'pending':0,'reject':1}
root_proofs={}
for pr in rerun['rows']:
    rec=read(R/pr['proof']);res=rec['result'];sid=res.get('source_id') or res.get('sample_id')
    if res['status']=='pass':root_proofs[sid]=pr['proof']
last=read(B/'root_rerun/last_gap_results.json');assert last['berriai__litellm-5232']['status']=='pass' and last['berriai__litellm-9828']['status']=='pass'
assert last['berriai__litellm-9828']['caller_propagation']['status']=='pass'
interfaces=read(B/'root_parent_interface_refs.json')
old_after={'huggingface__smolagents-285':'b3bee3ae99dbff68779f/src/smolagents/local_python_executor.py','huggingface__smolagents-891':'8c80268b2e679b8968d3/src/smolagents/local_python_executor.py','huggingface__smolagents-1088':'9d77a98c64528b8f68cb/src/smolagents/_function_type_hints_utils.py','huggingface__smolagents-478':'6c64d934baa83a12b609/src/smolagents/default_tools.py'}
provenance=[];reviews=[];seen_patches={};repo_splits={};roles=collections.Counter();partitions=collections.Counter()
for row in rows:
    sid=row['sample_id'];source=raw_by_id[sid];assert sid in root_proofs,sid
    view=json.loads(row['messages'][1]['content']);plan=json.loads(row['messages'][2]['content'])
    # Replace a Cursor-added post-repair description with actual parent declarations.
    if sid=='crewaiinc__crewai-1818':
        texts=[]
        for ref in interfaces['refs']:
            p=R/ref['artifact_path'];assert sha(p)==ref['file_sha256']
            text='\n'.join(p.read_text(encoding='utf8').splitlines()[ref['start_line']-1:ref['end_line']]);assert text==ref['excerpt']
            texts.append(ref['original_path']+'\n'+text)
            copy_file(p,D/'portable_sources'/sid/'before'/ref['original_path'])
        note='Read-only declarations at this parent revision: Agent.knowledge_sources and StringKnowledgeSource.content. These establish the supplied string-source interface; they do not establish every knowledge-source implementation.\n'+'\n\n'.join(texts)
        view['evidence']=[ev for ev in view['evidence'] if ev['id']!='ev_readonly_interface_context']+[{'id':'ev_readonly_interface_context','text':note}]
        plan['problem_hypothesis']='Current-code fact: CrewPlanner._create_tasks_summary includes task/agent/tool fields and omits knowledge. The supplied parent declarations expose Agent.knowledge_sources and StringKnowledgeSource.content. Reporter claim: loaded knowledge is missing from planning. Hypothesis: include available string-source content in the planning summary; solver benefit remains untested.'
        plan['intended_change']='In the same planning module, add _get_agent_knowledge(task) reading task.agent.knowledge_sources and each supplied source.content; append the nonempty agent_knowledge value to _create_tasks_summary. Scope this demonstration to sources exposing content, such as the supplied string-source interface.'
        plan['verification_plan']=['Compare actual summaries for an agent with supplied content-bearing knowledge sources: the edit must include that content while preserving task description, expected output, role, goal and tool fields.','For an agent without knowledge sources, compare the absence of agent_knowledge and preservation of the existing summary fields. This bounded fixture does not establish support for every knowledge source.']
    assert [m['role'] for m in row['messages']]==['system','user','assistant']
    assert set(view)<= {'module_code','evidence','history','permitted_modules','verification_catalog'},(sid,set(view))
    state={**view,'state_id':sid,'root_lineage_id':'external/'+source['repo'].lower(),'split':row['split'],'plan_policy_version':'proposal_single_plan_v6'}
    validate_plan(plan,state);assert public_view(state)==view
    assert len(plan)==9 and plan['action']=='edit'
    assert source['problem_statement'] in view['evidence'][0]['text']
    for ev in view['evidence']:
        if ev['id']=='ev_readonly_interface_context':assert 'applied' not in ev['text'].lower()
    input_text=row['messages'][0]['content']+'\n'+json.dumps(view,ensure_ascii=False)
    for forbidden in ['child_patch','future_trajectory','mc_label','reward_score','source_group','root_lineage_id']:
        assert '"'+forbidden+'"' not in input_text
    assert source['patch'] not in input_text
    repo=source['repo'].lower();assert repo_splits.setdefault(repo,row['split'])==row['split']
    patch_sha=hashlib.sha256(source['patch'].encode()).hexdigest();assert (repo,patch_sha) not in seen_patches
    seen_patches[repo,patch_sha]=sid
    usage='AGENT_INFRASTRUCTURE_AUXILIARY' if 'litellm' in repo or sid.endswith('dspy-8053') else 'FRAMEWORK_PLANNING_AUXILIARY'
    row['usage']=usage;row['local_acceptance']='BOUNDED_BEHAVIOR_AND_PRESERVATION_VERIFIED';row['formal_training_ready']=False
    row['messages'][1]['content']=json.dumps(view,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    row['messages'][2]['content']=json.dumps(plan,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    roles[plan['target_module']]+=1;partitions[usage]+=1
    pr=copy.deepcopy(old_prov[sid] if sid in old_prov else new_prov[sid])
    for ref in pr['source_refs']:
        path=R/ref['artifact_path'];assert sha(path)==ref.get('file_sha256',ref.get('sha256'))
        segment='\n'.join(path.read_text(encoding='utf8').splitlines()[ref['start_line']-1:ref['end_line']])
        h=ref['excerpt_sha256'];assert hashlib.sha256(segment.encode()).hexdigest()==h or hashlib.sha256((segment+'\n').encode()).hexdigest()==h,(sid,ref)
        dest=D/'refs'/sid/(sha(path)[:10]+'_'+path.name);copy_file(path,dest);ref['bundle_source_path']=str(dest.relative_to(D)).replace('\\','/')
    if sid in old_after:
        after=W/'open_dataset_integration_v1/rebench/replays'/old_after[sid]
        copy_file(after,D/'portable_sources'/sid/'applied'/('src/smolagents/'+after.name))
        (D/'portable_sources'/sid/'original.patch').write_text(source['patch'],encoding='utf8')
    pr.update({'source_repo':source['repo'],'source_base_commit':source['base_commit'],'patch_sha256':patch_sha,'local_behavior_execution_verified':True,'root_behavior_verified':True,'root_review_status':'accepted_bounded_auxiliary','root_proof':root_proofs[sid],'formal_research1_sft':False,'original_teacher_plan':False,'usage':usage,'independent_issue_id':sid,'scope':'Real published patch/source callables, bounded before/after behavior with explicit preservation; dependency IO stubs documented. Full upstream test suite and TerminalBench transfer not established.'})
    if sid=='crewaiinc__crewai-1818':pr['read_only_parent_interface_refs']=interfaces['refs']
    if sid in last:pr['supplementary_root_proof']='root_verification/last_gap_results.json'
    provenance.append(pr);write(D/'source_records'/(sid+'.json'),source)
    reviews.append({'sample_id':sid,'accepted':True,'root_proof':root_proofs[sid],'plan_editorial_overrides_preserved':True,'scope':pr['scope']})

for split in ['train','validation']:
    sr=[r for r in rows if r['split']==split];jsonl(D/f'evolver_aux.{split}.jsonl',sr);jsonl(D/f'evolver_aux.{split}.edit.jsonl',sr);jsonl(D/f'evolver_aux.{split}.keep.jsonl',[])
    for part,label in [('framework','FRAMEWORK_PLANNING_AUXILIARY'),('infrastructure','AGENT_INFRASTRUCTURE_AUXILIARY')]:jsonl(D/f'evolver_{part}_aux.{split}.jsonl',[r for r in sr if r['usage']==label])
write(D/'provenance.json',provenance);write(D/'curation_review.json',reviews)
pending={'stanfordnlp__dspy-1556','stanfordnlp__dspy-8235','stanfordnlp__dspy-8432','berriai__litellm-11006','berriai__litellm-6138','berriai__litellm-6563','crewaiinc__crewai-2477'}
failed={'huggingface__smolagents-226':'Observed wildcard constructor regression; v1 auxiliary positive revoked.','crewaiinc__crewai-2604':'Required memory compatibility checks failed in strengthened probe; not positive data.'}
old_replay={r['source_row_id']:r['patch_apply_check']['ok'] for r in map(json.loads,(W/'open_dataset_integration_v1/rebench/usable_candidates.jsonl').read_text(encoding='utf8').splitlines())}
selected=set(r['instance_id'] for r in map(json.loads,(B/'main_candidates/raw_records.jsonl').read_text(encoding='utf8').splitlines()))|set(r['instance_id'] for r in map(json.loads,(B/'additional_crewai/selected_raw_records.jsonl').read_text(encoding='utf8').splitlines()))
accepted={r['sample_id'] for r in rows};ledger=[]
for r in raw:
    sid=r['instance_id'];reconstructed=old_replay.get(sid,sid in selected)
    if sid in accepted:status='accepted_auxiliary';reason='Root bounded behavior/preservation and Plan/source/export audit passed.'
    elif sid in failed:status='failed_verification';reason=failed[sid]
    elif sid in pending:status='pending_verification';reason='Not admitted; incomplete scope, stream reproduction or dependency/caller checks. Earlier worker pass is insufficient.'
    elif sid in old_replay and not reconstructed:status='reconstruction_failed';reason='Original patch reconstruction incomplete; retained unchanged.'
    else:status='not_admitted';reason='Raw source retained outside positives; out of scope, curation rejected or not yet individually verified.'
    ledger.append({'sample_id':sid,'repo':r['repo'],'base_commit':r['base_commit'],'patch_sha256':hashlib.sha256(r['patch'].encode()).hexdigest(),'original_patch_reconstructed':reconstructed,'status':status,'reason':reason})
write(D/'acceptance_ledger.json',ledger)
for p in (B/'root_rerun').glob('*.json'):copy_file(p,D/'root_verification'/p.name)
for name in ['root_final_decisions.json','root_parent_interface_refs.json','root_existing_review.json'] :copy_file(B/name,D/'root_verification'/name)
for name in ['verify_extras.py','results.json','plan_overrides.json']:copy_file(B/'root_scope_extras'/name,D/'root_verification/curator_scope_extras'/name)
for p in [W/'build_open_dataset_v2.py',pathlib.Path(__file__),W/'root_verify_open_dataset_v2.py',W/'root_verify_last_gaps_v2.py',W/'root_freeze_missing_parent_interfaces.py',W/'check_open_dataset_tokenizer_only.py']:copy_file(p,D/'data_build_scripts'/p.name)
counts={'train':sum(r['split']=='train' for r in rows),'validation':sum(r['split']=='validation' for r in rows),'edit':23,'keep':0,'independent_issue_roots':23,'independent_parent_code_states':len({(r['source_repo'].lower(),r['source_base_commit']) for r in rows}),'source_repositories':len(repo_splits),'new_accepted_this_batch':19,'retained_v1':4,'revoked_v1':1,'formal_autonomous_sft':0,'mc_values':0}
manifest={'dataset_id':'research1-open-auxiliary-integrated-v2','root_approved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':counts,'partitions':dict(partitions),'module_roles':dict(roles),'raw_ledger_counts':dict(collections.Counter(r['status'] for r in ledger)),'raw_records':96,'strict_patch_reconstructions':sum(r['original_patch_reconstructed'] for r in ledger),'readiness':{'generic_schema_passed':True,'bounded_behavior_and_preservation_verified':True,'input_source_and_leakage_audit_passed':True,'repo_group_split_passed':True,'target_student_selected':False,'formal_training_ready':False,'full_upstream_test_suite_run':False,'terminal_bench_transfer_verified':False,'collection_complete':False},'no_training_or_inference':True,'budget':{'new_paid_evaluations':0,'research1_chain_regular_used':8,'research1_chain_retries_used':0,'total_cap':128},'label_policy':'Editorial public issue/change Plan demonstrations, auxiliary only. No MC/G/D/U labels inferred from published pass/fail outcomes. Compiler source records and gold patches excluded from policy input.','limitations':['Only edit demonstrations; keep count zero.','Direct framework and provider infrastructure corpora exported separately.','Before/after probes use explicitly recorded dependency/IO fixtures; no claim of full application validation.','No evidence of TerminalBench generalization or long-horizon credit assignment.'],'split_policy':'All DSPy samples validation; all other accepted repositories train. Same repo never spans splits.','files':{}}
write(D/'dataset_manifest.json',manifest)
readme=f'''# 研究一开源辅助规划数据 v2

已审核 **23 条 edit**：20 条训练、3 条按仓库隔离的验证样本；keep 为 0。本轮新增 19 条，保留旧版 4 条，撤销旧版 1 条。

来源为 smolagents、CrewAI、DSPy、LiteLLM 的已发布问题/变更记录。每条输入含当时源码和报告证据，答案是一份统一 JSON Plan。源码、原始补丁、测试记录、来源及许可另存，可追溯和撤销。

`evolver_framework_aux.*.jsonl`：直接框架规划辅助数据，训练 14 / 验证 2。
`evolver_infrastructure_aux.*.jsonl`：工具提供商等基础设施辅助数据，训练 6 / 验证 1。
`evolver_aux.*.jsonl`：两分区合并，训练 20 / 验证 3。`messages` 为训练字段，仅 assistant 答案计算损失。

验收包括实际固定版本的源码/补丁重建、局部修改前后行为、保持性、证据与答案对应、格式及按仓库分组。96 条原始来源记录中，66 条完成补丁重建；它们不会全部进入正向数据。完整状态见 acceptance_ledger.json。

这是辅助初始化数据。学生模型尚未锁定，未开始训练。完整上游测试、TerminalBench 迁移效果尚未验证；正式自主规划 SFT 和真实 MC 价值标签均为 0。不得把公开补丁结果补成 G、D、U。原始回复/记录与失败保留，正式采集预算本轮未消耗。

许可和来源见 source_cards.json；编译数据集与原仓库代码的许可分别保存。工作脚本和验收证据见 data_build_scripts/、root_verification/、portable_proofs/。局部验证依赖 fixtures；这些记录不声称已运行完整项目环境。
'''
(D/'README.md').write_text(readme,encoding='utf8')
source_cards=[]
for dataset,revision in sorted({(r['source_dataset'],r['source_revision']) for r in rows}):
    source_cards.append({'dataset':dataset,'revision':revision,'source_url':f'https://huggingface.co/datasets/{dataset}/tree/{revision}','compilation_license':'CC-BY-4.0','code_licenses_by_sample':{r['sample_id']:raw_by_id[r['sample_id']].get('license',raw_by_id[r['sample_id']].get('license_name')) for r in rows if r['source_dataset']==dataset}})
write(D/'source_cards.json',source_cards)
write(D/'DATASET_CARD.json',{'dataset_id':manifest['dataset_id'],'usage':'bounded public issue/change planning auxiliary','formal_training_ready':False,'counts':counts,'primary_manifest':'dataset_manifest.json'})
manifest['files']={str(p.relative_to(D)).replace('\\','/'):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='dataset_manifest.json' and '__pycache__' not in p.parts}
write(D/'dataset_manifest.json',manifest)
print(json.dumps({'counts':counts,'ledger':manifest['raw_ledger_counts'],'partition':dict(partitions),'module_roles':dict(roles),'files_hashed':len(manifest['files'])}))
