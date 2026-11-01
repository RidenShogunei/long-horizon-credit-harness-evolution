"""Root data curation/export; no model training or inference."""
import ast,copy,hashlib,json,pathlib,re,subprocess,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];W=R/'work';B=W/'open_dataset_integration_v1';D=B/'release_v1';D.mkdir(exist_ok=False)
sys.path.insert(0,str(W/'server_fix/src'))
from research1.data_pipeline.policy import validate_plan,render_messages
curated=json.loads((B/'curated/curated_examples.json').read_text(encoding='utf8'))
raw={r['source_row_id']:r for r in map(json.loads,(B/'rebench/raw_selected_rows.jsonl').read_text(encoding='utf8').splitlines())}
normalized={r['source_row_id']:r for r in map(json.loads,(B/'rebench/usable_candidates.jsonl').read_text(encoding='utf8').splitlines())}
profile=json.loads((W/'cursor_team_workspaces/C39_prompt_controls/approved_v2/prompt_profiles.json').read_text(encoding='utf8'))['P2']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj): (D/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def extract(path,names):
    text=path.read_text(encoding='utf8');tree=ast.parse(text);chunks=[];refs=[]
    for name in names:
        parts=name.split('.');pool=tree.body;n=None
        for part in parts:
            n=next(x for x in pool if getattr(x,'name',None)==part);pool=getattr(n,'body',[])
        segment=ast.get_source_segment(text,n)
        # AST source segments omit decorators; preserve exact complete declaration including them.
        if getattr(n,'decorator_list',None):
            first=min(x.lineno for x in n.decorator_list);segment='\n'.join(text.splitlines()[first-1:n.end_lineno])
        else:first=n.lineno
        chunks.append(segment)
        refs.append({'artifact_path':str(path.relative_to(R)).replace('\\','/'),'file_sha256':sha(path),'qualname':name,'start_line':first,'end_line':n.end_lineno,'excerpt_sha256':hashlib.sha256(segment.encode()).hexdigest()})
    return '\n\n'.join(chunks),refs
out=[];reviews=[];source_refs=[]
for draft in curated:
    sid=draft['source_id'];r=raw[sid];n=normalized[sid]
    assert n['patch_apply_check']['ok'] and r['base_commit']==draft['source_mapping']['base_commit']
    if sid.endswith('-877'):
        reviews.append({'source_id':sid,'admitted':False,'reason':'Published patch removes an attribute filter as well as noisy logging; unchanged filtering behavior is not established. Preserve as a curation negative, not a positive Plan.'});continue
    row=copy.deepcopy(draft);s=row['state'];plan=row['plan'];path=R/row['source_refs'][0]['artifact_path'];assert sha(path)==row['source_refs'][0]['sha256']
    if sid.endswith('-891'):
        code,refs=extract(path,['create_function','evaluate_function_def','evaluate_ast','evaluate_python_code'])
        plan['problem_hypothesis']='Reporter claim: a loop inside a user-defined function can evade the operation cap. Current code stores an integer _operations_count in state, and create_function copies state for function execution. Hypothesis: a shared mutable counter object will carry increments across those copied state mappings.'
    elif sid.endswith('-1088'):
        text=path.read_text(encoding='utf8');nodes=ast.parse(text).body;names={'description_re','args_re','args_split_re','returns_re'}
        assignments=[x for x in nodes if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in x.targets)]
        assert len(assignments)==4
        body,refs=extract(path,['_parse_google_format_docstring','get_json_schema'])
        code='\n\n'.join(ast.get_source_segment(text,x) for x in assignments)+'\n\n'+body
        for x in assignments:refs.append({'artifact_path':str(path.relative_to(R)).replace('\\','/'),'file_sha256':sha(path),'qualname':x.targets[0].id,'start_line':x.lineno,'end_line':x.end_lineno,'excerpt_sha256':hashlib.sha256(ast.get_source_segment(text,x).encode()).hexdigest()})
    elif sid.endswith('-226'):
        code,refs=extract(path,['CodeAgent'])
        runtime,rr=extract(path,['MultiStepAgent.__init__','MultiStepAgent.initialize_system_prompt','MultiStepAgent.run'])
        s['evidence'].append({'id':'ev_current_runtime_context','text':'Read-only complete current MultiStepAgent methods from the same source file; source is supplied for the existing prompt initialization call path, not as a repair recommendation:\n'+runtime});refs+=rr
    elif sid.endswith('-285'):
        code,refs=extract(path,['evaluate_augassign'])
        plan['problem_hypothesis']='Current-code fact: evaluate_augassign dispatches through binary operators rather than the target object in-place methods. Reporter claim: this bypasses __iadd__ on objects such as PuLP LpProblem. Hypothesis: invoking available in-place methods and retaining the existing AttributeError fallback addresses that reported dispatch limitation.'
        plan['preserve_behaviors']=['Operands without in-place methods retain the existing binary-operator fallback.','The result is assigned back through the existing set_value target handling.','Unsupported augmented assignment operators still raise InterpreterError.']
    elif sid.endswith('-478'):
        code,refs=extract(path,['SpeechToTextTool'])
        # Before-version dependency, fetched at the same dataset base commit; no future implementation.
        dep=B/'rebench/dependencies'/sid/'src/smolagents/tools.py';dep.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://raw.githubusercontent.com/{r['repo']}/{r['base_commit']}/src/smolagents/tools.py"
        if not dep.exists():
            with urllib.request.urlopen(url,timeout=20) as response:data=response.read(2*1024*1024+1)
            assert len(data)<=2*1024*1024;dep.write_bytes(data)
        dt=ast.parse(dep.read_text(encoding='utf8'));tool=next(x for x in dt.body if isinstance(x,ast.ClassDef) and x.name=='Tool')
        if any(getattr(x,'name',None)=='__new__' for x in tool.body):
            ctx,rr=extract(dep,['Tool.__new__']);refs+=rr
            s['evidence'].append({'id':'ev_current_base_constructor','text':'Read-only base Tool.__new__ from the same current source revision; PipelineTool inherits Tool. The Python class object is supplied as cls by this constructor contract:\n'+ctx})
    else:raise RuntimeError('unreviewed sample '+sid)
    alias=row['source_mapping']['visible_path'];s['module_code']={alias:code};s['root_lineage_id']='external/huggingface/smolagents';s['split']='train'
    # Preserve the actual published issue text, not a curator paraphrase, as the reported observation.
    s['evidence'][0]={'id':'ev_reported_issue','text':'Published reporter issue text (reporter assertions and examples, not a locally executed trajectory):\n'+r['problem_statement']}
    s['evidence'][1]={'id':'ev_exact_source','text':'Current source excerpts in module_code are complete selected AST declarations. Unshown code is not evidence of global absence. Read-only dependencies below are current-source context. Source provenance is stored outside policy messages.'}
    validate_plan(plan,s);messages=render_messages(s,plan);messages[0]['content']=profile
    record={'sample_id':sid,'source_dataset':'nebius/SWE-rebench','source_revision':'89cdfbab4ab1bd8f5a658bb212d1b63624f4f881','source_repo':r['repo'],'source_base_commit':r['base_commit'],'source_group':'external/huggingface/smolagents','split':'train','usage':'AUXILIARY_REPORTED_ISSUE_PLANNING','original_teacher_plan':False,'formal_research1_sft':False,'messages':messages}
    out.append(record);source_refs.append({'sample_id':sid,'source_refs':refs,'original_record_sha256':hashlib.sha256(json.dumps(r,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'original_patch_reconstructed':True,'local_behavior_execution_verified':None,'published_test_metadata_kept_outside_input':True,'license_compilation':'CC-BY-4.0','license_source_code':r.get('license_name')})
    reviews.append({'source_id':sid,'admitted':True,'usage':'auxiliary issue-to-Plan warm-start','formal':False,'scope':'source facts, exact parent and original patch reconstructed; behavior checks are proposed, not claimed as locally passed','root_curation_changes':'Complete source/context and original issue restored; current facts and preservation corrected where needed. Original Cursor draft remains unchanged.'})
assert len(out)==5 and len({x['sample_id'] for x in out})==5
(D/'evolver_aux.train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out),encoding='utf8')
(D/'evolver_aux.validation.jsonl').write_text('',encoding='utf8')
write('provenance.json',source_refs);write('curation_review.json',reviews)
# Portable actual source records, not source URLs presented as examples.
(D/'swe_rebench.raw.jsonl').write_text((B/'rebench/raw_selected_rows.jsonl').read_text(encoding='utf8'),encoding='utf8')
dgm=R/'open_dataset_integration_v1/dgm/dgm_records.jsonl'
(D/'dgm.raw.jsonl').write_bytes(dgm.read_bytes())
manifest={'dataset_id':'research1-open-auxiliary-pilot-v1','created_for':'integrated public-data corpus, not model-tuning experiment','sft_usage':'Auxiliary initialization for issue-grounded source reading and bounded Plan generation; not autonomous research-one SFT or MC value training','sources':[{'name':'nebius/SWE-rebench','url':'https://huggingface.co/datasets/nebius/SWE-rebench','revision':'89cdfbab4ab1bd8f5a658bb212d1b63624f4f881','license':'CC-BY-4.0 compilation; MIT/Apache-2.0 code per row','raw_rows':len(raw),'strict_original_patch_reconstructions':sum(bool(x['patch_apply_check']['ok']) for x in normalized.values())},{'name':'DGM published experiment logs','url':'https://github.com/jennyzzt/dgm','raw_decisions':len(dgm.read_text(encoding='utf8').splitlines()),'nonempty_patches':71,'training_admitted':0,'reason':'Experiment-time parent baseline not authenticated; code license Apache-2.0, archive scope recorded separately'}],'counts':{'new_auxiliary_plan_train':len(out),'edit':len(out),'keep':0,'independent_rebench_issues':len(out),'source_repositories_in_new_train':1,'positive_validation':0,'formal_sft':0,'mc_values':0},'split_policy':'All smolagents records remain train; OpenHands records remain dev candidates outside positive SFT. DGM connected evolution lineages remain together; no DGM row enters this train set.','no_future_input':'Gold/test patches, outcomes, success labels and provenance are outside messages; reported issue text may contain reporter suggestions, hence auxiliary reported-issue task labeling.','readiness':{'generic_schema_passed':True,'behavior_verified_locally_for_all_new_rows':False,'target_model_template_and_length':'pending tokenizer-only audit','terminal_bench_transfer_verified':False,'formal_training_ready':False},'no_new_training_or_inference':True,'paid_evaluations_this_collection':0,'files':{n:sha(D/n) for n in ['evolver_aux.train.jsonl','evolver_aux.validation.jsonl','provenance.json','curation_review.json','swe_rebench.raw.jsonl','dgm.raw.jsonl']}}
write('dataset_manifest.json',manifest)
print(json.dumps({'actual_published_source_records':len(raw)+manifest['sources'][1]['raw_decisions'],'new_auxiliary_plan_rows':len(out),'strict_rebench_patch_replays':manifest['sources'][0]['strict_original_patch_reconstructions'],'formal_sft':0,'new_training':0}))
