import collections,datetime,hashlib,json,pathlib,shutil,zipfile
W=pathlib.Path(__file__).resolve().parent;B=W/'open_dataset_integration_v2';D=B/'release_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((D/'dataset_manifest.json').read_text(encoding='utf8'))
audit=json.loads((D/'diagnostic_tokenizer_audit.json').read_text(encoding='utf8'))
assert len(audit['rows'])==23 and audit['all_within_cap'] and not audit['weights_loaded']
assert len({v['sample_id'] for v in audit['rows']})==23
for row in audit['rows']:assert row['dataset_sha256']==sha(D/f"evolver_aux.{row['split']}.jsonl")
manifest['readiness']['diagnostic_tokenizer_length_and_assistant_boundary_passed']=True
manifest['diagnostic_tokenizer_audit']={'file':'diagnostic_tokenizer_audit.json','tokenizer':'Qwen3-4B-Instruct-2507 (diagnostic only; student not locked)','max_tokens':max(r['total_tokens'] for r in audit['rows']),'cap':16384,'no_truncation':True,'template_kwargs':{'enable_thinking':False},'weights_loaded':False}
manifest['audit_attempts']=[{'status':'auditor_error_corrected','reason':'Initial tokenizer audit lacked Transformers 5 token-output normalization and explicit non-thinking template setting; it failed before admission. Corrected auditor passes exact prefix and assistant JSON checks for all 23 rows. No data truncation or student model execution.'}]
for p in [W/'check_open_dataset_tokenizer_only.py',pathlib.Path(__file__)]:shutil.copy2(p,D/'data_build_scripts'/p.name)
with (D/'README.md').open('a',encoding='utf8') as f:f.write('\n另使用服务器现有 Qwen3-4B-Instruct-2507 的 tokenizer 做只读格式检查：23 条均通过，最长 13,257 tokens（16,384 上限），未截断；只监督 assistant 的答案边界已核对。未加载模型权重，训练与推理次数为 0。该检查不等于已选定正式学生模型。\n')
manifest['files']={str(p.relative_to(D)).replace('\\','/'):sha(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='dataset_manifest.json' and '__pycache__' not in p.parts}
(D/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
for rel,h in manifest['files'].items():assert sha(D/rel)==h
archive=B/'research1_open_auxiliary_v2.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(D.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(D)).replace('\\','/'))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
receipt={'artifact':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'server_target':'/home/jinxu/arms_cache/open_dataset_integration_20261101/pilot_v2','counts':manifest['counts'],'files':len(manifest['files'])+1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(B/'release_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(receipt,ensure_ascii=False))
