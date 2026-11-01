import ast,concurrent.futures,hashlib,json,pathlib,re,subprocess,threading,urllib.parse,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];D=(R/'work/open_dataset_integration_v2/additional_crewai').resolve();(D/'cases').mkdir(exist_ok=True)
old=[json.loads(s) for s in (R/'work/open_dataset_integration_v2/main_candidates/raw_records.jsonl').read_text(encoding='utf8').splitlines()]
def canonical(row):
    n=row.get('pull_number');meta=row.get('meta') or {}
    if not n:
        u=(meta.get('pr_url') or '') if isinstance(meta,dict) else '';m=re.search(r'/pull/(\d+)',u);n=int(m.group(1)) if m else None
    return (row['repo'].lower(),n) if n else None
seen={canonical(r) for r in old if canonical(r)}
raw=[json.loads(s) for s in (D/'raw_records.jsonl').read_text(encoding='utf8').splitlines()]
targets={1009,1382,2370,2424,2428,2477,2637,2865,2890,1910,1791,2604}
selected=[r for r in raw if r.get('pull_number') in targets and canonical(r) not in seen]
seen_patches={(r['repo'].lower(),hashlib.sha256(r['patch'].encode()).hexdigest()) for r in old}
selected=[r for r in selected if (r['repo'].lower(),hashlib.sha256(r['patch'].encode()).hexdigest()) not in seen_patches]
(D/'selected_raw_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf8')
tree=ast.parse((R/'work/select_and_reconstruct_v2.py').read_text(encoding='utf8'));defs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['get','reconstruct']]
ns=globals();ns.update(total_bytes=0,lock=threading.Lock(),cache={},bytecap=60*1024**2)
exec(compile(ast.Module(body=defs,type_ignores=[]),'<reviewed-source-reconstructor>','exec'),ns)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(ns['reconstruct'],selected))
(D/'reconstruction_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
out={'raw_public_pr_records':len(raw),'selected_after_semantic_prefilter_and_pr_dedup':len(selected),'reconstructed':sum(r['status']=='reconstructed' for r in results),'selected_ids':[r['source_id'] for r in results],'cross_dataset_pr_overlap_excluded':[r['instance_id'] for r in raw if canonical(r) in seen]}
(D/'selected_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
