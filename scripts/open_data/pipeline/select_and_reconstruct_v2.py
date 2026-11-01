"""Filter real published tasks locally and reconstruct pinned before/after source, no code execution."""
import concurrent.futures,hashlib,json,pathlib,re,subprocess,threading,urllib.parse,urllib.request
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];D=R/'work/open_dataset_integration_v2/main_candidates';D.mkdir(parents=True,exist_ok=True)
D=D.resolve();(D/'cases').mkdir(exist_ok=True)
P=R/'work/open_dataset_integration_v2/published_files/swe_rebench_v2.parquet'
receipt=json.loads((P.parent/'download_receipt.json').read_text());assert receipt['actual_sha256']=='0e0bf9355f892ad74ae98d4e1c404f39fd6654a8e351ee3e6ab162e4a64cd3ad'
repos=['crewAIInc/crewAI','stanfordnlp/dspy','mem0ai/mem0','BerriAI/litellm']
cols=['instance_id','repo','base_commit','patch','problem_statement','pr_description','license','test_patch','FAIL_TO_PASS','PASS_TO_PASS','interface','meta']
table=pq.read_table(P,columns=cols,filters=[('repo','in',repos)],use_threads=False)
rows=table.to_pylist()
for row in rows:row['_source']={'dataset':'nebius/SWE-rebench-V2','revision':receipt['revision'],'parquet_sha256':receipt['actual_sha256'],'license_compilation':'CC-BY-4.0'}
(D/'raw_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
total_bytes=0;lock=threading.Lock();cache={};bytecap=60*1024**2
def get(repo,commit,path):
    global total_bytes
    key=(repo,commit,path)
    with lock:
        if key in cache:return cache[key]
    url='https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+urllib.parse.quote(path,safe='/')
    with urllib.request.urlopen(url,timeout=25) as f:blob=f.read(2*1024**2+1)
    if len(blob)>2*1024**2:raise RuntimeError('source file exceeds2MB limit')
    with lock:
        total_bytes+=len(blob);assert total_bytes<=bytecap;cache[key]=blob
    return blob
def reconstruct(row):
    sid=row['instance_id'];safe=re.sub('[^A-Za-z0-9_.-]','_',sid);dest=(D/'cases'/safe).resolve();assert dest.is_relative_to(D);dest.mkdir(parents=True,exist_ok=True)
    saved=dest/'reconstruction.json'
    if saved.exists():return json.loads(saved.read_text(encoding='utf8'))
    result={'source_id':sid,'repo':row['repo'],'base_commit':row['base_commit'],'status':'pending','source_files':{},'patch_sha256':hashlib.sha256(row['patch'].encode()).hexdigest()}
    try:
        assert re.fullmatch('[0-9a-f]{40}',row['base_commit'])
        if len(row['patch'])>400000:raise RuntimeError('patch exceeds400k pilot size')
        parts=[p for p in re.split(r'(?=^diff --git )',row['patch'],flags=re.M) if p.strip()]
        paths=[]
        for part in parts:
            old=re.search(r'^--- a/(.+)$',part,re.M);new=re.search(r'^\+\+\+ b/(.+)$',part,re.M)
            if old:
                path=old.group(1).strip();pp=pathlib.PurePosixPath(path);assert not pp.is_absolute() and not any(x in ['..','.git'] for x in pp.parts)
                blob=get(row['repo'],row['base_commit'],path);idx=re.search(r'^index ([0-9a-f]+)\.\.([0-9a-f]+)',part,re.M);obj=hashlib.sha1(('blob '+str(len(blob))+'\0').encode()+blob).hexdigest()
                if idx and not obj.startswith(idx.group(1)):raise RuntimeError('old gitblob mismatch '+path)
                before=dest/'before'/path;before.parent.mkdir(parents=True,exist_ok=True);before.write_bytes(blob)
                active=dest/'applied'/path;active.parent.mkdir(parents=True,exist_ok=True);active.write_bytes(blob)
                result['source_files'][path]={'before':str(before.relative_to(R)).replace('\\','/'),'before_sha256':hashlib.sha256(blob).hexdigest(),'before_gitblob':obj}
            if new:paths.append(new.group(1).strip())
        result['changed_paths']=paths
        work=dest/'applied';work.mkdir(exist_ok=True);subprocess.run(['git','init','--quiet',str(work)],check=True,capture_output=True)
        patch=dest/'original.patch';patch.write_text(row['patch'],encoding='utf8',newline='\n')
        chk=subprocess.run(['git','-C',str(work),'apply','--check','--whitespace=nowarn',str(patch)],capture_output=True,text=True,timeout=20)
        if chk.returncode:raise RuntimeError('patchcheck '+chk.stderr[-1200:])
        apply=subprocess.run(['git','-C',str(work),'apply','--whitespace=nowarn',str(patch)],capture_output=True,text=True,timeout=20)
        if apply.returncode:raise RuntimeError('patchapply '+apply.stderr[-1200:])
        for path in paths:
            f=(work/path).resolve();assert f.is_relative_to(work.resolve())
            if f.is_file():result['source_files'].setdefault(path,{}).update(after=str(f.relative_to(R)).replace('\\','/'),after_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        result.update(status='reconstructed',before_after_executed_behavior=False,original_patch=str(patch.relative_to(R)).replace('\\','/'))
    except Exception as e:result['error']=str(e)
    (dest/'reconstruction.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8');return result
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    for result in pool.map(reconstruct,rows):
        results.append(result)
        (D/'reconstruction_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
        if len(results)%5==0:print(json.dumps({'completed':len(results),'total':len(rows),'reconstructed':sum(r['status']=='reconstructed' for r in results),'http_source_bytes':total_bytes}),flush=True)
from collections import Counter
summary={'raw_rows':len(rows),'raw_repos':dict(Counter(r['repo'] for r in rows)),'reconstructed':sum(r['status']=='reconstructed' for r in results),'pending':sum(r['status']!='reconstructed' for r in results),'source_http_bytes':total_bytes,'no_training':True}
(D/'collection_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
