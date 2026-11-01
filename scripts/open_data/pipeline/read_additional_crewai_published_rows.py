"""Retrieve one already indexed official PR-data row group; no full 2.6GB download."""
import ast,hashlib,io,json,pathlib,urllib.request
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];D=R/'work/open_dataset_integration_v2/additional_crewai';D.mkdir(parents=True,exist_ok=True)
tree=ast.parse((R/'work/read_published_parquet_candidates.py').read_text(encoding='utf8'));defs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and getattr(n,'name',None) in ['fetch','RangeFile']]
ns=dict(io=io,urllib=urllib,BUDGET=60*1024**2,used=0,blockbytes=65536);exec(compile(ast.Module(body=defs,type_ignores=[]),'<reviewed-range-reader>','exec'),ns)
Base=ns['RangeFile']
class FastRange(Base):
    def read(self,n=-1):
        if n>131072:
            n=min(n,self.size-self.pos);start=self.pos
            b,status,cr=ns['fetch'](self.url,n,{'Range':f'bytes={start}-{start+n-1}'})
            assert status==206 and cr.startswith(f'bytes {start}-')
            self.pos+=len(b);return b
        return super().read(n)
url='https://huggingface.co/datasets/nebius/SWE-rebench-V2-PRs/resolve/fbf0ecf50f268d5344149e2f0097db6bede83737/data/train-00001-of-00003.parquet?source=bounded-crewai-slice'
f=FastRange(url,946882400);pf=pq.ParquetFile(f)
cols=[c for c in ['repo','instance_id','base_commit','patch','problem_statement','pr_description','pull_number','license','test_patch','FAIL_TO_PASS','PASS_TO_PASS','interface'] if c in pf.schema_arrow.names]
data=pf.read_row_group(5,columns=cols,use_threads=False).to_pylist();rows=[r for r in data if r['repo']=='crewAIInc/crewAI']
old={r['instance_id'] for r in map(json.loads,(R/'work/open_dataset_integration_v2/main_candidates/raw_records.jsonl').read_text(encoding='utf8').splitlines())}
new=[]
for row in rows:
    if row['instance_id'] in old:continue
    row['_source']={'dataset':'nebius/SWE-rebench-V2-PRs','revision':'fbf0ecf50f268d5344149e2f0097db6bede83737','file':'data/train-00001-of-00003.parquet','row_group':5,'license_compilation':'CC-BY-4.0','unverified_upstream_task':True};new.append(row)
(D/'raw_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new),encoding='utf8')
(D/'source_manifest.json').write_text(json.dumps({'rows_in_repo':len(rows),'new_after_main_dedup':len(new),'http_bytes':ns['used'],'no_training':True,'no_acceptance_claim':True},indent=2))
print(json.dumps({'crewai_public_pr_records':len(rows),'new_unique':len(new),'http_bytes':ns['used']}),flush=True)
