"""Read repository columns and selected row groups from pinned public Parquet by HTTP ranges."""
import collections,hashlib,io,json,pathlib,sys,time,urllib.request
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];D=R/'work/open_dataset_integration_v2/parquet_acquisition';D.mkdir(parents=True,exist_ok=True)
ALLOW={x.lower() for x in ['huggingface/smolagents','SWE-agent/SWE-agent','SWE-agent/mini-swe-agent','Aider-AI/aider','All-Hands-AI/OpenHands','OpenDevin/OpenDevin','langchain-ai/langchain','langchain-ai/langgraph','microsoft/autogen','ag2ai/ag2','crewAIInc/crewAI','run-llama/llama_index','langroid/langroid','openai/openai-agents-python','google/adk-python','mastra-ai/mastra','pydantic/pydantic-ai','deepset-ai/haystack']}
BUDGET=200*1024*1024;used=0;raw=[];inventory=[];sources=[];blockbytes=65536
def fetch(url,limit=2000000,headers=None):
    global used
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=40) as f:
        data=f.read(min(limit,BUDGET-used)+1);code=f.status;cr=f.headers.get('Content-Range')
    used+=len(data)
    if len(data)>limit or used>BUDGET:raise RuntimeError('HTTP byte budget exhausted')
    return data,code,cr
class RangeFile(io.RawIOBase):
    def __init__(self,url,size):super().__init__();self.url=url;self.size=size;self.pos=0;self.cache={}
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n> BUDGET-used:raise RuntimeError('one read exceeds remaining HTTP budget')
        out=[]
        while n>0:
            ix=self.pos//blockbytes;start=ix*blockbytes;end=min(start+blockbytes,self.size)-1
            if ix not in self.cache:
                data,status,cr=fetch(self.url,end-start+1,{'Range':f'bytes={start}-{end}','User-Agent':'research-data-integration'})
                if status!=206 or not cr or not cr.startswith(f'bytes {start}-'):raise RuntimeError(f'Range not honored: {status} {cr}')
                self.cache[ix]=data
            b=self.cache[ix];off=self.pos-start;take=min(n,len(b)-off)
            if take<=0:raise RuntimeError('short range read')
            out.append(b[off:off+take]);self.pos+=take;n-=take
        return b''.join(out)
    def readinto(self,buf):b=self.read(len(buf));buf[:len(b)]=b;return len(b)
def persist():
    (D/'raw_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw),encoding='utf8')
    (D/'inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inventory),encoding='utf8')
    (D/'sources.json').write_text(json.dumps({'http_bytes_used':used,'http_byte_cap':BUDGET,'sources':sources,'raw_rows':len(raw),'raw_repos':dict(collections.Counter(r['repo'] for r in raw)),'no_training':True},indent=2),encoding='utf8')
for ds in ['nebius/SWE-rebench-V2','nebius/SWE-rebench-V2-PRs']:
    try:
        meta=json.loads(fetch('https://huggingface.co/api/datasets/'+ds+'?blobs=true')[0]);rev=meta['sha']
        files=[x for x in meta['siblings'] if x['rfilename'].endswith('.parquet')]
        src={'dataset':ds,'revision':rev,'files':[{'name':x['rfilename'],'size':x.get('size',x.get('lfs',{}).get('size'))} for x in files],'errors':[]};sources.append(src)
        print(json.dumps({'dataset':ds,'files':src['files']}),flush=True)
        for entry in files:
            fn=entry['rfilename'];size=entry.get('size',entry.get('lfs',{}).get('size'));url=f'https://huggingface.co/datasets/{ds}/resolve/{rev}/{fn}'
            rf=RangeFile(url,size);pf=pq.ParquetFile(rf);cols=pf.schema_arrow.names
            assert 'repo' in cols
            matches=[]
            for rg in range(pf.num_row_groups):
                repos=pf.read_row_group(rg,columns=['repo','instance_id']).to_pylist()
                found=[(i,r) for i,r in enumerate(repos) if (r['repo'] or '').lower() in ALLOW]
                if found:
                    counts=collections.Counter(r['repo'] for _,r in found);matches.append((rg,found,counts))
                    inventory.extend({'dataset':ds,'revision':rev,'file':fn,'row_group':rg,'row_in_group':i,**r} for i,r in found)
            print(json.dumps({'dataset':ds,'file':fn,'groups':pf.num_row_groups,'matched_rows':sum(len(x[1]) for x in matches),'repos':dict(collections.Counter(x['repo'] for x in inventory if x['dataset']==ds)),'bytes':used}),flush=True);persist()
            # Diversify repositories, prioritize groups with the most relevant rows per compressed byte.
            required=[c for c in ['repo','instance_id','base_commit','patch','problem_statement','pr_description','license','test_patch','FAIL_TO_PASS','PASS_TO_PASS','pull_number','interface','meta','language'] if c in cols]
            matches.sort(key=lambda x:-len(x[1])/max(1,pf.metadata.row_group(x[0]).total_byte_size))
            perrepo=collections.Counter(r['repo'] for r in raw)
            for rg,found,counts in matches:
                if len(raw)>=150 or used>=BUDGET*0.92:break
                if all(perrepo[r]>=35 for r in counts):continue
                try:
                    table=pf.read_row_group(rg,columns=required);idx=[i for i,_ in found]
                    selected=table.take(idx).to_pylist()
                    for row in selected:
                        if perrepo[row['repo']]>=35 or len(raw)>=150:continue
                        if len(row.get('patch') or '')>400000:continue
                        row['_source']={'dataset':ds,'revision':rev,'file':fn,'row_group':rg,'license_compilation':'CC-BY-4.0','content_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
                        raw.append(row);perrepo[row['repo']]+=1
                    persist();print(json.dumps({'selected_rows':len(raw),'repos':dict(perrepo),'bytes':used}),flush=True)
                except Exception as e:src['errors'].append({'file':fn,'row_group':rg,'error':str(e)});persist();break
            del pf,rf
            if len(raw)>=150 or used>=BUDGET*0.92:break
    except Exception as e:
        sources.append({'dataset':ds,'error':str(e)});print(json.dumps({'dataset':ds,'error':str(e)}),flush=True);persist()
    if len(raw)>=150 or used>=BUDGET*0.92:break
persist();print(json.dumps({'status':'DONE','raw_rows':len(raw),'repos':dict(collections.Counter(r['repo'] for r in raw)),'bytes':used}),flush=True)
