"""One bounded, hash-verified official dataset download; local filtering, no model use."""
import hashlib,json,pathlib,time,urllib.request
import pyarrow.parquet as pq
R=pathlib.Path(__file__).resolve().parents[1];D=R/'work/open_dataset_integration_v2/published_files';D.mkdir(parents=True,exist_ok=True)
ds='nebius/SWE-rebench-V2'
meta=json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/'+ds+'?blobs=true',timeout=30));rev=meta['sha'];entry=next(r for r in meta['siblings'] if r['rfilename'].endswith('.parquet'))
size=entry['size'];expected=entry['lfs']['sha256'];assert size==428839266 and size<500*1024**2
approval={'source':ds,'revision':rev,'expected_bytes':size,'expected_sha256':expected,'download_cap':500*1024**2,'decision':'Replace failing online filter and slow per-column remote scan with one known 409MB compressed official file, then filter locally. This is data download only; no training, inference or paid evaluation.','official_file':entry['rfilename']}
(D/'download_approval.json').write_text(json.dumps(approval,indent=2))
file=D/'swe_rebench_v2.parquet';part=D/'swe_rebench_v2.parquet.partial'
if not file.exists():
    assert not part.exists(),'Preserve partial; review explicit resume instead of overwriting'
    url=f"https://huggingface.co/datasets/{ds}/resolve/{rev}/{entry['rfilename']}?download=true&source=bounded-open-data-v2"
    request=urllib.request.Request(url,headers={'Accept-Encoding':'identity'})
    h=hashlib.sha256();count=0;t=time.time();next_report=64*1024**2
    with urllib.request.urlopen(request,timeout=60) as response,part.open('xb') as output:
        assert response.status==200,(response.status,response.headers.get('Content-Range'))
        while True:
            b=response.read(4*1024**2)
            if not b:break
            count+=len(b);assert count<=size;output.write(b);h.update(b)
            if count>=next_report:print(json.dumps({'downloaded_mib':round(count/1024**2),'elapsed_seconds':round(time.time()-t)}),flush=True);next_report+=64*1024**2
    assert count==size and h.hexdigest()==expected,('download mismatch',count,h.hexdigest())
    assert part.resolve().is_relative_to(D.resolve()) and file.resolve().is_relative_to(D.resolve())
    part.rename(file)
assert file.stat().st_size==size
actual=hashlib.sha256(file.read_bytes()).hexdigest();assert actual==expected
pf=pq.ParquetFile(file);small=pf.read(columns=['repo','instance_id']).to_pylist();counts={}
for row in small:counts[row['repo']]=counts.get(row['repo'],0)+1
(D/'repository_inventory.json').write_text(json.dumps(counts,indent=2,sort_keys=True))
allow=['agent','aider','langchain','langgraph','crewai','autogen','llama','letta','dspy','haystack','litellm','pydantic-ai','openai','anthropic','agno','phidata','semantic-kernel','adk','mem0','dify','flowise','continue','promptflow','taskweaver','botpress']
matches={repo:n for repo,n in counts.items() if any(s in repo.lower() for s in allow)}
(D/'harness_repository_inventory.json').write_text(json.dumps(matches,indent=2,sort_keys=True))
result={**approval,'status':'DONE','actual_sha256':actual,'total_rows':pf.metadata.num_rows,'repository_matches':matches,'no_training':True}
(D/'download_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
