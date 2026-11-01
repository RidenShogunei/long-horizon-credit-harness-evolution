"""Freeze missing read-only interfaces at the authenticated parent, never at the repair."""
import ast,hashlib,json,pathlib,urllib.request
W=pathlib.Path(__file__).resolve().parent;B=W/'open_dataset_integration_v2';sid='crewaiinc__crewai-1818'
r=next(json.loads(s) for s in (B/'main_candidates/raw_records.jsonl').read_text(encoding='utf8').splitlines() if json.loads(s)['instance_id']==sid)
refs=[]
for rel in ['src/crewai/agent.py','src/crewai/knowledge/source/string_knowledge_source.py']:
    p=B/'main_candidates/cases'/sid/'before'/rel
    url=f"https://raw.githubusercontent.com/{r['repo']}/{r['base_commit']}/{rel}"
    if not p.exists():
        with urllib.request.urlopen(url,timeout=20) as response:data=response.read(256*1024+1)
        assert len(data)<=256*1024;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(data)
    text=p.read_text(encoding='utf8');tree=ast.parse(text)
    matches=[n for n in ast.walk(tree) if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in ['knowledge_sources','content']]
    assert matches,(rel,'missing parent field')
    for n in matches:
        segment='\n'.join(text.splitlines()[n.lineno-1:n.end_lineno])
        refs.append({'artifact_path':str(p.relative_to(W.parent)).replace('\\','/'),'original_path':rel,'field':n.target.id,'base_commit':r['base_commit'],'start_line':n.lineno,'end_line':n.end_lineno,'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'excerpt':segment,'url':url})
(B/'root_parent_interface_refs.json').write_text(json.dumps({'sample_id':sid,'refs':refs},indent=2),encoding='utf8')
print(json.dumps({'frozen_parent_interface_fields':[(v['original_path'],v['field']) for v in refs]}))
