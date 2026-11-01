"""Root checks for source-dependency binding and tool empty-argument preservation."""
import ast,hashlib,importlib.util,json,pathlib,sys,typing,urllib.request,copy
W=pathlib.Path(__file__).resolve().parent;B=W/'open_dataset_integration_v2';O=B/'root_rerun'
sys.path.insert(0,str(B/'litellm_strict'))
from runtime.pinned_import import extract_class_from_pinned,extract_function_from_pinned

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
case=B/'main_candidates/cases/berriai__litellm-9828'
row=next(json.loads(t) for t in (B/'main_candidates/raw_records.jsonl').read_text(encoding='utf8').splitlines() if json.loads(t)['instance_id']=='berriai__litellm-9828')
rel='litellm/litellm_core_utils/prompt_templates/common_utils.py';dep=case/'before'/rel
url=f"https://raw.githubusercontent.com/{row['repo']}/{row['base_commit']}/{rel}"
if not dep.exists():
    with urllib.request.urlopen(url,timeout=20) as response:content=response.read(512*1024+1)
    assert len(content)<=512*1024
    dep.parent.mkdir(exist_ok=True,parents=True);dep.write_bytes(content)
# The published patch has no change to this dependency; both variants use exact parent content.
assert not any(line.startswith('diff --git ') and rel in line for line in (case/'original.patch').read_text(encoding='utf8').splitlines())
after_dep=case/'applied'/rel;after_dep.parent.mkdir(parents=True,exist_ok=True);after_dep.write_bytes(dep.read_bytes())

class AnthropicError(Exception):pass
def empty_probe(variant):
    cls=extract_class_from_pinned('berriai__litellm-5232',variant,'llms/anthropic.py','ModelResponseIterator',{'AnthropicError':AnthropicError})
    it=cls(None,True)
    chunks=[{'type':'content_block_start','index':1,'content_block':{'type':'tool_use','id':'t1','name':'f1','input':{}}},{'type':'content_block_delta','index':1,'delta':{'type':'input_json_delta','partial_json':''}},{'type':'content_block_stop','index':1}]
    out=[it.chunk_parser(ch) for ch in chunks]
    return [{'arguments':r['tool_use']['function']['arguments'],'index':r['tool_use']['index']} for r in out if r.get('tool_use')]
b=empty_probe('before');a=empty_probe('applied')
tool_result={'status':'pass' if b[-1]['arguments']==a[-1]['arguments']=='{}' and b[0]['arguments']==a[0]['arguments']=='' and all(x['index']==0 for x in a) else 'reject','before':b,'after':a,'root_correction':'Worker selected first emitted tool chunk instead of stop chunk. Correct preservation compares the complete sequence; both emit empty start/delta and {} at stop.'}

# Use all functions from actual pinned 9828 code, including actual same-revision unpack_defs.
src=B/'root_scope_extras/verify_extras.py';spec=importlib.util.spec_from_file_location('extra_gap_module',src);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def pinned_vertex_extra(variant):
    extra=m._vertex_extra_9828(variant)
    extra['unpack_defs']=extract_function_from_pinned('berriai__litellm-9828',variant,'litellm_core_utils/prompt_templates/common_utils.py','unpack_defs',{'Set':set,'Any':typing.Any})
    return extra
m._vertex_extra_9828_original=m._vertex_extra_9828
def bound_vertex_extra(variant):
    extra=m._vertex_extra_9828_original(variant)
    extra['unpack_defs']=extract_function_from_pinned('berriai__litellm-9828',variant,'litellm_core_utils/prompt_templates/common_utils.py','unpack_defs',{'Set':set,'Any':typing.Any})
    return extra
m._vertex_extra_9828=bound_vertex_extra
vertex=m.extra_litellm_9828()
assert vertex['status']=='pass'
caller_observations={}
for variant in ['before','applied']:
    extra=bound_vertex_extra(variant)
    builder=extract_function_from_pinned('berriai__litellm-9828',variant,'llms/vertex_ai/common_utils.py','_build_vertex_schema',extra)
    caller=extract_function_from_pinned('berriai__litellm-9828',variant,'llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py','_map_response_schema',{'deepcopy':copy.deepcopy,'_build_vertex_schema':builder})
    caller_observations[variant]=caller(None,{'type':'object','properties':{'z':{'type':'string'},'a':{'type':'integer'}}})
assert 'propertyOrdering' not in caller_observations['before']
assert caller_observations['applied']['propertyOrdering']==['z','a']
vertex['caller_propagation']={'status':'pass','before':caller_observations['before'],'after':caller_observations['applied'],'method':'VertexGeminiConfig._map_response_schema'}
vertex['source_binding']={'repo':row['repo'],'base_commit':row['base_commit'],'dependency_url':url,'dependency_sha256':sha(dep),'same_pinned_dependency_for_both_variants':True}
result={'berriai__litellm-5232':tool_result,'berriai__litellm-9828':vertex,'reviewed_worker_script_sha256':sha(src),'models_loaded':False,'paid_evaluations':0}
(O/'last_gap_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({'5232':tool_result['status'],'9828':vertex['status']}))
