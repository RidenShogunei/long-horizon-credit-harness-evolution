"""CPU-only format/assistant-boundary audit. Does not import/load student weights."""
import argparse,datetime,hashlib,json,pathlib
from transformers import AutoTokenizer

ap=argparse.ArgumentParser();ap.add_argument('--dataset-dir',required=True);ap.add_argument('--tokenizer',required=True);ap.add_argument('--max-tokens',type=int,default=16384);a=ap.parse_args()
root=pathlib.Path(a.dataset_dir)
tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
def ids(value):
    if hasattr(value,'get'):value=value.get('input_ids')
    if isinstance(value,list) and value and isinstance(value[0],list):value=value[0]
    assert isinstance(value,list) and all(isinstance(i,int) for i in value)
    return value
rows=[]
for split in ['train','validation']:
    path=root/f'evolver_aux.{split}.jsonl'
    for text in path.read_text(encoding='utf8').splitlines():
        row=json.loads(text);m=row['messages'];assert [v['role'] for v in m]==['system','user','assistant']
        plan=json.loads(m[-1]['content']);assert len(plan)==9
        prefix=ids(tok.apply_chat_template(m[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False))
        full=ids(tok.apply_chat_template(m,tokenize=True,add_generation_prompt=False,enable_thinking=False))
        assert full[:len(prefix)]==prefix,'assistant prefix mismatch '+row['sample_id']
        assert len(full)>len(prefix),'empty assistant labels'
        decoded=tok.decode(full[len(prefix):],skip_special_tokens=True).strip()
        assert json.loads(decoded)==plan,'assistant serialization mismatch'
        rows.append({'sample_id':row['sample_id'],'split':split,'total_tokens':len(full),'prompt_tokens':len(prefix),'assistant_tokens':len(full)-len(prefix),'within_cap':len(full)<=a.max_tokens,'full_ids_sha256':hashlib.sha256(json.dumps(full).encode()).hexdigest(),'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
report={'schema':'generic_dataset.diagnostic_tokenizer_audit.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'diagnostic_tokenizer':a.tokenizer,'formal_student_selected':False,'template_kwargs':{'enable_thinking':False},'weights_loaded':False,'inference_calls':0,'optimizer_steps':0,'prompt_labels':'all -100','assistant_labels':'only exact assistant token suffix and terminator','max_tokens':a.max_tokens,'no_truncation':True,'all_within_cap':all(r['within_cap'] for r in rows),'rows':rows}
(root/'diagnostic_tokenizer_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'count':len(rows),'max_total_tokens':max((r['total_tokens'] for r in rows),default=0),'overflow':[r['sample_id'] for r in rows if not r['within_cap']],'weights_loaded':False,'training_steps':0}))
