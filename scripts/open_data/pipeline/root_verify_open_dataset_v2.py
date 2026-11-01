"""Independent root reruns of reviewed local data probes; no model/evaluation calls."""
import argparse,contextlib,datetime,hashlib,importlib.util,io,json,pathlib,subprocess,sys
from concurrent.futures import ThreadPoolExecutor

W=pathlib.Path(__file__).resolve().parent
B=W/'open_dataset_integration_v2'
OUT=B/'root_rerun';OUT.mkdir(exist_ok=True)
SPECS={
 'acceptance_strengthened':('verify_samples.py',['285','891','1088','478','226']),
 'dspy_strict':('verify_batch.py',['8089','8340','8053']),
 'crewai_strict':('verify_batch.py',['2570','3150','2588','1818','2408']),
 'litellm_strict':('verify_batch.py',['11964','9261','5232','12168','9828','8512']),
 'crewai_pr_strict':('verify_batch.py',['1910','2424','2890','2865','1791']),
}
def child(batch,case):
    path=B/batch/SPECS[batch][0]
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('root_probe',path)
    mod=importlib.util.module_from_spec(spec);sys.modules['root_probe']=mod
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf),contextlib.redirect_stderr(buf):
        spec.loader.exec_module(mod)
        if batch=='dspy_strict':
            mod.FIXTURES=OUT/'dspy_fixtures'
        if batch=='crewai_strict':mod.ensure_workspace_root()
        if batch=='litellm_strict':
            fn=mod.RUNNERS[next(k for k in mod.RUNNERS if k.endswith('-'+case))]
            checks=[v.to_json() for v in fn()]
            res={'source_id':'berriai__litellm-'+case,'status':'pass' if checks and all(x['status']=='pass' for x in checks) else 'reject','checks':checks}
        else:
            res=getattr(mod,'verify_'+case)()
        if case=='226':
            res['worker_original_status']=res['status'];res['status']='reject'
            res['root_reason']='Wildcard authorized_imports constructor regresses with missing logger; worker skipped failure is never PASS.'
    payload={'batch':batch,'case':case,'result':res,'captured_log':buf.getvalue()[-12000:],
      'probe_script_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'ran_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    print(json.dumps(payload,ensure_ascii=False))

def run_one(pair):
    batch,case=pair;target=OUT/(batch+'-'+case+'.json')
    try:
        p=subprocess.run([sys.executable,'-X','utf8',__file__,'--batch',batch,'--case',case],capture_output=True,text=True,encoding='utf8',timeout=45)
        if p.returncode:res={'batch':batch,'case':case,'result':{'status':'pending','reason':'probe_error'},'stdout':p.stdout[-3000:],'stderr':p.stderr[-4000:]}
        else:res=json.loads(p.stdout)
    except subprocess.TimeoutExpired:res={'batch':batch,'case':case,'result':{'status':'pending','reason':'root_child_timeout'}}
    target.write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf8')
    return {'batch':batch,'case':case,'status':res['result']['status'],'proof':str(target.relative_to(W.parent)).replace('\\','/')}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--batch');ap.add_argument('--case');a=ap.parse_args()
    if a.batch:child(a.batch,a.case)
    else:
        pairs=[(batch,case) for batch,(_,cases) in SPECS.items() for case in cases]
        with ThreadPoolExecutor(max_workers=3) as ex:rows=list(ex.map(run_one,pairs))
        result={'executed_cases':len(rows),'counts':{s:sum(r['status']==s for r in rows) for s in ['pass','pending','reject']},'rows':rows,'model_calls':0,'paid_evaluations':0}
        (OUT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
