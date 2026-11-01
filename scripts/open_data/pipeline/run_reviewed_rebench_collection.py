"""Run actual published dataset ingestion; no repo clone or model calls."""
import hashlib,importlib.util,json,pathlib,re,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];D=R/'work/open_dataset_integration_v1/rebench'
spec=importlib.util.spec_from_file_location('rebench',D/'collect_pilot.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
cache={};original=m.fetch_source_file
def fetch(repo,commit,path):
    assert re.fullmatch(r'[\w.-]+/[\w.-]+',repo) and re.fullmatch('[0-9a-f]{40}',commit)
    pp=pathlib.PurePosixPath(path);assert not pp.is_absolute() and not any(x in ['..','.git'] for x in pp.parts)
    key=(repo,commit,path)
    if key not in cache:cache[key]=original(repo,commit,path)
    return cache[key]
m.fetch_source_file=fetch
def apply_check(repo,commit,patch):
    sid=hashlib.sha256((repo+commit+patch).encode()).hexdigest()[:20]
    dest=(D/'replays'/sid).resolve();assert dest.is_relative_to(D.resolve());dest.mkdir(parents=True,exist_ok=False)
    checks=[]
    for part in re.split(r'(?=^diff --git )',patch,flags=re.M):
        old=re.search(r'^--- a/(.+)$',part,re.M)
        if old:
            path=old.group(1);blob=fetch(repo,commit,path)
            if blob is None:return {'ok':False,'error':'missing required before source','path':path}
            f=(dest/path).resolve();assert f.is_relative_to(dest);f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(blob)
            idx=re.search(r'^index ([0-9a-f]+)\.\.([0-9a-f]+)',part,re.M)
            gitsha=hashlib.sha1(('blob '+str(len(blob))+'\0').encode()+blob).hexdigest()
            if idx and not gitsha.startswith(idx.group(1)):return {'ok':False,'error':'base git blob mismatch','path':path,'actual':gitsha,'expected_prefix':idx.group(1)}
            checks.append({'path':path,'before_sha256':hashlib.sha256(blob).hexdigest(),'git_blob_sha1':gitsha})
    subprocess.run(['git','init','--quiet',str(dest)],check=True,capture_output=True)
    file=dest/'original_source_patch.diff';file.write_text(patch,encoding='utf8',newline='\n')
    cmd=['git','-C',str(dest),'apply','--check','--whitespace=nowarn',str(file)]
    chk=subprocess.run(cmd,capture_output=True,text=True)
    if chk.returncode:return {'ok':False,'stderr':chk.stderr[-1500:],'before_blob_checks':checks}
    applied=subprocess.run(['git','-C',str(dest),'apply','--whitespace=nowarn',str(file)],capture_output=True,text=True)
    return {'ok':applied.returncode==0,'source':'actual downloaded before files, original full dataset patch, no fuzz or repair','before_blob_checks':checks,'after_tree':str(dest.relative_to(D)),'stderr':applied.stderr[-1000:]}
m.git_apply_check=apply_check
raise SystemExit(m.main())
