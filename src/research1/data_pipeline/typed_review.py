"""Typed independent factual review; code is never relocated from a bad slot.

Historical review prompts/results are untouched. A new explicit policy is used
for a fresh review of an unchanged saved plan. Exact references establish what
was visible, not optimality, causal benefit or an implementation verdict.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from research1.data_pipeline.policy import (
    FACTUAL_WITNESS_VALIDATOR, canonical, digest, public_view, validate_plan,
    _atomic_json, _review_witnesses,
)

POLICY_ID = 'proposal-factual-review.v6.typed'
CODE_WITNESS_VALIDATOR = 'proposal-code-witness.v1'
_INSTRUCTIONS = (
    'Factual review policy: ' + POLICY_ID + '. Independently check this original single plan against '
    'ONLY its frozen decision-time proposal evidence, visible module source and read-only framework '
    'interfaces. Keep factual accuracy separate from action optimality. A bounded keep needs substantive '
    'actual execution facts and evidence limits; task success or merely not seeing errors is insufficient. '
    'A factually supported keep does not prove no improvement exists or that keep is best. For edit, '
    'check observed facts and current-code/interface facts separately from explicitly unproven expected '
    'effects. A concrete process limitation may support a plausible implementable improvement even '
    'when the baseline operates as designed. Proven code defect, causal responsibility for task outcomes, '
    'solver compliance, future performance gain or demonstrated benefit are not required in advance. '
    'Unsupported causal guarantees, global absence claims from bounded excerpts, uncertain source facts '
    'or impossible runtime data paths must remain pending_review. Check the declared single-module scope '
    'and whether needed runtime values can reach the proposed functions; do not silently rewrite the plan '
    'or add another hook. This is factual review, not executable behavior verification or optimality proof. '
    'Prior proposal outcome/failure_layer is only a recorded classification and does not prove a cause. '
    'Return final JSON with exactly these fields: '
    '{"status":"PASS|pending_review|FAIL","reason":"...","evidence_refs":["original evidence id"],'
    '"witnesses":[{"evidence_ref":"original evidence id","text":"exact original visible substring",'
    '"kind":"observed_tool_output|execution_status|check_status|failure_behavior|context"}],'
    '"code_witnesses":[{"path":"visible module path or read-only framework source path",'
    '"text":"exact source substring"}]}. '
    'IMPORTANT TYPE BOUNDARY: witnesses may quote only original supplied proposal evidence referenced by '
    'evidence_refs. Source-only quotations from module_code or framework_context MUST go in code_witnesses, '
    'using their actual source path, never a trajectory evidence_ref. Do not assign source text to a '
    'trajectory because the plan discusses both. Wrong-slot quotations will be rejected, not relocated. '
    'For PASS provide exact witnesses for the concrete proposal facts and exact code_witnesses for relevant '
    'source/dataflow facts. An edit needs at least one source quotation from its declared editing location. '
    'A keep may use an empty code_witnesses list when its factual claims need no source quotation. '
    'Quote original decoded characters, not JSON escape representations. Context quotes may identify task '
    'requirements, commands or metadata; commands do not establish their output or success. Actual captured '
    'output does not by itself establish its causal producer, freshness, HTTP/exit/test success or task '
    'correctness. Use execution_status/check_status only with the corresponding substantive status observation. '
    'A printed test claim is reported evidence, not proof that requirements are correct. No new tests, '
    'patches, changed plans, future trajectories or reward/dev information may be introduced.\n'
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text_sha(text: str) -> str:
    return _sha(text.encode('utf-8'))


def _read(path: Path) -> tuple[dict, str]:
    path=Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Typed review records must be existing absolute regular files')
    raw=path.read_bytes();value=json.loads(raw)
    if not isinstance(value,dict):
        raise ValueError('Typed review record must be an object')
    return value,_sha(raw)


def _metadata(metadata: dict, state: dict) -> None:
    if (not isinstance(metadata,dict) or metadata.get('raw_field')!='content'
            or metadata.get('finish_reason')!='stop' or metadata.get('temperature')!=0):
        raise ValueError('Typed factual review requires complete final JSON at temperature zero')
    for key in ('provider','model'):
        expected=state.get('review_'+key) or state.get('teacher_'+key)
        if expected and metadata.get(key)!=expected:
            raise ValueError('Typed factual reviewer differs from the frozen policy')


def _request(state: dict,plan: dict) -> dict:
    if state.get('factual_review_policy')!=POLICY_ID:
        raise ValueError('Typed factual review requires its explicitly frozen policy')
    view=public_view(state)
    prompt=_INSTRUCTIONS+'Frozen visible view:\n'+canonical(view)+'\nCandidate:\n'+canonical(plan)
    return {'review_policy':POLICY_ID,'request_id':digest({'policy':POLICY_ID,'view':view,'plan':plan}),
            'prompt':prompt,'prompt_sha256':_text_sha(prompt),
            'visible_view_sha256':digest(view),'plan_sha256':digest(plan)}


def _validate_code(value: Any,view: dict,plan: dict) -> list[dict]:
    if not isinstance(value,list):
        raise ValueError('code_witnesses must be a list')
    sources={path:[{'source_kind':'module_code','path':path,'text':text,
                   'source_sha256':_text_sha(text)}] for path,text in view['module_code'].items()}
    for fragment in view.get('framework_context',{}).get('source_fragments',[]):
        sources.setdefault(fragment['path'],[]).append({'source_kind':'framework_fragment',
            'path':fragment['path'],'symbol':fragment['symbol'],'text':fragment['text'],
            'source_sha256':fragment['source_sha256'],'excerpt_sha256':fragment['excerpt_sha256'],
            'start_line':fragment['start_line']})
    validated=[];seen=set()
    for row in value:
        if (not isinstance(row,dict) or set(row)!={'path','text'}
                or not isinstance(row.get('path'),str) or row['path'] not in sources
                or not isinstance(row.get('text'),str) or not row['text'].strip()):
            raise ValueError('Source witness must name an original visible source path and exact text')
        signature=(row['path'],row['text'])
        if signature in seen:
            raise ValueError('Repeated source witness')
        seen.add(signature)
        matches=[]
        for source in sources[row['path']]:
            cursor=0
            while True:
                start=source['text'].find(row['text'],cursor)
                if start<0:break
                matches.append({**{key:value for key,value in source.items() if key!='text'},
                    'offset_start':start,'offset_end':start+len(row['text'])})
                cursor=start+len(row['text'])
        if not matches:
            raise ValueError('Source witness is absent from the original visible source')
        validated.append(dict(row,validation={'validator':CODE_WITNESS_VALIDATOR,
            'evidence_type':'source_text_only','execution_success_established':False,'matches':matches}))
    if plan['action']=='edit' and not any(row['path'] in plan['locations'] for row in value):
        raise ValueError('Edit factual PASS needs a source witness for its declared editing location')
    return validated


def _base(state: dict,plan: dict,request: dict,raw_path: Path,metadata: dict|None=None) -> dict:
    metadata=metadata or {}
    return {'status':'pending_review','reason':'Typed factual review is not validated','evidence_refs':[],
        'independent':True,'review_type':'model_fact_review','review_policy':POLICY_ID,
        'witness_validator':FACTUAL_WITNESS_VALIDATOR,'code_witness_validator':CODE_WITNESS_VALIDATOR,
        'evidence_text_sha256':{row['id']:_text_sha(row['text']) for row in public_view(state)['evidence']},
        'reviewer_id':'factual_review_'+digest({'prompt':request['prompt_sha256'],
            'model':metadata.get('model'),'provider':metadata.get('provider')})[:24],
        'review_path':str(raw_path),'plan_sha256':request['plan_sha256'],
        'visible_view_sha256':request['visible_view_sha256'],'code_witnesses':[]}


def _result(saved: dict,state: dict,plan: dict,request: dict,raw_path: Path) -> dict:
    result=_base(state,plan,request,raw_path,saved.get('metadata'))
    try:
        _metadata(saved.get('metadata'),state)
        value=json.loads(saved['raw'])
        if (not isinstance(value,dict) or set(value)!={'status','reason','evidence_refs','witnesses','code_witnesses'}
                or value.get('status') not in {'PASS','pending_review','FAIL'}
                or not isinstance(value.get('reason'),str) or not value['reason'].strip()):
            raise ValueError('Typed factual reply differs from the frozen response shape')
        refs=value['evidence_refs'];visible={row['id'] for row in public_view(state)['evidence']}
        if (not isinstance(refs,list) or not refs or any(not isinstance(ref,str) or ref not in visible for ref in refs)
                or len(set(refs))!=len(refs)):
            raise ValueError('Typed factual review needs original proposal references')
        if value['status']=='PASS':
            claims=plan['problem_hypothesis']+' '+plan['intended_change']
            if re.search(r'\b(?:globally\s+defect[- ]free|defect[- ]free\s+(?:everywhere|globally)|no\s+(?:defects?|warnings?|errors?)\s+(?:exist\s+)?(?:anywhere|everywhere|in\s+all\s+tasks))\b',claims,re.I):
                raise ValueError('Blanket global absence claim is unsupported by a bounded view')
            # This validator never searches source code for a bad evidence quote.
            witnesses=_review_witnesses(value['witnesses'],state,plan['action'])
            if any(row['evidence_ref'] not in refs for row in witnesses):
                raise ValueError('Witness references must also be listed by the reviewer')
            code=_validate_code(value['code_witnesses'],public_view(state),plan)
            result.update(witnesses=witnesses,code_witnesses=code)
        elif not isinstance(value['witnesses'],list) or not isinstance(value['code_witnesses'],list):
            raise ValueError('Typed review witness fields must remain lists')
        result.update(status=value['status'],reason=value['reason'],evidence_refs=refs)
    except (ValueError,TypeError,KeyError) as exc:
        result['reason']=str(exc)
    return result


def _store_result(path: Path,result: dict) -> None:
    if path.exists():
        previous,_=_read(path)
        if previous!=result:
            raise ValueError('Immutable typed factual result changed')
    else:
        _atomic_json(path,result)


def review_plan_typed(state: dict,plan: dict,directory: Path) -> dict:
    """Persist a fresh typed review; no resend after an unknown request outcome."""
    from research1.distill import _tongji_chat
    plan=validate_plan(plan,state);request=_request(state,plan)
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    input_path=directory/'factual_review_input.json'
    raw_path=directory/'factual_review.json'
    result_path=directory/'factual_review_result.json'
    started_path=directory/'factual_review_started.json'
    if not input_path.exists() and any(path.exists() for path in (raw_path,result_path,started_path)):
        raise ValueError('Cannot reuse a historical/untyped review directory for a new typed policy')
    if input_path.exists():
        previous,_=_read(input_path)
        if previous!=request:
            raise ValueError('Immutable typed factual request changed')
    else:
        _atomic_json(input_path,request)
    _,input_hash=_read(input_path)
    if raw_path.exists():
        saved,_=_read(raw_path)
    elif started_path.exists():
        started,_=_read(started_path)
        if started.get('request_id')!=request['request_id'] or started.get('prompt_sha256')!=request['prompt_sha256']:
            raise ValueError('Typed factual start belongs to another request')
        if result_path.exists():
            previous,_=_read(result_path)
            if (previous.get('plan_sha256')!=request['plan_sha256']
                    or previous.get('visible_view_sha256')!=request['visible_view_sha256']
                    or previous.get('status')=='PASS'):
                raise ValueError('Typed factual outcome is unknown despite an inconsistent saved result')
            return previous
        result=_base(state,plan,request,raw_path)
        result['reason']='CALL_OUTCOME_UNKNOWN: started typed factual request has no saved raw reply; no resend'
        _store_result(result_path,result)
        return result
    else:
        started={'request_id':request['request_id'],'prompt_sha256':request['prompt_sha256'],
                 'input_record_sha256':input_hash,'started_unix':time.time(),'automatic_resend':False}
        try:
            with started_path.open('x',encoding='utf-8') as handle:
                handle.write(json.dumps(started,ensure_ascii=False,indent=2)+'\n');handle.flush();os.fsync(handle.fileno())
        except FileExistsError:
            result=_base(state,plan,request,raw_path)
            result['reason']='CALL_OUTCOME_UNKNOWN: typed factual request claimed by another writer; no resend'
            return result
        try:
            raw,metadata=_tongji_chat(request['prompt'],max_tokens=6000,role='research1_data_factual_review_typed',
                json_output=True,reasoning_effort='none',allow_http_fallback=False)
            _,started_hash=_read(started_path)
            saved={**request,'raw':raw,'raw_sha256':_text_sha(raw),'metadata':metadata,
                'metadata_sha256':digest(metadata),'review_type':'model_fact_review',
                'input_record_sha256':input_hash,'started_record_sha256':started_hash}
            _atomic_json(raw_path,saved)
        except Exception as exc:
            _atomic_json(directory/'factual_review_error.json',{'request_id':request['request_id'],
                'error_type':type(exc).__name__,'reason':str(exc)[:1000],'automatic_resend':False})
            result=_base(state,plan,request,raw_path)
            result['reason']='CALL_OUTCOME_UNKNOWN: typed factual request failed or response unavailable; no resend'
            _store_result(result_path,result)
            return result
    _verify_records(saved,state,plan,request,directory)
    result=_result(saved,state,plan,request,raw_path)
    _store_result(result_path,result)
    return result


def _verify_records(saved: dict,state: dict,plan: dict,request: dict,directory: Path) -> tuple[str,str]:
    for key,value in request.items():
        if saved.get(key)!=value:
            raise ValueError('Saved typed factual input/prompt changed')
    if (not isinstance(saved.get('raw'),str) or saved.get('raw_sha256')!=_text_sha(saved['raw'])
            or saved.get('metadata_sha256')!=digest(saved.get('metadata'))
            or saved.get('review_type')!='model_fact_review'):
        raise ValueError('Saved typed factual raw reply/metadata changed')
    input_record,input_hash=_read(directory/'factual_review_input.json')
    started,started_hash=_read(directory/'factual_review_started.json')
    if (input_record!=request or saved.get('input_record_sha256')!=input_hash
            or saved.get('started_record_sha256')!=started_hash
            or started.get('request_id')!=request['request_id']
            or started.get('prompt_sha256')!=request['prompt_sha256']
            or started.get('input_record_sha256')!=input_hash or started.get('automatic_resend') is not False):
        raise ValueError('Typed factual request/start provenance changed')
    return input_hash,started_hash


def verify_saved_typed_review(review: dict,state: dict,plan: dict) -> dict:
    """Return immutable-record proof or raise ValueError; never call a model."""
    plan=validate_plan(plan,state);request=_request(state,plan)
    if (not isinstance(review,dict) or review.get('status')!='PASS' or review.get('review_policy')!=POLICY_ID
            or review.get('independent') is not True or review.get('review_type')!='model_fact_review'):
        raise ValueError('A real typed factual PASS is required')
    raw_path=Path(review.get('review_path',''))
    saved,raw_hash=_read(raw_path)
    if raw_path.name!='factual_review.json':
        raise ValueError('Typed factual review_path must name its original raw record')
    _verify_records(saved,state,plan,request,raw_path.parent)
    result_path=raw_path.parent/'factual_review_result.json'
    result,result_hash=_read(result_path)
    reconstructed=_result(saved,state,plan,request,raw_path)
    if reconstructed.get('status')!='PASS' or review!=result or review!=reconstructed:
        raise ValueError('Typed factual PASS differs from its original saved raw reply or exact witnesses')
    return {'factual_review_sha256':digest(review),'factual_raw_path':str(raw_path),
            'factual_raw_record_sha256':raw_hash,'factual_result_path':str(result_path),
            'factual_result_record_sha256':result_hash}
