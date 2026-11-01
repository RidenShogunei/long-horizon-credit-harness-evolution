"""Exact, deterministic evidence selectors for a new opt-in factual review.

The complete decoded-segment catalog is not a diagnosis. Models select ids;
they cannot supply quotation text or observation kinds. Historical typed/v5
records are never repaired, relocated, promoted or rewritten by this module.
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
    _atomic_json, _review_witnesses, _witness_sources, probe_source_strength,
)
from research1.data_pipeline.observation_validator import audit_witnesses
from research1.data_pipeline.typed_review import _validate_code, CODE_WITNESS_VALIDATOR

POLICY_ID='proposal-factual-review.v7.selectors'
PROBE_POLICY_ID='proposal-factual-review.v8.probe_io'
LOCAL_API_VALIDATOR='proposal-local-api-capture-witness.v1'
CATALOG_SCHEMA='proposal-original-witness-catalog.v1'
_BASE_RAW_KEYS=frozenset({'status','reason','evidence_refs','witness_ids','code_witnesses'})
_CLAIM_VERDICTS=frozenset({'supported','unproven_expected_effect','planned_action','incorrect','uncertain'})
_INSTRUCTIONS=(
    'Factual review policy: '+POLICY_ID+'. Independently review only this original single plan, its '
    'frozen decision-time proposal view, and visible source/runtime interfaces. Separate factual accuracy '
    'from decision optimality. A bounded keep needs substantive observed execution facts and evidence '
    'limits; task success or not seeing errors does not prove keep is best. A plausible implementable '
    'process improvement may be justified even when baseline works as designed; a proven code defect, '
    'causal responsibility, future solver benefit or score gain is not required in advance. Distinguish '
    'observed facts, current-source facts and explicitly unproven expected effects. Unsupported causal '
    'guarantees, global absence claims or infeasible runtime dataflow stay pending_review. Check the '
    'original declared scope; do not rewrite the plan or silently add a hook. '
    'The witness catalog is a deterministic exhaustive index of every original decoded evidence '
    'segment. Items were not filtered by success, outcome or relevance. You may select witness_ids '
    'only. Do not copy quotations, assign kinds or invent ids. context means requirements, messages, '
    'commands, metadata or text without substantive observed payload. observed_tool_output means only '
    'that the unchanged existing validator found substantive captured observation at that original '
    'location. It NEVER proves execution/test/HTTP success, causal producer, freshness or task '
    'correctness; mixed observations may include displayed commands as context. A catalog entry itself '
    'is not a diagnosis or recommendation. PASS needs selected actual substantive observation. '
    'Source-only facts MUST use code_witnesses with an original visible module/framework path and exact '
    'source substring. Never use a trajectory id for code found only in module_code/framework_context. '
    'Wrong-slot or invented references are rejected, never relocated. An edit needs a code quotation '
    'from its declared editing location. Code quotations do not count as execution observations. '
    'Return final JSON with exactly these fields: '
    '{"status":"PASS|pending_review|FAIL","reason":"...","evidence_refs":["original evidence id"],'
    '"witness_ids":["id from catalog"],"code_witnesses":[{"path":"visible source path",'
    '"text":"exact original source substring"}]}. '
    'For PASS explain how the selected facts and source support the original plan. Do not introduce '
    'patches, new tests, future trajectories, reward/dev information or new source.\n'
)
_PROBE_INSTRUCTIONS=(
    'Factual review policy: '+PROBE_POLICY_ID+'. Independently review this original single plan using '
    'only its decision-time proposal evidence, current visible source and physically verified predecision '
    'Harness API records. Keep observed/current-source facts distinct from unproven expected effects. '
    'Neither edit nor keep is preferred; do not require prior proof of future utility or causal responsibility. '
    'A bounded keep is not optimality proof. Check the exact declared scope and reachable runtime values; '
    'do not rewrite a plan or silently add a hook. Unsupported facts, infeasible data paths, causal guarantees '
    'or global absence claims stay pending_review. The complete deterministic catalog includes three kinds: '
    'context, observed_tool_output (unchanged V5 trajectory validation), and local_api_observation. '
    'The last kind is available only when the physical publisher receipt, raw frozen-parent capture, '
    'original evidence ID/text, and non-null actual invoked API have all been independently rebound. '
    'It is a separate Harness-local return observation, NEVER an agent trajectory step or proof of shell '
    'success, task correctness, failure cause or performance benefit. fixture_defined_input is definition-derived '
    'where marked, not captured invocation input; missing inputs and unknown producers are not inferred. '
    'At least one substantive real trajectory observation OR one verified known-API capture must support PASS. '
    'Do not add an unrelated agent quote just to meet a gate. Context, source text, forged API labels and '
    'unknown-producer records cannot alone establish an observed fact. Select witness_ids only, without '
    'copying quotations or assigning kinds. A listed capture/check is not a diagnosis or recommendation. '
    'Use code_witnesses only for exact original module/framework source substrings; an edit needs a source '
    'quotation from its declared editing location. Return final JSON with exactly these fields: '
    '{"status":"PASS|pending_review|FAIL","reason":"...","evidence_refs":["original evidence id"],'
    '"witness_ids":["id from catalog"],"code_witnesses":[{"path":"visible source path",'
    '"text":"exact original source substring"}]}. Do not introduce patches, changed plans, future results '
    'or reward/dev information. Wrong quotation slots are rejected, never relocated.\n'
)
_PRECISION_INSTRUCTIONS=(
    ' Plan precision policy indexes every nonempty plan narrative field in plan_field_catalog. '
    'Independently assess EVERY catalog field and EVERY concrete factual assertion within its exact '
    'text, including source case or object identity, exact type or value, predeclared fixture identity, '
    'preconditions, and feasible declared runtime data paths within the plan scope. '
    'One supported clause cannot excuse another false clause in the same or another field. '
    'Keep explicit hypotheses and verification intentions distinct from observed or current-source facts. '
    'When mixed factual clauses appear in one field, verify each; if any concrete clause is false or '
    'not verifiable from the same sources, mark the entire field incorrect or uncertain respectively. '
    'Return final JSON with the same probe factual fields plus claim_audit: a list with exactly one row '
    'per catalog id and no missing, duplicate, or extra ids. Each row is '
    '{"field_id":"id from plan_field_catalog","verdict":"supported|unproven_expected_effect|'
    'planned_action|incorrect|uncertain","reason":"nonempty concrete checks performed"}. '
    'Use unproven_expected_effect for explicit hypotheses without asserting utility; planned_action for '
    'future checks without pretending execution occurred. Do not require future score gain or causal '
    'responsibility. Use the same sources only; do not reference prior review failures.\n'
)


def _precision_enabled(state: dict,review_policy: str) -> bool:
    profile=state.get('plan_precision_policy')
    if profile is None:
        return False
    from research1.data_pipeline.plan_precision import POLICY_ID as PRECISION_POLICY_ID
    if profile!=PRECISION_POLICY_ID:
        raise ValueError('Unknown plan precision policy')
    if review_policy!=PROBE_POLICY_ID:
        raise ValueError('Plan precision field audit requires the probe IO factual review policy')
    return True


def _validate_claim_audit(raw_audit: Any,catalog: dict) -> list[dict]:
    if not isinstance(raw_audit,list):
        raise ValueError('claim_audit must be a list')
    lookup={row['id']:row for row in catalog['items']};seen=set();rows=[]
    for entry in raw_audit:
        if (not isinstance(entry,dict) or set(entry)!={'field_id','verdict','reason'}
                or not isinstance(entry['field_id'],str) or entry['field_id'] not in lookup
                or entry['field_id'] in seen):
            raise ValueError('claim_audit row ids must match the plan field catalog exactly once')
        if entry['verdict'] not in _CLAIM_VERDICTS:
            raise ValueError('claim_audit verdict is not allowed')
        if not isinstance(entry['reason'],str) or not entry['reason'].strip():
            raise ValueError('claim_audit reason must be nonempty')
        seen.add(entry['field_id']);rows.append(dict(entry))
    if seen!=set(lookup):
        raise ValueError('claim_audit must include every catalog field id')
    return rows


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _text_sha(text: str) -> str:
    return _sha(text.encode('utf-8'))


def _read(path: Path) -> tuple[dict,str]:
    path=Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('Selector review records must be existing absolute regular files')
    raw=path.read_bytes();value=json.loads(raw)
    if not isinstance(value,dict):raise ValueError('Selector review record must be an object')
    return value,_sha(raw)


def _metadata(meta: dict,state: dict) -> None:
    if (not isinstance(meta,dict) or meta.get('raw_field')!='content'
            or meta.get('finish_reason')!='stop' or meta.get('temperature')!=0):
        raise ValueError('Selector factual review requires complete final JSON at temperature zero')
    for key in ('provider','model'):
        expected=state.get('review_'+key) or state.get('teacher_'+key)
        if expected and meta.get(key)!=expected:
            raise ValueError('Selector factual reviewer differs from the frozen policy')


def _substantive_at(source: dict,report: dict) -> bool:
    # An identical message/command quote elsewhere in an observation must not
    # upgrade this source. Require the exact original observation location.
    if source.get('field')!='observation_excerpt' or not report.get('eligible_witness_set'):
        return False
    if not report.get('substantive_execution_fact'):return False
    for row in report.get('witnesses',[]):
        for loc in row.get('validation',{}).get('locations',[]):
            if (loc.get('field')=='observation_excerpt' and loc.get('step_index')==source.get('step_index')
                    and loc.get('step_id')==source.get('step_id')
                    and loc.get('source_index')==source.get('source_index')
                    and loc.get('segment_index')==source.get('segment_index')
                    and loc.get('segment_offset')==0 and loc.get('length')==len(source['text'])
                    and loc.get('substantive_observed_fact') is True):
                return True
    return False


def _verified_api_sources(state: dict,view: dict) -> tuple[dict,dict]:
    from research1.data_pipeline.predecision_probe_io import SCHEMA,validate_probe_io_state
    if (state.get('predecision_probe_io_policy')!=SCHEMA or view.get('probe_evidence_policy')!=SCHEMA
            or digest(view)!=digest(public_view(state))):
        raise ValueError('API captures require the original physically validated publisher state/view')
    proof=validate_probe_io_state(state)
    refs=proof.get('evidence_ids_added')
    if (proof.get('schema')!=SCHEMA or proof.get('verified') is not True
            or not isinstance(refs,list) or refs!=view.get('predecision_probe_evidence_refs')
            or set((proof.get('artifacts') or {}))!={'raw_contract','receipt'}):
        raise ValueError('API capture proof does not bind the original evidence references/artifacts')
    evidence={row['id']:row['text'] for row in view['evidence']}
    original={row['id']:row['text'] for row in state['evidence']}
    known={}
    for ref in refs:
        if ref not in evidence or original.get(ref)!=evidence[ref]:
            raise ValueError('API evidence ID/text differs from its physically rebound projection')
        body=json.loads(evidence[ref])
        api=body.get('invoked_api') if isinstance(body,dict) else None
        if isinstance(api,str) and api.strip():
            known[ref]={'text':evidence[ref],'invoked_api':api}
    return known,proof


def build_witness_catalog(view: dict,*,review_policy: str=POLICY_ID,state: dict|None=None) -> dict:
    """One item for every original segment; no ranking, rewriting or omission."""
    if review_policy not in {POLICY_ID,PROBE_POLICY_ID}:raise ValueError('Unknown factual selector catalog policy')
    known={}
    if review_policy==PROBE_POLICY_ID:
        if state is None:raise ValueError('API catalog requires its physical publisher state')
        known,_=_verified_api_sources(state,view)
    items=[]
    for evidence in view['evidence']:
        for source_index,source in enumerate(_witness_sources(evidence['text'])):
            text=source['text']
            proposed={'evidence_ref':evidence['id'],'text':text,'kind':'observed_tool_output'}
            report=audit_witnesses([proposed],view,'edit')
            kind='observed_tool_output' if _substantive_at(source,report) else 'context'
            if kind=='observed_tool_output':
                # Retained complete input lines are known display context even
                # when the full original keystrokes field was budget-omitted.
                # This only removes weak eligibility; original text stays exact.
                from research1.data_pipeline.excerpt_commands import observation_excerpt_command_only
                try:
                    step=json.loads(evidence['text'])['steps'][source['step_index']]
                    if observation_excerpt_command_only(step,source['segment_index']):
                        kind='context'
                except (ValueError,TypeError,KeyError,IndexError):
                    pass
            item={'evidence_ref':evidence['id'],'serialized_evidence_sha256':_text_sha(evidence['text']),
                'source_ordinal':source_index,'field':source['field'],
                'step_id':source.get('step_id'),'step_index':source.get('step_index'),
                'source_index':source.get('source_index'),'visible_step_sha256':source.get('visible_step_sha256'),
                'segment_index':source.get('segment_index'),'segment_start':source.get('segment_start'),
                'offset_start':0,'offset_end':len(text),'text_sha256':_text_sha(text),
                'text':text,'kind':kind,'success_established':False,
                'producer_established':False,'freshness_established':False}
            if evidence['id'] in known:
                if source['field']!='plain_text' or text!=known[evidence['id']]['text']:
                    raise ValueError('Local API capture must remain its original complete standalone record')
                item.update(kind='local_api_observation',source_kind='verified_parent_harness_local_api_return',
                    invoked_api=known[evidence['id']]['invoked_api'],original_agent_step=False)
            item['id']='original_witness_'+digest(item)[:32]
            items.append(item)
    return {'schema':CATALOG_SCHEMA,'view_sha256':digest(view),'validator':FACTUAL_WITNESS_VALIDATOR,
            'selection':'all_original_decoded_segments_in_original_order','items':items}


def _request(state: dict,plan: dict) -> tuple[dict,dict]:
    policy=state.get('factual_review_policy')
    if policy not in {POLICY_ID,PROBE_POLICY_ID}:
        raise ValueError('Selector review requires its explicitly frozen policy')
    view=public_view(state);catalog=build_witness_catalog(view,review_policy=policy,state=state)
    payload={'policy':policy,'original_view':view,'witness_catalog':catalog,'plan':plan}
    instructions=_INSTRUCTIONS if policy==POLICY_ID else _PROBE_INSTRUCTIONS
    precision_catalog=None
    if _precision_enabled(state,policy):
        from research1.data_pipeline import plan_precision as precision
        precision_catalog=precision.build_plan_field_catalog(plan)
        payload['precision_policy']=precision.POLICY_ID
        payload['plan_field_catalog']=precision_catalog
        instructions+=_PRECISION_INSTRUCTIONS
    prompt=instructions+probe_source_strength(view)+canonical(payload)
    request={'review_policy':policy,'request_id':digest(payload),'prompt':prompt,
        'prompt_sha256':_text_sha(prompt),'visible_view_sha256':digest(view),
        'plan_sha256':digest(plan),'catalog_sha256':digest(catalog)}
    if precision_catalog is not None:
        from research1.data_pipeline import plan_precision as precision
        request['precision_policy']=precision.POLICY_ID
        request['plan_field_catalog_sha256']=precision.catalog_sha256(precision_catalog)
    return request,catalog


def _base(state: dict,plan: dict,request: dict,directory: Path,meta: dict|None=None) -> dict:
    meta=meta or {}
    return {'status':'pending_review','reason':'Selector factual review is not validated','evidence_refs':[],
        'independent':True,'review_type':'model_fact_review','review_policy':request['review_policy'],
        'witness_validator':FACTUAL_WITNESS_VALIDATOR,'code_witness_validator':CODE_WITNESS_VALIDATOR,
        'evidence_text_sha256':{row['id']:_text_sha(row['text']) for row in public_view(state)['evidence']},
        'reviewer_id':'factual_review_'+digest({'prompt':request['prompt_sha256'],
            'model':meta.get('model'),'provider':meta.get('provider')})[:24],
        'review_path':str(directory/'factual_review.json'),'plan_sha256':request['plan_sha256'],
        'visible_view_sha256':request['visible_view_sha256'],'catalog_sha256':request['catalog_sha256'],
        'witness_catalog_path':str(directory/'factual_witness_catalog.json'),
        'witness_ids':[],'code_witnesses':[]}


def _materialize(ids: Any,refs: list,catalog: dict) -> list[dict]:
    if (not isinstance(ids,list) or not ids or any(not isinstance(value,str) for value in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('PASS needs distinct original witness ids')
    lookup={row['id']:row for row in catalog['items']};result=[]
    for identifier in ids:
        if identifier not in lookup:raise ValueError('Unknown original witness id')
        row=lookup[identifier]
        if row['evidence_ref'] not in refs:raise ValueError('Selected witness is missing its original evidence reference')
        result.append({'evidence_ref':row['evidence_ref'],'text':row['text'],'kind':row['kind']})
    return result


def _local_api_witnesses(witnesses: list,state: dict,view: dict) -> list[dict]:
    known,proof=_verified_api_sources(state,view)
    validated=[]
    for witness in witnesses:
        original=known.get(witness['evidence_ref'])
        if (witness.get('kind')!='local_api_observation' or original is None
                or witness['text']!=original['text']):
            raise ValueError('Selected API observation is not an exact physically verified known-API capture')
        validated.append(dict(witness,validation={
            'validator':LOCAL_API_VALIDATOR,'publisher_policy':proof['schema'],
            'source_kind':'verified_parent_harness_local_api_return','invoked_api':original['invoked_api'],
            'capture_scope':'Predecision frozen-parent Harness API return values only; not the original task invocation.',
            'input_scope':'fixture_defined_input is definition-derived where marked, not captured invocation input; unknown input remains unknown.',
            'evidence_ref':witness['evidence_ref'],'text_sha256':_text_sha(witness['text']),
            'offset_start':0,'offset_end':len(witness['text']),
            'artifact_proofs':{label:dict(artifact) for label,artifact in proof['artifacts'].items()},
            'parent_code_hash':proof.get('parent_code_hash'),'original_checks_id':proof.get('original_checks_id'),
            'final_checks_id':proof.get('final_checks_id'),'original_agent_step':False,
            'captured_input_established':False,
            'task_success_established':False,'shell_success_established':False,'success_established':False,
            'performance_benefit_established':False,'producer_established':False}))
    return validated


def _result(saved: dict,state: dict,plan: dict,request: dict,catalog: dict,directory: Path) -> dict:
    result=_base(state,plan,request,directory,saved.get('metadata'))
    precision=_precision_enabled(state,request['review_policy'])
    expected_keys=_BASE_RAW_KEYS|({'claim_audit'} if precision else set())
    try:
        _metadata(saved.get('metadata'),state);value=json.loads(saved['raw'])
        if (not isinstance(value,dict) or set(value)!=expected_keys
                or value.get('status') not in {'PASS','pending_review','FAIL'}
                or not isinstance(value.get('reason'),str) or not value['reason'].strip()):
            raise ValueError('Selector reply differs from the frozen response shape; old quotation slots are not relocated')
        claim_rows=None
        if precision:
            from research1.data_pipeline import plan_precision as precision_mod
            field_catalog=precision_mod.build_plan_field_catalog(plan)
            if request.get('plan_field_catalog_sha256')!=precision_mod.catalog_sha256(field_catalog):
                raise ValueError('Plan field catalog binding changed')
            claim_rows=_validate_claim_audit(value['claim_audit'],field_catalog)
        refs=value['evidence_refs'];view=public_view(state);visible={row['id'] for row in view['evidence']}
        if (not isinstance(refs,list) or not refs or any(not isinstance(ref,str) or ref not in visible for ref in refs)
                or len(set(refs))!=len(refs)):
            raise ValueError('Selector factual review needs original proposal references')
        if value['status']=='PASS':
            claims=plan['problem_hypothesis']+' '+plan['intended_change']
            if re.search(r'\b(?:globally\s+defect[- ]free|defect[- ]free\s+(?:everywhere|globally)|no\s+(?:defects?|warnings?|errors?)\s+(?:exist\s+)?(?:anywhere|everywhere|in\s+all\s+tasks))\b',claims,re.I):
                raise ValueError('Blanket global absence claim is unsupported by a bounded view')
            materialized=_materialize(value['witness_ids'],refs,catalog)
            if request['review_policy']==POLICY_ID:
                report=audit_witnesses(materialized,view,plan['action'])
                if not report.get('eligible_witness_set') or not report.get('substantive_execution_fact'):
                    raise ValueError('PASS needs an actual substantive original observation; commands/context/source do not supply one')
                # Historical V7 follows its original V5 gate, unchanged.
                witnesses=_review_witnesses(materialized,state,plan['action'])
            else:
                local=[row for row in materialized if row['kind']=='local_api_observation']
                trajectory=[row for row in materialized if row['kind']!='local_api_observation']
                local_validated=_local_api_witnesses(local,state,view) if local else []
                original_validated=[];substantive=False
                if trajectory:
                    # Per-witness V5 validation remains exact. V8's aggregate
                    # observed-fact gate also recognizes the separately proven
                    # API capture, so unrelated agent quotations are unnecessary.
                    action='edit' if local_validated else plan['action']
                    report=audit_witnesses(trajectory,view,action)
                    if not report.get('eligible_witness_set'):
                        raise ValueError('Selected original trajectory witness fails unchanged V5 validation')
                    original_validated=_review_witnesses(trajectory,state,action)
                    substantive=report.get('substantive_execution_fact') is True
                if not substantive and not local_validated:
                    raise ValueError('PASS needs a substantive original trajectory observation or a physically verified known-API capture')
                originals=iter(original_validated);locals_=iter(local_validated)
                witnesses=[next(locals_) if row['kind']=='local_api_observation' else next(originals) for row in materialized]
                result['local_api_witnesses']=local_validated
            if precision and any(row['verdict'] in {'incorrect','uncertain'} for row in claim_rows or []):
                raise ValueError('PASS requires every field audit verdict to avoid incorrect or uncertain')
            code=_validate_code(value['code_witnesses'],view,plan)
            result.update(witnesses=witnesses,witness_ids=list(value['witness_ids']),code_witnesses=code)
        elif not isinstance(value['witness_ids'],list) or not isinstance(value['code_witnesses'],list):
            raise ValueError('Selector witness fields must remain lists')
        if precision and claim_rows is not None:
            result.update(precision_policy=request['precision_policy'],
                plan_field_catalog_sha256=request['plan_field_catalog_sha256'],claim_audit=claim_rows)
        result.update(status=value['status'],reason=value['reason'],evidence_refs=refs)
    except (ValueError,TypeError,KeyError) as exc:
        result['reason']=str(exc)
    return result


def _store(path: Path,value: dict) -> None:
    if path.exists():
        previous,_=_read(path)
        if previous!=value:raise ValueError('Immutable selector review artifact changed')
    else:_atomic_json(path,value)


def _verify_records(saved: dict,request: dict,catalog: dict,directory: Path) -> None:
    if any(saved.get(key)!=value for key,value in request.items()):
        raise ValueError('Selector review prompt/input/catalog binding changed')
    if (not isinstance(saved.get('raw'),str) or saved.get('raw_sha256')!=_text_sha(saved['raw'])
            or saved.get('metadata_sha256')!=digest(saved.get('metadata'))
            or saved.get('review_type')!='model_fact_review'):
        raise ValueError('Selector review raw reply/metadata changed')
    input_record,input_hash=_read(directory/'factual_review_input.json')
    actual_catalog,catalog_hash=_read(directory/'factual_witness_catalog.json')
    started,started_hash=_read(directory/'factual_review_started.json')
    if (input_record!=request or actual_catalog!=catalog
            or saved.get('input_record_sha256')!=input_hash
            or saved.get('catalog_record_sha256')!=catalog_hash
            or saved.get('started_record_sha256')!=started_hash
            or started.get('request_id')!=request['request_id']
            or started.get('prompt_sha256')!=request['prompt_sha256']
            or started.get('input_record_sha256')!=input_hash
            or started.get('catalog_record_sha256')!=catalog_hash or started.get('automatic_resend') is not False):
        raise ValueError('Original selector catalog/request/start provenance changed')


def review_plan_selected(state: dict,plan: dict,directory: Path) -> dict:
    """One saved request, with no speculative resend after an unknown outcome."""
    from research1.distill import _tongji_chat
    plan=validate_plan(plan,state);request,catalog=_request(state,plan)
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    input_path=directory/'factual_review_input.json';raw_path=directory/'factual_review.json'
    result_path=directory/'factual_review_result.json';start_path=directory/'factual_review_started.json'
    catalog_path=directory/'factual_witness_catalog.json'
    if not input_path.exists() and any(path.exists() for path in (raw_path,result_path,start_path)):
        raise ValueError('Cannot reuse an old review directory for a new selector policy')
    _store(input_path,request);_store(catalog_path,catalog)
    _,input_hash=_read(input_path);_,catalog_hash=_read(catalog_path)
    if raw_path.exists():saved,_=_read(raw_path)
    elif start_path.exists():
        started,_=_read(start_path)
        if started.get('request_id')!=request['request_id'] or started.get('prompt_sha256')!=request['prompt_sha256']:
            raise ValueError('Selector review start belongs to another request')
        if result_path.exists():
            previous,_=_read(result_path)
            if (previous.get('plan_sha256')!=request['plan_sha256']
                    or previous.get('visible_view_sha256')!=request['visible_view_sha256']
                    or previous.get('status')=='PASS'):
                raise ValueError('Selector review outcome is unknown despite an inconsistent saved result')
            return previous
        result=_base(state,plan,request,directory)
        result['reason']='CALL_OUTCOME_UNKNOWN: original selector request started without saved raw; no resend'
        _store(result_path,result);return result
    else:
        started={'request_id':request['request_id'],'prompt_sha256':request['prompt_sha256'],
            'input_record_sha256':input_hash,'catalog_record_sha256':catalog_hash,
            'started_unix':time.time(),'automatic_resend':False}
        try:
            with start_path.open('x',encoding='utf-8') as handle:
                handle.write(json.dumps(started,ensure_ascii=False,indent=2)+'\n');handle.flush();os.fsync(handle.fileno())
        except FileExistsError:
            result=_base(state,plan,request,directory)
            result['reason']='CALL_OUTCOME_UNKNOWN: selector request claimed by another writer; no resend'
            return result
        try:
            raw,meta=_tongji_chat(request['prompt'],max_tokens=6000,role='research1_data_factual_review_selectors',
                json_output=True,reasoning_effort='none',allow_http_fallback=False)
            _,started_hash=_read(start_path)
            saved={**request,'raw':raw,'raw_sha256':_text_sha(raw),'metadata':meta,'metadata_sha256':digest(meta),
                'review_type':'model_fact_review','input_record_sha256':input_hash,
                'catalog_record_sha256':catalog_hash,'started_record_sha256':started_hash}
            _atomic_json(raw_path,saved)
        except Exception as exc:
            _atomic_json(directory/'factual_review_error.json',{'request_id':request['request_id'],
                'error_type':type(exc).__name__,'reason':str(exc)[:1000],'automatic_resend':False})
            result=_base(state,plan,request,directory)
            result['reason']='CALL_OUTCOME_UNKNOWN: selector request failed or raw unavailable; no resend'
            _store(result_path,result);return result
    _verify_records(saved,request,catalog,directory)
    result=_result(saved,state,plan,request,catalog,directory)
    _store(result_path,result);return result


def verify_saved_selected_review(review: dict,state: dict,plan: dict) -> dict:
    """Zero transport/probe: reconstruct catalog, input and materialized witnesses."""
    plan=validate_plan(plan,state);request,catalog=_request(state,plan)
    if (not isinstance(review,dict) or review.get('status')!='PASS' or review.get('review_policy')!=request['review_policy']
            or review.get('independent') is not True or review.get('review_type')!='model_fact_review'):
        raise ValueError('A real saved selector factual PASS is required')
    path=Path(review.get('review_path',''));saved,raw_hash=_read(path)
    if path.name!='factual_review.json':raise ValueError('Selector review_path must name its original raw record')
    _verify_records(saved,request,catalog,path.parent)
    result_path=path.parent/'factual_review_result.json';result,result_hash=_read(result_path)
    reconstructed=_result(saved,state,plan,request,catalog,path.parent)
    if reconstructed.get('status')!='PASS' or review!=result or review!=reconstructed:
        raise ValueError('Selector PASS differs from its original raw IDs, exact source or V5-validated observations')
    return {'factual_review_sha256':digest(review),'factual_raw_path':str(path),
        'factual_raw_record_sha256':raw_hash,'factual_result_path':str(result_path),
        'factual_result_record_sha256':result_hash}
