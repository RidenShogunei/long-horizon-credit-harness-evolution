"""Read and validate saved teacher plans without any generation or new contract.

Only admission metadata may differ. Original source states, exact public views,
prompts, request settings, raw teacher bytes and preteacher contract stay bound.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from research1.data_pipeline.engine import ArtifactChangedError
from research1.data_pipeline.execution import tree_hash
from research1.data_pipeline.policy import parse_candidates, public_view, render_messages
from research1.data_pipeline.util import digest

ARMS = {"standard": "none", "reasoned": "low"}
LEGACY_REGISTRY = 'research1.module-contracts.v2'
STRICT_REGISTRY = 'research1.module-contracts.v3'
FACTUAL_POLICY = 'proposal-factual-review.v8.probe_io'
APPLICABILITY_POLICY = 'proposal-plan-applicability.v3.fact_ids'
FACTUAL_FILES = ('factual_review_input.json', 'factual_review_started.json',
    'factual_review.json', 'factual_review_result.json', 'factual_witness_catalog.json')


def _fail(message):
    raise ArtifactChangedError("Saved-teacher replay: " + message)


def _registry_version(value):
    if value in (None, 'v2', LEGACY_REGISTRY):
        return LEGACY_REGISTRY
    if value in ('v3', STRICT_REGISTRY):
        return STRICT_REGISTRY
    _fail('unsupported verification registry version')


def _regular_bytes(path):
    path = Path(path)
    if not path.is_absolute():
        _fail('source artifact path must be absolute')
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            _fail('source artifact cannot follow a symlink')
        cursor = cursor.parent
    if not path.is_file():
        _fail('source artifact must be an existing regular file')
    return path.read_bytes()


def _manifest_bytes(artifacts):
    if not isinstance(artifacts, dict) or not artifacts:
        _fail('source byte manifest is missing')
    loaded = {}
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) != {'path', 'sha256'}:
            _fail('artifact manifest has an invalid shape')
        if re.fullmatch(r'[0-9a-f]{64}', str(artifact['sha256'])) is None:
            _fail('artifact manifest needs a SHA-256')
        path = Path(artifact['path']).resolve()
        raw = _regular_bytes(Path(artifact['path']))
        if hashlib.sha256(raw).hexdigest() != artifact['sha256']:
            _fail('source artifact bytes changed: ' + str(label))
        loaded[path] = raw
    return loaded


def _source_sample(campaign, original, plan=None, expected_id=None):
    """Read mutable revocation/attempt state without opening a writing Ledger."""
    campaign = Path(campaign).resolve()
    db = campaign / 'ledger.sqlite3'
    _regular_bytes(db)
    try:
        with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True) as connection:
            connection.execute('PRAGMA query_only=ON')
            records = [(kind, key, json.loads(payload)) for kind, key, payload in
                connection.execute('SELECT kind,key,payload FROM records')]
            jobs = [json.loads(row[0]) for row in connection.execute('SELECT spec FROM jobs')]
    except (sqlite3.Error, ValueError) as exc:
        _fail('source ledger cannot be read unambiguously: ' + str(exc))
    matches = [row for kind, key, row in records if kind == 'samples'
        and row.get('state_id') == original['state_id']
        and row.get('teacher_record_path') == original['teacher_record_path']]
    if len(matches) != 1:
        _fail('original source sample is missing or ambiguous')
    sample = matches[0]
    sample_state=sample.get('state',{})
    extra=set(sample_state)-set(original)
    state_matches=(all(sample_state.get(key)==value for key,value in original.items())
        and extra<={'plan_factual_review','plan_contract_binding'}
        and ('plan_factual_review' not in sample_state or sample_state['plan_factual_review']==sample.get('qualification',{}).get('factual_review'))
        and ('plan_contract_binding' not in sample_state or sample_state['plan_contract_binding']==sample.get('qualification',{}).get('applicability')))
    if (sample.get('revoked') or sample.get('source_kind') != 'fresh_teacher_plan_from_reused_parent'
            or not state_matches or (plan is not None and sample.get('plan') != plan)
            or (expected_id is not None and sample.get('sample_id') != expected_id)):
        _fail('original source sample was revoked or its identity/plan changed')
    if sample.get('implementation') or sample.get('qualification', {}).get('implementation_task_result') is not None:
        _fail('original source sample already has an implementation or child attempt')
    arm = original['teacher_arm_id']
    for kind, key, row in records:
        if ((kind in {'edittrial_implementations', 'edittrial_verifications'} and key == arm)
                or kind == 'edittrial_call_starts' and key == 'edittrial_implementations:' + arm
                or kind == 'init_child_executions' and key == 'edittrial:arm:' + arm):
            _fail('original source arm already started implementation or child execution')
    if any(str(job.get('scope', '')).startswith('edittrial:child:') for job in jobs):
        _fail('original source trial already reserved a child evaluation')
    directory = campaign / 'implementations' / 'edittrial' / arm
    if directory.exists() and any(path.is_file() or path.is_symlink() for path in directory.rglob('*')):
        _fail('original source arm has saved implementation artifacts')
    return sample


def freeze_factual_replay_source(source):
    """Extend a preparation manifest; this never calls a model or runs a probe."""
    source = copy.deepcopy(source)
    campaign = Path(source['campaign']).resolve()
    def add(label, value):
        path = Path(value)
        raw = _regular_bytes(path)
        if not path.resolve().is_relative_to(campaign):
            _fail('factual replay artifact is outside the original campaign')
        source['artifacts'][label] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}
    add('source_protocol', campaign / 'protocol.json')
    for arm, specification in source['arms'].items():
        original = json.loads(_regular_bytes(Path(specification['state_path'])))
        sample = _source_sample(campaign, original)
        review = sample.get('qualification', {}).get('factual_review', {})
        if review.get('status') != 'PASS' or original.get('factual_review_policy') != FACTUAL_POLICY:
            _fail('source factual review must be original factual8 PASS')
        specification['original_sample_id'] = sample['sample_id']
        specification['factual_review_path'] = review['review_path']
        directory = Path(review['review_path']).parent
        for name in FACTUAL_FILES:
            add(arm + '_' + name, directory / name)
        for label in ('raw_contract', 'receipt'):
            add(arm + '_publisher_' + label, original['predecision_probe_artifacts'][label]['path'])
        add(arm + '_fixture_source', original['applicability_fixture_source']['path'])
    return source


def validate_replayed_factual_state(state, plan):
    """Pure live source validation, also called from admission/seal/export.

    In particular this checks the ORIGINAL trial ledger, never the new trial's
    implementation, and preserves the original decision protocol and fixture.
    """
    if state.get('replay_factual_from_source') is not True:
        _fail('factual reuse needs an explicit saved-source flag')
    loaded = _manifest_bytes(state.get('teacher_source_artifacts'))
    campaign = Path(state['teacher_source_campaign']).resolve()
    def read(value):
        path = Path(value).resolve()
        if path not in loaded or not path.is_relative_to(campaign):
            _fail('required original factual artifact missing from the byte manifest')
        return json.loads(loaded[path])
    original = read(state['original_state_path'])
    original_protocol = read(campaign / 'protocol.json')
    if (original_protocol.get('protocol_hash') != digest({k:v for k,v in original_protocol.items() if k!='protocol_hash'})
            or original_protocol.get('grounded_edit_trial', {}).get('teacher_source')
            or _registry_version(original_protocol.get('verification_registry_version')) != STRICT_REGISTRY
            or _registry_version(original.get('verification_registry_version')) != STRICT_REGISTRY
            or _registry_version(state.get('verification_registry_version')) != STRICT_REGISTRY
            or original_protocol.get('policies',{}).get('factual_reviewer',{}).get('version') != FACTUAL_POLICY
            or original_protocol.get('grounded_edit_trial',{}).get('publish_probe_io') is not True
            or original.get('decision_protocol_hash') != original_protocol.get('protocol_hash')
            or original.get('decision_protocol_hash') != state.get('decision_protocol_hash')
            or original.get('contract') != state.get('contract')
            or public_view(original) != public_view(state)
            or state.get('factual_review_policy') != FACTUAL_POLICY
            or original.get('factual_review_policy') != FACTUAL_POLICY):
        _fail('same-v3 source decision, contract, protocol or public view changed')
    sample = _source_sample(campaign, original, plan, state['original_sample_id'])
    review = sample.get('qualification', {}).get('factual_review')
    if review != state.get('original_factual_review') or review.get('status') != 'PASS':
        _fail('original factual review identity changed')
    if _parse_teacher(read(state['original_teacher_record_path']), state, original_protocol, state['teacher_arm_id']) != [plan]:
        _fail('original plan differs from its teacher raw reply')
    for value in original['predecision_probe_artifacts'].values():
        read(value['path'])
    fixture_path = Path(original['applicability_fixture_source']['path']).resolve()
    if fixture_path not in loaded or state.get('applicability_fixture_source') != original['applicability_fixture_source']:
        _fail('original fixture source is not preserved and manifest-bound')
    if read(campaign / 'edittrial_preteacher_contract.json') != state['contract']:
        _fail('original final preteacher contract changed')
    for name in FACTUAL_FILES:
        read(Path(review['review_path']).parent / name)
    from research1.data_pipeline.fixture_context import verify_probe_contract
    from research1.data_pipeline.predecision_probe_io import validate_probe_io_state
    from research1.data_pipeline.selector_review import verify_saved_selected_review
    try:
        verify_probe_contract(state['applicability_fixture_source'])
        validate_probe_io_state(state)
        return verify_saved_selected_review(review, state, plan)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        _fail('original physical factual/fixture proof no longer validates: ' + str(exc))


def _prompt(state):
    messages = render_messages(state)
    count = state.get("max_candidates", 2)
    return (messages[0]["content"] + "\nCollection request: generate at most " + str(count)
        + ' distinct plans before any candidate outcome is known. Return JSON {"candidates":[Plan,...]} instead of the deployment single-plan envelope. '
        + "Do not force a second edit or fill a candidate quota.\nFrozen proposal-only view:\n" + messages[1]["content"])


def _parse_teacher(record, state, protocol, arm):
    """Pure parsing; deliberately never call collect_candidates or its transport."""
    prompt = _prompt(state)
    if record.get("prompt") != prompt or record.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        _fail("exact original teacher prompt no longer matches the replay state")
    if record.get("visible_view_sha256") != digest(public_view(state)) or record.get("policy_id") != state.get("plan_policy_version"):
        _fail("teacher public-view/policy binding changed")
    if state.get("max_candidates") != 1 or state.get("teacher_reasoning_effort") != ARMS[arm] or state.get("teacher_arm_id") != arm:
        _fail("original arm candidate count or reasoning effort differs")
    settings = {"reasoning_effort": ARMS[arm], "max_tokens": state.get("teacher_max_tokens", 8000),
                "temperature": 0, "allow_http_fallback": state.get("allow_teacher_http_fallback") is not False}
    if (record.get("request_settings") != settings or settings["allow_http_fallback"] is not False
            or settings["max_tokens"] != protocol["grounded_edit_trial"].get("teacher_max_tokens", 8000)):
        _fail("teacher request settings differ from the frozen target")
    teacher = protocol["policies"]["teacher"]
    metadata = record.get("metadata", {})
    if (metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop"
            or metadata.get("temperature") != 0 or metadata.get("reasoning_effort") != ARMS[arm]
            or metadata.get("provider") != teacher.get("provider") or metadata.get("model") != teacher.get("model")
            or state.get("teacher_provider") != teacher.get("provider") or state.get("teacher_model") != teacher.get("model")):
        _fail("original teacher model/signature or final completion is invalid")
    try:
        plans = parse_candidates(record["raw"], state, 1)
    except (ValueError, KeyError, TypeError) as exc:
        _fail("original raw teacher reply is not a complete valid single plan: " + str(exc))
    if len(plans) != 1:
        _fail("original arm must contain exactly one plan")
    return plans


def load_teacher_replay(protocol: dict, parent_row: dict) -> dict:
    """Validate every frozen byte artifact and return target admission states.

    This must run afresh on every resume, before reading cached derived data.
    It does not run a fact extractor, AST indexer, probe, teacher or scorer.
    """
    target_registry = _registry_version(protocol.get('verification_registry_version'))
    replay_factual = protocol.get('grounded_edit_trial', {}).get('replay_factual_from_source') is True
    if target_registry != LEGACY_REGISTRY and not replay_factual:
        _fail('saved-plan admission currently permits only original registry v2; v3 requires fresh teacher generation')
    if replay_factual and (target_registry != STRICT_REGISTRY
            or protocol.get('policies', {}).get('factual_reviewer', {}).get('version') != FACTUAL_POLICY
            or protocol.get('policies', {}).get('applicability_reviewer', {}).get('version') != APPLICABILITY_POLICY
            or protocol.get('policies', {}).get('implementer', {}).get('version') != 'five-module-exact-replacements.v3'
            or protocol.get('grounded_edit_trial', {}).get('publish_probe_io')
            or protocol.get('grounded_edit_trial', {}).get('fact_source')):
        _fail('same-v3 factual reuse requires its exact independent admission policies and no generation flags')
    config = protocol["grounded_edit_trial"]
    source = config.get("teacher_source")
    if not isinstance(source, dict) or set(source.get("arms", {})) != set(ARMS):
        _fail("two original arm sources are required")
    campaign = Path(source["campaign"]).resolve()
    original_protocol_path = campaign / 'protocol.json'
    if not original_protocol_path.is_file() or original_protocol_path.is_symlink():
        _fail('original teacher protocol must be an existing regular file')
    original_protocol = json.loads(original_protocol_path.read_bytes())
    if original_protocol.get('protocol_hash') != digest({key:value for key,value in original_protocol.items() if key!='protocol_hash'}):
        _fail('original teacher protocol content binding changed')
    if _registry_version(original_protocol.get('verification_registry_version')) != target_registry:
        _fail('original teacher protocol and target verification registries differ')
    if replay_factual and original_protocol.get('grounded_edit_trial', {}).get('teacher_source'):
        _fail('factual replay must use the original generation trial, not a replay')
    if replay_factual and original_protocol.get('policies',{}).get('factual_reviewer') != protocol.get('policies',{}).get('factual_reviewer'):
        _fail('factual replay must preserve the original factual-review policy and model configuration')
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        _fail("source byte manifest is missing")
    loaded, hashes = {}, {}
    for label, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            _fail("artifact manifest has an invalid shape")
        path = Path(artifact["path"])
        if not path.is_absolute() or re.fullmatch(r"[0-9a-f]{64}", str(artifact["sha256"])) is None or not path.is_file():
            _fail("artifact needs an existing absolute file and a SHA-256")
        raw = _regular_bytes(path)
        path = path.resolve()
        if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            _fail("source artifact bytes changed: " + str(label))
        if path in loaded and loaded[path] != raw:
            _fail("duplicate source path has inconsistent bytes")
        loaded[path], hashes[path] = raw, artifact["sha256"]

    def source_file(value):
        path = Path(value)
        if not path.is_absolute():
            _fail("required source file paths must be absolute")
        path = path.resolve()
        if campaign not in path.parents or path not in loaded:
            _fail("required original file is outside its campaign or missing from the byte manifest")
        try:
            parsed = json.loads(loaded[path])
        except (ValueError, UnicodeError) as exc:
            _fail("source artifact is not valid original JSON: " + str(exc))
        return path, parsed

    contract_path, contract = source_file(source["contract_path"])
    if _registry_version(contract.get('registry', {}).get('version')) != target_registry:
        _fail('original preteacher contract and target verification registries differ')
    if contract.get("checks_id") != "checks_" + digest({key: value for key, value in contract.items() if key != "checks_id"})[:24]:
        _fail("original preteacher contract content binding changed")
    target_hash = protocol["modules_hash"]
    expected_parent = parent_row["parent_result"]
    states, standard_view = {}, None
    for arm in ARMS:
        specification = source["arms"][arm]
        state_path, original = source_file(specification["state_path"])
        teacher_path, record = source_file(specification["teacher_path"])
        if _registry_version(original.get('verification_registry_version')) != target_registry:
            _fail('original teacher state and target verification registries differ')
        before = public_view(original)
        if Path(original.get("teacher_record_path", "")).resolve() != teacher_path:
            _fail("original state points to a different teacher record")
        if (original.get("contract") != contract or original.get("code_hash") != target_hash
                or tree_hash(Path(original["modules_root"])) != target_hash
                or original.get("root_lineage_id") != parent_row["root_lineage_id"]
                or original.get("split") != parent_row["split"]
                or original.get("proposal_task_ids") != [parent_row["task"]]):
            _fail("original state contract, implementation or lineage differs from the frozen reused parent")
        if any((Path(original["modules_root"]) / path).read_text(encoding="utf-8") != text
               for path, text in original["module_code"].items()):
            _fail("original visible module source differs from its original code tree")
        provenance = original.get("parent_execution_provenance", {})
        if (provenance.get("original_evaluation_id") != expected_parent.get("evaluation_id")
                or provenance.get("original_result_sha256") != expected_parent.get("result_sha256")
                or provenance.get("original_trajectory_sha256") != expected_parent.get("trajectory_sha256")):
            _fail("original execution provenance differs from the reused parent")
        if original.get("plan_policy_version") != protocol.get("sft", {}).get("plan_policy_version"):
            _fail("replay may not change the original planning interface")
        if standard_view is not None and before != standard_view:
            _fail("the two original arms do not have the same frozen public view")
        standard_view = before
        state = copy.deepcopy(original)
        state.update(admission_protocol_hash=protocol["protocol_hash"],
            factual_review_policy=protocol["policies"]["factual_reviewer"]["version"],
            teacher_record_path=str(teacher_path), original_teacher_record_path=str(teacher_path),
            original_teacher_record_sha256=hashes[teacher_path], original_state_id=original["state_id"],
            original_state_path=str(state_path), original_state_sha256=hashes[state_path],
            original_preteacher_contract_sha256=hashes[contract_path], teacher_source_campaign=str(campaign),
            teacher_source_artifacts=copy.deepcopy(artifacts))
        applicability=protocol.get('policies',{}).get('applicability_reviewer',{})
        if replay_factual:
            sample = _source_sample(campaign, original, expected_id=specification.get('original_sample_id'))
            if not specification.get('original_sample_id') or sample.get('qualification', {}).get('factual_review', {}).get('review_path') != specification.get('factual_review_path'):
                _fail('source sample/factual-review manifest binding is missing or changed')
            state.update(replay_factual_from_source=True, original_sample_id=sample['sample_id'],
                original_factual_review=copy.deepcopy(sample['qualification']['factual_review']),
                applicability_policy=APPLICABILITY_POLICY)
            if applicability.get('fixture_source') != original.get('applicability_fixture_source'):
                _fail('target must preserve the original full fixture context')
        elif applicability.get('version')=='proposal-plan-applicability.v2.fixtures':
            from research1.data_pipeline.fixture_context import verify_probe_contract
            state['applicability_policy']=applicability['version']
            state['applicability_fixture_source']=verify_probe_contract(applicability['fixture_source'])
        state["state_id"] = "state_" + digest({"original_state_id": original["state_id"],
            "original_state_sha256": hashes[state_path], "admission_protocol_hash": protocol["protocol_hash"],
            "factual_review_policy": state["factual_review_policy"], "arm": arm})[:24]
        if public_view(state) != before:
            _fail("admission metadata changed the original public view")
        plans = _parse_teacher(record, state, protocol, arm)
        if replay_factual:
            validate_replayed_factual_state(state, plans[0])
        states[arm] = {"state": state, "plans": plans}
    return {"base": copy.deepcopy(states["standard"]["state"]), "contract": contract,
        "contract_bytes": loaded[contract_path], "arms": states,
        "source_campaign": str(campaign), "source_artifacts": copy.deepcopy(artifacts)}
