"""Frozen two-arm, one-parent edit trial; no reference/expert patch is an input.

Every raw arm is retained before any plan review or implementation. Existing
scores/roots are reused verbatim; only verified changed children may consume the
two reserved proposal evaluations. An uncertain external call is never retried
without its raw reply. Export remains the CLI's separate, audited operation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import time

from research1.data_pipeline.engine import ArtifactChangedError, _cached, _public_state, _usable
from research1.data_pipeline.execution import evaluate, tree_hash
from research1.data_pipeline.initialization import (
    _add_addressable_feedback, _check_attempt, _check_result_artifacts,
    _raise_policy_failure, _review_valid, _StopInitialization,
)
from research1.data_pipeline.ledger import BudgetExceeded, Ledger
from research1.data_pipeline.policy import collect_candidates, review_plan, validate_plan
from research1.data_pipeline.source_parents import source_provenance, validate_parent_sources
from research1.data_pipeline.util import atomic_json, digest

TRIAL_DRIVER = "grounded_edit_trial.v1"
ARMS = [{"id": "standard", "reasoning_effort": "none"},
        {"id": "reasoned", "reasoning_effort": "low"}]


class AmbiguousExternalCall(RuntimeError):
    pass


def _model_once(ledger, kind, key, raw_path, function):
    """Ledger commit + raw cache recovery; no raw means no speculative retry."""
    cached = dict(ledger.record_items(kind)).get(key)
    if cached is not None:
        if cached.get("raw_sha256") is not None:
            path = Path(cached["raw_path"])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != cached["raw_sha256"]:
                raise ArtifactChangedError("Saved external raw reply changed: " + kind)
        return copy.deepcopy(cached["value"])
    start_key = kind + ":" + key
    started = dict(ledger.record_items("edittrial_call_starts")).get(start_key)
    binding = {"kind": kind, "key": key, "raw_path": str(Path(raw_path).resolve())}
    if started is not None:
        if started != binding:
            raise ArtifactChangedError("Frozen external-call binding changed")
        if not Path(raw_path).is_file():
            raise AmbiguousExternalCall("External call started without a saved raw reply; automatic resend forbidden")
    else:
        ledger.put_record("edittrial_call_starts", start_key, binding)
    value = function()
    ledger.put_record(kind, key, {"_operation_key": key, "value": value,
        "raw_path": str(Path(raw_path).resolve()),
        "raw_sha256": hashlib.sha256(Path(raw_path).read_bytes()).hexdigest() if Path(raw_path).is_file() else None})
    return copy.deepcopy(value)


def _immutable_json(path, value):
    path = Path(path)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ArtifactChangedError("Frozen trial artifact changed: " + path.name)
        return
    atomic_json(path, value)


def run_grounded_edit_trial(campaign: Path, *, evaluator=None, proposer=None, implementer=None,
        verifier=None, reviewer=None, fact_collector=None, interface_builder=None,
        applicability_reviewer=None, contract_builder=None, source_validator=None,
        evidence_selector=None, probe_io_publisher=None, probe_io_validator=None) -> dict:
    """One fixed reused parent, both teacher arms first, at most two child jobs.

    Optional callbacks support deterministic offline tests. Production defaults
    always use the immutable source-parent validator and real adapters.
    """
    from research1.data_pipeline.protocol import load_protocol
    from research1.data_pipeline.module_adapters import get_adapter

    campaign = Path(campaign).resolve()
    protocol = load_protocol(campaign, verify_source=evaluator is None)
    module_adapter = get_adapter(protocol)
    config = protocol.get("grounded_edit_trial") or {}
    publish_probe_io=config.get('publish_probe_io',False)
    precision=protocol.get('plan_precision_policy')
    precision_reviewer=protocol.get('policies',{}).get('factual_reviewer',{})
    independent_policy=protocol.get('independent_plan_audit_policy')
    if independent_policy is not None or protocol.get('policies',{}).get('independent_plan_audit'):
        from research1.data_pipeline.independent_plan_audit import POLICY_ID as INDEPENDENT_AUDIT_POLICY
        applicability=protocol.get('policies',{}).get('applicability_reviewer',{})
        if (independent_policy!=INDEPENDENT_AUDIT_POLICY
                or precision!='proposal-plan-precision.v1.field_audit'
                or precision_reviewer.get('precision_policy')!=precision
                or precision_reviewer.get('version')!='proposal-factual-review.v8.probe_io'
                or applicability.get('version')!='proposal-plan-applicability.v3.fact_ids'
                or not applicability.get('fixture_source_ref')
                or module_adapter.REGISTRY_VERSION!='research1.module-contracts.v3'
                or publish_probe_io is not True or config.get('teacher_source')
                or config.get('replay_factual_from_source') or config.get('fact_source')):
            raise ValueError('Independent plan audit requires its frozen fresh published-v3 admission profile')
    if precision is not None or precision_reviewer.get('precision_policy') is not None:
        applicability=protocol.get('policies',{}).get('applicability_reviewer',{})
        if (precision!='proposal-plan-precision.v1.field_audit'
                or precision_reviewer.get('precision_policy')!=precision
                or precision_reviewer.get('version')!='proposal-factual-review.v8.probe_io'
                or applicability.get('version')!='proposal-plan-applicability.v3.fact_ids'
                or not applicability.get('fixture_source_ref')
                or module_adapter.REGISTRY_VERSION!='research1.module-contracts.v3'
                or publish_probe_io is not True or config.get('teacher_source')
                or config.get('replay_factual_from_source') or config.get('fact_source')):
            raise ValueError('Plan precision audit requires its frozen fresh published-v3 admission profile')
    if not isinstance(publish_probe_io,bool):
        raise ValueError('Parent probe publication requires an explicit boolean opt-in')
    if publish_probe_io:
        if (module_adapter.REGISTRY_VERSION!='research1.module-contracts.v3'
                or config.get('teacher_source') or config.get('fact_source')
                or not protocol.get('diagnostic_source_selection',{}).get('frozen_before_candidates')):
            raise ValueError('Parent probe publication is limited to fresh fixed-source registry-v3 diagnosis')
        if probe_io_publisher is None or probe_io_validator is None:
            from research1.data_pipeline.predecision_probe_io import publish_parent_probe_io, validate_probe_io_state
            probe_io_publisher=probe_io_publisher or publish_parent_probe_io
            probe_io_validator=probe_io_validator or validate_probe_io_state
    if (protocol.get("execution_mode") != "initialization"
            or protocol.get("trial_driver") != TRIAL_DRIVER
            or config.get("arms") != ARMS or config.get("stop_after_qualified_edit") != 1):
        raise ValueError("Grounded edit trial requires its exact preregistered two-arm protocol")
    if protocol.get("stage_budget", {}).get("proposal") != 2:
        raise ValueError("Grounded edit trial must reserve exactly two proposal evaluations")
    queue = protocol.get("initialization", {})
    tasks = queue.get("task_order", [])
    if queue.get("max_parents") != 1 or len(tasks) != 1:
        raise ValueError("Grounded edit trial requires exactly one frozen reused parent")
    sets = protocol["task_sets"]
    if (tasks[0] not in sets["proposal"] or set(sets["proposal"]) & (set(sets["reward"]) | set(sets["dev"]))):
        raise ValueError("Trial parent must belong exclusively to the proposal partition")
    if fact_collector is None or interface_builder is None:
        from research1.data_pipeline.fact_cards import collect_fact_card, build_interface_card
        if fact_collector is None and config.get('fact_source',{}).get('kind')=='saved_quote_replay':
            from research1.data_pipeline.fact_cards import replay_fact_card
            source=config['fact_source']
            fact_collector=lambda state,directory:replay_fact_card(state,Path(source['directory']),directory,
                old_state_path=Path(source['old_state_path']))
        fact_collector = fact_collector or collect_fact_card
        interface_builder = interface_builder or build_interface_card
    production_applicability = applicability_reviewer is None
    if production_applicability:
        from research1.data_pipeline.applicability import review_applicability
        applicability_reviewer = review_applicability
    evaluate_task, propose = evaluator or evaluate, proposer or collect_candidates
    implement, verify = implementer or module_adapter.implement, verifier or module_adapter.verify
    build_contract = contract_builder or module_adapter.build_contract
    validate_sources = source_validator or validate_parent_sources
    baseline = Path(protocol["modules_root"])
    ledger = Ledger(campaign)
    parent_scope = "init:parent:0"
    task = tasks[0]
    status = {"schema": "research1.grounded-edit-trial.v1", "campaign": str(campaign),
        "protocol_id": protocol["protocol_id"], "trial_driver": TRIAL_DRIVER,
        "status": "RUNNING", "phase": "source_validation", "task_order": tasks,
        "calibration_complete": False, "data_collection_complete": False,
        "training_started": False, "trial_goal_met": False, "started_unix": time.time()}

    def save(phase=None):
        if phase:
            status["phase"] = phase
        samples = [row for row in ledger.records("samples") if not row.get("revoked")]
        parents = ledger.records("init_parents")
        teachers = ledger.records("edittrial_teacher_outcomes")
        status["counts"] = {"raw_plans": len(samples),
            "qualified_edit": sum(row["plan"]["action"] == "edit" and row["qualification"]["status"] == "verified" for row in samples),
            "qualified_keep": sum(row["plan"]["action"] == "keep" and row["qualification"]["status"] == "verified" for row in samples),
            "pending": sum(row["qualification"]["status"] == "pending" for row in samples),
            "rejected": sum(row["qualification"]["status"] == "rejected" for row in samples),
            "failed_parents": sum(row.get("status") in {"UNSCORED", "EVIDENCE_PENDING"} for row in parents),
            "parents_processed": sum(row.get("status") == "DONE" for row in parents),
            "unique_parent_roots": len({row["root_lineage_id"] for row in parents if row.get("root_lineage_id")}),
            "reused_parent_executions": len({row["parent_result"]["evaluation_id"] for row in parents}),
            "new_independent_parents": 0, "teacher_arms_finished": len(teachers),
            "teacher_errors": sum(row["status"] != "COMPLETE" for row in teachers),
            "fact_index_errors": len(ledger.records("fact_index_error"))}
        status["trial_goal_met"] = status["counts"]["qualified_edit"] >= 1
        status["ledger"] = ledger.status()
        atomic_json(campaign / "status.json", status)

    def sample(state, plan, scope, parent, *, review=None, binding=None, attempt=None,
               behavior=None, child=None, pending_reason=None, independent_plan_audit=None):
        review = review or {"status": "pending_review", "reason": "Factual review not completed", "evidence_refs": []}
        attempt = attempt or {}
        behavior = behavior or {"status": "pending", "covers_plan": False, "executable": False}
        checked = (behavior.get("status") == "PASS" and behavior.get("covers_plan") is True
                   and behavior.get("executable") is True)
        if plan["action"] == "edit":
            checked = (checked and (binding or {}).get("status") == "PASS"
                and behavior.get("checks_id") == state["contract"]["checks_id"]
                and behavior.get("mechanism_id") == plan.get("verification_kind")
                and attempt.get("status") == "applied" and attempt.get("code_hash") != state["code_hash"])
        qualified = bool(_usable(parent) and _review_valid(review, state) and checked
                         and (plan["action"] == "keep" or (child is not None and _usable(child))))
        qualification = "verified" if qualified else "pending"
        if review.get("status") == "FAIL" or behavior.get("status") == "FAIL" or attempt.get("status") in {"rejected", "failed"}:
            qualification = "rejected"
        if independent_plan_audit and independent_plan_audit.get("decision") == "REJECT":
            qualification = "rejected"
        sample_id = "sample_" + digest({"state": state["state_id"], "plan": plan, "scope": scope})[:24]
        record = {"sample_id": sample_id, "candidate_id": scope, "state": copy.deepcopy(state),
            "plan": copy.deepcopy(plan), "protocol_id": protocol["protocol_id"],
            "root_lineage_id": state["root_lineage_id"], "state_id": state["state_id"],
            "source_kind": "fresh_teacher_plan_from_reused_parent",
            "parent_execution_provenance": copy.deepcopy(state["parent_execution_provenance"]),
            "qualification": {"status": qualification, "factual_review": review, "behavior": behavior,
                "applicability": binding, "proposal_task_result": parent,
                "implementation_task_result": child if plan["action"] == "edit" else None,
                "task_result": (child or {"status": "INCOMPLETE", "reward": None, "failure_layer": "unknown"}) if plan["action"] == "edit" else parent},
            "implementation": attempt, "mechanism_id": plan.get("verification_kind"),
            "teacher_path": state["teacher_record_path"], "teacher_record_path": state["teacher_record_path"],
            "source_refs": [state["state_id"], scope, parent["evaluation_id"],
                state["parent_execution_provenance"]["source_snapshot_sha256"],
                state["parent_execution_provenance"]["original_evaluation_id"]]
                + ([child["evaluation_id"]] if child and child.get("evaluation_id") else []),
            "revoked": False}
        if state.get("teacher_source_artifacts"):
            record["source_kind"] = "replayed_teacher_plan_admission"
            for field in ("original_teacher_record_path", "original_teacher_record_sha256", "original_state_id",
                    "original_state_path", "original_state_sha256", "original_preteacher_contract_sha256",
                    "teacher_source_campaign", "teacher_source_artifacts"):
                record[field] = copy.deepcopy(state[field])
            record["source_refs"] += [state["original_state_id"], state["original_teacher_record_sha256"], state["original_state_sha256"]]
            if state.get('replay_factual_from_source'):
                record['original_sample_id']=state['original_sample_id']
                record['replay_factual_from_source']=True
                record['source_refs'].append(state['original_sample_id'])
        if pending_reason:
            record["qualification"]["pending_reason"] = pending_reason
        if independent_plan_audit is not None:
            record["qualification"]["independent_plan_audit"] = copy.deepcopy(independent_plan_audit)
            if independent_plan_audit.get("request_sha256"):
                record["source_refs"].append(independent_plan_audit["request_sha256"])
            if independent_plan_audit.get("decision") == "REJECT":
                record["qualification"]["independent_rejection_reason"] = independent_plan_audit.get("reason")
        raw_path = Path(state["teacher_record_path"])
        if raw_path.is_file():
            record["teacher_prompt_sha256"] = json.loads(raw_path.read_text(encoding="utf-8")).get("prompt_sha256")
        previous = dict(ledger.record_items("samples")).get(sample_id)
        if previous and previous.get("revoked"):
            record.update(revoked=True, revocation_reason=previous.get("revocation_reason"))
        if qualification=='verified' and state.get('contract',{}).get('binding_mode')=='independent_plan_to_prefrozen_checks.v1':
            from research1.data_pipeline.admission_artifacts import seal_positive_artifacts
            if previous and previous.get('admission_artifacts'):
                record['admission_artifacts']=previous['admission_artifacts']
            record['admission_artifacts']=seal_positive_artifacts(record)
        ledger.put_record("samples", sample_id, record)
        return record

    with ledger.controller():
        try:
            save()
            rows = validate_sources(campaign, protocol)
            if len(rows) != 1 or rows[0]["task"] != task:
                raise ArtifactChangedError("Trial source is not its one frozen original parent")
            row = rows[0]
            parent = copy.deepcopy(row["parent_result"])
            if not _usable(parent):
                raise ArtifactChangedError("The frozen reused parent has no valid original score")
            _check_result_artifacts(parent, expected_task=task, expected_code_hash=tree_hash(baseline))
            provenance = source_provenance(row, protocol)
            prior = dict(ledger.record_items("init_parents")).get(parent_scope)
            if prior and (prior.get("parent_result") != parent or prior.get("parent_source") != provenance):
                raise ArtifactChangedError("Reused parent checkpoint changed")
            for old in ledger.records("samples"):
                _check_attempt(old["state"], old.get("implementation") or {})
                child = old.get("qualification", {}).get("implementation_task_result")
                if child and _usable(child):
                    _check_result_artifacts(child, expected_task=task, expected_code_hash=old["implementation"]["code_hash"])
            checkpoint = {"scope": parent_scope, "task": task, "status": "RUNNING", "parent_result": parent,
                "parent_source": provenance, "root_lineage_id": row["root_lineage_id"]}
            ledger.put_record("init_parent_executions", parent_scope, {"_operation_key": parent_scope, "value": parent})
            ledger.put_record("init_parent_source_reuses", parent_scope, provenance)
            ledger.put_record("init_parents", parent_scope, checkpoint)

            # A fresh fixture reference is hash-only in the protocol. Resolve
            # its full private review context after protocol freeze, and again
            # on resume, without running any fixture or model.
            fresh_fixture_source=None
            applicability_policy=protocol.get('policies',{}).get('applicability_reviewer',{})
            reference=applicability_policy.get('fixture_source_ref')
            if reference is not None:
                from research1.data_pipeline.fixture_context import freeze_probe_contract
                path=Path(reference['path'])
                if (not path.is_absolute() or not path.is_file() or path.is_symlink()
                        or hashlib.sha256(path.read_bytes()).hexdigest()!=reference.get('sha256')):
                    raise ArtifactChangedError('Fresh applicability fixture source reference changed')
                fresh_fixture_source=freeze_probe_contract(path)
                if fresh_fixture_source.get('sha256')!=reference['sha256']:
                    raise ArtifactChangedError('Fresh applicability fixture source changed during freezing')

            def base_state():
                if evidence_selector is not None:
                    evidence = evidence_selector(parent)
                else:
                    from research1.evidence import select_trajectory_evidence
                    evidence = select_trajectory_evidence(Path(parent["trajectory"]), **protocol["evidence_budget"])
                evidence = copy.deepcopy(evidence)
                for item in evidence:
                    item.pop("task_result", None)
                    item["id"] = task + ":" + item["id"]
                    item["proposal_task_id"] = task
                state = _public_state(baseline, evidence, [], row["root_lineage_id"], row["split"], [task], protocol)
                if protocol.get("framework_context") is not None:
                    from research1.data_pipeline.framework_context import build_framework_context
                    state["framework_context"] = build_framework_context(Path(protocol["upstream_root"]),
                        protocol, evaluation_spec=parent["evaluation_spec"])
                state.update(parent_execution_provenance=provenance, decision_protocol_hash=protocol["protocol_hash"],
                    origin_id=row["source_state_id"], allow_teacher_http_fallback=False)
                factual_policy=protocol.get('policies',{}).get('factual_reviewer',{}).get('version')
                if factual_policy:
                    state['factual_review_policy']=factual_policy
                if protocol.get('plan_precision_policy') is not None:
                    state['plan_precision_policy']=protocol['plan_precision_policy']
                if protocol.get('independent_plan_audit_policy') is not None:
                    state['independent_plan_audit_policy']=protocol['independent_plan_audit_policy']
                    state['independent_plan_audit_protocol_ref']=copy.deepcopy(
                        protocol.get('policies',{}).get('independent_plan_audit',{}))
                if applicability_policy.get('version'):
                    state['applicability_policy']=applicability_policy['version']
                if fresh_fixture_source is not None:
                    state['applicability_fixture_source']=copy.deepcopy(fresh_fixture_source)
                if config.get('fact_source'):
                    state['fact_source_artifacts']=copy.deepcopy(config['fact_source']['artifacts'])
                state["prior_proposal_outcomes"] = [{"proposal_task_id": task, "execution_id": parent["evaluation_id"],
                    "outcome": "PASS" if parent["reward"] == 1 else "FAIL" if parent["reward"] == 0 else "UNSCORED",
                    "failure_layer": parent["failure_layer"], "parent_code_hash": state["code_hash"],
                    "source_sha256": parent["result_sha256"], "finished_before_decision": True, "source_partition": "proposal"}]
                _add_addressable_feedback(state)
                return state
            teacher_replay = None
            if config.get("teacher_source"):
                from research1.data_pipeline.teacher_replay import load_teacher_replay
                teacher_replay = load_teacher_replay(protocol, row)
                base = copy.deepcopy(teacher_replay["base"])
                contract = copy.deepcopy(teacher_replay["contract"])
                ledger.put_record("edittrial_teacher_source", parent_scope,
                    {"source_campaign": teacher_replay["source_campaign"],
                     "artifacts": teacher_replay["source_artifacts"], "new_teacher_calls": 0})
                ledger.put_record("init_contracts", parent_scope, {"_operation_key": parent_scope, "value": contract})
                contract_copy = campaign / "edittrial_preteacher_contract.json"
                if contract_copy.exists():
                    if contract_copy.read_bytes() != teacher_replay["contract_bytes"]:
                        raise ArtifactChangedError("Copied original preteacher contract bytes changed")
                else:
                    import os, tempfile
                    with tempfile.NamedTemporaryFile("wb", dir=campaign, delete=False) as handle:
                        handle.write(teacher_replay["contract_bytes"])
                        temporary_contract = Path(handle.name)
                    os.replace(temporary_contract, contract_copy)
            else:
                base = _cached(ledger, "edittrial_base_state", parent_scope, base_state)
                if fresh_fixture_source is not None and base.get('applicability_fixture_source')!=fresh_fixture_source:
                    raise ArtifactChangedError('Cached fresh applicability fixture context changed')
                if publish_probe_io:
                    # Execute/cache the original parent probes once, before
                    # any fact-selector or teacher request. Publication only
                    # projects their saved I/O; it cannot change a check or
                    # rerun a probe.
                    initial_base=copy.deepcopy(base)
                    initial_contract=_cached(ledger,'edittrial_predecision_contracts',parent_scope,
                        lambda:build_contract(copy.deepcopy(initial_base)))
                    publication=_cached(ledger,'edittrial_probe_io_publication',parent_scope,
                        lambda:probe_io_publisher(copy.deepcopy(initial_base),copy.deepcopy(initial_contract),
                            campaign/'predecision_probe_io',protocol_hash=protocol['protocol_hash']))
                    if (not isinstance(publication,(tuple,list)) or len(publication)!=2
                            or not all(isinstance(value,dict) for value in publication)):
                        raise ArtifactChangedError('Parent probe publisher returned no bound state/contract')
                    base,contract=copy.deepcopy(publication)
                    for field in ('module_code','code_hash','modules_root','root_lineage_id','split',
                            'proposal_task_ids','history','permitted_modules','plan_policy_version',
                            'verification_registry_version','framework_context'):
                        if base.get(field)!=initial_base.get(field):
                            raise ArtifactChangedError('Parent probe publication changed frozen parent field: '+field)
                    evidence=base.get('evidence')
                    original_evidence=initial_base.get('evidence',[])
                    if not isinstance(evidence,list) or evidence[:len(original_evidence)]!=original_evidence:
                        raise ArtifactChangedError('Parent probe publication altered original proposal evidence')
                    allowed={'evidence_hashes','checks_id'}
                    if ({key:value for key,value in initial_contract.items() if key not in allowed}
                            !={key:value for key,value in contract.items() if key not in allowed}):
                        raise ArtifactChangedError('Parent probe publication changed frozen checks/scopes/probes')
                    if (contract.get('checks_id')!='checks_'+digest({key:value for key,value in contract.items() if key!='checks_id'})[:24]
                            or contract.get('evidence_hashes')!={row['id']:digest(row) for row in evidence}):
                        raise ArtifactChangedError('Published parent probe evidence/contract binding is invalid')
                    base['contract']=contract
                    base['verification_catalog']=module_adapter.proposal_catalog(base,contract)
                    try:
                        proofs=probe_io_validator(copy.deepcopy(base))
                    except (ValueError,TypeError,KeyError,OSError) as exc:
                        raise ArtifactChangedError('Saved parent probe publication no longer validates: '+str(exc)) from exc
                    if (not isinstance(proofs,(dict,list)) or not proofs
                            or isinstance(proofs,dict) and (proofs.get('status') in {'FAIL','ERROR'} or proofs.get('verified') is False)):
                        raise ArtifactChangedError('Saved parent probe publication no longer validates')
                    _immutable_json(campaign/'edittrial_probe_io_validation.json',proofs)
                    ledger.put_record('edittrial_probe_io_validations',parent_scope,{'proofs':proofs})
                    ledger.put_record('edittrial_contracts',parent_scope,{'_operation_key':parent_scope,'value':contract})
                    save('predecision_probe_io_published')
                facts_dir = campaign / "facts" / "edittrial-parent"
                try:
                    fact = _model_once(ledger, "edittrial_fact_cards", parent_scope, facts_dir / "fact_output.json",
                        lambda: fact_collector(copy.deepcopy(base), facts_dir))
                except (ArtifactChangedError, AmbiguousExternalCall):
                    raise
                except (ValueError, RuntimeError, OSError) as exc:
                    _raise_policy_failure(exc)
                    if re.search(r"immutable|different frozen input|frozen.*(?:source|input).*changed|artifact.*changed", str(exc), re.I):
                        raise ArtifactChangedError("Frozen fact input/source binding changed: " + str(exc)) from exc
                    raw_path = facts_dir / "fact_output.json"
                    if not raw_path.is_file():
                        raise AmbiguousExternalCall("Fact selection failed without a saved raw reply; no automatic resend") from exc
                    # Navigation is optional. Preserve an invalid selector reply and
                    # continue with AST interfaces plus all original evidence;
                    # never ask for a corrected quote or repair the model's facts.
                    reason = type(exc).__name__ + ": " + str(exc)
                    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    ledger.put_record("fact_index_error", parent_scope,
                        {"reason": reason, "raw_path": str(raw_path), "raw_sha256": raw_hash,
                         "disposition": "card omitted; original evidence retained; no retry"})
                    ledger.put_record("edittrial_fact_cards", parent_scope,
                        {"_operation_key": parent_scope, "value": None, "raw_path": str(raw_path),
                         "raw_sha256": raw_hash, "fact_index_error": reason})
                    fact = None
                interface = _cached(ledger, "edittrial_interface_cards", parent_scope,
                    lambda: interface_builder(copy.deepcopy(base)))
                base["fact_index"] = [value for value in (fact, interface) if value is not None]
                if config.get('submitted_code_index'):
                    from research1.data_pipeline.submitted_code import build_submitted_code_index
                    submitted,notes=_cached(ledger,'edittrial_submitted_code_index',parent_scope,
                        lambda:build_submitted_code_index(copy.deepcopy(base),max_facts=32))
                    if submitted is not None:
                        base['fact_index'].append(submitted)
                    base['submitted_code_index_metadata']=notes
                    _immutable_json(campaign/'facts'/'submitted_code_index_notes.json',notes)
                if not publish_probe_io:
                    contract = _cached(ledger, "edittrial_contracts", parent_scope,
                        lambda: build_contract(copy.deepcopy(base)))
                base["contract"] = contract
                base["verification_catalog"] = module_adapter.proposal_catalog(base, contract)
                ledger.put_record("init_contracts", parent_scope, {"_operation_key": parent_scope, "value": contract})
                _immutable_json(campaign / "edittrial_preteacher_contract.json", contract)
            prepared = []
            # All planned calls happen before any candidate review/child outcome.
            for arm in ARMS:
                arm_id = arm["id"]
                directory = campaign / "decisions" / "edittrial" / arm_id
                if teacher_replay is not None:
                    state = copy.deepcopy(teacher_replay["arms"][arm_id]["state"])
                else:
                    state = copy.deepcopy(base)
                    state.update(max_candidates=1, teacher_reasoning_effort=arm["reasoning_effort"],
                        teacher_arm_id=arm_id, teacher_record_path=str(directory / "teacher.json"))
                    if config.get('teacher_max_tokens'):
                        state['teacher_max_tokens']=config['teacher_max_tokens']
                    state["state_id"] = "state_" + digest({k: v for k, v in state.items() if k not in {"state_id", "modules_root"}})[:24]
                _immutable_json(campaign / "states" / state["state_id"] / "state.json", state)
                ledger.put_record("states", state["state_id"], state)
                scope = "edittrial:arm:" + arm_id
                try:
                    def propose_one():
                        plans = propose(copy.deepcopy(state), directory)
                        if not isinstance(plans, list) or len(plans) != 1:
                            raise ValueError("Each fixed trial arm must return exactly one honest plan")
                        return [validate_plan(plans[0], state)]
                    if teacher_replay is not None:
                        plans = _cached(ledger, "edittrial_replayed_proposals", arm_id,
                            lambda: copy.deepcopy(teacher_replay["arms"][arm_id]["plans"]))
                    else:
                        plans = _model_once(ledger, "edittrial_proposals", arm_id, directory / "teacher.json", propose_one)
                    outcome = {"arm_id": arm_id, "status": "COMPLETE", "plans": plans,
                               "teacher_record_path": state["teacher_record_path"], "state_id": state["state_id"]}
                    if teacher_replay is not None:
                        outcome.update(source_kind="replayed_teacher_plan_admission", new_teacher_calls=0,
                            original_state_id=state["original_state_id"],
                            original_teacher_record_sha256=state["original_teacher_record_sha256"])
                    plan = plans[0]
                    previous = dict(ledger.record_items("samples")).get("sample_" + digest({"state": state["state_id"], "plan": plan, "scope": scope})[:24])
                    if previous is None:
                        sample(state, plan, scope, parent)
                    prepared.append((arm_id, state, plan, scope))
                except ArtifactChangedError:
                    raise
                except (ValueError, RuntimeError, OSError) as exc:
                    _raise_policy_failure(exc)
                    outcome = {"arm_id": arm_id, "status": "AMBIGUOUS_CALL_NOT_RETRIED" if isinstance(exc, AmbiguousExternalCall) else "TEACHER_ERROR",
                        "reason": type(exc).__name__ + ": " + str(exc), "teacher_record_path": state["teacher_record_path"], "state_id": state["state_id"]}
                ledger.put_record("edittrial_teacher_outcomes", arm_id, outcome)
                save("all_teacher_arms")

            reviewed = []
            for arm_id, state, plan, scope in prepared:
                review_dir = campaign / "reviews" / "edittrial" / arm_id
                try:
                    if config.get('replay_factual_from_source'):
                        from research1.data_pipeline.teacher_replay import validate_replayed_factual_state
                        proof=validate_replayed_factual_state(state,plan)
                        review=copy.deepcopy(state['original_factual_review'])
                        replayed={'value':review,'proof':proof,'original_sample_id':state['original_sample_id'],
                            'original_state_id':state['original_state_id'],'new_factual_calls':0}
                        previous=dict(ledger.record_items('edittrial_replayed_factual_reviews')).get(arm_id)
                        if previous is not None and previous!=replayed:
                            raise ArtifactChangedError('Saved factual replay binding changed')
                        ledger.put_record('edittrial_replayed_factual_reviews',arm_id,replayed)
                    else:
                        review = _model_once(ledger, "edittrial_factual_reviews", arm_id, review_dir / "factual_review.json",
                            lambda: reviewer(copy.deepcopy(plan), copy.deepcopy(state)) if reviewer is not None
                            else review_plan(copy.deepcopy(state), copy.deepcopy(plan), review_dir))
                except ArtifactChangedError:
                    raise
                except (ValueError, RuntimeError, OSError) as exc:
                    _raise_policy_failure(exc)
                    review = {"status": "pending_review", "reason": type(exc).__name__ + ": " + str(exc), "evidence_refs": []}
                if not isinstance(review, dict):
                    review = {"status": "pending_review", "reason": "No structured factual review", "evidence_refs": []}
                state["plan_factual_review"] = review
                binding = None
                if plan["action"] == "edit":
                    binding = _cached(ledger, "edittrial_applicability", arm_id,
                        lambda: applicability_reviewer(copy.deepcopy(state), copy.deepcopy(plan), copy.deepcopy(review),
                            campaign / "applicability" / "edittrial" / arm_id))
                    state["plan_contract_binding"] = binding
                    if production_applicability and binding.get("status") == "PASS":
                        from research1.data_pipeline.applicability import validate_binding
                        if not validate_binding(binding, state, plan, contract):
                            raise ArtifactChangedError("Saved applicability binding is no longer valid")
                reviewed.append((arm_id, state, plan, scope, review, binding))
                old = dict(ledger.record_items("samples")).get("sample_" + digest({"state": state["state_id"], "plan": plan, "scope": scope})[:24])
                # On an audit-enabled resume, retain the committed audit and
                # its original sample until the barrier revalidates it. In
                # particular a terminal REJECT must not be erased by replaying
                # the already cached factual/applicability review.
                has_audit = independent_policy is not None and isinstance(
                    (old or {}).get('qualification', {}).get('independent_plan_audit'), dict)
                if not old or not old.get("implementation") and not has_audit:
                    sample(state, plan, scope, parent, review=review, binding=binding)
                save("all_factual_reviews")

            independent_arm_proofs = {}
            if independent_policy is not None:
                from research1.data_pipeline.independent_plan_audit import (
                    POLICY_ID as INDEPENDENT_AUDIT_POLICY, REPORT_NAME, RECEIPT_NAME,
                    freeze_audit_request, read_independent_audit, TERMINAL_DECISIONS,
                )
                protocol_path = campaign / "protocol.json"
                protocol_ref = {"path": str(protocol_path.resolve()),
                    "sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    "fixture_source_ref": protocol.get("policies", {}).get("applicability_reviewer", {}).get("fixture_source_ref")}
                awaiting = False
                for arm_id, state, plan, scope, review, binding in reviewed:
                    sample_id = "sample_" + digest({"state": state["state_id"], "plan": plan, "scope": scope})[:24]
                    existing_sample = dict(ledger.record_items("samples")).get(sample_id) or {}
                    committed = existing_sample.get("qualification", {}).get("independent_plan_audit")
                    if (not _review_valid(review, state)
                            or plan['action'] == 'edit' and (binding or {}).get('status') != 'PASS'):
                        if (existing_sample.get('implementation') or existing_sample.get('admission_artifacts')
                                or existing_sample.get('qualification', {}).get('implementation_task_result') is not None
                                or isinstance(committed, dict) and committed.get('decision') in TERMINAL_DECISIONS):
                            raise ArtifactChangedError('Previously admitted independent audit prerequisite no longer validates')
                        # Preserve the original DS/app outcome. An ineligible
                        # candidate needs no additional independent decision
                        # and can never enter the implementation branch.
                        continue
                    audit_dir = campaign / "independent_audits" / arm_id
                    request = freeze_audit_request(state, plan, review, binding, audit_dir,
                        protocol_ref=protocol_ref, campaign=campaign)
                    report_path = audit_dir / REPORT_NAME
                    receipt_path = audit_dir / RECEIPT_NAME
                    # Pin externally supplied bytes even before the companion
                    # file exists. A PENDING/invalid report cannot later be
                    # overwritten into a favorable decision on resume.
                    for external_path in (report_path, receipt_path):
                        observed_key = arm_id + ':' + external_path.name
                        observed = dict(ledger.record_items('independent_audit_observed_files')).get(observed_key)
                        if not external_path.is_file():
                            if observed is not None:
                                raise ArtifactChangedError('Previously observed independent audit artifact is missing')
                            continue
                        if external_path.is_symlink():
                            raise ArtifactChangedError('Independent audit artifact must be a regular file')
                        current = {'path': str(external_path.resolve()),
                            'sha256': hashlib.sha256(external_path.read_bytes()).hexdigest()}
                        if observed is not None and observed != current:
                            raise ArtifactChangedError('Previously observed independent audit artifact bytes changed')
                        if observed is None:
                            ledger.put_record('independent_audit_observed_files', observed_key, current)
                    request_path = str((audit_dir / "audit_request.json").resolve())
                    proof = {"schema": "research1.independent-plan-audit-proof.v1", "policy": INDEPENDENT_AUDIT_POLICY,
                        "request_sha256": request["request_sha256"], "request_path": request_path,
                        "status": "missing", "decision": None}
                    if report_path.is_file() and receipt_path.is_file():
                        try:
                            proof = read_independent_audit(request,
                                auditor_report_ref={"path": str(report_path.resolve())},
                                coordinator_receipt_ref={"path": str(receipt_path.resolve())},
                                record={"state": state, "plan": plan,
                                    "qualification": {"factual_review": review, "applicability": binding}})
                            proof["request_path"] = request_path
                            proof["request_record_sha256"] = request.get("request_record_sha256")
                        except ValueError:
                            proof = {"schema": "research1.independent-plan-audit-proof.v1",
                                "policy": INDEPENDENT_AUDIT_POLICY, "request_sha256": request["request_sha256"],
                                "request_path": request_path, "status": "invalid", "decision": None,
                                "reason": "External independent audit artifacts failed validation"}
                    if isinstance(committed, dict) and committed.get("decision") in TERMINAL_DECISIONS:
                        if proof.get("decision") not in TERMINAL_DECISIONS:
                            raise ArtifactChangedError("Committed independent audit missing on resume")
                        if proof.get("decision") != committed.get("decision"):
                            raise ArtifactChangedError("Committed independent audit decision cannot change")
                        for key in ("request_sha256", "request_record_sha256", "report_sha256", "coordinator_receipt_sha256",
                                    "auditor_task_provenance_sha256"):
                            if committed.get(key) and proof.get(key) != committed.get(key):
                                raise ArtifactChangedError("Committed independent audit artifact identity changed")
                    independent_arm_proofs[arm_id] = proof
                    if proof.get("decision") not in TERMINAL_DECISIONS:
                        awaiting = True
                    # Revalidation cannot replace an implemented or sealed
                    # sample with an empty preimplementation record. The
                    # existing patch, behavior, child result and seal remain
                    # intact; downstream _model_once still owns recovery of
                    # an incomplete implementation/child, without a new call.
                    preserve_existing = bool(existing_sample.get('implementation')
                        or existing_sample.get('admission_artifacts')
                        or existing_sample.get('qualification', {}).get('implementation_task_result') is not None
                        or existing_sample.get('qualification', {}).get('status') == 'verified')
                    if preserve_existing:
                        if not isinstance(committed, dict) or committed.get('decision') not in TERMINAL_DECISIONS:
                            raise ArtifactChangedError('Existing implementation lacks its original terminal independent audit')
                    else:
                        sample(state, plan, scope, parent, review=review, binding=binding, independent_plan_audit=proof)
                save("independent_plan_audit_frozen")
                if awaiting:
                    raise _StopInitialization("AWAITING_INDEPENDENT_PLAN_AUDIT",
                        "External independent plan audit required before any editor call", None)

            for arm_id, state, plan, scope, review, binding in reviewed:
                sample_id = "sample_" + digest({"state": state["state_id"], "plan": plan, "scope": scope})[:24]
                old = dict(ledger.record_items("samples")).get(sample_id)
                if old and old["qualification"]["status"] == "verified":
                    continue
                if independent_policy is not None:
                    audit = independent_arm_proofs.get(arm_id) or {}
                    if audit.get("decision") == "REJECT":
                        sample(state, plan, scope, parent, review=review, binding=binding,
                            independent_plan_audit=audit)
                        save("independent_plan_audit_rejected")
                        continue
                    if audit.get("decision") != "APPROVE":
                        continue
                if status["trial_goal_met"]:
                    if old and (old.get('implementation') or old.get('admission_artifacts')
                            or old.get('qualification', {}).get('implementation_task_result') is not None):
                        save("goal_reached_existing_arm_preserved")
                        continue
                    sample(state, plan, scope, parent, review=review, binding=binding,
                        pending_reason="One qualified edit reached the frozen trial goal; remaining fixed arm retained after generation and review, not implemented or replaced",
                        independent_plan_audit=independent_arm_proofs.get(arm_id) if independent_policy else None)
                    save("goal_reached_remaining_arm_preserved")
                    continue
                child, attempt = None, {}
                if plan["action"] == "keep":
                    attempt = {"status": "kept", "modules_root": state["modules_root"], "code_hash": state["code_hash"]}
                    behavior = {"status": "PASS", "covers_plan": True, "executable": True,
                        "mechanism_id": "explicit_keep", "reason": "Code unchanged; independent factual eligibility does not establish optimality"}
                elif not _review_valid(review, state) or (binding or {}).get("status") != "PASS":
                    behavior = {"status": "pending", "covers_plan": False, "executable": False,
                        "reason": "True factual PASS and valid pre-frozen applicability binding required before implementation"}
                else:
                    implementation_dir = campaign / "implementations" / "edittrial" / arm_id
                    if state.get('replay_factual_from_source'):
                        from research1.data_pipeline.teacher_replay import validate_replayed_factual_state
                        validate_replayed_factual_state(state,plan)
                    try:
                        attempt = _model_once(ledger, "edittrial_implementations", arm_id, implementation_dir / "editor_output.json",
                            lambda: implement(copy.deepcopy(plan), copy.deepcopy(state), implementation_dir))
                    except ArtifactChangedError:
                        raise
                    except (ValueError, RuntimeError, OSError) as exc:
                        _raise_policy_failure(exc)
                        attempt = {"status": "pending", "reason": type(exc).__name__ + ": " + str(exc)}
                    _check_attempt(state, attempt)
                    if attempt.get("status") == "applied":
                        behavior = _cached(ledger, "edittrial_verifications", arm_id,
                            lambda: verify(copy.deepcopy(plan), copy.deepcopy(state), copy.deepcopy(attempt), contract))
                        accepted = (behavior.get("status") == "PASS" and behavior.get("covers_plan") is True
                            and behavior.get("executable") is True and behavior.get("checks_id") == contract["checks_id"]
                            and behavior.get("mechanism_id") == plan.get("verification_kind")
                            and attempt["code_hash"] != state["code_hash"])
                        if accepted:
                            cached_child = dict(ledger.record_items("init_child_executions")).get(scope)
                            if cached_child is not None:
                                child = copy.deepcopy(cached_child["value"])
                            else:
                                child = evaluate_task(campaign, protocol, ledger, Path(attempt["modules_root"]), task,
                                    "proposal", "edittrial:child:" + arm_id, 1)
                                if child.get("in_progress"):
                                    raise _StopInitialization("WAITING_EXISTING_JOB", "Resume the existing child job; do not reserve a replacement", child)
                                ledger.put_record("init_child_executions", scope, {"_operation_key": scope, "value": child})
                            if child.get("fatal"):
                                raise _StopInitialization("BLOCKED_CONFIGURATION", child.get("error") or "Fatal child execution failure", child)
                            if _usable(child):
                                _check_result_artifacts(child, expected_task=task, expected_code_hash=attempt["code_hash"])
                    else:
                        behavior = {"status": "pending", "covers_plan": False, "executable": False,
                            "reason": attempt.get("reason", "No applied saved implementation")}
                sample(state, plan, scope, parent, review=review, binding=binding, attempt=attempt, behavior=behavior, child=child,
                    independent_plan_audit=independent_arm_proofs.get(arm_id) if independent_policy else None)
                save("qualifying_fixed_arm")
            checkpoint.update(status="DONE", raw_plans=status["counts"]["raw_plans"])
            ledger.put_record("init_parents", parent_scope, checkpoint)
            status.update(status="GROUNDED_EDIT_TRIAL_COMPLETE" if status["trial_goal_met"] else "GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT",
                phase="finished", finished_unix=time.time())
        except _StopInitialization as exc:
            status.update(status=exc.status, reason=str(exc), blocked_result=exc.result)
        except ArtifactChangedError as exc:
            for lineage in {row["root_lineage_id"] for row in ledger.records("samples")}:
                ledger.revoke(lineage, str(exc))
            status.update(status="BLOCKED_ARTIFACT_CHANGED", reason=str(exc))
        except BudgetExceeded as exc:
            status.update(status="BUDGET_EXHAUSTED", reason=str(exc))
        except AmbiguousExternalCall as exc:
            status.update(status="AMBIGUOUS_CALL_NOT_RETRIED", reason=str(exc))
        except Exception as exc:
            status.update(status="INCOMPLETE", reason=type(exc).__name__ + ": " + str(exc))
            ledger.add_event("edittrial_error", {"reason": status["reason"]})
        save()
    return status
