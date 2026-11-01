"""Single-plan deployment contract and immutable, proposal-only teacher calls.

Identity, admission and evaluation records never become model input. The
multi-candidate teacher request is a collection operation; student messages
always request one Plan, not a jointly generated candidate set.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

LEGACY_PLAN_POLICY_ID = "proposal_single_plan_v2"
PREVIOUS_PLAN_POLICY_ID = "proposal_single_plan_v3"
PREVIOUS_PROCESS_POLICY_ID = "proposal_single_plan_v4"
PREVIOUS_PLANNING_POLICY_ID = "proposal_single_plan_v5"
POLICY_ID = "proposal_single_plan_v6"
FACTUAL_REVIEW_POLICY = "proposal-factual-review.v5"
FACTUAL_WITNESS_VALIDATOR = "proposal-factual-witness.v5"
PROBE_EVIDENCE_POLICY_ID = 'research1.predecision-harness-api-io.v1'
_PROBE_SOURCE_STRENGTH = (
    ' Additional source-strength rule: probe_evidence_policy identifies real predecision observations '
    'of this parent Harness local API. They can support bounded API facts and testable improvement '
    'hypotheses. They do not establish that the original task actually triggered the shown condition, '
    'that it caused a task failure, or that changing it will improve solver behavior or performance. '
    'These records are not agent trajectory steps. Do not upgrade them to agent execution/test success '
    'or a causal producer relationship, and do not treat their availability as a recommended repair.\n'
)


def probe_source_strength(view: dict) -> str:
    return _PROBE_SOURCE_STRENGTH if view.get('probe_evidence_policy') == PROBE_EVIDENCE_POLICY_ID else ''
MODULES = {"agent_loop", "observation", "tools", "context_mgmt", "verification"}
PLAN_FIELDS = {"action", "target_module", "locations", "evidence_refs",
               "problem_hypothesis", "intended_change", "preserve_behaviors",
               "verification_plan", "verification_kind"}
_FORBIDDEN = {
    "reward", "rewards", "score", "scores", "reward_score", "reward_scores", "dev_score",
    "dev_scores", "task_result", "task_results", "result", "results", "evaluation",
    "evaluations", "future", "future_results", "future_trajectory", "future_trajectories",
    "child_diff", "child_patch", "patch", "patch_text", "child_result", "child_results",
    "hidden_tests", "reference_solution", "reference_solutions", "answer_key",
    "long_horizon_label", "mc_label", "terminal_gain", "cumulative_regression", "u_lambda",
    "api_key", "authorization", "password", "secret", "access_token", "refresh_token",
}
_SYSTEM_V2 = (
    "You are the Harness evolver. Read only the supplied decision-time proposal evidence, "
    "current visible module code, and this branch's history. Return exactly ONE JSON Plan. "
    "Do not invent a defect, infer a defect from a task failure alone, claim global absence "
    "from a bounded excerpt, or output a patch, task answer, predicted score or future result. "
    "Use keep with a factual bounded reason when no edit is justified. An edit may touch "
    "only one permitted module and existing visible files. Plan fields are action(edit|keep), "
    "target_module(module name|null), locations(list of visible file paths), evidence_refs(list "
    "of supplied evidence ids), problem_hypothesis, intended_change, preserve_behaviors(list), "
    "verification_plan(list), verification_kind(string|null). Available verification IDs "
    "are interfaces, not evidence that a repair is needed. Unknown mechanisms remain pending. "
    "Prior proposal binary outcomes, if supplied, are finished execution feedback, not proof of a Harness defect."
)
_SYSTEM_V3 = (_SYSTEM_V2 + " A keep must still name specific factual checks and evidence limits in its "
              "verification_plan; no code edit does not mean no factual verification is required.")
_SYSTEM_V4 = (_SYSTEM_V3 +
    " Objective clarification: improve the Harness task-solving process through evidence-grounded, "
    "implementable changes. A grounded edit may address a concrete observed process limitation "
    "even when the baseline operates as designed; a proven implementation defect or demonstrated "
    "benefit is not required in advance. Distinguish observed facts and current-code facts from a "
    "testable mechanism hypothesis and its expected effects. Support the connection between the "
    "observed limitation, relevant Harness code and the proposed change with specific visible "
    "evidence and applicable behavior and preservation checks. Treat future utility and causal "
    "benefit as unproven until subsequent evaluation. A task outcome, an application error, a "
    "catalog entry or missing evidence alone does not establish a Harness limitation. Choose "
    "keep when no sufficiently grounded change is available. Neither action is preferred.")

_SYSTEM_V5 = (
    "You are the Harness evolver. Your objective is to improve the task-solving process through "
    "evidence-grounded, implementable and falsifiable changes to the current Harness. Read only "
    "the supplied decision-time task requirements, proposal execution evidence, visible module "
    "code, framework interface context and this branch's history. Return exactly ONE JSON Plan. "
    "First compare what the task requires with the actual observed workflow. Then connect a "
    "specific observed process limitation or improvement opportunity to a visible Harness "
    "function and the supplied runtime interfaces. Form a mechanism hypothesis explaining how "
    "one bounded change could affect that workflow, and specify a check that could falsify it. "
    "Finally choose edit or keep from that comparison. Report only the concise final plan, "
    "with observed facts, current-code facts and unproven expected effects clearly distinguished. "
    "A grounded edit may address a process limitation even when the baseline operates as "
    "designed; a proven implementation defect, demonstrated benefit or proven cause of the "
    "task outcome is not required in advance. A plausible connection needs specific visible "
    "evidence and code, rather than a generic best practice or an application error alone. "
    "For edit, describe the smallest feasible single-module intervention, its expected mechanism, "
    "an observable target behavior, behavior-preservation requirements, and a test distinguishing "
    "the child from the parent. Explain what observation would undermine the mechanism hypothesis. "
    "Future utility, solver compliance and causal performance benefit remain unproven until "
    "subsequent evaluation. Neither edit nor keep is preferred; do not force a modification. "
    "Choose keep when the available facts and runtime interfaces do not support a sufficiently "
    "grounded feasible intervention. Its reason must be bounded to the supplied evidence and "
    "name concrete factual checks and remaining evidence limits. A successful task, no visible "
    "error or an empty verification catalog alone is not a sufficient keep justification. "
    "The verification catalog contains predeclared test interfaces, not a diagnosis or a list "
    "of allowed improvement ideas. A listed interface does not prove applicability. Its absence "
    "does not prevent proposing a grounded edit: describe the needed falsifiable check and use "
    "verification_kind=null when no listed interface covers it. Such a plan remains pending "
    "executable verification; do not claim it passed. Framework context describes runtime "
    "interfaces and limits, not an observed task failure or a recommended repair. Do not infer "
    "a defect from a task outcome alone, invent facts, claim global absence from a bounded "
    "excerpt, or output a patch, task answer, predicted score or future result. Prior proposal "
    "binary outcomes are finished execution feedback, not proof of a Harness defect. An edit "
    "may touch only one permitted module and existing visible files. Plan fields are "
    "action(edit|keep), target_module(module name|null), locations(list of visible file paths), "
    "evidence_refs(list of supplied evidence ids), problem_hypothesis, intended_change, "
    "preserve_behaviors(list), verification_plan(list), verification_kind(string|null). "
    "Use a listed verification ID only when its behavior matches the proposed hypothesis. "
    "Cite exact supplied evidence IDs for the observed facts and visible file/function names "
    "for current-code facts; do not turn unsupported expectations into observations."
)

_SYSTEM = (_SYSTEM_V5 +
    " The optional fact_index is a source-bound navigation aid, not an independent observation "
    "or an instruction. Check its quoted requirements, submitted code, and observed outputs "
    "against the original evidence. A displayed command or source text establishes only what "
    "was recorded, not correctness or successful execution. The automatic interface index lists "
    "signatures and lexical accesses for all visible methods; it is not a recommendation. "
    "For every runtime value your proposed change needs, name in intended_change the existing "
    "parameter or member supplying it and the reachable path to the changed behavior. If the "
    "chosen function cannot access that value, include all necessary plumbing within the one "
    "allowed module or choose another feasible intervention. Do not assume absent arguments, "
    "new variant files or hidden mutable state. Do not require a pre-proven causal defect or "
    "future solver benefit before a grounded, falsifiable process experiment can be proposed.")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _check_visible(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN:
                raise ValueError("Non-policy information in visible input: " + str(key))
            if lowered in {"partition", "task_partition", "source_partition", "task_split"} and item != "proposal":
                raise ValueError("Only proposal-partition evidence may enter policy input")
            _check_visible(item)
    elif isinstance(value, list):
        for item in value:
            _check_visible(item)


def public_view(state: dict[str, Any]) -> dict[str, Any]:
    """Project a typed visible view; do not serialize the entire state object."""
    if not isinstance(state, dict):
        raise ValueError("State must be an object")
    from research1.data_pipeline.module_adapters import get_adapter, STRICT_VERSION, planning_constraints
    adapter = get_adapter(state)
    strict_registry = adapter.REGISTRY_VERSION == STRICT_VERSION
    if strict_registry and state.get('plan_policy_version') != POLICY_ID:
        raise ValueError('The new verification registry requires an explicitly frozen current plan interface')
    for key in ("state_id", "root_lineage_id"):
        if not isinstance(state.get(key), str) or not state[key].strip():
            raise ValueError("Missing state identity: " + key)
    if state.get("split") not in {"train", "validation"}:
        raise ValueError("State split must be train or validation")
    modules = state.get("permitted_modules")
    if (not isinstance(modules, list) or not modules or any(not isinstance(m, str) or m not in MODULES for m in modules)
            or len(set(modules)) != len(modules)):
        raise ValueError("permitted_modules must name supported Harness modules")
    code = state.get("module_code")
    if not isinstance(code, dict) or not code or any(not _visible_path(p) or p.split("/", 1)[0] not in MODULES | {"tool_helper"}
            or not isinstance(v, str) or not v.strip() for p, v in code.items()):
        raise ValueError("module_code must contain existing visible relative files")
    evidence = state.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("State requires proposal evidence")
    rows = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Evidence must be an object")
        _check_visible(item)
        if not isinstance(item.get("id"), str) or not item["id"].strip() or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("Evidence needs a stable id and nonempty visible text")
        rows.append({"id": item["id"], "text": item["text"]})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evidence ids must be unique")
    history = state.get("history", [])
    if not isinstance(history, list):
        raise ValueError("History must be a list of decision-time records")
    _check_visible(history)
    # Explicit histories contain plans, their real application status, and
    # proposal-visible feedback, never result blobs or code patches.
    allowed_history = {"decision_id", "action", "plan", "implementation_status", "verification_feedback", "evidence_refs", "note"}
    for item in history:
        if not isinstance(item, dict) or set(item) - allowed_history:
            raise ValueError("History contains fields outside the visible decision contract")
    catalog = state.get("verification_catalog", [])
    if not isinstance(catalog, list):
        raise ValueError("verification_catalog must be a list of frozen interface ids")
    for item in catalog:
        if (not isinstance(item, dict) or set(item) != {"mechanism_id", "module"}
                or not isinstance(item["mechanism_id"], str) or not item["mechanism_id"].strip()
                or item["module"] not in modules):
            raise ValueError("Verification catalog exposes only a mechanism id and permitted module")
    if len({item["mechanism_id"] for item in catalog}) != len(catalog):
        raise ValueError("Verification interface ids must be unique")
    view={"module_code": code, "evidence": rows, "history": history,
          "permitted_modules": list(modules), "verification_catalog": catalog}
    if state.get('predecision_probe_io_policy') is not None:
        if state['predecision_probe_io_policy'] != PROBE_EVIDENCE_POLICY_ID:
            raise ValueError('Unknown predecision probe evidence policy')
        from research1.data_pipeline.predecision_probe_io import SCHEMA, validate_probe_io_state
        if SCHEMA != PROBE_EVIDENCE_POLICY_ID:
            raise ValueError('Predecision probe publisher policy differs from the frozen interface')
        proof=validate_probe_io_state(state)
        if not isinstance(proof,dict) or proof.get('schema')!=SCHEMA or proof.get('verified') is not True:
            raise ValueError('Predecision probe source did not establish its physical artifact binding')
        view['probe_evidence_policy'] = SCHEMA
        references=proof.get('evidence_ids_added')
        if (not isinstance(references,list) or any(not isinstance(ref,str) or not ref for ref in references)
                or len(set(references))!=len(references) or not set(references).issubset({row['id'] for row in rows})):
            raise ValueError('Physical predecision probe proof has invalid evidence references')
        view['predecision_probe_evidence_refs'] = list(references)
    if strict_registry:
        view['verification_registry_version'] = STRICT_VERSION
        view['planning_constraints'] = planning_constraints()
    if 'framework_context' in state:
        from research1.data_pipeline.framework_context import validate_framework_context
        validate_framework_context(state['framework_context'])
        view['framework_context'] = state['framework_context']
    if 'verification_interfaces' in state:
        if state['verification_interfaces'] != adapter.planning_interfaces(modules,
                include_scope=state.get('plan_policy_version')==POLICY_ID):
            raise ValueError('Verification interface descriptions differ from the frozen registry')
        view['verification_interfaces'] = state['verification_interfaces']
    if state.get('plan_policy_version') == POLICY_ID:
        from research1.data_pipeline.fact_cards import project_fact_index
        index = project_fact_index(state)
        if index is not None:
            view['fact_index'] = index
    outcomes=state.get('prior_proposal_outcomes',[])
    if not isinstance(outcomes,list):raise ValueError('Prior proposal outcomes must be a list')
    if outcomes:
        required={'proposal_task_id','execution_id','outcome','failure_layer','parent_code_hash','source_sha256','finished_before_decision','source_partition'}
        tasks=set(state.get('proposal_task_ids',[]))
        for item in outcomes:
            if not isinstance(item,dict) or set(item)!=required:raise ValueError('Prior proposal feedback may not contain scores or tests')
            if item['proposal_task_id'] not in tasks or item['source_partition']!='proposal' or item['finished_before_decision'] is not True:
                raise ValueError('Only finished prior proposal feedback is visible')
            if item['outcome'] not in {'PASS','FAIL','UNSCORED'}:raise ValueError('Unsupported proposal outcome encoding')
            if not isinstance(item['execution_id'],str) or not item['execution_id']:raise ValueError('Proposal execution provenance missing')
            if not isinstance(item['parent_code_hash'],str) or item['parent_code_hash']!=state.get('code_hash') or not re.fullmatch(r'[0-9a-f]{64}',item['parent_code_hash']):
                raise ValueError('Proposal feedback belongs to another parent')
            if not isinstance(item['source_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',item['source_sha256']):
                raise ValueError('Proposal source checksum missing')
            if item['failure_layer'] not in {'model','harness','infrastructure','container','scorer','unknown'}:raise ValueError('Unsupported proposal failure layer')
        if len({item['execution_id'] for item in outcomes})!=len(outcomes):raise ValueError('Duplicate prior proposal execution')
        view['prior_proposal_outcomes']=outcomes
    if state.get('plan_precision_policy') is not None:
        from research1.data_pipeline.plan_precision import POLICY_ID as PLAN_PRECISION_POLICY_ID, public_rules
        if state['plan_precision_policy'] != PLAN_PRECISION_POLICY_ID:
            raise ValueError('Unknown plan precision policy')
        view['plan_precision'] = {'policy': PLAN_PRECISION_POLICY_ID, 'rules': public_rules()}
    return view


def _visible_path(path: Any) -> bool:
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        return False
    parsed = PurePosixPath(path)
    return not parsed.is_absolute() and ".." not in parsed.parts and "." not in parsed.parts and str(parsed) == path


def validate_plan(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    view = public_view(state)
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        raise ValueError("Plan must contain exactly the single-plan contract fields")
    _check_visible(plan)
    if plan["action"] not in {"edit", "keep"}:
        raise ValueError("Invalid plan action")
    for key in ("problem_hypothesis", "intended_change"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError("Missing plan " + key)
    for key in ("locations", "evidence_refs", "preserve_behaviors", "verification_plan"):
        values = plan[key]
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError("Invalid plan list: " + key)
    refs = {e["id"] for e in view["evidence"]}
    if not plan["evidence_refs"] or not set(plan["evidence_refs"]).issubset(refs):
        raise ValueError("Plan references missing or unknown evidence")
    target = plan["target_module"]
    if target is not None and target not in view["permitted_modules"]:
        raise ValueError("Plan targets a non-permitted module")
    for path in plan["locations"]:
        if path not in view["module_code"]:
            raise ValueError("Plan location is not an existing visible file")
        module = path.split("/", 1)[0]
        if module == "tool_helper":
            module = "tools"
        if target != module:
            raise ValueError("Each modification is limited to its declared single module")
    kind = plan["verification_kind"]
    if kind is not None and (not isinstance(kind, str) or not kind.strip()):
        raise ValueError("verification_kind must be a nonempty string or null")
    if plan["action"] == "edit" and (target is None or not plan["locations"] or not plan["preserve_behaviors"] or not plan["verification_plan"]):
        raise ValueError("Edit needs a target, locations, preservation and verification requirements")
    return json.loads(canonical(plan))


def render_messages(state: dict[str, Any], plan: dict[str, Any] | None = None) -> list[dict[str, str]]:
    # Previously collected states did not store a separate instruction version.
    # Their original canonical instruction must survive later export/review.
    version = state.get("plan_policy_version", LEGACY_PLAN_POLICY_ID)
    systems = {LEGACY_PLAN_POLICY_ID: _SYSTEM_V2, PREVIOUS_PLAN_POLICY_ID: _SYSTEM_V3,
               PREVIOUS_PROCESS_POLICY_ID: _SYSTEM_V4, PREVIOUS_PLANNING_POLICY_ID: _SYSTEM_V5, POLICY_ID: _SYSTEM}
    if version not in systems:
        raise ValueError("Unsupported frozen plan instruction version")
    system=systems[version]
    if version==POLICY_ID and state.get('fact_index_projection')=='compact_v1':
        system += (' Read the compact fact_index as navigation first, then verify its facts against the original sources. '
            'fact_id and card_id identify navigation anchors; Plan.evidence_refs must cite the original evidence_ref IDs, '
            'not fact_id or card_id. Submitted-code quotations establish what commands contained, not that a file was '
            'successfully written or executed. No navigation entry is a recommended intervention.')
    from research1.data_pipeline.module_adapters import get_adapter, STRICT_VERSION
    if get_adapter(state).REGISTRY_VERSION == STRICT_VERSION:
        system += (' The explicit planning_constraints require one concrete feasible implementation path. '
            'Name exact existing fully-qualified entry-point symbols and only predeclared helper names; '
            'do not offer alternative or/or routes, add classes or registrations, or assume hidden runtime '
            'values or aliases between distinct runtime types. These are general feasibility constraints, '
            'not a diagnosis or a recommended repair.')
    view=public_view(state)
    system += probe_source_strength(view)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": canonical(view)}]
    if plan is not None:
        messages.append({"role": "assistant", "content": canonical(validate_plan(plan, state))})
    return messages


def parse_candidates(raw: str | dict[str, Any], state: dict[str, Any], max_candidates: int = 2) -> list[dict[str, Any]]:
    if max_candidates not in {1, 2}:
        raise ValueError("Candidate count must be 1 or 2")
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("Teacher output must be a JSON candidates object")
    plans = value["candidates"]
    if not isinstance(plans, list) or not 1 <= len(plans) <= max_candidates:
        raise ValueError("Teacher must return one or two honest candidates")
    result, seen = [], set()
    for plan in plans:
        plan = validate_plan(plan, state)
        signature = canonical(plan)
        if signature not in seen:
            seen.add(signature)
            result.append(plan)
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        name = handle.name
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(name, path)


def collect_candidates(state: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    """Save final teacher output before validating; failed replies are not retried."""
    from research1.distill import _tongji_chat

    max_candidates = state.get("max_candidates", 2)
    if max_candidates not in {1, 2}:
        raise ValueError("Candidate count must be 1 or 2")
    messages = render_messages(state)
    prompt = (messages[0]["content"] + "\nCollection request: generate at most " + str(max_candidates)
              + ' distinct plans before any candidate outcome is known. Return JSON {"candidates":[Plan,...]} instead of the deployment single-plan envelope. '
              + "Do not force a second edit or fill a candidate quota.\nFrozen proposal-only view:\n" + messages[1]["content"])
    path = Path(directory) / "teacher.json"
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    effort = state.get('teacher_reasoning_effort', 'none')
    if effort not in {'none', 'low'}:
        raise ValueError('Unsupported frozen teacher reasoning setting')
    output_limit=state.get('teacher_max_tokens',8000)
    if output_limit not in {8000,16000} or isinstance(output_limit,bool):
        raise ValueError('Unsupported frozen teacher output budget')
    settings = {'reasoning_effort':effort, 'max_tokens':output_limit, 'temperature':0,
                'allow_http_fallback':state.get('allow_teacher_http_fallback') is not False}
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("prompt_sha256") != prompt_hash or saved.get("prompt") != prompt:
            raise ValueError("Teacher input changed for an immutable collection directory")
        if 'teacher_reasoning_effort' in state and saved.get('request_settings') != settings:
            raise ValueError('Teacher request settings changed for an immutable collection directory')
        raw, metadata = saved["raw"], saved["metadata"]
    else:
        transport = ({'allow_http_fallback': False}
                     if state.get('allow_teacher_http_fallback') is False else {})
        raw, metadata = _tongji_chat(prompt, max_tokens=output_limit, role="research1_data_teacher",
                                     json_output=True, reasoning_effort=effort, **transport)
        saved = {"policy_id": state.get("plan_policy_version", LEGACY_PLAN_POLICY_ID),
                 "prompt": prompt, "prompt_sha256": prompt_hash,
                 "raw": raw, "metadata": metadata, "visible_view_sha256": digest(public_view(state)),
                 'request_settings':settings}
        _atomic_json(path, saved)
    if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop" or metadata.get("temperature") != 0:
        raise ValueError("Teacher requires complete final JSON at frozen temperature zero")
    expected = state.get("teacher_model")
    if expected and metadata.get("model") != expected:
        raise ValueError("Teacher model differs from the frozen policy")
    expected_provider = state.get("teacher_provider")
    if expected_provider and metadata.get("provider") != expected_provider:
        raise ValueError("Teacher provider differs from the frozen policy")
    if 'teacher_reasoning_effort' in state and metadata.get('reasoning_effort') != effort:
        raise ValueError('Teacher reasoning request differs from the frozen policy')
    return parse_candidates(raw, state, max_candidates)


def _witness_sources(text: str) -> list[dict[str, Any]]:
    """Decode the selector envelope once; never unescape a quoted witness.

    Selector text is a JSON serialization, while observation segments contain
    the actual visible characters. Exactness is checked against those strings,
    retaining their offsets and the hash of the unchanged serialization.
    """
    try:
        selected = json.loads(text)
    except (ValueError, TypeError):
        selected = None
    if not isinstance(selected, dict) or not isinstance(selected.get("steps"), list):
        return [{"field": "plain_text", "text": text}]
    sources = []
    for step_index, step in enumerate(selected["steps"]):
        if not isinstance(step, dict):
            continue
        fields = [(name, step.get(name)) for name in ("message_excerpt", "observation_excerpt")]
        for call_index, call in enumerate(step.get("tool_calls") or []):
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                continue
            fields.append((f"tool_calls/{call_index}/keystrokes_excerpt",
                           call["arguments"].get("keystrokes_excerpt")))
        for field, excerpt in fields:
            if not isinstance(excerpt, dict):
                continue
            for segment_index, segment in enumerate(excerpt.get("segments") or []):
                if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                    sources.append({"field": field, "step_index": step_index,
                        "step_id": step.get("step_id"), "source_index": step.get("source_index"),
                        "visible_step_sha256": step.get("visible_step_sha256"),
                        "segment_index": segment_index, "segment_start": segment.get("start"),
                        "text": segment["text"]})
    return sources


def _review_witnesses(value: Any, state: dict[str, Any], action: str) -> list[dict[str, Any]]:
    from research1.data_pipeline.observation_validator import validate_witnesses
    return validate_witnesses(value, public_view(state), action)


def review_plan(state: dict[str, Any], plan: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Independent model factual review; executable behavior checks remain separate.

    A PASS is a scoped model review, not a claim of human-verified truth or of
    long-term benefit. All reviews are saved separately from teacher output.
    """
    if state.get('factual_review_policy')=='proposal-factual-review.v6.typed':
        from research1.data_pipeline.typed_review import review_plan_typed
        return review_plan_typed(state,plan,directory)
    if state.get('factual_review_policy') in {'proposal-factual-review.v7.selectors','proposal-factual-review.v8.probe_io'}:
        from research1.data_pipeline.selector_review import review_plan_selected
        return review_plan_selected(state,plan,directory)
    from research1.distill import _tongji_chat

    plan = validate_plan(plan, state)
    view = public_view(state)
    prompt = (
        "Factual review policy: " + FACTUAL_REVIEW_POLICY + ". "
        "Independently check the candidate's factual justification against ONLY this frozen "
        "proposal-only view and visible module code. Do not invent evidence or infer correctness "
        "from a task score. Distinguish bounded abstention from global absence: 'no defect is "
        "evidenced in this supplied bounded view' means the available evidence does not justify "
        "a specific edit; it does not mean no defects exist. Do not demand global correctness "
        "proof for a bounded keep. Independently verify the candidate's concrete source and "
        "execution facts and whether they support that bounded decision. A score alone or "
        "merely not seeing errors is insufficient. A recorded failure_layer is a saved "
        "classification, not a proven cause; accurately reporting that classification is "
        "permitted, attributing causal responsibility requires separate evidence. "
        "Omitted observations cannot prove global absence of defects. Check claims that errors were diagnosed/resolved; "
        "For an edit, review concrete observations and current-code claims separately from its "
        "expected benefit. A proposed change may address an observed process limitation even "
        "when the baseline operates as designed. Check that specific visible facts and relevant "
        "Harness code support a plausible implementable mechanism and a plan to test its behavior "
        "and preservation; do not require a proven implementation defect or proof of future "
        "performance benefit. Explicitly qualified expected effects are hypotheses, not observed "
        "outcomes. Unsupported causal guarantees or invented facts must remain pending_review. "
        "Neither edit nor keep is preferred. "
        "unresolved, overstated or uncertain facts must stay pending_review. Do not repair or "
        "rewrite the candidate. Return JSON {status:PASS|pending_review|FAIL, reason:string, "
        "evidence_refs:[supplied ids], witnesses:[{evidence_ref:id,text:exact visible substring, "
        "kind:observed_tool_output|execution_status|check_status|failure_behavior|context}]}. "
        "For PASS provide exact witnesses for the concrete facts that justify the plan. "
        "KEEP needs a substantive actual execution observation supporting a specific bounded "
        "fact, not reward, absence, a command merely echoing success, or a reviewer verdict. "
        "Use observed_tool_output for an actual returned body or other output; this establishes "
        "what was observed and does not establish HTTP success, exit success, test success or "
        "task correctness. Use execution_status only with an explicit execution/HTTP status "
        "observation; check_status only with an actual check-result observation. Match the "
        "strength of every candidate claim to the evidence: arbitrary JSON is not success "
        "proof, while a factual bounded keep need not contain a particular success phrase. "
        "Quote decoded observation characters, never unexecuted command/message text as "
        "execution output. An observed_tool_output witness is captured terminal text at step granularity, not a causal producer or success claim. A command echo alone cannot justify keep; mixed command-plus-real-output quotations are allowed. Prefer context for displayed commands. Do "
        "not use a displayed command as evidence of its output or success. Use context for "
        "serialized envelope metadata such as omitted_step_ids. An empty-keystroke polling "
        "capture may establish what text was observed, but not which command produced it, "
        "whether it is fresh, or whether execution/checks succeeded. Do not upgrade a polling "
        "observation to causal, freshness or success claims. Do not attribute visible outcome metadata to an evidence id whose "
        "text does not contain it. This is a factual review only, not executable "
        "verification or optimality assessment.\nFrozen visible view:\n" + canonical(view)
        + "\nCandidate:\n" + canonical(plan)
    )
    path = Path(directory) / "factual_review.json"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("prompt") != prompt or saved.get("prompt_sha256") != prompt_hash:
            raise ValueError("Factual review input changed in an immutable directory")
        raw, metadata = saved["raw"], saved["metadata"]
    else:
        raw, metadata = _tongji_chat(prompt, max_tokens=5000, role="research1_data_factual_review",
                                     json_output=True, reasoning_effort="none")
        saved = {"prompt": prompt, "prompt_sha256": prompt_hash, "raw": raw, "metadata": metadata,
                 "review_policy": FACTUAL_REVIEW_POLICY,
                 "review_type": "model_fact_review", "visible_view_sha256": digest(view), "plan_sha256": digest(plan)}
        _atomic_json(path, saved)
    result = {"status": "pending_review", "reason": "Factual review is not validated", "evidence_refs": [],
              "independent": True, "review_type": "model_fact_review",
              "review_policy": FACTUAL_REVIEW_POLICY,
              "witness_validator": FACTUAL_WITNESS_VALIDATOR,
              "evidence_text_sha256": {item["id"]: hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                                       for item in view["evidence"]},
              "reviewer_id": "factual_review_" + digest({"prompt": prompt_hash, "model": metadata.get("model"),
                                                         "provider": metadata.get("provider")})[:24],
              "review_path": str(path), "plan_sha256": digest(plan), "visible_view_sha256": digest(view)}
    try:
        if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop" or metadata.get("temperature") != 0:
            raise ValueError("Review requires complete final content at frozen temperature zero")
        expected = state.get("review_model", state.get("teacher_model"))
        if expected and metadata.get("model") != expected:
            raise ValueError("Review model differs from the frozen review policy")
        expected_provider = state.get("review_provider")
        if expected_provider and metadata.get("provider") != expected_provider:
            raise ValueError("Review provider differs from the frozen review policy")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("status") not in {"PASS", "pending_review", "FAIL"}:
            raise ValueError("Invalid factual review status")
        reason, refs = value.get("reason"), value.get("evidence_refs")
        visible_refs = {item["id"] for item in view["evidence"]}
        if (not isinstance(reason, str) or not reason.strip() or not isinstance(refs, list) or not refs
                or any(not isinstance(ref, str) or ref not in visible_refs for ref in refs)):
            raise ValueError("Review needs a reason and supplied evidence references")
        if value["status"] == "PASS":
            witnesses = _review_witnesses(value.get("witnesses"), state, plan["action"])
            if any(witness["evidence_ref"] not in refs for witness in witnesses):
                raise ValueError("Witness references must also be cited by the review")
            claims = plan["problem_hypothesis"] + " " + plan["intended_change"]
            if re.search(r"\b(?:globally\s+defect[- ]free|defect[- ]free\s+(?:everywhere|globally)|no\s+(?:defects?|warnings?|errors?)\s+(?:exist\s+)?(?:anywhere|everywhere|in\s+all\s+tasks))\b", claims, re.I):
                raise ValueError("Blanket global absence claim is unsupported by a bounded view")
            result["witnesses"] = witnesses
        result.update(status=value["status"], reason=reason, evidence_refs=refs)
    except (ValueError, TypeError, KeyError) as exc:
        result["reason"] = str(exc)
    _atomic_json(Path(directory) / "factual_review_result.json", result)
    return result
