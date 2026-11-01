"""Independent parent-only attribution before candidate generation.

A model-reviewed witness is an attribution hypothesis, not causal proof. The
separate module-contract builder must reproduce its target and preservation
fixtures on the real immutable parent before selecting the mechanism.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
from typing import Any

from research1.data_pipeline.modules import frozen_registry
from research1.data_pipeline.policy import canonical, digest, public_view, _atomic_json, _witness_sources, FACTUAL_WITNESS_VALIDATOR

DIAGNOSTIC_POLICY = "proposal-parent-attribution.v4"
_FIELDS = {"module", "mechanism_id", "signal", "evidence_refs", "witness", "reason"}
_UNSUPPORTED_REASON = re.compile(
    r"\b(?:reward|score|scores|globally|global absence|defect[- ]free|all errors|only errors|no errors)\b", re.I
)


def _catalog() -> list[dict[str, str]]:
    return [{key: pack[key] for key in ("module", "mechanism_id", "signal", "coverage")}
            for pack in frozen_registry()["packs"].values()]


def _validate(rows: Any, state: dict[str, Any], *, request_id: str) -> tuple[list[dict], list[dict]]:
    if not isinstance(rows, list):
        raise ValueError("Parent diagnostic reply needs a diagnostics list")
    if len(rows) > len(frozen_registry()['packs']):
        raise ValueError("At most one diagnostic per registered mechanism")
    view = public_view(state)
    evidence = {row["id"]: row["text"] for row in view["evidence"]}
    packs = frozen_registry()["packs"]
    accepted, rejected, seen = [], [], set()
    for index, row in enumerate(rows):
        reason = None
        if (not isinstance(row, dict) or set(row) != _FIELDS
                or any(not isinstance(row[key], str) for key in _FIELDS - {"evidence_refs"})):
            reason = "Diagnostic shape differs from frozen attribution contract"
        else:
            pack = packs.get(row["mechanism_id"])
            refs = row["evidence_refs"]
            if (not pack or row["module"] != pack["module"] or row["signal"] != pack["signal"]
                    or row["module"] not in view["permitted_modules"]):
                reason = "Unknown or disallowed mechanism/module/signal"
            elif row["mechanism_id"] in seen:
                reason = "Duplicate mechanism; no second favourable review chosen"
            elif (not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in evidence for ref in refs)
                    or len(set(refs)) != len(refs)):
                reason = "Evidence references are missing or outside proposal view"
            elif (not isinstance(row["witness"], str) or not row["witness"].strip()
                    or not any(row["witness"] in segment["text"] for ref in refs
                               for segment in _witness_sources(evidence[ref]))):
                reason = "Witness is not an exact substring of referenced visible evidence"
            elif (not isinstance(row["reason"], str) or len(row["reason"].strip()) < 30
                    or _UNSUPPORTED_REASON.search(row["reason"])):
                reason = "Concrete bounded attribution required; reward/global absence cannot establish a process limitation"
            if pack:
                seen.add(row["mechanism_id"])
        if reason:
            rejected.append({"index": index, "reason": reason})
            continue
        item = dict(row)
        item['witness_validation'] = {'validator': FACTUAL_WITNESS_VALIDATOR,
            'serialized_evidence_sha256': {ref: hashlib.sha256(evidence[ref].encode('utf-8')).hexdigest() for ref in row['evidence_refs']},
            'matched_evidence_refs': [ref for ref in row['evidence_refs']
                if any(row['witness'] in segment['text'] for segment in _witness_sources(evidence[ref]))]}
        item["review"] = {"status": "PASS", "reviewer_id": DIAGNOSTIC_POLICY,
                          "independent": True, "review_type": "model_parent_attribution",
                          "request_id": request_id,
                          "limitation": "Proposal-visible hypothesis only; executable parent reproduction is separately required, not proof of task-score causation"}
        accepted.append(item)
    return accepted, rejected


def review_diagnostics(state: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    """Return reviewed diagnostic hypotheses, allowing an honest empty list.

    Only the policy-visible parent view and frozen registry descriptions are
    sent. State IDs, result blobs, current candidates and editor artifacts are
    not serialized. Invalid replies are retained but never repaired by retry.
    """
    from research1.distill import _tongji_chat
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    view = public_view(state)
    catalog = [row for row in _catalog() if row["module"] in view["permitted_modules"]]
    request = {"policy": DIAGNOSTIC_POLICY, "parent_view": view, "registry": catalog}
    request_id = digest(request)
    prompt = (
        "Independently review this parent decision-time Harness view and its supplied runtime interfaces, "
        "before any new candidate plan exists. Compare task requirements, actual observed workflow and "
        "visible Harness code to identify a concrete process limitation that could support a falsifiable "
        "improvement hypothesis. A limitation may be actionable even when the baseline operates as designed; "
        "a proven implementation defect, demonstrated benefit or cause of task outcomes is not required. "
        "Separate observed facts, current-code facts and unproven expected effects. Do not propose code or "
        "a change plan. Your specific role is to identify applicability of predeclared verification "
        "interfaces, not to decide whether the planner may propose any improvement. An empty diagnostics "
        "list means no supported attribution to this registry was found; it is not evidence for keep, "
        "absence of process-improvement opportunities, or permission to bypass executable verification. "
        "A task failure, a task reward, a generic error string, available test fixtures or absence in a bounded "
        "excerpt alone cannot establish a process limitation. Application code, its terminal echo or an "
        "application error alone is also insufficient; a specific connection between the observed workflow "
        "and the current Harness mechanism must be supported. A mechanism being listed is not evidence of "
        "an observed limitation or a reason to change it. Every diagnostic needs an exact "
        "unchanged witness substring from a referenced evidence text, concrete bounded reason, registered module, "
        "mechanism_id and signal. Do not invent or de-escape witness text. If evidence is uncertain, absent, "
        "outside these mechanisms, return an empty list and do not invent a registered match. The caller separately reproduces the target fixture "
        "on this exact parent. That fixture and your attribution do not establish that a proposed change "
        "will improve future task outcomes, solver compliance or long-run utility. Return final JSON only: "
        '{"diagnostics":[{"module":"...","mechanism_id":"...","signal":"...",'
        '"evidence_refs":["..."],"witness":"exact visible substring","reason":"bounded concrete attribution"}]}\n'
        + canonical(request)
    )
    input_path, output_path = directory / "diagnostic_input.json", directory / "diagnostic_output.json"
    input_record = {"request_id": request_id, "prompt": prompt, "policy": DIAGNOSTIC_POLICY}
    if input_path.exists() and json.loads(input_path.read_text(encoding="utf-8")) != input_record:
        raise ValueError("Frozen diagnostic input changed")
    _atomic_json(input_path, input_record)
    if output_path.exists():
        output = json.loads(output_path.read_text(encoding="utf-8"))
        if output.get("request_id") != request_id:
            raise ValueError("Saved diagnostics belong to another frozen parent")
        raw, metadata = output["raw"], output["metadata"]
    else:
        transport = ({'allow_http_fallback': False}
                     if state.get('allow_teacher_http_fallback') is False else {})
        raw, metadata = _tongji_chat(prompt, max_tokens=4000, role="data_parent_attribution",
                                    json_output=True, reasoning_effort="none", **transport)
        _atomic_json(output_path, {"request_id": request_id, "raw": raw, "metadata": metadata})
    if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop" or metadata.get("temperature") != 0:
        raise ValueError("Diagnostics require complete final JSON at frozen temperature zero")
    expected_model = state.get("diagnostic_model") or state.get("policy", {}).get("diagnostic_model")
    if expected_model and metadata.get("model") != expected_model:
        raise ValueError("Diagnostic model differs from frozen attribution policy")
    expected_provider = state.get("diagnostic_provider") or state.get("policy", {}).get("diagnostic_provider")
    if expected_provider and metadata.get("provider") != expected_provider:
        raise ValueError("Diagnostic provider differs from frozen attribution policy")
    signature = {key: metadata.get(key) for key in ("provider", "model", "temperature")}
    signature_path = directory.parent / "diagnostic_policy_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text(encoding="utf-8")) != signature:
        raise ValueError("Diagnostic provider/model/temperature changed within branch")
    _atomic_json(signature_path, signature)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or set(parsed) != {"diagnostics"}:
        raise ValueError("Parent attribution reply must contain only diagnostics")
    accepted, rejected = _validate(parsed["diagnostics"], state, request_id=request_id)
    _atomic_json(directory / "diagnostic_review.json", {"request_id": request_id, "diagnostics": accepted,
                 "rejected": rejected, "status": "REVIEWED", "causal_claim": False})
    return accepted
