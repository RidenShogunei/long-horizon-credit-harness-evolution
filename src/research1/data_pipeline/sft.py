"""Versioned SFT views, separate from raw evidence and long-horizon targets.

The controller passes the current admission revisions. This exporter never
promotes an execution merely because it has a score, and never puts outcome
metadata into the model's single-plan messages.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from research1.data_pipeline.policy import canonical, digest, render_messages, validate_plan

SCHEMA = "research1.sft.single_plan.v1"
_LEGACY_SOURCES = {"archived_teacher_plan_reimplementation", "directed_behavior_repair_init",
                   "auxiliary_warning_cleanup", "legacy", "legacy_init", "directed_repair"}


def _valid_task(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    score = result.get("reward")
    return bool(result.get("status") in {"PASS", "FAIL"}
                and result.get("failure_layer") in {"model", "harness"}
                and isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(score))


def _review_valid(review: Any, state: dict[str, Any]) -> bool:
    if not isinstance(review, dict):
        return False
    refs = review.get("evidence_refs", review.get("refs"))
    ids = {item["id"] for item in state["evidence"]}
    return bool(review.get("status") == "PASS" and review.get("independent") is True
                and isinstance(review.get("reviewer_id"), str) and review["reviewer_id"].strip()
                and isinstance(review.get("reason"), str) and review["reason"].strip()
                and isinstance(refs, list) and refs
                and all(isinstance(ref, str) and ref in ids for ref in refs))


def _qualification(record: dict[str, Any]) -> tuple[bool, str]:
    if record.get("revoked") is True:
        return False, "revoked"
    source = str(record.get("source_kind", ""))
    if not source or source in _LEGACY_SOURCES or "legacy" in source or "directed" in source:
        return False, "legacy_or_unspecified_source"
    if not isinstance(record.get("protocol_id"), str) or not record["protocol_id"].strip():
        return False, "missing_protocol"
    qualification = record.get("qualification") or {}
    if qualification.get("status") != "verified":
        return False, "admission_not_verified"
    state, plan = record.get("state"), record.get("plan")
    if not isinstance(state, dict) or not isinstance(plan, dict):
        return False, "invalid_policy_example: state and plan must be objects"
    try:
        validate_plan(plan, state)
    except (ValueError, TypeError, KeyError) as exc:
        return False, "invalid_policy_example: " + str(exc)
    review = qualification.get("factual_review")
    if not _review_valid(review, state):
        return False, "independent_factual_review_required"
    proposal_result = qualification.get("proposal_task_result")
    if proposal_result is not None and not _valid_task(proposal_result):
        return False, "valid_proposal_execution_required"
    if plan["action"] == "keep" and not _valid_task(proposal_result or qualification.get("task_result")):
        return False, "reviewed_keep_requires_valid_proposal_execution"
    if plan["action"] == "edit":
        implementation_result = qualification.get("implementation_task_result") or qualification.get("task_result")
        if not _valid_task(implementation_result):
            return False, "edit_requires_valid_implementation_execution"
        behavior = qualification.get("behavior") or {}
        if (behavior.get("status") != "PASS" or behavior.get("covers_plan") is not True
                or behavior.get("executable") is not True):
            return False, "executable_behavior_and_preservation_required"
        if not plan.get("verification_kind"):
            return False, "edit_mechanism_not_verified"
        mechanism = record.get("mechanism_id")
        if mechanism and mechanism != plan["verification_kind"]:
            return False, "verification_mechanism_mismatch"
        spec = implementation_result.get("evaluation_spec")
        implementation = record.get("implementation") or {}
        expected_hash = (implementation.get("code_hash") or qualification.get("implementation_code_hash")
                         or behavior.get("child_code_hash"))
        if spec is not None and (not isinstance(spec, dict) or not expected_hash or spec.get("code_hash") != expected_hash):
            return False, "implementation_execution_binding_mismatch"
        if behavior.get("child_code_hash") and expected_hash != behavior["child_code_hash"]:
            return False, "behavior_implementation_binding_mismatch"
    from research1.data_pipeline.admission_artifacts import validate_positive_artifacts
    valid, reason = validate_positive_artifacts(record)
    if not valid:
        return False, reason
    return True, "verified_single_plan_initialization"


def _tasks(state: dict[str, Any]) -> set[str]:
    tasks: set[str] = set()
    for key in ("proposal_task_ids", "task_names", "source_task_ids"):
        values = state.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(key + " must be a list of task ids")
        tasks.update(values)
    if state.get("task_name"):
        if not isinstance(state["task_name"], str):
            raise ValueError("task_name must be a string")
        tasks.add(state["task_name"])
    for evidence in state.get("evidence", []):
        if evidence.get("task_name"):
            if not isinstance(evidence["task_name"], str):
                raise ValueError("Evidence task_name must be a string")
            tasks.add(evidence["task_name"])
    return tasks


def _current_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold admission history: explicit revisions win; otherwise append order."""
    current: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str) or not record["sample_id"].strip():
            raise ValueError("Each admission record needs sample_id")
        previous = current.get(record["sample_id"])
        revision = record.get("revision")
        if revision is not None and (isinstance(revision, bool) or not isinstance(revision, int) or revision < 0):
            raise ValueError("Admission revision must be a nonnegative integer")
        if previous is not None and previous.get("revision") is not None:
            if revision is None:
                raise ValueError("Cannot mix versioned and unversioned admissions for one sample")
            if revision < previous["revision"]:
                continue
            if revision == previous["revision"] and canonical(record) != canonical(previous):
                raise ValueError("Conflicting admission content for the same revision")
        elif previous is not None and revision is not None:
            raise ValueError("Cannot mix versioned and unversioned admissions for one sample")
        current[record["sample_id"]] = record
    return [current[key] for key in sorted(current)]


def _check_splits(records: list[dict[str, Any]]) -> None:
    roots, tasks = {}, {}
    for record in records:
        if record.get("revoked") or not isinstance(record.get("state"), dict):
            continue
        state = record["state"]
        split, root = state.get("split"), state.get("root_lineage_id")
        if split not in {"train", "validation"} or not isinstance(root, str) or not root:
            # Invalid examples are reported by qualification, not silently
            # projected onto an implicit training fold.
            continue
        if root in roots and roots[root] != split:
            raise ValueError("Evolution lineage leaked across train/validation: " + root)
        roots[root] = split
        for task in _tasks(state):
            if task in tasks and tasks[task] != split:
                raise ValueError("Proposal task leaked across train/validation: " + task)
            tasks[task] = split


def _load_validator(profile: dict[str, Any] | None):
    if profile is None:
        return None, "student_model_and_template_not_selected"
    if not isinstance(profile.get("model_id"), str) or not profile["model_id"].strip():
        return None, "student_model_not_identified"
    limit = profile.get("max_context_tokens")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return None, "student_context_limit_not_frozen"
    if callable(profile.get("validator")):
        return profile["validator"], None
    tokenizer_path = profile.get("tokenizer_path")
    if not isinstance(tokenizer_path, str) or not Path(tokenizer_path).is_dir():
        return None, "local_student_tokenizer_not_available"
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            return None, "student_chat_template_missing"
    except Exception as exc:
        return None, "student_tokenizer_load_failed: " + type(exc).__name__

    def validate(messages):
        all_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        prefix_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        if all_ids[:len(prefix_ids)] != prefix_ids or len(all_ids) <= len(prefix_ids):
            return {"valid": False, "reason": "chat_template_has_no_verified_assistant_boundary"}
        labels = [-100] * len(prefix_ids) + list(all_ids[len(prefix_ids):])
        return {"valid": True, "input_ids": list(all_ids), "labels": labels,
                "template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()}
    return validate, None


def _validate_tokens(messages: list[dict[str, str]], validator, profile: dict[str, Any]) -> dict[str, Any]:
    try:
        result = validator(messages)
    except Exception as exc:
        return {"valid": False, "reason": "student_template_validation_failed: " + type(exc).__name__}
    if not isinstance(result, dict) or result.get("valid") is not True:
        return {"valid": False, "reason": (result or {}).get("reason", "student_template_invalid") if isinstance(result, dict) else "student_template_invalid"}
    ids, labels = result.get("input_ids"), result.get("labels")
    if (not isinstance(ids, list) or not ids or not isinstance(labels, list) or len(labels) != len(ids)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in ids)
            or any(isinstance(v, bool) or not isinstance(v, int) for v in labels)):
        return {"valid": False, "reason": "student_token_ids_and_assistant_loss_mask_required"}
    supervised = [i for i, value in enumerate(labels) if value != -100]
    if (not supervised or supervised[0] == 0
            or supervised != list(range(supervised[0], len(ids)))
            or any(labels[i] != ids[i] for i in supervised)):
        return {"valid": False, "reason": "invalid_assistant_only_loss_mask"}
    if len(ids) > profile["max_context_tokens"]:
        return {"valid": False, "reason": "student_context_overflow", "token_count": len(ids)}
    if not isinstance(result.get("template_sha256"), str) or not result["template_sha256"].strip():
        return {"valid": False, "reason": "student_template_identity_required"}
    return {"valid": True, "token_count": len(ids), "input_ids": ids, "labels": labels,
            "template_sha256": result.get("template_sha256")}


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(canonical(row) + "\n" for row in rows)


def _publish(campaign: Path, files: dict[str, str], report: dict[str, Any]) -> dict[str, Any]:
    revision = digest({"files": files, "manifest": report})
    base = Path(campaign) / "exports" / "sft"
    directory = base / "generations" / revision
    manifest = dict(report, dataset_revision=revision,
                    files={name: str((directory / name).resolve()) for name in files})
    payloads = dict(files, **{"manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"})
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        for name, text in payloads.items():
            if not (directory / name).is_file() or (directory / name).read_text(encoding="utf-8") != text:
                raise ValueError("Immutable SFT generation content changed")
    else:
        staging = Path(tempfile.mkdtemp(prefix=".sft-", dir=directory.parent))
        try:
            for name, text in payloads.items():
                (staging / name).write_text(text, encoding="utf-8")
            os.replace(staging, directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    # Publish the complete immutable generation in a single atomic pointer swap.
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=base, encoding="utf-8", delete=False) as handle:
        temp_name = handle.name
        handle.write(json.dumps({"dataset_revision": revision, "manifest": str((directory / "manifest.json").resolve())}, indent=2) + "\n")
    os.replace(temp_name, base / "CURRENT.json")
    return manifest


def export_dataset(campaign: Path, records: list[dict[str, Any]], target_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Export current, qualified single-plan messages; never request a model.

    ``target_profile.validator(messages)`` is an optional injected tokenizer
    adapter returning actual ``input_ids`` and assistant-only ``labels``. Without
    a selected local tokenizer/template, generic messages remain exportable but
    the dataset is explicitly not training-ready.
    """
    source_validation = None
    factual_recheck_validation = None
    protocol_path = Path(campaign) / 'protocol.json'
    if protocol_path.is_file() and json.loads(protocol_path.read_text(encoding='utf-8')).get('source_parents'):
        from research1.data_pipeline.protocol import load_protocol
        from research1.data_pipeline.ledger import Ledger
        from research1.data_pipeline.source_parents import audit_sources_for_export
        ledger = Ledger(campaign)
        source_validation = audit_sources_for_export(campaign, load_protocol(campaign), ledger)
        # Re-read the current admission revisions after any propagated source
        # revocations; caller-supplied stale records may never republish them.
        records = ledger.records('samples')
    if any(record.get('factual_recheck') for record in records):
        from research1.data_pipeline.ledger import Ledger
        from research1.data_pipeline.admission_recheck import audit_recheck_exports
        ledger = Ledger(campaign)
        factual_recheck_validation = audit_recheck_exports(campaign, ledger)
        records = ledger.records('samples')
    current = _current_records(records)
    _check_splits(current)
    validator, tokenizer_problem = _load_validator(target_profile)
    profile = target_profile or {}
    examples, metadata, rejected, duplicates = {}, [], [], []
    seen = {}
    for record in current:
        eligible, reason = _qualification(record)
        if not eligible:
            rejected.append({"sample_id": record["sample_id"], "reason": reason})
            continue
        state, plan = record["state"], record["plan"]
        if not _tasks(state):
            rejected.append({"sample_id": record["sample_id"], "reason": "proposal_task_identity_required_for_split_audit"})
            continue
        messages = render_messages(state, plan)
        duplicate_key = (state.get("origin_id", state["state_id"]), digest(plan))
        if duplicate_key in seen:
            duplicates.append({"sample_id": record["sample_id"], "canonical_sample_id": seen[duplicate_key]})
            continue
        token_result = _validate_tokens(messages, validator, profile) if validator else None
        if token_result is not None and not token_result["valid"]:
            rejected.append({"sample_id": record["sample_id"], "reason": token_result["reason"]})
            continue
        seen[duplicate_key] = record["sample_id"]
        examples[record["sample_id"]] = {"messages": messages}
        metadata.append({"sample_id": record["sample_id"], "state_id": state["state_id"],
            "root_lineage_id": state["root_lineage_id"], "origin_id": state.get("origin_id"),
            "split": state["split"], "action": plan["action"], "target_module": plan["target_module"],
            "source_kind": record["source_kind"], "protocol_id": record["protocol_id"],
            "mechanism_id": record.get("mechanism_id"), "proposal_task_ids": sorted(_tasks(state)),
            "plan_sha256": digest(plan), "visible_messages_sha256": digest(messages[:-1]),
            "teacher_path": record.get("teacher_path"), "teacher_prompt_sha256": record.get("teacher_prompt_sha256"),
            "qualification": record["qualification"], "token_validation": token_result})
        if record.get('parent_execution_provenance'):
            # Keep cross-protocol execution identity and compatibility binding
            # in sidecars, never inside training messages.
            metadata[-1]['parent_execution_provenance'] = record['parent_execution_provenance']
        if record.get('factual_recheck'):
            metadata[-1]['factual_recheck'] = record['factual_recheck']
    template_ids = {m["token_validation"]["template_sha256"] for m in metadata if m.get("token_validation")}
    if len(template_ids) > 1:
        raise ValueError("Student tokenizer/template changed within one dataset generation")
    files: dict[str, str] = {}
    counts = {}
    selections = {}
    for split in ("train", "validation"):
        by_action = {action: [m["sample_id"] for m in metadata if m["split"] == split and m["action"] == action]
                     for action in ("edit", "keep")}
        for action, sample_ids in by_action.items():
            files[f"{action}.{split}.jsonl"] = _jsonl([examples[key] for key in sample_ids])
        # K <= E/4 gives K/(E+K) <= 20%, with no keep-only stand-in dataset.
        mixed = by_action["edit"] + by_action["keep"][:len(by_action["edit"]) // 4]
        selections[split] = dict(by_action, mixed=mixed)
        files[f"mixed.{split}.jsonl"] = _jsonl([examples[key] for key in mixed])
        counts[split] = {"edit": len(by_action["edit"]), "keep": len(by_action["keep"]),
                         "mixed": len(mixed), "mixed_keep": len(mixed) - len(by_action["edit"])}
        tokenized = []
        if validator:
            by_id = {m["sample_id"]: m for m in metadata}
            for key in mixed:
                validated = by_id[key]["token_validation"]
                tokenized.append({"input_ids": validated["input_ids"], "attention_mask": [1] * validated["token_count"],
                                  "labels": validated["labels"]})
            files[f"mixed.{split}.tokenized.jsonl"] = _jsonl(tokenized)
    by_id = {m["sample_id"]: m for m in metadata}
    # Metadata sidecars preserve ordering; never embed ids/outcomes in messages.
    for split in ("train", "validation"):
        for action in ("edit", "keep", "mixed"):
            members = [by_id[key] for key in selections[split][action]]
            files[f"{action}.{split}.metadata.jsonl"] = _jsonl(members)
    files["metadata.jsonl"] = _jsonl(metadata)
    files["excluded.jsonl"] = _jsonl(rejected)
    files["duplicates.jsonl"] = _jsonl(duplicates)
    train_ready = bool(validator and counts["train"]["mixed"])
    readiness = []
    if tokenizer_problem:
        readiness.append(tokenizer_problem)
    if not counts["train"]["edit"]:
        readiness.append("no_verified_edit_training_examples_keep_cannot_fill_quota")
    if not counts["validation"]["mixed"]:
        readiness.append("no_heldout_validation_examples")
        train_ready = False
    report = {"schema": SCHEMA, "training_ready": train_ready,
        "readiness_reasons": readiness, "loss_scope": "assistant_plan_only", "keep_fraction_cap": 0.2,
        "counts": counts, "qualified_records": len(metadata), "excluded_records": len(rejected),
        "duplicate_records": len(duplicates), "independent_lineages": len({m["root_lineage_id"] for m in metadata}),
        "mechanisms": sorted({m["mechanism_id"] for m in metadata if m.get("mechanism_id")}),
        "module_counts": {module: sum(m["target_module"] == module for m in metadata) for module in sorted({m["target_module"] for m in metadata if m["target_module"]})},
        "student_profile": {key: profile.get(key) for key in ("model_id", "tokenizer_path", "max_context_tokens")},
        "contains_long_horizon_targets": False,
        "combined_selection": "all_qualified_edits_plus_at_most_floor(edit_count/4)_reviewed_keeps_per_split"}
    if source_validation is not None:
        report['source_parent_validation'] = source_validation
    if factual_recheck_validation is not None:
        report['factual_recheck_validation'] = factual_recheck_validation
    return _publish(Path(campaign), files, report)
