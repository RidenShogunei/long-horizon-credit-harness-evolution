"""Preregistered auxiliary prompt/context comparison, with at-most-once calls.

This runner cannot execute tasks, consume an evaluation authorization, or export
SFT.  Candidate actions are descriptive outputs, never qualification decisions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from research1.data_pipeline.policy import (
    canonical, collect_candidates, digest, parse_candidates, public_view, render_messages,
)
from research1.data_pipeline.util import atomic_json


SCHEMA = "research1.auxiliary.prompt-context-comparison.v1"
_VIEW_ENRICHMENTS = {"framework_context", "verification_catalog", "verification_interfaces"}


class TeacherRecordError(ValueError):
    pass


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collection_prompt(state: dict) -> str:
    """Exact collection envelope; verify against the saved raw record too."""
    messages = render_messages(state)
    count = state.get("max_candidates", 2)
    if count not in {1, 2}:
        raise ValueError("Candidate count must be one or two")
    return (messages[0]["content"] + "\nCollection request: generate at most " + str(count)
            + ' distinct plans before any candidate outcome is known. Return JSON {"candidates":[Plan,...]} instead of the deployment single-plan envelope. '
            + "Do not force a second edit or fill a candidate quota.\nFrozen proposal-only view:\n"
            + messages[1]["content"])


@contextmanager
def _controller(directory: Path):
    """A kernel lock is released by crashes, unlike a pid-file lock."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "comparison.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Comparison controller is already running") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Comparison controller is already running") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _same_parent(source: dict, state: dict, *, input_variant: str) -> None:
    for field in ("root_lineage_id", "split", "code_hash", "proposal_task_ids"):
        if source.get(field) != state.get(field):
            raise ValueError("Comparison changed parent identity: " + field)
    before, after = public_view(source), public_view(state)
    if input_variant == "original":
        if before != after:
            raise ValueError("Original-input condition differs from source public view")
    elif input_variant == "enriched":
        if ({k: v for k, v in before.items() if k not in _VIEW_ENRICHMENTS}
                != {k: v for k, v in after.items() if k not in _VIEW_ENRICHMENTS}):
            raise ValueError("Enrichment changed evidence, source code, or branch history")
    else:
        raise ValueError("Input variant must be original or enriched")


def prepare_comparison(directory: Path, cases: list[dict], *, metadata: dict | None = None) -> dict:
    """Freeze every case and exact prompt before the first model call.

    Cases require case_id, parent_id, condition, source_state_path, state_path,
    and input_variant (original|enriched). source_teacher_path is optional but
    is byte-bound when provided. The list order is the fixed execution order.
    """
    directory = Path(directory).resolve()
    with _controller(directory):
        if (directory / "manifest.json").exists():
            raise ValueError("Comparison is already prepared; run or inspect it")
        if not isinstance(cases, list) or not cases:
            raise ValueError("Comparison requires an explicit nonempty fixed case list")
        rows, ids, signatures = [], set(), set()
        frozen = []
        for index, case in enumerate(cases):
            case_id = case["case_id"]
            if (not isinstance(case_id, str) or not case_id or case_id in ids
                    or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in case_id)):
                raise ValueError("Case IDs must be unique safe names")
            ids.add(case_id)
            source_path = Path(case["source_state_path"]).resolve()
            state_path = Path(case["state_path"]).resolve()
            source, state = _read(source_path), _read(state_path)
            _same_parent(source, state, input_variant=case["input_variant"])
            if state.get("allow_teacher_http_fallback") is not False:
                raise ValueError("Comparison must explicitly disable teacher HTTP fallback")
            if case.get("expected_source_view_sha256") not in {None, digest(public_view(source))}:
                raise ValueError("Source public-view hash differs from preregistration request")
            signature = (state.get("teacher_provider"), state.get("teacher_model"), 0)
            if not all(signature[:2]):
                raise ValueError("Every case must freeze a teacher provider and model")
            signatures.add(signature)
            prompt = _collection_prompt(state)
            row = {"case_id": case_id, "order": index, "parent_id": case["parent_id"],
                   "condition": case["condition"], "input_variant": case["input_variant"],
                   "source_state_path": str(source_path), "source_state_sha256": _sha(source_path),
                   "state_path": str(state_path), "state_sha256": _sha(state_path),
                   "source_public_view_sha256": digest(public_view(source)),
                   "public_view_sha256": digest(public_view(state)),
                   "root_lineage_id": state["root_lineage_id"], "split": state["split"],
                   "plan_policy_version": state.get("plan_policy_version"),
                   "teacher_signature": {"provider": signature[0], "model": signature[1], "temperature": 0},
                   "allow_teacher_http_fallback": False,
                   "messages": render_messages(state), "prompt": prompt,
                   "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                   "frozen_state_path": "cases/" + case_id + "/state.json"}
            if case.get("source_teacher_path"):
                teacher_path = Path(case["source_teacher_path"]).resolve()
                row.update(source_teacher_path=str(teacher_path), source_teacher_sha256=_sha(teacher_path))
            frozen.append((row, state)); rows.append(row)
        if len(signatures) != 1:
            raise ValueError("Teacher provider/model/temperature must be identical in all conditions")
        manifest = {"schema": SCHEMA, "cases": rows, "metadata": metadata or {},
                    "fixed_order": [r["case_id"] for r in rows],
                    "evaluation_jobs": 0, "automatic_sft_admission": False,
                    "automatic_retry": False,
                    "stop_rule": "One pass over the fixed cases; preserve every result, including keep, parse failures and ambiguous calls."}
        manifest["manifest_sha256"] = digest(manifest)
        for row, state in frozen:
            atomic_json(directory / row["frozen_state_path"], state)
            row["frozen_state_sha256"] = _sha(directory / row["frozen_state_path"])
        # Include hashes of the serialized snapshots in the signed manifest.
        manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        atomic_json(directory / "manifest.json", manifest)
        return manifest


def _validated_manifest(directory: Path) -> dict:
    manifest = _read(directory / "manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("manifest_sha256")
            != digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})):
        raise ValueError("Comparison manifest changed")
    for row in manifest["cases"]:
        for path_key, hash_key in (("source_state_path", "source_state_sha256"),
                                   ("state_path", "state_sha256"),
                                   ("source_teacher_path", "source_teacher_sha256")):
            if path_key in row and _sha(Path(row[path_key])) != row[hash_key]:
                raise ValueError("Preregistered source bytes changed: " + row["case_id"])
        frozen = directory / row["frozen_state_path"]
        if _sha(frozen) != row["frozen_state_sha256"]:
            raise ValueError("Frozen condition state changed")
        state = _read(frozen)
        if state.get("allow_teacher_http_fallback") is not False or row.get("allow_teacher_http_fallback") is not False:
            raise ValueError("Comparison HTTP fallback prohibition changed")
        if (render_messages(state) != row["messages"] or _collection_prompt(state) != row["prompt"]
                or digest(public_view(state)) != row["public_view_sha256"]):
            raise ValueError("Renderer or model-facing input changed since preregistration")
    return manifest


def comparison_status(directory: Path) -> dict:
    directory = Path(directory).resolve()
    manifest = _validated_manifest(directory)
    rows, counts = [], {}
    for case in manifest["cases"]:
        outcome_path = directory / "cases" / case["case_id"] / "outcome.json"
        outcome = _read(outcome_path) if outcome_path.exists() else {"status": "PENDING", "plans": []}
        if outcome_path.exists():
            if outcome.get("case_id") != case["case_id"] or outcome.get("prompt_sha256") != case["prompt_sha256"]:
                raise ValueError("Outcome belongs to another preregistered case")
            raw_path = outcome_path.parent / "teacher.json"
            if outcome.get("raw_record_sha256") and _sha(raw_path) != outcome["raw_record_sha256"]:
                raise ValueError("Raw teacher record changed after outcome persistence")
            if outcome["status"] == "COMPLETE":
                _raw_signature(raw_path, case)
                state = _read(directory / case["frozen_state_path"])
                if parse_candidates(_read(raw_path)["raw"], state, state.get("max_candidates", 2)) != outcome["plans"]:
                    raise ValueError("Saved actions differ from unchanged raw teacher reply")
        counts.setdefault(case["condition"], {"cases": 0, "edit": 0, "keep": 0, "parse_error": 0,
                                              "uncertain": 0, "pending": 0, "other_error": 0})
        count = counts[case["condition"]]; count["cases"] += 1
        for plan in outcome.get("plans", []):
            count[plan["action"]] += 1
        status = outcome["status"]
        if status == "PARSE_ERROR": count["parse_error"] += 1
        elif status == "AMBIGUOUS_CALL_NOT_RETRIED": count["uncertain"] += 1
        elif status == "PENDING": count["pending"] += 1
        elif status != "COMPLETE": count["other_error"] += 1
        rows.append({"case_id": case["case_id"], "parent_id": case["parent_id"],
                     "condition": case["condition"], "status": status,
                     "actions": [p["action"] for p in outcome.get("plans", [])],
                     "outcome_path": str(outcome_path) if outcome_path.exists() else None})
    return {"schema": SCHEMA, "manifest_sha256": manifest["manifest_sha256"],
            "cases": rows, "by_condition": counts,
            "complete": all(r["status"] != "PENDING" for r in rows),
            "evaluation_jobs": 0, "qualified_edit": None, "qualified_keep": None,
            "sft_exported": False, "mc_labels": 0,
            "interpretation": "Descriptive auxiliary comparison; actions alone do not establish factual support, implementability, utility, or SFT qualification."}


def _raw_signature(path: Path, row: dict) -> None:
    saved = _read(path)
    if saved.get("prompt") != row["prompt"] or saved.get("prompt_sha256") != row["prompt_sha256"]:
        raise TeacherRecordError("Saved teacher input differs from preregistration")
    metadata = saved.get("metadata") or {}
    expected = row["teacher_signature"]
    if ({key: metadata.get(key) for key in expected} != expected
            or metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop"):
        raise TeacherRecordError("Teacher signature or completion status differs from preregistration")


def _preteacher_checks(directory: Path, manifest: dict) -> None:
    """Verify and once-bind required preteacher artifacts before any case call.

    Diagnostics remain outside the model-facing state. This guard checks their
    provenance and timing, without inventing or changing mechanism attribution.
    """
    required = manifest.get("metadata", {}).get("required_preteacher_checks")
    if required is None:
        return
    if not isinstance(required, str) or not required:
        raise ValueError("Required preteacher checks path is invalid")
    checks_path = (directory / required).resolve()
    if not checks_path.is_relative_to(directory):
        raise ValueError("Preteacher checks must stay inside the comparison directory")
    checks = _read(checks_path)
    if (checks.get("comparison_manifest_sha256") != manifest["manifest_sha256"]
            or checks.get("frozen_before_teacher_calls") is not True):
        raise ValueError("Preteacher checks do not bind this frozen comparison")
    parents = {row["parent_id"] for row in manifest["cases"]}
    if not isinstance(checks.get("parents"), dict) or set(checks["parents"]) != parents:
        raise ValueError("Preteacher checks do not cover the exact preregistered parents")
    signature = {"checks_path": str(checks_path), "checks_sha256": _sha(checks_path),
                 "comparison_manifest_sha256": manifest["manifest_sha256"], "contracts": {}}
    for parent in sorted(parents):
        rows = [row for row in manifest["cases"] if row["parent_id"] == parent and row["condition"] == "D"]
        if len(rows) != 1:
            raise ValueError("Required checks need exactly one condition D per parent")
        reference = _read(directory / rows[0]["frozen_state_path"])
        record = checks["parents"][parent]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Preteacher contract record is invalid")
        contract_path = (directory / record["path"]).resolve()
        if not contract_path.is_relative_to(directory):
            raise ValueError("Preteacher contract must stay inside the comparison directory")
        if _sha(contract_path) != record.get("sha256"):
            raise ValueError("Preteacher contract bytes changed")
        artifact = _read(contract_path)
        state, contract = artifact["state"], artifact["contract"]
        for field in ("state_id", "root_lineage_id", "split", "code_hash", "modules_root", "proposal_task_ids", "plan_policy_version"):
            if state.get(field) != reference.get(field):
                raise ValueError("Preteacher contract changed parent identity or code")
        if public_view(state) != public_view(reference):
            raise ValueError("Preteacher contract changed frozen condition D inputs")
        expected_id = "checks_" + digest({k: v for k, v in contract.items() if k != "checks_id"})[:24]
        if contract.get("checks_id") != expected_id or record.get("checks_id") != expected_id:
            raise ValueError("Preteacher checks ID does not bind contract content")
        expected_evidence = {row["id"]: digest(row) for row in reference["evidence"]}
        if contract.get("evidence_hashes") != expected_evidence:
            raise ValueError("Preteacher contract does not bind frozen proposal evidence")
        selected = [row["mechanism_id"] for row in contract.get("selected", [])]
        if record.get("selected_mechanisms") != selected:
            raise ValueError("Preteacher check summary differs from the frozen contract")
        signature["contracts"][parent] = {"path": str(contract_path), "sha256": record["sha256"], "checks_id": expected_id}
    binding_path = directory / "preteacher_binding.json"
    if binding_path.exists():
        if _read(binding_path) != signature:
            raise ValueError("Preteacher checks changed after their first execution binding")
    else:
        if any((directory / "cases" / row["case_id"] / "started.json").exists() for row in manifest["cases"]):
            raise ValueError("A teacher call started before preteacher checks were bound")
        atomic_json(binding_path, signature)


def run_comparison(directory: Path, *, proposer=None) -> dict:
    """Execute each case at most once; a started call without raw is not retried."""
    directory = Path(directory).resolve()
    proposer = proposer or collect_candidates
    with _controller(directory):
        manifest = _validated_manifest(directory)
        _preteacher_checks(directory, manifest)
        for row in manifest["cases"]:
            _preteacher_checks(directory, manifest)
            case_dir = directory / "cases" / row["case_id"]
            output = case_dir / "outcome.json"
            if output.exists():
                continue
            started, raw_path = case_dir / "started.json", case_dir / "teacher.json"
            if started.exists() and not raw_path.exists():
                atomic_json(output, {"case_id": row["case_id"], "prompt_sha256": row["prompt_sha256"],
                                     "status": "AMBIGUOUS_CALL_NOT_RETRIED", "plans": [],
                                     "reason": "Call started but no raw record survived; do not automatically repeat."})
                continue
            if started.exists() and _read(started).get("prompt_sha256") != row["prompt_sha256"]:
                raise ValueError("Call marker belongs to another prompt")
            if not started.exists():
                atomic_json(started, {"case_id": row["case_id"], "prompt_sha256": row["prompt_sha256"],
                                      "started_unix": time.time(), "automatic_retry": False})
            try:
                # collect_candidates reads a matching teacher.json without a
                # second network call when resuming after raw persistence.
                state = _read(directory / row["frozen_state_path"])
                plans = proposer(state, case_dir)
                if not raw_path.exists():
                    atomic_json(output, {"case_id": row["case_id"], "prompt_sha256": row["prompt_sha256"],
                                         "status": "AMBIGUOUS_CALL_NOT_RETRIED", "plans": [],
                                         "reason": "Proposer returned without preserving its raw response."})
                    continue
                _raw_signature(raw_path, row)
                if plans != parse_candidates(_read(raw_path)["raw"], state, state.get("max_candidates", 2)):
                    raise ValueError("Proposer actions differ from its preserved raw reply")
                outcome = {"status": "COMPLETE", "plans": plans,
                           "raw_record_sha256": _sha(raw_path), "qualification": "not_assessed"}
            except Exception as exc:
                # Exception strings can contain provider payloads or headers.
                # Preserve only the exception class; raw replies stay on disk.
                if raw_path.exists():
                    status = "PARSE_ERROR" if isinstance(exc, (ValueError, TypeError, KeyError)) else "CALL_ERROR"
                    try:
                        _raw_signature(raw_path, row)
                    except (TeacherRecordError, ValueError, TypeError, KeyError):
                        status = "SIGNATURE_ERROR"
                    outcome = {"status": status,
                               "plans": [], "error_type": type(exc).__name__,
                               "raw_record_sha256": _sha(raw_path)}
                else:
                    outcome = {"status": "AMBIGUOUS_CALL_NOT_RETRIED", "plans": [],
                               "error_type": type(exc).__name__,
                               "reason": "No persisted raw record; automatic retry forbidden."}
            outcome.update(case_id=row["case_id"], prompt_sha256=row["prompt_sha256"])
            atomic_json(output, outcome)
            atomic_json(directory / "summary.json", comparison_status(directory))
        _preteacher_checks(directory, manifest)
        summary = comparison_status(directory)
        atomic_json(directory / "summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auxiliary fixed prompt/context comparison")
    parser.add_argument("action", choices=("prepare", "run", "status"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        if not args.spec:
            parser.error("prepare requires --spec")
        spec = _read(args.spec)
        doc = prepare_comparison(args.directory, spec["cases"], metadata=spec.get("metadata"))
        print(canonical({"manifest_sha256": doc["manifest_sha256"], "cases": len(doc["cases"])}))
    elif args.action == "run":
        print(canonical(run_comparison(args.directory)))
    else:
        print(canonical(comparison_status(args.directory)))


if __name__ == "__main__":
    main()
