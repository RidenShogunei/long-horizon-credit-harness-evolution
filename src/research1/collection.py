"""Evidence-led plan initialization with immutable inputs and explicit admission.

This collector does not manufacture verification for a new mechanism. A plan
without a pre-frozen, applicable executable contract remains pending. Real task
execution is opt-in; no long-horizon labels are created here.
"""
from __future__ import annotations

import hashlib
import ast
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable

SCHEMA = "research1.collection.v2"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _id(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(_json(value).encode()).hexdigest()[:24]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def append_unique(path: Path, row: dict[str, Any], key: str) -> bool:
    """Idempotent export for a single collector process; not a cross-host lock."""
    if any(old.get(key) == row.get(key) for old in _read_rows(path)):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def sample_id_for_row(row: dict[str, Any]) -> str:
    if row.get("sample_id"):
        return str(row["sample_id"])
    return _id("legacy_", {k: row.get(k) for k in ("run_id", "prompt", "completion")})


def set_admission(
    root: Path, sample_id: str, status: str, reason: str, *,
    sft_row: dict[str, Any] | None = None, evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append an auditable admission/revocation event, then rebuild the SFT view.

    Existing unregistered rows remain visible until explicitly reviewed. A
    revocation retains the row in the registry and removes it from the SFT view.
    Raw attempts and immutable run artifacts are never removed.
    """
    if status not in {"verified", "revoked", "rejected", "pending", "pending_review"}:
        raise ValueError("unsupported admission status")
    registry = root / "datasets/research1/admission_registry.jsonl"
    rows = _read_rows(registry)
    previous = next((r for r in reversed(rows) if r.get("sample_id") == sample_id), None)
    if sft_row is None:
        sft_row = next((row for row in _read_rows(root / "datasets/research1/sft/verified_plans.jsonl")
                        if sample_id_for_row(row) == sample_id), None)
        if sft_row is None and previous:
            sft_row = previous.get("sft_row")
    event = {"schema": SCHEMA, "sample_id": sample_id, "status": status,
             "reason": reason, "evidence": evidence or {}, "sft_row": sft_row}
    event["event_id"] = _id("admission_", event)
    # Repeated calls do not append duplicate *current* events. A later change
    # back to an earlier status still needs its own ordering in the ledger.
    if previous != event:
        registry.parent.mkdir(parents=True, exist_ok=True)
        with registry.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    rebuild_verified_plans(root)
    return event


def rebuild_verified_plans(root: Path) -> dict[str, Any]:
    path = root / "datasets/research1/sft/verified_plans.jsonl"
    events = _read_rows(root / "datasets/research1/admission_registry.jsonl")
    latest = {row["sample_id"]: row for row in events}
    rows = {sample_id_for_row(row): row for row in _read_rows(path)}
    for key, event in latest.items():
        if event["status"] == "verified" and event.get("sft_row"):
            rows[key] = event["sft_row"]
        elif event["status"] != "verified":
            rows.pop(key, None)
    # Keep all candidates in the latest accepted view of an origin, not every
    # re-excerpt of the same trajectory. Raw plans and admission history persist.
    current_view = {}
    for event in events:
        row = event.get("sft_row") or {}
        if event["status"] == "verified" and latest[event["sample_id"]]["status"] == "verified" and row.get("origin_id"):
            current_view[row["origin_id"]] = row.get("state_id")
    rows = {key: row for key, row in rows.items()
            if not row.get("origin_id") or row.get("state_id") == current_view.get(row["origin_id"], row.get("state_id"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in sorted(rows)), encoding="utf-8")
    temp.replace(path)
    return {"verified_count": len(rows), "registry_count": len(latest), "path": str(path)}


def _public(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if any(word in lowered for word in ("api_key", "password", "secret", "token")):
                raise ValueError(f"private field is not permitted in the policy view: {key}")
            if lowered in {"child_reward", "observed_delta", "long_horizon_label", "future_results", "child_diff"}:
                raise ValueError(f"future information is not permitted in the policy view: {key}")
            _public(item)
    elif isinstance(value, list):
        for item in value:
            _public(item)


def freeze_parent(
    *, root: Path, modules_root: Path, task_name: str, evidence: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None, public_config: dict[str, Any] | None = None,
    checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze all module files plus the exact bounded teacher/student view."""
    modules_root = modules_root.resolve()
    if not modules_root.is_dir():
        raise ValueError("parent modules directory does not exist")
    if not evidence or any(not e.get("id") or not str(e.get("text", "")).strip() for e in evidence):
        raise ValueError("evidence requires stable ids and nonempty text")
    if len({e["id"] for e in evidence}) != len(evidence):
        raise ValueError("duplicate evidence ids")
    evidence = json.loads(_json(evidence))
    source_files: dict[str, Path] = {}
    for entry in evidence:
        if entry.get("path"):
            path = Path(entry["path"]).resolve()
            if not path.is_file():
                raise ValueError("evidence source path is not a readable file")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if entry.get("source_sha256") not in {None, digest}:
                raise ValueError("evidence source hash changed before capture")
            entry["source_sha256"] = digest
            source_files[digest] = path
    view = json.loads(_json({"task_name": task_name, "evidence": evidence, "history": history or [],
                            "config": public_config or {}}))
    _public(view)
    execution_config = None
    config_path = (public_config or {}).get("config_path")
    if config_path:
        from research1.config import load_pilot, resolve_path, train_task_names
        execution_config = load_pilot(Path(config_path))
        if task_name not in train_task_names(resolve_path(execution_config, "dataset_tb")):
            raise ValueError("formal collection task is absent from the manifest train split")
        expected_hash = (public_config or {}).get("config_hash")
        if expected_hash and execution_config["_config_hash"] != expected_hash:
            raise ValueError("configuration changed before parent capture")
        _public(execution_config)
        view["config"] = dict(view["config"], config_hash=execution_config["_config_hash"])
    hashes: dict[str, str] = {}
    source: dict[str, str] = {}
    for path in sorted(modules_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("parent snapshots cannot contain symlinks")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            rel = path.relative_to(modules_root).as_posix()
            hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix == ".py":
                source[rel] = path.read_text(encoding="utf-8")
    if not hashes:
        raise ValueError("empty parent snapshot")
    # Full code is preserved for recovery. The bounded view is deterministic;
    # selection/size limits must be chosen before the teacher is called.
    selected = (public_config or {}).get("visible_files", ["tools/baseline.py"])
    if not isinstance(selected, list) or not selected or any(p not in source for p in selected):
        raise ValueError("visible_files must name existing Python files")
    limit = int((public_config or {}).get("max_code_chars", 24000))
    if not 1000 <= limit <= 100000:
        raise ValueError("max_code_chars must be between 1000 and 100000")
    view["module_code"] = {p: source[p][:limit] for p in sorted(selected)}
    view["module_hashes"] = hashes
    # An alternate excerpt/check suite is a view revision of the same source,
    # not independent experience. File paths and selector settings are excluded.
    origin_id = _id("origin_", {
        "task_name": task_name, "module_hashes": hashes,
        "trajectory_hashes": sorted({e.get("source_sha256") or _id("text_", e["text"]) for e in evidence}),
        "execution_config_hash": (execution_config or {}).get("_config_hash"),
    })
    state_id = _id("state_", {"schema": SCHEMA, "view": view, "module_hashes": hashes,
                            "checks": checks})
    dest = root / "datasets/research1/states" / state_id
    doc = {"schema": SCHEMA, "state_id": state_id, "origin_id": origin_id,
           "view_revision": (public_config or {}).get("evidence_selector_id", "manual-view-v1"), "view": view,
           "module_hashes": hashes, "checks": checks, "checks_id": _id("checks_", checks),
           "execution_config": execution_config,
           "modules_root": str((dest / "modules").resolve())}
    if dest.exists():
        old = json.loads((dest / "state.json").read_text(encoding="utf-8"))
        # A verification suite is part of the experiment conditions; changing
        # it cannot silently overwrite a previously frozen state contract.
        if old != doc:
            raise ValueError("frozen state/check contract already exists with different content")
        for rel, digest in hashes.items():
            if hashlib.sha256((dest / "modules" / rel).read_bytes()).hexdigest() != digest:
                raise ValueError("frozen parent snapshot hash mismatch")
        return old
    dest.mkdir(parents=True)
    for rel in hashes:
        target = dest / "modules" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(modules_root / rel, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != hashes[rel]:
            raise ValueError("parent changed while snapshotting; retry from a stable version")
    for digest, path in source_files.items():
        target = dest / "evidence" / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("evidence changed while snapshotting")
    _write(dest / "state.json", doc)
    _write(dest / "visible_view.json", view)
    return doc


def prepare_parent(**kwargs: Any) -> dict[str, Any]:
    """Preview entry point: freeze inputs/contracts without model or task calls."""
    return freeze_parent(**kwargs)


def teacher_prompt(state: dict[str, Any], max_candidates: int) -> str:
    return (
        "You propose evidence-grounded Harness evolution plans. Diagnose the current "
        "state yourself. Do not optimize warning wording or invent a defect. "
        "Return JSON only: {\"candidates\":[...]} with at most " + str(max_candidates) +
        " distinct candidates, all generated before any candidate outcome is known. "
        "Each candidate must contain action(edit|keep), target_module, locations(list), "
        "evidence_refs(list of supplied evidence ids), problem_hypothesis, intended_change, "
        "preserve_behaviors(list), verification_plan(list), verification_kind(string|null). "
        "Use keep with a concrete reason when the evidence does not justify an edit. "
        "For edit, target_module is the module name (for example tools), and locations are existing visible code paths. "
        "Do not output a patch, reference solution, score prediction, or future result. "
        "A verification suite being available does not mean the evidence warrants its fix.\n"
        "Available frozen verification suite: " + str((state.get("checks") or {}).get("mechanism_id")) +
        "\nExact bounded policy view (also the student input):\n" + _json(state["view"])
    )


def parse_candidates(raw: str | dict[str, Any], state: dict[str, Any], max_candidates: int) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        value = json.loads(cleaned)
    else:
        value = raw
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= max_candidates:
        raise ValueError("teacher must return one or two honest candidates")
    evidence_ids = {row["id"] for row in state["view"]["evidence"]}
    seen: set[str] = set()
    result = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("action") not in {"edit", "keep"}:
            raise ValueError("invalid plan action")
        for key in ("problem_hypothesis", "intended_change"):
            if not isinstance(candidate.get(key), str) or not candidate[key].strip():
                raise ValueError(f"missing plan {key}")
        for key in ("evidence_refs", "locations", "preserve_behaviors", "verification_plan"):
            if not isinstance(candidate.get(key), list):
                raise ValueError(f"plan {key} must be a list")
        if not candidate["evidence_refs"] or not set(candidate["evidence_refs"]).issubset(evidence_ids):
            raise ValueError("plan references unknown or missing evidence")
        if candidate["action"] == "edit":
            if not candidate["locations"] or any(path not in state["view"]["module_code"] for path in candidate["locations"]):
                raise ValueError("edit locations must be in the frozen visible code")
            if not candidate.get("preserve_behaviors") or not candidate.get("verification_plan"):
                raise ValueError("edit needs preservation and verification requirements")
        signature = _json({k: candidate.get(k) for k in ("action", "target_module", "locations", "problem_hypothesis", "intended_change")})
        if signature in seen:
            continue
        seen.add(signature)
        result.append(candidate)
    return result


def valid_task_result(result: dict[str, Any]) -> bool:
    value = result.get("reward")
    return bool(result.get("status") in {"PASS", "FAIL"}
                and result.get("failure_layer") not in {"infrastructure", "container", "scorer"}
                and isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))


def _default_teacher(prompt: str) -> tuple[str, dict[str, Any]]:
    from research1.distill import _tongji_chat
    return _tongji_chat(prompt, max_tokens=8000, role="collection_teacher",
                       json_output=True, reasoning_effort="none")


def _default_editor(plan: dict[str, Any], state: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    from research1.fixed_editor import compact_editor
    return compact_editor(plan, state, attempt_dir)


def _default_verifier(plan: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    from research1.plan_init import judge_behavior, probe_checks
    bundle = state.get("checks") or {}
    if (bundle.get("kind") != "parser_contract" or not bundle.get("checks")
            or plan.get("verification_kind") != bundle.get("mechanism_id")
            or any(path not in bundle.get("allowed_targets", []) for path in plan["locations"])):
        return {"status": "pending", "reason": "no applicable pre-frozen executable verification contract"}
    scope = check_edit_scope(plan, state, attempt)
    if scope["status"] != "PASS":
        return scope
    before = probe_checks(Path(state["modules_root"]), bundle["checks"])
    after = probe_checks(Path(attempt["modules_root"]), bundle["checks"])
    result = judge_behavior(parent_warnings=before, child_warnings=after,
                            checks=bundle["checks"], edit_applied=True)
    return {"status": "PASS" if result["sft_eligible"] else "FAIL", "covers_plan": True,
            "checks_id": state["checks_id"], "behavior": result}


def _symbols(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["function:" + node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            other = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result["function:" + node.name + "." + item.name] = ast.dump(item, include_attributes=False)
                else:
                    other.append(ast.dump(item, include_attributes=False))
            result["global:" + node.name] = _json([ast.dump(b, include_attributes=False) for b in node.bases] + other)
        elif isinstance(node, ast.Import):
            for name in node.names:
                result["import:" + name.name] = ast.dump(name, include_attributes=False)
        elif isinstance(node, ast.ImportFrom):
            result["import:" + str(node.module)] = ast.dump(node, include_attributes=False)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            key = ",".join(ast.unparse(t) for t in targets)
            result["global:" + key] = ast.dump(node, include_attributes=False)
        else:
            # Imports/assignments/functions are the only authorized editable
            # top-level forms. Other executable statements need their own suite.
            result["other:" + str(len(result))] = ast.dump(node, include_attributes=False)
    return result


def check_edit_scope(plan: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    bundle = state.get("checks") or {}
    if not bundle.get("coverage_statement") or not bundle.get("allowed_changed_functions"):
        return {"status": "pending", "reason": "verification suite lacks frozen coverage and function scope"}
    allowed = {"function:" + name for name in bundle["allowed_changed_functions"]}
    allowed |= {"global:" + name for name in bundle.get("allowed_changed_globals", [])}
    allowed |= {"import:" + name for name in bundle.get("allowed_imports", [])}
    parent_root, child_root = Path(state["modules_root"]), Path(attempt["modules_root"])
    changed = []
    try:
        actual_files = {p.relative_to(child_root).as_posix() for p in child_root.rglob("*")
                        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        if actual_files != set(state["module_hashes"]):
            return {"status": "FAIL", "reason": "child added or removed files outside the frozen contract"}
        for rel, digest in state["module_hashes"].items():
            before_path, after_path = parent_root / rel, child_root / rel
            if hashlib.sha256(before_path.read_bytes()).hexdigest() != digest:
                return {"status": "FAIL", "reason": "immutable parent snapshot was modified"}
            if hashlib.sha256(after_path.read_bytes()).hexdigest() == digest:
                continue
            if rel not in plan["locations"]:
                return {"status": "FAIL", "reason": "edit changed a file outside plan locations"}
            before, after = _symbols(before_path.read_text(encoding="utf-8")), _symbols(after_path.read_text(encoding="utf-8"))
            changed.extend(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    except (OSError, SyntaxError) as exc:
        return {"status": "FAIL", "reason": f"scope check failed: {exc}"}
    if not changed or any(key not in allowed for key in changed):
        return {"status": "FAIL", "reason": "executable diff exceeds the frozen function/global/import scope", "changed_symbols": changed}
    return {"status": "PASS", "changed_symbols": changed}


def _default_evaluator(plan: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any], attempt_dir: Path) -> dict[str, Any]:
    from research1.plan_init import _run_child_task
    cfg = state.get("execution_config")
    if not cfg:
        raise ValueError("real execution requires a frozen config_path/config_hash snapshot")
    if cfg.get("_config_hash") != state["view"]["config"].get("config_hash"):
        raise ValueError("frozen execution configuration hash does not match the policy view")
    return _run_child_task(cfg=cfg, run_dir=attempt_dir, modules_root=Path(attempt["modules_root"]),
                           task_name=state["view"]["task_name"], run_id=attempt_dir.name,
                           version_id=str(attempt.get("child_version_id") or attempt_dir.name))


def collect_parent(
    *, root: Path, modules_root: Path, task_name: str, evidence: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None, public_config: dict[str, Any] | None = None,
    checks: dict[str, Any] | None = None, max_candidates: int = 2, run_task: bool = False,
    teacher: Callable | None = None, editor: Callable | None = None,
    verifier: Callable | None = None, evaluator: Callable | None = None,
    keep_reviewer: Callable | None = None,
) -> dict[str, Any]:
    """Generate all plans first, then implement and verify independent candidates.

    Callbacks: teacher(prompt)->(raw,metadata) or raw; editor(plan,state,dir)->dict;
    verifier(plan,state,attempt)->dict with PASS/covers_plan/checks_id;
    evaluator(plan,state,attempt,dir)->real TaskRun dict. Injected verifiers must
    use the frozen checks_id. Missing checks never produce a verified edit.
    keep_reviewer(plan,state)->{status:PASS,reason,evidence_refs} is an optional
    trusted, independent review. A scored parent alone never verifies a keep.
    Review metadata is stored separately and never added to either policy view.
    """
    if max_candidates not in {1, 2}:
        raise ValueError("max_candidates must be 1 or 2")
    root = Path(root)
    state = freeze_parent(root=root, modules_root=modules_root, task_name=task_name,
                          evidence=evidence, history=history, public_config=public_config, checks=checks)
    run_id = _id("collect_", {"state_id": state["state_id"], "checks_id": state["checks_id"],
                            "max_candidates": max_candidates, "collector": SCHEMA})
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = teacher_prompt(state, max_candidates)
    teacher_path = run_dir / "teacher.json"
    if teacher_path.exists():
        saved = json.loads(teacher_path.read_text(encoding="utf-8"))
        if saved["prompt"] != prompt:
            raise ValueError("teacher input changed for a frozen collection run")
        raw, metadata = saved["raw"], saved["metadata"]
    else:
        answer = (teacher or _default_teacher)(prompt)
        raw, metadata = answer if isinstance(answer, tuple) else (answer, {})
        _write(teacher_path, {"prompt": prompt, "raw": raw, "metadata": metadata,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    if teacher is None:
        if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop":
            raise ValueError("teacher requires complete final JSON content; reasoning or truncated outputs are not plans")
        expected_model = state["view"]["config"].get("teacher_model")
        if metadata.get("temperature") != 0.0 or (expected_model and metadata.get("model") != expected_model):
            raise ValueError("teacher model/temperature differs from the frozen adapter policy")
    # Parse/save the complete candidate set before editing any candidate.
    plans = parse_candidates(raw, state, max_candidates)
    _write(run_dir / "plans.json", plans)
    outcomes = []
    raw_root = root / "datasets/research1/raw"
    for plan in plans:
        plan_id = _id("plan_", {"state_id": state["state_id"], "plan": plan})
        sample_id = _id("sample_", {"state_id": state["state_id"], "plan_id": plan_id})
        decision = {"schema": SCHEMA, "decision_id": plan_id, "run_id": run_id,
                    "origin_id": state["origin_id"], "view_revision": state["view_revision"],
                    "state_id": state["state_id"], "plan_id": plan_id, "plan": plan,
                    "teacher_path": str(teacher_path), "checks_id": state["checks_id"],
                    "long_horizon_label": None}
        append_unique(raw_root / "decisions.jsonl", decision, "decision_id")
        sft = {"schema": SCHEMA, "sample_id": sample_id, "run_id": run_id,
               "origin_id": state["origin_id"], "view_revision": state["view_revision"],
               "state_id": state["state_id"], "plan_id": plan_id, "task_name": task_name,
               "prompt": prompt, "completion": _json({"candidates": [plan]}), "use": "plan_supervised_init",
               "long_horizon_label": None}
        status, reason = "pending", "no applicable executable verification contract"
        outcome: dict[str, Any] = {"plan_id": plan_id, "sample_id": sample_id, "action": plan["action"]}
        if plan["action"] == "keep":
            # Abstention is allowed, but parent validity does not establish that
            # the teacher's factual diagnosis or justification is correct.
            status = "pending_review"
            referenced = [e for e in state["view"]["evidence"] if e["id"] in plan["evidence_refs"]]
            parent_valid = any(valid_task_result(e.get("task_result") or {}) for e in referenced)
            review_path = run_dir / "reviews" / plan_id / "review.json"
            review_record = None
            if review_path.is_file():
                review_record = json.loads(review_path.read_text(encoding="utf-8"))
                if review_record.get("state_id") != state["state_id"] or review_record.get("plan_id") != plan_id:
                    raise ValueError("saved keep review does not match the frozen state and plan")
            elif keep_reviewer is not None:
                try:
                    # Reviewers cannot mutate the teacher output or student input.
                    review = keep_reviewer(json.loads(_json(plan)), json.loads(_json(state)))
                    if not isinstance(review, dict):
                        review = {"status": "pending_review", "reason": "reviewer returned a non-object result"}
                except Exception as exc:
                    review = {"status": "pending_review", "reason": f"keep review error: {type(exc).__name__}: {exc}"}
                review_record = {"schema": SCHEMA, "state_id": state["state_id"],
                                 "plan_id": plan_id, "review": review}
                _write(review_path, review_record)
            review = (review_record or {}).get("review") or {}
            review_refs = review.get("evidence_refs")
            visible_ids = {e["id"] for e in state["view"]["evidence"]}
            review_pass = bool(
                review.get("status") == "PASS"
                and isinstance(review.get("reason"), str) and review["reason"].strip()
                and isinstance(review_refs, list) and review_refs
                and all(isinstance(ref, str) and ref in visible_ids for ref in review_refs)
            )
            if review_record is not None:
                outcome["keep_review"] = review_record
                outcome["keep_review_path"] = str(review_path)
            if not parent_valid:
                reason = "keep requires a valid referenced parent result and independent evidence review"
            elif review_pass:
                status, reason = "verified", "valid referenced parent result and independent keep review; no optimality or long-horizon claim"
            else:
                reason = "keep retained pending independent review of its evidence and factual justification"
        elif checks:
            attempt_id = _id("attempt_", {"plan_id": plan_id, "editor": "fixed-editor-v1",
                                         "checks_id": state["checks_id"]})
            attempt_dir = run_dir / attempt_id
            attempt_dir.mkdir(parents=True, exist_ok=True)
            result_path = attempt_dir / "attempt.json"
            if result_path.exists():
                attempt = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                try:
                    attempt = (editor or _default_editor)(plan, state, attempt_dir)
                except Exception as exc:
                    attempt = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
                attempt.update({"attempt_id": attempt_id, "plan_id": plan_id, "state_id": state["state_id"]})
                _write(result_path, attempt)
            append_unique(raw_root / "attempts.jsonl", attempt, "attempt_id")
            outcome["attempt"] = attempt
            if attempt.get("status") == "applied":
                try:
                    verification = (verifier or _default_verifier)(plan, state, attempt)
                except Exception as exc:
                    verification = {"status": "pending", "reason": f"verification error: {type(exc).__name__}: {exc}"}
                _write(attempt_dir / "verification.json", verification)
                outcome["verification"] = verification
                verified = (verification.get("status") == "PASS" and verification.get("covers_plan") is True
                            and verification.get("checks_id") == state["checks_id"])
                if verified and run_task:
                    task_path = attempt_dir / "task_run.json"
                    if task_path.exists():
                        task = json.loads(task_path.read_text(encoding="utf-8"))
                    else:
                        try:
                            task = (evaluator or _default_evaluator)(plan, state, attempt, attempt_dir)
                        except Exception as exc:
                            task = {"status": "INCOMPLETE", "reward": None, "failure_layer": "infrastructure", "error": str(exc)}
                        task["task_run_id"] = _id("task_", {"attempt_id": attempt_id, "evaluation": state["view"]["config"]})
                        _write(task_path, task)
                    append_unique(raw_root / "task_runs.jsonl", task, "task_run_id")
                    outcome["task_run"] = task
                    if valid_task_result(task):
                        status, reason = "verified", "frozen behavior contract and valid real task execution"
                    else:
                        reason = "task execution lacks a valid benchmark result"
                elif verified:
                    reason = "behavior verified; real task execution not requested"
                elif verification.get("status") == "FAIL":
                    status, reason = "rejected", "frozen behavior or preservation contract failed"
            else:
                reason = str(attempt.get("reason") or attempt.get("reject_reason") or "edit not applied")
        set_admission(root, sample_id, status, reason, sft_row=sft,
                      evidence={"state_id": state["state_id"], "plan_id": plan_id, "run_id": run_id})
        outcome.update({"status": status, "reason": reason})
        outcomes.append(outcome)
    summary = {"schema": SCHEMA, "run_id": run_id, "state_id": state["state_id"],
               "candidates": outcomes, "verified": sum(r["status"] == "verified" for r in outcomes),
               "pending": sum(r["status"] in {"pending", "pending_review"} for r in outcomes),
               "pending_review": sum(r["status"] == "pending_review" for r in outcomes),
               "long_horizon_label": None}
    _write(run_dir / "summary.json", summary)
    return summary
