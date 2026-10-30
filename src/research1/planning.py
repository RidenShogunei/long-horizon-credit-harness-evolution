"""Fixed planner snapshot + fixed editor. Plans and diffs stored separately."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research1.contracts import EditAttempt, EditPlan, write_json
from research1.versions import apply_unified_diff, load_state

PLANNER_SNAPSHOT_ID = "planner_static_v1"
EDITOR_ID = "fixed_patch_applier_v1"


def plan_id_for(parent_version_id: str, candidate_label: str, plan_text: str) -> str:
    digest = hashlib.sha256(f"{parent_version_id}|{candidate_label}|{plan_text}".encode()).hexdigest()[:12]
    return f"plan_{candidate_label}_{digest}"


def make_plan(
    *,
    parent_version_id: str,
    locked_module: str,
    plan_text: str,
    candidate_label: str,
    intended_files: list[str] | None = None,
) -> EditPlan:
    return EditPlan(
        plan_id=plan_id_for(parent_version_id, candidate_label, plan_text),
        parent_version_id=parent_version_id,
        locked_module=locked_module,
        plan_text=plan_text,
        intended_files=list(intended_files or []),
        candidate_label=candidate_label,
    )


def store_plan(run_dir: Path, plan: EditPlan) -> Path:
    path = run_dir / "plans" / f"{plan.plan_id}.json"
    write_json(path, plan)
    # keep plan text separately for audit
    (run_dir / "plans" / f"{plan.plan_id}.txt").write_text(plan.plan_text)
    return path


def store_patch(run_dir: Path, attempt_id: str, patch_text: str) -> Path:
    path = run_dir / "patches" / f"{attempt_id}.diff"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(patch_text)
    return path


def apply_plan_patch(
    *,
    run_dir: Path,
    versions_root: Path,
    plan: EditPlan,
    patch_text: str,
    child_version_id: str,
) -> EditAttempt:
    """Deterministic editor: apply saved patch bytes only."""
    attempt_id = f"attempt_{plan.candidate_label}_{child_version_id}"
    store_plan(run_dir, plan)
    store_patch(run_dir, attempt_id, patch_text)
    parent = load_state(versions_root, plan.parent_version_id)
    state, reason = apply_unified_diff(
        parent_modules=Path(parent.modules_root),
        versions_root=versions_root,
        child_version_id=child_version_id,
        parent_id=plan.parent_version_id,
        patch_text=patch_text,
        locked_module=plan.locked_module,
    )
    attempt = EditAttempt(
        attempt_id=attempt_id,
        plan_id=plan.plan_id,
        parent_version_id=plan.parent_version_id,
        child_version_id=None if state is None else state.version_id,
        patch_text=patch_text,
        applied=state is not None,
        outcome_class="apply-failure" if state is None else "no-effect",
        reject_reason=reason,
    )
    write_json(run_dir / "attempts" / f"{attempt_id}.json", attempt)
    return attempt


def frozen_planner_meta() -> dict[str, Any]:
    return {
        "planner_snapshot_id": PLANNER_SNAPSHOT_ID,
        "editor_id": EDITOR_ID,
        "note": "Plans are frozen text; editor only re-applies stored patches.",
    }


def load_plan(path: Path) -> EditPlan:
    d = json.loads(path.read_text())
    return EditPlan(**{k: d[k] for k in EditPlan.__dataclass_fields__ if k in d})
