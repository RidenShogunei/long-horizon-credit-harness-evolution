"""Same-parent fork, continuations, and interrupt resume for P2B."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research1.contracts import ReturnLabel, Transition, write_json
from research1.planning import apply_plan_patch, make_plan, store_plan
from research1.returns import StepScores, label_branch, mean_score
from research1.versions import load_state, materialize_version


def ensure_h0_version(
    versions_root: Path,
    modules_source: Path,
    version_id: str = "H0",
) -> Any:
    dest = versions_root / version_id
    if (dest / "state.json").is_file():
        return load_state(versions_root, version_id)
    return materialize_version(
        versions_root=versions_root,
        version_id=version_id,
        source_modules=modules_source,
        parent_id=None,
        note="frozen H0 baseline",
    )


def fork_candidates(
    *,
    run_dir: Path,
    versions_root: Path,
    parent_version_id: str,
    locked_module: str,
    candidates: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """
    From the same parent snapshot, create plans A/B and apply patches.
    Each candidate dict: {label, plan_text, patch_text}.
    Isolation: no candidate may read another candidate's eval results here.
    """
    results: list[dict[str, Any]] = []
    parent = load_state(versions_root, parent_version_id)
    for cand in candidates:
        label = cand["label"]
        plan = make_plan(
            parent_version_id=parent.version_id,
            locked_module=locked_module,
            plan_text=cand["plan_text"],
            candidate_label=label,
        )
        store_plan(run_dir, plan)
        child_id = f"{parent_version_id}_{label}1"
        attempt = apply_plan_patch(
            run_dir=run_dir,
            versions_root=versions_root,
            plan=plan,
            patch_text=cand["patch_text"],
            child_version_id=child_id,
        )
        results.append(
            {
                "label": label,
                "plan": plan.to_dict(),
                "attempt": attempt.to_dict(),
                "child_version_id": attempt.child_version_id,
            }
        )
    write_json(run_dir / "fork_manifest.json", results)
    return results


def continue_once(
    *,
    run_dir: Path,
    versions_root: Path,
    parent_child_version_id: str,
    locked_module: str,
    label: str,
    plan_text: str,
    patch_text: str,
) -> dict[str, Any]:
    """One frozen-policy continuation from a candidate child (e.g. A1 → A2)."""
    plan = make_plan(
        parent_version_id=parent_child_version_id,
        locked_module=locked_module,
        plan_text=plan_text,
        candidate_label=f"{label}_cont",
    )
    child_id = parent_child_version_id.replace("1", "2") if parent_child_version_id.endswith("1") else f"{parent_child_version_id}_c2"
    # Prefer deterministic A1→A2 naming
    if parent_child_version_id.endswith("A1"):
        child_id = parent_child_version_id[:-1] + "2"
    elif parent_child_version_id.endswith("B1"):
        child_id = parent_child_version_id[:-1] + "2"
    attempt = apply_plan_patch(
        run_dir=run_dir,
        versions_root=versions_root,
        plan=plan,
        patch_text=patch_text,
        child_version_id=child_id,
    )
    out = {
        "label": label,
        "plan": plan.to_dict(),
        "attempt": attempt.to_dict(),
        "child_version_id": attempt.child_version_id,
    }
    write_json(run_dir / f"continuation_{label}.json", out)
    return out


def write_transition(
    run_dir: Path,
    *,
    transition_id: str,
    parent_version_id: str,
    child_version_id: str,
    plan_id: str,
    attempt_id: str,
    parent_scores: dict[str, float],
    child_scores: dict[str, float],
) -> Transition:
    t = Transition(
        transition_id=transition_id,
        parent_version_id=parent_version_id,
        child_version_id=child_version_id,
        plan_id=plan_id,
        attempt_id=attempt_id,
        parent_step_scores=parent_scores,
        child_step_scores=child_scores,
    )
    write_json(run_dir / "transitions" / f"{transition_id}.json", t)
    return t


def label_path_from_score_tables(
    run_dir: Path,
    *,
    branch_id: str,
    version_order: list[str],
    score_tables: dict[str, dict[str, float]],
    lambda_value: float = 0.5,
) -> ReturnLabel:
    path = [
        StepScores(vid, mean_score(score_tables[vid])) for vid in version_order
    ]
    lab = label_branch(
        label_id=f"return_{branch_id}",
        branch_id=branch_id,
        path=path,
        lambda_value=lambda_value,
    )
    write_json(run_dir / "returns" / f"{branch_id}.json", lab)
    return lab


def resume_marker(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resume.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"completed_steps": [], "updated_unix": time.time()}


def mark_step(run_dir: Path, step: str) -> None:
    m = resume_marker(run_dir)
    if step not in m["completed_steps"]:
        m["completed_steps"].append(step)
    m["updated_unix"] = time.time()
    write_json(run_dir / "resume.json", m)
