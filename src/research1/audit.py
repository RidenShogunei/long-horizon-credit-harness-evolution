"""Audit / report: recompute returns from stored tables; check isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS
from research1.contracts import validate_task_run
from research1.returns import IncompleteReturnError, recompute_from_tables


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def find_run_dir(runs_root: Path, run_id: str) -> Path:
    direct = runs_root / run_id
    if direct.is_dir():
        return direct
    # allow prefix search
    matches = sorted(runs_root.glob(f"{run_id}*"))
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"run not found: {run_id} under {runs_root}")


def audit_run(run_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    notes: list[str] = []

    # Task runs
    for tr in run_dir.rglob("task_run.json"):
        d = _load_json(tr)
        issues.extend(f"{tr}: {e}" for e in validate_task_run(d))
        if d.get("reward") is None and d.get("failure_layer") == "unknown":
            notes.append(f"{tr}: unknown kept null (good)")

    # Branch isolation: A must not embed B rewards in its own score tables wrongly
    score_path = run_dir / "score_tables.json"
    recomputed: list[dict[str, Any]] = []
    incomplete_returns = False
    if score_path.is_file():
        tables_doc = _load_json(score_path)
        lambda_value = float(tables_doc.get("lambda", 0.5))
        for branch in tables_doc.get("branches", []):
            stored = branch.get("stored_return")
            try:
                lab = recompute_from_tables(
                    branch_id=branch["branch_id"], version_order=branch["version_order"],
                    score_tables=tables_doc["scores"], lambda_value=lambda_value,
                    expected_tasks=tables_doc.get("expected_tasks"),
                    trial_counts=tables_doc.get("trial_counts"),
                    required_repetitions=tables_doc.get("required_repetitions"),
                    task_weights=tables_doc.get("task_weights"),
                    label_scope=tables_doc.get("label_scope", "legacy_diagnostic"),
                )
            except IncompleteReturnError as exc:
                incomplete_returns = True
                notes.append(f"branch {branch['branch_id']}: incomplete returns: {exc}")
                continue
            if not lab.is_complete or lab.label_scope != "real_evolution":
                incomplete_returns = True
                notes.append(f"branch {branch['branch_id']}: diagnostic/incomplete label, not formal evolution supervision")
            if stored and stored.get("metric_definition") == lab.metric_definition:
                for field in ("terminal_gain", "cumulative_regression", "u_lambda"):
                    if abs(float(stored.get(field, 0)) - float(getattr(lab, field))) > 1e-9:
                        issues.append(
                            f"branch {branch['branch_id']}: {field} mismatch "
                            f"stored={stored.get(field)} recomputed={getattr(lab, field)}"
                        )
            recomputed.append(lab.to_dict())
            # Write recomputation artifact
            out = run_dir / "returns" / f"recomputed_{branch['branch_id']}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(lab.to_dict(), indent=2) + "\n")
    else:
        notes.append("no score_tables.json — skip return recompute")

    # Fork isolation check
    fork = run_dir / "fork_manifest.json"
    if fork.is_file():
        notes.append("fork_manifest present")

    status = "FAIL" if issues else ("INCOMPLETE" if incomplete_returns else "PASS")
    return {
        "status": status,
        "run_dir": str(run_dir),
        "issues": issues,
        "notes": notes,
        "recomputed_returns": recomputed,
        "returns_complete": bool(recomputed) and not incomplete_returns,
    }


def report_run(run_dir: Path) -> dict[str, Any]:
    audit = audit_run(run_dir)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "audit_status": audit["status"],
        "issues": audit["issues"],
    }
    for name in ("summary.json", "fork_manifest.json", "tasks.lock.json", "evolve_invoke.json"):
        p = run_dir / name
        if p.is_file():
            summary[name] = _load_json(p)
    returns_dir = run_dir / "returns"
    if returns_dir.is_dir():
        summary["returns"] = {
            p.name: _load_json(p) for p in sorted(returns_dir.glob("*.json"))
        }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    # Markdown short report
    md = [
        f"# Report `{run_dir.name}`",
        "",
        f"- audit: **{audit['status']}**",
        f"- issues: {len(audit['issues'])}",
        "",
    ]
    if audit["recomputed_returns"]:
        md.append("## Recomputed returns")
        for lab in audit["recomputed_returns"]:
            md.append(
                f"- `{lab['branch_id']}`: terminal_gain={lab['terminal_gain']:.4f} "
                f"cum_reg={lab['cumulative_regression']:.4f} U={lab['u_lambda']:.4f}"
            )
    (run_dir / "report.md").write_text("\n".join(md) + "\n")
    return summary


def exit_for_audit(result: dict[str, Any]) -> int:
    if result.get("status") == "PASS":
        return EXIT_PASS
    if result.get("status") == "INCOMPLETE":
        return EXIT_INCOMPLETE
    return EXIT_FAIL
