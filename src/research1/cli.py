"""research1 CLI: preflight / calibrate / run-modularrsi / run-branches / distill-plan / audit / report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS, __version__
from research1.audit import audit_run, find_run_dir, report_run
from research1.calibrate import run_calibrate
from research1.config import REPO_ROOT, load_pilot, resolve_path
from research1.causal import run_causal
from research1.p2a import run_modularrsi
from research1.p2b import run_branches
from research1.preflight import run_preflight, write_h0_lock


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_preflight(args: argparse.Namespace) -> int:
    report = run_preflight(Path(args.config) if args.config else None)
    if args.write_h0_lock and report.status == "PASS":
        try:
            path = write_h0_lock(load_pilot(Path(args.config) if args.config else None))
            d = report.to_dict()
            d["h0_lock_written"] = str(path)
            _print(d)
            return report.exit_code
        except Exception as exc:  # noqa: BLE001
            d = report.to_dict()
            d["h0_lock_error"] = str(exc)
            _print(d)
            return EXIT_FAIL
    _print(report.to_dict())
    return report.exit_code


def cmd_calibrate(args: argparse.Namespace) -> int:
    code, doc = run_calibrate(
        Path(args.config) if args.config else None,
        max_tasks=args.max_tasks,
        max_trials=args.max_trials,
        probe=args.probe,
    )
    _print(doc)
    return code


def cmd_run_modularrsi(args: argparse.Namespace) -> int:
    code, doc = run_modularrsi(
        Path(args.config) if args.config else None,
        dry_run=args.dry_run,
        max_tasks=args.max_tasks,
    )
    _print(doc)
    return code


def cmd_run_causal(args: argparse.Namespace) -> int:
    if args.resume_dual:
        from research1.causal import resume_dual_branch_triggers

        code, doc = resume_dual_branch_triggers(
            Path(args.resume_dual),
            Path(args.config) if args.config else None,
            provider_mode=args.provider_mode,
            max_trigger_reps=args.max_trigger_reps,
        )
    elif args.dual_branch:
        from research1.causal import run_dual_branch

        code, doc = run_dual_branch(
            Path(args.config) if args.config else None,
            reps=args.reps,
            provider_mode=args.provider_mode,
            prior_h0=Path(args.prior_h0) if args.prior_h0 else None,
            max_trigger_reps=args.max_trigger_reps,
        )
    elif args.promote_false_concat:
        from research1.causal import run_false_concat_loop

        code, doc = run_false_concat_loop(
            Path(args.config) if args.config else None,
            reps=args.reps,
            provider_mode=args.provider_mode,
            max_h1_reps=args.max_h1_reps,
        )
    else:
        code, doc = run_causal(
            Path(args.config) if args.config else None,
            max_tasks=args.max_tasks,
            reps=args.reps,
            provider_mode=args.provider_mode,
        )
    _print(doc)
    return code


def cmd_distill_observed_batches(args: argparse.Namespace) -> int:
    if not _legacy_allowed(args):
        return EXIT_FAIL
    from research1.distill import run_observed_batches

    code, doc = run_observed_batches(
        Path(args.config) if args.config else None,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        max_per_task=args.max_per_task,
        provider_mode=args.provider_mode,
    )
    _print(doc)
    return code


def cmd_cover_parents(args: argparse.Namespace) -> int:
    from research1.plan_init import run_coverage_parents

    tasks = [item for item in args.tasks.split(",") if item] if args.tasks else None
    code, doc = run_coverage_parents(Path(args.config) if args.config else None, tasks=tasks)
    _print({k: doc[k] for k in ("run_id", "parent_states", "mechanism_count", "distinct_patches", "verified_passes")})
    return code


def cmd_init_plan_sample(args: argparse.Namespace) -> int:
    from research1.plan_init import run_plan_init_sample

    code, doc = run_plan_init_sample(
        trajectory=Path(args.trajectory),
        modules_root=Path(args.modules),
        task_name=args.task,
        parent_reward=args.parent_reward,
        source_kind=args.source_kind,
        run_task=not args.skip_task,
        config_path=Path(args.config) if args.config else None,
    )
    summary = {
        "run_id": doc["decision_row"]["run_id"],
        "action": doc["action"],
        "sft_eligible": doc["behavior"].get("sft_eligible"),
        "harmful_behavior": doc["behavior"].get("harmful_behavior"),
        "child_status": doc["child_task_run"].get("status"),
        "observed_delta": doc.get("observed_delta"),
        "exported_sft": bool(doc.get("exported", {}).get("sft")),
    }
    _print(summary)
    return code


def cmd_distill_llm_review(args: argparse.Namespace) -> int:
    if not _legacy_allowed(args):
        return EXIT_FAIL
    from research1.distill import run_llm_review

    code, doc = run_llm_review(
        Path(args.config) if args.config else None,
        provider_mode=args.provider_mode,
    )
    _print(doc)
    return code


def cmd_distill_observed(args: argparse.Namespace) -> int:
    if not _legacy_allowed(args):
        return EXIT_FAIL
    from research1.distill import run_observed_collection

    code, doc = run_observed_collection(
        Path(args.config) if args.config else None,
        target_positives=args.target_positives,
        max_candidates=args.max_candidates,
        max_per_task=args.max_per_task,
        provider_mode=args.provider_mode,
    )
    _print(doc)
    return code


def cmd_distill_verified(args: argparse.Namespace) -> int:
    if not _legacy_allowed(args):
        return EXIT_FAIL
    from research1.distill import run_verified_collection

    code, doc = run_verified_collection(
        Path(args.config) if args.config else None,
        target_positives=args.target_positives,
        max_candidates=args.max_candidates,
        max_per_task=args.max_per_task,
        provider_mode=args.provider_mode,
    )
    _print(doc)
    return code


def cmd_distill_plan(args: argparse.Namespace) -> int:
    if args.resume:
        from research1.distill import resume_distill_plan

        code, doc = resume_distill_plan(
            Path(args.resume),
            Path(args.config) if args.config else None,
            apply_one=not args.no_edit,
        )
    else:
        from research1.distill import run_distill_plan

        code, doc = run_distill_plan(
            Path(args.config) if args.config else None,
            limit=args.limit,
            apply_one=not args.no_edit,
            provider_mode=args.provider_mode,
        )
    _print(doc)
    return code


def cmd_run_branches(args: argparse.Namespace) -> int:
    code, doc = run_branches(
        Path(args.config) if args.config else None,
        continuations=args.continuations,
        reps=args.reps,
        skip_eval=args.skip_eval,
    )
    _print(doc)
    return code


def _legacy_allowed(args: argparse.Namespace) -> bool:
    """Old warning-only collectors are for auxiliary reproduction only."""
    if getattr(args, "allow_legacy", False):
        return True
    _print({
        "status": "FAIL",
        "reason": "This warning-only path is legacy auxiliary collection, not verified SFT.",
        "next": "Use collect-plan-init for verified initialization; --allow-legacy explicitly enables historical reproduction.",
    })
    return False


def cmd_label_continuations(args: argparse.Namespace) -> int:
    from research1.value_io import label_continuations

    code, result = label_continuations(
        Path(args.input), Path(args.output),
        expected_paths=args.paths_per_candidate, write=not args.dry_run,
    )
    _print(result)
    return code


def cmd_collect_plan_init(args: argparse.Namespace) -> int:
    from research1.collection import collect_parent, prepare_parent
    from research1.config import train_task_names

    if args.run_task and not args.execute:
        _print({"status": "FAIL", "reason": "--run-task requires --execute"})
        return EXIT_FAIL
    cfg = load_pilot(Path(args.config))
    state_path = Path(args.state).resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    task = str(state["task_name"])
    if task not in train_task_names(resolve_path(cfg, "dataset_tb")):
        _print({"status": "FAIL", "reason": "Task is absent from the manifest train split", "task": task})
        return EXIT_FAIL
    allowed = {"visible_files", "max_code_chars", "teacher_policy_id", "editor_policy_id"}
    supplied = state.get("public_config") or {}
    if set(supplied) - allowed:
        raise ValueError("Unsupported public_config fields: " + ", ".join(sorted(set(supplied) - allowed)))
    public = {
        **supplied,
        "config_path": str(Path(args.config).resolve()),
        "config_hash": cfg["_config_hash"],
        "solver_temperature": float(cfg.get("p2b_solver_temperature", 0.7)),
        "locked_module": cfg.get("locked_module") or cfg.get("LOCKED_MODULE") or "tools",
        "teacher_policy_id": supplied.get("teacher_policy_id", "grounded_plan_v2"),
        "editor_policy_id": supplied.get("editor_policy_id", "fixed_editor_preservation_v2"),
    }
    modules = Path(state["modules_root"])
    if not modules.is_absolute():
        modules = state_path.parent / modules
    checks = json.loads(Path(args.checks).read_text(encoding="utf-8")) if args.checks else None
    params = dict(root=REPO_ROOT, modules_root=modules.resolve(), task_name=task,
                  evidence=state["evidence"], history=state.get("history", []),
                  public_config=public, checks=checks)
    if not args.execute:
        frozen = prepare_parent(**params)
        _print({"status": "PREVIEW", "state_id": frozen["state_id"],
                "checks_id": frozen["checks_id"], "models_called": False,
                "next": "Add --execute to request plans; add --run-task to evaluate behavior-valid edits."})
        return EXIT_PASS
    result = collect_parent(**params, max_candidates=args.max_candidates, run_task=args.run_task)
    _print(result)
    return EXIT_PASS if result.get("verified", 0) else EXIT_INCOMPLETE


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_pilot(Path(args.config) if args.config else None)
    runs_root = resolve_path(cfg, "runs_root")
    run_dir = Path(args.run) if Path(args.run).is_dir() else find_run_dir(runs_root, args.run)
    result = audit_run(run_dir)
    _print(result)
    return {"PASS": EXIT_PASS, "INCOMPLETE": EXIT_INCOMPLETE}.get(result["status"], EXIT_FAIL)


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_pilot(Path(args.config) if args.config else None)
    runs_root = resolve_path(cfg, "runs_root")
    run_dir = Path(args.run) if Path(args.run).is_dir() else find_run_dir(runs_root, args.run)
    summary = report_run(run_dir)
    _print(summary)
    return {"PASS": EXIT_PASS, "INCOMPLETE": EXIT_INCOMPLETE}.get(summary.get("audit_status"), EXIT_FAIL)


def cmd_write_h0_lock(args: argparse.Namespace) -> int:
    try:
        path = write_h0_lock(load_pilot(Path(args.config) if args.config else None))
        _print({"status": "PASS", "h0_lock": str(path)})
        return EXIT_PASS
    except Exception as exc:  # noqa: BLE001
        _print({"status": "FAIL", "error": str(exc)})
        return EXIT_FAIL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="research1", description="research1-v1 foundation CLI")
    p.add_argument("--version", action="version", version=f"research1 {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    from research1.data_pipeline.cli import add_parser as add_data_parser
    add_data_parser(sub)

    sp = sub.add_parser("preflight", help="Check proxy/provider/disk/harbor/dataset/H0")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--write-h0-lock", action="store_true")
    sp.set_defaults(func=cmd_preflight)

    sp = sub.add_parser("write-h0-lock", help="Verify H0 hashes and write h0.lock.json")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.set_defaults(func=cmd_write_h0_lock)

    sp = sub.add_parser("calibrate", help="8→4 task calibrate + LOCKED_MODULE freeze")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--probe", action="store_true", help="Smaller probe (default 2×1)")
    sp.add_argument("--max-tasks", type=int, default=None)
    sp.add_argument("--max-trials", type=int, default=None)
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser(
        "run-modularrsi",
        help="P2A official ModularRSI Phase0 (smoke or n8 full protocol via config)",
    )
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Override config modularrsi_max_tasks (default: config or 2)",
    )
    sp.set_defaults(func=cmd_run_modularrsi)

    sp = sub.add_parser("run-causal", help="Track B: tools evidence, frozen plan, fixed editor, H1")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--max-tasks", type=int, default=3)
    sp.add_argument("--reps", type=int, default=1)
    sp.add_argument("--provider-mode", default="tongji_agi")
    sp.add_argument(
        "--promote-false-concat",
        action="store_true",
        help="After coverage found no parse error, patch the evidenced empty-keystroke warning",
    )
    sp.add_argument(
        "--dual-branch",
        action="store_true",
        help="Track B: same H0 → two functional candidates → evidence-gated continuation each",
    )
    sp.add_argument(
        "--resume-dual",
        default="",
        help="Resume an existing causal_dual_* run: re-eval until triggers, then re-gate continuations",
    )
    sp.add_argument(
        "--prior-h0",
        default="",
        help="Optional path to prior causal H0 modules (defaults to causal_edit_…/versions/H0/modules)",
    )
    sp.add_argument(
        "--max-trigger-reps",
        type=int,
        default=5,
        help="Dual-branch: keep evaluating a child until the trigger appears (cap)",
    )
    sp.add_argument("--max-h1-reps", type=int, default=3)
    sp.set_defaults(func=cmd_run_causal)

    sp = sub.add_parser(
        "distill-plan",
        help="PLAN-layer teacher data: trajectory + snapshot → plan I/O (Tongji)",
    )
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--limit", type=int, default=5, help="How many grounded plan calls (3–6)")
    sp.add_argument(
        "--no-edit",
        action="store_true",
        help="Record plan I/O only; skip the one fixed-editor proof",
    )
    sp.add_argument("--provider-mode", default="tongji_agi")
    sp.add_argument(
        "--resume",
        default="",
        help="Retry failed rows in an existing distill_plan_* directory",
    )
    sp.set_defaults(func=cmd_distill_plan)

    sp = sub.add_parser(
        "distill-verified",
        help="Collect probe-verified plan labels (edit + empty-keystroke probe)",
    )
    sp.add_argument("--allow-legacy", action="store_true", help="Explicitly reproduce auxiliary warning-only collection")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--target-positives", type=int, default=40)
    sp.add_argument("--max-candidates", type=int, default=120)
    sp.add_argument(
        "--max-per-task",
        type=int,
        default=0,
        help="Cap candidates per task (0 keeps every available case)",
    )
    sp.add_argument("--provider-mode", default="tongji_agi")
    sp.set_defaults(func=cmd_distill_verified)

    sp = sub.add_parser(
        "distill-observed",
        help="Collect plans whose quoted phenomenon disappears on replay",
    )
    sp.add_argument("--allow-legacy", action="store_true", help="Explicitly reproduce auxiliary warning-only collection")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/distill_verified_v1.yaml"))
    sp.add_argument("--target-positives", type=int, default=40)
    sp.add_argument("--max-candidates", type=int, default=18)
    sp.add_argument("--max-per-task", type=int, default=0)
    sp.add_argument("--provider-mode", default="tongji_agi")
    sp.set_defaults(func=cmd_distill_observed)

    sp = sub.add_parser(
        "distill-observed-batches",
        help="Run observed plan collection in successive batches",
    )
    sp.add_argument("--allow-legacy", action="store_true", help="Explicitly reproduce auxiliary warning-only collection")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/distill_verified_v1.yaml"))
    sp.add_argument("--batch-size", type=int, default=10)
    sp.add_argument("--max-batches", type=int, default=20)
    sp.add_argument("--max-per-task", type=int, default=4)
    sp.add_argument("--provider-mode", default="tongji_agi")
    sp.set_defaults(func=cmd_distill_observed_batches)

    sp = sub.add_parser(
        "init-plan-sample",
        help="Export one behavior-checked plan from a recoverable parent state",
    )
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--trajectory", required=True)
    sp.add_argument("--modules", required=True)
    sp.add_argument("--task", required=True)
    sp.add_argument("--parent-reward", type=float, default=0.0)
    sp.add_argument("--source-kind", default="historical_failure")
    sp.add_argument("--skip-task", action="store_true")
    sp.set_defaults(func=cmd_init_plan_sample)

    sp = sub.add_parser(
        "cover-parents",
        help="Run a fixed light-task set and record parent states without minting duplicate plans",
    )
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--tasks", default="", help="Comma-separated task names")
    sp.set_defaults(func=cmd_cover_parents)

    sp = sub.add_parser(
        "distill-llm-review",
        help="Review stored observed plans with the Tongji model",
    )
    sp.add_argument("--allow-legacy", action="store_true", help="Explicitly reproduce auxiliary LLM review; never a value label")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/distill_verified_v1.yaml"))
    sp.add_argument("--provider-mode", default="tongji_agi")
    sp.set_defaults(func=cmd_distill_llm_review)

    sp = sub.add_parser("run-branches", help="P2B mini branch loop")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--continuations", type=int, default=1)
    sp.add_argument("--reps", type=int, default=None)
    sp.add_argument(
        "--skip-eval",
        action="store_true",
        help="Wiring only; forces INCOMPLETE (never PASS)",
    )
    sp.set_defaults(func=cmd_run_branches)

    sp = sub.add_parser("collect-plan-init", help="Freeze a manifest-train parent and collect grounded edit/keep plans; preview calls no models")
    sp.add_argument("--state", required=True, help="JSON: modules_root, task_name, evidence, history, public_config")
    sp.add_argument("--checks", default="", help="Trusted behavior-contract JSON frozen before teacher generation")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--max-candidates", type=int, default=2, choices=(1, 2))
    sp.add_argument("--execute", action="store_true", help="Request teacher/editor plans using fixed adapters")
    sp.add_argument("--run-task", action="store_true", help="Run real benchmark only for behavior-valid edits")
    sp.set_defaults(func=cmd_collect_plan_init)

    sp = sub.add_parser("label-continuations", help="Recompute MC targets from completed real evolution paths; never launches experiments")
    sp.add_argument("--input", default=str(REPO_ROOT / "datasets/research1/value/continuations.jsonl"))
    sp.add_argument("--output", default=str(REPO_ROOT / "datasets/research1/value/targets.jsonl"))
    sp.add_argument("--paths-per-candidate", type=int, default=2)
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_label_continuations)

    sp = sub.add_parser("audit", help="Recompute returns / validate records")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--run", required=True, help="RUN_ID or path")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("report", help="Write report.json/md for a run")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--run", required=True, help="RUN_ID or path")
    sp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INCOMPLETE
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    raise SystemExit(main())
