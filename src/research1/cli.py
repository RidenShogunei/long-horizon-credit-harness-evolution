"""research1 CLI: preflight / calibrate / run-modularrsi / run-branches / audit / report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS, __version__
from research1.audit import audit_run, find_run_dir, report_run
from research1.calibrate import run_calibrate
from research1.config import REPO_ROOT, load_pilot, resolve_path
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


def cmd_run_branches(args: argparse.Namespace) -> int:
    code, doc = run_branches(
        Path(args.config) if args.config else None,
        continuations=args.continuations,
        reps=args.reps,
        skip_eval=args.skip_eval,
    )
    _print(doc)
    return code


def cmd_audit(args: argparse.Namespace) -> int:
    cfg = load_pilot(Path(args.config) if args.config else None)
    runs_root = resolve_path(cfg, "runs_root")
    run_dir = Path(args.run) if Path(args.run).is_dir() else find_run_dir(runs_root, args.run)
    result = audit_run(run_dir)
    _print(result)
    return EXIT_PASS if result["status"] == "PASS" else EXIT_FAIL


def cmd_report(args: argparse.Namespace) -> int:
    cfg = load_pilot(Path(args.config) if args.config else None)
    runs_root = resolve_path(cfg, "runs_root")
    run_dir = Path(args.run) if Path(args.run).is_dir() else find_run_dir(runs_root, args.run)
    summary = report_run(run_dir)
    _print(summary)
    return EXIT_PASS if summary.get("audit_status") == "PASS" else EXIT_FAIL


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

    sp = sub.add_parser("run-modularrsi", help="P2A official PROFILE=smoke")
    sp.add_argument("--config", default=str(REPO_ROOT / "configs/research1/pilot.yaml"))
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--max-tasks", type=int, default=2)
    sp.set_defaults(func=cmd_run_modularrsi)

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
