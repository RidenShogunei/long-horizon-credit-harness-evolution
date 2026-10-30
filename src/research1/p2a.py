"""P2A: official ModularRSI PROFILE=smoke with LOCKED_MODULE from calibrate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS
from research1.adapters import (
    apply_provider_to_environ,
    run_official_evolve_smoke,
    save_process_logs,
    with_llm_provider_env,
    write_local_adaptations_note,
)
from research1.config import load_json, load_pilot, resolve_path
from research1.contracts import write_json
from research1.preflight import run_preflight


def _locked_module(cfg: dict[str, Any]) -> str | None:
    if cfg.get("LOCKED_MODULE"):
        return str(cfg["LOCKED_MODULE"])
    stamp = Path(cfg["_config_path"]).parent / "calibrate.stamp.json"
    if stamp.is_file():
        return load_json(stamp).get("LOCKED_MODULE")
    lock_path = resolve_path(cfg, "tasks_lock") if cfg.get("tasks_lock") else (
        Path(cfg["_config_path"]).parent / "tasks.lock.json"
    )
    if not lock_path.is_absolute():
        from research1.config import REPO_ROOT

        lock_path = REPO_ROOT / lock_path
    if lock_path.is_file():
        return load_json(lock_path).get("LOCKED_MODULE")
    return None


def run_modularrsi(
    config_path: Path | None = None,
    *,
    dry_run: bool = False,
    max_tasks: int | None = 2,
) -> tuple[int, dict[str, Any]]:
    cfg = load_pilot(config_path)
    pf = run_preflight(config_path)
    locked = _locked_module(cfg)
    if locked is None:
        return EXIT_INCOMPLETE, {
            "status": "INCOMPLETE",
            "reason": "LOCKED_MODULE missing; run calibrate first",
            "preflight": pf.to_dict(),
        }
    if pf.status != "PASS":
        return EXIT_INCOMPLETE, {
            "status": "INCOMPLETE",
            "reason": "preflight_not_pass",
            "preflight": pf.to_dict(),
            "LOCKED_MODULE": locked,
        }

    run_id = time.strftime("modularrsi_%Y%m%d_%H%M%S", time.gmtime())
    runs_root = resolve_path(cfg, "runs_root")
    out = runs_root / run_id
    out.mkdir(parents=True, exist_ok=True)

    prov = with_llm_provider_env(cfg.get("provider_mode"))
    apply_provider_to_environ(prov)
    # P2A solver temp 0.0 (official Phase0)
    # evolve.sh / phase0 hardcodes SOLVER_TEMPERATURE=0.0

    write_local_adaptations_note(out / "A13_local_adaptations.md")
    mr = resolve_path(cfg, "modularrsi_root")
    ds = resolve_path(cfg, "dataset_tb")

    proc = run_official_evolve_smoke(
        modularrsi_root=mr,
        support_dataset_dir=ds,
        locked_module=locked,
        archive_root=out / "archive",
        model=prov.model,
        api_base=prov.api_base,
        profile="smoke",
        max_tasks=max_tasks,
        dry_run=dry_run,
    )
    save_process_logs(out, proc, name="evolve")
    summary = {
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "run_id": run_id,
        "LOCKED_MODULE": locked,
        "profile": "smoke",
        "solver_temperature_policy": "upstream_phase0_0.0",
        "provider_id": prov.provider_id,
        "model": prov.model,
        "exit_code": proc.returncode,
        "dry_run": dry_run,
        "archive": str(out / "archive"),
        "A13": str(out / "A13_local_adaptations.md"),
    }
    if proc.returncode != 0:
        # Quota / missing dataset → INCOMPLETE
        blob = (proc.stderr or "") + (proc.stdout or "")
        if any(s in blob.lower() for s in ("quota", "429", "rate limit", "dataset")):
            summary["status"] = "INCOMPLETE"
            summary["blocker"] = blob[-2000:]
    write_json(out / "summary.json", summary)
    code = {
        "PASS": EXIT_PASS,
        "FAIL": EXIT_FAIL,
        "INCOMPLETE": EXIT_INCOMPLETE,
    }[summary["status"]]
    return code, summary
