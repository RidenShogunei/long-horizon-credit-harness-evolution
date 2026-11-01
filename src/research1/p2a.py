"""P2A: official ModularRSI PROFILE=smoke with LOCKED_MODULE from calibrate."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS
from research1.adapters import (
    apply_provider_to_environ,
    merge_no_proxy,
    run_official_evolve_smoke,
    save_process_logs,
    with_llm_provider_env,
    write_local_adaptations_note,
)
from research1.config import load_json, load_pilot, resolve_path, train_task_names
from research1.contracts import write_json
from research1.preflight import run_preflight

# Upstream phase0: task_subset = train_tasks[:max_tasks], then Random(seed).shuffle.
# Smoke max_tasks=2, so every seed is the manifest prefix. Seed 0 is the
# upstream default; it does not drop build-ffmpeg-legacy.
SMOKE_TASK_SEED = 0
SMOKE_PROVIDER_MODE = "tongji_agi"


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


def smoke_task_queue(tasks: list[str], max_tasks: int, task_seed: int, epochs: int = 1) -> list[str]:
    """Match online_evo.py: slice to max_tasks, then shuffle each epoch."""
    subset = list(tasks[:max_tasks])
    rng = random.Random(task_seed)
    queued: list[str] = []
    for _ in range(max(1, epochs)):
        epoch = list(subset)
        rng.shuffle(epoch)
        queued.extend(epoch)
    return queued


def run_modularrsi(
    config_path: Path | None = None,
    *,
    dry_run: bool = False,
    max_tasks: int | None = None,
    task_seed: int | None = None,
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

    # Track A forces Tongji. pilot.yaml provider_mode=auto may fall back to
    # the official DeepSeek API; that is not this reproduction.
    prov = with_llm_provider_env(SMOKE_PROVIDER_MODE)
    if prov.provider_id != SMOKE_PROVIDER_MODE:
        return EXIT_FAIL, {
            "status": "FAIL",
            "reason": f"expected provider {SMOKE_PROVIDER_MODE}, got {prov.provider_id}",
            "LOCKED_MODULE": locked,
        }
    if "agi.tongji.edu.cn" not in (prov.api_base or ""):
        return EXIT_FAIL, {
            "status": "FAIL",
            "reason": f"Tongji api_base required, got {prov.api_base}",
            "LOCKED_MODULE": locked,
        }
    apply_provider_to_environ(prov)
    # P2A solver temp 0.0 (official Phase0)
    # evolve.sh / phase0 hardcodes SOLVER_TEMPERATURE=0.0

    write_local_adaptations_note(out / "A13_local_adaptations.md")
    mr = resolve_path(cfg, "modularrsi_root")
    ds = resolve_path(cfg, "dataset_tb")
    # Workload: config defaults, CLI max_tasks overrides when provided.
    profile = str(cfg.get("modularrsi_profile") or cfg.get("profile") or "smoke")
    cap = int(cfg.get("modularrsi_max_tasks") or cfg.get("max_tasks") or 2)
    if max_tasks is not None:
        cap = int(max_tasks)
    seed = int(cfg.get("modularrsi_task_seed") or cfg.get("task_seed") or SMOKE_TASK_SEED)
    if task_seed is not None:
        seed = int(task_seed)
    epochs = int(cfg.get("modularrsi_epochs") or 1)
    reflect_every = int(cfg.get("modularrsi_reflect_every") or (2 if profile == "smoke" else 10))
    task_concurrency = int(cfg.get("modularrsi_task_concurrency") or (2 if profile == "smoke" else 6))
    attempts = int(cfg.get("modularrsi_attempts") or 3)

    train = train_task_names(ds)
    queued = smoke_task_queue(train, cap, seed, epochs=epochs)
    write_json(
        out / "task_selection.json",
        {
            "task_seed": seed,
            "max_tasks": cap,
            "epochs": epochs,
            "profile": profile,
            "reflect_every": reflect_every,
            "task_concurrency": task_concurrency,
            "attempts": attempts,
            "selector": "upstream_slice_then_shuffle",
            "note": (
                "phase0 slices tasks[:max_tasks] before Random(task_seed).shuffle "
                "each epoch. Seed does not change the prefix membership."
            ),
            "train_prefix": train[:cap],
            "queued_tasks": queued,
            "heavy_image_tasks_in_queue": [
                name
                for name in queued
                if name in {"build-ffmpeg-legacy", "build-imagemagick-legacy"}
            ],
        },
    )

    proc = run_official_evolve_smoke(
        modularrsi_root=mr,
        support_dataset_dir=ds,
        locked_module=locked,
        archive_root=out / "archive",
        model=prov.model,
        api_base=prov.api_base,
        profile=profile,
        max_tasks=cap,
        task_seed=seed,
        epochs=epochs,
        reflect_every=reflect_every,
        task_concurrency=task_concurrency,
        attempts=attempts,
        dry_run=dry_run,
    )
    save_process_logs(out, proc, name="evolve")
    summary = {
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "run_id": run_id,
        "LOCKED_MODULE": locked,
        "profile": profile,
        "max_tasks": cap,
        "epochs": epochs,
        "reflect_every": reflect_every,
        "task_concurrency": task_concurrency,
        "attempts": attempts,
        "solver_temperature_policy": "upstream_phase0_0.0",
        "provider_id": prov.provider_id,
        "model": prov.model,
        "api_base": prov.api_base,
        "task_seed": seed,
        "queued_tasks": queued,
        "no_proxy": merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy")),
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
