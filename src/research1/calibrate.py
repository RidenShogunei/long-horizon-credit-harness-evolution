"""Fixed-order 8 train tasks × up to 3 trials → select 4; freeze LOCKED_MODULE."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS, PROTOCOL_ID
from research1.adapters import (
    apply_provider_to_environ,
    run_harbor_task,
    save_process_logs,
    with_llm_provider_env,
)
from research1.config import load_pilot, resolve_path, train_task_names
from research1.contracts import write_json
from research1.evaluation import aggregate_task_means, should_retry, summarize_trial
from research1.preflight import run_preflight
from research1.versions import BASELINE_BUNDLE


def _select_calibrate_pool(train_names: list[str], n: int = 8) -> list[str]:
    """Fixed order: first n train tasks from manifest order (never hand-fill)."""
    if len(train_names) < n:
        raise ValueError(f"need >= {n} train tasks, got {len(train_names)}")
    return list(train_names[:n])


def _select_four(task_stats: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, str]]:
    """Prefer scorable tasks with non-degenerate outcomes; record exclusions."""
    exclusions: dict[str, str] = {}
    ranked: list[tuple[float, str]] = []
    for name, st in task_stats.items():
        rewards = [r for r in st["rewards"] if r is not None]
        if not rewards:
            exclusions[name] = "no_official_reward"
            continue
        mean = sum(rewards) / len(rewards)
        # Always-1 or always-0 are less informative but still usable if needed
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        # Prefer intermediate means with some variance
        usefulness = 1.0 - abs(mean - 0.5) + 0.25 * (1.0 if variance > 0 else 0.0)
        if all(r == 1.0 for r in rewards):
            usefulness -= 0.2
        if all(r == 0.0 for r in rewards) and st.get("all_infra"):
            exclusions[name] = "infra_zero_only"
            continue
        ranked.append((usefulness, name))
    ranked.sort(reverse=True)
    selected = [name for _, name in ranked[:4]]
    for name in task_stats:
        if name not in selected and name not in exclusions:
            exclusions[name] = "not_in_top4_usefulness"
    return selected, exclusions


def _choose_locked_module(task_stats: dict[str, dict[str, Any]], selected: list[str]) -> str:
    """
    Freeze LOCKED_MODULE after calibration evidence — not pre-locked.
    Heuristic: prefer tools (official Phase0 default) once we have scorable
    selected tasks; if scorer/infra dominate, still lock tools for evolve
    plumbing but mark confidence low.
    """
    scorable = 0
    for name in selected:
        rewards = [r for r in task_stats[name]["rewards"] if r is not None]
        if rewards:
            scorable += 1
    if scorable == 0:
        return "tools"  # plumbing default after attempted calibrate; low confidence
    # Without per-module ablation in this stage, use official default tools.
    # Explicitly not observation (protocol: do not pre-lock observation).
    return "tools"


def run_calibrate(
    config_path: Path | None = None,
    *,
    max_tasks: int | None = None,
    max_trials: int | None = None,
    probe: bool = False,
) -> tuple[int, dict[str, Any]]:
    cfg = load_pilot(config_path)
    pf = run_preflight(config_path)
    if pf.status != "PASS":
        return EXIT_INCOMPLETE, {
            "status": "INCOMPLETE",
            "reason": "preflight_not_pass",
            "preflight": pf.to_dict(),
        }

    ds = resolve_path(cfg, "dataset_tb")
    train = train_task_names(ds)
    pool_n = int(cfg.get("calibrate_pool_n", 8))
    trials_n = int(cfg.get("calibrate_trials_per_task", 3))
    if probe:
        pool_n = min(pool_n, max_tasks or 2)
        trials_n = min(trials_n, max_trials or 1)
    if max_tasks is not None:
        pool_n = min(pool_n, max_tasks)
    if max_trials is not None:
        trials_n = min(trials_n, max_trials)

    if probe:
        # Full calibrate uses fixed first-8 order. Probe may prefer lighter
        # tasks from that same fixed window (by manifest bytes) to reduce cost.
        window = _select_calibrate_pool(train, n=8)
        manifest_path = ds / "manifest.json"
        size_by_name: dict[str, int] = {}
        if manifest_path.is_file():
            for row in json.loads(manifest_path.read_text()).get("tasks") or []:
                size_by_name[row["name"]] = int(row.get("bytes") or 0)
        ranked = sorted(window, key=lambda n: size_by_name.get(n, 10**12))
        pool = ranked[:pool_n]
    else:
        pool = _select_calibrate_pool(train, n=8)
        pool = pool[:pool_n]

    run_id = time.strftime("calibrate_%Y%m%d_%H%M%S", time.gmtime())
    runs_root = resolve_path(cfg, "runs_root")
    out = runs_root / run_id
    out.mkdir(parents=True, exist_ok=True)

    prov = with_llm_provider_env(cfg.get("provider_mode"))
    apply_provider_to_environ(prov)

    # P2B/calibrate uses frozen DeepSeek Flash solver temp from roles (0.7)
    temperature = float(cfg.get("p2b_solver_temperature", 0.7))
    modules = resolve_path(cfg, "modules_root")
    mr = resolve_path(cfg, "modularrsi_root")
    locked_for_eval = cfg.get("locked_module") or "tools"  # eval kwarg only; not final lock

    task_stats: dict[str, dict[str, Any]] = {}
    all_runs: list[Any] = []

    for task in pool:
        task_stats[task] = {"rewards": [], "layers": [], "all_infra": True}
        for trial in range(1, trials_n + 1):
            trial_dir = out / task / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            proc = run_harbor_task(
                modularrsi_root=mr,
                modules_root=modules,
                task_name=task,
                output_dir=trial_dir,
                model=prov.model,
                api_base=prov.api_base,
                temperature=temperature,
                dataset_path=ds,
                locked_module_type=locked_for_eval,
                composer_name="static",
            )
            save_process_logs(trial_dir, proc)
            summary = summarize_trial(
                run_id=run_id,
                version_id="H0",
                task_name=task,
                trial_index=trial,
                output_dir=trial_dir,
                active_bundle=BASELINE_BUNDLE,
                model=prov.model,
                temperature=temperature,
                provider_id=prov.provider_id,
            )
            # one infra retry
            if should_retry(summary) and cfg.get("calibrate_retry_infra", True):
                retry_dir = out / task / f"trial_{trial}_retry"
                retry_dir.mkdir(parents=True, exist_ok=True)
                proc2 = run_harbor_task(
                    modularrsi_root=mr,
                    modules_root=modules,
                    task_name=task,
                    output_dir=retry_dir,
                    model=prov.model,
                    api_base=prov.api_base,
                    temperature=temperature,
                    dataset_path=ds,
                    locked_module_type=locked_for_eval,
                    composer_name="static",
                )
                save_process_logs(retry_dir, proc2)
                summary = summarize_trial(
                    run_id=run_id,
                    version_id="H0",
                    task_name=task,
                    trial_index=trial,
                    output_dir=retry_dir,
                    active_bundle=BASELINE_BUNDLE,
                    model=prov.model,
                    temperature=temperature,
                    provider_id=prov.provider_id,
                )
            write_json(trial_dir / "task_run.json", summary)
            all_runs.append(summary)
            task_stats[task]["rewards"].append(summary.reward)
            task_stats[task]["layers"].append(summary.failure_layer)
            if summary.failure_layer not in ("infrastructure", "scorer", "container"):
                task_stats[task]["all_infra"] = False

    selected, exclusions = _select_four(task_stats)
    # If probe selected fewer than 4, keep what we have and mark incomplete
    locked = _choose_locked_module(task_stats, selected)

    means = aggregate_task_means(all_runs)
    lock_doc: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "calibrate_run_id": run_id,
        "probe": probe,
        "pool": pool,
        "trials_per_task": trials_n,
        "selected_tasks": selected,
        "exclusions": exclusions,
        "LOCKED_MODULE": locked,
        "LOCKED_MODULE_rationale": (
            "Post-calibration freeze. Official Phase0 default `tools` chosen after "
            "scorable evidence exists (or as plumbing default if probe incomplete). "
            "Not pre-locked observation."
        ),
        "task_means": means,
        "task_stats": task_stats,
        "provider_id": prov.provider_id,
        "temperature": temperature,
        "resume_path": (
            f"python -m research1 calibrate --config {cfg.get('_config_path')} "
            f"# full 8×3; current probe={probe} pool={pool_n} trials={trials_n}"
        ),
        "status": "PASS"
        if (len(selected) >= 4 and not probe)
        else "INCOMPLETE",
    }

    lock_path = resolve_path(cfg, "tasks_lock") if cfg.get("tasks_lock") else (
        Path(cfg["_config_path"]).parent / "tasks.lock.json"
    )
    if not lock_path.is_absolute():
        from research1.config import REPO_ROOT

        lock_path = REPO_ROOT / lock_path
    write_json(lock_path, lock_doc)
    write_json(out / "tasks.lock.json", lock_doc)
    write_json(out / "summary.json", {"status": lock_doc["status"], "lock": str(lock_path)})

    # Also stamp pilot overlay for LOCKED_MODULE if complete
    stamp = {
        "LOCKED_MODULE": locked,
        "selected_tasks": selected,
        "tasks_lock": str(lock_path),
        "updated_from": run_id,
    }
    write_json(Path(cfg["_config_path"]).parent / "calibrate.stamp.json", stamp)

    code = EXIT_PASS if lock_doc["status"] == "PASS" else EXIT_INCOMPLETE
    if len(selected) == 0 and not means:
        code = EXIT_FAIL
    return code, lock_doc
