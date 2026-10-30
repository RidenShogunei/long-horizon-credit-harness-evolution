"""P2B: same-parent fork → two candidates → 1 continuation each (mini loop)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS
from research1.adapters import (
    apply_provider_to_environ,
    run_harbor_task,
    save_process_logs,
    with_llm_provider_env,
)
from research1.config import load_json, load_pilot, resolve_path
from research1.contracts import write_json
from research1.evaluation import aggregate_task_means, summarize_trial
from research1.preflight import run_preflight
from research1.rollout import (
    continue_once,
    ensure_h0_version,
    fork_candidates,
    label_path_from_score_tables,
    mark_step,
    resume_marker,
    write_transition,
)
from research1.versions import BASELINE_BUNDLE


def _tasks_and_module(cfg: dict[str, Any]) -> tuple[list[str], str]:
    stamp = Path(cfg["_config_path"]).parent / "calibrate.stamp.json"
    lock_path = Path(cfg.get("tasks_lock") or (Path(cfg["_config_path"]).parent / "tasks.lock.json"))
    if not lock_path.is_absolute():
        from research1.config import REPO_ROOT

        lock_path = REPO_ROOT / lock_path
    selected: list[str] = []
    locked = cfg.get("LOCKED_MODULE")
    if stamp.is_file():
        st = load_json(stamp)
        selected = list(st.get("selected_tasks") or [])
        locked = locked or st.get("LOCKED_MODULE")
    if lock_path.is_file():
        lk = load_json(lock_path)
        selected = selected or list(lk.get("selected_tasks") or [])
        locked = locked or lk.get("LOCKED_MODULE")
    # Mini-loop uses first 2 selected (or pilot override)
    mini = list(cfg.get("p2b_tasks") or selected[:2])
    if not locked:
        locked = "tools"
    return mini, str(locked)


def _comment_patch(modules_root: Path, module: str, tag: str) -> str:
    """Minimal unified diff: prepend a comment to baseline.py (apply-safe)."""
    import difflib

    rel = f"{module}/baseline.py"
    path = modules_root / rel
    original = path.read_text().splitlines(keepends=True)
    if not original:
        original = ["\n"]
    updated = [f"# research1-p2b-marker: {tag}\n", *original]
    diff = difflib.unified_diff(
        original,
        updated,
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        n=3,
    )
    text = "".join(diff)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _eval_version(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    version_id: str,
    modules_root: Path,
    tasks: list[str],
    reps: int,
    prov: Any,
    temperature: float,
    locked_module: str,
) -> dict[str, float]:
    mr = resolve_path(cfg, "modularrsi_root")
    ds = resolve_path(cfg, "dataset_tb")
    runs = []
    for task in tasks:
        for rep in range(1, reps + 1):
            trial_dir = run_dir / "evals" / version_id / task / f"rep_{rep}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            proc = run_harbor_task(
                modularrsi_root=mr,
                modules_root=modules_root,
                task_name=task,
                output_dir=trial_dir,
                model=prov.model,
                api_base=prov.api_base,
                temperature=temperature,
                dataset_path=ds,
                locked_module_type=locked_module,
                composer_name="static",
            )
            save_process_logs(trial_dir, proc)
            summary = summarize_trial(
                run_id=run_dir.name,
                version_id=version_id,
                task_name=task,
                trial_index=rep,
                output_dir=trial_dir,
                active_bundle=BASELINE_BUNDLE,
                model=prov.model,
                temperature=temperature,
                provider_id=prov.provider_id,
            )
            write_json(trial_dir / "task_run.json", summary)
            runs.append(summary)
    return aggregate_task_means(runs)


def run_branches(
    config_path: Path | None = None,
    *,
    continuations: int = 1,
    reps: int | None = None,
    skip_eval: bool = False,
) -> tuple[int, dict[str, Any]]:
    """
    Mini loop: H0 → plans A/B → A1/B1 eval → each +1 continuation → returns.
    skip_eval is for wiring tests only; status becomes INCOMPLETE (never PASS).
    """
    cfg = load_pilot(config_path)
    pf = run_preflight(config_path)
    tasks, locked = _tasks_and_module(cfg)
    if len(tasks) < 1:
        return EXIT_INCOMPLETE, {
            "status": "INCOMPLETE",
            "reason": "no calibrated tasks; run calibrate first",
            "preflight": pf.to_dict(),
        }
    tasks = tasks[:2]
    if pf.status != "PASS":
        return EXIT_INCOMPLETE, {
            "status": "INCOMPLETE",
            "reason": "preflight_not_pass",
            "preflight": pf.to_dict(),
        }

    reps = int(reps if reps is not None else cfg.get("p2b_reps", 2))
    temperature = float(cfg.get("p2b_solver_temperature", 0.7))
    lambda_value = float(cfg.get("lambda", 0.5))

    run_id = time.strftime("branches_%Y%m%d_%H%M%S", time.gmtime())
    runs_root = resolve_path(cfg, "runs_root")
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    versions_root = run_dir / "versions"

    prov = with_llm_provider_env(cfg.get("provider_mode"))
    apply_provider_to_environ(prov)

    modules_src = resolve_path(cfg, "modules_root")
    h0 = ensure_h0_version(versions_root, modules_src, "H0")
    mark_step(run_dir, "h0_materialized")

    candidates = [
        {
            "label": "A",
            "plan_text": f"P2B candidate A: marker comment on {locked}/baseline.py",
            "patch_text": _comment_patch(Path(h0.modules_root), locked, "A1"),
        },
        {
            "label": "B",
            "plan_text": f"P2B candidate B: marker comment on {locked}/baseline.py (independent)",
            "patch_text": _comment_patch(Path(h0.modules_root), locked, "B1"),
        },
    ]
    # B must not read A's execution results — fork before any eval.
    forks = fork_candidates(
        run_dir=run_dir,
        versions_root=versions_root,
        parent_version_id=h0.version_id,
        locked_module=locked,
        candidates=candidates,
    )
    mark_step(run_dir, "forked")

    # If patches failed (fragile context), fall back to copy-only children via empty note files
    for f in forks:
        if not f.get("child_version_id"):
            return EXIT_FAIL, {
                "status": "FAIL",
                "reason": "apply-failure on candidate",
                "forks": forks,
                "run_dir": str(run_dir),
            }

    scores: dict[str, dict[str, float]] = {}
    if skip_eval:
        # Wiring only — never claim PASS
        scores = {
            "H0": {t: 0.0 for t in tasks},
            forks[0]["child_version_id"]: {t: 0.0 for t in tasks},
            forks[1]["child_version_id"]: {t: 0.0 for t in tasks},
        }
    else:
        scores["H0"] = _eval_version(
            cfg=cfg,
            run_dir=run_dir,
            version_id="H0",
            modules_root=Path(h0.modules_root),
            tasks=tasks,
            reps=reps,
            prov=prov,
            temperature=temperature,
            locked_module=locked,
        )
        mark_step(run_dir, "eval_H0")
        for f in forks:
            vid = f["child_version_id"]
            st = load_state_modules(versions_root, vid)
            scores[vid] = _eval_version(
                cfg=cfg,
                run_dir=run_dir,
                version_id=vid,
                modules_root=st,
                tasks=tasks,
                reps=reps,
                prov=prov,
                temperature=temperature,
                locked_module=locked,
            )
        mark_step(run_dir, "eval_A1_B1")

    # Continuations (frozen policy: another independent marker)
    cont_versions: dict[str, str] = {}
    if continuations >= 1:
        for f in forks:
            label = f["label"]
            parent_child = f["child_version_id"]
            cont = continue_once(
                run_dir=run_dir,
                versions_root=versions_root,
                parent_child_version_id=parent_child,
                locked_module=locked,
                label=label,
                plan_text=f"P2B continuation {label}2 marker",
                patch_text=_comment_patch(
                    load_state_modules(versions_root, parent_child),
                    locked,
                    f"{label}2",
                ),
            )
            if not cont.get("child_version_id"):
                return EXIT_FAIL, {
                    "status": "FAIL",
                    "reason": f"continuation apply-failure {label}",
                    "run_dir": str(run_dir),
                }
            cont_versions[label] = cont["child_version_id"]
            if skip_eval:
                scores[cont["child_version_id"]] = {t: 0.0 for t in tasks}
            else:
                st = load_state_modules(versions_root, cont["child_version_id"])
                scores[cont["child_version_id"]] = _eval_version(
                    cfg=cfg,
                    run_dir=run_dir,
                    version_id=cont["child_version_id"],
                    modules_root=st,
                    tasks=tasks,
                    reps=reps,
                    prov=prov,
                    temperature=temperature,
                    locked_module=locked,
                )
        mark_step(run_dir, "continuations")

    branches = []
    for f in forks:
        label = f["label"]
        order = ["H0", f["child_version_id"]]
        if label in cont_versions:
            order.append(cont_versions[label])
        # transitions
        write_transition(
            run_dir,
            transition_id=f"H0_to_{f['child_version_id']}",
            parent_version_id="H0",
            child_version_id=f["child_version_id"],
            plan_id=f["plan"]["plan_id"],
            attempt_id=f["attempt"]["attempt_id"],
            parent_scores=scores.get("H0", {}),
            child_scores=scores.get(f["child_version_id"], {}),
        )
        lab = label_path_from_score_tables(
            run_dir,
            branch_id=f"branch_{label}",
            version_order=order,
            score_tables=scores,
            lambda_value=lambda_value,
        )
        branches.append(
            {
                "branch_id": f"branch_{label}",
                "version_order": order,
                "stored_return": lab.to_dict(),
            }
        )

    score_doc = {
        "lambda": lambda_value,
        "regression_primary": "parent-step",
        "scores": scores,
        "branches": branches,
        "skip_eval": skip_eval,
    }
    write_json(run_dir / "score_tables.json", score_doc)

    status = "INCOMPLETE" if skip_eval else "PASS"
    # If no rewards at all on H0, mark incomplete
    if not skip_eval and not scores.get("H0"):
        status = "INCOMPLETE"

    summary = {
        "status": status,
        "run_id": run_id,
        "tasks": tasks,
        "reps": reps,
        "LOCKED_MODULE": locked,
        "continuations": continuations,
        "provider_id": prov.provider_id,
        "temperature": temperature,
        "lambda": lambda_value,
        "resume": resume_marker(run_dir),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    code = {
        "PASS": EXIT_PASS,
        "FAIL": EXIT_FAIL,
        "INCOMPLETE": EXIT_INCOMPLETE,
    }[status]
    return code, summary


def load_state_modules(versions_root: Path, version_id: str) -> Path:
    from research1.versions import load_state

    return Path(load_state(versions_root, version_id).modules_root)
