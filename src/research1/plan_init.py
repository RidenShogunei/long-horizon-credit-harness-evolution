"""One parent-state plan sample: frozen checks, fixed editor, real task result.

Score movement is stored as observed_delta. It does not decide whether the
plan enters the supervised-init set. Long-horizon labels stay empty.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_PASS
from research1.causal import probe_parse_warning
from research1.config import REPO_ROOT, load_pilot, resolve_path
from research1.contracts import write_json
from research1.distill import (
    _apply_edit,
    _tongji_chat,
    build_edit_prompt,
    extract_file_body,
    parse_plan_output,
)
from research1.versions import sha256_file

SCHEMA = "research1.plan_init.v1"
NEEDLE = "should end with newline"


def frozen_false_concat_checks() -> list[dict[str, Any]]:
    """Checks fixed before a plan is requested.

    The empty wait must stop warning. A real missing newline must still warn.
    Well-formed commands must stay quiet.
    """
    return [
        {
            "id": "empty_wait_mid_batch",
            "role": "fix",
            "after": "absent",
            "needle": NEEDLE,
            "payload": _payload(
                [
                    {"keystrokes": "echo hi\n", "duration": 0.5},
                    {"keystrokes": "", "duration": 5.0},
                    {"keystrokes": "echo bye\n", "duration": 0.5},
                ]
            ),
        },
        {
            "id": "nonempty_missing_newline",
            "role": "preserve",
            "after": "present",
            "needle": NEEDLE,
            "payload": _payload(
                [
                    {"keystrokes": "echo hi", "duration": 0.5},
                    {"keystrokes": "echo bye\n", "duration": 0.5},
                ]
            ),
        },
        {
            "id": "well_formed_commands",
            "role": "preserve",
            "after": "absent",
            "needle": NEEDLE,
            "payload": _payload(
                [
                    {"keystrokes": "echo hi\n", "duration": 0.5},
                    {"keystrokes": "echo bye\n", "duration": 0.5},
                ]
            ),
        },
        {
            "id": "mixed_empty_and_real",
            "role": "preserve",
            "after": "present",
            "needle": "Command 2 should end with newline",
            "absent": ["Command 1 should end with newline"],
            "payload": _payload([
                {"keystrokes": "", "duration": 0.5},
                {"keystrokes": "echo hi", "duration": 0.5},
                {"keystrokes": "echo bye\n", "duration": 0.5},
            ]),
        },
        *[{
            "id": "nonempty_" + label,
            "role": "preserve",
            "after": "present",
            "needle": "Command 1 should end with newline",
            "payload": _payload([
                {"keystrokes": keystrokes, "duration": 0.5},
                {"keystrokes": "echo bye\n", "duration": 0.5},
            ]),
        } for label, keystrokes in (("space", " "), ("tab", "\t"))],
    ]


def _payload(commands: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "probe",
        "plan": "frozen check",
        "commands": commands,
        "task_complete": False,
    }


def probe_checks(modules_root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain the complete parser contract, rather than just a warning marker."""
    warnings: dict[str, Any] = {}
    for check in checks:
        probe = probe_parse_warning(modules_root, check["payload"])
        warnings[check["id"]] = probe
    return warnings


def judge_behavior(
    *,
    parent_warnings: dict[str, Any],
    child_warnings: dict[str, Any] | None,
    checks: list[dict[str, Any]],
    edit_applied: bool,
) -> dict[str, Any]:
    """Behavior gate. A score is not an input."""
    rows: list[dict[str, Any]] = []
    fix_ok = True
    preserve_ok = True
    parent_reproduced = bool(checks) and any(c.get("role") == "fix" for c in checks)
    complete_contract = True
    for check in checks:
        parent = parent_warnings.get(check["id"])
        child = (child_warnings or {}).get(check["id"])
        # Historical string probes can still be displayed, but cannot attest to
        # command preservation or completion/error semantics.
        before_text = str(parent.get("warning") or "") if isinstance(parent, dict) else str(parent or "")
        after_text = str(child.get("warning") or "") if isinstance(child, dict) else str(child or "")
        required = {"warning", "commands", "error", "is_task_complete"}
        contract_ok = bool(
            isinstance(parent, dict) and isinstance(child, dict)
            and required.issubset(parent) and required.issubset(child)
            and json.dumps(parent["commands"], sort_keys=True) == json.dumps(child["commands"], sort_keys=True)
            and parent["error"] == child["error"]
            and parent["is_task_complete"] == child["is_task_complete"]
        )
        complete_contract = complete_contract and contract_ok
        needle = str(check["needle"])
        before = needle in before_text
        after = None
        if child is not None:
            after = needle in after_text
        if check["role"] == "fix" and not before:
            parent_reproduced = False
        expect_present = check["after"] == "present"
        ok = None
        if after is not None:
            ok = (after if expect_present else (not after)) and contract_ok
            ok = ok and not any(text in after_text for text in check.get("absent", []))
            if check["role"] == "fix":
                ok = bool(before and not after and contract_ok)
                fix_ok = fix_ok and ok
            else:
                preserve_ok = preserve_ok and bool(ok)
        rows.append(
            {
                "id": check["id"],
                "role": check["role"],
                "before": before,
                "after": after,
                "ok": ok,
                "contract_preserved": contract_ok,
                "parent_result": parent,
                "child_result": child,
            }
        )
    harmful = bool(edit_applied and child_warnings is not None and not preserve_ok)
    sft_eligible = bool(
        edit_applied and parent_reproduced and child_warnings is not None
        and fix_ok and preserve_ok and complete_contract
    )
    return {
        "checks": rows,
        "parent_reproduced": parent_reproduced,
        "complete_contract": complete_contract,
        "fix_ok": fix_ok if child_warnings is not None else False,
        "preserve_ok": preserve_ok if child_warnings is not None else False,
        "harmful_behavior": harmful,
        "sft_eligible": sft_eligible,
    }


def sft_record_allowed(decision: dict[str, Any]) -> bool:
    """Promotion rule. observed_delta is ignored. Long-horizon labels stay empty."""
    behavior = decision.get("behavior") or {}
    task = decision.get("child_task_run") or {}
    if decision.get("action") != "edit":
        return False
    if not behavior.get("sft_eligible"):
        return False
    if behavior.get("harmful_behavior"):
        return False
    from research1.collection import valid_task_result
    if not valid_task_result(task):
        return False
    return True


def observed_delta(parent_reward: float | None, child_reward: float | None) -> float | None:
    if parent_reward is None or child_reward is None:
        return None
    return float(child_reward) - float(parent_reward)


def build_teacher_prompt(*, evidence: str, module_excerpt: str, module_sha: str) -> str:
    return (
        "You are the plan module. You see only the current locked module and the "
        "failure evidence below. Do not invent a defect to fill a quota.\n"
        "If the evidence does not support a change to this module, set ACTION to keep.\n"
        "Otherwise set ACTION to edit and describe the behavior change. "
        "Diagnose only from the evidence and code; state the affected behavior "
        "and the normal behavior that must be preserved. Do not infer a fix "
        "from a task score, and do not remove unrelated feedback.\n"
        "Reply in exactly this shape:\n"
        "ACTION: edit or keep\n"
        "TARGET: tools/baseline.py or none\n"
        "INTENT:\n"
        "<what is wrong and what the edit should change>\n"
        "TRIGGER: <short new diagnostic string, or NONE>\n"
        "\n"
        f"Module sha256: {module_sha}\n"
        "Module excerpt:\n"
        f"{module_excerpt}\n"
        "\n"
        "Failure evidence:\n"
        f"{evidence}\n"
    )


def parse_teacher_action(text: str) -> dict[str, Any]:
    parsed = parse_plan_output(text)
    action_m = re.search(r"^ACTION:\s*(edit|keep)\s*$", text, re.M | re.I)
    if not action_m:
        raise ValueError("plan output missing ACTION: edit|keep")
    action = action_m.group(1).lower()
    if action == "keep":
        parsed["target"] = "none"
    parsed["action"] = action
    return parsed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    from research1.collection import append_unique, _id
    row = dict(row)
    row.setdefault("record_id", _id("raw_", row))
    append_unique(path, row, "record_id")


def export_decision(root: Path, decision: dict[str, Any]) -> dict[str, str]:
    from research1.collection import sample_id_for_row, set_admission
    raw = root / "datasets" / "research1" / "raw"
    append_jsonl(raw / "decisions.jsonl", decision["decision_row"])
    append_jsonl(raw / "attempts.jsonl", decision["attempt_row"])
    append_jsonl(raw / "task_runs.jsonl", decision["task_row"])
    exported = {"sft": ""}
    sample = dict(decision["sft_row"])
    sample["sample_id"] = sample_id_for_row(sample)
    status = "pending"
    if sft_record_allowed(decision):
        sft_path = root / "datasets" / "research1" / "sft" / "verified_plans.jsonl"
        exported["sft"] = str(sft_path)
        status = "verified"
    elif decision["behavior"].get("harmful_behavior"):
        status = "rejected"
    set_admission(root, sample["sample_id"], status,
                  "legacy plan-init behavior and task admission check", sft_row=sample,
                  evidence={"run_id": sample.get("run_id"), "behavior": decision["behavior"]})
    return exported


def continue_plan_init(
    run_dir: Path,
    *,
    modules_root: Path,
    trajectory: Path,
    task_name: str,
    parent_reward: float | None,
    source_kind: str,
    run_task: bool,
    config_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Resume after a saved plan. Does not call the teacher again."""
    cfg = load_pilot(config_path)
    checks = json.loads((run_dir / "frozen_checks.json").read_text())["checks"]
    parsed = json.loads((run_dir / "plan.json").read_text())
    parent_warnings = json.loads((run_dir / "parent_probe.json").read_text())
    prompt = (run_dir / "teacher_prompt.txt").read_text()
    snapshot = modules_root / "tools" / "baseline.py"
    module_text = snapshot.read_text()
    record_path = run_dir / "plan_record.json"
    edit_prompt = (
        "Do not explain. The first line must be the begin sentinel and the last line "
        "must be the end sentinel. Return the complete file.\n"
        + build_edit_prompt(plan=parsed, source=module_text)
    )
    edit_raw, _emeta = _tongji_chat(edit_prompt, max_tokens=32000, role="plan_init_editor_retry")
    (run_dir / "editor_raw.txt").write_text(edit_raw)
    replacement = extract_file_body(edit_raw)
    linkage = _apply_edit(
        run_dir=run_dir,
        record_path=record_path,
        plan=parsed,
        snapshot=snapshot,
        replacement=replacement,
    )
    edit_applied = linkage.get("status") == "applied"
    child_id = linkage.get("child_version_id")
    child_warnings = None
    if edit_applied and child_id:
        child_warnings = probe_checks(run_dir / "versions" / child_id / "modules", checks)
        write_json(run_dir / "child_probe.json", child_warnings)
    behavior = judge_behavior(
        parent_warnings=parent_warnings,
        child_warnings=child_warnings,
        checks=checks,
        edit_applied=edit_applied,
    )
    child_task: dict[str, Any] = {"status": "INCOMPLETE", "reward": None, "task_name": task_name}
    if run_task and edit_applied and child_id and behavior.get("sft_eligible"):
        child_task = _run_child_task(
            cfg=cfg,
            run_dir=run_dir,
            modules_root=run_dir / "versions" / str(child_id) / "modules",
            task_name=task_name,
            run_id=run_dir.name,
            version_id=str(child_id),
        )
    behavior = judge_behavior(
        parent_warnings=parent_warnings,
        child_warnings=child_warnings,
        checks=checks,
        edit_applied=edit_applied,
    )
    delta = observed_delta(parent_reward, child_task.get("reward"))
    decision = _pack_decision(
        run_id=run_dir.name,
        task_name=task_name,
        source_kind=source_kind,
        trajectory=trajectory,
        snapshot=snapshot,
        modules_root=modules_root,
        parent_reward=parent_reward,
        prompt=prompt,
        parsed=parsed,
        edit_applied=edit_applied,
        child_id=child_id,
        reject_reason=str(linkage.get("reject_reason") or ""),
        diff_path=linkage.get("diff_path"),
        behavior=behavior,
        child_task=child_task,
        delta=delta,
    )
    # Behavior-valid edits still need a finished task run before SFT export.
    exported = export_decision(REPO_ROOT, decision)
    decision["exported"] = exported
    write_json(run_dir / "decision.json", decision)
    _write_audit(resolve_path(cfg, "reports_root") / "distillation_audit.md", decision)
    return (EXIT_PASS if sft_record_allowed(decision) else EXIT_FAIL), decision


def run_plan_init_sample(
    *,
    trajectory: Path,
    modules_root: Path,
    task_name: str,
    parent_reward: float | None,
    source_kind: str,
    run_task: bool,
    config_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Collect one recoverable parent state through plan, edit, checks, and export."""
    cfg = load_pilot(config_path)
    run_id = time.strftime("plan_init_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = resolve_path(cfg, "runs_root") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot = modules_root / "tools" / "baseline.py"
    evidence_path = trajectory
    from research1.causal import false_concat_findings
    from research1.distill import relevant_component_excerpt

    findings = false_concat_findings(evidence_path)
    checks = frozen_false_concat_checks() if findings else []
    # The legacy single-sample command can diagnose other evidence, but only a
    # prepared executable contract can promote an edit. The general collector
    # accepts external frozen verification bundles for additional mechanisms.
    write_json(run_dir / "frozen_checks.json", {"schema": SCHEMA, "checks": checks})
    evidence = str(findings[0].get("excerpt") or "")[:4000] if findings else evidence_path.read_text()[:12000]
    module_text = snapshot.read_text()
    parent_warnings = probe_checks(modules_root, checks)
    write_json(run_dir / "parent_probe.json", parent_warnings)

    prompt = build_teacher_prompt(
        evidence=evidence,
        module_excerpt=relevant_component_excerpt(module_text),
        module_sha=sha256_file(snapshot),
    )
    (run_dir / "teacher_prompt.txt").write_text(prompt)
    raw, _meta = _tongji_chat(prompt, max_tokens=4000, role="plan_init_teacher")
    (run_dir / "teacher_raw.txt").write_text(raw)
    parsed = parse_teacher_action(raw)
    write_json(run_dir / "plan.json", parsed)

    edit_applied = False
    child_id = None
    reject_reason = ""
    diff_path = None
    child_warnings = None
    if parsed["action"] == "edit" and parsed.get("target") not in {None, "none"}:
        record_path = run_dir / "plan_record.json"
        write_json(record_path, {"plan": parsed})
        edit_prompt = build_edit_prompt(plan=parsed, source=module_text)
        edit_raw, _emeta = _tongji_chat(edit_prompt, max_tokens=16000, role="plan_init_editor")
        (run_dir / "editor_raw.txt").write_text(edit_raw)
        replacement = extract_file_body(edit_raw)
        linkage = _apply_edit(
            run_dir=run_dir,
            record_path=record_path,
            plan=parsed,
            snapshot=snapshot,
            replacement=replacement,
        )
        edit_applied = linkage.get("status") == "applied"
        child_id = linkage.get("child_version_id")
        reject_reason = str(linkage.get("reject_reason") or "")
        diff_path = linkage.get("diff_path")
        if edit_applied and child_id:
            child_root = run_dir / "versions" / child_id / "modules"
            child_warnings = probe_checks(child_root, checks)
            write_json(run_dir / "child_probe.json", child_warnings)

    behavior = judge_behavior(
        parent_warnings=parent_warnings,
        child_warnings=child_warnings,
        checks=checks,
        edit_applied=edit_applied,
    )
    child_task: dict[str, Any] = {"status": "INCOMPLETE", "reward": None, "task_name": task_name}
    if run_task and edit_applied and child_id and behavior.get("sft_eligible"):
        child_task = _run_child_task(
            cfg=cfg,
            run_dir=run_dir,
            modules_root=run_dir / "versions" / child_id / "modules",
            task_name=task_name,
            run_id=run_id,
            version_id=child_id,
        )
    elif not run_task:
        child_task = {"status": "SKIPPED", "reward": None, "task_name": task_name}

    delta = observed_delta(parent_reward, child_task.get("reward"))
    decision = _pack_decision(
        run_id=run_id,
        task_name=task_name,
        source_kind=source_kind,
        trajectory=trajectory,
        snapshot=snapshot,
        modules_root=modules_root,
        parent_reward=parent_reward,
        prompt=prompt,
        parsed=parsed,
        edit_applied=edit_applied,
        child_id=child_id,
        reject_reason=reject_reason,
        diff_path=diff_path,
        behavior=behavior,
        child_task=child_task,
        delta=delta,
    )
    exported = export_decision(REPO_ROOT, decision)
    decision["exported"] = exported
    write_json(run_dir / "decision.json", decision)
    _write_audit(resolve_path(cfg, "reports_root") / "distillation_audit.md", decision)
    code = EXIT_PASS if sft_record_allowed(decision) else EXIT_FAIL
    return code, decision


def _run_child_task(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    modules_root: Path,
    task_name: str,
    run_id: str,
    version_id: str,
) -> dict[str, Any]:
    from research1.adapters import apply_provider_to_environ, run_harbor_task, with_llm_provider_env
    from research1.evaluation import summarize_trial
    from research1.versions import BASELINE_BUNDLE

    prov = with_llm_provider_env("tongji_agi")
    apply_provider_to_environ(prov)
    trial_dir = (run_dir / "evals" / version_id / task_name / "rep_2").resolve()
    proc = run_harbor_task(
        modularrsi_root=resolve_path(cfg, "modularrsi_root"),
        modules_root=modules_root.resolve(),
        task_name=task_name,
        output_dir=trial_dir,
        model=prov.model,
        api_base=prov.api_base,
        temperature=float(cfg.get("p2b_solver_temperature", 0.7)),
        dataset_path=resolve_path(cfg, "dataset_tb"),
        locked_module_type="tools",
        composer_name="static",
    )
    (trial_dir / "harbor.stdout").write_text(proc.stdout or "")
    (trial_dir / "harbor.stderr").write_text(proc.stderr or "")
    summary = summarize_trial(
        run_id=run_id,
        version_id=version_id,
        task_name=task_name,
        trial_index=1,
        output_dir=trial_dir,
        active_bundle=BASELINE_BUNDLE,
        model=prov.model,
        temperature=float(cfg.get("p2b_solver_temperature", 0.7)),
        provider_id=prov.provider_id,
    )
    write_json(trial_dir / "task_run.json", summary.to_dict())
    return summary.to_dict()


def _pack_decision(**kwargs: Any) -> dict[str, Any]:
    parsed = kwargs["parsed"]
    behavior = kwargs["behavior"]
    child_task = kwargs["child_task"]
    decision_row = {
        "schema": SCHEMA,
        "run_id": kwargs["run_id"],
        "source_kind": kwargs["source_kind"],
        "task_name": kwargs["task_name"],
        "trajectory": str(kwargs["trajectory"]),
        "module_sha256": sha256_file(kwargs["snapshot"]),
        "modules_root": str(kwargs["modules_root"]),
        "parent_reward": kwargs["parent_reward"],
        "action": parsed.get("action"),
        "plan": parsed,
        "teacher_prompt": kwargs["prompt"],
        "long_horizon_label": None,
    }
    attempt_row = {
        "run_id": kwargs["run_id"],
        "edit_applied": kwargs["edit_applied"],
        "child_version_id": kwargs["child_id"],
        "reject_reason": kwargs["reject_reason"],
        "diff_path": kwargs["diff_path"],
        "behavior": behavior,
        "observed_delta": kwargs["delta"],
        "harmful_behavior": behavior.get("harmful_behavior"),
    }
    task_row = {
        "run_id": kwargs["run_id"],
        "parent_reward": kwargs["parent_reward"],
        "child": child_task,
        "observed_delta": kwargs["delta"],
        "note": "observed_delta is not an SFT gate",
    }
    sft_row = {
        "run_id": kwargs["run_id"],
        "source_kind": kwargs["source_kind"],
        "task_name": kwargs["task_name"],
        "prompt": kwargs["prompt"],
        "completion": json.dumps(parsed, ensure_ascii=False),
        "use": "plan_supervised_init",
        "long_horizon_label": None,
    }
    return {
        "action": parsed.get("action"),
        "behavior": behavior,
        "child_task_run": child_task,
        "observed_delta": kwargs["delta"],
        "decision_row": decision_row,
        "attempt_row": attempt_row,
        "task_row": task_row,
        "sft_row": sft_row,
    }


def _write_audit(path: Path, decision: dict[str, Any]) -> None:
    behavior = decision["behavior"]
    lines = [
        "# Distillation audit",
        "",
        f"- Run: `{decision['decision_row']['run_id']}`",
        f"- Source: `{decision['decision_row']['source_kind']}`",
        f"- Task: `{decision['decision_row']['task_name']}`",
        f"- Parent reward: `{decision['decision_row']['parent_reward']}`",
        f"- Action: `{decision['action']}`",
        f"- Edit applied: `{decision['attempt_row']['edit_applied']}`",
        f"- Behavior fix: `{behavior.get('fix_ok')}`",
        f"- Preservation: `{behavior.get('preserve_ok')}`",
        f"- Harmful: `{behavior.get('harmful_behavior')}`",
        f"- Child task status: `{decision['child_task_run'].get('status')}`",
        f"- Observed delta: `{decision.get('observed_delta')}`",
        f"- SFT exported: `{sft_record_allowed(decision)}`",
        "- Long-horizon label: empty",
        "",
        "Existing warning-cleanup rows were not expanded.",
        "Score movement is recorded and is not the admission rule.",
        "",
        "## Same plan, two child executions",
        "",
        "- `rep_1` is `incomplete`. Harbor raised `FileNotFoundError` because `modules_root` was resolved in the wrong directory. The verifier result is null. This is not a task score of 0.",
        "- `rep_2` finished normally. Reward is 1.0. It is the valid child execution for this plan.",
        "- The parent reward 0.0 is the earlier failed trial. The observed 0 to 1 difference does not show that the patch caused the task to succeed.",
        "- Whether the false warning caused that original failure is unknown.",
        "- Long-horizon label remains empty.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


COVERAGE_TASKS = (
    "chebyshev-poisson-task",
    "c-escapes-decode",
    "cron-schedule-conflict-resolver",
    "dep-conflict",
)


def coverage_skip_reason(dockerfile: str, local_images: set[str]) -> str:
    """Empty string means the task image can be built here.

    Private registry bases time out on this machine unless the image is already
    local. Heavy source builds are the same class as the ffmpeg abort.
    """
    from research1.causal import dockerfile_from, is_heavy_source_build, is_private_base_image

    if is_heavy_source_build(dockerfile):
        return "heavy source build"
    image = dockerfile_from(dockerfile)
    if is_private_base_image(image) and image not in local_images:
        return "private base image is not local"
    return ""


def classify_coverage(trial_dir: Path, reward: float | None) -> dict[str, Any]:
    """Record a parent state. Another false-concat slice is not a new mechanism."""
    from research1.causal import false_concat_findings, scan_trajectory, sufficient_findings

    trajectories = list(trial_dir.rglob("trajectory.json"))
    kinds: list[str] = []
    for traj in trajectories:
        for finding in false_concat_findings(traj):
            kinds.append(str(finding.get("kind")))
        for finding in sufficient_findings(scan_trajectory(traj)):
            kinds.append(str(finding.get("kind")))
    unique = sorted(set(kinds))
    if reward is None:
        action = "incomplete"
        reason = "no valid task result"
    elif not unique:
        action = "keep"
        reason = "task result has no tools-module evidence"
    elif unique == ["false_concat_warning_on_empty_keystrokes"]:
        action = "keep"
        reason = "same false-concat mechanism; not a new experience class"
    else:
        action = "evidence"
        reason = "module evidence is present and no frozen check was prepared for it"
    return {
        "action": action,
        "reason": reason,
        "mechanisms": unique,
        "trajectory_count": len(trajectories),
    }


def run_coverage_parents(
    config_path: Path | None = None,
    *,
    tasks: list[str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run one current-version trial per task and store the parent-state decision."""
    from research1.adapters import apply_provider_to_environ, run_harbor_task, with_llm_provider_env
    from research1.evaluation import summarize_trial
    from research1.versions import BASELINE_BUNDLE

    cfg = load_pilot(config_path)
    chosen = list(tasks or COVERAGE_TASKS)
    run_id = time.strftime("parent_cover_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = resolve_path(cfg, "runs_root") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    modules = resolve_path(cfg, "modules_root").resolve()
    prov = with_llm_provider_env("tongji_agi")
    apply_provider_to_environ(prov)
    rows: list[dict[str, Any]] = []
    print(f"[parent-cover] tasks={chosen}", flush=True)
    dataset = resolve_path(cfg, "dataset_tb")
    local_images = set()
    try:
        import subprocess

        listed = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        local_images = {line.strip() for line in (listed.stdout or "").splitlines() if line.strip()}
    except OSError:
        local_images = set()
    for task in chosen:
        print(f"[parent-cover] {task}", flush=True)
        dockerfile = dataset / "tasks" / task / "environment" / "Dockerfile"
        skip = coverage_skip_reason(dockerfile.read_text() if dockerfile.is_file() else "", local_images)
        if skip:
            row = {
                "run_id": run_id,
                "task_name": task,
                "reward": None,
                "status": "INCOMPLETE",
                "failure_layer": "infrastructure",
                "modules_root": str(modules),
                "output_dir": "",
                "action": "incomplete",
                "reason": skip,
                "mechanisms": [],
                "trajectory_count": 0,
                "sft_exported": False,
                "long_horizon_label": None,
            }
            rows.append(row)
            append_jsonl(REPO_ROOT / "datasets" / "research1" / "raw" / "parent_states.jsonl", row)
            print(f"[parent-cover] {task} skipped: {skip}", flush=True)
            continue
        trial_dir = (run_dir / "evals" / "H0" / task / "rep_1").resolve()
        proc = run_harbor_task(
            modularrsi_root=resolve_path(cfg, "modularrsi_root"),
            modules_root=modules,
            task_name=task,
            output_dir=trial_dir,
            model=prov.model,
            api_base=prov.api_base,
            temperature=float(cfg.get("p2b_solver_temperature", 0.7)),
            dataset_path=resolve_path(cfg, "dataset_tb"),
            locked_module_type="tools",
            composer_name="static",
        )
        (trial_dir / "harbor.stdout").write_text(proc.stdout or "")
        (trial_dir / "harbor.stderr").write_text(proc.stderr or "")
        summary = summarize_trial(
            run_id=run_id,
            version_id="H0",
            task_name=task,
            trial_index=1,
            output_dir=trial_dir,
            active_bundle=BASELINE_BUNDLE,
            model=prov.model,
            temperature=float(cfg.get("p2b_solver_temperature", 0.7)),
            provider_id=prov.provider_id,
        )
        write_json(trial_dir / "task_run.json", summary.to_dict())
        classified = classify_coverage(trial_dir, summary.reward)
        row = {
            "run_id": run_id,
            "task_name": task,
            "reward": summary.reward,
            "status": summary.status,
            "failure_layer": summary.failure_layer,
            "modules_root": str(modules),
            "output_dir": str(trial_dir),
            **classified,
            "sft_exported": False,
            "long_horizon_label": None,
        }
        rows.append(row)
        append_jsonl(REPO_ROOT / "datasets" / "research1" / "raw" / "parent_states.jsonl", row)
        print(f"[parent-cover] {task} reward={summary.reward} action={row['action']}", flush=True)
    mechanisms = sorted({m for row in rows for m in row["mechanisms"]})
    summary_doc = {
        "run_id": run_id,
        "parent_states": len(rows),
        "mechanisms": mechanisms,
        "mechanism_count": len(mechanisms),
        "distinct_patches": 0,
        "verified_passes": 0,
        "rows": rows,
    }
    write_json(run_dir / "summary.json", summary_doc)
    report = resolve_path(cfg, "reports_root") / "parent_coverage.md"
    lines = [
        "# Parent-state coverage",
        "",
        f"- Run: `{run_id}`",
        f"- Parent states: {len(rows)}",
        f"- Mechanisms: {mechanisms or 'none'}",
        f"- Distinct patches: 0",
        f"- Verified plan passes: 0",
        "",
        "No second false-concat plan was exported.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['task_name']}` reward={row['reward']} action={row['action']}: {row['reason']}"
        )
    lines.append("")
    report.write_text("\n".join(lines))
    return EXIT_PASS, summary_doc
