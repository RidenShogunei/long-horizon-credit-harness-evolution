"""Track B: failure evidence → frozen plan → fixed editor → H1 trigger.

The planner is a frozen chat call (not a trained evolver). The editor does not
call a model: it diffs the planner's replacement file, rejects comment-only
edits, and applies the stored unified diff.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS
from research1.adapters import (
    apply_provider_to_environ,
    merge_no_proxy,
    run_harbor_task,
    save_process_logs,
    with_llm_provider_env,
)
from research1.config import REPO_ROOT, load_json, load_pilot, resolve_path, train_task_names
from research1.contracts import write_json
from research1.evaluation import aggregate_task_means, summarize_trial
from research1.planning import apply_plan_patch, make_plan
from research1.preflight import run_preflight
from research1.rollout import ensure_h0_version, mark_step, write_transition
from research1.versions import BASELINE_BUNDLE, hash_tree, load_state

FILE_BEGIN = "<<<FILE tools/baseline.py>>>"
FILE_END = "<<<END tools/baseline.py>>>"
LOCKED = "tools"
TARGET_REL = "tools/baseline.py"


def score_direction(delta: float | None) -> str:
    """Mean direction only. Equal means are not a behavioral claim."""
    if delta is None:
        return "unknown"
    if delta > 1e-9:
        return "improved"
    if delta < -1e-9:
        return "regressed"
    return "no_score_change_measured"


def score_direction_label(direction: str) -> str:
    return {
        "improved": "improved",
        "regressed": "regressed",
        "no_score_change_measured": "no score change measured",
        "unknown": "unknown",
    }.get(direction, direction)


def outcome_for(*, applied: bool, triggered: bool, direction: str) -> str:
    """Map measurement onto the existing outcome vocabulary without lying.

    ``no-effect`` is used only when the new logic was not observed.
    A triggered edit with equal means is not labeled no-effect; the score
    direction is stored separately as ``no_score_change_measured``.
    """
    if not applied:
        return "apply-failure"
    if not triggered:
        return "no-effect"
    if direction == "regressed":
        return "regression"
    if direction == "improved":
        return "success"
    # Triggered, mean unchanged: behavioral effect happened.
    return "success"


def dockerfile_from(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if not s.upper().startswith("FROM "):
            continue
        for part in s.split()[1:]:
            if not part.startswith("--"):
                return part
    return ""


def is_heavy_source_build(dockerfile: str) -> bool:
    """Same class as the recorded ffmpeg abort: image builds from source."""
    return "build-essential" in dockerfile


def is_private_base_image(image: str) -> bool:
    """Registry host in the image name. Official `name:tag` refs are public."""
    image = (image or "").strip()
    if "/" not in image:
        return False
    return "." in image.split("/", 1)[0]


def _strip_docstrings(node: ast.AST) -> None:
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    for child in ast.iter_child_nodes(node):
        _strip_docstrings(child)


def functional_ast_changed(before: str, after: str) -> tuple[bool, str]:
    try:
        old = ast.parse(before)
        new = ast.parse(after)
    except SyntaxError as exc:
        return False, f"syntax-error: {exc}"
    _strip_docstrings(old)
    _strip_docstrings(new)
    if ast.dump(old) == ast.dump(new):
        return False, "comment-or-docstring-only"
    return True, ""


def new_string_constants(before: str, after: str) -> set[str]:
    def consts(src: str) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                out.add(node.value)
        return out

    return consts(after) - consts(before)


def unified_diff_for(rel: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        n=3,
    )
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def parse_planner_output(text: str) -> dict[str, str]:
    plan = ""
    if "PLAN:" in text:
        plan = text.split("PLAN:", 1)[1]
        plan = plan.split("TRIGGER:", 1)[0].strip()
    trig_m = re.search(r"^TRIGGER:\s*(.+)$", text, re.M)
    if not trig_m:
        raise ValueError("planner output missing TRIGGER: line")
    trigger = trig_m.group(1).strip()
    if FILE_BEGIN not in text or FILE_END not in text:
        raise ValueError("planner output missing file sentinels")
    body = text.split(FILE_BEGIN, 1)[1].split(FILE_END, 1)[0]
    if body.startswith("\n"):
        body = body[1:]
    if not body.endswith("\n"):
        body += "\n"
    return {"plan_text": plan, "trigger": trigger, "file_text": body}


def editor_from_replacement(
    *,
    before: str,
    replacement: str,
    trigger: str,
) -> tuple[str | None, str]:
    """Fixed editor: diff, reject non-functional edits, return unified diff."""
    changed, reason = functional_ast_changed(before, replacement)
    if not changed:
        return None, reason or "no functional change"
    for anchor in ("class BaselineTools", "def parse_llm_response", "def execute", "def register"):
        if anchor not in replacement:
            return None, f"replacement dropped required anchor: {anchor}"
    fresh = new_string_constants(before, replacement)
    if not any(trigger in s for s in fresh):
        return None, "trigger text is not a new string constant in the replacement"
    if trigger.startswith("#") or "research1-p2b-marker" in trigger or "research1-p2b-marker" in replacement:
        return None, "comment marker is not a functional edit"
    diff = unified_diff_for(TARGET_REL, before, replacement)
    if not diff.strip():
        return None, "empty diff"
    return diff, ""


def select_expansion_tasks(
    train_names: list[str],
    *,
    sizes: dict[str, int],
    heavy: set[str],
    covered: set[str],
    private: set[str],
    window_n: int = 8,
    limit: int = 3,
) -> list[str]:
    """Probe order inside the fixed manifest window, then manifest order."""
    window = list(train_names[:window_n])
    ranked = sorted(window, key=lambda n: sizes.get(n, 10**12))
    chosen: list[str] = []

    def take(name: str) -> None:
        if name in heavy or name in covered or name in private:
            return
        if name not in chosen:
            chosen.append(name)

    for name in ranked:
        take(name)
        if len(chosen) >= limit:
            return chosen
    for name in train_names[window_n:]:
        take(name)
        if len(chosen) >= limit:
            break
    return chosen


def _mean(vals: list[float | None]) -> float | None:
    nums = [float(v) for v in vals if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _task_in_path(path: Path, names: list[str]) -> str | None:
    parts = list(path.parts)
    joined = set(parts)
    for name in names:
        if name in joined:
            return name
        suffix = "_" + name
        if any(p.endswith(suffix) for p in parts):
            return name
    return None


def tasks_with_agent_traces(runs_root: Path, train_names: list[str]) -> set[str]:
    found: set[str] = set()
    if not runs_root.is_dir():
        return found
    for traj in runs_root.rglob("trajectory.json"):
        try:
            data = json.loads(traj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        steps = data.get("steps") or []
        if not any(s.get("source") == "agent" for s in steps):
            continue
        name = _task_in_path(traj, train_names)
        if name:
            found.add(name)
    return found


def _obs_text(step: dict[str, Any]) -> str:
    obs = step.get("observation") or {}
    if isinstance(obs, dict):
        results = obs.get("results") or []
        if results and isinstance(results[0], dict):
            return str(results[0].get("content") or "")
    if isinstance(obs, str):
        return obs
    return ""


def scan_trajectory(path: Path) -> list[dict[str, Any]]:
    """Tools-module signals. Score flips are not findings."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [{"kind": "unreadable", "sufficient": False, "detail": str(exc), "path": str(path)}]
    findings: list[dict[str, Any]] = []
    extra = (data.get("agent") or {}).get("extra") or {}
    trace = extra.get("module_trace") or []
    trace_blob = json.dumps(trace, ensure_ascii=False)
    if "PARSE-ERROR" in trace_blob or "failure_tag=parse_error" in trace_blob:
        findings.append(
            {
                "kind": "parse_error",
                "sufficient": True,
                "detail": "module trace recorded PARSE-ERROR",
                "excerpt": trace_blob[:1500],
                "path": str(path),
            }
        )
    steps = data.get("steps") or []
    for i, step in enumerate(steps):
        obs = _obs_text(step)
        if "Previous response had parsing errors" in obs or "ERROR:" in obs and "parsing errors" in obs:
            findings.append(
                {
                    "kind": "parse_error",
                    "sufficient": True,
                    "detail": f"step {step.get('step_id')} received parse-error feedback",
                    "excerpt": obs[:1500],
                    "path": str(path),
                    "step_id": step.get("step_id"),
                }
            )
        if "should end with newline" in obs and "concatenated" in obs:
            findings.append(
                {
                    "kind": "newline_warning_on_wait",
                    "sufficient": False,
                    "detail": (
                        "parser warned that a following command would be concatenated; "
                        "the warning is already shown to the agent and empty-keystroke "
                        "waits still elapsed. Not treated as a defect to patch."
                    ),
                    "excerpt": obs[:800],
                    "path": str(path),
                    "step_id": step.get("step_id"),
                }
            )
    for ev in trace:
        if not isinstance(ev, dict):
            continue
        summary = str(ev.get("summary") or "")
        if ev.get("call") == "execute" and "FAIL:" in summary:
            # Next observations should contain the error text if feedback works.
            err = summary.split("FAIL:", 1)[-1].strip()
            blob = json.dumps(steps, ensure_ascii=False)
            if err and err not in blob and "[tool error]" not in blob:
                findings.append(
                    {
                        "kind": "execute_error_not_surfaced",
                        "sufficient": True,
                        "detail": "execute FAIL text absent from trajectory steps",
                        "excerpt": summary[:800],
                        "path": str(path),
                    }
                )
    return findings


def _max_duration(text: str) -> float | None:
    vals = [float(x) for x in re.findall(r'"duration"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text or "")]
    return max(vals) if vals else None


def scan_saved_responses(trial_agent_dir: Path) -> list[dict[str, Any]]:
    """Duration asks above 60s. Sufficiency is decided by silent_duration_caps."""
    findings: list[dict[str, Any]] = []
    if not trial_agent_dir.is_dir():
        return findings
    for resp in sorted(trial_agent_dir.glob("episode-*/response.txt")):
        text = resp.read_text(errors="replace")
        asked = _max_duration(text)
        if asked is None or asked <= 60.0:
            continue
        findings.append(
            {
                "kind": "duration_above_cap_requested",
                "sufficient": False,
                "detail": f"{resp.parent.name} requested duration {asked}",
                "excerpt": text[:800],
                "path": str(resp),
            }
        )
    return findings


def silent_duration_caps(agent_dir: Path) -> list[dict[str, Any]]:
    """Model asked for >60s and the executed call was clamped with no feedback."""
    traj_path = agent_dir / "trajectory.json"
    if not traj_path.is_file():
        return []
    try:
        data = json.loads(traj_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    steps = data.get("steps") or []
    findings: list[dict[str, Any]] = []
    for resp in sorted(agent_dir.glob("episode-*/response.txt")):
        text = resp.read_text(errors="replace")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        commands = payload.get("commands") if isinstance(payload, dict) else None
        if not isinstance(commands, list):
            continue
        for cmd in commands:
            if not isinstance(cmd, dict):
                continue
            try:
                asked = float(cmd.get("duration"))
            except (TypeError, ValueError):
                continue
            if asked <= 60.0:
                continue
            ks = cmd.get("keystrokes")
            for step in steps:
                obs = _obs_text(step)
                for tc in step.get("tool_calls") or []:
                    args = tc.get("arguments") or {}
                    if args.get("keystrokes") != ks:
                        continue
                    executed = args.get("duration")
                    try:
                        executed_f = float(executed)
                    except (TypeError, ValueError):
                        continue
                    if executed_f < asked - 1e-6 and "capped" not in obs.lower() and str(asked) not in obs:
                        findings.append(
                            {
                                "kind": "silent_duration_cap",
                                "sufficient": True,
                                "detail": (
                                    f"{resp.parent.name} asked duration {asked} "
                                    f"but the tool call stored {executed_f} and the "
                                    "observation did not say the wait was shortened"
                                ),
                                "excerpt": text[:1000],
                                "path": str(traj_path),
                                "step_id": step.get("step_id"),
                            }
                        )
    return findings


def replay_parser_errors(runs_root: Path) -> list[dict[str, Any]]:
    """Run the upstream JSON parser on saved responses. Errors are sufficient."""
    import subprocess

    py = Path("/home/jinxu/arms_cache/modularrsi/ModularRSI/.venv/bin/python")
    if not py.is_file() or not runs_root.is_dir():
        return []
    script = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/jinxu/arms_cache/modularrsi/ModularRSI/src")
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
parser = TerminusJSONPlainParser()
root = Path(sys.argv[1])
out = []
for resp in root.rglob("episode-*/response.txt"):
    if "132345" in resp.parts:
        continue
    text = resp.read_text(errors="replace")
    try:
        result = parser.parse_response(text)
    except Exception as exc:
        out.append({"path": str(resp), "error": f"parser raised {exc}"})
        continue
    if result.error:
        out.append({"path": str(resp), "error": result.error[:800], "warning": (result.warning or "")[:400]})
print(json.dumps(out))
"""
    proc = subprocess.run(
        [str(py), "-c", script, str(runs_root)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        return [
            {
                "kind": "parser_replay_failed",
                "sufficient": False,
                "detail": (proc.stderr or proc.stdout or "replay failed")[-500:],
                "path": str(runs_root),
            }
        ]
    raw = (proc.stdout or "").strip().splitlines()
    line = raw[-1] if raw else "[]"
    try:
        rows = json.loads(line)
    except json.JSONDecodeError:
        return []
    return [
        {
            "kind": "parse_error",
            "sufficient": True,
            "detail": row.get("error"),
            "excerpt": row.get("error"),
            "path": row.get("path"),
        }
        for row in rows
    ]


def false_concat_findings(traj_path: Path) -> list[dict[str, Any]]:
    """Newline/concatenation warning on an empty-keystroke wait.

    ``""`` does not end with a newline, so the parser warns that the next
    command will be glued onto this one. Empty keystrokes type nothing; the
    duration is only a wait. When the warned command is empty and the
    observation still shows a shell prompt, the warning is false feedback
    forwarded by tools/baseline.py.
    """
    try:
        data = json.loads(traj_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    findings: list[dict[str, Any]] = []
    for step in data.get("steps") or []:
        if step.get("source") != "agent":
            continue
        content = _obs_text(step)
        warned = [int(n) for n in re.findall(r"Command (\d+) should end with newline", content)]
        if not warned:
            continue
        calls = step.get("tool_calls") or []
        for number in warned:
            if number < 1 or number > len(calls):
                continue
            args = (calls[number - 1].get("arguments") or {})
            if args.get("keystrokes") != "":
                continue
            if "root@" not in content and "New Terminal Output" not in content:
                continue
            call_blob = json.dumps(calls, ensure_ascii=False)[:2000]
            findings.append(
                {
                    "kind": "false_concat_warning_on_empty_keystrokes",
                    "sufficient": True,
                    "detail": (
                        f"step {step.get('step_id')}: tools forwarded a warning that "
                        f"command {number} would be concatenated because it does not end "
                        "with a newline, but that command's keystrokes are empty "
                        f"(duration={args.get('duration')}, a wait). The observation still "
                        "shows terminal output after the batch, so this is false feedback."
                    ),
                    "excerpt": content[:1800] + "\n\nTOOL_CALLS:\n" + call_blob,
                    "path": str(traj_path),
                    "step_id": step.get("step_id"),
                }
            )
    return findings


def scan_runs_for_tools(runs_root: Path) -> dict[str, Any]:
    trajs = list(runs_root.rglob("trajectory.json")) if runs_root.is_dir() else []
    findings: list[dict[str, Any]] = []
    for traj in trajs:
        if "132345" in traj.parts:
            continue  # proxy 403, no tools interaction
        findings.extend(scan_trajectory(traj))
        findings.extend(scan_saved_responses(traj.parent))
        findings.extend(silent_duration_caps(traj.parent))
    findings.extend(replay_parser_errors(runs_root))
    sufficient = [f for f in findings if f.get("sufficient")]
    return {
        "trajectories": len(trajs),
        "findings": findings,
        "sufficient": sufficient,
        "sufficient_count": len(sufficient),
        "parser_error_count": sum(1 for f in findings if f.get("kind") == "parse_error" and f.get("sufficient")),
        "insufficient_newline_warnings": sum(1 for f in findings if f.get("kind") == "newline_warning_on_wait"),
    }


def _classify_task_image(task_dir: Path) -> dict[str, Any]:
    docker = task_dir / "environment" / "Dockerfile"
    text = docker.read_text() if docker.is_file() else ""
    image = dockerfile_from(text)
    return {
        "from": image,
        "heavy_source_build": is_heavy_source_build(text),
        "private_base": is_private_base_image(image),
    }


def _local_images() -> set[str]:
    import subprocess

    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    return {ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()}


def build_task_selection(cfg: dict[str, Any], *, limit: int) -> dict[str, Any]:
    ds = resolve_path(cfg, "dataset_tb")
    train = train_task_names(ds)
    manifest = json.loads((ds / "manifest.json").read_text())
    sizes = {row["name"]: int(row.get("bytes") or 0) for row in manifest.get("tasks") or []}
    lock_path = resolve_path(cfg, "tasks_lock")
    heavy: set[str] = set()
    if lock_path.is_file():
        exclusions = load_json(lock_path).get("exclusions") or {}
        for name, reason in exclusions.items():
            if "heavy" in str(reason) or "docker_build" in str(reason):
                heavy.add(name)
    private: set[str] = set()
    image_notes: dict[str, Any] = {}
    local = _local_images()
    window = train[: int(cfg.get("calibrate_pool_n", 8))]
    for name in train:
        task_dir = ds / "tasks" / name
        if not task_dir.is_dir():
            continue
        info = _classify_task_image(task_dir)
        if name in window or info["heavy_source_build"] or info["private_base"]:
            image_notes[name] = info
        if info["heavy_source_build"]:
            heavy.add(name)
        if info["private_base"] and info["from"] not in local:
            private.add(name)
    covered = tasks_with_agent_traces(resolve_path(cfg, "runs_root"), train)
    chosen = select_expansion_tasks(
        train,
        sizes=sizes,
        heavy=heavy,
        covered=covered,
        private=private,
        window_n=int(cfg.get("calibrate_pool_n", 8)),
        limit=limit,
    )
    return {
        "train_head": train[:8],
        "heavy": sorted(heavy),
        "private_unpulled": sorted(private),
        "already_covered": sorted(covered),
        "chosen": chosen,
        "image_notes": {k: image_notes[k] for k in list(image_notes)[:12]},
        "rule": (
            "manifest order; probe ranks the fixed first-8 window by manifest bytes; "
            "skip tasks that already have an agent trace, source-build images "
            "(build-essential, same class as the ffmpeg abort), and private bases "
            "that are not present locally"
        ),
    }


def _eval_one(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    version_id: str,
    modules_root: Path,
    task: str,
    rep: int,
    prov: Any,
    temperature: float,
) -> Any:
    trial_dir = run_dir / "evals" / version_id / task / f"rep_{rep}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    proc = run_harbor_task(
        modularrsi_root=resolve_path(cfg, "modularrsi_root"),
        modules_root=modules_root,
        task_name=task,
        output_dir=trial_dir,
        model=prov.model,
        api_base=prov.api_base,
        temperature=temperature,
        dataset_path=resolve_path(cfg, "dataset_tb"),
        locked_module_type=LOCKED,
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
    return summary


def _collect_findings(trial_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for traj in trial_dir.rglob("trajectory.json"):
        found.extend(scan_trajectory(traj))
        found.extend(false_concat_findings(traj))
        found.extend(scan_saved_responses(traj.parent))
        found.extend(silent_duration_caps(traj.parent))
        found.extend(replay_parser_errors(traj.parent))
    # trial logs can record parse_error even if the trajectory is short
    for log in trial_dir.rglob("trial.log"):
        text = log.read_text(errors="replace")
        if "failure_tag=parse_error" in text or "parsing errors" in text:
            found.append(
                {
                    "kind": "parse_error",
                    "sufficient": True,
                    "detail": "trial.log recorded a parse error",
                    "excerpt": text[-1500:],
                    "path": str(log),
                }
            )
    return found


def sufficient_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only evidence that can ground a functional tools edit."""
    out: list[dict[str, Any]] = []
    for finding in findings:
        if finding.get("sufficient"):
            out.append(finding)
            continue
        # false_concat_findings always sets sufficient=True; keep a belt check.
        if finding.get("kind") == "false_concat_warning_on_empty_keystrokes":
            out.append({**finding, "sufficient": True})
    return out


def continuation_finding(
    findings: list[dict[str, Any]],
    *,
    already_handled_kinds: set[str],
) -> dict[str, Any] | None:
    """Pick a residual tools defect not already addressed by the parent edit."""
    for finding in sufficient_findings(findings):
        kind = str(finding.get("kind") or "")
        if kind in already_handled_kinds:
            continue
        if kind in {"unreadable", "parser_replay_failed", "duration_above_cap_requested"}:
            continue
        return finding
    # Incomplete fix: false concat still present after a suppress/normalize edit.
    for finding in sufficient_findings(findings):
        if finding.get("kind") == "false_concat_warning_on_empty_keystrokes":
            return finding
    return None


def _excerpt_for_planner(finding: dict[str, Any], source: str) -> str:
    excerpt = str(finding.get("excerpt") or "")[:4000]
    return (
        "Evidenced tools issue:\n"
        f"kind: {finding.get('kind')}\n"
        f"detail: {finding.get('detail')}\n"
        f"path: {finding.get('path')}\n"
        "excerpt:\n"
        f"{excerpt}\n\n"
        "Current tools/baseline.py (the only file you may replace):\n"
        f"{source}"
    )


BRANCH_STRATEGIES: dict[str, str] = {
    "A": (
        "Strategy A (suppress-only): After parse, detect empty-keystrokes commands "
        "that are followed by another command, remove only the matching false "
        "newline-concatenation warning text, and append this exact trigger when "
        "suppression occurs: "
        "BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED: empty-keystrokes wait command "
        "does not need a trailing newline. "
        "Do not rewrite keystrokes. Do not invent unrelated features."
    ),
    "B": (
        "Strategy B (normalize empty waits): After parse, when a command has empty "
        "keystrokes and is followed by another command, rewrite that command's "
        "keystrokes to a single newline so the wait still elapses but the false "
        "concatenation condition cannot arise at the source, clear the matching "
        "false warning, and append this exact trigger when normalization occurs: "
        "BASELINE_TOOLS_EMPTY_WAIT_NORMALIZED: empty-keystrokes wait rewritten to "
        "newline so following commands are not falsely marked as concatenated. "
        "Do not use the Strategy A suppress-only trigger string. Do not invent "
        "unrelated features."
    ),
}


def _planner_prompt(
    evidence: str,
    rejection: str = "",
    *,
    strategy: str | None = None,
) -> str:
    extra = ""
    if rejection:
        extra = (
            "\nThe previous replacement was rejected by the fixed editor:\n"
            f"{rejection}\n"
            "Return a corrected full file. Do not add comment markers.\n"
        )
    strategy_block = ""
    if strategy:
        strategy_block = (
            f"\nYou must implement ONLY this branch strategy:\n{strategy}\n"
            "Your TRIGGER line must be exactly the distinctive string named in the "
            "strategy (the part after 'append this exact trigger').\n"
        )
    return (
        "You are a frozen planner. You are not being trained. Read the tools-module "
        "failure evidence and replace tools/baseline.py with a functional fix for "
        "THAT evidence. Do not touch observation truncation or verification. Do not "
        "invent an unrelated feature. Do not add comment-only markers.\n"
        "The loaded implementation is BaselineTools in tools/baseline.py. Harbor's "
        "own parser files are outside this edit. Put the fix in BaselineTools "
        "(parse_llm_response, execute, or the feedback those paths return).\n"
        "The fix must emit a distinctive feedback string only when the evidenced "
        "condition occurs, so a later trial can show the change actually ran.\n"
        f"{strategy_block}"
        "Reply in exactly this shape:\n"
        "PLAN:\n"
        "<what is wrong and what the edit changes>\n"
        "TRIGGER: <the distinctive feedback string, one line>\n"
        f"{FILE_BEGIN}\n"
        "<complete replacement file>\n"
        f"{FILE_END}\n"
        "No markdown fences around the file.\n"
        f"{extra}\n"
        f"{evidence}"
    )


def _frozen_chat(prov: Any, prompt: str) -> tuple[str, dict[str, Any]]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from llm_provider_lib import record_usage, resolve_provider

    resolved = resolve_provider(mode="tongji_agi")
    no_proxy = merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    url = resolved.api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": resolved.api_model_id,
        "temperature": 0,
        "max_tokens": 32000,
        "thinking": {"type": "disabled"},
        "messages": [
            {
                "role": "system",
                "content": "You edit one Python module. Follow the output shape exactly.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {resolved.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        if "thinking" in detail.lower() or exc.code == 400:
            body.pop("thinking", None)
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {resolved.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=240) as resp:
                payload = json.loads(resp.read().decode())
        else:
            raise RuntimeError(f"planner HTTP {exc.code}: {detail}") from exc
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or message.get("reasoning_content") or ""
    usage = payload.get("usage") or {}
    try:
        record_usage(
            resolved.provider_id,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=usage.get("total_tokens"),
            meta={"role": "causal_planner"},
        )
    except Exception:
        pass
    return content, {"model": resolved.api_model_id, "usage": usage, "provider": resolved.provider_id}


def _modules_root_in_trial(trial_dir: Path) -> str | None:
    for cfg in trial_dir.rglob("config.json"):
        try:
            data = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        blob = json.dumps(data)
        m = re.search(r'"modules_root"\s*:\s*"([^"]+)"', blob)
        if m:
            return m.group(1)
    return None


def _trigger_in_trial(trial_dir: Path, trigger: str) -> list[str]:
    hits: list[str] = []
    for p in trial_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".json", ".log", ".txt"}:
            continue
        if "modules" in p.parts and p.name == "baseline.py":
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if trigger and trigger in text:
            hits.append(str(p))
    return hits


def _write_report(path: Path, record: dict[str, Any]) -> None:
    direction = record.get("score_direction")
    label = score_direction_label(str(direction or "unknown"))
    lines = [
        "# research1 scaffold: failure feedback to a functional tools edit",
        "",
        "Track B only. Upstream ModularRSI was not modified. Comment-marker runs stay controls.",
        "",
        "## Why comment markers were not enough",
        "",
        "`branches_20261029_134412` applied `# research1-p2b-marker` lines in `tools/baseline.py`.",
        "Harbor loaded each version directory (`modules_root` matched, composer `tools=baseline`),",
        "but the marker is a comment, so the tools path did not change.",
        "`apply_plan_patch` then stored `outcome_class=no-effect` because the patch applied,",
        "not because behavior was measured. Cron rewards flipped across identical markers",
        "(H0 cron reps 0 and 1; browser was 1 on every trial). That is score noise.",
        "132345 stopped on Tongji HTTP 403 (`llm_call_failed`) before any tool call.",
        "The smoke archive's ffmpeg trial died in `EnvironmentStartTimeoutError` during image start.",
        "",
        "## Existing traces",
        "",
        record.get("existing_scan_summary") or "",
        "",
        "## Task coverage",
        "",
        record.get("coverage_summary") or "",
        "",
        "## Automatic H0 to H1",
        "",
    ]
    if not record.get("loop_ran"):
        lines.append(record.get("loop_skip_reason") or "The automatic loop did not run.")
    else:
        lines.extend(
            [
                f"- Plan: `{record.get('plan_path')}`",
                f"- Diff: `{record.get('diff_path')}`",
                f"- H0 modules_root: `{record.get('h0_modules_root')}`",
                f"- H1 modules_root: `{record.get('h1_modules_root')}`",
                f"- apply_status: `{record.get('apply_status')}`",
                f"- Trigger text: `{record.get('trigger')}`",
                f"- Triggered in H1 trajectory/logs: `{record.get('triggered')}`",
                f"- Trigger hits: {record.get('trigger_hits')}",
                f"- Loaded modules_root matches H1: `{record.get('loaded_modules_root_ok')}`",
                "",
                "### Scores",
                "",
                f"- n_h0: {record.get('n_h0')}",
                f"- n_h1: {record.get('n_h1')}",
                f"- raw H0 rewards: {record.get('raw_h0_rewards')}",
                f"- raw H1 rewards: {record.get('raw_h1_rewards')}",
                f"- observed_delta (H1 mean − H0 mean): {record.get('observed_delta')}",
                f"- score direction: {label}",
                "",
                "Equal means are recorded as no score change measured.",
                "That phrase does not mean the edit had no behavioral effect.",
                f"Behavioral effect (trigger observed): `{record.get('triggered')}`.",
                f"outcome_class `{record.get('outcome_class')}` follows apply + trigger + score direction.",
                "When the trigger fired and the means were equal, outcome_class is success",
                "for the behavioral edit, and the score direction stays no score change measured.",
                "",
            ]
        )
    lines.append(f"Run directory: `{record.get('run_dir')}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_causal(
    config_path: Path | None = None,
    *,
    max_tasks: int = 3,
    reps: int = 1,
    provider_mode: str = "tongji_agi",
) -> tuple[int, dict[str, Any]]:
    cfg = load_pilot(config_path)
    pf = run_preflight(config_path)
    runs_root = resolve_path(cfg, "runs_root")
    existing = scan_runs_for_tools(runs_root)
    existing_summary = (
        f"Scanned {existing['trajectories']} trajectory files under runs/, skipping the "
        f"132345 proxy-403 stubs. Parser-replay errors: {existing.get('parser_error_count')}. "
        f"Newline-wait warnings kept as insufficient: {existing.get('insufficient_newline_warnings')}. "
        f"Sufficient tools defects: {existing['sufficient_count']}."
    )
    if existing["sufficient_count"]:
        existing_summary += " A sufficient finding was present in the archived runs."

    run_id = time.strftime("causal_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "existing_scan.json", existing)

    selection = build_task_selection(cfg, limit=max_tasks)
    write_json(run_dir / "task_selection.json", selection)

    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "existing_scan_summary": existing_summary,
        "loop_ran": False,
        "provider_mode": provider_mode,
    }

    if pf.status != "PASS":
        record["loop_skip_reason"] = "preflight did not pass"
        record["preflight"] = pf.to_dict()
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        return EXIT_INCOMPLETE, record

    prov = with_llm_provider_env(provider_mode)
    apply_provider_to_environ(prov)
    if "agi.tongji.edu.cn" not in os.environ.get("NO_PROXY", ""):
        record["loop_skip_reason"] = "NO_PROXY missing agi.tongji.edu.cn"
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        return EXIT_FAIL, record

    temperature = float(cfg.get("p2b_solver_temperature", 0.7))
    versions_root = run_dir / "versions"
    h0 = ensure_h0_version(versions_root, resolve_path(cfg, "modules_root"), "H0")
    mark_step(run_dir, "h0_materialized")

    defect: dict[str, Any] | None = None
    defect_task: str | None = None
    h0_runs: list[Any] = []
    coverage_notes: list[str] = []

    if existing["sufficient"]:
        defect = existing["sufficient"][0]
        defect_task = _task_in_path(Path(str(defect.get("path"))), train_task_names(resolve_path(cfg, "dataset_tb")))
        coverage_notes.append(
            "Archived runs already contained a sufficient tools finding; expansion not required for discovery."
        )

    tasks = list(selection["chosen"])
    if defect is None:
        if not tasks:
            record["loop_skip_reason"] = "no expansion task passed the selection rules"
            record["coverage_summary"] = json.dumps(selection, ensure_ascii=False)
            write_json(run_dir / "causal_record.json", record)
            _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
            return EXIT_INCOMPLETE, record
        for task in tasks:
            coverage_notes.append(f"H0 coverage eval: {task} x {reps}")
            print(f"[causal] H0 eval {task}", flush=True)
            for rep in range(1, reps + 1):
                summary = _eval_one(
                    cfg=cfg,
                    run_dir=run_dir,
                    version_id="H0",
                    modules_root=Path(h0.modules_root),
                    task=task,
                    rep=rep,
                    prov=prov,
                    temperature=temperature,
                )
                h0_runs.append(summary)
                found = _collect_findings(run_dir / "evals" / "H0" / task / f"rep_{rep}")
                write_json(
                    run_dir / "evals" / "H0" / task / f"rep_{rep}" / "tools_findings.json",
                    found,
                )
                sufficient = [f for f in found if f.get("sufficient")]
                if sufficient and defect is None:
                    defect = sufficient[0]
                    defect_task = task
            if defect is not None:
                break
        mark_step(run_dir, "coverage_h0")

    record["coverage_summary"] = (
        "Selection rule: " + selection["rule"] + ". "
        f"Chosen: {tasks}. "
        f"Skipped heavy: {selection['heavy']}. "
        f"Skipped private bases not on disk: {selection['private_unpulled']}. "
        f"Already had agent traces: {selection['already_covered']}. "
        + " ".join(coverage_notes)
    )

    if defect is None or not defect_task:
        record["loop_skip_reason"] = (
            "Coverage expansion finished without a sufficient tools-module defect "
            "(parse failure, hidden execute error, or silent duration cap). "
            "No patch was invented."
        )
        record["h0_rewards"] = [
            {"task": r.task_name, "rep": r.trial_index, "reward": r.reward} for r in h0_runs
        ]
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        write_json(run_dir / "summary.json", {"status": "INCOMPLETE", **record})
        return EXIT_INCOMPLETE, record

    # If the defect was archived, still run a fresh H0 on that task from this version tree
    # so H1 has a paired parent loaded the same way.
    if not any(r.task_name == defect_task for r in h0_runs):
        print(f"[causal] paired H0 {defect_task}", flush=True)
        for rep in range(1, reps + 1):
            h0_runs.append(
                _eval_one(
                    cfg=cfg,
                    run_dir=run_dir,
                    version_id="H0",
                    modules_root=Path(h0.modules_root),
                    task=defect_task,
                    rep=rep,
                    prov=prov,
                    temperature=temperature,
                )
            )

    source_path = Path(h0.modules_root) / TARGET_REL
    source = source_path.read_text()
    evidence = _excerpt_for_planner(defect, source)
    (run_dir / "planner_evidence.txt").write_text(evidence)

    rejection = ""
    diff_text = ""
    plan_text = ""
    trigger = ""
    replacement = ""
    editor_error = ""
    planner_raw_paths: list[str] = []
    for attempt_i in range(1, 3):
        print(f"[causal] planner attempt {attempt_i}", flush=True)
        raw, meta = _frozen_chat(prov, _planner_prompt(evidence, rejection))
        raw_path = run_dir / "plans" / f"planner_raw_{attempt_i}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw)
        planner_raw_paths.append(str(raw_path))
        write_json(run_dir / "plans" / f"planner_meta_{attempt_i}.json", meta)
        try:
            parsed = parse_planner_output(raw)
        except ValueError as exc:
            rejection = str(exc)
            editor_error = rejection
            continue
        diff_text, editor_error = editor_from_replacement(
            before=source,
            replacement=parsed["file_text"],
            trigger=parsed["trigger"],
        )
        if diff_text:
            plan_text = parsed["plan_text"]
            trigger = parsed["trigger"]
            replacement = parsed["file_text"]
            break
        rejection = editor_error

    record["loop_ran"] = True
    record["planner_raw"] = planner_raw_paths
    record["trigger"] = trigger
    record["defect"] = {k: defect.get(k) for k in ("kind", "detail", "path", "step_id")}
    record["defect_task"] = defect_task

    if not diff_text:
        record["apply_status"] = "rejected"
        record["editor_error"] = editor_error
        record["loop_skip_reason"] = f"fixed editor rejected the planner output: {editor_error}"
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        write_json(run_dir / "summary.json", {"status": "FAIL", **record})
        return EXIT_FAIL, record

    plan = make_plan(
        parent_version_id="H0",
        locked_module=LOCKED,
        plan_text=plan_text,
        candidate_label="causal",
        intended_files=[TARGET_REL],
    )
    attempt = apply_plan_patch(
        run_dir=run_dir,
        versions_root=versions_root,
        plan=plan,
        patch_text=diff_text,
        child_version_id="H1",
        outcome_class=None,
    )
    applied = bool(attempt.applied and attempt.child_version_id)
    record["apply_status"] = "applied" if applied else "apply-failure"
    record["reject_reason"] = attempt.reject_reason
    record["plan_path"] = str(run_dir / "plans" / f"{plan.plan_id}.txt")
    record["diff_path"] = str(run_dir / "patches" / f"{attempt.attempt_id}.diff")
    record["h0_modules_root"] = h0.modules_root
    if not applied:
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        write_json(run_dir / "summary.json", {"status": "FAIL", **record})
        return EXIT_FAIL, record

    h1_state = load_state(versions_root, "H1")
    record["h1_modules_root"] = h1_state.modules_root
    # Confirm the version tree itself changed functionally.
    h1_src = Path(h1_state.modules_root) / TARGET_REL
    changed, why = functional_ast_changed(source, h1_src.read_text())
    record["version_tree_functional_change"] = changed
    record["version_tree_note"] = why
    h1_hash = hash_tree(Path(h1_state.modules_root), [TARGET_REL]).get(TARGET_REL)
    record["h1_baseline_sha256"] = h1_hash

    print(f"[causal] H1 eval {defect_task}", flush=True)
    h1_runs = []
    for rep in range(1, reps + 1):
        h1_runs.append(
            _eval_one(
                cfg=cfg,
                run_dir=run_dir,
                version_id="H1",
                modules_root=Path(h1_state.modules_root),
                task=defect_task,
                rep=rep,
                prov=prov,
                temperature=temperature,
            )
        )
    mark_step(run_dir, "eval_H1")

    trigger_hits: list[str] = []
    loaded_ok = False
    seen_roots: list[str | None] = []
    for rep in range(1, reps + 1):
        trial_dir = run_dir / "evals" / "H1" / defect_task / f"rep_{rep}"
        trigger_hits.extend(_trigger_in_trial(trial_dir, trigger))
        root = _modules_root_in_trial(trial_dir)
        seen_roots.append(root)
        if root and Path(root).resolve() == Path(h1_state.modules_root).resolve():
            loaded_ok = True
    # A second rep only when the first execution did not exercise the new path.
    if not trigger_hits and reps < 2:
        print("[causal] H1 did not trigger; one extra rep", flush=True)
        extra = _eval_one(
            cfg=cfg,
            run_dir=run_dir,
            version_id="H1",
            modules_root=Path(h1_state.modules_root),
            task=defect_task,
            rep=reps + 1,
            prov=prov,
            temperature=temperature,
        )
        h1_runs.append(extra)
        trial_dir = run_dir / "evals" / "H1" / defect_task / f"rep_{reps + 1}"
        trigger_hits.extend(_trigger_in_trial(trial_dir, trigger))
        root = _modules_root_in_trial(trial_dir)
        seen_roots.append(root)
        if root and Path(root).resolve() == Path(h1_state.modules_root).resolve():
            loaded_ok = True

    h0_task_runs = [r for r in h0_runs if r.task_name == defect_task]
    raw_h0 = [r.reward for r in h0_task_runs]
    raw_h1 = [r.reward for r in h1_runs]
    mean_h0 = _mean(raw_h0)
    mean_h1 = _mean(raw_h1)
    delta = None if mean_h0 is None or mean_h1 is None else mean_h1 - mean_h0
    direction = score_direction(delta)
    triggered = bool(trigger_hits)
    klass = outcome_for(applied=True, triggered=triggered, direction=direction)

    # Rewrite the attempt record so no-effect is not left as an unmeasured default.
    attempt_path = run_dir / "attempts" / f"{attempt.attempt_id}.json"
    if attempt_path.is_file():
        attempt_doc = json.loads(attempt_path.read_text())
        attempt_doc["outcome_class"] = klass
        attempt_doc["apply_status"] = "applied"
        attempt_doc["score_direction"] = direction
        attempt_doc["score_direction_label"] = score_direction_label(direction)
        attempt_doc["behavioral_effect"] = triggered
        attempt_doc["observed_delta"] = delta
        attempt_doc["outcome_note"] = (
            "score_direction is only the measured mean. "
            "no score change measured does not mean there was no behavioral effect. "
            "no-effect is used only when the new logic was not observed in the trial."
        )
        attempt_path.write_text(json.dumps(attempt_doc, indent=2) + "\n")

    parent_scores = aggregate_task_means(h0_task_runs)
    child_scores = aggregate_task_means(h1_runs)
    write_transition(
        run_dir,
        transition_id="H0_to_H1",
        parent_version_id="H0",
        child_version_id="H1",
        plan_id=plan.plan_id,
        attempt_id=attempt.attempt_id,
        parent_scores=parent_scores,
        child_scores=child_scores,
    )

    record.update(
        {
            "triggered": triggered,
            "trigger_hits": trigger_hits[:12],
            "loaded_modules_root_ok": loaded_ok,
            "loaded_modules_roots": seen_roots,
            "n_h0": len(raw_h0),
            "n_h1": len(raw_h1),
            "raw_h0_rewards": raw_h0,
            "raw_h1_rewards": raw_h1,
            "observed_delta": delta,
            "score_direction": direction,
            "outcome_class": klass,
            "replacement_stored": True,
            "editor_kept_trigger_constant": True,
        }
    )
    # Keep the replacement for audit without treating it as a second source of truth.
    (run_dir / "plans" / "replacement_tools_baseline.py").write_text(replacement)
    status = "PASS" if triggered and loaded_ok and changed else "INCOMPLETE"
    record["status"] = status
    write_json(run_dir / "causal_record.json", record)
    write_json(run_dir / "summary.json", {"status": status, "run_id": run_id, "run_dir": str(run_dir)})
    _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
    code = EXIT_PASS if status == "PASS" else EXIT_INCOMPLETE
    return code, record


def _best_false_concat(runs_root: Path) -> dict[str, Any] | None:
    ranked: list[tuple[int, dict[str, Any]]] = []
    if not runs_root.is_dir():
        return None
    for traj in runs_root.rglob("trajectory.json"):
        if "132345" in traj.parts:
            continue
        text = str(traj)
        for finding in false_concat_findings(traj):
            rank = 0
            if "/evals/H0/" in text:
                rank += 2
            if "cron-env-debug" in text:
                rank += 1
            ranked.append((rank, finding))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def run_false_concat_loop(
    config_path: Path | None = None,
    *,
    reps: int = 1,
    provider_mode: str = "tongji_agi",
    max_h1_reps: int = 3,
) -> tuple[int, dict[str, Any]]:
    """Plan and apply a tools fix for the empty-keystroke concatenation warning.

    Coverage already showed no parse error. This uses the warning that is
    present in archived cron trajectories, then runs a fresh H0 and H1.
    """
    cfg = load_pilot(config_path)
    pf = run_preflight(config_path)
    runs_root = resolve_path(cfg, "runs_root")
    finding = _best_false_concat(runs_root)
    run_id = time.strftime("causal_edit_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "loop_ran": False,
        "provider_mode": provider_mode,
        "existing_scan_summary": (
            "Archived parser replay: 0 parse errors. Five newline warnings all sit on "
            "empty-keystroke waits and were promoted only after coverage "
            "(causal_20261030_022523: complex-python reward 1, wasm reward 0 on a "
            "task assertion, crypto reward 1) also found no parse error, hidden "
            "execute failure, or silent duration cap."
        ),
        "coverage_summary": (
            "Prior coverage run causal_20261030_022523 evaluated "
            "complex-python-memory-leak-concurrency-bug (1.0), wasm-crackme-reverse (0.0), "
            "and crypto-transaction-batching-optimizer (1.0) from the fixed train window. "
            "None produced a parse failure. The edit below targets the false concatenation "
            "warning on empty keystrokes in cron-env-debug, which is where that warning occurred."
        ),
    }
    if pf.status != "PASS":
        record["loop_skip_reason"] = "preflight did not pass"
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        return EXIT_INCOMPLETE, record
    if finding is None:
        record["loop_skip_reason"] = "no empty-keystroke concatenation warning in archived trajectories"
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        return EXIT_INCOMPLETE, record

    prov = with_llm_provider_env(provider_mode)
    apply_provider_to_environ(prov)
    if "agi.tongji.edu.cn" not in os.environ.get("NO_PROXY", ""):
        record["loop_skip_reason"] = "NO_PROXY missing agi.tongji.edu.cn"
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        return EXIT_FAIL, record

    task = "cron-env-debug"
    temperature = float(cfg.get("p2b_solver_temperature", 0.7))
    versions_root = run_dir / "versions"
    h0 = ensure_h0_version(versions_root, resolve_path(cfg, "modules_root"), "H0")
    source = (Path(h0.modules_root) / TARGET_REL).read_text()
    evidence = _excerpt_for_planner(finding, source)
    (run_dir / "planner_evidence.txt").write_text(evidence)
    write_json(run_dir / "promoted_finding.json", finding)

    rejection = ""
    diff_text = ""
    plan_text = ""
    trigger = ""
    replacement = ""
    editor_error = ""
    for attempt_i in range(1, 3):
        print(f"[causal] planner attempt {attempt_i}", flush=True)
        raw, meta = _frozen_chat(prov, _planner_prompt(evidence, rejection))
        raw_path = run_dir / "plans" / f"planner_raw_{attempt_i}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw)
        write_json(run_dir / "plans" / f"planner_meta_{attempt_i}.json", meta)
        try:
            parsed = parse_planner_output(raw)
        except ValueError as exc:
            rejection = str(exc)
            editor_error = rejection
            continue
        diff_text, editor_error = editor_from_replacement(
            before=source,
            replacement=parsed["file_text"],
            trigger=parsed["trigger"],
        )
        if diff_text:
            plan_text = parsed["plan_text"]
            trigger = parsed["trigger"]
            replacement = parsed["file_text"]
            break
        rejection = editor_error

    record["loop_ran"] = True
    record["trigger"] = trigger
    record["defect"] = {k: finding.get(k) for k in ("kind", "detail", "path", "step_id")}
    record["defect_task"] = task
    if not diff_text:
        record["apply_status"] = "rejected"
        record["editor_error"] = editor_error
        record["loop_skip_reason"] = f"fixed editor rejected the planner output: {editor_error}"
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        write_json(run_dir / "summary.json", {"status": "FAIL", **record})
        return EXIT_FAIL, record

    plan = make_plan(
        parent_version_id="H0",
        locked_module=LOCKED,
        plan_text=plan_text,
        candidate_label="causal",
        intended_files=[TARGET_REL],
    )
    attempt = apply_plan_patch(
        run_dir=run_dir,
        versions_root=versions_root,
        plan=plan,
        patch_text=diff_text,
        child_version_id="H1",
    )
    if not attempt.applied or not attempt.child_version_id:
        record["apply_status"] = "apply-failure"
        record["reject_reason"] = attempt.reject_reason
        write_json(run_dir / "causal_record.json", record)
        _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
        return EXIT_FAIL, record

    h1_state = load_state(versions_root, "H1")
    changed, why = functional_ast_changed(source, (Path(h1_state.modules_root) / TARGET_REL).read_text())
    record.update(
        {
            "apply_status": "applied",
            "plan_path": str(run_dir / "plans" / f"{plan.plan_id}.txt"),
            "diff_path": str(run_dir / "patches" / f"{attempt.attempt_id}.diff"),
            "h0_modules_root": h0.modules_root,
            "h1_modules_root": h1_state.modules_root,
            "version_tree_functional_change": changed,
            "version_tree_note": why,
            "h1_baseline_sha256": hash_tree(Path(h1_state.modules_root), [TARGET_REL]).get(TARGET_REL),
        }
    )
    (run_dir / "plans" / "replacement_tools_baseline.py").write_text(replacement)

    print(f"[causal] H0 eval {task}", flush=True)
    h0_runs = [
        _eval_one(
            cfg=cfg,
            run_dir=run_dir,
            version_id="H0",
            modules_root=Path(h0.modules_root),
            task=task,
            rep=rep,
            prov=prov,
            temperature=temperature,
        )
        for rep in range(1, reps + 1)
    ]
    h1_runs = []
    trigger_hits: list[str] = []
    seen_roots: list[str | None] = []
    loaded_ok = False
    for rep in range(1, max_h1_reps + 1):
        print(f"[causal] H1 eval {task} rep {rep}", flush=True)
        h1_runs.append(
            _eval_one(
                cfg=cfg,
                run_dir=run_dir,
                version_id="H1",
                modules_root=Path(h1_state.modules_root),
                task=task,
                rep=rep,
                prov=prov,
                temperature=temperature,
            )
        )
        trial_dir = run_dir / "evals" / "H1" / task / f"rep_{rep}"
        trigger_hits.extend(_trigger_in_trial(trial_dir, trigger))
        root = _modules_root_in_trial(trial_dir)
        seen_roots.append(root)
        if root and Path(root).resolve() == Path(h1_state.modules_root).resolve():
            loaded_ok = True
        if trigger_hits:
            break

    raw_h0 = [r.reward for r in h0_runs]
    raw_h1 = [r.reward for r in h1_runs]
    mean_h0 = _mean(raw_h0)
    mean_h1 = _mean(raw_h1)
    delta = None if mean_h0 is None or mean_h1 is None else mean_h1 - mean_h0
    direction = score_direction(delta)
    triggered = bool(trigger_hits)
    klass = outcome_for(applied=True, triggered=triggered, direction=direction)
    attempt_path = run_dir / "attempts" / f"{attempt.attempt_id}.json"
    if attempt_path.is_file():
        attempt_doc = json.loads(attempt_path.read_text())
        attempt_doc["outcome_class"] = klass
        attempt_doc["apply_status"] = "applied"
        attempt_doc["score_direction"] = direction
        attempt_doc["score_direction_label"] = score_direction_label(direction)
        attempt_doc["behavioral_effect"] = triggered
        attempt_doc["observed_delta"] = delta
        attempt_doc["outcome_note"] = (
            "score_direction is only the measured mean. "
            "no score change measured does not mean there was no behavioral effect."
        )
        attempt_path.write_text(json.dumps(attempt_doc, indent=2) + "\n")
    write_transition(
        run_dir,
        transition_id="H0_to_H1",
        parent_version_id="H0",
        child_version_id="H1",
        plan_id=plan.plan_id,
        attempt_id=attempt.attempt_id,
        parent_scores=aggregate_task_means(h0_runs),
        child_scores=aggregate_task_means(h1_runs),
    )
    record.update(
        {
            "triggered": triggered,
            "trigger_hits": trigger_hits[:12],
            "loaded_modules_root_ok": loaded_ok,
            "loaded_modules_roots": seen_roots,
            "n_h0": len(raw_h0),
            "n_h1": len(raw_h1),
            "raw_h0_rewards": raw_h0,
            "raw_h1_rewards": raw_h1,
            "observed_delta": delta,
            "score_direction": direction,
            "outcome_class": klass,
        }
    )
    status = "PASS" if triggered and loaded_ok and changed else "INCOMPLETE"
    record["status"] = status
    write_json(run_dir / "causal_record.json", record)
    write_json(run_dir / "summary.json", {"status": status, "run_id": run_id, "run_dir": str(run_dir)})
    _write_report(resolve_path(cfg, "reports_root") / "research1_scaffold_causal.md", record)
    return (EXIT_PASS if status == "PASS" else EXIT_INCOMPLETE), record


def _plan_one_edit(
    *,
    run_dir: Path,
    label: str,
    source: str,
    evidence: str,
    prov: Any,
    strategy: str,
    expected_trigger_substr: str,
) -> dict[str, Any]:
    """Frozen planner + fixed editor for one branch label."""
    rejection = ""
    out: dict[str, Any] = {
        "label": label,
        "ok": False,
        "plan_text": "",
        "trigger": "",
        "diff_text": "",
        "replacement": "",
        "editor_error": "",
    }
    for attempt_i in range(1, 3):
        print(f"[dual] planner {label} attempt {attempt_i}", flush=True)
        raw, meta = _frozen_chat(prov, _planner_prompt(evidence, rejection, strategy=strategy))
        raw_path = run_dir / "plans" / f"planner_raw_{label}_{attempt_i}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw)
        write_json(run_dir / "plans" / f"planner_meta_{label}_{attempt_i}.json", meta)
        try:
            parsed = parse_planner_output(raw)
        except ValueError as exc:
            rejection = str(exc)
            out["editor_error"] = rejection
            continue
        if expected_trigger_substr and expected_trigger_substr not in parsed["trigger"]:
            rejection = (
                f"TRIGGER must contain {expected_trigger_substr!r}; "
                f"got {parsed['trigger']!r}"
            )
            out["editor_error"] = rejection
            continue
        diff_text, editor_error = editor_from_replacement(
            before=source,
            replacement=parsed["file_text"],
            trigger=parsed["trigger"],
        )
        if diff_text:
            out.update(
                {
                    "ok": True,
                    "plan_text": parsed["plan_text"],
                    "trigger": parsed["trigger"],
                    "diff_text": diff_text,
                    "replacement": parsed["file_text"],
                    "editor_error": "",
                }
            )
            return out
        rejection = editor_error
        out["editor_error"] = editor_error
    return out


def _eval_version_task(
    *,
    cfg: dict[str, Any],
    run_dir: Path,
    version_id: str,
    modules_root: Path,
    task: str,
    reps: int,
    prov: Any,
    temperature: float,
) -> list[Any]:
    return [
        _eval_one(
            cfg=cfg,
            run_dir=run_dir,
            version_id=version_id,
            modules_root=modules_root,
            task=task,
            rep=rep,
            prov=prov,
            temperature=temperature,
        )
        for rep in range(1, reps + 1)
    ]


def _measure_trigger(
    *,
    run_dir: Path,
    version_id: str,
    modules_root: Path,
    task: str,
    trigger: str,
    reps: int,
) -> dict[str, Any]:
    trigger_hits: list[str] = []
    seen_roots: list[str | None] = []
    loaded_ok = False
    for rep in range(1, reps + 1):
        trial_dir = run_dir / "evals" / version_id / task / f"rep_{rep}"
        trigger_hits.extend(_trigger_in_trial(trial_dir, trigger))
        root = _modules_root_in_trial(trial_dir)
        seen_roots.append(root)
        if root and Path(root).resolve() == Path(modules_root).resolve():
            loaded_ok = True
    return {
        "trigger_hits": trigger_hits,
        "loaded_modules_roots": seen_roots,
        "loaded_modules_root_ok": loaded_ok,
        "triggered": bool(trigger_hits),
    }


def _write_dual_report(path: Path, record: dict[str, Any]) -> None:
    lines = [
        "# research1 dual-branch continuation (Track B)",
        "",
        "Same-parent H0 fork into two trajectory-grounded **functional** tools candidates, "
        "then one continuation attempt each. Comment markers are rejected. Score uplift is "
        "not the pass criterion; load + trigger evidence is.",
        "",
        f"- Run: `{record.get('run_dir')}`",
        f"- Parent H0 source: `{record.get('h0_source')}`",
        f"- Evidenced defect: `{((record.get('defect') or {}).get('kind'))}`",
        f"- Status: **{record.get('status')}**",
        f"- Early stop: `{record.get('early_stop_reason') or 'none'}`",
        "",
        "## Why two candidates from one defect class",
        "",
        "Coverage (`causal_20261030_022523`) and archived parser replay found no parse "
        "error, silent duration cap, or unsourced execute failure. The only sufficient "
        "tools defect remains false newline-concatenation warnings on **empty-keystroke** "
        "waits. Dual-branch therefore forks two **distinct functional strategies** for "
        "that same evidence (suppress-only vs normalize empty waits), not a second invented "
        "bug and not comment markers.",
        "",
        "## Branches",
        "",
    ]
    for branch in record.get("branches") or []:
        label = branch.get("label")
        lines.extend(
            [
                f"### Branch {label}",
                "",
                f"- Strategy: `{branch.get('strategy_id')}`",
                f"- Round-1 version: `{branch.get('v1_id')}`",
                f"- Plan: `{branch.get('plan_path')}`",
                f"- Diff: `{branch.get('diff_path')}`",
                f"- Trigger: `{branch.get('trigger')}`",
                f"- Applied: `{branch.get('applied')}` / functional AST change: "
                f"`{branch.get('functional_change')}`",
                f"- Loaded modules_root ok: `{branch.get('loaded_modules_root_ok')}`",
                f"- Behavioral trigger: `{branch.get('triggered')}`",
                f"- Raw rewards (task={branch.get('task')}): "
                f"`{branch.get('raw_rewards')}`",
                f"- Score direction vs H0: `{branch.get('score_direction')}` "
                f"({score_direction_label(str(branch.get('score_direction') or 'unknown'))})",
                f"- Outcome class: `{branch.get('outcome_class')}`",
                f"- Continuation: `{branch.get('continuation_status')}`",
                f"- Continuation reason: {branch.get('continuation_reason') or ''}",
                "",
            ]
        )
        if branch.get("v2_id"):
            lines.extend(
                [
                    f"- Round-2 version: `{branch.get('v2_id')}`",
                    f"- Continuation plan: `{branch.get('cont_plan_path')}`",
                    f"- Continuation diff: `{branch.get('cont_diff_path')}`",
                    f"- Continuation trigger: `{branch.get('cont_trigger')}`",
                    f"- Continuation triggered: `{branch.get('cont_triggered')}`",
                    f"- Continuation rewards: `{branch.get('cont_raw_rewards')}`",
                    "",
                ]
            )
    lines.extend(
        [
            "## H0 baseline this run",
            "",
            f"- Raw H0 rewards: `{record.get('raw_h0_rewards')}`",
            f"- Task: `{record.get('defect_task')}`",
            "",
            "## Hard-rule check",
            "",
            "- Upstream ModularRSI checkout was not modified.",
            "- Edits live only under this run's `versions/`.",
            "- No `_comment_patch` / marker-only A2/B2 padding.",
            "- If a continuation lacked residual sufficient tools evidence, it stopped "
            "and recorded why instead of inventing a second edit.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_dual_branch(
    config_path: Path | None = None,
    *,
    reps: int = 1,
    provider_mode: str = "tongji_agi",
    prior_h0: Path | None = None,
    max_trigger_reps: int = 3,
) -> tuple[int, dict[str, Any]]:
    """Same H0 → two functional candidates → one evidence-gated continuation each."""
    cfg = load_pilot(config_path)
    pf = run_preflight(config_path)
    runs_root = resolve_path(cfg, "runs_root")
    prior = prior_h0 or (
        runs_root / "causal_edit_20261030_025402" / "versions" / "H0" / "modules"
    )
    run_id = time.strftime("causal_dual_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep at least `reps` trials; keep going (up to max_trigger_reps) until the
    # distinctive trigger appears. Empty-keystroke waits that are alone in a batch
    # never produce the concat warning, so one lucky cron solve can miss the path.
    trigger_reps = max(int(reps), int(max_trigger_reps))
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "provider_mode": provider_mode,
        "h0_source": str(prior),
        "early_stop_reason": "",
        "branches": [],
        "status": "INCOMPLETE",
        "trigger_reps": trigger_reps,
    }
    if pf.status != "PASS":
        record["early_stop_reason"] = "preflight did not pass"
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        return EXIT_INCOMPLETE, record

    finding = _best_false_concat(runs_root)
    if finding is None:
        record["early_stop_reason"] = (
            "no empty-keystroke concatenation warning in archived trajectories; "
            "refusing to invent a second tools defect for dual-branch"
        )
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        return EXIT_INCOMPLETE, record

    prov = with_llm_provider_env(provider_mode)
    apply_provider_to_environ(prov)
    if "agi.tongji.edu.cn" not in os.environ.get("NO_PROXY", ""):
        record["early_stop_reason"] = "NO_PROXY missing agi.tongji.edu.cn"
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        return EXIT_FAIL, record

    if not prior.is_dir():
        record["early_stop_reason"] = f"prior H0 modules missing: {prior}"
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        return EXIT_FAIL, record

    task = "cron-env-debug"
    temperature = float(cfg.get("p2b_solver_temperature", 0.7))
    versions_root = run_dir / "versions"
    h0 = ensure_h0_version(versions_root, prior, "H0")
    source = (Path(h0.modules_root) / TARGET_REL).read_text()
    evidence = _excerpt_for_planner(finding, source)
    (run_dir / "planner_evidence.txt").write_text(evidence)
    write_json(run_dir / "promoted_finding.json", finding)
    record["defect"] = {k: finding.get(k) for k in ("kind", "detail", "path", "step_id")}
    record["defect_task"] = task

    expected = {
        "A": "BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED",
        "B": "BASELINE_TOOLS_EMPTY_WAIT_NORMALIZED",
    }
    planned: dict[str, dict[str, Any]] = {}
    for label in ("A", "B"):
        planned[label] = _plan_one_edit(
            run_dir=run_dir,
            label=label,
            source=source,
            evidence=evidence,
            prov=prov,
            strategy=BRANCH_STRATEGIES[label],
            expected_trigger_substr=expected[label],
        )

    if not planned["A"]["ok"] or not planned["B"]["ok"]:
        record["early_stop_reason"] = (
            "could not obtain two distinct functional plans from the evidenced defect: "
            f"A={planned['A'].get('editor_error')!r} B={planned['B'].get('editor_error')!r}"
        )
        record["planned"] = {
            label: {k: v for k, v in doc.items() if k != "replacement"}
            for label, doc in planned.items()
        }
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        write_json(run_dir / "summary.json", {"status": "INCOMPLETE", **record})
        return EXIT_INCOMPLETE, record

    if planned["A"]["trigger"] == planned["B"]["trigger"]:
        record["early_stop_reason"] = (
            "branch A and B produced the same TRIGGER; refusing non-distinct candidates"
        )
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        return EXIT_INCOMPLETE, record

    # Same-parent fork before any eval (B must not see A's scores).
    fork_rows: list[dict[str, Any]] = []
    for label in ("A", "B"):
        doc = planned[label]
        (run_dir / "plans" / f"replacement_{label}_tools_baseline.py").write_text(doc["replacement"])
        plan = make_plan(
            parent_version_id="H0",
            locked_module=LOCKED,
            plan_text=doc["plan_text"],
            candidate_label=label,
            intended_files=[TARGET_REL],
        )
        child_id = f"H0_{label}1"
        attempt = apply_plan_patch(
            run_dir=run_dir,
            versions_root=versions_root,
            plan=plan,
            patch_text=doc["diff_text"],
            child_version_id=child_id,
        )
        fork_rows.append(
            {
                "label": label,
                "plan_id": plan.plan_id,
                "attempt_id": attempt.attempt_id,
                "child_version_id": attempt.child_version_id,
                "applied": attempt.applied,
                "reject_reason": attempt.reject_reason,
                "trigger": doc["trigger"],
            }
        )
    write_json(run_dir / "fork_manifest.json", fork_rows)
    mark_step(run_dir, "forked")

    if any(not row["applied"] or not row["child_version_id"] for row in fork_rows):
        record["early_stop_reason"] = "one or both round-1 patches failed to apply"
        record["fork_manifest"] = fork_rows
        write_json(run_dir / "dual_record.json", record)
        _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
        return EXIT_FAIL, record

    print(f"[dual] H0 eval {task}", flush=True)
    h0_runs = _eval_version_task(
        cfg=cfg,
        run_dir=run_dir,
        version_id="H0",
        modules_root=Path(h0.modules_root),
        task=task,
        reps=reps,
        prov=prov,
        temperature=temperature,
    )
    raw_h0 = [r.reward for r in h0_runs]
    mean_h0 = _mean(raw_h0)
    record["raw_h0_rewards"] = raw_h0

    branches_out: list[dict[str, Any]] = []
    v1_runs_by_label: dict[str, list[Any]] = {}
    for row in fork_rows:
        label = row["label"]
        v1 = row["child_version_id"]
        assert v1 is not None
        state = load_state(versions_root, v1)
        before = source
        after = (Path(state.modules_root) / TARGET_REL).read_text()
        changed, why = functional_ast_changed(before, after)
        runs = []
        meas = {
            "trigger_hits": [],
            "loaded_modules_roots": [],
            "loaded_modules_root_ok": False,
            "triggered": False,
        }
        for rep in range(1, trigger_reps + 1):
            print(f"[dual] eval {v1} {task} rep {rep}/{trigger_reps}", flush=True)
            runs.append(
                _eval_one(
                    cfg=cfg,
                    run_dir=run_dir,
                    version_id=v1,
                    modules_root=Path(state.modules_root),
                    task=task,
                    rep=rep,
                    prov=prov,
                    temperature=temperature,
                )
            )
            meas = _measure_trigger(
                run_dir=run_dir,
                version_id=v1,
                modules_root=Path(state.modules_root),
                task=task,
                trigger=row["trigger"],
                reps=rep,
            )
            if meas["triggered"]:
                break
        v1_runs_by_label[label] = runs
        mean_child = _mean(raw)
        delta = None if mean_h0 is None or mean_child is None else mean_child - mean_h0
        direction = score_direction(delta)
        klass = outcome_for(
            applied=True,
            triggered=bool(meas["triggered"]),
            direction=direction,
        )
        attempt_path = run_dir / "attempts" / f"attempt_{label}_{v1}.json"
        if attempt_path.is_file():
            attempt_doc = json.loads(attempt_path.read_text())
            attempt_doc["outcome_class"] = klass
            attempt_doc["behavioral_effect"] = bool(meas["triggered"])
            attempt_doc["score_direction"] = direction
            attempt_doc["observed_delta"] = delta
            attempt_path.write_text(json.dumps(attempt_doc, indent=2) + "\n")
        write_transition(
            run_dir,
            transition_id=f"H0_to_{v1}",
            parent_version_id="H0",
            child_version_id=v1,
            plan_id=row["plan_id"],
            attempt_id=row["attempt_id"],
            parent_scores=aggregate_task_means(h0_runs),
            child_scores=aggregate_task_means(runs),
        )
        branches_out.append(
            {
                "label": label,
                "strategy_id": label,
                "strategy_text": BRANCH_STRATEGIES[label],
                "v1_id": v1,
                "plan_path": str(run_dir / "plans" / f"{row['plan_id']}.txt"),
                "diff_path": str(run_dir / "patches" / f"{row['attempt_id']}.diff"),
                "trigger": row["trigger"],
                "applied": True,
                "functional_change": changed,
                "functional_note": why,
                "task": task,
                "raw_rewards": raw,
                "observed_delta": delta,
                "score_direction": direction,
                "outcome_class": klass,
                "triggered": meas["triggered"],
                "trigger_hits": meas["trigger_hits"][:12],
                "loaded_modules_root_ok": meas["loaded_modules_root_ok"],
                "loaded_modules_roots": meas["loaded_modules_roots"],
                "continuation_status": "pending",
                "continuation_reason": "",
            }
        )

    # Continuations: only when residual sufficient tools evidence exists.
    handled_kinds = {"false_concat_warning_on_empty_keystrokes", "newline_warning_on_wait"}
    for branch in branches_out:
        label = branch["label"]
        v1 = branch["v1_id"]
        trial_dir = run_dir / "evals" / v1 / task
        findings = _collect_findings(trial_dir)
        write_json(run_dir / f"continuation_scan_{label}.json", findings)
        # If round-1 already triggered, empty-keystroke false concat is handled unless
        # residual false_concat findings remain in the child trial.
        residual = continuation_finding(findings, already_handled_kinds=handled_kinds)
        # Re-open false_concat only when still observed after the edit.
        still_false = [
            f
            for f in sufficient_findings(findings)
            if f.get("kind") == "false_concat_warning_on_empty_keystrokes"
        ]
        if still_false:
            residual = still_false[0]
        elif branch.get("triggered"):
            residual = continuation_finding(
                findings,
                already_handled_kinds=handled_kinds
                | {"false_concat_warning_on_empty_keystrokes"},
            )

        if residual is None:
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = (
                "no residual sufficient tools defect in round-1 trajectories after the "
                "functional edit; refusing marker-only A2/B2 padding"
            )
            continue

        parent_state = load_state(versions_root, v1)
        parent_src = (Path(parent_state.modules_root) / TARGET_REL).read_text()
        cont_evidence = _excerpt_for_planner(residual, parent_src)
        (run_dir / "plans" / f"continuation_evidence_{label}.txt").write_text(cont_evidence)
        # Continuation keeps the branch's strategy family but must target residual evidence.
        cont_strategy = (
            f"Continuation for branch {label}. Fix ONLY this residual evidenced tools "
            f"issue ({residual.get('kind')}). Emit a NEW distinctive TRIGGER string that "
            "does not match the round-1 trigger. Keep prior round-1 behavior working."
        )
        cont_plan = _plan_one_edit(
            run_dir=run_dir,
            label=f"{label}2",
            source=parent_src,
            evidence=cont_evidence,
            prov=prov,
            strategy=cont_strategy,
            expected_trigger_substr="BASELINE_TOOLS_",
        )
        if not cont_plan["ok"]:
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = (
                f"residual evidence existed ({residual.get('kind')}) but planner/editor "
                f"failed: {cont_plan.get('editor_error')}"
            )
            continue
        if cont_plan["trigger"] == branch["trigger"]:
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = "continuation TRIGGER matched round-1; not distinct"
            continue

        v2 = f"H0_{label}2"
        plan = make_plan(
            parent_version_id=v1,
            locked_module=LOCKED,
            plan_text=cont_plan["plan_text"],
            candidate_label=f"{label}_cont",
            intended_files=[TARGET_REL],
        )
        attempt = apply_plan_patch(
            run_dir=run_dir,
            versions_root=versions_root,
            plan=plan,
            patch_text=cont_plan["diff_text"],
            child_version_id=v2,
        )
        if not attempt.applied or not attempt.child_version_id:
            branch["continuation_status"] = "apply-failure"
            branch["continuation_reason"] = attempt.reject_reason
            continue
        (run_dir / "plans" / f"replacement_{label}2_tools_baseline.py").write_text(
            cont_plan["replacement"]
        )
        v2_state = load_state(versions_root, v2)
        print(f"[dual] eval {v2} {task}", flush=True)
        cont_runs = _eval_version_task(
            cfg=cfg,
            run_dir=run_dir,
            version_id=v2,
            modules_root=Path(v2_state.modules_root),
            task=task,
            reps=reps,
            prov=prov,
            temperature=temperature,
        )
        cont_meas = _measure_trigger(
            run_dir=run_dir,
            version_id=v2,
            modules_root=Path(v2_state.modules_root),
            task=task,
            trigger=cont_plan["trigger"],
            reps=reps,
        )
        write_transition(
            run_dir,
            transition_id=f"{v1}_to_{v2}",
            parent_version_id=v1,
            child_version_id=v2,
            plan_id=plan.plan_id,
            attempt_id=attempt.attempt_id,
            parent_scores=aggregate_task_means(v1_runs_by_label.get(label) or []),
            child_scores=aggregate_task_means(cont_runs),
        )
        branch.update(
            {
                "continuation_status": "applied",
                "continuation_reason": f"residual kind={residual.get('kind')}",
                "v2_id": v2,
                "cont_plan_path": str(run_dir / "plans" / f"{plan.plan_id}.txt"),
                "cont_diff_path": str(run_dir / "patches" / f"{attempt.attempt_id}.diff"),
                "cont_trigger": cont_plan["trigger"],
                "cont_triggered": cont_meas["triggered"],
                "cont_trigger_hits": cont_meas["trigger_hits"][:12],
                "cont_raw_rewards": [r.reward for r in cont_runs],
                "cont_loaded_ok": cont_meas["loaded_modules_root_ok"],
            }
        )

    record["branches"] = branches_out
    n_functional = sum(
        1
        for b in branches_out
        if b.get("functional_change") and b.get("triggered") and b.get("loaded_modules_root_ok")
    )
    cont_skipped = all(b.get("continuation_status") == "skipped" for b in branches_out)
    if n_functional >= 2:
        record["status"] = "PASS"
        if cont_skipped:
            record["early_stop_reason"] = (
                "two real round-1 branches produced; continuations skipped because no "
                "residual sufficient tools evidence remained (no marker padding)"
            )
    else:
        record["status"] = "INCOMPLETE"
        record["early_stop_reason"] = (
            record.get("early_stop_reason")
            or f"only {n_functional}/2 branches showed loaded+triggered functional edits"
        )

    write_json(run_dir / "dual_record.json", record)
    write_json(
        run_dir / "summary.json",
        {"status": record["status"], "run_id": run_id, "run_dir": str(run_dir)},
    )
    _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
    mark_step(run_dir, "dual_complete")
    code = EXIT_PASS if record["status"] == "PASS" else EXIT_INCOMPLETE
    return code, record


def resume_dual_branch_triggers(
    run_dir: Path,
    config_path: Path | None = None,
    *,
    provider_mode: str = "tongji_agi",
    max_trigger_reps: int = 5,
) -> tuple[int, dict[str, Any]]:
    """Continue evaluating already-forked dual branches until triggers appear."""
    cfg = load_pilot(config_path)
    run_dir = run_dir.resolve()
    record_path = run_dir / "dual_record.json"
    if not record_path.is_file():
        return EXIT_FAIL, {"status": "FAIL", "error": f"missing {record_path}"}
    record = json.loads(record_path.read_text())
    versions_root = run_dir / "versions"
    task = str(record.get("defect_task") or "cron-env-debug")
    temperature = float(cfg.get("p2b_solver_temperature", 0.7))
    prov = with_llm_provider_env(provider_mode)
    apply_provider_to_environ(prov)
    if "agi.tongji.edu.cn" not in os.environ.get("NO_PROXY", ""):
        record["early_stop_reason"] = "NO_PROXY missing agi.tongji.edu.cn"
        write_json(record_path, record)
        return EXIT_FAIL, record

    raw_h0 = list(record.get("raw_h0_rewards") or [])
    mean_h0 = _mean(raw_h0)
    branches = list(record.get("branches") or [])
    for branch in branches:
        if branch.get("triggered"):
            continue
        label = branch["label"]
        v1 = branch["v1_id"]
        state = load_state(versions_root, v1)
        existing = list((run_dir / "evals" / v1 / task).glob("rep_*"))
        start_rep = len(existing) + 1
        runs_extra = []
        for rep in range(start_rep, max_trigger_reps + 1):
            print(f"[dual-resume] eval {v1} {task} rep {rep}/{max_trigger_reps}", flush=True)
            runs_extra.append(
                _eval_one(
                    cfg=cfg,
                    run_dir=run_dir,
                    version_id=v1,
                    modules_root=Path(state.modules_root),
                    task=task,
                    rep=rep,
                    prov=prov,
                    temperature=temperature,
                )
            )
            meas = _measure_trigger(
                run_dir=run_dir,
                version_id=v1,
                modules_root=Path(state.modules_root),
                task=task,
                trigger=branch["trigger"],
                reps=rep,
            )
            branch["raw_rewards"] = list(branch.get("raw_rewards") or []) + [
                r.reward for r in runs_extra[-1:]
            ]
            # raw_rewards should include all reps — rebuild from all task_run files
            rewards = []
            for tr in sorted((run_dir / "evals" / v1 / task).glob("rep_*/task_run.json")):
                rewards.append(json.loads(tr.read_text()).get("reward"))
            branch["raw_rewards"] = rewards
            branch["triggered"] = bool(meas["triggered"])
            branch["trigger_hits"] = meas["trigger_hits"][:12]
            branch["loaded_modules_root_ok"] = meas["loaded_modules_root_ok"]
            branch["loaded_modules_roots"] = meas["loaded_modules_roots"]
            mean_child = _mean([r for r in rewards if isinstance(r, (int, float))])
            delta = None if mean_h0 is None or mean_child is None else mean_child - mean_h0
            branch["observed_delta"] = delta
            branch["score_direction"] = score_direction(delta)
            branch["outcome_class"] = outcome_for(
                applied=True,
                triggered=bool(meas["triggered"]),
                direction=branch["score_direction"],
            )
            if meas["triggered"]:
                break

        # Refresh continuation decision after new evidence.
        findings = _collect_findings(run_dir / "evals" / v1 / task)
        write_json(run_dir / f"continuation_scan_{label}.json", findings)
        still_false = [
            f
            for f in sufficient_findings(findings)
            if f.get("kind") == "false_concat_warning_on_empty_keystrokes"
        ]
        if still_false:
            branch["continuation_status"] = "pending_residual"
            branch["continuation_reason"] = (
                "false concat still observed after extra reps; residual edit warranted"
            )
        elif branch.get("triggered"):
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = (
                "trigger observed; no residual sufficient tools defect for a second "
                "functional edit; refusing marker-only A2/B2 padding"
            )
        else:
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = (
                f"after {max_trigger_reps} reps the distinctive trigger still did not "
                "appear (agent did not emit empty-keystroke waits followed by another "
                "command in those trials)"
            )

    record["branches"] = branches
    record["resumed_triggers"] = True
    record["max_trigger_reps"] = max_trigger_reps
    n_functional = sum(
        1
        for b in branches
        if b.get("functional_change") and b.get("triggered") and b.get("loaded_modules_root_ok")
    )
    if n_functional >= 2:
        record["status"] = "PASS"
        if all(b.get("continuation_status") == "skipped" for b in branches):
            record["early_stop_reason"] = (
                "two real round-1 branches triggered; continuations skipped because no "
                "residual sufficient tools evidence remained (no marker padding)"
            )
        else:
            record["early_stop_reason"] = ""
    else:
        record["status"] = "INCOMPLETE"
        record["early_stop_reason"] = (
            f"after resume, only {n_functional}/2 branches showed loaded+triggered "
            "functional edits"
        )

    # Attempt residual continuations if marked pending.
    for branch in branches:
        if branch.get("continuation_status") != "pending_residual":
            continue
        label = branch["label"]
        v1 = branch["v1_id"]
        findings = json.loads((run_dir / f"continuation_scan_{label}.json").read_text())
        residual = next(
            (
                f
                for f in sufficient_findings(findings)
                if f.get("kind") == "false_concat_warning_on_empty_keystrokes"
            ),
            None,
        )
        if residual is None:
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = "pending residual disappeared on rescan"
            continue
        parent_state = load_state(versions_root, v1)
        parent_src = (Path(parent_state.modules_root) / TARGET_REL).read_text()
        cont_evidence = _excerpt_for_planner(residual, parent_src)
        (run_dir / "plans" / f"continuation_evidence_{label}.txt").write_text(cont_evidence)
        cont_strategy = (
            f"Continuation for branch {label}. The round-1 edit did not fully clear the "
            f"evidenced false-concat case. Fix the residual ({residual.get('kind')}) with "
            "a NEW distinctive TRIGGER containing BASELINE_TOOLS_ and keep prior behavior."
        )
        cont_plan = _plan_one_edit(
            run_dir=run_dir,
            label=f"{label}2",
            source=parent_src,
            evidence=cont_evidence,
            prov=prov,
            strategy=cont_strategy,
            expected_trigger_substr="BASELINE_TOOLS_",
        )
        if not cont_plan["ok"] or cont_plan["trigger"] == branch["trigger"]:
            branch["continuation_status"] = "skipped"
            branch["continuation_reason"] = (
                f"residual existed but continuation plan failed/not distinct: "
                f"{cont_plan.get('editor_error')}"
            )
            continue
        v2 = f"H0_{label}2"
        plan = make_plan(
            parent_version_id=v1,
            locked_module=LOCKED,
            plan_text=cont_plan["plan_text"],
            candidate_label=f"{label}_cont",
            intended_files=[TARGET_REL],
        )
        attempt = apply_plan_patch(
            run_dir=run_dir,
            versions_root=versions_root,
            plan=plan,
            patch_text=cont_plan["diff_text"],
            child_version_id=v2,
        )
        if not attempt.applied or not attempt.child_version_id:
            branch["continuation_status"] = "apply-failure"
            branch["continuation_reason"] = attempt.reject_reason
            continue
        v2_state = load_state(versions_root, v2)
        cont_runs = []
        cont_meas = {
            "triggered": False,
            "trigger_hits": [],
            "loaded_modules_root_ok": False,
        }
        for rep in range(1, max_trigger_reps + 1):
            print(f"[dual-resume] eval {v2} {task} rep {rep}", flush=True)
            cont_runs.append(
                _eval_one(
                    cfg=cfg,
                    run_dir=run_dir,
                    version_id=v2,
                    modules_root=Path(v2_state.modules_root),
                    task=task,
                    rep=rep,
                    prov=prov,
                    temperature=temperature,
                )
            )
            cont_meas = _measure_trigger(
                run_dir=run_dir,
                version_id=v2,
                modules_root=Path(v2_state.modules_root),
                task=task,
                trigger=cont_plan["trigger"],
                reps=rep,
            )
            if cont_meas["triggered"]:
                break
        write_transition(
            run_dir,
            transition_id=f"{v1}_to_{v2}",
            parent_version_id=v1,
            child_version_id=v2,
            plan_id=plan.plan_id,
            attempt_id=attempt.attempt_id,
            parent_scores={task: float(_mean(branch.get("raw_rewards") or []) or 0.0)},
            child_scores=aggregate_task_means(cont_runs),
        )
        branch.update(
            {
                "continuation_status": "applied",
                "continuation_reason": f"residual kind={residual.get('kind')}",
                "v2_id": v2,
                "cont_plan_path": str(run_dir / "plans" / f"{plan.plan_id}.txt"),
                "cont_diff_path": str(run_dir / "patches" / f"{attempt.attempt_id}.diff"),
                "cont_trigger": cont_plan["trigger"],
                "cont_triggered": cont_meas["triggered"],
                "cont_trigger_hits": cont_meas["trigger_hits"][:12],
                "cont_raw_rewards": [r.reward for r in cont_runs],
                "cont_loaded_ok": cont_meas["loaded_modules_root_ok"],
            }
        )

    write_json(record_path, record)
    write_json(
        run_dir / "summary.json",
        {"status": record["status"], "run_id": record.get("run_id"), "run_dir": str(run_dir)},
    )
    _write_dual_report(resolve_path(cfg, "reports_root") / "research1_dual_branch.md", record)
    mark_step(run_dir, "dual_triggers_resumed")
    return (EXIT_PASS if record["status"] == "PASS" else EXIT_INCOMPLETE), record


def resume_dual_triggers(run_dir: Path, config_path: Path | None = None, **kwargs: Any) -> tuple[int, dict[str, Any]]:
    return resume_dual_branch_triggers(run_dir, config_path, **kwargs)


def probe_version_trigger(modules_root: Path, expected_substr: str) -> dict[str, Any]:
    """Deterministic probe: feed the evidenced empty-mid-batch pattern into BaselineTools."""
    import importlib.util
    import sys

    mr_src = "/home/jinxu/arms_cache/modularrsi/ModularRSI/src"
    if mr_src not in sys.path:
        sys.path.insert(0, mr_src)
    path = Path(modules_root) / TARGET_REL
    spec = importlib.util.spec_from_file_location(f"probe_{path.parent.parent.name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tools = mod.BaselineTools(parser_name="json")
    payload = {
        "analysis": "probe",
        "plan": "probe empty wait mid-batch",
        "commands": [
            {"keystrokes": "echo hi\n", "duration": 0.5},
            {"keystrokes": "", "duration": 5.0},
            {"keystrokes": "echo bye\n", "duration": 0.5},
        ],
        "task_complete": False,
    }
    result = tools.parse_llm_response(json.dumps(payload))
    warning = result.warning or ""
    cmds = [(c.keystrokes, float(c.duration_sec)) for c in (result.commands or [])]
    return {
        "modules_root": str(modules_root),
        "expected_substr": expected_substr,
        "warning": warning,
        "commands": cmds,
        "triggered": expected_substr in warning,
    }
