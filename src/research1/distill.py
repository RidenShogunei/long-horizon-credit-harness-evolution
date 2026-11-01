"""Plan-layer teacher data: trajectory + component snapshot → modification plan.

The evolver step is two layers. This module is the PLAN layer. It calls a
frozen chat model and stores the prompt, the raw reply, and the parsed plan
(target, intent, optional trigger). Score deltas are not the teacher label.

An edit step can consume one parsed plan. The fixed editor diffs a replacement
file, rejects comment-only edits, and applies the unified diff into an
independent version directory.

The historical warning-replay collection loops are legacy candidate-discovery
tools. Their local positive/negative probe labels are not verified training
targets or long-horizon values. Their exports stay in the auxiliary pool;
research1.plan_init owns behavioral validation and training admission.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research1 import EXIT_FAIL, EXIT_PASS
from research1.adapters import merge_no_proxy
from research1.causal import (
    FILE_BEGIN,
    FILE_END,
    _obs_text,
    editor_from_replacement,
    false_concat_findings,
    probe_parse_warning,
    probe_version_trigger,
)
from research1.config import REPO_ROOT, load_pilot, resolve_path
from research1.contracts import write_json
from research1.planning import apply_plan_patch, make_plan
from research1.versions import materialize_version, sha256_file

SCHEMA_ID = "research1.distill.plan_io.v1"
VERIFIED_SCHEMA_ID = "research1.distill.plan_verified.v1"
PLAN_TEMPERATURE = 0.0
MODULE_RELPATH = "tools/baseline.py"
VERIFY_KIND = "false_concat_warning_on_empty_keystrokes"
_METHOD_RE = re.compile(r"^    (?:async )?def (\w+)\(")
_KEEP_METHOD = re.compile(
    r"filter|false|suppress|normalize|empty_wait|parse_llm_response|^execute$",
    re.I,
)


def relevant_component_excerpt(source: str, *, limit: int = 12000) -> str:
    """Keep the tools methods a plan would actually modify."""
    lines = source.splitlines()
    chunks: list[str] = []
    seen: set[str] = set()

    def add(block: str) -> None:
        text = block.strip("\n")
        if not text or text in seen:
            return
        seen.add(text)
        chunks.append(text)

    for i, line in enumerate(lines):
        if "FALSE_CONCAT" in line or "EMPTY_WAIT" in line or "research1-p2b-marker" in line:
            add("\n".join(lines[i : i + 8]))

    i = 0
    while i < len(lines):
        match = _METHOD_RE.match(lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1)
        j = i + 1
        while j < len(lines) and not _METHOD_RE.match(lines[j]) and not lines[j].startswith("def "):
            j += 1
        if _KEEP_METHOD.search(name):
            add("\n".join(lines[i:j]))
        i = j

    text = "\n\n".join(chunks).strip() + ("\n" if chunks else "")
    if len(text) > limit:
        text = text[:limit] + "\n# ... excerpt truncated ...\n"
    return text


def build_plan_prompt(plan_input: dict[str, Any]) -> str:
    """Prompt actually sent. It embeds the recorded input fields."""
    return (
        "You are the PLAN module of a layered evolver. You are not the editor "
        "and you are not being scored. Read the error-trajectory excerpt and the "
        "current component snapshot. Output a modification plan for that "
        "component only.\n"
        "Propose a change only when the excerpt implicates this component. If it "
        "does not, set TARGET to none and TRIGGER to NONE, and say why in INTENT.\n"
        "Do not emit a replacement file. Do not add comment-only markers. Do not "
        "propose a change whose purpose is to move a score.\n"
        "Reply in exactly this shape:\n"
        "TARGET: <relative path such as tools/baseline.py, or none>\n"
        "INTENT:\n"
        "<what is wrong in the excerpt and what the edit should change>\n"
        "TRIGGER: <one distinctive feedback string emitted only when that "
        "condition occurs, or NONE>\n"
        "\n"
        f"Finding kind: {plan_input.get('finding_kind')}\n"
        f"Finding detail: {plan_input.get('finding_detail')}\n"
        f"Source trajectory: {plan_input.get('source_trajectory_path')}\n"
        "\n"
        "Error-trajectory excerpt:\n"
        f"{plan_input.get('error_trajectory_excerpt')}\n"
        "\n"
        "Current component snapshot:\n"
        f"path: {plan_input.get('module_snapshot_path')}\n"
        f"sha256: {plan_input.get('module_snapshot_sha256')}\n"
        f"relpath: {plan_input.get('module_relpath')}\n"
        "\n"
        "Relevant component excerpt:\n"
        f"{plan_input.get('module_excerpt')}\n"
    )


def parse_plan_output(text: str) -> dict[str, Any]:
    """Parse TARGET / INTENT / TRIGGER. Does not invent missing fields."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    target_m = re.search(r"^TARGET:\s*(.+)$", cleaned, re.M)
    if not target_m:
        raise ValueError("plan output missing TARGET: line")
    intent_m = re.search(r"^INTENT:\s*(.*)$", cleaned, re.M)
    if not intent_m:
        raise ValueError("plan output missing INTENT: line")
    rest = cleaned[intent_m.end() :]
    trig_m = re.search(r"^TRIGGER:\s*(.*)$", rest, re.M)
    if trig_m:
        intent = (intent_m.group(1) + "\n" + rest[: trig_m.start()]).strip()
        trigger_raw = trig_m.group(1).strip()
    else:
        intent = (intent_m.group(1) + "\n" + rest).strip()
        trigger_raw = ""
    if not intent:
        raise ValueError("plan output missing intent text")
    target = target_m.group(1).strip()
    if target.lower() in {"none", "null", "n/a"}:
        target = "none"
    trigger: str | None
    if not trigger_raw or trigger_raw.upper() in {"NONE", "NULL", "N/A"}:
        trigger = None
    else:
        trigger = trigger_raw
    return {"target": target, "intent": intent, "trigger": trigger}


def validate_plan_io_record(doc: dict[str, Any]) -> list[str]:
    """Return schema errors. Empty list means the record is usable teacher data or a recorded failure."""
    errors: list[str] = []
    if doc.get("schema") != SCHEMA_ID:
        errors.append(f"schema must be {SCHEMA_ID}")
    if not str(doc.get("record_id") or "").strip():
        errors.append("missing record_id")
    status = doc.get("status")
    if status not in {"ok", "error"}:
        errors.append("status must be ok or error")
    inp = doc.get("input")
    if not isinstance(inp, dict):
        errors.append("missing input object")
        inp = {}
    for key in (
        "error_trajectory_excerpt",
        "module_snapshot_path",
        "module_snapshot_sha256",
        "module_excerpt",
    ):
        if not str(inp.get(key) or "").strip():
            errors.append(f"input missing {key}")
    digest = str(inp.get("module_snapshot_sha256") or "")
    if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append("module_snapshot_sha256 is not 64 hex chars")
    meta = doc.get("metadata")
    if not isinstance(meta, dict):
        errors.append("missing metadata object")
        meta = {}
    for key in (
        "provider",
        "model",
        "temperature",
        "task",
        "source_trajectory_path",
        "timestamp",
    ):
        if meta.get(key) in (None, ""):
            errors.append(f"metadata missing {key}")
    if "temperature" in meta and meta.get("temperature") not in (None, ""):
        if not isinstance(meta.get("temperature"), (int, float)):
            errors.append("metadata.temperature must be a number")
    if status == "error" and not str(doc.get("error") or "").strip():
        errors.append("error status missing error")
    if status == "ok":
        prompt = str(doc.get("planner_prompt") or "")
        raw = str(doc.get("planner_raw") or "")
        if not prompt.strip():
            errors.append("ok record missing planner_prompt")
        if not raw.strip():
            errors.append("ok record missing planner_raw")
        excerpt = str(inp.get("error_trajectory_excerpt") or "")
        if excerpt and excerpt[:80] not in prompt:
            errors.append("planner_prompt does not contain the trajectory excerpt")
        if digest and digest not in prompt:
            errors.append("planner_prompt does not contain the module hash")
        parsed = doc.get("parsed_plan")
        if not isinstance(parsed, dict):
            errors.append("ok record missing parsed_plan")
        else:
            if not str(parsed.get("target") or "").strip():
                errors.append("parsed_plan missing target")
            if not str(parsed.get("intent") or "").strip():
                errors.append("parsed_plan missing intent")
            if "trigger" not in parsed:
                errors.append("parsed_plan missing trigger key")
    edit = doc.get("edit")
    if isinstance(edit, dict) and edit.get("status") == "applied":
        if not str(edit.get("plan_path") or "").strip():
            errors.append("applied edit missing plan_path")
        if not str(edit.get("diff_path") or "").strip():
            errors.append("applied edit missing diff_path")
    return errors


def _traj(root: Path, needle: str) -> Path | None:
    if not root.is_dir():
        return None
    hits = [p for p in root.rglob("trajectory.json") if needle in str(p)]
    hits.sort()
    return hits[0] if hits else None


def _assertion_excerpt(text: str) -> str:
    marker = "AssertionError:"
    idx = text.find(marker)
    if idx < 0:
        return text[-1500:]
    return text[max(0, idx - 500) : idx + 900]


def _coverage_wasm_case(repo: Path, snapshot: Path) -> dict[str, Any] | None:
    run = repo / "runs" / "causal_20261030_022523"
    wasm_stdout = None
    if run.is_dir():
        for path in sorted(run.rglob("verifier/test-stdout.txt")):
            if "wasm-crackme-reverse" in str(path):
                wasm_stdout = path
                break
    if wasm_stdout is None or not snapshot.is_file():
        return None
    trial_log = wasm_stdout.parents[1] / "trial.log"
    tool_lines: list[str] = []
    if trial_log.is_file():
        for line in trial_log.read_text(errors="replace").splitlines():
            if "tools:baseline" in line or "step " in line and "error=" in line:
                tool_lines.append(line.strip())
    reward_path = wasm_stdout.parent / "reward.txt"
    reward = reward_path.read_text(errors="replace").strip() if reward_path.is_file() else ""
    assertion = _assertion_excerpt(wasm_stdout.read_text(errors="replace"))
    excerpt = (
        f"Coverage H0 trial. Verifier reward={reward or 'missing'}.\n"
        "trial.log lines that mention tools or a parse error flag:\n"
        + "\n".join(tool_lines[:12])
        + "\n\nVerifier assertion excerpt:\n"
        + assertion
    )
    return {
        "case_id": "coverage_wasm_assertion",
        "task": "wasm-crackme-reverse",
        "source_run": "causal_20261030_022523",
        "snapshot_path": snapshot,
        "finding": {
            "kind": "verifier_assertion_tools_parse_clean",
            "detail": (
                "wasm-crackme-reverse verifier reward 0 on test_flag_matches_wasm_output "
                "(passphrase not extracted from a WAT file). tools:baseline lines in "
                "trial.log report parse error=False and execute ok."
            ),
            "excerpt": excerpt[:4000],
            "path": str(wasm_stdout),
            "step_id": None,
        },
    }


def select_distill_cases(repo: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    """Ground cases in local branches-cron, causal-edit, and coverage artifacts."""
    repo = repo.resolve()
    branches = repo / "runs" / "branches_20261029_134412"
    causal_edit = repo / "runs" / "causal_edit_20261030_025402"
    coverage = repo / "runs" / "causal_20261030_022523"
    h0 = causal_edit / "versions" / "H0" / "modules" / MODULE_RELPATH
    h1 = causal_edit / "versions" / "H1" / "modules" / MODULE_RELPATH
    cov_h0 = coverage / "versions" / "H0" / "modules" / MODULE_RELPATH
    specs: list[tuple[str, Path | None, Path, str, str, int | None]] = []
    if h0.is_file():
        specs.append(
            (
                "branches_cron_h0_rep1",
                _traj(branches, "/evals/H0/cron-env-debug/rep_1/"),
                h0,
                "cron-env-debug",
                "branches_20261029_134412",
                9,
            )
        )
        specs.append(
            (
                "branches_cron_h0_rep2",
                _traj(branches, "/evals/H0/cron-env-debug/rep_2/"),
                h0,
                "cron-env-debug",
                "branches_20261029_134412",
                8,
            )
        )
        specs.append(
            (
                "causal_edit_h0_cron",
                _traj(causal_edit, "/evals/H0/cron-env-debug/rep_1/"),
                h0,
                "cron-env-debug",
                "causal_edit_20261030_025402",
                9,
            )
        )
    cases: list[dict[str, Any]] = []
    for case_id, traj, snapshot, task, source_run, step_id in specs:
        if traj is None or not traj.is_file():
            continue
        found = false_concat_findings(traj)
        if step_id is not None:
            matched = [row for row in found if row.get("step_id") == step_id]
            if matched:
                found = matched
        if not found:
            continue
        cases.append(
            {
                "case_id": case_id,
                "task": task,
                "source_run": source_run,
                "snapshot_path": snapshot,
                "finding": found[0],
            }
        )
    if cov_h0.is_file():
        wasm = _coverage_wasm_case(repo, cov_h0)
        if wasm is not None:
            cases.append(wasm)
    if h1.is_file():
        traj = _traj(branches, "/evals/H0/cron-env-debug/rep_1/")
        if traj is not None:
            found = false_concat_findings(traj)
            step9 = [row for row in found if row.get("step_id") == 9]
            if step9:
                cases.append(
                    {
                        "case_id": "branches_cron_against_h1_snapshot",
                        "task": "cron-env-debug",
                        "source_run": "branches_20261029_134412",
                        "snapshot_path": h1,
                        "finding": step9[0],
                    }
                )
    return cases[:limit]


def _plan_input(case: dict[str, Any]) -> dict[str, Any]:
    snapshot: Path = case["snapshot_path"]
    source = snapshot.read_text()
    finding = case["finding"]
    return {
        "error_trajectory_excerpt": str(finding.get("excerpt") or "")[:4000],
        "finding_kind": finding.get("kind"),
        "finding_detail": finding.get("detail"),
        "source_trajectory_path": str(finding.get("path") or ""),
        "step_id": finding.get("step_id"),
        "module_snapshot_path": str(snapshot.resolve()),
        "module_snapshot_sha256": sha256_file(snapshot),
        "module_relpath": MODULE_RELPATH,
        "module_excerpt": relevant_component_excerpt(source),
    }


def _tongji_chat(prompt: str, *, max_tokens: int, role: str, json_output: bool = False,
                 reasoning_effort: str | None = None,
                 allow_http_fallback: bool = True) -> tuple[str, dict[str, Any]]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from llm_provider_lib import record_usage, resolve_provider

    resolved = resolve_provider(mode="tongji_agi")
    no_proxy = merge_no_proxy(os.environ.get("NO_PROXY") or os.environ.get("no_proxy"))
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    if "agi.tongji.edu.cn" not in no_proxy:
        raise RuntimeError("NO_PROXY missing agi.tongji.edu.cn")
    url = resolved.api_base.rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {
        "model": resolved.api_model_id,
        "temperature": PLAN_TEMPERATURE,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [
            {
                "role": "system",
                "content": "Follow the requested output shape exactly.",
            },
            {"role": "user", "content": prompt},
        ],
    }
    if json_output:
        body['response_format'] = {'type': 'json_object'}
    if reasoning_effort is not None:
        body.pop('thinking', None)
        body['reasoning_effort'] = reasoning_effort
    headers = {
        "Authorization": f"Bearer {resolved.api_key}",
        "Content-Type": "application/json",
    }

    def _post(payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode())

    http_attempts = 1
    try:
        data = _post(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:800]
        if allow_http_fallback and ("thinking" in detail.lower() or exc.code == 400):
            body.pop("thinking", None)
            http_attempts += 1
            try:
                data = _post(body)
            except urllib.error.HTTPError as exc2:
                detail2 = exc2.read().decode(errors="replace")[:800]
                raise RuntimeError(f"planner HTTP {exc2.code}: {detail2}") from exc2
        else:
            raise RuntimeError(f"planner HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"planner connection failed: {exc.reason}") from exc
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content, raw_field = _message_text(message)
    usage = data.get("usage") or {}
    try:
        record_usage(
            resolved.provider_id,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=usage.get("total_tokens"),
            meta={"role": role},
        )
    except Exception:
        pass
    meta = {
        "provider": resolved.provider_id,
        "model": resolved.api_model_id,
        "temperature": PLAN_TEMPERATURE,
        "usage": usage,
        "raw_field": raw_field,
        "finish_reason": choice.get("finish_reason"),
        "http_attempts": http_attempts,
        "reasoning_effort": reasoning_effort,
    }
    if not str(content).strip():
        keys = ",".join(sorted(str(k) for k in message.keys()))
        raise RuntimeError(
            "planner returned empty content "
            f"(finish={choice.get('finish_reason')}, message_keys={keys})"
        )
    return str(content), meta


def _message_text(message: dict[str, Any]) -> tuple[str, str]:
    """Prefer assistant content. Fall back to reasoning fields the gateway fills."""
    content = message.get("content")
    if isinstance(content, list):
        bits: list[str] = []
        for part in content:
            if isinstance(part, str):
                bits.append(part)
            elif isinstance(part, dict):
                bits.append(str(part.get("text") or part.get("content") or ""))
        content = "".join(bits)
    if isinstance(content, str) and content.strip():
        return content, "content"
    for key in ("reasoning_content", "reasoning"):
        alt = message.get(key)
        if isinstance(alt, str) and alt.strip():
            return alt, key
    return "", ""


def _record_id(case_id: str, plan_input: dict[str, Any]) -> str:
    blob = "|".join(
        [
            case_id,
            str(plan_input.get("source_trajectory_path")),
            str(plan_input.get("step_id")),
            str(plan_input.get("module_snapshot_sha256")),
        ]
    )
    return "plan_io_" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def _ensure_newline(body: str) -> str:
    body = body.strip("\n")
    lines = body.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines)
    if body and not body.endswith("\n"):
        body += "\n"
    return body


def _looks_like_tools_module(body: str) -> bool:
    return (
        "class BaselineTools" in body
        and "def parse_llm_response" in body
        and "def register" in body
    )


def extract_file_body(text: str) -> str:
    """Pull a replacement module out of an edit-module reply."""
    if FILE_BEGIN in text and FILE_END in text:
        body = text.split(FILE_BEGIN, 1)[1].split(FILE_END, 1)[0]
        if body.startswith("\n"):
            body = body[1:]
        return _ensure_newline(body)
    for block in re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S):
        if _looks_like_tools_module(block):
            return _ensure_newline(block)
    if _looks_like_tools_module(text):
        start = text.find('"""')
        future = text.find("from __future__")
        klass = text.find("class BaselineTools")
        candidates = [i for i in (start, future, klass) if i >= 0]
        body = text[min(candidates) :] if candidates else text
        return _ensure_newline(body)
    raise ValueError("edit output missing file sentinels")


def build_edit_prompt(*, plan: dict[str, Any], source: str) -> str:
    trigger = plan.get("trigger") or "NONE"
    return (
        "You are the edit module. Apply the modification plan to the current "
        "tools/baseline.py. Return the complete replacement file. The fixed "
        "editor will reject comment-only diffs, so the change must alter "
        "executable behavior.\n"
        "If TRIGGER is not NONE, that exact string must be a new Python string "
        "constant and must be emitted only for the condition the plan names.\n"
        "Do not add comment markers such as research1-p2b-marker.\n"
        "Preserve command keystrokes, durations, ordering, parser errors, and "
        "task-completion fields unless the plan explicitly requires changing them. "
        "When changing diagnostic feedback, preserve every unrelated diagnostic, "
        "including when several warnings share one string or command batch. "
        "Filter only the specific diagnostic item supported by the plan; never "
        "drop a whole multi-warning string because one item matches.\n"
        "The first line of your reply must be the begin sentinel. "
        "Do not explain the edit before it.\n"
        "Reply with the file between these sentinels and no other text:\n"
        f"{FILE_BEGIN}\n"
        "<complete file>\n"
        f"{FILE_END}\n"
        "\n"
        f"TARGET: {plan.get('target')}\n"
        "INTENT:\n"
        f"{plan.get('intent')}\n"
        f"TRIGGER: {trigger}\n"
        "\n"
        "Current file:\n"
        f"{source}"
    )


def _apply_edit(
    *,
    run_dir: Path,
    record_path: Path,
    plan: dict[str, Any],
    snapshot: Path,
    replacement: str,
) -> dict[str, Any]:
    before = snapshot.read_text()
    trigger = str(plan.get("trigger") or "")
    diff, err = editor_from_replacement(before=before, replacement=replacement, trigger=trigger)
    linkage: dict[str, Any] = {
        "plan_path": str(record_path),
        "status": "rejected",
        "reject_reason": err,
        "diff_path": None,
        "child_version_id": None,
        "harbor_eval": False,
        "note": (
            "Fixed editor only. No Harbor trial. Teacher data remains the plan "
            "input and parsed plan, not a score."
        ),
    }
    if diff is None:
        return linkage
    modules_root = snapshot.parent.parent
    versions_root = run_dir / "versions"
    parent_id = "parent_" + sha256_file(snapshot)[:12]
    if not (versions_root / parent_id / "state.json").is_file():
        materialize_version(
            versions_root=versions_root,
            version_id=parent_id,
            source_modules=modules_root,
            parent_id=None,
            note="distill parent snapshot; independent of the source run",
        )
    marker = "plan_io_"
    name = record_path.name.removesuffix(".json")
    child_id = (
        "edit_" + name.split(marker, 1)[1][:12]
        if marker in name
        else "edit_" + hashlib.sha256(name.encode()).hexdigest()[:12]
    )
    edit_plan = make_plan(
        parent_version_id=parent_id,
        locked_module="tools",
        plan_text=str(plan.get("intent") or ""),
        candidate_label="distill",
        intended_files=[MODULE_RELPATH],
    )
    attempt = apply_plan_patch(
        run_dir=run_dir,
        versions_root=versions_root,
        plan=edit_plan,
        patch_text=diff,
        child_version_id=child_id,
        outcome_class=None,
    )
    diff_path = run_dir / "patches" / f"{attempt.attempt_id}.diff"
    linkage.update(
        {
            "status": "applied" if attempt.applied else "rejected",
            "reject_reason": attempt.reject_reason,
            "diff_path": str(diff_path) if diff_path.is_file() else None,
            "child_version_id": attempt.child_version_id,
            "stored_plan_path": str(run_dir / "plans" / f"{edit_plan.plan_id}.json"),
            "attempt_id": attempt.attempt_id,
            "outcome_class": attempt.outcome_class,
        }
    )
    return linkage


def _edit_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    case = str((record.get("metadata") or {}).get("case_id") or "")
    path = str((record.get("input") or {}).get("module_snapshot_path") or "")
    return (1 if "wasm" in case else 0, 1 if "/versions/H1/" in path else 0, case)


def _eligible_for_edit(record: dict[str, Any]) -> bool:
    if record.get("status") != "ok":
        return False
    parsed = record.get("parsed_plan") or {}
    target = str(parsed.get("target") or "")
    trigger = parsed.get("trigger")
    if target in {"", "none"} or not trigger:
        return False
    if not target.endswith(MODULE_RELPATH) and target not in {MODULE_RELPATH, "tools"}:
        return False
    return True


_OLD_WARNING_MARKERS = (
    "should end with newline",
    "concatenated together on the same line",
)


def filter_empty_keystroke_warnings(warning: Any, commands: list[Any]) -> Any:
    """Reference repair: remove only standalone false empty-wait diagnostics.

    Keep all non-target warning text and container types. Unknown diagnostic
    formats or command shapes are retained conservatively. This pure function
    does not change the command objects or historical module snapshots.
    """
    if isinstance(warning, (list, tuple)):
        kept = []
        for item in warning:
            filtered = filter_empty_keystroke_warnings(item, commands)
            if filtered is not None:
                kept.append(filtered)
        return type(warning)(kept) if kept else None
    if not isinstance(warning, str) or not warning:
        return warning
    kept_lines = []
    for line in warning.splitlines(keepends=True):
        match = re.fullmatch(
            r"\s*(?:-\s*)?Command\s+(\d+)\s+should end with newline when followed "
            r"by another command\.(?:\s+Otherwise the two commands will be "
            r"concatenated together on the same line\.)?\s*",
            line.rstrip("\r\n"),
            re.IGNORECASE,
        )
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(commands):
                command = commands[index]
                if isinstance(command, dict):
                    keystrokes = command.get("keystrokes")
                elif isinstance(command, (list, tuple)) and command:
                    keystrokes = command[0]
                else:
                    keystrokes = getattr(command, "keystrokes", None)
                if keystrokes == "":
                    continue
        kept_lines.append(line)
    result = "".join(kept_lines)
    return result if result.strip() else None


def trigger_repeats_false_warning(trigger: str) -> bool:
    """True when the plan copied the false warning instead of naming a new phrase."""
    text = (trigger or "").lower()
    return any(marker in text for marker in _OLD_WARNING_MARKERS)


def score_false_concat_probe(warning: str, trigger: str) -> dict[str, Any]:
    """Pass when the false warning is gone.

    A trigger that only repeats the old warning is not required to appear.
    A new phrase must show up in the probe warning.
    """
    text = warning or ""
    expected = (trigger or "").strip()
    present = any(marker in text for marker in _OLD_WARNING_MARKERS)
    repeats = trigger_repeats_false_warning(expected)
    emitted = bool(expected) and expected in text
    if present:
        passed = False
        reason = "false warning is still present"
    elif expected and not repeats and not emitted:
        passed = False
        reason = "false warning is gone but the new trigger was not emitted"
    else:
        passed = True
        reason = "false warning is gone"
    return {
        "warning": text,
        "expected_substr": expected,
        "false_concat_gone": not present,
        "trigger_repeats_warning": repeats,
        "triggered": emitted,
        "passed": passed,
        "reason": reason,
    }


def verify_false_concat_edit(modules_root: Path, trigger: str) -> dict[str, Any]:
    """Probe the archived empty-wait pattern against an edited tools module."""
    expected = str(trigger or "").strip()
    probe = probe_version_trigger(modules_root, expected)
    probe.update(score_false_concat_probe(str(probe.get("warning") or ""), expected))
    return probe


def _snapshot_already_fixed(snapshot: Path) -> bool:
    """True when the current tools module already suppresses the false warning."""
    try:
        probe = probe_version_trigger(snapshot.parent.parent, "should end with newline")
    except Exception:
        return False
    return "should end with newline" not in str(probe.get("warning") or "")


def _label_none(record: dict[str, Any]) -> dict[str, Any]:
    """A none-target is a negative only when the snapshot already fixed the defect."""
    parsed = record.get("parsed_plan") or {}
    finding = str((record.get("input") or {}).get("finding_kind") or "")
    if str(parsed.get("target") or "") != "none" or finding != VERIFY_KIND:
        return {
            "schema": VERIFIED_SCHEMA_ID,
            "label": "unlabeled",
            "reason": "plan is not a none-target on the verified finding",
        }
    snapshot = Path(str((record.get("input") or {}).get("module_snapshot_path") or ""))
    if not snapshot.is_file() or not _snapshot_already_fixed(snapshot):
        return {
            "schema": VERIFIED_SCHEMA_ID,
            "label": "unlabeled",
            "reason": "none-target on a snapshot that still emits the false warning",
        }
    return {
        "schema": VERIFIED_SCHEMA_ID,
        "label": "negative",
        "reason": "none-target and the snapshot already suppresses the false warning",
        "probe": "parent_snapshot",
    }


def _verify_applied_edit(run_dir: Path, record: dict[str, Any], linkage: dict[str, Any]) -> dict[str, Any]:
    child = str(linkage.get("child_version_id") or "")
    modules = run_dir / "versions" / child / "modules"
    trigger = str((record.get("parsed_plan") or {}).get("trigger") or "")
    try:
        probe = verify_false_concat_edit(modules, trigger)
    except Exception as exc:  # noqa: BLE001
        probe = {"passed": False, "error": str(exc), "triggered": False, "false_concat_gone": False}
    if probe.get("error"):
        label = "unlabeled"
        reason = "probe failed before it could score the edit"
    elif probe.get("passed"):
        label = "positive"
        reason = str(probe.get("reason") or "false warning is gone")
    else:
        label = "negative"
        reason = str(probe.get("reason") or "edit applied but the probe did not confirm the plan")
    verified = {
        "schema": VERIFIED_SCHEMA_ID,
        "label": label,
        "reason": reason,
        "probe": probe,
        "child_version_id": child,
    }
    probe_path = run_dir / "probes" / f"{record.get('record_id')}.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(probe_path, verified)
    verified["probe_path"] = str(probe_path)
    return verified


def _record_path_for(run_dir: Path, record: dict[str, Any]) -> Path | None:
    hinted = Path(str((record.get("edit") or {}).get("plan_path") or ""))
    if hinted.is_file():
        return hinted
    matches = sorted((run_dir / "records").glob(f"*{record['record_id']}.json"))
    return matches[0] if matches else None


def _edit_one(
    run_dir: Path,
    record: dict[str, Any],
    verifier: Any | None = None,
) -> dict[str, Any]:
    """Edit one eligible plan, then probe the child version."""
    record_path = _record_path_for(run_dir, record)
    if record_path is None:
        return {"record_id": record.get("record_id"), "status": "error", "reject_reason": "record path missing"}
    parsed = record["parsed_plan"]
    snapshot = Path(record["input"]["module_snapshot_path"])
    print(f"[distill-plan] edit {record['record_id']}", flush=True)
    edit_raw = ""
    edit_meta: dict[str, Any] = {}
    raw_path = run_dir / "edits" / f"{record['record_id']}_edit_raw.txt"
    try:
        edit_prompt = build_edit_prompt(plan=parsed, source=snapshot.read_text())
        edit_raw, edit_meta = _tongji_chat(edit_prompt, max_tokens=16000, role="distill_editor_llm")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(edit_raw)
        replacement = extract_file_body(edit_raw)
    except Exception as exc:  # noqa: BLE001
        if edit_raw:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(edit_raw)
        linkage = {
            "plan_path": str(record_path),
            "status": "error",
            "reject_reason": str(exc),
            "error": str(exc),
            "harbor_eval": False,
            "edit_raw_path": str(raw_path) if edit_raw else "",
        }
        record["edit"] = linkage
        record["verified"] = {
            "schema": VERIFIED_SCHEMA_ID,
            "label": "negative",
            "reason": "edit module failed before a probe",
        }
        write_json(record_path, record)
        print(f"[distill-plan] edit error: {exc}", flush=True)
        return {
            "record_id": record["record_id"],
            "plan_path": str(record_path),
            "status": "error",
            "reject_reason": str(exc),
            "label": "negative",
            "diff_path": None,
            "child_version_id": None,
        }
    linkage = _apply_edit(
        run_dir=run_dir,
        record_path=record_path,
        plan=parsed,
        snapshot=snapshot,
        replacement=replacement,
    )
    linkage["edit_provider"] = edit_meta.get("provider")
    linkage["edit_model"] = edit_meta.get("model")
    linkage["editor"] = "fixed_patch_applier_v1"
    linkage["replacement_via"] = "edit module consumed parsed plan"
    linkage["edit_raw_path"] = str(raw_path)
    if linkage.get("status") == "applied":
        verify = verifier or _verify_applied_edit
        verified = verify(run_dir, record, linkage)
    else:
        verified = {
            "schema": VERIFIED_SCHEMA_ID,
            "label": "negative",
            "reason": linkage.get("reject_reason") or "edit rejected",
        }
    linkage["verified_label"] = verified["label"]
    record["edit"] = linkage
    record["verified"] = verified
    write_json(record_path, record)
    print(
        f"[distill-plan] edit {linkage.get('status')} label={verified['label']}",
        flush=True,
    )
    return {
        "record_id": record["record_id"],
        "plan_path": linkage.get("plan_path"),
        "status": linkage.get("status"),
        "reject_reason": linkage.get("reject_reason"),
        "diff_path": linkage.get("diff_path"),
        "child_version_id": linkage.get("child_version_id"),
        "label": verified["label"],
        "probe_path": verified.get("probe_path"),
    }


def _attempt_edits(
    run_dir: Path,
    records: list[dict[str, Any]],
    *,
    max_attempts: int = 2,
    verify_all: bool = False,
) -> dict[str, Any] | None:
    """Edit eligible plans.

    The default keeps the single edit proof. ``verify_all`` edits every
    eligible plan and probes each child; that is the collection loop.
    """
    summary: dict[str, Any] | None = None
    attempts = 0
    verified_rows: list[dict[str, Any]] = []
    ordered = sorted(
        (row for row in records if _eligible_for_edit(row)),
        key=_edit_sort_key,
    )
    for record in ordered:
        edit = record.get("edit") or {}
        if edit.get("status") == "applied" and record.get("verified"):
            row = {
                "record_id": record.get("record_id"),
                "plan_path": edit.get("plan_path"),
                "status": "applied",
                "reject_reason": "",
                "diff_path": edit.get("diff_path"),
                "child_version_id": edit.get("child_version_id"),
                "label": (record.get("verified") or {}).get("label"),
            }
            verified_rows.append(row)
            if not verify_all:
                return row
            continue
        if not verify_all and attempts >= max_attempts:
            break
        attempts += 1
        row = _edit_one(run_dir, record)
        verified_rows.append(row)
        summary = row
        if not verify_all and row.get("status") == "applied":
            break
    if verify_all:
        return {
            "mode": "verify_all",
            "attempted": attempts,
            "rows": verified_rows,
            "positive": sum(1 for row in verified_rows if row.get("label") == "positive"),
            "negative": sum(1 for row in verified_rows if row.get("label") == "negative"),
        }
    return summary


def _metadata(
    *,
    chat_meta: dict[str, Any] | None,
    task: str,
    source_path: str,
    case_id: str,
    source_run: str,
) -> dict[str, Any]:
    chat_meta = chat_meta or {}
    return {
        "provider": chat_meta.get("provider") or "tongji_agi",
        "model": chat_meta.get("model") or "",
        "temperature": PLAN_TEMPERATURE,
        "task": task,
        "source_trajectory_path": source_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "source_run": source_run,
        "usage": chat_meta.get("usage") or {},
        "raw_field": chat_meta.get("raw_field") or "",
        "data_status": "legacy_candidate_only",
        "training_eligible": False,
        "long_horizon_label": None,
    }


def collect_verified_cases(repo: Path) -> list[dict[str, Any]]:
    """One case per trajectory step that shows the false empty-keystroke warning.

    The snapshot is the tools module that produced the trajectory when that
    run kept a version tree, otherwise the frozen H0 tools module.
    """
    runs = repo / "runs"
    if not runs.is_dir():
        return []
    default_snapshot = (
        runs / "causal_edit_20261030_025402" / "versions" / "H0" / "modules" / MODULE_RELPATH
    )
    cases: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for traj in sorted(runs.rglob("trajectory.json")):
        if "132345" in traj.parts or "distill_plan_" in str(traj):
            continue
        try:
            rel = traj.relative_to(runs)
        except ValueError:
            continue
        source_run = rel.parts[0]
        version_id = ""
        if "versions" in rel.parts:
            version_id = rel.parts[rel.parts.index("versions") + 1]
        elif "evals" in rel.parts:
            version_id = rel.parts[rel.parts.index("evals") + 1]
        snapshot = runs / source_run / "versions" / version_id / "modules" / MODULE_RELPATH
        if not snapshot.is_file():
            snapshot = default_snapshot
        if not snapshot.is_file():
            continue
        task = ""
        for part in rel.parts:
            if part.startswith("task_") and "_" in part[5:]:
                task = part.split("_", 2)[-1]
                break
        if not task and "cron-env-debug" in str(traj):
            task = "cron-env-debug"
        for finding in false_concat_findings(traj):
            key = (str(traj), finding.get("step_id"))
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                {
                    "case_id": f"{source_run}_{version_id or 'h0'}_s{finding.get('step_id')}",
                    "task": task or source_run,
                    "source_run": source_run,
                    "snapshot_path": snapshot,
                    "finding": finding,
                }
            )
    return cases


def warning_findings(traj_path: Path) -> list[dict[str, Any]]:
    """Agent steps whose observation quotes a newline-concatenation warning."""
    try:
        data = json.loads(traj_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    findings: list[dict[str, Any]] = []
    for step in data.get("steps") or []:
        if step.get("source") != "agent":
            continue
        content = _obs_text(step)
        if "should end with newline" not in content or "concatenated" not in content:
            continue
        calls = step.get("tool_calls") or []
        findings.append(
            {
                "kind": "newline_concat_warning",
                "detail": f"step {step.get('step_id')}: newline concatenation warning",
                "excerpt": content[:1800] + "\n\nTOOL_CALLS:\n" + json.dumps(calls, ensure_ascii=False)[:2000],
                "path": str(traj_path),
                "step_id": step.get("step_id"),
            }
        )
    return findings


def case_key(case: dict[str, Any]) -> tuple[str, Any]:
    finding = case.get("finding") or {}
    return (str(finding.get("path") or ""), finding.get("step_id"))


def used_observed_keys(repo: Path) -> set[tuple[str, Any]]:
    """Trajectory steps already stored by an observed-collection batch."""
    used: set[tuple[str, Any]] = set()
    runs = repo / "runs"
    if not runs.is_dir():
        return used
    for record_path in runs.glob("distill_observed_*/records/*.json"):
        try:
            doc = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        inp = doc.get("input") or {}
        used.add((str(inp.get("source_trajectory_path") or ""), inp.get("step_id")))
    return used


def collect_observed_cases(repo: Path, *, skip_used: bool = True) -> list[dict[str, Any]]:
    """Warning steps not already collected, false-concat rows first."""
    priority = collect_verified_cases(repo)
    runs = repo / "runs"
    seen = {case_key(case) for case in priority}
    extra: list[dict[str, Any]] = []
    default_snapshot = (
        runs / "causal_edit_20261030_025402" / "versions" / "H0" / "modules" / MODULE_RELPATH
    )
    if runs.is_dir():
        for traj in sorted(runs.rglob("trajectory.json")):
            if "132345" in traj.parts or traj.parts[0].startswith("distill_"):
                continue
            try:
                rel = traj.relative_to(runs)
            except ValueError:
                continue
            source_run = rel.parts[0]
            version_id = ""
            if "evals" in rel.parts:
                version_id = rel.parts[rel.parts.index("evals") + 1]
            snapshot = runs / source_run / "versions" / version_id / "modules" / MODULE_RELPATH
            if not snapshot.is_file():
                snapshot = default_snapshot
            if not snapshot.is_file():
                continue
            task = ""
            for part in rel.parts:
                if part.startswith("task_") and "_" in part[5:]:
                    task = part.split("_", 2)[-1]
                    break
            if not task and "cron-env-debug" in str(traj):
                task = "cron-env-debug"
            for finding in warning_findings(traj):
                key = (str(traj), finding.get("step_id"))
                if key in seen:
                    continue
                seen.add(key)
                extra.append(
                    {
                        "case_id": f"{source_run}_{version_id or 'h0'}_w{finding.get('step_id')}",
                        "task": task or source_run,
                        "source_run": source_run,
                        "snapshot_path": snapshot,
                        "finding": finding,
                    }
                )
    cases = priority + extra
    if skip_used:
        used = used_observed_keys(repo)
        cases = [case for case in cases if case_key(case) not in used]
    return cases


def export_observed_jsonl(run_dir: Path, dest: Path) -> int:
    """Export legacy warning-replay candidates, never verified SFT targets."""
    normalized_parts = {part.lower() for part in dest.parts}
    if "plan_sft" in normalized_parts or "sft" in normalized_parts:
        raise ValueError("warning-only candidates cannot be exported to an SFT directory")
    seen: set[str] = set()
    if dest.is_file():
        for line in dest.read_text().splitlines():
            try:
                existing = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(existing, dict) and existing.get("record_id"):
                seen.add(str(existing["record_id"]))
    lines: list[str] = []
    for record_path in sorted((run_dir / "records").glob("*.json")):
        try:
            doc = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (doc.get("verified") or {}).get("label") != "positive":
            continue
        record_id = str(doc.get("record_id") or "")
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        parsed = doc.get("parsed_plan") or {}
        lines.append(
            json.dumps(
                {
                    "record_id": doc.get("record_id"),
                    "task": (doc.get("metadata") or {}).get("task"),
                    "prompt": doc.get("planner_prompt"),
                    "completion": doc.get("planner_raw"),
                    "phenomenon": parsed.get("phenomenon"),
                    "target": parsed.get("target"),
                    "use": "auxiliary_warning_cleanup",
                    "training_eligible": False,
                    "verification_scope": "quoted warning disappeared in parser replay only",
                    "long_horizon_label": None,
                },
                ensure_ascii=False,
            )
        )
    if not lines:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(lines)


_EMPTY_WAIT_PAYLOAD = {
    "analysis": "probe",
    "plan": "probe empty wait mid-batch",
    "commands": [
        {"keystrokes": "echo hi\n", "duration": 0.5},
        {"keystrokes": "", "duration": 5.0},
        {"keystrokes": "echo bye\n", "duration": 0.5},
    ],
    "task_complete": False,
}


def payloads_for_excerpt(excerpt: str) -> list[dict[str, Any]]:
    """Fixed command batches a program can replay. The empty-wait batch is always included."""
    payloads = [dict(_EMPTY_WAIT_PAYLOAD)]
    marker = "TOOL_CALLS:"
    if marker not in (excerpt or ""):
        return payloads
    blob = excerpt.split(marker, 1)[1].strip()
    try:
        calls = json.loads(blob)
    except json.JSONDecodeError:
        return payloads
    commands: list[dict[str, Any]] = []
    if isinstance(calls, list):
        for call in calls[:6]:
            args = call.get("arguments") if isinstance(call, dict) else None
            if not isinstance(args, dict) and isinstance(call, dict):
                args = call
            if not isinstance(args, dict) or "keystrokes" not in args:
                continue
            commands.append(
                {
                    "keystrokes": args.get("keystrokes") or "",
                    "duration": float(args.get("duration") or 0.5),
                }
            )
    if len(commands) >= 2:
        payloads.append(
            {
                "analysis": "probe",
                "plan": "replay trajectory batch",
                "commands": commands,
                "task_complete": False,
            }
        )
    return payloads


def build_observed_prompt(plan_input: dict[str, Any]) -> str:
    """Plan prompt plus an exact phenomenon the program can replay."""
    base = build_plan_prompt(plan_input)
    old = (
        "TRIGGER: <one distinctive feedback string emitted only when that "
        "condition occurs, or NONE>\n"
    )
    new = (
        "PHENOMENON: <exact substring of the excerpt that this component emits "
        "today and that a correct edit must stop emitting, or NONE>\n"
        "TRIGGER: <one new feedback string emitted only when that condition "
        "occurs, or NONE>\n"
        "Quote PHENOMENON from the excerpt. Do not paraphrase it.\n"
    )
    if old not in base:
        raise RuntimeError("plan prompt shape drifted")
    return base.replace(old, new, 1)


def parse_observed_output(text: str) -> dict[str, Any]:
    """Parse a plan plus the PHENOMENON line. Missing phenomenon is an error."""
    parsed = parse_plan_output(text)
    match = re.search(r"^PHENOMENON:\s*(.*)$", text, re.M)
    if not match:
        raise ValueError("plan output missing PHENOMENON: line")
    raw = match.group(1).strip()
    if not raw or raw.upper() in {"NONE", "NULL", "N/A"}:
        phenomenon = None
    else:
        phenomenon = raw
    parsed["intent"] = re.sub(r"^PHENOMENON:.*$", "", parsed["intent"], flags=re.M).strip()
    parsed["phenomenon"] = phenomenon
    return parsed


def phenomenon_grounded(phenomenon: str | None, excerpt: str) -> bool:
    text = (phenomenon or "").strip()
    return bool(text) and text in (excerpt or "")


def score_observed(
    parent_warning: str,
    child_warning: str,
    phenomenon: str,
    trigger: str | None,
) -> dict[str, Any]:
    """Positive only when the parent emits the phenomenon and the child stops."""
    parent = parent_warning or ""
    child = child_warning or ""
    text = (phenomenon or "").strip()
    expected = (trigger or "").strip()
    if not text or text not in parent:
        return {
            "passed": False,
            "label": "unlabeled",
            "reason": "parent does not reproduce the phenomenon",
            "parent_warning": parent,
            "child_warning": child,
        }
    gone = text not in child
    repeats = bool(expected) and (expected in text or text in expected)
    emitted = bool(expected) and expected in child
    if not gone:
        label, reason, passed = "negative", "phenomenon is still present", False
    else:
        label, reason, passed = "positive", "phenomenon is gone", True
    return {
        "passed": passed,
        "label": label,
        "reason": reason,
        "parent_warning": parent,
        "child_warning": child,
        "phenomenon_gone": gone,
        "trigger_repeats_phenomenon": repeats,
        "triggered": emitted,
    }


def _matching_payload(modules_root: Path, phenomenon: str, excerpt: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for payload in payloads_for_excerpt(excerpt):
        probe = probe_parse_warning(modules_root, payload)
        if phenomenon in str(probe.get("warning") or ""):
            return payload, probe
    return None


def _verify_observed(run_dir: Path, record: dict[str, Any], linkage: dict[str, Any]) -> dict[str, Any]:
    check = record.get("check") or {}
    child = str(linkage.get("child_version_id") or "")
    modules = run_dir / "versions" / child / "modules"
    try:
        child_probe = probe_parse_warning(modules, check["payload"])
        scored = score_observed(
            str((check.get("parent_probe") or {}).get("warning") or ""),
            str(child_probe.get("warning") or ""),
            str(check.get("phenomenon") or ""),
            (record.get("parsed_plan") or {}).get("trigger"),
        )
        scored["child_probe"] = child_probe
    except Exception as exc:  # noqa: BLE001
        scored = {
            "passed": False,
            "label": "unlabeled",
            "reason": "probe failed before it could score the edit",
            "error": str(exc),
        }
    verified = {
        "schema": VERIFIED_SCHEMA_ID,
        "label": scored["label"],
        "reason": scored["reason"],
        "probe": scored,
        "child_version_id": child,
    }
    probe_path = run_dir / "probes" / f"{record.get('record_id')}.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(probe_path, verified)
    verified["probe_path"] = str(probe_path)
    return verified


def run_observed_collection(
    config_path: Path | None = None,
    *,
    target_positives: int = 40,
    max_candidates: int = 18,
    max_per_task: int = 0,
    provider_mode: str = "tongji_agi",
    cases: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Plan, edit, and keep a row only when a replayed phenomenon disappears."""
    if provider_mode != "tongji_agi":
        raise ValueError("observed collection calls tongji_agi only")
    cfg = load_pilot(config_path)
    runs_root = resolve_path(cfg, "runs_root")
    reports_root = resolve_path(cfg, "reports_root")
    run_id = time.strftime("distill_observed_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    if cases is None:
        cases = collect_observed_cases(REPO_ROOT)
    if max_per_task > 0:
        kept: list[dict[str, Any]] = []
        seen_task: dict[str, int] = {}
        for case in cases:
            task = str(case.get("task") or "")
            if seen_task.get(task, 0) >= max_per_task:
                continue
            seen_task[task] = seen_task.get(task, 0) + 1
            kept.append(case)
        cases = kept
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    positives = 0
    considered = 0
    print(
        f"[distill-observed] pool={len(cases)} target_positives={target_positives} "
        f"max_candidates={max_candidates}",
        flush=True,
    )
    for case in cases:
        if positives >= target_positives or considered >= max_candidates:
            break
        considered += 1
        plan_input = _plan_input(case)
        prompt = build_observed_prompt(plan_input)
        record_id = _record_id(case["case_id"], plan_input)
        record_path = run_dir / "records" / f"{considered:03d}_{record_id}.json"
        print(f"[distill-observed] {considered} {case['case_id']}", flush=True)
        raw = ""
        chat_meta: dict[str, Any] | None = None
        try:
            raw, chat_meta = _tongji_chat(prompt, max_tokens=4000, role="distill_planner")
            parsed = parse_observed_output(raw)
        except Exception as exc:  # noqa: BLE001
            record = {
                "schema": SCHEMA_ID,
                "record_id": record_id,
                "status": "error",
                "error": str(exc),
                "input": plan_input,
                "planner_prompt": prompt,
                "planner_raw": raw,
                "parsed_plan": None,
                "metadata": _metadata(
                    chat_meta=chat_meta,
                    task=case["task"],
                    source_path=str(plan_input["source_trajectory_path"]),
                    case_id=case["case_id"],
                    source_run=case["source_run"],
                ),
                "verified": {
                    "schema": VERIFIED_SCHEMA_ID,
                    "label": "unlabeled",
                    "reason": "plan call failed",
                },
            }
            write_json(record_path, record)
            records.append(record)
            summaries.append(_summary_row(record, record_path))
            print(f"[distill-observed] plan error: {exc}", flush=True)
            continue
        record = {
            "schema": SCHEMA_ID,
            "record_id": record_id,
            "status": "ok",
            "error": "",
            "input": plan_input,
            "planner_prompt": prompt,
            "planner_raw": raw,
            "parsed_plan": parsed,
            "metadata": _metadata(
                chat_meta=chat_meta,
                task=case["task"],
                source_path=str(plan_input["source_trajectory_path"]),
                case_id=case["case_id"],
                source_run=case["source_run"],
            ),
            "edit": {"plan_path": str(record_path), "status": "pending"},
        }
        excerpt = str(plan_input.get("error_trajectory_excerpt") or "")
        phenomenon = parsed.get("phenomenon")
        target = str(parsed.get("target") or "")
        editable = target.endswith(MODULE_RELPATH) or target in {MODULE_RELPATH, "tools"}
        if not editable or not phenomenon_grounded(phenomenon, excerpt):
            record["verified"] = {
                "schema": VERIFIED_SCHEMA_ID,
                "label": "unlabeled",
                "reason": "phenomenon is not an exact excerpt substring or the plan is not editable",
            }
            record["edit"] = {
                "plan_path": str(record_path),
                "status": "skipped",
                "reject_reason": record["verified"]["reason"],
                "verified_label": "unlabeled",
            }
            write_json(record_path, record)
        else:
            snapshot = Path(plan_input["module_snapshot_path"])
            try:
                matched = _matching_payload(snapshot.parent.parent, str(phenomenon), excerpt)
            except Exception as exc:  # noqa: BLE001
                matched = None
                record["verified"] = {
                    "schema": VERIFIED_SCHEMA_ID,
                    "label": "unlabeled",
                    "reason": "probe failed before it could score the edit",
                    "error": str(exc),
                }
            if matched is None and "verified" not in record:
                record["verified"] = {
                    "schema": VERIFIED_SCHEMA_ID,
                    "label": "unlabeled",
                    "reason": "parent does not reproduce the phenomenon",
                }
            if record.get("verified", {}).get("label") == "unlabeled" and matched is None:
                record["edit"] = {
                    "plan_path": str(record_path),
                    "status": "skipped",
                    "reject_reason": record["verified"]["reason"],
                    "verified_label": "unlabeled",
                }
                write_json(record_path, record)
            else:
                payload, parent_probe = matched  # type: ignore[misc]
                record["check"] = {
                    "phenomenon": phenomenon,
                    "payload": payload,
                    "parent_probe": parent_probe,
                }
                write_json(record_path, record)
                row = _edit_one(run_dir, record, verifier=_verify_observed)
                if row.get("label") == "positive":
                    positives += 1
        records.append(record)
        summaries.append(_summary_row(record, record_path))
        print(f"[distill-observed] positives={positives}/{target_positives}", flush=True)
    negatives = sum(1 for rec in records if (rec.get("verified") or {}).get("label") == "negative")
    unlabeled = sum(1 for rec in records if (rec.get("verified") or {}).get("label") == "unlabeled")
    summary = {
        "schema": VERIFIED_SCHEMA_ID,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "provider_mode": provider_mode,
        "temperature": PLAN_TEMPERATURE,
        "pool_size": len(cases),
        "candidates": considered,
        "target_positives": target_positives,
        "max_candidates": max_candidates,
        "positive_count": positives,
        "negative_count": negatives,
        "unlabeled_count": unlabeled,
        "stopped": "target" if positives >= target_positives else "pool_or_cap",
        "records": summaries,
        "teacher_label": "replayed phenomenon disappeared after the edit",
        "harbor_eval": False,
        "verification": "parse_llm_response replay; unlabeled when the parent does not emit the phenomenon",
    }
    write_json(run_dir / "manifest.json", summary)
    report_path = reports_root / "research1_plan_observed.md"
    lines = [
        "# Observed-phenomenon plan collection",
        "",
        f"- Run: `{run_dir}`",
        f"- Pool: {len(cases)}",
        f"- Candidates: {considered}",
        f"- Positive: {positives}",
        f"- Negative: {negatives}",
        f"- Unlabeled: {unlabeled}",
        f"- Stop: {summary['stopped']}",
        "",
        "A positive label means the parent module emitted the quoted phenomenon",
        "and the edited module stopped emitting it.",
        "Rows the program cannot replay stay unlabeled and are not training labels.",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    summary["report"] = str(report_path)
    exported = export_observed_jsonl(
        run_dir, REPO_ROOT / "datasets" / "auxiliary_warning_cleanup" / "candidates.jsonl"
    )
    summary["exported_auxiliary_candidates"] = exported
    summary["exported_positives"] = 0
    summary["training_eligible"] = False
    summary["use"] = "auxiliary_warning_cleanup"
    return (EXIT_PASS if positives else EXIT_FAIL), summary


def run_observed_batches(
    config_path: Path | None = None,
    *,
    batch_size: int = 10,
    max_batches: int = 20,
    max_per_task: int = 4,
    provider_mode: str = "tongji_agi",
) -> tuple[int, dict[str, Any]]:
    """Run observed collection one batch at a time, skipping steps already stored."""
    batches: list[dict[str, Any]] = []
    for index in range(1, max_batches + 1):
        remaining = collect_observed_cases(REPO_ROOT)
        if max_per_task > 0:
            kept: list[dict[str, Any]] = []
            seen_task: dict[str, int] = {}
            for case in remaining:
                task = str(case.get("task") or "")
                if seen_task.get(task, 0) >= max_per_task:
                    continue
                seen_task[task] = seen_task.get(task, 0) + 1
                kept.append(case)
            remaining = kept
        if not remaining:
            print(f"[distill-batches] stop: no unused cases before batch {index}", flush=True)
            break
        chunk = remaining[:batch_size]
        print(
            f"[distill-batches] batch {index} size={len(chunk)} remaining={len(remaining)}",
            flush=True,
        )
        _code, doc = run_observed_collection(
            config_path,
            target_positives=batch_size,
            max_candidates=len(chunk),
            max_per_task=0,
            provider_mode=provider_mode,
            cases=chunk,
        )
        batches.append(
            {
                "batch": index,
                "run_id": doc.get("run_id"),
                "positive_count": doc.get("positive_count"),
                "negative_count": doc.get("negative_count"),
                "unlabeled_count": doc.get("unlabeled_count"),
                "exported_positives": doc.get("exported_positives"),
            }
        )
        print(
            f"[distill-batches] batch {index} positive={doc.get('positive_count')} "
            f"negative={doc.get('negative_count')} unlabeled={doc.get('unlabeled_count')}",
            flush=True,
        )
    summary = {"batches": batches, "batch_count": len(batches)}
    return (EXIT_PASS if batches else EXIT_FAIL), summary


def build_llm_review_prompt(record: dict[str, Any]) -> str:
    """Ask the reviewer model whether this plan fixes the fault in the excerpt."""
    inp = record.get("input") or {}
    parsed = record.get("parsed_plan") or {}
    edit = record.get("edit") or {}
    diff = ""
    diff_path = edit.get("diff_path")
    if diff_path and Path(str(diff_path)).is_file():
        diff = Path(str(diff_path)).read_text()[:2500]
    excerpt = str(inp.get("error_trajectory_excerpt") or "")[:4000]
    module = str(inp.get("module_excerpt") or "")[:4000]
    return (
        "You are reviewing a modification plan for a harness component. "
        "Do not write a new plan.\n"
        "Pass only when the excerpt shows a fault in this component and the "
        "plan tells an editor how to fix that fault.\n"
        "Fail when the excerpt does not show that fault, the plan names the "
        "wrong component, or the plan does not say what to change.\n"
        "Do not pass a plan whose purpose is only to change a task score.\n"
        "An edit that failed to apply does not by itself fail a plan that "
        "still states the right change.\n"
        "Reply in exactly this shape:\n"
        "VERDICT: pass\n"
        "or\n"
        "VERDICT: fail\n"
        "REASON: <one sentence>\n"
        "\n"
        f"Finding: {inp.get('finding_kind')}\n"
        f"Detail: {inp.get('finding_detail')}\n"
        "\n"
        "Excerpt:\n"
        f"{excerpt}\n"
        "\n"
        "Component excerpt:\n"
        f"{module}\n"
        "\n"
        f"Plan target: {parsed.get('target')}\n"
        "Plan intent:\n"
        f"{parsed.get('intent')}\n"
        f"Plan phenomenon: {parsed.get('phenomenon')}\n"
        f"Plan trigger: {parsed.get('trigger')}\n"
        "\n"
        f"Edit status: {edit.get('status')}\n"
        f"Edit reject reason: {edit.get('reject_reason') or ''}\n"
        "Diff:\n"
        f"{diff or '(no diff)'}\n"
    )


def parse_review_verdict(text: str) -> dict[str, str]:
    """Parse VERDICT and REASON. Does not invent a verdict."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    match = re.search(r"^VERDICT:\s*(pass|fail)\s*$", cleaned, re.I | re.M)
    if not match:
        raise ValueError("review output missing VERDICT: pass|fail")
    reason_m = re.search(r"^REASON:\s*(.+)$", cleaned, re.M)
    if not reason_m or not reason_m.group(1).strip():
        raise ValueError("review output missing REASON")
    return {"verdict": match.group(1).lower(), "reason": reason_m.group(1).strip()}


def run_llm_review(
    config_path: Path | None = None,
    *,
    provider_mode: str = "tongji_agi",
) -> tuple[int, dict[str, Any]]:
    """Review stored observed plans with the Tongji model."""
    if provider_mode != "tongji_agi":
        raise ValueError("llm review calls tongji_agi only")
    load_pilot(config_path)
    records = sorted(REPO_ROOT.glob("runs/distill_observed_*/records/*.json"))
    reviews: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    print(f"[llm-review] records={len(records)}", flush=True)
    for index, path in enumerate(records, start=1):
        doc = json.loads(path.read_text())
        if not isinstance(doc.get("parsed_plan"), dict):
            skipped += 1
            continue
        print(f"[llm-review] {index} {doc.get('record_id')}", flush=True)
        prompt = build_llm_review_prompt(doc)
        raw = ""
        try:
            raw, _meta = _tongji_chat(prompt, max_tokens=800, role="distill_reviewer")
            verdict = parse_review_verdict(raw)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            reviews.append({"record_id": doc.get("record_id"), "path": str(path), "verdict": "error", "reason": str(exc), "raw": raw})
            print(f"[llm-review] error {exc}", flush=True)
            continue
        if verdict["verdict"] == "pass":
            passed += 1
        else:
            failed += 1
        reviews.append(
            {
                "record_id": doc.get("record_id"),
                "path": str(path),
                "task": (doc.get("metadata") or {}).get("task"),
                "program_label": (doc.get("verified") or {}).get("label"),
                "verdict": verdict["verdict"],
                "reason": verdict["reason"],
                "raw": raw,
                "reviewer": _meta,
            }
        )
        print(f"[llm-review] {verdict['verdict']} pass={passed} fail={failed}", flush=True)
    out_dir = REPO_ROOT / "datasets" / "auxiliary_warning_cleanup"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "llm_reviews.jsonl"
    review_rows = [
        {
            **review,
            "use": "auxiliary_review",
            "training_eligible": False,
            "long_horizon_label": None,
        }
        for review in reviews
    ]
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows))
    summary = {
        "records": len(records),
        "reviewed": passed + failed,
        "skipped_no_plan": skipped,
        "pass": passed,
        "fail": failed,
        "error": errors,
        "training_eligible": False,
        "exported_positives": 0,
        "jsonl": str(jsonl),
        "reviews": reviews,
    }
    write_json(REPO_ROOT / "runs" / "llm_review_latest.json", summary)
    report = REPO_ROOT / "reports" / "research1_plan_llm_review.md"
    report.write_text(
        "\n".join(
            [
                "# LLM review of observed plans",
                "",
                f"- Reviewed: {passed + failed}",
                f"- Pass: {passed}",
                f"- Fail: {failed}",
                f"- Skipped (no plan): {skipped}",
                f"- Errors: {errors}",
                f"- Auxiliary reviews: `{jsonl}`",
                "- LLM review alone never exports a training target or long-horizon label.",
                "",
            ]
        )
        + "\n"
    )
    return (EXIT_PASS if passed else EXIT_FAIL), summary


def run_verified_collection(
    config_path: Path | None = None,
    *,
    target_positives: int = 40,
    max_candidates: int = 120,
    max_per_task: int = 0,
    provider_mode: str = "tongji_agi",
) -> tuple[int, dict[str, Any]]:
    """Plan, edit, and probe until the positive quota or the candidate cap."""
    if provider_mode != "tongji_agi":
        raise ValueError("verified collection calls tongji_agi only")
    cfg = load_pilot(config_path)
    runs_root = resolve_path(cfg, "runs_root")
    reports_root = resolve_path(cfg, "reports_root")
    run_id = time.strftime("distill_verified_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    cases = collect_verified_cases(REPO_ROOT)
    if max_per_task > 0:
        kept: list[dict[str, Any]] = []
        seen_task: dict[str, int] = {}
        for case in cases:
            task = str(case.get("task") or "")
            if seen_task.get(task, 0) >= max_per_task:
                continue
            seen_task[task] = seen_task.get(task, 0) + 1
            kept.append(case)
        cases = kept
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    positives = 0
    considered = 0
    print(
        f"[distill-verified] pool={len(cases)} target_positives={target_positives} "
        f"max_candidates={max_candidates}",
        flush=True,
    )
    for case in cases:
        if positives >= target_positives or considered >= max_candidates:
            break
        considered += 1
        plan_input = _plan_input(case)
        prompt = build_plan_prompt(plan_input)
        record_id = _record_id(case["case_id"], plan_input)
        record_path = run_dir / "records" / f"{considered:03d}_{record_id}.json"
        print(f"[distill-verified] {considered} {case['case_id']}", flush=True)
        chat_meta: dict[str, Any] | None = None
        raw = ""
        try:
            raw, chat_meta = _tongji_chat(prompt, max_tokens=4000, role="distill_planner")
            parsed = parse_plan_output(raw)
        except Exception as exc:  # noqa: BLE001
            record = {
                "schema": SCHEMA_ID,
                "record_id": record_id,
                "status": "error",
                "error": str(exc),
                "input": plan_input,
                "planner_prompt": prompt,
                "planner_raw": raw,
                "parsed_plan": None,
                "metadata": _metadata(
                    chat_meta=chat_meta,
                    task=case["task"],
                    source_path=str(plan_input["source_trajectory_path"]),
                    case_id=case["case_id"],
                    source_run=case["source_run"],
                ),
                "verified": {
                    "schema": VERIFIED_SCHEMA_ID,
                    "label": "unlabeled",
                    "reason": "plan call failed",
                },
            }
            write_json(record_path, record)
            records.append(record)
            summaries.append(_summary_row(record, record_path))
            print(f"[distill-verified] plan error: {exc}", flush=True)
            continue
        record = {
            "schema": SCHEMA_ID,
            "record_id": record_id,
            "status": "ok",
            "error": "",
            "input": plan_input,
            "planner_prompt": prompt,
            "planner_raw": raw,
            "parsed_plan": parsed,
            "metadata": _metadata(
                chat_meta=chat_meta,
                task=case["task"],
                source_path=str(plan_input["source_trajectory_path"]),
                case_id=case["case_id"],
                source_run=case["source_run"],
            ),
            "edit": {"plan_path": str(record_path), "status": "pending"},
        }
        problems = validate_plan_io_record(record)
        if problems:
            record["status"] = "error"
            record["error"] = "schema: " + "; ".join(problems)
            record["verified"] = {
                "schema": VERIFIED_SCHEMA_ID,
                "label": "unlabeled",
                "reason": "plan schema failed",
            }
            write_json(record_path, record)
            records.append(record)
            summaries.append(_summary_row(record, record_path))
            continue
        if _eligible_for_edit(record):
            write_json(record_path, record)
            row = _edit_one(run_dir, record)
            if row.get("label") == "positive":
                positives += 1
        else:
            record["verified"] = _label_none(record)
            record["edit"] = {
                "plan_path": str(record_path),
                "status": "skipped",
                "reject_reason": "not an editable tools plan",
                "verified_label": record["verified"]["label"],
            }
            write_json(record_path, record)
        records.append(record)
        summaries.append(_summary_row(record, record_path))
        print(
            f"[distill-verified] positives={positives}/{target_positives}",
            flush=True,
        )
    negatives = sum(1 for rec in records if (rec.get("verified") or {}).get("label") == "negative")
    unlabeled = sum(1 for rec in records if (rec.get("verified") or {}).get("label") == "unlabeled")
    summary = {
        "schema": VERIFIED_SCHEMA_ID,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "provider_mode": provider_mode,
        "temperature": PLAN_TEMPERATURE,
        "pool_size": len(cases),
        "candidates": considered,
        "target_positives": target_positives,
        "max_candidates": max_candidates,
        "positive_count": positives,
        "negative_count": negatives,
        "unlabeled_count": unlabeled,
        "stopped": "target" if positives >= target_positives else "pool_or_cap",
        "records": summaries,
        "teacher_label": "legacy local probe outcome; not a verified training target",
        "use": "auxiliary_warning_cleanup",
        "training_eligible": False,
        "harbor_eval": False,
        "verification": "probe_version_trigger on the empty-keystroke wait pattern",
    }
    write_json(run_dir / "manifest.json", summary)
    report_path = reports_root / "research1_plan_verified.md"
    report_path.write_text(
        "\n".join(
            [
                "# Legacy warning-replay candidate collection",
                "",
                f"- Run: `{run_dir}`",
                f"- Pool: {len(cases)}",
                f"- Candidates: {considered}",
                f"- Positive: {positives}",
                f"- Negative: {negatives}",
                f"- Unlabeled: {unlabeled}",
                f"- Stop: {summary['stopped']}",
                "",
                "A positive label is an applied edit whose probe no longer emits the",
                "false empty-keystroke concatenation warning. A trigger that only",
                "repeats that warning is not required. A new trigger phrase must appear.",
                "A negative label is a rejected edit, a probe miss, or `target=none`",
                "on a snapshot that already suppresses that warning.",
                "These local probe labels are auxiliary evidence, not verified SFT targets.",
                "",
            ]
        )
        + "\n"
    )
    summary["report"] = str(report_path)
    code = EXIT_PASS if positives else EXIT_FAIL
    return code, summary


def run_distill_plan(
    config_path: Path | None = None,
    *,
    limit: int = 5,
    apply_one: bool = True,
    provider_mode: str = "tongji_agi",
) -> tuple[int, dict[str, Any]]:
    """Call Tongji for a small batch of plan I/O records. One optional edit proof."""
    if provider_mode != "tongji_agi":
        raise ValueError("distill-plan calls tongji_agi only")
    cfg = load_pilot(config_path)
    runs_root = resolve_path(cfg, "runs_root")
    reports_root = resolve_path(cfg, "reports_root")
    run_id = time.strftime("distill_plan_%Y%m%d_%H%M%S", time.gmtime())
    run_dir = runs_root / run_id
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    cases = select_distill_cases(REPO_ROOT, limit=limit)
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        plan_input = _plan_input(case)
        prompt = build_plan_prompt(plan_input)
        record_id = _record_id(case["case_id"], plan_input)
        record_path = run_dir / "records" / f"{index:02d}_{record_id}.json"
        print(f"[distill-plan] {case['case_id']} {record_id}", flush=True)
        chat_meta: dict[str, Any] | None = None
        raw = ""
        try:
            raw, chat_meta = _tongji_chat(prompt, max_tokens=4000, role="distill_planner")
            parsed = parse_plan_output(raw)
        except Exception as exc:  # noqa: BLE001 — record the failure, do not invent a plan
            record = {
                "schema": SCHEMA_ID,
                "record_id": record_id,
                "status": "error",
                "error": str(exc),
                "input": plan_input,
                "planner_prompt": prompt,
                "planner_raw": raw,
                "parsed_plan": None,
                "metadata": _metadata(
                    chat_meta=chat_meta,
                    task=case["task"],
                    source_path=str(plan_input["source_trajectory_path"]),
                    case_id=case["case_id"],
                    source_run=case["source_run"],
                ),
                "edit": {
                    "plan_path": str(record_path),
                    "status": "skipped",
                    "reject_reason": "plan call failed",
                    "harbor_eval": False,
                },
            }
            write_json(record_path, record)
            records.append(record)
            summaries.append(
                {
                    "record_id": record_id,
                    "case_id": case["case_id"],
                    "status": "error",
                    "path": str(record_path),
                    "task": case["task"],
                    "error": str(exc),
                }
            )
            print(f"[distill-plan] error {case['case_id']}: {exc}", flush=True)
            continue
        record = {
            "schema": SCHEMA_ID,
            "record_id": record_id,
            "status": "ok",
            "error": "",
            "input": plan_input,
            "planner_prompt": prompt,
            "planner_raw": raw,
            "parsed_plan": parsed,
            "metadata": _metadata(
                chat_meta=chat_meta,
                task=case["task"],
                source_path=str(plan_input["source_trajectory_path"]),
                case_id=case["case_id"],
                source_run=case["source_run"],
            ),
            "edit": {
                "plan_path": str(record_path),
                "status": "skipped",
                "reject_reason": "edit proof runs on at most one eligible plan",
                "harbor_eval": False,
            },
        }
        problems = validate_plan_io_record(record)
        if problems:
            record["status"] = "error"
            record["error"] = "schema: " + "; ".join(problems)
            record["parsed_plan"] = None
        write_json(record_path, record)
        records.append(record)
        summaries.append(
            {
                "record_id": record_id,
                "case_id": case["case_id"],
                "status": record["status"],
                "path": str(record_path),
                "task": case["task"],
                "target": (parsed.get("target") if record["status"] == "ok" else None),
                "trigger": (parsed.get("trigger") if record["status"] == "ok" else None),
                "error": record.get("error") or "",
            }
        )

    edit_summary = _attempt_edits(run_dir, records) if apply_one else None

    success = sum(1 for row in records if row.get("status") == "ok")
    failed = sum(1 for row in records if row.get("status") == "error")
    summary = {
        "schema": SCHEMA_ID,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "provider_mode": provider_mode,
        "temperature": PLAN_TEMPERATURE,
        "cases_selected": len(cases),
        "success_count": success,
        "error_count": failed,
        "records": summaries,
        "edit": edit_summary,
        "teacher_label": "plan input and parsed plan (target, intent, trigger)",
        "harbor_eval": False,
    }
    write_json(run_dir / "manifest.json", summary)
    report_path = reports_root / "research1_plan_distill.md"
    _write_report(report_path, summary, records)
    summary["report"] = str(report_path)
    code = EXIT_PASS if success else EXIT_FAIL
    return code, summary


def _summary_row(record: dict[str, Any], path: Path) -> dict[str, Any]:
    parsed = record.get("parsed_plan") or {}
    ok = record.get("status") == "ok"
    return {
        "record_id": record.get("record_id"),
        "case_id": (record.get("metadata") or {}).get("case_id"),
        "status": record.get("status"),
        "path": str(path),
        "task": (record.get("metadata") or {}).get("task"),
        "target": parsed.get("target") if ok else None,
        "trigger": parsed.get("trigger") if ok else None,
        "label": (record.get("verified") or {}).get("label"),
        "error": record.get("error") or "",
        "prior_errors": record.get("prior_errors") or [],
    }


def resume_distill_plan(
    run_dir: Path,
    config_path: Path | None = None,
    *,
    apply_one: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Retry failed plan rows in an existing distill run, then one edit proof."""
    run_dir = Path(run_dir).resolve()
    cfg = load_pilot(config_path)
    reports_root = resolve_path(cfg, "reports_root")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    for row in manifest.get("records") or []:
        path = Path(row["path"])
        records.append(json.loads(path.read_text()))
        paths.append(path)
    for record, path in zip(records, paths):
        if record.get("status") == "ok":
            continue
        prompt = str(record.get("planner_prompt") or "")
        if not prompt.strip():
            continue
        case_id = (record.get("metadata") or {}).get("case_id")
        print(f"[distill-plan] retry {case_id} {record.get('record_id')}", flush=True)
        raw = ""
        chat_meta: dict[str, Any] | None = None
        try:
            raw, chat_meta = _tongji_chat(prompt, max_tokens=4000, role="distill_planner")
            parsed = parse_plan_output(raw)
        except Exception as exc:  # noqa: BLE001
            prior = list(record.get("prior_errors") or [])
            if record.get("error"):
                prior.append(str(record["error"]))
            record["prior_errors"] = prior
            record["status"] = "error"
            record["error"] = str(exc)
            record["planner_raw"] = raw
            record["parsed_plan"] = None
            write_json(path, record)
            print(f"[distill-plan] retry error: {exc}", flush=True)
            continue
        prior = list(record.get("prior_errors") or [])
        if record.get("error"):
            prior.append(str(record["error"]))
        record["prior_errors"] = prior
        record["status"] = "ok"
        record["error"] = ""
        record["planner_raw"] = raw
        record["parsed_plan"] = parsed
        meta = record.setdefault("metadata", {})
        if chat_meta:
            meta["provider"] = chat_meta.get("provider") or meta.get("provider")
            meta["model"] = chat_meta.get("model") or meta.get("model")
            meta["usage"] = chat_meta.get("usage") or {}
            meta["raw_field"] = chat_meta.get("raw_field") or ""
            meta["finish_reason"] = chat_meta.get("finish_reason")
        meta["timestamp"] = datetime.now(timezone.utc).isoformat()
        meta["retried"] = True
        problems = validate_plan_io_record(record)
        if problems:
            record["status"] = "error"
            record["error"] = "schema: " + "; ".join(problems)
            record["parsed_plan"] = None
        else:
            record["edit"] = {
                "plan_path": str(path),
                "status": "skipped",
                "reject_reason": "edit proof runs on at most one eligible plan",
                "harbor_eval": False,
            }
        write_json(path, record)
    edit_summary = _attempt_edits(run_dir, records) if apply_one else manifest.get("edit")
    success = sum(1 for row in records if row.get("status") == "ok")
    failed = sum(1 for row in records if row.get("status") == "error")
    summary = {
        "schema": SCHEMA_ID,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "provider_mode": manifest.get("provider_mode") or "tongji_agi",
        "temperature": PLAN_TEMPERATURE,
        "cases_selected": len(records),
        "success_count": success,
        "error_count": failed,
        "records": [_summary_row(rec, path) for rec, path in zip(records, paths)],
        "edit": edit_summary,
        "teacher_label": "plan input and parsed plan (target, intent, trigger)",
        "harbor_eval": False,
        "resumed": True,
    }
    write_json(run_dir / "manifest.json", summary)
    report_path = reports_root / "research1_plan_distill.md"
    _write_report(report_path, summary, records)
    summary["report"] = str(report_path)
    return (EXIT_PASS if success else EXIT_FAIL), summary


def _write_report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        "# Plan-layer distillation (teacher data)",
        "",
        "The evolution step is two layers. The PLAN module reads an error-trajectory",
        "excerpt and the current component snapshot, then emits a modification plan.",
        "The edit module consumes that plan. This distill batch stores the plan",
        "module's input and output. It does not use score uplift or comment-marker",
        "patches as the teacher label, and it does not run a Harbor H0/H1 trial.",
        "",
        f"Schema: `{SCHEMA_ID}`",
        "",
        "Each record under `records/` has:",
        "",
        "- `input.error_trajectory_excerpt` plus `input.module_snapshot_path`,",
        "  `input.module_snapshot_sha256`, and `input.module_excerpt`",
        "- `planner_prompt` — the text actually sent",
        "- `planner_raw` — the model reply",
        "- `parsed_plan.target`, `parsed_plan.intent`, `parsed_plan.trigger`",
        "  (`trigger` is null when the model says NONE)",
        "- `metadata.provider`, `metadata.model`, `metadata.temperature`,",
        "  `metadata.task`, `metadata.source_trajectory_path`, `metadata.timestamp`",
        "",
        "CLI: `python3 -m research1 distill-plan` (Tongji, temperature 0).",
        "`--resume runs/distill_plan_*` retries a row whose call failed and does not",
        "invent a plan for that row. Code: `src/research1/distill.py`.",
        "",
        "Files for a batch:",
        "",
        "- `runs/distill_plan_*/records/*.json` — one teacher pair each",
        "- `runs/distill_plan_*/manifest.json` — counts and paths",
        "- `runs/distill_plan_*/patches/*.diff` and `versions/distill_edit_1/` —",
        "  the single edit proof, linked by `edit.plan_path`",
        "",
        "An applied edit stores `edit.plan_path` pointing at that record, plus",
        "`edit.diff_path`. The fixed editor rejects comment-only replacements",
        "before `patch` runs. `edit.outcome_class` stays `no-effect` when a diff",
        "is applied without a Harbor measurement. That is not a score claim.",
        "",
        "## This batch",
        "",
        f"- Run directory: `{summary.get('run_dir')}`",
        f"- Successful plan I/O records: {summary.get('success_count')}",
        f"- Failed rows: {summary.get('error_count')}",
        f"- Temperature: {summary.get('temperature')}",
        f"- Manifest: `{summary.get('run_dir')}/manifest.json`",
        "",
    ]
    retried = [row for row in (summary.get("records") or []) if row.get("prior_errors")]
    if retried:
        lines.append(
            "A first Tongji reply that was empty is stored under `prior_errors` and was retried. The record's `planner_raw` is the later non-empty reply."
        )
        lines.append("")
    lines.extend(
        [
            "| Case | Status | Task | Record |",
            "| --- | --- | --- | --- |",
        ]
    )
    errors: list[str] = []
    for row in summary.get("records") or []:
        lines.append(
            f"| `{row.get('case_id')}` | {row.get('status')} | {row.get('task')} | `{row.get('path')}` |"
        )
        err = str(row.get("error") or "").strip()
        if err:
            errors.append(f"- `{row.get('case_id')}`: {err[:300]}")
    if errors:
        lines.extend(["", "Failed rows (no plan was invented):", ""])
        lines.extend(errors)
    example = next(
        (
            rec
            for rec in records
            if rec.get("status") == "ok"
            and str((rec.get("parsed_plan") or {}).get("target") or "") not in {"", "none"}
        ),
        None,
    )
    if example is None:
        example = next((rec for rec in records if rec.get("status") == "ok"), None)
    lines.extend(["", "## Example", ""])
    if example is None:
        lines.append("No successful plan I/O record. See row errors in the manifest.")
    else:
        parsed = example.get("parsed_plan") or {}
        inp = example.get("input") or {}
        meta = example.get("metadata") or {}
        intent = str(parsed.get("intent") or "").strip().replace("\n", " ")
        if len(intent) > 400:
            intent = intent[:400] + "..."
        lines.extend(
            [
                f"- Record: `{example.get('record_id')}`",
                f"- Task: `{meta.get('task')}`",
                f"- Source: `{meta.get('source_trajectory_path')}`",
                f"- Snapshot sha256: `{inp.get('module_snapshot_sha256')}`",
                f"- Finding: `{inp.get('finding_kind')}` — {inp.get('finding_detail')}",
                f"- TARGET: `{parsed.get('target')}`",
                f"- TRIGGER: `{parsed.get('trigger')}`",
                f"- INTENT: {intent}",
            ]
        )
    edit = summary.get("edit") or {}
    lines.extend(["", "## Edit proof", ""])
    if not edit:
        lines.append("No edit was attempted.")
    else:
        lines.extend(
            [
                f"- plan_path: `{edit.get('plan_path')}`",
                f"- status: `{edit.get('status')}`",
                f"- diff: `{edit.get('diff_path')}`",
                f"- child version: `{edit.get('child_version_id')}`",
                f"- reject_reason: `{edit.get('reject_reason') or ''}`",
            ]
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
