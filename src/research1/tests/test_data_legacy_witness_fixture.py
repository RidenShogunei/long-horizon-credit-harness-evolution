"""Single-plan deployment contract and immutable, proposal-only teacher calls.

Identity, admission and evaluation records never become model input. The
multi-candidate teacher request is a collection operation; student messages
always request one Plan, not a jointly generated candidate set.
"""
from __future__ import annotations

import hashlib
import ast
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import tempfile
from typing import Any

LEGACY_PLAN_POLICY_ID = "proposal_single_plan_v2"
POLICY_ID = "proposal_single_plan_v3"
FACTUAL_REVIEW_POLICY = "proposal-factual-review.v4"
FACTUAL_WITNESS_VALIDATOR = "proposal-factual-witness.v4"
MODULES = {"agent_loop", "observation", "tools", "context_mgmt", "verification"}
PLAN_FIELDS = {"action", "target_module", "locations", "evidence_refs",
               "problem_hypothesis", "intended_change", "preserve_behaviors",
               "verification_plan", "verification_kind"}
_FORBIDDEN = {
    "reward", "rewards", "score", "scores", "reward_score", "reward_scores", "dev_score",
    "dev_scores", "task_result", "task_results", "result", "results", "evaluation",
    "evaluations", "future", "future_results", "future_trajectory", "future_trajectories",
    "child_diff", "child_patch", "patch", "patch_text", "child_result", "child_results",
    "hidden_tests", "reference_solution", "reference_solutions", "answer_key",
    "long_horizon_label", "mc_label", "terminal_gain", "cumulative_regression", "u_lambda",
    "api_key", "authorization", "password", "secret", "access_token", "refresh_token",
}
_SYSTEM_V2 = (
    "You are the Harness evolver. Read only the supplied decision-time proposal evidence, "
    "current visible module code, and this branch's history. Return exactly ONE JSON Plan. "
    "Do not invent a defect, infer a defect from a task failure alone, claim global absence "
    "from a bounded excerpt, or output a patch, task answer, predicted score or future result. "
    "Use keep with a factual bounded reason when no edit is justified. An edit may touch "
    "only one permitted module and existing visible files. Plan fields are action(edit|keep), "
    "target_module(module name|null), locations(list of visible file paths), evidence_refs(list "
    "of supplied evidence ids), problem_hypothesis, intended_change, preserve_behaviors(list), "
    "verification_plan(list), verification_kind(string|null). Available verification IDs "
    "are interfaces, not evidence that a repair is needed. Unknown mechanisms remain pending. "
    "Prior proposal binary outcomes, if supplied, are finished execution feedback, not proof of a Harness defect."
)
_SYSTEM = (_SYSTEM_V2 + " A keep must still name specific factual checks and evidence limits in its "
           "verification_plan; no code edit does not mean no factual verification is required.")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _check_visible(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN:
                raise ValueError("Non-policy information in visible input: " + str(key))
            if lowered in {"partition", "task_partition", "source_partition", "task_split"} and item != "proposal":
                raise ValueError("Only proposal-partition evidence may enter policy input")
            _check_visible(item)
    elif isinstance(value, list):
        for item in value:
            _check_visible(item)


def public_view(state: dict[str, Any]) -> dict[str, Any]:
    """Project a typed visible view; do not serialize the entire state object."""
    if not isinstance(state, dict):
        raise ValueError("State must be an object")
    for key in ("state_id", "root_lineage_id"):
        if not isinstance(state.get(key), str) or not state[key].strip():
            raise ValueError("Missing state identity: " + key)
    if state.get("split") not in {"train", "validation"}:
        raise ValueError("State split must be train or validation")
    modules = state.get("permitted_modules")
    if (not isinstance(modules, list) or not modules or any(not isinstance(m, str) or m not in MODULES for m in modules)
            or len(set(modules)) != len(modules)):
        raise ValueError("permitted_modules must name supported Harness modules")
    code = state.get("module_code")
    if not isinstance(code, dict) or not code or any(not _visible_path(p) or p.split("/", 1)[0] not in MODULES | {"tool_helper"}
            or not isinstance(v, str) or not v.strip() for p, v in code.items()):
        raise ValueError("module_code must contain existing visible relative files")
    evidence = state.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("State requires proposal evidence")
    rows = []
    for item in evidence:
        if not isinstance(item, dict):
            raise ValueError("Evidence must be an object")
        _check_visible(item)
        if not isinstance(item.get("id"), str) or not item["id"].strip() or not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("Evidence needs a stable id and nonempty visible text")
        rows.append({"id": item["id"], "text": item["text"]})
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evidence ids must be unique")
    history = state.get("history", [])
    if not isinstance(history, list):
        raise ValueError("History must be a list of decision-time records")
    _check_visible(history)
    # Explicit histories contain plans, their real application status, and
    # proposal-visible feedback, never result blobs or code patches.
    allowed_history = {"decision_id", "action", "plan", "implementation_status", "verification_feedback", "evidence_refs", "note"}
    for item in history:
        if not isinstance(item, dict) or set(item) - allowed_history:
            raise ValueError("History contains fields outside the visible decision contract")
    catalog = state.get("verification_catalog", [])
    if not isinstance(catalog, list):
        raise ValueError("verification_catalog must be a list of frozen interface ids")
    for item in catalog:
        if (not isinstance(item, dict) or set(item) != {"mechanism_id", "module"}
                or not isinstance(item["mechanism_id"], str) or not item["mechanism_id"].strip()
                or item["module"] not in modules):
            raise ValueError("Verification catalog exposes only a mechanism id and permitted module")
    if len({item["mechanism_id"] for item in catalog}) != len(catalog):
        raise ValueError("Verification interface ids must be unique")
    view={"module_code": code, "evidence": rows, "history": history,
          "permitted_modules": list(modules), "verification_catalog": catalog}
    outcomes=state.get('prior_proposal_outcomes',[])
    if not isinstance(outcomes,list):raise ValueError('Prior proposal outcomes must be a list')
    if outcomes:
        required={'proposal_task_id','execution_id','outcome','failure_layer','parent_code_hash','source_sha256','finished_before_decision','source_partition'}
        tasks=set(state.get('proposal_task_ids',[]))
        for item in outcomes:
            if not isinstance(item,dict) or set(item)!=required:raise ValueError('Prior proposal feedback may not contain scores or tests')
            if item['proposal_task_id'] not in tasks or item['source_partition']!='proposal' or item['finished_before_decision'] is not True:
                raise ValueError('Only finished prior proposal feedback is visible')
            if item['outcome'] not in {'PASS','FAIL','UNSCORED'}:raise ValueError('Unsupported proposal outcome encoding')
            if not isinstance(item['execution_id'],str) or not item['execution_id']:raise ValueError('Proposal execution provenance missing')
            if not isinstance(item['parent_code_hash'],str) or item['parent_code_hash']!=state.get('code_hash') or not re.fullmatch(r'[0-9a-f]{64}',item['parent_code_hash']):
                raise ValueError('Proposal feedback belongs to another parent')
            if not isinstance(item['source_sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',item['source_sha256']):
                raise ValueError('Proposal source checksum missing')
            if item['failure_layer'] not in {'model','harness','infrastructure','container','scorer','unknown'}:raise ValueError('Unsupported proposal failure layer')
        if len({item['execution_id'] for item in outcomes})!=len(outcomes):raise ValueError('Duplicate prior proposal execution')
        view['prior_proposal_outcomes']=outcomes
    return view


def _visible_path(path: Any) -> bool:
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        return False
    parsed = PurePosixPath(path)
    return not parsed.is_absolute() and ".." not in parsed.parts and "." not in parsed.parts and str(parsed) == path


def validate_plan(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    view = public_view(state)
    if not isinstance(plan, dict) or set(plan) != PLAN_FIELDS:
        raise ValueError("Plan must contain exactly the single-plan contract fields")
    _check_visible(plan)
    if plan["action"] not in {"edit", "keep"}:
        raise ValueError("Invalid plan action")
    for key in ("problem_hypothesis", "intended_change"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError("Missing plan " + key)
    for key in ("locations", "evidence_refs", "preserve_behaviors", "verification_plan"):
        values = plan[key]
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError("Invalid plan list: " + key)
    refs = {e["id"] for e in view["evidence"]}
    if not plan["evidence_refs"] or not set(plan["evidence_refs"]).issubset(refs):
        raise ValueError("Plan references missing or unknown evidence")
    target = plan["target_module"]
    if target is not None and target not in view["permitted_modules"]:
        raise ValueError("Plan targets a non-permitted module")
    for path in plan["locations"]:
        if path not in view["module_code"]:
            raise ValueError("Plan location is not an existing visible file")
        module = path.split("/", 1)[0]
        if module == "tool_helper":
            module = "tools"
        if target != module:
            raise ValueError("Each modification is limited to its declared single module")
    kind = plan["verification_kind"]
    if kind is not None and (not isinstance(kind, str) or not kind.strip()):
        raise ValueError("verification_kind must be a nonempty string or null")
    if plan["action"] == "edit" and (target is None or not plan["locations"] or not plan["preserve_behaviors"] or not plan["verification_plan"]):
        raise ValueError("Edit needs a target, locations, preservation and verification requirements")
    return json.loads(canonical(plan))


def render_messages(state: dict[str, Any], plan: dict[str, Any] | None = None) -> list[dict[str, str]]:
    # Previously collected states did not store a separate instruction version.
    # Their original canonical instruction must survive later export/review.
    version = state.get("plan_policy_version", LEGACY_PLAN_POLICY_ID)
    systems = {LEGACY_PLAN_POLICY_ID: _SYSTEM_V2, POLICY_ID: _SYSTEM}
    if version not in systems:
        raise ValueError("Unsupported frozen plan instruction version")
    messages = [{"role": "system", "content": systems[version]},
                {"role": "user", "content": canonical(public_view(state))}]
    if plan is not None:
        messages.append({"role": "assistant", "content": canonical(validate_plan(plan, state))})
    return messages


def parse_candidates(raw: str | dict[str, Any], state: dict[str, Any], max_candidates: int = 2) -> list[dict[str, Any]]:
    if max_candidates not in {1, 2}:
        raise ValueError("Candidate count must be 1 or 2")
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("Teacher output must be a JSON candidates object")
    plans = value["candidates"]
    if not isinstance(plans, list) or not 1 <= len(plans) <= max_candidates:
        raise ValueError("Teacher must return one or two honest candidates")
    result, seen = [], set()
    for plan in plans:
        plan = validate_plan(plan, state)
        signature = canonical(plan)
        if signature not in seen:
            seen.add(signature)
            result.append(plan)
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        name = handle.name
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(name, path)


def collect_candidates(state: dict[str, Any], directory: Path) -> list[dict[str, Any]]:
    """Save final teacher output before validating; failed replies are not retried."""
    from research1.distill import _tongji_chat

    max_candidates = state.get("max_candidates", 2)
    if max_candidates not in {1, 2}:
        raise ValueError("Candidate count must be 1 or 2")
    messages = render_messages(state)
    prompt = (messages[0]["content"] + "\nCollection request: generate at most " + str(max_candidates)
              + ' distinct plans before any candidate outcome is known. Return JSON {"candidates":[Plan,...]} instead of the deployment single-plan envelope. '
              + "Do not force a second edit or fill a candidate quota.\nFrozen proposal-only view:\n" + messages[1]["content"])
    path = Path(directory) / "teacher.json"
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("prompt_sha256") != prompt_hash or saved.get("prompt") != prompt:
            raise ValueError("Teacher input changed for an immutable collection directory")
        raw, metadata = saved["raw"], saved["metadata"]
    else:
        raw, metadata = _tongji_chat(prompt, max_tokens=8000, role="research1_data_teacher",
                                     json_output=True, reasoning_effort="none")
        saved = {"policy_id": state.get("plan_policy_version", LEGACY_PLAN_POLICY_ID),
                 "prompt": prompt, "prompt_sha256": prompt_hash,
                 "raw": raw, "metadata": metadata, "visible_view_sha256": digest(public_view(state))}
        _atomic_json(path, saved)
    if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop" or metadata.get("temperature") != 0:
        raise ValueError("Teacher requires complete final JSON at frozen temperature zero")
    expected = state.get("teacher_model")
    if expected and metadata.get("model") != expected:
        raise ValueError("Teacher model differs from the frozen policy")
    expected_provider = state.get("teacher_provider")
    if expected_provider and metadata.get("provider") != expected_provider:
        raise ValueError("Teacher provider differs from the frozen policy")
    return parse_candidates(raw, state, max_candidates)


def _witness_sources(text: str) -> list[dict[str, Any]]:
    """Decode the selector envelope once; never unescape a quoted witness.

    Selector text is a JSON serialization, while observation segments contain
    the actual visible characters. Exactness is checked against those strings,
    retaining their offsets and the hash of the unchanged serialization.
    """
    try:
        selected = json.loads(text)
    except (ValueError, TypeError):
        selected = None
    if not isinstance(selected, dict) or not isinstance(selected.get("steps"), list):
        return [{"field": "plain_text", "text": text}]
    sources = []
    for step_index, step in enumerate(selected["steps"]):
        if not isinstance(step, dict):
            continue
        fields = [(name, step.get(name)) for name in ("message_excerpt", "observation_excerpt")]
        for call_index, call in enumerate(step.get("tool_calls") or []):
            if not isinstance(call, dict) or not isinstance(call.get("arguments"), dict):
                continue
            fields.append((f"tool_calls/{call_index}/keystrokes_excerpt",
                           call["arguments"].get("keystrokes_excerpt")))
        for field, excerpt in fields:
            if not isinstance(excerpt, dict):
                continue
            for segment_index, segment in enumerate(excerpt.get("segments") or []):
                if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                    sources.append({"field": field, "step_index": step_index,
                        "step_id": step.get("step_id"), "source_index": step.get("source_index"),
                        "visible_step_sha256": step.get("visible_step_sha256"),
                        "segment_index": segment_index, "segment_start": segment.get("start"),
                        "text": segment["text"]})
    return sources


_TERMINAL_PROMPT = re.compile(r"[\w.-]+@[\w.-]+:[^\n#$]{0,120}[#$] ?|(?m:^> )")


def _terminal_characters(text: str) -> tuple[str, list[int]]:
    """Ignore terminal prompts/wrapping only when locating executed keystrokes.

    Witness characters are never normalized: their final match remains exact.
    The map retains original observation offsets for the command/output boundary.
    """
    ignored = set()
    for match in _TERMINAL_PROMPT.finditer(text):
        ignored.update(range(match.start(), match.end()))
    indices = [i for i, character in enumerate(text) if not character.isspace() and i not in ignored]
    return "".join(text[i] for i in indices), indices


def _substantive_command(command: str) -> bool:
    """Reject obvious manufactured status output; not a proof of program semantics."""
    try:
        words = shlex.split(command.strip())
    except ValueError:
        return False
    if not words:
        return False
    while words and (words[0] in {"env", "command", "builtin"} or "=" in words[0]):
        words.pop(0)
    if not words:
        return False
    executable = PurePosixPath(words[0]).name
    if executable in {"echo", "printf", "true", "false", ":", "sleep", "clear", "pwd"}:
        return False
    if executable in {"sh", "bash", "dash"}:
        options = next((word for word in words[1:] if word.startswith("-") and "c" in word), None)
        return bool(options and words[-1] != options and _substantive_command(words[-1]))
    # A compound echo/printf is not promoted because another command happens to
    # share its tool call. A future adapter can provide separate result events.
    if re.search(r"(?:^|[;&|]\s*|\n\s*)(?:[\w./-]*/)?(?:echo|printf)\b", command):
        return False
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable) and "-c" in words:
        index = words.index("-c")
        try:
            tree = ast.parse(words[index + 1])
        except (SyntaxError, IndexError):
            return False
        if tree.body and all(isinstance(node, ast.Expr) and (
                isinstance(node.value, ast.Constant) or (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "print")) for node in tree.body):
            return False
    return True


def _execution_origins(serialized: str, source: dict[str, Any], quote: str) -> list[dict[str, Any]]:
    """Bind output to original executed tool calls, not later screen replays.

    Current selector observations aggregate a step's terminal output. Command
    echoes delimit individual calls; an unambiguous single-call raw stdout is
    also supported. Missing or ambiguous provenance remains unqualified.
    """
    if source["field"] != "observation_excerpt":
        return []
    selected = json.loads(serialized)
    step = selected["steps"][source["step_index"]]
    if (step.get("source") != "agent" or not isinstance(step.get("source_index"), int)
            or isinstance(step.get("source_index"), bool) or step["source_index"] < 0
            or (not isinstance(step.get("step_id"), (int, str))) or not str(step["step_id"]).strip()
            or not isinstance(source.get("segment_start"), int) or source["segment_start"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", str(step.get("visible_step_sha256", "")))):
        return []
    observation = source["text"]
    compact, indices = _terminal_characters(observation)
    calls, cursor = [], 0
    for call_index, call in enumerate(step.get("tool_calls") or []):
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        command = arguments.get("keystrokes") if isinstance(arguments, dict) else None
        if (call.get("function_name") != "bash_command" or not isinstance(command, str)
                or not command.endswith("\n") or not command.strip()
                or not isinstance(call.get("tool_call_id"), str) or not call["tool_call_id"].strip()
                or (isinstance(arguments.get("keystrokes_excerpt"), dict)
                    and arguments["keystrokes_excerpt"].get("truncated") is True)):
            continue
        normalized = "".join(command.split())
        at = compact.find(normalized, cursor)
        span = None
        if at >= 0:
            span = (indices[at], indices[at + len(normalized) - 1] + 1)
            cursor = at + len(normalized)
        calls.append({"tool_call_id": call["tool_call_id"], "tool_call_index": call_index,
                      "tool_call_position": call.get("position", call_index + 1),
                      "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                      "command": command, "command_span": span,
                      "substantive": _substantive_command(command)})
    origins = []
    for call in calls:
        if not call["substantive"]:
            continue
        span = call["command_span"]
        if span is not None:
            following = [other["command_span"][0] for other in calls
                         if other["command_span"] is not None and other["command_span"][0] > span[0]]
            start, end = span[1], min(following) if following else len(observation)
            association = "terminal_command_interval"
        elif (len(calls) == 1 and not _TERMINAL_PROMPT.search(observation)
              and quote not in call["command"]):
            start, end = 0, len(observation)
            association = "sole_tool_call_stdout"
        else:
            continue
        for match in re.finditer(re.escape(quote), observation):
            if match.start() < start or match.end() > end:
                continue
            # A bare prompt or a score/verdict is not substantive tool output.
            if (_TERMINAL_PROMPT.fullmatch(quote.strip()) or re.fullmatch(
                    r"\s*(?:reward|score|verdict|task_result)\s*[:=].*", quote, re.I)):
                continue
            origin = {key: item for key, item in call.items() if key not in {"command", "substantive"}}
            origin.update(association=association, segment_offset=match.start(),
                          observation_segment_sha256=hashlib.sha256(observation.encode("utf-8")).hexdigest())
            origins.append(origin)
    return origins


def _polling_observation_origins(serialized: str, source: dict[str, Any], quote: str) -> list[dict[str, Any]]:
    """Locate a real polling capture, without inventing its causal producer."""
    if source["field"] != "observation_excerpt" or _TERMINAL_PROMPT.match(quote.lstrip()):
        return []
    selected = json.loads(serialized)
    step = selected["steps"][source["step_index"]]
    if (step.get("source") != "agent" or not isinstance(step.get("source_index"), int)
            or isinstance(step.get("source_index"), bool) or step["source_index"] < 0
            or not isinstance(step.get("step_id"), (int, str)) or not str(step["step_id"]).strip()
            or not isinstance(source.get("segment_start"), int) or source["segment_start"] < 0
            or not re.fullmatch(r"[a-f0-9]{64}", str(step.get("visible_step_sha256", "")))):
        return []
    if re.fullmatch(r"\s*(?:reward|score|verdict|task_result)\s*[:=].*", quote, re.I):
        return []
    captures = []
    for index, call in enumerate(step.get("tool_calls") or []):
        if not isinstance(call, dict) or call.get("function_name") != "bash_command":
            continue
        arguments = call.get("arguments")
        duration = arguments.get("duration") if isinstance(arguments, dict) else None
        if (not isinstance(arguments, dict) or arguments.get("keystrokes") != ""
                or not isinstance(duration, (int, float)) or isinstance(duration, bool) or not 0 < duration < float("inf")
                or not isinstance(call.get("tool_call_id"), str) or not call["tool_call_id"].strip()):
            continue
        for match in re.finditer(re.escape(quote), source["text"]):
            captures.append({"association": "polling_observation", "capture_tool_call_id": call["tool_call_id"],
                "capture_tool_call_index": index, "capture_duration_seconds": duration,
                "producer_tool_call_id": None, "producer_relation": "unknown",
                "freshness_established": False, "success_established": False,
                "segment_offset": match.start(),
                "observation_segment_sha256": hashlib.sha256(source["text"].encode("utf-8")).hexdigest()})
    return captures


def _serialized_context_sources(serialized: str, quote: str) -> list[dict[str, Any]]:
    """Match top-level selector metadata in its original JSON representation.

    These offsets refer to serialized envelope characters, not decoded tool
    observations. Metadata can provide context, never execution support.
    """
    try:
        selected = json.loads(serialized)
    except (ValueError, TypeError):
        return []
    if not isinstance(selected, dict) or not isinstance(selected.get("steps"), list):
        return []
    decoder, sources, position = json.JSONDecoder(), [], serialized.index("{") + 1
    while position < len(serialized):
        while position < len(serialized) and serialized[position].isspace():
            position += 1
        if position >= len(serialized) or serialized[position] == "}":
            break
        key_start = position
        key, position = decoder.raw_decode(serialized, position)
        while serialized[position].isspace():
            position += 1
        position += 1  # colon, already validated by json.loads
        while serialized[position].isspace():
            position += 1
        _, position = decoder.raw_decode(serialized, position)
        field_text = serialized[key_start:position]
        if key != "steps" and quote in field_text:
            sources.append({"field": "serialized_context", "metadata_key": key,
                "source_representation": "serialized_json", "segment_start": key_start, "text": field_text})
        while position < len(serialized) and serialized[position].isspace():
            position += 1
        if position < len(serialized) and serialized[position] == ",":
            position += 1
    return sources


def _review_witnesses(value: Any, state: dict[str, Any], action: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("PASS needs exact visible evidence witnesses")
    evidence = {item["id"]: item["text"] for item in public_view(state)["evidence"]}
    result, execution_fact = [], False
    positive_pattern = re.compile(
        r"(?:\b(?:GET|POST|PUT|PATCH|DELETE|HTTP(?:/\d(?:\.\d)?)?)\b[^\n]{0,30}\b[23]\d\d\b"
        r"|\b(?:validations?|checks?|tests?)\b[^\n]{0,25}\b(?:passed|successful|ok)\b"
        r"|\b(?:all|final)\s+(?:assertions?|checks?|validations?)\s+passed\b)", re.I)
    status_pattern = re.compile(r"\bHTTP/\d(?:\.\d)?\s+[1-5]\d\d\b|\bexit(?:[_ ]+(?:code|status))?\s*[:=]?\s*-?\d+\b", re.I)
    check_pattern = re.compile(r"\b(?:validations?|checks?|tests?|assertions?)\b[^\n]{0,45}\b(?:passed|failed|successful|ok|error)\b|\b\d+\s+(?:passed|failed)\b", re.I)
    for witness in value:
        if (not isinstance(witness, dict) or set(witness) != {"evidence_ref", "text", "kind"}
                or witness["evidence_ref"] not in evidence or not isinstance(witness["text"], str)
                or len(witness["text"].strip()) < 12
                or witness["kind"] not in {"observed_tool_output", "execution_status", "check_status",
                                            "positive_execution", "failure_behavior", "context"}):
            raise ValueError("Review witness must quote an exact supplied evidence substring")
        serialized = evidence[witness["evidence_ref"]]
        matches = [source for source in _witness_sources(serialized) if witness["text"] in source["text"]]
        if witness["kind"] == "context":
            matches.extend(_serialized_context_sources(serialized, witness["text"]))
        if not matches:
            raise ValueError("Review witness must quote an exact supplied evidence substring")
        execution_kind = witness["kind"] in {"observed_tool_output", "execution_status", "check_status", "positive_execution"}
        origins = {}
        if execution_kind:
            observations = []
            try:
                json.loads(serialized)
                plaintext_adapter = False
            except ValueError:
                plaintext_adapter = True
            for source in matches:
                if source["field"] == "observation_excerpt":
                    bound = _execution_origins(serialized, source, witness["text"])
                    if not bound and witness["kind"] == "observed_tool_output":
                        bound = _polling_observation_origins(serialized, source, witness["text"])
                    if bound:
                        origins[id(source)] = bound
                        observations.append(source)
                elif (source["field"] == "plain_text" and plaintext_adapter and witness["kind"] == "positive_execution"
                      and positive_pattern.search(witness["text"])
                      and not re.search(r"(?:^|[\n#;$|]\s*)(?:echo|printf)\b", source["text"])):
                    # Compatibility for explicit plaintext proposal adapters.
                    # New observed-output witnesses require typed provenance.
                    observations.append(source)
            required_pattern = {"execution_status": status_pattern, "check_status": check_pattern,
                                "positive_execution": positive_pattern}.get(witness["kind"])
            if not observations or (required_pattern and not required_pattern.search(witness["text"])):
                raise ValueError("positive_execution_unproven: execution witnesses need exact substantive tool output and the claimed status/check evidence")
            matches = observations
            execution_fact = True
        locations = []
        for source in matches:
            bound_origins = origins.get(id(source), [None])
            for origin in bound_origins:
                offset = origin["segment_offset"] if origin else source["text"].find(witness["text"])
                location = {key: item for key, item in source.items() if key != "text"}
                location.update(segment_offset=offset, length=len(witness["text"]))
                if origin:
                    location["execution_origin"] = origin
                if isinstance(source.get("segment_start"), int):
                    location["source_start"] = source["segment_start"] + offset
                    location["source_end"] = location["source_start"] + len(witness["text"])
                locations.append(location)
        result.append(dict(witness, validation={"validator": FACTUAL_WITNESS_VALIDATOR,
            "serialized_evidence_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "evidence_level": {"execution_status": "observed_status_report", "check_status": "observed_check_report",
                               "positive_execution": "legacy_success_report"}.get(witness["kind"], witness["kind"]),
            "success_established": False,
            "semantic_support": "independent_reviewer_required",
            "locations": locations}))
    if action == "keep" and not execution_fact:
        raise ValueError("KEEP requires a substantive observed execution fact, not a score or absence alone")
    return result


def review_plan(state: dict[str, Any], plan: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Independent model factual review; executable behavior checks remain separate.

    A PASS is a scoped model review, not a claim of human-verified truth or of
    long-term benefit. All reviews are saved separately from teacher output.
    """
    from research1.distill import _tongji_chat

    plan = validate_plan(plan, state)
    view = public_view(state)
    prompt = (
        "Factual review policy: " + FACTUAL_REVIEW_POLICY + ". "
        "Independently check the candidate's factual justification against ONLY this frozen "
        "proposal-only view and visible module code. Do not invent evidence or infer correctness "
        "from a task score. Distinguish bounded abstention from global absence: 'no defect is "
        "evidenced in this supplied bounded view' means the available evidence does not justify "
        "a specific edit; it does not mean no defects exist. Do not demand global correctness "
        "proof for a bounded keep. Independently verify the candidate's concrete source and "
        "execution facts and whether they support that bounded decision. A score alone or "
        "merely not seeing errors is insufficient. A recorded failure_layer is a saved "
        "classification, not a proven cause; accurately reporting that classification is "
        "permitted, attributing causal responsibility requires separate evidence. "
        "Omitted observations cannot prove global absence of defects. Check claims that errors were diagnosed/resolved; "
        "unresolved, overstated or uncertain facts must stay pending_review. Do not repair or "
        "rewrite the candidate. Return JSON {status:PASS|pending_review|FAIL, reason:string, "
        "evidence_refs:[supplied ids], witnesses:[{evidence_ref:id,text:exact visible substring, "
        "kind:observed_tool_output|execution_status|check_status|failure_behavior|context}]}. "
        "For PASS provide exact witnesses for the concrete facts that justify the plan. "
        "KEEP needs a substantive actual execution observation supporting a specific bounded "
        "fact, not reward, absence, a command merely echoing success, or a reviewer verdict. "
        "Use observed_tool_output for an actual returned body or other output; this establishes "
        "what was observed and does not establish HTTP success, exit success, test success or "
        "task correctness. Use execution_status only with an explicit execution/HTTP status "
        "observation; check_status only with an actual check-result observation. Match the "
        "strength of every candidate claim to the evidence: arbitrary JSON is not success "
        "proof, while a factual bounded keep need not contain a particular success phrase. "
        "Quote decoded observation characters, never unexecuted command/message text as "
        "execution output. A terminal command echo is context, not observed_tool_output; do "
        "not use a displayed command as evidence of its output or success. Use context for "
        "serialized envelope metadata such as omitted_step_ids. An empty-keystroke polling "
        "capture may establish what text was observed, but not which command produced it, "
        "whether it is fresh, or whether execution/checks succeeded. Do not upgrade a polling "
        "observation to causal, freshness or success claims. Do not attribute visible outcome metadata to an evidence id whose "
        "text does not contain it. This is a factual review only, not executable "
        "verification or optimality assessment.\nFrozen visible view:\n" + canonical(view)
        + "\nCandidate:\n" + canonical(plan)
    )
    path = Path(directory) / "factual_review.json"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("prompt") != prompt or saved.get("prompt_sha256") != prompt_hash:
            raise ValueError("Factual review input changed in an immutable directory")
        raw, metadata = saved["raw"], saved["metadata"]
    else:
        raw, metadata = _tongji_chat(prompt, max_tokens=5000, role="research1_data_factual_review",
                                     json_output=True, reasoning_effort="none")
        saved = {"prompt": prompt, "prompt_sha256": prompt_hash, "raw": raw, "metadata": metadata,
                 "review_policy": FACTUAL_REVIEW_POLICY,
                 "review_type": "model_fact_review", "visible_view_sha256": digest(view), "plan_sha256": digest(plan)}
        _atomic_json(path, saved)
    result = {"status": "pending_review", "reason": "Factual review is not validated", "evidence_refs": [],
              "independent": True, "review_type": "model_fact_review",
              "review_policy": FACTUAL_REVIEW_POLICY,
              "witness_validator": FACTUAL_WITNESS_VALIDATOR,
              "evidence_text_sha256": {item["id"]: hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                                       for item in view["evidence"]},
              "reviewer_id": "factual_review_" + digest({"prompt": prompt_hash, "model": metadata.get("model"),
                                                         "provider": metadata.get("provider")})[:24],
              "review_path": str(path), "plan_sha256": digest(plan), "visible_view_sha256": digest(view)}
    try:
        if metadata.get("raw_field") != "content" or metadata.get("finish_reason") != "stop" or metadata.get("temperature") != 0:
            raise ValueError("Review requires complete final content at frozen temperature zero")
        expected = state.get("review_model", state.get("teacher_model"))
        if expected and metadata.get("model") != expected:
            raise ValueError("Review model differs from the frozen review policy")
        expected_provider = state.get("review_provider")
        if expected_provider and metadata.get("provider") != expected_provider:
            raise ValueError("Review provider differs from the frozen review policy")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("status") not in {"PASS", "pending_review", "FAIL"}:
            raise ValueError("Invalid factual review status")
        reason, refs = value.get("reason"), value.get("evidence_refs")
        visible_refs = {item["id"] for item in view["evidence"]}
        if (not isinstance(reason, str) or not reason.strip() or not isinstance(refs, list) or not refs
                or any(not isinstance(ref, str) or ref not in visible_refs for ref in refs)):
            raise ValueError("Review needs a reason and supplied evidence references")
        if value["status"] == "PASS":
            witnesses = _review_witnesses(value.get("witnesses"), state, plan["action"])
            if any(witness["evidence_ref"] not in refs for witness in witnesses):
                raise ValueError("Witness references must also be cited by the review")
            claims = plan["problem_hypothesis"] + " " + plan["intended_change"]
            if re.search(r"\b(?:globally\s+defect[- ]free|defect[- ]free\s+(?:everywhere|globally)|no\s+(?:defects?|warnings?|errors?)\s+(?:exist\s+)?(?:anywhere|everywhere|in\s+all\s+tasks))\b", claims, re.I):
                raise ValueError("Blanket global absence claim is unsupported by a bounded view")
            result["witnesses"] = witnesses
        result.update(status=value["status"], reason=reason, evidence_refs=refs)
    except (ValueError, TypeError, KeyError) as exc:
        result["reason"] = str(exc)
    _atomic_json(Path(directory) / "factual_review_result.json", result)
    return result
