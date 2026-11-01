"""Pure prototype: exact step observations, without inferred shell producers.

The selector records aggregate terminal observations. A command identifier in
the same step is context, not a causal link between that command and a quote.
No model calls, file writes, admission changes, or new runtime schema.
"""
from __future__ import annotations
import hashlib
import ast
import json
import re
import shlex
from typing import Any

VALIDATOR = "proposal-factual-witness.v5"
_PROMPT = re.compile(r"(?:[\w.-]+@[\w.-]+:[^\n#$]{0,120}[#$] ?|[$#] )")
_KINDS = {"observed_tool_output", "execution_status", "check_status", "positive_execution", "context", "failure_behavior"}
_STATUS_REPORT = re.compile(r"\bHTTP/\d(?:\.\d)?\s+[1-5]\d\d\b|\b(?:GET|POST|PUT|PATCH|DELETE)\b[^\n]{0,30}\b[1-5]\d\d\b|\bexit(?:[_ ]+(?:code|status))?\s*[:=]?\s*-?\d+\b", re.I)
_CHECK_REPORT = re.compile(r"\b(?:validations?|checks?|tests?|assertions?)\b[^\n]{0,45}\b(?:passed|failed|successful|ok|error)\b|\b\d+\s+(?:passed|failed)\b", re.I)
_LEGACY_POSITIVE = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|HTTP(?:/\d(?:\.\d)?)?)\b[^\n]{0,30}\b[23]\d\d\b|\b(?:validations?|checks?|tests?)\b[^\n]{0,25}\b(?:passed|successful|ok)\b|\b(?:all|final)\s+(?:assertions?|checks?|validations?)\s+passed\b", re.I)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fake_command_head(command):
    """Only obvious literal-output commands; never parse a heredoc body."""
    first = next((line.strip() for line in command.splitlines() if line.strip()), "")
    if re.match(r"^(?:[\w./-]*/)?(?:echo|printf)\b", first):
        return True
    try:
        words = shlex.split(first)
    except ValueError:
        return False
    while words and (words[0] in {"env", "command", "builtin"} or "=" in words[0]):
        words.pop(0)
    if not words:
        return False
    name = words[0].rsplit("/", 1)[-1]
    if name in {"echo", "printf", "true", "false", ":"}:
        return True
    if name in {"sh", "bash", "dash"} and len(words) > 2 and "c" in words[1]:
        return _fake_command_head(words[-1])
    if re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", name) and "-c" in words:
        try:
            tree = ast.parse(words[words.index("-c") + 1])
        except (SyntaxError, IndexError):
            return False
        return bool(tree.body) and all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
            and all(isinstance(arg, ast.Constant) for arg in node.value.args) for node in tree.body)
    return False


def _is_displayed_command_line(line, step):
    stripped = line.strip()
    if _PROMPT.match(stripped) or stripped == ">" or stripped.startswith("> "):
        return True
    # Unadorned command echoes are context too. This is a line-level check,
    # never a prerequisite that a whole multiline command be reconstructed.
    for call in step.get("tool_calls", []):
        args = call.get("arguments") if isinstance(call, dict) else None
        command = args.get("keystrokes") if isinstance(args, dict) else None
        if isinstance(command, str) and stripped in {part.strip() for part in command.splitlines()}:
            return True
    return False


def _has_payload_at(text, start, end, step):
    """Does the exact quote intersect observed text beyond command decorations?"""
    offset = 0
    for line in text.splitlines(keepends=True):
        line_end = offset + len(line)
        if max(start, offset) < min(end, line_end):
            piece = line[max(0, start-offset):min(len(line), end-offset)].strip()
            if (piece and piece != "New Terminal Output:" and not _is_displayed_command_line(line, step)
                    and not re.fullmatch(r"(?:reward|score|verdict|task_result)\s*[:=].*", piece, re.I)):
                return True
        offset = line_end
    return False


def _sources(serialized):
    try:
        selected = json.loads(serialized)
    except ValueError:
        return [], None
    if not isinstance(selected, dict) or not isinstance(selected.get("steps"), list):
        return [], None
    sources = []
    for index, step in enumerate(selected["steps"]):
        if not isinstance(step, dict):
            continue
        fields = [(name, step.get(name)) for name in ("observation_excerpt", "message_excerpt")]
        for call_index, call in enumerate(step.get("tool_calls", [])):
            arguments = call.get("arguments") if isinstance(call, dict) else None
            if isinstance(arguments, dict):
                fields.append((f"tool_calls/{call_index}/keystrokes_excerpt", arguments.get("keystrokes_excerpt")))
        for name, excerpt in fields:
            if not isinstance(excerpt, dict):
                continue
            for segment_index, segment in enumerate(excerpt.get("segments", [])):
                if isinstance(segment, dict) and isinstance(segment.get("text"), str):
                    sources.append({"field": name, "text": segment["text"], "segment": segment,
                        "step": step, "step_index": index, "segment_index": segment_index})
    return sources, selected


def _observation_location(source, start, end):
    step, segment = source["step"], source["segment"]
    capture_ids = [call["tool_call_id"] for call in step.get("tool_calls", [])
        if isinstance(call, dict) and isinstance(call.get("tool_call_id"), str) and call["tool_call_id"]]
    if (source["field"] != "observation_excerpt" or step.get("source") != "agent"
            or not isinstance(step.get("source_index"), int) or isinstance(step.get("source_index"), bool)
            or step["source_index"] < 0 or not isinstance(step.get("step_id"), (int, str))
            or isinstance(step.get("step_id"), bool) or not str(step["step_id"]).strip()
            or not re.fullmatch("[a-f0-9]{64}", str(step.get("visible_step_sha256", "")))
            or not isinstance(segment.get("start"), int) or isinstance(segment.get("start"), bool) or segment["start"] < 0
            or segment.get("end") != segment["start"] + len(source["text"]) or not capture_ids):
        return None
    return {"source_granularity": "step_terminal_observation", "field": source["field"],
        "step_id": step["step_id"], "source_index": step["source_index"], "step_index": source["step_index"],
        "visible_step_sha256": step["visible_step_sha256"], "recorded_step_tool_call_ids": capture_ids,
        "observation_segment_sha256": _sha(source["text"]), "segment_index": source["segment_index"],
        "segment_offset": start, "source_start": segment["start"] + start,
        "source_end": segment["start"] + end, "length": end-start,
        "producer_tool_call_id": None, "producer_relation": "not_provided_by_selector",
        "freshness_established": False, "success_established": False}


def audit_witnesses(value: Any, view: dict, action: str) -> dict:
    """Audit all witnesses. Exact but weak command quotes do not fail the set."""
    report = {"validator": VALIDATOR, "valid_references": True, "substantive_execution_fact": False,
              "eligible_witness_set": False, "issues": [], "witnesses": []}
    evidence = {item["id"]: item["text"] for item in view["evidence"]}
    if not isinstance(value, list) or not value:
        report["valid_references"] = False
        report["issues"].append({"index": None, "reason": "Independent PASS needs exact visible witnesses"})
        return report
    for index, witness in enumerate(value):
        if (not isinstance(witness, dict) or set(witness) != {"evidence_ref", "text", "kind"}
                or witness.get("evidence_ref") not in evidence or witness.get("kind") not in _KINDS
                or not isinstance(witness.get("text"), str) or not witness["text"].strip()):
            report["valid_references"] = False
            report["issues"].append({"index": index, "code": "invalid_reference", "reason": "Review witness must quote an exact supplied evidence substring"})
            continue
        quote, kind = witness["text"], witness["kind"]
        serialized = evidence[witness["evidence_ref"]]
        sources, selected = _sources(serialized)
        matches, substantive = [], False
        for source in sources:
            for match in re.finditer(re.escape(quote), source["text"]):
                if kind == "context":
                    context_location = {"field": source["field"], "step_index": source["step_index"],
                        "segment_index": source["segment_index"], "segment_offset": match.start(),
                        "source_representation": "decoded_segment", "segment_sha256": _sha(source["text"]),
                        "step_id": source["step"].get("step_id"), "source_index": source["step"].get("source_index"),
                        "visible_step_sha256": source["step"].get("visible_step_sha256"), "length": len(quote)}
                    if isinstance(source["segment"].get("start"), int):
                        context_location.update(source_start=source["segment"]["start"] + match.start(),
                                                source_end=source["segment"]["start"] + match.end())
                    matches.append(context_location)
                else:
                    location = _observation_location(source, match.start(), match.end())
                    if location:
                        commands = [call["arguments"].get("keystrokes") for call in source["step"].get("tool_calls", [])
                                    if isinstance(call, dict) and isinstance(call.get("arguments"), dict)]
                        nonempty = [command for command in commands if isinstance(command, str) and command.strip()]
                        fake_only = bool(nonempty) and (all(_fake_command_head(command) for command in nonempty)
                            or any(_fake_command_head(command) and quote in command for command in nonempty))
                        payload = _has_payload_at(source["text"], match.start(), match.end(), source["step"])
                        location["substantive_observed_fact"] = bool(payload and not fake_only)
                        location["available_level"] = "observed_terminal_text" if payload else "displayed_command_text"
                        if fake_only:
                            location["available_level"] = "obvious_literal_output_command"
                        substantive |= location["substantive_observed_fact"]
                        matches.append(location)
        # Existing explicit plaintext adapters retain their conservative legacy
        # positive-report interface. Untyped JSON is not silently an adapter.
        try:
            json.loads(serialized)
            plain_adapter = False
        except ValueError:
            plain_adapter = True
        if (plain_adapter and kind == "positive_execution" and quote in serialized
                and _LEGACY_POSITIVE.search(quote)
                and not re.search(r"(?:^|[\n#;$|]\s*)(?:echo|printf)\b", serialized)):
            matches.append({"field": "plain_text", "source_granularity": "explicit_adapter_text",
                "source_representation": "plain_text", "source_start": serialized.index(quote),
                "source_end": serialized.index(quote)+len(quote), "success_established": False,
                "producer_tool_call_id": None, "reported_only": True})
            substantive = True
        if kind == "context":
            # Context may quote visible selector metadata at any nesting level.
            # Keep its literal serialized representation and exact source span.
            for match in re.finditer(re.escape(quote), serialized):
                matches.append({"field": "plain_text" if plain_adapter else "serialized_context",
                    "source_representation": "plain_text" if plain_adapter else "serialized_json",
                    "source_start": match.start(), "source_end": match.end(), "length": len(quote),
                    "execution_support": False})
        if not matches:
            report["valid_references"] = False
            report["issues"].append({"index": index, "code": "invalid_reference", "reason": "Review witness must quote an exact supplied evidence substring"})
        pattern = {"execution_status": _STATUS_REPORT, "check_status": _CHECK_REPORT,
                   "positive_execution": _LEGACY_POSITIVE}.get(kind)
        status_report_supported = not pattern or bool(substantive and pattern.search(quote))
        if pattern and not status_report_supported:
            report["issues"].append({"index": index, "code": "positive_execution_unproven",
                "reason": "Status/check reports need substantive observed output with the corresponding status/check text"})
        if kind != "context" and substantive and status_report_supported:
            report["substantive_execution_fact"] = True
        report["witnesses"].append(dict(witness, validation={"validator": VALIDATOR,
            "serialized_evidence_sha256": _sha(serialized), "source_granularity": "step_observation",
            "substantive_observed_fact": substantive, "success_established": False,
            "reported_only": bool(pattern),
            "semantic_support": "original_independent_reviewer_required", "locations": matches}))
    if action == "keep" and not report["substantive_execution_fact"]:
        report["issues"].append({"index": None, "code": "positive_execution_unproven", "reason": "KEEP needs a substantive observed execution fact, not commands or fake success alone"})
    report["eligible_witness_set"] = not report["issues"] and report["valid_references"]
    return report


def validate_witnesses(value, view, action):
    report = audit_witnesses(value, view, action)
    if not report["eligible_witness_set"]:
        raise ValueError("; ".join(f"{item.get('code', 'invalid_witness')} witness[{item['index']}]: {item['reason']}" for item in report["issues"]))
    return report["witnesses"]
