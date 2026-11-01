"""Exact AST statement navigation for submitted Python heredocs; never execute.

Only recorded tool-input segments are inspected. Terminal echoes are not parsed
or repaired. The resulting proposal facts retain the existing quotation schema
and prove neither that a file was written nor that any Python statement ran.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from research1.data_pipeline.fact_cards import flat_proposal_sources, make_fact_card, make_proposal_fact

_DELIMITER = r"(?P<q>['\"]?)(?P<delimiter>[A-Za-z_][A-Za-z0-9_]{0,63})(?P=q)"
_PY_PATH = r"(?:[A-Za-z0-9_./-]+\.py|'[A-Za-z0-9_./ -]+\.py'|\"[A-Za-z0-9_./ -]+\.py\")"
_HEADERS = (
    re.compile(r"^[ \t]*cat[ \t]+>[ \t]*" + _PY_PATH + r"[ \t]*<<[ \t]*" + _DELIMITER + r"[ \t]*$"),
    re.compile(r"^[ \t]*(?:[A-Za-z0-9_./-]*/)?python(?:3(?:\.[0-9]+)?)?[ \t]+-[ \t]*<<[ \t]*" + _DELIMITER + r"[ \t]*$"),
)
# Unsupported heredocs are skipped as opaque input, so Python-looking text in
# their bodies cannot be mistaken for an independent command. This conservative
# detector is not a shell parser and can omit ambiguous input rather than guess.
_ANY_HEREDOC = re.compile(r"<<(?P<tabs>-?)[ \t]*" + _DELIMITER)
_STATEMENTS = (ast.Assign, ast.AnnAssign, ast.Return, ast.Assert)
STATEMENT_CHARACTER_LIMIT = 600


def _line_text(line: str) -> str:
    return line[:-2] if line.endswith("\r\n") else line[:-1] if line.endswith("\n") else line


def _matching_end(lines: list[str], start: int, delimiter: str, *, tabs: bool = False) -> int | None:
    for index in range(start, len(lines)):
        text = _line_text(lines[index])
        if (text.lstrip("\t") if tabs else text) == delimiter:
            return index
    return None


def _blocks(text: str) -> tuple[list[dict], dict]:
    lines = text.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    blocks = []
    notes = {"unsupported_heredoc_headers": 0, "unclosed_heredocs": 0}
    index = 0
    while index < len(lines):
        line = _line_text(lines[index])
        header = next((match for pattern in _HEADERS if (match := pattern.fullmatch(line)) is not None), None)
        if header is not None:
            end = _matching_end(lines, index + 1, header["delimiter"])
            if end is None:
                notes["unclosed_heredocs"] += 1
                break
            body_start = offsets[index] + len(lines[index])
            body_end = offsets[end]
            blocks.append({"header_line": index + 1, "body_start": body_start, "body_end": body_end,
                           "body": text[body_start:body_end], "delimiter_quoted": bool(header["q"])})
            index = end + 1
            continue
        opaque = list(_ANY_HEREDOC.finditer(line)) if not line.lstrip().startswith("#") else []
        if opaque:
            notes["unsupported_heredoc_headers"] += 1
            after = index + 1
            for candidate in opaque:
                end = _matching_end(lines, after, candidate["delimiter"], tabs=bool(candidate["tabs"]))
                if end is None:
                    notes["unclosed_heredocs"] += 1
                    after = len(lines)
                    break
                after = end + 1
            index = after
        else:
            index += 1
    return blocks, notes


def _character_column(line: str, byte_column: int) -> int:
    """Python AST columns are UTF-8 byte offsets, unlike proposal string slices."""
    return len(line.encode("utf-8")[:byte_column].decode("utf-8"))


def _statement_offsets(body: str, node: ast.AST) -> tuple[int, int]:
    lines = body.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    start = offsets[node.lineno - 1] + _character_column(lines[node.lineno - 1], node.col_offset)
    end = offsets[node.end_lineno - 1] + _character_column(lines[node.end_lineno - 1], node.end_col_offset)
    return start, end


def _source_order(source: dict) -> tuple:
    locator = source["locator"]
    match = re.fullmatch(r"tool_calls/(\d+)/keystrokes_excerpt", locator["field"])
    return (source["evidence_ref"], locator["source_index"], locator["step_index"],
            int(match[1]) if match else -1, locator["segment_index"])


def build_submitted_code_index(state: dict, max_facts: int = 24) -> tuple[dict | None, dict]:
    """Return the existing fact-card format and a separate transparent count sidecar.

    Every short Assign/AnnAssign/Return/Assert node is a candidate, regardless of
    names or contents. If capped, blocks receive round-robin slots; chosen facts
    are then restored to original source order. No semantic target is selected.
    """
    if isinstance(max_facts, bool) or not isinstance(max_facts, int) or not 1 <= max_facts <= 32:
        raise ValueError("Submitted-code fact cap must be between one and 32")
    all_sources = flat_proposal_sources(state)
    sources = sorted((source for source in all_sources if source["source_role"] == "recorded_tool_input"), key=_source_order)
    notes: dict[str, Any] = {
        "schema": "research1.submitted-code-navigation-notes.v1",
        "navigation_scope": "Submitted Python source statements from recorded tool input only; neither file write nor execution is verified.",
        "selection": "All short Assign/AnnAssign/Return/Assert nodes; deterministic block round-robin cap, then original source order.",
        "source_filter": "Inherits flat_proposal_sources exclusions and bounds; original evidence remains authoritative.",
        "max_facts": max_facts, "statement_character_limit": STATEMENT_CHARACTER_LIMIT,
        "recorded_tool_input_segments": len(sources), "recognized_blocks": 0,
        "unsupported_heredoc_headers": 0, "unclosed_heredocs": 0,
        "syntax_invalid_blocks": 0, "statement_nodes_total": 0,
        "overlength_statements": 0, "fact_validation_exclusions": 0, "duplicate_facts": 0,
        "candidate_total": 0, "selected_count": 0, "truncated": False, "blocks": [],
    }
    queues, seen = [], set()
    for source in sources:
        blocks, segment_notes = _blocks(source["text"])
        for key, count in segment_notes.items():
            notes[key] += count
        for block in blocks:
            block_index = len(notes["blocks"])
            locator = source["locator"]
            detail = {"block_index": block_index, "source_id": source["source_id"],
                "evidence_ref": source["evidence_ref"], "field": locator["field"],
                "step_index": locator["step_index"], "segment_index": locator["segment_index"],
                "header_line": block["header_line"], "body_start": block["body_start"], "body_end": block["body_end"],
                "delimiter_quoted": block["delimiter_quoted"], "syntax_valid": False,
                "statement_nodes_total": 0, "candidate_count": 0, "selected_count": 0}
            notes["blocks"].append(detail)
            notes["recognized_blocks"] += 1
            try:
                tree = ast.parse(block["body"])
            except (SyntaxError, ValueError, RecursionError):
                notes["syntax_invalid_blocks"] += 1
                continue
            detail["syntax_valid"] = True
            nodes = sorted((node for node in ast.walk(tree) if isinstance(node, _STATEMENTS)),
                key=lambda node: (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset))
            detail["statement_nodes_total"] = len(nodes)
            notes["statement_nodes_total"] += len(nodes)
            candidates = []
            for node in nodes:
                relative_start, relative_end = _statement_offsets(block["body"], node)
                start, end = block["body_start"] + relative_start, block["body_start"] + relative_end
                if end - start > STATEMENT_CHARACTER_LIMIT:
                    notes["overlength_statements"] += 1
                    continue
                try:
                    fact = make_proposal_fact(state, source["evidence_ref"], field=locator["field"],
                        step_index=locator["step_index"], segment_index=locator["segment_index"], start=start, end=end)
                except (ValueError, KeyError, TypeError):
                    notes["fact_validation_exclusions"] += 1
                    continue
                if fact["fact_id"] in seen:
                    notes["duplicate_facts"] += 1
                    continue
                seen.add(fact["fact_id"])
                candidates.append({"fact": fact, "order": (*_source_order(source), start, end), "block_index": block_index})
            detail["candidate_count"] = len(candidates)
            notes["candidate_total"] += len(candidates)
            queues.append(candidates)
    selected = []
    position = 0
    while len(selected) < max_facts and any(position < len(queue) for queue in queues):
        for queue in queues:
            if position < len(queue) and len(selected) < max_facts:
                candidate = queue[position]
                selected.append(candidate)
                notes["blocks"][candidate["block_index"]]["selected_count"] += 1
        position += 1
    selected.sort(key=lambda row: row["order"])
    notes["selected_count"] = len(selected)
    notes["truncated"] = len(selected) < notes["candidate_total"]
    facts = [row["fact"] for row in selected]
    return (make_fact_card(state, facts) if facts else None), notes


def build_submitted_code_card(state: dict, max_facts: int = 24) -> dict | None:
    return build_submitted_code_index(state, max_facts=max_facts)[0]
