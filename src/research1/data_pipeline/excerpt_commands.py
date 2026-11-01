"""Conservative, pure display guard for exact retained tool-input lines.

No shell recognition, source reconstruction, success or producer inference.
Each fragment is inspected independently. A command-looking observation with
no exact known input match remains untouched.
"""
from __future__ import annotations

from typing import Any

from research1.data_pipeline.observation_validator import _is_displayed_command_line


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _lines(excerpt: dict, segment: dict):
    """Yield original text lines and whether both boundaries are proven.

    AST/byte coordinates are unrelated here: selector offsets count Unicode
    characters. Only literal newline boundaries and explicit source ends count.
    """
    if not isinstance(excerpt, dict) or not isinstance(segment, dict):
        return
    text, start, end = segment.get("text"), segment.get("start"), segment.get("end")
    if (not isinstance(text, str) or not _integer(start) or start < 0 or not _integer(end)
            or end != start + len(text)):
        return
    source_chars = excerpt.get("source_chars")
    if source_chars is not None and (not _integer(source_chars) or source_chars < end):
        return
    relative = 0
    for line in text.splitlines(keepends=True):
        line_start, line_end = start + relative, start + relative + len(line)
        starts_at_boundary = line_start == 0 or (relative > 0 and text[relative - 1] == "\n")
        ends_at_boundary = line.endswith("\n") or (_integer(source_chars) and line_end == source_chars)
        yield line, bool(starts_at_boundary and ends_at_boundary)
        relative += len(line)


def known_complete_keystroke_lines(step: dict) -> set[str]:
    """Only complete exact lines from this same step's retained input pieces."""
    found = set()
    if not isinstance(step, dict):
        return found
    for call in step.get("tool_calls") or []:
        arguments = call.get("arguments") if isinstance(call, dict) else None
        excerpt = arguments.get("keystrokes_excerpt") if isinstance(arguments, dict) else None
        if not isinstance(excerpt, dict) or not isinstance(excerpt.get("segments"), list):
            continue
        for segment in excerpt["segments"]:
            for line, complete in _lines(excerpt, segment):
                if complete and line.strip():
                    found.add(line.strip())
    return found


def observation_excerpt_command_only(step: dict, segment_index: int) -> bool:
    """True only when every nonblank line is already known display/context.

    Existing prompt/command decorations use the unchanged old validator. The
    only new exclusion is an exact stripped complete observation line matching
    a proven complete retained input line in the same step. Mixed real text,
    partial observation lines and uncertain fragment boundaries stay eligible
    under the existing validator; no new positive evidence is introduced.
    """
    if not isinstance(step, dict) or not _integer(segment_index) or segment_index < 0:
        return False
    excerpt = step.get("observation_excerpt")
    if not isinstance(excerpt, dict) or not isinstance(excerpt.get("segments"), list) or segment_index >= len(excerpt["segments"]):
        return False
    segment = excerpt["segments"][segment_index]
    known = known_complete_keystroke_lines(step)
    nonempty = False
    for line, complete in _lines(excerpt, segment):
        stripped = line.strip()
        if not stripped:
            continue
        nonempty = True
        if stripped == "New Terminal Output:" or _is_displayed_command_line(line, step):
            continue
        if complete and stripped in known:
            continue
        return False
    return nonempty
