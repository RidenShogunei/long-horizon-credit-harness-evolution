"""Validate and label completed real continuation paths without running models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from research1 import EXIT_FAIL, EXIT_INCOMPLETE, EXIT_PASS


def label_continuations(input_path: Path, output_path: Path, *, expected_paths: int = 2, write: bool = True) -> tuple[int, dict]:
    from research1.contracts import MCPath
    from research1.returns import aggregate_mc_paths

    if not input_path.is_file():
        return EXIT_INCOMPLETE, {"status": "INCOMPLETE", "reason": "No completed continuation file", "input": str(input_path)}
    try:
        content = input_path.read_text(encoding="utf-8")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if not rows:
            return EXIT_INCOMPLETE, {"status": "INCOMPLETE", "reason": "No real continuation paths"}
        grouped: dict[str, list] = {}
        for row in rows:
            path = MCPath.from_dict(row)
            grouped.setdefault(path.candidate_id, []).append(path)
        targets = []
        for candidate_id, paths in sorted(grouped.items()):
            label = aggregate_mc_paths(paths, expected_paths=expected_paths).to_dict()
            payload = json.dumps(label, sort_keys=True, ensure_ascii=False)
            label_id = "mc_" + hashlib.sha256(payload.encode()).hexdigest()[:24]
            targets.append({
                "schema": "research1.mc_targets.v1", "label_id": label_id,
                "candidate_id": candidate_id, "label": label,
                "source_sha256": hashlib.sha256(content.encode()).hexdigest(),
            })
    except (ValueError, KeyError, TypeError) as exc:
        return EXIT_INCOMPLETE, {"status": "INCOMPLETE", "reason": str(exc), "targets_written": 0}

    if write:
        # Validate the complete input before touching any previous target file.
        existing = {}
        try:
            if output_path.is_file():
                for line in output_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        existing[row["label_id"]] = row
        except (ValueError, KeyError, TypeError) as exc:
            return EXIT_FAIL, {"status": "FAIL", "reason": f"Existing targets need repair: {exc}", "targets_written": 0}
        for row in targets:
            existing[row["label_id"]] = row
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, delete=False) as handle:
                temp_name = handle.name
                for row in existing.values():
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temp_name, output_path)
        finally:
            if temp_name and Path(temp_name).exists():
                Path(temp_name).unlink()
    return EXIT_PASS, {"status": "PASS", "candidates": len(targets), "paths": len(rows), "written": write, "output": str(output_path) if write else None}
