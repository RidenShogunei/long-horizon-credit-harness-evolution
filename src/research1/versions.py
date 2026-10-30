"""Immutable harness version directories with hash verification."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from research1.contracts import State, write_json

BASELINE_BUNDLE = {
    "agent_loop": "baseline",
    "observation": "baseline",
    "context_mgmt": "baseline",
    "tools": "baseline",
    "verification": "baseline",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(modules_root: Path, relative_files: list[str] | None = None) -> dict[str, str]:
    root = modules_root.resolve()
    out: dict[str, str] = {}
    if relative_files is not None:
        for rel in relative_files:
            p = root / rel
            if p.is_file():
                out[rel] = sha256_file(p)
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            out[str(p.relative_to(root))] = sha256_file(p)
    return out


def verify_against_snapshot(
    modules_root: Path, snapshot: dict[str, Any]
) -> tuple[bool, list[str]]:
    expected = snapshot.get("file_sha256") or {}
    mismatches: list[str] = []
    for rel, exp in expected.items():
        p = modules_root / rel
        if not p.is_file():
            mismatches.append(f"missing:{rel}")
            continue
        got = sha256_file(p)
        if got != exp:
            mismatches.append(f"mismatch:{rel}")
    return (len(mismatches) == 0), mismatches


def materialize_version(
    *,
    versions_root: Path,
    version_id: str,
    source_modules: Path,
    parent_id: str | None,
    active_bundle: dict[str, str] | None = None,
    note: str = "",
) -> State:
    """Copy modules into an immutable version directory."""
    dest = versions_root / version_id / "modules"
    if dest.exists():
        raise FileExistsError(f"version already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_modules, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = hash_tree(dest)
    bundle = dict(active_bundle or BASELINE_BUNDLE)
    state = State(
        version_id=version_id,
        parent_id=parent_id,
        modules_root=str(dest.resolve()),
        active_bundle=bundle,
        file_sha256=hashes,
        note=note,
    )
    write_json(versions_root / version_id / "state.json", state)
    pin = {"composer": "static", "bundle": bundle}
    (versions_root / version_id / "active_bundle.json").write_text(
        json.dumps(pin, indent=2) + "\n"
    )
    return state


def apply_unified_diff(
    *,
    parent_modules: Path,
    versions_root: Path,
    child_version_id: str,
    parent_id: str,
    patch_text: str,
    allowed_prefixes: list[str] | None = None,
    locked_module: str | None = None,
) -> tuple[State | None, str]:
    """Apply a unified diff under allowed paths. Returns (state, reject_reason)."""
    import subprocess
    import tempfile

    allowed = allowed_prefixes or [
        "agent_loop/",
        "observation/",
        "tools/",
        "context_mgmt/",
        "verification/",
        "tool_helper/",
    ]
    # Reject patches that touch outside allowed prefixes.
    for line in patch_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            # formats: --- a/path or --- path
            raw = line[4:].strip()
            if raw == "/dev/null":
                continue
            if raw.startswith("a/") or raw.startswith("b/"):
                raw = raw[2:]
            # strip timestamp
            path = raw.split("\t")[0]
            if path.startswith("/"):
                return None, f"absolute path in patch: {path}"
            if locked_module and not (
                path.startswith(f"{locked_module}/") or path == f"{locked_module}"
            ):
                # allow tool_helper when locked_module=tools
                if not (locked_module == "tools" and path.startswith("tool_helper/")):
                    return None, f"patch touches non-locked module path: {path}"
            if not any(path.startswith(p) or path == p.rstrip("/") for p in allowed):
                return None, f"patch outside editable paths: {path}"

    with tempfile.TemporaryDirectory(prefix="research1_patch_") as td:
        work = Path(td) / "modules"
        shutil.copytree(parent_modules, work, ignore=shutil.ignore_patterns("__pycache__"))
        patch_file = Path(td) / "change.diff"
        patch_file.write_text(patch_text)
        proc = subprocess.run(
            ["patch", "-p1", "--forward", "--batch", "-i", str(patch_file)],
            cwd=work,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return None, f"apply-failure: {proc.stderr.strip() or proc.stdout.strip()}"
        # Move into versions root via materialize from work
        dest_parent = versions_root / child_version_id
        if dest_parent.exists():
            return None, f"version exists: {child_version_id}"
        state = materialize_version(
            versions_root=versions_root,
            version_id=child_version_id,
            source_modules=work,
            parent_id=parent_id,
            note="patched",
        )
        return state, ""


def load_state(versions_root: Path, version_id: str) -> State:
    return State.from_dict(json.loads((versions_root / version_id / "state.json").read_text()))
