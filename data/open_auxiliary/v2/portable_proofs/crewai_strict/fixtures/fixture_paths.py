"""Safe fixture path helpers for crewai_strict verification."""
from __future__ import annotations

import shutil
from pathlib import Path

BATCH = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = (BATCH / "fixtures" / "workspace_root").resolve()


def ensure_workspace_root() -> Path:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    return WORKSPACE_ROOT


def assert_under_workspace(path: Path) -> Path:
    resolved = path.resolve()
    root = ensure_workspace_root()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"fixture path {resolved} escapes workspace root {root}")
    return resolved


def safe_rmtree(path: Path) -> None:
    target = assert_under_workspace(path)
    if target == WORKSPACE_ROOT:
        raise ValueError("refusing to delete workspace root")
    if target.exists():
        shutil.rmtree(target)


def fixture_path(*parts: str, mkdir: bool = True) -> Path:
    p = assert_under_workspace(WORKSPACE_ROOT.joinpath(*parts))
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p
