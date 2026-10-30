"""Immutable JSON-serializable contracts for research1-v1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from research1 import PROTOCOL_ID, SCHEMA_VERSION

FailureLayer = Literal["model", "harness", "container", "scorer", "infrastructure", "unknown"]
OutcomeClass = Literal["success", "no-effect", "regression", "apply-failure"]
RunStatus = Literal["PASS", "FAIL", "INCOMPLETE"]


@dataclass
class State:
    """Harness version node."""

    version_id: str
    parent_id: str | None
    modules_root: str
    active_bundle: dict[str, str]
    file_sha256: dict[str, str]
    protocol_id: str = PROTOCOL_ID
    schema_version: str = SCHEMA_VERSION
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "State":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EditPlan:
    plan_id: str
    parent_version_id: str
    locked_module: str
    plan_text: str
    intended_files: list[str] = field(default_factory=list)
    candidate_label: str = "A"
    protocol_id: str = PROTOCOL_ID
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EditAttempt:
    attempt_id: str
    plan_id: str
    parent_version_id: str
    child_version_id: str | None
    patch_text: str
    applied: bool
    outcome_class: OutcomeClass
    reject_reason: str = ""
    protocol_id: str = PROTOCOL_ID
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRun:
    run_id: str
    version_id: str
    task_name: str
    trial_index: int
    reward: float | None
    failure_layer: FailureLayer
    status: RunStatus
    output_dir: str
    active_bundle: dict[str, str] = field(default_factory=dict)
    model: str = ""
    temperature: float | None = None
    provider_id: str = ""
    notes: str = ""
    protocol_id: str = PROTOCOL_ID
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transition:
    """Parent → child edge after an edit attempt."""

    transition_id: str
    parent_version_id: str
    child_version_id: str
    plan_id: str
    attempt_id: str
    parent_step_scores: dict[str, float]
    child_step_scores: dict[str, float]
    protocol_id: str = PROTOCOL_ID
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReturnLabel:
    """Stored return components; always keep gain and regression separate."""

    label_id: str
    root_version_id: str
    leaf_version_id: str
    branch_id: str
    terminal_gain: float
    cumulative_regression: float
    parent_step_deltas: list[float]
    best_ancestor_deltas: list[float]
    u_lambda: float
    lambda_value: float = 0.5
    regression_primary: str = "parent-step"
    protocol_id: str = PROTOCOL_ID
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "to_dict"):
        payload = obj.to_dict()
    elif isinstance(obj, list):
        payload = [x.to_dict() if hasattr(x, "to_dict") else x for x in obj]
    else:
        payload = obj
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_task_run(d: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("run_id", "version_id", "task_name", "trial_index", "failure_layer", "status"):
        if key not in d:
            errors.append(f"missing {key}")
    if d.get("reward") is None and d.get("failure_layer") == "unknown":
        errors.append("unknown failure_layer must not be treated as reward=0; keep reward null")
    if d.get("reward") == 0 and d.get("failure_layer") == "unknown":
        errors.append("do not coerce unknown to reward=0")
    return errors
