"""Taskwise returns and pure aggregation of completed continuation paths.

The positive part is taken per task BEFORE weighting. Nothing here runs an
evaluation or model call. Scalar-only legacy paths remain diagnostic only.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

from research1.contracts import MCPath, MCValueLabel, ReturnLabel

TASKWISE_METRIC = "taskwise-parent-step-v2"


class IncompleteReturnError(ValueError):
    """Required task results or fixed R repetitions are missing."""


@dataclass(frozen=True)
class StepScores:
    """Legacy version mean: insufficient to recover taskwise regression."""
    version_id: str
    mean_score: float


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalized_task_weights(task_ids: Sequence[str], weights: Mapping[str, float] | None = None) -> dict[str, float]:
    tasks = list(task_ids)
    if not tasks or len(set(tasks)) != len(tasks) or any(not isinstance(t, str) or not t for t in tasks):
        raise ValueError("expected_tasks must be nonempty, unique task IDs")
    if weights is None:
        return {task: 1.0 / len(tasks) for task in tasks}
    if set(weights) != set(tasks):
        raise ValueError("task weights must have exactly the expected task IDs")
    raw = {task: _finite_number(weights[task], f"weight[{task}]") for task in tasks}
    if any(value <= 0 for value in raw.values()):
        raise ValueError("all frozen task weights must be positive")
    total = math.fsum(raw.values())
    if not math.isfinite(total):
        raise ValueError("task weight sum must be finite")
    return {task: value / total for task, value in raw.items()}


def mean_score(task_scores: Mapping[str, float]) -> float:
    if not task_scores:
        raise ValueError("empty score table")
    return math.fsum(_finite_number(v, "task score") for v in task_scores.values()) / len(task_scores)


def parent_step_deltas(path: Sequence[StepScores]) -> list[float]:
    """Legacy diagnostic: difference in aggregate version means."""
    return [path[i].mean_score - path[i - 1].mean_score for i in range(1, len(path))]


def best_ancestor_deltas(path: Sequence[StepScores]) -> list[float]:
    out: list[float] = []
    best = path[0].mean_score if path else 0.0
    for step in path[1:]:
        out.append(step.mean_score - best)
        best = max(best, step.mean_score)
    return out


def cumulative_regression_parent_step(deltas: Sequence[float]) -> float:
    """Legacy scalar diagnostic; formal labels take the per-task positive part."""
    return float(math.fsum(max(0.0, -d) for d in deltas))


def cumulative_regression_best_ancestor(deltas: Sequence[float]) -> float:
    return float(math.fsum(max(0.0, -d) for d in deltas))


def terminal_gain(path: Sequence[StepScores]) -> float:
    if not path:
        raise ValueError("empty path")
    return path[-1].mean_score - path[0].mean_score


def u_value(terminal: float, cum_reg: float, lambda_value: float = 0.5) -> float:
    penalty = _finite_number(lambda_value, "lambda_value")
    if penalty < 0:
        raise ValueError("lambda_value must be nonnegative")
    return terminal - penalty * cum_reg


def label_branch(*, label_id: str, branch_id: str, path: Sequence[StepScores], lambda_value: float = 0.5) -> ReturnLabel:
    """Keep scalar API compatibility, explicitly marked legacy and incomplete."""
    for step in path:
        _finite_number(step.mean_score, "mean_score")
    ps = parent_step_deltas(path)
    tg = terminal_gain(path)
    cr = cumulative_regression_parent_step(ps)
    return ReturnLabel(
        label_id=label_id, root_version_id=path[0].version_id, leaf_version_id=path[-1].version_id,
        branch_id=branch_id, terminal_gain=tg, cumulative_regression=cr, parent_step_deltas=ps,
        best_ancestor_deltas=best_ancestor_deltas(path), u_lambda=u_value(tg, cr, lambda_value),
        lambda_value=lambda_value, metric_definition="legacy-version-mean-v1", label_scope="legacy_diagnostic",
        is_complete=False, incomplete_reasons=["scalar means cannot recover taskwise regression or verify R"],
        version_order=[step.version_id for step in path],
    )


def recompute_from_tables(
    *, branch_id: str, version_order: Sequence[str],
    score_tables: Mapping[str, Mapping[str, float]], lambda_value: float = 0.5,
    label_id: str | None = None, expected_tasks: Sequence[str] | None = None,
    task_weights: Mapping[str, float] | None = None,
    trial_counts: Mapping[str, Mapping[str, int]] | None = None,
    required_repetitions: int | None = None, label_scope: str = "real_evolution",
) -> ReturnLabel:
    """Use one frozen task set at every version; never drop missing task scores.

    Missing scores or insufficient supplied counts raise IncompleteReturnError.
    Old calls without frozen tasks/counts can inspect arithmetic but receive an
    incomplete label. A complete label requires explicit expected_tasks, fixed R,
    and counts exactly equal to R (no optional-stopping subset selection).
    """
    order = list(version_order)
    if len(order) < 2 or len(set(order)) != len(order):
        raise ValueError("version_order must contain at least two distinct version IDs")
    if order[0] not in score_tables:
        raise IncompleteReturnError(f"missing score table for {order[0]}")
    tasks = list(expected_tasks if expected_tasks is not None else score_tables[order[0]])
    weights = normalized_task_weights(tasks, task_weights)
    scores: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    reasons: list[str] = []
    if expected_tasks is None:
        reasons.append("expected task set was inferred, not explicitly frozen")
    if required_repetitions is None or trial_counts is None:
        reasons.append("fixed R and complete per-version trial counts were not supplied")
    if required_repetitions is not None:
        _positive_int(required_repetitions, "required_repetitions")
    for version in order:
        table = score_tables.get(version)
        if table is None or set(table) != set(tasks):
            raise IncompleteReturnError(f"{version}: score task set differs from frozen expected_tasks")
        if any(table[task] is None for task in tasks):
            raise IncompleteReturnError(f"{version}: missing task reward")
        scores[version] = {task: _finite_number(table[task], f"{version}/{task}") for task in tasks}
        if trial_counts is not None:
            observed = trial_counts.get(version)
            if observed is None or set(observed) != set(tasks):
                raise IncompleteReturnError(f"{version}: trial count task set is incomplete")
            counts[version] = {}
            for task in tasks:
                count = observed[task]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"{version}/{task}: trial count must be a nonnegative integer")
                if required_repetitions is not None and count != required_repetitions:
                    raise IncompleteReturnError(f"{version}/{task}: scored trials {count}, fixed R={required_repetitions}")
                counts[version][task] = count
    deltas = [{task: scores[child][task] - scores[parent][task] for task in tasks}
              for parent, child in zip(order, order[1:])]
    # Take the positive part BEFORE weighting over tasks.
    regression = math.fsum(weights[t] * max(0.0, -step[t]) for step in deltas for t in tasks)
    gain = math.fsum(weights[t] * (scores[order[-1]][t] - scores[order[0]][t]) for t in tasks)
    scalar_path = [StepScores(v, math.fsum(weights[t] * scores[v][t] for t in tasks)) for v in order]
    return ReturnLabel(
        label_id=label_id or f"label_{branch_id}", branch_id=branch_id,
        root_version_id=order[0], leaf_version_id=order[-1], terminal_gain=gain,
        cumulative_regression=regression, parent_step_deltas=parent_step_deltas(scalar_path),
        best_ancestor_deltas=best_ancestor_deltas(scalar_path), u_lambda=u_value(gain, regression, lambda_value),
        lambda_value=lambda_value, metric_definition=TASKWISE_METRIC, label_scope=label_scope,
        is_complete=not reasons, incomplete_reasons=reasons, version_order=order,
        task_ids=tasks, task_weights=weights, per_task_deltas=deltas, per_task_scores=scores,
        trial_counts=counts, required_repetitions=required_repetitions,
    )


def aggregate_mc_paths(paths: Sequence[MCPath], *, expected_paths: int = 2) -> MCValueLabel:
    """Aggregate exactly fixed M complete real paths; never start a rollout.

    Every supplied path must be complete. Parent, first implementation, policy,
    L, tasks, weights, R and lambda must match. IDs/seeds check declared provenance;
    the collector remains responsible for actually running independent paths.
    """
    _positive_int(expected_paths, "expected_paths")
    if expected_paths < 2:
        raise ValueError("MC uncertainty requires expected_paths >= 2")
    if len(paths) != expected_paths:
        raise IncompleteReturnError(f"expected exactly M={expected_paths} paths, received {len(paths)}")
    labels: list[ReturnLabel] = []
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_seeds: set[str] = set()
    condition: tuple[object, ...] | None = None
    first_scores: tuple[object, ...] | None = None
    for path in paths:
        if path.path_kind != "real_evolution":
            raise ValueError(f"{path.path_id}: only real_evolution paths can form MC labels")
        if any(not isinstance(v, str) or not v.strip() for v in (path.path_id, path.candidate_id, path.policy_id)):
            raise ValueError("path_id, candidate_id and policy_id must be nonempty strings")
        _positive_int(path.horizon, "horizon")
        if len(path.version_order) != path.horizon + 2:
            raise IncompleteReturnError(f"{path.path_id}: expected parent + first edit + L={path.horizon} successors")
        if not isinstance(path.first_implementation_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", path.first_implementation_hash):
            raise ValueError("first_implementation_hash must be a SHA-256 hex digest")
        if isinstance(path.continuation_seed, bool) or not isinstance(path.continuation_seed, (str, int)) or not str(path.continuation_seed).strip():
            raise ValueError("continuation_seed must identify independent continuation randomness")
        seed = str(path.continuation_seed)
        if path.path_id in seen_ids or seed in seen_seeds:
            raise ValueError("duplicate MC path ID or continuation seed; R repetitions are not independent M paths")
        seen_ids.add(path.path_id)
        seen_seeds.add(seed)
        label = recompute_from_tables(
            branch_id=path.path_id, version_order=path.version_order, score_tables=path.score_tables,
            expected_tasks=path.expected_tasks, task_weights=path.task_weights,
            trial_counts=path.trial_counts, required_repetitions=path.required_repetitions,
            lambda_value=path.lambda_value,
        )
        if not label.is_complete:
            raise IncompleteReturnError(f"{path.path_id}: {label.incomplete_reasons}")
        signature = (path.candidate_id, label.root_version_id, path.version_order[1],
                     path.first_implementation_hash.lower(), path.policy_id, path.horizon,
                     tuple(sorted(label.task_weights.items())), path.required_repetitions, path.lambda_value)
        # Freeze prefix evaluations instead of selecting favorable measurements per continuation.
        prefix = tuple(tuple(sorted(label.per_task_scores[v].items())) for v in path.version_order[:2])
        if condition is not None and (signature != condition or prefix != first_scores):
            raise ValueError("MC paths mix parent/candidate/implementation, policy, horizon, tasks, weights, R, lambda, or shared prefix scores")
        condition, first_scores = signature, prefix
        labels.append(label)
        rows.append({"path_id": path.path_id, "continuation_seed": path.continuation_seed, "return_label": label.to_dict()})
    first, base = paths[0], labels[0]
    gains = [label.terminal_gain for label in labels]
    regressions = [label.cumulative_regression for label in labels]
    utilities = [label.u_lambda for label in labels]
    return MCValueLabel(
        candidate_id=first.candidate_id, root_version_id=base.root_version_id,
        first_implementation_hash=first.first_implementation_hash.lower(), policy_id=first.policy_id,
        horizon=first.horizon, n_paths=len(paths), expected_paths=expected_paths,
        mean_terminal_gain=statistics.mean(gains), std_terminal_gain=statistics.stdev(gains),
        mean_cumulative_regression=statistics.mean(regressions), std_cumulative_regression=statistics.stdev(regressions),
        mean_u_lambda=statistics.mean(utilities), std_u_lambda=statistics.stdev(utilities),
        lambda_value=first.lambda_value, task_ids=base.task_ids, task_weights=base.task_weights,
        required_repetitions=first.required_repetitions, paths=rows,
    )
