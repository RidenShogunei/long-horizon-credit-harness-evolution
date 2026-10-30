"""Recompute S / d_t / terminal_gain / U from per-task score tables.

Primary regression metric is parent-step. best_ancestor is diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from research1.contracts import ReturnLabel


@dataclass(frozen=True)
class StepScores:
    """Mean reward set score at a version along a branch path."""

    version_id: str
    mean_score: float


def mean_score(task_scores: Mapping[str, float]) -> float:
    if not task_scores:
        raise ValueError("empty score table")
    return sum(task_scores.values()) / len(task_scores)


def parent_step_deltas(path: Sequence[StepScores]) -> list[float]:
    """d_t relative to immediate parent: mean(child) - mean(parent)."""
    if len(path) < 2:
        return []
    return [path[i].mean_score - path[i - 1].mean_score for i in range(1, len(path))]


def best_ancestor_deltas(path: Sequence[StepScores]) -> list[float]:
    """Diagnostic: mean(current) - max(mean of ancestors including self's parent chain)."""
    out: list[float] = []
    best = path[0].mean_score if path else 0.0
    for i in range(1, len(path)):
        # best ancestor before this step
        out.append(path[i].mean_score - best)
        best = max(best, path[i].mean_score)
    return out


def cumulative_regression_parent_step(deltas: Sequence[float]) -> float:
    """Sum of positive drops vs parent (process regression primary)."""
    return float(sum(max(0.0, -d) for d in deltas))


def cumulative_regression_best_ancestor(deltas: Sequence[float]) -> float:
    """Diagnostic: sum of positive drops vs best ancestor so far."""
    return float(sum(max(0.0, -d) for d in deltas))


def terminal_gain(path: Sequence[StepScores]) -> float:
    if not path:
        raise ValueError("empty path")
    return path[-1].mean_score - path[0].mean_score


def u_value(terminal: float, cum_reg: float, lambda_value: float = 0.5) -> float:
    return terminal - lambda_value * cum_reg


def label_branch(
    *,
    label_id: str,
    branch_id: str,
    path: Sequence[StepScores],
    lambda_value: float = 0.5,
) -> ReturnLabel:
    ps = parent_step_deltas(path)
    ba = best_ancestor_deltas(path)
    tg = terminal_gain(path)
    cr = cumulative_regression_parent_step(ps)
    return ReturnLabel(
        label_id=label_id,
        root_version_id=path[0].version_id,
        leaf_version_id=path[-1].version_id,
        branch_id=branch_id,
        terminal_gain=tg,
        cumulative_regression=cr,
        parent_step_deltas=list(ps),
        best_ancestor_deltas=list(ba),
        u_lambda=u_value(tg, cr, lambda_value),
        lambda_value=lambda_value,
        regression_primary="parent-step",
    )


def recompute_from_tables(
    *,
    branch_id: str,
    version_order: Sequence[str],
    score_tables: Mapping[str, Mapping[str, float]],
    lambda_value: float = 0.5,
    label_id: str | None = None,
) -> ReturnLabel:
    """Independently recompute returns from version→task score maps."""
    path = [
        StepScores(version_id=vid, mean_score=mean_score(score_tables[vid]))
        for vid in version_order
    ]
    return label_branch(
        label_id=label_id or f"label_{branch_id}",
        branch_id=branch_id,
        path=path,
        lambda_value=lambda_value,
    )
