"""Deterministic plan-field index for opt-in semantic field-complete review.

The internal field catalog binds exact original plan strings for reviewer-side
audit only; it is not teacher input. Whole-field coverage indexes entire plan
strings and does not mechanically prove every atomic sub-clause; independent
semantic acceptance remains required.
"""
from __future__ import annotations

import hashlib
from typing import Any

from research1.data_pipeline.policy import canonical, digest

POLICY_ID = 'proposal-plan-precision.v1.field_audit'
CATALOG_SCHEMA = 'proposal-plan-field-catalog.v1'
_FIELD_PATHS = ('problem_hypothesis', 'intended_change')
_LIST_FIELDS = ('preserve_behaviors', 'verification_plan')
_PUBLIC_RULES = (
    'Every concrete source or object identity, index or count, literal value or type, '
    'and runtime dataflow claim in your plan must be accurate against the supplied '
    'decision-time sources.',
    'Distinguish exact quoted strings, absent or null values, and defined or captured inputs; '
    'do not infer missing producers or invocation inputs.',
    'Explicit hypotheses and future intentions remain unproven until separately observed; '
    'planned verification steps describe intended checks, not completed execution.',
    'Precise runtime values must actually reach the proposed function or behavior within '
    'the plan\'s declared single-module scope and stated data paths.',
    'One supported witness or clause does not excuse another unsupported concrete claim '
    'in the same or a different part of your plan.',
)


def public_rules() -> list[str]:
    return list(_PUBLIC_RULES)


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def build_plan_field_catalog(plan: dict[str, Any]) -> dict[str, Any]:
    """Index every nonempty string in the scoped plan narrative fields."""
    items: list[dict[str, Any]] = []

    def append(field_path: str, text: str) -> None:
        if not isinstance(text, str) or not text:
            return
        row = {'field_path': field_path, 'text': text, 'text_sha256': _text_sha(text)}
        row['id'] = 'plan_field_' + digest({'field_path': field_path, 'text': text})[:32]
        items.append(row)

    for key in _FIELD_PATHS:
        append(key, plan[key])
    for key in _LIST_FIELDS:
        for index, value in enumerate(plan[key]):
            append(f'{key}[{index}]', value)
    return {'schema': CATALOG_SCHEMA, 'plan_sha256': digest(plan), 'items': items}


def catalog_sha256(catalog: dict[str, Any]) -> str:
    return digest(catalog)
