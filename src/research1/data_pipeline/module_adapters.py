"""Explicit verification adapters. Historical protocols keep their default."""
from __future__ import annotations

LEGACY_VERSION='research1.module-contracts.v2'
STRICT_VERSION='research1.module-contracts.v3'


def get_adapter(value=None):
    if isinstance(value,dict):
        value=value.get('verification_registry_version')
    if value is None or value in (LEGACY_VERSION,'v2'):
        from research1.data_pipeline import modules
        return modules
    if value in (STRICT_VERSION,'v3'):
        from research1.data_pipeline import modules_v3
        return modules_v3
    raise ValueError('Unsupported verification registry version: '+str(value))


def planning_constraints() -> dict:
    """Generic feasibility rules, independent of any task or mechanism."""
    return {'schema':'research1.single-concrete-visible-path.v1','requirements':[
        'Choose exactly one concrete feasible implementation path; do not offer ambiguous alternative or/or routes.',
        'Name exact existing fully-qualified entry-point symbols and files. Use only frozen allowed symbols and explicitly predeclared helper names.',
        'Do not add or subclass classes, change registrations, create variant files or expand the declared single-module scope.',
        'For every needed runtime value name its actual existing parameter/member and reachable data path. Do not invent hidden state, absent arguments or aliases between distinct runtime types.',
        'State observable target behavior and preservation checks; expected future benefit remains a hypothesis, not a prerequisite proof.'
    ]}
