"""Retain directed historical repairs as auxiliary evidence, not formal SFT."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from research1.collection import append_unique,set_admission,sample_id_for_row

def quarantine_directed(root:Path)->dict:
    source=Path(root)/'datasets/research1/sft/verified_plans.jsonl'
    if not source.exists():return {'quarantined':[],'legacy_remaining':0}
    before=source.read_bytes()
    rows=[json.loads(line) for line in before.decode('utf-8-sig').splitlines() if line.strip()]
    changed=[]
    for row in rows:
        if row.get('source_kind')!='archived_teacher_plan_reimplementation' and row.get('use')!='directed_behavior_repair_init':continue
        sid=sample_id_for_row(row)
        auxiliary=dict(row,sample_id=sid,formal_eligible=False,use='auxiliary_directed_repair',
                       original_use=row.get('use'),original_source_kind=row.get('source_kind'))
        append_unique(Path(root)/'datasets/research1/auxiliary/directed_repairs.jsonl',auxiliary,'sample_id')
        set_admission(Path(root),sid,'revoked',
            'Directed archived repair is auxiliary only: prompted repair target and historical output-format mismatch are not autonomous single-plan SFT',
            evidence={'classification':'auxiliary_directed_repair','sample_id':sid})
        changed.append(sid)
    after=[s for s in source.read_text().splitlines() if s.strip()]
    return {'quarantined':changed,'legacy_remaining':len(after),
            'original_export_sha256':hashlib.sha256(before).hexdigest(),
            'formal_source':'new data campaign exports only; remaining legacy records are not auto-promoted',
            'raw_records_modified':False}

