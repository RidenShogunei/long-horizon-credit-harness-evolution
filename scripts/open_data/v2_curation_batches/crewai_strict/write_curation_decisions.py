#!/usr/bin/env python3
import json
from pathlib import Path

STRICT = Path(__file__).resolve().parent
WORK = STRICT.parent
rows = json.loads((STRICT / "curated_examples.json").read_text(encoding="utf-8"))
vr = json.loads((STRICT / "verification_results.json").read_text(encoding="utf-8"))
by_status = {s["source_id"]: s["status"] for s in vr["samples"]}
decisions = {}
for row in rows:
    sid = row["source_id"]
    decisions[sid] = {
        "source_id": sid,
        "decision": "accept" if by_status.get(sid) == "pass" else by_status.get(sid, "pending"),
        "verification_status": row.get("verification_status"),
        "original_teacher_plan": False,
        "plan": row["plan"],
        "curation_notes": row["curation_notes"],
        "source_mapping": row["source_mapping"],
    }
legacy = json.loads(
    (WORK / "crewai_batch" / "curation_decisions.json").read_text(encoding="utf-8")
)
for sid in ("crewaiinc__crewai-2495", "crewaiinc__crewai-2552"):
    entry = dict(legacy[sid])
    entry["immutable_source"] = "work/open_dataset_integration_v2/crewai_batch/curation_decisions.json"
    decisions[sid] = entry
(STRICT / "curation_decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
print(len(decisions))
