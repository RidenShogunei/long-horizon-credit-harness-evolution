"""Approved, quota-driven formal expansion after a real completed calibration.

Each child is one immutable preregistered K2/M2/L2 group. Only one child may
run while the formal controller owns its lock, so child budget ceilings and
the sum of all child ledgers enforce the approved global spend bound.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.protocol import load_protocol
from research1.data_pipeline.sft import export_dataset
from research1.data_pipeline.util import atomic_json, digest

_TARGETS = {"origins", "verified_edits", "complete_mc_labels", "mechanisms", "covered_modules", "validation_edits"}
_BUDGET = {"regular", "retries", "total"}
_SAMPLING = {"R", "reward_n", "dev_n"}
_PROFILE = {"model_id", "tokenizer_path", "max_context_tokens"}
_TERMINAL_GROUPS = {"NEEDS_SCALE_LOCK", "INSUFFICIENT_EVIDENCE", "COMPLETE", "DATA_COLLECTION_COMPLETE"}


def _integer(value: Any, name: str, *, zero=False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(name + " must be a " + ("nonnegative" if zero else "positive") + " integer")
    return value


def _validate_scale(config: dict, base: dict) -> dict:
    if not isinstance(config, dict) or set(config) - {"targets", "budget", "sampling", "max_groups", "target_profile"}:
        raise ValueError("Unknown formal scale configuration fields")
    for field, required in (("targets", _TARGETS), ("budget", _BUDGET), ("sampling", _SAMPLING)):
        if not isinstance(config.get(field), dict) or set(config[field]) != required:
            raise ValueError("Formal " + field + " must explicitly provide " + ",".join(sorted(required)))
    for key, value in config["targets"].items():
        _integer(value, "targets." + key)
    if config["targets"]["covered_modules"] > 5:
        raise ValueError("There are only five permitted modules")
    for key, value in config["budget"].items():
        _integer(value, "budget." + key, zero=key == "retries")
    if config["budget"]["total"] != config["budget"]["regular"] + config["budget"]["retries"]:
        raise ValueError("Total budget must equal the explicit regular and retry ceilings")
    for key, value in config["sampling"].items():
        _integer(value, "sampling." + key)
    if config["sampling"]["reward_n"] > len(base["task_sets"]["reward"]) or config["sampling"]["dev_n"] > len(base["task_sets"]["dev"]):
        raise ValueError("Formal task counts exceed the locked reward/dev partitions")
    _integer(config.get("max_groups"), "max_groups")
    profile = config.get("target_profile")
    if profile is not None:
        if not isinstance(profile, dict) or set(profile) - _PROFILE:
            raise ValueError("Student profile exposes only model, local tokenizer and context limit")
        if "max_context_tokens" in profile:
            _integer(profile["max_context_tokens"], "target_profile.max_context_tokens")
    # Round-trip also rejects callbacks and non-finite values in an approval.
    return json.loads(json.dumps(config, ensure_ascii=False, allow_nan=False))


def prepare_formal(campaign: Path, calibration: Path, scale_config: dict) -> dict:
    campaign, calibration = Path(campaign).resolve(), Path(calibration).resolve()
    if campaign == calibration:
        raise ValueError("Formal expansion cannot overwrite its calibration")
    base = load_protocol(calibration)
    status = json.loads((calibration / "status.json").read_text(encoding="utf-8"))
    if status.get("calibration_complete") is not True or base.get("formal_scale") is not None:
        raise ValueError("A real completed calibration is required before formal expansion")
    config = _validate_scale(scale_config, base)
    if (campaign / "protocol.json").exists():
        previous = load_protocol(campaign)
        if previous.get("calibration", {}).get("protocol_hash") != base["protocol_hash"] or previous.get("formal_scale") != config:
            raise ValueError("Formal approval changed; create a new campaign")
        return previous
    campaign.mkdir(parents=True, exist_ok=True)
    if (campaign / "contracts.json").exists():
        raise ValueError("Unfinished formal preparation; inspect the existing campaign")
    shutil.copyfile(calibration / "contracts.json", campaign / "contracts.json")
    protocol = deepcopy(base)
    protocol.pop("protocol_hash", None)
    protocol.update(campaign_id=campaign.name, formal_scale=config, budget=config["budget"], stage_budget={},
                    calibration={"campaign": str(calibration), "protocol_hash": base["protocol_hash"],
                                 "completed_status_sha256": digest(status)},
                    noise_enabled=False, collection_split="train",
                    completion_rule="DATA_COLLECTION_COMPLETE requires every approved quota and training-ready SFT")
    protocol["sampling"].update(config["sampling"], K=2, M=2, L_after=2)
    protocol["sft"]["target_profile"] = config.get("target_profile")
    protocol["protocol_hash"] = digest(protocol)
    atomic_json(campaign / "protocol.json", protocol)
    if config.get("target_profile") is not None:
        atomic_json(campaign / "student_profile.json", config["target_profile"])
    atomic_json(campaign / "status.json", {"status": "PREPARED", "phase": "formal_prepared",
                "protocol_hash": protocol["protocol_hash"], "calibration_complete": True,
                "data_collection_complete": False, "groups_started": 0})
    Ledger(campaign)
    return protocol


def _children(campaign: Path) -> list[Path]:
    return sorted(path.parent for path in (campaign / "groups").glob("group_*/protocol.json"))


def _validated_children(campaign: Path, protocol: dict) -> list[Path]:
    result, seen = [], set()
    for child in _children(campaign):
        spec = load_protocol(child)
        index = spec.get("formal_group_index")
        overrides = {"campaign_id", "formal_scale", "formal_group_of", "formal_group_index", "noise_enabled",
                     "collection_split", "seed", "budget", "stage_budget", "protocol_hash"}
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < protocol["formal_scale"]["max_groups"]
                or index in seen or child.name != f"group_{index + 1:05d}"
                or spec.get("formal_group_of") != protocol["protocol_hash"]
                or spec.get("formal_scale") is not None or spec.get("noise_enabled") is not False
                or spec.get("collection_split") != ("validation" if (index + 1) % 5 == 0 else "train")
                or spec.get("seed") != int(digest({"seed": protocol["seed"], "formal_group": index})[:16], 16)
                or spec.get("sampling") != protocol["sampling"]
                or spec.get("task_sets") != protocol["task_sets"]
                or spec.get("modules_hash") != protocol["modules_hash"]
                or spec.get("policies") != protocol["policies"]
                or any(spec.get(key) != value for key, value in protocol.items() if key not in overrides)
                or set(spec) - set(protocol) - {"formal_group_of", "formal_group_index"}):
            raise ValueError("Unregistered or changed formal child group: " + child.name)
        seen.add(index)
        result.append(child)
    return result


def _usage(campaign: Path, root: Ledger, protocol: dict) -> dict[str, int]:
    # There are normally no root evaluation jobs. Include any nevertheless,
    # so an external approved job cannot become unaccounted spend.
    used = root.status()["budget"]["used"]
    regular, retries = used.get("regular", 0), used.get("retry", 0)
    for child in _validated_children(campaign, protocol):
        used = Ledger(child).status()["budget"]["used"]
        regular += used.get("regular", 0)
        retries += used.get("retry", 0)
    return {"regular": regular, "retries": retries, "total": regular + retries}


def _remaining(protocol: dict, used: dict) -> dict[str, int]:
    budget = protocol["budget"]
    return {key: max(0, budget[key] - used[key]) for key in _BUDGET}


def _merge_child(root: Ledger, child: Path) -> None:
    source = Ledger(child)
    for kind in ("states", "samples", "paths", "values", "comparisons", "transitions"):
        old = dict(root.record_items(kind))
        for key, row in source.record_items(kind):
            previous = old.get(key)
            if previous and previous.get("revoked"):
                continue
            if previous and previous.get("root_lineage_id") and row.get("root_lineage_id") != previous["root_lineage_id"]:
                raise ValueError("Record identity collided across formal groups: " + key)
            root.put_record(kind, key, row)
    # A source reference may have been revoked before a later dependent record
    # existed. Reapply durable root revocations after importing that record.
    with root.connection() as conn:
        revocations = [json.loads(row[0]) for row in conn.execute("SELECT payload FROM events WHERE kind='revocation' ORDER BY created")]
    seen = set()
    for event in revocations:
        reference = event.get("reference")
        if reference and reference not in seen:
            seen.add(reference)
            root.revoke(reference, event.get("reason", "Previously revoked source"))


def _profile(campaign: Path, protocol: dict):
    path = campaign / "student_profile.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else protocol["formal_scale"].get("target_profile")


def _progress(campaign: Path, protocol: dict, ledger: Ledger) -> dict:
    exported = export_dataset(campaign, ledger.records("samples"), target_profile=_profile(campaign, protocol))
    metadata = [json.loads(line) for line in Path(exported["files"]["metadata.jsonl"]).read_text(encoding="utf-8").splitlines()]
    edits = [row for row in metadata if row["action"] == "edit"]
    values = [row for row in ledger.records("values") if not row.get("revoked") and row.get("is_complete") is True
              and row.get("status") == "COMPLETE"]
    counts = {"origins": len({row["root_lineage_id"] for row in metadata}), "verified_edits": len(edits),
              "complete_mc_labels": len(values), "mechanisms": len({row["mechanism_id"] for row in edits if row.get("mechanism_id")}),
              "covered_modules": len({row["target_module"] for row in edits}),
              "validation_edits": sum(row["split"] == "validation" for row in edits)}
    targets = protocol["formal_scale"]["targets"]
    deficits = {key: max(0, targets[key] - counts[key]) for key in _TARGETS}
    return {"counts": counts, "targets": targets, "deficits": deficits, "sft": exported,
            "quotas_met": not any(deficits.values()), "training_ready": exported["training_ready"]}


def _child_protocol(campaign: Path, protocol: dict, index: int, remaining: dict) -> Path:
    child = campaign / "groups" / f"group_{index + 1:05d}"
    if (child / "protocol.json").exists():
        saved = load_protocol(child)
        if saved.get("formal_group_of") != protocol["protocol_hash"] or saved.get("formal_group_index") != index:
            raise ValueError("Formal child conditions differ from their registered group")
        return child
    sample = protocol["sampling"]
    normal = sample["reward_n"] * sample["R"] * 11 + 18 + sample["dev_n"] * sample["R"] * 5
    regular = min(normal, remaining["regular"], remaining["total"])
    retries = min(remaining["retries"], remaining["total"] - regular)
    if regular < 1:
        raise ValueError("No regular evaluation budget remains for a new group")
    child.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(campaign / "contracts.json", child / "contracts.json")
    spec = deepcopy(protocol)
    spec.pop("protocol_hash", None)
    spec.update(campaign_id=child.name, formal_scale=None, formal_group_of=protocol["protocol_hash"],
                formal_group_index=index, noise_enabled=False,
                collection_split="validation" if (index + 1) % 5 == 0 else "train",
                seed=int(digest({"seed": protocol["seed"], "formal_group": index})[:16], 16),
                budget={"regular": regular, "retries": retries, "total": regular + retries},
                stage_budget={"noise": 0, "proposal": 18, "reward": sample["reward_n"] * sample["R"] * 11,
                              "dev": sample["dev_n"] * sample["R"] * 5})
    spec["protocol_hash"] = digest(spec)
    atomic_json(child / "protocol.json", spec)
    Ledger(child)
    return child


def run_formal(campaign: Path, group_runner=None) -> dict:
    campaign = Path(campaign).resolve()
    protocol = load_protocol(campaign, verify_source=group_runner is None)
    if not isinstance(protocol.get("formal_scale"), dict):
        raise ValueError("This campaign has no approved formal quota lock")
    ledger = Ledger(campaign)
    if group_runner is None:
        from research1.data_pipeline.engine import run_campaign
        group_runner = run_campaign
    config = protocol["formal_scale"]
    status = {"status": "RUNNING", "phase": "formal_collection", "protocol_hash": protocol["protocol_hash"],
              "calibration_complete": True, "data_collection_complete": False, "groups_started": 0}

    def save():
        try:
            used = _usage(campaign, ledger, protocol)
            status["aggregate_job_budget"] = {"limits": protocol["budget"], "used": used, "remaining": _remaining(protocol, used)}
        except ValueError as exc:
            status["aggregate_job_budget"] = {"limits": protocol["budget"], "used": None, "remaining": None,
                                               "error": "Cannot trust child accounting: " + str(exc)}
        atomic_json(campaign / "status.json", status)

    with ledger.controller():
        try:
            for child in _validated_children(campaign, protocol):
                _merge_child(ledger, child)
            for index in range(config["max_groups"] + 1):
                progress = _progress(campaign, protocol, ledger)
                status.update(progress)
                if progress["quotas_met"]:
                    status.update(status="DATA_COLLECTION_COMPLETE" if progress["training_ready"] else "NEEDS_STUDENT_PROFILE",
                                  phase="finished", data_collection_complete=progress["training_ready"])
                    break
                if index == config["max_groups"]:
                    status.update(status="EXHAUSTED_EVIDENCE", phase="finished", reason="Approved maximum number of independent groups reached")
                    break
                child = campaign / "groups" / f"group_{index + 1:05d}"
                checkpoint = child / "formal_group_result.json"
                if checkpoint.exists():
                    old = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if old.get("status") in _TERMINAL_GROUPS:
                        status["groups_started"] = max(status["groups_started"], index + 1)
                        continue
                used = _usage(campaign, ledger, protocol)
                remaining = _remaining(protocol, used)
                if remaining["regular"] < 1 or remaining["total"] < 1:
                    status.update(status="BUDGET_EXHAUSTED", phase="finished", reason="Approved global task budget exhausted")
                    break
                child = _child_protocol(campaign, protocol, index, remaining)
                status.update(current_group=str(child), groups_started=index + 1)
                save()
                child_result = group_runner(child)
                if not isinstance(child_result, dict) or not isinstance(child_result.get("status"), str):
                    raise ValueError("Group controller returned no structured status")
                atomic_json(checkpoint, child_result)
                _merge_child(ledger, child)
                ledger.put_record("formal_groups", str(index), {"group_index": index, "campaign": str(child),
                                  "status": child_result["status"], "ledger": Ledger(child).status()})
                save()
                if child_result["status"] == "BUDGET_EXHAUSTED":
                    status.update(status="BUDGET_EXHAUSTED", phase="finished", reason="Registered child group exhausted its frozen task budget")
                    break
                if child_result["status"] not in _TERMINAL_GROUPS:
                    status.update(status=child_result["status"], phase="blocked", reason="Resume this registered group; no replacement path or new seed was selected",
                                  blocked_group=child_result)
                    break
            # Always expose the latest completed records and deficits, even
            # when an incomplete group or resource limit stops the campaign.
            status.update(_progress(campaign, protocol, ledger))
            actual = _usage(campaign, ledger, protocol)
            if any(actual[key] > protocol["budget"][key] for key in _BUDGET):
                status.update(status="BUDGET_EXHAUSTED", phase="blocked", data_collection_complete=False,
                              reason="Observed spend exceeds the approved ceiling; controller refuses further groups")
            if status["status"] == "DATA_COLLECTION_COMPLETE" and not (status["quotas_met"] and status["training_ready"]):
                raise ValueError("Completion invariants no longer hold")
        except Exception as exc:
            status.update(status="INCOMPLETE", phase="blocked", data_collection_complete=False,
                          reason=type(exc).__name__ + ": " + str(exc))
            ledger.add_event("formal_controller_error", {"type": type(exc).__name__, "reason": str(exc)})
        save()
    return status
