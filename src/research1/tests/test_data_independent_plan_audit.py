"""Independent preimplementation audit gate: freeze, driver pause, seal validation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from unittest.mock import patch

from research1.data_pipeline import edit_trial as trial
from research1.data_pipeline import independent_plan_audit as audit
from research1.data_pipeline import plan_precision
from research1.data_pipeline.admission_artifacts import validate_positive_artifacts
from research1.data_pipeline.execution import tree_hash
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.util import atomic_json


class IndependentAuditTrialFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.modules = self.root / "modules"
        for module in ("agent_loop", "observation", "tools", "context_mgmt", "verification"):
            path = self.modules / module / "baseline.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x = 1\n")
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.contract = {
            "checks_id": "checks-fixture",
            "selected": [],
            "parent_hashes": {},
            "preprobed": ["completion_requirement_recheck"],
            "registry": {"packs": {"completion_requirement_recheck": {
                "goal_checks": ["goal"], "preserve_checks": ["keep"], "max_files": 1}}},
        }
        self.protocol = {
            "protocol_id": "offline-trial",
            "protocol_hash": "p" * 64,
            "execution_mode": "initialization",
            "trial_driver": trial.TRIAL_DRIVER,
            "grounded_edit_trial": {"arms": copy.deepcopy(trial.ARMS), "stop_after_qualified_edit": 1},
            "initialization": {"task_order": ["bloom"], "max_parents": 1},
            "modules_root": str(self.modules),
            "modules_hash": tree_hash(self.modules),
            "policies": {},
            "budget": {"regular": 2, "retries": 0, "total": 2},
            "stage_budget": {"proposal": 2},
            "task_sets": {
                "proposal": ["bloom"],
                "proposal_train": ["bloom"],
                "proposal_validation": [],
                "reward": ["reward"],
                "dev": ["dev"],
            },
            "source_parents": {
                "source_campaign": "/original/campaign",
                "source_protocol_hash": "s" * 64,
                "snapshot_sha256": "a" * 64,
            },
            "evidence_budget": {"max_total_chars": 96000, "max_step_chars": 16000, "max_steps": 12},
        }
        (self.campaign / "protocol.json").write_text(json.dumps(self.protocol))
        artifact = self.root / "original_result.json"
        artifact.write_text("original immutable score fixture\n")
        trajectory = self.root / "original_trajectory.json"
        trajectory.write_text("original immutable trajectory fixture\n")
        self.parent = {
            "status": "PASS",
            "reward": 0.0,
            "failure_layer": "model",
            "evaluation_id": "old-eval",
            "trial_result_path": str(artifact),
            "result_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "trajectory": str(trajectory),
            "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "evaluation_spec": {
                "role": "proposal",
                "task": "bloom",
                "modules_root": str(self.modules),
                "code_hash": tree_hash(self.modules),
            },
        }
        self.source = {
            "task": "bloom",
            "root_lineage_id": "original-root",
            "split": "train",
            "source_state_id": "original-state",
            "parent_result": self.parent,
        }
        self.trace = []
        self.actions = {"standard": "edit", "reasoned": "edit"}
        self.meta = {"provider": "fixed", "model": "fixed-audit-model", "temperature": 0,
            "raw_field": "content", "finish_reason": "stop"}
        self.load_protocol = patch("research1.data_pipeline.protocol.load_protocol", return_value=self.protocol)
        self.load_protocol.start()
        self.addCleanup(self.load_protocol.stop)
        self.freeze_fixture = patch(
            "research1.data_pipeline.fixture_context.freeze_probe_contract",
            side_effect=self._fake_fixture_context,
        )
        self.freeze_fixture.start()
        self.addCleanup(self.freeze_fixture.stop)
        self.probe_validate = patch(
            "research1.data_pipeline.predecision_probe_io.validate_probe_io_state",
            side_effect=self._probe_validate,
        )
        self.probe_validate.start()
        self.addCleanup(self.probe_validate.stop)

    def _probe_validate(self, state):
        evidence = state.get("evidence") or []
        refs = [row["id"] for row in evidence[:1]]
        return {
            "schema": "research1.predecision-harness-api-io.v1",
            "verified": True,
            "evidence_ids_added": refs,
            "artifacts": copy.deepcopy(state.get("predecision_probe_artifacts") or {}),
        }

    def _fake_fixture_context(self, path):
        raw = Path(path).read_bytes()
        return {
            "schema": "research1.original-prefrozen-fixture-source.v1",
            "path": str(Path(path).resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "limitations": ["fixture fixture"],
        }

    def enable_independent_policy(self):
        from research1.data_pipeline import modules_v3 as mod
        from research1.data_pipeline.predecision_probe_io import publish_parent_probe_io, validate_probe_io_state
        from research1.tests.test_data_predecision_probe_io import _probe as offline_probe_output

        self.probe_validate.stop()
        self._independent_gate_enabled = True
        self._publish_parent_probe_io = publish_parent_probe_io
        self._validate_probe_io_state = validate_probe_io_state
        self._build_contract = mod.build_contract
        self.probe_module_patch = patch.object(mod, "probe_module", side_effect=offline_probe_output)
        self.probe_module_patch.start()
        self.addCleanup(self.probe_module_patch.stop)

        self.fixture_source = self.campaign / "source_snapshot" / "src" / "research1" / "data_pipeline" / "modules_v3.py"
        self.fixture_source.parent.mkdir(parents=True, exist_ok=True)
        self.fixture_source.write_text("# frozen fixture source for independent audit tests\n", encoding="utf-8")
        fixture_sha = hashlib.sha256(self.fixture_source.read_bytes()).hexdigest()
        self.protocol.update(
            verification_registry_version="research1.module-contracts.v3",
            plan_precision_policy=plan_precision.POLICY_ID,
            independent_plan_audit_policy=audit.POLICY_ID,
            sft={"plan_policy_version": "proposal_single_plan_v6"},
            permitted_modules=["agent_loop", "observation", "tools", "context_mgmt", "verification"],
            diagnostic_source_selection={"tasks": ["bloom"], "frozen_before_candidates": True},
            policies={
                "factual_reviewer": {
                    "version": "proposal-factual-review.v8.probe_io",
                    "precision_policy": plan_precision.POLICY_ID,
                },
                "applicability_reviewer": {
                    "version": "proposal-plan-applicability.v3.fact_ids",
                    "fixture_source_ref": {"path": str(self.fixture_source.resolve()), "sha256": fixture_sha},
                },
                "independent_plan_audit": {
                    "version": audit.POLICY_ID,
                    "protocol_ref": {"path": str((self.campaign / "protocol.json").resolve())},
                },
            },
            grounded_edit_trial={
                "arms": copy.deepcopy(trial.ARMS),
                "stop_after_qualified_edit": 1,
                "publish_probe_io": True,
            },
        )
        (self.campaign / "protocol.json").write_text(json.dumps(self.protocol))
        adapter = patch("research1.data_pipeline.module_adapters.get_adapter")
        self.adapter_mock = adapter.start()
        self.addCleanup(adapter.stop)

        fixture = self

        from research1.data_pipeline import module_adapters

        class FakeAdapter:
            REGISTRY_VERSION = module_adapters.STRICT_VERSION
            __file__ = mod.__file__
            _digest = staticmethod(mod._digest)

            def frozen_registry(self):
                return mod.frozen_registry()

            def tree_hashes(self, root):
                return mod.tree_hashes(root)

            def code_hash(self, root):
                return mod.code_hash(root)

            def proposal_catalog(self, state, contract):
                return mod.proposal_catalog(state, contract)

            def implement(self, plan, state, directory):
                return fixture.implement(plan, state, directory)

            def verify(self, plan, state, attempt, contract):
                return fixture.verify(plan, state, attempt, contract)

            def build_contract(self, state):
                return mod.build_contract(state)

            def planning_interfaces(self, permitted_modules, *, include_scope=False):
                return mod.planning_interfaces(permitted_modules, include_scope=include_scope)

        self.adapter_mock.return_value = FakeAdapter()

    def facts(self, state, directory):
        self.trace.append(("facts", None))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fact_output.json").write_text(json.dumps({"raw": "offline no-card fixture"}))
        return None

    def make_plan(self, state, action):
        return {
            "action": action,
            "target_module": "agent_loop" if action == "edit" else None,
            "locations": ["agent_loop/baseline.py"] if action == "edit" else [],
            "evidence_refs": [state["evidence"][0]["id"]],
            "problem_hypothesis": "Offline fixture observed requirement",
            "intended_change": "Offline fixture bounded change" if action == "edit" else "Keep bounded observation",
            "preserve_behaviors": ["Retain transport"],
            "verification_plan": ["Frozen fixture checks"],
            "verification_kind": "completion_requirement_recheck" if action == "edit" else None,
        }

    def propose(self, state, directory):
        arm = state["teacher_arm_id"]
        self.trace.append(("teacher", arm))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "teacher.json").write_text(json.dumps({"raw": "offline plan fixture", "prompt_sha256": arm}))
        return [self.make_plan(state, self.actions[arm])]

    def review(self, plan, state):
        arm = state["teacher_arm_id"]
        review_dir = self.campaign / "reviews" / "edittrial" / arm
        review_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "PASS",
            "independent": True,
            "reviewer_id": "offline-fact-review",
            "reason": "Bounded observed fixture",
            "evidence_refs": plan["evidence_refs"],
        }
        raw_path = review_dir / "factual_review.json"
        raw_path.write_text(json.dumps({"raw": json.dumps(result)}))
        result["review_path"] = str(raw_path.resolve())
        catalog_path = review_dir / "factual_witness_catalog.json"
        catalog_path.write_text(json.dumps({"schema": "witness-catalog-fixture", "items": []}))
        result["witness_catalog_path"] = str(catalog_path.resolve())
        atomic_json(review_dir / "factual_review_result.json", result)
        self.trace.append(("review", arm))
        return result

    def review_selector_pass(self, plan, state):
        from research1.data_pipeline import selector_review as selector
        from research1.data_pipeline.policy import public_view, validate_plan
        from research1.data_pipeline.util import digest as pipeline_digest
        import os
        import time
        plan = validate_plan(plan, state)
        arm = state["teacher_arm_id"]
        review_dir = self.campaign / "reviews" / "edittrial" / arm
        review_dir.mkdir(parents=True, exist_ok=True)
        state["factual_review_policy"] = "proposal-factual-review.v8.probe_io"
        state["review_model"] = self.meta["model"]
        state["review_provider"] = self.meta["provider"]
        request, catalog = selector._request(state, plan)
        atomic_json(review_dir / "factual_review_input.json", request)
        atomic_json(review_dir / "factual_witness_catalog.json", catalog)
        input_hash = hashlib.sha256((review_dir / "factual_review_input.json").read_bytes()).hexdigest()
        catalog_hash = hashlib.sha256((review_dir / "factual_witness_catalog.json").read_bytes()).hexdigest()
        started = {"request_id": request["request_id"], "prompt_sha256": request["prompt_sha256"],
            "input_record_sha256": input_hash, "catalog_record_sha256": catalog_hash,
            "started_unix": time.time(), "automatic_resend": False}
        start_path = review_dir / "factual_review_started.json"
        with start_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(started, ensure_ascii=False, indent=2) + "\n")
        started_hash = hashlib.sha256(start_path.read_bytes()).hexdigest()
        api_rows = [row for row in catalog.get("items", []) if row.get("kind") == "local_api_observation"]
        witness_ids = [api_rows[0]["id"]] if api_rows else []
        evidence_refs = list(dict.fromkeys(row["evidence_ref"] for row in api_rows)) or list(plan["evidence_refs"])
        reply = {"status": "PASS",
            "reason": "The verified local API return supports this bounded factual review; future effects remain unproven.",
            "evidence_refs": evidence_refs, "witness_ids": witness_ids,
            "code_witnesses": [{"path": "agent_loop/baseline.py", "text": "x = 1"}]}
        if state.get("plan_precision_policy"):
            from research1.data_pipeline import plan_precision
            reply["claim_audit"] = [{"field_id": row["id"], "verdict": "supported",
                "reason": "Checked against supplied sources."} for row in plan_precision.build_plan_field_catalog(plan)["items"]]
        raw = json.dumps(reply)
        saved = {**request, "raw": raw, "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "metadata": self.meta, "metadata_sha256": pipeline_digest(self.meta),
            "review_type": "model_fact_review", "input_record_sha256": input_hash,
            "catalog_record_sha256": catalog_hash, "started_record_sha256": started_hash}
        atomic_json(review_dir / "factual_review.json", saved)
        result = selector._result(saved, state, plan, request, catalog, review_dir)
        atomic_json(review_dir / "factual_review_result.json", result)
        self.trace.append(("review", arm))
        return result

    def applicability(self, state, plan, review, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        binding = {"status": "PASS", "binding_id": "offline-binding-" + state["teacher_arm_id"]}
        input_path = directory / "applicability_input.json"
        started_path = directory / "applicability_started.json"
        result_path = directory / "applicability_result.json"
        for path in (input_path, started_path, result_path):
            atomic_json(path, {"fixture": path.name})
        output_path = directory / "applicability_output.json"
        output_path.write_text(json.dumps({"raw": "{}"}))
        binding.update(
            applicability_input_path=str(input_path.resolve()),
            applicability_started_path=str(started_path.resolve()),
            applicability_output_path=str(output_path.resolve()),
        )
        self.trace.append(("binding", state["teacher_arm_id"]))
        return binding

    def implement(self, plan, state, directory):
        arm = state["teacher_arm_id"]
        self.trace.append(("implement", arm))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "editor_output.json").write_text(json.dumps({"raw": "offline implementation"}))
        child = directory / "child"
        import shutil
        shutil.copytree(self.modules, child)
        (child / "agent_loop/baseline.py").write_text("x = 2\n")
        patch_path = directory / "edit.diff"
        patch_path.write_text("offline saved exact fixture patch\n")
        return {
            "status": "applied",
            "modules_root": str(child),
            "code_hash": tree_hash(child),
            "parent_code_hash": state["code_hash"],
            "patch_path": str(patch_path),
            "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
        }

    def verify(self, plan, state, attempt, contract):
        self.trace.append(("verify", state["teacher_arm_id"]))
        if getattr(self, '_independent_gate_enabled', False):
            return {'status': 'pending', 'covers_plan': False, 'executable': False,
                'checks_id': contract['checks_id'], 'mechanism_id': plan['verification_kind'],
                'reason': 'Orchestration-only fixture has no real child behavior proof.'}
        return {
            "status": "PASS",
            "covers_plan": True,
            "executable": True,
            "checks_id": contract["checks_id"],
            "mechanism_id": plan["verification_kind"],
        }

    def evaluate(self, campaign, protocol, ledger, code, task, role, scope, repeat):
        self.trace.append(("evaluate", scope))
        from research1.data_pipeline.util import digest
        spec = {"role": role, "task": task, "scope": scope, "code_hash": tree_hash(code)}
        key = digest(spec)
        ledger.reserve_job(key, spec)
        ledger.start_job(key, "offline-test")
        result = {
            "status": "PASS",
            "reward": 0.0,
            "failure_layer": "model",
            "evaluation_id": key,
            "evaluation_spec": {"role": role, "task": task, "modules_root": str(code), "code_hash": tree_hash(code)},
        }
        ledger.complete_job(key, result)
        return result

    def _probe_io_stub(self, base, contract, directory, **kwargs):
        from research1.data_pipeline.util import digest
        base = copy.deepcopy(base)
        contract = copy.deepcopy(contract)
        directory.mkdir(parents=True, exist_ok=True)
        receipt = directory / "receipt.json"
        raw = directory / "raw.json"
        receipt.write_text("{}")
        raw.write_text("{}")
        evidence = base.get("evidence") or []
        contract["evidence_hashes"] = {row["id"]: digest(row) for row in evidence}
        contract["checks_id"] = "checks_" + digest({key: value for key, value in contract.items() if key != "checks_id"})[:24]
        base["predecision_probe_io_policy"] = "research1.predecision-harness-api-io.v1"
        base["predecision_probe_artifacts"] = {
            "raw_contract": {"path": str(raw.resolve()), "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()},
            "receipt": {"path": str(receipt.resolve()), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()},
        }
        base["contract"] = contract
        return base, contract

    def run_trial(self, **overrides):
        kwargs = {
            "evaluator": self.evaluate,
            "proposer": self.propose,
            "implementer": self.implement,
            "verifier": self.verify,
            "reviewer": self.review,
            "fact_collector": self.facts,
            "interface_builder": lambda state: None,
            "applicability_reviewer": self.applicability,
            "contract_builder": lambda state: copy.deepcopy(self.contract),
            "source_validator": lambda campaign, protocol: [copy.deepcopy(self.source)],
            "evidence_selector": lambda parent: [{
                "id": "trajectory",
                "text": "Executed fixture returned the bounded expected body",
                "partition": "proposal",
            }],
            "probe_io_publisher": self._probe_io_stub,
            "probe_io_validator": lambda state: {"verified": True, "schema": "fixture"},
        }
        if getattr(self, "_independent_gate_enabled", False):
            kwargs["contract_builder"] = self._build_contract
            kwargs["probe_io_publisher"] = self._publish_parent_probe_io
            kwargs["probe_io_validator"] = self._validate_probe_io_state
        kwargs.update(overrides)
        return trial.run_grounded_edit_trial(self.campaign, **kwargs)

    def write_external_audit(self, arm_id, request, *, decision="APPROVE"):
        audit_dir = self.campaign / "independent_audits" / arm_id
        audit_dir.mkdir(parents=True, exist_ok=True)
        provenance_path = audit_dir / "cursor_task_done.json"
        task_id = "cursor-audit-" + arm_id
        atomic_json(provenance_path, {"status": "DONE", "task_id": task_id,
            "model": "fixed-audit-model", "exit_code": 0})
        report_path = audit_dir / audit.REPORT_NAME
        report = {
            "schema": audit.REPORT_SCHEMA,
            "request_sha256": request["request_sha256"],
            "decision": decision,
            "auditor": {
                "kind": "cursor_cli_blinded_plan_audit",
                "task_id": task_id,
                "model": "fixed-audit-model",
            },
            "findings": "Blinded independent review of original plan and public view only.",
            "reason": "Independent curator " + decision.lower() + " for fixture plan.",
        }
        atomic_json(report_path, report)
        receipt_path = audit_dir / audit.RECEIPT_NAME
        receipt = {
            "schema": audit.RECEIPT_SCHEMA,
            "request_sha256": request["request_sha256"],
            "report_path": str(report_path.resolve()),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "auditor_task_provenance": {
                "path": str(provenance_path.resolve()),
                "sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            },
            "auditor_task_id": task_id,
            "auditor_model": "fixed-audit-model",
            "coordinator": {"identity": "root-ops-fixture"},
            "decision": decision,
            "reason": "Coordinator recorded matching " + decision.lower() + ".",
        }
        atomic_json(receipt_path, receipt)


class IndependentAuditLegacyPathTests(IndependentAuditTrialFixture):
    def test_old_no_policy_path_unchanged(self):
        result = self.run_trial()
        self.assertEqual(result["status"], "GROUNDED_EDIT_TRIAL_COMPLETE")
        self.assertFalse((self.campaign / "independent_audits").exists())


class IndependentPlanAuditGateTests(IndependentAuditTrialFixture):
    def setUp(self):
        IndependentAuditTrialFixture.setUp(self)
        self.enable_independent_policy()
        self.binding_patch = patch("research1.data_pipeline.applicability.validate_binding", return_value=True)
        self.binding_patch.start()
        self.addCleanup(self.binding_patch.stop)
        self.binding_v3 = patch("research1.data_pipeline.applicability_v3.validate_binding", return_value=True)
        self.binding_v3.start()
        self.addCleanup(self.binding_v3.stop)

    def review(self, plan, state):
        return self.review_selector_pass(plan, state)

    def test_missing_audit_pauses_with_zero_editor_calls(self):
        result = self.run_trial()
        self.assertEqual(result["status"], "AWAITING_INDEPENDENT_PLAN_AUDIT")
        self.assertFalse(any(kind == "implement" for kind, _ in self.trace))
        self.assertTrue((self.campaign / "independent_audits" / "standard" / audit.REQUEST_NAME).is_file())

    def test_one_approved_one_missing_still_pauses(self):
        self.run_trial()
        request = json.loads(
            (self.campaign / "independent_audits" / "standard" / audit.REQUEST_NAME).read_text(encoding="utf-8"))
        self.write_external_audit("standard", request, decision="APPROVE")
        second = self.run_trial()
        self.assertEqual(second["status"], "AWAITING_INDEPENDENT_PLAN_AUDIT")
        self.assertFalse(any(kind == "implement" for kind, _ in self.trace))

    def test_matching_terminal_audits_resume_without_reteacher(self):
        self.run_trial()
        before = list(self.trace)
        for arm in ("standard", "reasoned"):
            request = json.loads(
                (self.campaign / "independent_audits" / arm / audit.REQUEST_NAME).read_text(encoding="utf-8"))
            self.write_external_audit(arm, request, decision="APPROVE")
        result = self.run_trial()
        self.assertEqual(result["status"], "GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT")
        self.assertEqual(self.trace[: len(before)], before)
        self.assertEqual([arm for kind, arm in self.trace if kind == "implement"], ["standard", "reasoned"])
        for kind in ('teacher', 'review', 'binding', 'facts'):
            self.assertEqual(sum(row[0] == kind for row in before), sum(row[0] == kind for row in self.trace))

    def test_independent_reject_leaves_ds_results_and_blocks_implementation(self):
        self.run_trial()
        for arm, decision in (("standard", "REJECT"), ("reasoned", "APPROVE")):
            request = json.loads(
                (self.campaign / "independent_audits" / arm / audit.REQUEST_NAME).read_text(encoding="utf-8"))
            self.write_external_audit(arm, request, decision=decision)
        result = self.run_trial()
        self.assertEqual(result["status"], "GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT")
        rows = Ledger(self.campaign).records("samples")
        rejected = next(row for row in rows if row["state"]["teacher_arm_id"] == "standard")
        approved = next(row for row in rows if row["state"]["teacher_arm_id"] == "reasoned")
        self.assertEqual(rejected["qualification"]["status"], "rejected")
        self.assertEqual(rejected["qualification"]["factual_review"]["status"], "PASS")
        # This orchestration fixture deliberately has no valid saved editor patch;
        # independent approval permits an attempt, not SFT admission.
        self.assertEqual(approved['implementation']['status'], 'applied')
        self.assertNotEqual(approved['qualification']['status'], 'verified')
        self.assertEqual([arm for kind, arm in self.trace if kind == "implement"], ["reasoned"])

    def test_bare_approve_dict_is_insufficient(self):
        self.run_trial()
        audit_dir = self.campaign / "independent_audits" / "standard"
        atomic_json(audit_dir / audit.REPORT_NAME, {"decision": "APPROVE"})
        atomic_json(audit_dir / audit.RECEIPT_NAME, {"decision": "APPROVE"})
        result = self.run_trial()
        self.assertNotEqual(result["status"], "GROUNDED_EDIT_TRIAL_COMPLETE")
        self.assertFalse(any(kind == 'implement' for kind, _ in self.trace))

    def test_keep_does_not_require_applicability_bindings(self):
        self.actions = {"standard": "keep", "reasoned": "keep"}
        self.run_trial()
        for arm in ("standard", "reasoned"):
            request = json.loads(
                (self.campaign / "independent_audits" / arm / audit.REQUEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(request.get("applicability_bindings"), {})
            self.write_external_audit(arm, request, decision="APPROVE")
        result = self.run_trial()
        # The minimal orchestration teacher/parent files cannot pass the real
        # positive seal. The audit/app-not-applicable path must still be reached.
        self.assertEqual(result['status'], 'INCOMPLETE')
        self.assertIn('positive-admission artifacts', result.get('reason', ''))
        self.assertFalse(any(kind == 'binding' for kind, _ in self.trace))
        self.assertFalse(any(kind == 'implement' for kind, _ in self.trace))

    def test_validate_independent_audit_requires_approve(self):
        record = {
            "state": {"independent_plan_audit_policy": audit.POLICY_ID, "plan_policy_version": "proposal_single_plan_v6",
                "contract": {"binding_mode": "independent_plan_to_prefrozen_checks.v1"}},
            "qualification": {"independent_plan_audit": {"policy": audit.POLICY_ID, "decision": "REJECT"}},
        }
        with self.assertRaises(ValueError):
            audit.validate_independent_audit(record)

    def test_committed_reject_cannot_be_replaced_with_approve_on_resume(self):
        self.run_trial()
        for arm in ('standard', 'reasoned'):
            request = json.loads((self.campaign / 'independent_audits' / arm / audit.REQUEST_NAME).read_bytes())
            self.write_external_audit(arm, request, decision='REJECT')
        self.assertEqual(self.run_trial()['status'], 'GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT')
        before = copy.deepcopy(Ledger(self.campaign).records('samples'))
        request = json.loads((self.campaign / 'independent_audits/standard' / audit.REQUEST_NAME).read_bytes())
        self.write_external_audit('standard', request, decision='APPROVE')
        result = self.run_trial()
        self.assertNotEqual(result['status'], 'GROUNDED_EDIT_TRIAL_COMPLETE')
        self.assertFalse(any(kind == 'implement' for kind, _ in self.trace))
        after = Ledger(self.campaign).records('samples')
        strip_revocation = lambda row: {key: value for key, value in row.items()
            if key not in {'revoked', 'revocation_reason'}}
        self.assertEqual([strip_revocation(row) for row in after], [strip_revocation(row) for row in before])
        self.assertTrue(all(row.get('revoked') for row in after))

    def test_pending_or_partial_external_file_bytes_are_pinned_before_approval(self):
        self.run_trial()
        directory = self.campaign / 'independent_audits/standard'
        request = json.loads((directory / audit.REQUEST_NAME).read_bytes())
        self.write_external_audit('standard', request, decision='PENDING')
        (directory / audit.RECEIPT_NAME).unlink()
        self.assertEqual(self.run_trial()['status'], 'AWAITING_INDEPENDENT_PLAN_AUDIT')
        self.write_external_audit('standard', request, decision='APPROVE')
        result = self.run_trial()
        self.assertNotEqual(result['status'], 'GROUNDED_EDIT_TRIAL_COMPLETE')
        self.assertFalse(any(kind == 'implement' for kind, _ in self.trace))

    def test_already_sealed_record_survives_resume_without_resealing(self):
        # Storage/control-flow sentinel only; real artifact acceptance is tested
        # separately with the physical positive fixture below.
        self.run_trial()
        for arm in ('standard', 'reasoned'):
            request = json.loads((self.campaign / 'independent_audits' / arm / audit.REQUEST_NAME).read_bytes())
            self.write_external_audit(arm, request, decision='APPROVE')
        self.run_trial()
        ledger = Ledger(self.campaign)
        row = next(item for item in ledger.records('samples') if item['state']['teacher_arm_id'] == 'standard')
        row['qualification']['status'] = 'verified'
        row['admission_artifacts'] = {'preserve_existing_seal': 'storage sentinel'}
        ledger.put_record('samples', row['sample_id'], row)
        before = copy.deepcopy(ledger.records('samples'))
        counts = list(self.trace)
        with patch('research1.data_pipeline.admission_artifacts.seal_positive_artifacts',
                side_effect=AssertionError('Resume must not reseal stored terminal records')):
            self.run_trial()
        self.assertEqual(Ledger(self.campaign).records('samples'), before)
        self.assertEqual(self.trace, counts)

    def test_noneligible_arm_needs_no_audit_and_only_approved_arm_is_attempted(self):
        def eligibility(state, plan, review, directory):
            if state['teacher_arm_id'] == 'reasoned':
                return {'status': 'pending', 'reason': 'Original unsupported fixture mechanism'}
            return self.applicability(state, plan, review, directory)
        result = self.run_trial(applicability_reviewer=eligibility)
        self.assertEqual(result['status'], 'AWAITING_INDEPENDENT_PLAN_AUDIT')
        self.assertFalse((self.campaign / 'independent_audits/reasoned').exists())
        request = json.loads((self.campaign / 'independent_audits/standard' / audit.REQUEST_NAME).read_bytes())
        self.write_external_audit('standard', request)
        result = self.run_trial(applicability_reviewer=eligibility)
        self.assertEqual(result['status'], 'GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT')
        self.assertEqual([arm for kind, arm in self.trace if kind == 'implement'], ['standard'])

    def test_all_noneligible_finish_without_audit_or_implementation(self):
        result = self.run_trial(applicability_reviewer=lambda *args: {'status': 'pending', 'reason': 'Unsupported original scope'})
        self.assertEqual(result['status'], 'GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT')
        self.assertFalse((self.campaign / 'independent_audits').exists())
        self.assertFalse(any(kind == 'implement' for kind, _ in self.trace))

    def test_terminal_audit_cannot_survive_invalidated_original_prerequisites(self):
        self.run_trial()
        for arm in ('standard', 'reasoned'):
            request = json.loads((self.campaign / 'independent_audits' / arm / audit.REQUEST_NAME).read_bytes())
            self.write_external_audit(arm, request, decision='REJECT')
        self.run_trial()
        trace = list(self.trace)
        with patch.object(trial, '_review_valid', return_value=False):
            result = self.run_trial()
        self.assertEqual(result['status'], 'BLOCKED_ARTIFACT_CHANGED')
        self.assertEqual(self.trace, trace)


class IndependentAuditValidatorTests(IndependentAuditTrialFixture):
    setUp = IndependentPlanAuditGateTests.setUp
    review = IndependentPlanAuditGateTests.review
    def _approved_bundle(self):
        self.run_trial()
        arm = "standard"
        request = json.loads(
            (self.campaign / "independent_audits" / arm / audit.REQUEST_NAME).read_text(encoding="utf-8"))
        self.write_external_audit(arm, request, decision="APPROVE")
        row = next(item for item in Ledger(self.campaign).records("samples")
            if item['state']['teacher_arm_id'] == arm)
        request['_request_path'] = str((self.campaign / 'independent_audits' / arm / audit.REQUEST_NAME).resolve())
        request['request_record_sha256'] = hashlib.sha256(Path(request['_request_path']).read_bytes()).hexdigest()
        proof = audit.read_independent_audit(request,
            auditor_report_ref={"path": str((self.campaign / "independent_audits" / arm / audit.REPORT_NAME).resolve())},
            coordinator_receipt_ref={"path": str((self.campaign / "independent_audits" / arm / audit.RECEIPT_NAME).resolve())},
            record=row)
        return request, proof, row

    def test_valid_bundle_passes_validate_frozen_request(self):
        request, proof, row = self._approved_bundle()
        audit.validate_frozen_request(request, record=row, request_path=request["_request_path"])
        self.assertEqual(proof["decision"], "APPROVE")

    def test_missing_factual_input_blocks_read(self):
        request, proof, row = self._approved_bundle()
        path = Path(request["review_bindings"]["factual_review_input"]["path"])
        path.unlink()
        with self.assertRaises(ValueError):
            audit.read_independent_audit(request,
                auditor_report_ref={"path": proof["report_path"]},
                coordinator_receipt_ref={"path": proof["coordinator_receipt_path"]},
                record=row)

    def test_borrowed_provenance_exit_code_blocks(self):
        self.run_trial()
        request = json.loads(
            (self.campaign / "independent_audits" / "standard" / audit.REQUEST_NAME).read_text(encoding="utf-8"))
        self.write_external_audit("standard", request, decision="APPROVE")
        prov = self.campaign / "independent_audits" / "standard" / "cursor_task_done.json"
        value = json.loads(prov.read_text(encoding="utf-8"))
        value["exit_code"] = 1
        atomic_json(prov, value)
        result = self.run_trial()
        self.assertNotEqual(result['status'], 'GROUNDED_EDIT_TRIAL_COMPLETE')
        self.assertFalse(any(kind == 'implement' for kind, _ in self.trace))

    def test_request_record_byte_mutation_blocks_validate(self):
        request, proof, row = self._approved_bundle()
        row["qualification"]["independent_plan_audit"] = proof
        path = Path(proof["request_path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            audit.validate_independent_audit(row)

    def test_cross_candidate_request_reuse_blocks(self):
        request, proof, row = self._approved_bundle()
        other = copy.deepcopy(row)
        other["plan"] = copy.deepcopy(row["plan"])
        other["plan"]["problem_hypothesis"] = "Different candidate plan text"
        with self.assertRaises(ValueError):
            audit.read_independent_audit(request,
                auditor_report_ref={"path": proof["report_path"]},
                coordinator_receipt_ref={"path": proof["coordinator_receipt_path"]},
                record=other)


class IndependentAuditPhysicalSealTests(unittest.TestCase):
    """Use existing real publisher/reviewer/editor/result fixtures, not gate mocks."""
    def setUp(self):
        import shutil
        from research1.data_pipeline import modules_v3 as mod, selector_review as selector
        from research1.data_pipeline import applicability_v3 as app3, fixture_context, predecision_probe_io as publisher
        from research1.data_pipeline.policy import public_view, review_plan, digest
        from research1.tests import test_data_admission_artifacts as positive
        from research1.tests import test_data_plan_contract_binding as binding_fixture
        from research1.tests import test_data_probe_evidence_admission as probes
        original = binding_fixture.PlanContractBindingTests
        owner = self

        def new_case(name):
            case = original(name)
            old_setup = case.setUp

            def setup():
                old_setup()
                case.state.update(verification_registry_version=mod.REGISTRY_VERSION,
                    factual_review_policy=selector.PROBE_POLICY_ID,
                    plan_precision_policy=plan_precision.POLICY_ID,
                    applicability_policy=app3.POLICY_ID,
                    independent_plan_audit_policy=audit.POLICY_ID,
                    teacher_arm_id='standard')
                relative = fixture_context.STRICT_RELATIVE_SOURCE
                source = case.directory / 'source_snapshot' / relative
                source.parent.mkdir(parents=True)
                shutil.copyfile(mod.__file__, source)
                protocol = {'verification_registry_version': mod.REGISTRY_VERSION,
                    'source_sha256': {relative: hashlib.sha256(source.read_bytes()).hexdigest()},
                    'registry_sha256': digest(mod.frozen_registry()),
                    'independent_plan_audit_policy': audit.POLICY_ID}
                protocol['protocol_hash'] = digest(protocol)
                atomic_json(case.directory / 'protocol.json', protocol)
                case.state['decision_protocol_hash'] = protocol['protocol_hash']
                case.state['applicability_fixture_source'] = fixture_context.freeze_probe_contract(source)
                owner.protocol = protocol

            def freeze():
                with patch.object(mod, 'probe_module', side_effect=probes.probe) as probe:
                    contract = mod.build_contract(case.state)
                case.state['contract'] = contract
                state, contract = publisher.publish_parent_probe_io(case.state, contract,
                    case.directory / 'published-probe', protocol_hash=owner.protocol['protocol_hash'])
                case.state.clear(); case.state.update(state)
                return contract, probe

            def facts():
                catalog = selector.build_witness_catalog(public_view(case.state),
                    review_policy=selector.PROBE_POLICY_ID, state=case.state)
                witness = next(row for row in catalog['items'] if row['kind'] == 'local_api_observation')
                owner.witness = witness
                reply = {'status': 'PASS', 'reason': 'Bounded original evidence; future utility unproven.',
                    'evidence_refs': [witness['evidence_ref']], 'witness_ids': [witness['id']],
                    'code_witnesses': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}],
                    'claim_audit': [{'field_id': row['id'], 'verdict': 'supported', 'reason': 'Checked fixture source.'}
                        for row in plan_precision.build_plan_field_catalog(case.plan)['items']]}
                path = case.directory / 'reviews' / 'edittrial' / 'standard'
                with patch('research1.distill._tongji_chat', return_value=(json.dumps(reply), case.meta)):
                    result = review_plan(case.state, case.plan, path)
                owner.assertEqual(result['status'], 'PASS', result)
                case.state['plan_factual_review'] = result
                return result

            def binding(reply=None, directory=None):
                pack = case.state['contract']['registry']['packs'][case.plan['verification_kind']]
                value = {'status': 'PASS', 'reason': 'Original data path is reachable; frozen checks cover the plan.',
                    'mechanism_id': case.plan['verification_kind'], 'witness_ids': [owner.witness['id']],
                    'code_refs': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}],
                    'covered_checks': pack['goal_checks'] + pack['preserve_checks']}
                with patch('research1.distill._tongji_chat', return_value=(json.dumps(value), case.meta)) as chat:
                    result = app3.review_applicability(case.state, case.plan, case.state['plan_factual_review'],
                        directory or case.directory / 'applicability' / 'standard')
                owner.assertEqual(result['status'], 'PASS', result)
                return result, chat

            case.setUp = setup; case.freeze = freeze; case.facts = facts; case.binding = binding
            return case

        self.positive = positive
        self.case = positive.AdmissionArtifactTests('test_real_saved_fixture_passes_without_model_probe_or_task_execution')
        # Existing fixture constructor seals immediately. Defer that one construction
        # call until external files exist; actual seal AND export checks below are real.
        with patch.object(binding_fixture, 'PlanContractBindingTests', side_effect=new_case), \
                patch.object(binding_fixture, 'mod', mod), patch.object(positive, 'modules', mod), \
                patch.object(positive.artifacts, 'seal_positive_artifacts', return_value={'construction_deferred': True}):
            self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.record = self.case.record
        self.record.pop('admission_artifacts')
        self.root = self.case.root
        state = self.record['state']
        self.record['qualification']['applicability'] = state['plan_contract_binding']
        atomic_json(self.root / 'states' / state['state_id'] / 'state.json', state)
        atomic_json(self.root / 'edittrial_preteacher_contract.json', state['contract'])
        self.directory = self.root / 'independent_audits' / 'standard'
        self.request = audit.freeze_audit_request(state, self.record['plan'], state['plan_factual_review'],
            state['plan_contract_binding'], self.directory,
            protocol_ref={'path': str((self.root / 'protocol.json').resolve())}, campaign=self.root)
        IndependentAuditTrialFixture.write_external_audit(self, 'standard', self.request)
        self.proof = audit.read_independent_audit(self.request,
            auditor_report_ref={'path': str((self.directory / audit.REPORT_NAME).resolve())},
            coordinator_receipt_ref={'path': str((self.directory / audit.RECEIPT_NAME).resolve())}, record=self.record)
        self.record['qualification']['independent_plan_audit'] = self.proof
        self.record['admission_artifacts'] = positive.artifacts.seal_positive_artifacts(self.record)

    @property
    def campaign(self):
        return self.root

    def test_real_approved_seal_and_export_accept_unchanged_physical_bundle(self):
        with patch('research1.distill._tongji_chat') as chat:
            self.assertTrue(self.positive.artifacts.validate_positive_artifacts(self.record)[0])
            self.assertTrue(self.positive.sft._qualification(self.record)[0])
            self.assertTrue(audit.validate_independent_audit(self.record)['verified'])
        chat.assert_not_called()

    def test_every_bound_file_byte_mutation_invalidates_existing_seal_and_export(self):
        paths = {Path(self.proof[key]) for key in ('request_path', 'report_path',
            'coordinator_receipt_path', 'auditor_task_provenance_path')}
        for group in ('source_bindings', 'review_bindings', 'applicability_bindings', 'probe_io_bindings'):
            paths.update(Path(meta['path']) for meta in self.request[group].values() if meta)
        for path in sorted(paths):
            with self.subTest(path=str(path)):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b' ')
                    self.assertFalse(self.positive.artifacts.validate_positive_artifacts(self.record)[0])
                    self.assertFalse(self.positive.sft._qualification(self.record)[0])
                finally:
                    path.write_bytes(original)
        self.assertTrue(self.positive.artifacts.validate_positive_artifacts(self.record)[0])

    def test_approval_cannot_be_reused_for_other_plan_state_or_checks(self):
        for mutation in ('plan', 'state', 'checks'):
            value = copy.deepcopy(self.record)
            if mutation == 'plan': value['plan']['problem_hypothesis'] += ' Another plan.'
            elif mutation == 'state': value['state']['state_id'] = 'another-state'
            else: value['state']['contract']['checks_id'] = 'another-checks'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_independent_audit(value)

    def test_true_provenance_identity_and_integer_success_are_required(self):
        path = Path(self.proof['auditor_task_provenance_path'])
        receipt_path = Path(self.proof['coordinator_receipt_path'])
        raw, receipt_raw = path.read_bytes(), receipt_path.read_bytes()
        for key, value in (('task_id', 'borrowed-task'), ('model', 'another-model'),
                           ('exit_code', 1), ('exit_code', False), ('exit_code', None)):
            with self.subTest(key=key, value=value):
                try:
                    document = json.loads(raw); document[key] = value; atomic_json(path, document)
                    receipt = json.loads(receipt_raw)
                    receipt['auditor_task_provenance']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    atomic_json(receipt_path, receipt)
                    with self.assertRaises(ValueError):
                        audit.read_independent_audit(self.request,
                            auditor_report_ref={'path': self.proof['report_path']},
                            coordinator_receipt_ref={'path': str(receipt_path)}, record=self.record)
                finally:
                    path.write_bytes(raw); receipt_path.write_bytes(receipt_raw)

    def test_missing_factual_input_and_failed_behavior_cannot_be_overridden(self):
        path = Path(self.request['review_bindings']['factual_review_input']['path'])
        original = path.read_bytes()
        try:
            path.unlink()
            self.assertFalse(self.positive.artifacts.validate_positive_artifacts(self.record)[0])
        finally:
            path.write_bytes(original)
        value = copy.deepcopy(self.record)
        value['qualification']['behavior']['status'] = 'FAIL'
        self.assertFalse(self.positive.sft._qualification(value)[0])
        blind = json.dumps(self.request['blind_reading_input'])
        self.assertNotIn(self.record['qualification']['factual_review']['reason'], blind)
        self.assertNotIn(self.record['qualification']['applicability']['reason'], blind)


if __name__ == "__main__":
    unittest.main()
