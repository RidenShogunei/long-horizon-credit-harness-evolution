"""Prospective registry opt-in, immutable source refs and unchanged budgets."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from research1.data_pipeline import modules
from research1.data_pipeline.cli import add_parser, command
from research1.data_pipeline.protocol import (
    prepare, load_protocol, _validate_registry_options, V2_REGISTRY, V3_REGISTRY,
)
from research1.data_pipeline.policy import public_view
from research1.data_pipeline.ledger import Ledger
from research1.tests import test_data_parent_replan as replan_fixtures
from research1.tests import test_data_edit_trial as trial_fixtures


class RegistryOptionTests(unittest.TestCase):
    def test_default_and_explicit_legacy_version_preserve_legacy_selection(self):
        self.assertEqual(_validate_registry_options(None, None), V2_REGISTRY)
        self.assertEqual(_validate_registry_options(V2_REGISTRY, None), V2_REGISTRY)

    def test_v3_requires_fresh_fixed_source_without_teacher_or_quote_replay(self):
        diagnostic = {"tasks": ["bloom"]}
        for kwargs in ({}, {"diagnostic_selection": None}, {"teacher_replay_from": Path("old")},
                       {"fact_replay_from": Path("quotes")}, {"fact_replay_state": Path("old-state")},
                       {"teacher_max_tokens": 8000}):
            options = {"version": V3_REGISTRY, "diagnostic_selection": diagnostic, "teacher_max_tokens": 16000}
            options.update(kwargs)
            if not kwargs:
                self.assertEqual(_validate_registry_options(**options), V3_REGISTRY)
            else:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    _validate_registry_options(**options)

    def test_fresh_v3_can_freeze_selector_review_and_fixture_visibility(self):
        kwargs = {"diagnostic_selection": {"tasks": ["bloom"]}, "teacher_max_tokens": 16000,
            "admission_review_policy": "proposal-factual-review.v7.selectors", "applicability_fixtures": True}
        self.assertEqual(_validate_registry_options(V3_REGISTRY, **kwargs), V3_REGISTRY)
        with self.assertRaisesRegex(ValueError, "Reviewer options"):
            _validate_registry_options(None, **kwargs)

    def test_probe_publication_is_explicit_and_limited_to_fresh_v3(self):
        kwargs = {"diagnostic_selection": {"tasks": ["bloom"]}, "teacher_max_tokens": 16000,
                  "publish_probe_io": True}
        self.assertEqual(_validate_registry_options(V3_REGISTRY, **kwargs), V3_REGISTRY)
        for version in (None, V2_REGISTRY):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "publication requires"):
                _validate_registry_options(version, **kwargs)
        with self.assertRaises(ValueError):
            _validate_registry_options(V3_REGISTRY, **dict(kwargs, teacher_replay_from=Path("old")))
        with self.assertRaises(ValueError):
            _validate_registry_options(V3_REGISTRY, **dict(kwargs, fact_replay_from=Path("old-quotes")))

    def test_v8_probe_factual_review_requires_the_new_published_source(self):
        kwargs={"diagnostic_selection":{"tasks":["bloom"]},"teacher_max_tokens":16000,
            "admission_review_policy":"proposal-factual-review.v8.probe_io"}
        self.assertEqual(_validate_registry_options(V3_REGISTRY,**dict(kwargs,publish_probe_io=True)),V3_REGISTRY)
        with self.assertRaisesRegex(ValueError,"v8 requires published"):
            _validate_registry_options(V3_REGISTRY,**kwargs)
        with self.assertRaisesRegex(ValueError,"v8 requires published"):
            _validate_registry_options(V2_REGISTRY,**dict(kwargs,teacher_replay_from=Path("old")))

    def test_fresh_app3_requires_published_v3_factual8_and_original_fixture_ref(self):
        options={'diagnostic_selection':{'tasks':['bloom']},'teacher_max_tokens':16000,
            'admission_review_policy':'proposal-factual-review.v8.probe_io',
            'applicability_review_policy':'proposal-plan-applicability.v3.fact_ids',
            'applicability_fixtures':True,'publish_probe_io':True}
        self.assertEqual(_validate_registry_options(V3_REGISTRY,**options),V3_REGISTRY)
        self.assertEqual(_validate_registry_options(V3_REGISTRY,**dict(options,
            plan_precision_policy='proposal-plan-precision.v1.field_audit')),V3_REGISTRY)
        for changed in ({'publish_probe_io':False},{'applicability_fixtures':False},
                {'admission_review_policy':'proposal-factual-review.v7.selectors'},
                {'replay_factual_from_source':True}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                _validate_registry_options(V3_REGISTRY,**dict(options,**changed))
        with self.assertRaises(ValueError):
            _validate_registry_options(V3_REGISTRY,**dict(options,
                applicability_review_policy=None,plan_precision_policy='proposal-plan-precision.v1.field_audit'))

    def test_cli_exposes_explicit_factual_replay_and_applicability3(self):
        parser=argparse.ArgumentParser();add_parser(parser.add_subparsers())
        args=parser.parse_args(['data','prepare','--campaign','offline-new','--replay-factual-from-source',
            '--applicability-review-policy','proposal-plan-applicability.v3.fact_ids',
            '--plan-precision-policy','proposal-plan-precision.v1.field_audit'])
        fake={'protocol_hash':'fixture','task_sets':{'proposal':[]},'budget':{'regular':88}}
        with patch('research1.data_pipeline.cli.prepare',return_value=fake) as call,redirect_stdout(io.StringIO()):
            self.assertEqual(command(args),0)
        self.assertTrue(call.call_args.kwargs['replay_factual_from_source'])
        self.assertEqual(call.call_args.kwargs['applicability_review_policy'],'proposal-plan-applicability.v3.fact_ids')
        self.assertEqual(call.call_args.kwargs['plan_precision_policy'],'proposal-plan-precision.v1.field_audit')

    def test_cli_passes_explicit_registry_selection_to_prepare(self):
        parser = argparse.ArgumentParser()
        add_parser(parser.add_subparsers())
        args = parser.parse_args(["data", "prepare", "--campaign", "offline-new", "--mode", "initialization",
            "--verification-registry-version", V3_REGISTRY, "--teacher-max-tokens", "16000",
            "--admission-review-policy", "proposal-factual-review.v8.probe_io", "--applicability-fixtures",
            "--publish-probe-io"])
        fake = {"protocol_hash": "fixture", "task_sets": {"proposal": []}, "budget": {"regular": 88}}
        with patch("research1.data_pipeline.cli.prepare", return_value=fake) as call, redirect_stdout(io.StringIO()):
            self.assertEqual(command(args), 0)
        self.assertEqual(call.call_args.kwargs["verification_registry_version"], V3_REGISTRY)
        self.assertEqual(call.call_args.kwargs["teacher_max_tokens"], 16000)
        self.assertTrue(call.call_args.kwargs["applicability_fixtures"])
        self.assertTrue(call.call_args.kwargs["publish_probe_io"])
        self.assertEqual(call.call_args.kwargs["admission_review_policy"],"proposal-factual-review.v8.probe_io")


class RegistryPreparationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = replan_fixtures.ReplanTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_new_v3_contract_and_hash_only_fixture_ref_are_frozen_without_budget_reset(self):
        f = self.fixture
        source = f.root / "src/research1/data_pipeline/modules_v3.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("offline_fixture_source = 1\n", encoding="utf-8")
        registry = {"version": V3_REGISTRY, "packs": {}}
        adapter = SimpleNamespace(frozen_registry=lambda: copy.deepcopy(registry))
        rationale = "Freeze a new typed runtime contract over a previously scored parent without changing its original split or result."
        kwargs = dict(root=f.root, seed=7, cached_images={"fixture:1"}, mode="initialization",
            supersedes=f.latest, replan_from=f.source, diagnostic_tasks=[f.order[1]], diagnostic_rationale=rationale,
            teacher_max_tokens=16000, verification_registry_version=V3_REGISTRY,
            admission_review_policy="proposal-factual-review.v7.selectors", applicability_fixtures=True,
            publish_probe_io=True)
        with patch("research1.data_pipeline.module_adapters.get_adapter", return_value=adapter):
            protocol = prepare(f.campaign, f.config, **kwargs)
            self.assertEqual(load_protocol(f.campaign), protocol)
            self.assertEqual(prepare(f.campaign, f.config, **kwargs), protocol)
        self.assertEqual(protocol["verification_registry_version"], V3_REGISTRY)
        self.assertEqual(json.loads((f.campaign / "contracts.json").read_text()), registry)
        self.assertEqual(protocol["budget"], {"regular": 90, "retries": 32, "total": 122})
        self.assertEqual(protocol["stage_budget"]["proposal"], 2)
        self.assertTrue(protocol["grounded_edit_trial"]["publish_probe_io"])
        with patch("research1.data_pipeline.module_adapters.get_adapter", return_value=adapter):
            with self.assertRaisesRegex(ValueError,"input or token budget differs"):
                prepare(f.campaign,f.config,**dict(kwargs,publish_probe_io=False))
        reviewer = protocol["policies"]["applicability_reviewer"]
        self.assertEqual(reviewer["version"], "proposal-plan-applicability.v2.fixtures")
        self.assertNotIn("fixture_source", reviewer)
        reference = reviewer["fixture_source_ref"]
        self.assertEqual(set(reference), {"path", "sha256"})
        self.assertEqual(reference["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(Path(reference["path"]).read_bytes(), source.read_bytes())
        self.assertEqual(protocol["policies"]["factual_reviewer"]["version"], "proposal-factual-review.v7.selectors")

    def test_fresh_app3_freezes_original_fixture_reference_without_teacher_replay(self):
        f=self.fixture
        source=f.root/'src/research1/data_pipeline/modules_v3.py';source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('offline_fixture_source = 1\n',encoding='utf-8')
        registry={'version':V3_REGISTRY,'packs':{}}
        adapter=SimpleNamespace(frozen_registry=lambda:copy.deepcopy(registry))
        options=dict(root=f.root,seed=7,cached_images={'fixture:1'},mode='initialization',
            supersedes=f.latest,replan_from=f.source,diagnostic_tasks=[f.order[1]],
            diagnostic_rationale='Freeze an independent factual and applicability review before any new teacher plans or candidate outcomes.',
            teacher_max_tokens=16000,verification_registry_version=V3_REGISTRY,
            admission_review_policy='proposal-factual-review.v8.probe_io',
            applicability_review_policy='proposal-plan-applicability.v3.fact_ids',
            plan_precision_policy='proposal-plan-precision.v1.field_audit',
            applicability_fixtures=True,publish_probe_io=True)
        with patch('research1.data_pipeline.module_adapters.get_adapter',return_value=adapter):
            protocol=prepare(f.campaign,f.config,**options)
            self.assertEqual(prepare(f.campaign,f.config,**options),protocol)
        reviewer=protocol['policies']['applicability_reviewer']
        self.assertEqual(reviewer['version'],'proposal-plan-applicability.v3.fact_ids')
        self.assertIn('fixture_source_ref',reviewer);self.assertNotIn('fixture_source',reviewer)
        self.assertNotIn('teacher_source',protocol['grounded_edit_trial'])
        self.assertNotIn('replay_factual_from_source',protocol['grounded_edit_trial'])
        self.assertEqual(protocol['stage_budget']['proposal'],2)
        self.assertEqual(protocol['plan_precision_policy'],'proposal-plan-precision.v1.field_audit')
        self.assertEqual(protocol['policies']['factual_reviewer']['precision_policy'],protocol['plan_precision_policy'])
        with patch('research1.data_pipeline.module_adapters.get_adapter',return_value=adapter):
            with self.assertRaisesRegex(ValueError,'precision policy differs'):
                prepare(f.campaign,f.config,**dict(options,plan_precision_policy=None))


class FreshFixtureDriverTests(unittest.TestCase):
    def setUp(self):
        self.fixture = trial_fixtures.EditTrialTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_driver_selects_protocol_adapter_and_freezes_private_fixture_before_calls(self):
        f = self.fixture
        f.protocol["verification_registry_version"] = V3_REGISTRY
        f.protocol["sft"] = {"plan_policy_version": "proposal_single_plan_v6"}
        path = f.campaign / "source_snapshot/src/research1/data_pipeline/modules_v3.py"
        path.parent.mkdir(parents=True)
        path.write_text("offline frozen source\n")
        reference = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        f.protocol["policies"]["factual_reviewer"] = {"version": "proposal-factual-review.v7.selectors"}
        f.protocol["policies"]["applicability_reviewer"] = {"version": "proposal-plan-applicability.v2.fixtures", "fixture_source_ref": reference}
        context = {"sha256": reference["sha256"], "path": str(path), "text": "PRIVATE-FIXTURE-SOURCE"}
        adapter = SimpleNamespace(REGISTRY_VERSION=V3_REGISTRY, build_contract=Mock(return_value=f.contract),
            implement=Mock(side_effect=f.implement), verify=Mock(side_effect=f.verify),
            proposal_catalog=Mock(return_value=[]), planning_interfaces=modules.planning_interfaces)
        original = f.propose
        def propose(state, directory):
            self.assertEqual(state["applicability_fixture_source"], context)
            self.assertEqual(state["applicability_policy"], "proposal-plan-applicability.v2.fixtures")
            self.assertEqual(state["factual_review_policy"], "proposal-factual-review.v7.selectors")
            self.assertNotIn("PRIVATE-FIXTURE-SOURCE", json.dumps(public_view(state)))
            return original(state, directory)
        with patch("research1.data_pipeline.module_adapters.get_adapter", return_value=adapter) as select, \
             patch("research1.data_pipeline.fixture_context.freeze_probe_contract", return_value=context) as freeze:
            result = f.run_trial(proposer=propose, implementer=None, verifier=None, contract_builder=None)
        self.assertEqual(result["counts"]["qualified_edit"], 1, result)
        select.assert_any_call(f.protocol)
        freeze.assert_called_once_with(path)
        self.assertEqual(adapter.build_contract.call_count, 1)
        self.assertEqual(adapter.implement.call_count, 1)
        self.assertEqual(adapter.verify.call_count, 1)
        self.assertEqual(adapter.proposal_catalog.call_count, 1)

    def test_changed_fresh_fixture_ref_stops_before_any_fact_or_teacher_call(self):
        f = self.fixture
        path = f.campaign / "source_snapshot/modules_v3.py"
        path.parent.mkdir()
        path.write_text("changed fixture\n")
        f.protocol["policies"]["applicability_reviewer"] = {"version": "proposal-plan-applicability.v2.fixtures",
            "fixture_source_ref": {"path": str(path), "sha256": "0" * 64}}
        result = f.run_trial()
        self.assertEqual(result["status"], "BLOCKED_ARTIFACT_CHANGED")
        self.assertEqual(f.trace, [])
        self.assertEqual(Ledger(f.campaign).status()["budget"]["used"], {})


class PrecisionProfileDriverTests(unittest.TestCase):
    def test_fresh_precision_flag_reaches_both_teacher_states_and_resume(self):
        from research1.tests import test_data_probe_io_driver as publication_fixtures
        fixture=publication_fixtures.ProbeIODriverTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        f=fixture.fixture;precision='proposal-plan-precision.v1.field_audit'
        source=f.campaign/'source_snapshot/src/research1/data_pipeline/modules_v3.py'
        source.parent.mkdir(parents=True);source.write_text('offline fixture source\n')
        reference={'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        context={**reference,'text':'PRIVATE-FIXTURE-CONTEXT'}
        f.protocol['plan_precision_policy']=precision
        f.protocol['policies']['factual_reviewer']={'version':'proposal-factual-review.v8.probe_io','precision_policy':precision}
        f.protocol['policies']['applicability_reviewer']={'version':'proposal-plan-applicability.v3.fact_ids','fixture_source_ref':reference}
        original=f.propose;seen=[]
        def propose(state,directory):
            seen.append(state['teacher_arm_id'])
            self.assertEqual(state['plan_precision_policy'],precision)
            self.assertEqual(state['applicability_policy'],'proposal-plan-applicability.v3.fact_ids')
            self.assertEqual(state['applicability_fixture_source'],context)
            return original(state,directory)
        with patch('research1.data_pipeline.fixture_context.freeze_probe_contract',return_value=context):
            result=fixture.run_trial(proposer=propose)
            fixture.run_trial(proposer=propose)
        self.assertEqual(result['counts']['raw_plans'],2,result)
        self.assertEqual(seen,['standard','reasoned'])
        self.assertEqual(fixture.parent_probe.call_count,6)
        self.assertEqual(Ledger(f.campaign).status()['budget']['used'],{})

    def test_incomplete_precision_profile_stops_before_any_call(self):
        f=trial_fixtures.EditTrialTests();f.setUp();self.addCleanup(f.doCleanups)
        f.protocol['plan_precision_policy']='proposal-plan-precision.v1.field_audit'
        with self.assertRaisesRegex(ValueError,'frozen fresh published-v3'):
            f.run_trial()
        self.assertEqual(f.trace,[])


if __name__ == "__main__":
    unittest.main()
