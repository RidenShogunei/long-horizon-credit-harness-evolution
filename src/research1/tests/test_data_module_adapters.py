"""Opt-in registry routing keeps legacy prompts and source bytes unchanged."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from research1.data_pipeline import modules,fixture_context
from research1.data_pipeline import module_adapters as adapters
from research1.data_pipeline.policy import canonical,render_messages,public_view
from research1.data_pipeline.engine import _public_state
from research1.data_pipeline.util import digest
from research1.tests.test_data_sft import record


class ModuleAdapterTests(unittest.TestCase):
    def strict_stub(self):
        return types.SimpleNamespace(REGISTRY_VERSION=adapters.STRICT_VERSION,
            planning_interfaces=modules.planning_interfaces)

    def test_factory_default_and_explicit_legacy_are_the_original_module(self):
        for value in [None,{},'v2',adapters.LEGACY_VERSION,{'verification_registry_version':'v2'}]:
            self.assertIs(adapters.get_adapter(value),modules)
        for value in ['v4',False,{'verification_registry_version':'unknown'}]:
            with self.assertRaisesRegex(ValueError,'Unsupported verification registry'):
                adapters.get_adapter(value)

    def test_factory_v3_requires_explicit_flag_without_inferring_private_contract(self):
        import research1.data_pipeline as package
        stub=self.strict_stub()
        with patch.object(package,'modules_v3',stub,create=True):
            for value in ['v3',adapters.STRICT_VERSION,{'verification_registry_version':'v3'}]:
                self.assertIs(adapters.get_adapter(value),stub)
        self.assertIs(adapters.get_adapter({'contract':{'registry':{'version':adapters.STRICT_VERSION}}}),modules)

    def test_all_prior_policy_message_bytes_and_legacy_registry_stay_identical(self):
        expected={
            'proposal_single_plan_v2':'3dc654d97f8b868d07045d94009631a10e73146ba35b2f887089a3bc81055386',
            'proposal_single_plan_v3':'a365028a28181568aedca6e5317450b06ea74137a3e20a57e522cf3bff73d020',
            'proposal_single_plan_v4':'6a6382352bf862e1c509f6ac918369b47515bd50d0c3c404a8a2e80cec4b9b1e',
            'proposal_single_plan_v5':'466c21fc587a881f2a7efd52f48d9a5478c4ba381c80858d38f8c931cfaccfca',
            'proposal_single_plan_v6':'ad5d31f46fb842061812be4930c06299f8fa4134d2f84c2255f4180e06d83326'}
        for version,sha in expected.items():
            state=copy.deepcopy(record()['state']);state['plan_policy_version']=version
            self.assertEqual(hashlib.sha256(canonical(render_messages(state)).encode()).hexdigest(),sha)
            state['verification_registry_version']='v2'
            self.assertEqual(hashlib.sha256(canonical(render_messages(state)).encode()).hexdigest(),sha)
        self.assertEqual(hashlib.sha256(Path(modules.__file__).read_bytes()).hexdigest(),
            '4a04740d5ae00ad847bcf3616afe00915ed81d2a4f4266b13a2e68f7d95d9b27')
        self.assertEqual(hashlib.sha256(canonical(modules.frozen_registry()).encode()).hexdigest(),
            'e78d6be6771c3d501b410c37b8527000984c90c18d2eb5bdfa60d409883640f6')

    def test_opt_in_view_and_prompt_add_only_generic_feasibility_constraints(self):
        state=record()['state'];state['plan_policy_version']='proposal_single_plan_v6'
        previous=public_view(state);previous_system=render_messages(state)[0]['content']
        state['verification_registry_version']='v3'
        with patch.object(adapters,'get_adapter',return_value=self.strict_stub()):
            view=public_view(state);messages=render_messages(state)
        self.assertEqual({key:value for key,value in view.items() if key not in {'verification_registry_version','planning_constraints'}},previous)
        self.assertEqual(view['verification_registry_version'],adapters.STRICT_VERSION)
        self.assertEqual(view['planning_constraints'],adapters.planning_constraints())
        self.assertTrue(messages[0]['content'].startswith(previous_system))
        self.assertIn('one concrete feasible implementation path',messages[0]['content'])
        self.assertIn('aliases between distinct runtime types',messages[0]['content'])
        for target_hint in ['gmail','bloom','completion_requirement_recheck','_next_prompt','AgentLoopState','ctx.state']:
            self.assertNotIn(target_hint,canonical(view['planning_constraints']))

    def test_strict_registry_cannot_be_silently_applied_to_a_prior_plan_interface(self):
        state=record()['state'];state['verification_registry_version']='v3'
        with patch.object(adapters,'get_adapter',return_value=self.strict_stub()):
            with self.assertRaisesRegex(ValueError,'explicitly frozen current plan interface'):
                public_view(state)

    def test_public_state_propagates_explicit_registry_without_exposing_private_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tools').mkdir();(root/'tools/baseline.py').write_text('def parse():\n    return 1\n')
            proto={'protocol_id':'p','policies':{},'sft':{'plan_policy_version':'proposal_single_plan_v6'},
                   'verification_registry_version':'v3'}
            with patch.object(adapters,'get_adapter',return_value=self.strict_stub()):
                state=_public_state(root,[{'id':'e','text':'original proposal','partition':'proposal'}],[],'r','train',['t'],proto)
                state['plan_factual_review']={'private':'DO_NOT_EXPOSE'}
                state['plan_contract_binding']={'private':'DO_NOT_EXPOSE'}
                view=public_view(state)
            self.assertEqual(state['verification_registry_version'],adapters.STRICT_VERSION)
            self.assertNotIn('DO_NOT_EXPOSE',canonical(view))

    def test_fixture_source_selection_supports_new_filename_without_changing_legacy_context_shape(self):
        import research1.data_pipeline as package
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=Path(modules.__file__).read_text(encoding='utf-8').replace(
                'REGISTRY_VERSION = "research1.module-contracts.v2"',
                'REGISTRY_VERSION = "research1.module-contracts.v3"',1)
            executing=root/'executing/modules_v3.py';executing.parent.mkdir();executing.write_text(source,encoding='utf-8')
            spec=importlib.util.spec_from_file_location('standalone_adapter_fixture',executing)
            stub=importlib.util.module_from_spec(spec);spec.loader.exec_module(stub)
            path=root/'campaign/source_snapshot'/fixture_context.STRICT_RELATIVE_SOURCE
            path.parent.mkdir(parents=True);path.write_bytes(executing.read_bytes())
            manifest={'source_sha256':{fixture_context.STRICT_RELATIVE_SOURCE:hashlib.sha256(path.read_bytes()).hexdigest()},
                      'registry_sha256':digest(stub.frozen_registry()),'verification_registry_version':adapters.STRICT_VERSION}
            manifest['protocol_hash']=digest(manifest)
            (root/'campaign/protocol.json').write_text(json.dumps(manifest),encoding='utf-8')
            with patch.object(package,'modules_v3',stub,create=True):
                context=fixture_context.freeze_probe_contract(path)
                self.assertEqual(fixture_context.verify_probe_contract(context),context)
            self.assertEqual(context['verification_registry_version'],adapters.STRICT_VERSION)
            self.assertTrue(context['read_only'])
            with self.assertRaisesRegex(ValueError,'adapter differ'):
                fixture_context.freeze_probe_contract(path,executing_module=modules)


if __name__=='__main__':unittest.main()
