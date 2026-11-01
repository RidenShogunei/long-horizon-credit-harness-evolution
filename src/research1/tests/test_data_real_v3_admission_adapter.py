"""Real v3 source/adapter wiring; model/task calls remain offline fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import modules as legacy,modules_v3 as strict
from research1.data_pipeline import applicability,fixture_context,module_adapters
from research1.data_pipeline.policy import digest
from research1.tests import test_data_admission_artifacts as positive_fixture
from research1.tests import test_data_plan_contract_binding as binding_fixture


class RealV3AdmissionAdapterTests(unittest.TestCase):
    def test_real_v3_original_snapshot_freezes_verifies_and_rejects_legacy_adapter(self):
        self.assertIs(module_adapters.get_adapter('v3'),strict)
        with tempfile.TemporaryDirectory() as directory:
            campaign=Path(directory)/'campaign'
            source=campaign/'source_snapshot'/fixture_context.STRICT_RELATIVE_SOURCE
            source.parent.mkdir(parents=True);source.write_bytes(Path(strict.__file__).read_bytes())
            manifest={'verification_registry_version':module_adapters.STRICT_VERSION,
                'source_sha256':{fixture_context.STRICT_RELATIVE_SOURCE:hashlib.sha256(source.read_bytes()).hexdigest()},
                'registry_sha256':digest(strict.frozen_registry())}
            manifest['protocol_hash']=digest(manifest)
            (campaign/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
            with patch('research1.distill._tongji_chat') as chat:
                context=fixture_context.freeze_probe_contract(source)
                self.assertEqual(fixture_context.verify_probe_contract(context),context)
            chat.assert_not_called()
            self.assertEqual(context['verification_registry_version'],module_adapters.STRICT_VERSION)
            self.assertEqual(context['executing_module_path'],str(Path(strict.__file__).resolve()))
            self.assertIn('second_completion_prompt_exact',context['text'])
            self.assertIn('class TaskConfig:',context['text'])
            with self.assertRaisesRegex(ValueError,'adapter differ'):
                fixture_context.freeze_probe_contract(source,executing_module=legacy)

    def test_real_v3_applicability_editor_and_positive_seal_use_selected_adapter(self):
        # Reuse the real saved-artifact fixture builder under an explicit v3
        # state. Parent probes, teacher/review/editor replies and physical trial
        # JSON are synthetic offline fixtures, never new collected examples.
        original_class=binding_fixture.PlanContractBindingTests
        def new_fixture(name):
            case=original_class(name);original_setup=case.setUp
            def setup():
                original_setup()
                case.state['verification_registry_version']=module_adapters.STRICT_VERSION
            case.setUp=setup
            return case
        case=positive_fixture.AdmissionArtifactTests('test_real_saved_fixture_passes_without_model_probe_or_task_execution')
        with patch.object(binding_fixture,'PlanContractBindingTests',side_effect=new_fixture), \
                patch.object(binding_fixture,'mod',strict),patch.object(positive_fixture,'modules',strict):
            case.setUp()
        self.addCleanup(case.doCleanups)
        state=case.record['state'];plan=case.record['plan'];contract=state['contract']
        self.assertEqual(contract['registry']['version'],module_adapters.STRICT_VERSION)
        self.assertEqual(case.record['implementation']['editor_policy_id'],'five-module-exact-replacements.v3')
        editor=json.loads((Path(case.record['implementation']['patch_path']).parent/'editor_input.json').read_text(encoding='utf-8'))
        self.assertEqual(editor['policy'],'five-module-exact-replacements.v3')
        self.assertIn('a bare helper name is a MODULE-LEVEL function',editor['prompt'])
        self.assertIn('ALL added text including headings and separators',editor['prompt'])
        self.assertIn('Read-only runtime interfaces:',editor['prompt'])
        with patch('research1.distill._tongji_chat') as chat, \
                patch.object(legacy,'_scope',side_effect=AssertionError('wrong legacy scope adapter')), \
                patch.object(legacy,'frozen_registry',side_effect=AssertionError('wrong legacy registry adapter')):
            self.assertTrue(applicability.validate_binding(state['plan_contract_binding'],state,plan,contract))
            self.assertEqual(positive_fixture.artifacts.validate_positive_artifacts(case.record),
                (True,'verified_v6_positive_artifact_binding'))
        chat.assert_not_called()
        self.assertEqual(case.record['qualification']['implementation_task_result']['reward'],0)


if __name__=='__main__':unittest.main()
