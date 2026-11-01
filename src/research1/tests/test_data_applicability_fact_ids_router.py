"""New router/source gates retain exact factual provenance and immutable source bytes."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability as router,applicability_v3,selector_review,teacher_replay
from research1.data_pipeline.engine import ArtifactChangedError
from research1.tests import test_data_applicability_fact_ids_v3 as fact_id_fixtures
from research1.tests import test_data_admission_artifacts as positive_fixtures


class ApplicabilityFactIdsRouterTests(unittest.TestCase):
    def setUp(self):
        self.case=fact_id_fixtures.ApplicabilityFactIdsV3Tests('test_api_only_witness_ids_bind_without_trajectory_requote')
        self.case.setUp();self.addCleanup(self.case.doCleanups)

    def test_router_dispatches_new_review_and_binding_without_reinterpreting_old_records(self):
        review=self.case.factual_pass()
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(self.case.applicability_reply()),self.case.meta)):
            binding=router.review_applicability(self.case.state,self.case.plan,review,self.case.root/'routed')
        self.assertEqual(binding['status'],'PASS',binding)
        self.assertEqual(binding['policy'],applicability_v3.POLICY_ID)
        self.assertTrue(router.validate_binding(binding,self.case.state,self.case.plan,self.case.contract))
        state=copy.deepcopy(self.case.state);state.pop('applicability_policy')
        with patch.object(applicability_v3,'validate_binding',wraps=applicability_v3.validate_binding) as backend:
            self.assertFalse(router.validate_binding(binding,state,self.case.plan,self.case.contract))
        self.assertEqual(backend.call_count,1)

    def test_valid_but_unselected_original_factual_catalog_id_cannot_be_added_by_app_reviewer(self):
        review=self.case.factual_pass()
        unselected=next(row['id'] for row in self.case.catalog()['items'] if row['id'] not in review['witness_ids'])
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(self.case.applicability_reply([unselected])),self.case.meta)):
            binding=router.review_applicability(self.case.state,self.case.plan,review,self.case.root/'unselected')
        self.assertEqual(binding['status'],'pending')
        self.assertIn('Unknown factual witness id',binding['reason'])

    def test_reused_fact_source_is_rechecked_and_revocation_invalidates_the_new_binding(self):
        state=self.case.state;plan=self.case.plan;review=self.case.factual_pass()
        proof=selector_review.verify_saved_selected_review(review,state,plan)
        state.update(replay_factual_from_source=True,original_factual_review=copy.deepcopy(review))
        with patch.object(teacher_replay,'validate_replayed_factual_state',return_value=proof,create=True) as source, \
                patch('research1.distill._tongji_chat',return_value=(json.dumps(self.case.applicability_reply()),self.case.meta)):
            binding=router.review_applicability(state,plan,review,self.case.root/'source-bound')
            self.assertEqual(binding['status'],'PASS',binding)
            self.assertTrue(router.validate_binding(binding,state,plan,self.case.contract))
        self.assertGreaterEqual(source.call_count,2)
        with patch.object(teacher_replay,'validate_replayed_factual_state',side_effect=ArtifactChangedError('Source sample revoked'),create=True):
            self.assertFalse(router.validate_binding(binding,state,plan,self.case.contract))
            self.assertFalse(router.validate_factual_review(review,state,plan))
        changed=dict(review,reason='Another review object')
        with patch.object(teacher_replay,'validate_replayed_factual_state',create=True) as source:
            with self.assertRaisesRegex(ValueError,'exact original review object'):
                router._factual_proof(changed,state,plan)
        source.assert_not_called()

    def test_python_fixture_manifest_bytes_are_sealed_and_export_retracts_on_drift(self):
        case=positive_fixtures.AdmissionArtifactTests('test_real_saved_fixture_passes_without_model_probe_or_task_execution')
        # Create a first-admission fixture; no changed positive is resealed.
        with patch.object(positive_fixtures.artifacts,'seal_positive_artifacts',return_value={}):
            case.setUp()
        self.addCleanup(case.doCleanups)
        source=case.root/'original_fixture_modules.py'
        source.write_bytes(b'def original_fixture():\n    return 1\n')
        record=case.record
        record['state']['teacher_source_artifacts']={'original_modules_fixture':{
            'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}
        record['admission_artifacts']=positive_fixtures.artifacts.seal_positive_artifacts(record)
        self.assertTrue(positive_fixtures.artifacts.validate_positive_artifacts(record)[0])
        campaign=case.root/'source-export'
        first=positive_fixtures.sft.export_dataset(campaign,[record])
        self.assertEqual(first['counts']['train']['edit'],1)
        source.write_bytes(source.read_bytes()+b'# altered original fixture\n')
        self.assertFalse(positive_fixtures.artifacts.validate_positive_artifacts(record)[0])
        changed=positive_fixtures.sft.export_dataset(campaign,[record])
        self.assertEqual(changed['counts']['train']['edit'],0)
        self.assertNotEqual(first['dataset_revision'],changed['dataset_revision'])

    def test_json_source_originals_still_require_object_shape(self):
        source=self.case.root/'original.json'
        source.write_text('[]',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'remain an object'):
            positive_fixtures.artifacts._read_source_artifact(str(source))
        source.write_text('{}',encoding='utf-8')
        path,sha=positive_fixtures.artifacts._read_source_artifact(str(source))
        self.assertEqual(path,source)
        self.assertEqual(sha,hashlib.sha256(source.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
