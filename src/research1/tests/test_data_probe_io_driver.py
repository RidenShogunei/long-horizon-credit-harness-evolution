"""Publish saved parent I/O before planning; never rerun or rewrite probes."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research1.data_pipeline.cli import _export
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.policy import public_view
from research1.data_pipeline.util import digest
from research1.data_pipeline import modules_v3
from research1.data_pipeline.predecision_probe_io import publish_parent_probe_io,validate_probe_io_state
from research1.tests import test_data_edit_trial as fixtures
from research1.tests.test_data_predecision_probe_io import _probe as offline_probe_output


class ProbeIODriverTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EditTrialTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        f.protocol['verification_registry_version']='research1.module-contracts.v3'
        f.protocol['sft']={'plan_policy_version':'proposal_single_plan_v6'}
        f.protocol['diagnostic_source_selection']={'tasks':['bloom'],'frozen_before_candidates':True}
        f.protocol['grounded_edit_trial']['publish_probe_io']=True
        self.probe_calls=0
        self.publication_calls=0
        self.validation_calls=0
        # Exercise the actual contract builder, publisher, physical validator
        # and policy projection. Only the six parent API outputs are offline
        # fixtures; no model, solver or task execution occurs in these tests.
        calls=patch.object(modules_v3,'probe_module',side_effect=offline_probe_output)
        self.parent_probe=calls.start();self.addCleanup(calls.stop)

    def build_contract(self,state):
        self.probe_calls+=1
        self.fixture.trace.append(('build_parent_probes',None))
        return modules_v3.build_contract(state)

    def publish(self,state,contract,directory,*,protocol_hash):
        self.publication_calls+=1
        self.fixture.trace.append(('publish_parent_io',None))
        self.assertEqual(self.probe_calls,1)
        self.assertFalse(any(kind in {'facts','teacher'} for kind,_ in self.fixture.trace))
        return publish_parent_probe_io(state,contract,directory,protocol_hash=protocol_hash)

    def validate(self,state):
        self.validation_calls+=1
        return validate_probe_io_state(state)

    def facts(self,state,directory):
        self.assertTrue(state['evidence'][-1]['id'].startswith('predecision.harness_api.io.'))
        self.assertEqual(state['contract']['evidence_hashes'],{row['id']:digest(row) for row in state['evidence']})
        self.assertEqual(self.validation_calls,1)
        return self.fixture.facts(state,directory)

    def run_trial(self,**kwargs):
        arguments={'contract_builder':self.build_contract,'probe_io_publisher':self.publish,
                   'probe_io_validator':self.validate,'fact_collector':self.facts,'reviewer':self.pending_review}
        arguments.update(kwargs)
        return self.fixture.run_trial(**arguments)

    def pending_review(self,plan,state):
        result=self.fixture.review(plan,state)
        result.update(status='pending_review',reason='Offline scheduling fixture does not claim positive admission')
        return result

    def test_probe_publication_precedes_facts_and_teacher_without_second_build(self):
        result=self.run_trial()
        self.assertEqual(result['counts']['raw_plans'],2,result)
        self.assertEqual(result['counts']['pending'],2,result)
        self.assertEqual(self.probe_calls,1)
        self.assertEqual(self.publication_calls,1)
        self.assertEqual(self.validation_calls,1)
        self.assertEqual(self.parent_probe.call_count,6)
        trace=[kind for kind,_ in self.fixture.trace]
        self.assertLess(trace.index('build_parent_probes'),trace.index('publish_parent_io'))
        self.assertLess(trace.index('publish_parent_io'),trace.index('facts'))
        self.assertLess(trace.index('facts'),trace.index('teacher'))
        rows=Ledger(self.fixture.campaign).records('samples')
        self.assertEqual(len(rows),2)
        for row in rows:
            state=row['state']
            self.assertIn('predecision_probe_artifacts',state)
            original=Ledger(self.fixture.campaign).records('edittrial_predecision_contracts')[0]['value']
            self.assertEqual(state['contract']['parent_probes'],original['parent_probes'])
            view=json.dumps(public_view(state))
            self.assertIn('BaselineTools.parse_llm_response',view)
            self.assertTrue(public_view(state)['predecision_probe_evidence_refs'])
            self.assertNotIn('predecision_probe_artifacts',view)
            self.assertNotIn('parent_probes',view)

    def test_resume_revalidates_receipt_without_republishing_or_reexecuting_probes(self):
        self.run_trial()
        before=list(self.fixture.trace)
        result=self.run_trial()
        self.assertEqual(result['counts']['raw_plans'],2,result)
        self.assertEqual(self.fixture.trace,before)
        self.assertEqual(self.probe_calls,1)
        self.assertEqual(self.publication_calls,1)
        self.assertEqual(self.validation_calls,2)
        self.assertEqual(self.parent_probe.call_count,6)

    def test_changed_receipt_on_resume_revokes_and_stops_before_more_model_calls(self):
        self.run_trial()
        row=Ledger(self.fixture.campaign).records('samples')[0]
        receipt=Path(row['state']['predecision_probe_artifacts']['receipt']['path'])
        receipt.write_text('changed receipt')
        before=list(self.fixture.trace)
        result=self.run_trial()
        self.assertEqual(result['status'],'BLOCKED_ARTIFACT_CHANGED')
        self.assertEqual(self.fixture.trace,before)
        self.assertEqual(self.probe_calls,1)
        self.assertTrue(all(row['revoked'] for row in Ledger(self.fixture.campaign).records('samples')))

    def test_publisher_cannot_change_check_or_scope_content(self):
        def invalid(state,contract,directory,**kwargs):
            state,final=self.publish(state,contract,directory,**kwargs)
            final['registry']['version']='changed'
            final['checks_id']='checks_'+digest({key:value for key,value in final.items() if key!='checks_id'})[:24]
            return state,final
        result=self.run_trial(probe_io_publisher=invalid)
        self.assertEqual(result['status'],'BLOCKED_ARTIFACT_CHANGED')
        self.assertFalse(any(kind in {'facts','teacher'} for kind,_ in self.fixture.trace))
        self.assertEqual(Ledger(self.fixture.campaign).status()['budget']['used'],{})

    def test_no_flag_keeps_the_original_facts_then_contract_order(self):
        self.fixture.protocol['grounded_edit_trial'].pop('publish_probe_io')
        def forbidden(*args,**kwargs):
            raise AssertionError('Disabled publication must not be called')
        result=self.fixture.run_trial(contract_builder=self.build_contract,
            probe_io_publisher=forbidden,probe_io_validator=forbidden,reviewer=self.pending_review)
        self.assertEqual(result['counts']['raw_plans'],2,result)
        trace=[kind for kind,_ in self.fixture.trace]
        self.assertLess(trace.index('facts'),trace.index('build_parent_probes'))


class ExportRevocationCountTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.EditTrialTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_export_recounts_active_qualifications_and_all_raw_after_revoke(self):
        f=self.fixture; ledger=Ledger(f.campaign)
        rows=[{'sample_id':'revoked-keep','plan':{'action':'keep'},'qualification':{'status':'verified'},'revoked':True},
              {'sample_id':'pending-edit','plan':{'action':'edit'},'qualification':{'status':'pending'},'revoked':False},
              {'sample_id':'rejected-edit','plan':{'action':'edit'},'qualification':{'status':'rejected'},'revoked':False},
              {'sample_id':'verified-edit','plan':{'action':'edit'},'qualification':{'status':'verified'},'revoked':False}]
        for row in rows:ledger.put_record('samples',row['sample_id'],row)
        before=ledger.records('samples')
        counts={'raw_plans':999,'qualified_edit':0,'qualified_keep':1,'pending':0,'rejected':9,
            'unique_parent_roots':1,'parents_processed':1,'reused_parent_executions':1,'teacher_errors':3}
        (f.campaign/'status.json').write_text(json.dumps({'status':'GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT','counts':counts}))
        manifest={'dataset_revision':'offline-export','counts':{},'training_ready':False,
                  'readiness_reasons':['student_model_and_template_not_selected']}
        with patch('research1.data_pipeline.cli.load_protocol',return_value=f.protocol), \
             patch('research1.data_pipeline.sft.export_dataset',return_value=manifest):
            _export(f.campaign)
            first=json.loads((f.campaign/'status.json').read_text())
            _export(f.campaign)
        second=json.loads((f.campaign/'status.json').read_text())
        self.assertEqual(first,second)
        actual=second['counts']
        self.assertEqual({key:actual[key] for key in ('raw_plans','qualified_edit','qualified_keep','pending','rejected','revoked')},
            {'raw_plans':4,'qualified_edit':1,'qualified_keep':0,'pending':1,'rejected':1,'revoked':1})
        for key in ('unique_parent_roots','parents_processed','reused_parent_executions','teacher_errors'):
            self.assertEqual(actual[key],counts[key])
        self.assertEqual(ledger.records('samples'),before)


if __name__=='__main__':
    unittest.main()
