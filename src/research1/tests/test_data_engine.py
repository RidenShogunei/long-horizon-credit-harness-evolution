"""Scheduler semantics and exact result binding; no model, Docker or task calls."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research1.data_pipeline.engine import run_campaign, _return_components
from research1.data_pipeline.execution import BindingError, bind_trial, tree_hash
from research1.data_pipeline.util import digest


class EngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.modules = self.root / 'baseline'
        for module in ['agent_loop','observation','tools','context_mgmt','verification']:
            (self.modules/module).mkdir(parents=True)
            (self.modules/module/'baseline.py').write_text('x = 1\n')
        self.campaign = self.root/'campaign';self.campaign.mkdir()
        self.protocol = {'protocol_id':'test-data-v1','modules_root':str(self.modules),
            'seed':42,'policies':{'solver':{'model':'fake','temperature':.7}},
            'task_sets':{'debug':['debug'+str(i) for i in range(8)],
                         'proposal':['support'+str(i) for i in range(24)],
                         'proposal_train':['support'+str(i) for i in range(19)],
                         'proposal_validation':['support'+str(i) for i in range(19,24)],
                         'reward':['reward0','reward1'],'dev':['dev0','dev1']},
            'sampling':{'K':2,'M':2,'L_after':2,'R':1,'reward_n':2,'dev_n':2,
                        'proposal_per_decision':2,'lambda':.5},
            'budget':{'regular':96,'retries':32,'total':128},'formal_scale':None,
            'evidence_budget':{'max_total_chars':96000,'max_step_chars':16000,'max_steps':12}}
        (self.campaign/'protocol.json').write_text(json.dumps(self.protocol))
        self.calls=[];self.teacher_calls=[];self.edit_calls=[]
        self.contract_patch=patch('research1.data_pipeline.modules.build_contract',return_value={'id':'pre-frozen','checks_id':'fixture','checks':[]})
        self.contract_patch.start();self.addCleanup(self.contract_patch.stop)
        self.protocol_patch=patch('research1.data_pipeline.protocol.load_protocol',return_value=self.protocol)
        self.protocol_patch.start();self.addCleanup(self.protocol_patch.stop)

    def evaluate(self,campaign,protocol,ledger,modules,task,role,scope,repeat):
        key=digest({'task':task,'role':role,'scope':scope,'repeat':repeat,'code':tree_hash(modules)})
        old=ledger.get_job(key)
        if old:return old['result']
        ledger.reserve_job(key,{'role':role},'regular');ledger.start_job(key,'test')
        self.calls.append((task,role,scope,repeat))
        modified='x = 2' in (modules/'tools/baseline.py').read_text()
        reward=(0.0 if modified else 1.0) if task=='reward0' else (1.0 if modified else 0.0) if task=='reward1' else 1.0
        result={'status':'PASS','reward':reward,'failure_layer':'model','evaluation_id':key,
                'evidence':[{'id':'trace','text':'Command transport and output observed'}]}
        ledger.complete_job(key,result);return result

    def plan(self,state,action='keep'):
        return {'action':action,'target_module':'tools','locations':['tools/baseline.py'] if action=='edit' else [],
            'evidence_refs':[state['evidence'][0]['id']],'problem_hypothesis':'Observed configured constant',
            'intended_change':'Use configured constant' if action=='edit' else 'Retain supported behavior',
            'preserve_behaviors':['Preserve command transport'],'verification_plan':['Frozen fixture'],
            'verification_kind':'configured_constant'}

    def proposer(self,state,directory):
        self.teacher_calls.append(copy.deepcopy(state))
        pool=self.protocol['task_sets']['proposal_validation' if self.protocol.get('collection_split')=='validation' else 'proposal_train']
        self.assertEqual(set(state['proposal_task_ids'])-set(pool),set())
        for item in state['history']:
            self.assertNotIn('scores',item);self.assertNotIn('reward',json.dumps(item))
        return [self.plan(state,'edit'),self.plan(state)] if not state['history'] else [self.plan(state)]

    def implement(self,plan,state,directory):
        self.edit_calls.append(str(directory))
        child=directory/'child';shutil.copytree(state['modules_root'],child)
        (child/'tools/baseline.py').write_text('x = 2\n')
        patch_path=directory/'saved.diff';patch_path.write_text('fixture stored patch\n')
        return {'status':'applied','modules_root':str(child),'code_hash':tree_hash(child),
                'parent_code_hash':tree_hash(Path(state['modules_root'])),
                'patch_path':str(patch_path),'patch_sha256':hashlib.sha256(patch_path.read_bytes()).hexdigest()}

    def verify(self,*args):return {'status':'PASS','covers_plan':True,'executable':True,'checks_id':'fixture','mechanism_id':'configured_constant'}

    def review(self,plan,state):
        return {'status':'PASS','reason':'Independent fixture evidence review',
                'independent':True,
                'evidence_refs':[state['evidence'][0]['id']],'reviewer_id':'independent-test'}

    def run_engine(self,**extra):
        args={'evaluator':self.evaluate,'proposer':self.proposer,'implementer':self.implement,
              'verifier':self.verify,'reviewer':self.review,'diagnostician':lambda state,directory:[]};args.update(extra)
        return run_campaign(self.campaign,**args)

    def test_real_decision_noops_fixed_paths_shared_implementation_and_resume(self):
        result=self.run_engine()
        self.assertEqual(result['status'],'NEEDS_SCALE_LOCK',result)
        self.assertFalse(result['data_collection_complete'])
        self.assertEqual(result['functional_first_edits'],1)
        self.assertEqual(result['complete_paths'],4)
        self.assertEqual(len(self.edit_calls),1)
        # 24 noise +18 proposal +4 reward for parent/edit +10 dev.
        self.assertEqual(len(self.calls),56)
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        values=ledger.records('values');self.assertEqual(len(values),2)
        edit=next(v for v in values if v['mean_cumulative_regression']>0)
        self.assertEqual(edit['mean_terminal_gain'],0)
        self.assertEqual(edit['mean_cumulative_regression'],.5)
        self.assertEqual(edit['mean_u_lambda'],-.25)
        paths=ledger.records('paths');self.assertEqual(len(paths),4)
        for path in paths:
            self.assertEqual(len(path['nodes']),4)
            self.assertEqual(len({n['decision_id'] for n in path['nodes']}),4)
            self.assertEqual(path['nodes'][1]['code_hash'],path['first_implementation_hash'])
            self.assertEqual(path['nodes'][1]['code_hash'],path['nodes'][3]['code_hash'])
        old_counts=(len(self.calls),len(self.teacher_calls),len(self.edit_calls))
        again=self.run_engine()
        self.assertEqual(again['status'],'NEEDS_SCALE_LOCK')
        self.assertEqual(old_counts,(len(self.calls),len(self.teacher_calls),len(self.edit_calls)))
        self.assertEqual(len(ledger.records('values')),2)
        self.assertEqual(ledger.records('comparisons')[0]['preference'],'undetermined')

    def test_keep_only_cannot_certify_functional_calibration(self):
        result=self.run_engine(proposer=lambda state,directory:[self.plan(state),dict(self.plan(state),intended_change='Another abstention')])
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertFalse(result['calibration_complete'])
        self.assertEqual(result['functional_first_edits'],0)

    def test_unreviewed_samples_remain_pending_even_with_valid_results(self):
        self.run_engine(reviewer=lambda *args:{'status':'pending','reason':'Not independently reviewed','evidence_refs':[]})
        from research1.data_pipeline.ledger import Ledger
        records=Ledger(self.campaign).records('samples')
        self.assertTrue(records)
        self.assertTrue(all(r['qualification']['status']=='pending' for r in records))

    def test_invalid_score_stops_without_value_zero_or_replacement_path(self):
        def fail(*args):return {'status':'INCOMPLETE','reward':None,'failure_layer':'scorer','error':'missing score'}
        result=self.run_engine(evaluator=fail)
        self.assertEqual(result['status'],'INCOMPLETE')
        from research1.data_pipeline.ledger import Ledger
        self.assertEqual(Ledger(self.campaign).records('values'),[])
        self.assertEqual(self.teacher_calls,[])

    def test_downstream_failed_apply_retains_state_and_completes_decision_horizon(self):
        def planner(state,directory):
            if not state['history']:return self.proposer(state,directory)
            return [self.plan(state,'edit')]
        def editor(plan,state,directory):
            if state['history']:return {'status':'error','reason':'patch did not match'}
            return self.implement(plan,state,directory)
        result=self.run_engine(proposer=planner,implementer=editor)
        self.assertEqual(result['status'],'NEEDS_SCALE_LOCK')
        from research1.data_pipeline.ledger import Ledger
        for path in Ledger(self.campaign).records('paths'):
            self.assertEqual(path['nodes'][1]['code_hash'],path['nodes'][-1]['code_hash'])
            self.assertFalse(path['nodes'][-1]['accepted'])

    def test_first_apply_failure_has_no_numeric_utility(self):
        result=self.run_engine(implementer=lambda *args:{'status':'error','reason':'patch failed'})
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        from research1.data_pipeline.ledger import Ledger
        invalid=next(v for v in Ledger(self.campaign).records('values') if not v['is_complete'])
        self.assertIsNone(invalid['mean_u_lambda'])

    def test_taskwise_regression_is_taken_before_weighting(self):
        nodes=[{'scores':{'a':1.,'b':0.}},{'scores':{'a':0.,'b':1.}}]
        result=_return_components(nodes,['a','b'],.5,.5)
        self.assertEqual(result,{'terminal_gain':0.,'cumulative_regression':.5,'u_lambda':-.25,'lambda_value':.5})

    def test_cached_implementation_tamper_blocks_resume_and_revokes_derived_views(self):
        result=self.run_engine();self.assertEqual(result['status'],'NEEDS_SCALE_LOCK')
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        attempt=ledger.records('implementations')[0]['value']
        (Path(attempt['modules_root'])/'tools/baseline.py').write_text('x = 999\n')
        before=len(self.calls)
        resumed=self.run_engine()
        self.assertEqual(resumed['status'],'BLOCKED_ARTIFACT_CHANGED')
        self.assertTrue(all(v['revoked'] for v in ledger.records('values')))
        self.assertEqual(len(self.calls),before)

    def test_validation_campaign_uses_own_support_tasks_and_can_skip_noise(self):
        self.protocol.update(collection_split='validation',noise_enabled=False)
        result=self.run_engine()
        self.assertEqual(result['status'],'NEEDS_SCALE_LOCK')
        self.assertFalse(any(role=='noise' for task,role,scope,repeat in self.calls))
        self.assertTrue(all(s['split']=='validation' for s in self.teacher_calls))
        self.assertTrue(all(set(s['proposal_task_ids'])<=set(self.protocol['task_sets']['proposal_validation']) for s in self.teacher_calls))

    def test_identical_realized_edits_do_not_count_as_a_distinct_counterfactual_group(self):
        def same_code(state,directory):
            return [self.plan(state,'edit'),dict(self.plan(state,'edit'),intended_change='Equivalent configuration')]
        result=self.run_engine(proposer=same_code)
        self.assertEqual(result['status'],'INSUFFICIENT_EVIDENCE')
        self.assertEqual(result['complete_paths'],0)
        self.assertIn('same action/code',result['reason'])


class TrialBindingTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.modules=self.root/'modules';self.modules.mkdir();(self.modules/'x.py').write_text('x=1\n')
        self.task=self.root/'dataset'/'task';self.task.mkdir(parents=True);(self.task/'task.toml').write_text('id="test"')
        self.output=self.root/'out';job=self.output/'job';self.trial=job/'task__a';self.trial.mkdir(parents=True)
        self.spec={'task':'task','task_path':str(self.task),'modules_root':str(self.modules),
                   'code_hash':tree_hash(self.modules),'task_hash':tree_hash(self.task),
                   'model':'fake','provider_id':'test','temperature':.7,'max_turns':200}
        (job/'config.json').write_text(json.dumps({'tasks':[{'path':str(self.task)}],'jobs_dir':str(self.output)}))
        self.config={'task':{'path':str(self.task)},'trial_name':self.trial.name,'trials_dir':str(job),
                     'agent':{'model_name':'fake','kwargs':{'modules_root':str(self.modules),
                       'temperature':.7,'max_turns':200,'composer_name':'static','composer_scope':'all'}}}
        (self.trial/'config.json').write_text(json.dumps(self.config))
        self.result={'task_name':'task','trial_name':self.trial.name,'config':self.config,
                     'finished_at':'now','verifier_result':{'rewards':{'reward':0.}},'exception_info':None}
        (self.trial/'result.json').write_text(json.dumps(self.result))
        (job/'result.json').write_text('{"reward": 1}')

    def test_uses_exact_concrete_trial_valid_zero_not_job_aggregate(self):
        result=bind_trial(self.output,self.spec)
        self.assertEqual(result['reward'],0.)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['trial_result_path'],str(self.trial/'result.json'))

    def test_rejects_ambiguous_trials_wrong_conditions_and_changed_assets(self):
        with self.assertRaises(BindingError):bind_trial(self.output,dict(self.spec,model='other'))
        (self.modules/'x.py').write_text('x=2\n')
        with self.assertRaises(BindingError):bind_trial(self.output,self.spec)
        (self.modules/'x.py').write_text('x=1\n')
        second=self.trial.parent/'task__b';second.mkdir();(second/'config.json').write_text('{}')
        with self.assertRaises(BindingError):bind_trial(self.output,self.spec)

    def test_missing_score_stays_null_and_error_classification_uses_exception_only(self):
        self.result['verifier_result']={};self.result['exception_info']={'exception_type':'AgentTimeoutError','exception_message':'agent execution timed out'}
        (self.trial/'result.json').write_text(json.dumps(self.result))
        result=bind_trial(self.output,self.spec)
        self.assertIsNone(result['reward']);self.assertEqual(result['status'],'INCOMPLETE')


class ExecutionRecoveryTests(TrialBindingTests):
    def setUp(self):
        super().setUp()
        self.campaign=self.root/'campaign';self.campaign.mkdir()
        self.config_path=self.root/'pilot.yaml'
        self.config_path.write_text('dataset_tb: '+str(self.task.parent.parent).replace('\\','/')+'\n')
        # The real adapter resolves dataset/tasks/<name>.
        actual=self.task.parent/'tasks'/'task';actual.parent.mkdir();shutil.copytree(self.task,actual)
        self.config_path.write_text('dataset_tb: '+str(self.task.parent).replace('\\','/')+'\n')
        self.task=actual
        self.protocol={'protocol_id':'test','config_path':str(self.config_path),'budget':{'regular':3,'retries':2,'total':5},
            'runtime':{'max_turns':200,'wall_seconds':60,'disk_floor_gb':0},
            'policies':{'solver':{'model':'fake','temperature':.7}},'stage_budget':{'reward':3}}
        (self.campaign/'protocol.json').write_text(json.dumps(self.protocol))
        self.commands=[];self.rewards=[0.]
        provider=SimpleNamespace(model='fake',provider_id='test',api_base='test')
        for target,value in [('research1.adapters.with_llm_provider_env',provider),
                             ('research1.adapters.apply_provider_to_environ',None)]:
            mocked=patch(target,return_value=value);mocked.start();self.addCleanup(mocked.stop)
        image=patch('research1.data_pipeline.execution._image_identity',side_effect=lambda task:
            {'dockerfile_sha256':'frozen','image_ids':{'hb__task':'built' if self.commands else None}})
        image.start();self.addCleanup(image.stop)
        command=patch('research1.data_pipeline.execution._command',side_effect=self.prepare_command)
        command.start();self.addCleanup(command.stop)

    def prepare_command(self,protocol,spec,output,provider):
        self.commands.append(str(output))
        job=output/'job';trial=job/'task__bound';trial.mkdir(parents=True)
        (job/'config.json').write_text(json.dumps({'tasks':[{'path':str(self.task)}],'jobs_dir':str(output)}))
        config=copy.deepcopy(self.config)
        config.update(task={'path':str(self.task)},trial_name=trial.name,trials_dir=str(job))
        config['agent']['kwargs']['model_info']=spec['model_info']
        (trial/'config.json').write_text(json.dumps(config))
        reward=self.rewards[len(self.commands)-1]
        result={'task_name':'task','trial_name':trial.name,'config':config,'finished_at':'now',
                'verifier_result':{'rewards':{'reward':reward}},
                'exception_info':None if reward is not None else {'exception_type':'ConnectionError','exception_message':'connection reset'}}
        (trial/'result.json').write_text(json.dumps(result))
        return ['never-executed'],{},self.root

    def test_completed_slot_reuses_original_binding_after_image_is_built(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=lambda **kw:0)) as launch:
            first=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
            again=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(first['reward'],0.)
        self.assertEqual(first,again)
        self.assertEqual(launch.call_count,1)
        self.assertEqual(ledger.status()['budget']['used'],{'regular':1})

    def test_infrastructure_retry_is_distinct_and_counted_never_retries_valid_zero(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        self.rewards=[None,0.]
        ledger=Ledger(self.campaign)
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=lambda **kw:0)):
            result=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(result['reward'],0.)
        self.assertEqual(len(self.commands),2)
        self.assertEqual(ledger.status()['budget']['used'],{'regular':1,'retry':1})

    def test_operational_wall_timeout_with_missing_score_never_uses_retry_budget(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        self.rewards=[None]
        ledger=Ledger(self.campaign)
        waits=iter([subprocess.TimeoutExpired('frozen-task',60),-15])
        def wait(**kwargs):
            value=next(waits)
            if isinstance(value,Exception):raise value
            return value
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=wait)) as launch, \
             patch('research1.data_pipeline.execution.cleanup_owned',return_value={'process_groups':[],'containers':[]}) as cleanup:
            result=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertIsNone(result['reward']);self.assertEqual(result['status'],'INCOMPLETE')
        self.assertTrue(result['wall_limit_reached']);self.assertFalse(result['retryable'])
        self.assertTrue(result['stopped_for_budget'])
        self.assertEqual(launch.call_count,1);cleanup.assert_called_once()
        self.assertEqual(ledger.status()['budget']['used'],{'regular':1})
        with patch('research1.data_pipeline.execution.subprocess.Popen') as resumed_launch:
            again=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(again,result);resumed_launch.assert_not_called()

    def test_operational_wall_timeout_retains_valid_zero_without_retry(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        calls=[]
        def wait(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:raise subprocess.TimeoutExpired('frozen-task',60)
            return -15
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=wait)) as launch, \
             patch('research1.data_pipeline.execution.cleanup_owned',return_value={'process_groups':[],'containers':[]}):
            result=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(result['reward'],0.);self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['wall_limit_reached']);self.assertFalse(result['retryable'])
        self.assertEqual(launch.call_count,1)
        self.assertEqual(ledger.status()['budget']['used'],{'regular':1})

    def test_resumed_live_job_past_its_wall_limit_records_no_retry_stop(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        def crash(**kwargs):raise KeyboardInterrupt()
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=crash)):
            with self.assertRaises(KeyboardInterrupt):
                evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        marker_path=next((self.campaign/'executions').glob('*/attempt_0/launch.json'))
        marker=json.loads(marker_path.read_text());marker['started_unix']=0
        marker_path.write_text(json.dumps(marker))
        with patch('research1.data_pipeline.execution._owned_process',return_value=True), \
             patch('research1.data_pipeline.execution.cleanup_owned',return_value={'process_groups':[],'containers':[]}), \
             patch('research1.data_pipeline.execution.subprocess.Popen') as launch:
            result=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(result['reward'],0.)
        self.assertTrue(result['wall_limit_reached']);self.assertFalse(result['retryable'])
        self.assertTrue(json.loads(marker_path.read_text())['wall_limit_reached'])
        launch.assert_not_called()

    def test_crash_after_trial_finished_recovers_without_relaunching(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        def crash(**kw):raise KeyboardInterrupt()
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=crash)):
            with self.assertRaises(KeyboardInterrupt):
                evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        with patch('research1.data_pipeline.execution.subprocess.Popen') as launch:
            recovered=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(recovered['reward'],0.)
        launch.assert_not_called()
        self.assertEqual(len(self.commands),1)

    def test_changed_committed_result_is_rejected_not_rerun_for_a_new_score(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=lambda **kw:0)):
            first=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        path=Path(first['trial_result_path']);data=json.loads(path.read_text())
        data['verifier_result']['rewards']['reward']=1.
        path.write_text(json.dumps(data))
        with patch('research1.data_pipeline.execution.subprocess.Popen') as launch:
            result=evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','fixed')
        self.assertEqual(result['status'],'INCOMPLETE');self.assertTrue(result['fatal'])
        launch.assert_not_called()

    def test_task_asset_drift_between_scopes_is_not_a_new_compatible_evaluation(self):
        from research1.data_pipeline.execution import evaluate
        from research1.data_pipeline.ledger import Ledger
        ledger=Ledger(self.campaign)
        with patch('research1.data_pipeline.execution.subprocess.Popen',return_value=SimpleNamespace(pid=99999999,wait=lambda **kw:0)):
            evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','first')
        (self.task/'task.toml').write_text('id="changed"')
        with patch('research1.data_pipeline.execution.subprocess.Popen') as launch:
            with self.assertRaisesRegex(BindingError,'changed between scopes'):
                evaluate(self.campaign,self.protocol,ledger,self.modules,'task','reward','second')
        launch.assert_not_called()

