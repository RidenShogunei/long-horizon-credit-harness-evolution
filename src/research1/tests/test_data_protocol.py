import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from research1.data_pipeline.protocol import partition_tasks,validate_partitions,exposed_tasks,load_protocol,handover_budget,initialization_queue,runtime_debug_partition,image_inventory,prepare
from research1.data_pipeline.util import atomic_json,digest,tree_hash

class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        # These fixtures isolate partition/budget behavior. The real source
        # context and its visibility boundary have separate integration tests.
        context = patch('research1.data_pipeline.framework_context.build_framework_context', return_value=None)
        context.start(); self.addCleanup(context.stop)

    def test_explicit_parent_queue_can_include_frozen_validation_without_replay_or_readiness_bypass(self):
        sets={'proposal_train':['old','train','unprepared'],'proposal_validation':['validation']}
        inventory={t:{'full_image_cached':True} for t in ['old','train','validation']}
        queue=['train','validation']
        actual=initialization_queue(sets,inventory,{},seed=7,max_parents=2,excluded_tasks={'old'},task_order=queue)
        self.assertEqual(actual,queue)
        self.assertEqual(sets['proposal_validation'],['validation'])
        for invalid in (['old'],['unprepared'],['reward-task'],['train','train']):
            with self.subTest(queue=invalid), self.assertRaises(ValueError):
                initialization_queue(sets,inventory,{},seed=7,max_parents=2,excluded_tasks={'old'},task_order=invalid)

    def test_prebuilt_task_latest_tag_is_eligible_without_base_or_build_readiness(self):
        with tempfile.TemporaryDirectory() as td:
            dataset=Path(td);task='heavy';env=dataset/'tasks'/task/'environment';env.mkdir(parents=True)
            (env/'Dockerfile').write_text('FROM unavailable:1\nRUN apt-get install build-essential\n')
            inventory=image_inventory(dataset,[task],{'hb__heavy:latest'})
            self.assertTrue(inventory[task]['full_image_cached'])
            self.assertFalse(inventory[task]['cached']);self.assertTrue(inventory[task]['heavy_build'])
            sets={'proposal_train':[task],'proposal_validation':[]}
            self.assertEqual(initialization_queue(sets,inventory,{},seed=7,max_parents=1),[task])
            for cached in ({'hb__heavy:other'},{'hb__heavy-other:latest'}):
                self.assertFalse(image_inventory(dataset,[task],cached)[task]['full_image_cached'])
            ready={task:{'status':'known_budget_incompatible'}}
            self.assertEqual(initialization_queue(sets,inventory,ready,seed=7,max_parents=1),[])
            self.assertEqual(initialization_queue(sets,inventory,{task:{'status':'env_unavailable'}},seed=7,max_parents=1),[])

    def test_inherited_initialization_preparation_preserves_folds_but_excludes_observed_parents(self):
        import hashlib
        from research1.data_pipeline.ledger import Ledger
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset=root/'dataset';(dataset/'tasks').mkdir(parents=True)
            tasks=['task'+str(i) for i in range(120)]
            atomic_json(dataset/'manifest.json',{'tasks':[{'name':t,'split':'train'} for t in tasks]})
            for task in tasks:
                env=dataset/'tasks'/task/'environment';env.mkdir(parents=True)
                (env/'Dockerfile').write_text('FROM python:3.11-slim\n')
            source=root/'h0';source.mkdir()
            for module in ('agent_loop','observation','tools','context_mgmt','verification'):
                (source/module).mkdir();(source/module/'baseline.py').write_text('value=1\n')
            (root/'configs').mkdir()
            hashes={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
            atomic_json(root/'configs/h0.snapshot.v0.json',{'modules_root':str(source),'file_sha256':hashes})
            config=root/'configs/pilot.yaml';config.write_text('dataset_tb: '+dataset.as_posix()+'\nmodularrsi_root: '+root.as_posix()+'\n')
            (root/'src/harbor').mkdir(parents=True);(root/'src/harbor/runtime.py').write_text('runtime_version=1\n')
            first=root/'runs/first'
            prior=prepare(first,config,root=root,cached_images={'python:3.11-slim'},mode='initialization',max_parents=4)
            observed=prior['initialization']['task_order'];self.assertEqual(len(observed),4)
            ledger=Ledger(first)
            for index,task in enumerate(observed):
                path=first/'executions'/task;path.mkdir(parents=True)
                result={'task_name':task,'reward':None if index==0 else float(index%2)}
                atomic_json(path/'task_run.json',result)
                ledger.reserve_job(task,{'role':'proposal','task':task});ledger.start_job(task,'owner')
                ledger.fail_job(task,result) if index==0 else ledger.complete_job(task,result)
            atomic_json(first/'status.json',{'status':'INIT_BATCH_COMPLETE'})
            successor=prepare(root/'runs/next',config,root=root,cached_images={'python:3.11-slim'},
                              mode='initialization',max_parents=6,supersedes=first)
            self.assertEqual(successor['task_sets'],prior['task_sets'])
            self.assertFalse(set(successor['initialization']['task_order'])&set(observed))
            self.assertTrue(set(observed)<=set(successor['historical_exposure']))
            self.assertEqual(len(successor['initialization']['task_order']),6)
            self.assertIn(successor['initialization']['task_order'][4],prior['task_sets']['proposal_validation'])
            self.assertEqual(successor['budget'],{'regular':92,'retries':32,'total':124})

    def test_initialization_queue_excludes_known_limits_without_changing_eval_folds(self):
        sets={'proposal_train':['slow','fast','unknown','heavy'],'proposal_validation':['validation']}
        inventory={t:{'cached':True,'heavy_build':t=='heavy'} for t in ['slow','fast','unknown','heavy','validation']}
        ready={'slow':{'status':'known_budget_incompatible'},'fast':{'status':'completed_compatible'}}
        queue=initialization_queue(sets,inventory,ready,seed=7,max_parents=6)
        self.assertEqual(queue[0],'fast');self.assertNotIn('slow',queue);self.assertNotIn('heavy',queue)
        self.assertIn('unknown',queue);self.assertIn('validation',queue)

    def test_noise_does_not_reuse_known_timeout_as_runtime_certified(self):
        tasks=['task-'+str(i) for i in range(120)]
        sets=partition_tasks(tasks,set(tasks[:20]),seed=7)
        with self.assertRaisesRegex(ValueError,'eight runtime-certified'):runtime_debug_partition(sets,tasks,{},seed=7)
        eligible=list(set(tasks)-set(sets['proposal'])-set(sets['reward'])-set(sets['dev']))[:8]
        ready={t:{'noise_eligible':True} for t in eligible}
        changed=runtime_debug_partition(sets,tasks,ready,seed=7)
        for role in ('proposal','reward','dev'):self.assertEqual(changed[role],sets[role])
        self.assertEqual(set(changed['debug']),set(eligible))
    def test_stopped_campaign_hands_remaining_budget_once_without_refunding_calls(self):
        from research1.data_pipeline.ledger import Ledger,BudgetExceeded
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=root/'old';old.mkdir();modules=old/'modules';modules.mkdir();(modules/'x.py').write_text('x=1')
            registry={'packs':[]};atomic_json(old/'contracts.json',registry)
            body={'modules_root':str(modules),'modules_hash':tree_hash(modules),'registry_sha256':digest(registry),
                  'budget':{'regular':96,'retries':32,'total':128}}
            atomic_json(old/'protocol.json',dict(body,protocol_hash=digest(body)))
            atomic_json(old/'status.json',{'status':'INTERRUPTED_SUPERSEDED'})
            ledger=Ledger(old);ledger.reserve_job('attempted',{'role':'noise'});ledger.start_job('attempted','owner')
            with self.assertRaisesRegex(ValueError,'unfinished'):handover_budget(old,root/'new')
            ledger.fail_job('attempted',{'status':'INCOMPLETE','reward':None})
            result=handover_budget(old,root/'new')
            self.assertEqual(result['remaining'],{'regular':95,'retries':32,'total':127})
            self.assertEqual(handover_budget(old,root/'new'),result)
            with self.assertRaisesRegex(ValueError,'different successor'):handover_budget(old,root/'another')
            with self.assertRaisesRegex(BudgetExceeded,'transferred'):ledger.reserve_job('fork',{'role':'noise'})

    def test_disjoint_seeded_folds_exclude_historical_tasks(self):
        tasks=['task-'+str(i) for i in range(120)];exposed=set(tasks[:20])
        first=partition_tasks(tasks,exposed,seed=7)
        self.assertEqual(first,partition_tasks(list(reversed(tasks)),exposed,seed=7))
        self.assertNotEqual(first,partition_tasks(tasks,exposed,seed=8))
        self.assertEqual(len(first['proposal_train']),19)
        self.assertEqual(len(first['proposal_validation']),5)
        for role in ('proposal','reward','dev'):self.assertFalse(set(first[role])&exposed)
        bad=dict(first,dev=first['reward'])
        with self.assertRaisesRegex(ValueError,'Overlapping'):validate_partitions(bad,tasks)
        with self.assertRaisesRegex(ValueError,'72 unexposed'):partition_tasks(tasks,set(tasks[:60]),seed=7)

    def test_exposure_uses_actual_executions_not_only_scored_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'runs'/'old'/'failedtask';path.mkdir(parents=True)
            (path/'task_run.json').write_text(json.dumps({'task_name':'failedtask','reward':None}))
            trial=root/'runs/old/othertask/agent';trial.mkdir(parents=True)
            (trial/'trajectory.json').write_text('{}')
            self.assertEqual(exposed_tasks(root,['failedtask','othertask','unseen']),{'failedtask','othertask'})

    def test_protocol_and_parent_registry_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);modules=root/'modules';modules.mkdir();(modules/'x.py').write_text('x=1')
            registry={'packs':['frozen']};atomic_json(root/'contracts.json',registry)
            body={'modules_root':str(modules),'modules_hash':tree_hash(modules),'registry_sha256':digest(registry)}
            doc=dict(body,protocol_hash=digest(body));atomic_json(root/'protocol.json',doc)
            self.assertEqual(load_protocol(root),doc)
            atomic_json(root/'contracts.json',{'packs':['changed']})
            with self.assertRaisesRegex(ValueError,'registry'):load_protocol(root)
            atomic_json(root/'contracts.json',registry)
            (modules/'x.py').write_text('x=2')
            with self.assertRaisesRegex(ValueError,'H0'):load_protocol(root)

