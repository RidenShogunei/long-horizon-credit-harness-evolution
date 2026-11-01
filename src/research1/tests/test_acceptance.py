import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from research1.acceptance import apply_replacements, compact_editor, run_acceptance

class AcceptanceTests(unittest.TestCase):
    def test_real_task_startup_error_is_recorded_without_admission(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'old'; source.mkdir()
            modules=root/'modules'; (modules/'tools').mkdir(parents=True)
            (modules/'tools/baseline.py').write_text('x=1\n')
            traj=root/'trajectory.json'
            traj.write_text(json.dumps({'steps':[{'step_id':1,'source':'agent','message':'visible','observation':{'results':[{'content':'WARNINGS: observed'}]}}]}))
            (source/'decision.json').write_text(json.dumps({'decision_row':{'modules_root':str(modules),'trajectory':str(traj),'task_name':'task'}}))
            (source/'plan.json').write_text(json.dumps({'action':'edit','target':'tools/baseline.py','intent':'Preserve real warnings'}))
            (source/'teacher_prompt.txt').write_text('ORIGINAL INPUT')
            (source/'teacher_raw.txt').write_text('ORIGINAL PLAN')
            ds=root/'dataset'; ds.mkdir(); (ds/'manifest.json').write_text(json.dumps({'tasks':[{'name':'task','split':'train'}]}))
            config=root/'pilot.yaml'; config.write_text('dataset_tb: '+ds.as_posix()+'\nmodularrsi_root: '+root.as_posix()+'\n')
            provider=SimpleNamespace(model='fixed',api_base='test',provider_id='test')
            with patch('research1.acceptance.compact_editor',return_value={'status':'applied','modules_root':str(modules),'child_version_id':'child'}), \
                 patch('research1.acceptance._default_verifier',return_value={'status':'PASS','covers_plan':True}), \
                 patch('research1.adapters.with_llm_provider_env',return_value=provider), \
                 patch('research1.adapters.apply_provider_to_environ'), \
                 patch('research1.adapters.run_harbor_task',side_effect=OSError('executor unavailable')):
                result=run_acceptance(source,root=root,config_path=config,max_attempts=1)
            self.assertEqual(result['status'],'INCOMPLETE')
            self.assertEqual(result['phase'],'finished')
            self.assertIsNone(result['task_run']['reward'])
            self.assertEqual(result['task_run']['execution_error']['type'],'OSError')
            self.assertFalse(result['exported_sft'])
            self.assertEqual((root/'datasets/research1/sft/verified_plans.jsonl').read_text(),'')
            self.assertTrue((root/'runs'/result['run_id']/'real_child_task/execution_error.json').is_file())

    def test_reasoning_field_cannot_be_used_as_editor_code(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'tools').mkdir(); (root/'tools/baseline.py').write_text('x = 1\n')
            directory=root/'attempt'; directory.mkdir()
            state={'modules_root':str(root),'checks':{}}
            plan={'intended_change':'fix','preserve_behaviors':['normal behavior']}
            with patch('research1.distill._tongji_chat',return_value=('{"edits":[]}',{'temperature':0.0,'raw_field':'reasoning','finish_reason':'stop'})):
                with self.assertRaisesRegex(ValueError,'complete final content'):
                    compact_editor(plan,state,directory)

    def test_exact_replacements_reject_ambiguous_or_overlapping_blocks(self):
        self.assertEqual(apply_replacements('x = 1\n',[{'old':'1','new':'2'}]),'x = 2\n')
        with self.assertRaises(ValueError): apply_replacements('x=1\ny=1\n',[{'old':'1','new':'2'}])
        with self.assertRaises(ValueError): apply_replacements('x = 123\n',[{'old':'123','new':'2'},{'old':'23','new':'3'}])
    def test_archived_teacher_input_and_valid_zero_survive_complete_loop(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/'old'; source.mkdir()
            modules=root/'modules'; (modules/'tools').mkdir(parents=True)
            (modules/'tools/baseline.py').write_text('def parse():\n    return 1\n')
            traj=root/'trajectory.json'; traj.write_text(json.dumps({'steps':[{'step_id':1,'source':'agent','message':'visible','observation':{'results':[{'content':'WARNINGS: observed'}]}}]}))
            original={'modules_root':str(modules),'trajectory':str(traj),'task_name':'task'}
            (source/'decision.json').write_text(json.dumps({'decision_row':original}))
            plan={'action':'edit','target':'tools/baseline.py','intent':'Repair observed behavior and preserve real warnings'}
            (source/'plan.json').write_text(json.dumps(plan))
            (source/'teacher_prompt.txt').write_text('ORIGINAL INPUT BEFORE RESULTS')
            (source/'teacher_raw.txt').write_text('ORIGINAL RAW PLAN')
            ds=root/'dataset'; ds.mkdir(); (ds/'manifest.json').write_text(json.dumps({'tasks':[{'name':'task','split':'train'}]}))
            config=root/'pilot.yaml'; config.write_text('dataset_tb: '+ds.as_posix()+'\nmodularrsi_root: '+root.as_posix()+'\n')
            def evaluate(**kw):
                kw['output_dir'].mkdir(parents=True,exist_ok=True)
                (kw['output_dir']/'result.json').write_text('{"reward":0}')
                return subprocess.CompletedProcess([],0,'','')
            provider=SimpleNamespace(model='fixed',api_base='test',provider_id='test')
            with patch('research1.acceptance.compact_editor',return_value={'status':'applied','modules_root':str(modules),'child_version_id':'child'}), \
                 patch('research1.acceptance._default_verifier',return_value={'status':'PASS','covers_plan':True}), \
                 patch('research1.adapters.with_llm_provider_env',return_value=provider), \
                 patch('research1.adapters.apply_provider_to_environ'), patch('research1.adapters.run_harbor_task',side_effect=evaluate):
                result=run_acceptance(source,root=root,config_path=config,max_attempts=1)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['task_run']['reward'],0)
            rows=[json.loads(s) for s in (root/'datasets/research1/sft/verified_plans.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['prompt'],'ORIGINAL INPUT BEFORE RESULTS')
            self.assertEqual(json.loads(rows[0]['completion']),plan)
            self.assertIsNone(rows[0]['long_horizon_label'])
