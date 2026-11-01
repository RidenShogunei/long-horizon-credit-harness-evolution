from contextlib import redirect_stdout
import hashlib,io,json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from research1.cli import build_parser
from research1.data_pipeline.cli import command
from research1.data_pipeline.protocol import prepare,load_protocol

class DataEntryTests(unittest.TestCase):
    def test_explicit_initialization_tasks_are_forwarded_without_changing_folds(self):
        args=build_parser().parse_args(['data','prepare','--campaign','new','--mode','initialization',
            '--parents','2','--parent-task','train','--parent-task','validation'])
        with patch('research1.data_pipeline.cli.prepare',return_value={'protocol_hash':'hash','task_sets':{},'budget':{}}) as prepare_call,redirect_stdout(io.StringIO()):
            self.assertEqual(command(args),0)
        self.assertEqual(prepare_call.call_args.kwargs['parent_tasks'],['train','validation'])
        self.assertEqual(prepare_call.call_args.kwargs['max_parents'],2)

    def test_prepare_freezes_assets_and_never_requests_models(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);dataset=root/'dataset';(dataset/'tasks').mkdir(parents=True)
            tasks=['task'+str(i) for i in range(120)]
            (dataset/'manifest.json').write_text(json.dumps({'tasks':[{'name':t,'split':'train'} for t in tasks]}))
            for task in tasks:
                env=dataset/'tasks'/task/'environment';env.mkdir(parents=True)
                (env/'Dockerfile').write_text('FROM python:3.11-slim\n')
            source=root/'h0';source.mkdir()
            for name in ('agent_loop','observation','tools','context_mgmt','verification'):
                (source/name).mkdir();(source/name/'baseline.py').write_text('value = 1\n')
            (root/'configs').mkdir()
            hashes={p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')}
            (root/'configs/h0.snapshot.v0.json').write_text(json.dumps({'modules_root':str(source),'file_sha256':hashes}))
            config=root/'configs/pilot.yaml';config.write_text('dataset_tb: '+dataset.as_posix()+'\nmodularrsi_root: '+root.as_posix()+'\n')
            (root/'src/harbor').mkdir(parents=True);(root/'src/harbor/runtime.py').write_text('runtime_version=1\n')
            campaign=root/'runs/data'
            with patch('research1.distill._tongji_chat',side_effect=AssertionError('prepare made a model call')), \
                    patch('research1.data_pipeline.framework_context.build_framework_context', return_value=None):
                protocol=prepare(campaign,config,root=root,cached_images={'python:3.11-slim'},mode='initialization')
            self.assertEqual(len(protocol['task_asset_hashes']),80)
            self.assertEqual(protocol['budget'],{'regular':96,'retries':32,'total':128})
            self.assertEqual(load_protocol(campaign,verify_source=True),protocol)
            self.assertEqual(prepare(campaign,config,root=root,cached_images=set(),mode='initialization'),protocol)
            config.write_text(config.read_text()+'max_turns: 4\n')
            with self.assertRaisesRegex(ValueError,'config changed'):load_protocol(campaign,verify_source=True)

    def test_public_data_actions_and_child_group_budget_guard(self):
        args=build_parser().parse_args(['data','run','--campaign','test-child'])
        with patch('research1.data_pipeline.cli.load_protocol',return_value={'formal_group_of':'parent'}),patch('research1.data_pipeline.cli.Ledger'):
            with self.assertRaisesRegex(ValueError,'parent campaign'):command(args)
        for name in ('status','resume','export'):
            args=build_parser().parse_args(['data',name,'--campaign','example'])
            self.assertEqual(args.data_action,name)

    def test_student_profile_cannot_copy_private_extra_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'profile.json';path.write_text(json.dumps({'model_id':'student','api_key':'not-a-real-key'}))
            args=build_parser().parse_args(['data','export','--campaign',td,'--target-profile',str(path)])
            with patch('research1.data_pipeline.cli.load_protocol',return_value={}),patch('research1.data_pipeline.cli.Ledger'),patch('research1.data_pipeline.cli._export') as export:
                with self.assertRaisesRegex(ValueError,'Student profile'):command(args)
                export.assert_not_called()
