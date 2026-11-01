import json
from pathlib import Path
import tempfile
import unittest
from research1.data_pipeline.legacy import quarantine_directed

class DataLegacyTests(unittest.TestCase):
    def test_quarantine_preserves_raw_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data=root/'datasets/research1';(data/'sft').mkdir(parents=True);(data/'raw').mkdir()
            raw=data/'raw/attempts.jsonl';raw.write_bytes(b'{"original":"immutable"}\n')
            rows=[{'sample_id':'repair','source_kind':'archived_teacher_plan_reimplementation','use':'directed_behavior_repair_init'},
                  {'sample_id':'old-keep','use':'plan_supervised_init'}]
            export=data/'sft/verified_plans.jsonl';export.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            first=quarantine_directed(root)
            self.assertEqual(first['quarantined'],['repair'])
            self.assertEqual([json.loads(s)['sample_id'] for s in export.read_text().splitlines()],['old-keep'])
            self.assertEqual(quarantine_directed(root)['quarantined'],[])
            aux=[json.loads(s) for s in (data/'auxiliary/directed_repairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(aux),1);self.assertFalse(aux[0]['formal_eligible'])
            self.assertEqual(raw.read_bytes(),b'{"original":"immutable"}\n')
