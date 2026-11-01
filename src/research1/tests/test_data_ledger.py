import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from research1.data_pipeline.ledger import Ledger,BudgetExceeded,ControllerBusy

class DataLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'protocol.json').write_text(json.dumps({'budget':{'regular':3,'retries':2,'total':4}}))
        self.ledger=Ledger(self.root)

    def test_atomic_global_and_retry_budgets_count_attempted_jobs(self):
        errors=[];created=[]
        def worker(i):
            try:
                ledger=Ledger(self.root)
                if ledger.reserve_job(str(i),{'task':'fixed','repeat':i}):created.append(i)
            except BudgetExceeded:pass
            except Exception as exc:errors.append(exc)
        workers=[threading.Thread(target=worker,args=(i,)) for i in range(12)]
        for worker in workers:worker.start()
        for worker in workers:worker.join()
        self.assertFalse(errors);self.assertEqual(len(created),3)
        self.assertTrue(self.ledger.reserve_job('retry',{'retry_of':str(created[0])},'retry'))
        with self.assertRaises(BudgetExceeded):self.ledger.reserve_job('retry2',{'retry_of':'x'},'retry')
        self.assertEqual(self.ledger.status()['budget']['remaining_regular'],0)
        self.assertEqual(self.ledger.status()['budget']['remaining_retry'],0)

    def test_resume_identity_completion_and_failure_are_immutable(self):
        self.ledger.reserve_job('j',{'model':'fixed'})
        self.assertFalse(self.ledger.reserve_job('j',{'model':'fixed'}))
        with self.assertRaises(ValueError):self.ledger.reserve_job('j',{'model':'changed'})
        self.ledger.start_job('j','original')
        with self.assertRaises(ValueError):self.ledger.start_job('j','new-owner')
        result={'reward':0,'status':'PASS'}
        self.ledger.complete_job('j',result);self.ledger.complete_job('j',result)
        with self.assertRaises(ValueError):self.ledger.complete_job('j',{'reward':1})
        self.ledger.reserve_job('f',{'task':'failed'})
        self.ledger.fail_job('f',{'reward':None,'status':'INCOMPLETE'})
        with self.assertRaises(ValueError):self.ledger.start_job('f','retry-as-new')
        self.assertEqual(self.ledger.get_job('j')['result'],result)

    def test_stage_reservations_are_enforced_before_launch(self):
        protocol={'budget':{'regular':3,'retries':2,'total':4},'stage_budget':{'proposal':1}}
        (self.root/'protocol.json').write_text(json.dumps(protocol))
        ledger=Ledger(self.root)
        ledger.reserve_job('p1',{'role':'proposal'})
        with self.assertRaisesRegex(BudgetExceeded,'stage'):ledger.reserve_job('p2',{'role':'proposal'})
        self.assertTrue(ledger.reserve_job('r1',{'role':'reward'}))

    def test_single_controller_lock_and_distinct_launcher_lock(self):
        with self.ledger.controller():
            with self.assertRaises(ControllerBusy):
                with Ledger(self.root).controller():pass
            with Ledger(self.root).controller('launcher'):pass
        with Ledger(self.root).controller():pass

    def test_revocation_cascades_and_label_revisions_remain_auditable(self):
        self.ledger.put_record('samples','s',{'sample_id':'s','plan_id':'p','qualification':{'status':'verified'}})
        self.ledger.put_record('values','v',{'candidate_id':'p','source_refs':['s'],'mean_u':.1})
        self.ledger.put_record('values','v',{'candidate_id':'p','source_refs':['s'],'mean_u':.2})
        self.ledger.put_record('comparisons','pair',{'candidate_ids':['p','other'],'source_refs':['v']})
        self.assertEqual(len(self.ledger.records('values')),1)
        changed=self.ledger.revoke('s','evidence rejected')
        self.assertEqual(len(changed),3)
        for kind in ('samples','values','comparisons'):self.assertTrue(self.ledger.records(kind)[0]['revoked'])
        with closing(sqlite3.connect(self.ledger.path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM record_history WHERE kind='values'").fetchone()[0],3)
        self.assertEqual(self.ledger.revoke('s','evidence rejected'),[])

