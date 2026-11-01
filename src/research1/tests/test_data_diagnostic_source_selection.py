"""Explicit diagnosis may narrow source reuse without inventing a new parent."""
import copy
import json
from pathlib import Path
import unittest

from research1.tests import test_data_parent_replan as fixtures
from research1.data_pipeline.protocol import prepare, load_protocol
from research1.data_pipeline.source_parents import validate_parent_sources


class DiagnosticSourceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ReplanTests('test_complete_source_set_is_new_decisions_not_new_independent_executions')
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)

    def test_declared_single_source_is_frozen_and_budget_inherited_once(self):
        f=self.fixture;task=f.order[1]
        rationale='Visible requirements and implementation show a concrete discrepancy for an interface feasibility diagnosis.'
        protocol=prepare(f.campaign,f.config,root=f.root,seed=7,cached_images={'fixture:1'},
            mode='initialization',supersedes=f.latest,replan_from=f.source,
            diagnostic_tasks=[task],diagnostic_rationale=rationale)
        rows=validate_parent_sources(f.campaign,protocol)
        self.assertEqual([row['task'] for row in rows],[task])
        self.assertEqual(rows[0]['root_lineage_id'],'original_root_1')
        self.assertEqual(rows[0]['split'],'validation')
        self.assertEqual(rows[0]['parent_result'],f.original_results[1])
        self.assertEqual(protocol['budget'],{'regular':90,'retries':32,'total':122})
        self.assertEqual(protocol['stage_budget']['proposal'],2)
        snapshot=json.loads((f.campaign/'source_parents.json').read_text())
        self.assertEqual(len(snapshot['all_scored_source_parents']),4)
        self.assertEqual(snapshot['new_independent_parent_increment'],0)
        self.assertEqual(protocol['grounded_edit_trial']['arms'],[
            {'id':'standard','reasoning_effort':'none'},{'id':'reasoned','reasoning_effort':'low'}])
        self.assertEqual(prepare(f.campaign,f.config,root=f.root,seed=7,cached_images=set(),
            mode='initialization',supersedes=f.latest,replan_from=f.source,
            diagnostic_tasks=[task],diagnostic_rationale=rationale),protocol)
        with self.assertRaisesRegex(ValueError,'selection differs'):
            prepare(f.campaign,f.config,root=f.root,seed=7,cached_images=set(),mode='initialization',
                supersedes=f.latest,replan_from=f.source,diagnostic_tasks=[f.order[0]],diagnostic_rationale=rationale)

    def test_undeclared_subset_and_non_source_tasks_fail_before_budget_transfer(self):
        f=self.fixture;before=load_protocol(f.latest)
        for tasks,rationale in (([f.order[0],f.order[1]],'A concrete predeclared source diagnosis that is long enough.'),
                                ([f.order[0]],'short')):
            with self.subTest(tasks=tasks),self.assertRaises(ValueError):
                prepare(f.campaign,f.config,root=f.root,seed=7,cached_images={'fixture:1'},
                    mode='initialization',supersedes=f.latest,replan_from=f.source,
                    diagnostic_tasks=tasks,diagnostic_rationale=rationale)
        self.assertEqual(load_protocol(f.latest),before)
