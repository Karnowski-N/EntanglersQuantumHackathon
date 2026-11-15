"""Checks for fair sweep accounting and preserving the established hourly algorithm."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import hourly_seed_shot_sweep as s

class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior=json.loads((Path(__file__).resolve().parent.parent/'data/test_reference_schedules.json').read_text())['prior_hourly']

    def test_fresh_runner_preserves_hourly_schedule_and_samples(self):
        for fleet in (5,10):
            prior=next(r for r in self.prior['runs'] if r['fleet']==fleet and r['seed']==7)
            row=s.run_trial(fleet,'reduced_qaoa',200,7,tuple(range(13,18)),self.prior['references'][str(fleet)]['cost'])
            self.assertTrue(row['valid']);np.testing.assert_array_equal(row['schedule'],prior['initial_schedule'])
            self.assertAlmostEqual(row['gap_pct'],prior['initial_gap_pct'],places=9)
            self.assertEqual(row['construction_milp_calls'],2)
            self.assertEqual(row['actual_quantum_circuits'],5)
            self.assertTrue(all(x['probability_entries']==2**x['qubits'] for x in row['circuit_trace']))

    def test_exact_hourly_is_independent_and_does_not_invoke_qaoa(self):
        for fleet,expected in ((5,0.),(10,.010097328447855602)):
            with patch.object(s.rq,'tune_qaoa',side_effect=AssertionError('Quantum in exact control')):
                row=s.run_trial(fleet,'exact_hourly',None,None,tuple(range(13,18)),self.prior['references'][str(fleet)]['cost'])
            self.assertTrue(row['valid']);self.assertAlmostEqual(row['gap_pct'],expected,places=8)
            self.assertEqual(row['construction_milp_calls'],2);self.assertEqual(row['actual_quantum_circuits'],0)

    def test_random_control_does_not_invoke_qaoa(self):
        with patch.object(s.rq,'tune_qaoa',side_effect=AssertionError('Quantum in random control')):
            row=s.run_trial(5,'random_locked',200,7,tuple(range(13,18)),327924.49)
        self.assertTrue(row['valid']);self.assertEqual(row['statevector_evaluations'],0)
        self.assertEqual(row['max_probability_entries'],0)

    def test_failed_trial_stays_failed_with_no_repair(self):
        with patch.object(s.rq,'schedule_hybrid',side_effect=RuntimeError('no sampled choice is feasible')) as run:
            row=s.run_trial(5,'reduced_qaoa',1,7,tuple(range(13,18)),327924.49)
        self.assertFalse(row['valid']);self.assertIsNone(row['cost']);self.assertIsNone(row['gap_pct'])
        self.assertIsNone(row['schedule']);run.assert_called_once()
        self.assertIn('no sampled choice',row['failure'])

    def test_failure_accounting_and_json_are_not_success_biased(self):
        common=dict(fleet=5,method='reduced_qaoa',shots=5,seed=7,total_seconds=.2,
                    statevector_evaluations=100,max_qubits=7,max_probability_entries=128,construction_milp_calls=1)
        a=common|dict(valid=True,gap_pct=.01)
        b=common|dict(seed=8,valid=False,gap_pct=None)
        result=dict(settings=dict(target_gap_pct=.1,fleets=[5],shots=[5],seeds=[7,8]),runs=[a,b])
        row=s.summarize(result)[0]
        self.assertEqual((row['attempted'],row['valid'],row['target_hits']),(2,1,1))
        self.assertEqual(row['valid_rate'],.5);self.assertEqual(row['target_rate'],.5)
        self.assertEqual(row['mean_valid_gap_pct'],.01)
        self.assertLess(row['valid_ci95_low'],.5);self.assertGreater(row['valid_ci95_high'],.5)
        json.dumps(s.clean_json(dict(a=np.nan,b=np.array([np.inf]),c=np.int64(7))),allow_nan=False)

if __name__=='__main__': unittest.main(verbosity=2)
