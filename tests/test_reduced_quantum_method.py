"""Behavioral checks for reduction, sampling provenance, and objective agreement."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import reduced_quantum_method as m

REFERENCE_DATA=Path(__file__).resolve().parent.parent/'data/test_reference_schedules.json'

class ReducedMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=json.loads(REFERENCE_DATA.read_text())['hourly_regression']
        cls.ns=m.make_namespace(5,strict=True)
        cls.baseline=np.array(cls.old['references']['gross5']['schedule'])

    def test_exact_restriction_and_bit_order(self):
        rng=np.random.default_rng(72)
        for n in (0,1,2,7):
            Q=np.triu(rng.normal(size=(n,n)))
            for fixed in ({},{i:i%2 for i in range(n)},{i:1 for i in range(0,n,2)}):
                q,c,free,values=m.substitute(Q,17.,fixed);b=m.bit_table(len(free));full=m.decode(np.arange(len(b)),free,values)
                np.testing.assert_allclose(np.einsum('bi,ij,bj->b',b,q,b)+c,np.einsum('bi,ij,bj->b',full,Q,full)+17,atol=1e-10)
                for i,v in fixed.items():self.assertTrue((full[:,i]==v).all())
                for j,i in enumerate(free):np.testing.assert_array_equal(full[:,i],b[:,j])

    def test_large_original_space_never_expanded(self):
        n=130;ns=m.make_namespace(5);tr=m.adapt(ns);Q=np.diag(np.linspace(-1,1,n))
        fixed={i:i%2 for i in range(n-3)}
        Q,off=ns['add_locks'](Q,0,[i for i,v in fixed.items() if v],[i for i,v in fixed.items() if not v],2)
        a,pr=ns['run_qaoa'](Q,off,maxiter=8)
        self.assertEqual(len(pr),8);self.assertEqual(tr[0]['state_table_shape'],[8,3]);self.assertEqual(tr[0]['state_table_bytes'],24)
        decoded=m.decode([0,7],tr[0]['free'],tr[0]['values'])
        self.assertEqual(decoded.shape,(2,130));self.assertTrue((decoded[0,-3:]==0).all());self.assertTrue((decoded[1,-3:]==1).all())
        m.audit_reduction(tr)
        big=m.make_namespace(100);self.assertIsNone(big['INDEX_BITS']);self.assertIsNone(big['PLANT_INDEX']);self.assertEqual(big['N'],100)

    def test_cap_precedes_exponential_allocation(self):
        ns=m.make_namespace(100);m.adapt(ns)
        with patch.object(m,'bit_table',side_effect=AssertionError('allocation attempted')):
            with self.assertRaisesRegex(ValueError,'requires 103 qubits'):ns['run_qaoa'](np.eye(103),0)
        with patch.object(m.np,'arange',side_effect=AssertionError('allocation attempted')):
            with self.assertRaises(ValueError):m.bit_table(19)

    def test_degenerate_spaces_honestly_flagged(self):
        for n in (0,1,3):
            a,p,w=m.tune_qaoa(np.zeros((n,n)),12)
            self.assertFalse(w['quantum_executed']);self.assertEqual(w['statevector_evaluations'],0)
            np.testing.assert_allclose(p,np.full(2**n,1/2**n))
        a,p,w=m.tune_qaoa(np.array([[-1.]]),0,maxiter=25)
        self.assertTrue(w['quantum_executed']);self.assertGreater(p[1],.9)

    def test_cost_phase_changes_distribution(self):
        Q=np.array([[-2.,.7],[0.,1.]])
        a,pr,w=m.tune_qaoa(Q,0,maxiter=40)
        qc,pars=m.qaoa_circuit(Q)
        no_phase=m.Statevector(qc.assign_parameters(dict(zip(pars,[0.,a[1]])))).probabilities()
        np.testing.assert_allclose(no_phase,np.full(4,.25),atol=1e-12)
        self.assertGreater(np.abs(pr-no_phase).sum(),.3)
        b=m.bit_table(2);energy=np.einsum('bi,ij,bj->b',b,Q,b)
        self.assertLess(pr@energy,no_phase@energy-.2)

    def test_batch_matches_official_checker(self):
        ns=self.ns;s=self.baseline
        for start,pair in ((0,(0,2)),(12,(1,4)),(20,(0,4))):
            scores=m.candidate_results(ns['plants'],ns['demand'],ns['reserve'],s,start,4,pair,np.arange(256))
            for i in range(256):
                candidate=s.copy();candidate[start:start+4,list(pair)]=m.bit_table(8)[i].reshape(4,2)
                scalar=m.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],candidate,verbose=False)
                self.assertEqual(scores[i]['valid'],scalar['valid']);self.assertAlmostEqual(scores[i]['total_cost'],scalar['total_cost'],places=6)

    def test_block_energy_and_prefix_fixing_preserve_valid_choices(self):
        ns=self.ns;s=self.baseline
        for start,pair in ((0,(0,2)),(6,(0,1)),(12,(1,4)),(20,(0,4))):
            Q,c=m.block_qubo(ns['plants'],ns['demand'],ns['reserve'],s,start,4,pair)
            fixed=m.boundary_fixed(ns,s,start,4,pair);states=m.bit_table(8)
            scores=m.candidate_results(ns['plants'],ns['demand'],ns['reserve'],s,start,4,pair,np.arange(256))
            energy=np.einsum('bi,ij,bj->b',states,Q,states)+c
            difference=[]
            for value,r in scores.items():
                if r['valid']:
                    for i,v in fixed.items():self.assertEqual(states[value,i],v)
                    difference.append(r['total_cost']-energy[value])
            self.assertTrue(difference);np.testing.assert_allclose(difference,difference[0],atol=1e-6)

    def test_no_milp_and_only_observed_changes_are_accepted(self):
        ns=self.ns;s=self.baseline.copy();pair=None
        for i in range(1,5):
            candidate=s.copy();candidate[:4,i]=1
            chk=m.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],candidate,verbose=False)
            if chk['valid'] and chk['total_cost']>self.old['references']['gross5']['cost']+1:
                s=candidate;pair=(0,i);break
        self.assertIsNotNone(pair)
        with patch.object(m,'combinations',return_value=[pair]):
            exact=m.refine_schedule(ns,s,sampler='exact',max_blocks=1)
        self.assertEqual(exact['accepted'],1);row=exact['trace'][0];winner=row['winner']
        original=s[:4,list(pair)].reshape(-1);free=row['free'];incumbent=sum(int(original[i])<<j for j,i in enumerate(free))
        def delta_sampler(chosen):
            def tune(Q,offset,**kw):
                p=np.zeros(1<<len(Q));p[chosen]=1
                return np.zeros(2),p,dict(statevector_evaluations=0,quantum_executed=False,logical_rzz=0)
            return tune
        with patch.dict(ns,{'scipy_milp':lambda *a,**k:self.fail('MILP in refinement')}),patch.object(m,'milp',side_effect=AssertionError('MILP called')),patch.object(m.ck,'solve_milp',side_effect=AssertionError('Checker MILP called')):
            for chosen,accept in ((incumbent,0),(winner,1)):
                with patch.object(m,'tune_qaoa',side_effect=delta_sampler(chosen)),patch.object(m,'combinations',return_value=[pair]):
                    r=m.refine_schedule(ns,s,max_blocks=1,shots=8)
                self.assertEqual(r['accepted'],accept);self.assertTrue(m.audit_refinement(ns,s,r))
                if accept:
                    self.assertEqual(r['trace'][0]['winning_count'],8)
                    corrupted=copy.deepcopy(r);corrupted['trace'][0]['measurement_counts']={incumbent:8}
                    with self.assertRaises(AssertionError):m.audit_refinement(ns,s,corrupted)

    def test_strict_hourly_sampler_does_not_repair_unobserved_failure(self):
        ns=m.make_namespace(5,strict=True);m.adapt(ns)
        def all_off(Q,offset,**kw):
            free=np.arange(len(Q));ns['_layout']=(free,np.zeros(len(Q),dtype=np.uint8))
            p=np.zeros(1<<len(Q));p[0]=1
            return np.zeros(2),p
        ns['run_qaoa']=all_off
        with self.assertRaisesRegex(RuntimeError,'no sampled choice is feasible'):
            ns['schedule_day_hybrid'](range(24),shots=1)

    def test_twelve_hourly_regressions_and_sample_provenance(self):
        count=0
        for old in self.old['runs']:
            if old['phase']!='hourly' or old['case'] not in ('gross5','gross10') or old['method']!='reduced_qaoa' or old['hours']!=list(range(13,18)):continue
            ns=m.make_namespace(int(old['case'][5:]),strict=True);trace=m.adapt(ns);stats=m.instrument_milp(ns)
            s,log=ns['schedule_day_hybrid'](range(13,18),seed=old['seed'])
            np.testing.assert_array_equal(s,old['schedule']);self.assertEqual(stats['calls'],2)
            for h in log:
                if h['method']=='qaoa':
                    self.assertGreater(h['selected_sample_count'],0);self.assertTrue(h['quantum_executed'])
                    self.assertEqual(sum(h['reduced_measurement_counts'].values()),200)
                    self.assertIn(h['pick'],[x['state'] for x in h['sampled_plant_counts']])
            for r in trace:self.assertEqual(r['probability_entries'],1<<r['qubits'])
            m.audit_reduction(trace);count+=1
        self.assertEqual(count,12)

if __name__=='__main__':unittest.main(verbosity=2)
