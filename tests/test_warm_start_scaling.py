"""Behavioral checks for the warm-start experiment; no provider calls."""
import unittest
import numpy as np
from qiskit.quantum_info import Statevector
import warm_start_scaling as w

class WarmStartTests(unittest.TestCase):
    def test_zero_phase_preserves_product(self):
        Q=np.array([[2.,3.,-1.],[0.,-4.,2.],[0.,0.,1.]])
        p=np.array([.1,.8,.3]);bits=w.m.rq.bit_table(3)
        expected=np.prod(np.where(bits,p,1-p),axis=1)
        for beta in (.1,1.1,2.5):
            actual=Statevector.from_instruction(w.circuit(Q,p,(0,beta),False)).probabilities()
            np.testing.assert_allclose(actual,expected,atol=1e-12)
    def test_uniform_reproduces_original_qaoa(self):
        Q=np.array([[2.,3.,-1.],[0.,-4.,2.],[0.,0.,1.]])
        old,params=w.m.rq.qaoa_circuit(Q)
        old=old.assign_parameters(dict(zip(params,(1.2,.4))))
        new=w.circuit(Q,[.5]*3,(1.2,.4),False)
        np.testing.assert_allclose(Statevector.from_instruction(old).probabilities(),Statevector.from_instruction(new).probabilities(),atol=1e-12)
    def test_cost_phase_changes_distribution(self):
        Q=np.array([[2.,3.],[0.,-4.]])
        a=Statevector.from_instruction(w.circuit(Q,[.2,.8],(1.2,.4),False)).probabilities()
        b=Statevector.from_instruction(w.circuit(Q,[.2,.8],(0,.4),False)).probabilities()
        self.assertGreater(np.linalg.norm(a-b),.01)
    def test_greedy_and_large_layout_do_not_call_milp_or_statevector(self):
        for fleet in w.FLEETS:
            ns=w.m.rq.make_namespace(fleet,strict=True)
            def forbidden(*a,**k):raise AssertionError('Unexpected solver/statevector call')
            ns['solve_milp_window']=ns['scipy_milp']=ns['Statevector']=forbidden
            history=[[int(p['on_at_start']) for p in ns['plants']]]*13
            t=w.layout(ns,13,history);x=t['seed_plant_state']
            on,off=ns['locks'](ns['plants'],history)
            self.assertTrue(ns['hour_ok'](ns['plants'],ns['demand'][13],ns['reserve'][13],x,on,off))
            self.assertTrue(np.all((t['probabilities_one']>0)&(t['probabilities_one']<1)))
            self.assertEqual(t['reduced_qubits'],fleet+3-len(t['fixed']))
    def test_forced_bits_and_large_sample_decoder(self):
        ns=w.m.rq.make_namespace(100);history=[[int(p['on_at_start']) for p in ns['plants']]]*13
        t=w.layout(ns,13,history);counts=w.counts_from_product(t,200,17)
        bits,weights=w.m.bit_rows(counts,t['reduced_qubits'])
        self.assertEqual(bits.shape[1],103);self.assertEqual(sum(weights),200)
        s=w.summarize(counts,t,ns);self.assertGreater(s['valid_shots'],0)
        self.assertIsNotNone(s['best_sampled_plant_state'])

if __name__=='__main__':unittest.main()
