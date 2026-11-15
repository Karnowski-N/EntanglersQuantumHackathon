"""Run the unchanged warm-start protocol using bounded local Aer MPS.

Fallback when the QPerfect cloud queue is unavailable. This does not submit
cloud/hardware jobs, and its results must be labeled separately from MIMIQ.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.metadata,time,resource,warnings
from qiskit import qasm3,transpile
from qiskit_aer import AerSimulator
import warm_start_scaling as w
from hourly_seed_shot_sweep import dump,clean_json

class LocalPilot:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.record=dict(device='local:qiskit_aer:matrix_product_state',backend_kind='local_ideal_mps',
            pricing=dict(perTask=0.,perShot=0.,perMinute=0.),created_utc=datetime.now(timezone.utc).isoformat(),
            jobs=[],complete=False,source_sha256={name:hashlib.sha256((w.m.ROOT/name).read_bytes()).hexdigest() for name in
            ['warm_start_scaling.py','warm_start_local_mps.py','mimiq_mps_pilot.py','mimiq_feasibility_training.py','reduced_quantum_method.py','rq_classical.py','check.py']},
            versions={name:importlib.metadata.version(name) for name in ('qiskit','qiskit-aer','numpy','scipy')},
            local_options=dict(max_parallel_threads=1,matrix_product_state_truncation_threshold=1e-12,
                mps_sample_measure_algorithm='mps_apply_measure',address_space_limit_bytes=2147483648,milp_threads=1),
            cloud_submission=False,physical_qpu=False,noise_model=False)
        self.save()
    def save(self):dump(self.output/'pilot.json',self.record)
    def submit(self,circuits,label,shots,bond,seed):
        circuits=circuits if isinstance(circuits,list) else [circuits]
        if len(self.record['jobs'])>=16:raise RuntimeError('Execution batch budget exhausted')
        start=time.perf_counter();info=dict(label=label,job_id=f'local-mps:{label}',shots_per_circuit=shots,
            circuits=len(circuits),options=dict(bonddim=bond,seed=seed),status='RUNNING',wall_seconds=0.,results=[])
        self.record['jobs'].append(info);self.save()
        sim=AerSimulator(method='matrix_product_state',max_parallel_threads=1,max_memory_mb=1024,
            matrix_product_state_max_bond_dimension=bond,matrix_product_state_truncation_threshold=1e-12,
            mps_sample_measure_algorithm='mps_apply_measure')
        for i,qc in enumerate(circuits):
            (self.output/f'{label}_{i}.qasm').write_text(qasm3.dumps(qc));t=time.perf_counter()
            # Explicit native gate list avoids Aer's conservative 63-qubit
            # transpiler target placeholder. No coupling map or hardware target.
            compiled=transpile(qc,basis_gates=['ry','rz','rzz','cx','measure'],optimization_level=0)
            result=sim.run(compiled,shots=shots,seed_simulator=seed).result()
            if not result.success:raise RuntimeError('Aer MPS execution failed')
            counts=result.get_counts();assert sum(counts.values())==shots
            metadata=clean_json(result.results[0].metadata)
            metadata.update(cost=0.,simulator='Qiskit Aer MPS',circuit_wall_seconds=time.perf_counter()-t,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            info['results'].append(dict(counts=counts,metadata=metadata,metadata_keys=list(metadata)))
            info['wall_seconds']=time.perf_counter()-start;self.save()
            print('LOCAL',label,i+1,'/',len(circuits),'qubits',qc.num_qubits,'seconds',round(metadata['circuit_wall_seconds'],3),flush=True)
        info['status']='COMPLETED';self.save();dump(self.output/f'{label}_results.json',info)
        return info['results'],info

def main(fleets=w.FLEETS):
    original=w.m.Pilot;original_namespace=w.m.rq.make_namespace
    def single_thread_namespace(*args,**kwargs):
        ns=original_namespace(*args,**kwargs);solver=ns['scipy_milp']
        def solve(*a,**kw):
            kw['options']=dict(kw.get('options',{}),threads=1)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='Unrecognized options detected.*threads')
                return solver(*a,**kw)
        ns['scipy_milp']=solve
        return ns
    try:
        w.m.Pilot=LocalPilot;w.m.rq.make_namespace=single_thread_namespace
        return w.main(fleets)
    finally:w.m.Pilot=original;w.m.rq.make_namespace=original_namespace

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fleets',type=int,nargs='+',default=list(w.FLEETS))
    main(tuple(parser.parse_args().fleets))
