"""Bounded, zero-credit QPerfect MIMIQ MPS pilot; no physical QPU/noise model.

Unlike the exact-statevector sweep, large circuits use a fixed finite-sample
parameter grid. No full-space bit/probability arrays are made for large circuits.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, time, sys
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qbraid.runtime import QbraidProvider
import reduced_quantum_method as rq
from hourly_seed_shot_sweep import dump, clean_json

DEVICE='qperfect:qperfect:sim:mimiq-mps'
ROOT=rq.HERE
GRID=[(.4,.4),(.8,.4),(1.2,.8),(1.8,1.2),(2.6,.4),(1.2,2.3),(2.6,2.3)]


def measured(Q,angles):
    qc,params=rq.qaoa_circuit(Q,p=1)
    qc=qc.assign_parameters(dict(zip(params,angles)));qc.measure_all()
    return qc


def bit_rows(counts,qubits):
    states=[];weights=[]
    for key,count in counts.items():
        key=key.replace(' ','')
        if len(key)!=qubits or set(key)-set('01'):raise ValueError('Unexpected measurement bitstring')
        states.append([int(v) for v in key[::-1]]);weights.append(int(count))
    return np.asarray(states,dtype=np.uint8).reshape(-1,qubits),np.asarray(weights,dtype=int)


def restore(bits,free,values):
    rows=np.broadcast_to(values,(len(bits),len(values))).copy()
    rows[:,free]=bits
    return rows


def sample_summary(counts,Q,offset,free,values,ns,hour,history,full_Q=None,full_offset=None):
    bits,weights=bit_rows(counts,len(Q));full=restore(bits,free,values)
    energies=np.einsum('bi,ij,bj->b',bits,Q,bits)+offset
    if full_Q is not None:
        np.testing.assert_allclose(energies,np.einsum('bi,ij,bj->b',full,full_Q,full)+full_offset,atol=1e-5,rtol=1e-10)
    shots=int(weights.sum());mean=float(np.average(energies,weights=weights))
    variance=float(np.average((energies-mean)**2,weights=weights))
    on,off=ns['locks'](ns['plants'],history)
    was=history[-1] if len(history) else [int(p['on_at_start']) for p in ns['plants']]
    plant_counts=Counter()
    for row,n in zip(full[:,:ns['N']],weights):plant_counts[tuple(map(int,row))]+=int(n)
    valid=[x for x in plant_counts if ns['hour_ok'](ns['plants'],ns['demand'][hour],ns['reserve'][hour],x,on,off)]
    valid.sort(key=lambda x:sum(int(b)<<i for i,b in enumerate(x)))
    best=min(valid,key=lambda x:ns['hour_cost'](ns['plants'],ns['demand'][hour],x,was)) if valid else None
    return dict(shots=shots,unique_reduced_patterns=len(counts),unique_plant_patterns=len(plant_counts),
        mean_qubo_energy=mean,energy_standard_error=float(np.sqrt(variance/max(shots-1,1))),
        valid_shots=sum(plant_counts[x] for x in valid),valid_share=sum(plant_counts[x] for x in valid)/shots,
        best_sampled_plant_state=best,selected_sample_count=None if best is None else plant_counts[best],
        best_sampled_hour_cost=None if best is None else float(ns['hour_cost'](ns['plants'],ns['demand'][hour],best,was)))


class Pilot:
    def __init__(self,output=None):
        self.output=Path(output) if output else ROOT/'results/mimiq_mps_pilot'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.output.mkdir(parents=True,exist_ok=False)
        self.device=QbraidProvider().get_device(DEVICE)
        price=self.device.profile.pricing
        rates={k:float(getattr(price,k)) for k in ('perTask','perShot','perMinute')}
        if any(rates.values()):raise RuntimeError('Pilot only authorizes zero-credit simulator jobs; pricing is no longer zero')
        self.record=dict(device=DEVICE,pricing=rates,created_utc=datetime.now(timezone.utc).isoformat(),
            settings=dict(training_shots=512,final_shots=200,training_bond=64,final_bond=128,
                convergence_bonds=[64,128,256],timelimit_minutes_per_job=2,max_jobs=16,
                model='gross demand; original reserve; no ramps; original 3 slack qubits',
                quantum_hours=list(range(13,18)),training='fixed 8-angle sampled-energy grid on MIMIQ; no exact large-state expectation',
                physical_qpu=False,noise_model=False,group_partition=False),jobs=[],complete=False,
            source_sha256={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in ['mimiq_mps_pilot.py','reduced_quantum_method.py','rq_classical.py','check.py']})
        self.save()
    def save(self):dump(self.output/'pilot.json',self.record)
    def submit(self,circuits,label,shots,bond,seed):
        if len(self.record['jobs'])>=16:raise RuntimeError('Pilot job budget exhausted')
        price=self.device.profile.pricing
        if any(float(getattr(price,k)) for k in ('perTask','perShot','perMinute')):raise RuntimeError('Nonzero price')
        circuits=circuits if isinstance(circuits,list) else [circuits]
        for i,qc in enumerate(circuits):(self.output/f'{label}_{i}.qasm').write_text(qasm3.dumps(qc))
        options=dict(bonddim=bond,seed=seed,timelimit=2)
        start=time.perf_counter()
        job=self.device.run(circuits if len(circuits)>1 else circuits[0],shots=shots,
            name=f'RQ hourly MPS pilot: {label}',runtime_options=options,as_batch=len(circuits)>1)
        info=dict(label=label,job_id=job.id,shots_per_circuit=shots,circuits=len(circuits),
                  options=options,status='SUBMITTED',submit_seconds=time.perf_counter()-start)
        self.record['jobs'].append(info);self.save()
        print('SUBMITTED',label,job.id,'circuits',len(circuits),'qubits',[q.num_qubits for q in circuits],flush=True)
        waiting=time.perf_counter();last=None
        while True:
            status=job.status();name=getattr(status,'name',str(status))
            if name!=last:print('STATUS',label,name,flush=True);last=name
            info['status']=name;info['wall_seconds']=time.perf_counter()-start;self.save()
            if name in ('COMPLETED','FAILED','CANCELLED','CANCELED'):break
            if time.perf_counter()-waiting>360:
                raise TimeoutError('Cloud pilot is still queued/running after 6 minutes; job ID retained for retrieval')
            time.sleep(5)
        result=job.result(timeout=30)
        if not result.success:
            info['error']='Simulator job did not complete successfully';self.save()
            raise RuntimeError(f'MIMIQ job failed: {job.id}')
        results=result.results if hasattr(result,'results') else [result]
        if len(results)!=len(circuits):raise AssertionError('Batch result count mismatch')
        safe=[]
        for item in results:
            counts=item.data.get_counts();assert sum(counts.values())==shots
            details=item.details or {}
            selected={k:details[k] for k in ('fidelity','cost','status','time_stamps','execution_time','executionTime','timings','simulator') if k in details}
            # Other numeric simulator fields are safe to retain; no account/config records are read.
            safe.append(dict(counts=counts,metadata=json.loads(json.dumps(clean_json(selected),default=str)),metadata_keys=list(details)))
        info.update(status='COMPLETED',wall_seconds=time.perf_counter()-start,results=safe)
        self.save();dump(self.output/f'{label}_results.json',info)
        print('DONE',label,'wall',round(info['wall_seconds'],2),'metadata',safe[0]['metadata_keys'],flush=True)
        return safe,info


def initial_problem(fleet):
    ns=rq.make_namespace(fleet,strict=True);stats=rq.instrument_milp(ns)
    cost,plan,_=ns['solve_milp_window'](ns['plants'],ns['demand'],ns['reserve'],
        [int(p['on_at_start']) for p in ns['plants']],np.zeros(fleet,int),np.zeros(fleet,int))
    chk=rq.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],plan,verbose=False)
    assert chk['valid'] and abs(chk['total_cost']-cost)<1e-4
    return ns,stats,float(cost),plan


def problem(ns,hour,history):
    on,off=ns['locks'](ns['plants'],history);was=history[-1]
    Q,c=ns['build_qubo'](ns['plants'],ns['demand'][hour],ns['reserve'][hour],was)
    Q,c=ns['add_locks'](Q,c,on,off,2*ns['plant_hour_costs'](ns['plants'],was).max())
    fixed={i:1 for i in on}|{i:0 for i in off}
    small,constant,free,values=rq.substitute(Q,c,fixed)
    return Q,c,small,constant,free,values,fixed


def run():
    pilot=Pilot();start=time.perf_counter()
    ns,stats,baseline,plan=initial_problem(10)
    history=plan[:13].tolist();Q,c,small,constant,free,values,fixed=problem(ns,13,history)
    angles,probabilities,work=rq.tune_qaoa(small,constant)
    basis=QuantumCircuit(5);basis.x(0);basis.x(3);basis.measure_all()
    small_results,job=pilot.submit([basis,measured(small,angles)],'validation10',4096,64,7101)
    if small_results[0]['counts']!={'01001':4096}:raise AssertionError('Provider bit ordering differs from the decoder')
    check=sample_summary(small_results[1]['counts'],small,constant,free,values,ns,13,history,Q,c)
    bits=rq.bit_table(len(small));energies=np.einsum('bi,ij,bj->b',bits,small,bits)+constant
    exact_mean=float(probabilities@energies);variance=float(probabilities@((energies-exact_mean)**2))
    stderr=float(np.sqrt(variance/4096));z=abs(check['mean_qubo_energy']-exact_mean)/max(stderr,1e-9)
    full=rq.decode(np.arange(len(bits)),free,values,columns=ns['N']);on,off=ns['locks'](ns['plants'],history)
    exact_valid=float(sum(p for p,x in zip(probabilities,full) if ns['hour_ok'](ns['plants'],ns['demand'][13],ns['reserve'][13],x,on,off)))
    pilot.record['validation10']=dict(qubits=len(small),angles=angles,local_training=work,baseline_cost=baseline,
        bit_order_passed=True,exact_mean_energy=exact_mean,exact_valid_probability=exact_valid,
        energy_sampling_standard_error=stderr,energy_z_score=z,observed=check,reported_metadata=small_results[1]['metadata'])
    pilot.save()
    if z>5:raise AssertionError('Small-case MPS mean energy disagrees with exact statevector beyond 5 sampling SE')
    if abs(check['valid_share']-exact_valid)>5*np.sqrt(exact_valid*(1-exact_valid)/4096)+5/4096:
        raise AssertionError('Small-case valid probability disagrees beyond sampling uncertainty')
    print('VALIDATED small case:',len(small),'qubits; energy z-score',round(z,2),'valid sample share',check['valid_share'],flush=True)

    ns,stats,baseline,plan=initial_problem(100);history=plan[:13].tolist()
    large=dict(fleet=100,reference_cost=baseline,reference_schedule=plan,initial_milp_seconds=stats['seconds'],
               hourly=[],completed_day=False,valid=False,training_shots=0,final_shots=0)
    pilot.record['hundred_plant']=large;pilot.save()
    previous=list(map(float,angles))
    for hour in range(13,18):
        Q,c,small,constant,free,values,fixed=problem(ns,hour,history)
        grid=[tuple(previous)]+GRID
        trace=dict(hour=hour,reduced_qubits=len(small),original_qubits=len(Q),forced=len(fixed),fixed=fixed,free=free,values=values,
            Q=Q,offset=c,small_Q=small,small_offset=constant,history_before=history.copy(),angles_tested=grid,
            logical_rzz=int(measured(small,grid[0]).count_ops().get('rzz',0)))
        large['hourly'].append(trace);pilot.save()
        training,job=pilot.submit([measured(small,a) for a in grid],f'h{hour}_training',512,64,7200+hour)
        summaries=[sample_summary(r['counts'],small,constant,free,values,ns,hour,history,Q,c) for r in training]
        winner=min(range(len(grid)),key=lambda i:summaries[i]['mean_qubo_energy'])
        chosen=grid[winner];previous=list(chosen)
        trace.update(training_summary=summaries,chosen_index=winner,chosen_angles=chosen,training_job_id=job['job_id'])
        large['training_shots']+=512*len(grid);pilot.save()
        final,job=pilot.submit(measured(small,chosen),f'h{hour}_final',200,128,7300+hour)
        observed=sample_summary(final[0]['counts'],small,constant,free,values,ns,hour,history,Q,c)
        trace.update(final_summary=observed,final_metadata=final[0]['metadata'],final_job_id=job['job_id'])
        large['final_shots']+=200
        # A matched random control on the identical history; no quantum/MILP optimization.
        rng=np.random.default_rng(7300+hour);random_bits=rng.integers(0,2,size=(200,len(small)),dtype=np.uint8)
        random_counts=Counter(''.join(map(str,x[::-1])) for x in random_bits)
        trace['random_same_history']=sample_summary(random_counts,small,constant,free,values,ns,hour,history,Q,c)
        pilot.save()
        if hour==13:
            trace['bond_checks']={128:dict(summary=observed,metadata=final[0]['metadata'],job_id=job['job_id'])}
            for bond in (64,256):
                samples,bond_job=pilot.submit(measured(small,chosen),f'h13_bond{bond}',200,bond,7300+hour)
                trace['bond_checks'][bond]=dict(summary=sample_summary(samples[0]['counts'],small,constant,free,values,ns,hour,history,Q,c),
                     metadata=samples[0]['metadata'],job_id=bond_job['job_id'])
                pilot.save()
        if observed['best_sampled_plant_state'] is None:
            large['failure']=f'hour {hour}: zero feasible choices among the fresh 200 final samples; no repair or training-sample substitution'
            print('FAILED SCHEDULE',large['failure'],flush=True);break
        history.append(list(observed['best_sampled_plant_state']))
        print('SELECTED',hour,'valid share',observed['valid_share'],'sampled hour cost',observed['best_sampled_hour_cost'],flush=True)
    if len(history)==18:
        on,off=ns['remaining_locks'](ns['plants'],history)
        try:
            _,tail,_=ns['solve_milp_window'](ns['plants'],ns['demand'][18:],ns['reserve'][18:],history[-1],on,off)
            schedule=np.vstack([history,tail]);checked=rq.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],schedule,milp_cost=baseline,verbose=False)
            large.update(completed_day=True,valid=checked['valid'],schedule=schedule,checked=checked)
        except RuntimeError as exc:large['failure']=str(exc)
    large['milp_stats']=stats
    pilot.record.update(complete=True,total_wall_seconds=time.perf_counter()-start)
    pilot.save();print('PILOT COMPLETE',pilot.output,flush=True)

if __name__=='__main__':run()
