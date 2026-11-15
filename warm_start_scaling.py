"""Greedy warm-start variant of reduced hourly QAOA, ideal MIMIQ MPS only.

The classical seed is NOT the optimal hourly MILP row. The unchanged hybrid
still uses the full-day MILP for its prefix and a conditional MILP for its tail.
Every quantum-hour selected state must occur in fresh final measurements.
No cloud work occurs on import. Run this file explicitly to launch the pilot.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse, hashlib, json, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import mimiq_mps_pilot as m
import mimiq_feasibility_training as ft
from hourly_seed_shot_sweep import dump

FLEETS=(5,10,25,50,100)
HOURS=tuple(range(13,18))
ANGLES=tuple((g,b) for g in (.4,1.2,2.6) for b in (.4,1.1))
TRAIN_SHOTS=256
FINAL_SHOTS=200
TOTAL_DRAWS=TRAIN_SHOTS*len(ANGLES)+FINAL_SHOTS
SEED=9100


def greedy_seed(ns,hour,history):
    """Capacity-filling merit-order heuristic. No MILP call or optimum input."""
    plants=ns['plants'];on,off=ns['locks'](plants,history)
    demand=float(ns['demand'][hour]);required=demand+ns['reserve'][hour]
    was=history[-1];x=np.zeros(ns['N'],dtype=np.uint8);x[on]=1
    fixed=set(on)|set(off)
    cap=np.array([p['pmax_mw'] for p in plants]);floor=np.array([p['pmin_mw'] for p in plants])
    if floor@x>demand:raise RuntimeError('Forced minimum generation exceeds demand')
    # Marginal energy plus startup/no-load cost per maximum MW. Not a relaxed
    # global optimum; this greedy adaptation has no inherited optimality guarantee.
    merit=lambda i:(plants[i]['marginal_cost_per_mwh']+
        (plants[i]['no_load_cost_per_h']+plants[i]['startup_cost']*(1-was[i]))/cap[i],i)
    for i in sorted((i for i in range(ns['N']) if i not in fixed),key=merit):
        if cap@x>=required:break
        if floor@x+floor[i]<=demand:x[i]=1
    if not ns['hour_ok'](plants,demand,ns['reserve'][hour],x,on,off):
        raise RuntimeError('Greedy seed failed; no MILP repair')
    return x


def layout(ns,hour,history):
    Q,c,small,constant,free,values,fixed=m.problem(ns,hour,history)
    seed=greedy_seed(ns,hour,history)
    # Choose the nearest representable slack for this feasible plant seed.
    # Keep the original three-bit slack formulation, including its limitations.
    residual=sum(p['pmax_mw']*v for p,v in zip(ns['plants'],seed))/ns['BLOCK_MW']-np.ceil((ns['demand'][hour]+ns['reserve'][hour])/ns['BLOCK_MW'])
    slack=int(np.clip(np.rint(residual),0,2**ns['N_SLACK']-1))
    full=np.r_[seed,[(slack>>k)&1 for k in range(ns['N_SLACK'])]].astype(np.uint8)
    free_plants=int(sum(free<ns['N']))
    epsilon=min(.15,1./max(1,free_plants))
    flip=np.where(free<ns['N'],epsilon,.15)
    probabilities=np.where(full[free]==1,1-flip,flip)
    return dict(hour=hour,history_before=[list(x) for x in history],Q=Q,offset=c,small_Q=small,
        small_offset=constant,free=free,values=values,fixed=fixed,reduced_qubits=len(small),
        original_qubits=len(Q),seed_plant_state=seed,seed_hour_cost=float(ns['hour_cost'](ns['plants'],ns['demand'][hour],seed,history[-1])),
        seed_slack=slack,slack_residual=float(residual),slack_clipped=bool(residual<0 or residual>7),
        probabilities_one=probabilities,plant_flip_probability=epsilon,free_plants=free_plants,
        seed_plant_probability=float((1-epsilon)**free_plants))


def circuit(Q,probabilities,angles,measure=True):
    """p=1 cost evolution and mixer aligned with the initial product state.

    At gamma=0 the mixer preserves that product distribution exactly. At
    probabilities=.5 this is the original H-initialized, Rx-mixer circuit.
    """
    Q=np.asarray(Q,float);p=np.asarray(probabilities,float)
    if Q.shape!=(len(p),len(p)) or np.any((p<=0)|(p>=1)):
        raise ValueError('Interior probabilities and matching QUBO required')
    gamma,beta=map(float,angles);theta=2*np.arcsin(np.sqrt(p));n=len(p)
    h=-np.diag(Q)/2.;J={}
    for i in range(n):
        for j in range(i+1,n):
            if Q[i,j]:J[i,j]=Q[i,j]/4.;h[i]-=Q[i,j]/4.;h[j]-=Q[i,j]/4.
    scale=max(np.max(abs(h),initial=0),max(map(abs,J.values()),default=0)) or 1.
    qc=QuantumCircuit(n)
    for i,t in enumerate(theta):qc.ry(float(t),i)
    if gamma:
        for i,v in enumerate(h):
            if v:qc.rz(2*gamma*v/scale,i)
        for (i,j),v in J.items():qc.rzz(2*gamma*v/scale,i,j)
    for i,t in enumerate(theta):
        qc.ry(-float(t),i);qc.rz(2*beta,i);qc.ry(float(t),i)
    if measure:qc.measure_all()
    return qc


def from_trace(trace,angles,measure=True):
    return circuit(trace['small_Q'],trace['probabilities_one'],angles,measure)


def counts_from_product(trace,shots,seed):
    rng=np.random.default_rng(seed)
    bits=(rng.random((shots,len(trace['free'])))<trace['probabilities_one']).astype(np.uint8)
    return Counter(''.join(map(str,row[::-1])) for row in bits)


def summarize(counts,trace,ns):
    return ft.score(counts,ns,trace['hour'],trace['history_before'],np.asarray(trace['small_Q']),
        trace['small_offset'],np.asarray(trace['free']),np.asarray(trace['values']),np.asarray(trace['Q']),trace['offset'])


def local_search(ns,hour,history,seed_state,budget,seed):
    """Classical improving random 1/2-bit proposals, seeded by the same greedy row."""
    on,off=ns['locks'](ns['plants'],history);fixed=set(on)|set(off)
    free=[i for i in range(ns['N']) if i not in fixed]
    x=np.array(seed_state,dtype=np.uint8);cost=float(ns['hour_cost'](ns['plants'],ns['demand'][hour],x,history[-1]))
    rng=np.random.default_rng(seed);accepted=0;valid=0
    for _ in range(budget):
        if not free:break
        y=x.copy();k=1 if len(free)==1 or rng.random()<.5 else 2
        y[rng.choice(free,size=k,replace=False)]^=1
        if ns['hour_ok'](ns['plants'],ns['demand'][hour],ns['reserve'][hour],y,on,off):
            valid+=1;c=float(ns['hour_cost'](ns['plants'],ns['demand'][hour],y,history[-1]))
            if c<cost-1e-9:x,cost=y,c;accepted+=1
    return dict(selected_state=x,hour_cost=cost,proposals=budget if free else 0,valid_proposals=valid,accepted=accepted)


def complete_tail(ns,history,baseline):
    if len(history)!=18:raise RuntimeError('Five hourly choices required before tail')
    on,off=ns['remaining_locks'](ns['plants'],history)
    _,tail,_=ns['solve_milp_window'](ns['plants'],ns['demand'][18:],ns['reserve'][18:],history[-1],on,off)
    schedule=np.vstack([history,tail])
    checked=m.rq.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],schedule,milp_cost=baseline,verbose=False)
    if not checked['valid']:raise AssertionError('Constructed schedule invalid')
    return dict(valid=True,completed_day=True,schedule=schedule,checked=checked)


def classical_run(fleet,plan,baseline,initial_seconds,method):
    ns=m.rq.make_namespace(fleet,strict=True);stats=m.rq.instrument_milp(ns)
    history=plan[:13].tolist();start=time.perf_counter()
    result=dict(method=method,valid=False,completed_day=False,hourly=[],initial_full_day_milp_wall_seconds=initial_seconds)
    try:
        for hour in HOURS:
            # Shared initializer; controls do not build or evaluate a statevector.
            trace=layout(ns,hour,history)
            row=dict(hour=hour,history_before=[list(x) for x in history],seed_state=trace['seed_plant_state'])
            if method=='greedy':pick=trace['seed_plant_state']
            elif method.startswith('biased_random'):
                shots=TOTAL_DRAWS if method.endswith('budget') else FINAL_SHOTS
                counts=counts_from_product(trace,shots,SEED+hour)
                scored=summarize(counts,trace,ns);row.update(summary=scored,counts=counts,draws=shots)
                pick=scored['best_sampled_plant_state']
                if pick is None:raise RuntimeError(f'hour {hour}: no feasible biased random sample')
            elif method=='local_search':
                local=local_search(ns,hour,history,trace['seed_plant_state'],TOTAL_DRAWS,SEED+hour)
                row['search']=local;pick=local['selected_state']
            else:raise ValueError(method)
            row['selected_state']=pick;result['hourly'].append(row);history.append(list(map(int,pick)))
        result.update(complete_tail(ns,history,baseline))
    except RuntimeError as e:result['failure']=str(e)
    result.update(post_prefix_wall_seconds=time.perf_counter()-start,milp_tail_stats=stats)
    result['accounted_wall_seconds']=initial_seconds+result['post_prefix_wall_seconds']
    return result


def small_exact_validation():
    ns,_,baseline,plan=m.initial_problem(5);trace=layout(ns,13,plan[:13].tolist())
    q=from_trace(trace,ANGLES[0],False);probs=Statevector.from_instruction(q).probabilities()
    q0=from_trace(trace,(0,ANGLES[0][1]),False);p0=Statevector.from_instruction(q0).probabilities()
    assert np.sum(abs(probs-p0))/2>1e-6
    return ns,trace,probs,dict(qubits=q.num_qubits,cost_phase_distribution_tv=float(np.sum(abs(probs-p0))/2))


def main(fleets=FLEETS):
    output=m.ROOT/'results/warm_start_scaling'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    pilot=m.Pilot(output);started=time.perf_counter()
    settings=dict(fleets=fleets,quantum_hours=HOURS,p=1,training_angles=ANGLES,training_shots_per_angle=TRAIN_SHOTS,
        final_shots=FINAL_SHOTS,training_bond=64,final_bond=128,seed=SEED,trajectories_per_fleet=1,
        selection='feasibility LCB then feasible true-cost lower-tail mean; fixed before held-out final samples',
        warm_seed='greedy marginal/startup/no-load merit order, no optimal hourly row',
        epsilon='min(0.15, 1 / number_of_free_plants); slack flip probability 0.15',
        exact_forced_reduction=True,group_partition=False,renewables=False,ramps=False,physical_qpu=False,noise_model=False,
        initial_full_day_milp=True,milp_tail=True,slack_qubits=3,final_failure_policy='stop, no repair or retry',
        controls=['greedy','biased_random_200','biased_random_budget','local_search'],classical_total_budget=TOTAL_DRAWS,
        cloud_batching='all fleet sizes in shared hour batches; wall time is not attributable to individual fleets')
    pilot.record.update(experiment='warm-start reduced hourly scaling',settings=settings,fleets={},validation={})
    pilot.record['source_sha256']['warm_start_scaling.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pilot.record['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for n in fleets for p in
        [m.ROOT/'data'/('plants'+('' if n==5 else str(n))+'.csv'),m.ROOT/'data'/('demand'+('' if n==5 else str(n))+'.csv')]}
    dump(output/'prespecified_policy.json',dict(settings=settings,source_sha256=pilot.record['source_sha256'],input_sha256=pilot.record['input_sha256']))
    pilot.save();print('OUTPUT',output,flush=True)
    # Small MPS cross-check of the new mixer and bit convention before large runs.
    ns,t,exact,validation=small_exact_validation()
    results,job=pilot.submit([from_trace(t,ANGLES[0]),from_trace(t,(0,ANGLES[0][1]))],'warm_validation5',4096,64,9001)
    exact_valid=[]
    for j,angles in enumerate([ANGLES[0],(0,ANGLES[0][1])]):
        exact=Statevector.from_instruction(from_trace(t,angles,False)).probabilities()
        bits=m.rq.bit_table(len(t['small_Q']));energies=np.einsum('bi,ij,bj->b',bits,t['small_Q'],bits)+t['small_offset']
        expected=float(exact@energies);se=float(np.sqrt(exact@((energies-expected)**2)/4096))
        observed=summarize(results[j]['counts'],t,ns);z=abs(observed['mean_qubo_energy']-expected)/max(se,1e-8)
        if z>5:raise AssertionError('Warm-start MPS energy mismatch beyond 5 standard errors')
        full=m.restore(bits,t['free'],t['values']);on,off=ns['locks'](ns['plants'],t['history_before'])
        ep=float(sum(p for x,p in zip(full,exact) if ns['hour_ok'](ns['plants'],ns['demand'][13],ns['reserve'][13],x[:ns['N']],on,off)))
        if abs(observed['valid_share']-ep)>5*np.sqrt(ep*(1-ep)/4096)+5/4096:raise AssertionError('Warm-start MPS feasibility mismatch')
        exact_valid.append(dict(angles=angles,expected_valid_probability=ep,observed=observed,energy_z_score=z))
    validation.update(results=exact_valid,job_id=job['job_id']);pilot.record['validation']=validation;pilot.save()
    contexts={}
    for fleet in fleets:
        start=time.perf_counter();ns,stats,baseline,plan=m.initial_problem(fleet);initial=time.perf_counter()-start
        history=plan[:13].tolist()
        row=dict(fleet=fleet,baseline_cost=baseline,baseline_schedule=plan,baseline_wall_seconds=initial,baseline_solver_seconds=stats['seconds'],
            baseline_statuses=stats['statuses'].copy(),controls={},quantum=dict(valid=False,completed_day=False,hourly=[]))
        pilot.record['fleets'][str(fleet)]=row;contexts[fleet]=(ns,stats,history)
        for method in settings['controls']:
            row['controls'][method]=classical_run(fleet,plan,baseline,initial,method)
        pilot.save();print('CLASSICAL',fleet,{k:(v.get('checked',{}).get('gap_pct'),v['valid']) for k,v in row['controls'].items()},flush=True)
    for hour in HOURS:
        active=[];circuits=[]
        for fleet,(ns,stats,history) in contexts.items():
            qr=pilot.record['fleets'][str(fleet)]['quantum']
            if 'failure' in qr:continue
            try:
                local_start=time.perf_counter();trace=layout(ns,hour,history)
                trace.update(training_angles=ANGLES,local_prepare_seconds=time.perf_counter()-local_start)
                built=[from_trace(trace,a) for a in ANGLES]
                trace['logical_rzz']=int(built[0].count_ops().get('rzz',0))
                qr['hourly'].append(trace);active.append((fleet,ns,history,trace));circuits.extend(built)
            except RuntimeError as e:qr['failure']=str(e)
        pilot.save()
        if not active:break
        training,job=pilot.submit(circuits,f'warm_h{hour}_training',TRAIN_SHOTS,64,SEED+100+hour)
        finals=[]
        for index,(fleet,ns,history,trace) in enumerate(active):
            scores=[summarize(r['counts'],trace,ns) for r in training[index*len(ANGLES):(index+1)*len(ANGLES)]]
            chosen=ft.choose(scores)
            trace.update(training_scores=scores,chosen_index=chosen,chosen_angles=ANGLES[chosen],training_job_id=job['job_id'],training_result_start=index*len(ANGLES))
            finals.append(from_trace(trace,ANGLES[chosen]))
        pilot.save();dump(output/f'h{hour}_choices_before_final.json',[dict(fleet=f,angles=t['chosen_angles']) for f,ns,h,t in active])
        heldout,job=pilot.submit(finals,f'warm_h{hour}_final',FINAL_SHOTS,128,SEED+hour)
        for index,(fleet,ns,history,trace) in enumerate(active):
            observed=summarize(heldout[index]['counts'],trace,ns)
            trace.update(final=observed,final_job_id=job['job_id'],final_result_index=index,final_metadata=heldout[index]['metadata'])
            controls={}
            for shots in (FINAL_SHOTS,TOTAL_DRAWS):
                counts=counts_from_product(trace,shots,SEED+hour)
                controls[f'biased_random_{shots}']=dict(counts=counts,summary=summarize(counts,trace,ns))
            controls['local_search']=local_search(ns,hour,history,trace['seed_plant_state'],TOTAL_DRAWS,SEED+hour)
            trace['same_history_controls']=controls
            qr=pilot.record['fleets'][str(fleet)]['quantum']
            if observed['best_sampled_plant_state'] is None:qr['failure']=f'hour {hour}: zero feasible final samples; no repair or retry'
            else:history.append(list(observed['best_sampled_plant_state']))
            pilot.save();print('QUANTUM',fleet,hour,'qubits',trace['reduced_qubits'],'valid',observed['valid_shots'],'cost',observed['best_sampled_hour_cost'],'seed cost',trace['seed_hour_cost'],flush=True)
    for fleet,(ns,stats,history) in contexts.items():
        row=pilot.record['fleets'][str(fleet)];qr=row['quantum']
        start=time.perf_counter()
        if len(history)==18:
            try:qr.update(complete_tail(ns,history,row['baseline_cost']))
            except RuntimeError as e:qr['failure']=str(e)
        qr.update(tail_wall_seconds=time.perf_counter()-start,milp_stats=stats)
        pilot.save();print('FULL DAY',fleet,qr['valid'],qr.get('checked',{}).get('gap_pct'),flush=True)
    # One preregistered larger-bond check on the largest fleet's latest sampled circuit.
    largest=max(fleets);last=pilot.record['fleets'][str(largest)]['quantum']['hourly'][-1]
    if 'final' in last:
        result,job=pilot.submit(from_trace(last,last['chosen_angles']),'largest_bond256',FINAL_SHOTS,256,SEED+last['hour'])
        ns=contexts[largest][0];prior=next(j for j in pilot.record['jobs'] if j['job_id']==last['final_job_id'])['results'][last['final_result_index']]['counts']
        pilot.record['bond_check']=dict(fleet=largest,hour=last['hour'],job_id=job['job_id'],summary=summarize(result[0]['counts'],last,ns),
            counts_identical_to_bond128=result[0]['counts']==prior,metadata=result[0]['metadata'])
    pilot.record.update(complete=True,total_wall_seconds=time.perf_counter()-started)
    pilot.save();print('COMPLETE',output,flush=True)
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fleets',type=int,nargs='+',default=list(FLEETS))
    main(tuple(parser.parse_args().fleets))
