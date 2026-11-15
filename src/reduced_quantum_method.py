"""Allocation-safe reduced hourly QAOA and sampled multi-hour refinement.
Only reduced state spaces are enumerated. No cloud/provider calls.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib
import importlib.util
import json
import time
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from rq_candidate_checks import candidate_results

SOURCE=Path(__file__).resolve().parent
HERE=SOURCE.parent
spec=importlib.util.spec_from_file_location('_rq_local_checker',SOURCE/'check.py')
ck=importlib.util.module_from_spec(spec);spec.loader.exec_module(ck)
DEFAULT_MAX_QUBITS=18


def bit_table(qubits,max_qubits=DEFAULT_MAX_QUBITS):
    if qubits<0 or qubits>max_qubits or qubits>=63:
        raise ValueError(f'Reduced circuit requires {qubits} qubits; local allocation limit is {max_qubits}. No state table was allocated. Use stronger valid reduction, smaller blocks, or explicitly choose an appropriate simulation limit.')
    return ((np.arange(1<<qubits,dtype=np.uint64)[:,None]>>np.arange(qubits,dtype=np.uint64))&1).astype(np.uint8)


def substitute(Q,offset,fixed):
    Q=np.triu(np.asarray(Q,dtype=float))
    if Q.ndim!=2 or Q.shape[0]!=Q.shape[1]:raise ValueError('Q must be square')
    values=np.zeros(len(Q),dtype=np.uint8)
    for i,v in fixed.items():
        if not isinstance(i,(int,np.integer)) or not 0<=i<len(Q) or v not in (0,1):raise ValueError('Invalid fixed assignment')
        values[i]=v
    free=np.array([i for i in range(len(Q)) if i not in fixed],dtype=int)
    reduced=Q[np.ix_(free,free)].copy()
    reduced[np.diag_indices(len(free))]+=((Q+Q.T)@values)[free]
    return reduced,float(offset+values@Q@values),free,values


def decode(indices,free,values,columns=None):
    """Decode only sampled rows; never construct integer indices into a full statevector."""
    ids=np.asarray(indices,dtype=np.uint64)
    if np.any(ids >= (1<<len(free))):raise ValueError('Sample outside reduced space')
    n=len(values) if columns is None else columns
    out=np.broadcast_to(values[:n],(len(ids),n)).copy()
    for j,i in enumerate(free):
        if i<n:out[:,i]=((ids>>np.uint64(j))&1).astype(np.uint8)
    return out


def qaoa_circuit(Q,p=1):
    n=len(Q);h=-np.diag(Q).copy()/2;J={}
    for j in range(n):
        for k in range(j+1,n):
            if Q[j,k]!=0:
                J[j,k]=Q[j,k]/4;h[j]-=Q[j,k]/4;h[k]-=Q[j,k]/4
    scale=max(np.max(np.abs(h),initial=0),max((abs(v) for v in J.values()),default=0)) or 1.
    gamma,beta=ParameterVector('gamma',p),ParameterVector('beta',p)
    qc=QuantumCircuit(n);qc.h(range(n))
    for layer in range(p):
        for j,hj in enumerate(h):
            if hj:qc.rz(2*gamma[layer]*hj/scale,j)
        for (j,k),v in J.items():qc.rzz(2*gamma[layer]*v/scale,j,k)
        qc.rx(2*beta[layer],range(n))
    return qc,list(gamma)+list(beta)


def tune_qaoa(Q,offset,p=1,start=None,maxiter=60,grid=(15,8),max_qubits=DEFAULT_MAX_QUBITS,statevector=None):
    """Exact-statevector angle training; finite measurement draws happen separately."""
    statevector=Statevector if statevector is None else statevector
    bits=bit_table(len(Q),max_qubits)
    energy=np.einsum('bi,ij,bj->b',bits,np.triu(Q),bits)+offset
    spread=np.ptp(energy)
    work=dict(statevector_evaluations=0,energy_states=len(bits),state_table_shape=list(bits.shape),state_table_bytes=bits.nbytes,probability_entries=len(bits),quantum_executed=False,logical_rzz=0)
    if not len(Q) or spread<1e-12:
        return np.zeros(2*p),np.full(len(bits),1/len(bits)),work
    energy=(energy-energy.min())/spread
    qc,params=qaoa_circuit(Q,p)
    work['logical_rzz']=int(qc.count_ops().get('rzz',0));work['quantum_executed']=True
    def probabilities(angles):
        work['statevector_evaluations']+=1
        return statevector(qc.assign_parameters(dict(zip(params,angles)))).probabilities()
    def objective(angles):return float(probabilities(angles)@energy)
    if start is None or len(start)!=2*p:
        points=[(g,b) for g in np.linspace(.1,3.,grid[0]) for b in np.linspace(.1,1.5,grid[1])]
        g,b=min(points,key=lambda gb:objective([gb[0]]*p+[gb[1]]*p));start=[g]*p+[b]*p
    fit=minimize(objective,start,method='COBYLA',options={'maxiter':maxiter,'rhobeg':.2})
    return fit.x,probabilities(fit.x),work


def make_namespace(fleet_count=10,n_slack=3,max_qubits=DEFAULT_MAX_QUBITS,strict=False):
    """Polynomial-size setup. INDEX_BITS/PLANT_INDEX are deliberately absent tables."""
    suffix='' if fleet_count==5 else str(fleet_count)
    plants=ck.load_plants(HERE/'data'/f'plants{suffix}.csv')
    demand,reserve=ck.load_demand(HERE/'data'/f'demand{suffix}.csv')
    ns=dict(np=np,time=time,Bounds=Bounds,LinearConstraint=LinearConstraint,scipy_milp=milp,lil_matrix=lil_matrix,
            plants=plants,demand=demand,reserve=reserve,N=len(plants),N_SLACK=n_slack,N_QUBITS=len(plants)+n_slack,
            BLOCK_MW=50,hours=np.arange(len(demand)),names=[p['plant_id'] for p in plants],INDEX_BITS=None,PLANT_INDEX=None,
            Statevector=Statevector,check_schedule=ck.check_schedule,check_hour=ck.check_hour,hour_ok=ck.hour_ok,
            hour_cost=ck.hour_cost,locks=ck.locks,qaoa_circuit=qaoa_circuit,max_qubits=max_qubits,strict=strict)
    exec(compile((SOURCE/'rq_classical.py').read_text(),'rq_classical.py','exec'),ns)
    def run_qaoa(Q,offset,p=1,start=None,maxiter=60):
        a,pr,info=tune_qaoa(Q,offset,p,start,maxiter,max_qubits=ns['max_qubits'],statevector=ns['Statevector'])
        ns['_fit_info']=info
        return a,pr
    ns['run_qaoa']=run_qaoa
    ns['schedule_day_hybrid']=lambda *a,**kw: schedule_hybrid(ns,*a,**kw)
    ns['schedule_day']=lambda **kw:schedule_hybrid(ns,range(len(ns['demand'])),**kw)
    return ns


def adapt(ns,reduced=True):
    pending,trace={},[];old_add,old_run=ns['add_locks'],ns['run_qaoa']
    ns['_reduce']=reduced
    def add(Q,off,must_on,must_off,weight):
        pending.clear();pending.update({int(i):1 for i in must_on});pending.update({int(i):0 for i in must_off})
        return old_add(Q,off,must_on,must_off,weight)
    def hourly(Q,off,p=1,start=None,maxiter=60):
        fixed=dict(pending) if reduced else {}
        small,constant,free,values=substitute(Q,off,fixed)
        # Guard before tuning, bit tables, statevectors or probability allocation.
        if len(small)>ns['max_qubits']:raise ValueError(f'Reduced circuit requires {len(small)} qubits; limit {ns["max_qubits"]}. No state table allocated.')
        ns['_fit_info']={}
        angles,pr=old_run(small,constant,p=p,start=start,maxiter=maxiter)
        if len(pr)!=(1<<len(free)):raise AssertionError('Probability layout mismatch')
        ns['_layout']=(free,values)
        work=ns.get('_fit_info',{})
        row=dict(qubits=len(free),forced=len(fixed),fixed=fixed,free=free,values=values,full_Q=Q.copy(),full_offset=off,
                 small_Q=small,small_offset=constant,logical_rzz=p*int(np.count_nonzero(np.triu(small,1))),
                 statevector_entries=len(pr),probability_entries=len(pr),angles=np.asarray(angles).copy())
        row.update(work)
        trace.append(row)
        return angles,pr  # REDUCED probabilities; no 2**len(Q) compatibility expansion.
    ns['add_locks'],ns['run_qaoa']=add,hourly
    ns['_trace']=trace
    return trace


def audit_reduction(trace,full_bits=None):
    """Small-instance development audit; full_bits is an ignored compatibility argument."""
    t=time.perf_counter()
    for r in trace:
        small=bit_table(r['qubits'])
        full=decode(np.arange(len(small)),r['free'],r['values'])
        a=np.einsum('bi,ij,bj->b',full,r['full_Q'],full)+r['full_offset']
        b=np.einsum('bi,ij,bj->b',small,r['small_Q'],small)+r['small_offset']
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-6)
        for i,v in r['fixed'].items():assert (full[:,i]==v).all()
    return time.perf_counter()-t


def _ordered_counts(rows):
    counts=Counter(tuple(int(v) for v in row) for row in rows)
    # Python integers avoid overflow for fleets with >63 plants; only sampled rows are encoded.
    return dict(sorted(counts.items(),key=lambda kv:sum(b<<i for i,b in enumerate(kv[0]))))


def schedule_hybrid(ns,qaoa_hours,p=1,shots=200,sampler='qaoa',seed=7,penalty=None,milp_horizon='day'):
    if shots<1 or sampler not in ('qaoa','random'):raise ValueError('Positive shots and qaoa/random required')
    if '_trace' not in ns:adapt(ns,True)
    chosen=set(int(x) for x in qaoa_hours);T=len(ns['demand']);N=ns['N']
    if not chosen.issubset(range(T)):raise ValueError('Hour outside horizon')
    rng=np.random.default_rng(seed);history=[];log=[];angles=None;t=0
    was=[int(x['on_at_start']) for x in ns['plants']]
    while t<T:
        if t in chosen:
            mo,mf=ns['locks'](ns['plants'],history)
            Q,off=ns['build_qubo'](ns['plants'],ns['demand'][t],ns['reserve'][t],was,penalty=penalty)
            Q,off=ns['add_locks'](Q,off,mo,mf,weight=2*ns['plant_hour_costs'](ns['plants'],was).max())
            if sampler=='qaoa':
                angles,pr=ns['run_qaoa'](Q,off,p=p,start=angles)
                free,values=ns['_layout'];draws=rng.choice(len(pr),size=shots,p=pr/pr.sum())
                states=decode(draws,free,values,columns=N)
                work=ns.get('_fit_info',{});qubits=len(free)
            else:
                fixed={i:1 for i in mo}|{i:0 for i in mf} if ns.get('_reduce',True) else {}
                free=np.array([i for i in range(N) if i not in fixed]);values=np.zeros(N,dtype=np.uint8)
                for i,v in fixed.items():values[i]=v
                states=np.broadcast_to(values,(shots,N)).copy();states[:,free]=rng.integers(0,2,size=(shots,len(free)))
                draws=None;work={};qubits=0
            counts=_ordered_counts(states)
            feasible=[x for x in counts if ns['hour_ok'](ns['plants'],ns['demand'][t],ns['reserve'][t],x,mo,mf)]
            if not feasible and ns['strict']:raise RuntimeError(f'hour {t}: no sampled choice is feasible; no unsampled repair substituted')
            if not feasible:feasible=[x for x in counts if ns['hour_ok'](ns['plants'],ns['demand'][t],ns['reserve'][t],x)]
            pick=min(feasible,key=lambda x:ns['hour_cost'](ns['plants'],ns['demand'][t],x,was)) if feasible else max(counts,key=counts.get)
            log.append(dict(hour=t,method=sampler,pick=list(pick),plants_decided=N,qubits=qubits,
                valid_share=sum(v for x,v in counts.items() if ns['hour_ok'](ns['plants'],ns['demand'][t],ns['reserve'][t],x,mo,mf))/shots,
                p_best=np.nan,matches_best=None,milp_seconds=0.,angles=angles,
                selected_sample_count=counts[pick],sampled_plant_counts=[dict(state=list(x),count=v) for x,v in counts.items()],
                reduced_measurement_counts=None if draws is None else dict(Counter(map(int,draws))),
                statevector_evaluations=work.get('statevector_evaluations',0),quantum_executed=work.get('quantum_executed',False)))
            history.append(list(pick));was=list(pick);t+=1
        else:
            stop=t
            while stop<T and stop not in chosen:stop+=1
            end=T if milp_horizon=='day' else stop
            on,off=ns['remaining_locks'](ns['plants'],history)
            try:_,window,secs=ns['solve_milp_window'](ns['plants'],ns['demand'][t:end],ns['reserve'][t:end],was,on,off)
            except RuntimeError:
                if ns['strict']:raise
                _,window,secs=ns['solve_milp_window'](ns['plants'],ns['demand'][t:end],ns['reserve'][t:end],was,np.zeros(N,int),np.zeros(N,int))
            for k in range(t,stop):
                pick=list(map(int,window[k-t]));history.append(pick);was=pick
                log.append(dict(hour=k,method='milp',pick=pick,qubits=0,valid_share=np.nan,p_best=np.nan,matches_best=None,
                                milp_seconds=secs if k==t else 0.,angles=None,quantum_executed=False,statevector_evaluations=0))
            t=stop
    return np.asarray(history,dtype=int),log


def block_qubo(plants,demand,reserve,schedule,start,width,pair):
    stop=start+width;m=2*width;Q=np.zeros((m,m));offset=0.;operating=[];bad=[]
    for t in range(start,stop):
        costs=[];invalid=[]
        for a,b in ((0,0),(1,0),(0,1),(1,1)):
            on=schedule[t].copy();on[list(pair)]=(a,b)
            costs.append(ck.hour_cost(plants,demand[t],on,was_on=on));invalid.append(not ck.hour_ok(plants,demand[t],reserve[t],on))
        operating.append(np.array(costs));bad.append(np.array(invalid,dtype=float))
    penalty=1+sum(np.ptp(c) for c in operating)+(width+1)*sum(plants[i]['startup_cost'] for i in pair)
    for h,(costs,invalid) in enumerate(zip(operating,bad)):
        e00,e10,e01,e11=costs+penalty*invalid;j,k=2*h,2*h+1
        offset+=e00;Q[j,j]+=e10-e00;Q[k,k]+=e01-e00;Q[j,k]+=e11-e10-e01+e00
    for a,i in enumerate(pair):
        startup=plants[i]['startup_cost']
        for h in range(width):
            j=2*h+a
            if h==0:Q[j,j]+=startup*(1-(schedule[start-1,i] if start else int(plants[i]['on_at_start'])))
            else:Q[j,j]+=startup;Q[j-2,j]-=startup
        if stop<len(schedule):offset+=startup*schedule[stop,i];Q[m-2+a,m-2+a]-=startup*schedule[stop,i]
    return Q,offset


def boundary_fixed(ns,schedule,start,width,pair):
    """Only immutable PREFIX history may force bits; never use incumbent in-block locks."""
    on,off=ns['remaining_locks'](ns['plants'],schedule[:start])
    return {2*h+a:(1 if on[i]>h else 0) for a,i in enumerate(pair) for h in range(width) if on[i]>h or off[i]>h}


def refine_schedule(ns,initial,sampler='qaoa',seed=7,shots=64,width=4,max_passes=1,max_blocks=None,reduce=True):
    if sampler not in ('qaoa','random','exact') or shots<1 or max_passes<1 or not 2<=width<=6:raise ValueError('Invalid refinement settings')
    plants,demand,reserve=ns['plants'],ns['demand'],ns['reserve'];s=np.asarray(initial,dtype=int).copy()
    initial_result=ck.check_schedule(plants,demand,reserve,s,verbose=False)
    if not initial_result['valid']:raise ValueError('Refinement requires a valid starting day')
    current=initial_result['total_cost'];began=time.perf_counter();rng=np.random.default_rng(seed);logs=[]
    if len(demand)<width:raise ValueError("Block width exceeds horizon")
    starts=list(range(0,len(demand)-width+1,max(1,width//2)))
    if starts[-1]!=len(demand)-width:starts.append(len(demand)-width)
    blocks=[(t,pair) for t in starts for pair in combinations(range(len(plants)),2)]
    if max_blocks is not None:blocks=blocks[:max_blocks]
    for pass_id in range(max_passes):
        accepted_this_pass=0
        for start,pair in blocks:
            Q,offset=block_qubo(plants,demand,reserve,s,start,width,pair)
            fixed=boundary_fixed(ns,s,start,width,pair) if reduce else {}
            small,constant,free,values=substitute(Q,offset,fixed);m=len(free)
            # All samplers share the exact same safely restricted search domain.
            work=dict(statevector_evaluations=0,quantum_executed=False,logical_rzz=0)
            if sampler=='qaoa':
                angles,pr,work=tune_qaoa(small,constant,maxiter=40,grid=(5,4),max_qubits=ns['max_qubits'])
                draws=rng.choice(len(pr),size=shots,p=pr/pr.sum())
            elif sampler=='random':angles=None;draws=rng.integers(0,1<<m,size=shots)
            else:angles=None;draws=np.arange(len(bit_table(m,ns['max_qubits'])))
            seen,multiplicity=np.unique(draws,return_counts=True);full=decode(seen,free,values)
            full_values=full.astype(np.uint64)@(np.uint64(1)<<np.arange(2*width,dtype=np.uint64))
            results=candidate_results(plants,demand,reserve,s,start,width,pair,full_values)
            before=current;winner=None;winning_full=None
            # Sorted compressed values are monotonic in full bit order, preserving deterministic ties.
            for value,full_value in zip(seen,full_values):
                r=results[int(full_value)]
                if r['valid'] and r['total_cost']<current-1e-6:current=r['total_cost'];winner=int(value);winning_full=int(full_value)
            if winner is not None:
                s[start:start+width,list(pair)]=decode([winner],free,values)[0].reshape(width,2);accepted_this_pass+=1
            counts=dict(zip(map(int,seen),map(int,multiplicity)))
            logs.append(dict(pass_id=pass_id,start=start,width=width,pair=list(pair),qubits=m,full_block_qubits=2*width,
                fixed=fixed,free=free,values=values,angles=angles,sampler=sampler,
                measurement_counts=counts if sampler!='exact' else None,enumerated=seen if sampler=='exact' else None,
                accepted=winner is not None,winner=winner,winning_full=winning_full,winning_count=counts.get(winner,0),
                cost_before=before,cost_after=current,unique_candidates=len(seen),**work))
        if accepted_this_pass==0:break
    final=ck.check_schedule(plants,demand,reserve,s,verbose=False)
    assert final['valid'] and abs(final['total_cost']-current)<1e-6
    return dict(schedule=s,initial_cost=initial_result['total_cost'],cost=current,valid=True,seconds=time.perf_counter()-began,
                sampler=sampler,seed=seed,blocks=len(logs),accepted=sum(x['accepted'] for x in logs),
                quantum_blocks=sum(x['quantum_executed'] for x in logs),
                quantum_accepted=sum(x['accepted'] and x['quantum_executed'] for x in logs),
                statevector_evaluations=sum(x['statevector_evaluations'] for x in logs),
                min_qubits=min((x['qubits'] for x in logs),default=0),max_qubits=max((x['qubits'] for x in logs),default=0),
                milp_calls=0,trace=logs)


def audit_refinement(ns,initial,run):
    s=np.asarray(initial).copy();previous=ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],s,verbose=False)['total_cost']
    for row in run['trace']:
        assert abs(row['cost_before']-previous)<1e-6
        if row['accepted']:
            winner=row['winner'];counts=row['measurement_counts']
            if counts is not None:assert int(counts.get(winner,counts.get(str(winner),0)))>0
            else:assert winner in row['enumerated']
            legal_fixed=boundary_fixed(ns,s,row['start'],row['width'],row['pair'])
            for i,v in row['fixed'].items():assert legal_fixed[int(i)]==v
            patch=decode([winner],np.asarray(row['free']),np.asarray(row['values']))[0].reshape(row['width'],2)
            s[row['start']:row['start']+row['width'],row['pair']]=patch
            check=ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],s,verbose=False)
            assert check['valid'] and check['total_cost']<previous-1e-6
            previous=check['total_cost']
        assert abs(row['cost_after']-previous)<1e-6
    assert np.array_equal(s,run['schedule']) and abs(previous-run['cost'])<1e-6
    return True


def instrument_milp(ns):
    stats={'calls':0,'seconds':0.,'statuses':[]};old=ns['scipy_milp']
    def solve(*a,**kw):
        stats['calls']+=1;t=time.perf_counter();kw['options']=dict(kw.get('options',{}),mip_rel_gap=0.)
        try:
            res=old(*a,**kw);stats['statuses'].append(int(res.status))
            if res.x is not None and res.status!=0:raise RuntimeError('MILP did not prove optimality')
            return res
        finally:stats['seconds']+=time.perf_counter()-t
    ns['scipy_milp']=solve
    return stats


def json_default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def run_final_study(fleets=(5,10),seeds=(7,8,9,17,18,19),quantum_hours=tuple(range(13,18)),shots=200,block_shots=64,max_passes=1,controls=True,max_qubits=DEFAULT_MAX_QUBITS,output_dir=None):
    """Gross-demand study. No solar/wind data are loaded. Baseline is scoring-only for refinement."""
    from datetime import datetime,timezone
    output=Path(output_dir) if output_dir else HERE/'results/final_reduced_refinement'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    result=dict(settings=dict(fleets=fleets,seeds=seeds,quantum_hours=list(quantum_hours),hourly_shots=shots,block_shots=block_shots,max_passes=max_passes,controls=controls,max_qubits=max_qubits,model='gross demand; original reserve; no ramp constraints',quantum_execution='local ideal statevector; exact-expectation tuning; sampled outcomes'),references={},runs=[],output_dir=str(output))
    import importlib.metadata as metadata
    result['software']={name:metadata.version(name) for name in ('numpy','scipy','qiskit')}
    sources=['reduced_quantum_method.py','rq_classical.py','rq_candidate_checks.py','check.py']
    for fleet in fleets:
        suffix='' if fleet==5 else str(fleet)
        sources.extend([f'data/plants{suffix}.csv',f'data/demand{suffix}.csv'])
    result['source_sha256']={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in sources}
    def save():
        (output/'results.json').write_text(json.dumps(result,indent=2,default=json_default)+'\n')
    for fleet in fleets:
        ns=make_namespace(fleet,max_qubits=max_qubits,strict=True);stats=instrument_milp(ns)
        baseline,bs,_=ns['solve_milp_window'](ns['plants'],ns['demand'],ns['reserve'],[int(p['on_at_start']) for p in ns['plants']],np.zeros(ns['N'],int),np.zeros(ns['N'],int))
        assert ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],bs,verbose=False)['valid']
        scored_baseline=ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],bs,verbose=False)['total_cost']
        assert abs(scored_baseline-baseline)<1e-5
        result['references'][fleet]=dict(cost=baseline,solver_statuses=stats['statuses'],seconds=stats['seconds'])
        print(f'\n{fleet} plants: MILP reference ${baseline:,.2f}; quantum hours {list(quantum_hours)}',flush=True)
        for seed in seeds:
            ns=make_namespace(fleet,max_qubits=max_qubits,strict=True);trace=adapt(ns,True);stats=instrument_milp(ns);t=time.perf_counter()
            try:
                start,hour_log=ns['schedule_day_hybrid'](quantum_hours,shots=shots,seed=seed)
                chk=ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],start,milp_cost=baseline,verbose=False)
                if not chk['valid']:raise RuntimeError('Starting schedule invalid')
            except RuntimeError as exc:
                result['runs'].append(dict(fleet=fleet,seed=seed,valid=False,error=str(exc)));save();print('FAILED',seed,str(exc),flush=True);continue
            initial_seconds=time.perf_counter()-t;audit_seconds=audit_reduction(trace)
            entry=dict(fleet=fleet,seed=seed,valid=True,initial_schedule=start,initial_cost=chk['total_cost'],initial_gap_pct=chk['gap_pct'],initial_seconds=initial_seconds,hourly_trace=hour_log,hourly_qubits=[x['qubits'] for x in trace],hourly_probability_entries=[x['probability_entries'] for x in trace],substitution_audit_seconds=audit_seconds,construction_milp_calls=stats['calls'],construction_milp_seconds=stats['seconds'],refinements={})
            for sampler in (('qaoa','random','exact') if controls else ('qaoa',)):
                before_calls=stats['calls'];run=refine_schedule(ns,start,sampler=sampler,seed=seed,shots=block_shots,max_passes=max_passes,reduce=True)
                assert stats['calls']==before_calls
                audit_refinement(ns,start,run)
                run['gap_pct']=100*(run['cost']/baseline-1);run['total_seconds']=initial_seconds+run['seconds']
                entry['refinements'][sampler]=run
                print(f"seed {seed}, {sampler:6s}: {chk['gap_pct']:.4f}% -> {run['gap_pct']:.4f}%; accepted {run['accepted']}/{run['blocks']}; quantum accepted {run['quantum_accepted']}; reduced block qubits {run['min_qubits']}–{run['max_qubits']}; added {run['seconds']:.2f}s",flush=True)
            result['runs'].append(entry);save()
    print('\nAll accepted changes were replayed from recorded sampled outcomes and independently checked.\nSaved:',output,flush=True)
    return result


def summarize_study(result):
    """Export all paired comparisons, including failures in attempted-run counts."""
    import csv
    rows=[]
    for fleet in result['settings']['fleets']:
        attempted=[r for r in result['runs'] if r['fleet']==fleet]
        good=[r for r in attempted if r['valid']]
        for method in ('hourly','qaoa','random','exact'):
            selected=good if method=='hourly' else [r for r in good if method in r['refinements']]
            if not selected:continue
            gaps=[r['initial_gap_pct'] if method=='hourly' else r['refinements'][method]['gap_pct'] for r in selected]
            times=[r['initial_seconds'] if method=='hourly' else r['refinements'][method]['total_seconds'] for r in selected]
            accepted=0 if method=='hourly' else sum(r['refinements'][method]['accepted'] for r in selected)
            quantum_accepted=0 if method=='hourly' else sum(r['refinements'][method]['quantum_accepted'] for r in selected)
            rows.append(dict(fleet=fleet,method=method,valid=len(selected),attempted=len(attempted),mean_gap_pct=float(np.mean(gaps)),worst_gap_pct=float(max(gaps)),median_total_seconds=float(np.median(times)),accepted=accepted,quantum_accepted=quantum_accepted))
    output=Path(result['output_dir'])
    if rows:
        with (output/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows
