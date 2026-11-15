"""Feasibility-focused MIMIQ training, with fresh matched validation and no repair.

The first experiment freezes the archived failed hour-14 QUBO/history and compares
selection policies. A conditional continuation uses fresh samples, never the best
validation seed or an unsampled/MILP-repaired quantum choice.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib,json,time
import numpy as np
import mimiq_mps_pilot as m
from hourly_seed_shot_sweep import dump,wilson

ROOT=m.ROOT
ARCHIVE=ROOT/'results/mimiq_mps_pilot/20261115T071630549053Z/pilot.json'
VALID_LOWER_BOUND=.05
CVAR_FRACTION=.25
REFINE_OFFSETS=[(-.2,0),(.2,0),(0,-.15),(0,.15),(-.2,.15),(.2,-.15)]
HOLDOUT_SEEDS=(8101,8102,8103)


def weighted_lower_tail(costs,weights,fraction=.25):
    if len(costs)==0:return None
    order=np.argsort(costs,kind='stable');remaining=float(sum(weights))*fraction
    target=remaining;total=0.
    for i in order:
        take=min(remaining,float(weights[i]));total+=take*float(costs[i]);remaining-=take
        if remaining<=1e-12:break
    return total/target


def score(counts,ns,hour,history,Q,c,free,values,full_Q=None,full_c=None):
    base=m.sample_summary(counts,Q,c,free,values,ns,hour,history,full_Q,full_c)
    bits,weights=m.bit_rows(counts,len(Q));states=m.restore(bits,free,values)[:,:ns['N']]
    on,off=ns['locks'](ns['plants'],history)
    demand=ns['demand'][hour];required=demand+ns['reserve'][hour]
    costs=[];valid_weights=[];violation=0.
    for x,w in zip(states,weights):
        cap=sum(p['pmax_mw']*v for p,v in zip(ns['plants'],x))
        floor=sum(p['pmin_mw']*v for p,v in zip(ns['plants'],x))
        violation+=float(w)*(max(0.,required-cap)/required+max(0.,floor-demand)/demand)
        if ns['hour_ok'](ns['plants'],demand,ns['reserve'][hour],x,on,off):
            costs.append(ns['hour_cost'](ns['plants'],demand,x,history[-1]));valid_weights.append(int(w))
    lower,upper=wilson(base['valid_shots'],base['shots'])
    return base|dict(valid_lcb95=lower,valid_ucb95=upper,mean_normalized_violation=violation/base['shots'],
        feasible_cost_cvar25=weighted_lower_tail(np.asarray(costs),np.asarray(valid_weights)),
        mean_feasible_hour_cost=None if not costs else float(np.average(costs,weights=valid_weights)))


def choose(scores):
    """Prespecified, feasibility-aware ranking; no MILP/optimal cost is an input."""
    reliable=[i for i,s in enumerate(scores) if s['valid_lcb95']>=VALID_LOWER_BOUND]
    if reliable:
        return min(reliable,key=lambda i:(scores[i]['feasible_cost_cvar25'],-scores[i]['valid_lcb95'],i))
    observed=[i for i,s in enumerate(scores) if s['valid_shots']>0]
    if observed:
        return min(observed,key=lambda i:(-scores[i]['valid_lcb95'],scores[i]['feasible_cost_cvar25'],i))
    return min(range(len(scores)),key=lambda i:(scores[i]['mean_normalized_violation'],scores[i]['mean_qubo_energy'],i))


def nearby(angle):
    g,b=angle
    return [(max(.02,g+dg),float((b+db)%np.pi)) for dg,db in REFINE_OFFSETS]


def as_arrays(trace):
    return tuple(np.asarray(trace[key]) for key in ('small_Q','free','values'))


def gather(pilot,ns,hour,history,trace,angles,label,seed):
    Q,free,values=as_arrays(trace);c=trace['small_offset']
    result,job=pilot.submit([m.measured(Q,a) for a in angles],label,512,64,seed)
    summaries=[score(r['counts'],ns,hour,history,Q,c,free,values,np.asarray(trace['Q']),trace['offset']) for r in result]
    return summaries,job['job_id']


def new_trace(ns,hour,history):
    Q,c,small,constant,free,values,fixed=m.problem(ns,hour,history)
    return dict(hour=hour,Q=Q,offset=c,small_Q=small,small_offset=constant,free=free,values=values,
                fixed=fixed,history_before=history.copy(),reduced_qubits=len(small))


def main():
    archive=json.loads(ARCHIVE.read_text());old=archive['hundred_plant']['hourly'][1]
    output=ROOT/'results/mimiq_feasibility_training'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    pilot=m.Pilot(output);started=time.perf_counter();ns=m.rq.make_namespace(100,strict=True)
    pilot.record.update(experiment='feasibility-focused training on frozen failed hour 14',
        archive=str(ARCHIVE),archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        settings=dict(valid_lcb95_target=VALID_LOWER_BOUND,cost_tiebreak='mean cheapest 25% of feasible sampled hourly costs',
            training_shots=512,training_bond=64,validation_shots=200,validation_bond=128,
            holdout_seeds=HOLDOUT_SEEDS,refine_offsets=REFINE_OFFSETS,comparison=['old_energy','feasible_same_grid','feasible_expanded'],
            continuation_rule='attempt only if feasible_expanded produces a legal sample in at least two of three validation batches; use a separate fresh seed for every construction hour',
            max_extra_training_rounds_per_continuation_hour=1,max_jobs=16,physical_qpu=False,noise_model=False,
            group_partition=False,renewables=False,model='original gross load and reserve, no ramps, 3 slack qubits'),
        fixed_problem=old)
    pilot.record['source_sha256']['mimiq_feasibility_training.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(output/'prespecified_policy.json',{k:pilot.record[k] for k in ('settings','source_sha256','archive','archive_sha256')})
    Q,free,values=as_arrays(old);c=old['small_offset'];history=old['history_before']
    training_job=next(j for j in archive['jobs'] if j['label']=='h14_training')
    old_scores=[score(r['counts'],ns,14,history,Q,c,free,values,np.array(old['Q']),old['offset']) for r in training_job['results']]
    old_index=min(range(len(old_scores)),key=lambda i:old_scores[i]['mean_qubo_energy'])
    feasible_index=choose(old_scores);centre=old['angles_tested'][feasible_index]
    pilot.record['archived_grid']=dict(angles=old['angles_tested'],scores=old_scores,old_index=old_index,feasible_index=feasible_index)
    pilot.save();print('ARCHIVED policy choices:',old_index,feasible_index,flush=True)
    extra_angles=nearby(centre)
    extra_scores,training_id=gather(pilot,ns,14,history,old,extra_angles,'h14_local_training',8014)
    all_angles=old['angles_tested']+extra_angles;all_scores=old_scores+extra_scores
    expanded_index=choose(all_scores)
    methods=dict(old_energy=old['angles_tested'][old_index],feasible_same_grid=centre,feasible_expanded=all_angles[expanded_index])
    pilot.record.update(extra_training=dict(angles=extra_angles,scores=extra_scores,job_id=training_id),
                        chosen_methods=methods,expanded_index=expanded_index,holdout=[])
    pilot.save();dump(output/'choices_before_holdout.json',dict(methods=methods,expanded_index=expanded_index,scores=all_scores))
    print('CHOSEN',methods,'expanded valid training fraction',all_scores[expanded_index]['valid_share'],flush=True)
    unique=[];mapping={}
    for label,a in methods.items():
        key=tuple(a)
        if key not in unique:unique.append(key)
        mapping[label]=unique.index(key)
    pilot.record['holdout_circuit_mapping']=mapping
    for seed in HOLDOUT_SEEDS:
        results,job=pilot.submit([m.measured(Q,a) for a in unique],f'h14_holdout_{seed}',200,128,seed)
        summaries={label:score(results[index]['counts'],ns,14,history,Q,c,free,values,np.array(old['Q']),old['offset']) for label,index in mapping.items()}
        rng=np.random.default_rng(seed);random_bits=rng.integers(0,2,size=(200,len(Q)),dtype=np.uint8)
        random_counts=Counter(''.join(map(str,x[::-1])) for x in random_bits)
        summaries['uniform']=score(random_counts,ns,14,history,Q,c,free,values,np.array(old['Q']),old['offset'])
        pilot.record['holdout'].append(dict(seed=seed,job_id=job['job_id'],summaries=summaries));pilot.save()
        print('HOLDOUT',seed,{k:v['valid_shots'] for k,v in summaries.items()},flush=True)
    successes=sum(x['summaries']['feasible_expanded']['valid_shots']>0 for x in pilot.record['holdout'])
    continuation=dict(attempted=successes>=2,valid=False,completed_day=False,hourly=[],
        provenance='continuation of archived quantum-selected hour 13; not an independent new full-day restart',
        archived_prefix=history,reference_cost=archive['hundred_plant']['reference_cost'])
    pilot.record['continuation']=continuation;pilot.save()
    if successes>=2:
        history=[list(x) for x in history];angles=methods['feasible_expanded'];stats=m.rq.instrument_milp(ns)
        for hour in range(14,18):
            trace={k:old[k] for k in ('hour','Q','offset','small_Q','small_offset','free','values','fixed','history_before','reduced_qubits')} if hour==14 else new_trace(ns,hour,history)
            trace['history_before']=[list(x) for x in history]
            if hour>14:
                tested=[tuple(angles)]+nearby(angles)
                scores,job_id=gather(pilot,ns,hour,history,trace,tested,f'h{hour}_training',8300+hour)
                chosen=choose(scores)
                trace.update(training_angles=tested,training_scores=scores,training_jobs=[job_id])
                # A fixed extra budget can refine a poorly feasible training result; not a retry of final shots.
                if scores[chosen]['valid_lcb95']<VALID_LOWER_BOUND and len(pilot.record['jobs'])+2*(18-hour)+1<16:
                    extra=nearby(tested[chosen]);more,job_id=gather(pilot,ns,hour,history,trace,extra,f'h{hour}_training_extra',8400+hour)
                    tested+=extra;scores+=more;chosen=choose(scores)
                    trace.update(training_angles=tested,training_scores=scores)
                    trace['training_jobs'].append(job_id)
                angles=tested[chosen]
            trace['chosen_angles']=angles;continuation['hourly'].append(trace);pilot.save()
            small,free,values=as_arrays(trace);constant=trace['small_offset']
            result,job=pilot.submit(m.measured(small,angles),f'h{hour}_continuation',200,128,8200+hour)
            final=score(result[0]['counts'],ns,hour,history,small,constant,free,values,np.array(trace['Q']),trace['offset'])
            trace.update(final=final,job_id=job['job_id'],metadata=result[0]['metadata']);pilot.save()
            print('CONTINUATION',hour,'valid samples',final['valid_shots'],flush=True)
            if final['best_sampled_plant_state'] is None:
                continuation['failure']=f'hour {hour}: no legal sampled choice; no repair and no final-shot retry';break
            history.append(list(final['best_sampled_plant_state']))
        if len(history)==18:
            hold_on,hold_off=ns['remaining_locks'](ns['plants'],history)
            try:
                _,tail,_=ns['solve_milp_window'](ns['plants'],ns['demand'][18:],ns['reserve'][18:],history[-1],hold_on,hold_off)
                schedule=np.vstack([history,tail]);checked=m.rq.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],schedule,milp_cost=continuation['reference_cost'],verbose=False)
                continuation.update(completed_day=True,valid=checked['valid'],schedule=schedule,checked=checked)
            except RuntimeError as exc:continuation['failure']=str(exc)
        continuation['milp_stats']=stats
    else:continuation['failure']='Held-out feasibility gate not met; continuation not attempted'
    pilot.record.update(complete=True,total_wall_seconds=time.perf_counter()-started)
    pilot.save();print('EXPERIMENT COMPLETE',output,'valid complete continuation',continuation['valid'],flush=True)

if __name__=='__main__':main()
