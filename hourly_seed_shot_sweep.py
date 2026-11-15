"""Ideal local seed/shot experiments for the gross-demand hourly hybrid.

No refinement, renewables, provider SDKs, device/noise models or job submission.
All sampled methods call the already validated strict hourly implementation.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import time
import numpy as np
import reduced_quantum_method as rq

LABELS = {'reduced_qaoa': 'Reduced QAOA', 'unreduced_qaoa': 'Unreduced QAOA',
          'random_locked': 'Random, same locks', 'exact_hourly': 'Exact hourly'}
METHODS = tuple(LABELS)[:3]


def clean_json(value):
    """Write standards-compliant JSON, including failed/undefined measurements."""
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, dict): return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wilson(successes, attempts):
    """95% Wilson interval across seed trials, not across correlated hourly shots."""
    if not attempts: return (None, None)
    z = 1.959963984540054
    p = successes / attempts
    denominator = 1 + z*z/attempts
    centre = (p + z*z/(2*attempts)) / denominator
    half = z * math.sqrt(p*(1-p)/attempts + z*z/(4*attempts**2)) / denominator
    return (max(0., centre-half), min(1., centre+half))


def exact_hourly(ns, quantum_hours):
    """Exact local commitment selection; still a greedy hourly hybrid, not day optimum.

    Enumerate little-endian plant patterns in the same tie order as sampled runs.
    This explicit small-fleet reference is never used to repair a sampled trial.
    """
    if ns['N'] > 10: raise ValueError('Exact hourly control is limited to 10 plants')
    selected = set(quantum_hours)
    T, N = len(ns['demand']), ns['N']
    history, log = [], []
    was = [int(p['on_at_start']) for p in ns['plants']]
    t = 0
    while t < T:
        if t in selected:
            on, off = ns['locks'](ns['plants'], history)
            fixed = {i: 1 for i in on} | {i: 0 for i in off}
            free = [i for i in range(N) if i not in fixed]
            values = np.zeros(N, dtype=np.uint8)
            for i, v in fixed.items(): values[i] = v
            states = rq.decode(np.arange(1 << len(free)), free, values)
            feasible = [x.tolist() for x in states if ns['hour_ok'](
                ns['plants'], ns['demand'][t], ns['reserve'][t], x, on, off)]
            if not feasible: raise RuntimeError(f'hour {t}: no feasible exact local choice')
            pick = min(feasible, key=lambda x: ns['hour_cost'](ns['plants'], ns['demand'][t], x, was))
            log.append(dict(hour=t, method='exact_hourly', pick=pick,
                            candidates_enumerated=len(states), feasible_candidates=len(feasible)))
            history.append(pick); was = pick; t += 1
        else:
            stop = t
            while stop < T and stop not in selected: stop += 1
            hold_on, hold_off = ns['remaining_locks'](ns['plants'], history)
            _, plan, _ = ns['solve_milp_window'](ns['plants'], ns['demand'][t:T],
                ns['reserve'][t:T], was, hold_on, hold_off)
            for hour in range(t, stop):
                pick = list(map(int, plan[hour-t])); history.append(pick); was = pick
                log.append(dict(hour=hour, method='milp', pick=pick))
            t = stop
    return np.asarray(history, dtype=int), log


def audit_samples(ns, schedule, log, shots):
    """Require every non-MILP selected row to be an observed legal cheapest sample."""
    for row in log:
        if row['method'] not in ('qaoa', 'random'): continue
        t = row['hour']; counts = row['sampled_plant_counts']
        assert sum(item['count'] for item in counts) == shots
        assert row['selected_sample_count'] > 0
        matches = [item for item in counts if item['state'] == row['pick']]
        assert len(matches) == 1 and matches[0]['count'] == row['selected_sample_count']
        np.testing.assert_array_equal(schedule[t], row['pick'])
        on, off = ns['locks'](ns['plants'], schedule[:t])
        was = schedule[t-1] if t else [int(p['on_at_start']) for p in ns['plants']]
        feasible = [item['state'] for item in counts if ns['hour_ok'](
            ns['plants'], ns['demand'][t], ns['reserve'][t], item['state'], on, off)]
        assert feasible
        chosen = min(feasible, key=lambda x: ns['hour_cost'](ns['plants'], ns['demand'][t], x, was))
        assert chosen == row['pick']


def reference(fleet, max_qubits):
    ns = rq.make_namespace(fleet, max_qubits=max_qubits, strict=True)
    stats = rq.instrument_milp(ns)
    cost, schedule, _ = ns['solve_milp_window'](ns['plants'], ns['demand'], ns['reserve'],
        [int(p['on_at_start']) for p in ns['plants']], np.zeros(ns['N'], int), np.zeros(ns['N'], int))
    checked = rq.ck.check_schedule(ns['plants'], ns['demand'], ns['reserve'], schedule, verbose=False)
    assert checked['valid'] and abs(checked['total_cost']-cost) < 1e-5
    assert stats['statuses'] == [0]
    return dict(cost=float(cost), schedule=schedule, seconds=stats['seconds'], solver_status=0)


def run_trial(fleet, method, shots, seed, quantum_hours, baseline, max_qubits=18, layers=1):
    """Fresh setup, angle training and MILP calls per trial; no cross-trial cache."""
    t0 = time.perf_counter()
    ns = rq.make_namespace(fleet, max_qubits=max_qubits, strict=True)
    stats = rq.instrument_milp(ns)
    trace = rq.adapt(ns, reduced=method != 'unreduced_qaoa')
    row = dict(fleet=fleet, method=method, shots=shots, seed=seed, valid=False,
               cost=None, gap_pct=None, schedule=None, hourly_log=[], failure=None)
    try:
        if method == 'exact_hourly':
            schedule, log = exact_hourly(ns, quantum_hours)
        else:
            sampler = 'random' if method == 'random_locked' else 'qaoa'
            schedule, log = ns['schedule_day_hybrid'](quantum_hours, p=layers, shots=shots,
                                                     seed=seed, sampler=sampler)
        checked = rq.ck.check_schedule(ns['plants'], ns['demand'], ns['reserve'], schedule,
                                       milp_cost=baseline, ramps=False, verbose=False)
        row.update(valid=bool(checked['valid']), schedule=schedule, hourly_log=log,
                   candidate_cost=float(checked['total_cost']), violations=checked['violations'])
        if checked['valid']:
            gap = 100*(checked['total_cost']-baseline)/baseline
            assert gap >= -1e-7, 'A valid schedule scored below the certified baseline'
            row.update(cost=float(checked['total_cost']), gap_pct=0. if abs(gap)<1e-10 else float(gap))
        else:
            row['failure'] = 'Returned day violates full-day constraints'
    except (RuntimeError, ValueError) as exc:
        # Sampling failures/infeasible completions are data, not repaired/retried successes.
        row['failure'] = f'{type(exc).__name__}: {exc}'
    row['total_seconds'] = time.perf_counter()-t0
    row.update(construction_milp_calls=stats['calls'], construction_milp_seconds=stats['seconds'],
        milp_statuses=stats['statuses'], circuit_trace=trace,
        statevector_evaluations=sum(x.get('statevector_evaluations', 0) for x in trace),
        max_qubits=max((x['qubits'] for x in trace), default=0),
        min_qubits=min((x['qubits'] for x in trace), default=0),
        max_probability_entries=max((x['probability_entries'] for x in trace), default=0),
        total_logical_rzz=sum(x.get('logical_rzz', 0) for x in trace),
        attempted_quantum_hours=len(trace),
        actual_quantum_circuits=sum(bool(x.get('quantum_executed')) for x in trace))
    audit_start = time.perf_counter()
    if trace: rq.audit_reduction(trace)
    if row['schedule'] is not None: audit_samples(ns, row['schedule'], row['hourly_log'], shots)
    row['audit_seconds'] = time.perf_counter()-audit_start
    return row


def summarize(result):
    """Conditional gap and unconditional success must always be shown together."""
    target = result['settings']['target_gap_pct']
    groups = {}
    for r in result['runs']:
        groups.setdefault((r['fleet'], r['method'], r['shots']), []).append(r)
    rows = []
    for (fleet, method, shots), trials in groups.items():
        good = [r for r in trials if r['valid']]
        gap = [r['gap_pct'] for r in good]
        hits = sum(r['gap_pct'] <= target + 1e-10 for r in good)
        lo, hi = wilson(len(good), len(trials))
        row = dict(fleet=fleet, method=method, shots=shots, attempted=len(trials), valid=len(good),
            failures=len(trials)-len(good), valid_rate=len(good)/len(trials),
            valid_ci95_low=lo, valid_ci95_high=hi,
            target_gap_pct=target, target_hits=hits, target_rate=hits/len(trials),
            mean_valid_gap_pct=float(np.mean(gap)) if gap else None,
            median_valid_gap_pct=float(np.median(gap)) if gap else None,
            worst_valid_gap_pct=float(max(gap)) if gap else None,
            optimum_hits=sum(abs(g) < 1e-7 for g in gap),
            median_total_seconds=float(np.median([r['total_seconds'] for r in trials])),
            median_statevector_evaluations=float(np.median([r['statevector_evaluations'] for r in trials])),
            max_qubits=max(r['max_qubits'] for r in trials),
            max_probability_entries=max(r['max_probability_entries'] for r in trials),
            median_milp_calls=float(np.median([r['construction_milp_calls'] for r in trials])))
        if method == 'exact_hourly': row.update(valid_ci95_low=None, valid_ci95_high=None)
        rows.append(row)
    return rows


def paired_summary(result):
    """Report common-valid costs and validity differences instead of hiding failures."""
    rows = []
    for fleet in result['settings']['fleets']:
        for shots in result['settings']['shots']:
            group = {(r['method'], r['seed']): r for r in result['runs']
                     if r['fleet']==fleet and r['shots']==shots}
            for control in ('random_locked', 'unreduced_qaoa'):
                wins=ties=losses=both_bad=rq_only=control_only=0; differences=[]
                for seed in result['settings']['seeds']:
                    a, b = group.get(('reduced_qaoa', seed)), group.get((control, seed))
                    if a is None or b is None: continue
                    if a['valid'] and b['valid']:
                        d = b['gap_pct']-a['gap_pct']; differences.append(d)
                        wins += d>1e-8; losses += d < -1e-8; ties += abs(d)<=1e-8
                    elif a['valid']: rq_only += 1
                    elif b['valid']: control_only += 1
                    else: both_bad += 1
                rows.append(dict(fleet=fleet, shots=shots, comparator=control,
                    common_valid=len(differences), reduced_lower_gap=int(wins), ties=int(ties),
                    reduced_higher_gap=int(losses), reduced_only_valid=rq_only,
                    comparator_only_valid=control_only, both_failed=both_bad,
                    mean_gap_reduction_pp=float(np.mean(differences)) if differences else None))
    return rows


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_sweep(fleets=(5,10), seeds=tuple(range(101,111)), shots=(5,10,20,50,100,200),
              quantum_hours=tuple(range(13,18)), max_qubits=18, layers=1,
              target_gap_pct=.1, output_dir=None):
    """360 sampled trials plus two deterministic references with the default settings.

    Seeds change final Born draws, not deterministic angle optimization. Shot sweeps
    vary final samples per hour, not training measurements. Different histories can
    cause later circuits to differ. Trial ordering rotates methods to limit order bias.
    """
    fleets, seeds, shots, quantum_hours = map(tuple, (fleets, seeds, shots, quantum_hours))
    if not fleets or any(n not in (5,10) for n in fleets): raise ValueError('This sweep is restricted to the supplied 5/10-plant gross-demand cases')
    for name, values in [('fleets',fleets),('seeds',seeds),('shots',shots),('quantum_hours',quantum_hours)]:
        if not values or len(set(values)) != len(values): raise ValueError(f'{name} must be nonempty and unique')
        if any(not isinstance(x,(int,np.integer)) for x in values): raise ValueError(f'{name} must contain integers')
    if min(shots)<1 or min(seeds)<0 or min(quantum_hours)<0 or max(quantum_hours)>23: raise ValueError('Invalid shots, seeds or hours')
    if layers != 1: raise ValueError('Keep p=1 fixed in this seed/shot study')
    if not 1 <= max_qubits <= 18 or not math.isfinite(target_gap_pct) or target_gap_pct < 0: raise ValueError('Invalid cap/target')
    output = Path(output_dir) if output_dir else rq.HERE/'results/hourly_seed_shot_sweeps'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(fleets=fleets, seeds=seeds, shots=shots, quantum_hours=quantum_hours,
        methods=METHODS, layers=layers, max_qubits=max_qubits, target_gap_pct=target_gap_pct,
        training='deterministic exact-statevector expectations; all training evaluations timed/counted',
        seed_scope='final measurement draws only; different outcomes alter later histories/circuits',
        model='gross demand; original reserve; no ramp constraints; 3 slack qubits',
        timing='fresh trial setup + MILP construction + training + sampling + checking; reference/audits separate',
        strict=True, refinement=False, renewables=False, noise_model=False, cloud_jobs=False)
    sources=['hourly_seed_shot_sweep.py','reduced_quantum_method.py','rq_classical.py','rq_candidate_checks.py','check.py']
    for n in fleets:
        suffix='' if n==5 else str(n)
        sources += [f'data/plants{suffix}.csv',f'data/demand{suffix}.csv']
    result=dict(settings=settings,references={},runs=[],output_dir=str(output),complete=False,
        software={name:importlib.metadata.version(name) for name in ('numpy','scipy','qiskit')},
        source_sha256={name:hashlib.sha256((rq.HERE/name).read_bytes()).hexdigest() for name in sources})
    dump(output/'manifest.json', {k:v for k,v in result.items() if k!='runs'})
    started=time.perf_counter(); attempted=0
    def record(row):
        nonlocal attempted
        result['runs'].append(row);attempted+=1
        with (output/'runs.jsonl').open('a') as f:
            f.write(json.dumps(clean_json(row),allow_nan=False)+'\n')
        dump(output/'progress.json',dict(completed_trials=attempted,
            expected_trials=len(fleets)*(len(seeds)*len(shots)*len(METHODS)+1),
            last_trial={k:row[k] for k in ('fleet','method','shots','seed','valid','failure')},
            elapsed_seconds=time.perf_counter()-started))
    print('IDEAL LOCAL SIMULATION — gross-demand hourly hybrid; no refinement or provider calls.',flush=True)
    print('Fresh angle training per trial. Failed trials stay in the denominator. Results:',output,flush=True)
    for fleet_index,fleet in enumerate(fleets):
        ref=reference(fleet,max_qubits);result['references'][str(fleet)]=ref
        dump(output/'manifest.json',{k:v for k,v in result.items() if k!='runs'})
        print(f'\n{fleet} plants — certified MILP baseline ${ref["cost"]:,.2f}',flush=True)
        record(run_trial(fleet,'exact_hourly',None,None,quantum_hours,ref['cost'],max_qubits,layers))
        for shot_index,count in enumerate(shots):
            for seed_index,seed in enumerate(seeds):
                rotate=(fleet_index+shot_index+seed_index)%len(METHODS)
                order=METHODS[rotate:]+METHODS[:rotate]
                outcomes=[]
                for method in order:
                    row=run_trial(fleet,method,count,seed,quantum_hours,ref['cost'],max_qubits,layers)
                    record(row)
                    outcomes.append(f'{method}: {row["gap_pct"]:.4f}%' if row['valid'] else f'{method}: FAILED')
                print(f'  shots={count:3d}, seed={seed}: '+ '; '.join(outcomes),flush=True)
            write_csv(output/'summary.csv',summarize(result))
            write_csv(output/'paired.csv',paired_summary(result))
    result['complete']=True;result['elapsed_seconds']=time.perf_counter()-started
    dump(output/'results.json',result)
    dump(output/'manifest.json',{k:v for k,v in result.items() if k!='runs'})
    write_csv(output/'summary.csv',summarize(result));write_csv(output/'paired.csv',paired_summary(result))
    print(f'\nCompleted {attempted} trials in {result["elapsed_seconds"]:.1f}s. All returned schedules and sampled selections checked.',flush=True)
    return result


def display_sweep(result):
    """Render notebook tables and publication-exportable figures from saved trials."""
    from html import escape
    from IPython.display import display, HTML, Image
    import matplotlib.pyplot as plt
    summary=summarize(result);pairs=paired_summary(result);output=Path(result['output_dir'])
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
    formatted=[]
    for row in summary:
        fmt=lambda x:'—' if x is None else f'{x:.4f}'
        formatted.append([row['fleet'],LABELS[row['method']],row['shots'] if row['shots'] is not None else 'all local patterns',
            f'{row["valid"]}/{row["attempted"]}',fmt(row['mean_valid_gap_pct']),fmt(row['worst_valid_gap_pct']),
            f'{row["target_hits"]}/{row["attempted"]}',f'{row["median_total_seconds"]:.3f}',row['max_qubits']])
    display(HTML(table(['Plants','Method','Shots/hour','Valid/attempted','Mean valid gap %','Worst valid gap %',
                        f'Valid & gap ≤ {result["settings"]["target_gap_pct"]}%','Median total s','Max qubits'],formatted)))
    colors={'reduced_qaoa':'#2563eb','unreduced_qaoa':'#7c3aed','random_locked':'#d97706'}
    fleets=result['settings']['fleets'];fig,axes=plt.subplots(3,len(fleets),figsize=(6*len(fleets),10),squeeze=False)
    for col,fleet in enumerate(fleets):
        for method in METHODS:
            rows=sorted([r for r in summary if r['fleet']==fleet and r['method']==method],key=lambda r:r['shots'])
            x=[r['shots'] for r in rows]
            axes[0,col].plot(x,[100*r['valid_rate'] for r in rows],'-o',label=LABELS[method],color=colors[method])
            axes[1,col].plot(x,[r['mean_valid_gap_pct'] if r['mean_valid_gap_pct'] is not None else np.nan for r in rows],'-o',color=colors[method])
            axes[2,col].plot(x,[100*r['target_rate'] for r in rows],'-o',color=colors[method])
        exact=next(r for r in summary if r['fleet']==fleet and r['method']=='exact_hourly')
        if exact['valid']:axes[1,col].axhline(exact['mean_valid_gap_pct'],color='#16a34a',ls='--',label='Exact hourly reference')
        axes[0,col].set(title=f'{fleet} plants — {len(result["settings"]["seeds"])} seeds',ylabel='Valid complete days (%)',ylim=(-3,103))
        axes[1,col].set(ylabel='Mean gap among valid days (%)')
        axes[2,col].set(ylabel=f'Valid & gap ≤ {result["settings"]["target_gap_pct"]}% (%)',ylim=(-3,103))
        for ax in axes[:,col]:
            ax.set_xscale('log');ax.set_xticks(result['settings']['shots'],labels=list(map(str,result['settings']['shots'])))
            ax.set_xlabel('Final measurement shots per quantum hour');ax.grid(alpha=.2)
        axes[0,col].legend(fontsize=8);axes[1,col].legend(fontsize=8)
    fig.suptitle('Reduced hourly hybrid — ideal simulation, gross demand\nGap is conditional on validity; success panels count every attempted seed',fontsize=13)
    fig.tight_layout();fig.savefig(output/'seed_shot_sweep.png',dpi=170);fig.savefig(output/'seed_shot_sweep.pdf');plt.close(fig)
    display(Image(filename=str(output/'seed_shot_sweep.png')))
    print('Gap baseline: matching certified full-day MILP. Exact hourly is a separate greedy classical reference.')
    print('QAOA and random select only sampled candidates. Failure means no valid day; no fallback or rerun.')
    print('Shots vary final draws, not angle-training shots. Exact-expectation training is timed and counted.')
    print('MILP constructs other hours, including initial full-day lookahead. This is not a MILP-free solver.')
    print('Seeds vary draws on the same fixed datasets, not independent demand instances. Confidence intervals are in summary.csv.')
    print('Reduced/unreduced circuits can sample differently despite exact energy substitution; no identical-distribution claim.')
    print('Saved traces, summaries, paired comparisons, versions/hashes, PNG and PDF:',output)
    return summary,pairs
