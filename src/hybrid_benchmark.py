"""Local ideal-QAOA proposals + MILP neighborhood refinement; no cloud calls.

Same constraints/costs as check.solve_milp with ramps=False. A refinement may
change at most `radius` commitment bits; it is NOT a SciPy MIP warm start.
"""
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.sparse import lil_matrix
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from check import (load_plants, load_demand, check_schedule, hour_ok, hour_cost, locks)

ROOT = Path(__file__).resolve().parent.parent


def refine(plants, demand, reserve, proposal=None, radius=None, time_limit=10):
    """Full MILP when proposal=None; otherwise a Hamming-ball MILP.

    Solver optimality refers to this solve's search domain, which is restricted
    when a proposal is supplied. Always validate returned schedules independently.
    """
    n, T = len(plants), len(demand)
    U = lambda i, t: i * T + t
    V = lambda i, t: n * T + i * T + t
    P = lambda i, t: 2 * n * T + i * T + t
    c = np.zeros(3 * n * T)
    rows, low, high = [], [], []

    def add(terms, lo, hi):
        rows.append(terms)
        low.append(lo)
        high.append(hi)

    for t in range(T):
        add([(P(i, t), 1) for i in range(n)], demand[t], demand[t])
        add([(U(i, t), p['pmax_mw']) for i, p in enumerate(plants)],
            demand[t] + reserve[t], np.inf)
    for i, p in enumerate(plants):
        up, down, initial = int(p['min_up_h']), int(p['min_down_h']), int(p['on_at_start'])
        for t in range(T):
            c[U(i, t)] = p['no_load_cost_per_h']
            c[V(i, t)] = p['startup_cost']
            c[P(i, t)] = p['marginal_cost_per_mwh']
            add([(P(i, t), 1), (U(i, t), -p['pmin_mw'])], 0, np.inf)
            add([(P(i, t), 1), (U(i, t), -p['pmax_mw'])], -np.inf, 0)
            before = [(U(i, t - 1), 1)] if t else []
            add([(V(i, t), 1), (U(i, t), -1)] + before,
                -initial if t == 0 else 0, np.inf)
            add([(V(i, s), 1) for s in range(max(0, t-up+1), t+1)] + [(U(i, t), -1)],
                -np.inf, 0)
            window = [(V(i, s), 1) for s in range(max(0, t-down+1), t+1)]
            if t >= down:
                add(window + [(U(i, t-down), 1)], -np.inf, 1)
            else:
                add(window, -np.inf, 1-initial)
    if proposal is not None:
        proposal = np.asarray(proposal, dtype=int)
        if proposal.shape != (T, n) or not np.isin(proposal, [0, 1]).all():
            raise ValueError('Proposal must be a binary hours-by-plants array')
        if radius is None or not 0 <= radius <= T*n:
            raise ValueError('Radius must be between 0 and hours * plants')
        # sum_{proposal=0} u + sum_{proposal=1} (1-u) <= radius
        add([(U(i, t), 1-2*int(proposal[t, i])) for t in range(T) for i in range(n)],
            -np.inf, radius-int(proposal.sum()))
    A = lil_matrix((len(rows), len(c)))
    for k, terms in enumerate(rows):
        for j, value in terms:
            A[k, j] += value
    upper = np.r_[np.ones(2*n*T), [p['pmax_mw'] for p in plants for _ in range(T)]]
    result = milp(c, integrality=np.r_[np.ones(2*n*T), np.zeros(n*T)],
                  bounds=Bounds(0, upper),
                  constraints=LinearConstraint(A.tocsr(), low, high),
                  options={'time_limit': time_limit, 'mip_rel_gap': 0.0})
    schedule = None if result.x is None else np.array(
        [[round(result.x[U(i, t)]) for i in range(n)] for t in range(T)])
    checked = None if schedule is None else check_schedule(
        plants, demand, reserve, schedule, verbose=False)
    valid = checked is not None and checked['valid']
    if valid and not np.isclose(checked['total_cost'], result.fun, rtol=1e-6):
        raise AssertionError('MILP and checker costs disagree')
    return dict(schedule=schedule, valid=valid,
                cost=checked['total_cost'] if valid else None,
                proven_in_domain=bool(valid and result.status == 0),
                status=int(result.status), message=result.message,
                solver_gap=getattr(result, 'mip_gap', None))


def propose(plants, demand, reserve, sampler='qaoa', shots=50, seed=0,
            maxiter=30, layers=1):
    """Hourly sampler, classical locks/filtering/dispatch, then cheapest choice.

    QAOA uses an approximate hourly cost, exhaustive statevector expectations,
    an initial angle grid and COBYLA. This explicitly counts classical simulation
    and exhaustive scoring in proposal runtime; it cannot demonstrate QPU speedup.
    Random shares the same constraints, shot budget and postprocessing.
    Slack size adapts to the fleet and hour (not hard-coded to three bits).
    """
    if sampler not in ('qaoa', 'random'):
        raise ValueError('sampler must be qaoa or random')
    rng = np.random.default_rng(seed)
    n = len(plants)
    history, warm_angles, max_qubits, gates, evals = [], None, 0, 0, 0
    floor_price = min(p['marginal_cost_per_mwh'] for p in plants)
    for t in range(len(demand)):
        was = history[-1] if history else [int(p['on_at_start']) for p in plants]
        must_on, must_off = locks(plants, history)
        if sampler == 'random':
            candidates = rng.integers(0, 2, size=(shots, n))
        else:
            costs = np.array([p['startup_cost']*(1-was[i]) + p['no_load_cost_per_h']
                              + p['pmin_mw']*(p['marginal_cost_per_mwh']-floor_price)
                              for i, p in enumerate(plants)])
            capacity = np.array([p['pmax_mw']/50 for p in plants])
            if not np.allclose(capacity, np.round(capacity)):
                raise ValueError('This QUBO requires capacities in 50 MW blocks')
            required = math.ceil((demand[t]+reserve[t])/50)
            spare = max(0, int(capacity.sum())-required)
            slack = math.ceil(math.log2(spare+1)) if spare else 0
            b = np.r_[capacity, -2.0**np.arange(slack)]
            m = len(b)
            if m > 16:
                raise ValueError('Local statevector guard: at most 16 qubits in this benchmark')
            max_qubits = max(max_qubits, m)
            penalty = 2*costs.max()
            Q = np.triu(2*penalty*np.outer(b, b), 1)
            Q[np.diag_indices(m)] = penalty*(b*b-2*required*b)
            Q[np.arange(n), np.arange(n)] += costs
            for i in must_on:
                Q[i, i] -= penalty
            for i in must_off:
                Q[i, i] += penalty
            bits = (np.arange(2**m)[:, None] >> np.arange(m)) & 1
            energy = np.einsum('bi,ij,bj->b', bits, Q, bits)
            energy = (energy-energy.min()) / max(float(np.ptp(energy)), 1e-12)
            pairs = [(i, j, Q[i, j]/4) for i in range(m) for j in range(i+1, m)
                     if Q[i, j] != 0]
            h = -np.diag(Q)/2
            for i, j, value in pairs:
                h[i] -= value
                h[j] -= value
            scale = max(float(np.max(np.abs(h))), max((abs(v) for _, _, v in pairs), default=0), 1e-12)
            gamma, beta = ParameterVector('g', layers), ParameterVector('b', layers)
            qc = QuantumCircuit(m)
            qc.h(range(m))
            for layer in range(layers):
                for i, value in enumerate(h):
                    if value:
                        qc.rz(2*gamma[layer]*value/scale, i)
                for i, j, value in pairs:
                    qc.rzz(2*gamma[layer]*value/scale, i, j)
                qc.rx(2*beta[layer], range(m))
            params = list(gamma)+list(beta)
            gates += len(pairs)*layers  # logical RZZs across one circuit per hour

            def probs(angles):
                return Statevector(qc.assign_parameters(dict(zip(params, angles)))).probabilities()

            def objective(angles):
                nonlocal evals
                evals += 1
                return float(probs(angles) @ energy)

            if warm_angles is None:
                grid = [[g]*layers+[v]*layers for g in (0.2, 1.0, 2.0, 3.0)
                        for v in (0.2, 0.7, 1.2)]
                warm_angles = min(grid, key=objective)
            opt = minimize(objective, warm_angles, method='COBYLA',
                           options={'maxiter': maxiter, 'rhobeg': 0.2})
            warm_angles = opt.x
            weights = probs(warm_angles)
            samples = rng.choice(len(weights), shots, p=weights/weights.sum())
            candidates = bits[samples, :n]
        candidates = np.unique(candidates, axis=0)
        valid = [s for s in candidates if hour_ok(plants, demand[t], reserve[t], s, must_on, must_off)]
        # Invalid proposals are permitted; the MILP must repair them or report failure.
        if not valid:
            valid = [s for s in candidates if hour_ok(plants, demand[t], reserve[t], s)]
        pick = min(valid, key=lambda s: hour_cost(plants, demand[t], s, was)) if valid else candidates[0]
        history.append(list(map(int, pick)))
    return np.array(history), dict(max_qubits=max_qubits, logical_rzz_per_sweep=gates,
                                   objective_evaluations=evals)


def run_hybrid_benchmark(fleets=(5,), horizons=(6, 12, 24), seeds=(0, 1, 2),
                         shots=50, change_fraction=0.15, time_limit=10,
                         maxiter=30, output_dir='results/hybrid_benchmark',
                         proposer=None):
    """Return (raw records, grouped records), also write CSVs and metadata.

    For each seed, perturb the given hourly load by +/-5% and recompute 10%
    reserve; the same instance goes to all methods. Seeds vary both load and
    sampling, so reliability is scenario + sampling reliability. No fallback to
    full MILP is hidden in the hybrids. Refinement gets the same per-solve limit
    as full MILP, but hybrids additionally pay proposal time: this is NOT an
    equal-total-time-budget experiment. Horizons are prefixes, not distinct days.

    Optional proposer(plants, demand, reserve, sampler, shots, seed, maxiter)
    returns (binary schedule, diagnostics dict); useful for your other tab.
    """
    if shots < 1 or not 0 <= change_fraction <= 1:
        raise ValueError('Positive shots and change_fraction in [0,1] required')
    seeds, fleets, horizons = tuple(seeds), tuple(fleets), tuple(horizons)
    if not seeds or not fleets or not horizons or not set(fleets) <= {5, 10}:
        raise ValueError('Use nonempty seeds/horizons and fleets drawn from (5,10)')
    generator = propose if proposer is None else proposer
    records = []
    # Warm up HiGHS once; all reported timings are subsequent wall-clock runs.
    warm_plants, (warm_d, warm_r) = load_plants(), load_demand()
    refine(warm_plants, warm_d[:6], warm_r[:6], time_limit=time_limit)
    for fleet in fleets:
        suffix = '' if fleet == 5 else '10'
        plants = load_plants(ROOT/'data'/f'plants{suffix}.csv')
        base, _ = load_demand(ROOT/'data'/f'demand{suffix}.csv')
        for T in horizons:
            if not 1 <= T <= len(base):
                raise ValueError('Horizon must be between 1 and 24 hours')
            for seed in seeds:
                rng = np.random.default_rng(seed)
                demand = np.round(base[:T]*rng.uniform(0.95, 1.05, T))
                reserve = np.round(0.1*demand)
                start = perf_counter()
                reference = refine(plants, demand, reserve, time_limit=time_limit)
                baseline_time = perf_counter()-start
                ref_cost = reference['cost']
                radius = math.ceil(change_fraction*T*fleet)
                for method in ('full_milp', 'random+milp', 'qaoa+milp'):
                    proposal_time, refine_time, raw_valid, raw_cost, changes, diag = 0., 0., None, None, None, {}
                    proposal_schedule = None
                    if method == 'full_milp':
                        result, total = reference, baseline_time
                        refine_time = baseline_time
                    else:
                        start = perf_counter()
                        proposal_schedule, diag = generator(
                            plants, demand, reserve, sampler=method.split('+')[0],
                            shots=shots, seed=seed, maxiter=maxiter)
                        proposal_time = perf_counter()-start
                        before = check_schedule(plants, demand, reserve, proposal_schedule, verbose=False)
                        raw_valid = before['valid']
                        raw_cost = before['total_cost'] if raw_valid else None
                        refine_start = perf_counter()
                        result = refine(plants, demand, reserve, proposal_schedule, radius, time_limit)
                        refine_time = perf_counter()-refine_start
                        total = perf_counter()-start
                        if result['valid']:
                            changes = int(np.count_nonzero(result['schedule'] != proposal_schedule))
                            assert changes <= radius
                    cost = result['cost']
                    gap = None if cost is None or ref_cost is None else 100*(cost/ref_cost-1)
                    improvement = None if cost is None or raw_cost is None else 100*(raw_cost-cost)/raw_cost
                    records.append(dict(
                        plants=fleet, hours=T, seed=seed, method=method,
                        valid=result['valid'], cost=cost, gap_to_reference_pct=gap,
                        reference_proven_optimal=reference['proven_in_domain'],
                        proposal_valid=raw_valid, proposal_cost=raw_cost,
                        refinement_saving_pct=improvement, changed_bits=changes,
                        allowed_changes=radius if method != 'full_milp' else None,
                        total_s=total, proposal_s=proposal_time, refinement_s=refine_time,
                        time_saved_vs_full_pct=100*(1-total/baseline_time),
                        status=result['status'], solver_message=result['message'],
                        proven_in_search_domain=result['proven_in_domain'],
                        max_qubits=diag.get('max_qubits', 0),
                        logical_rzz_per_sweep=diag.get('logical_rzz_per_sweep', 0),
                        objective_evaluations=diag.get('objective_evaluations', 0),
                        scheduled_sample_shots=shots*T if method != 'full_milp' else 0,
                        paid_qpu_shots=0,
                        proposed_schedule=json.dumps(proposal_schedule.tolist()) if proposal_schedule is not None else None,
                        final_schedule=json.dumps(result['schedule'].tolist()) if result['schedule'] is not None else None,
                    ))
                print(f'Finished {fleet} plants, {T} hours, seed {seed}', flush=True)
    summary = []
    groups = dict.fromkeys((r['plants'], r['hours'], r['method']) for r in records)
    for fleet, T, method in groups:
        group = [r for r in records if (r['plants'], r['hours'], r['method']) == (fleet, T, method)]
        gaps = [r['gap_to_reference_pct'] for r in group if r['gap_to_reference_pct'] is not None]
        savings = [r['refinement_saving_pct'] for r in group if r['refinement_saving_pct'] is not None]
        summary.append(dict(
            plants=fleet, hours=T, method=method, runs=len(group),
            valid_rate_pct=100*mean(r['valid'] for r in group),
            mean_gap_pct=mean(gaps) if gaps else None,
            worst_gap_pct=max(gaps) if gaps else None,
            median_total_s=median(r['total_s'] for r in group),
            median_proposal_s=median(r['proposal_s'] for r in group),
            median_refinement_s=median(r['refinement_s'] for r in group),
            median_time_saved_pct=median(r['time_saved_vs_full_pct'] for r in group),
            mean_refinement_saving_pct=mean(savings) if savings else None,
            reference_proven_rate_pct=100*mean(r['reference_proven_optimal'] for r in group),
        ))
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in (('runs.csv', records), ('summary.csv', summary)):
        with (folder/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    import scipy, qiskit
    (folder/'metadata.json').write_text(json.dumps(dict(
        fleets=fleets, horizons=horizons, seeds=seeds, shots=shots,
        change_fraction=change_fraction, per_milp_time_limit_s=time_limit,
        maxiter=maxiter, qaoa_layers=1, scipy=scipy.__version__, qiskit=qiskit.__version__,
        environment='local ideal statevector; no paid jobs; ramps disabled',
        formulation='hourly approximate QUBO + whole-horizon MILP Hamming neighborhood',
        reference='full same-horizon MILP; optimal only when reference_proven_optimal=True',
        timing='includes proposals, exhaustive simulation, model build and validation; one HiGHS warmup',
        reliability='three methods see identical seed-perturbed load; prefixes of the source day',
    ), indent=2)+'\n')
    return records, summary


def display_hybrid_summary(summary):
    """Show a notebook table, also return Markdown for export."""
    columns = [('plants', 'Plants'), ('hours', 'Hours'), ('method', 'Method'),
               ('runs', 'Runs'), ('valid_rate_pct', 'Valid %'),
               ('mean_gap_pct', 'Mean gap %'), ('worst_gap_pct', 'Worst gap %'),
               ('median_total_s', 'Total s'), ('median_proposal_s', 'Proposal s'),
               ('median_refinement_s', 'MILP s'),
               ('mean_refinement_saving_pct', 'Repair saving %')]
    def fmt(v):
        return '—' if v is None else (f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v))
    text = '| ' + ' | '.join(label for _, label in columns) + ' |\n'
    text += '| ' + ' | '.join('---' for _ in columns) + ' |\n'
    text += '\n'.join('| ' + ' | '.join(fmt(r[key]) for key, _ in columns) + ' |' for r in summary)
    try:
        from IPython.display import Markdown, display
        display(Markdown(text))
    except ImportError:
        print(text)
    return text
