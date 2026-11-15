"""Shared study definitions extracted from starter.ipynb.

Importing this module runs no experiments and submits no jobs. Original algorithms,
defaults and study functions are preserved; notebook-parsing loaders are replaced by
isolated Python namespaces. Use study_context() to keep each experiment's settings
and mutable instrumentation separate. The original starter remains a local archive.
"""
from types import FunctionType
import time
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, milp as scipy_milp
from scipy.sparse import lil_matrix
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from check import (DATA, load_plants, load_demand, check_schedule, check_hour,
                   hour_ok, hour_cost, locks)


import ast as nd_ast

import json as nd_json

from pathlib import Path as nd_Path

from time import perf_counter as nd_clock

from datetime import datetime as nd_datetime, timezone as nd_timezone

import numpy as nd_np

import matplotlib.pyplot as nd_plt

from IPython.display import display as nd_display

ND_WIDTHS = (5, 6)

ND_SEEDS = (7, 8, 9)

ND_HOLDOUT_SEEDS = (17, 18, 19)

ND_SHOTS = 200

ND_SLACK = 3

ND_REFRESH_QUOTES = True

ND_COMPUTE_CREDITS_PER_HOUR = None

import ast as rq_ast

import json as rq_json

from pathlib import Path as rq_Path

from time import perf_counter as rq_clock

import numpy as rq_np

import matplotlib.pyplot as rq_plt

RQ_FLEETS = (5, 10)

RQ_SEEDS = (7, 8, 9, 17, 18, 19)

RQ_HOURS = tuple(range(13,18))

RQ_SHOTS = 200

RQ_FULL_DAY_CONTROL = True

RQ_MAX_QUBITS = 18

import ast as sw_ast

import csv as sw_csv

import json as sw_json

from pathlib import Path as sw_Path

import numpy as sw_np

import matplotlib.pyplot as sw_plt

SW_SEEDS = (7, 8, 9, 17, 18, 19)

SW_HOURS = tuple(range(13, 18))

SW_SHOTS = 200

import ast as ww_ast

import csv as ww_csv

import json as ww_json

from collections import OrderedDict as ww_OrderedDict

from pathlib import Path as ww_Path

from time import perf_counter as ww_clock

import numpy as ww_np

import matplotlib.pyplot as ww_plt

WW_LENGTHS = (4, 5, 6)

WW_SELECTION_SEEDS = (7, 8, 9)

WW_VALIDATION_SEEDS = (101, 102, 103, 104, 105, 106, 107, 108)

WW_TOP_PER_LENGTH = 3

WW_SHOTS = 200

WW_SCREEN_CACHE = True

import ast as aw_ast

import csv as aw_csv

import json as aw_json

from collections import OrderedDict as aw_OrderedDict

from pathlib import Path as aw_Path

from time import perf_counter as aw_clock

import numpy as aw_np

import matplotlib.pyplot as aw_plt

AW_LENGTHS = tuple(range(1, 25))

AW_SELECTION_SEEDS = (7, 8, 9)

AW_VALIDATION_SEEDS = tuple(range(201, 209))

AW_TOP_OVERALL = 5

AW_SHOTS = 200

AW_TUNING_CACHE = True


def _bind(function, namespace):
    """Preserve a function's code/defaults while binding its experiment globals."""
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    bound.__doc__ = function.__doc__
    return bound

def study_context(**settings):
    """Independent study settings; does not read or execute any notebook."""
    namespace = dict(globals())
    namespace.update(settings)
    for name, value in list(namespace.items()):
        if isinstance(value, FunctionType) and value.__module__ == __name__:
            namespace[name] = _bind(value, namespace)
    return namespace

def reduced_context(shots=200):
    return study_context(RQ_SHOTS=shots)

def install_renewable_solver(namespace):
    # Bind to this run's instrumented solver, preserving its optimality checks.
    namespace['solve_milp_renewables'] = _bind(solve_milp_renewables, namespace)

def make_legacy_namespace(fleet_count=10, n_slack=3):
    """Historical full-bit-table implementation, used only by the older window study."""
    suffix = '' if fleet_count == 5 else str(fleet_count)
    fleet = load_plants(DATA / f'plants{suffix}.csv')
    load, margin = load_demand(DATA / f'demand{suffix}.csv')
    n, nq = len(fleet), len(fleet) + n_slack
    bits = (np.arange(2 ** nq)[:, None] >> np.arange(nq)) & 1
    namespace = dict(globals())
    namespace.update(plants=fleet, demand=load, reserve=margin, N=n,
                     N_SLACK=n_slack, N_QUBITS=nq, BLOCK_MW=50,
                     hours=np.arange(len(load)), names=[p['plant_id'] for p in fleet],
                     INDEX_BITS=bits, PLANT_INDEX=bits[:, :n] @ (2 ** np.arange(n)))
    for name in LEGACY_FUNCTIONS:
        namespace[name] = _bind(globals()[name], namespace)
    return namespace


LEGACY_FUNCTIONS = ('plant_hour_costs', 'build_qubo', 'all_energies', 'qubo_to_ising', 'qaoa_circuit', 'run_qaoa', 'choice_probs', 'add_locks', 'schedule_day', 'remaining_locks', 'solve_milp_window', 'schedule_day_hybrid')

def plant_hour_costs(plants, was_on):
    c = np.zeros(len(plants))

    marginal_cheapest = min(
        p["marginal_cost_per_mwh"] for p in plants
    )

    for i, p in enumerate(plants):
        startup = p["startup_cost"] * (1 - was_on[i])
        no_load = p["no_load_cost_per_h"]
        extra_energy = p["pmin_mw"] * (
            p["marginal_cost_per_mwh"] - marginal_cheapest
        )

        c[i] = startup + no_load + extra_energy

    return c

def build_qubo(plants, demand_mw, reserve_mw, was_on, penalty=None):
    c = plant_hour_costs(plants, was_on)
    A = 2 * c.max() if penalty is None else penalty

    # Plant capacity weights, followed by negative slack weights.
    b = np.array(
        [p["pmax_mw"] / BLOCK_MW for p in plants]
        + [-(2 ** k) for k in range(N_SLACK)]
    )

    # Required capacity in 50 MW blocks.
    r = np.ceil((demand_mw + reserve_mw) / BLOCK_MW)

    Q = np.zeros((N_QUBITS, N_QUBITS))
    offset = 0.0

    # Approximate operating costs.
    for i in range(len(plants)):
        Q[i, i] += c[i]

    # Capacity penalty: A * (sum(b[j] * y[j]) - r)**2.
    for j in range(N_QUBITS):
        Q[j, j] += A * (b[j] ** 2 - 2 * r * b[j])

        for k in range(j + 1, N_QUBITS):
            Q[j, k] += 2 * A * b[j] * b[k]

    offset += A * r ** 2

    return Q, offset

def all_energies(Q, offset):
    # QUBO energy of every bitstring in the configured state table.
    U = np.triu(Q)
    return np.einsum("bj,jk,bk->b", INDEX_BITS, U, INDEX_BITS) + offset

def qubo_to_ising(Q):
    # y = (1 - z) / 2 turns the QUBO into  sum_j h_j Z_j + sum_jk J_jk Z_j Z_k + constant.
    h, J = np.zeros(len(Q)), {}
    for j in range(len(Q)):
        h[j] -= Q[j, j] / 2
        for k in range(j + 1, len(Q)):
            if Q[j, k] != 0:
                J[(j, k)] = Q[j, k] / 4
                h[j] -= Q[j, k] / 4
                h[k] -= Q[j, k] / 4
    return h, J

def qaoa_circuit(Q, p=1):
    # Parameterized QAOA circuit with p layers. Returns (circuit, parameters).
    h, J = qubo_to_ising(Q)
    scale = max(np.abs(h).max(), max(abs(v) for v in J.values()))   # keeps angles in a sane range
    gammas, betas = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = QuantumCircuit(len(Q))
    qc.h(range(len(Q)))
    for layer in range(p):
        for j, hj in enumerate(h):
            if hj != 0:
                qc.rz(2 * gammas[layer] * hj / scale, j)
        for (j, k), Jjk in J.items():
            qc.rzz(2 * gammas[layer] * Jjk / scale, j, k)
        qc.rx(2 * betas[layer], range(len(Q)))
    return qc, list(gammas) + list(betas)

def run_qaoa(Q, offset, p=1, start=None, maxiter=60):
    # Tune angles on the exact statevector; return angles and basis-state probabilities.
    E = all_energies(Q, offset)
    E = (E - E.min()) / (E.max() - E.min())
    qc, params = qaoa_circuit(Q, p)

    def probs_at(angles):
        return Statevector(qc.assign_parameters(dict(zip(params, angles)))).probabilities()

    def average_energy(angles):
        return probs_at(angles) @ E

    if start is None or len(start) != 2 * p:     # coarse grid for one (gamma, beta), used in every layer
        grid = [(g, b) for g in np.linspace(0.1, 3.0, 15) for b in np.linspace(0.1, 1.5, 8)]
        g, b = min(grid, key=lambda gb: average_energy([gb[0]] * p + [gb[1]] * p))
        start = [g] * p + [b] * p
    result = minimize(average_energy, start, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.2})
    return result.x, probs_at(result.x)

def choice_probs(probs):
    # Plant on/off probabilities, marginalized over the slack qubits.
    return np.bincount(PLANT_INDEX, weights=probs, minlength=2 ** N)

def add_locks(Q, offset, must_on, must_off, weight):
    # Penalty weight*(1 - x_i) for a plant that must stay on, weight*x_i for one that must stay off.
    Q = Q.copy()
    for i in must_on:
        Q[i, i] -= weight
        offset += weight
    for i in must_off:
        Q[i, i] += weight
    return Q, offset

def schedule_day(p=1, shots=200, sampler="qaoa", seed=7, penalty=None):
    rng = np.random.default_rng(seed)
    history, log, angles = [], [], None
    was_on = [int(x["on_at_start"]) for x in plants]
    for t in hours:
        must_on, must_off = locks(plants, history)
        Q, offset = build_qubo(plants, demand[t], reserve[t], was_on, penalty=penalty)
        Q, offset = add_locks(Q, offset, must_on, must_off, weight=2 * plant_hour_costs(plants, was_on).max())
        if sampler == "qaoa":
            angles, probs = run_qaoa(Q, offset, p=p, start=angles)
        else:                                   # random guessing, for comparison
            probs = np.full(2 ** N_QUBITS, 1 / 2 ** N_QUBITS)
        samples = rng.choice(2 ** N_QUBITS, size=shots, p=probs / probs.sum())
        seen = [list(INDEX_BITS[k][:N]) for k in np.unique(PLANT_INDEX[samples])]
        ok = [c for c in seen if hour_ok(plants, demand[t], reserve[t], c, must_on, must_off)]
        if not ok:                              # nothing fully valid: allow breaking a lock
            ok = [c for c in seen if hour_ok(plants, demand[t], reserve[t], c)]
        if ok:
            pick = min(ok, key=lambda c: hour_cost(plants, demand[t], c, was_on))
        else:                                   # still nothing: take the most common sample
            pick = list(INDEX_BITS[np.bincount(PLANT_INDEX[samples]).argmax()][:N])
        ref = check_hour(plants, demand[t], reserve[t], pick, was_on, must_on, must_off, verbose=False)
        best_k = None if ref["best_choice"] is None else int(np.dot(ref["best_choice"], 2 ** np.arange(N)))
        valid_share = np.mean([hour_ok(plants, demand[t], reserve[t], INDEX_BITS[k][:N], must_on, must_off)
                               for k in samples])
        log.append(dict(hour=t, pick=pick, matches_best=ref["matches_best"], valid_share=valid_share,
                        p_best=0.0 if best_k is None else choice_probs(probs)[best_k],
                        Q=Q, offset=offset, angles=angles))
        history.append(pick)
        was_on = pick
    return np.array(history), log

def remaining_locks(plants, history):
    # Like check.locks, but with the number of hours each plant must still stay on / off.
    hold_on, hold_off = np.zeros(len(plants), dtype=int), np.zeros(len(plants), dtype=int)
    for i, p in enumerate(plants):
        states = [int(p["on_at_start"])] + [int(h[i]) for h in history]
        now, held = states[-1], 0
        for s in reversed(states):
            if s != now:
                break
            held += 1
        if held == len(states):          # never switched today: free, as in check.locks
            continue
        if now == 1:
            hold_on[i] = max(0, int(p["min_up_h"]) - held)
        else:
            hold_off[i] = max(0, int(p["min_down_h"]) - held)
    return hold_on, hold_off

def solve_milp_window(plants, demand, reserve, was_on, hold_on, hold_off):
    """check.solve_milp for a stretch of hours that starts in the middle of the day.

    was_on: who was on the hour before the stretch (decides who pays a startup).
    hold_on[i] / hold_off[i]: hours plant i must still stay on / off at the start of the stretch
    (see remaining_locks). Returns (cost, schedule [hours x plants], seconds).
    """
    n, T = len(plants), len(demand)
    U = lambda i, t: i * T + t
    V = lambda i, t: n * T + i * T + t
    P = lambda i, t: 2 * n * T + i * T + t
    cost = np.zeros(3 * n * T)
    lower = np.zeros(3 * n * T)
    upper = np.r_[np.ones(2 * n * T), [p["pmax_mw"] for p in plants for _ in range(T)]]
    rows, lo, hi = [], [], []

    def add(terms, low, high):            # low <= sum(coef * x[j]) <= high
        rows.append(terms)
        lo.append(low)
        hi.append(high)

    for t in range(T):
        add([(P(i, t), 1) for i in range(n)], demand[t], demand[t])
        add([(U(i, t), p["pmax_mw"]) for i, p in enumerate(plants)], demand[t] + reserve[t], np.inf)
    for i, p in enumerate(plants):
        up, down, u0 = int(p["min_up_h"]), int(p["min_down_h"]), int(was_on[i])
        for t in range(T):
            cost[U(i, t)] = p["no_load_cost_per_h"]
            cost[V(i, t)] = p["startup_cost"]
            cost[P(i, t)] = p["marginal_cost_per_mwh"]
            add([(P(i, t), 1), (U(i, t), -p["pmin_mw"])], 0, np.inf)
            add([(P(i, t), 1), (U(i, t), -p["pmax_mw"])], -np.inf, 0)
            before = [(U(i, t - 1), 1)] if t > 0 else []          # v_t >= u_t - u_(t-1)
            add([(V(i, t), 1), (U(i, t), -1)] + before, -u0 if t == 0 else 0, np.inf)
            add([(V(i, s), 1) for s in range(max(0, t - up + 1), t + 1)] + [(U(i, t), -1)], -np.inf, 0)
            window = [(V(i, s), 1) for s in range(max(0, t - down + 1), t + 1)]
            if t - down >= 0:
                add(window + [(U(i, t - down), 1)], -np.inf, 1)
            else:
                add(window, -np.inf, 1 - u0)
        # Min up / min down time still owed from before the stretch: pin the first hours.
        for t in range(min(T, int(hold_on[i]))):
            lower[U(i, t)] = 1
        for t in range(min(T, int(hold_off[i]))):
            upper[U(i, t)] = 0
    A = lil_matrix((len(rows), 3 * n * T))
    for k, terms in enumerate(rows):
        for j, a in terms:
            A[k, j] += a
    integrality = np.r_[np.ones(2 * n * T), np.zeros(n * T)]
    start = time.perf_counter()
    res = scipy_milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi),
                     integrality=integrality, bounds=Bounds(lower, upper))
    seconds = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(f"The MILP found no schedule for this stretch: {res.message}")
    schedule = np.array([[round(res.x[U(i, t)]) for i in range(n)] for t in range(T)])
    return res.fun, schedule, seconds

def schedule_day_hybrid(qaoa_hours, p=1, shots=200, sampler="qaoa", seed=7, penalty=None,
                        milp_horizon="day"):
    """schedule_day, except that the hours not in qaoa_hours are decided by a MILP.

    Returns (schedule, log). Each log row has method="qaoa" or "milp"; MILP rows have NaN
    for p_best and valid_share (there are no samples). The log also records the MILP seconds.
    """
    qaoa_hours = {int(t) for t in qaoa_hours}
    rng = np.random.default_rng(seed)
    history, log, angles = [], [], None
    was_on = [int(x["on_at_start"]) for x in plants]
    T, t = len(hours), 0
    while t < T:
        if t in qaoa_hours:                       # ---- the same steps as schedule_day ----
            must_on, must_off = locks(plants, history)
            Q, offset = build_qubo(plants, demand[t], reserve[t], was_on, penalty=penalty)
            Q, offset = add_locks(Q, offset, must_on, must_off,
                                  weight=2 * plant_hour_costs(plants, was_on).max())
            # The whole fleet is in the QUBO every QAOA hour: one qubit per plant plus the slack qubits.
            # Locked plants stay in it (with a penalty); nothing is dropped to make the problem smaller.
            assert N == len(plants) and Q.shape == (N + N_SLACK, N + N_SLACK), "QAOA must decide all plants"
            if sampler == "qaoa":
                angles, probs = run_qaoa(Q, offset, p=p, start=angles)
            else:                                 # random guessing, for comparison
                probs = np.full(2 ** N_QUBITS, 1 / 2 ** N_QUBITS)
            samples = rng.choice(2 ** N_QUBITS, size=shots, p=probs / probs.sum())
            seen = [list(INDEX_BITS[k][:N]) for k in np.unique(PLANT_INDEX[samples])]
            ok = [c for c in seen if hour_ok(plants, demand[t], reserve[t], c, must_on, must_off)]
            if not ok:                            # nothing fully valid: allow breaking a lock
                ok = [c for c in seen if hour_ok(plants, demand[t], reserve[t], c)]
            if ok:
                pick = min(ok, key=lambda c: hour_cost(plants, demand[t], c, was_on))
            else:                                 # still nothing: take the most common sample
                pick = list(INDEX_BITS[np.bincount(PLANT_INDEX[samples]).argmax()][:N])
            ref = check_hour(plants, demand[t], reserve[t], pick, was_on, must_on, must_off, verbose=False)
            best_k = None if ref["best_choice"] is None else int(np.dot(ref["best_choice"], 2 ** np.arange(N)))
            valid_share = np.mean([hour_ok(plants, demand[t], reserve[t], INDEX_BITS[k][:N], must_on, must_off)
                                   for k in samples])
            log.append(dict(hour=t, method=sampler, pick=pick, matches_best=ref["matches_best"],
                            plants_decided=N, qubits=N + N_SLACK, valid_share=valid_share,
                            p_best=0.0 if best_k is None else choice_probs(probs)[best_k],
                            Q=Q, offset=offset, angles=angles, milp_seconds=0.0))
            history.append(pick)
            was_on = pick
            t += 1
        else:                                     # ---- a MILP stretch ----
            stop = t
            while stop < T and stop not in qaoa_hours:
                stop += 1
            end = T if milp_horizon == "day" else stop
            hold_on, hold_off = remaining_locks(plants, history)
            try:
                _, window, secs = solve_milp_window(plants, demand[t:end], reserve[t:end], was_on, hold_on, hold_off)
            except RuntimeError:                  # no schedule honours the locks: allow breaking them
                _, window, secs = solve_milp_window(plants, demand[t:end], reserve[t:end], was_on,
                                                    np.zeros(N, int), np.zeros(N, int))
                print(f"  hour {t}: the MILP had to break a min up/down lock", flush=True)
            for k in range(t, stop):
                must_on, must_off = locks(plants, history)
                pick = [int(x) for x in window[k - t]]
                ref = check_hour(plants, demand[k], reserve[k], pick, was_on, must_on, must_off, verbose=False)
                log.append(dict(hour=k, method="milp", pick=pick, matches_best=ref["matches_best"],
                                valid_share=np.nan, p_best=np.nan, Q=None, offset=None, angles=None,
                                milp_seconds=secs if k == t else 0.0))
                history.append(pick)
                was_on = pick
            t = stop
    return np.array(history), log

def solve_milp_renewables(plants, demand, reserve, renewable):
    """check.solve_milp plus one curtailment variable per hour: the renewable MW spilled.

    thermal output + used renewables = demand, 0 <= used <= forecast; headroom (capacity on - output)
    >= reserve, which is the starter's reserve rule when nothing is curtailed; the usual pmin/pmax,
    startup and min up/down rules. Returns (cost, schedule, dispatch, curtailed_mw [per hour], seconds).
    """
    n, T = len(plants), len(demand)
    U = lambda i, t: i * T + t
    V = lambda i, t: n * T + i * T + t
    P = lambda i, t: 2 * n * T + i * T + t
    C = lambda t: 3 * n * T + t
    n_var = 3 * n * T + T
    cost = np.zeros(n_var)
    rows, lo, hi = [], [], []

    def add(terms, low, high):            # low <= sum(coef * x[j]) <= high
        rows.append(terms)
        lo.append(low)
        hi.append(high)

    for t in range(T):
        add([(P(i, t), 1) for i in range(n)] + [(C(t), -1)], demand[t] - renewable[t], demand[t] - renewable[t])
        add([(U(i, t), p["pmax_mw"]) for i, p in enumerate(plants)] + [(P(i, t), -1) for i in range(n)],
            reserve[t], np.inf)
    for i, p in enumerate(plants):
        up, down, u0 = int(p["min_up_h"]), int(p["min_down_h"]), int(p["on_at_start"])
        for t in range(T):
            cost[U(i, t)] = p["no_load_cost_per_h"]
            cost[V(i, t)] = p["startup_cost"]
            cost[P(i, t)] = p["marginal_cost_per_mwh"]
            add([(P(i, t), 1), (U(i, t), -p["pmin_mw"])], 0, np.inf)
            add([(P(i, t), 1), (U(i, t), -p["pmax_mw"])], -np.inf, 0)
            before = [(U(i, t - 1), 1)] if t > 0 else []
            add([(V(i, t), 1), (U(i, t), -1)] + before, -u0 if t == 0 else 0, np.inf)
            add([(V(i, s), 1) for s in range(max(0, t - up + 1), t + 1)] + [(U(i, t), -1)], -np.inf, 0)
            window = [(V(i, s), 1) for s in range(max(0, t - down + 1), t + 1)]
            if t - down >= 0:
                add(window + [(U(i, t - down), 1)], -np.inf, 1)
            else:
                add(window, -np.inf, 1 - u0)
    A = lil_matrix((len(rows), n_var))
    for k, terms in enumerate(rows):
        for j, a in terms:
            A[k, j] += a
    integrality = np.r_[np.ones(2 * n * T), np.zeros(n * T + T)]
    upper = np.r_[np.ones(2 * n * T), [p["pmax_mw"] for p in plants for _ in range(T)], np.maximum(renewable, 0)]
    started = time.perf_counter()
    res = scipy_milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi),
                     integrality=integrality, bounds=Bounds(0, upper))
    seconds = time.perf_counter() - started
    if res.x is None:
        raise RuntimeError(f"The MILP found no schedule even with curtailment: {res.message}")
    schedule = np.array([[round(res.x[U(i, t)]) for i in range(n)] for t in range(T)])
    dispatch = np.array([[res.x[P(i, t)] for i in range(n)] for t in range(T)])
    curtailed = np.array([res.x[C(t)] for t in range(T)])
    return res.fun, schedule, dispatch, curtailed, seconds

def nd_make_namespace():
    return make_legacy_namespace(10, ND_SLACK)


def nd_instrument(ns):
    """Nested timings: statevector time is INSIDE QAOA time; solver time INSIDE MILP time."""
    stats = {}
    for name, key in [('run_qaoa', 'qaoa_s'), ('solve_milp_window', 'milp_s'),
                      ('scipy_milp', 'milp_solver_s'), ('Statevector', 'statevector_build_s')]:
        original = ns[name]
        def measured(*args, _fn=original, _key=key, **kwargs):
            started = nd_clock()
            stats[_key + '_calls'] = stats.get(_key + '_calls', 0) + 1
            if _key == 'milp_solver_s':
                kwargs['options'] = {**kwargs.get('options', {}), 'mip_rel_gap': 0.0}
            try:
                result = _fn(*args, **kwargs)
                if _key == 'milp_solver_s':
                    stats.setdefault('solver_statuses', []).append(int(result.status))
                    if result.x is not None and result.status != 0:
                        raise RuntimeError('MILP did not prove optimality for its domain')
                return result
            finally:
                stats[_key] = stats.get(_key, 0.0) + nd_clock() - started
        ns[name] = measured
    return stats

def nd_one_run(ns, stats, label, selected, sampler, seed, reference, phase):
    stats.clear()
    started = nd_clock()
    schedule, log, error = None, [], None
    try:
        if sampler == 'milp':
            _, schedule, _ = ns['solve_milp_window'](
                ns['plants'], ns['demand'], ns['reserve'],
                [int(p['on_at_start']) for p in ns['plants']],
                nd_np.zeros(ns['N'], int), nd_np.zeros(ns['N'], int))
        else:
            schedule, log = ns['schedule_day_hybrid'](
                selected, p=1, shots=ND_SHOTS, sampler=sampler,
                seed=seed, milp_horizon='day')
        result = ns['check_schedule'](ns['plants'], ns['demand'], ns['reserve'],
                                     schedule, milp_cost=reference, verbose=False)
    except RuntimeError as exc:
        error, result = str(exc), None
    elapsed = nd_clock() - started
    valid = result is not None and result['valid']
    if valid and reference is not None and result['total_cost'] < reference - 0.01:
        raise AssertionError('Valid result undercuts reference; inspect model/cost consistency')
    qtime, mtime = stats.get('qaoa_s', 0.0), stats.get('milp_s', 0.0)
    row = dict(phase=phase, window=label, sampled_hours=len(selected), sampler=sampler,
               seed=seed, valid=valid, violations=None if result is None else len(result['violations']),
               cost=None if result is None else result['total_cost'],
               gap_pct=result['gap_pct'] if valid else None,
               total_s=elapsed, qaoa_s=qtime, milp_s=mtime,
               other_classical_s=max(0.0, elapsed-qtime-mtime),
               milp_solver_s=stats.get('milp_solver_s', 0.0),
               statevector_build_s=stats.get('statevector_build_s', 0.0),
               statevector_evaluations=stats.get('statevector_build_s_calls', 0),
               qaoa_hour_calls=stats.get('qaoa_s_calls', 0),
               milp_calls=stats.get('milp_s_calls', 0),
               milp_domain_optimal=bool(stats.get('solver_statuses')) and
                   all(status == 0 for status in stats.get('solver_statuses', [])),
               quantum_job_credits=0.0,
               allocated_compute_credits=None if ND_COMPUTE_CREDITS_PER_HOUR is None else
                   elapsed * ND_COMPUTE_CREDITS_PER_HOUR / 3600,
               error=error)
    # Schedules/checker findings stay in memory for auditing; don't retain large Q matrices.
    return row, dict(schedule=schedule, result=result,
                     methods=[r['method'] for r in log], solver_statuses=stats.get('solver_statuses', []).copy())

def nd_table(records):
    from html import escape
    from IPython.display import HTML
    if not records:
        return
    def fmt(value):
        if value is None or isinstance(value, float) and not nd_np.isfinite(value):
            return '—'
        return f'{value:.4f}' if isinstance(value, float) else str(value)
    keys = list(records[0])
    html = '<div style="overflow:auto;max-height:650px"><table><thead><tr>'
    html += ''.join('<th>'+escape(k)+'</th>' for k in keys) + '</tr></thead><tbody>'
    for row in records:
        html += '<tr>'+''.join('<td>'+escape(fmt(row.get(k)))+'</td>' for k in keys)+'</tr>'
    nd_display(HTML(html+'</tbody></table></div>'))

def nd_summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('phase','window','sampled_hours','sampler'))
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        gaps = [r['gap_pct'] for r in group if r['valid'] and r['gap_pct'] is not None]
        summary.append(dict(zip(('phase','window','sampled_hours','sampler'), key),
            runs=len(group), valid_days=sum(r['valid'] for r in group),
            mean_valid_gap_pct=float(nd_np.mean(gaps)) if gaps else float('nan'),
            worst_valid_gap_pct=max(gaps) if gaps else float('nan'),
            **{out:float(nd_np.median([r[field] for r in group])) for out,field in (
                ('median_total_s','total_s'), ('median_qaoa_s','qaoa_s'),
                ('median_milp_s','milp_s'), ('median_other_s','other_classical_s'),
                ('median_statevectors','statevector_evaluations'), ('median_milp_calls','milp_calls'))},
            quantum_job_credits=sum(r['quantum_job_credits'] for r in group)))
    return rows, summary

def nd_credit_quotes():
    """Read-only qBraid quotes in credits, with a fallback for the older notebook SDK."""
    if not ND_REFRESH_QUOTES:
        return []
    # qBraid Lab's notebook SDK can predate estimate_cost. The image's system
    # SDK supports it; use its existing credential resolution without copying tokens.
    script = '''import json,sys
from datetime import datetime,timezone
from qbraid.runtime import QbraidProvider
p=QbraidProvider()
rows=[]
for device in ('ionq:ionq:sim:simulator','aws:ionq:qpu:forte-1'):
    try:
        q=p.client.estimate_cost(device,shots=int(sys.argv[1])).model_dump(mode='json')
        rows.append(dict(device=device,utc=datetime.now(timezone.utc).isoformat(),**q))
    except Exception as e:
        rows.append(dict(device=device,pricingAvailable=False,error=type(e).__name__))
print(json.dumps(rows))
'''
    try:
        from qbraid.runtime import QbraidProvider
        provider = QbraidProvider()
        if not hasattr(provider.client, 'estimate_cost'):
            import subprocess
            response = subprocess.run(['/usr/bin/python', '-c', script, str(ND_SHOTS)],
                                      capture_output=True, text=True, check=True, timeout=45)
            return nd_json.loads(response.stdout)
        quotes = []
        for device in ('ionq:ionq:sim:simulator', 'aws:ionq:qpu:forte-1'):
            try:
                quote = provider.client.estimate_cost(device, shots=ND_SHOTS).model_dump(mode='json')
                quotes.append(dict(device=device, utc=nd_datetime.now(nd_timezone.utc).isoformat(), **quote))
            except Exception as exc:
                quotes.append(dict(device=device, pricingAvailable=False, error=type(exc).__name__))
        return quotes
    except Exception as exc:
        print('Live quotes unavailable:', type(exc).__name__, '(not treated as zero credits)')
        return []

def run_night_day_study():
    print(f'NIGHT/DAY WINDOW study — 10 plants, 24 hours, {10+ND_SLACK} qubits, {ND_SHOTS} shots, p=1', flush=True)
    print('Earlier study: 24 individual hours, then SIX candidate windows with seeds 7–9.')
    print('Its 13–18 label means FIVE hours. That was candidate selection, not an exhaustive window test.')
    print('MILP looks through hour 23 at every MILP stretch; all that planning is timed.')
    print('Three slack bits retain the original 350 MW spare-capacity limit. This study ranks that model.')
    print('QAOA timings are classical IDEAL SIMULATION + tuning, not measured QPU execution times.')
    print('Statevector and solver timings are nested diagnostics; do not add them to total time.')
    ns = nd_make_namespace()
    stats = nd_instrument(ns)
    rows, artifacts, choices = [], {}, {}
    for width in ND_WIDTHS:
        for start in range(25-width):
            choices[f'{start:02}:00–{start+width:02}:00 ({width}h)'] = tuple(range(start, start+width))
    choices['DAY 06:00–18:00 (12h)'] = tuple(range(6,18))
    choices['NIGHT 18:00–06:00 (12h)'] = tuple(range(6)) + tuple(range(18,24))
    # Night is two blocks within this single-day horizon, NOT a multi-day cyclic model.
    all_hours = tuple(range(24))
    ref_row, ref_art = nd_one_run(ns, stats, 'FULL MILP', (), 'milp', ND_SEEDS[0], None, 'reference')
    if not ref_row['valid'] or not ref_row['milp_domain_optimal']:
        raise RuntimeError('Need a valid, proven full-day MILP reference')
    reference = ref_row['cost']
    ref_row['gap_pct'] = 0.0
    print(f'Reference: ${reference:,.2f}; total {ref_row["total_s"]:.3f}s', flush=True)
    def run(label, selected, sampler, seed, phase):
        row, art = nd_one_run(ns, stats, label, selected, sampler, seed, reference, phase)
        rows.append(row)
        artifacts[(phase, label, sampler, seed)] = art
        return row
    began = nd_clock()
    # No distributions or MILP plans cached between runs. Seed changes sampling, not optimizer start.
    for seed in ND_SEEDS:
        run('FULL MILP', (), 'milp', seed, 'selection')
        run('FULL QAOA', all_hours, 'qaoa', seed, 'selection')
        order = list(choices)
        nd_np.random.default_rng(seed).shuffle(order)
        for label in order:
            for sampler in ('qaoa', 'random'):
                run(label, choices[label], sampler, seed, 'selection')
            print(f'Finished seed={seed}, {label}; elapsed {nd_clock()-began:.0f}s', flush=True)
    raw, summary = nd_summarize(rows)
    winners = {}
    for width in ND_WIDTHS:
        eligible = [r for r in summary if r['sampler']=='qaoa' and r['sampled_hours']==width
                    and r['valid_days']==r['runs']]
        if eligible:
            best = min(eligible, key=lambda r:(r['mean_valid_gap_pct'],r['median_total_s']))
            winners[width] = best['window']
            print(f"Best observed {width}h window: {best['window']}, mean gap {best['mean_valid_gap_pct']:.4f}%")
        else:
            print(f'No {width}h window valid in every selection seed.')
    # Select BEFORE examining these seeds; same demand day, so this is not new-day validation.
    holdouts = set(winners.values()) | {'13:00–18:00 (5h)', '13:00–19:00 (6h)'}
    for seed in ND_HOLDOUT_SEEDS:
        run('FULL MILP', (), 'milp', seed, 'holdout')
        run('FULL QAOA', all_hours, 'qaoa', seed, 'holdout')
        for label in sorted(holdouts & choices.keys()):
            for sampler in ('qaoa', 'random'):
                run(label, choices[label], sampler, seed, 'holdout')
        print(f'Finished holdout seed={seed}; elapsed {nd_clock()-began:.0f}s', flush=True)
    raw, summary = nd_summarize(rows)
    print('\nValid gaps only; NaN = no valid days. Rank equal-duration windows separately.')
    nd_table(summary)
    fig, axes = nd_plt.subplots(len(ND_WIDTHS), 2, figsize=(13, 4*len(ND_WIDTHS)), squeeze=False)
    for idx, width in enumerate(ND_WIDTHS):
        part = [r for r in summary if r['phase']=='selection' and r['sampled_hours']==width]
        for sampler in ('qaoa', 'random'):
            sub = sorted((r for r in part if r['sampler']==sampler), key=lambda r:r['window'])
            x = [int(r['window'][:2]) for r in sub]
            axes[idx,0].plot(x, [r['mean_valid_gap_pct'] for r in sub], 'o-', label=f'{sampler}: valid-run mean')
            axes[idx,1].plot(x, [r['median_total_s'] for r in sub], 'o-', label=f'{sampler} + MILP')
            bad = [r for r in sub if r['valid_days']<r['runs']]
            axes[idx,0].scatter([int(r['window'][:2]) for r in bad], [r['mean_valid_gap_pct'] for r in bad],
                                marker='x', s=90, color='red', label=f'{sampler}: some invalid runs')
        for label, color in [('FULL MILP','black'), ('FULL QAOA','purple')]:
            base = next(r for r in summary if r['phase']=='selection' and r['window']==label)
            axes[idx,0].axhline(base['mean_valid_gap_pct'], color=color, linestyle='--', label=label)
            axes[idx,1].axhline(base['median_total_s'], color=color, linestyle='--', label=label)
        axes[idx,0].set(title=f'{width}-hour windows: cost (inspect validity table)', xlabel='Start hour', ylabel='Gap (%)')
        axes[idx,1].set(title=f'{width}-hour windows: ALL computation', xlabel='Start hour', ylabel='Median seconds', yscale='log')
        for ax in axes[idx]:
            ax.axvline(13, color='gray', linestyle=':')
            ax.legend(fontsize=7)
    fig.suptitle('NIGHT/DAY WINDOW study — local ideal simulation')
    fig.tight_layout()
    nd_plt.show()
    quotes = nd_credit_quotes()
    print('\nCREDITS: actual quantum-job charges for every run above = 0 (no remote jobs).')
    print('Local runtime uses the existing instance. Subscription and AI costs are not included;')
    print('optional allocated_compute_credits uses the configured rate, not an account balance delta.')
    quote_rows = []
    for quote in quotes:
        print('Live read-only quote:', quote)
        if quote.get('pricingAvailable'):
            price = float(quote['estimatedCost'])
            for calls in (5,6,24):
                quote_rows.append(dict(device=quote['device'], shots_per_job=ND_SHOTS,
                    final_sampling_jobs=calls, estimated_credits=calls*price))
    if quote_rows:
        nd_table(quote_rows)
    print('These hypothetical credits assume ONE separately billed final-sampling job per QAOA hour,')
    print('after LOCAL angle tuning. They exclude hardware tuning, queue time, and any batching effects.')
    print('Hardware tuning would require many further circuit evaluations. statevector_evaluations')
    print('counts the local workload, but is not a reliable prediction of hardware convergence or time.')
    print('Quotes are estimates, not caps; availability/noise and circuit acceptance are not tested.')
    print('MILP has no quantum-job charge. No QPU speedup is measured by this local study.')
    print('\nFRESH-SEED COMPARISON (same demand day; three unseen sampling seeds)')
    compact_keys = ('window','sampler','valid_days','runs','mean_valid_gap_pct','median_total_s',
                    'median_qaoa_s','median_milp_s')
    nd_table([{k:r[k] for k in compact_keys} for r in summary
              if r['phase']=='holdout' and r['sampler'] in ('qaoa','milp')])
    print('Selection winners can change on fresh seeds; this does not prove a universally best window.')
    print('All earlier MILP lookahead and all classical simulation/checking are included in total_s.')
    return dict(runs=raw, summary=summary, artifacts=artifacts, winners=winners,
                reference=ref_row, reference_artifact=ref_art, credit_quotes=quotes,
                settings=dict(widths=ND_WIDTHS, seeds=ND_SEEDS, holdout_seeds=ND_HOLDOUT_SEEDS,
                              shots=ND_SHOTS, slack=ND_SLACK, milp_horizon='day'),
                experiment_s=nd_clock()-began)

def rq_helpers(fleet_count=10):
    if fleet_count not in (5, 10):
        raise ValueError('This study supports the supplied 5- and 10-plant datasets.')
    context = study_context(ND_SLACK=3, ND_SHOTS=RQ_SHOTS,
                            ND_COMPUTE_CREDITS_PER_HOUR=None)
    import reduced_quantum_method as rq_engine
    def make_fleet_namespace():
        return rq_engine.make_namespace(fleet_count, n_slack=3,
                                        max_qubits=RQ_MAX_QUBITS, strict=False)
    context['nd_make_namespace'] = make_fleet_namespace
    return context


def rq_substitute(Q, offset, fixed):
    from reduced_quantum_method import substitute
    return substitute(Q, offset, fixed)

def rq_adapt(ns, reduced):
    from reduced_quantum_method import adapt
    return adapt(ns, reduced)

def rq_audit(trace, full_bits=None):
    # full_bits is accepted for API compatibility; the audit uses reduced states.
    # Exhaustively compare energies only within the small restricted domain.
    from reduced_quantum_method import audit_reduction
    return audit_reduction(trace)

def run_reduced_plant_study(fleet_count=10):
    helper = rq_helpers(fleet_count)
    original_variant = f'original {fleet_count+3} qubits'
    ns = helper['nd_make_namespace']()
    stats = helper['nd_instrument'](ns)
    ref, art = helper['nd_one_run'](ns,stats,'FULL MILP',(),'milp',RQ_SEEDS[0],None,'reduction')
    assert ref['valid'] and ref['milp_domain_optimal']
    reference = ref['cost']
    print(f'REDUCED-QUBIT {fleet_count} PLANT study: MILP reference ${reference:,.2f}',flush=True)
    print(f'QAOA hours {list(RQ_HOURS)}; p=1, {RQ_SHOTS} shots, 3 slack bits; seeds {RQ_SEEDS}')
    print('Only genuine minimum-up/down locks are fixed; no MILP choices are used to fix free plants.')
    print('Full and reduced runs can develop different histories. Warm-start angles are retained.')
    print('Reduced circuits cannot violate a fixed lock. Other original fallbacks remain and are checked.')
    rows, artifacts, hourly_rows, traces = [], {}, [], {}
    # Primary result: hourly QAOA during 13..17, MILP for the other hours.
    # MILP still looks ahead through the remaining day; all that work is timed.
    # Full-day QAOA is a separate control to show what the time split contributes.
    scenarios = [('13:00–18:00',RQ_HOURS)]
    if RQ_FULL_DAY_CONTROL:
        scenarios.append(('FULL DAY',tuple(range(24))))
    for seed in RQ_SEEDS:
        # Repeat baseline as well so timing is not just a single cold-start observation.
        ns = helper['nd_make_namespace'](); stats=helper['nd_instrument'](ns)
        row,art=helper['nd_one_run'](ns,stats,'FULL MILP',(),'milp',seed,reference,'reduction')
        row.update(fleet=fleet_count,variant='MILP',min_qubits=0,max_qubits=0,mean_qubits=0,mean_rzz=0,
                   mean_statevector_entries=0,audit_s=0)
        rows.append(row);artifacts[('MILP','FULL MILP',seed)]=art
        for label, selected in scenarios:
            for reduced in (False,True):
                variant = 'fixed plants removed' if reduced else original_variant
                ns=helper['nd_make_namespace']()
                trace=rq_adapt(ns,reduced)
                stats=helper['nd_instrument'](ns)
                row,art=helper['nd_one_run'](ns,stats,label,selected,'qaoa',seed,reference,'reduction')
                row.update(fleet=fleet_count,variant=variant,
                    min_qubits=min((r['qubits'] for r in trace),default=0),
                    max_qubits=max((r['qubits'] for r in trace),default=0),
                    mean_qubits=float(rq_np.mean([r['qubits'] for r in trace])) if trace else 0,
                    mean_rzz=float(rq_np.mean([r['logical_rzz'] for r in trace])) if trace else 0,
                    mean_statevector_entries=float(rq_np.mean([r['statevector_entries'] for r in trace])) if trace else 0,
                    audit_s=rq_audit(trace,ns['INDEX_BITS']))
                assert len(trace)<=len(selected)
                for t,item in zip(selected,trace):
                    hourly_rows.append(dict(fleet=fleet_count,window=label,variant=variant,seed=seed,hour=t,
                        qubits=item['qubits'],forced_plants=item['forced'],logical_rzz=item['logical_rzz'],
                        statevector_entries=item['statevector_entries'],
                        forced_names=', '.join(f"{ns['names'][i]}={v}" for i,v in item['fixed'].items())))
                rows.append(row);artifacts[(variant,label,seed)]=art;traces[(variant,label,seed)]=trace
                gap=f"{row['gap_pct']:.4f}%" if row['valid'] else 'INVALID'
                print(f"seed {seed}, {label}, {variant}: gap {gap}, "
                      f"qubits {row['min_qubits']}–{row['max_qubits']}, total {row['total_s']:.2f}s",flush=True)
    summary=[]
    for label,variant in dict.fromkeys((r['window'],r['variant']) for r in rows):
        group=[r for r in rows if r['window']==label and r['variant']==variant]
        gaps=[r['gap_pct'] for r in group if r['valid']]
        summary.append(dict(fleet=fleet_count,window=label,variant=variant,runs=len(group),
            valid_days=sum(r['valid'] for r in group),
            mean_valid_gap_pct=float(rq_np.mean(gaps)) if gaps else None,
            worst_valid_gap_pct=max(gaps) if gaps else None,
            min_qubits=min(r['min_qubits'] for r in group),max_qubits=max(r['max_qubits'] for r in group),
            mean_qubits=float(rq_np.mean([r['mean_qubits'] for r in group])),
            mean_logical_rzz=float(rq_np.mean([r['mean_rzz'] for r in group])),
            mean_statevector_entries=float(rq_np.mean([r['mean_statevector_entries'] for r in group])),
            median_total_s=float(rq_np.median([r['total_s'] for r in group])),
            median_qaoa_s=float(rq_np.median([r['qaoa_s'] for r in group])),
            median_milp_s=float(rq_np.median([r['milp_s'] for r in group])),
            quantum_job_credits=0))
    print('\nCOST AND RESOURCE COMPARISON (invalid days excluded from gaps, not from timings)')
    helper['nd_table'](summary)
    print('\nPAIRED COST-GAP DIFFERENCE: reduced minus original; negative is better.')
    pairs=[]
    for label,selected in scenarios:
        for seed in RQ_SEEDS:
            a=next(r for r in rows if r['window']==label and r['seed']==seed and r['variant']==original_variant)
            b=next(r for r in rows if r['window']==label and r['seed']==seed and r['variant']=='fixed plants removed')
            pairs.append(dict(fleet=fleet_count,window=label,seed=seed,original_valid=a['valid'],reduced_valid=b['valid'],
                gap_change_percentage_points=b['gap_pct']-a['gap_pct'] if a['valid'] and b['valid'] else None,
                total_time_ratio=b['total_s']/a['total_s']))
    helper['nd_table'](pairs)
    print('\nPER-HOUR REDUCED CIRCUITS: 13:00–18:00 (all seeds; actual locks in each evolving history)')
    helper['nd_table']([r for r in hourly_rows if r['window']=='13:00–18:00' and r['variant']=='fixed plants removed'])
    fig,axes=rq_plt.subplots(1,2,figsize=(12,4))
    for seed in RQ_SEEDS:
        sub=[r for r in hourly_rows if r['window']=='13:00–18:00' and r['variant']=='fixed plants removed' and r['seed']==seed]
        axes[0].plot([r['hour'] for r in sub],[r['qubits'] for r in sub],'o-',label=f'seed {seed}')
    axes[0].axhline(fleet_count+3,color='black',linestyle='--',label='Original')
    axes[0].set(xlabel='Planning hour',ylabel='Qubits per circuit',title='Fixed-plant substitution',xticks=list(RQ_HOURS))
    axes[0].legend(fontsize=7)
    labels=[f"{r['window']}\n{r['variant']}" for r in summary]
    axes[1].bar(range(len(summary)),[r['mean_valid_gap_pct'] if r['mean_valid_gap_pct'] is not None else float('nan') for r in summary])
    axes[1].set_xticks(range(len(summary)),labels,rotation=25,ha='right',fontsize=8)
    axes[1].set(ylabel='Mean valid daily cost gap (%)',title=f'{fleet_count} plants: check valid-day counts in table')
    fig.suptitle(f'Reduced-qubit comparison: {fleet_count} plants')
    fig.tight_layout();rq_plt.show()
    print('All substitutions passed exhaustive energy-equivalence checks, including offsets.')
    print('Logical RZZ counts are before hardware transpilation/routing; statevector entries are not runtime predictions.')
    print('Total timing includes substitution, sampled decoding, MILP lookahead, local tuning and scheduling checks.')
    print('The extra exhaustive substitution audit is separate: total audit seconds =',sum(r['audit_s'] for r in rows))
    print('No actual hardware sampling/noise was tested. Local ideal simulation job credits = 0.')
    print('Fixing locks changes the variational search and enforces those locks exactly; lower qubits do not guarantee lower cost.')
    print('Three slack bits and the original approximate hourly objective remain; this does not fix their modeling limitations.')
    return dict(fleet=fleet_count,reference_cost=reference,summary=summary,runs=rows,hourly=hourly_rows,
                paired=pairs,artifacts=artifacts,traces=traces,
                settings=dict(fleet=fleet_count,seeds=RQ_SEEDS,hours=RQ_HOURS,shots=RQ_SHOTS,slack=3,layers=1))

def run_solar_wind_rq_study():
    from check import DATA
    private = reduced_context(SW_SHOTS)
    helper = private['rq_helpers'](10)
    original = helper['nd_make_namespace']()
    gross, reserve = original['demand'].copy(), original['reserve'].copy()
    fleet = original['plants']
    with open(DATA / 'renewables.csv', newline='') as f:
        forecast = list(sw_csv.DictReader(f))
    assert [int(r['hour']) for r in forecast] == list(range(24))
    scale = gross.max() / max(float(r['system_load_mw']) for r in forecast)
    solar = sw_np.array([float(r['solar_mw']) for r in forecast]) * scale
    wind = sw_np.array([float(r['wind_mw']) for r in forecast]) * scale
    net = gross - solar - wind
    assert sw_np.all(net >= 0), 'Negative net load needs explicit curtailment modeling.'
    top_hours = tuple(sorted(sw_np.argsort(-net)[:len(SW_HOURS)].tolist()))
    # Keep reserve at the original gross-demand value for BOTH algorithms.
    # Check the net-load model against a curtailment-enabled MILP:
    # at this share, a zero-spillage optimum must match plain net-load MILP.
    curt_ns = helper['nd_make_namespace']()
    helper['nd_instrument'](curt_ns)  # require proven MILP termination / zero relative gap
    install_renewable_solver(curt_ns)
    curt_cost, _, _, spill, _ = curt_ns['solve_milp_renewables'](fleet, gross, reserve, solar + wind)
    assert sw_np.abs(spill).sum() < 1e-6, 'This share needs a curtailment-aware hourly scheduler.'

    scenarios = [('No renewables / 13–17', gross, SW_HOURS),
                 ('Solar + wind / 13–17', net, SW_HOURS),
                 ('Solar + wind / top net hours', net, top_hours)]
    rows, artifacts, summaries, references = [], {}, [], {}
    print('SOLAR + WIND — matched 10-plant comparison; local simulation only')
    print(f'Seeds {SW_SEEDS}; {SW_SHOTS} shots, p=1, 3 slack bits.')
    print(f'Primary hours {list(SW_HOURS)}; top net-load hours {list(top_hours)}.')
    print(f'Renewable energy share {(solar + wind).sum()/gross.sum():.2%}; reserve unchanged; no curtailment.')
    print('Each scenario uses its OWN full-day MILP reference. Invalid runs never enter gap averages.')

    def namespace(load):
        ns = helper['nd_make_namespace']()
        ns.update(demand=load.copy(), reserve=reserve.copy())
        return ns

    for label, load, selected in scenarios:
        ns = namespace(load)
        stats = helper['nd_instrument'](ns)
        ref, art = helper['nd_one_run'](ns, stats, label, (), 'milp', SW_SEEDS[0], None, 'solar/wind')
        assert ref['valid'] and ref['milp_domain_optimal']
        reference = references[label] = ref['cost']
        if label.startswith('Solar'):
            assert abs(reference-curt_cost) < 0.01, 'Net-load and curtailment baselines disagree.'
        for seed in SW_SEEDS:
            for method in ('Full-day MILP', 'Original hybrid', 'Reduced hybrid'):
                ns = namespace(load)
                trace = [] if method == 'Full-day MILP' else private['rq_adapt'](ns, method == 'Reduced hybrid')
                stats = helper['nd_instrument'](ns)
                row, art = helper['nd_one_run'](
                    ns, stats, label, () if method == 'Full-day MILP' else selected,
                    'milp' if method == 'Full-day MILP' else 'qaoa', seed, reference, 'solar/wind')
                row.update(method=method, reference_cost=reference,
                           qubits=[t['qubits'] for t in trace],
                           audit_s=private['rq_audit'](trace, ns['INDEX_BITS']))
                if row['valid']:
                    assert art['schedule'].shape == (24, 10)
                    used_renewables = gross - load
                    balance = art['result']['dispatch'].sum(axis=1) + used_renewables - gross
                    assert sw_np.max(sw_np.abs(balance)) < 1e-6
                rows.append(row)
                artifacts[(label, method, seed)] = art
            print(f'{label}, seed {seed}: comparisons complete', flush=True)
        for method in ('Full-day MILP', 'Original hybrid', 'Reduced hybrid'):
            group = [r for r in rows if r['window'] == label and r['method'] == method]
            valid = [r for r in group if r['valid']]
            qubits = [q for r in group for q in r['qubits']]
            summaries.append(dict(scenario=label, method=method,
                valid_days=f'{len(valid)}/{len(group)}', milp_cost=reference,
                mean_valid_gap_pct=float(sw_np.mean([r['gap_pct'] for r in valid])) if valid else None,
                mean_valid_cost=float(sw_np.mean([r['cost'] for r in valid])) if valid else None,
                mean_extra_dollars=float(sw_np.mean([r['cost']-reference for r in valid])) if valid else None,
                min_qubits=min(qubits, default=0), max_qubits=max(qubits, default=0),
                mean_qubits=float(sw_np.mean(qubits)) if qubits else 0,
                median_total_s=float(sw_np.median([r['total_s'] for r in group])),
                quantum_job_credits=0))
    helper['nd_table'](summaries)

    # Bars show actual computed results. Dots show seed variability, NOT confidence
    # intervals. Qubit whiskers are observed min/max, NOT uncertainty estimates.
    labels = [s[0] for s in scenarios]
    x = sw_np.arange(len(labels))
    fig, axes = sw_plt.subplots(2, 2, figsize=(14, 9))
    colors = {'Full-day MILP':'#374151', 'Original hybrid':'#d97706', 'Reduced hybrid':'#2563eb'}
    for j, method in enumerate(colors):
        selected_rows = [next(r for r in summaries if r['scenario']==label and r['method']==method) for label in labels]
        pos = x + (j-1)*0.25
        cost = [r['mean_valid_cost']/1000 if r['mean_valid_cost'] is not None else sw_np.nan for r in selected_rows]
        axes[0,1].bar(pos, cost, width=.24, color=colors[method], label=method)
        times = [r['median_total_s'] for r in selected_rows]
        axes[1,1].bar(pos, times, width=.24, color=colors[method], label=method)
        if method == 'Full-day MILP':
            continue
        pos = x + (-.18 if method=='Original hybrid' else .18)
        gaps = [r['mean_valid_gap_pct'] if r['mean_valid_gap_pct'] is not None else sw_np.nan for r in selected_rows]
        bars = axes[0,0].bar(pos, gaps, width=.34, color=colors[method], label=method)
        axes[0,0].bar_label(bars, labels=[f'{g:.3f}%' for g in gaps], padding=4, fontsize=8)
        for k,label in enumerate(labels):
            sample = [r['gap_pct'] for r in rows if r['window']==label and r['method']==method and r['valid']]
            axes[0,0].scatter(pos[k]+sw_np.linspace(-.06,.06,len(sample)), sample, s=16, color='black', zorder=3)
        means = sw_np.array([r['mean_qubits'] for r in selected_rows])
        lows = means-sw_np.array([r['min_qubits'] for r in selected_rows])
        highs = sw_np.array([r['max_qubits'] for r in selected_rows])-means
        axes[1,0].bar(pos, means, width=.34, color=colors[method], label=method,
                      yerr=sw_np.vstack([lows,highs]), capsize=4)
    axes[0,0].axhline(0, color=colors['Full-day MILP'], linewidth=1)
    axes[0,0].set(title='Gap to each scenario’s MILP (dots = seeds)', ylabel='Mean valid daily cost gap (%)')
    axes[0,1].set(title='Renewables lower total operating cost', ylabel='Mean valid daily cost ($ thousands)')
    axes[1,0].set(title='Circuit size (whiskers = observed range)', ylabel='Mean logical qubits per circuit', ylim=(0,15))
    axes[1,1].set(title='Measured local runtime (includes MILP and tuning)', ylabel='Median seconds per daily schedule')
    for ax in axes.flat:
        ax.set_xticks(x, ['No renewables\nhours 13–17', 'Solar + wind\nhours 13–17', 'Solar + wind\ntop net hours'])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('SOLAR + WIND — original and reduced-qubit hourly hybrids', fontsize=14)
    fig.tight_layout()
    sw_plt.show()
    print('Interpretation: smaller circuits do not guarantee smaller scheduling gaps. Compare dollars AND percentages.')
    print('Top net-load hours are a heuristic, not a proven best window. One forecast day and six seeds do not establish generality.')
    print('No QPU jobs submitted. Quantum-job credits = 0; subscription/AI charges are not allocated here.')
    print('Existing limits remain: approximate hourly QUBO, 3 slack bits, no ramp/forecast-error constraints in this experiment.')
    return dict(summary=summaries, runs=rows, artifacts=artifacts, references=references,
                gross=gross, net=net, solar=solar, wind=wind, reserve=reserve,
                settings=dict(fleet=10,seeds=SW_SEEDS,hours=SW_HOURS,top_hours=top_hours,shots=SW_SHOTS,layers=1,slack=3))

def run_solar_wind_window_study():
    from check import DATA
    began = ww_clock()
    assert not set(WW_SELECTION_SEEDS) & set(WW_VALIDATION_SEEDS)
    assert set(WW_LENGTHS) == {4, 5, 6}
    assert len(set(WW_SELECTION_SEEDS)) == len(WW_SELECTION_SEEDS)
    assert len(set(WW_VALIDATION_SEEDS)) == len(WW_VALIDATION_SEEDS)
    private = reduced_context(WW_SHOTS)
    helper = private['rq_helpers'](10)
    original = helper['nd_make_namespace']()
    fleet, gross, reserve = original['plants'], original['demand'].copy(), original['reserve'].copy()
    with open(DATA / 'renewables.csv', newline='') as f:
        forecast = list(ww_csv.DictReader(f))
    assert [int(r['hour']) for r in forecast] == list(range(24))
    scale = gross.max() / max(float(r['system_load_mw']) for r in forecast)
    solar = ww_np.array([float(r['solar_mw']) for r in forecast]) * scale
    wind = ww_np.array([float(r['wind_mw']) for r in forecast]) * scale
    net = gross - solar - wind
    assert ww_np.all(net >= 0), 'Negative net load needs explicit curtailment.'

    def namespace():
        ns = helper['nd_make_namespace']()
        ns.update(demand=net.copy(), reserve=reserve.copy())
        return ns

    ns = namespace()
    stats = helper['nd_instrument'](ns)
    ref, ref_art = helper['nd_one_run'](ns, stats, 'FULL MILP', (), 'milp', 0, None, 'reference')
    assert ref['valid'] and ref['milp_domain_optimal']
    reference = ref['cost']
    # Guard against silently treating a curtailment scenario as fixed net load.
    install_renewable_solver(ns)
    curt_cost, _, _, spill, _ = ns['solve_milp_renewables'](fleet, gross, reserve, solar+wind)
    assert ww_np.abs(spill).sum() < 1e-6 and abs(curt_cost-reference) < .01

    # Screening has repeated identical QUBOs/starting angles across seeds/windows.
    # Cache only the deterministic optimizer output, never sampled choices, locks,
    # or schedules. This preserves the algorithm and all 200 independent samples.
    # Keys include the full reduced QUBO, offset, angles, depth and iteration limit.
    # Every finalist/validation run DISABLES this cache. Report only those timings
    # as per-schedule runtime; screening speed is an experiment-running convenience.
    cache = ww_OrderedDict()
    cache_counts = dict(hits=0, misses=0)
    def install_cache(ns):
        original_run = ns['run_qaoa']
        def cached(Q, offset, p=1, start=None, maxiter=60):
            angles_key = None if start is None else ww_np.asarray(start, dtype=float).tobytes()
            key = (Q.shape, ww_np.asarray(Q, dtype=float).tobytes(), float(offset), p, angles_key, maxiter)
            if key in cache:
                cache_counts['hits'] += 1
                cache.move_to_end(key)
                a, b = cache[key]
                return a.copy(), b.copy()
            cache_counts['misses'] += 1
            a, b = original_run(Q, offset, p=p, start=start, maxiter=maxiter)
            cache[key] = (a.copy(), b.copy())
            if len(cache) > 128:
                cache.popitem(last=False)
            return a, b
        ns['run_qaoa'] = cached

    rows, artifacts, hourly = [], {}, []
    def label(start, length):
        return f'{start:02d}:00–{start+length:02d}:00'

    def run(start, length, seed, phase, cached=False, milp=False):
        ns = namespace()
        if cached:
            install_cache(ns)
        trace = [] if milp else private['rq_adapt'](ns, True)
        stats = helper['nd_instrument'](ns)
        selected = tuple(range(start, start+length)) if not milp else ()
        row, art = helper['nd_one_run'](ns, stats, 'FULL MILP' if milp else label(start,length),
                                      selected, 'milp' if milp else 'qaoa', seed, reference, phase)
        row.update(start=start, length=length, screening_cache_enabled=cached,
                   min_qubits=min((t['qubits'] for t in trace), default=0),
                   max_qubits=max((t['qubits'] for t in trace), default=0),
                   mean_qubits=float(ww_np.mean([t['qubits'] for t in trace])) if trace else 0,
                   audit_s=private['rq_audit'](trace, ns['INDEX_BITS']))
        if row['valid']:
            assert art['schedule'].shape == (24, 10)
            balance = art['result']['dispatch'].sum(axis=1) + solar + wind - gross
            assert ww_np.max(ww_np.abs(balance)) < 1e-6
            assert milp or len(trace) == length
        for hour, t in zip(selected, trace):
            hourly.append(dict(phase=phase,start=start,length=length,seed=seed,hour=hour,
                               qubits=t['qubits'],forced=t['forced'],logical_rzz=t['logical_rzz']))
        rows.append(row)
        artifacts[(phase,start,length,seed)] = art
        return row, art

    def summarize(phase, start, length):
        group = [r for r in rows if r['phase']==phase and r['start']==start and r['length']==length]
        valid = [r for r in group if r['valid']]
        gaps = [r['gap_pct'] for r in valid]
        return dict(phase=phase,window='FULL MILP' if length==0 else label(start,length),
                    start=start,length=length,valid_days=len(valid),runs=len(group),
                    mean_valid_gap_pct=float(ww_np.mean(gaps)) if gaps else None,
                    worst_valid_gap_pct=max(gaps) if gaps else None,
                    mean_valid_cost=float(ww_np.mean([r['cost'] for r in valid])) if valid else None,
                    min_qubits=min((r['min_qubits'] for r in group),default=0),
                    max_qubits=max((r['max_qubits'] for r in group),default=0),
                    median_total_s=float(ww_np.median([r['total_s'] for r in group])),
                    timing_cached=any(r['screening_cache_enabled'] for r in group))

    candidates = [(start,length) for length in WW_LENGTHS for start in range(25-length)]
    assert len(candidates)==60 and len(set(candidates))==60
    print('SOLAR + WIND — exhaustive contiguous 4/5/6-hour window study')
    print(f'10 plants; p=1; {WW_SHOTS} shots/hour; own full-day MILP ${reference:,.2f}.')
    print('Solar/wind are fixed forecasts; reserve stays based on gross demand. Existing reduced-qubit method only.')
    print(f'Screen 60 windows on seeds {WW_SELECTION_SEEDS}; validate top {WW_TOP_PER_LENGTH} per length on {WW_VALIDATION_SEEDS}.')
    print('Only all-valid windows qualify. Invalid runs remain visible; invalid costs never enter gap means.')
    selection = []
    for k,(start,length) in enumerate(candidates,1):
        for seed in WW_SELECTION_SEEDS:
            run(start,length,seed,'selection',cached=WW_SCREEN_CACHE)
        s = summarize('selection',start,length)
        selection.append(s)
        gap = 'n/a' if s['mean_valid_gap_pct'] is None else f"{s['mean_valid_gap_pct']:.4f}%"
        print(f"Screen {k:02d}/60: {length}h {s['window']}, valid {s['valid_days']}/{s['runs']}, mean gap {gap}",flush=True)

    finalists, chosen = [], {}
    for length in WW_LENGTHS:
        eligible = sorted([s for s in selection if s['length']==length and s['valid_days']==s['runs']],
                          key=lambda s:(s['mean_valid_gap_pct'],s['worst_valid_gap_pct'],s['start']))
        if not eligible:
            print(f'No all-valid {length}-hour candidate; no winner declared.')
            continue
        chosen[length] = (eligible[0]['start'],length)
        finalists.extend((s['start'],length) for s in eligible[:WW_TOP_PER_LENGTH])
    # Keep the existing 13–17 schedule and the earlier top-net-hours comparison,
    # regardless of whether they rank well. These controls are not chosen on validation.
    top_net = sorted(ww_np.argsort(-net)[:5].tolist())
    controls = [(13,5)]
    if top_net == list(range(top_net[0],top_net[0]+5)):
        controls.append((top_net[0],5))
    finalists = list(dict.fromkeys(finalists+controls))
    print('Finalists FIXED before validation:',[label(s,l) for s,l in finalists],flush=True)
    print('Screening winners by length:',{l:label(*w) for l,w in chosen.items()},flush=True)
    print('Selection table (timing_cached=True rows are not cold runtime measurements):')
    helper['nd_table'](selection)

    # Explicit regression check: rerun every finalist on ALL screening seeds with
    # caching OFF. Exact schedule agreement checks the optimization reuse, not just cost.
    for start,length in finalists:
        for seed in WW_SELECTION_SEEDS:
            check_row, check_art = run(start,length,seed,'uncached_recheck')
            old = next(r for r in rows if r['phase']=='selection' and (r['start'],r['length'],r['seed'])==(start,length,seed))
            previous = artifacts[('selection',start,length,seed)]
            assert check_row['valid']==old['valid']
            if check_art['schedule'] is not None:
                assert ww_np.array_equal(check_art['schedule'],previous['schedule'])
                assert abs(check_row['cost']-old['cost']) < 1e-6
        print(f'Uncached reproduction passed: {label(start,length)}',flush=True)
    cache.clear()

    validation = []
    for k,(start,length) in enumerate(finalists,1):
        for seed in WW_VALIDATION_SEEDS:
            run(start,length,seed,'validation')
        s = summarize('validation',start,length)
        s['selection_winner'] = chosen.get(length)==(start,length)
        validation.append(s)
        print(f"Validate {k}/{len(finalists)}: {s['window']}, valid {s['valid_days']}/{s['runs']}, mean gap {s['mean_valid_gap_pct']}",flush=True)
    for seed in WW_VALIDATION_SEEDS:
        run(0,0,seed,'validation',milp=True)
    baseline = summarize('validation',0,0)
    print('\nFRESH-SEED VALIDATION (no tuning cache; lower valid gap is better; inspect valid counts)')
    helper['nd_table'](validation+[baseline])
    print('\nPRESELECTED WINNERS — each chosen using screening seeds only')
    winners = [s for s in validation if s['selection_winner']]
    helper['nd_table'](winners)
    # Paired differences use the SAME validation seeds; negative means cheaper
    # than our existing 13:00–18:00 window. Do not silently drop failed pairs.
    paired = []
    for length, (start,_) in chosen.items():
        for seed in WW_VALIDATION_SEEDS:
            a=next(r for r in rows if r['phase']=='validation' and (r['start'],r['length'],r['seed'])==(start,length,seed))
            b=next(r for r in rows if r['phase']=='validation' and (r['start'],r['length'],r['seed'])==(13,5,seed))
            paired.append(dict(window=label(start,length),length=length,seed=seed,
                               candidate_valid=a['valid'],original_valid=b['valid'],
                               gap_change_percentage_points=a['gap_pct']-b['gap_pct'] if a['valid'] and b['valid'] else None))
    helper['nd_table'](paired)

    # All 60 cells in the heatmap are tested; impossible start/length pairs are gray.
    # Invalid windows are excluded from ranking and marked with a red cross.
    fig,axes=ww_plt.subplots(2,2,figsize=(16,10))
    matrix=ww_np.full((3,21),ww_np.nan)
    for s in selection:
        if s['valid_days']==s['runs']:
            matrix[list(WW_LENGTHS).index(s['length']),s['start']]=s['mean_valid_gap_pct']
    cmap=ww_plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    im=axes[0,0].imshow(matrix,aspect='auto',cmap=cmap)
    for s in selection:
        if s['valid_days']<s['runs']:
            axes[0,0].plot(s['start'],list(WW_LENGTHS).index(s['length']),'rx')
    axes[0,0].set(xticks=range(21),yticks=range(3),yticklabels=[f'{l} hours' for l in WW_LENGTHS],
                  xlabel='Start hour (end is exclusive)',title='All 60 windows: selection mean gap')
    fig.colorbar(im,ax=axes[0,0],label='Gap (%)')

    positions=ww_np.arange(len(validation));width=.38
    screen_means=[next(t['mean_valid_gap_pct'] for t in selection if (t['start'],t['length'])==(s['start'],s['length'])) for s in validation]
    validate_means=[s['mean_valid_gap_pct'] if s['mean_valid_gap_pct'] is not None else ww_np.nan for s in validation]
    axes[0,1].bar(positions-width/2,screen_means,width,color='#94a3b8',label='Selection (3 seeds)')
    axes[0,1].bar(positions+width/2,validate_means,width,color='#2563eb',label='Validation (8 fresh seeds)')
    for i,s in enumerate(validation):
        values=[r['gap_pct'] for r in rows if r['phase']=='validation' and (r['start'],r['length'])==(s['start'],s['length']) and r['valid']]
        axes[0,1].scatter(i+width/2+ww_np.linspace(-.06,.06,len(values)),values,color='black',s=12,zorder=3)
    tick_labels=[s['window']+f"\n{s['length']}h"+(' *' if s['selection_winner'] else '') for s in validation]
    axes[0,1].set(xticks=positions,xticklabels=tick_labels,ylabel='Mean valid cost gap (%)',
                  title='Finalists: * = chosen before validation; dots = fresh seeds')
    axes[0,1].tick_params(axis='x',labelrotation=50,labelsize=8);axes[0,1].legend(fontsize=8)

    curves=[s for s in validation if s['selection_winner'] or (s['start'],s['length'])==(13,5)]
    xx=ww_np.arange(len(curves));means=[];lo=[];hi=[]
    for s in curves:
        qs=[q['qubits'] for q in hourly if q['phase']=='validation' and (q['start'],q['length'])==(s['start'],s['length'])]
        means.append(ww_np.mean(qs));lo.append(min(qs));hi.append(max(qs))
    means=ww_np.array(means)
    axes[1,0].bar(xx,means,color='#0d9488',yerr=ww_np.array([means-ww_np.array(lo),ww_np.array(hi)-means]),capsize=4)
    axes[1,0].axhline(13,color='black',linestyle='--',label='Unreduced: 13 qubits')
    axes[1,0].set(xticks=xx,xticklabels=[s['window']+f"\n{s['length']}h" for s in curves],
                  ylabel='Qubits per hourly circuit',ylim=(0,15),title='Circuit size (observed min–max, fresh seeds)')
    axes[1,0].legend(fontsize=8)
    timings=curves+[baseline]
    axes[1,1].bar(range(len(timings)),[s['median_total_s'] for s in timings],color='#7c3aed')
    axes[1,1].set(xticks=range(len(timings)),xticklabels=[s['window'] for s in timings],
                  ylabel='Median local seconds',title='Fresh-seed runtime: tuning + MILP + validation; no cache')
    for ax in axes.flat:
        if ax is not axes[0,0]: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('SOLAR + WIND — 4/5/6-hour QAOA windows; one circuit per selected hour',fontsize=14)
    fig.tight_layout();ww_plt.show()

    print('All reduced-QUBO substitutions audited. Valid schedules passed demand, reserve and min up/down checks.')
    print('Best means best among tested contiguous windows on this day; not proof across other days or arbitrary hour subsets.')
    print('Compare within the same duration first: a 4-hour window uses fewer QAOA jobs/shots than a 6-hour window.')
    print('The separate future-cost prototype is NOT included. Existing MILP stretches already plan through the remaining day.')
    print('Quantum-job credits spent: 0 (local ideal simulation). No hardware/noise or quantum speed advantage measured.')
    print('Screening tuning reuse:',cache_counts,'; finalist schedules reproduced exactly without caching.')
    print(f'Total study time {ww_clock()-began:.1f} s; {len(rows)} schedule runs plus the reference/curtailment checks.')
    return dict(selection=selection,validation=validation,winners=winners,paired=paired,baseline=baseline,
                runs=rows,artifacts=artifacts,hourly=hourly,reference_cost=reference,
                forecast=dict(gross=gross,net=net,solar=solar,wind=wind,reserve=reserve),
                settings=dict(lengths=WW_LENGTHS,selection_seeds=WW_SELECTION_SEEDS,
                              validation_seeds=WW_VALIDATION_SEEDS,shots=WW_SHOTS,finalists=finalists,
                              chosen_before_validation=chosen,screen_cache=WW_SCREEN_CACHE,cache_counts=cache_counts))

def run_reduced_qubit_hybrid_hourly_window_study():
    from check import DATA
    began = aw_clock()
    assert not set(AW_SELECTION_SEEDS) & set(AW_VALIDATION_SEEDS)
    assert tuple(AW_LENGTHS) == tuple(range(1,25))
    assert len(set(AW_SELECTION_SEEDS)) == len(AW_SELECTION_SEEDS)
    assert len(set(AW_VALIDATION_SEEDS)) == len(AW_VALIDATION_SEEDS)
    private = reduced_context(AW_SHOTS)
    helper = private['rq_helpers'](10)
    original = helper['nd_make_namespace']()
    fleet, gross, reserve = original['plants'], original['demand'].copy(), original['reserve'].copy()
    with open(DATA / 'renewables.csv', newline='') as f:
        forecast = list(aw_csv.DictReader(f))
    assert [int(r['hour']) for r in forecast] == list(range(24))
    scale = gross.max() / max(float(r['system_load_mw']) for r in forecast)
    solar = aw_np.array([float(r['solar_mw']) for r in forecast]) * scale
    wind = aw_np.array([float(r['wind_mw']) for r in forecast]) * scale
    net = gross - solar - wind
    assert aw_np.all(net >= 0), 'Negative net load needs explicit curtailment.'

    def namespace():
        ns = helper['nd_make_namespace']()
        ns.update(demand=net.copy(), reserve=reserve.copy())
        return ns

    ns = namespace()
    stats = helper['nd_instrument'](ns)
    ref, ref_art = helper['nd_one_run'](ns, stats, 'FULL MILP', (), 'milp', 0, None, 'reference')
    assert ref['valid'] and ref['milp_domain_optimal']
    reference = ref['cost']
    # Guard against silently treating a curtailment scenario as fixed net load.
    install_renewable_solver(ns)
    curt_cost, _, _, spill, _ = ns['solve_milp_renewables'](fleet, gross, reserve, solar+wind)
    assert aw_np.abs(spill).sum() < 1e-6 and abs(curt_cost-reference) < .01

    # Screening has repeated identical QUBOs/starting angles across seeds/windows.
    # Cache only the deterministic optimizer output, never sampled choices, locks,
    # or schedules. This preserves the algorithm and all 200 independent samples.
    # Keys include the full reduced QUBO, offset, angles, depth and iteration limit.
    # Validation uses fresh sampling seeds with the same exact tuning cache. This
    # does NOT reuse sampled decisions or read the MILP reference schedule. Timing
    # is reported only from separate UNCACHED repeats; their schedules must match.
    cache = aw_OrderedDict()
    cache_counts = dict(hits=0, misses=0)
    def install_cache(ns):
        original_run = ns['run_qaoa']
        def cached(Q, offset, p=1, start=None, maxiter=60):
            angles_key = None if start is None else aw_np.asarray(start, dtype=float).tobytes()
            key = (Q.shape, aw_np.asarray(Q, dtype=float).tobytes(), float(offset), p, angles_key, maxiter)
            if key in cache:
                cache_counts['hits'] += 1
                cache.move_to_end(key)
                a, b = cache[key]
                return a.copy(), b.copy()
            cache_counts['misses'] += 1
            a, b = original_run(Q, offset, p=p, start=start, maxiter=maxiter)
            cache[key] = (a.copy(), b.copy())
            if len(cache) > 128:
                cache.popitem(last=False)
            return a, b
        ns['run_qaoa'] = cached

    rows, artifacts, hourly = [], {}, []
    def label(start, length):
        return f'{start:02d}:00–{start+length:02d}:00'

    def run(start, length, seed, phase, cached=False, milp=False):
        ns = namespace()
        if cached:
            install_cache(ns)
        trace = [] if milp else private['rq_adapt'](ns, True)
        stats = helper['nd_instrument'](ns)
        selected = tuple(range(start, start+length)) if not milp else ()
        row, art = helper['nd_one_run'](ns, stats, 'FULL MILP' if milp else label(start,length),
                                      selected, 'milp' if milp else 'qaoa', seed, reference, phase)
        row.update(start=start, length=length, screening_cache_enabled=cached,
                   min_qubits=min((t['qubits'] for t in trace), default=0),
                   max_qubits=max((t['qubits'] for t in trace), default=0),
                   mean_qubits=float(aw_np.mean([t['qubits'] for t in trace])) if trace else 0,
                   audit_s=private['rq_audit'](trace, ns['INDEX_BITS']))
        if row['valid']:
            assert art['schedule'].shape == (24, 10)
            balance = art['result']['dispatch'].sum(axis=1) + solar + wind - gross
            assert aw_np.max(aw_np.abs(balance)) < 1e-6
            assert milp or len(trace) == length
        for hour, t in zip(selected, trace):
            hourly.append(dict(phase=phase,start=start,length=length,seed=seed,hour=hour,
                               qubits=t['qubits'],forced=t['forced'],logical_rzz=t['logical_rzz']))
        rows.append(row)
        artifacts[(phase,start,length,seed)] = art
        return row, art

    def summarize(phase, start, length):
        group = [r for r in rows if r['phase']==phase and r['start']==start and r['length']==length]
        valid = [r for r in group if r['valid']]
        gaps = [r['gap_pct'] for r in valid]
        return dict(phase=phase,window='FULL MILP' if length==0 else label(start,length),
                    start=start,length=length,valid_days=len(valid),runs=len(group),
                    mean_valid_gap_pct=float(aw_np.mean(gaps)) if gaps else None,
                    worst_valid_gap_pct=max(gaps) if gaps else None,
                    mean_valid_cost=float(aw_np.mean([r['cost'] for r in valid])) if valid else None,
                    min_qubits=min((r['min_qubits'] for r in group),default=0),
                    max_qubits=max((r['max_qubits'] for r in group),default=0),
                    median_total_s=float(aw_np.median([r['total_s'] for r in group])),
                    timing_cached=any(r['screening_cache_enabled'] for r in group))

    # Group by start hour, then grow the window. This maximizes exact prefix-tuning
    # reuse; every full schedule is still independently sampled and physically checked.
    candidates = [(start,length) for start in range(24) for length in AW_LENGTHS if start+length<=24]
    assert len(candidates)==300 and len(set(candidates))==300
    print('REDUCED QUBIT HYBRID HOURLY METHOD — ALL 300 CONTIGUOUS WINDOWS')
    print(f'10 plants with solar/wind; p=1; {AW_SHOTS} shots per selected hour; MILP reference ${reference:,.2f}.')
    print('One circuit per selected hour, even for the full-day 24-hour candidate.')
    print(f'Screening seeds {AW_SELECTION_SEEDS}; fresh validation seeds {AW_VALIDATION_SEEDS}.')
    print('A window must be valid in every screening seed to qualify as a winner. Failures remain visible.')
    selection=[]
    for k,(start,length) in enumerate(candidates,1):
        for seed in AW_SELECTION_SEEDS:
            run(start,length,seed,'selection',cached=AW_TUNING_CACHE)
        s=summarize('selection',start,length)
        selection.append(s)
        if k%10==0 or length==24:
            gap='n/a' if s['mean_valid_gap_pct'] is None else f"{s['mean_valid_gap_pct']:.4f}%"
            print(f"Screen {k}/300: {s['window']} ({length} hourly circuits), valid {s['valid_days']}/{s['runs']}, gap {gap}",flush=True)

    def rank(s):
        # Explicit tie-break: lower mean, then lower worst gap, then fewer quantum
        # hours, then earlier start. This is a COST-gap objective, not quantum advantage.
        return (s['mean_valid_gap_pct'],s['worst_valid_gap_pct'],s['length'],s['start'])
    eligible=sorted([s for s in selection if s['valid_days']==s['runs']],key=rank)
    assert eligible, 'No all-valid nonempty QAOA window; no best window can be declared.'
    chosen={}
    for length in AW_LENGTHS:
        group=[s for s in eligible if s['length']==length]
        if group:
            chosen[length]=(group[0]['start'],length)
    overall=(eligible[0]['start'],eligible[0]['length'])
    finalists=list(chosen.values())+[(s['start'],s['length']) for s in eligible[:AW_TOP_OVERALL]]
    # Include the reference window, restricted-search winners, and all-day hourly
    # QAOA, even if a control failed screening or did not rank near the best.
    controls=[(13,5),(15,4),(15,5),(15,6),(0,24)]
    finalists=sorted(set(finalists+controls))
    print('Overall choice FIXED before validation:',label(*overall),'duration',overall[1],flush=True)
    print('Best screening window per duration:',{l:label(*w) for l,w in chosen.items()},flush=True)
    print('Finalists/controls FIXED before validation:',[label(s,l) for s,l in finalists],flush=True)
    print('ALL 300 SCREENING RESULTS: cached timings are experiment wall times, not standalone runtime.')
    helper['nd_table'](sorted(selection,key=lambda s:(s['length'],s['start'])))
    # Clear between phases. Seeds 201–208 have not been used to select the finalists.
    cache.clear()
    validation=[]
    for k,(start,length) in enumerate(finalists,1):
        for seed in AW_VALIDATION_SEEDS:
            run(start,length,seed,'validation',cached=AW_TUNING_CACHE)
        s=summarize('validation',start,length)
        s.update(selection_winner=chosen.get(length)==(start,length),overall_preselected=(start,length)==overall)
        validation.append(s)
        print(f"Validate {k}/{len(finalists)}: {s['window']} ({length}h), valid {s['valid_days']}/{s['runs']}, mean gap {s['mean_valid_gap_pct']}",flush=True)

    # Independently repeat the preselected overall winner, original 5-hour window,
    # and full-day QAOA without optimizer-output reuse. This provides honest cold
    # runtimes and an exact check of cached schedules for short/medium/full-day cases.
    # These are benchmark repeats, not extra independent seed evidence.
    benchmark_keys=list(dict.fromkeys([overall,(13,5),(0,24)]))
    benchmarks=[]
    for start,length in benchmark_keys:
        for seed in AW_VALIDATION_SEEDS[:3]:
            row,art=run(start,length,seed,'cold_benchmark')
            old=next(r for r in rows if r['phase']=='validation' and (r['start'],r['length'],r['seed'])==(start,length,seed))
            previous=artifacts[('validation',start,length,seed)]
            assert row['valid']==old['valid']
            if art['schedule'] is not None:
                assert aw_np.array_equal(art['schedule'],previous['schedule'])
                assert abs(row['cost']-old['cost'])<1e-6
            else:
                assert previous['schedule'] is None
        benchmarks.append(summarize('cold_benchmark',start,length))
        print('Exact uncached reproduction passed:',label(start,length),flush=True)
    cache.clear()
    for seed in AW_VALIDATION_SEEDS:
        run(0,0,seed,'validation',milp=True)
    baseline=summarize('validation',0,0)
    print('\nFRESH-SEED VALIDATION: cached timing columns are NOT standalone runtime')
    helper['nd_table'](validation+[baseline])
    print('\nUNCACHED RUNTIME BENCHMARKS (3 repeated seeds; MILP has 8 repeats)')
    helper['nd_table'](benchmarks+[baseline])
    best=next(s for s in validation if (s['start'],s['length'])==overall)
    winners=[s for s in validation if s['selection_winner']]
    # Report the quantum-hour/cost tradeoff explicitly. Thresholds below are just
    # reporting guides, NOT guarantees or selection inputs. These summaries use
    # validation outcomes and would need another independent test before adoption.
    exposure=[]
    for threshold in (0.1,0.5,1.0):
        acceptable=[s for s in validation if s['valid_days']==s['runs'] and s['mean_valid_gap_pct']<=threshold]
        if acceptable:
            s=min(acceptable,key=lambda s:(-s['length'],s['mean_valid_gap_pct'],s['start']))
            exposure.append(dict(mean_gap_threshold_pct=threshold,window=s['window'],quantum_hours=s['length'],mean_valid_gap_pct=s['mean_valid_gap_pct']))
    print('\nPRESELECTED OVERALL COST-GAP CHOICE, evaluated on fresh seeds:')
    helper['nd_table']([best])
    print('\nEXPLORATORY: most quantum hours among validated candidates below stated mean-gap thresholds')
    helper['nd_table'](exposure)

    # Six panels separate window coverage, duration tradeoffs, single-hour results,
    # validation, circuit size and genuinely uncached runtime.
    fig,axes=aw_plt.subplots(3,2,figsize=(17,17))
    matrix=aw_np.full((24,24),aw_np.nan)
    for s in selection:
        if s['valid_days']==s['runs']:
            matrix[s['length']-1,s['start']]=s['mean_valid_gap_pct']
    cmap=aw_plt.get_cmap('viridis').copy();cmap.set_bad('#eeeeee')
    im=axes[0,0].imshow(matrix,aspect='auto',cmap=cmap,origin='lower')
    for s in selection:
        if s['valid_days']<s['runs']:
            axes[0,0].plot(s['start'],s['length']-1,'rx',markersize=3)
    axes[0,0].set(xticks=range(0,24,2),yticks=range(0,24,2),yticklabels=range(1,25,2),
                  xlabel='Start hour',ylabel='Number of separate hourly QAOA circuits',title='All 300 windows: selection gap; red × = some invalid')
    fig.colorbar(im,ax=axes[0,0],label='Mean valid gap (%)')
    ordered=sorted(winners,key=lambda s:s['length'])
    x=[s['length'] for s in ordered]
    sg=[next(t['mean_valid_gap_pct'] for t in selection if (t['start'],t['length'])==(s['start'],s['length'])) for s in ordered]
    vg=[s['mean_valid_gap_pct'] if s['valid_days']==s['runs'] else aw_np.nan for s in ordered]
    axes[0,1].plot(x,sg,'o-',color='#94a3b8',label='Best screening candidate at each duration')
    axes[0,1].plot(x,vg,'o-',color='#2563eb',label='Same candidates, fresh seeds (all-valid only)')
    axes[0,1].set(xlabel='Quantum hours per day (one circuit each)',ylabel='Mean gap (%)',xticks=range(1,25,2),title='More quantum hours versus daily cost gap')
    axes[0,1].legend(fontsize=8)

    singles=sorted([s for s in selection if s['length']==1],key=lambda s:s['start'])
    single_gaps=[s['mean_valid_gap_pct'] if s['valid_days']==s['runs'] else aw_np.nan for s in singles]
    axes[1,0].bar(range(24),single_gaps,color='#0d9488')
    axes[1,0].set(xticks=range(24),xlabel='The one hour given to QAOA (other 23 hours use MILP)',ylabel='Selection mean gap (%)',title='Every single-hour option — screening results')
    axes[1,0].tick_params(axis='x',labelsize=8)

    comparison_keys=list(dict.fromkeys([overall,(15,4),(15,5),(13,5),(0,24)]))
    comparisons=[next(s for s in validation if (s['start'],s['length'])==key) for key in comparison_keys]+[baseline]
    values=[s['mean_valid_gap_pct'] if s['mean_valid_gap_pct'] is not None else aw_np.nan for s in comparisons]
    positions=aw_np.arange(len(comparisons))
    bars=axes[1,1].bar(positions,values,color=['#2563eb']*(len(comparisons)-1)+['#374151'])
    axes[1,1].bar_label(bars,labels=[f'{v:.3f}%' if aw_np.isfinite(v) else 'no valid runs' for v in values],padding=3,fontsize=8)
    for i,s in enumerate(comparisons):
        gaps=[r['gap_pct'] for r in rows if r['phase']=='validation' and (r['start'],r['length'])==(s['start'],s['length']) and r['valid']]
        axes[1,1].scatter(i+aw_np.linspace(-.09,.09,len(gaps)),gaps,color='black',s=12,zorder=3)
    axes[1,1].set(xticks=positions,xticklabels=[s['window']+f"\n{s['length']}h; valid {s['valid_days']}/{s['runs']}" for s in comparisons],
                  ylabel='Mean VALID gap (%) — inspect validity counts',title='Fresh seeds: selected window, old settings and full day')
    axes[1,1].tick_params(axis='x',labelrotation=25,labelsize=8)

    means=[];low=[];high=[]
    for s in ordered:
        qs=[q['qubits'] for q in hourly if q['phase']=='validation' and (q['start'],q['length'])==(s['start'],s['length'])]
        means.append(float(aw_np.mean(qs)));low.append(min(qs));high.append(max(qs))
    axes[2,0].plot(x,means,'o-',color='#0d9488',label='Mean qubits per hourly circuit')
    axes[2,0].fill_between(x,low,high,color='#0d9488',alpha=.2,label='Observed min–max')
    axes[2,0].axhline(13,color='black',linestyle='--',label='Unreduced 10 plants + 3 slack')
    axes[2,0].set(xticks=range(1,25,2),xlabel='Quantum hours in preselected window',ylabel='Qubits per circuit (NOT multiplied by hours)',title='Circuits stay hourly at every window length',ylim=(0,15))
    axes[2,0].legend(fontsize=8)
    timing=benchmarks+[baseline]
    axes[2,1].bar(range(len(timing)),[s['median_total_s'] for s in timing],color='#7c3aed')
    axes[2,1].set(xticks=range(len(timing)),xticklabels=[s['window'] for s in timing],ylabel='Median local seconds per schedule',title='Uncached runtime: exact schedule reproduction checked')
    for ax in axes.flat:
        if ax is not axes[0,0]:
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('REDUCED QUBIT HYBRID HOURLY METHOD — all 1–24-hour windows with solar + wind',fontsize=15)
    fig.tight_layout();aw_plt.show()

    print('Best COST gap is not proof of quantum advantage: MILP is the zero-gap baseline; few QAOA hours leave most work to MILP.')
    print('The all-day case is 24 sequential hourly circuits, NOT one 24-hour circuit.')
    print('All 300 contiguous windows were screened; arbitrary scattered subsets and other forecast days are outside this study.')
    print('Same constraints/objective as before: gross-demand reserve, min up/down locks, approximate QUBO, no added forecast uncertainty or ramps.')
    print('Every reduced QUBO passed energy-equivalence auditing. Invalid days are not accepted as low-cost results.')
    print('The future-cost scoring prototype remains separate. Quantum-job credits spent: 0 (local simulation only).')
    print('Tuning reuse statistics:',cache_counts,'; cold benchmark schedules matched cached results exactly.')
    print(f'Total study time {aw_clock()-began:.1f} s; {len(rows)} schedule executions plus reference/curtailment checks.')
    return dict(selection=selection,validation=validation,winners=winners,overall_preselected=best,
                exposure=exposure,benchmarks=benchmarks,baseline=baseline,runs=rows,artifacts=artifacts,
                hourly=hourly,reference_cost=reference,forecast=dict(gross=gross,net=net,solar=solar,wind=wind,reserve=reserve),
                settings=dict(method_name='reduced qubit hybrid hourly method',lengths=AW_LENGTHS,
                              selection_seeds=AW_SELECTION_SEEDS,validation_seeds=AW_VALIDATION_SEEDS,
                              shots=AW_SHOTS,finalists=finalists,chosen_before_validation=chosen,
                              overall_before_validation=overall,tuning_cache=AW_TUNING_CACHE,cache_counts=cache_counts))

def confirm_reduced_qubit_hybrid_hourly_windows(study, seeds=tuple(range(301,317))):
    import ast, csv, json
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from check import DATA
    assert not set(seeds) & (set(study['settings']['selection_seeds']) | set(study['settings']['validation_seeds']))
    private = reduced_context(study['settings']['shots'])
    helper=private['rq_helpers'](10)
    initial=helper['nd_make_namespace']()
    gross,reserve=initial['demand'].copy(),initial['reserve'].copy()
    with open(DATA/'renewables.csv',newline='') as f:
        forecast=list(csv.DictReader(f))
    assert [int(r['hour']) for r in forecast]==list(range(24))
    scale=gross.max()/max(float(r['system_load_mw']) for r in forecast)
    renewable=np.array([float(r['solar_mw'])+float(r['wind_mw']) for r in forecast])*scale
    net=gross-renewable
    def namespace():
        ns=helper['nd_make_namespace']()
        ns.update(demand=net.copy(),reserve=reserve.copy())
        return ns
    # Recheck the baseline so changed input files cannot silently reuse old costs.
    ns=namespace();stats=helper['nd_instrument'](ns)
    ref,_=helper['nd_one_run'](ns,stats,'FULL MILP',(),'milp',0,None,'confirmation reference')
    assert ref['valid'] and ref['milp_domain_optimal'] and abs(ref['cost']-study['reference_cost'])<.01
    reference=ref['cost']
    eligible=sorted([s for s in study['validation'] if s['valid_days']==s['runs']],
                    key=lambda s:(s['mean_valid_gap_pct'],s['worst_valid_gap_pct'],s['length'],s['start']))
    chosen=list(dict.fromkeys([(s['start'],s['length']) for s in eligible[:3]]+[(13,5)]))
    rows,artifacts,summaries=[],{},[]
    print('FINAL CONFIRMATION — candidates fixed from the earlier validation results:',chosen)
    print('New seeds:',seeds,'; no tuning cache; same forecast, constraints, shots and reduced-hourly algorithm.')
    for start,length in chosen:
        label=f'{start:02d}:00–{start+length:02d}:00'
        for seed in seeds:
            ns=namespace();trace=private['rq_adapt'](ns,True);stats=helper['nd_instrument'](ns)
            row,art=helper['nd_one_run'](ns,stats,label,tuple(range(start,start+length)),
                                       'qaoa',seed,reference,'confirmation')
            row.update(start=start,length=length,qubits=[t['qubits'] for t in trace],
                       audit_s=private['rq_audit'](trace,ns['INDEX_BITS']))
            if row['valid']:
                assert len(trace)==length
                assert np.max(np.abs(art['result']['dispatch'].sum(axis=1)+renewable-gross))<1e-6
            rows.append(row);artifacts[(start,length,seed)]=art
        group=[r for r in rows if (r['start'],r['length'])==(start,length)]
        valid=[r for r in group if r['valid']]
        qs=[q for r in group for q in r['qubits']]
        s=dict(window=label,start=start,length=length,valid_days=len(valid),runs=len(group),
               mean_valid_gap_pct=float(np.mean([r['gap_pct'] for r in valid])) if valid else None,
               worst_valid_gap_pct=max((r['gap_pct'] for r in valid),default=None),
               mean_valid_cost=float(np.mean([r['cost'] for r in valid])) if valid else None,
               min_qubits=min(qs,default=0),max_qubits=max(qs,default=0),
               median_total_s=float(np.median([r['total_s'] for r in group])))
        summaries.append(s)
        print(f"Confirm {label}: valid {len(valid)}/{len(group)}, mean gap {s['mean_valid_gap_pct']}",flush=True)
    helper['nd_table'](summaries)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    x=np.arange(len(summaries))
    gaps=[s['mean_valid_gap_pct'] if s['mean_valid_gap_pct'] is not None else np.nan for s in summaries]
    bars=axes[0].bar(x,gaps,color=['#2563eb']*(len(summaries)-1)+['#d97706'])
    axes[0].bar_label(bars,labels=[f'{g:.3f}%' for g in gaps],padding=3,fontsize=9)
    for i,s in enumerate(summaries):
        values=[r['gap_pct'] for r in rows if (r['start'],r['length'])==(s['start'],s['length']) and r['valid']]
        axes[0].scatter(i+np.linspace(-.1,.1,len(values)),values,color='black',s=13,zorder=3)
    axes[0].set(ylabel='Mean valid gap (%)',title='16 new seeds: final confirmation; dots = seed results')
    axes[1].bar(x,[s['median_total_s'] for s in summaries],color='#7c3aed')
    axes[1].set(ylabel='Median local seconds',title='Uncached runtime for the same confirmation runs')
    for ax in axes:
        ax.set_xticks(x,[s['window']+f"\n{s['length']}h; valid {s['valid_days']}/{s['runs']}" for s in summaries])
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('REDUCED QUBIT HYBRID HOURLY METHOD — short-window confirmation')
    fig.tight_layout();plt.show()
    print('These are measured seed averages on one forecast day, not proof of a unique optimal window for every run/day.')
    return dict(summary=summaries,runs=rows,artifacts=artifacts,settings=dict(seeds=seeds,candidates=chosen))
