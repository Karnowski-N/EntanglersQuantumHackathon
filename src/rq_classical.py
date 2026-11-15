"""Original classical formulas, loaded into isolated fleet namespaces. No execution on import."""

def plant_hour_costs(plants, was_on):
    c = np.zeros(len(plants))
    marginal_cheapest = min((p['marginal_cost_per_mwh'] for p in plants))
    for i, p in enumerate(plants):
        startup = p['startup_cost'] * (1 - was_on[i])
        no_load = p['no_load_cost_per_h']
        extra_energy = p['pmin_mw'] * (p['marginal_cost_per_mwh'] - marginal_cheapest)
        c[i] = startup + no_load + extra_energy
    return c

def build_qubo(plants, demand_mw, reserve_mw, was_on, penalty=None):
    c = plant_hour_costs(plants, was_on)
    A = 2 * c.max() if penalty is None else penalty
    b = np.array([p['pmax_mw'] / BLOCK_MW for p in plants] + [-2 ** k for k in range(N_SLACK)])
    r = np.ceil((demand_mw + reserve_mw) / BLOCK_MW)
    Q = np.zeros((N_QUBITS, N_QUBITS))
    offset = 0.0
    for i in range(len(plants)):
        Q[i, i] += c[i]
    for j in range(N_QUBITS):
        Q[j, j] += A * (b[j] ** 2 - 2 * r * b[j])
        for k in range(j + 1, N_QUBITS):
            Q[j, k] += 2 * A * b[j] * b[k]
    offset += A * r ** 2
    return (Q, offset)

def add_locks(Q, offset, must_on, must_off, weight):
    Q = Q.copy()
    for i in must_on:
        Q[i, i] -= weight
        offset += weight
    for i in must_off:
        Q[i, i] += weight
    return (Q, offset)

def remaining_locks(plants, history):
    hold_on, hold_off = (np.zeros(len(plants), dtype=int), np.zeros(len(plants), dtype=int))
    for i, p in enumerate(plants):
        states = [int(p['on_at_start'])] + [int(h[i]) for h in history]
        now, held = (states[-1], 0)
        for s in reversed(states):
            if s != now:
                break
            held += 1
        if held == len(states):
            continue
        if now == 1:
            hold_on[i] = max(0, int(p['min_up_h']) - held)
        else:
            hold_off[i] = max(0, int(p['min_down_h']) - held)
    return (hold_on, hold_off)

def solve_milp_window(plants, demand, reserve, was_on, hold_on, hold_off):
    """check.solve_milp for a stretch of hours that starts in the middle of the day.

    was_on: who was on the hour before the stretch (decides who pays a startup).
    hold_on[i] / hold_off[i]: hours plant i must still stay on / off at the start of the stretch
    (see remaining_locks). Returns (cost, schedule [hours x plants], seconds).
    """
    n, T = (len(plants), len(demand))
    U = lambda i, t: i * T + t
    V = lambda i, t: n * T + i * T + t
    P = lambda i, t: 2 * n * T + i * T + t
    cost = np.zeros(3 * n * T)
    lower = np.zeros(3 * n * T)
    upper = np.r_[np.ones(2 * n * T), [p['pmax_mw'] for p in plants for _ in range(T)]]
    rows, lo, hi = ([], [], [])

    def add(terms, low, high):
        rows.append(terms)
        lo.append(low)
        hi.append(high)
    for t in range(T):
        add([(P(i, t), 1) for i in range(n)], demand[t], demand[t])
        add([(U(i, t), p['pmax_mw']) for i, p in enumerate(plants)], demand[t] + reserve[t], np.inf)
    for i, p in enumerate(plants):
        up, down, u0 = (int(p['min_up_h']), int(p['min_down_h']), int(was_on[i]))
        for t in range(T):
            cost[U(i, t)] = p['no_load_cost_per_h']
            cost[V(i, t)] = p['startup_cost']
            cost[P(i, t)] = p['marginal_cost_per_mwh']
            add([(P(i, t), 1), (U(i, t), -p['pmin_mw'])], 0, np.inf)
            add([(P(i, t), 1), (U(i, t), -p['pmax_mw'])], -np.inf, 0)
            before = [(U(i, t - 1), 1)] if t > 0 else []
            add([(V(i, t), 1), (U(i, t), -1)] + before, -u0 if t == 0 else 0, np.inf)
            add([(V(i, s), 1) for s in range(max(0, t - up + 1), t + 1)] + [(U(i, t), -1)], -np.inf, 0)
            window = [(V(i, s), 1) for s in range(max(0, t - down + 1), t + 1)]
            if t - down >= 0:
                add(window + [(U(i, t - down), 1)], -np.inf, 1)
            else:
                add(window, -np.inf, 1 - u0)
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
    res = scipy_milp(cost, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integrality, bounds=Bounds(lower, upper))
    seconds = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(f'The MILP found no schedule for this stretch: {res.message}')
    schedule = np.array([[round(res.x[U(i, t)]) for i in range(n)] for t in range(T)])
    return (res.fun, schedule, seconds)
