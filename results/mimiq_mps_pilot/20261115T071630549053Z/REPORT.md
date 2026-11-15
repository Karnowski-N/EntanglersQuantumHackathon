# QPerfect MIMIQ MPS — 100-plant reduced-hourly pilot

**Result: MIMIQ simulated the large reduced circuits successfully, but the limited scheduling pilot failed to complete a day.** Hour 13 returned feasible proposals; hour 14 returned none in its 200 fresh final samples. This is a failed scheduling trial, not a valid daily-cost result.

The pilot took **6.71 minutes** wall time, including local setup/training validation, cloud submission, queues, sampling, and bond checks. There were **7 cloud jobs**, and every returned per-circuit cost was zero. These are ideal MPS simulations on `qperfect:qperfect:sim:mimiq-mps`; no physical QPU or device noise model was used.

## What was tested

- Gross demand, original reserve and min-up/down constraints, no ramps or renewables. Quantum hours were intended to be 13–17; the run stopped at the first failed hour.
- Full-day MILP supplies the initial prefix, just as in the existing hourly hybrid. Its certified cost for this 100-plant instance is $5,037,376.32.
- Only legally forced variables were substituted. All remaining free decisions stayed together; there was no partition into smaller groups.
- The first circuit contains 66 free plant variables plus three slack variables (69 qubits). The next contains 75 free plant variables plus three slack variables (78 qubits).
- Large-state exact expectation training is infeasible locally. This pilot therefore changes training to an eight-angle grid evaluated with 512 MIMIQ samples per angle at bond dimension 64. The grid includes transferred ten-plant angles and seven fixed candidates. Lowest sampled mean QUBO energy chooses the angle.
- Each scheduling decision then uses **200 fresh samples** at bond dimension 128. Training samples and bond-check samples are not substituted when the final samples fail.
- This is one limited scaling pilot, not the same training budget as the earlier exact-statevector sweep, not an exhaustive angle search, and not evidence that every QAOA training strategy fails at 100 plants.

| Hour | Reduced qubits | Logical RZZ gates | Feasible final QAOA samples | Random on same history | MIMIQ reported fidelity estimate |
|---|---:|---:|---:|---:|---:|
| 13 | 69 | 2346 | 102/200 | 0/200 | 0.999999999958 |
| 14 | 78 | 3003 | 0/200 | 0/200 | 0.999999999996 |

Random controls use the identical current history and legal forced values. They are conditional hourly controls, not independently constructed random full-day runs. Reported fidelity describes the simulator approximation; it is not a physical-QPU fidelity measurement.

## Correctness and approximation checks

- A deterministic basis-state circuit verified provider bit ordering. An 11-qubit scheduling circuit agreed with the exact local statevector in energy and feasible-sample probability within sampling uncertainty (energy discrepancy 0.78 standard errors, 4,096 shots).
- For hour 13, bond dimensions 64, 128 and 256 produced identical 200-shot counts at the same seed. Reported fidelity estimates were approximately 0.999999957, 0.999999999958 and 0.999999999998. This supports accuracy for that selected circuit; it is not a general guarantee for all larger circuits.
- Independent validation checked **20 large-circuit result sets / 8992 samples**, reconstructed reduced/full QUBO energies, checked feasibility, and verified that selected plant states were observed samples and the cheapest legal observed choice.

## Failure diagnosis

All 200 final samples at hour 14 failed the demand-plus-reserve capacity requirement; none failed the forced locks or minimum-output check. Feasible continuation is possible: available capacity after forced-off locks is 11,700 MW, versus an 11,004 MW requirement. A diagnostic MILP found a feasible optimal continuation of the chosen prefix in about 0.52 seconds.
That diagnostic is not a quantum-generated solution and was not used to repair the pilot. The missing feasible final sample is a sampling/training limitation in this experiment, not evidence that the instance has no solution or that MPS cannot execute the circuit.
One alternative grid angle produced 3 feasible training samples out of 512 at hour 14, but had worse mean QUBO energy than the chosen angle. This illustrates that minimizing mean surrogate energy and maximizing useful feasible proposal probability can disagree under a limited training budget. Changing that selection objective would be a new, explicitly labeled experiment.

## What bounded groups would mean

For the first 100-plant hour, the current exact reduction leaves 66 free plant decisions. A grouped variant could allow ten of those decisions to change at a time, hold the other 56 at their current assignments, and retain the three slack variables: about 13 qubits per subproblem instead of 69. Contributions from held-fixed plants still enter the QUBO and the full schedule checker. After checking a proposed update, the procedure would move to another group, potentially using overlapping groups or additional passes.
This can reduce circuit width, entangling gates and exact-simulation allocation. It can also keep proposals closer to an existing feasible schedule. However, fixing a free decision for one search step is a heuristic restriction, unlike removing a variable whose value is logically forced by the fixed history. It can miss improvements requiring coordinated changes across groups and may require many additional steps. It does not guarantee either better quality or lower total runtime.
The MIMIQ pilot shows that grouping is not required merely to fit these two 100-plant circuits. The immediate observed issue is useful feasible sampling under the limited angle-training procedure. No 500-plant experiment was run, and no quantum computational advantage is established.

## Reproducibility

`pilot.json` preserves model settings, matrices, mappings, angles, counts, source hashes, costs, timing and job IDs. Each submitted circuit has an OpenQASM file. `independent_validation.json` and `failure_diagnosis.json` contain checks. The standalone runner is `mimiq_mps_pilot.py`; it refuses nonzero advertised simulator rates. Rerunning it submits new cloud simulator jobs. The existing starter sweep was not edited or rerun.

Primary service references: [qBraid MIMIQ provider](https://docs.qbraid.com/v2/sdk/user-guide/providers/qperfect), [QPerfect simulation parameters](https://docs.qperfect.io/mimiqcircuits-python/manual/simulation_parameters.html).

| Job label | Job ID | Wall seconds |
|---|---|---:|
| validation10 | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95c50c8f5265fb3f24f62` | 44.18 |
| h13_training | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95c9dc8f5265fb3f24f81` | 98.76 |
| h13_final | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95ce5c8f5265fb3f24fa1` | 26.18 |
| h13_bond64 | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95cffc8f5265fb3f24faf` | 46.04 |
| h13_bond256 | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95d2dc8f5265fb3f24fb6` | 25.55 |
| h14_training | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95d6ec8f5265fb3f24fc9` | 119.99 |
| h14_final | `qperfect:qperfect:sim:mimiq-mps-69e6-qjob-6af95dc1c8f5265fb3f24feb` | 36.97 |
