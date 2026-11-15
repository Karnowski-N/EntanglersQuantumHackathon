# Hybrid Quantum-Classical Optimization for Power Plant Scheduling

## Project Overview

Large energy providers such as the Tennessee Valley Authority (TVA) must determine which generating units to operate and how much electricity each unit should produce. These decisions must minimize operating costs while meeting electricity demand, reserve requirements, and operational constraints.

Traditionally, this problem is solved using classical optimization methods such as **Mixed-Integer Linear Programming (MILP)**.

Our project explores a **hybrid quantum-classical optimization framework**, combining Quadratic Unconstrained Binary Optimization (QUBO), the Quantum Approximate Optimization Algorithm (QAOA), and classical optimization techniques.

Rather than replacing MILP entirely, we investigate how quantum-assisted optimization can complement classical methods while reducing unnecessary quantum resource usage.

## Why It Matters

Large-scale power systems require utilities to coordinate hundreds of generating units across multiple scheduling periods while satisfying electricity demand, reliability requirements, and complex operational constraints.

As the number of generating units and interdependent scheduling decisions increases, unit commitment becomes increasingly challenging.

Quantum optimization offers an alternative approach to exploring combinatorial solution spaces. Although quantum advantage over classical methods has not yet been demonstrated for this problem, hybrid quantum-classical frameworks offer a promising direction for future research as quantum hardware and algorithms continue to develop.

Our goal is to explore how quantum optimization can be incorporated into existing classical workflows while maintaining operational feasibility and computational efficiency.

## Mathematical Formulation

### Unit Commitment and Economic Dispatch

Our optimization problem consists of two related decisions:

- **Unit Commitment (UC):** Determine which power plants should be ON or OFF during each hour.
- **Economic Dispatch (ED):** Determine how much electricity each committed plant should generate.

For generator \(i\) and hour \(t\), define:

| Variable | Description |
|---|---|
| \(u_{it}\) | Binary commitment status (1 = ON, 0 = OFF) |
| \(v_{it}\) | Binary startup indicator |
| \(w_{it}\) | Binary shutdown indicator |
| \(p_{it}\) | Electricity generation in MW |

### Objective Function

The general objective is to minimize total operating cost:

$$
\min C_{\text{total}} =
\sum_{t=1}^{T}\sum_{i=1}^{N}
\left(
c_i p_{it}
+F_i u_{it}
+S_i v_{it}
+H_i w_{it}
\right)
$$

where:

- \(c_i\): Marginal generation cost ($/MWh)
- \(F_i\): No-load operating cost ($/hour)
- \(S_i\): Startup cost
- \(H_i\): Shutdown cost, when modeled
- \(N\): Number of generating units
- \(T\): Number of scheduling hours

### Operational Constraints

**Power balance**

$$
\sum_i p_{it}=D_t
$$

**Generation limits**

$$
P_i^{\min}u_{it}\le p_{it}\le P_i^{\max}u_{it}
$$

**Reserve requirement**

$$
\sum_i P_i^{\max}u_{it}\ge D_t+R_t
$$

**Startup and shutdown transitions**

$$
u_{it}-u_{i,t-1}=v_{it}-w_{it}
$$

Minimum up-time and down-time constraints also restrict when generators may change operating states.

These equations describe the general mathematical model. The exact implemented cost terms and constraints depend on the solver and validation functions used in our experiments.

### QUBO Formulation

For quantum-assisted optimization, binary commitment decisions are represented using a QUBO objective.

A representative hourly formulation is:

$$
E(x,s)=
\sum_i \hat{c}_i x_i+
A\left(\sum_i a_i x_i-s-r_t\right)^2
$$

where:

- \(x_i\): Binary generator commitment decision
- \(\hat{c}_i\): Estimated commitment cost
- \(a_i\): Discretized generator capacity
- \(s\): Binary-encoded slack variable
- \(r_t\): Required capacity, including reserve
- \(A\): Constraint penalty weight

The QUBO objective guides the quantum algorithm toward promising commitment combinations.

The resulting schedules are evaluated using classical feasibility checks and operating-cost calculations.

---

## What We Did

We developed and tested a hybrid quantum-classical optimization framework for power plant scheduling, extending the original five-plant starter problem toward a **10-plant, 24-hour scenario**.

Our development consisted of the original starter implementation followed by three optimization strategies.

### 1. Starter: Classical MILP vs. QUBO-QAOA

We began by comparing the provided classical MILP baseline against the guided QUBO-QAOA implementation.

**TODO 1 — Classical Cost Heuristic**

We developed a classical cost-based heuristic that prioritizes lower-cost generating units when constructing candidate schedules.

**TODO 2 — Hourly Commitment Cost Coefficients**

We defined an estimated hourly commitment cost coefficient for each generating unit, accounting for factors such as:

- Startup cost
- No-load operating cost
- Minimum generation requirement
- Relative marginal generation cost

These coefficients were used to construct the QUBO objective.

**TODO 3 — QUBO and QAOA Implementation**

The original model included five generator commitment variables and three additional slack bits, resulting in an eight-qubit optimization problem.

The slack bits help represent capacity inequalities as equality penalties.

The QUBO was transformed into an Ising cost Hamiltonian, which was then used in the QAOA cost layer.

QAOA alternates between two parameterized operations:

- **Cost layer (\(\gamma\)):** Encodes the optimization objective.
- **Mixer layer (\(\beta\)):** Redistributes probability amplitudes among candidate solutions.

These layers are repeated \(p\) times. A classical optimizer adjusts the parameters to reduce the expected objective value, aiming to increase the probability of sampling low-cost solutions.

We compared QUBO-QAOA against MILP for individual hours and across a 12-hour scheduling experiment.

In our tested cases, QUBO-QAOA achieved comparable or higher operating costs and did not demonstrate a runtime advantage over MILP.

We also compared QAOA against random candidate sampling. In our experiments, configurations using at least 50 shots consistently produced schedules satisfying the tested operational constraints.

These observations are specific to our experimental settings and do not establish a general minimum shot requirement.

### 2. Hybrid Optimization I: Selective Temporal Quantum Optimization

Next, we investigated how to combine QAOA and MILP within a complete 24-hour scheduling horizon.

Rather than applying QAOA to every hour, we evaluated selected hourly intervals and compared their results against classical MILP.

Our experiments identified the period between approximately **1 PM and 6 PM (hours 13–18)** as a promising interval for quantum-assisted optimization.

During the selected period, the observed cost difference between the quantum-assisted and classical approaches was less than approximately **0.08%** in our tested configuration.

Based on these findings, we developed a hybrid scheduling approach that applies QAOA selectively while using MILP for the remaining scheduling decisions.

This strategy reduces the number of quantum optimization executions while retaining classical methods for feasibility checking and economic dispatch.

### 3. Hybrid Optimization II: Reduced-Qubit Hybrid Hourly Method (RQHH)

Quantum optimization becomes increasingly resource-intensive as the number of qubits and quantum gates increases.

To address this challenge, we developed the **Reduced-Qubit Hybrid Hourly (RQHH)** method.

RQHH uses generator operating constraints and commitment history to identify decisions that are already fixed and therefore do not require independent quantum optimization variables.

In particular, minimum up-time constraints determine how long a generator must remain online after startup.

For example:

Suppose a gas generator starts operating at hour 8 and has a minimum up-time requirement of eight hours.

The generator must remain online through hour 15.

Therefore:

- At hour 13, its commitment status is already fixed to ON.
- No additional commitment qubit is required to determine that status.
- At hour 16, the generator may become eligible for shutdown, so its commitment status can again become an optimization decision.

Importantly, RQHH preserves the generator's operating history even when the quantum-assisted scheduling period begins later in the day.

By eliminating commitment variables fixed by operating constraints, this method aims to reduce:

- Active commitment qubits
- Quantum search-space size
- Circuit complexity
- Unnecessary quantum computation

Auxiliary slack variables may still be required, depending on the QUBO formulation.

### 4. Hybrid Optimization III: Flexible-Capacity Economic Dispatch

Our third improvement focuses on allowing generating units to operate at partial capacity rather than assuming that every committed unit must generate at maximum output.

In real power systems, a committed generator can typically operate within its minimum and maximum generation limits.

We incorporated **Linear Programming (LP)** to determine the economically efficient output levels of committed generators.

The workflow is:

1. Generate candidate ON/OFF commitment decisions through the hybrid QUBO-QAOA framework.
2. Identify the committed generating units.
3. Use classical LP to determine the generation output of each committed unit.
4. Enforce generation limits and electricity demand requirements.
5. Evaluate the resulting dispatch cost and feasibility.

For committed unit \(i\), the dispatch must satisfy:

$$
P_i^{\min}u_{it}\le p_{it}\le P_i^{\max}u_{it}
$$

This allows the optimization framework to distinguish between deciding whether a plant operates and deciding how much electricity it produces.

Flexible-capacity dispatch improves the physical representation of the scheduling problem and allows generation to better match demand.

It does not imply that electricity can be stored or carried into later hours without an explicit storage model.

---

## Results and Evaluation

Our experiments investigate the tradeoffs between classical optimization and quantum-assisted scheduling.

We evaluate:

- Total operating cost
- Relative cost gap against MILP
- Operational constraint violations
- Runtime
- Number of quantum evaluations and measurement shots
- Qubit requirements and circuit complexity

The relative cost gap is calculated as:

$$
\text{Cost Gap (\%)}=
100\frac{C_{\text{hybrid}}-C_{\text{MILP}}}
{C_{\text{MILP}}}
$$

In one of our initial hybrid experiments, the resulting schedule was approximately **0.19% more expensive than the MILP baseline**.

This demonstrates comparable operating costs for that tested configuration, but does not establish quantum computational advantage.

All comparisons should use consistent generator data, operating constraints, demand profiles, and cost definitions.

## Limitations and Future Work

Our experiments demonstrate both the potential and current limitations of hybrid quantum-classical optimization.

MILP remains a strong classical benchmark, particularly for the relatively small scheduling instances studied here.

Our QUBO formulation uses approximate commitment costs, while classical optimization and feasibility checks handle additional operational requirements.

Future research directions include:

1. **Scalability:** Evaluate larger scheduling problems involving more generating units and longer planning horizons.
2. **Adaptive Quantum Assistance:** Investigate how to identify scheduling periods where QAOA provides the greatest measurable benefit.
3. **Reduced-Qubit Optimization:** Extend RQHH to reduce circuit complexity while preserving valid commitment decisions.
4. **Renewable Integration:** Improve scheduling under variable solar and wind generation, including changes in net demand and ramping requirements.
5. **Battery Storage:** Incorporate state-of-charge dynamics, charging and discharging constraints, and energy shifting across multiple hours.

Our long-term goal is to develop a resource-efficient hybrid optimization framework that can complement established classical power-system scheduling methods.

Rather than assuming quantum computing will replace MILP, we aim to determine where quantum-assisted optimization can provide measurable value as power-system optimization problems become more complex.
