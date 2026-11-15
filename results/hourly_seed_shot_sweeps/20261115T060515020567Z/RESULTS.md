# Completed ideal local hourly seed/shot sweep

Completed 362 trials in 404.0 seconds. Independent verification checked 295 returned schedules and replayed 1085 quantum measurement logs. 67 trials failed and remain in every attempted-run denominator.

Gross demand, five/ten plants, hours 13–17, ten seeds (101–110). No refinement, renewables, noise model, cloud simulation or hardware. The exact-hourly row is one deterministic classical reference, not ten seed trials.

| Plants | Method | Shots/hour | Valid/attempted | Mean valid gap (%) | Worst valid gap (%) | Valid & gap ≤ 0.1% | Median trial seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | Exact hourly | all local patterns | 1/1 | 0.0000 | 0.0000 | 1/1 | 0.087 |
| 5 | Reduced QAOA | 5 | 10/10 | 1.5742 | 2.6351 | 1/10 | 0.767 |
| 5 | Unreduced QAOA | 5 | 7/10 | 2.1032 | 3.1912 | 0/10 | 0.887 |
| 5 | Random, same locks | 5 | 3/10 | 3.3010 | 3.7077 | 0/10 | 0.062 |
| 5 | Unreduced QAOA | 10 | 7/10 | 1.5909 | 2.4875 | 0/10 | 0.850 |
| 5 | Random, same locks | 10 | 5/10 | 2.7653 | 3.3097 | 0/10 | 0.073 |
| 5 | Reduced QAOA | 10 | 9/10 | 1.2081 | 2.6351 | 1/10 | 0.791 |
| 5 | Random, same locks | 20 | 10/10 | 1.6875 | 2.5272 | 1/10 | 0.120 |
| 5 | Reduced QAOA | 20 | 10/10 | 0.1402 | 0.6641 | 6/10 | 1.177 |
| 5 | Unreduced QAOA | 20 | 10/10 | 0.2522 | 1.1198 | 5/10 | 1.299 |
| 5 | Reduced QAOA | 50 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.784 |
| 5 | Unreduced QAOA | 50 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.858 |
| 5 | Random, same locks | 50 | 10/10 | 0.5123 | 1.8235 | 5/10 | 0.074 |
| 5 | Unreduced QAOA | 100 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.856 |
| 5 | Random, same locks | 100 | 10/10 | 0.1823 | 1.8235 | 9/10 | 0.074 |
| 5 | Reduced QAOA | 100 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.783 |
| 5 | Random, same locks | 200 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.075 |
| 5 | Reduced QAOA | 200 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.789 |
| 5 | Unreduced QAOA | 200 | 10/10 | 0.0000 | 0.0000 | 10/10 | 0.865 |
| 10 | Exact hourly | all local patterns | 1/1 | 0.0101 | 0.0101 | 1/1 | 0.192 |
| 10 | Unreduced QAOA | 5 | 1/10 | 2.7068 | 2.7068 | 0/10 | 2.118 |
| 10 | Random, same locks | 5 | 0/10 | — | — | 0/10 | 0.084 |
| 10 | Reduced QAOA | 5 | 8/10 | 2.7161 | 3.9064 | 0/10 | 1.455 |
| 10 | Random, same locks | 10 | 0/10 | — | — | 0/10 | 0.083 |
| 10 | Reduced QAOA | 10 | 10/10 | 2.0260 | 3.2557 | 0/10 | 1.560 |
| 10 | Unreduced QAOA | 10 | 5/10 | 2.7997 | 3.6627 | 0/10 | 3.066 |
| 10 | Reduced QAOA | 20 | 10/10 | 1.1979 | 1.8697 | 0/10 | 1.662 |
| 10 | Unreduced QAOA | 20 | 10/10 | 2.1023 | 3.2210 | 0/10 | 3.093 |
| 10 | Random, same locks | 20 | 1/10 | 6.4216 | 6.4216 | 0/10 | 0.084 |
| 10 | Unreduced QAOA | 50 | 10/10 | 0.9319 | 1.7206 | 0/10 | 3.161 |
| 10 | Random, same locks | 50 | 8/10 | 2.2769 | 2.7978 | 0/10 | 0.112 |
| 10 | Reduced QAOA | 50 | 10/10 | 0.6052 | 1.2298 | 0/10 | 1.775 |
| 10 | Random, same locks | 100 | 9/10 | 1.4325 | 2.5044 | 0/10 | 0.121 |
| 10 | Reduced QAOA | 100 | 10/10 | 0.3964 | 0.7276 | 2/10 | 1.774 |
| 10 | Unreduced QAOA | 100 | 10/10 | 0.9543 | 2.2681 | 0/10 | 3.237 |
| 10 | Reduced QAOA | 200 | 10/10 | 0.0804 | 0.3032 | 7/10 | 1.764 |
| 10 | Unreduced QAOA | 200 | 10/10 | 0.2016 | 0.3933 | 4/10 | 3.150 |
| 10 | Random, same locks | 200 | 10/10 | 0.9078 | 2.2263 | 1/10 | 0.140 |

Read the conditional gap together with validity: unsuccessful trials have no reportable valid-day gap. At low shots, comparing only surviving trials can reverse the apparent ranking. `paired.csv` separately records cost comparisons on common-valid seeds and the cases where only one method returned a valid day.

Angle training uses exact simulated expectations and is timed/counted; shot budgets refer only to final sampling per hour. The methods use fresh training and full-day MILP lookahead during construction. Sampling-budget comparisons are not equal-compute comparisons, and these results do not establish a quantum speed or cost advantage.

The recorded seeds are repeated draws on the same two fixed datasets. Wider confidence intervals and held-out fleet/demand instances matter for any general claim. Results with zero failures in ten trials do not guarantee future success.

The source and dataset hashes, detailed schedules, angles, and sampled counts are in `results.json`. `runs.jsonl` preserves each attempt. Charts are `seed_shot_sweep.png` and `seed_shot_sweep.pdf`. The independent validation report is `independent_validation.json`.
