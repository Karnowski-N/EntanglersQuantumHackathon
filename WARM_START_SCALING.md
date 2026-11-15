# Warm-start scaling results

Latest completed pilot: [20261115T083253067634Z](results/warm_start_scaling/20261115T083253067634Z/REPORT.md).

- [Chart (PNG)](results/warm_start_scaling/20261115T083253067634Z/warm_start_scaling.png) · [PDF](results/warm_start_scaling/20261115T083253067634Z/warm_start_scaling.pdf) · [CSV](results/warm_start_scaling/20261115T083253067634Z/summary.csv)
- [Full results and methods](results/warm_start_scaling/20261115T083253067634Z/REPORT.md)
- [Independent validation](results/warm_start_scaling/20261115T083253067634Z/independent_validation.json)
- Implementation: `warm_start_scaling.py`; local backend: `warm_start_local_mps.py`; focused tests: `test_warm_start_scaling.py`.

| Plants | Valid whole day | Warm QAOA gap | Executed qubits | Feasible final shots | Full MILP wall time |
|---:|:---:|---:|---:|---:|---:|
| 5 | yes | 0.0000% | 7–8 | 400/1000 | 0.060 s |
| 10 | yes | 0.0101% | 11–12 | 460/1000 | 0.098 s |
| 25 | yes | 0.0853% | 21–26 | 601/1000 | 0.303 s |
| 50 | yes | 0.0828% | 39–50 | 530/1000 | 0.817 s |
| 100 | yes | 0.0590% | 69–99 | 545/1000 | 1.154 s |


Backend: **Qiskit Aer MPS (local fallback)**. One trajectory per fleet. All methods retain the original MILP prefix and tail. The original reduced hourly sweep and notebook cells remain available. No hardware was run.

To explicitly rerun the local fallback (no cloud or hardware submission):

```bash
timeout 900s prlimit --as=2147483648 -- env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/jovyan/.qbraid/environments/qbraid_8lh8/pyenv/bin/python warm_start_local_mps.py --fleets 5 10 25 50 100
```

For QPerfect MIMIQ instead, the original `warm_start_scaling.py` entry point submits cloud jobs only when its advertised pricing is zero. Saved queued jobs can be resumed without resubmission using `results/warm_start_scaling/resume_experiment.py <result-folder>`.

This warm start uses a greedy feasible seed with a compatible mixer, inspired by [Egger et al.](https://arxiv.org/abs/2009.10095). Its role is to improve useful sampling, not to reduce additional free qubits. Compare to the provided classical controls before attributing gains to quantum evolution.

The saved results and chart are also appended to `starter.ipynb` under **Warm-start scaling pilot: 5–100 plants**. Its display cell submits no jobs. All 99 pre-existing cells were preserved.
