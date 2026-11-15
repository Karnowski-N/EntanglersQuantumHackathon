"""Local, ideal-simulator comparison of the starter's scheduling methods."""
import csv
from pathlib import Path
from statistics import mean, median
from time import perf_counter


def run_benchmark(schedule_day, check_schedule, plants, demand, reserve, milp_cost,
                  output_dir="results", shots_grid=(5, 10, 20, 50, 200),
                  seeds=range(5)):
    """Use the same p=1 scheduler and checker for both samplers; no cloud jobs.

    Seeds vary shot sampling, not optimizer initialization. Each QAOA run includes
    fresh tuning; runtime includes scheduling and final validation, but excludes
    the separately computed full-day MILP baseline. This is an ideal simulation.
    """
    seeds = tuple(seeds)
    rows = []
    for shots in shots_grid:
        for sampler in ("qaoa", "random"):
            for seed in seeds:
                start = perf_counter()
                schedule, log = schedule_day(p=1, shots=shots, sampler=sampler, seed=seed)
                result = check_schedule(plants, demand, reserve, schedule,
                                        milp_cost=milp_cost, verbose=False)
                elapsed = perf_counter() - start
                rows.append(dict(
                    sampler=sampler, shots=shots, seed=seed, layers=1,
                    valid=result["valid"], violations=len(result["violations"]),
                    total_cost=result["total_cost"],
                    valid_gap_pct=result["gap_pct"] if result["valid"] else None,
                    runtime_s=elapsed,
                    mean_valid_sample_pct=100 * mean(r["valid_share"] for r in log),
                ))
            print(f"Finished {sampler:6s}, {shots:3d} shots ({len(seeds)} seeds)", flush=True)

    summary = []
    for shots in shots_grid:
        for sampler in ("qaoa", "random"):
            group = [r for r in rows if r["shots"] == shots and r["sampler"] == sampler]
            gaps = [r["valid_gap_pct"] for r in group if r["valid"]]
            summary.append(dict(
                sampler=sampler, shots=shots, runs=len(group), valid_days=len(gaps),
                valid_rate_pct=100 * len(gaps) / len(group),
                mean_valid_gap_pct=mean(gaps) if gaps else None,
                min_valid_gap_pct=min(gaps) if gaps else None,
                max_valid_gap_pct=max(gaps) if gaps else None,
                median_runtime_s=median(r["runtime_s"] for r in group),
                mean_valid_sample_pct=mean(r["mean_valid_sample_pct"] for r in group),
            ))
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, records in (("benchmark_runs.csv", rows), ("benchmark_summary.csv", summary)):
        with (folder / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return rows, summary


def display_summary(summary):
    """Render the table in Jupyter, with a plain-text fallback."""
    def fmt(x):
        return "—" if x is None else f"{x:.2f}"
    header = ("Sampler", "Shots", "Valid days", "Mean valid gap (%)",
              "Valid gap range (%)", "Median runtime (s)", "Valid samples (%)")
    data = []
    for r in summary:
        interval = "—" if r["min_valid_gap_pct"] is None else (
            f'{fmt(r["min_valid_gap_pct"])} to {fmt(r["max_valid_gap_pct"])}')
        data.append((r["sampler"], str(r["shots"]), f'{r["valid_days"]}/{r["runs"]}',
                     fmt(r["mean_valid_gap_pct"]), interval, fmt(r["median_runtime_s"]),
                     fmt(r["mean_valid_sample_pct"])))
    table = "| " + " | ".join(header) + " |\n"
    table += "| " + " | ".join("---" for _ in header) + " |\n"
    table += "\n".join("| " + " | ".join(row) + " |" for row in data)
    try:
        from IPython.display import Markdown, display
        display(Markdown(table))
    except ImportError:
        print(table)
    return table
