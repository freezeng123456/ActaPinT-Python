#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from actapint.experiments import (
    run_burgers_solution,
    run_iterative_paradiag_ade,
    run_linear_solution,
    run_mgrit_linear,
    run_parareal_burgers,
    run_parareal_linear,
    run_stmg_linear,
    run_wave_solution,
)
from actapint.paper_validation import run_paper_validation


EXPERIMENTS = {
    "solution_heat_ade": run_linear_solution,
    "solution_burgers": run_burgers_solution,
    "solution_wave": run_wave_solution,
    "parareal_heat_ade": run_parareal_linear,
    "parareal_burgers": run_parareal_burgers,
    "mgrit_heat_ade": run_mgrit_linear,
    "iterative_paradiag_ade": run_iterative_paradiag_ade,
    "stmg_heat_ade": run_stmg_linear,
    "paper_validation": run_paper_validation,
}


def main():
    parser = argparse.ArgumentParser(description="Run Python reproductions of ActaPinT experiments.")
    parser.add_argument("experiment", choices=["all", *EXPERIMENTS])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="Use smaller grids for a fast validation run.")
    args = parser.parse_args()
    output = Path(args.output_dir)
    selected = EXPERIMENTS.items() if args.experiment == "all" else [(args.experiment, EXPERIMENTS[args.experiment])]
    for name, function in selected:
        print(f"[run] {name}", flush=True)
        metric_path = function(output, quick=args.quick)
        print(f"[ok]  {metric_path}", flush=True)


if __name__ == "__main__":
    main()
