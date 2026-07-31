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
from actapint.section2 import (
    run_figure_2_1,
    run_figure_2_2,
    run_figure_2_3,
    run_figure_2_4,
    run_section2,
)


EXPERIMENTS = {
    "section2": run_section2,
    "figure_2_1": run_figure_2_1,
    "figure_2_2": run_figure_2_2,
    "figure_2_3": run_figure_2_3,
    "figure_2_4": run_figure_2_4,
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

GPU_EXPERIMENTS = {"parareal_burgers", "paper_validation"}


def main():
    parser = argparse.ArgumentParser(description="Run Python reproductions of ActaPinT experiments.")
    parser.add_argument("experiment", choices=["all", *EXPERIMENTS])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="Use smaller grids for a fast validation run.")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Use CuPy for supported batched fine propagators when available.",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    selected = EXPERIMENTS.items() if args.experiment == "all" else [(args.experiment, EXPERIMENTS[args.experiment])]
    for name, function in selected:
        print(f"[run] {name}", flush=True)
        kwargs = {"quick": args.quick}
        if name in GPU_EXPERIMENTS:
            kwargs["backend"] = args.backend
        metric_path = function(output, **kwargs)
        print(f"[ok]  {metric_path}", flush=True)


if __name__ == "__main__":
    main()
