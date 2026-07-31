#!/usr/bin/env python3
"""Benchmark the independent Burgers fine propagators on CPU and GPU."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from actapint.gpu import CupyBurgersFineStepper, backend_metadata
from actapint.integrators import burgers_fine_step
from actapint.operators import burgers_operators


def elapsed(function, repeats: int) -> tuple[float, np.ndarray]:
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-points", type=int, default=128)
    parser.add_argument("--time-slices", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--viscosity", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    n = args.space_points
    batch = args.time_slices
    dt = (4.0 / batch) / args.substeps
    a1, a2 = burgers_operators(n, 1 / n, args.viscosity, periodic=True)
    values = np.random.default_rng(5).random((n, batch))
    cpu_step = lambda value: burgers_fine_step(
        a1,
        a2,
        value,
        dt,
        args.substeps,
        "sdirk22",
        1e-12,
    )
    gpu_step = CupyBurgersFineStepper(
        a1,
        a2,
        dt,
        args.substeps,
        tol=1e-14,
    )

    gpu_step(values)
    gpu_seconds, gpu_result = elapsed(lambda: gpu_step(values), args.repeats)
    cpu_seconds, cpu_result = elapsed(
        lambda: np.column_stack(
            [cpu_step(values[:, j]) for j in range(batch)]
        ),
        args.repeats,
    )
    report = {
        "parameters": {
            "space_points": n,
            "time_slices": batch,
            "substeps": args.substeps,
            "viscosity": args.viscosity,
        },
        "device": backend_metadata("gpu"),
        "cpu_seconds": cpu_seconds,
        "gpu_seconds": gpu_seconds,
        "speedup": cpu_seconds / gpu_seconds,
        "max_abs_difference": float(np.max(np.abs(cpu_result - gpu_result))),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
