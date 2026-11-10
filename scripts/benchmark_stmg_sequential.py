"""Forward-substitution baseline using iterative spatial MG, not direct LU."""

import argparse
import hashlib
import json
import os
import platform
import socket
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from actapint.stmg_fem import space_hierarchy
from actapint.stmg_sequential import ForwardSubstitution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--refinements", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if min(args.steps, args.repeats) < 1 or args.warmups < 0:
        p.error("positive steps/repeats and nonnegative warmups required")
    start = perf_counter()
    solver = ForwardSubstitution(space_hierarchy(args.refinements))
    setup = perf_counter() - start
    exact = np.array(
        [
            np.random.default_rng(np.random.SeedSequence([20261111, j])).uniform(
                -1, 1, (solver.q, solver.n)
            )
            for j in range(args.steps)
        ]
    )
    rhs = solver.apply(exact)
    samples = []
    for repeat in range(-args.warmups, args.repeats):
        start = perf_counter()
        u, iterations, block_residuals = solver.solve(rhs)
        elapsed = perf_counter() - start
        residual = float(np.linalg.norm(rhs - solver.apply(u)) / np.linalg.norm(rhs))
        error = float(np.linalg.norm(u - exact) / np.linalg.norm(exact))
        if residual > 1e-8 or error > 1e-7:
            raise ArithmeticError(f"acceptance failed: {residual=}, {error=}")
        row = dict(
            repeat=repeat,
            seconds=elapsed,
            relative_residual=residual,
            relative_error=error,
            spatial_iterations=iterations,
            block_relative_residuals=block_residuals,
        )
        if repeat >= 0:
            samples.append(row)
        print(
            dict(steps=args.steps, repeat=repeat, seconds=elapsed, residual=residual),
            flush=True,
        )
    result = dict(
        status="passed",
        method="sequential forward substitution with iterative spatial V-cycles",
        steps=args.steps,
        spatial_dofs=solver.n,
        dofs=exact.size,
        dt=0.1,
        degree=3,
        setup_seconds=setup,
        warmups=args.warmups,
        samples=samples,
        median_seconds=float(np.median([s["seconds"] for s in samples])),
        seed=20261111,
        host=socket.gethostname(),
        tolerance="1e-8 relative residual for each spatial solve; independently accept global residual <= 1e-8 and relative solution error <= 1e-7",
        source_sha256={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [
                Path("actapint/stmg_fem.py"),
                Path("actapint/stmg_sequential.py"),
                Path(__file__),
            ]
        },
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            affinity=(
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            threads={
                k: os.environ.get(k)
                for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
            },
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
