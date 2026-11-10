"""Measured MPI weak/strong scaling. Example:
mpiexec -n 4 python -m scripts.benchmark_stmg_mpi --steps 512 --output strong-p4.json
Omit --steps for weak scaling with two slabs per rank. Setup and verification
are outside solve timings. The maximum rank time is recorded for every repeat.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import mpi4py
from mpi4py import MPI
from actapint.stmg_fem import space_hierarchy
from actapint.stmg_mpi import DistributedTimeMultigrid, distributed_random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--refinements", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats >= 1 and warmups >= 0 required")
    comm = MPI.COMM_WORLD
    steps = args.steps or 2 * comm.size
    comm.Barrier()
    t0 = perf_counter()
    levels = space_hierarchy(args.refinements)
    mg = DistributedTimeMultigrid(levels, steps)
    setup = comm.allreduce(perf_counter() - t0, op=MPI.MAX)
    # Fixed, partition-independent manufactured problem. This preserves nonzero
    # forcing and supplies exact global error without gathering any vectors.
    exact = distributed_random(mg, args.seed)
    rhs = mg.apply(exact)
    rows = []
    for repeat in range(-args.warmups, args.repeats):
        comm.Barrier()
        start = perf_counter()
        solution, history = mg.solve(rhs)
        seconds = comm.allreduce(perf_counter() - start, op=MPI.MAX)
        error = mg.norm(solution - exact) / mg.norm(exact)
        if history[-1] > 1e-8 or error > 1e-7:
            raise ArithmeticError(
                f"acceptance failed: residual={history[-1]}, error={error}"
            )
        if repeat >= 0:
            rows.append(
                dict(
                    repeat=repeat,
                    seconds=seconds,
                    iterations=len(history) - 1,
                    relative_residual=history[-1],
                    relative_error=error,
                    residual_history=history,
                )
            )
        if comm.rank == 0:
            print(
                dict(
                    ranks=comm.size,
                    steps=steps,
                    repeat=repeat,
                    seconds=seconds,
                    iterations=len(history) - 1,
                    residual=history[-1],
                ),
                flush=True,
            )
    # Separate homogeneous error-reduction check from the measured RHS solve.
    initial = distributed_random(mg, args.seed + 1, lower=0)
    u = initial.copy()
    norm = mg.norm(initial)
    errors = [1.0]
    for _ in range(100):
        u = mg.cycle(np.zeros_like(u), u)
        errors.append(mg.norm(u) / norm)
        if errors[-1] < 1e-8:
            break
    if errors[-1] >= 1e-8:
        raise ArithmeticError("homogeneous test failed")
    hosts = comm.gather(MPI.Get_processor_name(), root=0)
    affinities = comm.gather(
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        root=0,
    )
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path("actapint/stmg_fem.py"),
            Path("actapint/stmg_mpi.py"),
            Path(__file__),
        ]
    }
    if comm.rank == 0:
        payload = dict(
            status="passed",
            scaling="strong" if args.steps else "weak",
            ranks=comm.size,
            hosts=hosts,
            unique_hosts=len(set(hosts)),
            affinity=affinities,
            steps=steps,
            local_steps=mg.local_steps,
            spatial_dofs=mg.n,
            dofs=steps * mg.q * mg.n,
            tetrahedra=levels[-1].tetrahedra,
            degree=3,
            dt=0.1,
            refinements=args.refinements,
            seed=args.seed,
            warmups=args.warmups,
            setup_seconds=setup,
            samples=rows,
            median_seconds=float(np.median([r["seconds"] for r in rows])),
            homogeneous_iterations=len(errors) - 1,
            homogeneous_error_history=errors,
            timing="maximum rank wall time; includes halo exchange, coarse rank agglomeration and convergence reductions; excludes setup and verification",
            source_sha256=hashes,
            environment=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                scipy=scipy.__version__,
                mpi4py=mpi4py.__version__,
                mpi=MPI.Get_library_version(),
                platform=platform.platform(),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                threads={
                    k: os.environ.get(k)
                    for k in [
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    ]
                },
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    mg.close()


if __name__ == "__main__":
    main()
