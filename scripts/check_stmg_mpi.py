"""Run with mpiexec -n {1,2,4,8} python -m scripts.check_stmg_mpi."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from actapint.stmg_fem import space_hierarchy, TimeMultigrid
from actapint.stmg_mpi import DistributedTimeMultigrid, distributed_random


def main():
    comm = MPI.COMM_WORLD
    levels = space_hierarchy(1)
    for steps in (comm.size, 2 * comm.size, 4 * comm.size):
        mg = DistributedTimeMultigrid(levels, steps)
        exact = distributed_random(mg)
        b = mg.apply(exact)
        # Gathering occurs only in this small independent verification, never
        # in the solver or scaling benchmark.
        global_b = comm.gather(b, root=0)
        global_exact = comm.gather(exact, root=0)
        u, history = mg.solve(b, tol=1e-11)
        global_u = comm.gather(u, root=0)
        cycle = comm.gather(mg.cycle(b), root=0)
        ok = True
        if comm.rank == 0:
            b0, e0, u0 = map(np.concatenate, (global_b, global_exact, global_u))
            serial = TimeMultigrid(levels, steps)
            sp = levels[-1]
            diag = sparse.kron(serial.kt, sp.mass) + sparse.kron(
                serial.mt, sp.stiffness
            )
            sub = sparse.kron(serial.nt, sp.mass)
            a = sparse.kron(sparse.eye(steps), diag) - sparse.kron(
                sparse.diags([np.ones(steps - 1)], [-1], shape=(steps, steps)), sub
            )
            ref = spsolve(a.tocsc(), b0.ravel()).reshape(b0.shape)
            ok = (
                np.allclose(b0, (a @ e0.ravel()).reshape(b0.shape), atol=1e-12)
                and np.allclose(u0, ref, atol=1e-9)
                and np.allclose(np.concatenate(cycle), serial.cycle(b0), atol=1e-11)
                and history[-1] <= 1e-11
            )
            print(
                dict(
                    ranks=comm.size,
                    steps=steps,
                    passed=ok,
                    residual=history[-1],
                    global_difference=float(abs(u0 - ref).max()),
                ),
                flush=True,
            )
        if not comm.bcast(ok, root=0):
            raise AssertionError("MPI/global/serial comparison failed")
        zero, zh = mg.solve(np.zeros_like(b))
        assert zh == [0.0] and not zero.any()
        mg.close()


if __name__ == "__main__":
    main()
