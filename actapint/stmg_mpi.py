"""Time-distributed dG(3)/P1 STMG with nearest-neighbour MPI communication.

Each rank owns contiguous time slabs and the complete spatial mesh. Pairwise
rank agglomeration keeps coarse time grids distributed until one slab remains.
No iteration gathers a global space-time vector. mpi4py is an optional extra.
"""

from __future__ import annotations
import numpy as np
from mpi4py import MPI
from .stmg_fem import TimeMultigrid


class DistributedTimeMultigrid:
    def __init__(self, levels, steps, comm=MPI.COMM_WORLD, dt=0.1, degree=3):
        self.comm, self.steps = comm, steps
        self.rank, self.size = comm.rank, comm.size
        if (
            steps < 1
            or steps & (steps - 1)
            or self.size & (self.size - 1)
            or steps < self.size
            or steps % self.size
        ):
            raise ValueError(
                "steps and ranks must be powers of two, with steps >= ranks"
            )
        self.local_steps = steps // self.size
        self.offset = self.rank * self.local_steps
        self.block = TimeMultigrid(levels, steps, dt, degree, build_hierarchy=False)
        self.q, self.n = self.block.q, self.block.n
        self.shape = (self.local_steps, self.q, self.n)
        self.agglomerate = self.local_steps == 1 and self.size > 1
        self.coarse = None
        self.coarse_comm = None
        if steps == 1:
            self.block.prepare_direct()
        elif self.agglomerate:
            self.coarse_comm = comm.Split(
                0 if self.rank % 2 == 0 else MPI.UNDEFINED, self.rank
            )
            if self.rank % 2 == 0:
                self.coarse = DistributedTimeMultigrid(
                    levels, steps // 2, self.coarse_comm, dt * 2, degree
                )
        else:
            self.coarse = DistributedTimeMultigrid(
                levels, steps // 2, comm, dt * 2, degree
            )

    def apply(self, u):
        out = self.block.apply(u)
        previous = np.zeros((self.q, self.n))
        self.comm.Sendrecv(
            np.ascontiguousarray(u[-1]),
            dest=self.rank + 1 if self.rank + 1 < self.size else MPI.PROC_NULL,
            sendtag=11,
            recvbuf=previous,
            source=self.rank - 1 if self.rank else MPI.PROC_NULL,
            recvtag=11,
        )
        if self.rank:
            out[0] -= self.block.nt @ (self.block.space.mass @ previous.T).T
        return out

    def norm(self, u):
        return float(
            np.sqrt(self.comm.allreduce(float(np.vdot(u, u).real), op=MPI.SUM))
        )

    def smooth(self, u, b):
        r = b - self.apply(u)
        return u + np.array([self.block.block_solve(row) for row in r])

    def cycle(self, b, u=None):
        if self.steps == 1:
            return self.block.block_solve(b[0], direct=True)[None]
        u = np.zeros_like(b) if u is None else u.copy()
        u = self.smooth(u, b)
        residual = b - self.apply(u)
        p = self.block.prolong
        if self.agglomerate:
            if self.rank % 2:
                self.comm.Send(np.ascontiguousarray(residual[0]), self.rank - 1, tag=21)
                change = np.empty_like(u)
                self.comm.Recv(change[0], self.rank - 1, tag=22)
            else:
                pair = np.empty((2 * self.q, self.n))
                pair[: self.q] = residual[0]
                self.comm.Recv(pair[self.q :], self.rank + 1, tag=21)
                ec = self.coarse.cycle((p.T @ pair)[None])
                expanded = p @ ec[0]
                self.comm.Send(
                    np.ascontiguousarray(expanded[self.q :]), self.rank + 1, tag=22
                )
                change = expanded[: self.q][None]
        else:
            paired = residual.reshape(self.local_steps // 2, 2 * self.q, self.n)
            rc = np.einsum("ji,tjx->tix", p, paired)
            ec = self.coarse.cycle(rc)
            change = np.einsum("ij,tjx->tix", p, ec).reshape(u.shape)
        return self.smooth(u + change, b)

    def solve(self, b, tol=1e-8, maxiter=100, initial=None):
        valid = (
            b.shape == self.shape
            and np.isfinite(b).all()
            and tol > 0
            and maxiter > 0
            and (
                initial is None
                or (initial.shape == self.shape and np.isfinite(initial).all())
            )
        )
        if not self.comm.allreduce(bool(valid), op=MPI.LAND):
            raise ValueError("invalid local rhs, initial state, tolerance or maxiter")
        u = np.zeros_like(b) if initial is None else initial.copy()
        scale = self.norm(b - self.apply(u))
        if scale == 0:
            return u, [0.0]
        history = [1.0]
        for _ in range(maxiter):
            u = self.cycle(b, u)
            history.append(self.norm(b - self.apply(u)) / scale)
            if not np.isfinite(history[-1]):
                raise ArithmeticError("nonfinite global residual")
            if history[-1] <= tol:
                break
        return u, history

    def close(self):
        if self.coarse is not None:
            self.coarse.close()
        if self.coarse_comm is not None and self.coarse_comm != MPI.COMM_NULL:
            self.coarse_comm.Free()


def distributed_random(mg, seed=20261111, lower=-1.0):
    """Identical global problem for every rank count; one seed per time slab."""
    return np.array(
        [
            np.random.default_rng(
                np.random.SeedSequence([seed, mg.offset + i])
            ).uniform(lower, 1, (mg.q, mg.n))
            for i in range(mg.local_steps)
        ]
    )
