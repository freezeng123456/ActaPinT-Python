"""P1 tetrahedral FEM / dG(p) time multigrid for Gander--Neumueller (2016).

This is a new implementation of the discretization behind Acta Table 4.1,
not the backward-Euler / finite-difference two-grid example in section4.stmg.
Time slabs use block Jacobi; each block application is one spatial V-cycle
with damped block Gauss-Seidel. Time coarsening uses exact polynomial injection and its
transpose. The coarsest time problem is solved by sparse direct spatial solves.
All quantities are computed; historical timings are never used as output.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from scipy.sparse.linalg import splu


def dg_matrices(degree=3, dt=0.1):
    if degree < 0 or dt <= 0:
        raise ValueError("degree >= 0 and dt > 0 are required")
    nodes, weights = leggauss(degree + 1)
    nodes, weights = (nodes + 1) / 2, weights / 2
    basis = []
    for j, node in enumerate(nodes):
        others = np.delete(nodes, j)
        poly = Polynomial.fromroots(others) if len(others) else Polynomial([1.0])
        basis.append(poly / poly(node))
    derivative = np.array([p.deriv()(nodes) for p in basis]).T
    left, right = np.array([p(0) for p in basis]), np.array([p(1) for p in basis])
    mass = np.diag(weights)
    stiffness = mass @ derivative + np.outer(left, left)
    coupling = np.outer(left, right)
    prolong = np.vstack(
        [np.array([p((nodes + half) / 2) for p in basis]).T for half in (0, 1)]
    )
    return stiffness, dt * mass, coupling, prolong


def cube_mesh():
    points = np.array(
        [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
        + [(0.5,) * 3]
    )
    tets = []
    for axis in range(3):
        for side in (0.0, 1.0):
            face = np.flatnonzero(points[:8, axis] == side)
            # Four vertices sorted by the remaining two coordinates. The
            # same face diagonal is shared by both adjacent tetrahedra.
            a, b, c, d = face
            tets += [(8, a, b, d), (8, a, c, d)]
    return points, np.asarray(tets)


def refine_mesh(points, tets):
    new = points.tolist()
    edges = {}
    children = []
    for tet in tets:
        mids = {}
        for i, j in combinations(range(4), 2):
            edge = tuple(sorted((int(tet[i]), int(tet[j]))))
            if edge not in edges:
                edges[edge] = len(new)
                new.append(((points[edge[0]] + points[edge[1]]) / 2).tolist())
            mids[i, j] = edges[edge]
        a, b, c, d = tet
        e, f, g, h, i, j = (
            mids[k] for k in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        )
        children += [
            (a, e, f, g),
            (b, e, h, i),
            (c, f, h, j),
            (d, g, i, j),
            (e, j, f, g),
            (e, j, g, i),
            (e, j, i, h),
            (e, j, h, f),
        ]
    rows, cols, vals = (
        list(range(len(points))),
        list(range(len(points))),
        [1.0] * len(points),
    )
    for (a, b), idx in edges.items():
        rows += [idx, idx]
        cols += [a, b]
        vals += [0.5, 0.5]
    transfer = sparse.csr_matrix((vals, (rows, cols)), shape=(len(new), len(points)))
    return np.asarray(new), np.asarray(children), transfer


def interior_nodes(points):
    return np.flatnonzero(np.all((points > 0) & (points < 1), axis=1))


def assemble_fem(points, tets):
    coords = points[tets]
    affine = np.concatenate([np.ones((*coords.shape[:2], 1)), coords], axis=2)
    inv = np.linalg.inv(affine)
    volume = np.abs(np.linalg.det(coords[:, 1:] - coords[:, 0, None, :])) / 6
    gradients = inv[:, 1:, :]
    local_k = volume[:, None, None] * np.einsum("tik,tij->tkj", gradients, gradients)
    local_m = volume[:, None, None] * (np.ones((4, 4)) + np.eye(4))[None, :, :] / 20
    rows = np.broadcast_to(tets[:, :, None], (*tets.shape, 4)).ravel()
    cols = np.broadcast_to(tets[:, None, :], (len(tets), 4, 4)).ravel()
    shape = (len(points),) * 2
    mass = sparse.coo_matrix((local_m.ravel(), (rows, cols)), shape=shape).tocsr()
    stiff = sparse.coo_matrix((local_k.ravel(), (rows, cols)), shape=shape).tocsr()
    ids = interior_nodes(points)
    return mass[ids][:, ids].tocsc(), stiff[ids][:, ids].tocsc()


@dataclass
class SpaceLevel:
    mass: sparse.csc_matrix
    stiffness: sparse.csc_matrix
    prolong: sparse.csr_matrix | None
    nodes: np.ndarray
    tetrahedra: int


def space_hierarchy(refinements=4):
    if refinements < 0:
        raise ValueError("refinements must be nonnegative")
    points, tets = cube_mesh()
    levels = []
    transfer = None
    for lev in range(refinements + 1):
        ids = interior_nodes(points)
        mass, stiff = assemble_fem(points, tets)
        levels.append(SpaceLevel(mass, stiff, transfer, points[ids], len(tets)))
        if lev < refinements:
            points, tets, p = refine_mesh(points, tets)
            transfer = p[interior_nodes(points)][:, ids].tocsr()
    return levels


class SpatialVCycle:
    """One V(1,1), with q time coefficients per space node.

    Forward Gauss-Seidel is used for both sweeps. ``damped_gs`` applies
    omega*(D+L)^-1 to the residual; ``sor`` applies
    omega*(D+omega*L)^-1. These differ when omega != 1.
    """

    def __init__(
        self, levels, kt, mt, omega=1.285, relaxation="damped_gs", post_sweep="forward"
    ):
        if relaxation not in ("damped_gs", "sor") or post_sweep not in (
            "forward",
            "backward",
        ):
            raise ValueError("invalid spatial relaxation or post-sweep direction")
        q = kt.shape[0]
        self.q = q
        self.levels = levels
        self.omega = omega
        self.a = []
        self.p = []
        self.forward = []
        self.backward = []
        for lev, space in enumerate(levels):
            a = (sparse.kron(space.mass, kt) + sparse.kron(space.stiffness, mt)).tocsc()
            self.a.append(a)
            if lev == 0:
                self.direct = splu(a)
                continue
            coo = a.tocoo()
            br, bc = coo.row // q, coo.col // q
            sweeps = [("lower", self.forward)]
            if post_sweep == "backward":
                sweeps.append(("upper", self.backward))
            for side, sink in sweeps:
                keep = br >= bc if side == "lower" else br <= bc
                values = coo.data[keep] * (
                    np.where(br[keep] == bc[keep], 1.0, omega)
                    if relaxation == "sor"
                    else 1.0
                )
                triangular = sparse.coo_matrix(
                    (values, (coo.row[keep], coo.col[keep])), shape=a.shape
                ).tocsc()
                sink.append(splu(triangular, permc_spec="NATURAL", diag_pivot_thresh=0))
            self.p.append(sparse.kron(space.prolong, sparse.eye(q), format="csr"))
        if post_sweep == "forward":
            self.backward = self.forward

    def cycle(self, b, lev=None):
        if lev is None:
            lev = len(self.a) - 1
        if lev == 0:
            return self.direct.solve(b)
        a, p = self.a[lev], self.p[lev - 1]
        u = self.omega * self.forward[lev - 1].solve(b)
        u += p @ self.cycle(p.T @ (b - a @ u), lev - 1)
        u += self.omega * self.backward[lev - 1].solve(b - a @ u)
        return u


class TimeMultigrid:
    def __init__(
        self,
        levels,
        steps,
        dt=0.1,
        degree=3,
        damping=1.0,
        spatial_solver="vcycle",
        spatial_relaxation="damped_gs",
        post_sweep="forward",
    ):
        if steps < 1 or steps & (steps - 1):
            raise ValueError("steps must be a positive power of two")
        if spatial_solver not in ("vcycle", "direct"):
            raise ValueError("spatial_solver must be vcycle or direct")
        self.steps = steps
        self.dt = dt
        self.q = degree + 1
        self.space = levels[-1]
        self.n = self.space.mass.shape[0]
        self.damping = damping
        kt, mt, nt, self.prolong = dg_matrices(degree, dt)
        self.kt, self.mt, self.nt = kt, mt, nt
        self.spatial = (
            None
            if spatial_solver == "direct" or steps == 1
            else SpatialVCycle(
                levels, kt, mt, relaxation=spatial_relaxation, post_sweep=post_sweep
            )
        )
        # The direct block solve is a diagonalization of the small DG time
        # matrix. It avoids a fill-heavy LU on all q*Nx unknowns at once.
        eig, v = np.linalg.eig(np.linalg.solve(mt, kt))
        self.v = v
        self.transform = np.linalg.solve(v, np.linalg.inv(mt))
        self.factors = [
            splu((self.space.stiffness + lam * self.space.mass).astype(complex).tocsc())
            for lam in eig
        ]
        self.spatial_solver = spatial_solver
        self.coarse = (
            None
            if steps == 1
            else TimeMultigrid(
                levels,
                steps // 2,
                dt * 2,
                degree,
                damping,
                spatial_solver,
                spatial_relaxation,
                post_sweep,
            )
        )

    def diagonal_apply(self, u):
        mu = (self.space.mass @ u.reshape(-1, self.n).T).T.reshape(u.shape)
        ku = (self.space.stiffness @ u.reshape(-1, self.n).T).T.reshape(u.shape)
        return np.einsum("ij,tjx->tix", self.kt, mu) + np.einsum(
            "ij,tjx->tix", self.mt, ku
        )

    def apply(self, u):
        out = self.diagonal_apply(u)
        if len(u) > 1:
            mu = (self.space.mass @ u[:-1].reshape(-1, self.n).T).T.reshape(
                u[:-1].shape
            )
            out[1:] -= np.einsum("ij,tjx->tix", self.nt, mu)
        return out

    def block_solve(self, b, direct=False):
        if direct or self.spatial is None:
            transformed = self.transform @ b
            w = np.array(
                [
                    lu.solve(row.astype(complex))
                    for lu, row in zip(self.factors, transformed)
                ]
            )
            return (self.v @ w).real
        return self.spatial.cycle(b.T.ravel()).reshape(self.n, self.q).T

    def smooth(self, u, b):
        r = b - self.apply(u)
        return u + self.damping * np.array([self.block_solve(row) for row in r])

    def cycle(self, b, u=None):
        if self.steps == 1:
            return self.block_solve(b[0], direct=True)[None, :, :]
        u = np.zeros_like(b) if u is None else u.copy()
        u = self.smooth(u, b)
        r = (b - self.apply(u)).reshape(self.steps // 2, 2 * self.q, self.n)
        rc = np.einsum("ji,tjx->tix", self.prolong, r)
        ec = self.coarse.cycle(rc)
        correction = np.einsum("ij,tjx->tix", self.prolong, ec).reshape(b.shape)
        return self.smooth(u + correction, b)

    def solve(self, b, tol=1e-8, maxiter=100, initial=None):
        if b.shape != (self.steps, self.q, self.n):
            raise ValueError("rhs must have shape (steps, degree + 1, space nodes)")
        if not np.isfinite(b).all() or tol <= 0 or maxiter < 1:
            raise ValueError("finite rhs, positive tolerance and maxiter are required")
        u = np.zeros_like(b) if initial is None else initial.copy()
        scale = np.linalg.norm(b - self.apply(u))
        history = [1.0]
        if scale == 0:
            return u, [0.0]
        for _ in range(maxiter):
            u = self.cycle(b, u)
            history.append(float(np.linalg.norm(b - self.apply(u)) / scale))
            if not np.isfinite(history[-1]):
                raise ArithmeticError("multigrid residual is not finite")
            if history[-1] <= tol:
                break
        return u, history

    def sequential_solve(self, b):
        out = np.zeros_like(b)
        for n in range(len(b)):
            rhs = b[n].copy()
            if n:
                rhs += self.nt @ (self.space.mass @ out[n - 1].T).T
            out[n] = self.block_solve(rhs, direct=True)
        return out


def run_stmg_fem(output_dir, quick=False):
    """Measured local numerical baseline; not a historical scaling-table claim."""
    levels = space_hierarchy(1 if quick else 4)
    rows = []
    for steps in (2, 4) if quick else (2, 4, 8):
        t0 = perf_counter()
        mg = TimeMultigrid(levels, steps)
        setup = perf_counter() - t0
        rng = np.random.default_rng(20261110)
        exact = rng.uniform(-1, 1, (steps, 4, mg.n))
        rhs = mg.apply(exact)
        t0 = perf_counter()
        solution, history = mg.solve(rhs)
        elapsed = perf_counter() - t0
        t0 = perf_counter()
        sequential = mg.sequential_solve(rhs)
        sequential_seconds = perf_counter() - t0
        if history[-1] > 1e-8:
            raise ArithmeticError(f"STMG did not converge for {steps} time steps")
        rows.append(
            dict(
                time_steps=steps,
                spatial_dofs=mg.n,
                dofs=solution.size,
                iterations=len(history) - 1,
                relative_residual=history[-1],
                relative_solution_error=float(
                    np.linalg.norm(solution - exact) / np.linalg.norm(exact)
                ),
                sequential_difference=float(
                    np.linalg.norm(solution - sequential) / np.linalg.norm(sequential)
                ),
                setup_seconds=setup,
                solve_seconds=elapsed,
                sequential_seconds=sequential_seconds,
                residual_history=history,
            )
        )
        # Homogeneous problem and random (0,1) initial vector, as in the
        # source's convergence tests; stop on error rather than residual.
        initial = rng.uniform(0, 1, exact.shape)
        u = initial.copy()
        norm = np.linalg.norm(initial)
        error_history = [1.0]
        t0 = perf_counter()
        for _ in range(100):
            u = mg.cycle(np.zeros_like(u), u)
            error_history.append(float(np.linalg.norm(u) / norm))
            if error_history[-1] < 1e-8:
                break
        if error_history[-1] >= 1e-8:
            raise ArithmeticError(
                f"homogeneous STMG did not converge for {steps} time steps"
            )
        rows[-1]["homogeneous_test"] = dict(
            iterations=len(error_history) - 1,
            relative_error_history=error_history,
            solve_seconds=perf_counter() - t0,
        )
    payload = dict(
        source="Gander and Neumueller (2016), doi:10.1137/15M1046605, Example 6.1",
        execution="serial local numerical baseline; no parallel speedup claimed",
        mesh=dict(tetrahedra=levels[-1].tetrahedra, interior_vertices=mg.n),
        configuration=dict(
            degree=3,
            time_step=0.1,
            space_gauss_seidel_damping=1.285,
            pre=1,
            post=1,
            time_damping=1.0,
            time_cycle="V",
            spatial_relaxation="damped_gs",
            spatial_post_smoothing="forward",
            coarsest_time_solver="direct DG diagonalization",
        ),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            platform=platform.platform(),
        ),
        rows=rows,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stmg_fem_3d.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)
