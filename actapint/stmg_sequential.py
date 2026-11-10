"""Sequential time stepping with the same iterative spatial V-cycle as STMG."""

import numpy as np
from .stmg_fem import SpatialVCycle, dg_matrices


class ForwardSubstitution:
    def __init__(self, levels, dt=0.1, degree=3):
        self.kt, self.mt, self.nt, _ = dg_matrices(degree, dt)
        self.space = levels[-1]
        self.n = self.space.mass.shape[0]
        self.q = degree + 1
        self.spatial = SpatialVCycle(levels, self.kt, self.mt)

    def apply(self, u):
        out = np.array(
            [
                (self.spatial.a[-1] @ row.T.ravel()).reshape(self.n, self.q).T
                for row in u
            ]
        )
        for j in range(1, len(u)):
            out[j] -= self.nt @ (self.space.mass @ u[j - 1].T).T
        return out

    def solve(self, b, tol=1e-8, maxiter=100):
        if (
            b.ndim != 3
            or b.shape[1:] != (self.q, self.n)
            or not np.isfinite(b).all()
            or tol <= 0
            or maxiter < 1
        ):
            raise ValueError(
                "finite time-by-degree-by-space rhs and positive limits required"
            )
        solution = np.zeros_like(b)
        iterations, final_relative = [], []
        a = self.spatial.a[-1]
        for j in range(len(b)):
            rhs = b[j].copy()
            if j:
                rhs += self.nt @ (self.space.mass @ solution[j - 1].T).T
            rhs = rhs.T.ravel()
            x = np.zeros_like(rhs)
            r = rhs.copy()
            scale = np.linalg.norm(rhs)
            count = 0
            relative = 0.0 if scale == 0 else 1.0
            while relative > tol and count < maxiter:
                x += self.spatial.cycle(r)
                r = rhs - a @ x
                relative = float(np.linalg.norm(r) / scale)
                count += 1
                if not np.isfinite(relative):
                    raise ArithmeticError("nonfinite spatial residual")
            if relative > tol:
                raise ArithmeticError(f"spatial solve failed at slab {j}: {relative}")
            solution[j] = x.reshape(self.n, self.q).T
            iterations.append(count)
            final_relative.append(relative)
        return solution, iterations, final_relative
