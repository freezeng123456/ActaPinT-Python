from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _solver(matrix):
    return splu(sparse.csc_matrix(matrix))


def linear_step_matrix(a, dt: float, method: str = "sdirk22"):
    """Build the one-step propagation matrix for u_t + A u = 0."""
    n = a.shape[0]
    eye = sparse.eye(n, format="csc")
    a = sparse.csc_matrix(a)
    if method == "ie":
        solve = _solver(eye + dt * a)
        return solve.solve(np.eye(n))
    if method == "tr":
        solve = _solver(eye + 0.5 * dt * a)
        return solve.solve((eye - 0.5 * dt * a).toarray())
    if method == "sdirk22":
        gamma = (2.0 - np.sqrt(2.0)) / 2.0
        solve = _solver(eye + gamma * dt * a)
        g1 = solve.solve(np.eye(n))
        return solve.solve(np.eye(n) - (1.0 - gamma) * dt * (a @ g1))
    if method == "sdirk23":
        gamma = (3.0 + np.sqrt(3.0)) / 6.0
        solve = _solver(eye + gamma * dt * a)
        g1 = solve.solve(np.eye(n))
        g2 = solve.solve(np.eye(n) - (1.0 - 2.0 * gamma) * dt * (a @ g1))
        return np.eye(n) - 0.5 * dt * (a @ (g1 + g2))
    raise ValueError(f"unknown integrator: {method}")


def repeated_linear_step(a, dt: float, substeps: int, method: str = "sdirk22"):
    p = linear_step_matrix(a, dt, method)
    return np.linalg.matrix_power(p, substeps)


def burgers_ie_step(a1, a2, tau: float, y, tol: float = 1e-12, max_iter: int = 80):
    y = np.asarray(y, dtype=float)
    z = y.copy()
    eye = sparse.eye(y.size, format="csc")
    for _ in range(max_iter):
        residual = z + tau * (a2 @ z + a1 @ (z**2)) - y
        if np.linalg.norm(residual, np.inf) <= tol:
            return z
        jacobian = eye + tau * (a2 + 2.0 * a1 @ sparse.diags(z))
        z -= splu(jacobian.tocsc()).solve(residual)
    raise RuntimeError("Newton iteration did not converge")


def burgers_fine_step(
    a1,
    a2,
    y,
    dt: float,
    substeps: int,
    method: str = "sdirk22",
    tol: float = 1e-12,
):
    z = np.asarray(y, dtype=float).copy()
    gamma = (2.0 - np.sqrt(2.0)) / 2.0
    for _ in range(substeps):
        if method == "ie":
            z = burgers_ie_step(a1, a2, dt, z, tol)
        elif method == "tr":
            rhs = z - 0.5 * dt * (a2 @ z + a1 @ (z**2))
            z = burgers_ie_step(a1, a2, 0.5 * dt, rhs, tol)
        elif method == "sdirk22":
            g1 = burgers_ie_step(a1, a2, gamma * dt, z, tol)
            rhs = z - (1.0 - gamma) * dt * (a2 @ g1 + a1 @ (g1**2))
            z = burgers_ie_step(a1, a2, gamma * dt, rhs, tol)
        else:
            raise ValueError(f"unknown integrator: {method}")
    return z

