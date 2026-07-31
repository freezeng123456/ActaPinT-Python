from __future__ import annotations

import numpy as np
from scipy import sparse


def periodic_first_derivative(n: int, dx: float, scale: float = 1.0):
    e = np.ones(n)
    a = sparse.diags((-e, e), (-1, 1), shape=(n, n), format="lil")
    a[0, -1] = -1.0
    a[-1, 0] = 1.0
    return (scale / (2.0 * dx) * a).tocsr()


def periodic_negative_laplacian(n: int, dx: float, scale: float = 1.0):
    e = np.ones(n)
    a = sparse.diags((-e, 2.0 * e, -e), (-1, 0, 1), shape=(n, n), format="lil")
    a[0, -1] = -1.0
    a[-1, 0] = -1.0
    return (scale / dx**2 * a).tocsr()


def dirichlet_first_derivative(n: int, dx: float, scale: float = 1.0):
    e = np.ones(n)
    return (scale * sparse.diags((-e, e), (-1, 1), shape=(n, n)) / (2.0 * dx)).tocsr()


def dirichlet_negative_laplacian(n: int, dx: float, scale: float = 1.0):
    e = np.ones(n)
    return (scale * sparse.diags((-e, 2.0 * e, -e), (-1, 0, 1), shape=(n, n)) / dx**2).tocsr()


def linear_ade_operator(n: int, dx: float, nu: float, periodic: bool = True):
    """Return A for u_t + A u = 0, matching the MATLAB sign convention."""
    if periodic:
        d1 = periodic_first_derivative(n, dx)
        d2 = periodic_negative_laplacian(n, dx)
    else:
        d1 = dirichlet_first_derivative(n, dx)
        d2 = dirichlet_negative_laplacian(n, dx)
    if nu < 0:
        return abs(nu) * d2
    return d1 + nu * d2


def burgers_operators(n: int, dx: float, nu: float, periodic: bool = False):
    """Return A1, A2 for u_t + A2 u + A1(u**2) = 0."""
    if periodic:
        a1 = periodic_first_derivative(n, dx, scale=0.5)
        a2 = periodic_negative_laplacian(n, dx, scale=nu)
    else:
        a1 = dirichlet_first_derivative(n, dx, scale=0.5)
        a2 = dirichlet_negative_laplacian(n, dx, scale=nu)
    return a1.tocsr(), a2.tocsr()

