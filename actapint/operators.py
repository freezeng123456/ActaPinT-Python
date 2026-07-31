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


def neumann_first_derivative(n: int, dx: float, scale: float = 1.0):
    """Centred first derivative with the one-sided rows used by the upstream scripts.

    The end rows come from eliminating a ghost node with a second-order
    reflection, which turns the stencil into (-4/3, 4/3)/(2*dx).
    """
    e = np.ones(n)
    a = sparse.diags((-e, e), (-1, 1), shape=(n, n), format="lil")
    a[0, 0], a[0, 1] = -4.0 / 3.0, 4.0 / 3.0
    a[-1, -2], a[-1, -1] = -4.0 / 3.0, 4.0 / 3.0
    return (scale / (2.0 * dx) * a).tocsr()


def neumann_negative_laplacian(
    n: int, dx: float, scale: float = 1.0, corner: float = 1.0
):
    """Negative Laplacian with reflected end rows.

    ``corner`` is the end-row diagonal that replaces the interior value 2.
    The upstream scripts are not consistent here: the heat and wave branches use
    a first-order reflection (``corner=1``) while the Burgers branch uses the
    second-order variant (``corner=2/3``), so the choice is left explicit.
    """
    e = np.ones(n)
    a = sparse.diags((-e, 2.0 * e, -e), (-1, 0, 1), shape=(n, n), format="lil")
    a[0, 0] = corner
    a[-1, -1] = corner
    return (scale / dx**2 * a).tocsr()


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

