"""Shared pieces of the Section 3 reproduction.

Section 3 states its discretizations once, in (3.12), and then reuses them for
every method family: centred finite differences on the unit interval with
periodic boundary conditions, assembled into the linear system of ODEs (2.1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..operators import (
    dirichlet_first_derivative,
    dirichlet_negative_laplacian,
    periodic_first_derivative,
    periodic_negative_laplacian,
)


def ade_generator(n, dx, nu, boundary_condition="periodic"):
    """Return A of u'(t) = A u(t) + g(t) for the advection-diffusion equation.

    Equation (3.12) writes this as ``nu/dx**2 * A_xx - 1/(2*dx) * A_x``, whose
    spectrum lies in the left half-plane, so the semigroup is ``exp(A t)``.
    """
    if boundary_condition == "periodic":
        diffusion = periodic_negative_laplacian(n, dx, nu)
        advection = periodic_first_derivative(n, dx, 1.0)
    elif boundary_condition == "dirichlet":
        diffusion = dirichlet_negative_laplacian(n, dx, nu)
        advection = dirichlet_first_derivative(n, dx, 1.0)
    else:
        raise ValueError(f"unsupported boundary condition {boundary_condition!r}")
    return (-diffusion - advection).tocsr()


def heat_generator(n, dx, boundary_condition="dirichlet"):
    """Return A of u'(t) = A u(t) + g(t) for the heat equation (2.3)."""
    if boundary_condition == "periodic":
        return (-periodic_negative_laplacian(n, dx, 1.0)).tocsr()
    if boundary_condition == "dirichlet":
        return (-dirichlet_negative_laplacian(n, dx, 1.0)).tocsr()
    raise ValueError(f"unsupported boundary condition {boundary_condition!r}")


def grid(dx, boundary_condition="periodic"):
    """Return the unknown positions on Omega = (0, 1).

    The periodic grid carries x = 0 but not x = 1, so that (3.12) closes with a
    single wrap-around.  The Dirichlet grid carries only interior nodes.
    """
    n_cells = int(round(1.0 / dx))
    if boundary_condition == "periodic":
        return np.arange(n_cells) * dx
    if boundary_condition == "dirichlet":
        return np.arange(1, n_cells) * dx
    raise ValueError(f"unsupported boundary condition {boundary_condition!r}")


def relative_sup_error(approx, reference):
    """Max-norm error relative to the largest reference value, as in Section 3."""
    scale = np.max(np.abs(reference))
    if scale == 0.0:
        return float(np.max(np.abs(approx - reference)))
    return float(np.max(np.abs(approx - reference)) / scale)


def write_json(output_dir, name, payload):
    path = Path(output_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def finish(output_dir, name, caption, payload, artifacts):
    """Write one figure's metrics next to its rendered panels."""
    body = {"figure": caption, "artifacts": artifacts}
    body.update(payload)
    return write_json(output_dir, name, body)
