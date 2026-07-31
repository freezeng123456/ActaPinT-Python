"""Schwarz waveform relaxation, Section 3.2 (Figures 3.1 and 3.3).

Two very different messages live here.  For the parabolic advection-diffusion
equation, OSWR converges faster as the diffusion parameter shrinks, so PinT gets
*easier* in the advection-dominated limit (Figure 3.1).  For the wave equation,
SWR with plain Dirichlet transmission conditions converges in a finite number of
steps, and running it red-black is exactly unmapped tent pitching, which advances
the exact solution slab by slab (Figure 3.3).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.optimize import brentq
from scipy.sparse.linalg import splu

from ..operators import dirichlet_first_derivative, dirichlet_negative_laplacian
from ..plotting import save_figure, style_axis
from .common import finish

# Constant of Theorem 3.1 separating the two branches of the optimized Robin
# parameter, printed in the article as 1.618386576...
Y_CRITICAL = 1.618386576

# Parameters of the Figure 3.1 experiment, from the paragraph preceding it.
DOMAIN_LENGTH = 8.2
T_FINAL = 5.0
DT = 0.01
DX = 0.02
SUBDOMAINS = 4
OVERLAP_CELLS = 2
TOLERANCE = 1e-8
ADVECTION_SPEED = 1.0

# The eight diffusion parameters of Figure 3.1, which `SWR_for_ADE.m` writes as
# `nnu = 2.^(-7:0)`.
FIGURE_3_1_NU = tuple(2.0 ** np.arange(0, -8, -1))

# How the dimensionless p_tilde of Theorem 3.1 becomes the Robin parameter p of
# equation (3.1).  Theorem 3.1 prints ``p = p_tilde * s / nu``, but
# `SWR_for_ADE.m` computes ``p = tp*nu/Ldx``, that is ``p_tilde * nu / l``, and
# only the latter reproduces Figure 3.1(b).  See docs/SECTION3_REPRODUCTION.md.
ROBIN_SCALING = "upstream"
ROBIN_SCALING_PRINTED = "theorem"

# `SWR_for_ADE.m` scales the centred advection difference by ``dt/dx``, but a
# centred first derivative needs ``dt/(2*dx)``, so the committed script transports
# at speed 2 rather than the speed 1 of equation (2.5).  The published iteration
# counts are reproduced by the consistent coefficient, so that is the default and
# the committed one stays available.  See docs/SECTION3_REPRODUCTION.md.
ADVECTION_SCALE = 0.5
ADVECTION_SCALE_UPSTREAM = 1.0


def _r0(y, p, y0):
    """Convergence factor of OSWR in Fourier space, Theorem 3.1."""
    y = np.asarray(y, dtype=float)
    numerator = (y - p) ** 2 + y**2 - y0**2
    denominator = (y + p) ** 2 + y**2 - y0**2
    return np.abs(numerator / denominator) * np.exp(-y)


def _y_bar(y0, p):
    """The interior maximizer y_bar(y0, p_tilde) of Theorem 3.1."""
    discriminant = p * (-(p**3) - 4.0 * p**2 + (4.0 + 2.0 * y0**2) * p + 8.0 * y0**2)
    return np.sqrt((y0**2 + 2.0 * p + np.sqrt(max(discriminant, 0.0))) / 2.0)


def optimized_robin_parameter(
    overlap, nu, speed=ADVECTION_SPEED, scaling=ROBIN_SCALING
):
    """Return (p_star, p_tilde_star) of Theorem 3.1 for an overlap and nu.

    Below the critical overlap ratio the optimum equioscillates between y0 and
    the interior maximizer, equation (3.2a); at or above it the optimum is pinned
    by equation (3.2b) instead.  ``scaling`` selects how p_tilde becomes p; see
    ``ROBIN_SCALING``.
    """
    y0 = overlap / nu
    if y0 < Y_CRITICAL:

        def residual(p):
            return _r0(y0, p, y0) - _r0(_y_bar(y0, p), p, y0)

    else:

        def residual(p):
            return y0 - p * np.sqrt(p / (4.0 + p))

    lo, hi = 1e-12, 1e6
    if residual(lo) * residual(hi) > 0.0:
        raise RuntimeError(f"no sign change for the Robin parameter at nu={nu}")
    p_tilde = abs(brentq(residual, lo, hi, xtol=1e-15, rtol=8.9e-16))
    if scaling == "theorem":
        return p_tilde * speed / nu, p_tilde
    if scaling == "upstream":
        return p_tilde * nu / overlap, p_tilde
    raise ValueError(f"unknown Robin scaling {scaling!r}")


def convergence_factors(
    nu, *, overlap=OVERLAP_CELLS * DX, t_final=T_FINAL, dt=DT, samples=200001
):
    """Theoretical OSWR convergence factors for Dirichlet and Robin TCs.

    ``dirichlet`` is equation (3.3): setting ``p = infinity`` in ``R0`` leaves
    ``exp(-y)``, whose maximum over the resolved modes sits at ``y_min``.

    ``robin`` is the right-hand side of (3.2c), the value at which the optimized
    factor equioscillates.  That bound, not the maximum over ``[y_min, y_max]``,
    is what Figure 3.1(a) plots; ``robin_interval_max`` reports the maximum for
    comparison, and it is larger, so the inequality in (3.2c) does not hold as
    printed.  See docs/SECTION3_REPRODUCTION.md.
    """
    y0 = overlap / nu
    y_min = overlap * np.pi / (nu * t_final)
    y_max = overlap * np.pi / (nu * dt)
    y = np.logspace(np.log10(y_min), np.log10(y_max), samples)
    p_star, p_tilde = optimized_robin_parameter(overlap, nu)
    return {
        "nu": float(nu),
        "dirichlet": float(np.exp(-y_min)),
        "robin": float(_r0(_y_bar(y0, p_tilde), p_tilde, y0)),
        "robin_interval_max": float(np.max(_r0(y, p_tilde, y0))),
        "p_star": float(p_star),
        "p_tilde": float(p_tilde),
        "y0": float(y0),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "branch": "equioscillation" if y0 < Y_CRITICAL else "pinned",
    }


def initial_profile(x, length=DOMAIN_LENGTH):
    """u0(x) = exp(-10 (x - L/2)^2), the initial value used for Figure 3.1."""
    return np.exp(-10.0 * (x - length / 2.0) ** 2)


def _blocks(nu, dx, dt, n_intervals, count, q, advection_scale):
    """Assemble the block-diagonal Q and the coupling W of `SWR_for_ADE.m`.

    Each subdomain carries ``m`` unknowns and overlaps its neighbours by two
    cells.  ``q = 1/(1 + dx*p)`` parameterizes the transmission condition, with
    ``q = 0`` the Dirichlet limit ``p = infinity``.
    """
    m = int(round(1 + (n_intervals - 2) / count))
    r1 = nu * dt / dx**2
    r2 = advection_scale * dt / dx
    e = np.ones(m)
    size = count * m
    q_matrix = sparse.lil_matrix((size, size))
    coupling = sparse.lil_matrix((size, size))
    for s in range(count):
        a1 = sparse.diags((-e, 2.0 * e, -e), (-1, 0, 1), shape=(m, m), format="lil")
        a2 = sparse.diags((-e, e), (-1, 1), shape=(m, m), format="lil")
        if s > 0:
            a1[0, 0] = 2.0 - q
            a2[0, 0] = q
        if s < count - 1:
            a1[m - 1, m - 1] = 2.0 - q
            a2[m - 1, m - 1] = q
        block = sparse.identity(m, format="lil") + r1 * a1 + r2 * a2
        q_matrix[s * m : (s + 1) * m, s * m : (s + 1) * m] = block
        if s < count - 1:
            row = s * m + m - 1
            coupling[row, (s + 1) * m + 0] = -(r1 - r2) * q
            coupling[row, (s + 1) * m + 1] = r1 - r2
        if s > 0:
            row = s * m
            coupling[row, (s - 1) * m + m - 1] = -(r1 + r2) * q
            coupling[row, (s - 1) * m + m - 2] = r1 + r2
    return m, splu(q_matrix.tocsc()), coupling.tocsr()


def subdomain_nodes(x, count=SUBDOMAINS, m=None):
    """Global node positions of each subdomain's unknowns, as `SWR_for_ADE.m` cuts them."""
    n_intervals = x.size - 1
    if m is None:
        m = int(round(1 + (n_intervals - 2) / count))
    return [x[s * (m - 1) + 1 : s * (m - 1) + 1 + m] for s in range(count)]


def oswr_iterations(
    nu,
    transmission,
    *,
    dx=DX,
    dt=DT,
    length=DOMAIN_LENGTH,
    t_final=T_FINAL,
    count=SUBDOMAINS,
    tolerance=TOLERANCE,
    max_iterations=1000,
    robin_scaling=ROBIN_SCALING,
    robin_parameter=None,
    advection_scale=ADVECTION_SCALE,
):
    """Port of `SWR_for_ADE.m`: parallel SWR on four overlapping subdomains.

    The upstream script starts from a zero guess and stops on the difference
    between successive iterates, not on the distance to the converged solution
    that the article's text describes.  Returns the iteration count and history.
    """
    n_intervals = int(round(length / dx))
    x = np.arange(n_intervals + 1) * dx
    steps = int(round(t_final / dt))

    if transmission == "dirichlet":
        q = 0.0
        p_star = np.inf
    elif transmission == "robin":
        if robin_parameter is None:
            p_star, _ = optimized_robin_parameter(
                OVERLAP_CELLS * dx, nu, scaling=robin_scaling
            )
        else:
            p_star = float(robin_parameter)
        q = 1.0 / (1.0 + dx * p_star)
    else:
        raise ValueError(f"unknown transmission condition {transmission!r}")

    m, lu, coupling = _blocks(nu, dx, dt, n_intervals, count, q, advection_scale)
    nodes = np.concatenate(subdomain_nodes(x, count, m))
    previous = np.zeros((count * m, steps + 1))
    previous[:, 0] = initial_profile(nodes, length)
    current = previous.copy()

    history = []
    for iteration in range(1, max_iterations + 1):
        for step in range(steps):
            current[:, step + 1] = lu.solve(
                current[:, step] + coupling.dot(previous[:, step + 1])
            )
        increment = float(np.max(np.abs(current - previous)))
        history.append(increment)
        previous = current.copy()
        if increment <= tolerance:
            return iteration, history
    return max_iterations, history


def monolithic_solution(
    nu, *, dx=DX, dt=DT, length=DOMAIN_LENGTH, t_final=T_FINAL,
    advection_scale=ADVECTION_SCALE,
):
    """Backward Euler on the whole domain, as `SWR_for_ADE.m` computes `U_ref`."""
    n_intervals = int(round(length / dx))
    x = np.arange(n_intervals + 1) * dx
    n = n_intervals - 1
    e = np.ones(n)
    r1 = nu * dt / dx**2
    r2 = advection_scale * dt / dx
    matrix = (
        sparse.identity(n)
        + r1 * sparse.diags((-e, 2.0 * e, -e), (-1, 0, 1), shape=(n, n))
        + r2 * sparse.diags((-e, e), (-1, 1), shape=(n, n))
    )
    lu = splu(matrix.tocsc())
    steps = int(round(t_final / dt))
    u = np.zeros((x.size, steps + 1))
    u[:, 0] = initial_profile(x, length)
    u[0, 0] = u[-1, 0] = 0.0
    for step in range(steps):
        u[1:-1, step + 1] = lu.solve(u[1:-1, step])
    return x, u


def run_figure_3_1(output_dir, quick=False):
    """Figure 3.1: OSWR gets easier as the advection-diffusion problem stiffens."""
    nus = FIGURE_3_1_NU[::2] if quick else FIGURE_3_1_NU
    theory = [convergence_factors(nu) for nu in nus]
    measured = {}
    for transmission in ("dirichlet", "robin"):
        counts = []
        for nu in nus:
            iterations, _ = oswr_iterations(
                nu, transmission, t_final=T_FINAL if not quick else 1.0
            )
            counts.append(iterations)
        measured[transmission] = counts

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    styles = {
        "dirichlet": dict(color="#C0392B", marker="o", label="Dirichlet"),
        "robin": dict(color="#2C4B8B", marker="+", label="Robin"),
    }
    axes[0].loglog(
        nus, [t["dirichlet"] for t in theory], linestyle="-.", **styles["dirichlet"]
    )
    axes[0].loglog(nus, [t["robin"] for t in theory], linestyle="-.", **styles["robin"])
    axes[0].set_xlabel(r"$\nu$")
    axes[0].set_ylabel(r"Convergence factor $\rho$")
    axes[0].set_title("(a) theoretical convergence factor")
    for transmission, counts in measured.items():
        axes[1].loglog(nus, counts, linestyle="-.", **styles[transmission])
    axes[1].set_xlabel(r"$\nu$")
    axes[1].set_ylabel("Iteration number")
    axes[1].set_title("(b) four-subdomain OSWR, tolerance $10^{-8}$")
    for ax in axes:
        style_axis(ax)
        ax.legend()
    artifacts = save_figure(fig, f"{output_dir}/figure_3_1_oswr")

    return finish(
        output_dir,
        "figure_3_1_oswr",
        "OSWR for the advection-diffusion equation: both the theoretical "
        "convergence factor and the measured four-subdomain iteration count fall "
        "as the diffusion parameter shrinks",
        {
            "parameters": {
                "domain_length": DOMAIN_LENGTH,
                "t_final": T_FINAL,
                "dt": DT,
                "dx": DX,
                "subdomains": SUBDOMAINS,
                "overlap": OVERLAP_CELLS * DX,
                "tolerance": TOLERANCE,
                "robin_scaling": ROBIN_SCALING,
            },
            "nu": [float(nu) for nu in nus],
            "theory": theory,
            "iterations": measured,
        },
        artifacts,
    )
