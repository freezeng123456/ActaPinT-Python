"""Section 2 of the paper: model problems linking the parabolic and hyperbolic world.

Reproduces Figures 2.1-2.4, that is 24 space-time solution panels, from the
upstream scripts ``Solution_Heat_ADE.m``, ``Solution_Bergers.m``,
``Solution_Wave.m`` and the source term ``g.m``.

Section 2 carries no convergence tables; its evidence is entirely the shape of
the space-time solutions, so the panel layout, axis orientation and colour
scaling are matched to the published figures and the per-panel scalars are
written to JSON for machine-readable comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .colormaps import parula
from .operators import (
    dirichlet_first_derivative,
    dirichlet_negative_laplacian,
    neumann_first_derivative,
    neumann_negative_laplacian,
    periodic_first_derivative,
    periodic_negative_laplacian,
)
from .plotting import save_figure

# Source term (2.4): four Gaussian heating events in the middle of the domain.
SOURCE_AMPLITUDE = 10.0
SOURCE_SHARPNESS = 200.0
SOURCE_TIMES = (0.1, 0.6, 1.35, 1.85)
SOURCE_CENTRE = 0.5

# Oscillating initial condition u0(x) = sin^2(8*pi*(1-x)^2) used by the
# zero-source panels 2.1(d), 2.2(d,h), 2.3(d,h) and 2.4(d).
IC_FREQUENCY = 8.0

# Coefficient of the Burgers flux.  Equation (2.6) and ``Solution_Bergers.m``
# both state 0.5, but the published panels of Figure 2.3 are only reproduced with
# 1.0; see docs/SECTION2_REPRODUCTION.md for the evidence.  The default therefore
# matches the figures rather than the printed equation.
BURGERS_FLUX_COEFFICIENT = 1.0
BURGERS_FLUX_COEFFICIENT_UPSTREAM = 0.5

T_FINAL = 3.0


def source_term(nodes, t, sharpness=SOURCE_SHARPNESS):
    """Evaluate g(x, t) of equation (2.4) on ``nodes``.

    Equation (2.4) is a sum of isotropic Gaussians, which factors into a fixed
    spatial envelope times a sum of pulses in time.  ``sharpness`` is the sigma
    of that equation: Section 2 uses 200, while Section 3 uses 1000 for a
    low-regularity solution and 5 for a smooth one.
    """
    envelope = SOURCE_AMPLITUDE * np.exp(-sharpness * (nodes - SOURCE_CENTRE) ** 2)
    pulses = sum(np.exp(-sharpness * (t - centre) ** 2) for centre in SOURCE_TIMES)
    return envelope * pulses


def initial_condition(nodes):
    """Evaluate u0(x) = sin^2(8*pi*(1-x)^2)."""
    return np.sin(IC_FREQUENCY * np.pi * (1.0 - nodes) ** 2) ** 2


def _grid(dt):
    """Upstream scripts always tie dx to dt and sample the closed interval."""
    dx = dt
    x = np.arange(0.0, 1.0 + 0.5 * dx, dx)
    steps = int(round(T_FINAL / dt))
    t = np.arange(steps + 1) * dt
    return x, dx, t, steps


def _node_view(x, boundary_condition):
    """Return the unknowns carried by the scheme for a boundary condition.

    Dirichlet eliminates both end points.  Neumann and periodic keep every
    sampled node, so the periodic grid inherits the upstream quirk of carrying
    both x=0 and x=1 as separate unknowns.
    """
    if boundary_condition == "dirichlet":
        return x[1:-1], slice(1, -1)
    return x, slice(None)


@dataclass
class PanelResult:
    """One space-time solution panel together with its scalar diagnostics."""

    label: str
    title: str
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    parameters: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def metrics(self):
        u = self.u
        final = u[:, -1]
        return {
            "label": self.label,
            "title": self.title,
            "parameters": self.parameters,
            "grid": {"nx": int(u.shape[0]), "nt": int(u.shape[1])},
            "min": float(u.min()),
            "max": float(u.max()),
            "final_time_linf": float(np.abs(final).max()),
            "final_time_mean": float(final.mean()),
            "peak_time": float(self.t[int(np.argmax(np.abs(u).max(axis=0)))]),
            **self.diagnostics,
        }


# --------------------------------------------------------------------------
# Heat equation (2.3) and advection-diffusion equation (2.5)
# --------------------------------------------------------------------------


def solve_heat_ade(
    *,
    nu,
    advection,
    boundary_condition,
    with_source,
    dt=1e-3,
):
    """Solve u_t + a u_x - nu u_xx = g on (0,1) x (0,T].

    Mirrors ``Solution_Heat_ADE.m``: the Dirichlet and periodic branches use
    IMEX Euler with implicit diffusion and explicit advection, while the Neumann
    branch uses the trapezoidal rule on the full operator.  ``advection=0``
    together with ``nu=1`` is the heat equation of Figure 2.1.
    """
    x, dx, t, steps = _grid(dt)
    nodes, interior = _node_view(x, boundary_condition)
    n = nodes.size

    if boundary_condition == "dirichlet":
        d1 = dirichlet_first_derivative(n, dx, advection)
        laplacian = -dirichlet_negative_laplacian(n, dx, nu)
        scheme = "imex-euler"
    elif boundary_condition == "periodic":
        d1 = periodic_first_derivative(n, dx, advection)
        laplacian = -periodic_negative_laplacian(n, dx, nu)
        scheme = "imex-euler"
    elif boundary_condition == "neumann":
        d1 = neumann_first_derivative(n, dx, advection)
        laplacian = -neumann_negative_laplacian(n, dx, nu, corner=1.0)
        scheme = "trapezoidal"
    else:
        raise ValueError(f"unknown boundary condition: {boundary_condition}")

    eye = sparse.eye(n, format="csc")
    u = np.zeros((x.size, steps + 1))
    if not with_source:
        u[:, 0] = initial_condition(x)

    if scheme == "imex-euler":
        solve = splu((eye - dt * laplacian).tocsc())
        for k in range(steps):
            state = u[interior, k]
            rhs = state - dt * (d1 @ state)
            if with_source:
                rhs = rhs + dt * source_term(nodes, t[k + 1])
            u[interior, k + 1] = solve.solve(rhs)
    else:
        operator = laplacian - d1
        solve = splu((eye - 0.5 * dt * operator).tocsc())
        explicit = (eye + 0.5 * dt * operator).tocsr()
        for k in range(steps):
            rhs = explicit @ u[interior, k]
            if with_source:
                rhs = rhs + 0.5 * dt * (
                    source_term(nodes, t[k]) + source_term(nodes, t[k + 1])
                )
            u[interior, k + 1] = solve.solve(rhs)

    return x, t, u, {"scheme": scheme, "dt": dt, "dx": dx, "nu": nu,
                     "advection": advection, "boundary_condition": boundary_condition,
                     "source": bool(with_source)}


# --------------------------------------------------------------------------
# Viscous Burgers equation (2.6)
# --------------------------------------------------------------------------


def solve_burgers(
    *,
    nu,
    boundary_condition,
    with_source,
    dt=1.0 / 400.0,
    newton_max=100,
    flux_coefficient=None,
):
    """Solve u_t - nu u_xx + c (u^2)_x = g with Crank-Nicolson and Newton.

    Mirrors ``Solution_Bergers.m`` including its Newton tolerance
    max(dt^2, dx^2)/10 and its iteration cap.  ``flux_coefficient`` defaults to
    the value that reproduces Figure 2.3; pass
    ``BURGERS_FLUX_COEFFICIENT_UPSTREAM`` to run the committed script instead.
    """
    x, dx, t, steps = _grid(dt)
    nodes, interior = _node_view(x, boundary_condition)
    n = nodes.size

    flux = BURGERS_FLUX_COEFFICIENT if flux_coefficient is None else flux_coefficient
    if boundary_condition == "dirichlet":
        a1 = dirichlet_first_derivative(n, dx, flux)
        a2 = dirichlet_negative_laplacian(n, dx, nu)
    elif boundary_condition == "periodic":
        a1 = periodic_first_derivative(n, dx, flux)
        a2 = periodic_negative_laplacian(n, dx, nu)
    elif boundary_condition == "neumann":
        a1 = neumann_first_derivative(n, dx, flux)
        a2 = neumann_negative_laplacian(n, dx, nu, corner=2.0 / 3.0)
    else:
        raise ValueError(f"unknown boundary condition: {boundary_condition}")

    eye = sparse.eye(n, format="csc")
    tol = max(dt**2, dx**2) / 10.0
    theta = 0.5

    u = np.zeros((x.size, steps + 1))
    if not with_source:
        u[:, 0] = initial_condition(x)

    newton_counts = []
    for k in range(steps):
        old = u[interior, k]
        forcing = (
            0.5 * (source_term(nodes, t[k]) + source_term(nodes, t[k + 1]))
            if with_source
            else 0.0
        )
        z = old.copy()
        for iteration in range(1, newton_max + 1):
            residual = (
                z
                - old
                + dt * (a2 @ (theta * z + (1.0 - theta) * old))
                + dt * (a1 @ (theta * z**2 + (1.0 - theta) * old**2))
                - dt * forcing
            )
            if np.linalg.norm(residual, np.inf) <= tol:
                break
            jacobian = eye + dt * theta * a2 + 2.0 * dt * theta * (a1 @ sparse.diags(z))
            z = z - splu(jacobian.tocsc()).solve(residual)
        newton_counts.append(iteration)
        u[interior, k + 1] = z

    return x, t, u, {
        "scheme": "crank-nicolson-newton",
        "dt": dt,
        "dx": dx,
        "nu": nu,
        "flux_coefficient": flux,
        "boundary_condition": boundary_condition,
        "source": bool(with_source),
        "newton_tolerance": tol,
        "newton_iterations_max": int(max(newton_counts)),
        "newton_iterations_mean": float(np.mean(newton_counts)),
    }


# --------------------------------------------------------------------------
# Second-order wave equation (2.7)
# --------------------------------------------------------------------------


def solve_wave(
    *,
    wave_speed_squared,
    boundary_condition,
    with_source,
    dt=1.0 / 400.0,
):
    """Solve u_tt = c^2 u_xx + g by trapezoidal rule on the first-order system.

    Mirrors ``Solution_Wave.m``.  The upstream Neumann branch assigns
    ``A2(1,1)=nu/dx^2`` although ``nu`` is never defined in that script, so the
    end-row diagonal is taken as ``c^2/dx^2``, which is the coefficient the
    surrounding branches use.
    """
    x, dx, t, steps = _grid(dt)
    nodes, interior = _node_view(x, boundary_condition)
    n = nodes.size

    if boundary_condition == "dirichlet":
        stiffness = dirichlet_negative_laplacian(n, dx, wave_speed_squared)
    elif boundary_condition == "periodic":
        stiffness = periodic_negative_laplacian(n, dx, wave_speed_squared)
    elif boundary_condition == "neumann":
        stiffness = neumann_negative_laplacian(n, dx, wave_speed_squared, corner=1.0)
    else:
        raise ValueError(f"unknown boundary condition: {boundary_condition}")

    zero = sparse.csc_matrix((n, n))
    eye = sparse.eye(n, format="csc")
    operator = sparse.bmat([[zero, -eye], [stiffness, zero]], format="csc")
    eye2 = sparse.eye(2 * n, format="csc")
    solve = splu((eye2 + 0.5 * dt * operator).tocsc())
    explicit = (eye2 - 0.5 * dt * operator).tocsr()

    u = np.zeros((x.size, steps + 1))
    if not with_source:
        u[:, 0] = initial_condition(x)

    # u_t(x,0) = 0, so the velocity block starts at zero.
    z = np.concatenate([u[interior, 0], np.zeros(n)])
    for k in range(steps):
        rhs = explicit @ z
        if with_source:
            rhs[n:] += 0.5 * dt * (
                source_term(nodes, t[k]) + source_term(nodes, t[k + 1])
            )
        z = solve.solve(rhs)
        u[interior, k + 1] = z[:n]

    return x, t, u, {
        "scheme": "trapezoidal-first-order-system",
        "dt": dt,
        "dx": dx,
        "wave_speed_squared": wave_speed_squared,
        "boundary_condition": boundary_condition,
        "source": bool(with_source),
    }


# --------------------------------------------------------------------------
# Panel specifications taken from the published figure captions and titles
# --------------------------------------------------------------------------

_BC_TITLE = {
    "dirichlet": "Dirichlet BCs",
    "neumann": "Neumann BCs",
    "periodic": "Periodic BCs",
}

FIGURE_2_1 = [
    ("a", {"boundary_condition": "dirichlet", "with_source": True}),
    ("b", {"boundary_condition": "neumann", "with_source": True}),
    ("c", {"boundary_condition": "periodic", "with_source": True}),
    ("d", {"boundary_condition": "periodic", "with_source": False}),
]

FIGURE_2_2 = [
    ("a", {"boundary_condition": "dirichlet", "with_source": True, "nu": 1.0}),
    ("b", {"boundary_condition": "dirichlet", "with_source": True, "nu": 1e-2}),
    ("c", {"boundary_condition": "dirichlet", "with_source": True, "nu": 5e-4}),
    ("d", {"boundary_condition": "dirichlet", "with_source": False, "nu": 5e-4}),
    ("e", {"boundary_condition": "periodic", "with_source": True, "nu": 1.0}),
    ("f", {"boundary_condition": "periodic", "with_source": True, "nu": 1e-2}),
    ("g", {"boundary_condition": "periodic", "with_source": True, "nu": 5e-4}),
    ("h", {"boundary_condition": "periodic", "with_source": False, "nu": 5e-4}),
]

FIGURE_2_3 = FIGURE_2_2

FIGURE_2_4 = [
    ("a", {"boundary_condition": "dirichlet", "with_source": True}),
    ("b", {"boundary_condition": "neumann", "with_source": True}),
    ("c", {"boundary_condition": "periodic", "with_source": True}),
    ("d", {"boundary_condition": "periodic", "with_source": False}),
]


def _panel(label, spec, solver, extra):
    x, t, u, parameters = solver(**{**spec, **extra})
    return PanelResult(
        label=label,
        title=_BC_TITLE[spec["boundary_condition"]],
        x=x,
        t=t,
        u=u,
        parameters=parameters,
    )


def _plot_panels(panels, rows, columns, output_path, *, max_pixels=700):
    """Lay panels out as in the paper: x horizontal, t vertical, parula colours."""
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.35 * columns, 5.4 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, panel in zip(axes, panels):
        u = panel.u
        step_x = max(1, u.shape[0] // max_pixels)
        step_t = max(1, u.shape[1] // (2 * max_pixels))
        field = u[::step_x, ::step_t]
        ax.imshow(
            field.T,
            extent=(panel.x[0], panel.x[-1], panel.t[0], panel.t[-1]),
            origin="lower",
            aspect="auto",
            cmap=parula(),
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(panel.title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_yticks(np.arange(0.0, T_FINAL + 0.01, 0.5))
        ax.tick_params(labelsize=8.5)
        ax.text(
            0.5,
            -0.115,
            f"({panel.label})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    return save_figure(fig, output_path)


def _write(output_dir, name, payload):
    path = Path(output_dir) / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _finish(output_dir, name, caption, panels, rows, columns):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _plot_panels(panels, rows, columns, output_dir / name)
    return _write(
        output_dir,
        name,
        {
            "figure": caption,
            "panels": [panel.metrics() for panel in panels],
            "artifacts": artifacts,
        },
    )


def run_figure_2_1(output_dir, quick=False):
    """Figure 2.1: heat equation under Dirichlet, Neumann and periodic BCs."""
    dt = 4e-3 if quick else 1e-3
    panels = [
        _panel(label, spec, solve_heat_ade,
               {"nu": 1.0, "advection": 0.0, "dt": dt})
        for label, spec in FIGURE_2_1
    ]
    return _finish(
        output_dir,
        "figure_2_1_heat",
        "Heat equation: locality in time under Dirichlet versus accumulation "
        "under Neumann and periodic boundary conditions",
        panels,
        1,
        4,
    )


def run_figure_2_2(output_dir, quick=False):
    """Figure 2.2: advection-diffusion with an ever smaller diffusion parameter."""
    dt = 4e-3 if quick else 1e-3
    panels = [
        _panel(label, spec, solve_heat_ade, {"advection": 1.0, "dt": dt})
        for label, spec in FIGURE_2_2
    ]
    return _finish(
        output_dir,
        "figure_2_2_advection_diffusion",
        "Advection-diffusion equation: fine information travels far in time once "
        "the diffusion parameter is small and the boundary is periodic",
        panels,
        2,
        4,
    )


def run_figure_2_3(output_dir, quick=False):
    """Figure 2.3: viscous Burgers equation and shock formation."""
    dt = 1.0 / (160 if quick else 400)
    panels = [
        _panel(label, spec, solve_burgers, {"dt": dt})
        for label, spec in FIGURE_2_3
    ]
    return _finish(
        output_dir,
        "figure_2_3_burgers",
        "Burgers' equation: nonlinear steepening generates high-frequency shocks "
        "that travel far in space and time",
        panels,
        2,
        4,
    )


def run_figure_2_4(output_dir, quick=False):
    """Figure 2.4: second-order wave equation for all three boundary conditions."""
    dt = 1.0 / (160 if quick else 400)
    panels = [
        _panel(label, spec, solve_wave,
               {"wave_speed_squared": 0.2, "dt": dt})
        for label, spec in FIGURE_2_4
    ]
    return _finish(
        output_dir,
        "figure_2_4_wave",
        "Second-order wave equation: detailed long-time propagation for every "
        "boundary condition",
        panels,
        1,
        4,
    )


def run_section2(output_dir, quick=False):
    """Run every Figure 2.x panel and write a combined summary."""
    output_dir = Path(output_dir)
    figures = {
        "figure_2_1": run_figure_2_1(output_dir, quick=quick),
        "figure_2_2": run_figure_2_2(output_dir, quick=quick),
        "figure_2_3": run_figure_2_3(output_dir, quick=quick),
        "figure_2_4": run_figure_2_4(output_dir, quick=quick),
    }
    summary = {}
    total = 0
    for name, path in figures.items():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        summary[name] = {
            "figure": payload["figure"],
            "panels": len(payload["panels"]),
            "metrics": str(path),
        }
        total += len(payload["panels"])
    summary["panels_total"] = total
    summary["quick"] = bool(quick)
    return _write(output_dir, "section2_summary", summary)
