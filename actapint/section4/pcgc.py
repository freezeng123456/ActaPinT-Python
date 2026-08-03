"""Parareal with a diagonalization-based coarse grid correction.

Port of `PCGC_Parareal_Heat_ADE.m` and `PCGC_Parareal_Burgers.m`, which produce
Figures 4.12 and 4.13.  The variant (4.15) replaces the sequential coarse grid
correction by a head-tail coupled one, ``U_0^{k+1} = alpha U_{N_t}^{k+1} + u_0``,
so that the whole correction becomes the all-at-once system (4.16) with an
alpha-circulant in time, which ParaDiag solves in parallel by the three steps
(4.17).  Theorem 4.7 says the price is nothing at all as long as
``alpha <= rho / (1 + rho)``, and the two figures are a test of that threshold:
each panel draws three values of ``alpha`` around it, plus the standard
correction, which is the same iteration with ``alpha = 0``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from ..plotting import save_figure, style_axis
from .common import (
    BURGERS_FLUX_SCALE_AS_COMMITTED,
    SDIRK23_GAMMA,
    BurgersStepper,
    alpha_circulant_solve,
    burgers_operators,
    convergence_factors,
    finish,
    linear_operator,
    linear_propagator,
    matlab_random_iterate,
    parareal,
    sequential_solution,
)

# Both scripts share the discretization the text states for this subsection: the
# heat equation and the ADE with periodic ends and ``u(x,0) = sin(2 pi x)``,
# backward Euler for G and an SDIRK for F.  The scripts assemble the operator on
# ``x = -1:dx:1``, which is 257 unknowns, but the stated convergence factors
# ``rho ~ 0.22`` and ``rho ~ 0.39`` belong to the 128 unknowns of the interval
# (0,1) that the text describes, so that is what is used here.
DX = 1.0 / 128
T_FINAL = 4.0
COARSE_DT = 0.1
SUBSTEPS = 10
FINE_METHOD = "sdirk23"  # `time_integrator_F=23`; the text says SDIRK22, and the
FINE_METHOD_IN_TEXT = "sdirk22"  # two give the same rho to three digits here
ALPHAS = (0.4, 0.25, 0.1, 0.0)
TOLERANCE = 1e-10
ITERATIONS = 35

# The published annotations: the convergence factor of standard Parareal and the
# threshold rho/(1+rho) of Theorem 4.7, for the two linear panels.
PUBLISHED_RHO = {"heat": 0.22, "ade": 0.39}
PUBLISHED_THRESHOLD = {"heat": 0.18, "ade": 0.28}

# Figure 4.13 uses the same set-up for Burgers' equation with two diffusion
# parameters; `PCGC_Parareal_Burgers.m` ships ``nu = 0.01``, which reproduces the
# right panel, and the left one is the weaker nonlinearity of ``nu = 0.1``.
#
# These two panels are the one place in the chapter where the flux of the script,
# ``0.5 (u**2)_x``, is also the flux of the published figure: with the ``(u**2)_x``
# that Figures 4.5 and 4.11 need, the quasi-Newton iteration (4.19) diverges from
# the random starting iterate instead of reproducing the panels.
BURGERS_NU_VALUES = (0.1, 0.01)
BURGERS_NU_IN_SCRIPT = 0.01
BURGERS_ITERATIONS = 44
NEWTON_MAX = 100

CURVE_STYLES = (
    (0.4, "#B33F62", "s", r"Diag-based CGC: $\alpha=0.4$"),
    (0.25, "#17324D", "+", r"Diag-based CGC: $\alpha=0.25$"),
    (0.1, "#2F6BFF", "o", r"Diag-based CGC: $\alpha=0.1$"),
    (0.0, "#D1495B", "*", "standard CGC"),
)


def initial_condition(x):
    """``u(x,0) = sin(2 pi x)``, as both scripts and the text set it."""
    return np.sin(2.0 * np.pi * x)


def _linear_history(nu, alpha, *, iterations=ITERATIONS, tolerance=TOLERANCE):
    """Error history of one curve of Figure 4.12.

    Each iteration evaluates the defect ``F - G`` in parallel, assembles the
    right-hand side of (4.16) -- whose first block carries the head-tail term
    ``alpha U_{N_t}^k`` -- and solves the alpha-circulant system in one shot.
    With ``alpha = 0`` the system is block lower triangular and the iteration
    degenerates to the sequential correction, which is standard Parareal.
    """
    steps = int(round(T_FINAL / COARSE_DT))
    operator, x = linear_operator(DX, nu)
    fine_matrix = linear_propagator(
        operator, COARSE_DT / SUBSTEPS, FINE_METHOD, substeps=SUBSTEPS
    )
    coarse_matrix = linear_propagator(operator, COARSE_DT, "ie")
    fine = fine_matrix.__matmul__
    coarse = coarse_matrix.__matmul__
    u0 = initial_condition(x)
    reference = sequential_solution(fine_matrix, u0, steps)
    # Both scripts draw the starting iterate from ``random('unif',-1,1,...)``.
    start = matlab_random_iterate(u0, steps, -1.0, 1.0)
    if alpha == 0.0:
        _, errors = parareal(
            fine, coarse, reference, start, iterations=iterations, tolerance=tolerance
        )
        return errors

    eye = sparse.eye(u0.size, format="csr")
    coarse_operator = eye + COARSE_DT * operator
    current = np.array(start, dtype=float)
    errors = [float(np.abs(current - reference).max())]
    for _ in range(iterations):
        rhs = np.zeros((steps, u0.size))
        for n in range(steps):
            tail = alpha * current[steps] if n == 0 else current[n]
            rhs[n] = coarse_operator @ (fine(current[n]) - coarse(tail))
        current = np.vstack(
            [u0, alpha_circulant_solve(operator, COARSE_DT, alpha, rhs)]
        )
        errors.append(float(np.abs(current - reference).max()))
        if errors[-1] <= tolerance:
            break
    return errors


def _panel(ax, curves, title, *, tolerance, xmax):
    """One panel of Figures 4.12 and 4.13: four error curves against k."""
    top = max(max(errors) for errors in curves.values())
    for alpha, colour, marker, label in CURVE_STYLES:
        errors = curves.get(alpha)
        if errors is None:
            continue
        ax.semilogy(
            np.arange(len(errors)),
            errors,
            "-.",
            color=colour,
            marker=marker,
            markersize=7,
            markerfacecolor="none",
            linewidth=1.0,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("iteration index $k$")
    ax.set_ylabel("error")
    ax.set_xlim(0, xmax)
    ax.set_ylim(tolerance, top)
    style_axis(ax)
    ax.legend(fontsize=8.5)


def run_figure_4_12(output_dir, quick=False):
    """Figure 4.12: the threshold of Theorem 4.7, measured.

    For the heat equation ``rho = 0.23`` and the threshold is ``0.19``, so of the
    three values only ``alpha = 0.1`` keeps the standard rate and ``0.25`` and
    ``0.4`` are visibly slower.  For the ADE with ``nu = 0.1`` the factor is
    larger, ``rho = 0.40``, so the threshold moves up to ``0.28`` and
    ``alpha = 0.25`` now converges at the standard rate as well.  That is the
    whole point of the figure: the admissible ``alpha`` is not small, and the
    round-off amplification of the diagonalization stays negligible.
    """
    cases = (("heat", -1.0), ("ade", 0.1))
    if quick:
        cases = cases[:1]
    alphas = ALPHAS[-2:] if quick else ALPHAS
    iterations = 8 if quick else ITERATIONS

    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.9 * len(cases), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, (label, nu) in zip(axes, cases):
        operator, _ = linear_operator(DX, nu)
        factors = convergence_factors(operator, COARSE_DT, SUBSTEPS, FINE_METHOD)
        rho = factors["rho_l_max"]
        curves = {
            alpha: _linear_history(nu, alpha, iterations=iterations)
            for alpha in alphas
        }
        title = (
            "heat equation"
            if label == "heat"
            else rf"ADE with $\nu={nu:g}$"
        )
        _panel(ax, curves, title, tolerance=TOLERANCE, xmax=20 if label == "heat" else 22)
        panels.append(
            {
                "equation": label,
                "nu": nu,
                "rho": rho,
                "rho_published": PUBLISHED_RHO[label],
                "threshold": rho / (1.0 + rho),
                "threshold_published": PUBLISHED_THRESHOLD[label],
                "curves": {
                    f"alpha_{alpha:g}": errors for alpha, errors in curves.items()
                },
                "iterations_to_tolerance": {
                    f"alpha_{alpha:g}": len(errors) - 1
                    for alpha, errors in curves.items()
                },
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_12_diag_cgc")

    return finish(
        output_dir,
        "figure_4_12_diag_cgc",
        "Measured error of Parareal with the diagonalization-based coarse grid "
        "correction (4.15) for three values of alpha, against the standard "
        "correction, for the heat equation and the advection-diffusion equation",
        {
            "parameters": {
                "dx": DX,
                "unknowns": int(round(1.0 / DX)),
                "t_final": T_FINAL,
                "coarse_dt": COARSE_DT,
                "substeps": SUBSTEPS,
                "steps": int(round(T_FINAL / COARSE_DT)),
                "coarse_solver": "backward Euler",
                "fine_solver": FINE_METHOD,
                "fine_solver_stated_in_text": FINE_METHOD_IN_TEXT,
                "initial_condition": "sin(2 pi x)",
                "initial_iterate": "uniform random on (-1,1), MATLAB rng('default')",
                "alphas": list(alphas),
                "tolerance": TOLERANCE,
            },
            "panels": panels,
        },
        artifacts,
    )


def _burgers_history(nu, alpha, *, iterations=BURGERS_ITERATIONS, tolerance=TOLERANCE):
    """Error history of one curve of Figure 4.13.

    The nonlinear correction is (4.18), solved by the quasi-Newton method (4.19):
    the Jacobian of the whole space-time system is replaced by
    ``C_alpha kron I - DT I kron J``, with ``J`` the average of the Jacobians over
    the coarse levels, so every Newton step is one alpha-circulant solve.  The
    inner tolerance is the script's ``max(Tol, 10**-k)``, which lets the early
    iterations stop long before the correction is solved accurately.
    """
    steps = int(round(T_FINAL / COARSE_DT))
    a1, a2, x = burgers_operators(DX, nu, flux_scale=BURGERS_FLUX_SCALE_AS_COMMITTED)
    stepper = BurgersStepper(a1, a2, tolerance / 3.0, NEWTON_MAX)
    fine = _burgers_fine(stepper)
    coarse = lambda u: stepper(COARSE_DT, u)  # noqa: E731 - reads as the script's G
    u0 = initial_condition(x)
    reference = sequential_solution(fine, u0, steps)
    start = matlab_random_iterate(u0, steps, -1.0, 1.0)
    if alpha == 0.0:
        _, errors = parareal(
            fine, coarse, reference, start, iterations=iterations, tolerance=tolerance
        )
        return errors

    current = np.array(start, dtype=float)
    errors = [float(np.abs(current - reference).max())]
    a1_dense, a2_dense = a1.toarray(), a2.toarray()
    for k in range(1, iterations + 1):
        defect = np.zeros((steps, u0.size))
        for n in range(steps):
            tail = alpha * current[steps] + u0 if n == 0 else current[n]
            defect[n] = fine(current[n]) - coarse(tail)
        interior = current[1:].copy()
        inner_tolerance = max(tolerance, 10.0 ** (-k))
        for _ in range(NEWTON_MAX):
            shifted = interior - defect
            residual = np.zeros_like(interior)
            for n in range(steps):
                flux = a2 @ shifted[n] + a1 @ shifted[n] ** 2
                previous = alpha * interior[steps - 1] + u0 if n == 0 else interior[n - 1]
                residual[n] = shifted[n] - previous + COARSE_DT * flux
            if np.abs(residual).max() <= inner_tolerance:
                break
            jacobian = a2_dense + 2.0 * (a1_dense * shifted.mean(axis=0)[None, :])
            interior = interior - alpha_circulant_solve(
                jacobian, COARSE_DT, alpha, residual
            )
        current = np.vstack([u0, interior])
        errors.append(float(np.abs(current - reference).max()))
        if errors[-1] <= tolerance:
            break
    return errors


def _burgers_fine(stepper):
    """F: ``SUBSTEPS`` steps of the script's SDIRK23 for Burgers' equation."""
    dt = COARSE_DT / SUBSTEPS
    gamma = SDIRK23_GAMMA

    def propagate(y):
        z = np.asarray(y, dtype=float)
        for _ in range(SUBSTEPS):
            stage1 = stepper(gamma * dt, z)
            stage2 = stepper(gamma * dt, z - (1.0 - 2.0 * gamma) * dt * stepper.flux(stage1))
            z = z - 0.5 * dt * (stepper.flux(stage1) + stepper.flux(stage2))
        return z

    return propagate


def run_figure_4_13(output_dir, quick=False):
    """Figure 4.13: the same threshold survives the nonlinearity.

    The correction is now the nonlinear system (4.18), and the quasi-Newton
    method (4.19) keeps one alpha-circulant solve per step by averaging the
    Jacobians over the coarse grid.  The influence of ``alpha`` is the one the
    linear panels show, which is the point of the figure.
    """
    nus = BURGERS_NU_VALUES[:1] if quick else BURGERS_NU_VALUES
    alphas = ALPHAS[-2:] if quick else ALPHAS
    iterations = 6 if quick else BURGERS_ITERATIONS

    fig, axes = plt.subplots(
        1, len(nus), figsize=(5.9 * len(nus), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, nu in zip(axes, nus):
        curves = {
            alpha: _burgers_history(nu, alpha, iterations=iterations)
            for alpha in alphas
        }
        _panel(
            ax,
            curves,
            rf"Burgers' equation with $\nu={nu:g}$",
            tolerance=TOLERANCE,
            xmax=20,
        )
        panels.append(
            {
                "nu": nu,
                "curves": {
                    f"alpha_{alpha:g}": errors for alpha, errors in curves.items()
                },
                "iterations_to_tolerance": {
                    f"alpha_{alpha:g}": len(errors) - 1
                    for alpha, errors in curves.items()
                },
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_13_diag_cgc_burgers")

    return finish(
        output_dir,
        "figure_4_13_diag_cgc_burgers",
        "Measured error of Parareal with the diagonalization-based and the "
        "standard coarse grid correction for Burgers' equation with two values "
        "of the diffusion parameter",
        {
            "parameters": {
                "dx": DX,
                "t_final": T_FINAL,
                "coarse_dt": COARSE_DT,
                "substeps": SUBSTEPS,
                "steps": int(round(T_FINAL / COARSE_DT)),
                "coarse_solver": "implicit Euler with Newton",
                "fine_solver": FINE_METHOD,
                "initial_condition": "sin(2 pi x)",
                "boundary_condition": "periodic",
                "burgers_flux": "(u**2)_x, not the 0.5 (u**2)_x of the script",
                "initial_iterate": "uniform random on (-1,1), MATLAB rng('default')",
                "alphas": list(alphas),
                "nu_in_script": BURGERS_NU_IN_SCRIPT,
                "tolerance": TOLERANCE,
            },
            "panels": panels,
        },
        artifacts,
    )
