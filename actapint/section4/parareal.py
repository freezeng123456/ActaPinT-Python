"""Parareal: the two convergence regimes, the fine solver, and weak diffusion.

Port of `Parareal_Heat_ADE.m` and `Parareal_Burgers.m`, which between them produce
Figures 4.2 to 4.5.  The script carries the parameters of Figure 4.2 as commented
alternatives to those of Figures 4.4 and 4.5, and its two plotting blocks -- the
active three-dimensional scatter of the convergence factor and the commented
``semilogy`` of the two bounds against the measured error -- are the two figure
styles of this subsection.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import save_figure, style_axis
from .common import (
    burgers_implicit_step,
    burgers_operators,
    burgers_propagator,
    coarse_initial_iterate,
    constant_initial_iterate,
    convergence_factors,
    finish,
    linear_operator,
    linear_propagator,
    parareal,
    sequential_solution,
    superlinear_bound,
)

# Figure 4.2: the heat equation on a deliberately coarse space mesh, where both
# propagators are backward Euler, so that the two regimes are visible at all.
# The text states a coarsening factor of 10, but the published panels are only
# reproduced with 4; see docs/SECTION4_REPRODUCTION.md for the measurements.
FIGURE_4_2_DX = 1.0 / 5
FIGURE_4_2_SUBSTEPS = 4
FIGURE_4_2_SUBSTEPS_TEXT = 10
FIGURE_4_2_CASES = ((0.02, 6), (0.5, 64))
FIGURE_4_2_TOLERANCE = 2e-14
FIGURE_4_2_ITERATIONS = 50

# Figure 4.3: the same equation with diffusion 0.1 on a fine mesh, comparing
# three fine integrators at three coarsening factors.
FIGURE_4_3_DX = 1.0 / 256
FIGURE_4_3_NU = -0.1  # negative selects the heat equation with diffusion 0.1
FIGURE_4_3_T = 4.0
FIGURE_4_3_DT = 0.1
FIGURE_4_3_SUBSTEPS = (2, 10, 50)
FIGURE_4_3_METHODS = (("tr", "trapezoidal rule"), ("sdirk22", "SDIRK22"), ("sdirk23", "SDIRK23"))
FIGURE_4_3_PREDICTED_FACTOR = 0.3  # the constant factor of (4.8)
FIGURE_4_3_ITERATIONS = 40
FIGURE_4_3_TOLERANCE = 2e-14

# Figures 4.4 and 4.5: the advection-diffusion equation as diffusion weakens.
FIGURE_4_4_DX = 1.0 / 128
FIGURE_4_4_T = 4.0
FIGURE_4_4_DT = 0.1
FIGURE_4_4_SUBSTEPS = 32
NU_VALUES = (1.0, 0.1, 0.02)
# The published annotations of Figure 4.4, one per panel.
PUBLISHED_RHO_L_MAX = (0.23, 0.39, 0.79)
FIGURE_4_5_ITERATIONS = 44
FIGURE_4_5_TOLERANCE = 2e-14
BURGERS_TOLERANCE = 1e-13  # `Parareal_Burgers.m` stops one decade earlier

PANEL_LABELS = ("a", "b", "c")
CURVE_COLOURS = ("#D1495B", "#17324D", "#2F6BFF")
CURVE_MARKERS = ("o", "+", "o")


def initial_condition(x, square=True):
    """The scripts' ``sin(2 pi x)**2``; the text of Figure 4.5 says ``sin(2 pi x)``."""
    values = np.sin(2.0 * np.pi * x)
    return values**2 if square else values


def _linear_case(nu, dx, coarse_dt, substeps, method, *, boundary_condition="periodic"):
    """Propagators, reference trajectory, and starting iterate of one panel."""
    operator, x = linear_operator(dx, nu, boundary_condition=boundary_condition)
    fine = linear_propagator(operator, coarse_dt / substeps, method, substeps=substeps)
    coarse = linear_propagator(operator, coarse_dt, "ie")
    return operator, x, fine, coarse


def run_figure_4_2(output_dir, quick=False):
    """Figure 4.2: superlinear on a short interval, linear on a long one.

    The mesh is so coarse in space that the coarse propagator resolves the whole
    spectrum, which is what lets the superlinear bound (4.5a) be sharp on the
    short interval.  On the long interval the same bound is useless, while the
    linear factor (4.5b) describes the measured error.

    This is the one figure of the chapter that starts from the sequential coarse
    solution rather than from the initial condition, which is what puts its first
    measured error at the coarse-solver error rather than at O(1).
    """
    cases = FIGURE_4_2_CASES[:1] if quick else FIGURE_4_2_CASES
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.6 * len(cases), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, (t_final, steps) in zip(axes, cases):
        coarse_dt = t_final / steps
        operator, x, fine, coarse = _linear_case(
            -1.0, FIGURE_4_2_DX, coarse_dt, FIGURE_4_2_SUBSTEPS, "ie"
        )
        factors = convergence_factors(
            operator, coarse_dt, FIGURE_4_2_SUBSTEPS, "ie"
        )
        text_factors = convergence_factors(
            operator, coarse_dt, FIGURE_4_2_SUBSTEPS_TEXT, "ie"
        )
        u0 = initial_condition(x)
        reference = sequential_solution(fine, u0, steps)
        start = coarse_initial_iterate(lambda u: coarse @ u, u0, steps)
        _, errors = parareal(
            lambda u: fine @ u,
            lambda u: coarse @ u,
            reference,
            start,
            iterations=FIGURE_4_2_ITERATIONS,
            tolerance=FIGURE_4_2_TOLERANCE,
        )
        iterations = len(errors) - 1
        linear = errors[0] * factors["rho_l_max"] ** np.arange(iterations + 1)
        superlinear = errors[0] * superlinear_bound(
            factors["rho_s_base"], steps, iterations
        )
        ax.semilogy(
            np.arange(iterations + 1),
            superlinear,
            ":",
            color="#2F6BFF",
            linewidth=1.3,
            label=r"bound $\varrho_s$",
        )
        ax.semilogy(
            np.arange(iterations + 1),
            linear,
            "-.",
            color="#D1495B",
            linewidth=1.3,
            label=r"bound $\varrho_l$",
        )
        ax.semilogy(
            np.arange(iterations + 1),
            errors,
            "--",
            color="#17324D",
            marker="o",
            markersize=7,
            markerfacecolor="none",
            linewidth=1.2,
            label="measured error",
        )
        ax.set_title(rf"$(T, N_t)=({t_final:g}, {steps})$")
        ax.set_xlabel("iteration index")
        ax.set_xlim(0, iterations)
        ax.set_ylim(1e-16, 10.0)
        style_axis(ax)
        ax.legend(fontsize=9)
        panels.append(
            {
                "t_final": t_final,
                "steps": steps,
                "coarse_dt": coarse_dt,
                "rho_l_max": factors["rho_l_max"],
                "rho_s_base": factors["rho_s_base"],
                "rho_l_max_with_substeps_from_text": text_factors["rho_l_max"],
                "iterations_to_tolerance": iterations,
                "errors": errors,
                "bound_linear": linear.tolist(),
                "bound_superlinear": superlinear.tolist(),
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_2_parareal_regimes")

    return finish(
        output_dir,
        "figure_4_2_parareal_regimes",
        "Parareal for the heat equation in its two convergence regimes: "
        "superlinear on a short time interval, linear on a long one, against the "
        "bounds (4.5a) and (4.5b)",
        {
            "parameters": {
                "equation": "heat, periodic, u0 = sin(2 pi x)**2",
                "dx": FIGURE_4_2_DX,
                "substeps": FIGURE_4_2_SUBSTEPS,
                "substeps_stated_in_text": FIGURE_4_2_SUBSTEPS_TEXT,
                "coarse_solver": "backward Euler",
                "fine_solver": "backward Euler",
                "initial_iterate": "sequential coarse solution",
                "tolerance": FIGURE_4_2_TOLERANCE,
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_4_3(output_dir, quick=False):
    """Figure 4.3: the fine solver decides the rate only while J is small.

    All three integrators end up contracting by about 0.3 once the coarsening
    factor is large, which is the content of (4.8); the trapezoidal rule, which
    is A-stable but not L-stable, is the one that suffers at small J.
    """
    substep_counts = FIGURE_4_3_SUBSTEPS[:1] if quick else FIGURE_4_3_SUBSTEPS
    steps = int(round(FIGURE_4_3_T / FIGURE_4_3_DT))
    dx = 1.0 / 64 if quick else FIGURE_4_3_DX
    iterations = 12 if quick else FIGURE_4_3_ITERATIONS

    fig, axes = plt.subplots(
        1, len(substep_counts), figsize=(4.8 * len(substep_counts), 4.3),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, substeps in zip(axes, substep_counts):
        curves = []
        for (method, name), colour, marker in zip(
            FIGURE_4_3_METHODS, CURVE_COLOURS, ("o", "*", "s")
        ):
            operator, x, fine, coarse = _linear_case(
                FIGURE_4_3_NU, dx, FIGURE_4_3_DT, substeps, method
            )
            factors = convergence_factors(operator, FIGURE_4_3_DT, substeps, method)
            u0 = initial_condition(x)
            reference = sequential_solution(fine, u0, steps)
            start = constant_initial_iterate(u0, steps)
            _, errors = parareal(
                lambda u, fine=fine: fine @ u,
                lambda u, coarse=coarse: coarse @ u,
                reference,
                start,
                iterations=iterations,
                tolerance=FIGURE_4_3_TOLERANCE,
            )
            ax.semilogy(
                np.arange(len(errors)),
                errors,
                "-.",
                color=colour,
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                linewidth=1.1,
                label=name,
            )
            curves.append(
                {
                    "fine_solver": name,
                    "rho_l_max": factors["rho_l_max"],
                    "iterations_to_tolerance": len(errors) - 1,
                    "measured_rate": float(
                        np.exp(np.diff(np.log(np.maximum(errors, 1e-300)))[1:6].mean())
                    ),
                    "errors": errors,
                }
            )
        longest = max(len(curve["errors"]) for curve in curves)
        predicted = curves[0]["errors"][0] * FIGURE_4_3_PREDICTED_FACTOR ** np.arange(longest)
        ax.semilogy(
            np.arange(longest),
            predicted,
            "-",
            color="#B14AED",
            linewidth=1.2,
            label=r"prediction $\mathrm{Err}_0\times 0.3^k$",
        )
        ax.set_title(rf"$J={substeps}$")
        ax.set_xlabel("iteration index")
        ax.set_ylabel("error")
        ax.set_xlim(0, longest - 1)
        ax.set_ylim(1e-14, 10.0)
        style_axis(ax)
        ax.legend(fontsize=8)
        panels.append({"substeps": substeps, "curves": curves})
    artifacts = save_figure(fig, f"{output_dir}/figure_4_3_fine_solver")

    return finish(
        output_dir,
        "figure_4_3_fine_solver",
        "Parareal for the heat equation with three fine solvers at three "
        "coarsening factors; the straight line is the constant factor 0.3 of (4.8)",
        {
            "parameters": {
                "equation": "heat with diffusion 0.1, periodic, u0 = sin(2 pi x)**2",
                "dx": dx,
                "t_final": FIGURE_4_3_T,
                "coarse_dt": FIGURE_4_3_DT,
                "steps": steps,
                "coarse_solver": "backward Euler",
                "predicted_factor": FIGURE_4_3_PREDICTED_FACTOR,
                "tolerance": FIGURE_4_3_TOLERANCE,
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_4_4(output_dir, quick=False):
    """Figure 4.4: the convergence factor at every eigenvalue of DT A.

    The script plots ``rho_l`` of (4.5b) as a surface over the spectrum and
    annotates its maximum, which is the number that predicts the rate of
    Figure 4.5(a).  As diffusion weakens the spectrum turns from a segment on
    the real axis into an ellipse, and the maximum moves towards one.
    """
    nus = NU_VALUES[:1] if quick else NU_VALUES
    dx = 1.0 / 64 if quick else FIGURE_4_4_DX
    substeps = 8 if quick else FIGURE_4_4_SUBSTEPS

    fig = plt.figure(figsize=(4.6 * len(nus), 4.0), constrained_layout=True)
    panels = []
    for index, (nu, published) in enumerate(zip(nus, PUBLISHED_RHO_L_MAX)):
        operator, _ = linear_operator(dx, nu)
        factors = convergence_factors(operator, FIGURE_4_4_DT, substeps, "sdirk22")
        eigenvalues = factors["eigenvalues"]
        ax = fig.add_subplot(1, len(nus), index + 1, projection="3d")
        ax.plot(
            eigenvalues.real,
            eigenvalues.imag,
            factors["rho_l"],
            "+",
            color="#2F6BFF",
            markersize=5,
        )
        ax.set_xlabel(r"Re($\lambda(A)$)", fontsize=9, labelpad=-2)
        ax.set_ylabel(r"Im($\lambda(A)$)", fontsize=9, labelpad=-2)
        ax.set_zlabel(r"$\varrho_l(J,z)$", fontsize=9, labelpad=-4)
        ax.set_zlim(0.0, 1.0)
        ax.set_title(rf"$\nu={nu:g}$", fontsize=11)
        ax.tick_params(labelsize=7, pad=-1)
        ax.text2D(
            0.28,
            0.74,
            rf"$\varrho_{{l,\max}}={factors['rho_l_max']:.2f}$",
            transform=ax.transAxes,
            fontsize=10,
        )
        panels.append(
            {
                "nu": nu,
                "rho_l_max": factors["rho_l_max"],
                "published_rho_l_max": published,
                "rho_s_base": factors["rho_s_base"],
                "spectrum_real_range": [
                    float(eigenvalues.real.min()),
                    float(eigenvalues.real.max()),
                ],
                "spectrum_imaginary_range": [
                    float(eigenvalues.imag.min()),
                    float(eigenvalues.imag.max()),
                ],
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_4_convergence_factor")

    return finish(
        output_dir,
        "figure_4_4_convergence_factor",
        "The Parareal convergence factor of (4.5b) at every eigenvalue of DT A "
        "for the advection-diffusion equation; its maximum approaches one as the "
        "diffusion parameter shrinks",
        {
            "parameters": {
                "equation": "advection diffusion, periodic",
                "dx": dx,
                "coarse_dt": FIGURE_4_4_DT,
                "substeps": substeps,
                "coarse_solver": "backward Euler",
                "fine_solver": "SDIRK22",
            },
            "panels": panels,
        },
        artifacts,
    )


def _burgers_panel(nu, dx, coarse_dt, steps, substeps, iterations, tolerance):
    """Parareal on Burgers' equation, with Newton inside both propagators.

    `Parareal_Burgers.m` closes the periodic stencil, matching the boundary
    condition the text states for Figure 4.5, and drives Newton to ``Tol/3``.
    """
    a1, a2, x = burgers_operators(dx, nu, boundary_condition="periodic")
    a1, a2 = a1.toarray(), a2.toarray()
    u0 = initial_condition(x)
    fine = burgers_propagator(
        a1, a2, coarse_dt / substeps, substeps, tolerance / 3.0, "sdirk22"
    )

    def coarse(u):
        return burgers_implicit_step(a1, a2, coarse_dt, u, tolerance / 3.0)

    reference = sequential_solution(fine, u0, steps)
    start = constant_initial_iterate(u0, steps)
    return parareal(
        fine, coarse, reference, start, iterations=iterations, tolerance=tolerance
    )


def run_figure_4_5(output_dir, quick=False):
    """Figure 4.5: Parareal deteriorates as the problem becomes advective.

    Both panels use the same discretization; only the equation changes.  In the
    published panels the curves for the largest and the smallest diffusion carry
    each other's legend entry: the fastest curve there is labelled ``nu = 0.02``
    even though it decays at the rate that Figure 4.4 predicts for ``nu = 1``.
    The curves below are labelled by the diffusion parameter they were computed
    with, so the panel reads the way the surrounding text describes it.
    """
    nus = NU_VALUES[:1] if quick else NU_VALUES
    dx = 1.0 / 64 if quick else FIGURE_4_4_DX
    substeps = 8 if quick else FIGURE_4_4_SUBSTEPS
    steps = 10 if quick else int(round(FIGURE_4_4_T / FIGURE_4_4_DT))
    iterations = 12 if quick else FIGURE_4_5_ITERATIONS

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    panels = []
    for ax, title, kind in zip(
        axes, ("Advection diffusion equation", "Burgers' equation"), ("ade", "burgers")
    ):
        curves = []
        for nu, colour, marker in zip(nus, CURVE_COLOURS, CURVE_MARKERS):
            if kind == "ade":
                operator, x, fine, coarse = _linear_case(
                    nu, dx, FIGURE_4_4_DT, substeps, "sdirk22"
                )
                factors = convergence_factors(
                    operator, FIGURE_4_4_DT, substeps, "sdirk22"
                )
                u0 = initial_condition(x)
                reference = sequential_solution(fine, u0, steps)
                start = constant_initial_iterate(u0, steps)
                _, errors = parareal(
                    lambda u, fine=fine: fine @ u,
                    lambda u, coarse=coarse: coarse @ u,
                    reference,
                    start,
                    iterations=iterations,
                    tolerance=FIGURE_4_5_TOLERANCE,
                )
                rho = factors["rho_l_max"]
            else:
                _, errors = _burgers_panel(
                    nu, dx, FIGURE_4_4_DT, steps, substeps, iterations,
                    BURGERS_TOLERANCE,
                )
                rho = None
            ax.semilogy(
                np.arange(len(errors)),
                errors,
                "-.",
                color=colour,
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                linewidth=1.1,
                label=rf"$\nu={nu:g}$",
            )
            curves.append(
                {
                    "nu": nu,
                    "rho_l_max": rho,
                    "iterations_to_tolerance": len(errors) - 1,
                    "errors": errors,
                }
            )
        ax.set_title(title)
        ax.set_xlabel("iteration index")
        ax.set_ylabel("error")
        ax.set_xlim(0, max(len(curve["errors"]) for curve in curves) - 1)
        ax.set_ylim(1e-14, 10.0)
        style_axis(ax)
        ax.legend(fontsize=9)
        panels.append({"equation": kind, "curves": curves})
    artifacts = save_figure(fig, f"{output_dir}/figure_4_5_parareal_diffusion")

    return finish(
        output_dir,
        "figure_4_5_parareal_diffusion",
        "Parareal for the advection-diffusion equation and for Burgers' equation "
        "as the diffusion parameter shrinks",
        {
            "parameters": {
                "dx": dx,
                "t_final": FIGURE_4_4_T,
                "coarse_dt": FIGURE_4_4_DT,
                "steps": steps,
                "substeps": substeps,
                "coarse_solver": "backward Euler",
                "fine_solver": "SDIRK22",
                "initial_condition": "sin(2 pi x)**2",
                "initial_condition_stated_in_text": "sin(2 pi x)",
                "boundary_condition": "periodic",
                "burgers_flux": "(u**2)_x, not the 0.5 (u**2)_x of the script",
                "published_legend": "the nu = 1 and nu = 0.02 entries are swapped",
            },
            "panels": panels,
        },
        artifacts,
    )
