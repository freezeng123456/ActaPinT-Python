"""Iterative ParaExp against standard Parareal on Burgers' equation.

Port of `ParaExp_Burgers.m`, which produces Figure 3.8.  Both methods are the
Parareal iteration of (3.20); they differ only in the coarse propagator.  Standard
Parareal takes one backward Euler step of the full nonlinear problem, while
iterative ParaExp applies the exponential of the linear diffusion part exactly and
leaves the nonlinearity entirely to the iteration.  That is why ParaExp wins when
diffusion dominates and diverges once advection takes over.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from ..operators import burgers_operators
from ..plotting import save_figure, style_axis
from .common import finish

# Parameters of `ParaExp_Burgers.m`.
DX = 1.0 / 100
T_FINAL = 2.0
COARSE_STEP = 0.01
SUBSTEPS = 20
THETA = 1.0  # theta = 1 is backward Euler
NEWTON_TOLERANCE = 1e-13
NEWTON_MAX = 100
ITERATIONS = 11  # iteration numbers 0 to 10, the range Figure 3.8 shows
STOP_TOLERANCE = 1e-10
# Once ParaExp diverges there is nothing left to measure, and every further
# iteration makes Newton's method inside the fine solver much more expensive.
DIVERGENCE_LIMIT = 1e6
NU_VALUES = (1.0, 0.1, 0.02)
PANEL_LABELS = ("a", "b", "c")


def setup(nu, dx=DX):
    """Operators and grid of (3.21): f(u) = A u + B u**2 with periodic wrap.

    The script keeps unknowns at both ends of [0, 1] and closes the periodic
    stencil across them, so there are 101 unknowns for dx = 1/100.  ``a1`` is
    ``-B`` and ``a2`` is ``-A``, matching the script's u' + a2 u + a1 u**2 = 0.
    """
    x = np.arange(0.0, 1.0 + dx / 2, dx)
    a1, a2 = burgers_operators(x.size, dx, nu, periodic=True)
    return a1.toarray(), a2.toarray(), x


def initial_condition(x):
    """The square pulse u(x, 0) = 1 on |x - 1/2| <= 1/5 of the script."""
    return (np.abs(x - 0.5) <= 0.2).astype(float)


def implicit_step(u, a1, a2, dt, theta=THETA):
    """One theta step of u' = -a2 u - a1 u**2, solved by Newton's method."""
    identity = np.eye(u.size)
    z = u.copy()
    quadratic = u**2
    iterations = 0
    for iterations in range(1, NEWTON_MAX + 1):
        residual = (
            z
            - u
            + dt * a2.dot(theta * z + (1.0 - theta) * u)
            + dt * a1.dot(theta * z**2 + (1.0 - theta) * quadratic)
        )
        if np.abs(residual).max() <= NEWTON_TOLERANCE:
            break
        jacobian = identity + dt * theta * a2 + 2.0 * dt * theta * a1 * z[None, :]
        z = z - np.linalg.solve(jacobian, residual)
    return z, iterations


def fine_propagator(u, a1, a2, coarse_step=COARSE_STEP, substeps=SUBSTEPS):
    """F of (3.20): ``substeps`` backward Euler steps across one coarse interval."""
    dt = coarse_step / substeps
    z = u
    total = 0
    for _ in range(substeps):
        z, iterations = implicit_step(z, a1, a2, dt)
        total += iterations
    return z, total


def reference_solution(a1, a2, u0, intervals, coarse_step=COARSE_STEP):
    """Serial fine solution, which the script uses as the exact answer."""
    u = np.zeros((intervals + 1, u0.size))
    u[0] = u0
    newton = np.zeros(intervals + 1, dtype=int)
    for n in range(intervals):
        u[n + 1], newton[n + 1] = fine_propagator(u[n], a1, a2, coarse_step)
    return u, newton


def parareal_errors(
    nu,
    *,
    dx=DX,
    t_final=T_FINAL,
    coarse_step=COARSE_STEP,
    substeps=SUBSTEPS,
    iterations=ITERATIONS,
    tolerance=STOP_TOLERANCE,
):
    """Errors of both coarse propagators, iteration by iteration.

    Both start from the same initial guess, the exponential coarse solution, so
    that iteration zero is common; that is what the script plots.
    """
    a1, a2, x = setup(nu, dx)
    intervals = int(round(t_final / coarse_step))
    u0 = initial_condition(x)
    reference, newton = reference_solution(a1, a2, u0, intervals, coarse_step)
    propagate_exp = expm(-coarse_step * a2)

    def coarse(kind, u):
        if kind == "exp":
            return propagate_exp.dot(u)
        return implicit_step(u, a1, a2, coarse_step)[0]

    guess = np.zeros((intervals + 1, x.size))
    guess[0] = u0
    for n in range(intervals):
        guess[n + 1] = coarse("exp", guess[n])

    errors = {}
    for kind in ("exp", "euler"):
        current = guess.copy()
        stored = guess.copy()  # coarse values of the previous iterate
        history = [float(np.abs(reference - current).max())]
        for _ in range(iterations - 1):
            nxt = np.zeros_like(current)
            nxt[0] = u0
            for n in range(intervals):
                advanced = coarse(kind, nxt[n])
                nxt[n + 1] = (
                    advanced
                    + fine_propagator(current[n], a1, a2, coarse_step, substeps)[0]
                    - stored[n + 1]
                )
                stored[n + 1] = advanced
            current = nxt
            history.append(float(np.abs(reference - current).max()))
            if history[-1] <= tolerance or history[-1] >= DIVERGENCE_LIMIT:
                break
        errors[kind] = history
    truncation = max(coarse_step / substeps, dx**2)
    return errors, truncation, intervals, int(newton[1:].mean())


def run_figure_3_8(output_dir, quick=False):
    """Figure 3.8: ParaExp beats Parareal only while diffusion dominates."""
    nus = NU_VALUES[:1] if quick else NU_VALUES
    iterations = 4 if quick else ITERATIONS
    t_final = 0.2 if quick else T_FINAL

    fig, axes = plt.subplots(
        1, len(nus), figsize=(4.6 * len(nus), 4.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    metrics = []
    for ax, label, nu in zip(axes, PANEL_LABELS, nus):
        errors, truncation, intervals, newton = parareal_errors(
            nu, iterations=iterations, t_final=t_final
        )
        for kind, colour, marker, name in (
            ("exp", "#2F6BFF", "o", "iterative ParaExp"),
            ("euler", "#D1495B", "+", "standard Parareal"),
        ):
            history = errors[kind]
            ax.semilogy(
                np.arange(len(history)),
                history,
                "-.",
                color=colour,
                marker=marker,
                markersize=8,
                markerfacecolor="none",
                linewidth=1.2,
                label=name,
            )
        ax.axhline(truncation, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(rf"({label}) $\nu={nu:g}$")
        ax.set_xlabel("iteration number")
        ax.set_ylabel("error")
        ax.set_xlim(0, max(len(errors["exp"]), len(errors["euler"])) - 1)
        style_axis(ax)
        ax.legend(fontsize=9)
        metrics.append(
            {
                "panel": label,
                "nu": nu,
                "truncation_error": truncation,
                "coarse_intervals": intervals,
                "mean_newton_per_coarse_interval": newton,
                "paraexp": errors["exp"],
                "parareal": errors["euler"],
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_3_8_paraexp")

    return finish(
        output_dir,
        "figure_3_8_paraexp",
        "Convergence of iterative ParaExp and standard Parareal for Burgers' "
        "equation as the diffusion parameter shrinks; the horizontal line is the "
        "truncation error max(dt, dx**2) at which one would stop",
        {
            "parameters": {
                "dx": DX,
                "t_final": t_final,
                "coarse_step": COARSE_STEP,
                "fine_step": COARSE_STEP / SUBSTEPS,
                "newton_tolerance": NEWTON_TOLERANCE,
                "iterations": iterations,
                "stop_tolerance": STOP_TOLERANCE,
            },
            "panels": metrics,
        },
        artifacts,
    )
