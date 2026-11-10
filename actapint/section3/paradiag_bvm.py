"""ParaDiag I on a boundary value method discretization.

Port of `Direct_ParaDiag_Burgers.m` (Figure 3.13, Figure 3.14 and Table 3.2) plus
Figure 3.12, which the scripts do not cover.  The geometric mesh of
`paradiag_direct` buys diagonalizability at the price of a badly conditioned
eigenvector matrix; the boundary value method (3.38) instead keeps the mesh
uniform and centres the time derivative, closing the system with one backward
Euler step at the end, and its time-stepping matrix has Cond(V) = O(N_t^2).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..operators import periodic_first_derivative, periodic_negative_laplacian
from ..plotting import save_figure, style_axis
from .common import finish
from .paradiag_direct import (
    error_at,
    exponential_reference,
    uniform_error,
    wave_operator,
)

# Parameters of `Direct_ParaDiag_Burgers.m`.
BURGERS_DX = 0.01
BURGERS_STEPS = 200
BURGERS_NU = (0.1, 0.002)
BURGERS_HORIZONS = (0.1, 0.2, 0.4, 0.8, 1.6)
MAX_ITERATIONS = 30
NEWTON_TOLERANCE = BURGERS_DX**2 / 100.0
DIVERGENCE_LIMIT = 1e8
RANDOM_SEED = 20240501

# Parameters of Figure 3.14.  The text states the fine mesh, dx = 1/200, the
# coarse mesh the scaling factors are computed on, DX = 1/20, and the two
# horizons; it states neither the time-step nor the starting iterate, and the
# horizontal line of the panels is drawn at 6.4e-5 and 5.6e-5 rather than at any
# max(dt^2, dx^2), so it fixes neither.  What the digitized panels do fix is the
# observable and the starting iterate.  The two variants of a panel share their
# first value to the last digit of the vector art, and the first value cannot
# depend on the variant unless it is measured at the starting iterate, so the
# panels plot the distance to the solution of the all-at-once system and not the
# difference of two successive iterates that `Direct_ParaDiag_Burgers.m` plots.
# At nu = 0.02 both variants start at 1.99, which is the
# max-norm distance from a uniform random iterate to the solution of the
# all-at-once system, 1 + max|U| = 1.992, and no other start comes near it.  Of
# the two random ranges only (-1, 1), the range of the initial condition, has the
# zero mean that makes the averaged Jacobian right from the first iterate, and it
# puts the whole averaged curve of panel (a) on the published one; (0, 1) spends
# three iterations at 1.3 that the published panel does not have.  At nu = 0.002
# every random start diverges for all four curves while the published panel has
# three of them converging, so that panel is read as the constant start; both are
# run at both viscosities and both are in the metrics file.
NKA_DX = 1.0 / 200
NKA_COARSE_DX = 1.0 / 20
NKA_DT = 0.01
NKA_NU = (0.02, 0.002)
NKA_HORIZONS = (1.3, 0.7)
NKA_MAX_ITERATIONS = 80
NKA_TOLERANCE = 1e-6
NKA_STARTS = {0.02: "symmetric", 0.002: "initial"}

# The weight scan runs on a coarser fine mesh than the figure: the contraction
# factor of both variants is a property of the continuous problem, and moving
# dx from 1/200 to 1/50 changes the measured decay per iteration by 2%, while
# the scan costs sixteen times less there.
NKA_SCAN_DX = 1.0 / 50
NKA_SCAN_RATIOS = (2, 5, 10, 20, 50)

# The published curves of Figure 3.14, read off the article page by
# `scripts/digitize_figure_3_14.py`; the calibration is in
# docs/data/figure_3_14_published.json.  The figure is vector art in the PDF, so
# these are the plotted values themselves rather than a reading of a raster.
# Panel (b) is drawn with a stride, a vertex every third or every second
# iteration.
PUBLISHED_FIGURE_3_14 = {
    (0.02, "average", 1.3): {
        0: 1.993, 1: 0.5843, 2: 0.24, 3: 0.1111, 4: 0.04734, 5: 0.01534,
        6: 0.006006, 7: 0.00373, 8: 0.002032, 9: 0.001196, 10: 0.0005605,
        11: 0.0003012, 12: 0.0001266, 13: 6.173e-05, 14: 2.483e-05, 15: 1.076e-05,
        16: 4.574e-06, 17: 1.627e-06, 18: 7.17e-07,
    },
    (0.02, "average", 0.7): {
        0: 1.961, 1: 0.6179, 2: 0.1514, 3: 0.04477, 4: 0.01085, 5: 0.00275,
        6: 0.0005309, 7: 0.0001103, 8: 3.148e-05, 9: 9.165e-06, 10: 2.996e-06,
        11: 7.976e-07,
    },
    (0.02, "nka", 1.3): {
        0: 1.993, 1: 0.03525, 2: 0.0109, 3: 0.00451, 4: 0.001261, 5: 0.0004617,
        6: 0.0001231, 7: 3.899e-05, 8: 1.048e-05, 9: 2.811e-06, 10: 7.9e-07,
    },
    (0.02, "nka", 0.7): {
        0: 1.961, 1: 0.0163, 2: 0.002424, 3: 0.000424, 4: 6.695e-05, 5: 9.212e-06,
        6: 1.767e-06,
    },
    (0.002, "average", 1.3): {
        0: 3.091, 1: 2.629, 2: 3.078, 3: 3.993, 4: 5.08, 5: 9.904, 6: 86.18,
    },
    (0.002, "average", 0.7): {
        0: 2.703, 3: 0.7012, 6: 0.3063, 9: 0.4659, 12: 0.8383, 15: 0.4091,
        18: 0.1632, 21: 0.04853, 24: 0.01069, 27: 0.002179, 30: 0.0003467,
        33: 4.171e-05, 36: 4.681e-06,
    },
    (0.002, "nka", 1.3): {
        0: 3.091, 3: 0.07473, 6: 0.04592, 9: 0.02845, 12: 0.01653, 15: 0.008702,
        18: 0.004076, 21: 0.001707, 24: 0.0006317, 27: 0.0002006, 30: 5.447e-05,
        33: 2.163e-05, 36: 1.261e-05, 39: 6.774e-06, 42: 3.384e-06, 45: 1.556e-06,
        48: 6.584e-07,
    },
    (0.002, "nka", 0.7): {
        0: 2.703, 2: 0.2091, 4: 0.03169, 6: 0.006312, 8: 0.002693, 10: 0.001047,
        12: 0.0003191, 14: 9.222e-05, 16: 2.874e-05, 18: 8.233e-06, 20: 2.393e-06,
    },
}

# Parameters of Figure 3.12.  The text states dx = 1/40 and T = 0.5 for step
# counts 2^2 to 2^8, but the published axes say otherwise: the markers sit at
# N_t = 5, 8, 16, ..., 256 in panel (a) and carry on to 512 in panel (b), and the
# horizon is 0.4.  With those, all seven trapezoidal markers, all seven BVM
# markers and all eight BVM condition numbers come out within one per cent.
FIGURE_3_12_DX = 1.0 / 40
FIGURE_3_12_T = 0.4
FIGURE_3_12_STEPS = (5, 8, 16, 32, 64, 128, 256, 512)
FIGURE_3_12_ERROR_STEPS = FIGURE_3_12_STEPS[:-1]
# The mesh ratio of the geometric curve is not stated, and neither the closed form
# (3.37) nor the minimizer over a scan reproduces the published pair of panels.
# Cond(V) does pin it down, because it falls by a factor of thirty for every ten
# per cent added to the ratio; these are the values it gives, read against the
# published panel (b) whose calibration the BVM branch of the same panel confirms
# to two per cent.  N_t = 64 sits in a crowd of markers and is interpolated.
FIGURE_3_12_RHO = {
    5: 0.095,
    8: 0.160,
    16: 0.244,
    32: 0.283,
    64: 0.295,
    128: 0.303,
    256: 0.303,
    512: 0.303,
}

PANEL_LABELS = ("a", "b", "c", "d")


def burgers_operators(nu, dx):
    """The two operators of `Direct_ParaDiag_Burgers.m` for u' = -f(u).

    Burgers' equation (2.6) is written there as u' - nu u_xx + (u^2/2)_x = 0, so
    f(u) = advection u^2 + diffusion u with the advection matrix carrying the
    factor one half of the flux.
    """
    x = np.arange(0.0, 1.0 + dx / 2, dx)
    advection = 0.5 * periodic_first_derivative(x.size, dx, 1.0).toarray()
    diffusion = periodic_negative_laplacian(x.size, dx, nu).toarray()
    return advection, diffusion, x


def initial_condition(x):
    return np.sin(2.0 * np.pi * x)


def residual_operator(advection, diffusion):
    """f(u) and its Jacobian, as the `f` and `df` helpers of the script."""

    def f(u):
        return advection.dot(u**2) + diffusion.dot(u)

    def df(u):
        return 2.0 * advection * u[None, :] + diffusion

    return f, df


def trapezoidal_steps(advection, diffusion, u0, steps, dt, tolerance=NEWTON_TOLERANCE):
    """Sequential trapezoidal rule with Newton, and the work it costs.

    The counter follows the script: it counts residual evaluations, which is one
    more than the number of Jacobian solves at each step, and that is the number
    Table 3.2 reports for the sequential method.
    """
    f, df = residual_operator(advection, diffusion)
    identity = np.eye(u0.size)
    solution = np.zeros((steps + 1, u0.size))
    solution[0] = u0
    evaluations = np.zeros(steps, dtype=int)
    for n in range(steps):
        previous = solution[n]
        state = previous.copy()
        for count in range(1, MAX_ITERATIONS + 1):
            residual = previous - 0.5 * dt * f(previous) - state - 0.5 * dt * f(state)
            if np.abs(residual).max() <= tolerance:
                break
            jacobian = identity + 0.5 * dt * df(state)
            state = state + np.linalg.solve(jacobian, residual)
        solution[n + 1] = state
        evaluations[n] = count
    return solution, evaluations


def bvm_matrix(steps, dt):
    """B of (3.39b): centred differences closed by one backward Euler step."""
    matrix = np.zeros((steps, steps))
    index = np.arange(steps - 1)
    matrix[index, index + 1] = 0.5
    matrix[index + 1, index] = -0.5
    matrix[-1, -2] = -1.0
    matrix[-1, -1] = 1.0
    return matrix / dt


def _diagonalized_solve(spectrum, rhs, shift_matrix, *, scaled_shift=False):
    """The three steps of (3.46), with ``rhs`` shaped (steps, unknowns)."""
    eigenvalues, eigenvectors = spectrum
    transformed = np.linalg.solve(eigenvectors, rhs.astype(complex))
    identity = np.eye(shift_matrix.shape[0])
    for n, value in enumerate(eigenvalues):
        block = (
            identity + value * shift_matrix
            if scaled_shift
            else value * identity + shift_matrix
        )
        transformed[n] = np.linalg.solve(block, transformed[n])
    return (eigenvectors @ transformed).real


def quasi_newton(
    advection,
    diffusion,
    u0,
    steps,
    dt,
    *,
    variant="average",
    scaling=None,
    iterations=MAX_ITERATIONS,
    tolerance=NEWTON_TOLERANCE,
    start="random",
    seed=RANDOM_SEED,
    reference=None,
):
    """ParaDiag I as a quasi-Newton iteration on the all-at-once system (3.42).

    The averaged Jacobian (3.44) replaces every block by a single matrix, which
    is what makes the diagonalization possible.  ``variant="nka"`` keeps the
    nearest Kronecker product weights (3.48) in front of it, so that the time
    matrix to diagonalize becomes B^-1 Phi rather than B.  The reported error is
    the difference between two successive iterates, as in the script, unless a
    ``reference`` solution is given, in which case it is the max-norm distance to
    it starting at the initial iterate, the convention used here for Figure 3.14.

    `Direct_ParaDiag_Burgers.m` offers two starting iterates, a uniform random
    vector, which is the one its code uses, and the initial condition repeated at
    every time level, which is left commented out.  The random one reproduces
    which entries of Table 3.2 diverge; the constant one is needed on the fine
    mesh of Figure 3.14, where the random start diverges throughout.
    ``start="symmetric"`` draws the random vector from (-1, 1) instead of (0, 1),
    which is the range of the initial condition and the only start that puts the
    averaged variant of Figure 3.14(a) on the published curve.
    """
    f, df = residual_operator(advection, diffusion)
    b = np.zeros((steps, u0.size))
    b[0] = u0 / (2.0 * dt)
    matrix = bvm_matrix(steps, dt)
    scaled_shift = variant.startswith("nka")
    online = variant == "nka_online"
    inverse = np.linalg.inv(matrix) if scaled_shift else None
    if scaled_shift and not online:
        time_matrix = inverse * scaling[None, :]
    else:
        time_matrix = matrix
    # With weights from the reduced model the mesh, and hence the time matrix, is
    # the same at every iteration, so it is diagonalized once; recomputing the
    # weights from the current iterate is the expensive variant of (3.48).
    spectrum = None if online else np.linalg.eig(time_matrix)

    if start == "initial":
        iterate = np.tile(u0, (steps, 1))
    else:
        low = -1.0 if start == "symmetric" else 0.0
        iterate = np.random.default_rng(seed).uniform(low, 1.0, size=(steps, u0.size))
    history = [] if reference is None else [float(np.abs(iterate - reference).max())]
    for _ in range(iterations):
        averaged = df(iterate.mean(axis=0))
        nonlinear = np.array([f(row) for row in iterate])
        if online:
            denominator = float((averaged * averaged).sum())
            weights = np.array(
                [float((df(row) * averaged).sum()) / denominator for row in iterate]
            )
            time_matrix = inverse * weights[None, :]
            spectrum = np.linalg.eig(time_matrix)
        if scaled_shift:
            rhs = inverse.dot(b - nonlinear) + time_matrix.dot(iterate).dot(averaged.T)
        else:
            rhs = b - nonlinear + iterate.dot(averaged.T)
        nxt = _diagonalized_solve(spectrum, rhs, averaged, scaled_shift=scaled_shift)
        change = float(
            np.abs(nxt - (iterate if reference is None else reference)).max()
        )
        history.append(change)
        iterate = nxt
        if change <= tolerance or not np.isfinite(change) or change >= DIVERGENCE_LIMIT:
            break
    converged = bool(history and history[-1] <= tolerance)
    return iterate, history, converged


def all_at_once_solution(advection, diffusion, u0, steps, dt, tolerance=1e-12):
    """The solution of the all-at-once system (3.42), by Newton on the sparse form.

    The Jacobian is block tridiagonal in time, so one sparse solve per Newton step
    settles it; the quasi-Newton iterations of Figure 3.14 are measured against
    this under the max-norm comparison convention.
    """
    f, df = residual_operator(advection, diffusion)
    matrix = bvm_matrix(steps, dt)
    time_part = sparse.kron(sparse.csr_matrix(matrix), sparse.eye(u0.size), "csr")
    b = np.zeros((steps, u0.size))
    b[0] = u0 / (2.0 * dt)
    iterate = np.tile(u0, (steps, 1))
    for _ in range(MAX_ITERATIONS):
        residual = b - matrix.dot(iterate) - np.array([f(row) for row in iterate])
        if np.abs(residual).max() < tolerance:
            break
        blocks = sparse.block_diag([sparse.csr_matrix(df(row)) for row in iterate], format="csr")
        update = spsolve((time_part + blocks).tocsc(), residual.ravel())
        iterate = iterate + update.reshape(steps, u0.size)
    # The last Newton update may have exhausted the iteration limit. Always
    # report the residual of the returned solution, not of its predecessor.
    residual = b - matrix.dot(iterate) - np.array([f(row) for row in iterate])
    return iterate, float(np.abs(residual).max())


def nka_weights(nu, dt, steps, *, coarse_dx=NKA_COARSE_DX, part="jacobian",
                against="mean", levels=None):
    """The weights (3.48) of the reduced model on the coarse space mesh.

    The paper computes them once, before the iteration, from a trapezoidal rule
    solve on a ten times coarser grid, so they cost almost nothing.  The
    keywords are the parts of the construction the text leaves open and the scan
    varies: ``part="advective"`` fits only the term the Jacobian varies in,
    ``against="initial"`` measures the blocks against the first one rather than
    against the average (3.44), and ``levels`` runs the reduced model on fewer
    time levels and interpolates the weights back.
    """
    advection, diffusion, x = burgers_operators(nu, coarse_dx)
    _, df = residual_operator(advection, diffusion)
    solution, _ = trapezoidal_steps(
        advection, diffusion, initial_condition(x), steps, dt
    )
    solution = solution[1:]
    index = np.arange(steps)
    if levels is not None:
        index = np.linspace(0, steps - 1, levels).round().astype(int)
        solution = solution[index]
    if part == "advective":
        blocks = [2.0 * advection * row[None, :] for row in solution]
    else:
        blocks = [df(row) for row in solution]
    averaged = (
        blocks[0]
        if against == "initial"
        else (2.0 * advection * solution.mean(axis=0)[None, :]
              if part == "advective" else df(solution.mean(axis=0)))
    )
    denominator = float((averaged * averaged).sum())
    weights = np.array([float((block * averaged).sum()) / denominator
                        for block in blocks])
    return weights if levels is None else np.interp(np.arange(steps), index, weights)


def nka_weight_scan(nu, dx, dt, steps, *, iterations=NKA_MAX_ITERATIONS):
    """Iteration counts of the weight constructions the article leaves open.

    Each candidate is run from both starting iterates the script offers and
    measured against the same observable as the published panels, the max-norm
    distance to the solution of the all-at-once system.  The mesh ratios are
    against the fine mesh of the figure, so that the coarse meshes are the same
    ones whatever mesh ``dx`` the iteration itself is run on.  The last three
    entries are not constructions at all but the text's weights pushed away from
    one by a factor, which is what shows that no diagonal Phi of this shape
    reaches the published gain at nu = 0.02.
    """
    advection, diffusion, x = burgers_operators(nu, dx)
    u0 = initial_condition(x)
    reference, residual = all_at_once_solution(advection, diffusion, u0, steps, dt)
    assert residual < 1e-9, residual
    text = nka_weights(nu, dt, steps)
    candidates = [("averaged Jacobian (3.44), Phi = I", None)]
    for ratio in NKA_SCAN_RATIOS:
        coarse = ratio * NKA_DX
        candidates.append(
            (f"reduced model, DX/dx = {ratio} (DX = 1/{round(1 / coarse)})",
             nka_weights(nu, dt, steps, coarse_dx=coarse))
        )
    try:
        candidates.append(("reduced model at nu/10", nka_weights(nu / 10.0, dt, steps)))
    except np.linalg.LinAlgError:
        # At nu = 0.002 the coarse mesh cannot carry a tenth of the viscosity.
        pass
    candidates += [
        ("reduced model at 10 nu", nka_weights(nu * 10.0, dt, steps)),
        ("normalized to mean one", text / text.mean()),
        ("normalized to first block one", text / text[0]),
        ("unit Frobenius norm", text * np.sqrt(steps) / np.linalg.norm(text)),
        ("measured against the first block",
         nka_weights(nu, dt, steps, against="initial")),
        ("advective part only", nka_weights(nu, dt, steps, part="advective")),
        ("ten time levels", nka_weights(nu, dt, steps, levels=10)),
        ("exact Jacobians on the fine mesh", nka_weights(nu, dt, steps, coarse_dx=dx)),
    ]
    candidates += [
        (f"the text's weights spread by {factor:g}", 1.0 + factor * (text - 1.0))
        for factor in (2.0, 5.0, 10.0)
    ]

    records = []
    for name, weights in candidates:
        record = {"variant": name}
        if weights is not None:
            record["weights"] = [float(weights.min()), float(weights.max())]
        for start in ("symmetric", "initial"):
            _, history, _ = quasi_newton(
                advection, diffusion, u0, steps, dt,
                variant="average" if weights is None else "nka",
                scaling=weights, iterations=iterations, tolerance=NKA_TOLERANCE,
                start=start, reference=reference,
            )
            record[start] = _decay(history)
        records.append(record)
    return records


def _decay(history):
    """Decay per iteration and the count to six decades, or nothing if it diverges."""
    values = np.asarray(history)
    if not np.isfinite(values).all() or values[-1] > values[0]:
        return {"iterations": None, "decay": float("nan"), "final": float(values[-1])}
    logs = np.log10(values)
    below = np.nonzero(logs <= logs[0] - 6.0)[0]
    window = slice(max(1, len(logs) // 2), len(logs))
    fit = np.arange(len(logs))[window]
    return {
        "iterations": int(below[0]) if len(below) else None,
        "decay": float(np.polyfit(fit, logs[window], 1)[0]) if len(fit) > 2 else
        float("nan"),
        "final": float(values[-1]),
    }


def _error_history_panel(ax, curves, title, tolerance_line):
    """Draw one panel of Figures 3.13 and 3.14."""
    for curve in curves:
        history = curve["history"]
        ax.semilogy(
            np.arange(len(history)),
            history,
            curve["style"],
            color=curve["colour"],
            marker=curve["marker"],
            markersize=7,
            markerfacecolor="none",
            linewidth=1.1,
            label=curve["label"],
        )
    ax.axhline(tolerance_line, color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("iteration index")
    ax.set_ylabel("error")
    ax.set_ylim(NEWTON_TOLERANCE, 100.0)
    style_axis(ax)
    ax.legend(fontsize=8)


def run_figure_3_13(output_dir, quick=False):
    """Figure 3.13: ParaDiag I with BVM on Burgers' equation, for five horizons."""
    horizons = BURGERS_HORIZONS[:2] if quick else BURGERS_HORIZONS
    steps = 40 if quick else BURGERS_STEPS
    palette = ("#D1495B", "#00A6A6", "#2F6BFF", "#17324D", "#B14AED")
    markers = ("+", "o", "s", "*", "d")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    panels = []
    for ax, label, nu in zip(axes, PANEL_LABELS, BURGERS_NU):
        advection, diffusion, x = burgers_operators(nu, BURGERS_DX)
        u0 = initial_condition(x)
        curves = []
        records = []
        for horizon, colour, marker in zip(horizons, palette, markers):
            dt = horizon / steps
            _, history, converged = quasi_newton(
                advection, diffusion, u0, steps, dt,
                iterations=MAX_ITERATIONS,
            )
            curves.append(
                {
                    "history": history,
                    "style": ":",
                    "colour": colour,
                    "marker": marker,
                    "label": rf"$T={horizon:g}$",
                }
            )
            records.append(
                {
                    "horizon": horizon,
                    "dt": dt,
                    "iterations": len(history),
                    "converged": converged,
                    "errors": history,
                }
            )
        _error_history_panel(
            ax, curves, rf"({label}) $\nu={nu:g}$", max(BURGERS_DX**2, (horizons[-1] / steps) ** 2)
        )
        panels.append({"panel": label, "nu": nu, "runs": records})

    artifacts = save_figure(fig, f"{output_dir}/figure_3_13_paradiag_bvm_burgers")
    return finish(
        output_dir,
        "figure_3_13_paradiag_bvm_burgers",
        "Error of ParaDiag I with the BVM discretization (3.38) for Burgers' "
        "equation with two diffusion parameters; the number of steps is fixed at "
        "N_t = T/dt = 200 and the horizontal line is the space-time discretization "
        "error max(dt^2, dx^2) = 1e-4",
        {
            "parameters": {
                "dx": BURGERS_DX,
                "steps": steps,
                "tolerance": NEWTON_TOLERANCE,
                "max_iterations": MAX_ITERATIONS,
                "seed": RANDOM_SEED,
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_3_14(output_dir, quick=False):
    """Figure 3.14: the averaged Jacobian against the nearest Kronecker product."""
    horizons = NKA_HORIZONS[:1] if quick else NKA_HORIZONS
    dx = 1.0 / 50 if quick else NKA_DX
    iterations = 20 if quick else NKA_MAX_ITERATIONS

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    panels = []
    for ax, label, nu in zip(axes, PANEL_LABELS, NKA_NU):
        advection, diffusion, x = burgers_operators(nu, dx)
        u0 = initial_condition(x)
        start = NKA_STARTS[nu]
        curves = []
        records = []
        plan = [(v, h) for v in ("average", "nka") for h in horizons]
        for (variant, horizon), marker in zip(plan, ("+", "x", "o", "s")):
            colour = "#2F6BFF" if variant == "average" else "#D1495B"
            steps = int(round(horizon / NKA_DT))
            reference, residual = all_at_once_solution(
                advection, diffusion, u0, steps, NKA_DT
            )
            weights = None if variant == "average" else nka_weights(nu, NKA_DT, steps)
            record = {"variant": variant, "horizon": horizon, "steps": steps,
                      "all_at_once_residual": residual}
            for name in ("symmetric", "initial"):
                _, history, converged = quasi_newton(
                    advection, diffusion, u0, steps, NKA_DT,
                    variant=variant, scaling=weights, iterations=iterations,
                    tolerance=NKA_TOLERANCE, start=name, reference=reference,
                )
                record[name] = {
                    "iterations": len(history) - 1,
                    "converged": converged,
                    "errors": history,
                }
                if name == start:
                    curves.append(
                        {
                            "history": history,
                            "style": ":",
                            "colour": colour,
                            "marker": marker,
                            "label": rf"({'average' if weights is None else 'NKA'})"
                                     rf" $T={horizon:g}$",
                        }
                    )
            if weights is not None:
                record["weights"] = [float(weights.min()), float(weights.max())]
            record["published"] = PUBLISHED_FIGURE_3_14[(nu, variant, horizon)]
            record["against_published"] = _against_published(
                record[start]["errors"], record["published"]
            )
            records.append(record)
        _error_history_panel(ax, curves, rf"({label}) $\nu={nu:g}$", 1e-4)
        panels.append({"panel": label, "nu": nu, "start": start, "runs": records})

    scan = []
    if not quick:
        for nu in NKA_NU:
            for horizon in horizons:
                scan.append(
                    {
                        "nu": nu,
                        "horizon": horizon,
                        "dx": NKA_SCAN_DX,
                        "candidates": nka_weight_scan(
                            nu, NKA_SCAN_DX, NKA_DT, int(round(horizon / NKA_DT))
                        ),
                    }
                )

    artifacts = save_figure(fig, f"{output_dir}/figure_3_14_paradiag_nka")
    return finish(
        output_dir,
        "figure_3_14_paradiag_nka",
        "Error of the two quasi-Newton versions of ParaDiag I with the BVM "
        "discretization for Burgers' equation with two diffusion parameters",
        {
            "parameters": {
                "dx": dx,
                "coarse_dx": NKA_COARSE_DX,
                "dt": NKA_DT,
                "tolerance": NKA_TOLERANCE,
                "max_iterations": iterations,
                "seed": RANDOM_SEED,
                "starts": {str(key): value for key, value in NKA_STARTS.items()},
            },
            "panels": panels,
            "weight_scan": scan,
        },
        artifacts,
    )


def _against_published(errors, published):
    """Reproduced over published, iteration by iteration, at the drawn vertices."""
    ratios = {
        k: errors[k] / value
        for k, value in published.items()
        if k < len(errors) and errors[k] > 0.0
    }
    if not ratios:
        return None
    return {
        "ratio_to_published": ratios,
        "worst_ratio": max(max(ratios.values()), 1.0 / min(ratios.values())),
    }


def run_table_3_2(output_dir, quick=False):
    """Table 3.2: Jacobian solves for the sequential and the parallel method."""
    horizons = BURGERS_HORIZONS[:2] if quick else BURGERS_HORIZONS
    steps = 40 if quick else BURGERS_STEPS
    rows = []
    for nu in BURGERS_NU:
        advection, diffusion, x = burgers_operators(nu, BURGERS_DX)
        u0 = initial_condition(x)
        sequential = []
        parallel = []
        from_initial = []
        for horizon in horizons:
            dt = horizon / steps
            _, evaluations = trapezoidal_steps(advection, diffusion, u0, steps, dt)
            sequential.append(int(evaluations.sum()))
            for start, sink in (("random", parallel), ("initial", from_initial)):
                _, history, converged = quasi_newton(
                    advection, diffusion, u0, steps, dt,
                    iterations=MAX_ITERATIONS, start=start,
                )
                sink.append(len(history) if converged else None)
        rows.append(
            {
                "nu": nu,
                "horizons": list(horizons),
                "trapezoidal_rule": sequential,
                "paradiag_i": parallel,
                "paradiag_i_from_initial_condition": from_initial,
            }
        )

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for row, colour, marker in zip(rows, ("#D1495B", "#2F6BFF"), ("o", "+")):
        ax.loglog(
            row["horizons"],
            row["trapezoidal_rule"],
            "-.",
            color=colour,
            marker=marker,
            markersize=9,
            markerfacecolor="none",
            linewidth=1.2,
            label=rf"trapezoidal rule, $\nu={row['nu']:g}$",
        )
        finite = [
            (horizon, value)
            for horizon, value in zip(row["horizons"], row["paradiag_i"])
            if value is not None
        ]
        if finite:
            ax.loglog(
                [item[0] for item in finite],
                [item[1] for item in finite],
                "--",
                color=colour,
                marker=marker,
                markersize=9,
                linewidth=1.2,
                label=rf"ParaDiag I, $\nu={row['nu']:g}$",
            )
    ax.set_xlabel(r"$T$")
    ax.set_ylabel("total Jacobian solves")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{horizon:g}" for horizon in horizons])
    ax.set_title("sequential Newton work against parallel ParaDiag work")
    style_axis(ax)
    ax.legend(fontsize=8)
    artifacts = save_figure(fig, f"{output_dir}/table_3_2_jacobian_solves")

    return finish(
        output_dir,
        "table_3_2_jacobian_solves",
        "Number of total Jacobian solves for the sequential trapezoidal rule and "
        "for ParaDiag I with the BVM discretization in parallel; a missing entry "
        "means the iteration diverged",
        {"parameters": {"dx": BURGERS_DX, "steps": steps}, "rows": rows},
        artifacts,
    )


def wave_bvm_error(a, u0, v0, steps, t_final):
    """Error and Cond(V) of ParaDiag I with the BVM discretization (3.41).

    The second-order system is kept in terms of u alone, so the time matrix is
    the square of (3.39b) and the right-hand side is the one derived just above
    (3.41): the initial position enters the second block, the initial velocity
    the first.
    """
    dt = t_final / steps
    matrix = bvm_matrix(steps, dt)
    rhs = np.zeros((steps, u0.size))
    rhs[0] = v0 / (2.0 * dt)
    rhs[1] = -u0 / (4.0 * dt**2)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    transformed = np.linalg.solve(eigenvectors, rhs.astype(complex))
    identity = np.eye(u0.size)
    for n, value in enumerate(eigenvalues):
        transformed[n] = np.linalg.solve(value**2 * identity - a, transformed[n])
    solution = (eigenvectors @ transformed).real
    reference = exponential_reference(
        np.block([[np.zeros_like(a), -identity], [-a, np.zeros_like(a)]]),
        np.concatenate([u0, v0]),
        dt * np.arange(1, steps + 1),
    )[:, : u0.size]
    error = float(np.abs(reference - solution).max())
    return error, float(np.linalg.cond(eigenvectors))


def run_figure_3_12(output_dir, quick=False):
    """Figure 3.12: the geometric mesh against the BVM discretization."""
    step_counts = FIGURE_3_12_STEPS[:3] if quick else FIGURE_3_12_STEPS
    problem = wave_operator(FIGURE_3_12_DX)
    a, x, w0 = problem
    laplacian = -a[x.size :, : x.size]
    u0 = w0[: x.size]
    v0 = w0[x.size :]

    rows = []
    for steps in step_counts:
        rho = FIGURE_3_12_RHO[steps]
        geometric, condition = error_at(
            problem, steps, rho, FIGURE_3_12_T, condition=True
        )
        bvm, bvm_condition = wave_bvm_error(
            laplacian, u0, v0, steps, FIGURE_3_12_T
        )
        rows.append(
            {
                "steps": steps,
                "rho": rho,
                "geometric_error": geometric,
                "geometric_condition": condition,
                "bvm_error": bvm,
                "bvm_condition": bvm_condition,
                "trapezoidal_error": uniform_error(
                    a, w0, steps, FIGURE_3_12_T,
                    trapezoidal=True, components=x.size,
                ),
            }
        )

    counts = np.array([row["steps"] for row in rows], dtype=float)
    # The published error panel stops one step count short of the condition panel.
    error_rows = [row for row in rows if row["steps"] in FIGURE_3_12_ERROR_STEPS]
    error_counts = np.array([row["steps"] for row in error_rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    series = (
        ("geometric_error", "#D1495B", "+", "-.", r"ParaDiag I with $\{\Delta t_n\}$"),
        ("bvm_error", "#2F6BFF", "o", "--", r"ParaDiag I with $\Delta t$"),
        ("trapezoidal_error", "#17324D", "s", "-", r"Trapezoidal Rule with $\Delta t$"),
    )
    for key, colour, marker, style, label in series:
        axes[0].loglog(
            error_counts,
            [row[key] for row in error_rows],
            style,
            color=colour,
            marker=marker,
            markersize=7,
            markerfacecolor="none",
            linewidth=1.1,
            label=label,
        )
    usable = [
        row[key] for row in error_rows for key, *_ in series if row[key] <= 1.0
    ]
    axes[0].set_ylim(min(usable) / 3.0, 3e1)
    axes[0].set_title("(a) measured error")
    axes[0].set_xlabel(r"$N_t$")
    axes[0].set_ylabel("measured error")
    for key, colour, marker, style, label in series[:2]:
        axes[1].loglog(
            counts,
            [row[key.replace("_error", "_condition")] for row in rows],
            style,
            color=colour,
            marker=marker,
            markersize=7,
            markerfacecolor="none",
            linewidth=1.1,
            label=label,
        )
    axes[1].set_title("(b) condition number of the eigenvector matrix")
    axes[1].set_xlabel(r"$N_t$")
    axes[1].set_ylabel(r"${\rm Cond}(V)$")
    for ax, limits in zip(axes, (error_counts, counts)):
        ax.set_xlim(limits.min() / 1.3, limits.max() * 1.3)
        style_axis(ax)
        ax.legend(fontsize=8)

    artifacts = save_figure(fig, f"{output_dir}/figure_3_12_paradiag_bvm_wave")
    return finish(
        output_dir,
        "figure_3_12_paradiag_bvm_wave",
        "(a) Measured error for the wave equation using ParaDiag I with the "
        "trapezoidal rule on geometric step sizes and with the BVM discretization "
        "on a uniform step; (b) the condition number of the eigenvector matrix of "
        "the two time-stepping matrices",
        {
            "parameters": {
                "dx": FIGURE_3_12_DX,
                "t_final": FIGURE_3_12_T,
                "error_panel_steps": list(FIGURE_3_12_ERROR_STEPS),
            },
            "rows": rows,
        },
        artifacts,
    )
