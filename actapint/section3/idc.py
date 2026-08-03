"""Time-parallel integral deferred correction.

Port of `IDC_PIDC_ADE.m`, which produces Figure 3.5.  The script implements IDC
and its pipelined version PIDC in a single routine, switching on a flag: the two
differ only in which initial value each sweep of a window starts from.  IDC waits
for the fully corrected value of the previous window, while PIDC starts sweep k
from the sweep-k value of the previous window, so all sweeps run concurrently.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import splu

import matplotlib.pyplot as plt

from ..plotting import save_figure, style_axis
from ..section2 import source_term
from .common import ade_generator, finish

# Parameters of `IDC_PIDC_ADE.m`.
DX = 1.0 / 64
T_FINAL = 3.0
WINDOW = 1.0 / 10
NODES = 5
SWEEPS = 3
THETA = 1.0  # theta = 1 is backward Euler, theta = 1/2 the trapezoidal rule

# Lagrange quadrature weights of `LagrangeWeights.m`, one row per subinterval.
LAGRANGE_WEIGHTS = {
    2: np.array([[1 / 2, 1 / 2]]),
    3: np.array([[5 / 24, 1 / 3, -1 / 24], [-1 / 24, 1 / 3, 5 / 24]]),
    4: np.array(
        [
            [9 / 72, 19 / 72, -5 / 72, 1 / 72],
            [-1 / 72, 13 / 72, 13 / 72, -1 / 72],
            [1 / 72, -5 / 72, 19 / 72, 9 / 72],
        ]
    ),
    5: np.array(
        [
            [251 / 2880, 323 / 1440, -11 / 120, 53 / 1440, -19 / 2880],
            [-19 / 2880, 173 / 1440, 19 / 120, -37 / 1440, 11 / 2880],
            [11 / 2880, -37 / 1440, 19 / 120, 173 / 1440, -19 / 2880],
            [-19 / 2880, 53 / 1440, -11 / 120, 323 / 1440, 251 / 2880],
        ]
    ),
    6: np.array(
        [
            [19 / 288, 1427 / 7200, -399 / 3600, 241 / 3600, -173 / 7200, 3 / 800],
            [-3 / 800, 637 / 7200, 511 / 3600, -129 / 3600, 77 / 7200, -11 / 7200],
            [11 / 7200, -31 / 2400, 401 / 3600, 401 / 3600, -31 / 2400, 11 / 7200],
            [-11 / 7200, 77 / 7200, -129 / 3600, 511 / 3600, 637 / 7200, -3 / 800],
            [3 / 800, -173 / 7200, 241 / 3600, -399 / 3600, 1427 / 7200, 19 / 288],
        ]
    ),
}

# The four panels of Figure 3.5.
FIGURE_3_5_PANELS = (
    ("a", 1.0, 1000.0),
    ("b", 1e-3, 1000.0),
    ("c", 1.0, 5.0),
    ("d", 1e-3, 5.0),
)

# The k_max = 0 curve of Figure 3.5, read off page 26 of the article at 600 dpi
# with the seven decades of the frame running from 1 down to 1e-7.  That curve is
# the error of the constant predictor, so it depends on the reference solution
# alone and on no part of the method, which makes it the sharpest available check
# on the parameters of each panel.  Windows whose marker is hidden under another
# curve are left out.  IDC and PIDC share it, and RIDC does too, so Figure 3.6
# repeats these values.
PUBLISHED_INITIAL_ERROR = {
    "a": {1: 0.1273, 2: 0.1529, 3: 0.0905, 4: 0.085, 5: 0.085, 13: 0.1745,
          30: 0.3392},
    "b": {1: 0.4301, 2: 0.679, 3: 0.6355, 4: 0.5725, 5: 0.5273, 6: 0.491,
          7: 0.6794, 8: 0.6351, 9: 0.573, 10: 0.5274, 11: 0.4893, 12: 0.4566,
          14: 0.6785, 15: 0.6781, 16: 0.5984, 17: 0.5473, 18: 0.5103,
          19: 0.6807, 20: 0.6798, 21: 0.6006, 22: 0.5504, 23: 0.5115},
    "c": {1: 0.0488, 2: 0.0976, 3: 0.1487, 4: 0.2009, 5: 0.2528, 6: 0.3021,
          7: 0.3482, 8: 0.3902, 9: 0.429, 10: 0.4662, 11: 0.5045, 12: 0.5451,
          13: 0.5906, 14: 0.6392, 15: 0.6905, 16: 0.7429, 17: 0.7896,
          18: 0.7873, 19: 0.8171, 20: 0.8436, 21: 0.8686, 22: 0.8891,
          23: 0.8954, 24: 0.9, 25: 0.9007, 26: 0.9021, 27: 0.9007, 28: 0.9042,
          30: 0.8934},
    "d": {1: 0.0602, 2: 0.1254, 3: 0.1911, 4: 0.2545, 5: 0.3119, 6: 0.3607,
          7: 0.3999, 8: 0.4296, 9: 0.4501, 10: 0.4673, 11: 0.5, 12: 0.5548,
          13: 0.6157, 14: 0.6783, 15: 0.7404, 16: 0.776, 17: 0.7873,
          18: 0.8128, 19: 0.8314, 20: 0.8487, 21: 0.8605, 22: 0.8781,
          23: 0.8931, 24: 0.8992, 25: 0.8965, 26: 0.9025, 27: 0.9032,
          28: 0.9033},
}

# Factor between the reproduced curves and the published ones.  err^k_n divides
# by the largest value the reference solution takes anywhere, so the k_max = 0
# curve has to touch one in the window where that value is attained.  It does in
# the two smooth panels, and in the reproduction of all four, but the two
# low-regularity panels top out at 0.41 and 0.68, so they are drawn against a
# constant this many times the maximum of the solution they plot.
PUBLISHED_OFFSET = {"a": 2.447, "b": 1.469, "c": 0.983, "d": 0.984}


def ade_setup(nu, dx=DX):
    """Operator and grid of (3.12).

    `IDC_PIDC_ADE.m` puts unknowns on ``x = 0:dx:1``, so both endpoints carry an
    unknown and the periodic wrap-around closes a ring of 65 nodes rather than 64.
    """
    x = np.arange(0.0, 1.0 + dx / 2, dx)
    return ade_generator(x.size, dx, nu), x


def node_times(t_final=T_FINAL, window=WINDOW, nodes=NODES):
    """All quadrature nodes of all windows, and the windows count."""
    windows = int(round(t_final / window))
    return np.linspace(0.0, t_final, windows * (nodes - 1) + 1), windows


def reference_solution(a, x, t, sharpness, tolerance=1e-13, method="LSODA"):
    """Reference solution of (2.1), the role of ODE45 in the script.

    The script asks ODE45 for thirteen digits.  The diffusion term makes (2.1)
    stiff for nu = 1, so an implicit-capable integrator is used here instead;
    ``method`` is exposed so that the agreement can be checked against an
    explicit one.
    """
    jacobian = a.toarray()
    solution = solve_ivp(
        lambda time, u: a.dot(u) + source_term(x, time, sharpness),
        (t[0], t[-1]),
        np.zeros(x.size),
        t_eval=t,
        method=method,
        jac=(lambda time, u: jacobian) if method in {"LSODA", "Radau", "BDF"} else None,
        rtol=tolerance,
        atol=tolerance,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y


def idc_sweeps(
    a,
    x,
    t,
    windows,
    sharpness,
    sweeps,
    *,
    pipeline,
    nodes=NODES,
    theta=THETA,
    u0=None,
):
    """Run IDC or PIDC and return the solution after ``sweeps - 1`` corrections.

    Sweep index 0 is the initial approximation, which the script takes constant on
    each window, equal to that window's initial value.  Each later sweep applies
    (3.11) from left to right across the nodes.
    """
    nx = x.size
    dt = t[1] - t[0]
    weights = LAGRANGE_WEIGHTS[nodes]
    identity = sparse.identity(nx, format="csc")
    step = splu((identity - theta * dt * a).tocsc())
    u0 = np.zeros(nx) if u0 is None else u0

    def rhs(time, state):
        return a.dot(state) + source_term(x, time, sharpness)

    # Initial value of each sweep on the next window: one column per sweep for
    # PIDC, a single shared column for IDC.
    carried = np.tile(u0[:, None], (1, sweeps if pipeline else 1))
    u = np.zeros((sweeps, nodes, nx))
    out = np.zeros((nx, windows * (nodes - 1) + 1))
    out[:, 0] = u0

    for window in range(windows):
        base = window * (nodes - 1)
        u[0, :, :] = carried[:, 0]
        for k in range(1, sweeps):
            u[k, 0] = carried[:, k] if pipeline else carried[:, 0]
            for m in range(nodes - 1):
                integral = sum(
                    (nodes - 1) * dt * weights[m, mm] * rhs(t[base + mm], u[k - 1, mm])
                    for mm in range(nodes)
                )
                rest = (
                    u[k, m]
                    - theta * dt * a.dot(u[k - 1, m + 1])
                    + (1.0 - theta) * dt * a.dot(u[k, m] - u[k - 1, m])
                    + integral
                )
                u[k, m + 1] = step.solve(rest)
        if pipeline:
            carried = u[:, -1, :].T.copy()
        else:
            carried[:, 0] = u[-1, -1]
        out[:, base + 1 : base + nodes] = u[-1, 1:, :].T
    return out


def ridc_sweeps(
    a,
    x,
    t,
    sharpness,
    sweeps,
    *,
    nodes=NODES,
    theta=THETA,
    u0=None,
    startup="corrected",
):
    """Run RIDC and return the solution after ``sweeps - 1`` corrections.

    Section 3.3.2 describes RIDC as IDC with a sliding quadrature interval: there
    are no windows, and the correction for step j uses the M previous values of
    the coarser level, so that the interval to be integrated is always the last
    subinterval of the stencil.  Only the first M - 1 steps, where the stencil
    cannot slide any further left, need a different rule, and the choice matters
    because RIDC never restarts: whatever error the start commits stays.  With
    ``startup="corrected"`` those steps reuse the earlier rows of the weight
    matrix, as IDC does inside its first window.  With ``startup="inherited"``
    they are copied from the coarser level, which is what a processor that cannot
    begin before M values are available would hold.
    """
    nx = x.size
    dt = t[1] - t[0]
    weights = LAGRANGE_WEIGHTS[nodes]
    step = splu((sparse.identity(nx, format="csc") - theta * dt * a).tocsc())
    u0 = np.zeros(nx) if u0 is None else u0

    def rhs(time, state):
        return a.dot(state) + source_term(x, time, sharpness)

    previous = np.tile(u0, (t.size, 1))
    inherited = startup == "inherited"
    for _ in range(1, sweeps):
        current = previous.copy() if inherited else np.zeros_like(previous)
        current[0] = u0
        for j in range(nodes - 2 if inherited else 0, t.size - 1):
            if j < nodes - 2:
                start, row = 0, j
            else:
                start, row = j - (nodes - 2), nodes - 2
            integral = sum(
                (nodes - 1) * dt * weights[row, mm] * rhs(t[start + mm], previous[start + mm])
                for mm in range(nodes)
            )
            rest = (
                current[j]
                - theta * dt * a.dot(previous[j + 1])
                + (1.0 - theta) * dt * a.dot(current[j] - previous[j])
                + integral
            )
            current[j + 1] = step.solve(rest)
        previous = current
    return previous.T


def window_errors(approx, reference, windows, nodes=NODES):
    """err^k_n of Section 3.3: the window's max error, scaled by the global max."""
    scale = np.abs(reference).max()
    errors = np.empty(windows)
    for window in range(windows):
        block = slice(window * (nodes - 1), window * (nodes - 1) + nodes)
        errors[window] = np.abs(reference[:, block] - approx[:, block]).max() / scale
    return errors


def published_offset(panel, initial_errors):
    """The reproduced k_max = 0 curve against the digitized published one."""
    published = PUBLISHED_INITIAL_ERROR[panel]
    ratios = np.array(
        [initial_errors[window - 1] / value for window, value in published.items()]
    )
    median = float(np.median(ratios))
    return {
        "windows_read": len(published),
        "median": median,
        "min": float(ratios.min()),
        "max": float(ratios.max()),
        "worst_deviation_from_median": float(np.abs(ratios / median - 1).max()),
    }


def error_table(nu, sharpness, variants, sweeps=SWEEPS, dx=DX, t_final=T_FINAL, window=WINDOW):
    """Per-window errors of the requested method variants, sweep by sweep.

    ``variants`` maps a label to ``"idc"``, ``"pidc"`` or ``"ridc"``.  The script
    reruns the whole method for every value of k_max rather than recording the
    intermediate sweeps of one run, and that is followed here.
    """
    a, x = ade_setup(nu, dx)
    t, windows = node_times(t_final, window)
    reference = reference_solution(a, x, t, sharpness)

    def run(kind, k):
        if kind == "ridc":
            return ridc_sweeps(a, x, t, sharpness, k + 1)
        return idc_sweeps(
            a, x, t, windows, sharpness, k + 1, pipeline=(kind == "pidc")
        )

    table = {
        label: np.array(
            [window_errors(run(kind, k), reference, windows) for k in range(sweeps)]
        )
        for label, kind in variants.items()
    }
    return table, windows, float(np.abs(reference).max())


def _run_error_figure(output_dir, name, variants, caption, quick):
    """Draw one of the two four-panel error figures of Section 3.3."""
    sweeps = 2 if quick else SWEEPS
    window = 1.0 / 3 if quick else WINDOW
    panels = FIGURE_3_5_PANELS[:2] if quick else FIGURE_3_5_PANELS

    rows = 1 if quick else 2
    columns = len(panels) // rows
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.6 * columns, 4.4 * rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()

    styles = {"IDC": ("#1F3FBF", "+"), "PIDC": ("#D1495B", "o"), "RIDC": ("#D1495B", "o")}
    dashes = ["-.", "-", ":"]
    metrics = []
    for ax, (label, nu, sharpness) in zip(axes, panels):
        table, windows, scale = error_table(
            nu, sharpness, variants, sweeps=sweeps, t_final=T_FINAL, window=window
        )
        index = np.arange(1, windows + 1)
        for tag, curves in table.items():
            colour, marker = styles[tag]
            for k, errors in enumerate(curves):
                suffix = " (initial)" if k == 0 else ""
                ax.semilogy(
                    index,
                    errors,
                    dashes[k % len(dashes)],
                    color=colour,
                    marker=marker,
                    markersize=6,
                    markerfacecolor="none",
                    linewidth=1.0,
                    label=rf"{tag} $k_{{\max}}={k}${suffix}",
                )
        ax.set_title(rf"({label}) $\nu={nu:g}$, $\sigma={sharpness:g}$")
        ax.set_xlabel("window index")
        ax.set_ylabel("max error on each window")
        ax.set_xlim(1, windows)
        ax.set_xticks(index[::2])
        ax.set_ylim(1e-7, 1e0)
        style_axis(ax)
        if label == "a":
            ax.legend(fontsize=8, loc="lower left", ncol=2)
        initial = next(iter(table.values()))[0]
        metrics.append(
            {
                "panel": label,
                "nu": nu,
                "sharpness": sharpness,
                "reference_scale": scale,
                "published_initial_error": PUBLISHED_INITIAL_ERROR[label],
                "initial_error_against_published": (
                    published_offset(label, initial) if windows == 30 else None
                ),
                **{tag.lower(): curves.tolist() for tag, curves in table.items()},
                "final_window": {
                    tag.lower(): [float(curve[-1]) for curve in curves]
                    for tag, curves in table.items()
                },
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/{name}")

    return finish(
        output_dir,
        name,
        caption,
        {
            "parameters": {
                "dx": DX,
                "t_final": T_FINAL,
                "window": window,
                "nodes": NODES,
                "sweeps": sweeps,
                "time_integrator": "backward Euler (theta = 1)",
                "space_unknowns": ade_setup(1.0)[1].size,
            },
            "panels": metrics,
        },
        artifacts,
    )


def run_figure_3_5(output_dir, quick=False):
    """Figure 3.5: PIDC needs a regular solution to match IDC."""
    return _run_error_figure(
        output_dir,
        "figure_3_5_idc_pidc",
        {"IDC": "idc", "PIDC": "pidc"},
        "Maximum relative error on each time window for IDC and its pipelined "
        "version PIDC, for a low-regularity source (a, b) and a smooth one (c, d), "
        "with large (a, c) and small (b, d) diffusion",
        quick,
    )


def run_figure_3_6(output_dir, quick=False):
    """Figure 3.6: RIDC, whose sliding interval also needs a regular solution."""
    return _run_error_figure(
        output_dir,
        "figure_3_6_idc_ridc",
        {"IDC": "idc", "RIDC": "ridc"},
        "Maximum relative error on each time window for IDC and its parallel "
        "version RIDC, for a low-regularity source (a, b) and a smooth one (c, d), "
        "with large (a, c) and small (b, d) diffusion",
        quick,
    )
