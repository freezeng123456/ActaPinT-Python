"""The six schematics of the article: Figures 1.1, 3.2, 3.4, 3.7, 4.1 and 4.7.

These carry no measurement, so there is nothing to compare against a published
number.  What can still be reproduced is their content, and three of them have
content that is computed rather than drawn:

* Figure 3.2 shows where red-black SWR already has the exact solution after each
  half-sweep.  That region is not drawn here by hand but propagated: on a grid
  with ``dt = dx`` the domain of dependence of the wave equation is the pair of
  neighbours one row below, so a point is exact exactly when both of them are,
  with the physical boundaries always exact and an artificial interface exact
  only where a previous solve has already made it so.  The tents of panel (a) and
  the rhomboids the caption of panel (c) mentions come out of that recursion.
* Figure 3.4 shows IDC sweeps on a pipeline of time windows.  The curves are real
  sweeps of (3.11) on a scalar problem, so the sweep histories converge to the
  exact solution the way the published ones do.
* Figure 3.7 shows the two ParaExp steps.  The red curves are the solutions
  ``v_n`` of (3.13) with a rough source and zero initial value, and the blue
  dashed ones are the homogeneous propagations ``w_n`` of (3.14), both computed.

Figures 1.1, 4.1 and 4.7 are pure geometry.  Figure 1.1 is the one whose geometry
carries information — the shape of a forward Euler trajectory — so its thirteen
values are read off the vector art of the page: they sit on a uniform grid, at
``FORWARD_EULER`` units above the time axis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .plotting import COLORS, save_figure
from .section3.common import finish

# Figure 1.1.  The thirteen values, in units of the grid the page draws them on:
# the markers sit 16.98 pt apart horizontally and their heights above the time
# axis are integer multiples of 1.869 pt.
FORWARD_EULER = (10, 12, 15, 17, 17, 15, 14, 13, 14, 15, 16, 17, 16)
FORWARD_EULER_ASPECT = 1.869 / 16.98
FORWARD_EULER_TOP = 29.55
FORWARD_EULER_RIGHT = 12.95

# Figure 3.2.  Seven interfaces x_0 .. x_6 one unit apart, so that a wave of
# speed one needs one unit of time to cross half a subdomain and the tents of the
# red solve close exactly at T_1.
INTERFACES = tuple(range(7))
RED_SUBDOMAINS = ((0, 2), (2, 4), (4, 6))
BLACK_SUBDOMAINS = ((1, 3), (3, 5))
WAVE_SPEED = 1.0
SWR_LEVELS = 4
SWR_RESOLUTION = 24  # grid points per unit, in space and in time alike

# Figure 3.4.  Six windows, four sweeps, the M = 6 nodes per window of the text.
PIDC_WINDOWS = 6
PIDC_SWEEPS = 4
PIDC_NODES = 6
PIDC_HORIZON = 6.0

# Figure 3.7.  Four windows of the ParaExp decomposition, and a source rough
# enough that the particular solutions are visibly not smooth.
PARAEXP_WINDOWS = 4
PARAEXP_GROWTH = 0.45
PARAEXP_HORIZON = 4.0
PARAEXP_INITIAL = 1.0
PARAEXP_SEED = 20240501

# Figure 4.1.  The two grids, with the J = 4 the page draws.
PARAREAL_COARSE_STEPS = 2
PARAREAL_J = 4

# Figure 4.7.  Eight coarse points, and F running over two coarse intervals at a
# time from every point, which is what makes MGRiT with FCF-relaxation an
# overlapping Parareal.
FCF_POINTS = 8
FCF_OVERLAP = 2


def _bare_axis(ax):
    """A schematic has no frame, no ticks and no grid."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _arrow(ax, start, end, **kwargs):
    kwargs.setdefault("arrowprops", {})
    ax.annotate("", xy=end, xytext=start, arrowprops=kwargs.pop("arrowprops"), **kwargs)


def run_figure_1_1(output_dir, quick=False):
    """Figure 1.1: the sequential nature of forward Euler."""
    values = np.array(FORWARD_EULER, dtype=float)
    steps = np.arange(values.size, dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    _bare_axis(ax)
    ax.plot(steps, values, color=COLORS["ink"], linewidth=1.0, marker="o",
            markersize=4.0, markerfacecolor=COLORS["ink"], zorder=3)
    for index, value in enumerate(values):
        offset, align = ((-9, -4), "right") if index == 0 else ((-1, 6), "center")
        ax.annotate(f"$u_{{{index}}}$", (index, value),
                    textcoords="offset points", xytext=offset,
                    ha=align, va="bottom", fontsize=10, color=COLORS["ink"])
        label = "0" if index == 0 else f"$t_{{{index}}}$"
        ax.annotate(label, (index, 0.0), textcoords="offset points",
                    xytext=(0, -14), ha="center", va="top", fontsize=10,
                    color=COLORS["ink"])
    arrow = {"arrowstyle": "-|>", "color": COLORS["ink"], "linewidth": 1.1,
             "mutation_scale": 11}
    _arrow(ax, (0.0, 0.0), (FORWARD_EULER_RIGHT, 0.0), arrowprops=arrow)
    _arrow(ax, (0.0, 0.0), (0.0, FORWARD_EULER_TOP), arrowprops=arrow)
    ax.annotate("$t$", (FORWARD_EULER_RIGHT, 0.0), textcoords="offset points",
                xytext=(8, -3), fontsize=11, color=COLORS["ink"])
    ax.annotate("$u$", (0.0, FORWARD_EULER_TOP), textcoords="offset points",
                xytext=(-3, 4), ha="center", fontsize=11, color=COLORS["ink"])
    ax.set_xlim(-0.7, FORWARD_EULER_RIGHT + 0.9)
    ax.set_ylim(-6.0, FORWARD_EULER_TOP + 3.0)
    ax.set_aspect(FORWARD_EULER_ASPECT)
    ax.set_title("Sequential nature of time integration using forward Euler",
                 fontsize=11)
    fig.tight_layout()

    artifacts = save_figure(fig, f"{output_dir}/figure_1_1_forward_euler.png")
    return finish(
        output_dir,
        "figure_1_1_forward_euler",
        "Sequential nature of time integration using forward Euler",
        {
            "parameters": {
                "steps": int(values.size - 1),
                "source": "the marker heights of the published page, in units of "
                          "the grid they sit on",
            },
            "values": [float(v) for v in values],
        },
        artifacts,
    )


def swr_exact_region(levels=SWR_LEVELS, resolution=SWR_RESOLUTION, speed=WAVE_SPEED):
    """Where red-black SWR has the exact solution after each half-sweep.

    The grid has ``dt = dx / speed``, so the domain of dependence of a point is
    its two neighbours one row below and the recursion is exact rather than a
    drawing: a point of a subdomain is exact when both neighbours below it are,
    an artificial interface is exact only where an earlier solve has made it so,
    and the physical boundaries are exact throughout.
    """
    x = np.linspace(INTERFACES[0], INTERFACES[-1],
                    (INTERFACES[-1] - INTERFACES[0]) * resolution + 1)
    t = np.linspace(0.0, levels, levels * resolution + 1)
    exact = np.zeros((t.size, x.size), dtype=bool)
    exact[0] = True

    schedule, frames = [], []
    for step in range(1, levels + 1):
        subdomains = RED_SUBDOMAINS if step % 2 else BLACK_SUBDOMAINS
        schedule.append((step, "red" if step % 2 else "black", subdomains, step))
        before = exact.copy()
        gained = np.zeros_like(exact)
        rows = int(round(step * resolution))
        for a, b in subdomains:
            left, right = int(round(a * resolution)), int(round(b * resolution))
            local = np.zeros((rows + 1, right - left + 1), dtype=bool)
            local[0] = True  # the initial condition of the whole solve
            for row in range(1, rows + 1):
                interior = local[row - 1, :-2] & local[row - 1, 2:]
                local[row, 1:-1] = interior
                # An interface is exact where a previous solve already is, and a
                # physical boundary is exact always.
                local[row, 0] = a == INTERFACES[0] or before[row, left]
                local[row, -1] = b == INTERFACES[-1] or before[row, right]
            gained[: rows + 1, left : right + 1] |= local
        exact = before | gained
        frames.append({"before": before, "after": exact.copy(),
                       "gained": exact & ~before})
    return x, t, frames, schedule


def _swr_panel(ax, x, t, frame, subdomains, band, colour, label, title):
    grid_x, grid_t = np.meshgrid(x, t)
    blue = COLORS["blue"]

    # Hatched: where the solution is already exact when this half-sweep starts,
    # which is what the published panel of the same index shows.
    if frame["before"].any():
        filled = ax.contourf(grid_x, grid_t, frame["before"].astype(float),
                             levels=[0.5, 1.5], colors=[blue], alpha=0.13,
                             hatches=["///"], zorder=1)
        filled.set_edgecolor(blue)
        filled.set_linewidth(0.0)
        ax.contour(grid_x, grid_t, frame["before"].astype(float), levels=[0.5],
                   colors=[blue], linewidths=0.8, zorder=2)

    # The space-time subdomains this half-sweep solves, over the band the page
    # draws: what lies below it this family has solved already.
    left = min(a for a, _ in subdomains)
    right = max(b for _, b in subdomains)
    for a, b in subdomains:
        ax.add_patch(Rectangle((a, band[0]), b - a, band[1] - band[0],
                               facecolor=colour, alpha=0.20, edgecolor=colour,
                               linewidth=1.5, zorder=3))
    # The characteristics inside those solves: they close the tents in which the
    # solve is already exact and bound the redundant computations around them.
    height = band[1] - band[0]
    for interface in sorted({end for pair in subdomains for end in pair}):
        for direction in (-1.0, 1.0):
            reach = min(height, (interface - left) if direction < 0 else (right - interface))
            if reach <= 0:
                continue
            ax.plot([interface, interface + direction * reach],
                    [band[0], band[0] + reach], color=COLORS["ink"],
                    linewidth=0.6, alpha=0.75, zorder=4)

    ax.plot([INTERFACES[0], INTERFACES[-1]], [0.0, 0.0], color=COLORS["ink"],
            linewidth=1.0, zorder=4)
    for interface in INTERFACES:
        ax.plot([interface], [0.0], marker="o", markersize=3.4,
                color=COLORS["ink"], zorder=5)
        ax.annotate(f"$x_{{{interface}}}$", (interface, 0.0),
                    textcoords="offset points", xytext=(3, 4), fontsize=8,
                    color=COLORS["ink"], zorder=6)
        style = "-" if interface in (INTERFACES[0], INTERFACES[-1]) else "--"
        ax.plot([interface, interface], [0.0, SWR_LEVELS + 0.35], linestyle=style,
                color=COLORS["ink"] if style == "-" else COLORS["slate"],
                linewidth=0.9 if style == "-" else 0.6, alpha=1.0 if style == "-" else 0.6,
                zorder=1)
    for level in range(1, SWR_LEVELS + 1):
        ax.plot([-0.16, 0.0], [level, level], color=COLORS["ink"], linewidth=0.8)
        ax.annotate(f"$T_{{{level}}}$", (-0.22, level), ha="right", va="center",
                    fontsize=8, color=COLORS["ink"])

    # The two families of subdomains: the black ones labelled above their span
    # and the red ones below, as the page has them.
    for family, colour_name, y in ((BLACK_SUBDOMAINS, "ink", -0.85),
                                   (RED_SUBDOMAINS, "rose", -1.75)):
        for a, b in family:
            ax.annotate("", xy=(b, y), xytext=(a, y),
                        arrowprops={"arrowstyle": "<|-|>", "color": COLORS[colour_name],
                                    "linewidth": 0.8, "mutation_scale": 7})
            ax.annotate(f"$\\Omega_{{{a + 1}}}$", ((a + b) / 2, y),
                        textcoords="offset points", xytext=(0, -6), ha="center",
                        va="top", fontsize=8.5, color=COLORS[colour_name])

    ax.set_title(title, fontsize=10)
    ax.annotate(label, (0.5 * INTERFACES[-1], -2.85), ha="center", va="top",
                fontsize=9, color=COLORS["slate"])
    ax.set_xlim(-0.95, INTERFACES[-1] + 0.6)
    ax.set_ylim(-3.3, SWR_LEVELS + 0.7)
    _bare_axis(ax)


def run_figure_3_2(output_dir, quick=False):
    """Figure 3.2: red-black SWR with generous overlap."""
    resolution = 8 if quick else SWR_RESOLUTION
    x, t, frames, schedule = swr_exact_region(resolution=resolution)
    bands = ((0.0, 1.0), (1.0, 2.0), (1.0, 3.0), (2.0, 4.0))
    labels = ("(a)", "(b)", "(c)", "(d)")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2))
    for ax, frame, (step, family, subdomains, _), band, label in zip(
            axes.ravel(), frames, schedule, bands, labels):
        colour = COLORS["rose"] if family == "red" else COLORS["slate"]
        _swr_panel(ax, x, t, frame, subdomains, band, colour, label, f"$k = {step}$")
    # The page points out the redundant computations in the first two panels.
    for ax, anchor, targets in ((axes[0, 0], (3.0, 1.75), ((2.0, 0.72), (4.0, 0.72))),
                                (axes[0, 1], (3.0, 2.95),
                                 ((1.45, 1.72), (2.55, 1.72), (3.45, 1.72), (4.55, 1.72)))):
        ax.annotate("regions of redundant computations", anchor, ha="center",
                    va="bottom", fontsize=8, color=COLORS["ink"])
        for target in targets:
            ax.annotate("", xy=target, xytext=(anchor[0], anchor[1] - 0.06),
                        arrowprops={"arrowstyle": "-|>", "color": COLORS["ink"],
                                    "linewidth": 0.6, "mutation_scale": 7})
    fig.suptitle("Red-black SWR with generous overlap: where the solution is "
                 "already exact", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    artifacts = save_figure(fig, f"{output_dir}/figure_3_2_red_black_swr.png")
    cell = (x[1] - x[0]) * (t[1] - t[0])
    return finish(
        output_dir,
        "figure_3_2_red_black_swr",
        "Illustration of red-black SWR with generous overlap",
        {
            "parameters": {
                "interfaces": list(INTERFACES),
                "red_subdomains": [list(pair) for pair in RED_SUBDOMAINS],
                "black_subdomains": [list(pair) for pair in BLACK_SUBDOMAINS],
                "wave_speed": WAVE_SPEED,
                "resolution": resolution,
                "levels": SWR_LEVELS,
            },
            "steps": [
                {
                    "k": step,
                    "family": family,
                    "solved_to": float(horizon),
                    "exact_area": float(frame["after"].sum() * cell),
                    "exact_area_gained": float(frame["gained"].sum() * cell),
                    "exact_up_to": float(t[np.flatnonzero(frame["after"].any(axis=1)).max()]),
                }
                for frame, (step, family, _, horizon) in zip(frames, schedule)
            ],
        },
        artifacts,
    )


def _pidc_sweeps(windows=PIDC_WINDOWS, sweeps=PIDC_SWEEPS, nodes=PIDC_NODES,
                 horizon=PIDC_HORIZON):
    """IDC sweeps of (3.11) on every window of a scalar test problem.

    The problem is ``u' = cos(3 t) - u`` with ``u(0) = 1``, whose solution is
    smooth and oscillating, so that the sweeps of a window are visibly ordered:
    each one lies closer to the exact solution than the previous.
    """
    def rhs(time, value):
        return np.cos(3.0 * time) - value

    def exact(time):
        # u' = cos(3 t) - u with u(0) = 1, integrated in closed form.
        return (0.1 * (np.cos(3.0 * time) + 3.0 * np.sin(3.0 * time))
                + 0.9 * np.exp(-time))

    width = horizon / windows
    per_window = []
    start = exact(0.0)
    for window in range(windows):
        t0 = window * width
        nodes_t = np.linspace(t0, t0 + width, nodes)
        step = np.diff(nodes_t)
        weights = np.zeros((nodes - 1, nodes))
        for row in range(nodes - 1):
            # Lagrange quadrature of (3.11) over one subinterval.
            for column in range(nodes):
                basis = np.poly1d([1.0])
                for other in range(nodes):
                    if other != column:
                        basis *= np.poly1d([1.0, -nodes_t[other]]) / (
                            nodes_t[column] - nodes_t[other])
                antiderivative = basis.integ()
                weights[row, column] = (antiderivative(nodes_t[row + 1])
                                        - antiderivative(nodes_t[row]))
        iterate = np.full(nodes, start)
        history = [iterate.copy()]
        for _ in range(sweeps):
            residual = weights.dot(rhs(nodes_t, iterate))
            nxt = iterate.copy()
            for row in range(nodes - 1):
                # (3.11) with theta = 1, i.e. backward Euler on the correction:
                # u^{k+1}_{m+1} = u^{k+1}_m + dt (f(u^{k+1}_{m+1}) - f(u^k_{m+1}))
                #                 + the quadrature of f(u^k) over the subinterval.
                # Here f(u, t) = cos(3 t) - u, so the implicit term is linear and
                # the update can be solved for in closed form.
                nxt[row + 1] = ((nxt[row] + step[row] * iterate[row + 1]
                                 + residual[row]) / (1.0 + step[row]))
            iterate = nxt
            history.append(iterate.copy())
        per_window.append({"t": nodes_t, "history": history,
                           "exact": exact(nodes_t)})
        start = history[-1][-1]
    return per_window


def _pidc_panel(ax, per_window, active, offset, windows, labels):
    """One panel: which (window, sweep) pairs are running, at their k level."""
    span = 0.62
    started = min((window for window, _ in active), default=0)
    for column, window in enumerate(range(offset, offset + windows)):
        data = per_window[window]
        t0, t1 = data["t"][0], data["t"][-1]
        scale = span / max(1e-12, np.ptp(np.concatenate(
            [np.concatenate(data["history"]), data["exact"]])))
        centre = float(np.mean(data["exact"]))
        for sweep in range(1, PIDC_SWEEPS + 1):
            curve = (data["history"][sweep] - centre) * scale + sweep
            truth = (data["exact"] - centre) * scale + sweep
            if (window, sweep) in active:
                ax.plot(data["t"], truth, color=COLORS["ink"], linewidth=1.0, zorder=3)
                ax.plot(data["t"], curve, color=COLORS["rose"], linewidth=1.2,
                        marker="o", markersize=2.6, zorder=4)
            elif window < started or any(w == window and s > sweep for w, s in active):
                # A sweep history: this window has already gone past this sweep.
                ax.plot(data["t"], curve, color=COLORS["ink"], linewidth=0.8,
                        linestyle=(0, (3, 2)), alpha=0.75, zorder=2)
        ax.plot([t1, t1], [0.35, PIDC_SWEEPS + 0.7], linestyle=(0, (4, 3)),
                color=COLORS["slate"], linewidth=0.7, alpha=0.7, zorder=1)
        if column == 0:
            ax.plot([t0, t0], [0.35, PIDC_SWEEPS + 0.7], linestyle=(0, (4, 3)),
                    color=COLORS["slate"], linewidth=0.7, alpha=0.7, zorder=1)
    for sweep in range(1, PIDC_SWEEPS + 1):
        ax.annotate(f"$k = {sweep}$", (per_window[offset]["t"][0], sweep),
                    textcoords="offset points", xytext=(-8, 0), ha="right",
                    va="center", fontsize=8, color=COLORS["ink"])
    edges = [per_window[w]["t"][0] for w in range(offset, offset + windows)]
    edges.append(per_window[offset + windows - 1]["t"][-1])
    for edge, label in zip(edges, labels):
        ax.annotate(label, (edge, 0.35), textcoords="offset points",
                    xytext=(0, -10), ha="center", va="top", fontsize=8,
                    color=COLORS["ink"])
    ax.set_xlim(edges[0] - 0.13 * (edges[-1] - edges[0]), edges[-1] + 0.04 * (edges[-1] - edges[0]))
    ax.set_ylim(-0.35, PIDC_SWEEPS + 0.9)
    _bare_axis(ax)


def run_figure_3_4(output_dir, quick=False):
    """Figure 3.4: the PIDC pipeline, four sweeps deep."""
    per_window = _pidc_sweeps()
    # The initialization fills the pipeline one window at a time, and then every
    # further stage runs k on window n, k-1 on n+1, and so on.
    stages = []
    for stage in range(1, PIDC_SWEEPS + 1):
        stages.append({"active": {(stage - sweep, sweep)
                                  for sweep in range(1, stage + 1)},
                       "offset": 0, "windows": PIDC_WINDOWS,
                       "labels": [f"$T_{{{i}}}$" for i in range(PIDC_WINDOWS + 1)]})
    pipeline_labels = ["$T_{n-1}$", "$T_n$", "$T_{n+1}$", "$T_{n+2}$",
                       "$T_{n+3}$", "$T_{n+4}$"]
    for shift in range(2):
        stages.append({
            "active": {(shift + PIDC_SWEEPS - sweep, sweep)
                       for sweep in range(1, PIDC_SWEEPS + 1)},
            "offset": 0,
            "windows": len(pipeline_labels) - 1,
            "labels": pipeline_labels,
        })

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 8.4))
    for ax, stage, label in zip(axes.ravel(), stages, "abcdef"):
        _pidc_panel(ax, per_window, stage["active"], stage["offset"],
                    stage["windows"], stage["labels"])
        ax.annotate(f"({label})", (0.5, -0.22), xycoords="axes fraction",
                    ha="center", fontsize=9, color=COLORS["slate"])
    fig.suptitle("PIDC with $k_{\\max} = 4$: the sweeps of four windows run at "
                 "once once the pipeline is full", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    artifacts = save_figure(fig, f"{output_dir}/figure_3_4_pidc_pipeline.png")
    errors = [
        {
            "window": index,
            "errors": [float(np.max(np.abs(sweep - data["exact"])))
                       for sweep in data["history"]],
        }
        for index, data in enumerate(per_window)
    ]
    return finish(
        output_dir,
        "figure_3_4_pidc_pipeline",
        "In PIDC with k_max = 4, the sweeps on the four time windows can run "
        "simultaneously once the initialization phase is completed",
        {
            "parameters": {
                "problem": "u' = cos(3 t) - u, u(0) = 1",
                "windows": PIDC_WINDOWS,
                "sweeps": PIDC_SWEEPS,
                "nodes": PIDC_NODES,
                "horizon": PIDC_HORIZON,
                "integrator": "backward Euler, theta = 1 in (3.11)",
            },
            "stages": [
                {"panel": label,
                 "running": sorted([list(pair) for pair in stage["active"]])}
                for label, stage in zip("abcdef", stages)
            ],
            "sweep_errors": errors,
        },
        artifacts,
    )


def _paraexp_pieces(windows=PARAEXP_WINDOWS, growth=PARAEXP_GROWTH,
                    horizon=PARAEXP_HORIZON, initial=PARAEXP_INITIAL,
                    seed=PARAEXP_SEED):
    """The particular and homogeneous pieces (3.13) and (3.14) of ParaExp.

    A scalar problem is enough for the picture: ``u' = a u + g(t)`` with a rough
    ``g``, so that the particular solutions of the windows are visibly rough and
    the homogeneous ones, which are exponentials, are visibly smooth.
    """
    edges = np.linspace(0.0, horizon, windows + 1)
    rng = np.random.default_rng(seed)
    # A source that is only piecewise constant, so that the particular solutions
    # are visibly rough while the homogeneous ones stay exponentials.
    cells = 240
    levels = rng.uniform(-1.0, 1.0, size=cells)
    breaks = np.linspace(0.0, horizon, cells + 1)

    def source(time):
        index = np.clip(np.searchsorted(breaks, np.atleast_1d(time), side="right") - 1,
                        0, cells - 1)
        return 0.95 + 0.85 * levels[index]

    # One grid for every piece, so that (3.15) holds to roundoff rather than only
    # up to the difference between two discretizations.
    per_window = 400
    grid = np.linspace(0.0, horizon, windows * per_window + 1)
    step = grid[1] - grid[0]
    forcing = source(grid)

    def march(first, last, value, with_source):
        """Trapezoidal rule on u' = a u + g over grid[first:last + 1]."""
        out = np.empty(last - first + 1)
        out[0] = value
        for index in range(1, out.size):
            source_term = (0.5 * step * (forcing[first + index] + forcing[first + index - 1])
                           if with_source else 0.0)
            out[index] = ((1.0 + 0.5 * step * growth) * out[index - 1] + source_term) / (
                1.0 - 0.5 * step * growth)
        return out

    starts = [window * per_window for window in range(windows + 1)]
    particular, homogeneous = [], []
    for window in range(windows):
        first, last = starts[window], starts[window + 1]
        values = march(first, last, 0.0, True)
        particular.append((grid[first : last + 1], values))
        # (3.14): the homogeneous piece carries the initial value of the window on
        # to the horizon, the first one carrying u_0 itself.
        head = initial if window == 0 else particular[window - 1][1][-1]
        homogeneous.append((grid[first:], march(first, grid.size - 1, head, False)))

    # (3.15): the sum of the pieces must be the solution the sequential solver
    # would have produced.
    direct = march(0, grid.size - 1, initial, True)
    identity = 0.0
    for window in range(windows):
        first, last = starts[window], starts[window + 1]
        total = particular[window][1].copy()
        for earlier in range(window + 1):
            offset = first - starts[earlier]
            total += homogeneous[earlier][1][offset : offset + total.size]
        identity = max(identity, float(np.max(np.abs(total - direct[first : last + 1]))))
    return edges, particular, homogeneous, identity


def run_figure_3_7(output_dir, quick=False):
    """Figure 3.7: the two steps of the ParaExp solver."""
    edges, particular, homogeneous, identity = _paraexp_pieces()

    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    _bare_axis(ax)
    for t, values in particular:
        ax.plot(t, values, color=COLORS["rose"], linewidth=1.2, zorder=3)
    for t, values in homogeneous:
        ax.plot(t, values, color=COLORS["blue"], linewidth=1.2,
                linestyle=(0, (5, 3)), zorder=2)
    top = max(values.max() for _, values in homogeneous)
    arrow = {"arrowstyle": "-|>", "color": COLORS["ink"], "linewidth": 1.0,
             "mutation_scale": 10}
    _arrow(ax, (0.0, 0.0), (edges[-1] * 1.06, 0.0), arrowprops=arrow)
    _arrow(ax, (0.0, 0.0), (0.0, top * 1.12), arrowprops=arrow)
    start = homogeneous[0][1][0]
    ax.plot([0.0], [start], marker="o", markersize=3.6, color=COLORS["ink"], zorder=4)
    ax.annotate("$u_0$", (0.0, start), textcoords="offset points", xytext=(-6, 0),
                ha="right", va="center", fontsize=10, color=COLORS["blue"])
    for index, edge in enumerate(edges):
        ax.plot([edge, edge], [0.0, top * 1.02], color=COLORS["slate"],
                linewidth=0.6, alpha=0.45, zorder=1)
        ax.annotate(f"$T_{{{index}}}$", (edge, 0.0), textcoords="offset points",
                    xytext=(0, -13), ha="center", va="top", fontsize=10,
                    color=COLORS["ink"])
    ax.set_xlim(-0.42, edges[-1] * 1.12)
    ax.set_ylim(-top * 0.13, top * 1.2)
    ax.set_title("Two steps in the ParaExp solver: rough particular solutions in "
                 "red, smooth homogeneous ones dashed", fontsize=11)
    fig.tight_layout()

    artifacts = save_figure(fig, f"{output_dir}/figure_3_7_paraexp_steps.png")
    return finish(
        output_dir,
        "figure_3_7_paraexp_steps",
        "Two steps in the ParaExp solver",
        {
            "parameters": {
                "problem": "u' = a u + g(t) with a rough source",
                "growth": PARAEXP_GROWTH,
                "windows": PARAEXP_WINDOWS,
                "horizon": PARAEXP_HORIZON,
                "initial_value": PARAEXP_INITIAL,
                "seed": PARAEXP_SEED,
            },
            "window_edges": [float(edge) for edge in edges],
            "particular_at_window_end": [float(values[-1]) for _, values in particular],
            "homogeneous_at_horizon": [float(values[-1]) for _, values in homogeneous],
            "solution_at_horizon": float(
                particular[-1][1][-1] + sum(values[-1] for _, values in homogeneous)),
            "decomposition_error": identity,
        },
        artifacts,
    )


def run_figure_4_1(output_dir, quick=False):
    """Figure 4.1: the two time grids of Parareal."""
    coarse = np.arange(PARAREAL_COARSE_STEPS + 1, dtype=float)
    fine = np.linspace(0.0, PARAREAL_COARSE_STEPS, PARAREAL_COARSE_STEPS * PARAREAL_J + 1)

    fig, ax = plt.subplots(figsize=(8.0, 2.5))
    _bare_axis(ax)
    ax.plot([-0.32, PARAREAL_COARSE_STEPS + 0.32], [0.0, 0.0],
            color=COLORS["slate"], linewidth=1.0, zorder=1)
    for point in fine:
        ax.plot([point, point], [-0.055, 0.055], color=COLORS["slate"],
                linewidth=0.9, zorder=2)
    labels = ("$T_{n-1}$", "$T_n$", "$T_{n+1}$")
    for point, label in zip(coarse, labels):
        ax.plot([point, point], [-0.105, 0.105], color=COLORS["rose"],
                linewidth=2.6, solid_capstyle="butt", zorder=3)
        ax.annotate(label, (point, 0.16), ha="center", va="bottom", fontsize=11,
                    color=COLORS["rose"])
    for left, right in zip(fine[:-1], fine[1:]):
        ax.annotate("$\\Delta t$", (0.5 * (left + right), 0.055),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    va="bottom", fontsize=8.5, color=COLORS["ink"])
    for left, right in zip(coarse[:-1], coarse[1:]):
        for point in (left, right):
            ax.plot([point, point], [-0.42, -0.09], linestyle=(0, (3, 3)),
                    color=COLORS["slate"], linewidth=0.7, zorder=1)
        ax.annotate("", xy=(right, -0.34), xytext=(left, -0.34),
                    arrowprops={"arrowstyle": "<|-|>", "color": COLORS["rose"],
                                "linewidth": 0.9, "mutation_scale": 8})
        ax.annotate("$\\Delta T$", (0.5 * (left + right), -0.34),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    va="bottom", fontsize=10.5, color=COLORS["rose"])
    ax.set_xlim(-0.5, PARAREAL_COARSE_STEPS + 0.5)
    ax.set_ylim(-0.62, 0.44)
    ax.set_title(f"Parareal uses two time grids: one $\\Delta T$ holds "
                 f"$J = {PARAREAL_J}$ steps $\\Delta t$", fontsize=11)
    fig.tight_layout()

    artifacts = save_figure(fig, f"{output_dir}/figure_4_1_two_grids.png")
    return finish(
        output_dir,
        "figure_4_1_two_grids",
        "Parareal uses two time grids, where each large time-step size Delta T "
        "contains J small time-step sizes Delta t",
        {
            "parameters": {
                "coarse_intervals": PARAREAL_COARSE_STEPS,
                "coarsening_factor": PARAREAL_J,
            },
            "coarse_points": [float(point) for point in coarse],
            "fine_points": [float(point) for point in fine],
        },
        artifacts,
    )


def run_figure_4_7(output_dir, quick=False):
    """Figure 4.7: MGRiT with FCF-relaxation as an overlapping Parareal."""
    points = np.arange(FCF_POINTS, dtype=float)
    spans = [(start, start + FCF_OVERLAP)
             for start in range(FCF_POINTS - FCF_OVERLAP)]

    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    _bare_axis(ax)
    ax.plot([-0.4, FCF_POINTS - 0.6], [0.0, 0.0], color=COLORS["slate"],
            linewidth=1.0, zorder=1)
    for point in points:
        ax.plot([point], [0.0], marker="o", markersize=7.0,
                markerfacecolor=COLORS["paper"], markeredgecolor=COLORS["ink"],
                markeredgewidth=1.4, zorder=3)
        ax.annotate(f"$T_{{{int(point)}}}$", (point, 0.0),
                    textcoords="offset points", xytext=(-9, -4), ha="right",
                    va="top", fontsize=10, color=COLORS["ink"])
    for span in spans:
        above = span[0] % 2 == 0
        y = 0.52 if above else -0.52
        tip = 0.10 if above else -0.10
        for point in span:
            ax.plot([point, point], [tip, y], color=COLORS["slate"],
                    linewidth=0.8, zorder=2)
        ax.plot([span[0], span[1]], [y, y], color=COLORS["slate"], linewidth=0.8,
                zorder=2)
        ax.annotate(f"$\\mathcal{{F}} : T_{{{span[0]}}} \\to T_{{{span[1]}}}$",
                    (0.5 * sum(span), y), textcoords="offset points",
                    xytext=(0, 5 if above else -14), ha="center",
                    va="bottom" if above else "top", fontsize=9.5,
                    color=COLORS["rose"])
    ax.set_xlim(-0.9, FCF_POINTS - 0.1)
    ax.set_ylim(-1.02, 1.02)
    ax.set_title("MGRiT with FCF-relaxation: the fine solves overlap by one "
                 "coarse interval", fontsize=11)
    fig.tight_layout()

    artifacts = save_figure(fig, f"{output_dir}/figure_4_7_fcf_relaxation.png")
    return finish(
        output_dir,
        "figure_4_7_fcf_relaxation",
        "Geometric representation of MGRiT with FCF-relaxation as an overlapping "
        "variant of Parareal",
        {
            "parameters": {
                "coarse_points": FCF_POINTS,
                "fine_solve_span_in_coarse_intervals": FCF_OVERLAP,
                "drawn_points": "every one of the eight is a coarse point; the "
                                "published page draws them all alike and does not "
                                "draw the fine grid",
            },
            "fine_solves": [[int(a), int(b)] for a, b in spans],
        },
        artifacts,
    )


SCHEMATIC_EXPERIMENTS = {
    "figure_1_1": run_figure_1_1,
    "figure_3_2": run_figure_3_2,
    "figure_3_4": run_figure_3_4,
    "figure_3_7": run_figure_3_7,
    "figure_4_1": run_figure_4_1,
    "figure_4_7": run_figure_4_7,
}


def run_schematics(output_dir, quick=False):
    """Draw all six schematics in figure order."""
    return [function(output_dir, quick=quick)
            for function in SCHEMATIC_EXPERIMENTS.values()]


__all__ = ["SCHEMATIC_EXPERIMENTS", "run_schematics", *SCHEMATIC_EXPERIMENTS]
