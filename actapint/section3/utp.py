"""Red-black SWR, equivalently unmapped tent pitching, for the wave equation.

Port of `UTP_DD_Method.m` and `RedBlackSubdomains.m`, which produce Figure 3.3.
The whole space-time problem is assembled at once, and red-black restricted
additive Schwarz is applied to it. Because the wave equation has a finite speed
of propagation, each local solve is already exact inside a triangular tent, so
the iteration advances the exact solution slab by slab instead of converging
asymptotically everywhere.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..colormaps import parula
from ..plotting import save_figure
from .common import finish

# Parameters of `UTP_DD_Method.m`.
WAVE_SPEED = 1.0
SPACE_UNKNOWNS = 400
T_FINAL = 3.0
RED_SUBDOMAINS = 5

# The four panels of Figure 3.3, as the script's titles name them.
FIGURE_3_3_PANELS = (
    ("a", "initial", 0, "Initial error"),
    ("b", "red", 4, "4 red iteration"),
    ("c", "black", 8, "8 black iteration"),
    ("d", "red", 12, "12 red iteration"),
)


def wave_grid(n=SPACE_UNKNOWNS, t_final=T_FINAL, speed=WAVE_SPEED):
    """Interior space nodes and the CFL time step chosen by the script."""
    dx = 1.0 / (n + 1)
    steps = int(np.ceil(t_final / (dx / speed)))
    dt = t_final / steps
    x = np.arange(1, n + 1) * dx
    t = np.arange(1, steps + 1) * dt
    return x, dx, t, dt, steps


def initial_condition(x):
    """u0(x) = sin^2(8 pi (1 - x)^2), the oscillating signal of Figure 2.4(d)."""
    return np.sin(8.0 * np.pi * (1.0 - x) ** 2) ** 2


def space_time_system(n=SPACE_UNKNOWNS, t_final=T_FINAL, speed=WAVE_SPEED):
    """Assemble the all-at-once wave operator A and right-hand side f.

    ``A`` is the explicit three-level scheme written as one space-time matrix, so
    it is block lower triangular with diagonal blocks ``I/(c dt)^2``.  The initial
    displacement and velocity enter ``f`` through the first two time blocks, using
    the Taylor start of the script.
    """
    x, dx, t, dt, steps = wave_grid(n, t_final, speed)
    e = np.ones(n)
    laplacian = sparse.diags((e, -2.0 * e, e), (-1, 0, 1), shape=(n, n)) / dx**2
    et = np.ones(steps)
    second_time = sparse.diags(
        (et, -2.0 * et, et), (-2, -1, 0), shape=(steps, steps)
    ) / (speed**2 * dt**2)
    shift = sparse.diags(et, -1, shape=(steps, steps))
    a = sparse.kron(second_time, sparse.identity(n)) - sparse.kron(shift, laplacian)

    u0 = initial_condition(x)
    u0t = np.zeros(n)
    f = np.zeros(n * steps)
    f[:n] = u0t / dt + u0 / dt**2 + 0.5 * laplacian.dot(u0)
    f[n : 2 * n] = -u0 / dt**2
    return a.tocsr(), f, x, t, dt, steps


def red_black_subdomains(n, steps, nx=RED_SUBDOMAINS):
    """Port of `RedBlackSubdomains.m`: index sets of the space-time tents.

    Red subdomains tile space; black ones are offset by half a subdomain and do
    not reach the outer boundaries.  The time slabs are two tent heights tall,
    with the first red slab only one, which is what staggers the two colours.
    """
    red_interfaces = np.round(n / nx * np.arange(nx + 1)).astype(int)
    red_space = [
        np.arange(red_interfaces[i], red_interfaces[i + 1]) for i in range(nx)
    ]
    black_interfaces = np.round(n / nx * (np.arange(nx) + 0.5)).astype(int)
    black_space = [
        np.arange(black_interfaces[i], black_interfaces[i + 1]) for i in range(nx - 1)
    ]
    width = min(
        [*np.diff(red_interfaces).tolist(), *np.diff(black_interfaces).tolist()]
    )
    tent = min(width // 2, steps)

    def slabs(first):
        out = [np.arange(0, first)]
        end = first
        while end < steps:
            nxt = min(end + 2 * tent, steps)
            out.append(np.arange(end, nxt))
            end = nxt
        return out

    return red_space, black_space, slabs(tent), slabs(min(2 * tent, steps)), tent


class _BoxSolver:
    """Exact solve on one space-time box, cached by its shape.

    Restricting the all-at-once operator to a box keeps it block lower triangular,
    so the local problem is as cheap as time stepping inside the box.
    """

    def __init__(self, a, n):
        self.a = a
        self.n = n
        self._cache = {}

    def indices(self, space, times):
        return (times[:, None] * self.n + space[None, :]).ravel()

    def solve(self, space, times, residual):
        key = (space[0], space.size, times.size)
        index = self.indices(space, times)
        if key not in self._cache:
            block = self.a[index, :][:, index]
            self._cache[key] = splu(block.tocsc())
        return index, self._cache[key].solve(residual[index])


def run_red_black_swr(
    n=SPACE_UNKNOWNS,
    t_final=T_FINAL,
    speed=WAVE_SPEED,
    nx=RED_SUBDOMAINS,
    sweeps=12,
    seed=0,
    snapshots=(),
):
    """Iterate red-black SWR on the space-time wave system.

    ``snapshots`` lists ``(colour, sweep)`` pairs at which to record the error
    field.  Returns those fields together with the grid and the per-sweep error.
    """
    a, f, x, t, dt, steps = space_time_system(n, t_final, speed)
    exact = _forward_solve(a, f, n, steps)
    rng = np.random.default_rng(seed)
    u = rng.random(n * steps) - 0.5

    red_space, black_space, red_slabs, black_slabs, tent = red_black_subdomains(
        n, steps, nx
    )
    solver = _BoxSolver(a, n)
    wanted = set(snapshots)
    fields = {}
    history = []

    def record(colour, sweep):
        if (colour, sweep) in wanted:
            fields[(colour, sweep)] = np.abs(exact - u).reshape(steps, n).T.copy()

    record("initial", 0)
    history.append(float(np.max(np.abs(exact - u))))
    for sweep in range(1, sweeps + 1):
        if sweep <= len(red_slabs):
            residual = f - a.dot(u)
            for space in red_space:
                index, correction = solver.solve(
                    space, red_slabs[sweep - 1], residual
                )
                u[index] += correction
                residual = f - a.dot(u)
        record("red", sweep)
        if sweep <= len(black_slabs):
            residual = f - a.dot(u)
            for space in black_space:
                index, correction = solver.solve(
                    space, black_slabs[sweep - 1], residual
                )
                u[index] += correction
                residual = f - a.dot(u)
        record("black", sweep)
        history.append(float(np.max(np.abs(exact - u))))

    return {
        "fields": fields,
        "x": x,
        "t": t,
        "dt": dt,
        "steps": steps,
        "tent": tent,
        "red_slabs": red_slabs,
        "black_slabs": black_slabs,
        "history": history,
        "exact": exact.reshape(steps, n).T,
    }


def _forward_solve(a, f, n, steps):
    """Solve the block lower triangular space-time system by time stepping."""
    u = np.zeros(n * steps)
    for step in range(steps):
        rows = slice(step * n, (step + 1) * n)
        block = a[rows, : (step + 1) * n]
        rhs = f[rows] - block[:, : step * n].dot(u[: step * n])
        diagonal = block[:, step * n : (step + 1) * n].diagonal()
        u[rows] = rhs / diagonal
    return u


def front_profile(field, t, tolerance=1e-8):
    """Time up to which each spatial node has converged.

    The tents make this a sawtooth, whose teeth are what Figure 3.3 shows: the
    valleys sit at the top of the slab just solved and the peaks half a subdomain
    further, where two tents overlap.
    """
    converged = field < tolerance
    front = np.zeros(field.shape[0])
    for node in range(field.shape[0]):
        column = converged[node]
        if not column[0]:
            continue
        bad = np.flatnonzero(~column)
        front[node] = t[bad[0] - 1] if bad.size else t[-1]
    return front


def converged_front(field, t, tolerance=1e-8):
    """Latest time up to which the error has fallen below ``tolerance`` everywhere."""
    return float(front_profile(field, t, tolerance).min())


def run_figure_3_3(output_dir, quick=False):
    """Figure 3.3: red-black SWR advances the exact wave solution slab by slab."""
    n = 200 if quick else SPACE_UNKNOWNS
    panels = FIGURE_3_3_PANELS[:2] if quick else FIGURE_3_3_PANELS
    sweeps = max(sweep for _, _, sweep, _ in panels) or 1
    result = run_red_black_swr(
        n=n,
        sweeps=sweeps,
        snapshots=[(colour, sweep) for _, colour, sweep, _ in panels],
    )

    x, t = result["x"], result["t"]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.0 * len(panels), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    metrics = []
    for ax, (label, colour, sweep, title) in zip(axes, panels):
        field = result["fields"][(colour, sweep)]
        image = ax.imshow(
            field.T,
            extent=(x[-1], x[0], t[0], t[-1]),
            origin="lower",
            aspect="auto",
            cmap=parula(),
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(f"({label}) {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(image, ax=ax)
        front = front_profile(field, t)
        metrics.append(
            {
                "panel": label,
                "colour": colour,
                "sweep": sweep,
                "max_error": float(field.max()),
                "front_valley": float(front.min()),
                "front_peak": float(front.max()),
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_3_3_utp")

    return finish(
        output_dir,
        "figure_3_3_utp",
        "Red-black SWR, equivalently unmapped tent pitching, applied to the "
        "second-order wave equation: the exact solution is recovered inside the "
        "tents and advances by one slab per iteration",
        {
            "parameters": {
                "space_unknowns": n,
                "time_steps": result["steps"],
                "dt": result["dt"],
                "wave_speed": WAVE_SPEED,
                "t_final": T_FINAL,
                "red_subdomains": RED_SUBDOMAINS,
                "tent_height_steps": result["tent"],
            },
            "panels": metrics,
            "error_history": result["history"],
        },
        artifacts,
    )
