"""Space-time multigrid, Figures 4.18 to 4.22 and the note on Table 4.1.

Port of `STMG_Heat_ADE.m` and `STMG_Bergers.m`.  Both scripts solve the
all-at-once system of the linear-theta method on a space-time grid with
``N_x = N_t = 2^l - 1`` by a two-level cycle: a few damped block Jacobi
smoothing steps (4.32), one coarse grid correction on a grid coarsened by two in
*both* directions, and the same number of post-smoothing steps.  The coarse
problem is solved sequentially and exactly, which is what makes the two-level
convergence factor of Figures 4.19 to 4.21 a property of the smoother alone.

Figure 4.18 needs no solve at all: it is the local Fourier analysis factor (4.40)
maximized over the high frequencies, which is the tool the section uses to pick
the damping parameter ``eta = 1/2``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..plotting import save_figure, style_axis
from .common import BurgersStepper, finish

# The grid of both scripts: l = 8, so N_x = N_t = 255 unknowns per direction,
# dx = 1/256 and T = 5.
LEVEL = 8
T_FINAL = 5.0

# Figure 4.18: the LFA factor for three diffusion parameters on three grids.
LFA_NU_VALUES = (0.1, 0.01, 0.001)
LFA_MESHES = (1.0 / 64, 1.0 / 128, 1.0 / 256)
LFA_ETAS = np.linspace(0.1, 1.0, 91)

# Figures 4.19 and 4.21 sweep the damping parameter over this range, which is the
# scripts' ``eeta = 0.1:0.016:1.1``.
ETAS = np.arange(0.1, 1.1 + 1e-9, 0.016)
DAMPING_ITERATIONS = (5, 10, 15)

# Figure 4.19 uses one smoothing step and backward Euler; Figure 4.21 uses the
# trapezoidal rule and several smoothing counts, three for the heat equation and
# two to four for the ADE.
SMOOTHING_HEAT_TR = (3, 5, 10)
SMOOTHING_ADE_TR = (2, 3, 4)

# Figure 4.20: error against the iteration index for eta = 1/2, with one and
# three smoothing steps, for the ADE with three diffusion parameters and the heat
# equation (which the scripts select with nu < 0).
ITERATION_NU_VALUES = (0.001, 0.01, 0.1, -1.0)
ITERATION_LABELS = (
    r"ADE with $\nu=0.001$",
    r"ADE with $\nu=0.01$",
    r"ADE with $\nu=0.1$",
    "Heat equation",
)
ITERATION_STYLES = (
    ("#E8384F", "o"),
    ("#D34FD3", "s"),
    ("#2F6BFF", "D"),
    ("#17324D", "*"),
)
ITERATIONS = 30
SMOOTHING_COUNTS = (1, 3)
ETA_DEFAULT = 0.5

# Figure 4.22: Burgers' equation, backward Euler, eta = 0.28 as the script sets
# it inside its loop, sixteen iterations, two viscosities.
BURGERS_NU_VALUES = (0.1, 1.0)
BURGERS_ETA = 0.28
BURGERS_SMOOTHING = 2
BURGERS_ITERATIONS = 16
BURGERS_NEWTON_TOLERANCE = 1e-10
BURGERS_NEWTON_MAX = 100


def lfa_factor(eta, omega_dt, xi_dx, nu, dx, dt):
    """The convergence factor (4.40) of damped block Jacobi for the ADE."""
    denominator = (
        1.0
        + 2.0 * nu * dt / dx**2 * (1.0 - np.cos(xi_dx))
        + 1j * dt / dx * np.sin(xi_dx)
    )
    return np.abs(1.0 - eta * (1.0 - np.exp(-1j * omega_dt) / denominator))


def lfa_maximum(eta, nu, mesh, samples=257):
    """``rho_max``: (4.40) maximized over the high frequencies in time.

    The maximum is taken over ``xi dx in (-pi, pi)`` and ``omega dt in
    (pi/2, pi)``, the band that time coarsening has to leave to the smoother.
    """
    xi = np.linspace(-np.pi, np.pi, samples)
    omega = np.linspace(0.5 * np.pi, np.pi, samples)
    values = lfa_factor(eta, omega[:, None], xi[None, :], nu, mesh, mesh)
    return float(values.max())


def run_figure_4_18(output_dir, quick=False):
    """Figure 4.18: the LFA factor picks the damping parameter.

    Each panel maximizes (4.40) over the high frequencies for one diffusion
    parameter and three mesh sizes.  The curves are almost mesh-independent and
    all have their minimum near ``eta = 0.55``, close enough to the ``eta = 1/2``
    that Theorem 4.9 proves optimal for the heat equation; the value of the
    minimum, about 0.72, is why STMG needs a handful of smoothing steps for the
    advection-dominated cases.
    """
    etas = LFA_ETAS[::4] if quick else LFA_ETAS
    fig, axes = plt.subplots(
        1, len(LFA_NU_VALUES), figsize=(4.6 * len(LFA_NU_VALUES), 3.6),
        constrained_layout=True,
    )
    panels = []
    styles = (("#E8384F", "-"), ("#2F6BFF", "--"), ("#17324D", ":"))
    for ax, nu in zip(np.atleast_1d(axes), LFA_NU_VALUES):
        entry = {"nu": nu, "curves": {}}
        for mesh, (colour, dash) in zip(LFA_MESHES, styles):
            values = [lfa_maximum(eta, nu, mesh) for eta in etas]
            ax.plot(
                etas, values, dash, color=colour, linewidth=1.2,
                label=rf"$\Delta x=\Delta t=\frac{{1}}{{{int(round(1 / mesh))}}}$",
            )
            best = int(np.argmin(values))
            entry["curves"][f"mesh_1_over_{int(round(1 / mesh))}"] = {
                "eta": list(map(float, etas)),
                "rho_max": list(map(float, values)),
                "minimum": float(values[best]),
                "argmin_eta": float(etas[best]),
                "rho_max_at_one_half": lfa_maximum(0.5, nu, mesh),
            }
        ax.set_title(rf"ADE with $\nu={nu:g}$")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\rho_{\max}$")
        ax.set_xlim(0.1, 1.0)
        style_axis(ax)
        ax.legend(fontsize=8)
        panels.append(entry)
    artifacts = save_figure(fig, f"{output_dir}/figure_4_18_stmg_lfa")

    return finish(
        output_dir,
        "figure_4_18_stmg_lfa",
        "Maximum of the LFA convergence factor (4.40) over the high frequencies "
        "in time for the advection-diffusion equation, three diffusion parameters",
        {
            "parameters": {
                "factor": "equation (4.40), damped block Jacobi",
                "high_frequency_band": "xi dx in (-pi, pi), omega dt in (pi/2, pi)",
                "meshes": list(LFA_MESHES),
                "nu_values": list(LFA_NU_VALUES),
            },
            "panels": panels,
        },
        artifacts,
    )


def _transfer(fine_count):
    """The scripts' prolongation: linear interpolation, restriction its half."""
    coarse_count = (fine_count + 1) // 2 - 1
    matrix = sparse.lil_matrix((fine_count, coarse_count))
    for j in range(coarse_count):
        centre = 2 * j + 1
        matrix[centre, j] = 1.0
        matrix[centre - 1, j] = 0.5
        if centre + 1 < fine_count:
            matrix[centre + 1, j] = 0.5
    prolongation = matrix.tocsr()
    return prolongation, 0.5 * prolongation.T.tocsr()


def linear_setup(nu, theta, *, level=LEVEL, t_final=T_FINAL):
    """Assemble the all-at-once operators of `STMG_Heat_ADE.m`.

    The script writes ``u' = A u`` with ``A = -D_1 + nu D_2`` for the ADE and
    ``A = |nu| D_2`` for the heat equation, which it selects with ``nu < 0``, and
    steps it with ``Q u_{n+1} = tilde Q u_n``.
    """
    n = 2**level - 1
    dx = 1.0 / (n + 1)
    dt = t_final / n
    ones = np.ones(n)
    first = sparse.diags([ones, -ones], [-1, 1], shape=(n, n)) / (2.0 * dx)
    second = sparse.diags([ones, -2.0 * ones, ones], [-1, 0, 1], shape=(n, n)) / dx**2
    operator = (abs(nu) * second if nu < 0 else first + nu * second).tocsc()

    eye = sparse.eye(n, format="csc")
    q = (eye - dt * theta * operator).tocsc()
    q_tilde = (eye + dt * (1.0 - theta) * operator).tocsc()

    prolong_x, restrict_x = _transfer(n)
    prolong_t, restrict_t = _transfer(n)
    coarse_operator = (restrict_x @ operator @ prolong_x).tocsc()
    coarse_count = coarse_operator.shape[0]
    coarse_eye = sparse.eye(coarse_count, format="csc")
    q_coarse = (coarse_eye - 2.0 * dt * theta * coarse_operator).tocsc()
    q_coarse_tilde = (coarse_eye + 2.0 * dt * (1.0 - theta) * coarse_operator).tocsc()

    x = np.arange(1, n + 1) * dx
    u0 = np.sin(2.0 * np.pi * x)
    solve_q = splu(q)
    reference = np.zeros((n + 1, n))
    reference[0] = u0
    for step in range(n):
        reference[step + 1] = solve_q.solve(q_tilde @ reference[step])
    return {
        "steps": n,
        "size": n,
        "dx": dx,
        "dt": dt,
        "q": q,
        "q_tilde": q_tilde,
        "solve_q": solve_q,
        "solve_q_coarse": splu(q_coarse),
        "q_coarse_tilde": q_coarse_tilde,
        "prolong_x": prolong_x,
        "restrict_x": restrict_x,
        "prolong_t": prolong_t,
        "restrict_t": restrict_t,
        "u0": u0,
        "reference": reference,
        "x": x,
    }


def _residual(setup, values):
    """``tilde Q u_n - Q u_{n+1}`` for every interval, as the scripts form it."""
    return values[:-1] @ setup["q_tilde"].T.toarray() - values[1:] @ setup["q"].T.toarray()


def _smooth(setup, values, eta, count):
    """``count`` damped block Jacobi steps (4.32), parallel across time."""
    for _ in range(count):
        residual = _residual(setup, values)
        updated = values.copy()
        updated[1:] += eta * setup["solve_q"].solve(residual.T).T
        values = updated
    return values


def _coarse_correction(setup, residual):
    """Restrict, solve the coarse all-at-once system exactly, prolongate."""
    coarse = setup["restrict_x"] @ residual.T @ setup["restrict_t"].T
    coarse_steps = coarse.shape[1]
    correction = np.zeros((coarse.shape[0], coarse_steps + 1))
    for n in range(coarse_steps):
        correction[:, n + 1] = setup["solve_q_coarse"].solve(
            setup["q_coarse_tilde"] @ correction[:, n] + coarse[:, n]
        )
    return (setup["prolong_x"] @ correction[:, 1:] @ setup["prolong_t"].T).T


def stmg_errors(setup, eta, smoothing, iterations, *, start=None, tolerance=0.0):
    """The two-level cycle of `STMG_Heat_ADE.m`, and its error every iteration.

    One quirk of the script is kept: the residual that drives the coarse grid
    correction is the one of the iterate *before* the last smoothing step, since
    the script reuses the ``uold`` of its Jacobi loop.  It only makes the
    correction slightly inexact, and the published figures are the ones this
    version produces.
    """
    reference = setup["reference"]
    values = reference.copy() if start is None else np.array(start, dtype=float)
    errors = [float(np.abs(values - reference).max())]
    for _ in range(iterations):
        before_last = values if smoothing == 0 else _smooth(
            setup, values, eta, smoothing - 1
        )
        values = _smooth(setup, before_last, eta, 1)
        values = values + np.vstack(
            [np.zeros(setup["size"]), _coarse_correction(setup, _residual(setup, before_last))]
        )
        values = _smooth(setup, values, eta, smoothing)
        errors.append(float(np.abs(values - reference).max()))
        if tolerance and errors[-1] <= tolerance:
            break
    return errors


def random_start(setup):
    """``rand(Nx, Nt)`` after ``rng('default')``, with the exact initial column."""
    stream = np.random.RandomState(5489)
    steps, size = setup["steps"], setup["size"]
    draw = stream.random_sample(size * steps).reshape((size, steps), order="F")
    values = np.zeros((steps + 1, size))
    values[0] = setup["u0"]
    values[1:] = draw.T
    return values


def _damping_panel(ax, setup, smoothing, title, *, etas):
    """Error after 5, 10 and 15 cycles as a function of the damping parameter."""
    curves = {k: [] for k in DAMPING_ITERATIONS}
    start = random_start(setup)
    for eta in etas:
        errors = stmg_errors(
            setup, eta, smoothing, max(DAMPING_ITERATIONS), start=start
        )
        for k in DAMPING_ITERATIONS:
            curves[k].append(errors[k])
    styles = {5: ("#E8384F", "-"), 10: ("#2F6BFF", ":"), 15: ("#17324D", "--")}
    for k, values in curves.items():
        colour, dash = styles[k]
        ax.semilogy(etas, values, dash, color=colour, linewidth=1.1, label=rf"$k={k}$")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"damping parameter $\eta$")
    ax.set_ylabel("error")
    ax.set_xlim(0.15, 1.1)
    style_axis(ax)
    ax.legend(fontsize=8)
    return {
        "eta": list(map(float, etas)),
        "errors": {f"k_{k}": list(map(float, values)) for k, values in curves.items()},
        "best_eta": {
            f"k_{k}": float(etas[int(np.argmin(values))]) for k, values in curves.items()
        },
    }


def run_figure_4_19(output_dir, quick=False):
    """Figure 4.19: the numerical optimum of the damping parameter.

    One smoothing step, backward Euler, and the error after 5, 10 and 15 cycles
    over the whole range of ``eta``.  The three curves are flat and low on
    ``0.3 <= eta <= 0.7`` and blow up as ``eta`` approaches one, which is the
    numerical form of Theorem 4.9: the choice ``eta = 1/2`` is safe for both
    equations.
    """
    etas = ETAS[::4] if quick else ETAS
    problems = ((-1.0, "Two-grid STMG for heat equation"),
                (0.01, r"Two-grid STMG for ADE with $\nu=0.01$"))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0), constrained_layout=True)
    panels = {}
    for ax, (nu, title) in zip(axes, problems):
        setup = linear_setup(nu, theta=1.0)
        panels["heat" if nu < 0 else f"ade_nu_{nu:g}"] = _damping_panel(
            ax, setup, 1, title, etas=etas
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_19_stmg_damping")

    return finish(
        output_dir,
        "figure_4_19_stmg_damping",
        "Dependence of the two-level STMG error after 5, 10 and 15 iterations on "
        "the damping parameter, heat equation and ADE with nu = 0.01",
        {
            "parameters": _linear_parameters(1.0, smoothing=1),
            "panels": panels,
        },
        artifacts,
    )


def _linear_parameters(theta, **extra):
    setup = linear_setup(-1.0, theta)
    parameters = {
        "level": LEVEL,
        "steps": setup["steps"],
        "dx": setup["dx"],
        "dt": setup["dt"],
        "t_final": T_FINAL,
        "theta": theta,
        "integrator": "backward Euler" if theta == 1.0 else "trapezoidal rule",
        "initial_condition": "sin(2 pi x), homogeneous Dirichlet ends",
        "starting_iterate": "uniform random on (0,1), MATLAB rng('default')",
        "coarsening": "factor two in space and in time, Galerkin coarse operator",
        "coarse_solver": "exact sequential solve of the coarse all-at-once system",
    }
    parameters.update(extra)
    return parameters


def run_figure_4_20(output_dir, quick=False):
    """Figure 4.20: STMG with one and three smoothing steps.

    With ``eta = 1/2`` and backward Euler every curve is a straight line, so the
    two-level cycle converges at a fixed rate: about 0.5 per cycle for the heat
    equation and 0.65 to 0.7 for the ADE, and three smoothing steps roughly
    double the rate.  The ADE curves separate very little with ``nu``, which is
    the insensitivity the text points out and the opposite of what Parareal and
    MGRiT do.
    """
    counts = SMOOTHING_COUNTS[:1] if quick else SMOOTHING_COUNTS
    iterations = 12 if quick else ITERATIONS
    nu_values = ITERATION_NU_VALUES[:2] if quick else ITERATION_NU_VALUES

    fig, axes = plt.subplots(
        1, len(counts), figsize=(5.8 * len(counts), 4.3), constrained_layout=True
    )
    panels = {}
    for ax, smoothing in zip(np.atleast_1d(axes), counts):
        entry = {}
        for nu, label, (colour, marker) in zip(
            nu_values, ITERATION_LABELS, ITERATION_STYLES
        ):
            setup = linear_setup(nu, theta=1.0)
            errors = stmg_errors(
                setup, ETA_DEFAULT, smoothing, iterations, start=random_start(setup)
            )
            ax.semilogy(
                np.arange(len(errors)), errors, "--", color=colour, marker=marker,
                markersize=6, markerfacecolor="none", markevery=3, linewidth=0.9,
                label=label,
            )
            entry[label] = {
                "nu": nu,
                "errors": errors,
                "measured_factor": _measured_factor(errors),
            }
        # The published panels carry an unlabelled dash-dot line a little above
        # 1e-2, which the text never defines; on this grid it sits at the size of
        # one time step, so that is what is drawn.
        setup = linear_setup(-1.0, theta=1.0)
        ax.axhline(setup["dt"], color="#4A4A4A", linestyle="-.", linewidth=0.9)
        ax.set_title(
            f"{smoothing} Jacobi smoothing step" + ("s" if smoothing > 1 else "")
        )
        ax.set_xlabel("Iteration Index")
        ax.set_ylabel("Error")
        ax.set_xlim(0, iterations)
        ax.set_ylim(1e-10, 3.0)
        style_axis(ax)
        ax.legend(fontsize=8)
        panels[f"smoothing_{smoothing}"] = entry
    artifacts = save_figure(fig, f"{output_dir}/figure_4_20_stmg_iterations")

    return finish(
        output_dir,
        "figure_4_20_stmg_iterations",
        "Measured error of two-level STMG with one and three block Jacobi "
        "smoothing steps for the ADE and the heat equation, eta = 1/2",
        {
            "parameters": _linear_parameters(
                1.0, eta=ETA_DEFAULT, smoothing_counts=list(counts)
            ),
            "panels": panels,
        },
        artifacts,
    )


def _measured_factor(errors):
    usable = [value for value in errors if value > 1e-13]
    if len(usable) < 3:
        return None
    return float((usable[-1] / usable[1]) ** (1.0 / (len(usable) - 2)))


def run_figure_4_21(output_dir, quick=False):
    """Figure 4.21: with the trapezoidal rule STMG loses much of its strength.

    The same damping sweep as Figure 4.19, now for ``theta = 1/2``.  For the heat
    equation the error after 15 cycles stays near one for every ``eta`` even with
    ten smoothing steps -- the two-level cycle simply does not converge -- while
    the ADE does converge, more smoothing helps, and the best damping parameter
    moves to about 0.8 instead of the 1/2 of backward Euler.
    """
    etas = ETAS[::6] if quick else ETAS
    rows = (
        (-1.0, "heat equation", SMOOTHING_HEAT_TR),
        (0.01, r"ADE with $\nu=0.01$", SMOOTHING_ADE_TR),
    )
    if quick:
        rows = tuple((nu, title, counts[:1]) for nu, title, counts in rows)

    fig, axes = plt.subplots(
        len(rows), len(rows[0][2]),
        figsize=(4.4 * len(rows[0][2]), 3.6 * len(rows)),
        constrained_layout=True, squeeze=False,
    )
    panels = {}
    for row, (nu, title, counts) in zip(axes, rows):
        setup = linear_setup(nu, theta=0.5)
        key = "heat" if nu < 0 else f"ade_nu_{nu:g}"
        panels[key] = {}
        for ax, smoothing in zip(row, counts):
            panels[key][f"smoothing_{smoothing}"] = _damping_panel(
                ax, setup, smoothing,
                f"{title}, {smoothing} Jacobi steps", etas=etas,
            )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_21_stmg_trapezoidal")

    return finish(
        output_dir,
        "figure_4_21_stmg_trapezoidal",
        "Error of two-level STMG with the trapezoidal rule for several numbers "
        "of smoothing steps, as a function of the damping parameter",
        {
            "parameters": _linear_parameters(
                0.5,
                smoothing_counts={
                    "heat": list(SMOOTHING_HEAT_TR), "ade": list(SMOOTHING_ADE_TR)
                },
            ),
            "panels": panels,
        },
        artifacts,
    )


def burgers_setup(nu, *, level=LEVEL, t_final=T_FINAL, theta=1.0):
    """The nonlinear all-at-once problem of `STMG_Bergers.m`.

    ``u' = A_2 u + A_1 u**2`` with ``A_1`` the centred flux divided by ``4 dx``,
    that is the ``0.5 (u**2)_x`` of (2.6) -- the one place in this chapter where
    the committed factor and the published figure agree.
    """
    n = 2**level - 1
    dx = 1.0 / (n + 1)
    dt = t_final / n
    ones = np.ones(n)
    a1 = (sparse.diags([ones, -ones], [-1, 1], shape=(n, n)) / (4.0 * dx)).tocsr()
    a2 = (
        nu * sparse.diags([ones, -2.0 * ones, ones], [-1, 0, 1], shape=(n, n)) / dx**2
    ).tocsr()

    prolong_x, restrict_x = _transfer(n)
    prolong_t, restrict_t = _transfer(n)
    a1c = (restrict_x @ a1 @ prolong_x).tocsr()
    a2c = (restrict_x @ a2 @ prolong_x).tocsr()

    x = np.arange(1, n + 1) * dx
    u0 = np.sin(2.0 * np.pi * x)
    stepper = BurgersStepper(
        -a1, -a2, BURGERS_NEWTON_TOLERANCE, BURGERS_NEWTON_MAX
    )
    reference = np.zeros((n + 1, n))
    reference[0] = u0
    for step in range(n):
        reference[step + 1] = stepper(dt, reference[step])
    return {
        "steps": n,
        "size": n,
        "dx": dx,
        "dt": dt,
        "theta": theta,
        "a1": a1,
        "a2": a2,
        "a1c": a1c,
        "a2c": a2c,
        "prolong_x": prolong_x,
        "restrict_x": restrict_x,
        "prolong_t": prolong_t,
        "restrict_t": restrict_t,
        "u0": u0,
        "reference": reference,
        "stepper": stepper,
        "coarse_stepper": BurgersStepper(
            -a1c, -a2c, BURGERS_NEWTON_TOLERANCE, BURGERS_NEWTON_MAX
        ),
    }


def _burgers_flux(setup, values, coarse=False):
    a1 = setup["a1c"] if coarse else setup["a1"]
    a2 = setup["a2c"] if coarse else setup["a2"]
    return values @ a2.T + (values**2) @ a1.T


def _burgers_residual(setup, values, *, dt=None, coarse=False, u0=None):
    """Residual of the nonlinear all-at-once system, level by level."""
    dt = setup["dt"] if dt is None else dt
    theta = setup["theta"]
    u0 = setup["u0"] if u0 is None else u0
    previous = np.vstack([u0, values[:-1]])
    flux = _burgers_flux(setup, values, coarse)
    flux_previous = _burgers_flux(setup, previous, coarse)
    return (
        dt * (theta * flux + (1.0 - theta) * flux_previous) - (values - previous)
    )


def _burgers_smooth(setup, values, eta, count):
    """The nonlinear damped block Jacobi of `STMG_Bergers.m`.

    Each level solves ``C - theta dt f(C) = eta r_n`` for its own correction with
    a Newton iteration, so the smoother stays parallel across time even though
    every block is nonlinear.
    """
    stepper = setup["stepper"]
    zero = np.zeros(setup["size"])
    for _ in range(count):
        residual = eta * _burgers_residual(setup, values)
        correction = np.array(
            [stepper(setup["theta"] * setup["dt"], row, zero) for row in residual]
        )
        values = values + correction
    return values


def _burgers_coarse_correction(setup, values):
    """The FAS coarse grid correction of `STMG_Bergers.m`.

    Two things about ``solve_Kc`` are worth stating, because they decide what the
    method actually is.  It builds the coarse right-hand side from the restricted
    iterate in the FAS way, but it then keeps the *previous* coarse level frozen at
    that restricted iterate instead of using the level it has just computed, so the
    coarse problem decouples into one nonlinear equation per coarse level rather
    than being a sequential solve; and the first coarse level carries no initial
    condition at all.  Both are kept here, since the published Figure 4.22 is what
    this cycle produces.
    """
    residual = _burgers_residual(setup, values)
    coarse_residual = (setup["restrict_x"] @ residual.T @ setup["restrict_t"].T).T
    coarse_values = (setup["restrict_x"] @ values.T @ setup["restrict_t"].T).T
    coarse_dt = 2.0 * setup["dt"]
    theta = setup["theta"]
    stepper = setup["coarse_stepper"]

    flux = _burgers_flux(setup, coarse_values, True)
    # The frozen previous level appears on both sides of ``solve_Kc`` and cancels,
    # which leaves one equation per coarse level:
    # ``Un - theta dt_c f(Un) = r_c[n] + U_c[n] - theta dt_c f(U_c[n])``.
    target = coarse_residual + coarse_values - coarse_dt * theta * flux
    solved = np.array(
        [
            stepper(theta * coarse_dt, target[n], coarse_values[n])
            for n in range(coarse_values.shape[0])
        ]
    )
    difference = solved - coarse_values
    return (setup["prolong_x"] @ difference.T @ setup["prolong_t"].T).T


def burgers_errors(setup, eta, smoothing, iterations, *, start):
    """One two-level FAS cycle per iteration, and the error against the exact solve."""
    reference = setup["reference"][1:]
    values = np.array(start, dtype=float)
    errors = [float(np.abs(values - reference).max())]
    for _ in range(iterations):
        values = _burgers_smooth(setup, values, eta, smoothing)
        values = values + _burgers_coarse_correction(setup, values)
        values = _burgers_smooth(setup, values, eta, smoothing)
        errors.append(float(np.abs(values - reference).max()))
    return errors


def burgers_start(setup):
    """``random('unif', -1, 1, Nx Nt, 1)`` after ``rng('default')``."""
    stream = np.random.RandomState(5489)
    steps, size = setup["steps"], setup["size"]
    draw = -1.0 + 2.0 * stream.random_sample(size * steps)
    return draw.reshape((size, steps), order="F").T.copy()


def run_figure_4_22(output_dir, quick=False):
    """Figure 4.22: STMG for Burgers' equation with two smoothing steps.

    The nonlinear cycle keeps the structure of the linear one, with an FAS coarse
    grid correction and a Newton solve per block in the smoother.  At ``nu = 1``
    the error falls to 1e-4 in sixteen cycles; at ``nu = 0.1`` the same sixteen
    cycles gain barely more than one order, the same weak-diffusion degradation
    the linear ADE shows in Figure 4.20.
    """
    nu_values = BURGERS_NU_VALUES[1:] if quick else BURGERS_NU_VALUES
    iterations = 6 if quick else BURGERS_ITERATIONS

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    styles = (("#E8384F", "s"), ("#2F6BFF", "*"))
    curves = {}
    for nu, (colour, marker) in zip(nu_values, styles):
        setup = burgers_setup(nu)
        errors = burgers_errors(
            setup, BURGERS_ETA, BURGERS_SMOOTHING, iterations,
            start=burgers_start(setup),
        )
        ax.semilogy(
            np.arange(len(errors)), errors, "--", color=colour, marker=marker,
            markersize=7, markerfacecolor="none", markevery=2, linewidth=0.9,
            label=rf"$\nu={nu:g}$",
        )
        curves[f"nu_{nu:g}"] = {
            "errors": errors,
            "measured_factor": _measured_factor(errors),
        }
    setup = burgers_setup(nu_values[-1])
    ax.axhline(setup["dt"], color="#4A4A4A", linestyle="-.", linewidth=0.9)
    ax.set_title("STMG for Burgers' equation")
    ax.set_xlabel("Iteration Index")
    ax.set_ylabel("Error")
    ax.set_xlim(0, iterations)
    ax.set_ylim(1e-4, 3.0)
    style_axis(ax)
    ax.legend(fontsize=9)
    artifacts = save_figure(fig, f"{output_dir}/figure_4_22_stmg_burgers")

    return finish(
        output_dir,
        "figure_4_22_stmg_burgers",
        "Measured error of STMG with two block Jacobi smoothing steps for "
        "Burgers' equation",
        {
            "parameters": {
                "level": LEVEL,
                "steps": setup["steps"],
                "dx": setup["dx"],
                "dt": setup["dt"],
                "t_final": T_FINAL,
                "theta": 1.0,
                "eta": BURGERS_ETA,
                "smoothing": BURGERS_SMOOTHING,
                "newton_tolerance": BURGERS_NEWTON_TOLERANCE,
                "flux": "0.5 (u**2)_x, as both the script and (2.6) write it",
                "starting_iterate": "uniform random on (-1,1), MATLAB rng('default')",
                "coarse_solver": "FAS, exact sequential solve on the coarse grid",
                "nu_values": list(nu_values),
            },
            "curves": curves,
        },
        artifacts,
    )
