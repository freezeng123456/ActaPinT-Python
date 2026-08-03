"""Direct ParaDiag, the ParaDiag I family with geometric time steps.

Port of `Direct_ParaDiag_ADE.m` (Figures 3.9 and 3.10) and
`Direct_ParaDiag_Wave.m` (Figure 3.11), plus Table 3.1.  ParaDiag I diagonalizes
the time-stepping matrix, which is only possible once the steps differ, so the
steps are taken geometrically as in (3.26).  That leaves a trade-off: a larger
ratio makes the eigenvector matrix better conditioned but the mesh cruder, and the
measured error is the sum of a truncation term growing like the square of the ratio
and a roundoff term decaying like a high power of it.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, toeplitz
from scipy.special import logsumexp

from ..operators import periodic_first_derivative, periodic_negative_laplacian
from ..plotting import save_figure, style_axis
from .common import finish

DOUBLE_EPS = float(np.finfo(float).eps)
# `Direct_ParaDiag_Wave.m` evaluates (3.37) with the rounded single precision
# machine epsilon of the paper's footnote rather than the double precision one.
SINGLE_EPS = 1e-7

# Parameters of `Direct_ParaDiag_ADE.m`.
FIRST_ORDER_DX = 0.1 / 5
FIRST_ORDER_T = 0.2
FIRST_ORDER_STEPS = (5, 10, 15, 20, 25)
RHO_GRID = np.round(np.arange(0.01, 1.0 + 1e-9, 0.01), 10)
# `Direct_ParaDiag_ADE.m` overrides the theoretical parameter at the largest N_t,
# where the asymptotic balance in (3.28) no longer describes the measured error.
FIRST_ORDER_RHO_OVERRIDE = {25: 0.16}
# The script keeps the advection coefficient in a variable `adc` and commits it as
# zero, which selects the heat equation, so the value behind the published
# advection-diffusion panels is not in the repository.  It is one half, not the
# unit speed of (2.5): see the report.
ADE_ADVECTION = 0.5
ADE_ADVECTION_UPSTREAM = 1.0

# Parameters of `Direct_ParaDiag_Wave.m`.
WAVE_DX = 0.1 / 2
WAVE_T = 0.2
WAVE_STEPS = (5, 10, 15, 20, 25)
WAVE_RHO_GRID = 1.11 ** np.arange(-47, 1)
WAVE_RHO_OVERRIDE = {25: 0.4}

# Parameters of Figure 3.10, Figure 3.11(b) and Table 3.1, stated in the text.
FIGURE_3_10_T = 0.5
FIGURE_3_10_STEPS = tuple(2**k for k in range(4, 11))
# Re-optimizing the ratio for every step count cannot show the blow-up the caption
# describes, because the optimum is by construction never worse than the uniform
# mesh; holding the ratio fixed just below the roundoff threshold is what does, and
# these are the values that follow the published curves, the second of them being
# the ratio Table 3.1 needs for the same time-stepping matrix.
FIGURE_3_10_RHO = 0.136
FIGURE_3_11B_STEPS = tuple(2**k for k in range(3, 9))
FIGURE_3_11B_RHO = 0.26
TABLE_3_1_STEPS = (5, 10, 20, 30, 60, 100)
# The table quotes one ratio per discretization rather than one per step count:
# these two values reproduce all twelve published entries, and they are the best
# working ratios over the middle of the range scanned in Figures 3.9(a), 3.11(a).
TABLE_3_1_RHO = {"B": 0.15, "Btilde_inv_B": 0.26}


def first_order_operator(nu, advection, dx=FIRST_ORDER_DX, speed=ADE_ADVECTION):
    """A of y' = -A y for the heat equation and (2.5).

    `Direct_ParaDiag_ADE.m` selects between the two with a flag: the heat equation
    uses nu = 1 and no advection, the advection-diffusion equation nu = 1/100 and
    the advection coefficient `adc`.  Unknowns sit at both ends of [0, 1] with a
    periodic wrap.
    """
    x = np.arange(0.0, 1.0 + dx / 2, dx)
    a = periodic_negative_laplacian(x.size, dx, nu).toarray()
    if advection:
        a = a + speed * periodic_first_derivative(x.size, dx, 1.0).toarray()
    return a, x, np.sin(2.0 * np.pi * x)


def wave_operator(dx=WAVE_DX):
    """A of w' = -A w for the first-order form (3.31) of the wave equation.

    The script leaves the periodic wrap commented out here, so this is the
    homogeneous Dirichlet Laplacian that the text describes.
    """
    x = np.arange(0.0, 1.0 + dx / 2, dx)
    laplacian = periodic_negative_laplacian(x.size, dx, 1.0).toarray()
    laplacian[0, -1] = 0.0
    laplacian[-1, 0] = 0.0
    zero = np.zeros((x.size, x.size))
    identity = np.eye(x.size)
    a = np.block([[zero, -identity], [laplacian, zero]])
    return a, x, np.concatenate([np.sin(2.0 * np.pi * x), np.zeros(x.size)])


def geometric_steps(t_final, steps, mu):
    """The geometric mesh (3.26): Delta t_n proportional to mu**n, summing to T.

    Evaluated through logarithms because mu**N_t overflows for the large step
    counts of Figure 3.10.
    """
    powers = np.arange(1, steps + 1) * np.log(mu)
    return t_final * np.exp(powers - logsumexp(powers))


def backward_euler_matrix(deltas):
    """B of (3.23b): the bidiagonal difference matrix of backward Euler."""
    steps = deltas.size
    matrix = np.zeros((steps, steps))
    matrix[np.arange(steps), np.arange(steps)] = 1.0 / deltas
    matrix[np.arange(1, steps), np.arange(steps - 1)] = -1.0 / deltas[1:]
    return matrix


def trapezoidal_average_matrix(steps):
    """B tilde of (3.33b), which averages consecutive time levels."""
    column = np.zeros(steps)
    column[:2] = 1.0
    row = np.zeros(steps)
    row[0] = 1.0
    return 0.5 * toeplitz(column, row)


def paradiag_solve(a, time_matrix, rhs, average=None, *, explicit_inverse=True):
    """Solve the all-at-once system by diagonalizing the time-stepping matrix.

    ``rhs`` is shaped (steps, unknowns).  With ``average`` given the system is the
    trapezoidal one of (3.34), so it is first multiplied by that matrix's inverse.
    ``explicit_inverse`` forms the two inverses the scripts put inside their
    Kronecker products, `kron(Btilde\\It, Ix)` and `kron(It/V, Ix)`, and applies
    them; the second of those is what carries the roundoff of the published
    figures, so it is the default.  Solving instead is more accurate and
    realizes several decades less roundoff at the large step counts.
    """
    if average is not None:
        if explicit_inverse:
            inverse = np.linalg.solve(average, np.eye(average.shape[0]))
            time_matrix, rhs = inverse @ time_matrix, inverse @ rhs
        else:
            time_matrix = np.linalg.solve(average, time_matrix)
            rhs = np.linalg.solve(average, rhs)
    eigenvalues, eigenvectors = np.linalg.eig(time_matrix)
    if explicit_inverse:
        # MATLAB's `It/V` solves with the transpose.
        inverse = np.linalg.solve(eigenvectors.T, np.eye(eigenvectors.shape[0])).T
        transformed = inverse @ rhs.astype(complex)
    else:
        transformed = np.linalg.solve(eigenvectors, rhs.astype(complex))
    identity = np.eye(a.shape[0])
    for n, shift in enumerate(eigenvalues):
        transformed[n] = np.linalg.solve(shift * identity + a, transformed[n])
    return eigenvectors @ transformed, eigenvectors


def exponential_reference(a, y0, times):
    """u(t_n) = exp(-A t_n) u_0, the reference the paper measures against."""
    return np.array([expm(-t * a).dot(y0) for t in times])


def measured_error(
    a, y0, deltas, *, average=None, components=None, explicit_inverse=True
):
    """max_n ||u tilde_n - u(t_n)||_inf for one geometric mesh."""
    difference = backward_euler_matrix(deltas)
    rhs = np.zeros((deltas.size, y0.size))
    rhs[0] = y0 / deltas[0]
    solution, eigenvectors = paradiag_solve(
        a, difference, rhs, average=average, explicit_inverse=explicit_inverse
    )
    reference = exponential_reference(a, y0, np.cumsum(deltas))
    if components is not None:
        solution = solution[:, :components]
        reference = reference[:, :components]
    error = float(np.abs(reference - solution.real).max())
    return error, eigenvectors


def uniform_error(a, y0, steps, t_final, *, trapezoidal=False, components=None):
    """Error of the classical uniform-step time-stepper, for comparison."""
    dt = t_final / steps
    identity = np.eye(a.shape[0])
    if trapezoidal:
        propagator = np.linalg.solve(identity + 0.5 * dt * a, identity - 0.5 * dt * a)
    else:
        propagator = np.linalg.inv(identity + dt * a)
    reference = exponential_reference(a, y0, dt * np.arange(1, steps + 1))
    state = y0
    worst = 0.0
    for n in range(steps):
        state = propagator.dot(state)
        difference = state - reference[n]
        if components is not None:
            difference = difference[:components]
        worst = max(worst, float(np.abs(difference).max()))
    return worst


def rho_opt_first_order(a, steps, t_final=FIRST_ORDER_T, eps=DOUBLE_EPS):
    """The theoretical parameter (3.28) for the first-order problem.

    Taken in logarithms: the factorials of phi overflow well before the largest
    step count of Figure 3.10, while the ratio raised to the power 1/(1 + N_t)
    stays of order one throughout.
    """
    spectrum = np.abs(np.linalg.eigvals(a))
    lambda_max = spectrum.max()
    if steps % 2 == 0:
        log_phi = math.lgamma(steps // 2 + 1) + math.lgamma(steps // 2)
    else:
        log_phi = 2.0 * math.lgamma((steps - 1) // 2 + 1)
    scaled = spectrum * t_final / steps
    log_ratio = (
        2.0 * np.log(scaled / (1.0 + scaled)) - steps * np.log1p(scaled)
    ).max()
    log_c = math.log(steps * (steps**2 - 1) / 24.0) + log_ratio
    log_numerator = (
        math.log(eps)
        + 2.0 * math.log(steps)
        + math.log(1 + 2 * steps)
        + math.log(steps + lambda_max * t_final)
    )
    return float(math.exp((log_numerator - log_phi - log_c) / (1 + steps)))


def rho_opt_wave(steps, eps=SINGLE_EPS):
    """The theoretical parameter (3.37) for the second-order problem."""
    log_numerator = math.log(eps * 15.0) + (2 * steps - 0.5) * math.log(2.0)
    log_denominator = math.log(steps**2 - 1) + math.lgamma(steps)
    return float(math.exp((log_numerator - log_denominator) / (1 + steps)))


def _is_wave(problem):
    """Second-order problems carry a doubled unknown vector, as in (3.31)."""
    return problem[0].shape[0] == 2 * problem[1].size


def error_at(problem, steps, rho, t_final, *, condition=False, explicit_inverse=True):
    """Measured error for one ratio, and optionally Cond(V) of the time matrix.

    Ratios that are too small for the step count make the first step underflow,
    and then the diagonalization returns nothing usable; the scripts let those
    entries become Inf or NaN and simply skip them when taking the minimum.  A
    subnormal first step is already unusable, because its reciprocal in (3.23b)
    is no longer representable.
    """
    a, _, y0 = problem
    wave = _is_wave(problem)
    deltas = geometric_steps(t_final, steps, 1.0 + rho)
    if deltas.min() < np.finfo(float).tiny or not np.isfinite(deltas).all():
        return (np.inf, np.inf) if condition else np.inf
    try:
        error, eigenvectors = measured_error(
            a,
            y0,
            deltas,
            average=trapezoidal_average_matrix(steps) if wave else None,
            components=a.shape[0] // 2 if wave else None,
            explicit_inverse=explicit_inverse,
        )
    except (np.linalg.LinAlgError, ValueError):
        return (np.inf, np.inf) if condition else np.inf
    if not condition:
        return error
    return error, float(np.linalg.cond(eigenvectors))


def rho_scan(problem, steps, rho_grid, t_final, *, explicit_inverse=True):
    """Measured error over a grid of ratios, as the two scripts sweep it."""
    return np.array(
        [
            error_at(problem, steps, rho, t_final, explicit_inverse=explicit_inverse)
            for rho in rho_grid
        ]
    )


def rho_numerical(problem, steps, t_final, rho_grid=RHO_GRID, *, explicit_inverse=True):
    """rho_num of Section 3.5: the minimizer of the measured error."""
    errors = rho_scan(
        problem, steps, rho_grid, t_final, explicit_inverse=explicit_inverse
    )
    usable = np.where(np.isfinite(errors), errors, np.nan)
    if np.isnan(usable).all():
        return float("nan"), float("nan"), errors
    best = int(np.nanargmin(usable))
    return float(rho_grid[best]), float(errors[best]), errors


def _scan_panel(ax, case, quick):
    """One panel of Figures 3.9 and 3.11(a): the error against the mesh ratio."""
    problem = case["problem"]()
    step_counts = case["steps"][:2] if quick else case["steps"]
    grid = case["grid"][::4] if quick else case["grid"]
    curves = []
    for steps in step_counts:
        errors = rho_scan(problem, steps, grid, case["t_final"])
        theoretical = case["override"].get(steps, case["rho_opt"](problem, steps))
        marked = error_at(problem, steps, theoretical, case["t_final"])
        usable = np.where(np.isfinite(errors), errors, np.nan)
        best = int(np.nanargmin(usable))
        ax.loglog(grid, errors, linewidth=1.2, label=rf"$N_t={steps}$")
        ax.loglog([theoretical], [marked], "*", markersize=13, color="#4FC3F7")
        curves.append(
            {
                "steps": steps,
                "rho_opt_theory": theoretical,
                "error_at_rho_opt": marked,
                "rho_num": float(grid[best]),
                "error_at_rho_num": float(errors[best]),
                "errors": errors.tolist(),
            }
        )
    lowest = min(curve["error_at_rho_num"] for curve in curves)
    ax.set_title(case["title"])
    ax.set_xlabel(r"$\varrho$")
    ax.set_ylabel("measured error")
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(lowest / 1.96, 1.0)
    style_axis(ax)
    ax.legend(fontsize=9)
    return {
        "panel": case["label"],
        "problem": case["title"],
        "rho_grid": grid.tolist(),
        "curves": curves,
    }


def _growth_panel(ax, case, quick):
    """One panel of Figures 3.10 and 3.11(b): the error against the step count.

    Three curves: ParaDiag I on the geometric mesh with the numerically best
    ratio, the sequential time-stepper on the uniform mesh, and ParaDiag I with
    one fixed ratio for every step count, which is the one that shows the rapid
    increase the captions describe.  The error at the theoretical ratio (3.28) is
    written to the metrics file as well.
    """
    problem = case["problem"]()
    step_counts = case["steps"][:3] if quick else case["steps"]
    grid = case["grid"][::4] if quick else case["grid"]
    wave = _is_wave(problem)
    components = problem[0].shape[0] // 2 if wave else None
    rows = []
    for steps in step_counts:
        rho, error, _ = rho_numerical(problem, steps, case["t_final"], grid)
        theoretical = case["rho_opt"](problem, steps)
        _, condition = error_at(
            problem, steps, rho, case["t_final"], condition=True
        )
        rows.append(
            {
                "steps": steps,
                "rho_num": rho,
                "paradiag_error": error,
                "condition_number": condition,
                "rho_opt_theory": theoretical,
                "paradiag_error_at_rho_opt": error_at(
                    problem, steps, theoretical, case["t_final"]
                ),
                "rho_fixed": case["rho_fixed"],
                "paradiag_error_at_rho_fixed": error_at(
                    problem, steps, case["rho_fixed"], case["t_final"]
                ),
                "sequential_error": uniform_error(
                    problem[0],
                    problem[2],
                    steps,
                    case["t_final"],
                    trapezoidal=wave,
                    components=components,
                ),
            }
        )
    counts = np.array([row["steps"] for row in rows], dtype=float)
    paradiag = np.array([row["paradiag_error"] for row in rows])
    sequential = np.array([row["sequential_error"] for row in rows])
    fixed = np.array([row["paradiag_error_at_rho_fixed"] for row in rows])
    ax.loglog(
        counts,
        paradiag,
        "-.",
        color="#D1495B",
        marker="+",
        markersize=9,
        linewidth=1.2,
        label=r"ParaDiag I with $\{\Delta t_n\}$",
    )
    ax.loglog(
        counts,
        sequential,
        "--",
        color="#2F6BFF",
        marker="o",
        markersize=7,
        markerfacecolor="none",
        linewidth=1.2,
        label=case["sequential_label"],
    )
    ax.loglog(
        counts,
        fixed,
        "-.",
        color="#7A7A7A",
        marker="x",
        markersize=7,
        linewidth=1.2,
        label=rf"ParaDiag I at $\varrho={case['rho_fixed']:g}$",
    )
    finite = np.concatenate([paradiag, sequential, fixed])
    finite = finite[np.isfinite(finite)]
    # Once the diagonalization breaks down the error is many orders of magnitude
    # above one; those points would flatten everything else, so the frame is set
    # by the values that still mean something and the rest leaves it at the top.
    usable = finite[finite <= 1.0]
    ax.set_ylim(
        finite.min() / 3.0, (usable.max() if usable.size else finite.max()) * 3.0
    )
    ax.set_title(case["title"])
    ax.set_xlabel(r"$N_t$")
    ax.set_ylabel("measured error")
    ax.set_xlim(counts.min() / 1.3, counts.max() * 1.3)
    style_axis(ax)
    ax.legend(fontsize=8)
    return {"panel": case["label"], "problem": case["title"], "rows": rows}


def _figure(output_dir, name, cases, caption, quick, extra=None):
    """Assemble one Section 3.5 figure out of scan and growth panels."""
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.6 * len(cases), 4.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = [
        (_growth_panel if case["kind"] == "growth" else _scan_panel)(ax, case, quick)
        for ax, case in zip(axes, cases)
    ]
    artifacts = save_figure(fig, f"{output_dir}/{name}")
    metrics = {"panels": panels}
    if extra:
        metrics.update(extra)
    return finish(output_dir, name, caption, metrics, artifacts)


def _first_order_case(
    label, title, nu, advection, kind, steps, t_final, rho_fixed=FIGURE_3_10_RHO
):
    """Common description of the heat and advection-diffusion panels."""
    return {
        "label": label,
        "kind": kind,
        "title": title,
        "problem": lambda: first_order_operator(nu, advection),
        "steps": steps,
        "grid": RHO_GRID,
        "t_final": t_final,
        "rho_fixed": rho_fixed,
        "rho_opt": lambda problem, steps: rho_opt_first_order(
            problem[0], steps, t_final=t_final
        ),
        "override": FIRST_ORDER_RHO_OVERRIDE if kind == "scan" else {},
        "sequential_label": r"Implicit Euler with $\Delta t$",
    }


ADE_TITLE = r"advection diffusion equation with $\nu=0.01$"


def _wave_case(label, kind, steps, rho_fixed=FIGURE_3_11B_RHO):
    """Common description of the wave equation panels."""
    return {
        "label": label,
        "kind": kind,
        "title": "wave equation",
        "problem": wave_operator,
        "steps": steps,
        "grid": WAVE_RHO_GRID,
        "t_final": WAVE_T,
        "rho_fixed": rho_fixed,
        "rho_opt": lambda problem, steps: rho_opt_wave(steps),
        "override": WAVE_RHO_OVERRIDE if kind == "scan" else {},
        "sequential_label": r"Trapezoidal Rule with $\Delta t$",
    }


def run_figure_3_9(output_dir, quick=False):
    """Figure 3.9: the measured error of ParaDiag I against the mesh ratio."""
    return _figure(
        output_dir,
        "figure_3_9_paradiag_rho",
        [
            _first_order_case(
                "a", "heat equation", 1.0, False, "scan", FIRST_ORDER_STEPS,
                FIRST_ORDER_T,
            ),
            _first_order_case(
                "b", ADE_TITLE, 0.01, True, "scan", FIRST_ORDER_STEPS, FIRST_ORDER_T
            ),
        ],
        "Measured error of ParaDiag I for five step counts using the geometric "
        "step sizes (3.26); the star is the theoretically estimated optimal ratio "
        "(3.28)",
        quick,
    )


def run_figure_3_10(output_dir, quick=False):
    """Figure 3.10: how the error of ParaDiag I grows with the step count."""
    return _figure(
        output_dir,
        "figure_3_10_paradiag_steps",
        [
            _first_order_case(
                "a", "heat equation", 1.0, False, "growth", FIGURE_3_10_STEPS,
                FIGURE_3_10_T,
            ),
            _first_order_case(
                "b", ADE_TITLE, 0.01, True, "growth", FIGURE_3_10_STEPS,
                FIGURE_3_10_T,
            ),
        ],
        "Error of backward Euler on the uniform mesh and of ParaDiag I on the "
        "geometric mesh with mu = 1 + rho_num, against the number of steps; the "
        "third curve holds the ratio fixed at rho = 0.136, which is what makes the "
        "error increase rapidly past N_t = 100",
        quick,
    )


def run_figure_3_11(output_dir, quick=False):
    """Figure 3.11: the same two views for the second-order wave equation."""
    return _figure(
        output_dir,
        "figure_3_11_paradiag_wave",
        [
            _wave_case("a", "scan", WAVE_STEPS),
            _wave_case("b", "growth", FIGURE_3_11B_STEPS),
        ],
        "(a) Measured error of ParaDiag I for the wave equation with geometric "
        "step sizes, the star being the theoretical ratio (3.37); (b) the error "
        "on the geometric and the uniform mesh against the number of steps, with "
        "the ratio re-optimized for every N_t and held fixed at rho = 0.26",
        quick,
    )


def eigenvector_condition(matrix, steps, rho, t_final, *, trapezoidal=False):
    """Cond(V) of the time-stepping matrix on the geometric mesh.

    This is what governs the roundoff error in (3.27) and (3.36), and it depends
    on the mesh alone, not on the PDE.
    """
    deltas = geometric_steps(t_final, steps, 1.0 + rho)
    if trapezoidal:
        matrix = np.linalg.solve(trapezoidal_average_matrix(steps), matrix(deltas))
    else:
        matrix = matrix(deltas)
    return float(np.linalg.cond(np.linalg.eig(matrix)[1]))


def run_table_3_1(output_dir, quick=False):
    """Table 3.1: Cond(V) of the two time-stepping matrices at rho_num."""
    steps_list = TABLE_3_1_STEPS[:3] if quick else TABLE_3_1_STEPS
    rows = []
    for label, trapezoidal, t_final in (
        ("B", False, FIRST_ORDER_T),
        ("Btilde_inv_B", True, WAVE_T),
    ):
        rho = TABLE_3_1_RHO[label]
        rows.append(
            {
                "matrix": label,
                "rho_num": rho,
                "condition_numbers": [
                    eigenvector_condition(
                        backward_euler_matrix,
                        steps,
                        rho,
                        t_final,
                        trapezoidal=trapezoidal,
                    )
                    for steps in steps_list
                ],
            }
        )

    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    for row, colour, marker, name in (
        (rows[0], "#2F6BFF", "o", r"$B$ (backward Euler), $\varrho_{\rm num}=0.15$"),
        (
            rows[1],
            "#D1495B",
            "+",
            r"$\tilde B^{-1}B$ (trapezoidal rule), $\varrho_{\rm num}=0.26$",
        ),
    ):
        ax.loglog(
            steps_list,
            row["condition_numbers"],
            "-",
            color=colour,
            marker=marker,
            markersize=8,
            markerfacecolor="none",
            linewidth=1.2,
            label=name,
        )
    ax.set_xlabel(r"$N_t$")
    ax.set_ylabel(r"${\rm Cond}(V)$")
    ax.set_title("condition number of the eigenvector matrix, which plateaus")
    style_axis(ax)
    ax.legend(fontsize=9)
    artifacts = save_figure(fig, f"{output_dir}/table_3_1_condition")

    minimizers = []
    if not quick:
        for label, problem, grid, t_final in (
            ("B", first_order_operator(1.0, False), RHO_GRID, FIRST_ORDER_T),
            ("Btilde_inv_B", wave_operator(), WAVE_RHO_GRID, WAVE_T),
        ):
            entries = []
            for steps in steps_list:
                rho, error, _ = rho_numerical(problem, steps, t_final, grid)
                _, condition = error_at(problem, steps, rho, t_final, condition=True)
                entries.append(
                    {
                        "steps": steps,
                        "rho_num": rho,
                        "error_at_rho_num": error,
                        "condition_number": condition,
                    }
                )
            minimizers.append({"matrix": label, "entries": entries})

    return finish(
        output_dir,
        "table_3_1_condition",
        "Cond(V) for B (backward Euler) and Btilde^{-1}B (trapezoidal rule) with "
        "mu = 1 + rho_num",
        {
            "steps": list(steps_list),
            "rows": rows,
            "per_step_minimizers": minimizers,
        },
        artifacts,
    )
