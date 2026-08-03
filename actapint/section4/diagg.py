"""Parareal with a diagonalization-based coarse solver.

Port of `DiagG_Parareal_Heat_Wave.m`, which produces Figures 4.14 to 4.16, and of
the Burgers' experiment of Figure 4.17, which the repository does not ship.

The variant (4.26) keeps the coarse grid correction of standard Parareal but
changes the coarse solver: instead of taking one large step, ``F*_alpha`` solves
the *fine* system over the whole interval ``[T_n, T_{n+1}]`` -- the same
integrator, the same ``J`` steps -- after replacing its initial condition by the
alpha-coupled one ``u_0 = alpha u_J + (1 - alpha) z`` of (4.28).  That makes the
system an alpha-circulant one, so a single diagonalization solves all ``J`` fine
steps at once, at roughly ``1/J`` of the cost of the sequential fine solver.

Because the coarse solver is then as accurate as the fine one up to the coupling,
the convergence factor (4.29) is ``alpha`` for negative real spectra and
``2 alpha N_t / (1 + alpha)`` for imaginary ones, which is why this is the one
Parareal variant of the chapter that also works for the wave equation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve

from ..operators import (
    dirichlet_negative_laplacian,
    periodic_negative_laplacian,
)
from ..plotting import save_figure, style_axis
from .common import (
    BURGERS_FLUX_SCALE_AS_COMMITTED,
    BurgersStepper,
    alpha_circulant_factors,
    burgers_operators,
    finish,
    matlab_random_iterate,
    sequential_solution,
    truncation_error,
)

# The discretization of Figures 4.14 to 4.16: the trapezoidal rule for both F and
# F*, ``DT = 1/12``, ``J = 10``, ``dx = 1/100``.  The heat panels are the exception
# on ``J``; see `HEAT_SUBSTEPS`.
DX = 1.0 / 100
COARSE_DT = 1.0 / 12
SUBSTEPS = 10
THETA = 0.5

# Figure 4.14: the heat equation with homogeneous Dirichlet ends, two grid
# lengths and three values of alpha, against the predicted factor rho = alpha.
HEAT_STEPS = (36, 72)
HEAT_ALPHAS = (1e-1, 1e-2, 1e-3)
# The heat panels need ``J = 20`` rather than the ``J = 10`` of the text, the
# same kind of recovery Figure 4.2 needed.  The asymptotic factor of the
# variant is ``alpha`` times a function of ``max |R^J|`` over the spectrum, and
# for the trapezoidal rule that maximum is ``exp(-J**2 dx**2 / DT)``, attained at
# the top of the spectrum where ``R -> -1``; it is 0.887 at ``J = 10`` and 0.619
# at ``J = 20``.  Sweeping ``J`` against the published curves gives a sharp
# minimum at 20 -- the deviation over all six curves is 0.05 decades there, 0.24
# at 19, 0.60 at 21 and 3.47 at the text's 10 -- and ``dx = 1/100`` is fixed
# independently by the discretization-error line the panels draw at 1e-4.  The
# wave panels are the other way round and keep ``SUBSTEPS``; see `run_figure_4_15`.
HEAT_SUBSTEPS = 20

# Figure 4.14 prints no numbers, so the targets are the published curves
# themselves, digitized from the vector content of the article page: every panel
# of this figure is drawn as paths rather than as a raster, so the plotted
# coordinates are recovered exactly once the axis frame and its ticks are read
# off the same page.  See docs/SECTION4_REPRODUCTION.md for the calibration.
PUBLISHED_HEAT = {
    (36, 1e-1): (1.282, 5.806e-2, 3.871e-3, 3.408e-4, 3.611e-5, 3.776e-6,
                 4.030e-7, 3.910e-8, 3.537e-9, 3.467e-10, 3.435e-11),
    (36, 1e-2): (1.224, 4.911e-3, 3.281e-5, 2.589e-7, 1.929e-9, 1.553e-11),
    (36, 1e-3): (1.220, 4.919e-4, 3.428e-7, 2.607e-10),
    (72, 1e-1): (1.222, 6.337e-2, 4.031e-3, 3.375e-4, 3.441e-5, 3.727e-6,
                 4.084e-7, 4.524e-8, 5.057e-9, 5.700e-10, 6.464e-11),
    (72, 1e-2): (1.300, 5.075e-3, 3.418e-5, 2.709e-7, 2.161e-9, 1.892e-11),
    (72, 1e-3): (1.274, 6.774e-4, 4.382e-7, 3.276e-10),
}

# Figures 4.15 and 4.16: the wave equation with periodic ends, written as the
# first-order system of (4.30) whose spectrum is imaginary.
WAVE_ALPHAS = (1e-2, 1e-4)
WAVE_STEPS = (24, 48, 96)
WAVE_STEPS_SMALL_ALPHA = (24, 48, 96, 960)
# Figure 4.16 draws the three pairs (alpha, N_t) whose products alpha N_t differ
# by two orders of magnitude each.
WAVE_GROUPS = (((1e-2, 24), (1e-4, 24)), ((1e-4, 960), (1e-4, 24)))

# The seven curves of Figure 4.15, digitized from page 86 as `PUBLISHED_HEAT` was
# from page 85.  They keep ``SUBSTEPS`` and the script's committed start, and both
# are pinned by these numbers: the starting errors are one value per ``N_t``,
# shared by the two alphas, which is what ``u0`` in every window gives and a draw
# does not, and they are the ``J = 10`` fine solution -- at ``J = 20`` the first
# curve would start at 6.075 instead of 6.272.
PUBLISHED_WAVE = {
    (1e-2, 24): (6.274, 1.343, 1.680e-1, 1.025e-2, 5.900e-4, 2.218e-5, 4.831e-7,
                 1.787e-8, 3.089e-10),
    (1e-2, 48): (6.384, 3.196, 7.905e-1, 1.126e-1, 1.114e-2, 1.165e-3, 7.143e-5,
                 4.110e-6, 2.419e-7, 8.469e-9, 3.684e-10),
    (1e-2, 96): (6.384, 7.322, 3.888, 1.222, 2.729e-1, 5.530e-2, 7.855e-3,
                 9.210e-4, 1.210e-4, 1.063e-5, 8.112e-7, 8.065e-8, 4.948e-9,
                 2.696e-10),
    (1e-4, 24): (6.272, 1.323e-2, 1.639e-5, 1.007e-8),
    (1e-4, 48): (6.381, 2.918e-2, 7.006e-5, 9.954e-8),
    (1e-4, 96): (6.381, 5.412e-2, 2.463e-4, 8.090e-7, 1.786e-9),
    (1e-4, 960): (6.498, 6.133e-1, 2.888e-2, 9.061e-4, 2.147e-5, 4.048e-7,
                  6.334e-9),
}

TOLERANCE = 1e-10
ITERATIONS = 20

# Figure 4.17: Burgers' equation with periodic ends, ``DT = 0.1``, ``J = 10``,
# ``dx = 1/100``, counting iterations until the global error reaches 1e-8.
BURGERS_DX = 1.0 / 100
BURGERS_COARSE_DT = 0.1
BURGERS_SUBSTEPS = 10
BURGERS_NU_VALUES = (1.0, 0.01, 0.0001)
BURGERS_ALPHAS = tuple(2.0 ** -exponent for exponent in (13, 11, 9, 7, 5, 3))
BURGERS_STEPS = (10, 20, 40, 80, 160)
BURGERS_STEPS_FIXED = 40
BURGERS_ALPHA_FIXED = 1e-3
BURGERS_TARGET = 1e-8
BURGERS_ITERATIONS = 25
BURGERS_NEWTON_STEPS = 12
BURGERS_TOLERANCE = 1e-12

# Figure 4.17 labels its three viscosities red, blue, black, in that order.
COUNT_STYLES = (("#D1495B", "o"), ("#2F6BFF", "+"), ("#17324D", "s"))

CURVE_STYLES = (
    ("#D1495B", "o"),
    ("#17324D", "+"),
    ("#2F6BFF", "s"),
    ("#B33F62", "*"),
)


def heat_system(dx):
    """``A`` of ``u' + A u = 0`` for the heat equation with Dirichlet ends."""
    n = int(round(1.0 / dx)) + 1
    return dirichlet_negative_laplacian(n, dx, 1.0).tocsr(), np.linspace(0.0, 1.0, n)


def wave_system(dx):
    """``A`` of the first-order wave system, whose eigenvalues are imaginary.

    The script assembles ``[[0, -I], [A_1, 0]]`` with ``A_1`` the periodic
    negative Laplacian, so that ``u' + A u = 0`` is ``u_t = v``, ``v_t = u_xx``.
    """
    n = int(round(1.0 / dx)) + 1
    laplacian = periodic_negative_laplacian(n, dx, 1.0)
    eye = sparse.eye(n)
    operator = sparse.bmat([[None, -eye], [laplacian, None]], format="csr")
    return operator, np.linspace(0.0, 1.0, n)


def _theta_blocks(operator, fine_dt):
    """The two blocks of one linear-theta step: ``r1 y_j + r2 y_{j-1} = 0``."""
    size = operator.shape[0]
    eye = sparse.eye(size, format="csr")
    r1 = (eye + THETA * fine_dt * operator).toarray()
    r2 = (-eye + (1.0 - THETA) * fine_dt * operator).toarray()
    return r1, r2


def _shift_spectrum(substeps, alpha):
    """Scaling and eigenvalues of the alpha-circulant shift over the fine grid."""
    scale = alpha ** (np.arange(substeps) / substeps)
    stencil = np.zeros(substeps)
    stencil[1 % substeps] = 1.0
    return scale, np.fft.fft(scale * stencil)


def _circulant_solve(r1, r2, alpha, rhs):
    """Solve ``(I kron r1 + C_alpha kron r2) y = rhs`` by diagonalizing C_alpha.

    ``rhs`` is indexed by fine step first; its remaining axes are passed through,
    so the same routine solves for one right-hand side or for a whole matrix.
    """
    substeps = rhs.shape[0]
    scale, eigenvalues = _shift_spectrum(substeps, alpha)
    shape = (substeps,) + (1,) * (rhs.ndim - 1)
    transformed = np.fft.fft(scale.reshape(shape) * rhs, axis=0)
    solved = np.empty_like(transformed)
    for j in range(substeps):
        solved[j] = np.linalg.solve(r1 + eigenvalues[j] * r2, transformed[j])
    return np.fft.ifft(solved, axis=0) / scale.reshape(shape)


def propagators(operator, coarse_dt, substeps, alpha, *, scale_initial=True):
    """Return the fine propagator F and the diagonalized coarse solver F*.

    Both are dense matrices, because every experiment of this subsection applies
    them a few thousand times to the same small system.

    ``scale_initial`` is the factor ``1 - alpha`` that (4.28) puts in front of the
    initial condition and that `DiagG_Parareal_Heat_Wave.m` leaves out.  Write
    ``x = R^J`` for the fine amplification over one window.  With the factor,
    ``F* = (1 - alpha) x / (1 - alpha x)``, the defect is
    ``alpha x (1 - x) / (1 - alpha x)`` and the error of the sweep contracts, per
    mode, by ``defect / (1 - F*) = alpha x`` -- the bound ``rho = alpha`` of
    Theorem 4.8, sharp only for modes the fine solver leaves untouched.  Without
    it, ``F* = x / (1 - alpha x)`` and the same ratio is
    ``alpha x**2 / (1 - (1 + alpha) x)``, which exceeds ``alpha`` as soon as
    ``x > 0.6`` or so and is only reached after ``O(1/(1 - x))`` windows, so the
    curves bend and depend on ``N_t``.  The published panels of Figures 4.14 to
    4.16 all show the second behaviour, that is the committed script.
    """
    fine_dt = coarse_dt / substeps
    r1, r2 = _theta_blocks(operator, fine_dt)
    size = operator.shape[0]
    single = np.linalg.solve(r1, -r2)
    fine = np.linalg.matrix_power(single, substeps)

    rhs = np.zeros((substeps, size, size))
    rhs[0] = -r2 * ((1.0 - alpha) if scale_initial else 1.0)
    coarse = _circulant_solve(r1, r2, alpha, rhs)[-1].real
    return fine, coarse


def diag_g_parareal(fine, coarse, reference, start, *, iterations, tolerance):
    """The iteration (4.26), as `DiagG_Parareal_Heat_Wave.m` sweeps it.

    ``coarse`` is linear here, so it is applied to the increment
    ``U^{k+1}_n - U^k_n`` exactly as the script does.
    """
    steps = reference.shape[0] - 1
    previous = np.array(start, dtype=float)
    current = previous.copy()
    errors = [float(np.abs(previous - reference).max())]
    for _ in range(iterations):
        fine_values = previous @ fine.T
        for n in range(steps):
            current[n + 1] = coarse @ (current[n] - previous[n]) + fine_values[n]
        errors.append(float(np.abs(current - reference).max()))
        if errors[-1] <= tolerance:
            break
        previous = current.copy()
    return current, errors


def _sequential(matrix, u0, steps):
    trajectory = np.zeros((steps + 1, u0.size))
    trajectory[0] = u0
    for n in range(steps):
        trajectory[n + 1] = matrix @ trajectory[n]
    return trajectory


def _history(
    operator, u0, steps, alpha, *, iterations=ITERATIONS, random_start,
    scale_initial, substeps=SUBSTEPS,
):
    """Error history of one curve of Figures 4.14 to 4.16.

    ``random_start`` is the one place where `DiagG_Parareal_Heat_Wave.m` and the
    published figures disagree.  The script draws ``random('unif',-1,1,...)`` for
    the first iterate and then overwrites every column with ``u0``, so only the
    smooth initial error survives.  The wave panels are the committed state: their
    three curves start at 6.27, 6.38 and 6.38, which is what ``u0`` in every
    column gives, and they grow with ``N_t`` because a longer window samples the
    solution at more times.  The heat panels are the draw itself: their six curves
    start at 1.220 to 1.300, all different at the same ``N_t``, which only a
    random iterate redrawn per run can do.

    ``scale_initial`` is the ``1 - alpha`` factor of (4.28); see `propagators`.
    Both figures need it left out, as the script has it.
    """
    fine, coarse = propagators(
        operator, COARSE_DT, substeps, alpha, scale_initial=scale_initial
    )
    reference = _sequential(fine, u0, steps)
    if random_start:
        start = matlab_random_iterate(u0, steps, -1.0, 1.0)
    else:
        start = np.tile(u0, (steps + 1, 1))
    _, errors = diag_g_parareal(
        fine, coarse, reference, start, iterations=iterations, tolerance=TOLERANCE
    )
    return errors


def _error_panel(ax, curves, title, *, xmax, level=None):
    """Measured errors and their predicted factors, one panel."""
    top = max(max(errors) for errors, _, _ in curves)
    for (errors, bound, label), (colour, marker) in zip(curves, CURVE_STYLES):
        index = np.arange(len(errors))
        ax.semilogy(
            index, errors, "-.", color=colour, marker=marker, markersize=7,
            markerfacecolor="none", linewidth=1.0, label=label,
        )
        if bound is not None:
            full = np.arange(xmax + 1)
            ax.semilogy(
                full, errors[0] * bound**full, ":", color=colour, linewidth=1.2
            )
    if level is not None:
        ax.axhline(level, color="#4A4A4A", linestyle="-.", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("iteration index $k$")
    ax.set_ylabel("error")
    ax.set_xlim(0, xmax)
    ax.set_ylim(TOLERANCE, max(top, 1.0))
    style_axis(ax)
    ax.legend(fontsize=9)


def run_figure_4_14(output_dir, quick=False):
    """Figure 4.14: for the heat equation the rate is alpha, whatever N_t is.

    The coarse solver differs from the fine one only through the alpha-coupling
    of (4.28), so the defect it leaves behind is ``O(alpha)`` and the factor
    (4.29) is simply ``alpha``.  Both panels show the same three slopes, which is
    the ``N_t``-independence the text points out.

    The panels are drawn with ``J = 20``, not the ``J = 10`` of the text; see
    `HEAT_SUBSTEPS`.  The curves are then not the clean ``alpha^k``: the ratio
    climbs from ``0.45 alpha`` to ``1.1 alpha`` for ``alpha = 0.1`` and settles at
    ``0.8 alpha`` for the two small ones, and both the published curves and these
    do it.
    """
    steps_list = HEAT_STEPS[:1] if quick else HEAT_STEPS
    alphas = HEAT_ALPHAS[:2] if quick else HEAT_ALPHAS
    operator, x = heat_system(DX)
    u0 = np.sin(2.0 * np.pi * x) ** 2
    level = truncation_error(DX, COARSE_DT / HEAT_SUBSTEPS)

    fig, axes = plt.subplots(
        1, len(steps_list), figsize=(5.8 * len(steps_list), 4.3), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, steps in zip(axes, steps_list):
        curves = []
        entry = {"steps": steps, "t_final": steps * COARSE_DT, "curves": {}}
        for alpha in alphas:
            errors = _history(
                operator, u0, steps, alpha, random_start=True, scale_initial=False,
                substeps=HEAT_SUBSTEPS,
            )
            published = PUBLISHED_HEAT[(steps, alpha)]
            curves.append((errors, alpha, rf"$\alpha={alpha:g}$"))
            entry["curves"][f"alpha_{alpha:g}"] = {
                "errors": errors,
                "predicted_factor": alpha,
                "measured_factor": _measured_factor(errors),
                "published_errors": list(published),
                "published_factor": _tail_factor(published),
                "tail_factor": _tail_factor(errors[: len(published)]),
                "published_deviation_decades": _deviation(errors, published),
                "iterations_to_tolerance": len(errors) - 1,
            }
        _error_panel(
            ax, curves, rf"heat equation with $N_t={steps}$", xmax=10, level=level
        )
        panels.append(entry)
    artifacts = save_figure(fig, f"{output_dir}/figure_4_14_diag_coarse_heat")

    return finish(
        output_dir,
        "figure_4_14_diag_coarse_heat",
        "Error of the Parareal variant (4.26) for the heat equation with three "
        "values of alpha and two grid lengths, against the predicted factor "
        "rho = alpha",
        {
            "parameters": _parameters(
                alphas,
                substeps=HEAT_SUBSTEPS,
                coupling="v_0 = alpha v_J + u_n, the committed line of the script",
            ),
            "panels": panels,
        },
        artifacts,
    )


def _measured_factor(errors):
    """The geometric mean rate over the part of the curve above the tolerance."""
    usable = [value for value in errors if value > TOLERANCE]
    if len(usable) < 3:
        return None
    return float((usable[-1] / usable[1]) ** (1.0 / (len(usable) - 2)))


def _tail_factor(errors):
    """The factor of the last decade of a curve, fitted where it is straight.

    The first two iterations are a transient in every panel of this figure: the
    starting error is spread over the whole spectrum and the modes the coarse
    solver already resolves are removed at once.  The fit therefore begins at the
    third point, or at the second when the curve reaches the tolerance too fast
    for that, and the published curves are read the same way.
    """
    usable = [value for value in errors if value > TOLERANCE]
    start = 2 if len(usable) > 4 else 1
    if len(usable) - start < 2:
        return None
    exponents = np.log10(usable[start:])
    slope = np.polyfit(np.arange(start, len(usable)), exponents, 1)[0]
    return float(10.0**slope)


def _deviation(errors, published):
    """Root mean square distance in decades between a curve and the published one."""
    common = min(len(errors), len(published))
    model = np.log10(np.asarray(errors[:common]))
    target = np.log10(np.asarray(published[:common]))
    return float(np.sqrt(np.mean((model - target) ** 2)))


def _parameters(
    alphas,
    *,
    equation="heat, Dirichlet, u0 = sin(2 pi x)**2",
    initial_iterate="uniform random on (-1,1), MATLAB rng('default')",
    coupling="v_0 = alpha v_J + (1 - alpha) u_n, as in (4.28)",
    substeps=SUBSTEPS,
):
    return {
        "equation": equation,
        "dx": DX,
        "coarse_dt": COARSE_DT,
        "substeps": substeps,
        "fine_dt": COARSE_DT / substeps,
        "integrator": "trapezoidal rule for F and F*",
        "initial_iterate": initial_iterate,
        "coupling": coupling,
        "alphas": list(alphas),
        "tolerance": TOLERANCE,
    }


def run_figure_4_15(output_dir, quick=False):
    """Figure 4.15: the wave equation, where the factor grows with N_t.

    The spectrum is imaginary, so (4.29) predicts ``2 alpha N_t / (1 + alpha)``:
    for ``alpha = 0.01`` the panels separate as ``N_t`` grows, but for
    ``alpha = 1e-4`` even ``N_t = 960`` costs only about two extra iterations.

    These panels keep the ``J = 10`` of the text and the committed start of the
    script, both against Figure 4.14, and `PUBLISHED_WAVE` is what pins them: every
    plotted point above ``TOLERANCE`` comes out within a per cent, and neither
    ``J = 20`` nor a random start does that here.
    """
    operator, x = wave_system(DX)
    u0 = np.concatenate([np.sin(2.0 * np.pi * x) ** 2, np.zeros(x.size)])
    level = truncation_error(DX, COARSE_DT / SUBSTEPS)
    alphas = WAVE_ALPHAS[:1] if quick else WAVE_ALPHAS

    fig, axes = plt.subplots(
        1, len(alphas), figsize=(5.8 * len(alphas), 4.3), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, alpha in zip(axes, alphas):
        steps_list = WAVE_STEPS if alpha > 1e-3 else WAVE_STEPS_SMALL_ALPHA
        if quick:
            steps_list = steps_list[:2]
        curves = []
        entry = {"alpha": alpha, "curves": {}}
        for steps in steps_list:
            errors = _history(
                operator, u0, steps, alpha, random_start=False, scale_initial=False
            )
            published = PUBLISHED_WAVE[(alpha, steps)]
            curves.append((errors, None, rf"$N_t={steps}$"))
            entry["curves"][f"steps_{steps}"] = {
                "errors": errors,
                "predicted_factor": 2.0 * alpha * steps / (1.0 + alpha),
                "measured_factor": _measured_factor(errors),
                "published_errors": list(published),
                "published_deviation_decades": _deviation(errors, published),
                "iterations_to_tolerance": len(errors) - 1,
            }
        _error_panel(
            ax, curves, rf"wave equation, $\alpha={alpha:g}$",
            xmax=14 if alpha > 1e-3 else 10, level=level,
        )
        panels.append(entry)
    artifacts = save_figure(fig, f"{output_dir}/figure_4_15_diag_coarse_wave")

    return finish(
        output_dir,
        "figure_4_15_diag_coarse_wave",
        "Measured error of the Parareal variant (4.26) for the wave equation "
        "with two values of alpha and several grid lengths",
        {
            "parameters": _parameters(
                alphas,
                equation="wave, periodic, u0 = sin(2 pi x)**2, v0 = 0",
                initial_iterate="u0 in every window, as the committed script",
                coupling="v_0 = alpha v_J + u_n, as the committed script",
            ),
            "panels": panels,
        },
        artifacts,
    )


def _burgers_window(a1, a2, coarse_dt, substeps, alpha, tolerance, newton_max):
    """The two window solvers of (4.26) for Burgers' equation.

    ``fine`` is ``substeps`` implicit Euler steps, each with its own Newton
    solve.  ``coarse`` solves the same ``substeps`` steps together from the
    alpha-coupled initial condition (4.28), which is the all-at-once system
    (4.23) restricted to one window: with the Jacobian frozen the coefficient
    matrix is ``C_alpha kron I + dt I kron J`` and one diagonalization per
    quasi-Newton step solves all ``substeps`` stages at once, which is where the
    parallelism of the variant comes from.  Both forms are written out below,
    with the note on why the nonlinear runs use the fixed-point one.
    """
    fine_dt = coarse_dt / substeps
    stepper = BurgersStepper(a1, a2, tolerance, newton_max)
    flux = stepper.flux
    size = a1.shape[0]
    scale, eigenvalues = alpha_circulant_factors(substeps, alpha)
    eye = np.eye(size)

    def fine(u):
        z = np.asarray(u, dtype=float)
        for _ in range(substeps):
            z = stepper(fine_dt, z)
        return z

    def coarse(u):
        """``F*_alpha``: the window system (4.23) with the coupling (4.28).

        The same system `_diagonalized_coarse` below solves by diagonalization,
        solved instead by its own fixed point: propagate the window, re-couple the
        head to the tail, repeat.  The coupling contributes a factor ``alpha`` per
        pass, so three to eight passes reach round-off over the range of alpha used
        here, and the result is the solution of the very same system -- what is
        given up is the parallelism across the ``J`` fine steps, not the answer.
        The diagonalized form is used for the linear figures, where it is exact,
        but its nonlinear quasi-Newton stalls once a window has steepened: at
        ``nu = 1e-4`` it cannot push the residual below ``5e-2``, and the outer
        iteration then diverges.
        """
        u = np.asarray(u, dtype=float)
        tail = u
        for _ in range(newton_max):
            z = alpha * tail + (1.0 - alpha) * u
            for _ in range(substeps):
                z = stepper(fine_dt, z)
            if np.abs(z - tail).max() <= tolerance:
                return z
            tail = z
        return tail

    def _diagonalized_coarse(u):
        u = np.asarray(u, dtype=float)
        values = np.tile(u, (substeps, 1))
        source = np.zeros((substeps, size))
        source[0] = (1.0 - alpha) * u

        def residual_of(state):
            residual = (
                state
                - np.roll(state, 1, axis=0)
                + fine_dt * np.array([flux(row) for row in state])
                - source
            )
            residual[0] += (1.0 - alpha) * state[-1]
            return residual

        residual = residual_of(values)
        norm = float(np.abs(residual).max())
        for _ in range(newton_max):
            if norm <= tolerance:
                break
            # Refreshing the frozen state each step -- the window average of the
            # current iterate -- keeps the system alpha-circulant while tracking
            # the nonlinearity, and backtracking on the step length keeps the
            # quasi-Newton iteration a contraction once the window has steepened,
            # which it is not at nu = 1e-4 with the full step.
            jacobian = (
                a2 + 2.0 * a1.multiply(values.mean(axis=0)[None, :])
            ).toarray()
            factors = [
                lu_factor(value * eye + fine_dt * jacobian) for value in eigenvalues
            ]
            transformed = np.fft.fft(scale[:, None] * residual, axis=0)
            solved = np.array(
                [lu_solve(factors[n], transformed[n]) for n in range(substeps)]
            )
            step = (np.fft.ifft(solved, axis=0) / scale[:, None]).real
            if not np.all(np.isfinite(step)):
                break
            for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
                trial = values - damping * step
                trial_residual = residual_of(trial)
                trial_norm = float(np.abs(trial_residual).max())
                if np.isfinite(trial_norm) and trial_norm < norm:
                    values, residual, norm = trial, trial_residual, trial_norm
                    break
            else:
                break
        return values[-1]

    return fine, coarse


def _burgers_iterations(fine, coarse, reference, u0, steps, target, iterations):
    """Sweeps of (4.26) until the error reaches ``target``, and the error history."""
    current = np.tile(u0, (steps + 1, 1))
    stored = np.array([coarse(row) for row in current[:-1]])
    errors = [float(np.abs(current - reference).max())]
    reached = None
    for k in range(1, iterations + 1):
        fine_values = np.array([fine(current[n]) for n in range(steps)])
        updated = current.copy()
        for n in range(steps):
            advanced = coarse(updated[n])
            updated[n + 1] = advanced + fine_values[n] - stored[n]
            stored[n] = advanced
        current = updated
        errors.append(float(np.abs(current - reference).max()))
        if errors[-1] <= target:
            reached = k
            break
    return reached, errors


def _burgers_count(nu, alpha, steps, *, target=BURGERS_TARGET):
    a1, a2, x = burgers_operators(
        BURGERS_DX, nu, boundary_condition="periodic",
        flux_scale=BURGERS_FLUX_SCALE_AS_COMMITTED,
    )
    u0 = np.sin(2.0 * np.pi * x) ** 2
    fine, coarse = _burgers_window(
        a1, a2, BURGERS_COARSE_DT, BURGERS_SUBSTEPS, alpha,
        BURGERS_TOLERANCE, BURGERS_NEWTON_STEPS,
    )
    reference = sequential_solution(fine, u0, steps)
    reached, errors = _burgers_iterations(
        fine, coarse, reference, u0, steps, target, BURGERS_ITERATIONS
    )
    return reached, errors


def _count_panel(ax, curves, title, xvalues, xlabel, *, logx, logy):
    for (counts, label), (colour, marker) in zip(curves, COUNT_STYLES):
        drawn = [value if value is not None else np.nan for value in counts]
        ax.plot(
            xvalues, drawn, "-.", color=colour, marker=marker, markersize=8,
            markerfacecolor="none", linewidth=1.0, label=label,
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("iteration number")
    style_axis(ax)
    ax.legend(fontsize=9)


def run_figure_4_17(output_dir, quick=False):
    """Figure 4.17: iterations of (4.26) for Burgers' equation to reach 1e-8.

    Panel (a) fixes ``N_t = 40`` and sweeps alpha: the count grows with alpha, and
    it grows faster for small viscosity, which is the deterioration the text
    points out.  Panel (b) fixes ``alpha = 1e-3`` and sweeps ``N_t``: the counts
    are flat, so the variant is robust in the number of windows, exactly as for
    the linear heat equation of Figure 4.14.
    """
    nus = BURGERS_NU_VALUES[:2] if quick else BURGERS_NU_VALUES
    alphas = BURGERS_ALPHAS[:3] if quick else BURGERS_ALPHAS
    steps_list = BURGERS_STEPS[:3] if quick else BURGERS_STEPS

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    panels = {"alpha_sweep": {}, "steps_sweep": {}}

    curves = []
    for nu in nus:
        counts = []
        for alpha in alphas:
            reached, errors = _burgers_count(nu, alpha, BURGERS_STEPS_FIXED)
            counts.append(reached)
            panels["alpha_sweep"][f"nu_{nu:g}_alpha_{alpha:g}"] = {
                "iterations": reached,
                "errors": errors,
            }
        curves.append((counts, rf"$\nu={nu:g}$"))
    _count_panel(
        axes[0], curves, rf"Burgers' equation with $N_t={BURGERS_STEPS_FIXED}$",
        list(alphas), r"$\alpha$", logx=True, logy=True,
    )

    curves = []
    for nu in nus:
        counts = []
        for steps in steps_list:
            reached, errors = _burgers_count(nu, BURGERS_ALPHA_FIXED, steps)
            counts.append(reached)
            panels["steps_sweep"][f"nu_{nu:g}_steps_{steps}"] = {
                "iterations": reached,
                "errors": errors,
            }
        curves.append((counts, rf"$\nu={nu:g}$"))
    _count_panel(
        axes[1], curves,
        rf"Burgers' equation, $\alpha={BURGERS_ALPHA_FIXED:g}$",
        list(steps_list), r"$N_t$", logx=False, logy=False,
    )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_17_diag_coarse_burgers")

    return finish(
        output_dir,
        "figure_4_17_diag_coarse_burgers",
        "Iteration numbers of the Parareal variant (4.26) for Burgers' equation "
        "when the global error reaches 1e-8, for three values of the viscosity",
        {
            "parameters": {
                "equation": "Burgers, periodic, u0 = sin(2 pi x)**2",
                "dx": BURGERS_DX,
                "coarse_dt": BURGERS_COARSE_DT,
                "substeps": BURGERS_SUBSTEPS,
                "integrator": "implicit Euler for F and F*",
                "coarse_solver": "alpha-circulant all-at-once window solve of "
                "(4.23) and (4.28), Jacobian frozen at the window's initial value",
                "newton_steps": BURGERS_NEWTON_STEPS,
                "newton_tolerance": BURGERS_TOLERANCE,
                "target_error": BURGERS_TARGET,
                "nu_values": list(nus),
                "alphas": list(alphas),
                "steps": list(steps_list),
                "steps_fixed": BURGERS_STEPS_FIXED,
                "alpha_fixed": BURGERS_ALPHA_FIXED,
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_4_16(output_dir, quick=False):
    """Figure 4.16: the wave-equation factor is sharp only when alpha N_t is small.

    Each panel draws two of the three pairs ``(alpha, N_t)`` together with the
    predicted ``2 alpha N_t / (1 + alpha)``.  For ``alpha N_t = 2.4e-3`` the
    prediction sits right on the measured curve; for the two larger products the
    measured error leaves the bound and turns superlinear.
    """
    groups = WAVE_GROUPS[:1] if quick else WAVE_GROUPS
    operator, x = wave_system(DX)
    u0 = np.concatenate([np.sin(2.0 * np.pi * x) ** 2, np.zeros(x.size)])
    level = truncation_error(DX, COARSE_DT / SUBSTEPS)

    fig, axes = plt.subplots(
        1, len(groups), figsize=(5.8 * len(groups), 4.3), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    for ax, group in zip(axes, groups):
        curves = []
        entry = {"curves": {}}
        for alpha, steps in group:
            errors = _history(
                operator, u0, steps, alpha, random_start=False, scale_initial=False
            )
            bound = 2.0 * alpha * steps / (1.0 + alpha)
            fixed = group[0][1] == group[1][1]
            label = rf"$\alpha={alpha:g}$" if fixed else rf"$N_t={steps}$"
            curves.append((errors, bound, label))
            entry["curves"][f"alpha_{alpha:g}_steps_{steps}"] = {
                "errors": errors,
                "predicted_factor": bound,
                "measured_factor": _measured_factor(errors),
                "iterations_to_tolerance": len(errors) - 1,
            }
        title = (
            rf"wave equation with $N_t={group[0][1]}$"
            if group[0][1] == group[1][1]
            else rf"wave equation, $\alpha={group[0][0]:g}$"
        )
        _error_panel(ax, curves, title, xmax=10, level=level)
        panels.append(entry)
    artifacts = save_figure(fig, f"{output_dir}/figure_4_16_diag_coarse_sharpness")

    return finish(
        output_dir,
        "figure_4_16_diag_coarse_sharpness",
        "The predicted convergence factor of the Parareal variant (4.26) for the "
        "wave equation against the measured error, for three pairs (alpha, N_t)",
        {
            "parameters": _parameters(
                sorted({alpha for group in groups for alpha, _ in group}),
                equation="wave, periodic, u0 = sin(2 pi x)**2, v0 = 0",
                initial_iterate="u0 in every window, as the committed script",
                coupling="v_0 = alpha v_J + u_n, as the committed script",
            ),
            "panels": panels,
        },
        artifacts,
    )
