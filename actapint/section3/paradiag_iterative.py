"""Iterative ParaDiag, the ParaDiag II family built on alpha-circulant preconditioners.

Port of `Iterative_ParaDiag_ADE.m` (Figure 3.15a-d and, with periodic boundaries,
Figure 3.16), `Iterative_ParaDiag_Wave.m` (Figure 3.15e,f and Figure 3.17) and
`Iterative_ParaDiag_RoundError.m` (Figure 3.18).  ParaDiag II leaves the mesh
alone and instead replaces the Toeplitz time-stepping matrix K of (3.62a) by the
alpha-circulant matrix P_alpha of (3.58c), which a scaled Fourier matrix
diagonalizes, so the whole family lives or dies by the spectrum of P^-1 K.

Two conventions are worth spelling out, because both scripts use them silently.
The scripts assemble the positive definite operator A_s = Tri[-1 2 -1]/dx^2 and
write the PDEs as u' = -A_s u and u'' = -A_s u, whereas the paper writes u' = A u
with A = -A_s; the functions below build the paper's A.  And the scripts place
the initial condition on `linspace(0, 1, Nx)` while assembling A with dx = 1/Nx,
so the sampling grid is marginally coarser than the differencing grid.  That is
kept, since it changes the initial condition by O(dx) only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres

from ..operators import (
    dirichlet_first_derivative,
    dirichlet_negative_laplacian,
    periodic_first_derivative,
    periodic_negative_laplacian,
)
from ..plotting import save_figure, style_axis
from .common import finish

# Section 3.5.2, the data shared by Figures 3.15 and 3.16.
FIRST_ORDER_DT = 1.0 / 50
FIRST_ORDER_DX = 1.0 / 100
FIRST_ORDER_T = 2.0
GAMMA = 1.0 / 100
ADE_NU = (1e-3, 1e-6)
INITIAL_SCALE = 0.125
GMRES_TOLERANCE = 1e-12
WAVE_GMRES_TOLERANCE = 1e-14

# Figure 3.16, whose legend uses a different pair of diffusion parameters.
FIGURE_3_16_ALPHAS = (0.1, 0.001)
FIGURE_3_16_NU = (1e-2, 1e-6)
FIGURE_3_16_ITERATIONS = 14
FIGURE_3_16_TOLERANCE = 1e-12

# `Iterative_ParaDiag_Wave.m`, Figure 3.17.
WAVE_DT = 1.0 / 16
WAVE_DX = 1.0 / 128
WAVE_ALPHA = 0.02
WAVE_HORIZONS = (0.5, 10.0, 20.0)
WAVE_GAMMAS = (1.0 / 120, 1.0 / 120.01)

# `Iterative_ParaDiag_RoundError.m`, Figure 3.18.
ROUNDOFF_GAMMA = 1.0 / 100
ROUNDOFF_T = 4.0
ROUNDOFF_ALPHAS = (1e-3, 1e-6, 1e-11)
ROUNDOFF_ITERATIONS = 10
RANDOM_SEED = 20240501


def script_grid(dx, periodic):
    """The scripts' grid: N_x = 1/dx unknowns sampled on `linspace(0, 1, N_x)`."""
    n = int(round(1.0 / dx))
    if periodic:
        return n, np.arange(n) * dx
    return n, np.linspace(0.0, 1.0, n)


def first_order_problem(nu, dx, *, advection, periodic=False):
    """A of u' = A u for the heat equation and for (2.5), plus u_0.

    The scripts encode the heat equation as the advection-free case with unit
    diffusion, and the advection diffusion equation as A_1 + nu A_2.
    """
    n, x = script_grid(dx, periodic)
    laplacian = periodic_negative_laplacian if periodic else dirichlet_negative_laplacian
    derivative = periodic_first_derivative if periodic else dirichlet_first_derivative
    operator = laplacian(n, dx, nu)
    if advection:
        operator = operator + derivative(n, dx, 1.0)
    return -operator.toarray(), x, np.sin(2.0 * np.pi * x)


def wave_problem(dx, *, periodic=False):
    """A of u'' = A u for the wave equation (3.53), plus u_0."""
    n, x = script_grid(dx, periodic)
    laplacian = periodic_negative_laplacian if periodic else dirichlet_negative_laplacian
    return -laplacian(n, dx, 1.0).toarray(), x, np.sin(2.0 * np.pi * x)


def first_order_pair(a, dt, theta):
    """r1 and r2 of (3.57): backward Euler for theta = 1, trapezoidal for 1/2."""
    identity = np.eye(a.shape[0])
    return identity - theta * dt * a, identity + (1.0 - theta) * dt * a


def numerov_pair(a, dt, gamma):
    """r1 and r2 of the parametrized Numerov-type method (3.54) for u'' = A u."""
    identity = np.eye(a.shape[0])
    scaled = dt**2 * a
    square = scaled.dot(scaled)
    r1 = identity - scaled / 12.0 + 10.0 * gamma * square / 12.0
    r2 = -2.0 * identity - 10.0 * scaled / 12.0 - 20.0 * gamma * square / 12.0
    return r1, r2


def first_order_symbols(dt, theta):
    """The scalars r1, r2 of (3.57) for a single spatial eigenvalue."""

    def symbols(value):
        return 1.0 - theta * dt * value, 1.0 + (1.0 - theta) * dt * value

    return symbols


def numerov_symbols(dt, gamma):
    """The scalars r1, r2 of (3.54) for a single spatial eigenvalue."""

    def symbols(value):
        scaled = dt**2 * value
        r1 = 1.0 - scaled / 12.0 + 10.0 * gamma * scaled**2 / 12.0
        r2 = -2.0 - 10.0 * scaled / 12.0 - 20.0 * gamma * scaled**2 / 12.0
        return r1, r2

    return symbols


def time_stencils(second_order):
    """Offsets of the r1 and the r2 blocks in K.

    The first-order system of (3.62a) couples u_n to u_{n-1}; the symmetric
    two-step method (3.54) couples u_n to u_{n-1} and u_{n-2}.
    """
    if second_order:
        return (0, 2), (1,)
    return (0,), (1,)


def coupling_sign(second_order):
    """Sign of the r2 block: negative for u' = A u, positive for u'' = A u."""
    return 1.0 if second_order else -1.0


def all_at_once(iterate, r1, r2, second_order):
    """K U, applied block by block as the scripts do."""
    sign = coupling_sign(second_order)
    result = iterate.dot(r1.T)
    result[1:] += sign * iterate[:-1].dot(r2.T)
    if second_order:
        result[2:] += iterate[:-2].dot(r1.T)
    return result


def toeplitz_blocks(steps, alpha, second_order):
    """The two scalar Toeplitz matrices of K and their alpha-circulant versions."""
    diagonal, subdiagonal = time_stencils(second_order)
    first = sum(np.eye(steps, k=-shift) for shift in diagonal)
    second = sum(np.eye(steps, k=-shift) for shift in subdiagonal)
    circulant_first, circulant_second = first.copy(), second.copy()
    for shifts, target in ((diagonal, circulant_first), (subdiagonal, circulant_second)):
        for shift in shifts:
            if shift:
                rows = np.arange(shift)
                target[rows, steps - shift + rows] = alpha
    return first, second, circulant_first, circulant_second


def preconditioner_solver(r1, r2, steps, alpha, second_order, *, inverted=False):
    """The three steps of (3.60): a scaled FFT, N_t spatial solves, its inverse.

    The scripts solve each of the N_t complex blocks, which is what the stored LU
    factors reproduce bit for bit.  With ``inverted`` the blocks are inverted once
    and applied as matrices instead; that makes the roundoff a fixed perturbation
    of the preconditioner rather than a fresh one per right-hand side, and costs
    the direct implementation of Figure 3.18 about a decade of measured error.
    """
    scaling = alpha ** (np.arange(steps) / steps)
    diagonal, subdiagonal = time_stencils(second_order)
    first, second = np.zeros(steps), np.zeros(steps)
    first[list(diagonal)] = 1.0
    second[list(subdiagonal)] = 1.0
    symbol_first = np.fft.fft(scaling * first)
    symbol_second = np.fft.fft(scaling * second)
    sign = coupling_sign(second_order)
    blocks = [
        symbol_first[n] * r1 + sign * symbol_second[n] * r2 for n in range(steps)
    ]
    apply_block = (
        (lambda block, vector: block.dot(vector))
        if inverted
        else (lambda block, vector: lu_solve(block, vector))
    )
    blocks = [np.linalg.inv(b) if inverted else lu_factor(b) for b in blocks]

    def solve(rhs):
        transformed = np.fft.fft(scaling[:, None] * rhs, axis=0)
        for n, block in enumerate(blocks):
            transformed[n] = apply_block(block, transformed[n])
        return np.real(np.fft.ifft(transformed, axis=0) / scaling[:, None])

    return solve


def preconditioned_spectrum(a, symbols, steps, alpha, second_order):
    """Eigenvalues of P^-1 K, one spatial mode at a time.

    K and P share their spatial blocks, so diagonalizing A first turns one
    eigenvalue problem of size N_t N_x into N_x problems of size N_t, which is
    what makes the spectrum of a space-time matrix with 10^4 rows computable.
    """
    first, second, circulant_first, circulant_second = toeplitz_blocks(
        steps, alpha, second_order
    )
    sign = coupling_sign(second_order)
    spectrum = []
    for value in np.linalg.eigvals(a):
        r1, r2 = symbols(value)
        stiff = r1 * first + sign * r2 * second
        precondition = r1 * circulant_first + sign * r2 * circulant_second
        spectrum.append(np.linalg.eigvals(np.linalg.solve(precondition, stiff)))
    return np.concatenate(spectrum)


def iteration_spectrum(a, symbols, steps, alpha, second_order):
    """The nonzero eigenvalues of M = I - P^-1 K, computed without cancellation.

    P_alpha - K is the alpha-weighted corner of the circulant, of rank one for
    the one-step and rank two for the two-step method, so M has at most two
    nonzero eigenvalues per spatial mode and they follow from a 2 x 2 matrix.
    Computing them this way avoids the noise that swamps a direct eigenvalue
    computation, whose N_t - 2 unit eigenvalues are defective.
    """
    _, _, circulant_first, circulant_second = toeplitz_blocks(
        steps, alpha, second_order
    )
    sign = coupling_sign(second_order)
    columns = 2 if second_order else 1
    unit = np.eye(steps)[:, :columns]
    spectrum = []
    for value in np.linalg.eigvals(a):
        r1, r2 = symbols(value)
        precondition = r1 * circulant_first + sign * r2 * circulant_second
        solved = np.linalg.solve(precondition, unit)
        if second_order:
            small = alpha * np.array(
                [
                    [
                        r1 * solved[-2, 0] + r2 * solved[-1, 0],
                        r1 * solved[-2, 1] + r2 * solved[-1, 1],
                    ],
                    [r1 * solved[-1, 0], r1 * solved[-1, 1]],
                ]
            )
            spectrum.extend(np.linalg.eigvals(small))
        else:
            spectrum.append(alpha * sign * r2 * solved[-1, 0])
    return np.array(spectrum)


def wave_start(a, u0, dt):
    """u_1 from u'(0) = 0, the way the scripts start the two-step method."""
    return np.linalg.solve(np.eye(a.shape[0]) - 0.5 * dt**2 * a, u0)


def build_system(problem, dt, steps, *, second_order, gamma=GAMMA, theta=1.0):
    """r1, r2 and the right-hand side b of K U = b for one of the three PDEs."""
    a, _, u0 = problem
    rhs = np.zeros((steps, u0.size))
    if second_order:
        r1, r2 = numerov_pair(a, dt, gamma)
        u1 = wave_start(a, u0, dt)
        rhs[0] = -r2.dot(u1) - r1.dot(u0)
        rhs[1] = -r1.dot(u1)
    else:
        r1, r2 = first_order_pair(a, dt, theta)
        rhs[0] = r2.dot(u0)
    return r1, r2, rhs


def sequential_solution(r1, r2, rhs, second_order):
    """The exact solution of K U = b, by forward substitution in time."""
    steps = rhs.shape[0]
    sign = coupling_sign(second_order)
    solution = np.zeros_like(rhs)
    factor = np.linalg.inv(r1)
    for n in range(steps):
        right = rhs[n].copy()
        if n >= 1:
            right -= sign * r2.dot(solution[n - 1])
        if second_order and n >= 2:
            right -= r1.dot(solution[n - 2])
        solution[n] = factor.dot(right)
    return solution


def initial_guess(problem, steps):
    """The scripts' starting iterate, u_0/8 repeated over all time-steps."""
    return np.tile(INITIAL_SCALE * problem[2], (steps, 1))


def stationary_iteration(
    r1,
    r2,
    rhs,
    steps,
    alpha,
    second_order,
    *,
    iterations,
    start=None,
    mode="refinement",
    reference=None,
    tolerance=None,
    inverted=False,
):
    """The preconditioned stationary iteration (3.61), in either implementation.

    ``mode="refinement"`` applies the preconditioner to the residual and updates,
    which is the iterative refinement form; ``mode="direct"`` instead moves the
    alpha-coupled corner to the right-hand side and solves for U^k outright,
    which is the head-tail coupled waveform relaxation (3.58) and, as alpha
    shrinks, loses accuracy to roundoff.
    """
    solve = preconditioner_solver(r1, r2, steps, alpha, second_order, inverted=inverted)
    iterate = np.zeros_like(rhs) if start is None else start.copy()
    increments, errors = [], []
    for _ in range(iterations):
        if mode == "refinement":
            update = solve(rhs - all_at_once(iterate, r1, r2, second_order))
            iterate = iterate + update
            increments.append(float(np.abs(update).max()))
        else:
            coupling = np.zeros_like(rhs)
            if second_order:
                coupling[0] = alpha * (r1.dot(iterate[-2]) + r2.dot(iterate[-1]))
                coupling[1] = alpha * r1.dot(iterate[-1])
            else:
                coupling[0] = -alpha * r2.dot(iterate[-1])
            following = solve(rhs + coupling)
            increments.append(float(np.abs(following - iterate).max()))
            iterate = following
        if reference is not None:
            errors.append(float(np.abs(iterate - reference).max()))
        if tolerance is not None and increments[-1] <= tolerance:
            break
    return iterate, increments, errors


def gmres_residuals(r1, r2, rhs, steps, alpha, second_order, *, start, tolerance):
    """Preconditioned GMRES on K U = b, unrestarted as in the scripts."""
    shape = rhs.shape
    unknowns = rhs.size
    solve = preconditioner_solver(r1, r2, steps, alpha, second_order)
    operator = LinearOperator(
        (unknowns, unknowns),
        matvec=lambda v: all_at_once(v.reshape(shape), r1, r2, second_order).ravel(),
        dtype=float,
    )
    preconditioner = LinearOperator(
        (unknowns, unknowns),
        matvec=lambda v: solve(v.reshape(shape)).ravel(),
        dtype=float,
    )
    relative = []
    solution, info = gmres(
        operator,
        rhs.ravel(),
        x0=start.ravel(),
        M=preconditioner,
        rtol=tolerance,
        atol=0.0,
        restart=min(unknowns, 400),
        maxiter=1000,
        callback=lambda value: relative.append(float(value)),
        callback_type="pr_norm",
    )
    # MATLAB's resvec reports absolute preconditioned residual norms and starts
    # at the initial guess, so undo scipy's normalization and prepend it.
    scale = float(np.linalg.norm(solve(rhs)))
    residual = solve(rhs - all_at_once(start, r1, r2, second_order))
    history = [float(np.linalg.norm(residual))] + [value * scale for value in relative]
    exact = sequential_solution(r1, r2, rhs, second_order)
    error = float(np.abs(solution.reshape(shape) - exact).max())
    return history, error, int(info)


def _cases(
    dx, dt, *, periodic, gamma=GAMMA, theta=1.0, include_heat=True, nus=ADE_NU
):
    """The three representative PDEs of Section 3.5.2 with their time-integrators."""
    steps = int(round(FIRST_ORDER_T / dt))
    cases = []
    if include_heat:
        cases.append(
            {
                "name": "heat equation",
                "problem": first_order_problem(
                    1.0, dx, advection=False, periodic=periodic
                ),
                "second_order": False,
                "steps": steps,
                "symbols": first_order_symbols(dt, theta),
                "theta": theta,
                "dt": dt,
            }
        )
    for nu in nus:
        cases.append(
            {
                "name": rf"adv diff equation, $\nu=10^{{{int(round(np.log10(nu)))}}}$",
                "problem": first_order_problem(nu, dx, advection=True, periodic=periodic),
                "second_order": False,
                "steps": steps,
                "symbols": first_order_symbols(dt, theta),
                "theta": theta,
                "dt": dt,
            }
        )
    cases.append(
        {
            "name": "wave equation",
            "problem": wave_problem(dx, periodic=periodic),
            "second_order": True,
            "steps": steps - 1,
            "symbols": numerov_symbols(dt, gamma),
            "theta": None,
            "dt": dt,
        }
    )
    return cases


def run_figure_3_15(output_dir, quick=False):
    """Figure 3.15: spectra of P^-1 K and the GMRES convergence they imply."""
    dx = 1.0 / 40 if quick else FIRST_ORDER_DX
    dt = 1.0 / 20 if quick else FIRST_ORDER_DT
    cases = _cases(dx, dt, periodic=False, theta=1.0)
    rows = ((0,), (1, 2), (3,))
    titles = ("Heat equation", "Advection diffusion equation", "Wave equation")
    colours = ("#17324D", "#D1495B")
    markers = ("+", "o")

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.4), constrained_layout=True)
    panels = []
    for row, (group, title) in enumerate(zip(rows, titles)):
        for offset, index in enumerate(group):
            case = cases[index]
            colour, marker = colours[offset], markers[offset]
            a = case["problem"][0]
            spectrum = preconditioned_spectrum(
                a, case["symbols"], case["steps"], 1.0, case["second_order"]
            )
            analytic = iteration_spectrum(
                a, case["symbols"], case["steps"], 1.0, case["second_order"]
            )
            r1, r2, rhs = build_system(
                case["problem"],
                case["dt"],
                case["steps"],
                second_order=case["second_order"],
                theta=1.0,
            )
            history, error, info = gmres_residuals(
                r1,
                r2,
                rhs,
                case["steps"],
                1.0,
                case["second_order"],
                start=initial_guess(case["problem"], case["steps"]),
                tolerance=(
                    WAVE_GMRES_TOLERANCE if case["second_order"] else GMRES_TOLERANCE
                ),
            )
            axes[row, 0].plot(
                spectrum.real,
                spectrum.imag,
                marker,
                color=colour,
                markersize=5,
                markerfacecolor="none",
                label=case["name"],
                # One marker per spatial mode and time level: as vectors these
                # clouds make the file tens of megabytes.
                rasterized=True,
            )
            axes[row, 1].semilogy(
                np.arange(len(history)),
                history,
                "-.",
                color=colour,
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                linewidth=1.1,
                label=case["name"],
            )
            if len(group) > 1 and offset == 0:
                # The clustered spectrum of the mildly advective case is only
                # visible on its own scale, so the paper insets it.
                inset = axes[row, 0].inset_axes([0.3, 0.58, 0.26, 0.34])
                inset.plot(
                    spectrum.real,
                    spectrum.imag,
                    marker,
                    color=colour,
                    markersize=3,
                    markerfacecolor="none",
                    rasterized=True,
                )
                inset.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
                inset.tick_params(labelsize=6)
                inset.set_title("zoom in", fontsize=7)
            panels.append(
                {
                    "problem": case["name"],
                    "steps": case["steps"],
                    "gmres_iterations": len(history) - 1,
                    "gmres_residuals": history,
                    "solution_error": error,
                    "gmres_info": info,
                    "spectrum_unit_distance_max": float(np.abs(spectrum - 1.0).max()),
                    "spectrum_real_range": [
                        float(spectrum.real.min()),
                        float(spectrum.real.max()),
                    ],
                    "spectrum_imag_range": [
                        float(spectrum.imag.min()),
                        float(spectrum.imag.max()),
                    ],
                    "analytic_unit_distance_max": float(np.abs(analytic).max()),
                }
            )
        axes[row, 0].set_title(title)
        axes[row, 0].set_xlabel(r"Re$(\lambda(\mathcal{P}^{-1}\mathcal{K}))$")
        axes[row, 0].set_ylabel(r"Im$(\lambda(\mathcal{P}^{-1}\mathcal{K}))$")
        axes[row, 1].set_title(title)
        axes[row, 1].set_xlabel("iteration index")
        axes[row, 1].set_ylabel("GMRES residual")
        axes[row, 1].set_ylim(1e-14, None)
        for column in (0, 1):
            style_axis(axes[row, column])
            if len(group) > 1:
                axes[row, column].legend(fontsize=8)

    artifacts = save_figure(fig, f"{output_dir}/figure_3_15_paradiag_ii_spectra")
    return finish(
        output_dir,
        "figure_3_15_paradiag_ii_spectra",
        "Spectra of the preconditioned matrix P^-1 K (a, c, e) and the measured "
        "convergence of preconditioned GMRES (b, d, f) for the three "
        "representative PDEs, with alpha = 1",
        {
            "parameters": {
                "dx": dx,
                "dt": dt,
                "t_final": FIRST_ORDER_T,
                "gamma": GAMMA,
                "alpha": 1.0,
                "time_integrator": "backward Euler / Numerov",
                "boundary_condition": "Dirichlet",
                "initial_guess": "u_0 / 8 repeated",
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_3_16(output_dir, quick=False):
    """Figure 3.16: head-tail coupled waveform relaxation for two values of alpha."""
    dx = 1.0 / 40 if quick else FIRST_ORDER_DX
    dt = 1.0 / 20 if quick else FIRST_ORDER_DT
    iterations = 6 if quick else FIGURE_3_16_ITERATIONS
    cases = _cases(
        dx, dt, periodic=True, theta=1.0, include_heat=False, nus=FIGURE_3_16_NU
    )
    colours = ("#D1495B", "#17324D", "#2F6BFF")
    markers = ("o", "s", "+")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    panels = []
    for ax, alpha in zip(axes, FIGURE_3_16_ALPHAS):
        records = []
        for case, colour, marker in zip(cases, colours, markers):
            r1, r2, rhs = build_system(
                case["problem"],
                case["dt"],
                case["steps"],
                second_order=case["second_order"],
                theta=1.0,
            )
            reference = sequential_solution(r1, r2, rhs, case["second_order"])
            start = initial_guess(case["problem"], case["steps"])
            _, increments, errors = stationary_iteration(
                r1,
                r2,
                rhs,
                case["steps"],
                alpha,
                case["second_order"],
                iterations=iterations,
                start=start,
                mode="refinement",
                reference=reference,
                tolerance=FIGURE_3_16_TOLERANCE,
            )
            # The script plots the increment norms, which is why its first value
            # depends on alpha even though the starting iterate does not.
            history = increments
            analytic = iteration_spectrum(
                case["problem"][0],
                case["symbols"],
                case["steps"],
                alpha,
                case["second_order"],
            )
            ax.semilogy(
                np.arange(len(history)),
                history,
                "-.",
                color=colour,
                marker=marker,
                markersize=6,
                markerfacecolor="none",
                linewidth=1.1,
                label=case["name"],
            )
            rate = np.exp(np.diff(np.log(np.maximum(history, 1e-300)))[:3].mean())
            records.append(
                {
                    "problem": case["name"],
                    "increments": increments,
                    "errors": errors,
                    "measured_rate": float(rate),
                    "spectral_radius": float(np.abs(analytic).max()),
                }
            )
        ax.set_title(rf"$\alpha={alpha:g}$")
        ax.set_xlabel("iteration index")
        ax.set_ylabel(r"Error $\max_n\|{\bf u}^k_n-{\bf u}_n\|$")
        ax.set_xlim(0, FIGURE_3_16_ITERATIONS - 1)
        ax.set_ylim(1e-12, 10.0)
        style_axis(ax)
        ax.legend(fontsize=8)
        panels.append({"alpha": alpha, "runs": records})

    artifacts = save_figure(fig, f"{output_dir}/figure_3_16_head_tail")
    return finish(
        output_dir,
        "figure_3_16_head_tail",
        "Error against iteration for the head-tail coupled waveform relaxation "
        "method (3.55) with two values of alpha and periodic boundary conditions",
        {
            "parameters": {
                "dx": dx,
                "dt": dt,
                "t_final": FIRST_ORDER_T,
                "boundary_condition": "periodic",
                "gamma": GAMMA,
                "nu": list(FIGURE_3_16_NU),
                "time_integrator": "backward Euler / Numerov",
                "tolerance": FIGURE_3_16_TOLERANCE,
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_3_17(output_dir, quick=False):
    """Figure 3.17: eigenvalues of the iteration matrix M for the wave equation."""
    dx = 1.0 / 32 if quick else WAVE_DX
    horizons = WAVE_HORIZONS[:2] if quick else WAVE_HORIZONS
    a = wave_problem(dx)[0]
    radius = WAVE_ALPHA / (1.0 - WAVE_ALPHA)
    angle = np.linspace(0.0, 2.0 * np.pi, 400)

    fig, axes = plt.subplots(
        2, len(horizons), figsize=(3.9 * len(horizons), 7.8), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    panels = []
    for row, gamma in enumerate(WAVE_GAMMAS):
        for column, horizon in enumerate(horizons):
            steps = int(round(horizon / WAVE_DT)) - 1
            symbols = numerov_symbols(WAVE_DT, gamma)
            spectrum = preconditioned_spectrum(a, symbols, steps, WAVE_ALPHA, True)
            analytic = iteration_spectrum(a, symbols, steps, WAVE_ALPHA, True)
            ax = axes[row, column]
            # The script plots Re(lambda(P^-1 K) - 1) against Im(lambda(P^-1 K)),
            # which is the spectrum of M reflected through the origin.
            ax.plot(
                spectrum.real - 1.0,
                spectrum.imag,
                "+",
                color="#17324D",
                markersize=6,
                # One marker per spatial mode and time level, up to 319 time
                # levels in the last column.
                rasterized=True,
            )
            ax.plot(
                radius * np.cos(angle),
                radius * np.sin(angle),
                "--",
                color="#D1495B",
                linewidth=1.3,
            )
            ax.set_xlim(-3.85 * radius, 1.2 * radius)
            ax.set_ylim(-2.0 * radius, 2.0 * radius)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(rf"$T={horizon:g}$")
            ax.set_xlabel(r"Re$(\lambda(\mathcal{M}))$")
            ax.set_ylabel(r"Im$(\lambda(\mathcal{M}))$")
            style_axis(ax)
            panels.append(
                {
                    "gamma": gamma,
                    "t_final": horizon,
                    "steps": steps,
                    "bound": radius,
                    "spectral_radius": float(np.abs(analytic).max()),
                    "inside_fraction": float(
                        np.mean(np.abs(analytic) <= radius * (1.0 + 1e-8))
                    ),
                    "outside_count": int(
                        np.sum(np.abs(analytic) > radius * (1.0 + 1e-8))
                    ),
                    "most_negative_real": float((spectrum.real - 1.0).min()),
                }
            )

    artifacts = save_figure(fig, f"{output_dir}/figure_3_17_iteration_matrix")
    return finish(
        output_dir,
        "figure_3_17_iteration_matrix",
        "Eigenvalues of the iteration matrix M of (3.67) for the wave equation "
        "with gamma = 1/120 (a-c) and gamma = 1/120.01 (d-f); the dashed line is "
        "the circle of radius alpha/(1 - alpha)",
        {
            "parameters": {
                "dx": dx,
                "dt": WAVE_DT,
                "alpha": WAVE_ALPHA,
                "gammas": list(WAVE_GAMMAS),
                "horizons": list(horizons),
                "boundary_condition": "Dirichlet",
            },
            "panels": panels,
        },
        artifacts,
    )


def run_figure_3_18(output_dir, quick=False):
    """Figure 3.18: the direct and the refinement implementation as alpha shrinks."""
    dx = 1.0 / 32 if quick else WAVE_DX
    iterations = 5 if quick else ROUNDOFF_ITERATIONS
    problem = wave_problem(dx)
    steps = int(round(ROUNDOFF_T / WAVE_DT)) - 1
    r1, r2, rhs = build_system(
        problem, WAVE_DT, steps, second_order=True, gamma=ROUNDOFF_GAMMA
    )
    start = np.random.default_rng(RANDOM_SEED).uniform(
        -1.0, 1.0, size=(steps, problem[1].size)
    )
    colours = ("#D1495B", "#17324D", "#2F6BFF")
    markers = ("+", "*", "o")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    panels = []
    for ax, mode, title in zip(
        axes, ("direct", "refinement"), ("Direct", "IR framework")
    ):
        records = []
        for alpha, colour, marker in zip(ROUNDOFF_ALPHAS, colours, markers):
            _, increments, _ = stationary_iteration(
                r1,
                r2,
                rhs,
                steps,
                alpha,
                True,
                iterations=iterations,
                start=start,
                mode=mode,
            )
            ax.semilogy(
                np.arange(len(increments)),
                increments,
                "-.",
                color=colour,
                marker=marker,
                markersize=7,
                markerfacecolor="none",
                linewidth=1.1,
                label=rf"$\alpha=10^{{{int(round(np.log10(alpha)))}}}$",
            )
            _, inverted, _ = stationary_iteration(
                r1,
                r2,
                rhs,
                steps,
                alpha,
                True,
                iterations=iterations,
                start=start,
                mode=mode,
                inverted=True,
            )
            records.append(
                {
                    "alpha": alpha,
                    "errors": increments,
                    "stagnation_level": float(min(increments[-3:])),
                    "errors_inverted_blocks": inverted,
                    "stagnation_level_inverted_blocks": float(min(inverted[-3:])),
                }
            )
        ax.set_title(title)
        ax.set_xlabel("iteration index")
        ax.set_ylabel("Measured Error")
        ax.set_ylim(1e-16, 10.0)
        ax.set_xticks(np.arange(iterations))
        style_axis(ax)
        ax.legend(fontsize=8)
        panels.append({"mode": mode, "runs": records})

    artifacts = save_figure(fig, f"{output_dir}/figure_3_18_roundoff")
    return finish(
        output_dir,
        "figure_3_18_roundoff",
        "Measured error of ParaDiag II for the wave equation (2.7) implemented in "
        "the direct and in the iterative refinement mode, for three values of alpha",
        {
            "parameters": {
                "dx": dx,
                "dt": WAVE_DT,
                "t_final": ROUNDOFF_T,
                "steps": steps,
                "gamma": ROUNDOFF_GAMMA,
                "alphas": list(ROUNDOFF_ALPHAS),
                "seed": RANDOM_SEED,
            },
            "panels": panels,
        },
        artifacts,
    )
