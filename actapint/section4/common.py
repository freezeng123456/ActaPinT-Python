"""Shared pieces of the Section 4 reproduction.

Section 4 runs every method on the same two-level time grid: a coarse grid of
``N_t`` intervals of length ``DT``, each subdivided into ``J`` fine steps of
length ``dt = DT / J`` (Figure 4.1).  The coarse propagator G is one backward
Euler step across ``DT``, and the fine propagator F is ``J`` steps of a
higher-order scheme, so the whole chapter is parametrized by the pair
(integrator of F, coarsening factor J).

The scripts of this chapter share one convention worth stating once: they write
the semi-discrete problem as ``u' + A u = 0``, so ``A`` is the *positive*
operator ``advection + nu * (-Laplacian)``, and a negative ``nu`` selects the
heat equation with diffusion ``|nu|``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import solve_banded
from scipy.sparse.linalg import splu

from ..operators import (
    dirichlet_first_derivative,
    dirichlet_negative_laplacian,
    periodic_first_derivative,
    periodic_negative_laplacian,
)

# Butcher coefficients of the integrators offered by the scripts.  SDIRK22 and
# SDIRK23 are the two tableaux printed in (4.9).
SDIRK22_GAMMA = (2.0 - np.sqrt(2.0)) / 2.0
SDIRK23_GAMMA = (3.0 + np.sqrt(3.0)) / 6.0
SDIRK33_GAMMA = 0.4358665215


def linear_operator(dx, nu, *, boundary_condition="periodic", advection=1.0):
    """Return A of ``u' + A u = 0`` and the grid the scripts build with it.

    ``nu < 0`` selects the heat equation ``u_t = |nu| u_xx``, which is how every
    script of this chapter switches between the two linear model problems.  The
    grid is the scripts' ``linspace(0, 1, Nx)`` with ``Nx = 1/dx``: it carries
    both endpoints, so its true spacing is ``1/(Nx - 1)`` while the operator is
    assembled with ``dx``.  The inconsistency is upstream and only shifts the
    initial condition by half a cell, but it is kept so that the reproduced
    numbers can be compared with the scripts directly.
    """
    n = int(round(1.0 / dx))
    if boundary_condition == "periodic":
        first = periodic_first_derivative(n, dx, advection)
        second = periodic_negative_laplacian(n, dx, 1.0)
    elif boundary_condition == "dirichlet":
        first = dirichlet_first_derivative(n, dx, advection)
        second = dirichlet_negative_laplacian(n, dx, 1.0)
    else:
        raise ValueError(f"unsupported boundary condition {boundary_condition!r}")
    operator = abs(nu) * second if nu < 0 else first + nu * second
    return operator.tocsr(), np.linspace(0.0, 1.0, n)


# The Burgers scripts divide the centred difference of the flux by ``4 dx``,
# which is the ``0.5 (u**2)_x`` of (2.6), but their published figures are only
# reproduced with ``2 dx``, that is with the flux ``(u**2)_x``.  Section 2 hit the
# same factor of two in `Solution_Bergers.m`; see docs/SECTION2_REPRODUCTION.md.
BURGERS_FLUX_SCALE = 1.0
BURGERS_FLUX_SCALE_AS_COMMITTED = 0.5


def burgers_operators(
    dx, nu, *, boundary_condition="dirichlet", flux_scale=BURGERS_FLUX_SCALE
):
    """Return A1, A2 of ``u' + A2 u + A1 u**2 = 0`` and the grid."""
    n = int(round(1.0 / dx))
    if boundary_condition == "periodic":
        a1 = periodic_first_derivative(n, dx, flux_scale)
        a2 = periodic_negative_laplacian(n, dx, nu)
    elif boundary_condition == "dirichlet":
        a1 = dirichlet_first_derivative(n, dx, flux_scale)
        a2 = dirichlet_negative_laplacian(n, dx, nu)
    else:
        raise ValueError(f"unsupported boundary condition {boundary_condition!r}")
    return a1.tocsr(), a2.tocsr(), np.linspace(0.0, 1.0, n)


def stability_function(method, z, *, substeps=1):
    """Amplification factor of one coarse step, as the scripts compute it.

    ``z`` is ``DT * lambda``; the fine integrators are applied to ``z/substeps``
    and raised to the power ``substeps``, so the result is comparable with the
    coarse factor at the same eigenvalue.
    """
    z = np.asarray(z, dtype=complex)
    if method == "ie":
        single = 1.0 / (1.0 + z / substeps)
    elif method == "tr":
        w = z / substeps
        single = (1.0 - 0.5 * w) / (1.0 + 0.5 * w)
    elif method == "sdirk22":
        w = z / substeps
        gamma = SDIRK22_GAMMA
        single = (1.0 - (1.0 - 2.0 * gamma) * w) / (1.0 + gamma * w) ** 2
    elif method == "sdirk23":
        w = z / substeps
        gamma = SDIRK23_GAMMA
        single = (np.sqrt(3.0) + w - gamma * w**2) / (
            np.sqrt(3.0) * (1.0 + gamma * w) ** 2
        )
    elif method == "sdirk33":
        w = z / substeps
        gamma = SDIRK33_GAMMA
        tau = 0.5 * (1.0 + gamma)
        b1 = -0.25 * (6.0 * gamma**2 - 16.0 * gamma + 1.0)
        b2 = 0.25 * (6.0 * gamma**2 - 20.0 * gamma + 5.0)
        matrix = np.array([[gamma, 0.0, 0.0], [tau - gamma, gamma, 0.0], [b1, b2, gamma]])
        weights = np.array([b1, b2, gamma])
        single = np.array(
            [
                1.0
                - value
                * weights
                @ np.linalg.solve(np.eye(3) + value * matrix, np.ones(3))
                for value in w
            ]
        )
    else:
        raise ValueError(f"unknown integrator: {method}")
    return single**substeps


def convergence_factors(operator, coarse_dt, substeps, method):
    """The two factors of Theorem 4.2, evaluated at every eigenvalue of DT A.

    ``rho_l`` is (4.5b), the factor that governs long time intervals, and the
    returned ``rho_s_base`` is the ``|R_g - R_f|`` of (4.5a), which the
    superlinear bound multiplies by the falling factorial of the step count.
    """
    eigenvalues = np.linalg.eigvals(coarse_dt * operator.toarray())
    coarse = stability_function("ie", eigenvalues)
    fine = stability_function(method, eigenvalues, substeps=substeps)
    difference = np.abs(coarse - fine)
    slack = 1.0 - np.abs(coarse)
    rho_l = np.full(difference.shape, np.inf)
    # The constant mode of a periodic advection operator has R_g = R_f = 1, so
    # it neither contracts nor contributes error; the scripts divide by zero
    # there and get NaN, which their max() then propagates.
    resolved = slack > 1e-12
    rho_l[resolved] = difference[resolved] / slack[resolved]
    rho_l[(~resolved) & (difference <= 1e-10)] = 0.0
    return {
        "eigenvalues": eigenvalues / coarse_dt,
        "rho_l": rho_l,
        "rho_l_max": float(np.max(rho_l)),
        "rho_s_base": float(np.max(difference)),
    }


def superlinear_bound(rho_s_base, steps, iterations):
    """The bound (4.5a): ``|R_g - R_f|**k / k! * prod_{j=1..k} (N_t - j)``.

    Evaluated through logarithms, since the falling factorial overflows well
    before the largest step count used in the chapter, and returned relative to
    the initial error so that it can be scaled by it.
    """
    values = np.ones(iterations + 1)
    logarithm = 0.0
    for k in range(1, iterations + 1):
        if steps - k <= 0:
            values[k:] = 0.0
            break
        logarithm += np.log(rho_s_base) + np.log(steps - k) - np.log(k)
        values[k] = np.exp(logarithm)
    return values


def linear_propagator(operator, dt, method, *, substeps=1):
    """Dense propagation matrix of ``substeps`` steps of ``method``.

    The scripts apply their integrators as matrices, so the propagator is built
    once and reused; this is also what makes the Parareal sweeps cheap enough to
    run every experiment of the chapter on one core.
    """
    n = operator.shape[0]
    identity = np.eye(n)
    eye = sparse.eye(n, format="csc")
    matrix = sparse.csc_matrix(operator)
    if method == "ie":
        step = splu(eye + dt * matrix).solve(identity)
    elif method == "tr":
        step = splu(eye + 0.5 * dt * matrix).solve((eye - 0.5 * dt * matrix).toarray())
    elif method == "sdirk22":
        gamma = SDIRK22_GAMMA
        solve = splu(eye + gamma * dt * matrix)
        stage = solve.solve(identity)
        step = solve.solve(identity - (1.0 - gamma) * dt * (matrix @ stage))
    elif method == "sdirk23":
        gamma = SDIRK23_GAMMA
        solve = splu(eye + gamma * dt * matrix)
        stage1 = solve.solve(identity)
        stage2 = solve.solve(identity - (1.0 - 2.0 * gamma) * dt * (matrix @ stage1))
        step = identity - 0.5 * dt * (matrix @ (stage1 + stage2))
    elif method == "sdirk33":
        gamma = SDIRK33_GAMMA
        tau = 0.5 * (1.0 + gamma)
        b1 = -0.25 * (6.0 * gamma**2 - 16.0 * gamma + 1.0)
        b2 = 0.25 * (6.0 * gamma**2 - 20.0 * gamma + 5.0)
        solve = splu(eye + gamma * dt * matrix)
        stage1 = solve.solve(identity)
        stage2 = solve.solve(identity - (tau - gamma) * dt * (matrix @ stage1))
        step = solve.solve(
            identity - b1 * dt * (matrix @ stage1) - b2 * dt * (matrix @ stage2)
        )
    else:
        raise ValueError(f"unknown integrator: {method}")
    return np.linalg.matrix_power(step, substeps) if substeps > 1 else step


# `Pro_IE` caps Newton at 1000 steps in the Parareal scripts and at 50 in
# `MGRiT_Burgers.m`; the cap is visible in the figures, because a run that leaves
# the region of convergence stops at whatever the last iterate was.
NEWTON_MAX = 1000
NEWTON_MAX_MGRIT = 50


def _tridiagonal_bands(matrix):
    """The three bands of ``matrix``, or None when it has periodic corners."""
    matrix = sparse.csr_matrix(matrix)
    n = matrix.shape[0]
    if n < 3 or abs(matrix[0, n - 1]) > 0.0 or abs(matrix[n - 1, 0]) > 0.0:
        return None
    return np.array(
        [
            np.append(0.0, matrix.diagonal(1)),
            matrix.diagonal(0),
            np.append(matrix.diagonal(-1), 0.0),
        ]
    )


class BurgersStepper:
    """Implicit Euler steps of ``u' + a2 u + a1 u**2 = 0`` by Newton's method.

    This mirrors the scripts' ``Pro_IE``: a bounded number of Newton steps, and
    no error when they are not enough.  That matters for the divergent runs of
    this chapter, where MATLAB simply keeps whatever the last iterate was and
    lets the plotted error grow, so a port that raised instead would stop early.

    With homogeneous Dirichlet ends the Jacobian is tridiagonal, and solving it
    as a banded system rather than a dense one is what makes the nonlinear
    experiments of this chapter run in minutes instead of hours.  The bands are
    extracted once here because the Newton solve sits in the innermost loop.
    """

    def __init__(self, a1, a2, tolerance, newton_max=NEWTON_MAX):
        self.a1 = a1
        self.a2 = a2
        self.tolerance = tolerance
        self.newton_max = newton_max
        self.bands1 = _tridiagonal_bands(a1)
        self.bands2 = _tridiagonal_bands(a2)
        self.banded = self.bands1 is not None and self.bands2 is not None
        self.identity = None if self.banded else np.eye(a1.shape[0])

    def flux(self, u):
        return self.a2 @ u + self.a1 @ u**2

    def __call__(self, tau, y, guess=None):
        """Solve ``z + tau (a2 z + a1 z**2) = y`` for ``z``.

        ``guess`` is the Newton starting point, ``y`` itself by default because
        for a time step ``y`` is the previous solution.  The STMG smoother instead
        solves for a *correction* whose right-hand side is a residual, and there
        the scripts start from zero; starting from ``y`` diverges immediately.
        """
        y = np.asarray(y, dtype=float)
        z = y.copy() if guess is None else np.array(guess, dtype=float)
        best = np.inf
        for _ in range(self.newton_max):
            residual = z + tau * self.flux(z) - y
            if not np.all(np.isfinite(residual)):
                break
            norm = float(np.abs(residual).max())
            if norm <= self.tolerance:
                break
            # The scripts ask for a residual below ``Tol/3``, which for the fine
            # steps of this chapter sits below the round-off level of the
            # residual itself, so Newton stagnates and MATLAB spends its whole
            # iteration budget.  Stopping at stagnation returns the same iterate
            # to round-off and cuts the nonlinear runs by an order of magnitude.
            if norm >= best:
                break
            best = norm
            if self.banded:
                jacobian = tau * (
                    self.bands2 + 2.0 * self.bands1 * _shifted_diagonals(z)
                )
                jacobian[1] += 1.0
                z = z - solve_banded((1, 1), jacobian, residual)
            else:
                # ``a1`` reaches this branch dense for the Parareal experiments and
                # sparse for the periodic ones, and only sparse matrices need the
                # explicit elementwise product.
                scaled = (
                    self.a1.multiply(z[None, :])
                    if sparse.issparse(self.a1)
                    else self.a1 * z[None, :]
                )
                jacobian = self.identity + tau * (self.a2 + 2.0 * scaled)
                z = z - np.linalg.solve(jacobian, residual)
        return z


def burgers_implicit_step(a1, a2, tau, y, tolerance, newton_max=NEWTON_MAX):
    """One implicit Euler step, for callers that only need a single step."""
    return BurgersStepper(a1, a2, tolerance, newton_max)(tau, y)


def _shifted_diagonals(z):
    """``diag(z)`` in banded layout: the columns each band multiplies."""
    return np.array([np.append(0.0, z[1:]), z, np.append(z[:-1], 0.0)])


def burgers_propagator(
    a1, a2, dt, substeps, tolerance, method="sdirk22", newton_max=NEWTON_MAX
):
    """Return F: ``substeps`` steps of ``method`` for Burgers' equation."""
    step = BurgersStepper(a1, a2, tolerance, newton_max)
    flux = step.flux

    def propagate(y):
        z = np.asarray(y, dtype=float)
        for _ in range(substeps):
            if method == "ie":
                z = step(dt, z)
            elif method == "tr":
                z = step(0.5 * dt, z - 0.5 * dt * flux(z))
            elif method == "sdirk22":
                gamma = SDIRK22_GAMMA
                stage = step(gamma * dt, z)
                z = step(gamma * dt, z - (1.0 - gamma) * dt * flux(stage))
            elif method == "sdirk23":
                gamma = SDIRK23_GAMMA
                stage1 = step(gamma * dt, z)
                stage2 = step(gamma * dt, z - (1.0 - 2.0 * gamma) * dt * flux(stage1))
                z = z - 0.5 * dt * (flux(stage1) + flux(stage2))
            else:
                raise ValueError(f"unknown integrator: {method}")
        return z

    return propagate


def sequential_solution(propagator, u0, steps):
    """The fine solution the scripts measure every iteration against.

    ``propagator`` may be a matrix or a callable, so that the linear and the
    nonlinear experiments share this routine.
    """
    advance = propagator if callable(propagator) else propagator.__matmul__
    trajectory = np.zeros((steps + 1, u0.size))
    trajectory[0] = u0
    for n in range(steps):
        trajectory[n + 1] = advance(trajectory[n])
    return trajectory


def constant_initial_iterate(u0, steps):
    """The committed scripts' starting iterate: ``u0`` at every level.

    They draw a uniform random field first and then overwrite every column with
    ``u0``, so the random draw has no effect on the result.  This choice costs
    one iteration, because the stored coarse values then do not belong to the
    starting iterate, and it inflates the initial error; the published figures
    use ``coarse_initial_iterate`` instead.
    """
    return np.tile(u0, (steps + 1, 1))


def matlab_random_iterate(u0, steps, low=0.0, high=1.0):
    """``random('unif', low, high, Nx, Nt+1)`` after ``rng('default')``.

    MGRiT and STMG start from a random iterate rather than from the initial
    condition, so the published error curves depend on the draw.  MATLAB's
    default generator is MT19937 seeded with 5489, which NumPy's legacy
    ``RandomState`` reproduces bit for bit, and MATLAB fills the array column by
    column; with both conventions matched the reproduced curves are the ones the
    scripts plot rather than a statistically similar sample.
    """
    stream = np.random.RandomState(5489)
    draw = stream.random_sample(u0.size * (steps + 1))
    values = low + (high - low) * draw.reshape((u0.size, steps + 1), order="F")
    values[:, 0] = u0
    return values.T


def coarse_initial_iterate(coarse, u0, steps):
    """The sequential coarse solution, which is the standard Parareal start.

    The scripts carry it as the commented-out alternative
    ``U_Ini(:,n+1)=Pro_IE(INV_AC,U_Ini(:,n))``.  It makes the stored coarse
    values consistent with the starting iterate, so the first sweep already
    contracts and the published error curves begin at the coarse-solver error.
    """
    trajectory = np.zeros((steps + 1, u0.size))
    trajectory[0] = u0
    for n in range(steps):
        trajectory[n + 1] = coarse(trajectory[n])
    return trajectory


def parareal(fine, coarse, reference, start, *, iterations, tolerance=0.0):
    """The Parareal iteration (4.1), written exactly as the scripts sweep it.

    One subtlety is carried over: the coarse values are updated in place inside
    the sequential sweep, so ``G`` is applied to the *new* iterate while the
    subtracted coarse term still belongs to the previous one.
    """
    current = np.array(start, dtype=float)
    stored_coarse = current.copy()
    errors = [float(np.abs(current - reference).max())]
    steps = reference.shape[0] - 1
    for _ in range(iterations):
        fine_values = np.array([fine(current[n]) for n in range(steps)])
        updated = current.copy()
        for n in range(steps):
            advanced = coarse(updated[n])
            updated[n + 1] = advanced + fine_values[n] - stored_coarse[n + 1]
            stored_coarse[n + 1] = advanced
        current = updated
        errors.append(float(np.abs(current - reference).max()))
        if tolerance and errors[-1] <= tolerance:
            break
    return current, errors


def mgrit_fcf(fine, coarse, reference, start, *, iterations, tolerance=0.0):
    """MGRiT with FCF-relaxation (4.12), as `MGRiT_Heat_ADE.m` codes it.

    Two fine solves per interval replace the one of Parareal, which is why the
    comparison in Figures 4.10 and 4.11 plots one MGRiT iteration against two
    Parareal iterations.  The script initializes the first coarse point with an
    exact fine solve and measures the error only over the interior levels, both
    of which are kept here.
    """
    steps = reference.shape[0] - 1
    current = np.array(start, dtype=float)
    current[1] = fine(reference[0])
    # The script measures the starting error over all levels and every later one
    # over the interior levels, which are the only ones the sweep updates.
    errors = [float(np.abs(current - reference).max())]
    for _ in range(iterations):
        updated = current.copy()
        two_fine = np.zeros_like(current)
        coarse_of_fine = np.zeros_like(current)
        for n in range(1, steps - 1):
            middle = fine(current[n - 1])
            two_fine[n + 1] = fine(middle)
            coarse_of_fine[n + 1] = coarse(middle)
        for n in range(1, steps - 1):
            updated[n + 1] = coarse(updated[n]) + two_fine[n + 1] - coarse_of_fine[n + 1]
        current = updated
        errors.append(float(np.abs(current[2:steps] - reference[2:steps]).max()))
        if tolerance and errors[-1] <= tolerance:
            break
    return current, errors


def alpha_circulant_factors(steps, alpha):
    """The scaling and the eigenvalues of the alpha-circulant of (4.16).

    ``C_alpha`` is the lower bidiagonal difference matrix of the coarse grid with
    ``-alpha`` in its top right corner, so that
    ``Gamma_alpha C_alpha Gamma_alpha^{-1}`` is a circulant and one Fourier
    transform diagonalizes it.
    """
    scale = alpha ** (np.arange(steps) / steps)
    stencil = np.zeros(steps)
    stencil[0] = 1.0
    if steps > 1:
        stencil[1] = -1.0
    return scale, np.fft.fft(scale * stencil)


def alpha_circulant_solve(operator, coarse_dt, alpha, rhs):
    """Solve ``(C_alpha kron I + I kron DT A) x = rhs``, the CGC (4.16).

    This is the three-step procedure (4.17): scale and transform along the coarse
    grid, solve one shifted system per frequency -- these are the ``N_t``
    independent solves that make the coarse grid correction parallel -- and
    transform back.  ``rhs`` and the result are indexed by coarse level first, as
    everywhere in this section.
    """
    steps, size = rhs.shape
    scale, eigenvalues = alpha_circulant_factors(steps, alpha)
    transformed = np.fft.fft(scale[:, None] * rhs, axis=0)
    solved = np.empty_like(transformed)
    is_sparse = sparse.issparse(operator)
    eye = sparse.eye(size, format="csc") if is_sparse else np.eye(size)
    for n in range(steps):
        shifted = eigenvalues[n] * eye + coarse_dt * operator
        if is_sparse:
            solved[n] = splu(shifted.tocsc()).solve(transformed[n])
        else:
            solved[n] = np.linalg.solve(shifted, transformed[n])
    return (np.fft.ifft(solved, axis=0) / scale[:, None]).real


def truncation_error(dx, dt):
    """The line ``max(dx**2, dt**2)`` that Figure 4.10 draws as a stopping level."""
    return max(dx**2, dt**2)


def write_json(output_dir, name, payload):
    path = Path(output_dir) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def finish(output_dir, name, caption, payload, artifacts):
    """Write one figure's metrics next to its rendered panels."""
    body = {"figure": caption, "artifacts": artifacts}
    body.update(payload)
    return write_json(output_dir, name, body)
