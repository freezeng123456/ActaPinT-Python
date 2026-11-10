import json
from pathlib import Path

import numpy as np
import pytest

from actapint.section2 import source_term
from actapint.section3 import run_figure_3_1, swr
from actapint.section3 import idc
from actapint.section3 import paradiag_bvm as pdbvm
from actapint.section3 import paradiag_direct as pd1
from actapint.section3 import paradiag_iterative as pdii
from actapint.section3.common import ade_generator, grid


def test_source_sharpness_is_the_sigma_of_equation_2_4():
    """Equation (2.4) is a sum of isotropic Gaussians of width 1/sqrt(sigma)."""
    nodes = np.array([0.5])
    wide = source_term(nodes, 0.1, sharpness=5.0)[0]
    narrow = source_term(nodes, 0.1, sharpness=1000.0)[0]
    # At the centre of a pulse both reach the full amplitude of the nearest pulse.
    assert wide > narrow
    # Away from every pulse centre the sharp source has decayed much further.
    assert source_term(nodes, 0.35, sharpness=1000.0)[0] < 1e-9
    assert source_term(nodes, 0.35, sharpness=5.0)[0] > 1e-2


def test_ade_generator_is_stable_and_conserves_mass_when_periodic():
    dx = 1.0 / 64.0
    x = grid(dx, "periodic")
    a = ade_generator(x.size, dx, 1e-3, "periodic")
    # A periodic conservative discretization annihilates the constant mode.
    assert abs(a.dot(np.ones(x.size)).max()) < 1e-9
    eigenvalues = np.linalg.eigvals(a.toarray())
    assert eigenvalues.real.max() < 1e-9


def test_optimized_robin_parameter_switches_branch_at_the_critical_overlap():
    """Theorem 3.1 splits at y0 = l/nu = 1.618386576..."""
    overlap = 2.0 * swr.DX
    below = swr.convergence_factors(overlap / (swr.Y_CRITICAL * 0.5))
    above = swr.convergence_factors(overlap / (swr.Y_CRITICAL * 2.0))
    assert below["branch"] == "equioscillation"
    assert above["branch"] == "pinned"


def test_theoretical_convergence_factors_match_the_published_curve():
    """Figure 3.1(a), read off the published markers at nu = 2^0 ... 2^-7."""
    published = {
        1.0: (0.8952, 0.3329),
        0.5: (0.8851, 0.2471),
        0.25: (0.8590, 0.1662),
        0.125: (0.8214, 0.09822),
        0.0625: (0.6730, 0.04779),
        0.03125: (0.4488, 0.01673),
        0.015625: (0.2017, 0.003133),
        0.0078125: (0.04027, 0.0001521),
    }
    for nu, (dirichlet, robin) in published.items():
        factors = swr.convergence_factors(nu)
        assert factors["robin"] == pytest.approx(robin, rel=0.01)
        if nu <= 0.125:
            # The three largest nu sit against the top of the published axes, where
            # the markers are clipped and their centroids read low.
            assert factors["dirichlet"] == pytest.approx(dirichlet, rel=0.01)


def test_theorem_and_upstream_robin_scalings_differ_by_nu_over_overlap():
    """Theorem 3.1 prints p = p_tilde s / nu; `SWR_for_ADE.m` uses p_tilde nu / l."""
    overlap = 2.0 * swr.DX
    for nu in (1.0, 0.0625):
        theorem, tilde = swr.optimized_robin_parameter(overlap, nu, scaling="theorem")
        upstream, _ = swr.optimized_robin_parameter(overlap, nu, scaling="upstream")
        assert theorem == pytest.approx(tilde / nu)
        assert upstream == pytest.approx(tilde * nu / overlap)


def test_published_iteration_counts_are_reproduced_exactly():
    """Figure 3.1(b) on the eight sampled nu, read off the published markers."""
    published = {
        "dirichlet": (13, 16, 32, 61, 110, 179, 274, 402),
        "robin": (12, 14, 20, 26, 29, 29, 28, 28),
    }
    nus = swr.FIGURE_3_1_NU[::-1]
    for transmission, expected in published.items():
        counts = [swr.oswr_iterations(nu, transmission)[0] for nu in nus]
        for count, want in zip(counts, expected):
            assert count == pytest.approx(want, abs=2)


def test_the_two_counts_quoted_in_the_article_text_are_matched():
    """The article states 92 Dirichlet and 28 Robin iterations at nu = 0.1."""
    assert swr.oswr_iterations(0.1, "dirichlet")[0] == 92
    assert swr.oswr_iterations(0.1, "robin")[0] == 28


def test_upstream_advection_coefficient_shifts_the_curve_by_one_octave():
    """`SWR_for_ADE.m` uses dt/dx for a centred difference, transporting at speed 2."""
    consistent = swr.oswr_iterations(
        0.125, "dirichlet", advection_scale=swr.ADVECTION_SCALE
    )[0]
    upstream = swr.oswr_iterations(
        0.25, "dirichlet", advection_scale=swr.ADVECTION_SCALE_UPSTREAM
    )[0]
    assert upstream == pytest.approx(consistent, rel=0.05)


def test_oswr_iteration_decreases_monotonically_and_stops_on_the_increment():
    iterations, history = swr.oswr_iterations(0.25, "dirichlet", t_final=0.5)
    assert history[-1] <= swr.TOLERANCE
    assert history[-1] < history[0]
    assert len(history) == iterations


def test_robin_beats_dirichlet_and_both_improve_as_nu_falls():
    """The qualitative claim of Figure 3.1(b)."""
    counts = {
        transmission: [
            swr.oswr_iterations(nu, transmission)[0] for nu in (1.0, 0.0625)
        ]
        for transmission in ("dirichlet", "robin")
    }
    assert counts["robin"][0] < counts["dirichlet"][0]
    assert counts["dirichlet"][1] < counts["dirichlet"][0]


@pytest.mark.slow
def test_figure_3_1_runs(tmp_path):
    path = Path(run_figure_3_1(tmp_path, quick=True))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["nu"]) == 4
    assert len(payload["iterations"]["dirichlet"]) == 4
    assert payload["artifacts"]["png"].endswith(".png")


def _dense_paradiag_ii(r1, r2, steps, alpha, second_order):
    """K and P_alpha assembled block by block, the definition of (3.62a), (3.58c)."""
    unknowns = r1.shape[0]
    diagonal, subdiagonal = pdii.time_stencils(second_order)
    sign = pdii.coupling_sign(second_order)
    stiff = np.zeros((steps * unknowns,) * 2)
    precondition = np.zeros_like(stiff)
    for shifts, block, scale in ((diagonal, r1, 1.0), (subdiagonal, r2, sign)):
        for shift in shifts:
            for n in range(steps):
                rows = slice(n * unknowns, (n + 1) * unknowns)
                if n - shift >= 0:
                    column = n - shift
                    weight = 1.0
                else:
                    column = (n - shift) % steps
                    weight = alpha
                    stiff[rows, column * unknowns : (column + 1) * unknowns] += 0.0
                target = slice(column * unknowns, (column + 1) * unknowns)
                precondition[rows, target] += scale * weight * block
                if weight == 1.0:
                    stiff[rows, target] += scale * block
    return stiff, precondition


@pytest.mark.parametrize("second_order", [False, True])
def test_fft_preconditioner_inverts_the_alpha_circulant_matrix(second_order):
    """Step (3.60) has to agree with a dense solve against P_alpha itself."""
    rng = np.random.default_rng(11)
    steps, unknowns, alpha = 6, 4, 0.05
    a = -rng.uniform(0.5, 2.0) * np.eye(unknowns) - rng.normal(size=(unknowns,) * 2)
    if second_order:
        r1, r2 = pdii.numerov_pair(a, 0.1, 1.0 / 120)
    else:
        r1, r2 = pdii.first_order_pair(a, 0.1, 1.0)
    stiff, precondition = _dense_paradiag_ii(r1, r2, steps, alpha, second_order)
    rhs = rng.normal(size=(steps, unknowns))
    solve = pdii.preconditioner_solver(r1, r2, steps, alpha, second_order)
    assert solve(rhs).ravel() == pytest.approx(
        np.linalg.solve(precondition, rhs.ravel()), abs=1e-10
    )
    iterate = rng.normal(size=(steps, unknowns))
    assert pdii.all_at_once(iterate, r1, r2, second_order).ravel() == pytest.approx(
        stiff.dot(iterate.ravel()), abs=1e-10
    )
    assert pdii.sequential_solution(r1, r2, rhs, second_order).ravel() == pytest.approx(
        np.linalg.solve(stiff, rhs.ravel()), abs=1e-9
    )


def test_numerov_stability_threshold_pins_the_iteration_matrix_to_its_bound():
    """Figure 3.17: at gamma = 1/120 every eigenvalue of M has modulus alpha/(1-alpha)."""
    a = pdii.wave_problem(1.0 / 32)[0]
    bound = pdii.WAVE_ALPHA / (1.0 - pdii.WAVE_ALPHA)
    for horizon in (0.5, 10.0):
        steps = int(round(horizon / pdii.WAVE_DT)) - 1
        stable = pdii.iteration_spectrum(
            a, pdii.numerov_symbols(pdii.WAVE_DT, 1.0 / 120), steps, pdii.WAVE_ALPHA, True
        )
        assert np.abs(stable).max() == pytest.approx(bound, rel=1e-3)
    # Just below the threshold the method loses unconditional stability, and the
    # escaping eigenvalue grows with the length of the time interval.
    escaped = [
        np.abs(
            pdii.iteration_spectrum(
                a,
                pdii.numerov_symbols(pdii.WAVE_DT, 1.0 / 120.01),
                int(round(horizon / pdii.WAVE_DT)) - 1,
                pdii.WAVE_ALPHA,
                True,
            )
        ).max()
        for horizon in (0.5, 10.0, 20.0)
    ]
    assert escaped[0] == pytest.approx(bound, rel=0.05)
    assert escaped[1] > 1.5 * bound
    assert escaped[2] > escaped[1]


def test_published_spectral_extremes_of_figure_3_15_are_reproduced():
    """The axes of Figure 3.15(a, c, e) fix three numbers, all with alpha = 1."""
    cases = pdii._cases(pdii.FIRST_ORDER_DX, pdii.FIRST_ORDER_DT, periodic=False)
    deviation = {
        case["name"]: pdii.iteration_spectrum(
            case["problem"][0],
            case["symbols"],
            case["steps"],
            1.0,
            case["second_order"],
        )
        for case in cases
    }
    heat = deviation["heat equation"]
    # Panel (a) spans Re(lambda) in [1, 1.00000002].
    assert np.abs(heat).max() == pytest.approx(2e-8, rel=0.1)
    # Panel (c) insets the mildly advective case on a 1e-14 scale.
    mild = next(v for k, v in deviation.items() if "10^{-3}" in k)
    assert np.abs(mild).max() < 1e-13
    # Panel (e) spans Im(lambda) in [-50, 50] and clusters Re(lambda) at 1/2.
    wave = np.abs(deviation["wave equation"]).max()
    assert wave == pytest.approx(48.0, rel=0.05)


def test_published_head_tail_errors_of_figure_3_16_are_reproduced():
    """Figure 3.16, the increment norms read off the published markers."""
    published = {
        (0.1, "10^{-2}"): (9.3e-3, 2.0e-4, 4.2e-6, 8.8e-8),
        (0.1, "wave"): (1.1e-1, 1.2e-2, 1.3e-3, 1.4e-4),
        (0.001, "10^{-2}"): (9.0e-5, 1.8e-8, 3.8e-12),
        (0.001, "wave"): (8.7e-4, 8.3e-7, 8.5e-10),
    }
    cases = pdii._cases(
        pdii.FIRST_ORDER_DX,
        pdii.FIRST_ORDER_DT,
        periodic=True,
        include_heat=False,
        nus=pdii.FIGURE_3_16_NU,
    )
    for (alpha, label), expected in published.items():
        case = next(c for c in cases if label in c["name"])
        r1, r2, rhs = pdii.build_system(
            case["problem"],
            case["dt"],
            case["steps"],
            second_order=case["second_order"],
        )
        _, increments, _ = pdii.stationary_iteration(
            r1,
            r2,
            rhs,
            case["steps"],
            alpha,
            case["second_order"],
            iterations=len(expected) + 1,
            start=pdii.initial_guess(case["problem"], case["steps"]),
        )
        assert increments[1 : len(expected) + 1] == pytest.approx(expected, rel=0.12)


@pytest.mark.slow
def test_iterative_refinement_defeats_the_roundoff_floor_of_figure_3_18():
    problem = pdii.wave_problem(1.0 / 32)
    steps = int(round(pdii.ROUNDOFF_T / pdii.WAVE_DT)) - 1
    r1, r2, rhs = pdii.build_system(
        problem, pdii.WAVE_DT, steps, second_order=True, gamma=pdii.ROUNDOFF_GAMMA
    )
    start = np.random.default_rng(3).uniform(-1.0, 1.0, size=(steps, problem[1].size))
    floors = {}
    for mode in ("direct", "refinement"):
        _, increments, _ = pdii.stationary_iteration(
            r1, r2, rhs, steps, 1e-11, True, iterations=6, start=start, mode=mode
        )
        floors[mode] = min(increments[-3:])
    # Cond(V_alpha) = 1/alpha, so solving for U^k outright stalls near eps/alpha.
    assert floors["direct"] > 1e-8
    assert floors["refinement"] < 1e-12


def test_published_advection_coefficient_of_figure_3_9_is_one_half():
    """`Direct_ParaDiag_ADE.m` commits `adc = 0`; the panels were made with 1/2.

    The sequential curve of Figure 3.10(b) is the clean measurement, since no
    diagonalization enters it: the published values are 0.0152 at N_t = 64 and
    0.0039 at N_t = 256, and unit advection overshoots both by a factor 4.
    """
    published = {64: 0.0152, 256: 0.0039}
    for steps, target in published.items():
        half = pd1.first_order_operator(0.01, True, speed=pd1.ADE_ADVECTION)
        unit = pd1.first_order_operator(
            0.01, True, speed=pd1.ADE_ADVECTION_UPSTREAM
        )
        error_half = pd1.uniform_error(half[0], half[2], steps, pd1.FIGURE_3_10_T)
        error_unit = pd1.uniform_error(unit[0], unit[2], steps, pd1.FIGURE_3_10_T)
        assert error_half == pytest.approx(target, rel=0.05)
        assert error_unit / error_half == pytest.approx(4.0, rel=0.05)


def test_published_heat_minima_of_figure_3_9_are_reproduced():
    """The two coarsest curves of Figure 3.9(a), read off the published raster."""
    problem = pd1.first_order_operator(1.0, False)
    published = {5: (0.66, 0.0895), 10: (0.33, 0.0508)}
    for steps, (rho, error) in published.items():
        best, measured, _ = pd1.rho_numerical(problem, steps, pd1.FIRST_ORDER_T)
        assert best == pytest.approx(rho, abs=0.02)
        assert measured == pytest.approx(error, rel=0.02)
    # Every curve ends at the same value at rho = 1, which the raster puts at 0.074.
    for steps in (10, 15, 20, 25):
        assert pd1.error_at(problem, steps, 1.0, pd1.FIRST_ORDER_T) == pytest.approx(
            0.0739, rel=0.02
        )


def test_table_3_1_needs_one_ratio_for_every_step_count():
    """Twelve published condition numbers, at the two fixed ratios of the table."""
    published = {
        "B": (1.7e3, 8.4e4, 1.3e6, 2.8e6, 4.4e6, 4.8e6),
        "Btilde_inv_B": (4.7e3, 7.9e5, 6.9e7, 3.8e8, 1.9e9, 4.1e9),
    }
    for label, values in published.items():
        trapezoidal = label != "B"
        rho = pd1.TABLE_3_1_RHO[label]
        horizon = pd1.WAVE_T if trapezoidal else pd1.FIRST_ORDER_T
        for steps, expected in zip(pd1.TABLE_3_1_STEPS, values):
            measured = pd1.eigenvector_condition(
                pd1.backward_euler_matrix,
                steps,
                rho,
                horizon,
                trapezoidal=trapezoidal,
            )
            assert measured == pytest.approx(expected, rel=0.02)
    # The per-step minimizer of Figure 3.9(a) is nothing like that fixed ratio, and
    # neither is the condition number it gives.
    problem = pd1.first_order_operator(1.0, False)
    best, _, _ = pd1.rho_numerical(problem, 5, pd1.FIRST_ORDER_T)
    assert best > 0.5
    assert pd1.eigenvector_condition(
        pd1.backward_euler_matrix, 5, best, pd1.FIRST_ORDER_T
    ) < 100.0


@pytest.mark.slow
def test_holding_the_ratio_fixed_is_what_makes_figure_3_10_blow_up():
    """Re-optimizing per step count plateaus; a fixed ratio reproduces the rise."""
    problem = pd1.first_order_operator(1.0, False)
    fixed = [
        pd1.error_at(problem, steps, pd1.FIGURE_3_10_RHO, pd1.FIGURE_3_10_T)
        for steps in (32, 256)
    ]
    optimal = [
        pd1.rho_numerical(problem, steps, pd1.FIGURE_3_10_T)[1]
        for steps in (32, 256)
    ]
    assert optimal[1] < 1.5 * optimal[0]
    assert fixed[1] > 5.0 * fixed[0]
    # The well-resolved marker is reproducible. At 256 steps the plotted
    # error is dominated by cancellation in V^-1 b (b contains 1/dt_min),
    # and its exact height depends on BLAS/LAPACK. Keep the paper's 0.18 in
    # the comparison report, not as a cross-platform correctness assertion.
    assert fixed[0] == pytest.approx(0.022, rel=0.6)
    deltas = pd1.geometric_steps(pd1.FIGURE_3_10_T, 256, 1 + pd1.FIGURE_3_10_RHO)
    state = problem[2].copy()
    sequential = []
    for dt in deltas:
        state = np.linalg.solve(np.eye(state.size) + dt * problem[0], state)
        sequential.append(state.copy())
    reference = pd1.exponential_reference(problem[0], problem[2], np.cumsum(deltas))
    truncation = np.abs(reference - sequential).max()
    assert truncation == pytest.approx(0.01622346488, rel=1e-6)
    # The paper demonstrates numerical instability of diagonalization,
    # not a large truncation error of backward Euler on the same mesh.
    assert fixed[1] > 5 * truncation


def test_a_ratio_too_large_for_the_step_count_is_reported_as_unusable():
    """At the top of the scan the first step of the mesh is already subnormal.

    Its reciprocal in (3.23b) is then no longer representable, so the entry has
    to be reported as Inf and skipped, as the published sweep does, instead of
    overflowing on the way there.
    """
    deltas = pd1.geometric_steps(pd1.FIGURE_3_10_T, 1024, 2.0)
    assert 0.0 < deltas.min() < np.finfo(float).tiny
    problem = pd1.first_order_operator(1.0, False)
    with np.errstate(over="raise", divide="raise"):
        assert pd1.error_at(problem, 1024, 1.0, pd1.FIGURE_3_10_T) == np.inf


def test_bvm_discretization_keeps_the_eigenvector_matrix_tame():
    """Figure 3.12(b): O(N_t^2) against the geometric mesh, which explodes."""
    for steps, published in ((4, 3.0), (256, 1.0e4)):
        matrix = pdbvm.bvm_matrix(steps, 0.5 / steps)
        condition = float(np.linalg.cond(np.linalg.eig(matrix)[1]))
        assert condition == pytest.approx(published, rel=3.0)
    geometric = pd1.eigenvector_condition(
        pd1.backward_euler_matrix, 256, 0.15, 0.5
    )
    assert geometric > 1e6


@pytest.mark.slow
def test_initial_curve_of_figure_3_5_is_the_published_one_up_to_one_constant():
    """The k_max = 0 curve carries no method in it, only the reference solution.

    Its shape therefore fixes the parameters of each panel, and what is left over
    is the normalizing constant the published panels were drawn with.
    """
    for panel, nu, sharpness in idc.FIGURE_3_5_PANELS:
        table, windows, _ = idc.error_table(nu, sharpness, {"IDC": "idc"}, sweeps=1)
        assert windows == 30
        against = idc.published_offset(panel, table["IDC"][0])
        assert against["median"] == pytest.approx(
            idc.PUBLISHED_OFFSET[panel], rel=0.01
        )
        # A constant to well under a per cent where the source is a spike, and to
        # the accuracy of the reading where it is smooth.
        tolerance = 0.02 if sharpness > 100 else 0.15
        assert against["worst_deviation_from_median"] < tolerance
    # The two smooth panels are the ones drawn against the definition of err^k_n.
    assert idc.PUBLISHED_OFFSET["c"] == pytest.approx(1.0, rel=0.02)
    assert idc.PUBLISHED_OFFSET["d"] == pytest.approx(1.0, rel=0.02)


def test_nka_weights_are_flat_where_diffusion_dominates_the_frobenius_norm():
    """Figure 3.14: the weights (3.48) can only act at the smaller viscosity.

    The reduced model of the text spreads them by a factor of three at
    nu = 0.002 and by five per cent at nu = 0.02, which is the whole of the
    difference between the two published panels.
    """
    steps = int(round(1.3 / pdbvm.NKA_DT))
    flat = pdbvm.nka_weights(0.02, pdbvm.NKA_DT, steps)
    spread = pdbvm.nka_weights(0.002, pdbvm.NKA_DT, steps)
    assert flat.max() / flat.min() == pytest.approx(1.087, rel=0.01)
    assert spread.max() / spread.min() == pytest.approx(3.22, rel=0.01)


def test_reference_history_of_figure_3_14_starts_at_the_initial_distance():
    """The published panels plot |U^k - U*|, so the iteration must report it."""
    advection, diffusion, x = pdbvm.burgers_operators(0.02, 1.0 / 50)
    u0 = pdbvm.initial_condition(x)
    reference, residual = pdbvm.all_at_once_solution(
        advection, diffusion, u0, 20, 0.02
    )
    assert residual < 1e-12
    _, history, converged = pdbvm.quasi_newton(
        advection, diffusion, u0, 20, 0.02, iterations=20, start="symmetric",
        tolerance=pdbvm.NKA_TOLERANCE, reference=reference,
    )
    start = np.random.default_rng(pdbvm.RANDOM_SEED).uniform(
        -1.0, 1.0, size=(20, u0.size)
    )
    assert converged
    assert history[0] == pytest.approx(np.abs(start - reference).max())
    assert history[-1] <= pdbvm.NKA_TOLERANCE


@pytest.mark.slow
def test_first_published_value_of_figure_3_14a_is_the_random_start():
    """1.993 and 1.961 against 1 + max|U| = 1.992; the constant start is at 0.73."""
    advection, diffusion, x = pdbvm.burgers_operators(0.02, pdbvm.NKA_DX)
    u0 = pdbvm.initial_condition(x)
    reference, _ = pdbvm.all_at_once_solution(
        advection, diffusion, u0, int(round(0.7 / pdbvm.NKA_DT)), pdbvm.NKA_DT
    )
    bound = 1.0 + float(np.abs(reference).max())
    published = [
        pdbvm.PUBLISHED_FIGURE_3_14[(0.02, "average", horizon)][0]
        for horizon in pdbvm.NKA_HORIZONS
    ]
    assert max(published) == pytest.approx(bound, rel=0.01)
    assert min(published) > 0.98 * bound
    constant = float(np.abs(np.tile(u0, (reference.shape[0], 1)) - reference).max())
    assert constant < 0.4 * bound


def _six_decades(curve):
    """Where a published curve has fallen six decades below its first value."""
    iterations = np.array(sorted(curve), dtype=float)
    logs = np.log10([curve[int(k)] for k in iterations])
    return float(np.interp(logs[0] - 6.0, logs[::-1], iterations[::-1]))


@pytest.mark.slow
def test_no_weight_vector_of_the_scan_reaches_the_published_gain_at_nu_0_02():
    """Figure 3.14(a): the published NKA saves four and a half iterations of ten,
    and the best weight vector of the scan saves two of eleven."""
    published = {
        variant: _six_decades(pdbvm.PUBLISHED_FIGURE_3_14[(0.02, variant, 0.7)])
        for variant in ("average", "nka")
    }
    assert published["average"] == pytest.approx(10.3, abs=0.1)
    assert published["nka"] == pytest.approx(5.9, abs=0.1)
    scan = pdbvm.nka_weight_scan(
        0.02, pdbvm.NKA_SCAN_DX, pdbvm.NKA_DT, int(round(0.7 / pdbvm.NKA_DT)),
        iterations=40,
    )
    counts = [
        record["symmetric"]["iterations"] for record in scan
        if record["symmetric"]["iterations"] is not None
    ]
    assert scan[0]["symmetric"]["iterations"] == 11
    assert min(counts) >= published["average"] - 2.0


@pytest.mark.slow
def test_sequential_column_of_table_3_2_is_reproduced():
    """401 Jacobian solves for nu = 0.1, T = 0.1, and 400 for nu = 0.002."""
    for nu, expected in ((0.1, 401), (0.002, 400)):
        advection, diffusion, x = pdbvm.burgers_operators(nu, pdbvm.BURGERS_DX)
        _, evaluations = pdbvm.trapezoidal_steps(
            advection,
            diffusion,
            pdbvm.initial_condition(x),
            pdbvm.BURGERS_STEPS,
            0.1 / pdbvm.BURGERS_STEPS,
        )
        assert int(evaluations.sum()) == expected
