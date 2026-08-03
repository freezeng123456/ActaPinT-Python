import functools

import numpy as np
import pytest

from actapint.section4 import diagg, mgrit, parareal, pcgc, pfasst, stmg
from actapint.section4.common import (
    alpha_circulant_solve,
    convergence_factors,
    linear_operator,
    matlab_random_iterate,
    stability_function,
    superlinear_bound,
)


def test_stability_functions_match_their_closed_forms():
    """The five integrators of (4.9), checked against the printed factors."""
    z = np.array([0.3 + 0.2j, 2.0, -0.5])
    assert stability_function("ie", z) == pytest.approx(1.0 / (1.0 + z))
    assert stability_function("tr", z) == pytest.approx((1.0 - 0.5 * z) / (1.0 + 0.5 * z))
    # Every scheme is consistent, so all of them agree with exp(-z) to their order.
    small = np.array([1e-3])
    for method, order in (("ie", 1), ("tr", 2), ("sdirk22", 2), ("sdirk23", 3)):
        error = abs(stability_function(method, small)[0] - np.exp(-small[0]))
        assert error < 1e-3 ** (order + 1) * 10


def test_substeps_raise_the_fine_factor_to_the_power_of_the_coarsening():
    """The fine propagator of one coarse step is J steps of the fine scheme."""
    z = np.array([0.7])
    single = stability_function("sdirk22", z / 5)
    assert stability_function("sdirk22", z, substeps=5) == pytest.approx(single**5)


def test_superlinear_bound_is_the_falling_factorial_of_theorem_4_2():
    bound = superlinear_bound(0.3, 10, 3)
    assert bound[0] == 1.0
    assert bound[1] == pytest.approx(0.3 * 9)
    assert bound[2] == pytest.approx(0.3**2 * 9 * 8 / 2)
    assert bound[3] == pytest.approx(0.3**3 * 9 * 8 * 7 / 6)


def test_negative_nu_selects_the_heat_equation():
    """Every script of the chapter switches problems through the sign of nu."""
    heat, _ = linear_operator(1 / 16, -0.5)
    ade, _ = linear_operator(1 / 16, 0.5)
    assert np.allclose(heat.toarray(), heat.toarray().T)
    assert not np.allclose(ade.toarray(), ade.toarray().T)


def test_matlab_random_iterate_reproduces_the_default_stream():
    """MATLAB's rng('default') is MT19937 seeded with 5489, filled column-wise."""
    values = matlab_random_iterate(np.zeros(3), 2, -1.0, 1.0)
    expected = np.random.RandomState(5489).random_sample(9)[3:]
    assert values.shape == (3, 3)
    assert np.all(values[0] == 0.0)
    assert values[1:].T.ravel(order="F") == pytest.approx(-1.0 + 2.0 * expected)


def test_published_convergence_factors_of_figure_4_9():
    """Five of the six printed maxima, to the four digits the titles give."""
    published = {
        ("heat", "mgrit"): 0.08375,
        ("ade_0.1", "mgrit"): 0.2718,
        ("ade_0.01", "mgrit"): 0.9021,
        ("heat", "parareal"): 0.2822,
        ("ade_0.1", "parareal"): 0.4453,
    }
    keys = {"mgrit": "rho_mgrit_max", "parareal": "rho_l_max"}
    factors = {
        "heat": mgrit._factors(-1.0, "dirichlet")[2],
        "ade_0.1": mgrit._factors(0.1, "periodic")[2],
        "ade_0.01": mgrit._factors(0.01, "periodic")[2],
    }
    for (case, method), value in published.items():
        assert factors[case][keys[method]] == pytest.approx(value, rel=3e-3)


def test_the_printed_parareal_factor_at_nu_0_01_is_above_one():
    """The published 0.9986 is the one value of Figure 4.9 that cannot be matched.

    The same routine reproduces the other five maxima and both divergent factors
    at nu = 0.002 to four digits, so the printed value is treated as a typo; what
    matters for the chapter's argument is that Parareal does not converge there.
    """
    factors = mgrit._factors(0.01, "periodic")[2]
    assert factors["rho_l_max"] > 1.0
    divergent = mgrit._factors(0.002, "periodic")[2]
    assert divergent["rho_l_max"] == pytest.approx(1.4211, rel=1e-3)
    assert divergent["rho_mgrit_max"] == pytest.approx(1.2812, rel=1e-3)


FIGURE_4_10_CASES = {
    "heat": (-1.0, "dirichlet"),
    "ade_0.1": (0.1, "periodic"),
    "ade_0.01": (0.01, "periodic"),
    "ade_0.002": (0.002, "periodic"),
}
# Below this the two runs differ in round-off rather than in iteration: the
# published curves flatten out around 1e-14 while the reproduced ones keep
# falling, since a precomputed propagator matrix accumulates less than the J
# successive solves of the script.  It is four decades under the truncation
# level the figure draws, so nothing of the comparison lives there.
FIGURE_4_10_FLOOR = 1e-9


@functools.lru_cache(maxsize=None)
def _figure_4_10_panel(panel):
    """Both curves of one panel of Figure 4.10, at the published cost matching."""
    nu, boundary = FIGURE_4_10_CASES[panel]
    _, parareal_errors, mgrit_errors = mgrit._linear_history(
        nu, boundary, mgrit.ITERATIONS, panel
    )
    return {"parareal": mgrit._cost_matched(parareal_errors), "mgrit": mgrit_errors}


def _against_published(panel, method, errors=None, floor=FIGURE_4_10_FLOOR):
    """Reproduced over published at every digitized marker above the floor."""
    if errors is None:
        errors = _figure_4_10_panel(panel)[method]
    return {
        k: errors[k] / value
        for k, value in mgrit.PUBLISHED_FIGURE_4_10[(panel, method)].items()
        if value > floor
    }


@pytest.mark.parametrize("panel", sorted(FIGURE_4_10_CASES))
def test_figure_4_10_reproduces_every_published_marker(panel):
    """Both curves of every panel, marker by marker, not just the rate.

    The digitizer is good to 0.018 decades on a marker it reads in full, so four
    per cent is the width of the reading; the panels come out inside it
    everywhere above the round-off floor, which is what fixes the two setups of
    `mgrit.FIGURE_4_10_SETUPS` uniquely.
    """
    for method in ("parareal", "mgrit"):
        ratios = _against_published(panel, method)
        assert len(ratios) >= 8
        for k, ratio in ratios.items():
            assert ratio == pytest.approx(1.0, rel=0.06), (method, k, ratio)


def test_figure_4_10_setups_are_not_interchangeable():
    """Neither setup draws the other's panels, which is why the figure needs both.

    Running `nu = 0.1` as the script now stands, from the random draw and the
    chirped initial condition, leaves the MGRiT curve a factor of four above the
    published one from the fourth iteration on, and running the divergent panel
    from `u0` at every level misses the first marker by a third.
    """
    assert mgrit.FIGURE_4_10_SETUPS["ade_0.1"] == ("plain", "constant")
    assert mgrit.FIGURE_4_10_SETUPS["ade_0.002"] == ("chirp", "random")

    mgrit.FIGURE_4_10_SETUPS["ade_0.1"] = ("chirp", "random")
    mgrit.FIGURE_4_10_SETUPS["ade_0.002"] = ("plain", "constant")
    try:
        _, parareal_errors, mgrit_errors = mgrit._linear_history(
            0.1, "periodic", mgrit.ITERATIONS, "ade_0.1"
        )
        swapped = _against_published("ade_0.1", "mgrit", mgrit_errors)
        assert all(3.8 < swapped[k] < 4.5 for k in range(4, 11))
        _, divergent, _ = mgrit._linear_history(
            0.002, "periodic", mgrit.ITERATIONS, "ade_0.002"
        )
        matched = mgrit._cost_matched(divergent)
        assert _against_published("ade_0.002", "parareal", matched)[1] > 1.5
    finally:
        mgrit.FIGURE_4_10_SETUPS["ade_0.1"] = ("plain", "constant")
        mgrit.FIGURE_4_10_SETUPS["ade_0.002"] = ("chirp", "random")


def test_parareal_factor_crosses_one_below_nu_one_hundredth():
    """Figure 4.4: the linear bound is what fails first as the diffusion weakens."""
    operator, _ = linear_operator(parareal.FIGURE_4_4_DX, 0.1)
    strong = convergence_factors(operator, 0.1, 32, "sdirk22")["rho_l_max"]
    operator, _ = linear_operator(parareal.FIGURE_4_4_DX, 0.005)
    weak = convergence_factors(operator, 0.1, 32, "sdirk22")["rho_l_max"]
    assert strong < 1.0 < weak


def test_alpha_circulant_solve_inverts_the_coarse_grid_correction():
    """(4.17) has to solve the system (4.16) it diagonalizes.

    ``C_alpha`` carries one on the diagonal, minus one on the subdiagonal and
    minus ``alpha`` in the top right corner, which is the head-tail coupling.
    """
    operator, _ = linear_operator(1 / 16, 0.1)
    steps, size = 8, operator.shape[0]
    alpha, coarse_dt = 0.25, 0.1
    rhs = np.random.default_rng(0).standard_normal((steps, size))
    solution = alpha_circulant_solve(operator, coarse_dt, alpha, rhs)
    residual = solution + coarse_dt * (operator @ solution.T).T
    residual[1:] -= solution[:-1]
    residual[0] -= alpha * solution[-1]
    assert residual.ravel() == pytest.approx(rhs.ravel(), abs=1e-10)


def test_pfasst_quadrature_matrices_match_the_printed_values():
    """Radau IIA with three and two nodes, as the text prints them."""
    fine = pfasst.quadrature_matrix(pfasst.FINE_NODES)
    root = np.sqrt(6.0)
    assert fine[0] == pytest.approx(
        [(88 - 7 * root) / 360, (296 - 169 * root) / 1800, (-2 + 3 * root) / 225]
    )
    assert fine[2] == pytest.approx([(16 - root) / 36, (16 + root) / 36, 1 / 9])
    coarse = pfasst.quadrature_matrix(pfasst.COARSE_NODES)
    assert coarse.ravel() == pytest.approx([5 / 12, -1 / 12, 3 / 4, 1 / 4])


def test_pfasst_transfer_matrices_match_the_printed_values():
    to_fine = pfasst.transfer_matrix(pfasst.COARSE_NODES, pfasst.FINE_NODES)
    assert to_fine.ravel() == pytest.approx(
        [1.2674, -0.2674, 0.5325, 0.4674, 0.0, 1.0], abs=1e-4
    )
    to_coarse = pfasst.transfer_matrix(pfasst.FINE_NODES, pfasst.COARSE_NODES)
    assert to_coarse.ravel() == pytest.approx(
        [0.5018, 0.6833, -0.1851, 0.0, 0.0, 1.0], abs=1e-4
    )


def test_pfasst_converges_to_the_collocation_solution():
    """The fixed point of the block iteration is the sequential solve.

    This is the check that pins the three block operators: if any of them is
    assembled wrongly the iteration still converges, but to something else.
    """
    operators = pfasst.block_operators(1.0, 0.0, dx=1 / 16, block_dt=1 / 16)
    sources = pfasst._sources(operators["x"], 1 / 16, 6)
    reference = pfasst._sequential(operators["fine"], sources, operators["size"])
    errors = pfasst.pfasst(operators, sources, iterations=200, tolerance=1e-13)
    assert errors[-1] < 1e-13
    # The collocation solution solves its own block system, level by level.
    previous = np.zeros(operators["size"])
    for level in range(reference.shape[0]):
        right = np.tile(previous, len(pfasst.FINE_NODES)) + sources[level]
        assert operators["fine"] @ reference[level] == pytest.approx(right, abs=1e-10)
        previous = reference[level, -operators["size"] :]


def test_pfasst_matches_the_published_curves_of_figure_4_6():
    """Figure 4.6: the digitized counts to reach 1e-12 and the tail factors.

    The factor is the quantity the figure is about, and it is what the
    advection-dominated curves are sensitive to: an upwind advection term or a
    grid coarser by two moves it by 4 to 18 per cent, so a per cent here pins
    the discretization down.
    """
    steps = int(round(pfasst.T_FINAL / pfasst.BLOCK_DT))
    for label, nu, advection in (
        ("heat equation", 1.0, 0.0),
        (r"ADE with $\nu=0.001$", 1e-3, 1.0),
        (r"ADE with $\nu=0.0001$", 1e-4, 1.0),
    ):
        operators = pfasst.block_operators(nu, advection)
        sources = pfasst._sources(operators["x"], pfasst.BLOCK_DT, steps)
        errors = pfasst.pfasst(operators, sources)
        assert len(errors) - 1 == pytest.approx(
            pfasst.PUBLISHED_ITERATIONS[label], abs=3
        )
        assert pfasst._measured_factor(errors) == pytest.approx(
            pfasst.PUBLISHED_FACTOR[label], rel=0.01
        )


def test_diagonalized_coarse_solver_leaves_a_defect_of_order_alpha():
    """(4.29) rests on F - F* being O(alpha); this is that statement, measured."""
    operator, x = diagg.heat_system(diagg.DX)
    state = np.sin(2.0 * np.pi * x) ** 2
    defects = []
    for alpha in (1e-2, 1e-3):
        fine, coarse = diagg.propagators(
            operator, diagg.COARSE_DT, diagg.SUBSTEPS, alpha
        )
        defects.append(np.abs((fine - coarse) @ state).max())
    assert defects[0] / defects[1] == pytest.approx(10.0, rel=0.05)


def _figure_4_14_history(steps, alpha, substeps=diagg.HEAT_SUBSTEPS):
    operator, x = diagg.heat_system(diagg.DX)
    u0 = np.sin(2.0 * np.pi * x) ** 2
    return diagg._history(
        operator, u0, steps, alpha, random_start=True, scale_initial=False,
        substeps=substeps,
    )


@pytest.mark.parametrize("steps,alpha", sorted(diagg.PUBLISHED_HEAT))
def test_diagonalized_coarse_solver_matches_the_published_heat_curves(steps, alpha):
    """Figure 4.14 against the curves digitized from the vector content of page 85."""
    published = diagg.PUBLISHED_HEAT[(steps, alpha)]
    errors = _figure_4_14_history(steps, alpha)
    assert diagg._deviation(errors, published) < 0.1
    assert diagg._tail_factor(errors[: len(published)]) == pytest.approx(
        diagg._tail_factor(published), rel=0.1
    )


def test_figure_4_14_needs_the_recovered_coarsening_factor():
    """The published panels are drawn with J = 20, not the J = 10 of the text.

    The factor per iteration is governed by ``max |R^J|`` over the spectrum, which
    is ``exp(-J**2 dx**2 / DT)``: 0.62 at ``J = 20``, where the curves are straight
    and just under ``alpha``, and 0.89 at ``J = 10``, where they bend and run
    several times ``alpha``.  Halving ``J`` costs more than a decade on every curve.
    """
    for steps, alpha in sorted(diagg.PUBLISHED_HEAT):
        published = diagg.PUBLISHED_HEAT[(steps, alpha)]
        recovered = diagg._deviation(_figure_4_14_history(steps, alpha), published)
        stated = diagg._deviation(
            _figure_4_14_history(steps, alpha, substeps=10), published
        )
        assert recovered < 0.1 < 1.0 < stated


def test_wave_system_has_a_purely_imaginary_spectrum():
    """Why the second variant is the one that survives a hyperbolic problem."""
    operator, _ = diagg.wave_system(1 / 20)
    eigenvalues = np.linalg.eigvals(operator.toarray())
    assert np.abs(eigenvalues.real).max() < 1e-8
    assert np.abs(eigenvalues.imag).max() > 1.0


@pytest.mark.parametrize("alpha,steps", sorted(diagg.PUBLISHED_WAVE))
def test_diagonalized_coarse_solver_matches_the_published_wave_errors(alpha, steps):
    """Figure 4.15 against the curves digitized from the vector content of page 86.

    These keep ``SUBSTEPS`` and the script's committed start, unlike Figure 4.14,
    and every plotted point of them is reproduced to a per cent.
    """
    operator, x = diagg.wave_system(diagg.DX)
    u0 = np.concatenate([np.sin(2.0 * np.pi * x) ** 2, np.zeros(x.size)])
    errors = diagg._history(
        operator, u0, steps, alpha, random_start=False, scale_initial=False
    )
    for k, value in enumerate(diagg.PUBLISHED_WAVE[(alpha, steps)]):
        assert errors[k] == pytest.approx(value, rel=0.01)


def test_burgers_window_solver_of_figure_4_17_approaches_the_fine_solver():
    """F*_alpha solves the same window system as F up to the alpha-coupling."""
    from actapint.section4.common import (
        BURGERS_FLUX_SCALE_AS_COMMITTED,
        burgers_operators,
    )

    a1, a2, x = burgers_operators(
        1 / 50, 0.1, boundary_condition="periodic",
        flux_scale=BURGERS_FLUX_SCALE_AS_COMMITTED,
    )
    state = np.sin(2.0 * np.pi * x) ** 2
    defects = []
    for alpha in (1e-2, 1e-3):
        fine, coarse = diagg._burgers_window(a1, a2, 0.1, 10, alpha, 1e-12, 12)
        defects.append(np.abs(fine(state) - coarse(state)).max())
    assert defects[0] / defects[1] == pytest.approx(10.0, rel=0.1)


def test_pcgc_reduces_to_standard_parareal_as_alpha_vanishes():
    """The head-tail coupled CGC (4.15) becomes the standard CGC (4.14) at alpha = 0.

    The two are compared through their asymptotic rate rather than iterate by
    iterate: the ``alpha = 0`` curve goes through ``parareal``, which carries the
    scripts' quirk of subtracting the starting iterate itself in the first sweep,
    so the two histories are offset by one sweep even though the operator is the
    same.
    """
    assert pcgc.ALPHAS[-1] == 0.0
    small = pcgc._linear_history(-1.0, 1e-6, iterations=10)
    exact = pcgc._linear_history(-1.0, 0.0, iterations=10)

    def rate(errors):
        return (errors[-1] / errors[-4]) ** (1 / 3)

    assert rate(small) == pytest.approx(rate(exact), rel=0.1)


def test_too_small_an_alpha_stalls_the_circulant_solve_at_roundoff():
    """The trade-off of the diagonalization: accuracy is lost like eps / alpha."""
    errors = pcgc._linear_history(-1.0, 1e-12, iterations=10)
    assert min(errors) > 1e-5
    assert errors[-1] == pytest.approx(np.finfo(float).eps / 1e-12, rel=1.0)


def test_lfa_factor_reproduces_theorem_4_9_for_the_heat_equation():
    """At eta = 1/2 every high frequency in time is damped by at least 1/sqrt(2)."""
    omega = np.linspace(0.5 * np.pi, np.pi, 65)
    xi = np.linspace(-np.pi, np.pi, 65)
    # The heat equation is the advection-free limit of (4.40); take dt/dx**2 large
    # enough that the space coarsening condition of the theorem also holds.
    values = stmg.lfa_factor(0.5, omega[:, None], xi[None, :], 1.0, 1 / 64, 1 / 64)
    assert values.max() <= 1.0 / np.sqrt(2.0) + 1e-12


def test_lfa_maximum_has_its_minimum_near_one_half():
    """Figure 4.18: the optimum drifts up with weaker diffusion but stays near 1/2."""
    for nu, expected in ((0.1, 0.50), (0.01, 0.54), (0.001, 0.58)):
        values = [(stmg.lfa_maximum(eta, nu, 1 / 128), eta) for eta in stmg.LFA_ETAS]
        best, eta = min(values)
        assert eta == pytest.approx(expected, abs=0.02)
        assert 0.7 < best < 0.8


def test_stmg_reproduces_the_published_error_of_figure_4_20():
    """One smoothing step, eta = 1/2, at the published marker positions."""
    published = {
        -1.0: {3: 6.5e-2, 6: 4.0e-3, 12: 5.4e-5, 21: 1.2e-7},
        0.001: {3: 6.0e-1, 6: 9.0e-2, 12: 3.0e-3, 21: 6.0e-5},
    }
    for nu, markers in published.items():
        setup = stmg.linear_setup(nu, theta=1.0)
        errors = stmg.stmg_errors(setup, 0.5, 1, 21, start=stmg.random_start(setup))
        for k, value in markers.items():
            assert errors[k] == pytest.approx(value, rel=0.35)


def test_stmg_smoother_alone_does_not_converge_without_the_coarse_correction():
    """The point of the coarse grid correction: block Jacobi only damps."""
    setup = stmg.linear_setup(-1.0, theta=1.0, level=6)
    start = stmg.random_start(setup)
    smoothed = stmg._smooth(setup, start, 0.5, 20)
    reference = setup["reference"]
    only_smoothing = np.abs(smoothed - reference).max()
    cycled = stmg.stmg_errors(setup, 0.5, 1, 20, start=start)[-1]
    assert only_smoothing > 1e-3
    assert cycled < 1e-8 * only_smoothing


def test_stmg_burgers_reproduces_the_published_error_of_figure_4_22():
    """Two smoothing steps at eta = 0.28, nu = 1, at the published markers."""
    setup = stmg.burgers_setup(1.0)
    errors = stmg.burgers_errors(
        setup, stmg.BURGERS_ETA, stmg.BURGERS_SMOOTHING, 8,
        start=stmg.burgers_start(setup),
    )
    published = {0: 1.3, 2: 1.2e-1, 4: 2.0e-2, 8: 1.6e-3}
    for k, value in published.items():
        assert errors[k] == pytest.approx(value, rel=0.6)
    assert errors[8] < errors[4] < errors[2] < errors[0]
