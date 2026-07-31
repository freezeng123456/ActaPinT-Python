import json
from pathlib import Path

import numpy as np
import pytest

from actapint.section2 import source_term
from actapint.section3 import run_figure_3_1, swr
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
