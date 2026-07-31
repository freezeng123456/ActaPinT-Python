"""Tests for the Section 2 model problems.

Section 2 publishes no numbers, so these tests pin the discretizations through
the conservation and decay properties the chapter's argument relies on: parabolic
Dirichlet problems forget their history, Neumann and periodic problems retain the
constant mode, and the wave equation conserves energy.
"""

import json

import numpy as np
import pytest
from scipy import sparse

from actapint.operators import (
    neumann_first_derivative,
    neumann_negative_laplacian,
)
from actapint.section2 import (
    BURGERS_FLUX_COEFFICIENT_UPSTREAM,
    FIGURE_2_1,
    FIGURE_2_2,
    FIGURE_2_3,
    FIGURE_2_4,
    SOURCE_AMPLITUDE,
    SOURCE_TIMES,
    initial_condition,
    run_figure_2_1,
    run_section2,
    solve_burgers,
    solve_heat_ade,
    solve_wave,
    source_term,
)


def test_panel_specifications_cover_every_published_panel():
    assert len(FIGURE_2_1) == 4
    assert len(FIGURE_2_2) == 8
    assert len(FIGURE_2_3) == 8
    assert len(FIGURE_2_4) == 4
    labels = [label for label, _ in FIGURE_2_2]
    assert labels == list("abcdefgh")


def test_source_term_peaks_at_the_published_times_and_centre():
    centre = np.array([0.5])
    for time in SOURCE_TIMES:
        assert source_term(centre, time)[0] == pytest.approx(SOURCE_AMPLITUDE, rel=1e-9)
    # Half way between two pulses the forcing is negligible.
    assert source_term(centre, 0.35)[0] < 1e-3
    # The Gaussian is centred in space.
    assert source_term(np.array([0.0]), 0.1)[0] < 1e-15


def test_initial_condition_is_compatible_with_dirichlet_and_periodic_ends():
    ends = initial_condition(np.array([0.0, 1.0]))
    assert np.allclose(ends, 0.0, atol=1e-12)
    sampled = initial_condition(np.linspace(0.0, 1.0, 4001))
    assert sampled.min() >= 0.0
    assert sampled.max() == pytest.approx(1.0, abs=1e-6)


def test_neumann_operators_annihilate_constants():
    n, dx = 40, 1 / 40
    one = np.ones(n)
    assert np.abs(neumann_negative_laplacian(n, dx) @ one).max() < 1e-12
    assert np.abs(neumann_first_derivative(n, dx) @ one).max() < 1e-12


def test_dirichlet_heat_forgets_its_history():
    """Figure 2.1(a): the solution is local in time under Dirichlet conditions."""
    _, _, u, _ = solve_heat_ade(
        nu=1.0,
        advection=0.0,
        boundary_condition="dirichlet",
        with_source=True,
        dt=4e-3,
    )
    assert np.abs(u[:, -1]).max() < 1e-4 * u.max()


@pytest.mark.parametrize("boundary_condition", ["neumann", "periodic"])
def test_closed_heat_problems_accumulate_and_retain_the_constant(boundary_condition):
    """Figure 2.1(b,c): heat cannot escape, so the final state is near the maximum."""
    _, _, u, _ = solve_heat_ade(
        nu=1.0,
        advection=0.0,
        boundary_condition=boundary_condition,
        with_source=True,
        dt=4e-3,
    )
    final = u[:, -1]
    assert final.min() > 0.9 * u.max()
    # Only the constant mode survives.
    assert final.max() - final.min() < 0.02 * u.max()


def test_periodic_heat_conserves_mass():
    _, _, u, _ = solve_heat_ade(
        nu=1.0,
        advection=0.0,
        boundary_condition="periodic",
        with_source=False,
        dt=4e-3,
    )
    mass = u.sum(axis=0)
    assert abs(mass[-1] - mass[0]) / mass[0] < 1e-8


def test_periodic_burgers_flux_is_conservative():
    """The nonlinear flux is in divergence form, so periodic mass is invariant."""
    _, _, u, _ = solve_burgers(
        nu=5e-4, boundary_condition="periodic", with_source=False, dt=1 / 160
    )
    mass = u.sum(axis=0)
    assert abs(mass[-1] - mass[0]) / mass[0] < 1e-12


def test_burgers_flux_coefficient_controls_the_shock_speed():
    """The published panels need a larger flux than the committed script states."""
    fronts = {}
    for coefficient in (BURGERS_FLUX_COEFFICIENT_UPSTREAM, 1.0):
        x, t, u, parameters = solve_burgers(
            nu=5e-4,
            boundary_condition="dirichlet",
            with_source=False,
            dt=1 / 400,
            flux_coefficient=coefficient,
        )
        assert parameters["flux_coefficient"] == coefficient
        column = int(np.argmin(np.abs(t - 1.0)))
        # Mass remaining in the domain measures how fast the shock drained it.
        fronts[coefficient] = u[:, column].mean()
    assert fronts[1.0] < 0.75 * fronts[BURGERS_FLUX_COEFFICIENT_UPSTREAM]


def test_wave_trapezoidal_rule_conserves_energy():
    from actapint.operators import dirichlet_negative_laplacian

    dt = 1 / 200
    x, t, u, parameters = solve_wave(
        wave_speed_squared=0.2,
        boundary_condition="dirichlet",
        with_source=False,
        dt=dt,
    )
    interior = u[1:-1]
    stiffness = dirichlet_negative_laplacian(interior.shape[0], parameters["dx"], 0.2)
    # Reconstruct the potential energy; it alone is not invariant, but the
    # trapezoidal rule keeps the total bounded, so the amplitude must not grow.
    potential = np.array([column @ (stiffness @ column) for column in interior.T])
    assert potential.max() <= potential[0] * (1.0 + 1e-8)
    assert np.abs(u).max() == pytest.approx(1.0, abs=5e-2)


def test_wave_solution_is_not_local_in_time():
    """Figure 2.4: hyperbolic solutions keep detail for every boundary condition."""
    for boundary_condition in ("dirichlet", "neumann", "periodic"):
        _, _, u, _ = solve_wave(
            wave_speed_squared=0.2,
            boundary_condition=boundary_condition,
            with_source=True,
            dt=1 / 160,
        )
        assert np.abs(u[:, -1]).max() > 0.2 * u.max()


def test_figure_2_1_writes_four_panels_and_artifacts(tmp_path):
    path = run_figure_2_1(tmp_path, quick=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["panels"]) == 4
    assert [panel["label"] for panel in payload["panels"]] == list("abcd")
    for extension in ("png", "svg"):
        assert (tmp_path / f"figure_2_1_heat.{extension}").exists()


@pytest.mark.slow
def test_section2_summary_reports_every_panel(tmp_path):
    path = run_section2(tmp_path, quick=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["panels_total"] == 24
