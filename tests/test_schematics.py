import json

import numpy as np
import pytest

from actapint import schematics


def test_forward_euler_values_sit_on_the_grid_of_the_page():
    """Figure 1.1 puts its thirteen markers on a uniform grid, 1.869 pt apart."""
    values = np.array(schematics.FORWARD_EULER, dtype=float)
    assert values.size == 13
    assert np.all(values == np.round(values))
    # The shape of the published trajectory: it plateaus at u_3, u_4 and at u_11.
    assert values.argmax() == 3
    assert values[3] == values[4] == values[11] == values.max()
    assert values[0] == values.min()


def _at(x, t, mask, point):
    """The value of a space-time mask at one point of the schematic."""
    return bool(mask[np.abs(t - point[1]).argmin(), np.abs(x - point[0]).argmin()])


def test_first_red_solve_is_exact_inside_the_tents_only():
    """Theorem 3.2: the first solve is exact in the tents and nowhere above them."""
    x, t, frames, _ = schematics.swr_exact_region(resolution=16)
    after = frames[0]["after"]
    assert _at(x, t, after, (1.0, 0.9))     # inside the tent over Omega_1
    assert _at(x, t, after, (0.4, 0.95))    # the outer boundary is known here
    assert not _at(x, t, after, (2.0, 0.5))  # the gap between two tents
    assert not _at(x, t, after, (3.0, 1.1))  # nothing above T_1 yet
    exact_rows = np.flatnonzero(after.any(axis=1))
    assert t[exact_rows.max()] == pytest.approx(1.0)


def test_black_solve_gains_the_rhomboids_the_caption_names():
    """The second solve closes the gaps and reaches T_2 over x_2 and x_4."""
    x, t, frames, _ = schematics.swr_exact_region(resolution=16)
    gained, after = frames[1]["gained"], frames[1]["after"]
    assert _at(x, t, gained, (2.0, 0.5))    # the gap the first solve left
    assert _at(x, t, gained, (2.0, 1.9))    # the tip of the rhomboid, just under T_2
    assert _at(x, t, gained, (4.0, 1.9))
    assert not _at(x, t, after, (2.0, 2.1))  # and not a step further
    # The exact region only ever grows, and reaches one more level each solve.
    areas = [frame["after"].sum() for frame in frames]
    assert areas == sorted(areas)
    horizons = [t[np.flatnonzero(frame["after"].any(axis=1)).max()] for frame in frames]
    assert horizons == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_idc_sweeps_of_a_window_converge():
    """Figure 3.4 draws sweep histories, so each sweep must beat the last."""
    per_window = schematics._pidc_sweeps()
    assert len(per_window) == schematics.PIDC_WINDOWS
    for data in per_window:
        errors = [np.max(np.abs(sweep - data["exact"])) for sweep in data["history"]]
        assert errors == sorted(errors, reverse=True)


def test_paraexp_pieces_add_up_to_the_sequential_solution():
    """Equation (3.15): the red and blue curves of Figure 3.7 sum to the solution."""
    _, particular, homogeneous, identity = schematics._paraexp_pieces()
    assert len(particular) == len(homogeneous) == schematics.PARAEXP_WINDOWS
    assert identity < 1e-10
    # Every particular piece starts from zero, as (3.13) requires.
    assert all(values[0] == 0.0 for _, values in particular)


def test_fcf_fine_solves_overlap_by_one_coarse_interval(tmp_path):
    """Figure 4.7 is Parareal with overlap: F spans two coarse intervals."""
    path = schematics.run_figure_4_7(tmp_path)
    body = json.loads(path.read_text())
    spans = body["fine_solves"]
    assert spans[0] == [0, 2]
    assert all(b - a == schematics.FCF_OVERLAP for a, b in spans)
    assert all(later[0] - earlier[0] == 1 for earlier, later in zip(spans, spans[1:]))


@pytest.mark.parametrize("name", sorted(schematics.SCHEMATIC_EXPERIMENTS))
def test_every_schematic_writes_a_figure_and_its_metrics(name, tmp_path):
    path = schematics.SCHEMATIC_EXPERIMENTS[name](tmp_path, quick=True)
    body = json.loads(path.read_text())
    assert body["figure"]
    assert body["parameters"]
    for kind in ("png", "svg"):
        assert (tmp_path / body["artifacts"][kind].split("/")[-1]).exists()
