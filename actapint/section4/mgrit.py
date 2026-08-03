"""MGRiT with FCF-relaxation against Parareal at equal cost.

Port of `MGRiT_Heat_ADE.m` and `MGRiT_Burgers.m`, which produce Figures 4.9 to
4.11, together with the analytical regions of Figure 4.8.  One MGRiT iteration
with FCF-relaxation uses two fine solves, so every comparison here plots one
MGRiT iteration against two Parareal iterations; that is also why the MGRiT
convergence factor is compared with the *square* of the Parareal one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..plotting import COLORS, save_figure, style_axis
from .common import (
    burgers_implicit_step,
    burgers_operators,
    burgers_propagator,
    constant_initial_iterate,
    convergence_factors,
    finish,
    linear_operator,
    linear_propagator,
    matlab_random_iterate,
    mgrit_fcf,
    NEWTON_MAX_MGRIT,
    parareal,
    sequential_solution,
    stability_function,
    truncation_error,
)

# Figure 4.8 works with the exact fine propagator, so the two factors are
# analytic functions of z alone and no discretization enters.
FIGURE_4_8_REAL = (-5.0, 0.0)
FIGURE_4_8_IMAGINARY = (-10.0, 10.0)
FIGURE_4_8_BOUNDS = (0.2, 0.4, 0.6)
FIGURE_4_8_RESOLUTION = 480
# The two constants of Theorem 4.6, for backward Euler and for LobattoIIIC-2 as
# the coarse solver.
PUBLISHED_THEOREM_4_6 = {
    "backward_euler": {"parareal": 0.2984, "mgrit": 0.1115},
    "lobatto_iiic_2": {"parareal": 0.0817, "mgrit": 0.0197},
}

# Figures 4.9 and 4.10: the parameters of `MGRiT_Heat_ADE.m`.
DX = 1.0 / 160
COARSE_DT = 0.125
SUBSTEPS = 20
T_FINAL = 5.0
ITERATIONS = 25
TOLERANCE = 2e-16
# `MGRiT_Heat_ADE.m` sets `nu=0.002max`, which is not valid MATLAB; the sweep of
# the published figures is the heat equation and these three ADE cases.
NU_VALUES = (0.1, 0.01, 0.002)
FIGURE_4_9_CASES = (("heat", -1.0), ("ade", 0.1), ("ade", 0.01))
PUBLISHED_RHO = {
    ("parareal", "heat"): 0.2822,
    ("parareal", 0.1): 0.4453,
    ("parareal", 0.01): 0.9986,
    ("parareal", 0.002): 1.4211,
    ("mgrit", "heat"): 0.08375,
    ("mgrit", 0.1): 0.2718,
    ("mgrit", 0.01): 0.9021,
    ("mgrit", 0.002): 1.2812,
}

# The published curves of Figure 4.10, read off the article page by
# `scripts/digitize_figure_4_10.py`; the calibration and the two independent
# readings per point are in docs/data/figure_4_10_published.json.  Only the
# markers the panel draws in full are kept: where the axis cuts a marker the
# reading is a lower bound, which is the first point of three of the panels.
PUBLISHED_FIGURE_4_10 = {
    ("heat", "parareal"): {
        1: 0.2224, 2: 0.0192, 3: 1.547e-3, 4: 1.246e-4, 5: 9.867e-6, 6: 7.813e-7,
        7: 6.294e-8, 8: 4.984e-9, 9: 3.947e-10, 10: 3.180e-11, 11: 2.518e-12,
        12: 2.064e-13, 13: 1.634e-14,
    },
    ("heat", "mgrit"): {
        1: 0.05692, 2: 4.747e-3, 3: 3.958e-4, 4: 3.358e-5, 5: 2.801e-6, 6: 2.335e-7,
        7: 1.948e-8, 8: 1.652e-9, 9: 1.378e-10, 10: 1.149e-11, 11: 9.582e-13,
        12: 8.129e-14,
    },
    ("ade_0.1", "parareal"): {
        0: 0.518, 1: 0.2425, 2: 0.03059, 3: 3.859e-3, 4: 4.954e-4, 5: 6.250e-5,
        6: 8.161e-6, 7: 1.048e-6, 8: 1.350e-7, 9: 1.802e-8, 10: 2.333e-9,
        11: 3.100e-10, 12: 4.048e-11, 13: 5.379e-12, 14: 7.526e-13, 15: 1.209e-13,
        16: 2.045e-14,
    },
    ("ade_0.1", "mgrit"): {
        0: 0.518, 1: 0.02619, 2: 1.657e-3, 3: 1.559e-4, 4: 2.638e-5, 5: 5.780e-6,
        6: 1.357e-6, 7: 3.241e-7, 8: 7.877e-8, 9: 1.948e-8, 10: 4.815e-9,
        11: 1.211e-9, 12: 3.047e-10, 13: 7.664e-11, 14: 1.962e-11, 15: 5.020e-12,
        16: 1.285e-12, 17: 3.403e-13, 18: 9.016e-14, 19: 2.838e-14,
    },
    ("ade_0.01", "parareal"): {
        3: 0.4214, 4: 0.3367, 5: 0.2984, 6: 0.2883, 7: 0.2883, 8: 0.2984, 9: 0.3036,
        10: 0.3197, 11: 0.2834, 12: 0.1631, 13: 0.06205, 14: 0.01643, 15: 3.416e-3,
        16: 5.483e-4, 17: 6.339e-5, 18: 4.368e-6, 19: 1.032e-7, 20: 3.654e-10,
    },
    ("ade_0.01", "mgrit"): {
        1: 0.4214, 2: 0.2555, 3: 0.1659, 4: 0.1078, 5: 0.07502, 6: 0.05221,
        7: 0.03827, 8: 0.02854, 9: 0.02128, 10: 0.01615, 11: 0.01246, 12: 9.959e-3,
        13: 8.096e-3, 14: 6.142e-3, 15: 2.777e-3, 16: 6.294e-4, 17: 6.124e-5,
        18: 1.662e-6,
    },
    ("ade_0.002", "parareal"): {
        0: 0.9837, 1: 0.6864, 2: 0.8217, 3: 1.036, 4: 1.523, 5: 2.635, 6: 4.521,
        7: 8.026, 8: 14.25, 9: 26.41, 10: 53.31, 11: 101.4, 12: 176.9,
    },
    ("ade_0.002", "mgrit"): {
        0: 0.9837, 1: 0.5196, 2: 0.4978, 3: 0.4628, 4: 0.4018, 5: 0.4123, 6: 0.4415,
        7: 0.481, 8: 0.5661, 9: 0.6548, 10: 0.7839, 11: 0.9384, 12: 1.086,
    },
}

# The four panels were not drawn by one run of the script.  Three of them come
# from the plain `sin(2 pi (1-x))**2` started from `u0` at every level, which is
# the line `U_Ini(:,n+1)=u0` the committed script carries commented out; the
# divergent one comes from the committed chirp started from the random draw, as
# the script stands.  Each setup reproduces its own panels at every digitized
# marker and misses the others by factors of up to four, so the two are not
# interchangeable.  See docs/SECTION4_REPRODUCTION.md.
FIGURE_4_10_SETUPS = {
    "heat": ("plain", "constant"),
    "ade_0.1": ("plain", "constant"),
    "ade_0.01": ("plain", "constant"),
    "ade_0.002": ("chirp", "random"),
}

# Figure 4.11: the parameters of `MGRiT_Burgers.m`, whose published panels use
# three diffusion parameters rather than the single committed one.  Only the
# first is known: `nu=1/2` is what the script sets, and it reproduces the left
# panel to three digits.  The caption names no values and the panels carry no
# titles, so the other two were chosen by matching the published curves; they
# reproduce the rates but sit a factor of two to three below the middle panel and
# about three iterations ahead of the right one.  See
# docs/SECTION4_REPRODUCTION.md.
BURGERS_DX = 1.0 / 160
BURGERS_COARSE_DT = 1.0 / 16
BURGERS_SUBSTEPS = 10
BURGERS_T_FINAL = 5.0
BURGERS_NU_VALUES = (0.5, 1.0 / 64, 0.006)
BURGERS_NU_IN_SCRIPT = 0.5
BURGERS_ITERATIONS = 60
BURGERS_TOLERANCE = 1e-14


def heat_initial_condition(x, frequency=2.0):
    """The scripts' ``sin(2 pi (1-x)**2)**2``.

    The text of Figures 4.9 to 4.11 states ``sin(8 pi (1-x)**2)**2`` instead, and
    `MGRiT_Burgers.m` does use the frequency 8; only `MGRiT_Heat_ADE.m` uses 2.
    """
    return np.sin(frequency * np.pi * (1.0 - x) ** 2) ** 2


def plain_initial_condition(x, frequency=2.0):
    """The same without the chirp, ``sin(2 pi (1-x))**2``.

    Three of the four panels of Figure 4.10 were drawn with this one; it is the
    committed line with the inner square dropped, and it is a single harmonic,
    ``(1 - cos(4 pi x))/2``, where the committed one spreads over the spectrum.
    """
    return np.sin(frequency * np.pi * (1.0 - x)) ** 2


def _factors(nu, boundary_condition):
    """Convergence factors of both methods over the spectrum of DT A."""
    operator, x = linear_operator(DX, nu, boundary_condition=boundary_condition)
    factors = convergence_factors(operator, COARSE_DT, SUBSTEPS, "sdirk22")
    fine = stability_function(
        "sdirk22", COARSE_DT * factors["eigenvalues"], substeps=SUBSTEPS
    )
    factors["rho_mgrit"] = np.abs(fine) * factors["rho_l"]
    factors["rho_mgrit_max"] = float(np.max(factors["rho_mgrit"]))
    return operator, x, factors


def run_figure_4_8(output_dir, quick=False):
    """Figure 4.8: where one MGRiT iteration beats two Parareal iterations.

    With the exact propagator for F the two factors are functions of z alone:
    ``rho_mgrit = |e^z| rho_parareal``.  Comparing ``rho_mgrit`` with the square
    of ``rho_parareal`` is the fair comparison, since both then cost two fine
    solves, and the shaded regions show that MGRiT's region is the larger one
    except in a thin strip along the imaginary axis.
    """
    resolution = 160 if quick else FIGURE_4_8_RESOLUTION
    real = np.linspace(*FIGURE_4_8_REAL, resolution)
    imaginary = np.linspace(*FIGURE_4_8_IMAGINARY, resolution)
    grid_real, grid_imaginary = np.meshgrid(real, imaginary)
    z = grid_real + 1j * grid_imaginary
    with np.errstate(divide="ignore", invalid="ignore"):
        coarse = 1.0 / (1.0 - z)
        fine = np.exp(z)
        parareal_factor = np.abs(coarse - fine) / (1.0 - np.abs(coarse))
    mgrit_factor = np.abs(fine) * parareal_factor
    quantities = (
        ("mgrit", mgrit_factor, r"$\varrho_l\leq\hat\varrho$", "MGRiT"),
        ("parareal", parareal_factor**2, r"$\varrho_l^2\leq\hat\varrho$", "Parareal"),
    )

    fig, axes = plt.subplots(
        2, len(FIGURE_4_8_BOUNDS), figsize=(4.3 * len(FIGURE_4_8_BOUNDS), 7.0),
        constrained_layout=True,
    )
    panels = []
    for row, (name, quantity, label, title) in enumerate(quantities):
        for column, bound in enumerate(FIGURE_4_8_BOUNDS):
            ax = axes[row, column]
            inside = np.isfinite(quantity) & (quantity <= bound)
            ax.contourf(
                grid_real, grid_imaginary, inside.astype(float),
                levels=[0.5, 1.5], colors=["#D9B8E8"],
            )
            ax.contour(
                grid_real, grid_imaginary, quantity, levels=[bound],
                colors=[COLORS["navy"]], linewidths=1.0,
            )
            ax.set_title(f"{title}")
            ax.set_xlabel(r"Re$(z)$")
            ax.set_ylabel(r"Im$(z)$")
            ax.text(
                0.06, 0.5, f"{label[:-1]}={bound}$".replace("$=", " = "),
                transform=ax.transAxes, fontsize=10,
            )
            style_axis(ax, grid=False)
            area = float(inside.mean())
            panels.append(
                {
                    "method": name,
                    "bound": bound,
                    "fraction_of_window_inside": area,
                    "widest_excluded_real_part": float(
                        np.max(grid_real[~inside]) if (~inside).any() else float("nan")
                    ),
                }
            )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_8_regions")

    return finish(
        output_dir,
        "figure_4_8_regions",
        "Regions of the left half-plane where one MGRiT iteration with "
        "FCF-relaxation, and two Parareal iterations, contract by at least a "
        "given factor, with backward Euler for G and the exact propagator for F",
        {
            "parameters": {
                "coarse_solver": "backward Euler, R_g = 1/(1-z)",
                "fine_solver": "exact, R_f = exp(z)",
                "real_range": list(FIGURE_4_8_REAL),
                "imaginary_range": list(FIGURE_4_8_IMAGINARY),
                "resolution": resolution,
                "bounds": list(FIGURE_4_8_BOUNDS),
            },
            "published_theorem_4_6": PUBLISHED_THEOREM_4_6,
            "panels": panels,
        },
        artifacts,
    )


def run_figure_4_9(output_dir, quick=False):
    """Figure 4.9: the two convergence factors over the discrete spectrum.

    For the heat equation and for the ADE with plenty of diffusion the MGRiT
    factor is close to the square of the Parareal one, so the two methods
    contract equally per fine solve.  Once the advection dominates both factors
    approach and then exceed one.
    """
    cases = FIGURE_4_9_CASES[:1] if quick else FIGURE_4_9_CASES

    fig = plt.figure(figsize=(4.5 * len(cases), 7.4), constrained_layout=True)
    panels = []
    for row, method in enumerate(("mgrit", "parareal")):
        for column, (equation, nu) in enumerate(cases):
            key = "heat" if equation == "heat" else nu
            boundary = "dirichlet" if equation == "heat" else "periodic"
            _, _, factors = _factors(nu, boundary)
            quantity = factors["rho_mgrit"] if method == "mgrit" else factors["rho_l"]
            maximum = (
                factors["rho_mgrit_max"] if method == "mgrit" else factors["rho_l_max"]
            )
            eigenvalues = factors["eigenvalues"]
            ax = fig.add_subplot(2, len(cases), row * len(cases) + column + 1,
                                 projection="3d")
            ax.plot(
                eigenvalues.real, eigenvalues.imag, quantity, "+",
                color="#2F6BFF", markersize=4,
            )
            ax.set_xlabel(r"Re($\lambda(A)$)", fontsize=8, labelpad=-2)
            ax.set_ylabel(r"Im($\lambda(A)$)", fontsize=8, labelpad=-2)
            ax.set_zlabel(r"$\varrho_l(J,z)$", fontsize=8, labelpad=-4)
            ax.set_zlim(0.0, 1.0)
            name = "MGRiT" if method == "mgrit" else "Parareal"
            title = (
                f"{name} (heat equation)" if equation == "heat"
                else rf"{name} (ADE with $\nu={nu:g}$)"
            )
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=6, pad=-1)
            ax.text2D(
                0.2, 0.78, rf"$\varrho_{{l,\max}}={maximum:.4g}$",
                transform=ax.transAxes, fontsize=9,
            )
            panels.append(
                {
                    "method": method,
                    "equation": equation,
                    "nu": nu,
                    "rho_max": maximum,
                    "published_rho_max": PUBLISHED_RHO[(method, key)],
                }
            )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_9_convergence_factors")

    # The divergent case is quoted in the text rather than plotted.
    divergent = {}
    if not quick:
        _, _, factors = _factors(0.002, "periodic")
        divergent = {
            "nu": 0.002,
            "parareal": factors["rho_l_max"],
            "mgrit": factors["rho_mgrit_max"],
            "published_parareal": PUBLISHED_RHO[("parareal", 0.002)],
            "published_mgrit": PUBLISHED_RHO[("mgrit", 0.002)],
        }

    return finish(
        output_dir,
        "figure_4_9_convergence_factors",
        "Convergence factors of MGRiT with FCF-relaxation and of Parareal at "
        "every eigenvalue of DT A, for the heat equation and for the "
        "advection-diffusion equation with two diffusion parameters",
        {
            "parameters": {
                "dx": DX,
                "coarse_dt": COARSE_DT,
                "substeps": SUBSTEPS,
                "coarse_solver": "backward Euler",
                "fine_solver": "SDIRK22",
                "boundary_condition": "Dirichlet for heat, periodic for ADE",
            },
            "panels": panels,
            "divergent_case": divergent,
        },
        artifacts,
    )


def _linear_history(nu, boundary_condition, iterations, panel):
    """Parareal and MGRiT error histories for one linear panel."""
    condition, iterate = FIGURE_4_10_SETUPS[panel]
    operator, x, factors = _factors(nu, boundary_condition)
    fine_matrix = linear_propagator(
        operator, COARSE_DT / SUBSTEPS, "sdirk22", substeps=SUBSTEPS
    )
    coarse_matrix = linear_propagator(operator, COARSE_DT, "ie")
    steps = int(round(T_FINAL / COARSE_DT))
    u0 = (
        heat_initial_condition(x) if condition == "chirp"
        else plain_initial_condition(x)
    )
    reference = sequential_solution(fine_matrix, u0, steps)
    start = (
        matlab_random_iterate(u0, steps) if iterate == "random"
        else constant_initial_iterate(u0, steps)
    )

    def fine(u):
        return fine_matrix @ u

    def coarse(u):
        return coarse_matrix @ u

    _, parareal_errors = parareal(
        fine, coarse, reference, start, iterations=2 * iterations, tolerance=TOLERANCE
    )
    _, mgrit_errors = mgrit_fcf(
        fine, coarse, reference, start, iterations=iterations, tolerance=TOLERANCE
    )
    return factors, parareal_errors, mgrit_errors


def _cost_matched(errors):
    """Two Parareal iterations cost one MGRiT iteration, so plot every second."""
    return errors[::2]


def _error_panel(ax, parareal_errors, mgrit_errors, truncation, title, limit=None):
    matched = _cost_matched(parareal_errors)
    ax.semilogy(
        np.arange(len(matched)), matched, "--", color="#2F6BFF", marker="o",
        markersize=6, markerfacecolor="none", linewidth=1.0, label="Parareal",
    )
    ax.semilogy(
        np.arange(len(mgrit_errors)), mgrit_errors, "--", color="#D1495B", marker="+",
        markersize=7, linewidth=1.0, label="MGRiT",
    )
    ax.axhline(truncation, color="black", linestyle="-.", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("iteration index")
    ax.set_ylabel("error")
    ax.set_xlim(0, limit if limit else max(len(matched), len(mgrit_errors)) - 1)
    style_axis(ax)
    ax.legend(fontsize=9)


def _against_published(panel, method, errors):
    """Reproduced over published, point by point, at the digitized markers."""
    published = PUBLISHED_FIGURE_4_10.get((panel, method))
    if published is None:
        return None
    ratios = {
        k: errors[k] / value
        for k, value in published.items()
        if k < len(errors) and errors[k] > 0.0
    }
    return {
        "published": published,
        "ratio_to_published": ratios,
        "worst_ratio": max(max(ratios.values()), 1.0 / min(ratios.values())),
    }


def run_figure_4_10(output_dir, quick=False):
    """Figure 4.10: measured errors of both methods at equal cost.

    The dash-dotted line is the truncation error ``max(dt**2, dx**2)``, below
    which iterating further buys nothing.  The last panel is the divergent case,
    where the ADE has so little diffusion that both methods grow.

    Every panel is also compared with the published curve point by point; the
    reading is in docs/data/figure_4_10_published.json and what the comparison
    settles, including the two setups the four panels need, is in
    docs/SECTION4_REPRODUCTION.md.
    """
    cases = (("heat", -1.0),) if quick else (("heat", -1.0), *(("ade", nu) for nu in NU_VALUES))
    iterations = 6 if quick else ITERATIONS
    columns = 2 if len(cases) > 1 else 1
    rows = int(np.ceil(len(cases) / columns))

    fig, axes = plt.subplots(
        rows, columns, figsize=(5.6 * columns, 4.3 * rows), constrained_layout=True,
        squeeze=False,
    )
    panels = []
    truncation = truncation_error(DX, COARSE_DT / SUBSTEPS)
    for index, (equation, nu) in enumerate(cases):
        boundary = "dirichlet" if equation == "heat" else "periodic"
        name = "heat" if equation == "heat" else f"ade_{nu:g}"
        factors, parareal_errors, mgrit_errors = _linear_history(
            nu, boundary, iterations, name
        )
        ax = axes[index // columns][index % columns]
        title = (
            "Heat equation" if equation == "heat" else rf"ADE with $\nu={nu:g}$"
        )
        _error_panel(ax, parareal_errors, mgrit_errors, truncation, title)
        condition, iterate = FIGURE_4_10_SETUPS[name]
        matched = _cost_matched(parareal_errors)
        panels.append(
            {
                "equation": equation,
                "nu": nu,
                "panel": name,
                "initial_condition": (
                    "sin(2 pi (1-x)**2)**2" if condition == "chirp"
                    else "sin(2 pi (1-x))**2"
                ),
                "initial_iterate": (
                    "uniform random on (0,1), MATLAB rng('default')"
                    if iterate == "random" else "u0 at every level"
                ),
                "rho_parareal_max": factors["rho_l_max"],
                "rho_mgrit_max": factors["rho_mgrit_max"],
                "truncation_error": truncation,
                "parareal_errors": parareal_errors,
                "parareal_at_matched_cost": matched,
                "mgrit_errors": mgrit_errors,
                "parareal_against_published": _against_published(
                    name, "parareal", matched
                ),
                "mgrit_against_published": _against_published(name, "mgrit", mgrit_errors),
            }
        )
    for index in range(len(cases), rows * columns):
        axes[index // columns][index % columns].axis("off")
    artifacts = save_figure(fig, f"{output_dir}/figure_4_10_measured_errors")

    return finish(
        output_dir,
        "figure_4_10_measured_errors",
        "Measured errors of Parareal and MGRiT with FCF-relaxation for the heat "
        "equation and for the advection-diffusion equation with three diffusion "
        "parameters, at equal cost in fine solves",
        {
            "parameters": {
                "dx": DX,
                "t_final": T_FINAL,
                "coarse_dt": COARSE_DT,
                "fine_dt": COARSE_DT / SUBSTEPS,
                "substeps": SUBSTEPS,
                "steps": int(round(T_FINAL / COARSE_DT)),
                "coarse_solver": "backward Euler",
                "fine_solver": "SDIRK22",
                "initial_condition_stated_in_text": "sin(8 pi (1-x)**2)**2",
                "initial_condition_and_iterate": "per panel, see the panels below",
                "tolerance": TOLERANCE,
            },
            "panels": panels,
        },
        artifacts,
    )


def _burgers_history(nu, iterations):
    """Parareal and MGRiT error histories for one Burgers panel."""
    a1, a2, x = burgers_operators(BURGERS_DX, nu, boundary_condition="dirichlet")
    steps = int(round(BURGERS_T_FINAL / BURGERS_COARSE_DT))
    tolerance = BURGERS_TOLERANCE / 3.0
    fine = burgers_propagator(
        a1, a2, BURGERS_COARSE_DT / BURGERS_SUBSTEPS, BURGERS_SUBSTEPS, tolerance,
        newton_max=NEWTON_MAX_MGRIT,
    )

    def coarse(u):
        return burgers_implicit_step(
            a1, a2, BURGERS_COARSE_DT, u, tolerance, NEWTON_MAX_MGRIT
        )

    u0 = heat_initial_condition(x, 8.0)
    reference = sequential_solution(fine, u0, steps)
    start = matlab_random_iterate(u0, steps)
    _, parareal_errors = parareal(
        fine, coarse, reference, start, iterations=2 * iterations,
        tolerance=BURGERS_TOLERANCE,
    )
    # `MGRiT_Burgers.m` starts the first coarse point from the fine solution
    # instead of the random draw, which `mgrit_fcf` reproduces.
    _, mgrit_errors = mgrit_fcf(
        fine, coarse, reference, start, iterations=iterations,
        tolerance=BURGERS_TOLERANCE,
    )
    return parareal_errors, mgrit_errors


def run_figure_4_11(output_dir, quick=False):
    """Figure 4.11: the same comparison for Burgers' equation.

    The conclusion of the linear panels survives the nonlinearity: one MGRiT
    iteration with FCF-relaxation contracts like two Parareal iterations, as long
    as the coarse solver still resolves the dynamics.
    """
    nus = BURGERS_NU_VALUES[:1] if quick else BURGERS_NU_VALUES
    iterations = 5 if quick else 30

    fig, axes = plt.subplots(
        1, len(nus), figsize=(4.9 * len(nus), 4.3), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = []
    truncation = truncation_error(BURGERS_DX, BURGERS_COARSE_DT / BURGERS_SUBSTEPS)
    for ax, nu in zip(axes, nus):
        parareal_errors, mgrit_errors = _burgers_history(nu, iterations)
        _error_panel(
            ax, parareal_errors, mgrit_errors, truncation,
            rf"Burgers' equation with $\nu={nu:g}$",
        )
        panels.append(
            {
                "nu": nu,
                "truncation_error": truncation,
                "parareal_errors": parareal_errors,
                "parareal_at_matched_cost": _cost_matched(parareal_errors),
                "mgrit_errors": mgrit_errors,
            }
        )
    artifacts = save_figure(fig, f"{output_dir}/figure_4_11_burgers")

    return finish(
        output_dir,
        "figure_4_11_burgers",
        "Measured errors of Parareal and MGRiT with FCF-relaxation for Burgers' "
        "equation with three diffusion parameters, at equal cost in fine solves",
        {
            "parameters": {
                "dx": BURGERS_DX,
                "t_final": BURGERS_T_FINAL,
                "coarse_dt": BURGERS_COARSE_DT,
                "fine_dt": BURGERS_COARSE_DT / BURGERS_SUBSTEPS,
                "substeps": BURGERS_SUBSTEPS,
                "steps": int(round(BURGERS_T_FINAL / BURGERS_COARSE_DT)),
                "coarse_solver": "implicit Euler with Newton",
                "fine_solver": "SDIRK22 with Newton",
                "initial_condition": "sin(8 pi (1-x)**2)**2",
                "boundary_condition": "homogeneous Dirichlet",
                "burgers_flux": "(u**2)_x, not the 0.5 (u**2)_x of the script",
                "tolerance": BURGERS_TOLERANCE,
                "newton_max": NEWTON_MAX_MGRIT,
                "nu_in_script": BURGERS_NU_IN_SCRIPT,
                "nu_of_other_panels": "inferred, the paper states no values",
            },
            "panels": panels,
        },
        artifacts,
    )
