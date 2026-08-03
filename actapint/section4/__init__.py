"""Reproduction of Section 4, PinT methods designed for parabolic problems."""

from __future__ import annotations

from .diagg import (
    run_figure_4_14,
    run_figure_4_15,
    run_figure_4_16,
    run_figure_4_17,
)
from .mgrit import (
    run_figure_4_8,
    run_figure_4_9,
    run_figure_4_10,
    run_figure_4_11,
)
from .pfasst import run_figure_4_6
from .stmg import (
    run_figure_4_18,
    run_figure_4_19,
    run_figure_4_20,
    run_figure_4_21,
    run_figure_4_22,
)
from .parareal import (
    run_figure_4_2,
    run_figure_4_3,
    run_figure_4_4,
    run_figure_4_5,
)
from .pcgc import run_figure_4_12, run_figure_4_13

SECTION4_EXPERIMENTS = {
    "figure_4_2": run_figure_4_2,
    "figure_4_3": run_figure_4_3,
    "figure_4_4": run_figure_4_4,
    "figure_4_5": run_figure_4_5,
    "figure_4_6": run_figure_4_6,
    "figure_4_8": run_figure_4_8,
    "figure_4_9": run_figure_4_9,
    "figure_4_10": run_figure_4_10,
    "figure_4_11": run_figure_4_11,
    "figure_4_12": run_figure_4_12,
    "figure_4_13": run_figure_4_13,
    "figure_4_14": run_figure_4_14,
    "figure_4_15": run_figure_4_15,
    "figure_4_16": run_figure_4_16,
    "figure_4_17": run_figure_4_17,
    "figure_4_18": run_figure_4_18,
    "figure_4_19": run_figure_4_19,
    "figure_4_20": run_figure_4_20,
    "figure_4_21": run_figure_4_21,
    "figure_4_22": run_figure_4_22,
}


def run_section4(output_dir, quick=False):
    """Run every reproducible experiment of Section 4 in figure order."""
    return [
        function(output_dir, quick=quick) for function in SECTION4_EXPERIMENTS.values()
    ]


__all__ = ["SECTION4_EXPERIMENTS", "run_section4", *SECTION4_EXPERIMENTS]
