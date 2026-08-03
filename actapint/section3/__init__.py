"""Reproduction of Section 3, effective PinT methods for hyperbolic problems."""

from __future__ import annotations

from .idc import run_figure_3_5, run_figure_3_6
from .paradiag_bvm import (
    run_figure_3_12,
    run_figure_3_13,
    run_figure_3_14,
    run_table_3_2,
)
from .paradiag_direct import (
    run_figure_3_9,
    run_figure_3_10,
    run_figure_3_11,
    run_table_3_1,
)
from .paradiag_iterative import (
    run_figure_3_15,
    run_figure_3_16,
    run_figure_3_17,
    run_figure_3_18,
)
from .paraexp import run_figure_3_8
from .swr import run_figure_3_1
from .utp import run_figure_3_3

SECTION3_EXPERIMENTS = {
    "figure_3_1": run_figure_3_1,
    "figure_3_3": run_figure_3_3,
    "figure_3_5": run_figure_3_5,
    "figure_3_6": run_figure_3_6,
    "figure_3_8": run_figure_3_8,
    "figure_3_9": run_figure_3_9,
    "figure_3_10": run_figure_3_10,
    "figure_3_11": run_figure_3_11,
    "figure_3_12": run_figure_3_12,
    "figure_3_13": run_figure_3_13,
    "figure_3_14": run_figure_3_14,
    "figure_3_15": run_figure_3_15,
    "figure_3_16": run_figure_3_16,
    "figure_3_17": run_figure_3_17,
    "figure_3_18": run_figure_3_18,
    "table_3_1": run_table_3_1,
    "table_3_2": run_table_3_2,
}


def run_section3(output_dir, quick=False):
    """Run every reproducible experiment of Section 3 in figure order."""
    return [
        function(output_dir, quick=quick) for function in SECTION3_EXPERIMENTS.values()
    ]


__all__ = ["SECTION3_EXPERIMENTS", "run_section3", *SECTION3_EXPERIMENTS]
