"""Colour map used by the Section 2 space-time panels.

The published figures use MATLAB's default ``parula``, which Matplotlib does not
ship.  ``actapint/data/parula.csv`` holds a table sampled from a panel raster of
the article (see ``scripts/extract_parula.py``); ``viridis`` is the fallback when
that file is absent, since it covers the same blue-green-yellow range.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

PARULA_TABLE = Path(__file__).with_name("data") / "parula.csv"


@lru_cache(maxsize=1)
def parula():
    """Return the sampled parula colour map, or viridis if it is unavailable."""
    if not PARULA_TABLE.exists():
        return plt.get_cmap("viridis")
    table = np.loadtxt(PARULA_TABLE, delimiter=",")
    return LinearSegmentedColormap.from_list("parula", np.clip(table, 0.0, 1.0))
