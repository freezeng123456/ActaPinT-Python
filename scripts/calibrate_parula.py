#!/usr/bin/env python3
"""Calibrate the parula table against published panels using the reproduced fields.

``scripts/extract_parula.py`` recovers the palette *path* by sorting panel colours
by luminance, but its parameterization is only approximate because the sorted
samples carry noise.  This script fixes the parameterization: it solves panels
(a), (b) and (c) of Figure 2.1 on the paper's grid, maps every pixel of the
published rasters to a solution value, and reads off the colour MATLAB assigned.

Panels are pooled because ``mesh`` blends neighbouring colours inside one pixel,
which biases any single panel wherever its field is steep.  Panel (d) is a
constant field and is therefore held out as an independent check.

Usage::

    pdfimages -f 6 -l 6 -png paper.pdf /tmp/pg6
    python scripts/calibrate_parula.py /tmp/pg6 actapint/data/parula.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg

from actapint.section2 import solve_heat_ade

# The upstream heat script plots every second grid point in both directions.
PLOT_STRIDE = 2


def data_area(image, *, min_chroma=0.15, inset=3):
    """Bounding box of the coloured axes interior, excluding white margin and text."""
    chroma = image.max(axis=2) - image.min(axis=2)
    rows = np.where((chroma > min_chroma).any(axis=1))[0]
    columns = np.where((chroma > min_chroma).any(axis=0))[0]
    return (
        rows[0] + inset,
        rows[-1] - inset,
        columns[0] + inset,
        columns[-1] - inset,
    )


CALIBRATION_PANELS = (
    ("a", "dirichlet", "000"),
    ("b", "neumann", "004"),
    ("c", "periodic", "008"),
)
HOLDOUT_PANEL = ("d", "periodic", "012")


def panel_samples(panel_path, boundary_condition, *, with_source=True):
    """Pair every data pixel of a published panel with its solution value."""
    x, t, u, _ = solve_heat_ade(
        nu=1.0,
        advection=0.0,
        boundary_condition=boundary_condition,
        with_source=with_source,
    )
    field = u[::PLOT_STRIDE, ::PLOT_STRIDE]
    xs, ts = x[::PLOT_STRIDE], t[::PLOT_STRIDE]
    lo, hi = float(field.min()), float(field.max())

    image = mpimg.imread(panel_path)[..., :3]
    top, bottom, left, right = data_area(image)
    patch = image[top : bottom + 1, left : right + 1]
    height, width = patch.shape[:2]

    # Pixel centres map linearly onto the axes limits; t grows upwards.
    px = np.linspace(xs[0], xs[-1], width)
    pt = np.linspace(ts[-1], ts[0], height)
    column_index = np.clip(np.searchsorted(xs, px), 0, len(xs) - 1)
    row_index = np.clip(np.searchsorted(ts, pt), 0, len(ts) - 1)
    sampled = field[np.ix_(column_index, row_index)].T

    values = ((sampled - lo) / (hi - lo)).reshape(-1)
    colours = patch.reshape(-1, 3)
    # Drop achromatic pixels: axis decorations bleeding into the inset box.
    chroma = colours.max(axis=1) - colours.min(axis=1)
    keep = chroma > 0.15
    return values[keep], colours[keep], (lo, hi)


def calibrate(prefix, *, entries=256):
    values, colours = [], []
    for label, boundary_condition, index in CALIBRATION_PANELS:
        panel_values, panel_colours, (lo, hi) = panel_samples(
            f"{prefix}-{index}.png", boundary_condition
        )
        print(
            f"panel 2.1({label}) {boundary_condition:>9}: "
            f"{len(panel_values):>7} pixels, field range [{lo:.6f}, {hi:.6f}]"
        )
        values.append(panel_values)
        colours.append(panel_colours)
    values = np.concatenate(values)
    colours = np.concatenate(colours)

    bins = np.clip((values * entries).astype(int), 0, entries - 1)
    table = np.zeros((entries, 3))
    populated = np.zeros(entries, bool)
    for index in range(entries):
        members = colours[bins == index]
        if len(members) >= 5:
            table[index] = np.median(members, axis=0)
            populated[index] = True

    grid = np.arange(entries)
    for channel in range(3):
        table[:, channel] = np.interp(grid, grid[populated], table[populated, channel])
    print(f"calibrated bins: {populated.sum()}/{entries}")
    return table


def validate(table, prefix):
    """Panel 2.1(d) is uniform, so its colour pins one exact palette position."""
    label, boundary_condition, index = HOLDOUT_PANEL
    image = mpimg.imread(f"{prefix}-{index}.png")[..., :3]
    top, bottom, left, right = data_area(image)
    patch = image[top : bottom + 1, left : right + 1].reshape(-1, 3)
    unique, counts = np.unique(
        (patch * 255).round().astype(np.uint8), axis=0, return_counts=True
    )
    dominant = unique[np.argmax(counts)] / 255.0

    x, t, u, _ = solve_heat_ade(
        nu=1.0,
        advection=0.0,
        boundary_condition=boundary_condition,
        with_source=False,
    )
    field = u[::PLOT_STRIDE, ::PLOT_STRIDE]
    expected = (u[:, -1].mean() - field.min()) / (field.max() - field.min())
    recovered = float(np.argmin(np.linalg.norm(table - dominant, axis=1))) / (
        len(table) - 1
    )
    print(
        f"holdout 2.1({label}): expected position {expected:.4f}, "
        f"recovered {recovered:.4f}, error {abs(expected - recovered):.4f}"
    )
    return abs(expected - recovered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", help="pdfimages prefix for page 6, e.g. /tmp/pg6")
    parser.add_argument("output", help="destination CSV")
    parser.add_argument("--entries", type=int, default=256)
    args = parser.parse_args()

    table = calibrate(args.prefix, entries=args.entries)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, table, fmt="%.6f", delimiter=",", header="r,g,b")
    print(f"wrote {len(table)} entries to {destination}")
    print(f"endpoints {table[0].round(4)} -> {table[-1].round(4)}")
    steps = np.linalg.norm(np.diff(table, axis=0), axis=1)
    print(f"max step {steps.max():.4f}  median step {np.median(steps):.5f}")
    validate(table, args.prefix)


if __name__ == "__main__":
    main()
