#!/usr/bin/env python3
"""Recover the MATLAB ``parula`` palette from a published Section 2 figure panel.

The upstream scripts plot with MATLAB's default colour map, which Matplotlib does
not ship.  Rather than transcribe the table from memory, this script samples it
from a lossless panel raster embedded in the article PDF.

Usage::

    pdfimages -f 6 -l 6 -png paper.pdf /tmp/pg6
    python scripts/extract_parula.py /tmp/pg6-000.png actapint/data/parula.csv

Page 6 of the article holds Figure 2.1; its panel (a) spans the full colour
range and is stored losslessly, so its pixels trace the whole palette.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg

# Relative luminance weights (Rec. 709), used to order colours along the palette.
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def candidate_colours(path, *, min_count=20, min_chroma=0.15):
    """Return colours that fill sizeable areas and are not part of the axis decor."""
    image = mpimg.imread(path)[..., :3]
    # Trim the axis labels, tick text and white margin around the data area.
    height, width = image.shape[:2]
    patch = image[int(0.03 * height) : int(0.845 * height), int(0.23 * width) : width - 6]
    rgb = (patch * 255).round().astype(np.uint8).reshape(-1, 3)
    colours, counts = np.unique(rgb, axis=0, return_counts=True)
    scaled = colours.astype(float) / 255.0
    chroma = scaled.max(axis=1) - scaled.min(axis=1)
    keep = (counts >= min_count) & (chroma > min_chroma)
    return scaled[keep]


def reject_outliers(curve, *, window=9, tol=0.05):
    """Drop colours that do not sit on the smooth palette curve."""
    keep = np.ones(len(curve), bool)
    for i in range(len(curve)):
        lo, hi = max(0, i - window), min(len(curve), i + window + 1)
        neighbours = [j for j in range(lo, hi) if j != i]
        if len(neighbours) < 4:
            continue
        if np.linalg.norm(curve[i] - curve[neighbours].mean(axis=0)) > tol:
            keep[i] = False
    return curve[keep]


def build_table(path, *, entries=256):
    colours = candidate_colours(path)
    ordered = colours[np.argsort(colours @ LUMINANCE)]
    clean = reject_outliers(ordered)
    steps = np.linalg.norm(np.diff(clean, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    arc /= arc[-1]
    target = np.linspace(0.0, 1.0, entries)
    return np.column_stack([np.interp(target, arc, clean[:, k]) for k in range(3)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", help="lossless PNG of Figure 2.1(a)")
    parser.add_argument("output", help="destination CSV")
    parser.add_argument("--entries", type=int, default=256)
    args = parser.parse_args()

    table = build_table(args.panel, entries=args.entries)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination, table, fmt="%.6f", delimiter=",", header="r,g,b")
    print(f"wrote {len(table)} entries to {destination}")
    print(f"endpoints {table[0].round(4)} -> {table[-1].round(4)}")


if __name__ == "__main__":
    main()
