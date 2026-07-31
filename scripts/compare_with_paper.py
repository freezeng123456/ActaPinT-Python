#!/usr/bin/env python3
"""Quantitative comparison of the reproduced Section 2 panels with the article.

Section 2 states no numbers, so a reproduction can only be checked against the
published rasters.  This script renders the article pages that carry Figures
2.1-2.4, locates each panel, inverts its pixels through the calibrated parula
table to recover normalised solution values, and compares them with the
reproduced fields.

Reported per panel:

``pearson`` / ``rms``
    pointwise agreement of the recovered and reproduced normalised fields.  These
    are only meaningful for smooth panels: where the solution carries stripes
    close to the pixel pitch, the published raster aliases them and pointwise
    statistics collapse even for an exact reproduction.
``pearson_smoothed`` / ``rms_smoothed``
    the same statistics after Gaussian low-pass filtering, which removes the
    aliasing sensitivity and measures whether the large-scale structure agrees.
``decile_mad``
    mean absolute difference of the two value distributions' deciles.  This is
    invariant to any spatial shift, so it separates "same solution, aliased
    rendering" from "different solution".
``within_2_steps``
    fraction of pixels agreeing to within two colour-map entries.

Usage::

    python scripts/compare_with_paper.py paper.pdf --output docs/data/section2_paper_comparison.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg
from scipy import ndimage

from actapint.colormaps import parula
from actapint.section2 import (
    FIGURE_2_1,
    FIGURE_2_2,
    FIGURE_2_3,
    FIGURE_2_4,
    solve_burgers,
    solve_heat_ade,
    solve_wave,
)

# The article starts on printed page 385, so page 390 is the sixth PDF page.
FIGURE_PAGES = {"2.1": 6, "2.2": 8, "2.3": 10, "2.4": 11}

# ``Solution_Heat_ADE.m`` plots every second grid point (``dxt=2``); the Burgers
# and wave scripts plot the full array.  The stride changes both the colour
# limits and which node each pixel shows, so it has to match per figure.
PLOT_STRIDES = {"2.1": 2, "2.2": 2, "2.3": 1, "2.4": 1}
RENDER_DPI = 300
MIN_CHROMA = 0.15
SMOOTHING_SIGMA = 4.0


def _correlate(a, b):
    """Pearson correlation, or None when either field is effectively constant."""
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def _show(value):
    return " const" if value is None else f"{value:6.3f}"


def render_page(pdf_path, page, directory, dpi=RENDER_DPI):
    prefix = Path(directory) / f"page{page}"
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png",
         str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
    )
    matches = sorted(Path(directory).glob(f"page{page}-*.png"))
    if not matches:
        raise FileNotFoundError(f"pdftoppm produced no output for page {page}")
    return matches[0]


def locate_panels(image, expected, *, min_area=20000):
    """Return panel bounding boxes in reading order.

    Panels are the only strongly chromatic regions on the page; axis text,
    captions and body text are achromatic.
    """
    chroma = image.max(axis=2) - image.min(axis=2)
    mask = chroma > MIN_CHROMA
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    labels, count = ndimage.label(mask)
    boxes = []
    for index in range(1, count + 1):
        rows, columns = np.where(labels == index)
        if len(rows) < min_area:
            continue
        boxes.append((rows.min(), rows.max(), columns.min(), columns.max()))
    if len(boxes) != expected:
        raise RuntimeError(f"found {len(boxes)} panels, expected {expected}")
    # Group into rows, then order left to right inside each row.
    boxes.sort(key=lambda box: box[0])
    heights = [box[1] - box[0] for box in boxes]
    tolerance = 0.5 * min(heights)
    ordered, row = [], [boxes[0]]
    for box in boxes[1:]:
        if abs(box[0] - row[0][0]) <= tolerance:
            row.append(box)
        else:
            ordered.extend(sorted(row, key=lambda item: item[2]))
            row = [box]
    ordered.extend(sorted(row, key=lambda item: item[2]))
    return ordered


def recover_values(patch, table):
    """Invert panel pixels through the palette, returning normalised values."""
    flat = patch.reshape(-1, 3)
    distances = ((flat[:, None, :] - table[None, :, :]) ** 2).sum(axis=2)
    index = distances.argmin(axis=1)
    residual = np.sqrt(distances[np.arange(len(flat)), index])
    values = index / (len(table) - 1)
    return values.reshape(patch.shape[:2]), residual.reshape(patch.shape[:2])


def reproduce(figure, spec):
    """Solve one panel with the parameters of its figure."""
    if figure == "2.1":
        return solve_heat_ade(nu=1.0, advection=0.0, **spec)
    if figure == "2.2":
        return solve_heat_ade(advection=1.0, **spec)
    if figure == "2.3":
        return solve_burgers(**spec)
    if figure == "2.4":
        return solve_wave(wave_speed_squared=0.2, **spec)
    raise ValueError(figure)


def sample_on_pixels(x, t, u, height, width, stride):
    """Resample a solution onto a panel's pixel grid, oriented as in the paper."""
    field = u[::stride, ::stride]
    xs, ts = x[::stride], t[::stride]
    px = np.linspace(xs[0], xs[-1], width)
    pt = np.linspace(ts[-1], ts[0], height)
    columns = np.clip(np.searchsorted(xs, px), 0, len(xs) - 1)
    rows = np.clip(np.searchsorted(ts, pt), 0, len(ts) - 1)
    sampled = field[np.ix_(columns, rows)].T
    lo, hi = float(field.min()), float(field.max())
    return (sampled - lo) / (hi - lo), lo, hi


def compare_figure(pdf_path, figure, panels, directory, table):
    page = FIGURE_PAGES[figure]
    image = mpimg.imread(render_page(pdf_path, page, directory))[..., :3]
    boxes = locate_panels(image, len(panels))
    stride = PLOT_STRIDES[figure]
    results = []
    for (label, spec), (top, bottom, left, right) in zip(panels, boxes):
        # Inset by a few pixels to avoid antialiased panel edges.
        patch = image[top + 3 : bottom - 2, left + 3 : right - 2]
        recovered, residual = recover_values(patch, table)
        x, t, u, _ = reproduce(figure, spec)
        mine, lo, hi = sample_on_pixels(x, t, u, *patch.shape[:2], stride)

        difference = recovered - mine
        smooth_recovered = ndimage.gaussian_filter(recovered, SMOOTHING_SIGMA)
        smooth_mine = ndimage.gaussian_filter(mine, SMOOTHING_SIGMA)
        deciles = np.arange(10, 100, 10)
        decile_mad = float(
            np.abs(
                np.percentile(recovered, deciles) - np.percentile(mine, deciles)
            ).mean()
        )
        results.append(
            {
                "panel": label,
                "boundary_condition": spec["boundary_condition"],
                "nu": spec.get("nu"),
                "source": spec["with_source"],
                "pixels": int(recovered.size),
                "reproduced_range": [lo, hi],
                "reproduced_is_constant": bool(hi - lo < 1e-12),
                "pearson": _correlate(recovered, mine),
                "rms": float(np.sqrt((difference**2).mean())),
                "pearson_smoothed": _correlate(smooth_recovered, smooth_mine),
                "rms_smoothed": float(
                    np.sqrt(((smooth_recovered - smooth_mine) ** 2).mean())
                ),
                "decile_mad": decile_mad,
                "median_abs": float(np.median(np.abs(difference))),
                "within_2_steps": float((np.abs(difference) <= 2 / 255).mean()),
                "palette_residual_median": float(np.median(residual)),
            }
        )
        entry = results[-1]
        print(
            f"  {figure}({label}) {spec['boundary_condition']:>9} "
            f"pearson {_show(entry['pearson'])}  smoothed {_show(entry['pearson_smoothed'])}  "
            f"rms {entry['rms']:.4f}  decileMAD {decile_mad:.4f}"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="path to the article PDF")
    parser.add_argument("--output", default="docs/data/section2_paper_comparison.json")
    args = parser.parse_args()

    table = parula()(np.linspace(0.0, 1.0, 256))[:, :3]
    figures = {
        "2.1": FIGURE_2_1,
        "2.2": FIGURE_2_2,
        "2.3": FIGURE_2_3,
        "2.4": FIGURE_2_4,
    }
    payload = {}
    with tempfile.TemporaryDirectory() as directory:
        for figure, panels in figures.items():
            print(f"Figure {figure}")
            payload[f"figure_{figure.replace('.', '_')}"] = compare_figure(
                args.pdf, figure, panels, directory, table
            )

    every = [panel for panels in payload.values() for panel in panels]
    smoothed = [
        panel["pearson_smoothed"]
        for panel in every
        if panel["pearson_smoothed"] is not None
    ]
    payload["summary"] = {
        "panels": len(every),
        "pearson_smoothed_min": min(smoothed),
        "pearson_smoothed_median": float(np.median(smoothed)),
        "decile_mad_max": max(panel["decile_mad"] for panel in every),
        "decile_mad_median": float(
            np.median([panel["decile_mad"] for panel in every])
        ),
        "note": (
            "Colour inversion is quantised to 256 palette entries and some panels "
            "are stored as JPEG in the PDF, so a perfect reproduction still shows "
            "a small non-zero rms. Panels whose stripes approach the raster pixel "
            "pitch are aliased in the published figure; for those, read "
            "pearson_smoothed and decile_mad rather than the pointwise values."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {destination}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
