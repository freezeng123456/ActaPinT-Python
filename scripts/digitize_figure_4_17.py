#!/usr/bin/env python3
"""Read the published iteration counts of Figure 4.17 off the article page.

Figure 4.17 prints no numbers, and the reproduction of Section 4.5 turns on
differences of a single iteration, so the published counts have to be recovered
from the raster rather than eyeballed.  They are integers on a small logarithmic
axis, which makes the reading decisive: every marker of both panels lands within
0.05 of an integer.

The pipeline is the one `compare_with_paper.py` uses for Section 2, specialized
to a line plot.  The page is rendered at 1200 dpi; the axis frame is the pair of
row and column runs that are dark across the whole panel; the tick marks are the
inward stubs on the bottom and left spines, and they carry the calibration.  The
three curves are separated by colour -- the article draws them pure red, pure
blue and black -- and each marker is located as the peak of the local density of
its colour, which is what distinguishes a marker from the dash-dot line drawn
through it: an open circle, a plus or an open square fills a third to a half of a
window of its own size, a line of the same stroke width fills a fifth.

``--validate`` runs the same extraction on a synthetic figure of known counts
drawn with the same axis scales, markers and legend placement, which is what
fixes the density threshold and shows that the reading is exact.

Usage::

    python scripts/digitize_figure_4_17.py paper.pdf --output docs/data/figure_4_17_published.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = None

# The article starts on printed page 385, so printed page 471 is the 87th.
PAGE = 87
RENDER_DPI = 1200
DARK = 0.55
DENSITY_WINDOW = 51
DENSITY_THRESHOLD = 0.28

# The alphas of panel (a) are not printed.  The marker columns come out evenly
# spaced by 0.600 decades, that is by a factor four, and the sweep then lands on
# the powers of two below; see the calibration the script reports.
ALPHA_EXPONENTS = (13, 11, 9, 7, 5, 3)
STEPS = (10, 20, 40, 80, 160)
CURVES = ("nu=1", "nu=0.01", "nu=0.0001")

# Column windows of the two panels on the rendered page, wide enough to contain
# the axis frame and nothing of the neighbouring panel.
PANEL_COLUMNS = {"a": (1050, 3700), "b": (4120, 6760)}


def render_page(pdf_path, directory, dpi=RENDER_DPI):
    prefix = Path(directory) / "page"
    subprocess.run(
        ["pdftoppm", "-f", str(PAGE), "-l", str(PAGE), "-r", str(dpi), "-png",
         str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
    )
    matches = sorted(Path(directory).glob("page-*.png"))
    if not matches:
        raise FileNotFoundError(f"pdftoppm produced no output for page {PAGE}")
    return matches[0]


def _run_centres(index):
    """Centre of every maximal run of consecutive integers."""
    centres, start, previous = [], index[0], index[0]
    for value in index[1:]:
        if value != previous + 1:
            centres.append(0.5 * (start + previous))
            start = value
        previous = value
    return centres + [0.5 * (start + previous)]


def panel_frame(dark, columns):
    """The four spines of one panel, as sub-pixel row and column positions."""
    left_edge, right_edge = columns
    rows = dark[:, left_edge:right_edge].sum(axis=1)
    index = np.where(rows > 0.8 * (right_edge - left_edge))[0]
    top, bottom = _run_centres(index)
    band = dark[int(top):int(bottom) + 1]
    index = np.where(band.sum(axis=0) > 0.8 * (bottom - top))[0]
    index = index[(index >= left_edge - 80) & (index <= right_edge + 80)]
    left, right = _run_centres(index)
    return top, bottom, left, right


def axis_ticks(frame, dark, *, x_depth=14, y_depth=22, margin=30):
    """The inward tick stubs of the bottom and the left spine.

    The two depths differ because the article's ticks are longer on the value
    axis than on the parameter axis; both are read as the columns, respectively
    rows, that are dark all the way through a band of that depth, which excludes
    the curves crossing the band and the legend frame further inside.
    """
    top, bottom, left, right = frame
    band = dark[int(bottom) - x_depth - 3:int(bottom) - 3, int(left) + 6:int(right) - 5]
    columns = _run_centres(np.where(band.all(axis=0))[0] + int(left) + 6)
    band = dark[int(top) + 5:int(bottom) - 4, int(left) + 4:int(left) + 4 + y_depth]
    rows = _run_centres(np.where(band.all(axis=1))[0] + int(top) + 5)
    # A run flush against a spine is the spine itself, not a tick.
    columns = [value for value in columns
               if left + margin < value < right - margin]
    rows = [value for value in rows if top + margin < value < bottom - margin]
    return columns, rows


def legend_box(dark, frame, *, minimum=300):
    """Bounding box of the legend frame, which has to be kept out of the curves."""
    top, bottom, left, right = frame
    inside = dark[int(top) + 8:int(bottom) - 8, int(left) + 8:int(right) - 8]
    filled = ndimage.binary_closing(inside, np.ones((1, 9)))
    labels, count = ndimage.label(filled, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    widths = ndimage.sum(filled, labels, range(1, count + 1))
    edges = [
        objects
        for objects, width in zip(ndimage.find_objects(labels), widths)
        if width >= minimum
    ]
    if not edges:
        raise RuntimeError("no legend frame found")
    rows = [slices[0].start for slices in edges] + [slices[0].stop for slices in edges]
    columns = [slices[1].start for slices in edges] + [slices[1].stop for slices in edges]
    return (
        min(rows) + int(top) + 8,
        max(rows) + int(top) + 8,
        min(columns) + int(left) + 8,
        max(columns) + int(left) + 8,
    )


def log_axis(positions, values):
    """Fit ``log(value) = a + b * position`` and report how well it holds.

    The residual is the check on the tick labels: the panels are read as if the
    value axis carried the consecutive powers of two, respectively the
    consecutive integers, up from 2, and if that reading were wrong the uneven
    spacing of a logarithmic axis would not fit a straight line.
    """
    coefficients = np.polyfit(positions, np.log(values), 1)
    residual = float(
        np.abs(np.polyval(coefficients, positions) - np.log(values)).max()
    )
    return coefficients, residual


def curve_masks(patch):
    """The three curves of the figure, separated by colour."""
    red, green, blue = patch[..., 0], patch[..., 1], patch[..., 2]
    return {
        "nu=1": (red - np.maximum(green, blue) > 0.2) & (red > 0.4),
        "nu=0.01": (blue - np.maximum(green, red) > 0.2) & (blue > 0.4),
        "nu=0.0001": (patch.max(axis=2) < 0.6)
        & (patch.max(axis=2) - patch.min(axis=2) < 0.15),
    }


def marker_rows(image, frame, legend, columns):
    """Row of every marker of every curve, at the given data columns."""
    top, bottom, left, right = frame
    r0, r1 = int(top) + 3, int(bottom) - 2
    c0, c1 = int(left) + 3, int(right) - 2
    rows, peaks = {}, {}
    for label, mask in curve_masks(image[r0:r1, c0:c1]).items():
        mask = ndimage.binary_closing(mask, np.ones((5, 5)))
        mask[legend[0] - r0:legend[1] - r0, legend[2] - c0:legend[3] - c0] = False
        density = ndimage.uniform_filter(mask.astype(float), size=DENSITY_WINDOW)
        found, height = [], []
        for column in columns:
            strip = density[:, int(column) - c0 - 6:int(column) - c0 + 7].mean(axis=1)
            peak = int(strip.argmax())
            window = slice(max(peak - 30, 0), peak + 31)
            weights = strip[window]
            centre = np.arange(len(density))[window]
            found.append(float((weights * centre).sum() / weights.sum()) + r0)
            height.append(float(strip[peak]))
        rows[label] = found
        peaks[label] = height
    return rows, peaks


def read_panels(image):
    """Both panels of Figure 4.17, with the calibration each reading rests on."""
    dark = image.max(axis=2) < DARK
    report, tables = {}, {}

    # Panel (a): the parameter axis carries one tick per decade of alpha from
    # 1e-4 to 1e-1, the value axis the powers of two up from 2.
    frame = panel_frame(dark, PANEL_COLUMNS["a"])
    columns, rows = axis_ticks(frame, dark)
    parameter, x_residual = log_axis(
        columns, [10.0**-exponent for exponent in range(4, 4 - len(columns), -1)]
    )
    value, y_residual = log_axis(rows, [2 ** (len(rows) - i) for i in range(len(rows))])
    marker_columns = [
        (np.log(2.0**-exponent) - parameter[1]) / parameter[0]
        for exponent in ALPHA_EXPONENTS
    ]
    legend = legend_box(dark, frame)
    found, peaks = marker_rows(image, frame, legend, marker_columns)
    tables["a"] = {
        label: [float(np.exp(np.polyval(value, row))) for row in values]
        for label, values in found.items()
    }
    report["a"] = {
        "frame": [float(item) for item in frame],
        "legend": [int(item) for item in legend],
        "x_ticks": [float(item) for item in columns],
        "y_ticks": [float(item) for item in rows],
        "pixels_per_decade_in_x": float(np.log(10.0) / parameter[0]),
        "pixels_per_doubling_in_y": float(np.log(2.0) / value[0]),
        "x_tick_fit_residual": x_residual,
        "y_tick_fit_residual": y_residual,
        "marker_columns": [float(item) for item in marker_columns],
        "density_peaks": peaks,
    }

    # Panel (b): the five values of N_t sit on the ticks, and the value axis
    # carries the consecutive integers up from 2, which is what their uneven
    # spacing identifies -- the gaps are in the ratio log(3/2) : log(4/3).
    frame = panel_frame(dark, PANEL_COLUMNS["b"])
    columns, rows = axis_ticks(frame, dark)
    value, y_residual = log_axis(rows, [len(rows) + 1 - i for i in range(len(rows))])
    legend = legend_box(dark, frame)
    found, peaks = marker_rows(image, frame, legend, columns)
    tables["b"] = {
        label: [float(np.exp(np.polyval(value, row))) for row in values]
        for label, values in found.items()
    }
    report["b"] = {
        "frame": [float(item) for item in frame],
        "legend": [int(item) for item in legend],
        "x_ticks": [float(item) for item in columns],
        "y_ticks": [float(item) for item in rows],
        "pixels_per_log_unit_in_y": float(-1.0 / value[0]),
        "y_tick_fit_residual": y_residual,
        "density_peaks": peaks,
    }
    return tables, report


def synthetic_page(path, counts_a, counts_b):
    """Redraw Figure 4.17 with known counts, in the article's own conventions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    alphas = [2.0**-exponent for exponent in ALPHA_EXPONENTS]
    styles = (("red", "o"), ("blue", "+"), ("black", "s"))
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 1.94), dpi=100)
    for ax, values, x in ((axes[0], counts_a, alphas), (axes[1], counts_b, STEPS)):
        for label, (colour, marker) in zip(CURVES, styles):
            ax.plot(
                x, values[label], "-.", color=colour, marker=marker, markersize=3.4,
                markerfacecolor="none", linewidth=0.5, label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.legend(fontsize=4, loc="upper left", fancybox=False, edgecolor="black")
        ax.tick_params(direction="in", labelsize=5, length=2.0)
        ax.minorticks_off()
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
    axes[0].set_yscale("log", base=2)
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    axes[0].set_xlim(10**-4.09, 10**-0.73)
    axes[0].set_yticks([2, 4, 8, 16])
    axes[0].set_ylim(1.5, 23.0)
    axes[1].set_xticks(list(STEPS))
    axes[1].set_xlim(8.4, 190.0)
    axes[1].set_yticks([2, 3, 4, 5, 6])
    axes[1].set_ylim(1.5, 6.0)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.2, wspace=0.45)
    fig.savefig(path, dpi=RENDER_DPI, facecolor="white")
    width = fig.get_figwidth() * RENDER_DPI
    windows = {
        name: (int(ax.get_position().x0 * width) - 12,
               int(ax.get_position().x1 * width) + 12)
        for name, ax in zip(("a", "b"), axes)
    }
    plt.close(fig)
    return windows


def validate():
    """Extract a synthetic figure of known counts through the same pipeline."""
    planted_a = {"nu=1": [2, 2, 2, 3, 4, 5], "nu=0.01": [3, 4, 4, 6, 9, 17],
                 "nu=0.0001": [3, 4, 5, 7, 11, 21]}
    planted_b = {"nu=1": [2, 2, 2, 2, 2], "nu=0.01": [3, 4, 4, 4, 4],
                 "nu=0.0001": [4, 4, 4, 5, 5]}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.png"
        windows = synthetic_page(path, planted_a, planted_b)
        image = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    global PANEL_COLUMNS
    keep = PANEL_COLUMNS
    PANEL_COLUMNS = windows
    try:
        tables, report = read_panels(image)
    finally:
        PANEL_COLUMNS = keep
    worst = 0.0
    for panel in ("a", "b"):
        print(f"  ({panel}) frame {report[panel]['frame']} "
              f"x ticks {report[panel]['x_ticks']} y ticks {report[panel]['y_ticks']}")
    for panel, planted in (("a", planted_a), ("b", planted_b)):
        for label, values in tables[panel].items():
            for read, expected in zip(values, planted[label]):
                worst = max(worst, abs(read - expected))
            print(f"  ({panel}) {label:<12}"
                  + "".join(f"{value:8.3f}" for value in values))
    print(f"  worst deviation from the planted integers: {worst:.3f}")
    return worst, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", help="path to the article PDF")
    parser.add_argument("--output", default="docs/data/figure_4_17_published.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    validation = None
    if args.validate:
        print("synthetic figure of known counts:")
        worst, _ = validate()
        validation = {"worst_deviation_from_planted_integers": worst}
    if args.pdf is None:
        return

    with tempfile.TemporaryDirectory() as directory:
        path = render_page(args.pdf, directory)
        image = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    tables, report = read_panels(image)

    alphas = [2.0**-exponent for exponent in ALPHA_EXPONENTS]
    print("\npanel (a), N_t = 40")
    print("  alpha       " + "".join(f"{value:>10.4g}" for value in alphas))
    for label in CURVES:
        print(f"  {label:<12}" + "".join(f"{v:>10.3f}" for v in tables["a"][label]))
    print("\npanel (b), alpha = 1e-3")
    print("  N_t         " + "".join(f"{value:>10}" for value in STEPS))
    for label in CURVES:
        print(f"  {label:<12}" + "".join(f"{v:>10.3f}" for v in tables["b"][label]))

    residual = max(
        abs(value - round(value))
        for panel in tables.values()
        for values in panel.values()
        for value in values
    )
    print(f"\nlargest distance from an integer: {residual:.3f}")

    payload = {
        "source": f"page {PAGE} of the article rendered at {RENDER_DPI} dpi",
        "alphas": alphas,
        "steps": list(STEPS),
        "panel_a": {label: [round(value) for value in tables["a"][label]]
                    for label in CURVES},
        "panel_b": {label: [round(value) for value in tables["b"][label]]
                    for label in CURVES},
        "raw": {panel: {label: [round(value, 4) for value in values]
                        for label, values in table.items()}
                for panel, table in tables.items()},
        "largest_distance_from_an_integer": residual,
        "calibration": report,
        "validation": validation,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
