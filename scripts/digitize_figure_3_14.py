#!/usr/bin/env python3
"""Read the published error histories of Figure 3.14 off the article page.

Figure 3.14 prints no numbers, and what the reproduction of Section 3.5.1 turns
on is not the shape of the curves but the factor by which each of them decays per
iteration, so the published curves have to be measured rather than eyeballed.

Unlike the rasterized panels of Section 2 and the line plots of Section 4, this
figure is stored in the PDF as vector art: the two quasi-Newton variants are
drawn as two dotted and two dash-dot polylines whose vertices are exactly the
numbers Matlab plotted.  The reading is therefore lossless up to the affine map
back to data coordinates, and that map is taken from the tick marks, not from the
axis frame -- the frame of panel (a) sits half a decade above the topmost tick.
The curves are separated by colour, the article draws the averaged variant pure
blue and the NKA variant pure red, and the two time intervals of each variant are
separated by their marker glyphs: a plus and a cross on the blue curves, a circle
and a square on the red ones, in the order the legend lists them.

``--validate`` runs the same extraction on a synthetic figure of known histories
drawn with the same axis scales, markers, line styles and legend placement, which
is what shows that the reading is exact rather than merely repeatable.

Usage::

    python scripts/digitize_figure_3_14.py paper.pdf --output docs/data/figure_3_14_published.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

# The article starts on printed page 385, so printed page 430 is the 46th.
PAGE = 46
BLUE = "stroke:rgb(0%,0%,100%)"
RED = "stroke:rgb(100%,0%,0%)"

# Tick labels of the two panels, which carry the whole calibration.
PANELS = (
    {"panel": "a", "nu": 0.02,
     "x_ticks": tuple(range(0, 19, 2)), "y_ticks": tuple(range(0, -7, -1))},
    {"panel": "b", "nu": 0.002,
     "x_ticks": tuple(range(0, 46, 5)), "y_ticks": tuple(range(2, -7, -1))},
)

# The legend of both panels, top to bottom.  Each entry is drawn with its own
# glyph, which is how a polyline is attributed to a variant and a horizon.
LEGEND = (
    ("plus", "average", 1.3),
    ("cross", "average", 0.7),
    ("circle", "nka", 1.3),
    ("square", "nka", 0.7),
)


def page_svg(pdf_path, page=PAGE):
    """The page as SVG, with the drawing kept as paths rather than pixels."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "page.svg"
        subprocess.run(
            ["pdftocairo", "-f", str(page), "-l", str(page), "-svg",
             str(pdf_path), str(target)],
            check=True,
            capture_output=True,
        )
        return target.read_text(encoding="utf-8")


def _vertices(data, matrix):
    """Vertices of every M...L run of one path, in page coordinates."""
    a, b, c, d, e, f = matrix
    runs, current = [], []
    for token in re.finditer(r"([MLCZ])([^MLCZ]*)", data):
        command = token.group(1)
        numbers = [float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", token.group(2))]
        if command == "M":
            if len(current) > 1:
                runs.append(current)
            current = [(numbers[0], numbers[1])]
        elif command == "L":
            current.extend(zip(numbers[0::2], numbers[1::2]))
        elif command == "C":
            # Only the on-curve endpoints are needed; markers are classified by
            # their extent, not by their outline.
            current.extend(zip(numbers[4::6], numbers[5::6]))
    if len(current) > 1:
        runs.append(current)
    mapped = []
    for run in runs:
        points = np.asarray(run)
        mapped.append(np.column_stack([
            a * points[:, 0] + c * points[:, 1] + e,
            b * points[:, 0] + d * points[:, 1] + f,
        ]))
    return mapped


def parse_paths(svg):
    """Every stroked path of the page as (style, list of vertex runs)."""
    out = []
    for attributes in re.findall(r"<path([^>]*?)/>", svg, re.S):
        data = re.search(r'\sd="([^"]*)"', attributes)
        style = re.search(r'style="([^"]*)"', attributes)
        if data is None or style is None or "stroke:none" in style.group(1):
            continue
        transform = re.search(r'transform="matrix\(([^)]*)\)"', attributes)
        matrix = ([float(v) for v in transform.group(1).split(",")]
                  if transform else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        out.append((style.group(1), _vertices(data.group(1), matrix)))
    return out


def segments(paths, coloured=False):
    """Every straight piece of every axis path, as pairs of endpoints.

    Matlab draws the axis frame as four separate lines and Matplotlib as one
    closed rectangle, so both are decomposed into their pieces here.
    """
    out = []
    for style, runs in paths:
        if (BLUE in style or RED in style) != coloured:
            continue
        for run in runs:
            out.extend(zip(run[:-1], run[1:]))
    return out


def panel_boxes(pieces):
    """The axes boxes of the page, as (left, right, top, bottom)."""
    horizontal, vertical = [], []
    for (x0, y0), (x1, y1) in pieces:
        if abs(y1 - y0) < 0.05 and abs(x1 - x0) > 50:
            horizontal.append((round(min(x0, x1), 2), round(max(x0, x1), 2), y0))
        if abs(x1 - x0) < 0.05 and abs(y1 - y0) > 40:
            vertical.append((round(min(y0, y1), 2), round(max(y0, y1), 2), x0))
    boxes = []
    for left, right in sorted({(item[0], item[1]) for item in horizontal}):
        rows = sorted({item[2] for item in horizontal
                       if abs(item[0] - left) < 0.05 and abs(item[1] - right) < 0.05})
        columns = sorted({item[2] for item in vertical
                          if left - 0.05 <= item[2] <= right + 0.05})
        if len(rows) == 2 and len(columns) == 2:
            boxes.append((columns[0], columns[1], rows[0], rows[1]))
    return sorted(set(boxes))


def ticks(pieces, box, axis):
    """Major tick positions on one spine, separated from the minor ones.

    A logarithmic Matlab axis carries eight minor ticks per decade, drawn
    shorter than the labelled ones; the split is on tick length.
    """
    left, right, top, bottom = box
    found = []
    for (x0, y0), (x1, y1) in pieces:
        length = np.hypot(x1 - x0, y1 - y0)
        if length > 8 or length < 0.2:
            continue
        if not (left - 2 <= x0 <= right + 2 and top - 2 <= y0 <= bottom + 2):
            continue
        if axis == "x" and abs(x1 - x0) < 0.1 and abs(max(y0, y1) - bottom) < 2:
            found.append((x0, length))
        if axis == "y" and abs(y1 - y0) < 0.1 and abs(min(x0, x1) - left) < 2:
            found.append((y0, length))
    if not found:
        return np.array([])
    lengths = np.array([item[1] for item in found])
    major = (np.ones(len(found), dtype=bool) if lengths.max() - lengths.min() < 0.1
             else lengths > 0.5 * (lengths.min() + lengths.max()))
    values = np.array(sorted(v for (v, _), keep in zip(found, major) if keep))
    kept = [values[0]]
    for value in values[1:]:
        if value - kept[-1] > 1.0:
            kept.append(value)
    return np.array(kept)


def calibrate(box, pieces, spec):
    """Affine maps from page coordinates to iteration index and log10 error."""
    x_ticks = ticks(pieces, box, "x")
    y_ticks = ticks(pieces, box, "y")
    if len(x_ticks) != len(spec["x_ticks"]) or len(y_ticks) != len(spec["y_ticks"]):
        raise RuntimeError(
            f"panel ({spec['panel']}): found {len(x_ticks)} x and {len(y_ticks)} y "
            f"major ticks, expected {len(spec['x_ticks'])} and {len(spec['y_ticks'])}"
        )
    x_fit = np.polyfit(x_ticks, np.asarray(spec["x_ticks"], dtype=float), 1)
    y_fit = np.polyfit(y_ticks, np.asarray(spec["y_ticks"], dtype=float), 1)
    residuals = (
        float(np.abs(np.polyval(x_fit, x_ticks) - np.asarray(spec["x_ticks"])).max()),
        float(np.abs(np.polyval(y_fit, y_ticks) - np.asarray(spec["y_ticks"])).max()),
    )
    return x_fit, y_fit, x_ticks, y_ticks, residuals


def glyph(run):
    """Classify one marker outline as one of the article's legend glyphs.

    A plus is two axis-aligned bars and a cross two slanted ones.  The closed
    outlines are told apart by their edges: a square is axis-aligned, whereas a
    circle reduces to a slanted polygon whatever its number of Bezier pieces.
    """
    if len(run) == 2:
        (x0, y0), (x1, y1) = run
        return "bar" if abs(x1 - x0) < 0.05 or abs(y1 - y0) < 0.05 else "stroke"
    if len(run) < 4:
        return "other"
    steps = np.diff(run, axis=0)
    aligned = (np.abs(steps[:, 0]) < 0.05) | (np.abs(steps[:, 1]) < 0.05)
    return "square" if aligned.all() else "circle"


def markers(paths, box):
    """Marker centres inside one panel, with the glyph each belongs to.

    Matlab writes one path per marker and Matplotlib one path per series, so the
    outlines are collected first and then clustered on their centres: the two
    strokes of a plus or a cross share theirs, and no two markers of one curve
    are within a marker width of each other.
    """
    left, right, top, bottom = box
    outlines = []
    for style, runs in paths:
        if "dasharray" in style or (BLUE not in style and RED not in style):
            continue
        variant = "average" if BLUE in style else "nka"
        for run in runs:
            centre = run.mean(axis=0)
            if not (left - 3 <= centre[0] <= right + 3
                    and top - 3 <= centre[1] <= bottom + 3):
                continue
            if max(np.ptp(run[:, 0]), np.ptp(run[:, 1])) > 12:
                continue
            outlines.append((variant, glyph(run), centre))

    found, used = [], set()
    for index, (variant, shape, centre) in enumerate(outlines):
        if index in used:
            continue
        cluster = [shape]
        used.add(index)
        for other in range(index + 1, len(outlines)):
            if other in used or outlines[other][0] != variant:
                continue
            if np.hypot(*(outlines[other][2] - centre)) < 1.0:
                cluster.append(outlines[other][1])
                used.add(other)
        if cluster.count("bar") == 2:
            kind = "plus"
        elif cluster.count("stroke") == 2:
            kind = "cross"
        elif "circle" in cluster:
            kind = "circle"
        elif "square" in cluster:
            kind = "square"
        else:
            continue
        found.append((kind, variant, centre))
    return found


def polylines(paths, box):
    """The dashed curves of one panel, as their vertices in page coordinates."""
    left, right, top, bottom = box
    out = []
    for style, runs in paths:
        if "dasharray" not in style:
            continue
        if BLUE not in style and RED not in style:
            continue
        for run in runs:
            inside = ((run[:, 0] >= left - 3) & (run[:, 0] <= right + 3)
                      & (run[:, 1] >= top - 3) & (run[:, 1] <= bottom + 3))
            if len(run) < 4 or inside.mean() < 0.9 or np.ptp(run[:, 1]) < 5:
                continue
            out.append((run, "average" if BLUE in style else "nka"))
    return out


def read_panel(paths, box, spec):
    """One panel of Figure 3.14: four histories with the calibration behind them."""
    axis_pieces = segments(paths)
    x_fit, y_fit, x_ticks, y_ticks, residuals = calibrate(box, axis_pieces, spec)
    glyphs = markers(paths, box)
    curves = {}
    for run, variant in polylines(paths, box):
        # Attribute the polyline to a legend entry through the markers that sit
        # on its vertices; the legend's own sample line carries no vertices.
        votes = {}
        for kind, colour, centre in glyphs:
            if colour != variant:
                continue
            distance = np.hypot(*(run - centre[None, :]).T).min()
            if distance < 1.0:
                votes[kind] = votes.get(kind, 0) + 1
        if not votes:
            continue
        kind = max(votes, key=votes.get)
        entry = next(item for item in LEGEND if item[0] == kind)
        if entry[1] != variant:
            raise RuntimeError(f"glyph {kind} on a {variant} curve")
        iterations = np.polyval(x_fit, run[:, 0])
        values = np.polyval(y_fit, run[:, 1])
        # A curve leaving the axis box is clipped, which adds one vertex that is
        # not a data point; every real vertex sits on an integer iteration.
        on_grid = np.abs(iterations - np.round(iterations)) < 0.05
        curves[f"{entry[1]}_T{entry[2]:g}"] = {
            "variant": entry[1],
            "horizon": entry[2],
            "marker": kind,
            "iterations": [int(round(value)) for value in iterations[on_grid]],
            "log10_error": [float(value) for value in values[on_grid]],
            "clipped_at": [[float(i), float(v)]
                           for i, v in zip(iterations[~on_grid], values[~on_grid])],
        }
    report = {
        "panel": spec["panel"],
        "nu": spec["nu"],
        "box": [float(value) for value in box],
        "x_ticks": [float(value) for value in x_ticks],
        "y_ticks": [float(value) for value in y_ticks],
        "points_per_iteration": float(1.0 / x_fit[0]),
        "points_per_decade": float(-1.0 / y_fit[0]),
        "x_tick_fit_residual": residuals[0],
        "y_tick_fit_residual": residuals[1],
        "axis_limits": {
            "iteration": [float(np.polyval(x_fit, box[0])), float(np.polyval(x_fit, box[1]))],
            "log10_error": [float(np.polyval(y_fit, box[3])), float(np.polyval(y_fit, box[2]))],
        },
    }
    return curves, report


def reference_line(paths, box, x_fit, y_fit):
    """The horizontal line the panel draws across the axis, in data coordinates."""
    left, right, top, bottom = box
    for style, runs in paths:
        if BLUE in style or RED in style or "dasharray" in style:
            continue
        for run in runs:
            if len(run) != 2 or abs(run[0, 1] - run[1, 1]) > 0.05:
                continue
            if np.ptp(run[:, 0]) < 0.8 * (right - left):
                continue
            if not (left - 1 <= run[:, 0].min() and run[:, 0].max() <= right + 1):
                continue
            if not (top + 2 < run[0, 1] < bottom - 2):
                continue
            return float(10.0 ** np.polyval(y_fit, run[0, 1]))
    return None


def read_page(svg, specs=PANELS):
    """Both panels, in the order the page draws them."""
    paths = parse_paths(svg)
    boxes = panel_boxes(segments(paths))
    if len(boxes) != len(specs):
        raise RuntimeError(f"found {len(boxes)} axes boxes, expected {len(specs)}")
    panels, reports = {}, {}
    for box, spec in zip(boxes, specs):
        curves, report = read_panel(paths, box, spec)
        x_fit, y_fit = calibrate(box, segments(paths), spec)[:2]
        report["reference_line"] = reference_line(paths, box, x_fit, y_fit)
        panels[spec["panel"]] = curves
        reports[spec["panel"]] = report
    return panels, reports


def decay(curve):
    """Least-squares decay per iteration over the straight part of a curve."""
    iterations = np.asarray(curve["iterations"], dtype=float)
    values = np.asarray(curve["log10_error"])
    if len(values) < 4:
        return float("nan")
    keep = slice(1, len(values))
    return float(np.polyfit(iterations[keep], values[keep], 1)[0])


def synthetic_page(path, planted):
    """Redraw Figure 3.14 with known histories, in the article's conventions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"plus": ("blue", "+", ":"), "cross": ("blue", "x", ":"),
              "circle": ("red", "o", "-."), "square": ("red", "s", "-.")}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.1), dpi=100)
    for ax, spec in zip(axes, PANELS):
        for kind, variant, horizon in LEGEND:
            values = planted[spec["panel"]][f"{variant}_T{horizon:g}"]
            colour, marker, line = styles[kind]
            ax.semilogy(np.arange(len(values)), values, line, color=colour,
                        marker=marker, markersize=3.0, markerfacecolor="none",
                        linewidth=0.5, label=f"({variant}) T={horizon}")
        ax.set_xticks(list(spec["x_ticks"]))
        ax.set_yticks([10.0**value for value in spec["y_ticks"]])
        ax.set_xlim(spec["x_ticks"][0], spec["x_ticks"][-1] + 1)
        ax.set_ylim(10.0 ** (spec["y_ticks"][-1] - 0.21), 10.0 ** (spec["y_ticks"][0] + 0.48))
        ax.tick_params(direction="in", labelsize=5, length=2.0)
        ax.tick_params(which="minor", direction="in", length=1.0)
        ax.legend(fontsize=4, fancybox=False, edgecolor="black")
        for spine in ax.spines.values():
            spine.set_linewidth(0.3)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.2, wspace=0.3)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def validate():
    """Extract a synthetic figure of known histories through the same pipeline."""
    rng = np.random.default_rng(3141)
    planted = {}
    for spec in PANELS:
        panel = {}
        for _, variant, horizon in LEGEND:
            length = 12 if horizon == 1.3 else 9
            factor = 10.0 ** (-0.3 - 0.2 * rng.random())
            values = 2.0 * factor ** np.arange(length)
            values[1:-1] *= 1.0 + 0.3 * rng.standard_normal(length - 2)
            panel[f"{variant}_T{horizon:g}"] = values
        planted[spec["panel"]] = panel
    with tempfile.TemporaryDirectory() as directory:
        pdf = Path(directory) / "synthetic.pdf"
        synthetic_page(pdf, planted)
        panels, reports = read_page(page_svg(pdf, page=1))
    worst = 0.0
    for panel, curves in panels.items():
        for name, curve in curves.items():
            expected = np.log10(planted[panel][name])
            read = np.asarray(curve["log10_error"])
            index = np.asarray(curve["iterations"])
            worst = max(worst, float(np.abs(read - expected[index]).max()))
            print(f"  ({panel}) {name:<12} {len(index):2d} of {len(expected):2d} points, "
                  f"worst {np.abs(read - expected[index]).max():.2e} decades")
    for panel, report in reports.items():
        print(f"  ({panel}) tick fit residuals "
              f"{report['x_tick_fit_residual']:.2e} iterations, "
              f"{report['y_tick_fit_residual']:.2e} decades")
    print(f"  worst deviation from the planted values: {worst:.2e} decades")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", help="path to the article PDF")
    parser.add_argument("--output", default="docs/data/figure_3_14_published.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    validation = None
    if args.validate:
        print("synthetic figure of known histories:")
        validation = {"worst_deviation_from_planted_values_in_decades": validate()}
    if args.pdf is None:
        return

    panels, reports = read_page(page_svg(args.pdf))
    for spec in PANELS:
        panel = spec["panel"]
        report = reports[panel]
        print(f"\npanel ({panel}), nu = {spec['nu']}: "
              f"{report['points_per_iteration']:.3f} points per iteration, "
              f"{report['points_per_decade']:.3f} points per decade, "
              f"tick fit residuals {report['x_tick_fit_residual']:.1e} and "
              f"{report['y_tick_fit_residual']:.1e}")
        print(f"  horizontal reference line at {report['reference_line']:.4g}")
        for _, variant, horizon in LEGEND:
            curve = panels[panel][f"{variant}_T{horizon:g}"]
            values = " ".join(f"{10.0**value:9.3g}" for value in curve["log10_error"])
            print(f"  {variant:>7} T={horizon}, decay {decay(curve):+.3f} per iteration"
                  f"\n      k = {curve['iterations']}\n      {values}")

    payload = {
        "source": f"page {PAGE} of the article, read from its vector paths",
        "note": (
            "log10_error is the vertical coordinate of the polyline vertices, "
            "which are the plotted data points; clipped_at holds the extra "
            "vertex a curve gets where it leaves the axis box.  Panel (b) is "
            "drawn with a stride: its curves carry a vertex every third or "
            "every second iteration."
        ),
        "panels": panels,
        "decay_per_iteration": {
            panel: {name: decay(curve) for name, curve in curves.items()}
            for panel, curves in panels.items()
        },
        "calibration": reports,
        "validation": validation,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
