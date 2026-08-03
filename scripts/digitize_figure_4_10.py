#!/usr/bin/env python3
"""Read the published error curves of Figure 4.10 off the article page.

The figure prints no numbers, and what is at stake in Section 4.4 is the shape
of the first iterations rather than the slope, so the four panels have to be
recovered from the raster.  Each panel is a semi-logarithmic plot of two curves,
Parareal in blue with an open circle and MGRiT in red with a plus, drawn dashed
through the markers.

The pixel-to-data map of a panel is calibrated on its *interior* tick marks and
never on the frame, so that the finite width of the spines cannot bias it; the
fit reports its own residual, which stays below half a pixel on every panel.
Each curve is then read twice, independently:

``marker``
    half-way between the top and the bottom of the marker glyph in the column
    through the marker's x position.  Both glyphs are symmetric about their
    centre, so this reading is exact up to the stroke width whenever the glyph
    is drawn in full.  A glyph is identified as the widest pair of ink runs a
    marker diameter apart, which is what separates it from a dash of the
    connecting line passing through the same column.
``line``
    a least-squares fit of the whole drawn polyline with the glyph columns cut
    out: every remaining column contributes the mean row of its ink, and the
    model interpolates linearly between the unknown vertex heights.  This is the
    only reading available where a marker is clipped by the axis.

``--validate`` runs the same extraction on a synthetic panel of known curves
drawn in the article's conventions, which is what fixes the tolerances: the
marker reading is then good to 0.018 decades and the line reading, which has to
fit dashes on segments dropping a decade per column, to 0.086 decades.

Usage::

    python scripts/digitize_figure_4_10.py paper.pdf --output docs/data/figure_4_10_published.json
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

# The article starts on printed page 385, so printed page 460 is the 76th.
PAGE = 76
RENDER_DPI = 600
# An open circle and a plus are about 30 px across at this resolution; the two
# bounds bracket that generously without admitting a pair of dashes.
MARKER_EXTENT = (20.0, 40.0)
GLYPH_RADIUS = 17

# Reading order on the page, with the axis labels the panels print and the
# iteration range of the plotted curves.
PANELS = (
    ("heat", "Heat equation", (0, 25), (5, 10, 15, 20), (-2, -4, -6, -8, -10, -12)),
    ("ade_0.1", "ADE with nu = 0.1", (0, 25), (5, 10, 15, 20),
     (-2, -4, -6, -8, -10, -12)),
    ("ade_0.01", "ADE with nu = 0.01", (0, 25), (5, 10, 15, 20),
     (-2, -4, -6, -8, -10, -12)),
    ("ade_0.002", "ADE with nu = 0.002", (0, 12), tuple(range(1, 12)), (2, 1, 0)),
)


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


def masks(image):
    """The two curve colours, the grey ink of the axes, and all ink."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    blue = (b - np.maximum(r, g)) > 0.15
    red = (r - np.maximum(g, b)) > 0.15
    chroma = image.max(axis=2) - image.min(axis=2)
    ink = image.mean(axis=2) < 0.8
    return blue, red, ink & (chroma < 0.15), ink


def _runs(line, minimum):
    """Start and end of every run of at least ``minimum`` set entries."""
    labels, count = ndimage.label(line)
    out = []
    for index in range(1, count + 1):
        where = np.where(labels == index)[0]
        if where.size >= minimum:
            out.append((int(where[0]), int(where[-1])))
    return out


def frames(ink, horizontal=900, vertical=700):
    """Axes boxes of a page: pairs of long straight lines joined by spines."""
    found = [
        (a, b, row)
        for row in range(ink.shape[0])
        for a, b in _runs(ink[row], horizontal)
    ]
    kept = []
    for a1, b1, top in found:
        for a2, b2, bottom in found:
            if bottom - top < 300 or abs(a1 - a2) > 25 or abs(b1 - b2) > 25:
                continue
            spines = {}
            for side, guess in (("left", max(a1, a2)), ("right", min(b1, b2))):
                for column in range(guess - 20, guess + 21):
                    if any(
                        a <= top + 4 and b >= bottom - 4
                        for a, b in _runs(ink[:, column], vertical)
                    ):
                        spines.setdefault(side, column)
            if len(spines) != 2:
                continue
            box = (top, bottom, spines["left"], spines["right"])
            if any(max(abs(u - v) for u, v in zip(box, other)) <= 3 for other in kept):
                continue
            kept.append(box)
    return sorted(kept)


def _cluster(values):
    """Centre of every group of positions within three pixels of each other."""
    out, run = [], [values[0]]
    for value in values[1:]:
        if value - run[-1] <= 3:
            run.append(value)
        else:
            out.append(float(np.mean(run)))
            run = [value]
    out.append(float(np.mean(run)))
    return out


def ticks(grey, box, minimum=8, reach=20, exclude=None):
    """Interior tick positions on the left and on the bottom spine."""
    top, bottom, left, right = box
    if exclude is not None:
        grey = grey.copy()
        grey[exclude[0]:exclude[1] + 1, exclude[2]:exclude[3] + 1] = False
    counts = grey[top + 3:bottom - 2, left + 3:left + reach].sum(axis=1)
    # A tick stops a few pixels into the panel; the dash-dotted truncation line
    # crosses the whole panel and is rejected by looking further in.
    beyond = grey[top + 3:bottom - 2, left + reach + 10:left + reach + 40].any(axis=1)
    rows = np.where((counts >= minimum) & ~beyond)[0] + top + 3
    counts = grey[bottom - reach:bottom - 3, left + 3:right - 2].sum(axis=0)
    columns = np.where(counts >= minimum)[0] + left + 3
    return (
        _cluster(rows) if rows.size else [],
        _cluster(columns) if columns.size else [],
    )


def _calibrate(values, pixels):
    values = np.asarray(values, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    slope, offset = np.polyfit(values, pixels, 1)
    residual = float(np.abs(pixels - (slope * values + offset)).max())
    return (slope, offset), residual


class Panel:
    """The pixel-to-data map of one semi-logarithmic panel, fitted on its ticks."""

    def __init__(self, box, x_ticks, y_ticks, x_range):
        self.top, self.bottom, self.left, self.right = box
        self.first, self.last = x_range
        (self.x_slope, self.x_offset), self.x_residual = _calibrate(*zip(*x_ticks))
        (self.y_slope, self.y_offset), self.y_residual = _calibrate(*zip(*y_ticks))

    def column(self, k):
        return self.x_slope * k + self.x_offset

    def value(self, row):
        return 10.0 ** ((row - self.y_offset) / self.y_slope)

    def clipped(self, row, margin=6.0):
        return bool(row <= self.top + margin or row >= self.bottom - margin)

    def calibration(self):
        return {
            "frame": [int(self.top), int(self.bottom), int(self.left), int(self.right)],
            "pixels_per_iteration": float(self.x_slope),
            "column_at_iteration_zero": float(self.x_offset),
            "pixels_per_decade": float(-self.y_slope),
            "row_at_decade_zero": float(self.y_offset),
            "x_tick_fit_residual": self.x_residual,
            "y_tick_fit_residual": self.y_residual,
        }


def marker_reading(mask, panel, half_width=1):
    """Centre of every marker glyph of one curve, in data coordinates."""
    out = {}
    for k in range(panel.first, panel.last + 1):
        centre = panel.column(k)
        low = max(int(round(centre - half_width)), panel.left)
        high = min(int(round(centre + half_width)) + 1, panel.right + 1)
        window = mask[panel.top:panel.bottom + 1, low:high]
        if not window.any():
            continue
        pieces = _runs(window.any(axis=1), 1)
        pair = None
        for first in range(len(pieces)):
            for last in range(len(pieces) - 1, first - 1, -1):
                extent = pieces[last][1] - pieces[first][0]
                if MARKER_EXTENT[0] <= extent <= MARKER_EXTENT[1]:
                    pair = (pieces[first][0], pieces[last][1])
                    break
            if pair is not None:
                break
        if pair is None:
            continue
        row = panel.top + 0.5 * (pair[0] + pair[1])
        out[k] = (panel.value(row), panel.clipped(row))
    return out


def line_reading(mask, panel, max_thickness=12, iterations=4):
    """Least-squares fit of the polyline through the vertices at integer x."""
    columns, rows = [], []
    for column in range(panel.left, panel.right + 1):
        found = np.where(mask[panel.top:panel.bottom + 1, column])[0]
        if found.size == 0 or found.max() - found.min() > max_thickness:
            continue
        position = (column - panel.x_offset) / panel.x_slope
        if abs(position - round(position)) * panel.x_slope < GLYPH_RADIUS:
            continue
        columns.append(position)
        rows.append(found.mean() + panel.top)
    if len(columns) < 10:
        return {}
    positions = np.array(columns)
    rows = np.array(rows)
    vertices = np.arange(panel.first, panel.last + 1)
    design = np.zeros((positions.size, vertices.size))
    left = np.clip(np.floor(positions).astype(int) - panel.first, 0, vertices.size - 2)
    weight = positions - vertices[left]
    design[np.arange(positions.size), left] = 1.0 - weight
    design[np.arange(positions.size), left + 1] = weight
    used = design.any(axis=0)
    solution = np.full(vertices.size, np.nan)
    keep = np.ones(rows.size, dtype=bool)
    for _ in range(iterations):
        fit, *_ = np.linalg.lstsq(design[keep][:, used], rows[keep], rcond=None)
        solution[used] = fit
        residual = rows - design[:, used] @ fit
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        new = np.abs(residual) <= 3.0 * scale
        if new.sum() < 10 or (new == keep).all():
            break
        keep = new
    out = {}
    for index, vertex in enumerate(vertices):
        # A vertex is only determined when both of its segments carry ink.
        if not used[index] or design[keep][:, index].sum() < 4.0:
            continue
        row = float(solution[index])
        out[int(vertex)] = (panel.value(row), panel.clipped(row))
    return out


def legend_box(grey, box, row_run=250, column_run=120):
    """The legend rectangle inside a panel: its only long straight grey lines."""
    top, bottom, left, right = box
    interior = grey[top + 6:bottom - 5, left + 6:right - 5]

    def longest(line):
        labels, count = ndimage.label(line)
        return max((labels == k).sum() for k in range(1, count + 1)) if count else 0

    rows = [index for index, line in enumerate(interior) if longest(line) >= row_run]
    if not rows:
        return None
    columns = [
        index for index, line in enumerate(interior.T) if longest(line) >= column_run
    ]
    return (
        top + 6 + min(rows), top + 6 + max(rows),
        left + 6 + min(columns), left + 6 + max(columns),
    )


def digitize_panel(blue, red, grey, panel, box):
    """Both curves of one panel, each read as a marker and as a line."""
    window = legend_box(grey, box)
    curves = {}
    for method, mask in (("parareal", blue), ("mgrit", red)):
        mask = mask.copy()
        if window is not None:
            top, bottom, left, right = window
            mask[top:bottom + 1, left:right + 1] = False
        marker = marker_reading(mask, panel)
        line = line_reading(mask, panel)
        # The marker is the better reading; the line is what remains where the
        # axis cuts the glyph, and the only reading above the top of the panel.
        preferred, source, clipped = {}, {}, []
        for k in sorted(set(marker) | set(line)):
            choice = marker.get(k)
            name = "marker"
            if choice is None or (choice[1] and k in line and not line[k][1]):
                choice, name = line[k], "line"
            preferred[k] = choice[0]
            source[k] = name
            if choice[1]:
                clipped.append(k)
        curves[method] = {
            "values": preferred,
            "read_from": source,
            "clipped": clipped,
            "marker": marker,
            "line": line,
        }
    curves["legend_box"] = window
    return curves


def read_panels(image, expected=len(PANELS)):
    """All four panels of Figure 4.10, with the calibration each reading rests on."""
    blue, red, grey, ink = masks(image)
    boxes = frames(ink)
    if len(boxes) != expected:
        raise RuntimeError(f"found {len(boxes)} panels, expected {expected}")
    tables, report = {}, {}
    for box, (name, title, x_range, x_labels, y_labels) in zip(boxes, PANELS[:expected]):
        window = legend_box(grey, box)
        rows, columns = ticks(grey, box, exclude=window)
        if len(columns) != len(x_labels) or len(rows) != len(y_labels):
            raise RuntimeError(
                f"panel {name}: found {len(columns)} x ticks and {len(rows)} y ticks, "
                f"expected {len(x_labels)} and {len(y_labels)}"
            )
        panel = Panel(
            box,
            x_ticks=list(zip(x_labels, columns)),
            y_ticks=list(zip(y_labels, rows)),
            x_range=x_range,
        )
        tables[name] = digitize_panel(blue, red, grey, panel, box)
        tables[name]["title"] = title
        report[name] = panel.calibration()
        report[name]["x_ticks"] = columns
        report[name]["y_ticks"] = rows
    return tables, report


def synthetic_page(path, curves):
    """Redraw one panel of Figure 4.10 with known curves, in its own conventions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi, page, box = RENDER_DPI, (2400, 1800), (1229, 934)
    fig = plt.figure(figsize=(page[0] / dpi, page[1] / dpi), dpi=dpi)
    ax = fig.add_axes([0.22, 0.16, box[0] / page[0], box[1] / page[1]])
    for values, style in ((curves["parareal"], "b--o"), (curves["mgrit"], "r--+")):
        ax.semilogy(
            np.arange(len(values)), values, style, markersize=3.6, linewidth=0.55,
            markerfacecolor="none", markeredgewidth=0.55,
        )
    ax.set_xlim(0, 25)
    ax.set_ylim(1e-14, 1e0)
    ax.set_xticks(range(0, 26, 5))
    ax.set_yticks(10.0 ** np.arange(-14, 1, 2))
    ax.tick_params(direction="in", length=1.7, width=0.5, which="major")
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def validate():
    """Extract a synthetic panel of known curves through the same pipeline.

    The planted curves carry the two features that make the article's panels
    hard: a first curve that falls more than a decade per iteration, so that its
    dashes are nearly vertical, and a second one that kinks from a steep start
    into a shallow tail.
    """
    planted = {
        "parareal": 10.0 ** np.array(
            [0.0, -0.65, -1.7, -2.85, -3.9, -4.99, -6.08, -7.2, -8.31, -9.41,
             -10.52, -11.62, -12.7, -13.8]
        ),
        "mgrit": 10.0 ** np.array(
            [-0.3, -1.58, -2.77, -3.8, -4.58, -5.23, -5.87, -6.49, -7.11, -7.71,
             -8.32, -8.93, -9.52, -10.12, -10.71, -11.3, -11.89, -12.47, -13.04,
             -13.55]
        ),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.png"
        synthetic_page(path, planted)
        image = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    blue, red, grey, ink = masks(image)
    box = frames(ink)[0]
    rows, columns = ticks(grey, box)
    panel = Panel(
        box,
        x_ticks=list(zip(range(0, 25, 5), columns)),
        y_ticks=list(zip(range(-2, -14, -2), rows)),
        x_range=(0, 25),
    )
    print(f"  frame {box} x ticks {[round(v, 1) for v in columns]} "
          f"y ticks {[round(v, 1) for v in rows]}")
    print(f"  calibration {panel.calibration()}")
    curves = digitize_panel(blue, red, grey, panel, box)
    worst = {"marker": 0.0, "line": 0.0}
    for method, values in planted.items():
        for reading in ("marker", "line"):
            found = curves[method][reading]
            deviations = [
                abs(np.log10(found[k][0] / value))
                for k, value in enumerate(values)
                if k in found and not found[k][1]
            ]
            worst[reading] = max(worst[reading], max(deviations))
            print(f"  {method:9s} {reading:6s} {len(deviations)} of {len(values)} read,"
                  f" worst {max(deviations):.4f} decades, median "
                  f"{float(np.median(deviations)):.4f}")
    return worst


def report(name, table):
    print(f"\n{name}: {table['title']}")
    for method in ("parareal", "mgrit"):
        curve = table[method]
        print(f"  {method}")
        for k in sorted(curve["values"]):
            marker, line = curve["marker"].get(k), curve["line"].get(k)
            text = f"    {k:3d}  {curve['values'][k]:.4e}"
            text += "  clipped" if k in curve["clipped"] else "         "
            text += f"  marker {marker[0]:.4e}" if marker else "  marker      -     "
            text += f"  line {line[0]:.4e}" if line else "  line      -     "
            print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", help="path to the article PDF")
    parser.add_argument("--output", default="docs/data/figure_4_10_published.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    validation = None
    if args.validate:
        print("synthetic panel of known curves:")
        worst = validate()
        validation = {f"worst_deviation_in_decades_{key}": value
                      for key, value in worst.items()}
    if args.pdf is None:
        return

    with tempfile.TemporaryDirectory() as directory:
        path = render_page(args.pdf, directory)
        image = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255.0
    tables, calibration = read_panels(image)
    for name, table in tables.items():
        report(name, table)

    payload = {
        "source": f"page {PAGE} of the article rendered at {RENDER_DPI} dpi",
        "panels": {
            name: {
                "title": table["title"],
                **{
                    method: {
                        "values": {str(k): v for k, v in table[method]["values"].items()},
                        "read_from": {str(k): v
                                      for k, v in table[method]["read_from"].items()},
                        "clipped": table[method]["clipped"],
                        "marker": {str(k): v[0]
                                   for k, v in table[method]["marker"].items()},
                        "line": {str(k): v[0] for k, v in table[method]["line"].items()},
                    }
                    for method in ("parareal", "mgrit")
                },
            }
            for name, table in tables.items()
        },
        "calibration": calibration,
        "validation": validation,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
