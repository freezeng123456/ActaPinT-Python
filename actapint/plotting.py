"""Publication-oriented plotting helpers shared by all experiments."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler


COLORS = {
    "navy": "#17324D",
    "blue": "#2F6BFF",
    "teal": "#0F8B8D",
    "amber": "#E09F3E",
    "rose": "#D1495B",
    "indigo": "#6D5BD0",
    "slate": "#52616B",
    "ink": "#17212B",
    "grid": "#D9E2EC",
    "paper": "#FFFFFF",
}

PALETTE = [
    COLORS["blue"],
    COLORS["teal"],
    COLORS["amber"],
    COLORS["rose"],
    COLORS["indigo"],
    COLORS["slate"],
]


def configure_plots() -> None:
    """Apply a deterministic style suitable for papers, slides, and web pages."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 240,
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["paper"],
            "axes.edgecolor": "#9AA9B8",
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["navy"],
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.prop_cycle": cycler(color=PALETTE),
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": COLORS["grid"],
            "legend.facecolor": COLORS["paper"],
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "xtick.color": COLORS["slate"],
            "ytick.color": COLORS["slate"],
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.72,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax, *, grid: bool = True) -> None:
    """Remove visual clutter and add a restrained major/minor grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, which="major")
        ax.grid(True, which="minor", alpha=0.28, linewidth=0.5)
        ax.set_axisbelow(True)


def _figure_accessible_text(fig, fallback: str) -> tuple[str, str]:
    """Return concise accessible text derived from visible figure titles."""
    titles: list[str] = []
    if fig._suptitle is not None:
        text = fig._suptitle.get_text().strip()
        if text:
            titles.append(text)
    for ax in fig.axes:
        text = ax.get_title().strip()
        if text and text not in titles:
            titles.append(text)

    title = titles[0] if titles else fallback
    if len(titles) > 1:
        description = "Numerical experiment with panels: " + "; ".join(titles)
    else:
        description = (
            f"Numerical experiment plot titled {title}. "
            "Quantitative values are provided in the accompanying JSON and report."
        )
    return title, description


def normalize_svg(
    svg_path: str | Path,
    *,
    title: str,
    description: str,
) -> None:
    """Make a Matplotlib SVG responsive and accessible in web documents."""
    svg_path = Path(svg_path)
    source = svg_path.read_text(encoding="utf-8")
    match = re.search(r"<svg\b[^>]*>", source)
    if match is None:
        raise ValueError(f"SVG root element not found in {svg_path}")

    root = match.group(0)
    root = re.sub(r'width="([0-9.]+)pt"', r'width="\1"', root)
    root = re.sub(r'height="([0-9.]+)pt"', r'height="\1"', root)
    root = re.sub(r'\srole="[^"]*"', "", root)
    root = re.sub(r'\saria-labelledby="[^"]*"', "", root)
    root = root[:-1] + ' role="img" aria-labelledby="figure-title figure-description">'

    source = re.sub(
        r'\s*<title id="figure-title">.*?</title>',
        "",
        source,
        flags=re.DOTALL,
    )
    source = re.sub(
        r'\s*<desc id="figure-description">.*?</desc>',
        "",
        source,
        flags=re.DOTALL,
    )
    match = re.search(r"<svg\b[^>]*>", source)
    if match is None:
        raise ValueError(f"SVG root element not found in {svg_path}")
    accessible_text = (
        f'\n <title id="figure-title">{escape(title)}</title>'
        f'\n <desc id="figure-description">{escape(description)}</desc>'
    )
    source = source[: match.start()] + root + accessible_text + source[match.end() :]
    svg_path.write_text(source, encoding="utf-8")


def save_figure(fig, output_path, *, dpi: int = 240) -> dict[str, str]:
    """Save both a web PNG and an editable, resolution-independent SVG."""
    output_path = Path(output_path)
    stem = output_path.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    title, description = _figure_accessible_text(fig, stem.name.replace("_", " "))
    normalize_svg(svg_path, title=title, description=description)
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


configure_plots()
