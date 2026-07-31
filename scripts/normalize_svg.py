#!/usr/bin/env python3
"""Normalize cached Matplotlib SVG files for responsive web rendering."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from actapint.plotting import normalize_svg


ACRONYMS = {
    "ade": "ADE",
    "mgrit": "MGRIT",
    "paradiag": "ParaDiag",
    "stmg": "STMG",
}


def readable_title(path: Path) -> str:
    """Turn a result filename into a concise figure title."""
    name = path.stem.replace("_", "-")
    name = re.sub(
        r"figure-(\d+)-(\d+)",
        lambda match: f"Figure {match.group(1)}.{match.group(2)}",
        name,
    )
    words = [
        ACRONYMS.get(word.lower(), word.capitalize())
        for word in name.replace("-", " ").split()
    ]
    return " ".join(words)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.glob("*.svg")) if path.is_dir() else [path])

    for path in files:
        title = readable_title(path)
        normalize_svg(
            path,
            title=title,
            description=(
                f"{title}, reproduced with the Python conversion of ActaPinT. "
                "The surrounding report provides parameters and quantitative values."
            ),
        )
        print(path)


if __name__ == "__main__":
    main()
