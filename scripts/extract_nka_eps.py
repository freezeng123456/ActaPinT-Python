"""Extract numeric polyline markers from the paper's original Figure 3.14 EPS.

Input: Figures/Fig_ParaDiagI_NKA_Burgers_{a,b}.eps from
https://arxiv.org/src/2503.13526. Only numeric coordinates are redistributed.
Calibration uses the labelled logarithmic ticks; noninteger clipped endpoints
are excluded. EPS rounds polyline coordinates to 0.001 PostScript units.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def extract(path, panel):
    text = path.read_text()
    config = {
        "a": (0.02, 22.0, 51.601, 0, (348.394 - 51.601) / 6),
        "b": (0.002, 418 / 49, 26.0, 2, (348.360 - 26) / 8),
    }
    nu, xstep, yref, logref, decade = config[panel]
    paths = re.findall(
        r"(0 0 1|1 0 0) RC\n\[.*?\] 0 setdash\n2 LJ\nN\n(.*?)\nS\n", text, re.S
    )
    curves = []
    for i, (color, geometry) in enumerate(paths[:4]):
        markers = []
        for x, y in re.findall(r"([\d.]+) ([\d.]+) [ML]\n?", geometry):
            x, y = float(x), float(y)
            k = (x - 62) / xstep
            if abs(k - round(k)) < 0.0001:
                markers.append(
                    dict(
                        iteration=round(k),
                        value=10 ** (logref - (y - yref) / decade),
                        x=x,
                        y=y,
                    )
                )
        assert len(markers) > 3 and markers[0]["iteration"] == 0
        curves.append(
            dict(
                nu=nu,
                horizon=1.3 if i % 2 == 0 else 0.7,
                method="average" if color == "0 0 1" else "nka",
                markers=markers,
            )
        )
    assert len(curves) == 4
    for i in (0, 1):
        assert curves[i]["markers"][0]["y"] == curves[i + 2]["markers"][0]["y"]
    return dict(
        file=path.name,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        x_origin=62,
        x_units_per_iteration=xstep,
        y_reference=yref,
        log10_reference=logref,
        y_units_per_decade=decade,
        curves=curves,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = dict(
        source="https://arxiv.org/src/2503.13526",
        coordinate_rounding=0.001,
        interpretation="numeric plot coordinates, not recomputed solver outputs",
        panels=[
            extract(a.source_dir / f"Fig_ParaDiagI_NKA_Burgers_{panel}.eps", panel)
            for panel in "ab"
        ],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
