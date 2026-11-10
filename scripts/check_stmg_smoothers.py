"""Record the effect of spatial ordering, GS definition and sweep direction.

Run as ``python -m scripts.check_stmg_smoothers --output results/sensitivity.json``.
This is a sensitivity check on 855 spatial nodes, separate from the paper mesh.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from actapint.stmg_fem import SpaceLevel, TimeMultigrid, space_hierarchy


def run():
    levels = space_hierarchy(3)
    rows = []
    for ordering in ("hierarchical", "lexicographic"):
        ordered, previous = [], None
        for level in levels:
            indices = (
                np.lexsort(level.nodes.T)
                if ordering == "lexicographic"
                else np.arange(len(level.nodes))
            )
            ordered.append(
                SpaceLevel(
                    level.mass[indices][:, indices].tocsc(),
                    level.stiffness[indices][:, indices].tocsc(),
                    None if previous is None else level.prolong[indices][:, previous],
                    level.nodes[indices],
                    level.tetrahedra,
                )
            )
            previous = indices
        for relaxation in ("damped_gs", "sor"):
            for post in ("forward", "backward"):
                mg = TimeMultigrid(
                    ordered, 8, spatial_relaxation=relaxation, post_sweep=post
                )
                initial = np.random.default_rng(1).uniform(0, 1, (8, 4, mg.n))
                u, norm = initial.copy(), np.linalg.norm(initial)
                history = [1.0]
                for _ in range(50):
                    u = mg.cycle(np.zeros_like(u), u)
                    history.append(float(np.linalg.norm(u) / norm))
                    if history[-1] < 1e-8:
                        break
                rows.append(
                    dict(
                        ordering=ordering,
                        relaxation=relaxation,
                        post_sweep=post,
                        iterations=len(history) - 1,
                        relative_error_history=history,
                    )
                )
    return dict(
        spatial_dofs=855,
        tetrahedra=6144,
        time_steps=8,
        degree=3,
        time_step=0.1,
        seed=1,
        tolerance=1e-8,
        rows=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
