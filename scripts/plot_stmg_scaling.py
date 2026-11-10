"""Validate raw timing files and plot measured medians with min/max bars."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_runs(folder, expected_ranks):
    runs = {}
    for mode in ("weak", "strong"):
        rows = []
        for rank in expected_ranks:
            d = json.loads((folder / f"{mode}-p{rank}.json").read_text())
            assert d["status"] == "passed" and d["ranks"] == rank
            assert d["steps"] == (2 * rank if mode == "weak" else 512)
            assert d["spatial_dofs"] == 7471 and d["tetrahedra"] == 49152
            assert d["warmups"] == 1 and len(d["samples"]) == 3
            assert len(d["hosts"]) == rank and d["unique_hosts"] == len(set(d["hosts"]))
            for s in d["samples"]:
                assert s["relative_residual"] <= 1e-8 and s["relative_error"] <= 1e-7
            t = [s["seconds"] for s in d["samples"]]
            rows.append(
                dict(
                    ranks=rank,
                    steps=d["steps"],
                    dofs=d["dofs"],
                    hosts=d["unique_hosts"],
                    median=float(np.median(t)),
                    minimum=min(t),
                    maximum=max(t),
                    iterations=[s["iterations"] for s in d["samples"]],
                    maximum_residual=max(s["relative_residual"] for s in d["samples"]),
                    maximum_error=max(s["relative_error"] for s in d["samples"]),
                )
            )
        base = rows[0]["median"]
        for row in rows:
            row["speedup"] = base / row["median"]
            row["efficiency"] = (
                base / row["median"] / (row["ranks"] if mode == "strong" else 1)
            )
        runs[mode] = rows
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--cluster", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    args = p.parse_args()
    result = {
        "single_host": read_runs(args.local, (1, 2, 4, 8)),
        "two_node_allocation": read_runs(args.cluster, (1, 2, 4, 8, 16)),
    }
    for row in result["single_host"]["weak"] + result["single_host"]["strong"]:
        assert row["hosts"] == 1
    for row in (
        result["two_node_allocation"]["weak"] + result["two_node_allocation"]["strong"]
    ):
        assert row["hosts"] == (1 if row["ranks"] == 1 else 2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    for key, label, color, marker in [
        ("single_host", "Mac: one host", "#3178ad", "o"),
        ("two_node_allocation", "SCNet: two nodes (P>=2)", "#d86425", "s"),
    ]:
        for mode, a in [("strong", axes[0, 0]), ("weak", axes[1, 0])]:
            rows = result[key][mode]
            x = [r["ranks"] for r in rows]
            y = np.array([r["median"] for r in rows])
            err = np.array(
                [y - [r["minimum"] for r in rows], [r["maximum"] for r in rows] - y]
            )
            a.errorbar(
                x,
                y,
                yerr=err,
                label=label,
                color=color,
                marker=marker,
                capsize=4,
                lw=1.5,
            )
        rows = result[key]["strong"]
        axes[0, 1].plot(
            [r["ranks"] for r in rows],
            [r["speedup"] for r in rows],
            marker + "-",
            color=color,
            label=label,
        )
        rows = result[key]["weak"]
        axes[1, 1].plot(
            [r["ranks"] for r in rows],
            [100 * r["efficiency"] for r in rows],
            marker + "-",
            color=color,
            label=label,
        )
    axes[0, 0].set(
        title="Strong scaling: 15,300,608 unknowns", ylabel="Solve wall time (s)"
    )
    axes[0, 0].set_yscale("log", base=2)
    axes[0, 1].plot(
        [1, 2, 4, 8, 16], [1, 2, 4, 8, 16], "--", color="gray", label="Ideal"
    )
    axes[0, 1].set(title="Strong speedup relative to P=1 STMG", ylabel="Speedup")
    axes[1, 0].set(
        title="Weak scaling: two time slabs per rank", ylabel="Solve wall time (s)"
    )
    axes[1, 1].axhline(100, ls="--", color="gray")
    axes[1, 1].set(title="Weak efficiency: t(1) / t(P)", ylabel="Efficiency (%)")
    for a in axes.ravel():
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
        a.set_xlabel("MPI ranks")
        a.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "P1 tetrahedral FEM / dG(3), dt=0.1; medians and full ranges of three solves",
        fontsize=12,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg"]:
        fig.savefig(args.output_prefix.with_suffix("." + suffix), dpi=180)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
