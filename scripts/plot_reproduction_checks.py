"""Plot computed checks and published NKA markers without curve fitting."""

import argparse
import json
from pathlib import Path

from actapint.plotting import plt, save_figure, style_axis
from actapint.section3.paradiag_bvm import PUBLISHED_FIGURE_3_14


def plot(data_dir, output):
    stmg = json.loads((data_dir / "stmg_fem_3d.json").read_text())
    nka = json.loads((data_dir / "figure_3_14_audit.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    for row in stmg["rows"]:
        axes[0, 0].semilogy(
            row["residual_history"], marker="o", label=f'{row["time_steps"]} time steps'
        )
        axes[0, 1].semilogy(
            row["homogeneous_test"]["relative_error_history"],
            marker="o",
            label=f'{row["time_steps"]} time steps',
        )
    axes[0, 0].set(
        title="3D STMG: manufactured right-hand side", ylabel="Relative residual"
    )
    axes[0, 1].set(
        title="3D STMG: homogeneous error propagation", ylabel="Relative error"
    )
    for ax in axes[0]:
        ax.axhline(1e-8, color="gray", linestyle=":", label="Stopping tolerance")
    for ax, horizon in zip(axes[1], (1.3, 0.7)):
        for variant, color in (("average", "#2F6BFF"), ("nka", "#D1495B")):
            run = next(
                r
                for r in nka["runs"]
                if r["nu"] == 0.02
                and r["horizon"] == horizon
                and r["start"] == "symmetric"
                and r["variant"] == variant
            )
            ax.semilogy(
                [r["iteration"] for r in run["records"]],
                [r["error"]["linf"] for r in run["records"]],
                color=color,
                label=f"Computed {variant}",
            )
            published = PUBLISHED_FIGURE_3_14[0.02, variant, horizon]
            ax.semilogy(
                list(published),
                list(published.values()),
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                color=color,
                label=f"Published {variant}",
            )
        ax.set(
            title=rf"Figure 3.14: $\nu=0.02$, $T={horizon}$",
            ylabel="Error (computed: max norm)",
            ylim=(1e-7, 4),
        )
    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        style_axis(ax)
        ax.legend(fontsize=8)
    paths = save_figure(fig, output)
    svg = Path(paths["svg"])
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(plot(args.data_dir, args.output))
