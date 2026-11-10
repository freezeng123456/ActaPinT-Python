"""Plot recomputed increments and original EPS coordinates with honest indexing."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir", type=Path, default=Path("docs/data/reproduction-2026-11-11")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads((a.data_dir / "nka_periodic.json").read_text())
    eps = json.loads((a.data_dir / "figure_3_14_eps.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=True)
    for row, nu in enumerate((0.02, 0.002)):
        for col, T in enumerate((1.3, 0.7)):
            ax = axes[row, col]
            for r in data["runs"]:
                if r["nu"] != nu or r["horizon"] != T or r["start"] == "initial":
                    continue
                color = "#2863ac" if r["method"] == "average" else "#c84536"
                style = "-" if r["start"] == "symmetric" else "--"
                label = f"{r['method']}, {'random' if r['start']=='symmetric' else 'coarse TR'}"
                values = [v["increment"]["linf"] for v in r["records"][1:]]
                ax.semilogy(
                    range(len(values)), values, style, color=color, lw=1.6, label=label
                )
            for panel in eps["panels"]:
                for curve in panel["curves"]:
                    if curve["nu"] == nu and curve["horizon"] == T:
                        ax.semilogy(
                            [v["iteration"] for v in curve["markers"]],
                            [v["value"] for v in curve["markers"]],
                            "o" if curve["method"] == "nka" else "+",
                            ms=4,
                            mfc="none",
                            color="black",
                            label="paper " + curve["method"],
                        )
            ax.set(
                title=f"nu={nu}, T={T}",
                xlabel="Update index (first update = 0)",
                ylabel="Max-norm increment / published Error",
                ylim=(3e-7, 1e2),
            )
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "NKA reconstruction: both methods use the same starting vector\n200 distinct periodic nodes, 200 time steps; paper Error definition unspecified",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160)
    fig.savefig(a.output.with_suffix(".png"), dpi=160)


if __name__ == "__main__":
    main()
