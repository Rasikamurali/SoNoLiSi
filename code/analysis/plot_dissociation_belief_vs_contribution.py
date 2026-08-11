"""
plot_dissociation_belief_vs_contribution.py
----------------------------------------------
Regenerates figures/2026-03-22/paper_stats/dissociation_belief_vs_contribution.png/.pdf
from the saved aggregate (exports/dissociations/dissociation_pooled_agg.csv): belief
(perceived injunctive norm) vs. behavior (contribution) recovery around a
group-composition injection event, split by discussion on/off.

The original plotting script that produced this figure is not present
anywhere in the repo (searched by filename and by distinctive on-figure
text) - only its data outputs were saved. Rebuilt from
dissociation_pooled_agg.csv, which contains exactly what's plotted: mean
and SEM of the non-replaced-minus-replaced gap by (discussion_on,
outcome_type, offset).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "exports/dissociations/dissociation_pooled_agg.csv"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"
OUT_STUB = os.path.join(OUT_DIR, "dissociation_belief_vs_contribution")

PANELS = [(0, None), (1, None)]
SERIES = [("belief", "Injunctive norm", "#4C72B0", "-"),
         ("contribution", "Contribution", "#C44E52", "--")]


def main():
    d = pd.read_csv(DATA_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for ax, (disc_val, panel_title) in zip(axes, PANELS):
        sub_panel = d[d["discussion_on"] == disc_val]
        for outcome, label, color, ls in SERIES:
            sub = sub_panel[sub_panel["outcome_type"] == outcome].sort_values("offset")
            x = sub["offset"].to_numpy()
            y = sub["mean"].to_numpy()
            ci = 1.96 * sub["sem"].to_numpy()
            ax.plot(x, y, marker="o", color=color, linestyle=ls, label=label, linewidth=2)
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)
        ax.axvline(0, color="gray", lw=1, ls=":")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Round offset from injection", fontsize=30)
        ax.set_xticks(sorted(d["offset"].unique()))
        ax.tick_params(axis="both", labelsize=26)

    axes[0].set_ylabel("Gap to group mean\n(non-replaced $-$ replaced)", fontsize=27)
    axes[1].legend(fontsize=30, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_STUB + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_STUB + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_STUB}.png / .pdf")


if __name__ == "__main__":
    main()
