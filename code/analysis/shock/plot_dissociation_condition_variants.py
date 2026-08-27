"""
plot_dissociation_condition_variants.py
------------------------------------------
Same 2x2 layout as plot_dissociation_belief_vs_contribution.py (rows =
injection round 20/10; col 1 = adversarial-minus-non-adversarial gap, col 2 =
non-adversarial agents' own raw level), but one figure per condition instead
of FULL: BASELINE, NO_SELECTION, NO_DISCUSSION ("no social learning"). Data
built by build_condition_variants.py -- see that script's docstring for which
panels reuse the saved/trusted round-20 & round-10 NO_DISCUSSION aggregates
vs. which are reconstructed from raw logs (BASELINE, NO_SELECTION -- no
saved ground truth exists for those), and
build_dissociation_round10.py for the underlying methodology/caveats.

random-selection replacement variant throughout (consistent with the FULL
and NO_DISCUSSION figures already produced), one output file per condition.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "exports/dissociations"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"

CONDITIONS = [
    ("baseline", "Baseline", "baseline"),
    ("no_selection", "No selection", "no_selection"),
    ("no_discussion", "No social learning", "no_social_learning"),
]
COL_TITLES = ["Gap: non-adversarial $-$ adversarial",
              "Non-adversarial agents level"]
SERIES = [("belief", "NE", "#4C72B0", "-"),
          ("contribution", "Contribution", "#C44E52", "--")]


def make_figure(cond_tag, cond_label, out_suffix):
    rows = [
        ("Injection at round 10",
         f"{DATA_DIR}/dissociation_gap_{cond_tag}_round10.csv",
         f"{DATA_DIR}/nonadversarial_levels_{cond_tag}_round10.csv"),
        ("Injection at round 20",
         f"{DATA_DIR}/dissociation_gap_{cond_tag}_round20.csv",
         f"{DATA_DIR}/nonadversarial_levels_{cond_tag}_round20.csv"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 13), sharex=True, sharey="col")
    fig.suptitle(cond_label, fontsize=28, y=1.01)

    for row_idx, (row_label, gap_path, level_path) in enumerate(rows):
        gap_d = pd.read_csv(gap_path)
        level_d = pd.read_csv(level_path)

        for col_idx, d in enumerate([gap_d, level_d]):
            ax = axes[row_idx, col_idx]
            for outcome, label, color, ls in SERIES:
                sub = d[d["outcome_type"] == outcome].sort_values("offset")
                x = sub["offset"].to_numpy()
                y = sub["mean"].to_numpy()
                ci = 1.96 * sub["sem"].to_numpy()
                ax.plot(x, y, marker="o", color=color, linestyle=ls, label=label, linewidth=2)
                ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)
            ax.axvline(0, color="gray", lw=1, ls=":")
            if col_idx == 0:
                ax.axhline(0, color="black", lw=1)
            ax.set_xticks(sorted(d["offset"].unique()))
            ax.tick_params(axis="both", labelsize=22)
            if row_idx == 0:
                ax.set_title(COL_TITLES[col_idx], fontsize=24, pad=12)
            if col_idx == 1 and row_idx == 0:
                ax.legend(fontsize=24, loc="lower left")

    for row_idx, (row_label, _, _) in enumerate(rows):
        axes[row_idx, 0].set_ylabel(f"{row_label}", fontsize=24)

    for col_idx in range(2):
        axes[1, col_idx].set_xlabel("Round offset from injection", fontsize=26)

    fig.tight_layout()
    out_stub = os.path.join(OUT_DIR, f"dissociation_belief_vs_contribution_{out_suffix}")
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_stub}.png / .pdf")


def main():
    for cond_tag, cond_label, out_suffix in CONDITIONS:
        make_figure(cond_tag, cond_label, out_suffix)


if __name__ == "__main__":
    main()
