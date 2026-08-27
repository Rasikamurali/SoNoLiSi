"""
plot_dissociation_top_replacement.py
---------------------------------------
Same 2x2 layout as plot_dissociation_belief_vs_contribution.py (rows =
injection round 20/10, FULL condition only; col 1 = non-replaced-minus-replaced
gap, col 2 = the 8 untouched agents' own raw level), but using the `top`
replacement-selection variant (4 most-connected agents hijacked into
adversarial behavior) instead of `random`. Data built by
build_top_replacement_variant.py -- see that script's and
build_dissociation_round10.py's docstrings for methodology/caveats.

Kept as a separate figure (not overwriting the `random` version) since
`random` was the variant that best matched the one piece of preserved ground
truth (round-20 dissociation_pooled_agg.csv); this is a robustness/comparison
figure, not a replacement.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "exports/dissociations"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"
OUT_STUB = os.path.join(OUT_DIR, "dissociation_belief_vs_contribution_top_replacement")

ROWS = [
    ("Injection at round 10",
     f"{DATA_DIR}/dissociation_gap_full_top_round10.csv",
     f"{DATA_DIR}/untouched_levels_full_top_round10.csv"),
    ("Injection at round 20",
     f"{DATA_DIR}/dissociation_gap_full_top_round20.csv",
     f"{DATA_DIR}/untouched_levels_full_top_round20.csv"),
]
COL_TITLES = ["Gap: non-adversarial $-$ adversarial",
              "Non-adversarial agents level"]
SERIES = [("belief", "NE", "#4C72B0", "-"),
          ("contribution", "Contribution", "#C44E52", "--")]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), sharex=True, sharey="col")

    for row_idx, (row_label, gap_path, level_path) in enumerate(ROWS):
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
                ax.set_title(COL_TITLES[col_idx], fontsize=22, pad=12)
            if col_idx == 1 and row_idx == 0:
                ax.legend(fontsize=24, loc="lower left")

    for row_idx, (row_label, _, _) in enumerate(ROWS):
        axes[row_idx, 0].set_ylabel(f"{row_label}", fontsize=24)

    for col_idx in range(2):
        axes[1, col_idx].set_xlabel("Round offset from injection", fontsize=26)

    fig.tight_layout()
    fig.savefig(OUT_STUB + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_STUB + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_STUB}.png / .pdf")


if __name__ == "__main__":
    main()
