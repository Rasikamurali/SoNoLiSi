"""
plot_dissociation_belief_vs_contribution.py
----------------------------------------------
Regenerates figures/2026-03-22/paper_stats/dissociation_belief_vs_contribution.png/.pdf
as a 2x2 grid -- rows = injection round (20 / 10), FULL condition only
(discussion + selection both on).

Column 1: the non-replaced-minus-replaced GAP (as before), FULL condition only.
  Row 1 (round 20) comes from the saved aggregate
  (exports/dissociations/dissociation_pooled_agg.csv) -- the original script
  that produced it from raw logs is lost, only this aggregate output survived.
  Row 2 (round 10) is the reconstruction from build_dissociation_round10.py
  (see that script's docstring for methodology + validation caveats: fit to
  the round-20 saved aggregate to within ~0.13-0.18 mean abs deviation, not
  bit-exact, since the original round-20 pipeline is unrecoverable).

Column 2: what happens to the 8 untouched (non-replaced) agents themselves --
  their own raw mean contribution and injunctive-norm level (not a gap) around
  the round where 4 agents carrying a conflicting (low-contribution)
  expectation are hijacked into adversarial behavior. Built by
  build_untouched_agents_levels.py from the same raw logs, FULL condition
  only, both injection rounds (this metric has no saved original to compare
  against -- it's new for both rows).
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROUND20_GAP = "exports/dissociations/dissociation_pooled_agg.csv"
DATA_ROUND10_GAP = "exports/dissociations/dissociation_round10_pooled_agg.csv"
DATA_ROUND20_LEVEL = "exports/dissociations/untouched_levels_full_round20.csv"
DATA_ROUND10_LEVEL = "exports/dissociations/untouched_levels_full_round10.csv"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"
OUT_STUB = os.path.join(OUT_DIR, "dissociation_belief_vs_contribution")

ROWS = [
    ("Injection at round 10", DATA_ROUND10_GAP, DATA_ROUND10_LEVEL),
    ("Injection at round 20", DATA_ROUND20_GAP, DATA_ROUND20_LEVEL),
]
SERIES = [("belief", "NE", "#4C72B0", "-"),
          ("contribution", "Contribution", "#DD8452", "--")]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), sharex=True, sharey="col")

    for row_idx, (row_label, gap_path, level_path) in enumerate(ROWS):
        gap_d = pd.read_csv(gap_path)
        gap_d = gap_d[gap_d["discussion_on"] == 1]
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
            ax.tick_params(axis="both", labelsize=26)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            if col_idx == 1 and row_idx == 0:
                ax.legend(fontsize=28, loc="lower left")

    for row_idx, (row_label, _, _) in enumerate(ROWS):
        axes[row_idx, 0].set_ylabel(f"{row_label}", fontsize=28, labelpad=14)

    for col_idx in range(2):
        axes[1, col_idx].set_xlabel("Round offset from injection", fontsize=30)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(OUT_STUB + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_STUB + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_STUB}.png / .pdf")


if __name__ == "__main__":
    main()
