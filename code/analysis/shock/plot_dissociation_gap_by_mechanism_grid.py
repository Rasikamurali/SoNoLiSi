"""
plot_dissociation_gap_by_mechanism_grid.py
-------------------------------------------
Combines the per-condition dissociation-gap panels (built by
build_condition_variants.py for BASELINE/NO_SELECTION/NO_DISCUSSION, and
already present in dissociation_pooled_agg.csv / dissociation_round10_pooled_agg.csv
for FULL via discussion_on==1) into two grids of
"Gap: non-adversarial - adversarial" only (no non-adversarial-level column --
see plot_dissociation_condition_variants.py for that companion panel):

  dissociation_gap_grid_3mechanisms.{png,pdf}: 2 rows (injection at round 10,
      round 20) x 3 columns (Baseline / No Discussion / No Selection).
  dissociation_gap_grid_4mechanisms.{png,pdf}: same, + a 4th Full column.

Condition -> mechanism-notation mapping (E = expectations only, SS = social
selection, SL = social learning/discussion):
  Baseline      = E        (no selection, no discussion)
  No Discussion = E+SS     (selection on, discussion off)
  No Selection  = E+SL     (discussion on, selection off)
  Full          = E+SS+SL  (both on)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "/data3/rasimura/social-norm-evo/code/analysis/exports/dissociations"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/SUPPORTING_MAIN_RESULTS/8_shock_variants/figures"

SERIES = [("belief", "NE", "#4C72B0", "-"),
          ("contribution", "Contribution", "#C44E52", "--")]

CONDITIONS_3 = [
    ("baseline", "E"),
    ("no_discussion", "E+SS"),
    ("no_selection", "E+SL"),
]
CONDITIONS_4 = CONDITIONS_3 + [("full", "E+SS+SL")]

ROWS = [("round10", "Injection at round 10"), ("round20", "Injection at round 20")]


def load_gap(cond_tag, pivot_tag):
    if cond_tag == "full":
        src = "dissociation_pooled_agg.csv" if pivot_tag == "round20" else "dissociation_round10_pooled_agg.csv"
        d = pd.read_csv(os.path.join(DATA_DIR, src))
        return d[d["discussion_on"] == 1][["outcome_type", "offset", "mean", "sem", "n"]]
    return pd.read_csv(os.path.join(DATA_DIR, f"dissociation_gap_{cond_tag}_{pivot_tag}.csv"))


def make_grid(conditions, out_stub):
    ncols = len(conditions)
    fig, axes = plt.subplots(2, ncols, figsize=(6.4 * ncols, 12.5), sharex=True, sharey="row")

    for row_idx, (pivot_tag, row_label) in enumerate(ROWS):
        for col_idx, (cond_tag, cond_label) in enumerate(conditions):
            d = load_gap(cond_tag, pivot_tag)
            ax = axes[row_idx, col_idx]
            for outcome, label, color, ls in SERIES:
                sub = d[d["outcome_type"] == outcome].sort_values("offset")
                x = sub["offset"].to_numpy()
                y = sub["mean"].to_numpy()
                ci = 1.96 * sub["sem"].to_numpy()
                ax.plot(x, y, marker="o", color=color, linestyle=ls, label=label, linewidth=2)
                ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)
            ax.axvline(0, color="gray", lw=1, ls=":")
            ax.axhline(0, color="black", lw=1)
            ax.set_xticks(sorted(d["offset"].unique()))
            ax.tick_params(axis="both", labelsize=22)
            if row_idx == 0:
                ax.set_title(cond_label, fontsize=26, pad=14)
            if row_idx == 0 and col_idx == ncols - 1:
                ax.legend(fontsize=22, loc="lower left")

    for row_idx, (_, row_label) in enumerate(ROWS):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=22, labelpad=12)

    for col_idx in range(ncols):
        axes[1, col_idx].set_xlabel("Round offset from injection", fontsize=24)

    fig.tight_layout(rect=(0.03, 0, 1, 1))
    fig.supylabel("Gap: non-adversarial $-$ adversarial", fontsize=24, x=0.005)
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_stub}.png / .pdf")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_grid(CONDITIONS_3, os.path.join(OUT_DIR, "dissociation_gap_grid_3mechanisms"))
    make_grid(CONDITIONS_4, os.path.join(OUT_DIR, "dissociation_gap_grid_4mechanisms"))


if __name__ == "__main__":
    main()
