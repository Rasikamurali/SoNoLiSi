"""
Side-by-side version: the original 4 per-model columns (GPT, Llama-7B,
Mistral-7B, Qwen-7B — each Holm-corrected within its own family across the
4 conditions) plus a 5th "Pooled" column (all four models' round-level rows
combined, Holm-corrected across the 4 conditions on the pooled fit).

This is the table form of the check that the pooled result doesn't overturn
any individual model's early->late direction — every row's per-model values
and the pooled value sit next to each other so the direction/significance
can be read off directly, model by model.

Output: figures/2026-03-22/paper_stats/perception_consensus_combined_small_with_avg.tex
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from temp_eval_perception import (  # noqa: E402
    _load_perception_full, _per_family_ols, _PA_CONDS, _PA_COND_LABEL,
    _stars_tex, _PA_OUT,
)
from perception_consensus_avg7b import pooled_ols  # noqa: E402

SMALL_FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
OUT_FNAME      = "perception_consensus_combined_small_with_avg.tex"


def _panel_rows(family_results, pooled_results):
    rows = []
    for ci, cond in enumerate(_PA_CONDS):
        for period in ("early", "late"):
            cells = []
            for family in SMALL_FAMILIES:
                r    = family_results.get((family, cond), {})
                mean = r.get(f"{period}_mean", np.nan)
                n    = r.get(f"n_{period}", 0)
                if np.isnan(mean):
                    cells.append("---")
                    continue
                thin  = r"^{\circ}" if n < 15 else ""
                stars = _stars_tex(r.get("p_holm", 1.0)) if period == "late" else ""
                cells.append(f"${mean:.3f}{thin}{stars}$")

            r    = pooled_results.get(cond, {})
            mean = r.get(f"{period}_mean", np.nan)
            n    = r.get(f"n_{period}", 0)
            if np.isnan(mean):
                cells.append("---")
            else:
                thin  = r"^{\circ}" if n < 15 else ""
                stars = _stars_tex(r.get("p_holm", 1.0)) if period == "late" else ""
                cells.append(f"${mean:.3f}{thin}{stars}$")

            cond_str   = _PA_COND_LABEL[cond] if period == "early" else ""
            period_str = "Early" if period == "early" else "Late"
            rows.append(rf"  {cond_str} & {period_str} & " + " & ".join(cells) + r" \\")
        if ci < len(_PA_CONDS) - 1:
            rows.append(r"\midrule")
    return rows


def make_table(dn_family, dn_pooled, in_family, in_pooled, fname):
    os.makedirs(_PA_OUT, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Perceptual consensus (GPT and 7B models, plus pooled average): "
        r"mean cross-agent SD of \textit{DN} (Panel A) and \textit{IN} (Panel B) "
        r"per round, by period and condition. Lower SD $=$ greater within-run "
        r"agreement on the norm. Early $=$ rounds 1--3; Late $=$ last 3 rounds "
        r"per run. Per-model columns: significance stars Holm-corrected within "
        r"model across the 4 conditions. Pooled column: all four models' runs "
        r"combined, Holm-corrected across the 4 conditions on the pooled fit. "
        r"$^{\dagger}p{<}0.10$, $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$; "
        r"$^\circ N{<}15$.}",
        r"\label{tab:perception_consensus_combined_small_with_avg}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{ll ccccc}",
        r"\toprule",
        r"  Condition & Period & GPT & Llama-7B & Mistral-7B & Qwen-7B & Pooled \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Panel A: DN consensus (cross-agent SD of descriptive norm perceptions)}} \\[2pt]",
    ]
    lines += _panel_rows(dn_family, dn_pooled)
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{7}{l}{\textit{Panel B: IN consensus (cross-agent SD of injunctive norm perceptions)}} \\[2pt]"
    )
    lines += _panel_rows(in_family, in_pooled)
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]

    path = os.path.join(_PA_OUT, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved -> {path}")


def main():
    print("Loading perception data (all models; filtering to 7B family) ...")
    agent_df, round_df = _load_perception_full()
    round_small = round_df[round_df["family"].isin(SMALL_FAMILIES)].copy()
    sub = round_small[round_small["period"].isin(["early", "late"])].copy()

    # Per-family results reuse _per_family_ols as-is (it iterates the full
    # 9-family order internally; we only read out the 4 small-model entries).
    full_sub = round_df[round_df["period"].isin(["early", "late"])].copy()
    dn_family = _per_family_ols(full_sub, "DN_SD")
    in_family = _per_family_ols(full_sub, "IN_SD")

    dn_pooled = pooled_ols(sub, "DN_SD")
    in_pooled = pooled_ols(sub, "IN_SD")

    make_table(dn_family, dn_pooled, in_family, in_pooled, OUT_FNAME)


if __name__ == "__main__":
    main()
