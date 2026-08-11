"""
Averaged version of perception_consensus_combined_small.tex: instead of one
column per 7B model (GPT, Llama-7B, Mistral-7B, Qwen-7B), pool all four
models' round-level rows together and fit a single early-vs-late OLS
(clustered by run_id) per condition. This is a proper pooled estimate, not a
naive average of the four already-rounded per-model numbers in the original
table — pooling first means each run contributes its own residual/cluster
rather than treating four point estimates as equally-weighted observations.

Holm correction is applied across the 4 conditions within each panel (DN/IN),
same as the per-family table, just with no family axis left to condition on.

Output: figures/2026-03-22/paper_stats/perception_consensus_combined_small_avg.tex
"""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from temp_eval_perception import (  # noqa: E402
    _load_perception_full, _PA_CONDS, _PA_COND_LABEL, _stars_tex, _PA_OUT,
)

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

SMALL_FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
OUT_FNAME      = "perception_consensus_combined_small_avg.tex"


def pooled_ols(df, outcome):
    """
    Pool rows from all SMALL_FAMILIES together. One OLS per condition
    (early vs late, clustered by run_id — run_id already encodes family, so
    clusters stay model-specific even though the fit is pooled). Holm-correct
    across the 4 conditions.
    """
    results = {}
    p_raws  = {}

    for cond in _PA_CONDS:
        sub = (df[df["condition"] == cond]
               .loc[lambda x: x["period"].isin(["early", "late"])]
               .dropna(subset=[outcome]))
        early_sub = sub[sub["period"] == "early"]
        late_sub  = sub[sub["period"] == "late"]
        n_early, n_late = len(early_sub), len(late_sub)

        coef, se, p_raw = np.nan, np.nan, 1.0
        if n_early >= 2 and n_late >= 2 and sub["run_id"].nunique() >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = smf.ols(
                        f"{outcome} ~ C(period, Treatment('early'))", data=sub
                    ).fit(cov_type="cluster", cov_kwds={"groups": sub["run_id"]})
                key   = "C(period, Treatment('early'))[T.late]"
                coef  = float(res.params.get(key, np.nan))
                se    = float(res.bse.get(key, np.nan))
                p_raw = float(res.pvalues.get(key, 1.0))
            except Exception:
                pass

        results[cond] = {
            "early_mean": early_sub[outcome].mean() if n_early else np.nan,
            "late_mean":  late_sub[outcome].mean()  if n_late  else np.nan,
            "n_early": n_early, "n_late": n_late,
            "coef": coef, "se": se, "p_raw": p_raw,
        }
        p_raws[cond] = p_raw

    cond_list = list(p_raws)
    _, p_holm_arr, _, _ = multipletests(
        [p_raws[c] for c in cond_list], alpha=0.05, method="holm"
    )
    for cond, p_h in zip(cond_list, p_holm_arr):
        results[cond]["p_holm"] = float(p_h)

    return results


def _panel_rows(results):
    rows = []
    for ci, cond in enumerate(_PA_CONDS):
        for period in ("early", "late"):
            r    = results.get(cond, {})
            mean = r.get(f"{period}_mean", np.nan)
            n    = r.get(f"n_{period}", 0)
            if np.isnan(mean):
                cell = "---"
            else:
                thin  = r"^{\circ}" if n < 15 else ""
                stars = _stars_tex(r.get("p_holm", 1.0)) if period == "late" else ""
                cell  = f"${mean:.3f}{thin}{stars}$"
            cond_str   = _PA_COND_LABEL[cond] if period == "early" else ""
            period_str = "Early" if period == "early" else "Late"
            rows.append(rf"  {cond_str} & {period_str} & {cell} \\")
        if ci < len(_PA_CONDS) - 1:
            rows.append(r"\midrule")
    return rows


def make_table(dn_res, in_res, fname):
    os.makedirs(_PA_OUT, exist_ok=True)

    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Perceptual consensus, pooled across the four 7B models "
        r"(GPT, Llama-7B, Mistral-7B, Qwen-7B): mean cross-agent SD of "
        r"\textit{DN} (Panel A) and \textit{IN} (Panel B) per round, by period "
        r"and condition. Lower SD $=$ greater within-run agreement on the norm. "
        r"Early $=$ rounds 1--3; Late $=$ last 3 rounds per run. "
        r"Significance stars on Late rows: Holm-corrected (across the 4 "
        r"conditions, within panel) period effect from OLS clustered by run "
        r"(runs from all four models pooled). "
        r"$^{\dagger}p{<}0.10$, $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$; "
        r"$^\circ N{<}15$.}",
        r"\label{tab:perception_consensus_combined_small_avg}",
        r"\begin{tabular}{ll c}",
        r"\toprule",
        r"  Condition & Period & Mean (7B, pooled) \\",
        r"\midrule",
        r"\multicolumn{3}{l}{\textit{Panel A: DN consensus (cross-agent SD of descriptive norm perceptions)}} \\[2pt]",
    ]
    lines += _panel_rows(dn_res)
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{3}{l}{\textit{Panel B: IN consensus (cross-agent SD of injunctive norm perceptions)}} \\[2pt]"
    )
    lines += _panel_rows(in_res)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(_PA_OUT, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved -> {path}")


def main():
    print("Loading perception data (all models; filtering to 7B family) ...")
    agent_df, round_df = _load_perception_full()
    round_small = round_df[round_df["family"].isin(SMALL_FAMILIES)].copy()
    sub = round_small[round_small["period"].isin(["early", "late"])].copy()
    print(f"  Pooled round-level rows (7B, early+late): {len(sub):,}")

    dn_res = pooled_ols(sub, "DN_SD")
    in_res = pooled_ols(sub, "IN_SD")
    make_table(dn_res, in_res, OUT_FNAME)


if __name__ == "__main__":
    main()
