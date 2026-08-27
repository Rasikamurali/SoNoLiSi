"""
community_group_contribution_tables.py
----------------------------------------
Compact statistical tables for community-size (N=12/16/20) and group-size
(G=3/4/6/8, two matched strata) structural robustness -- the contribution
(behavioral) half. Reports condition contrasts at each structural setting,
omnibus Condition x Setting interaction tests, and direct tests of whether
those contrasts change relative to the matched default (N=12 for community
size, G=4 for group size).

The central question is NOT "does N/G have a main effect on contribution."
It is: do the substantive experimental-condition effects change relative to
the matched default social structure?

Reuses analyze_structural_robustness.py's (asr) data build and OLS/Wald
machinery wholesale rather than rebuilding it -- that script already
implements almost everything here:
  - asr.load_axis / asr.qc_for_axis  -- run-level loader + QC (10 seeds/cell,
    latest-timestamp-wins dedup, one row per model x N/G x condition x seed).
  - asr.community_condition_contrasts -- PAPER_PAIRS contrasts per N, pooled
    with model FE, HC3, Bonferroni-corrected.
  - asr.community_interaction_models -- pooled Condition*N+ModelFE Wald test
    AND the Condition*N*Tier three-way interaction test (tier heterogeneity).
  - asr.group_interaction_models -- Condition*G+ModelFE Wald test fit
    SEPARATELY per stratum (N12: G in {3,4,6}; N16: G in {4,8}) -- never
    pools the two matched designs into one naive G=3/4/6/8 sweep.
This script adds only what didn't already exist: direct N-vs-12 / G-vs-4
contrast-difference tests (reading interaction coefficients off the
already-fitted pooled models, same covariance-aware trick as the MPCR
table), tier-specific contrasts (only built if the 3-way interaction test
is significant), and the compact table/CSV/notes formatting.

Output: figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/{community,group}/
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats

# Walk up from this file to find model_specs.py (shared across every theme
# folder), then add every code/analysis/ subfolder to sys.path so bare local
# imports keep working regardless of which theme folder a module lives in.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/behavioral/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# analyze_structural_robustness.py lives alongside this file, already on
# sys.path via the anchor-walk above.
import analyze_structural_robustness as asr  # noqa: E402  (the structural-robustness analysis this reuses)
from model_specs import stars, SIG_TEX  # noqa: E402

OUT_ROOT = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr"
COMMUNITY_OUT = os.path.join(OUT_ROOT, "community")
GROUP_OUT = os.path.join(OUT_ROOT, "group")

# Same 3 target contrasts as the MPCR table -- PAPER_PAIRS[0], [4], [3].
TARGET_CONTRASTS = [
    ("PURE_BASELINE", "BASELINE", "Baseline $-$ Pure Baseline"),
    ("NO_DISCUSSION",  "FULL",    "Full $-$ No Discussion"),
    ("NO_SELECTION",   "FULL",    "Full $-$ No Selection"),
]
assert all((a, b) in asr.PAPER_PAIRS for a, b, _ in TARGET_CONTRASTS)

TIER_ORDER = ["7B", "13B/14B", "70B/72B"]


# ═════════════════════════════════════════════════════════════════════════
# General linear-combination test (subsumes wald_contrast for both "contrast
# at a level" and "difference of contrast between levels" queries).
# ═════════════════════════════════════════════════════════════════════════

def linear_combo_test(result, weights):
    """weights: {term_name: coefficient} (a term absent from result.params
    is treated as the omitted/reference level, i.e. coefficient 0 and
    excluded from the covariance lookup)."""
    params, cov = result.params, result.cov_params()
    terms = [t for t in weights if t in params.index]
    if not terms:
        return 0.0, 0.0, np.nan, 1.0
    w = np.array([weights[t] for t in terms])
    beta = np.array([params[t] for t in terms])
    est = float(np.dot(w, beta))
    V = cov.loc[terms, terms].values
    var = float(w @ V @ w)
    se = float(np.sqrt(max(var, 0.0)))
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, z, p


def _inter_key(cond, level_val, axis_name, ref_level, ref_cond=None):
    ref_cond = ref_cond or asr.REF_COND
    if cond == ref_cond:
        return None
    return (f"C(condition, Treatment('{ref_cond}'))[T.{cond}]:"
            f"C({axis_name}, Treatment('{ref_level}'))[T.{level_val}]")


def contrast_change_test(result, cA, cB, level_val, axis_name, ref_level):
    """Delta_{cA->cB}(level) - Delta_{cA->cB}(ref_level): the change in one
    condition contrast relative to the matched reference level. Main-effect
    terms cancel algebraically (Treatment coding), leaving exactly the
    interaction-term difference between the two structural-axis levels."""
    key_b = _inter_key(cB, level_val, axis_name, ref_level)
    key_a = _inter_key(cA, level_val, axis_name, ref_level)
    weights = {}
    if key_b:
        weights[key_b] = weights.get(key_b, 0) + 1
    if key_a:
        weights[key_a] = weights.get(key_a, 0) - 1
    return linear_combo_test(result, weights)


# ═════════════════════════════════════════════════════════════════════════
# Part I.A -- Community size
# ═════════════════════════════════════════════════════════════════════════

def run_community(log):
    os.makedirs(COMMUNITY_OUT, exist_ok=True)
    log("\n" + "#" * 74 + "\n# COMMUNITY SIZE -- CONTRIBUTION\n" + "#" * 74)

    comm_df, comm_qc = asr.load_axis(asr.COMMUNITY_SOURCES, "N")
    asr.qc_for_axis(comm_df, comm_qc, asr.COMMUNITY_SOURCES, "N", "COMMUNITY SIZE", log)
    assert "gpt" not in comm_df["model"].unique(), "GPT must not appear in the community-size sweep"

    contrasts = asr.community_condition_contrasts(comm_df, log)
    contrasts.to_csv(os.path.join(COMMUNITY_OUT, "contribution_contrasts.csv"), index=False)

    pooled_res, pooled_het, res3, het3 = asr.community_interaction_models(comm_df, log)

    diff_rows = []
    for cA, cB, label in TARGET_CONTRASTS:
        for n in ["16", "20"]:
            est, se, z, p = contrast_change_test(pooled_res, cA, cB, n, "N", "12")
            diff_rows.append({"contrast": label, "comparison": f"N={n} vs N=12",
                              "estimate_diff": est, "se": se, "ci_lower": est - 1.96 * se,
                              "ci_upper": est + 1.96 * se, "z": z, "p_value": p})
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(COMMUNITY_OUT, "contribution_difference_tests.csv"), index=False)

    inter_rows = [{"scope": "Pooled (8 models, model FE)", "statistic": pooled_het[0],
                   "df": pooled_het[2], "p_value": pooled_het[1]} if pooled_het else
                  {"scope": "Pooled (8 models, model FE)", "statistic": np.nan, "df": np.nan, "p_value": np.nan},
                  {"scope": "Condition x N x Tier (3-way)", "statistic": het3[0] if het3 else np.nan,
                   "df": het3[2] if het3 else np.nan, "p_value": het3[1] if het3 else np.nan}]

    # Tier-specific contrasts: only built out if the 3-way interaction test is significant.
    tier_sig = het3 is not None and het3[1] < 0.05
    log(f"\n3-way Condition x N x Tier test p={het3[1]:.4g} ({'SIGNIFICANT' if tier_sig else 'not significant'}) "
        f"-> {'building' if tier_sig else 'skipping'} tier-specific contrast table.")
    tier_contrasts = pd.DataFrame()
    if tier_sig:
        tier_parts = []
        for tier in TIER_ORDER:
            tdf = comm_df[comm_df.tier == tier]
            if tdf.empty:
                continue
            tc = asr.community_condition_contrasts(tdf, log)
            tc["tier"] = tier
            tier_parts.append(tc)
        tier_contrasts = pd.concat(tier_parts, ignore_index=True) if tier_parts else pd.DataFrame()
    tier_contrasts.to_csv(os.path.join(COMMUNITY_OUT, "contribution_tier_heterogeneity.csv"), index=False)

    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_csv(os.path.join(COMMUNITY_OUT, "contribution_interaction_tests.csv"), index=False)

    tex = make_axis_table(
        contrasts, diff_df, levels=["12", "16", "20"], level_label="N",
        scope="Pooled8", pooled_het=pooled_het, het3=het3,
        title="Community-size", out_path=os.path.join(COMMUNITY_OUT, "contribution_statistical_table.tex"),
    )
    return dict(df=comm_df, contrasts=contrasts, diff_df=diff_df, pooled_het=pooled_het,
               het3=het3, tier_sig=tier_sig, tex=tex)


# ═════════════════════════════════════════════════════════════════════════
# Part I.B -- Group size (two matched strata, never pooled)
# ═════════════════════════════════════════════════════════════════════════

def run_group(log):
    os.makedirs(GROUP_OUT, exist_ok=True)
    log("\n" + "#" * 74 + "\n# GROUP SIZE -- CONTRIBUTION\n" + "#" * 74)

    g12_df, g12_qc = asr.load_axis(asr.GROUP_N12_SOURCES, "G")
    g12_df["stratum"] = "N12"
    g16_df, g16_qc = asr.load_axis(asr.GROUP_N16_SOURCES, "G")
    g16_df["stratum"] = "N16"
    asr.qc_for_axis(g12_df, g12_qc, asr.GROUP_N12_SOURCES, "G", "GROUP SIZE (N=12)", log)
    asr.qc_for_axis(g16_df, g16_qc, asr.GROUP_N16_SOURCES, "G", "GROUP SIZE (N=16)", log)
    group_df = pd.concat([g12_df, g16_df], ignore_index=True)
    for _df in (g12_df, g16_df):
        assert set(_df["model"].unique()) <= set(asr.SEVENB_MODELS), \
            "group-size sweep must be restricted to the 7B trio"

    strata_results = asr.group_interaction_models(group_df, log)  # {"N12": (res, het), "N16": (res, het)}

    strata_gvals = {"N12": ["3", "4", "6"], "N16": ["4", "8"]}
    contrasts_rows = []
    diff_rows = []
    inter_rows = []
    for stratum, gvals in strata_gvals.items():
        res, het = strata_results[stratum]
        inter_rows.append({"stratum": stratum, "statistic": het[0] if het else np.nan,
                           "df": het[2] if het else np.nan, "p_value": het[1] if het else np.nan})
        for cA, cB, label in TARGET_CONTRASTS:
            for g in gvals:
                # Contrast AT this G (absolute), via linear_combo_test on the SAME fitted model.
                key_bB, key_bA = asr.cond_param(cB), asr.cond_param(cA)
                weights = {}
                for k, sign in [(key_bB, 1), (key_bA, -1)]:
                    if k:
                        weights[k] = weights.get(k, 0) + sign
                if g != "4":
                    for k, sign in [(_inter_key(cB, g, "G", "4"), 1), (_inter_key(cA, g, "G", "4"), -1)]:
                        if k:
                            weights[k] = weights.get(k, 0) + sign
                est, se, z, p = linear_combo_test(res, weights)
                contrasts_rows.append({"stratum": stratum, "G": g, "contrast": label,
                                       "diff": est, "se": se, "p": p, "sig": stars(p)})
                if g != "4":
                    dest, dse, dz, dp = contrast_change_test(res, cA, cB, g, "G", "4")
                    diff_rows.append({"contrast": label, "comparison": f"{stratum}: G={g} vs G=4",
                                      "estimate_diff": dest, "se": dse, "ci_lower": dest - 1.96 * dse,
                                      "ci_upper": dest + 1.96 * dse, "z": dz, "p_value": dp})
    contrasts_df = pd.DataFrame(contrasts_rows)
    diff_df = pd.DataFrame(diff_rows)
    inter_df = pd.DataFrame(inter_rows)
    contrasts_df.to_csv(os.path.join(GROUP_OUT, "contribution_contrasts.csv"), index=False)
    diff_df.to_csv(os.path.join(GROUP_OUT, "contribution_difference_tests.csv"), index=False)
    inter_df.to_csv(os.path.join(GROUP_OUT, "contribution_interaction_tests.csv"), index=False)

    # Per-model (Llama/Mistral/Qwen) heterogeneity check -- only a table if divergent.
    model_rows = []
    for model in asr.SEVENB_MODELS:
        for stratum, gvals in strata_gvals.items():
            sub = group_df[(group_df.stratum == stratum) & (group_df.model == model)]
            if sub.empty or sub["G"].nunique() < 2:
                continue
            formula = f"mean_contribution ~ C(condition, Treatment('{asr.REF_COND}')) * C(G, Treatment('4'))"
            mres = asr.fit_ols_hc3(sub, formula)
            inter = [t for t in mres.params.index if ":" in t]
            if inter:
                wt = mres.wald_test(", ".join(f"{t} = 0" for t in inter), scalar=True)
                model_rows.append({"model": model, "stratum": stratum,
                                   "statistic": float(np.squeeze(wt.statistic)),
                                   "df": len(inter), "p_value": float(np.squeeze(wt.pvalue))})
    model_het_df = pd.DataFrame(model_rows)
    model_het_df.to_csv(os.path.join(GROUP_OUT, "contribution_model_heterogeneity.csv"), index=False)
    model_sig = bool((model_het_df["p_value"] < 0.05).any()) if not model_het_df.empty else False
    log(f"\nPer-model (Llama/Mistral/Qwen) Condition x G heterogeneity: "
        f"{'at least one model diverges (p<0.05)' if model_sig else 'no model diverges from pooled'}.")

    tex = make_group_table(contrasts_df, diff_df, inter_df,
                           out_path=os.path.join(GROUP_OUT, "contribution_statistical_table.tex"))
    return dict(group_df=group_df, contrasts=contrasts_df, diff_df=diff_df,
               inter_df=inter_df, model_het=model_het_df, model_sig=model_sig, tex=tex)


# ═════════════════════════════════════════════════════════════════════════
# Table builders
# ═════════════════════════════════════════════════════════════════════════

def make_axis_table(contrasts, diff_df, levels, level_label, scope, pooled_het, het3, title, out_path):
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(levels) + "}", r"\toprule",
        " & " + " & ".join(f"${level_label}={lv}$" for lv in levels) + r" \\", r"\midrule",
        r"\multicolumn{" + str(1 + len(levels)) + r"}{l}{\textit{Condition contrasts (pooled, model FE)}} \\",
    ]
    for cA, cB, label in TARGET_CONTRASTS:
        cells = []
        for lv in levels:
            row = contrasts[(contrasts[level_label] == lv) & (contrasts.scope == scope) &
                            (contrasts.contrast == f"{cA} -> {cB}")]
            cells.append(f"{row.iloc[0]['diff']:.3f} ({row.iloc[0]['se']:.3f}){SIG_TEX[row.iloc[0]['sig']]}"
                        if not row.empty else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{" + str(1 + len(levels)) + r"}{l}{\textit{Change relative to " +
                 f"${level_label}={levels[0]}$" + r"}} \\")
    for cA, cB, label in TARGET_CONTRASTS:
        cells = ["--"]
        for lv in levels[1:]:
            row = diff_df[(diff_df.contrast == label) & (diff_df.comparison == f"{level_label}={lv} vs {level_label}={levels[0]}")]
            if row.empty:
                row = diff_df[(diff_df.contrast == label) & (diff_df.comparison.str.contains(f"{level_label}={lv}"))]
            cells.append(f"{row.iloc[0]['estimate_diff']:+.3f} ({row.iloc[0]['se']:.3f}){stars(row.iloc[0]['p_value'])}"
                        if not row.empty else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    het_txt = (rf"Wald $\chi^2({pooled_het[2]})={pooled_het[0]:.2f}$, $p={pooled_het[1]:.3g}$"
              if pooled_het else "not estimable")
    het3_txt = (rf"Wald $\chi^2({het3[2]})={het3[0]:.2f}$, $p={het3[1]:.3g}$" if het3 else "not estimable")
    lines.append(rf"Condition $\times$ {level_label} & \multicolumn{{{len(levels)}}}{{c}}{{{het_txt}}} \\")
    lines.append(rf"Condition $\times$ {level_label} $\times$ Tier & \multicolumn{{{len(levels)}}}{{c}}{{{het3_txt}}} \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (rf"\caption{{\textbf{{{title} robustness (contribution).}} Entries report condition contrasts in "
         r"run-level mean contribution (SE in parentheses), pooled across available model variants with "
         r"model fixed effects and HC3-robust SEs. GPT is not part of this sweep. \textit{Change relative} "
         rf"rows test $\Delta_k({level_label})-\Delta_k({levels[0]})$ directly via the pooled interaction "
         r"model's covariance matrix, not inferred from significance switching. "
         r"$^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def make_group_table(contrasts_df, diff_df, inter_df, out_path):
    cols = [("N12", "3"), ("N12", "6"), ("N16", "8")]
    headers = [r"$G=3$ vs.\ $G=4$ (N=12)", r"$G=6$ vs.\ $G=4$ (N=12)", r"$G=8$ vs.\ $G=4$ (N=16)"]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
        " & " + " & ".join(headers) + r" \\", r"\midrule",
    ]
    for cA, cB, label in TARGET_CONTRASTS:
        cells = []
        for stratum, g in cols:
            row = diff_df[(diff_df.contrast == label) & (diff_df.comparison == f"{stratum}: G={g} vs G=4")]
            cells.append(f"{row.iloc[0]['estimate_diff']:+.3f} ({row.iloc[0]['se']:.3f}){stars(row.iloc[0]['p_value'])}"
                        if not row.empty else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    for _, r in inter_df.iterrows():
        cells = rf"\multicolumn{{{len(cols)}}}{{c}}{{Wald $\chi^2({int(r['df']) if pd.notna(r['df']) else 0})={r['statistic']:.2f}$, $p={r['p_value']:.3g}$}}"
        lines.append(rf"Condition $\times$ G ({r['stratum']}) & {cells} \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{\textbf{Group-size robustness (contribution).} Each cell is the change in that "
         r"condition contrast relative to its matched $G=4$ reference "
         r"($\Delta_k(G)-\Delta_k(4)$, SE in parentheses) -- not a raw contribution difference between "
         r"group sizes. $G3$/$G6$ vs.\ $G4$ use the $N=12$ stratum; $G8$ vs.\ $G4$ uses the $N=16$ "
         r"stratum (never compared to the $N=12,G=4$ default). Pooled across the 7B trio (Llama, "
         r"Mistral, Qwen) with model FE, HC3-robust SEs. "
         r"$^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    log_lines = []
    def log(msg):
        log_lines.append(str(msg))
        print(msg)

    comm = run_community(log)
    group = run_group(log)

    os.makedirs(COMMUNITY_OUT, exist_ok=True)
    with open(os.path.join(COMMUNITY_OUT, "contribution_qc_and_stats.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n{'=' * 74}\nCOMMUNITY-SIZE TABLE\n{'=' * 74}")
    print(comm["tex"])
    print(f"\n{'=' * 74}\nGROUP-SIZE TABLE\n{'=' * 74}")
    print(group["tex"])

    print(f"\nCommunity outputs -> {COMMUNITY_OUT}")
    print(f"Group outputs -> {GROUP_OUT}")
    return comm, group


if __name__ == "__main__":
    main()
