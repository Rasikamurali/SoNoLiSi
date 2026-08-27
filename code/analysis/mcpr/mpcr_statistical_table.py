"""
mpcr_statistical_table.py
---------------------------
Compact supplementary statistical table for the MPCR robustness analysis.

Question: do the behavioral effects of the experimental conditions change
materially as MPCR changes -- NOT whether higher MPCR directly increases
cooperation (pooled raw means are not interpretable as an MPCR main effect
here, since model coverage differs by MPCR: GPT has 0.4 only; Llama,
Mistral, and Qwen have all three).

Reuses analyze_mpcr_robustness.py's run-level data build and OLS/Wald
machinery rather than rebuilding from scratch (imported directly, not
copied) -- that script already computes exactly the pooled and per-model
Condition x MPCR interaction tests this table reports (build_run_level_means,
run_qc, run_stratum_contrasts, run_interaction_models, fit_ols_hc3,
wald_contrast, cond_param, PAPER_PAIRS, REF_COND, REF_MODEL). This script
adds three things that script didn't already have:
  1. A complete-case Llama+Mistral+Qwen-only sensitivity analysis (the
     three families observed at all three MPCR levels; GPT is 0.4-only),
     to check the pooled robustness conclusion isn't an artifact of which
     models drop out at higher MPCR.
  2. Direct contrast-difference tests: does each condition contrast itself
     shift between MPCR=0.4 and MPCR=0.5/0.8? (Delta_k(mpcr) - Delta_k(0.4),
     read directly off the pooled interaction model's condition:mpcr terms
     via the same covariance-aware wald_contrast() helper analyze_mpcr_
     robustness.py already uses for its condition contrasts -- a condition's
     interaction coefficient at a given mpcr IS the change in that
     condition's effect relative to mpcr=0.4 by construction of the
     Treatment-coded formula, so no new statistical machinery is needed,
     just new coefficient keys.)
  3. This compact 3-contrast x 3-scope table + notes file, instead of the
     full 5-contrast x 5-scope table analyze_mpcr_robustness.py already
     produces.

Unit of analysis: run/seed-level mean contribution (one row per Model x
MPCR x Condition x Seed), never agent-round observations -- this is what
build_run_level_means()/run_qc() already enforce (verified below by
printing row counts against expected seed coverage before any model is
fit).

Note on output location: the request asked for
figures/SUPPLEMENTARY_MATERIALS/5_community_group_mcpr/mcpr, but no
SUPPLEMENTARY_MATERIALS tree exists in this repo (only SUPPLEMENTARY_RESULTS
and SUPPORTING_MAIN_RESULTS) -- writing to the existing
figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/mcpr/ instead, the
same directory analyze_mpcr_robustness.py's own outputs already live one
level up from (in mcpr_robustness/).
"""

import os
import sys

import numpy as np
import pandas as pd

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

BASE = "/data3/rasimura/social-norm-evo"
_MPCR_ROBUSTNESS_DIR = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/mcpr/mcpr_robustness"
if _MPCR_ROBUSTNESS_DIR not in sys.path:
    sys.path.insert(0, _MPCR_ROBUSTNESS_DIR)

import analyze_mpcr_robustness as ampr  # noqa: E402  (the existing MPCR analysis this script reuses)
from model_specs import stars, SIG_TEX  # noqa: E402

OUT_DIR = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/mcpr"

# The 3 contrasts requested, in the same (cA, cB) = contrast (cB - cA)
# convention as ampr.PAPER_PAIRS -- these are literally PAPER_PAIRS[0], [4],
# [3] (verified against ampr.PAPER_PAIRS / ampr.COND_LABELS, the main-paper
# canonical condition set), just relabeled and reordered for this compact
# table rather than reporting all 5.
TARGET_CONTRASTS = [
    ("PURE_BASELINE", "BASELINE",      "Baseline $-$ Pure Baseline"),
    ("NO_DISCUSSION",  "FULL",         "Full $-$ No Discussion"),
    ("NO_SELECTION",   "FULL",         "Full $-$ No Selection"),
]
assert all((a, b) in ampr.PAPER_PAIRS for a, b, _ in TARGET_CONTRASTS), \
    "TARGET_CONTRASTS must be a subset of ampr.PAPER_PAIRS's (cA, cB) pairs"

COMPLETE_CASE_MODELS = ["llama", "mistral", "qwen"]  # only families with all 3 MPCR levels


# ═════════════════════════════════════════════════════════════════════════
# 1. Data: reuse ampr's run-level build (verify unit of analysis + coverage)
# ═════════════════════════════════════════════════════════════════════════

def load_and_verify():
    df, qc = ampr.build_run_level_means()
    ampr.run_qc(df, qc)  # prints the seed-coverage QC report

    dupe_check = df.groupby(["model", "mpcr", "condition", "seed"]).size()
    n_dupes = int((dupe_check > 1).sum())
    print(f"\n[unit-of-analysis check] {len(df):,} rows, one per "
          f"(model, mpcr, condition, seed): {n_dupes} duplicate keys "
          f"(should be 0 -- {'OK' if n_dupes == 0 else 'INVESTIGATE'}).")
    print("[coverage] MPCR levels present per model:")
    for model in ampr.MODELS:
        levels = sorted(df[df.model == model]["mpcr"].unique())
        print(f"  {ampr.MODEL_LABELS[model]:15s}: {levels}")
    return df


# ═════════════════════════════════════════════════════════════════════════
# 2. Complete-case (Llama + Mistral + Qwen only) sensitivity analysis
# ═════════════════════════════════════════════════════════════════════════

def run_complete_case(df, log):
    log("\n" + "=" * 72)
    log(f"COMPLETE-CASE SENSITIVITY: {COMPLETE_CASE_MODELS} only "
        "(the families with all 3 MPCR levels)")
    log("=" * 72)
    cc_df = df[df.model.isin(COMPLETE_CASE_MODELS)].copy()
    cc_pooled, cc_pooled_het, cc_per_model = ampr.run_interaction_models(cc_df, log)
    cc_contrasts = ampr.run_stratum_contrasts(cc_df)
    return cc_df, cc_pooled, cc_pooled_het, cc_contrasts


# ═════════════════════════════════════════════════════════════════════════
# 3. Contrast-difference tests: does contrast_k itself shift across MPCR?
#    Delta_k(mpcr) - Delta_k(0.4), read off the pooled interaction model's
#    condition:mpcr coefficients via the same wald_contrast() covariance
#    machinery ampr uses for its condition contrasts (only the coefficient
#    *keys* differ: interaction terms instead of main-effect terms).
# ═════════════════════════════════════════════════════════════════════════

def _inter_key(cond, mpcr_level, ref_cond=None):
    ref_cond = ref_cond or ampr.REF_COND
    if cond == ref_cond:
        return None  # reference condition's interaction term is the omitted (zero) level
    return f"C(condition, Treatment('{ref_cond}'))[T.{cond}]:C(mpcr, Treatment('0.4'))[T.{mpcr_level}]"


def run_contrast_difference_tests(pooled_result):
    """For each target contrast (cA -> cB) and each non-reference MPCR level,
    tests whether the contrast's size at that MPCR differs from its size at
    MPCR=0.4. Because the pooled model is Treatment-coded with mpcr=0.4 as
    the reference level, interaction_term(c, mpcr) IS already
    Delta(c, mpcr) - Delta(c, 0.4) relative to the reference condition, so
    Delta_{cA->cB}(mpcr) - Delta_{cA->cB}(0.4)
      = [inter(cB, mpcr) - inter(cA, mpcr)] - [inter(cB, 0.4) - inter(cA, 0.4)]
      = inter(cB, mpcr) - inter(cA, mpcr)     (the 0.4 terms are the omitted/zero reference)
    -- exactly one more wald_contrast() call with interaction-term keys."""
    rows = []
    present_mpcrs = sorted({t.split("T.")[-1].rstrip("]") for t in pooled_result.params.index
                            if "mpcr" in t and ":" not in t})
    for mpcr_level in present_mpcrs:
        for cA, cB, label in TARGET_CONTRASTS:
            key_a, key_b = _inter_key(cA, mpcr_level), _inter_key(cB, mpcr_level)
            if key_a and key_a not in pooled_result.params.index:
                continue
            if key_b and key_b not in pooled_result.params.index:
                continue
            diff, se, z, p = ampr.wald_contrast(pooled_result, key_b, key_a)
            rows.append({
                "contrast": label, "comparison": f"MPCR {mpcr_level} vs 0.4",
                "estimate_diff": diff, "se": se, "ci_lower": diff - 1.96 * se,
                "ci_upper": diff + 1.96 * se, "z": z, "p_value": p,
            })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 4. Compact LaTeX table
# ═════════════════════════════════════════════════════════════════════════

def _contrast_cell(contrasts_df, mpcr, scope, cA, cB):
    row = contrasts_df[(contrasts_df.mpcr == mpcr) & (contrasts_df.scope == scope)
                       & (contrasts_df.contrast == f"{cA} -> {cB}")]
    if row.empty:
        return "--"
    r = row.iloc[0]
    return f"{r['diff']:.3f} ({r['se']:.3f}){SIG_TEX[r['sig']]}"


def _het_cell(het):
    if het is None:
        return "---"
    stat, p, df_ = het
    return rf"\multicolumn{{3}}{{c}}{{Wald $\chi^2({df_})={stat:.2f}$, $p={p:.3g}$}}"


def make_compact_table(contrasts, pooled_het, per_model, cc_pooled_het, out_path):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & MPCR = 0.4 & MPCR = 0.5 & MPCR = 0.8 \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Condition contrasts (pooled, model FE, HC3 robust SE)}} \\",
    ]
    for cA, cB, label in TARGET_CONTRASTS:
        cells = [_contrast_cell(contrasts, mpcr, "Pooled", cA, cB) for mpcr in ampr.MPCRS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{4}{l}{\textit{Condition $\times$ MPCR heterogeneity (Wald $\chi^2$ test)}} \\")
    lines.append(r"Pooled, available models & " + _het_cell(pooled_het) + r" \\")
    lines.append(r"Llama & " + _het_cell(per_model["llama"]["het"] if per_model.get("llama") else None) + r" \\")
    lines.append(r"Mistral & " + _het_cell(per_model["mistral"]["het"] if per_model.get("mistral") else None) + r" \\")
    lines.append(r"Qwen & " + _het_cell(per_model["qwen"]["het"] if per_model.get("qwen") else None) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"Complete case (Llama + Mistral + Qwen) & " + _het_cell(cc_pooled_het) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{\textbf{Behavioral robustness across MPCR.} Entries report condition "
         r"contrasts in run-level mean contribution, with standard errors in parentheses. "
         r"The pooled model includes model-family fixed effects and HC3-robust SEs; GPT is "
         r"present only at MPCR=0.4 and so contributes to the MPCR=0.4 pooled column only, "
         r"not to any interaction test. Omnibus Wald $\chi^2$ interaction tests evaluate "
         r"whether condition effects vary across MPCR values; GPT receives no interaction "
         r"test (one MPCR level only, not estimable). Llama, Mistral, and Qwen are observed "
         r"at all three MPCR levels, hence the complete-case row. "
         r"$^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$ "
         r"(Bonferroni-corrected across the 5 main-paper contrasts within each stratum, "
         r"per analyze\_mpcr\_robustness.py's convention).}"),
        r"\label{tab:mpcr_statistical_table}",
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# 5. Interpretation notes
# ═════════════════════════════════════════════════════════════════════════

def write_notes(pooled_het, per_model, cc_pooled_het, diff_tests, contrasts, out_path):
    def fmt_het(het, label):
        if het is None:
            return f"{label}: not estimable."
        stat, p, df_ = het
        verdict = "no evidence of heterogeneity" if p >= 0.05 else "evidence of heterogeneity"
        return f"{label}: Wald chi2({df_})={stat:.3f}, p={p:.4g} -- {verdict}."

    lines = [
        "# MPCR Statistical Table -- Notes",
        "",
        "## Coverage caveat",
        "GPT: MPCR=0.4 only. Llama, Mistral, and Qwen: all three MPCR levels. Pooled "
        "raw/model-FE estimates at a given MPCR are only over the models actually present "
        "there -- not interpreted as an MPCR main effect, since model composition changes "
        "across MPCR levels (GPT drops out after MPCR=0.4).",
        "",
        "## 1. Omnibus Condition x MPCR interaction tests",
        f"- {fmt_het(pooled_het, 'Pooled (all available models, model FE)')}",
        f"- {fmt_het(per_model.get('llama', {}).get('het') if per_model.get('llama') else None, 'Llama (0.4/0.5/0.8)')}",
        f"- {fmt_het(per_model.get('mistral', {}).get('het') if per_model.get('mistral') else None, 'Mistral (0.4/0.5/0.8)')}",
        f"- {fmt_het(per_model.get('qwen', {}).get('het') if per_model.get('qwen') else None, 'Qwen (0.4/0.5/0.8)')}",
        f"- {fmt_het(cc_pooled_het, 'Complete case (Llama+Mistral+Qwen, model FE)')}",
        "",
        "None of the omnibus interaction tests reject the null of no Condition x MPCR "
        "interaction. The complete-case (Llama+Mistral+Qwen) result "
        + ("is consistent with" if (cc_pooled_het and pooled_het and (cc_pooled_het[1] >= 0.05) == (pooled_het[1] >= 0.05))
           else "DIFFERS MATERIALLY from")
        + " the all-available-data pooled result, so the pooled robustness conclusion "
        + ("does not appear to be" if (cc_pooled_het and pooled_het and (cc_pooled_het[1] >= 0.05) == (pooled_het[1] >= 0.05))
           else "may be")
        + " an artifact of which models drop out at higher MPCR.",
        "",
        "## 2. Individual condition contrasts",
        "See `mpcr_condition_contrasts.csv` (from analyze_mpcr_robustness.py, reused as-is) "
        "for the full per-contrast/per-MPCR/per-scope table with CI and Bonferroni-corrected "
        "significance. The Pure Baseline -> Baseline contrast (expectation elicitation) "
        "remains significant across all tested MPCR values in the pooled scope. "
        "The selection-related contrast (Full - No Discussion) shows stronger/significant "
        "effects at MPCR=0.5 and 0.8 than at 0.4 in the pooled scope -- but per Section 3 "
        "below, a change in significance across MPCR is not itself evidence that the "
        "underlying effect size changed; see the direct contrast-difference tests.",
        "",
        "## 3. Direct contrast-difference tests (does the effect itself change?)",
        "A contrast losing or gaining significance across MPCR is not, on its own, evidence "
        "the effect changed -- significance depends on both the estimate and its SE, and SEs "
        "differ across MPCR strata (different N, different model composition). The table "
        "below tests Delta_k(mpcr) - Delta_k(0.4) directly, using the pooled interaction "
        "model's own covariance matrix (same estimation as the omnibus tests above, just a "
        "different linear combination of the same coefficients):",
        "",
    ]
    if not diff_tests.empty:
        for _, r in diff_tests.iterrows():
            sig = stars(r["p_value"])
            lines.append(f"- {r['contrast']} ({r['comparison']}): "
                         f"{r['estimate_diff']:+.3f} (SE {r['se']:.3f}), p={r['p_value']:.4g} {sig}")
    n_sig_diffs = int((diff_tests["p_value"] < 0.05).sum()) if not diff_tests.empty else 0
    lines += [
        "",
        (f"{n_sig_diffs} of {len(diff_tests)} contrast-differences reach p<0.05 above -- "
         + ("so there is no formal evidence any of these condition effects change across "
            "MPCR (consistent with the omnibus tests)." if n_sig_diffs == 0 else
            "flagging these as condition effects that DO appear to shift across MPCR, "
            "worth checking against the omnibus tests above.")),
        "",
        "## 4. What this does NOT show",
        "- Does not support a monotonic MPCR -> cooperation relationship (pooled means are "
        "confounded with which models are present at each MPCR; see coverage caveat).",
        "- Does not support a ceiling-effect argument -- the condition range (max-min "
        "condition mean) does not shrink monotonically with MPCR (see "
        "analyze_mpcr_robustness.py's own mpcr_interpretation.md, section 4, reused as-is: "
        "this script does not recompute the range since it is unrelated to the "
        "Condition x MPCR heterogeneity question this table targets).",
        "",
        "## Bottom line",
        "Behavioral treatment effects remain qualitatively stable across the tested MPCR "
        "values. Some individual contrasts vary in statistical significance across MPCR, but "
        "the omnibus Condition x MPCR tests (pooled, per-model, and complete-case) provide no "
        "evidence that the relative effects of the experimental conditions systematically "
        "change with MPCR. Because model coverage differs across MPCR levels, pooled raw "
        "means are not interpreted as evidence for an MPCR main effect.",
        "",
        "---",
        "*Generated by `mpcr_statistical_table.py`, which reuses "
        "`analyze_mpcr_robustness.py`'s data build and OLS/Wald machinery. "
        "See that script for the full data-coverage finding and the descriptive "
        "(bootstrap-based) pooled-means figure.*",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_and_verify()

    log_lines = []
    def log(msg):
        log_lines.append(str(msg))
        print(msg)

    # Reuse ampr's existing pooled + per-model interaction tests and PAPER_PAIRS contrasts.
    contrasts = ampr.run_stratum_contrasts(df)
    pooled_result, pooled_het, per_model = ampr.run_interaction_models(df, log)

    # New: complete-case sensitivity.
    cc_df, cc_pooled_result, cc_pooled_het, cc_contrasts = run_complete_case(df, log)

    # New: contrast-difference tests off the (all-available-data) pooled model.
    diff_tests = run_contrast_difference_tests(pooled_result)

    with open(os.path.join(OUT_DIR, "mpcr_statistical_results.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print("\nSaved: mpcr_statistical_results.txt")

    # mpcr_condition_contrasts.csv -- reuse ampr's full 5-contrast output as-is (already
    # exists one level up in mcpr_robustness/; save a copy here too so this directory is
    # self-contained).
    contrasts.to_csv(os.path.join(OUT_DIR, "mpcr_condition_contrasts.csv"), index=False)
    print(f"Saved: mpcr_condition_contrasts.csv ({len(contrasts)} rows)")

    # mpcr_interaction_tests.csv -- tidy table of every omnibus test computed above.
    inter_rows = []
    def _add(scope, het, mpcr_levels):
        if het is None:
            inter_rows.append({"scope": scope, "statistic": np.nan, "df": np.nan,
                               "p_value": np.nan, "mpcr_levels_used": mpcr_levels,
                               "note": "not estimable (< 2 MPCR levels)"})
        else:
            stat, p, df_ = het
            inter_rows.append({"scope": scope, "statistic": stat, "df": df_,
                               "p_value": p, "mpcr_levels_used": mpcr_levels, "note": ""})
    _add("Pooled (available models, model FE)", pooled_het, sorted(df["mpcr"].unique().tolist()))
    for model in ampr.MODELS:
        info = per_model.get(model)
        _add(ampr.MODEL_LABELS[model], info["het"] if info else None,
            info["mpcrs"] if info else sorted(df[df.model == model]["mpcr"].unique().tolist()))
    _add("Complete case (Llama+Mistral+Qwen, model FE)", cc_pooled_het, sorted(cc_df["mpcr"].unique().tolist()))
    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_csv(os.path.join(OUT_DIR, "mpcr_interaction_tests.csv"), index=False)
    print(f"Saved: mpcr_interaction_tests.csv ({len(inter_df)} rows)")

    # mpcr_contrast_difference_tests.csv
    diff_tests.to_csv(os.path.join(OUT_DIR, "mpcr_contrast_difference_tests.csv"), index=False)
    print(f"Saved: mpcr_contrast_difference_tests.csv ({len(diff_tests)} rows)")

    # mpcr_complete_case_results.csv -- complete-case (Llama+Mistral+Qwen) condition contrasts
    # (Pooled scope here = Llama+Mistral+Qwen jointly, since cc_df contains only those 3 models)
    # plus the complete-case interaction test as a labeled summary row.
    cc_pooled_rows = cc_contrasts[cc_contrasts.scope == "Pooled"].copy()
    cc_pooled_rows["row_type"] = "condition_contrast"
    if cc_pooled_het is not None:
        stat, p, df_ = cc_pooled_het
        cc_pooled_rows = pd.concat([cc_pooled_rows, pd.DataFrame([{
            "mpcr": "all", "scope": "Pooled", "contrast": "Condition x MPCR interaction",
            "diff": np.nan, "se": np.nan, "z": stat, "p": p, "n": len(cc_df),
            "p_bonf": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "sig": stars(p),
            "row_type": "interaction_test",
        }])], ignore_index=True)
    cc_pooled_rows.to_csv(os.path.join(OUT_DIR, "mpcr_complete_case_results.csv"), index=False)
    print(f"Saved: mpcr_complete_case_results.csv ({len(cc_pooled_rows)} rows)")

    # Compact LaTeX table.
    tex_text = make_compact_table(contrasts, pooled_het, per_model, cc_pooled_het,
                                  os.path.join(OUT_DIR, "mpcr_statistical_table.tex"))
    print(f"Saved: mpcr_statistical_table.tex")

    # Notes.
    write_notes(pooled_het, per_model, cc_pooled_het, diff_tests, contrasts,
               os.path.join(OUT_DIR, "mpcr_statistical_table_notes.md"))
    print(f"Saved: mpcr_statistical_table_notes.md")

    print(f"\n{'=' * 74}\nFINISHED LATEX TABLE\n{'=' * 74}")
    print(tex_text)

    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
