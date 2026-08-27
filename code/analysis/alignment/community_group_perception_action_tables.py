"""
community_group_perception_action_tables.py
----------------------------------------------
Compact statistical tables for community-size (N=12/16/20) and group-size
(G=3/4/6/8, two matched strata) structural robustness -- the
perception-action gap half. Same architecture as
community_group_contribution_tables.py (Part I) and the MPCR statistical
table: condition contrasts at each structural setting, omnibus Condition x
Setting interaction tests, direct tests of whether those contrasts change
relative to the matched default (N=12 / G=4).

Gap definition: unchanged from the rest of this codebase --
  NE_gap = injunctive_norm  - contribution   ("NE - Contribution")
  EE_gap = descriptive_norm - contribution   ("EE - Contribution")
(alignment/perception_action_gap_plot.py's GAP_LABELS/compute_seed_gaps,
alignment/gap_based_alignment.py's in_gap/dn_gap -- same measure, not
reinvented.) Analyzed separately throughout, never averaged together.

Scalar summary: late-round (rounds 16-20) mean gap per (model, N/G,
condition, seed) -- the same "late" convention used everywhere else in this
codebase (stability_analysis.py's LAST_N=5, analyze_structural_robustness.py's
LATE_ROUNDS, perception_consensus.py's panel 2), chosen because it's the
existing convention, not because it produces significance.

Reuse strategy: builds a run-level gap dataframe shaped exactly like
analyze_structural_robustness.py's (asr) load_axis() output (N/G, model,
tier, family, condition, seed columns), then feeds a copy of it -- with the
gap column temporarily renamed to "mean_contribution" -- into asr's
existing community_condition_contrasts / community_interaction_models /
group_interaction_models unchanged (they only look at the column *name*
"mean_contribution", not its semantics, so this reuses the exact same
formula/Wald-test code without duplicating it). The difference-test helpers
(linear_combo_test / contrast_change_test) are imported from
community_group_contribution_tables.py rather than redefined.

Output: figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/{community,group}/
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
_STRUCTURAL_DIR = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/community_group_robusntess"
if _STRUCTURAL_DIR not in sys.path:
    sys.path.insert(0, _STRUCTURAL_DIR)

import analyze_structural_robustness as asr  # noqa: E402
from gap_based_alignment import (  # noqa: E402
    COMMUNITY_SOURCES, GROUP_N12_SOURCES, GROUP_N16_SOURCES, load_latest_logs_for_seed,
)
from community_group_contribution_tables import (  # noqa: E402
    linear_combo_test, contrast_change_test, _inter_key,
)
from model_specs import stars, SIG_TEX  # noqa: E402

OUT_ROOT = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr"
COMMUNITY_OUT = os.path.join(OUT_ROOT, "community")
GROUP_OUT = os.path.join(OUT_ROOT, "group")

LATE_ROUNDS = list(range(16, 21))
GAP_OUTCOMES = [("NE_gap", "injunctive_norm", "Normative expectation (NE)"),
               ("EE_gap", "descriptive_norm", "Empirical expectation (EE)")]

# Only 2 of the 3 target contrasts apply here: PURE_BASELINE has NO
# perceptions field recorded at all (established throughout this codebase --
# gap_based_alignment.py, perception_consensus.py, etc. all exclude it for
# the same reason), so "Baseline - Pure Baseline" is not estimable for a
# perception-action-gap outcome. asr.wald_contrast/community_condition_contrasts
# silently treat a genuinely-absent (non-reference) condition coefficient the
# same as a reference-level zero, which would otherwise print a fabricated
# "0.000 (0.000)" for this row instead of "not estimable" -- excluded here
# rather than displayed wrong.
TARGET_CONTRASTS = [
    ("NO_DISCUSSION", "FULL", "Full $-$ No Discussion"),
    ("NO_SELECTION",  "FULL", "Full $-$ No Selection"),
]


# ═════════════════════════════════════════════════════════════════════════
# Data: late-round mean NE_gap/EE_gap, shaped like asr.load_axis()'s output
# (N/G, model, tier, family, condition, seed) so it slots directly into
# asr's existing condition-contrast / interaction-model functions.
# ═════════════════════════════════════════════════════════════════════════

def load_late_gap_axis(sources, axis_col):
    """sources: {axis_value: {model_key: {dir, seeds, family}}} (same shape
    as asr.COMMUNITY_SOURCES / GROUP_N12/N16_SOURCES). Returns one row per
    (axis_value, model, condition, seed) with late-round (16-20) mean
    NE_gap/EE_gap, averaged first across agents within a round then across
    the late rounds -- matching compute_seed_gaps' per-round agent-mean,
    just summarized further to one scalar per run instead of a trajectory."""
    rows = []
    for axis_val, model_map in sources.items():
        for model_key, src in model_map.items():
            for seed in src["seeds"]:
                seed_dir = os.path.join(src["dir"], f"seed{seed}")
                found = load_latest_logs_for_seed(seed_dir)
                for cond, d in found.items():
                    ne_vals, ee_vals = [], []
                    for r in d["round_logs"]:
                        if r["round"] not in LATE_ROUNDS:
                            continue
                        contribs = {k: float(v) for k, v in r["contributions"].items()}
                        percs = r.get("perceptions") or {}
                        round_ne, round_ee = [], []
                        for aid, perc in percs.items():
                            if not perc:
                                continue
                            actual = contribs.get(aid)
                            if actual is None:
                                continue
                            inj, desc = perc.get("injunctive_norm"), perc.get("descriptive_norm")
                            if inj is not None:
                                round_ne.append(float(inj) - actual)
                            if desc is not None:
                                round_ee.append(float(desc) - actual)
                        if round_ne:
                            ne_vals.append(float(np.mean(round_ne)))
                        if round_ee:
                            ee_vals.append(float(np.mean(round_ee)))
                    if not ne_vals and not ee_vals:
                        continue
                    rows.append({
                        axis_col: axis_val, "model": model_key,
                        "tier": asr.MODEL_TIER.get(model_key, "7B"),
                        "family": src["family"], "condition": cond, "seed": seed,
                        "NE_gap": float(np.mean(ne_vals)) if ne_vals else np.nan,
                        "EE_gap": float(np.mean(ee_vals)) if ee_vals else np.nan,
                        "n_late_rounds": len(ne_vals),
                    })
    return pd.DataFrame(rows)


def _qc_late_gap(df, sources, axis_col, axis_label, log):
    log(f"\n[{axis_label}] late-round ({LATE_ROUNDS[0]}-{LATE_ROUNDS[-1]}) gap coverage:")
    for axis_val, model_map in sources.items():
        for model_key, src in model_map.items():
            n = df[(df[axis_col] == axis_val) & (df.model == model_key)]["seed"].nunique()
            expected = len(src["seeds"])
            flag = "OK" if n == expected else f"MISSING ({n}/{expected})"
            log(f"  {axis_col}={axis_val:4s} {model_key:12s}: {n}/{expected} seeds -- {flag}")


def _for_asr(df, outcome_col):
    """Copy of df with `outcome_col` renamed to 'mean_contribution', the
    column name asr's community_condition_contrasts/interaction_models/
    group_interaction_models are hardcoded to look for -- they only use the
    column by name, not by semantic meaning, so this reuses those functions
    unchanged for a non-contribution outcome."""
    return df.rename(columns={outcome_col: "mean_contribution"}).dropna(subset=["mean_contribution"])


# ═════════════════════════════════════════════════════════════════════════
# Community size
# ═════════════════════════════════════════════════════════════════════════

def run_community_gap(log):
    os.makedirs(COMMUNITY_OUT, exist_ok=True)
    log("\n" + "#" * 74 + "\n# COMMUNITY SIZE -- PERCEPTION-ACTION GAP\n" + "#" * 74)

    gap_df = load_late_gap_axis(COMMUNITY_SOURCES, "N")
    _qc_late_gap(gap_df, COMMUNITY_SOURCES, "N", "COMMUNITY SIZE", log)
    assert "gpt" not in gap_df["model"].unique()

    results, contrasts_all, diff_all, inter_all = {}, [], [], []
    for outcome_col, _raw, outcome_label in GAP_OUTCOMES:
        log(f"\n{'-'*72}\n{outcome_label}\n{'-'*72}")
        sub = _for_asr(gap_df, outcome_col)
        contrasts = asr.community_condition_contrasts(sub, log)
        # Drop PAPER_PAIRS not in this outcome's TARGET_CONTRASTS -- specifically
        # PURE_BASELINE ones, which asr's contrast fn can't tell apart from a
        # genuine zero (see TARGET_CONTRASTS comment above).
        valid_contrast_names = {f"{a} -> {b}" for a, b, _ in TARGET_CONTRASTS}
        contrasts = contrasts[contrasts["contrast"].isin(valid_contrast_names)].copy()
        contrasts["outcome"] = outcome_col
        pooled_res, pooled_het, res3, het3 = asr.community_interaction_models(sub, log)

        diff_rows = []
        for cA, cB, label in TARGET_CONTRASTS:
            for n in ["16", "20"]:
                est, se, z, p = contrast_change_test(pooled_res, cA, cB, n, "N", "12")
                diff_rows.append({"outcome": outcome_col, "contrast": label, "comparison": f"N={n} vs N=12",
                                  "estimate_diff": est, "se": se, "ci_lower": est - 1.96 * se,
                                  "ci_upper": est + 1.96 * se, "z": z, "p_value": p})
        diff_df = pd.DataFrame(diff_rows)

        inter_rows = [
            {"outcome": outcome_col, "scope": "Pooled (8 models, model FE)",
             "statistic": pooled_het[0] if pooled_het else np.nan,
             "df": pooled_het[2] if pooled_het else np.nan, "p_value": pooled_het[1] if pooled_het else np.nan},
            {"outcome": outcome_col, "scope": "Condition x N x Tier (3-way)",
             "statistic": het3[0] if het3 else np.nan, "df": het3[2] if het3 else np.nan,
             "p_value": het3[1] if het3 else np.nan},
        ]

        results[outcome_col] = dict(contrasts=contrasts, diff_df=diff_df, pooled_het=pooled_het, het3=het3)
        contrasts_all.append(contrasts)
        diff_all.append(diff_df)
        inter_all += inter_rows

        # Tier-specific contrasts, only if the 3-way interaction is significant.
        tier_sig = het3 is not None and het3[1] < 0.05
        log(f"\n[{outcome_col}] 3-way Condition x N x Tier test p={het3[1] if het3 else float('nan'):.4g} "
            f"({'SIGNIFICANT' if tier_sig else 'not significant'}) -> "
            f"{'building' if tier_sig else 'skipping'} tier-specific contrast table.")
        if tier_sig:
            tier_parts = []
            for tier in ["7B", "13B/14B", "70B/72B"]:
                tdf = sub[sub.tier == tier]
                if tdf.empty:
                    continue
                tc = asr.community_condition_contrasts(tdf, log)
                tc = tc[tc["contrast"].isin(valid_contrast_names)].copy()
                tc["tier"] = tier
                tc["outcome"] = outcome_col
                tier_parts.append(tc)
            if tier_parts:
                pd.concat(tier_parts, ignore_index=True).to_csv(
                    os.path.join(COMMUNITY_OUT, f"perception_action_tier_heterogeneity_{outcome_col}.csv"),
                    index=False)

    contrasts_df = pd.concat(contrasts_all, ignore_index=True)
    diff_df = pd.concat(diff_all, ignore_index=True)
    inter_df = pd.DataFrame(inter_all)
    contrasts_df.to_csv(os.path.join(COMMUNITY_OUT, "perception_action_results.csv"), index=False)
    diff_df.to_csv(os.path.join(COMMUNITY_OUT, "perception_action_difference_tests.csv"), index=False)
    inter_df.to_csv(os.path.join(COMMUNITY_OUT, "perception_action_interaction_tests.csv"), index=False)

    tex = make_axis_table_split(results, levels=["12", "16", "20"], level_label="N", scope="Pooled8",
                                title="Community-size",
                                out_path=os.path.join(COMMUNITY_OUT, "perception_action_statistical_table.tex"))
    return dict(gap_df=gap_df, results=results, tex=tex)


# ═════════════════════════════════════════════════════════════════════════
# Group size
# ═════════════════════════════════════════════════════════════════════════

def run_group_gap(log):
    os.makedirs(GROUP_OUT, exist_ok=True)
    log("\n" + "#" * 74 + "\n# GROUP SIZE -- PERCEPTION-ACTION GAP\n" + "#" * 74)

    g12_df = load_late_gap_axis(GROUP_N12_SOURCES, "G")
    g12_df["stratum"] = "N12"
    g16_df = load_late_gap_axis(GROUP_N16_SOURCES, "G")
    g16_df["stratum"] = "N16"
    _qc_late_gap(g12_df, GROUP_N12_SOURCES, "G", "GROUP SIZE (N=12)", log)
    _qc_late_gap(g16_df, GROUP_N16_SOURCES, "G", "GROUP SIZE (N=16)", log)
    group_df = pd.concat([g12_df, g16_df], ignore_index=True)

    strata_gvals = {"N12": ["3", "4", "6"], "N16": ["4", "8"]}
    all_contrasts, all_diff, all_inter = [], [], []
    per_outcome = {}
    for outcome_col, _raw, outcome_label in GAP_OUTCOMES:
        log(f"\n{'-'*72}\n{outcome_label}\n{'-'*72}")
        sub = _for_asr(group_df, outcome_col)
        strata_results = asr.group_interaction_models(sub, log)

        contrasts_rows, diff_rows, inter_rows = [], [], []
        for stratum, gvals in strata_gvals.items():
            res, het = strata_results[stratum]
            inter_rows.append({"outcome": outcome_col, "stratum": stratum,
                               "statistic": het[0] if het else np.nan,
                               "df": het[2] if het else np.nan, "p_value": het[1] if het else np.nan})
            for cA, cB, label in TARGET_CONTRASTS:
                for g in gvals:
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
                    contrasts_rows.append({"outcome": outcome_col, "stratum": stratum, "G": g, "contrast": label,
                                           "diff": est, "se": se, "p": p, "sig": stars(p)})
                    if g != "4":
                        dest, dse, dz, dp = contrast_change_test(res, cA, cB, g, "G", "4")
                        diff_rows.append({"outcome": outcome_col, "contrast": label,
                                          "comparison": f"{stratum}: G={g} vs G=4",
                                          "estimate_diff": dest, "se": dse, "ci_lower": dest - 1.96 * dse,
                                          "ci_upper": dest + 1.96 * dse, "z": dz, "p_value": dp})
        per_outcome[outcome_col] = dict(contrasts=pd.DataFrame(contrasts_rows), diff_df=pd.DataFrame(diff_rows),
                                        inter_df=pd.DataFrame(inter_rows))
        all_contrasts.append(pd.DataFrame(contrasts_rows))
        all_diff.append(pd.DataFrame(diff_rows))
        all_inter.append(pd.DataFrame(inter_rows))

        # Per-model (Llama/Mistral/Qwen) heterogeneity, only if any stratum's
        # pooled interaction test is significant.
        any_sig = any(pd.notna(r["p_value"]) and r["p_value"] < 0.05 for r in inter_rows)
        log(f"\n[{outcome_col}] Condition x G interaction significant in >=1 stratum: {any_sig} -> "
            f"{'building' if any_sig else 'skipping'} per-model heterogeneity check.")
        if any_sig:
            model_rows = []
            for model in asr.SEVENB_MODELS:
                for stratum, gvals in strata_gvals.items():
                    msub = sub[(sub.stratum == stratum) & (sub.model == model)]
                    if msub.empty or msub["G"].nunique() < 2:
                        continue
                    formula = f"mean_contribution ~ C(condition, Treatment('{asr.REF_COND}')) * C(G, Treatment('4'))"
                    mres = asr.fit_ols_hc3(msub, formula)
                    minter = [t for t in mres.params.index if ":" in t]
                    if minter:
                        wt = mres.wald_test(", ".join(f"{t} = 0" for t in minter), scalar=True)
                        model_rows.append({"outcome": outcome_col, "model": model, "stratum": stratum,
                                           "statistic": float(np.squeeze(wt.statistic)), "df": len(minter),
                                           "p_value": float(np.squeeze(wt.pvalue))})
            if model_rows:
                pd.DataFrame(model_rows).to_csv(
                    os.path.join(GROUP_OUT, f"perception_action_model_heterogeneity_{outcome_col}.csv"), index=False)

    contrasts_df = pd.concat(all_contrasts, ignore_index=True)
    diff_df = pd.concat(all_diff, ignore_index=True)
    inter_df = pd.concat(all_inter, ignore_index=True)
    contrasts_df.to_csv(os.path.join(GROUP_OUT, "perception_action_results.csv"), index=False)
    diff_df.to_csv(os.path.join(GROUP_OUT, "perception_action_difference_tests.csv"), index=False)
    inter_df.to_csv(os.path.join(GROUP_OUT, "perception_action_interaction_tests.csv"), index=False)

    tex = make_group_table_split(per_outcome, out_path=os.path.join(GROUP_OUT, "perception_action_statistical_table.tex"))
    return dict(group_df=group_df, per_outcome=per_outcome, tex=tex)


# ═════════════════════════════════════════════════════════════════════════
# Table builders (NE/EE split panels)
# ═════════════════════════════════════════════════════════════════════════

def make_axis_table_split(results, levels, level_label, scope, title, out_path):
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(levels) + "}", r"\toprule",
        " & " + " & ".join(f"${level_label}={lv}$" for lv in levels) + r" \\", r"\midrule",
    ]
    for outcome_col, _raw, outcome_label in GAP_OUTCOMES:
        r_ = results[outcome_col]
        lines.append(r"\multicolumn{" + str(1 + len(levels)) + rf"}}{{l}}{{\textit{{{outcome_label}}}}} \\")
        for cA, cB, label in TARGET_CONTRASTS:
            cells = []
            for lv in levels:
                row = r_["contrasts"][(r_["contrasts"][level_label] == lv) & (r_["contrasts"].scope == scope) &
                                      (r_["contrasts"].contrast == f"{cA} -> {cB}")]
                cells.append(f"{row.iloc[0]['diff']:.3f} ({row.iloc[0]['se']:.3f}){SIG_TEX[row.iloc[0]['sig']]}"
                            if not row.empty else "--")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        pooled_het, het3 = r_["pooled_het"], r_["het3"]
        het_txt = rf"Wald $\chi^2({pooled_het[2]})={pooled_het[0]:.2f}$, $p={pooled_het[1]:.3g}$" if pooled_het else "n/a"
        het3_txt = rf"Wald $\chi^2({het3[2]})={het3[0]:.2f}$, $p={het3[1]:.3g}$" if het3 else "n/a"
        lines.append(rf"\quad Condition $\times$ {level_label} & \multicolumn{{{len(levels)}}}{{c}}{{{het_txt}}} \\")
        lines.append(rf"\quad Condition $\times$ {level_label} $\times$ Tier & \multicolumn{{{len(levels)}}}{{c}}{{{het3_txt}}} \\")
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (rf"\caption{{\textbf{{{title} robustness (perception-action gap).}} NE $=$ injunctive-norm "
         r"$-$ contribution; EE $=$ descriptive-norm $-$ contribution, both late-round "
         rf"(rounds {LATE_ROUNDS[0]}--{LATE_ROUNDS[-1]}) means. Entries are condition contrasts pooled "
         r"across available model variants with model fixed effects and HC3-robust SEs; GPT is not "
         r"part of this sweep. $^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def make_group_table_split(per_outcome, out_path):
    cols = [("N12", "3"), ("N12", "6"), ("N16", "8")]
    headers = [r"$G=3$ vs.\ $G=4$ (N=12)", r"$G=6$ vs.\ $G=4$ (N=12)", r"$G=8$ vs.\ $G=4$ (N=16)"]
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
        " & " + " & ".join(headers) + r" \\", r"\midrule",
    ]
    for outcome_col, _raw, outcome_label in GAP_OUTCOMES:
        r_ = per_outcome[outcome_col]
        lines.append(r"\multicolumn{" + str(1 + len(cols)) + rf"}}{{l}}{{\textit{{{outcome_label}}}}} \\")
        for cA, cB, label in TARGET_CONTRASTS:
            cells = []
            for stratum, g in cols:
                row = r_["diff_df"][(r_["diff_df"].contrast == label) &
                                    (r_["diff_df"].comparison == f"{stratum}: G={g} vs G=4")]
                cells.append(f"{row.iloc[0]['estimate_diff']:+.3f} ({row.iloc[0]['se']:.3f}){stars(row.iloc[0]['p_value'])}"
                            if not row.empty else "--")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        for _, r2 in r_["inter_df"].iterrows():
            cell = rf"\multicolumn{{{len(cols)}}}{{c}}{{Wald $\chi^2({int(r2['df']) if pd.notna(r2['df']) else 0})={r2['statistic']:.2f}$, $p={r2['p_value']:.3g}$}}"
            lines.append(rf"\quad Condition $\times$ G ({r2['stratum']}) & {cell} \\")
        lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{\textbf{Group-size robustness (perception-action gap).} NE $=$ injunctive-norm "
         rf"$-$ contribution; EE $=$ descriptive-norm $-$ contribution, late-round means "
         rf"(rounds {LATE_ROUNDS[0]}--{LATE_ROUNDS[-1]}). Each cell is the change in that condition "
         r"contrast relative to its matched $G=4$ reference. $G3$/$G6$ vs.\ $G4$ use the $N=12$ "
         r"stratum; $G8$ vs.\ $G4$ uses the $N=16$ stratum. Pooled across the 7B trio with model FE, "
         r"HC3-robust SEs. $^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
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

    comm = run_community_gap(log)
    group = run_group_gap(log)

    with open(os.path.join(COMMUNITY_OUT, "perception_action_qc_and_stats.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n{'=' * 74}\nCOMMUNITY-SIZE PERCEPTION-ACTION TABLE\n{'=' * 74}")
    print(comm["tex"])
    print(f"\n{'=' * 74}\nGROUP-SIZE PERCEPTION-ACTION TABLE\n{'=' * 74}")
    print(group["tex"])

    print(f"\nCommunity outputs -> {COMMUNITY_OUT}")
    print(f"Group outputs -> {GROUP_OUT}")
    return comm, group


if __name__ == "__main__":
    main()
