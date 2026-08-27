"""
perception_consensus.py
------------------------
Cross-agent consensus (SD) of injunctive/descriptive norm perceptions, two
panels:

  Panel 1 (early-vs-late OLS): SD ~ C(period, Treatment('early')), fit per
      family and pooled across the tier's families, Holm-corrected across
      the 4 conditions. Early = rounds 1-3, Late = last 3 rounds per run.
  Panel 2 (continuous slope + final-round level): per-seed OLS slope of
      SD ~ round (centered), plus mean SD over the final 5 rounds, plus a
      mixed-effects (random intercept per seed) robustness check on the
      slope.

Both panels use the same underlying metric: contrib_sd / in_sd / dn_sd =
cross-agent SD per round per seed (>=2 valid values required). Lower SD =
faster/stronger consensus toward the shared norm.

Tiers (--tier flag, default 7b): 7b / s2 / pooled (family-based, same
family lists as gap_based_alignment.py) and community / group (new -- same
COMMUNITY_SOURCES / GROUP_N12_SOURCES / GROUP_N16_SOURCES data sources).

Output:
  7b        -> figures/MAIN_RESULTS/3_dn_in_convergence/
  s2        -> figures/SUPPLEMENTARY_RESULTS/2_bigger_models/3_dn_in_convergence/
  pooled    -> figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/subset_all_families/
  community -> figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/community/3_dn_in_convergence/<setting>/
  group     -> figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/group/3_dn_in_convergence/<setting>/
"""

import os
import sys
import argparse
import glob
import json
import shutil
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_specs import MODEL_SPECS  # noqa: E402
from gap_based_alignment import TIER_SPECS, STRUCTURAL_SETTINGS  # noqa: E402

warnings.filterwarnings("ignore")

# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/perception/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_LABEL = {"BASELINE": "Baseline", "NO_SELECTION": "No Selection",
             "NO_DISCUSSION": "No Discussion", "FULL": "Full"}
METRICS = ["contrib_sd", "in_sd", "dn_sd"]
METRIC_TEX = {"contrib_sd": "Contribution SD", "in_sd": "IN SD", "dn_sd": "DN SD"}
LAST_N = 5

EXPORT_ROOT = f"{BASE}/code/analysis/exports/perception_consensus"
PUBLISH_MAIN = f"{BASE}/figures/MAIN_RESULTS/3_dn_in_convergence"
PUBLISH_S2 = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/2_bigger_models/3_dn_in_convergence"
PUBLISH_POOLED = f"{BASE}/figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/subset_all_families"
PUBLISH_13B = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/3_13b_tier/3_dn_in_convergence"
PUBLISH_70B = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/4_70b_tier/3_dn_in_convergence"
PUBLISH_STRUCTURAL_ROOT = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr"


# ═════════════════════════════════════════════════════════════════════════
# 1. Data loading -- one round-level loader shared by both panels
# ═════════════════════════════════════════════════════════════════════════

def load_latest_logs_for_seed(seed_dir):
    best = {}
    for p in sorted(glob.glob(os.path.join(seed_dir, "log_*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        cond = d.get("condition")
        if cond in CONDITIONS:
            best[cond] = d
    return best


def _rows_from_log(family, run_id, cond, seed, d):
    rows = []
    max_round = max(r["round"] for r in d["round_logs"])
    for r in d["round_logs"]:
        cont = [float(v) for v in r["contributions"].values()]
        percs = r.get("perceptions") or {}
        in_vals, dn_vals = [], []
        for perc in percs.values():
            if not perc:
                continue
            inj, desc = perc.get("injunctive_norm"), perc.get("descriptive_norm")
            if inj is not None:
                in_vals.append(float(inj))
            if desc is not None:
                dn_vals.append(float(desc))
        rows.append({
            "family": family, "condition": cond, "seed": seed, "run_id": run_id,
            "round": r["round"], "max_round": max_round,
            "contrib_sd": float(np.std(cont, ddof=1)) if len(cont) >= 2 else np.nan,
            "in_sd": float(np.std(in_vals, ddof=1)) if len(in_vals) >= 2 else np.nan,
            "dn_sd": float(np.std(dn_vals, ddof=1)) if len(dn_vals) >= 2 else np.nan,
            "n_agents": len(percs),
        })
    return rows


def _assign_period(df):
    df = df.copy()
    df["period"] = "mid"
    df.loc[df["round"] <= 3, "period"] = "early"
    df.loc[df["round"] >= df["max_round"] - 2, "period"] = "late"
    return df


def load_family_tier_data(families):
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        if families is not None and family not in families:
            continue
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant, f"seed{seed}", "log_*.json")
            cond_data = {}
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                    cond_data[d["condition"]] = d
                except Exception:
                    continue
            for cond in CONDITIONS:
                d = cond_data.get(cond)
                if d is None:
                    continue
                run_id = f"{family}_s{seed}_{cond}"
                rows += _rows_from_log(family, run_id, cond, seed, d)
    return _assign_period(pd.DataFrame(rows))


def load_structural_setting_data(sources, axis_label):
    rows = []
    for model_key, src in sources.items():
        for seed in src["seeds"]:
            found = load_latest_logs_for_seed(os.path.join(src["dir"], f"seed{seed}"))
            for cond, d in found.items():
                run_id = f"{src['family']}_{axis_label}_s{seed}_{cond}"
                rows += _rows_from_log(src["family"], run_id, cond, seed, d)
    return _assign_period(pd.DataFrame(rows))


# ═════════════════════════════════════════════════════════════════════════
# 2. Panel 1: early-vs-late OLS (per family + pooled), Holm-corrected
# ═════════════════════════════════════════════════════════════════════════

def _fit_period_ols(sub):
    early_sub = sub[sub["period"] == "early"]
    late_sub = sub[sub["period"] == "late"]
    n_early, n_late = len(early_sub), len(late_sub)
    coef, se, p_raw = np.nan, np.nan, 1.0
    if n_early >= 2 and n_late >= 2 and sub["run_id"].nunique() >= 2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.ols("value ~ C(period, Treatment('early'))", data=sub).fit(
                    cov_type="cluster", cov_kwds={"groups": sub["run_id"]})
            key = "C(period, Treatment('early'))[T.late]"
            coef = float(res.params.get(key, np.nan))
            se = float(res.bse.get(key, np.nan))
            p_raw = float(res.pvalues.get(key, 1.0))
        except Exception:
            pass
    return {"early_mean": early_sub["value"].mean() if n_early else np.nan,
            "late_mean": late_sub["value"].mean() if n_late else np.nan,
            "n_early": n_early, "n_late": n_late, "coef": coef, "se": se, "p_raw": p_raw}


def per_family_ols(df, outcome, family_order):
    results = {}
    for family in family_order:
        fdf = df[df["family"] == family].copy()
        fdf["value"] = fdf[outcome]
        p_raws = {}
        for cond in CONDITIONS:
            sub = fdf[(fdf["condition"] == cond) & fdf["period"].isin(["early", "late"])].dropna(subset=["value"])
            results[(family, cond)] = _fit_period_ols(sub)
            p_raws[cond] = results[(family, cond)]["p_raw"]
        cond_list = list(p_raws)
        _, p_holm_arr, _, _ = multipletests([p_raws[c] for c in cond_list], alpha=0.05, method="holm")
        for cond, p_h in zip(cond_list, p_holm_arr):
            results[(family, cond)]["p_holm"] = float(p_h)
    return results


def pooled_ols(df, outcome):
    d = df.copy()
    d["value"] = d[outcome]
    results, p_raws = {}, {}
    for cond in CONDITIONS:
        sub = d[(d["condition"] == cond) & d["period"].isin(["early", "late"])].dropna(subset=["value"])
        results[cond] = _fit_period_ols(sub)
        p_raws[cond] = results[cond]["p_raw"]
    cond_list = list(p_raws)
    _, p_holm_arr, _, _ = multipletests([p_raws[c] for c in cond_list], alpha=0.05, method="holm")
    for cond, p_h in zip(cond_list, p_holm_arr):
        results[cond]["p_holm"] = float(p_h)
    return results


def _stars_tex(p):
    if p < 0.001: return r"^{***}"
    if p < 0.01: return r"^{**}"
    if p < 0.05: return r"^{*}"
    if p < 0.10: return r"^{\dagger}"
    return ""


def _panel_rows(fam_results, pooled_results, family_order):
    rows = []
    for ci, cond in enumerate(CONDITIONS):
        for period in ("early", "late"):
            cells = []
            for family in family_order:
                r = fam_results.get((family, cond), {})
                mean = r.get(f"{period}_mean", np.nan)
                n = r.get(f"n_{period}", 0)
                if np.isnan(mean):
                    cells.append("---")
                else:
                    thin = r"^{\circ}" if n < 15 else ""
                    stars = _stars_tex(r.get("p_holm", 1.0)) if period == "late" else ""
                    cells.append(f"${mean:.3f}{thin}{stars}$")
            pr = pooled_results.get(cond, {})
            pmean = pr.get(f"{period}_mean", np.nan)
            pn = pr.get(f"n_{period}", 0)
            if np.isnan(pmean):
                pooled_cell = "---"
            else:
                thin = r"^{\circ}" if pn < 15 else ""
                stars = _stars_tex(pr.get("p_holm", 1.0)) if period == "late" else ""
                pooled_cell = f"${pmean:.3f}{thin}{stars}$"
            cond_str = COND_LABEL[cond] if period == "early" else ""
            period_str = "Early" if period == "early" else "Late"
            rows.append(rf"  {cond_str} & {period_str} & {' & '.join(cells)} & {pooled_cell} \\")
        if ci < len(CONDITIONS) - 1:
            rows.append(r"\midrule")
    return rows


def make_panel1_table(dn_fam, dn_pooled, in_fam, in_pooled, family_order, tier_label, out_path):
    col_spec = "ll " + "c" * len(family_order) + " c"
    header = " & ".join(family_order) + " & Pooled"
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        rf"\caption{{Perceptual consensus ({tier_label}): mean cross-agent SD of "
        r"\textit{DN} (Panel A) and \textit{IN} (Panel B) per round, by period and "
        r"condition, per family and pooled. Lower SD $=$ greater within-run "
        r"agreement. Early $=$ rounds 1--3; Late $=$ last 3 rounds per run. "
        r"Significance stars on Late rows: Holm-corrected (across the 4 "
        r"conditions, within family/pooled column) period effect from OLS "
        r"clustered by run\_id. "
        r"$^{\dagger}p{<}0.10$, $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$; "
        r"$^\circ N{<}15$.}",
        r"\label{tab:perception_consensus_early_late}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"  Condition & Period & {header} \\",
        r"\midrule",
        r"\multicolumn{" + str(3 + len(family_order)) + r"}{l}{\textit{Panel A: DN consensus}} \\[2pt]",
    ]
    lines += _panel_rows(dn_fam, dn_pooled, family_order)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(3 + len(family_order)) + r"}{l}{\textit{Panel B: IN consensus}} \\[2pt]")
    lines += _panel_rows(in_fam, in_pooled, family_order)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved -> {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# 3. Panel 2: continuous slope + final-round SD + mixed-effects robustness
# ═════════════════════════════════════════════════════════════════════════

def per_seed_slope(df, family, cond, metric):
    slopes = []
    sub = df[(df["family"] == family) & (df["condition"] == cond)]
    for seed, g in sub.groupby("seed"):
        g = g.dropna(subset=[metric])
        if len(g) < 3:
            continue
        x = g["round"].values.astype(float)
        x = x - x.mean()  # center round so the closed-form OLS slope below applies
        y = g[metric].values
        # Closed-form single-predictor OLS slope: sum(x*y) / sum(x^2), equivalent
        # to fitting "metric ~ round" per seed without pulling in statsmodels.
        slopes.append(float(np.dot(x, y) / np.dot(x, x)))
    return np.array(slopes)


def per_seed_final_sd(df, family, cond, metric):
    sub = df[(df["family"] == family) & (df["condition"] == cond) & (df["round"] >= df["max_round"] - LAST_N + 1)]
    vals = sub.groupby("seed")[metric].mean().dropna().values
    return np.array(vals)


def compute_panel2_summary(df, family_order):
    slope_rows, final_rows = [], []
    for family in family_order:
        for cond in CONDITIONS:
            for metric in METRICS:
                s = per_seed_slope(df, family, cond, metric)
                if len(s) > 0:
                    slope_rows.append({"family": family, "condition": cond, "metric": metric,
                                       "mean": float(np.mean(s)), "se": float(np.std(s, ddof=1) / np.sqrt(len(s))),
                                       "n": len(s)})
                f = per_seed_final_sd(df, family, cond, metric)
                if len(f) > 0:
                    final_rows.append({"family": family, "condition": cond, "metric": metric,
                                       "mean": float(np.mean(f)), "se": float(np.std(f, ddof=1) / np.sqrt(len(f))),
                                       "n": len(f)})
    return pd.DataFrame(slope_rows), pd.DataFrame(final_rows)


def run_mixed_effects(df, family_order):
    rows = []
    round_mean = float(df["round"].mean())
    for family in family_order:
        for cond in CONDITIONS:
            for metric in METRICS:
                sub = df[(df["family"] == family) & (df["condition"] == cond)].dropna(subset=[metric])
                records = pd.DataFrame({"seed": sub["seed"].astype(str),
                                        "round_c": sub["round"].astype(float) - round_mean,
                                        "sd": sub[metric]})
                if len(records) < 10:
                    rows.append({"family": family, "condition": cond, "metric": metric,
                                 "fe_slope": np.nan, "fe_se": np.nan, "fe_pval": np.nan,
                                 "n_obs": len(records), "n_groups": 0, "converged": False})
                    continue
                fe_slope = fe_se = fe_pval = np.nan
                converged = False
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        me = smf.mixedlm("sd ~ round_c", records, groups=records["seed"]).fit(
                            reml=True, method="lbfgs")
                    fe_slope, fe_se, fe_pval = float(me.fe_params["round_c"]), float(me.bse["round_c"]), float(me.pvalues["round_c"])
                    converged = me.converged
                except Exception:
                    pass
                rows.append({"family": family, "condition": cond, "metric": metric,
                             "fe_slope": fe_slope, "fe_se": fe_se, "fe_pval": fe_pval,
                             "n_obs": len(records), "n_groups": records["seed"].nunique(), "converged": converged})
    return pd.DataFrame(rows)


def _pstar(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def make_panel2_table(slope_df, final_df, me_df, family_order, out_dir):
    def cell(df, family, cond, metric, mean_col="mean", se_col="se", star_col=None):
        row = df[(df["family"] == family) & (df["condition"] == cond) & (df["metric"] == metric)]
        if row.empty or pd.isna(row.iloc[0][mean_col]):
            return "--"
        r = row.iloc[0]
        stars = _pstar(r[star_col]) if star_col else ""
        return f"{r[mean_col]:.3f} ({r[se_col]:.3f}){stars}"

    col_spec = "ll" + "r" * len(family_order)
    header = " & ".join(family_order)
    for name, df_, caption, label, star_col in [
        (slope_df, slope_df, "Convergence rate: OLS slope of SD $\\sim$ round (centred).", "tab:pc_slope", None),
        (final_df, final_df, f"Stability: mean SD over final {LAST_N} rounds.", "tab:pc_final_sd", None),
    ]:
        lines = [r"\begin{table}[ht]", r"\centering", r"\small",
                 rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
                 rf"Condition & Metric & {header} \\", r"\midrule"]
        for ci, cond in enumerate(CONDITIONS):
            for mi, metric in enumerate(METRICS):
                cells = [cell(df_, fam, cond, metric) for fam in family_order]
                prefix = rf"\multirow{{{len(METRICS)}}}{{*}}{{{COND_LABEL[cond]}}}" if mi == 0 else ""
                lines.append(rf"{prefix} & {METRIC_TEX[metric]} & {' & '.join(cells)} \\")
            if ci < len(CONDITIONS) - 1:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}",
                 rf"\caption{{{caption} Cells show mean (SE) across seeds.}}",
                 rf"\label{{{label}}}", r"\end{table}"]
        fname = "stability_slope.tex" if name is slope_df else "stability_final_sd.tex"
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  Saved -> {os.path.join(out_dir, fname)}")

    # OLS vs ME side-by-side
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
             rf"Condition & Metric & {header} \\", r"\midrule"]
    for ci, cond in enumerate(CONDITIONS):
        for mi, metric in enumerate(METRICS):
            cells = [cell(slope_df, fam, cond, metric) for fam in family_order]
            prefix = rf"\multirow{{{len(METRICS)*2}}}{{*}}{{{COND_LABEL[cond]}}}" if mi == 0 else ""
            lines.append(rf"{prefix} & {METRIC_TEX[metric]} \textit{{(OLS)}} & {' & '.join(cells)} \\")
        for mi, metric in enumerate(METRICS):
            cells = [cell(me_df, fam, cond, metric, mean_col="fe_slope", se_col="fe_se", star_col="fe_pval")
                     for fam in family_order]
            lines.append(rf" & {METRIC_TEX[metric]} \textit{{(ME)}} & {' & '.join(cells)} \\")
        if ci < len(CONDITIONS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Convergence rate: OLS slope (mean $\pm$ SE across seeds) and "
             r"mixed-effects fixed-effect slope (SE) for SD $\sim$ round (centred). "
             r"ME model includes a random intercept per seed. "
             r"$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$ (ME).}",
             r"\label{tab:pc_slope_robustness}", r"\end{table}"]
    path = os.path.join(out_dir, "stability_slope_robustness.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved -> {path}")


METRIC_COLORS = {"contrib_sd": "#1f77b4", "in_sd": "#e377c2", "dn_sd": "#17becf"}
METRIC_LABELS_PLOT = {"contrib_sd": "Contribution SD", "in_sd": "IN SD", "dn_sd": "DN SD"}


def plot_convergence(df, family_order, out_dir):
    """len(family_order) rows x len(CONDITIONS) cols. Each subplot: contrib/IN/DN
    SD trajectories (mean +/- SE across seeds)."""
    rounds = np.array(sorted(df["round"].unique()))
    fig, axes = plt.subplots(len(family_order), len(CONDITIONS),
                             figsize=(4 * len(CONDITIONS), 4 * len(family_order)),
                             sharex=True, sharey=True, squeeze=False)

    for row, family in enumerate(family_order):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            sub = df[(df["family"] == family) & (df["condition"] == cond)]
            for metric in METRICS:
                series = [g.set_index("round")[metric].reindex(rounds).values
                          for _, g in sub.groupby("seed")]
                if not series:
                    continue
                arr = np.array(series, dtype=float)
                n = np.sum(~np.isnan(arr), axis=0)
                mean = np.nanmean(arr, axis=0)
                se = np.nanstd(arr, axis=0) / np.sqrt(np.where(n > 0, n, np.nan))
                ax.plot(rounds, mean, color=METRIC_COLORS[metric], linewidth=2,
                        label=METRIC_LABELS_PLOT[metric])
                ax.fill_between(rounds, mean - se, mean + se, color=METRIC_COLORS[metric], alpha=0.15)
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(family, fontsize=12)
            if row == 0:
                ax.set_title(COND_LABEL[cond], fontsize=12)
            if row == len(family_order) - 1:
                ax.set_xlabel("Round", fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METRICS),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"stability_analysis.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# 4. Per-tier pipeline
# ═════════════════════════════════════════════════════════════════════════

def _run_pipeline(df, family_order, out_dir, publish_dir, tier_label):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[{tier_label}] {len(df):,} round-level obs, families={family_order}")

    plot_convergence(df, family_order, out_dir)

    # Panel 1
    dn_fam = per_family_ols(df, "dn_sd", family_order)
    dn_pooled = pooled_ols(df, "dn_sd")
    in_fam = per_family_ols(df, "in_sd", family_order)
    in_pooled = pooled_ols(df, "in_sd")
    make_panel1_table(dn_fam, dn_pooled, in_fam, in_pooled, family_order, tier_label,
                      os.path.join(out_dir, "perception_consensus_early_late.tex"))

    # Panel 2
    slope_df, final_df = compute_panel2_summary(df, family_order)
    slope_df.to_csv(os.path.join(out_dir, "stability_slopes.csv"), index=False)
    final_df.to_csv(os.path.join(out_dir, "stability_final_sd.csv"), index=False)
    me_df = run_mixed_effects(df, family_order)
    me_df.to_csv(os.path.join(out_dir, "stability_slopes_me.csv"), index=False)
    make_panel2_table(slope_df, final_df, me_df, family_order, out_dir)

    if publish_dir and os.path.abspath(publish_dir) != os.path.abspath(out_dir):
        os.makedirs(publish_dir, exist_ok=True)
        for fname in os.listdir(out_dir):
            src = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(publish_dir, fname))
        print(f"[{tier_label}] Published -> {publish_dir}")


def run_family_tier(tier):
    spec = TIER_SPECS[tier]
    families = spec["families"]
    df = load_family_tier_data(families)
    family_order = families if families is not None else [s[1] for s in MODEL_SPECS]
    family_order = [f for f in family_order if f in set(df["family"])]

    out_dir = os.path.join(EXPORT_ROOT, f"tier_{tier}")
    publish_dir = {"7b": PUBLISH_MAIN, "s2": PUBLISH_S2, "pooled": PUBLISH_POOLED,
                   "13b": PUBLISH_13B, "70b": PUBLISH_70B}[tier]
    _run_pipeline(df, family_order, out_dir, publish_dir, tier)


def run_structural_tier(axis):
    for stratum, val, sources in STRUCTURAL_SETTINGS[axis]:
        label = f"{axis}_{stratum}_{val}"
        df = load_structural_setting_data(sources, axis_label=f"{stratum}{val}")
        if df.empty:
            print(f"\n[{label}] [WARN] no data assembled, skipping.")
            continue
        family_order = sorted(df["family"].unique())
        out_dir = os.path.join(EXPORT_ROOT, f"tier_{axis}", label)
        publish_dir = os.path.join(PUBLISH_STRUCTURAL_ROOT, axis, "3_dn_in_convergence", label)
        _run_pipeline(df, family_order, out_dir, publish_dir, label)


def main():
    parser = argparse.ArgumentParser(description="Perceptual consensus: early/late OLS + convergence slope (item a).")
    parser.add_argument("--tier", choices=["7b", "s2", "13b", "70b", "pooled", "community", "group", "all"], default="7b")
    args = parser.parse_args()

    tiers = ["7b", "s2", "13b", "70b", "pooled", "community", "group"] if args.tier == "all" else [args.tier]
    for tier in tiers:
        if tier in TIER_SPECS:
            run_family_tier(tier)
        else:
            run_structural_tier(tier)


if __name__ == "__main__":
    main()
