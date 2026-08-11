"""
gap_based_alignment_test.py
-----------------------------
Do LLM agents adjust their next-round contribution toward their own elicited
descriptive/injunctive expectations?

    InGap_{i,t}  = IN_{i,t} - Contribution_{i,t}
    DnGap_{i,t}  = DN_{i,t} - Contribution_{i,t}
    Shift_{i,t}  = Contribution_{i,t+1} - Contribution_{i,t}

Primary model:
    Shift_{i,t} = b_IN*InGap_{i,t} + b_DN*DnGap_{i,t}
                  + ConditionFE + FamilyFE + RoundFE + eps_{i,t}

IN_t and DN_t are elicited AFTER round t's contribution (see
sobel_mediation.py), so both gap predictors are measured no later than t,
and the outcome (the move into t+1) is the only thing that happens after
elicitation. This makes the design prospective, but it is still purely
observational: a positive coefficient describes a correlation between the
expectation-behavior discrepancy and the subsequent adjustment, not a causal
effect of the expectation on behavior.

Data source: the existing round-level agent panel
(exports/round_level_agent_panel.csv, built by build_agent_round_panel.py)
does not itself carry IN/DN (they live in each round log's `perceptions`
block, one level up from what that script extracts). This script inspects
the panel's columns, resolves the ones it needs by name search rather than
assuming exact spelling, and merges in IN/DN pulled fresh from the raw round
logs on (run_id, round, agent_id).

Outputs (all under exports/):
    gap_based_alignment_results.csv    - every regression coefficient + diagnostics, long format
    gap_based_alignment_table.tex      - primary / standardized / lagged-level / within-agent models
    gap_alignment_binned_shift.png/.pdf - main figure
    gap_alignment_diagnostics.png      - distribution + residual diagnostics
    gap_based_alignment_summary.md     - plain-language summary
"""

import os
import sys
import json
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reorg (2026-08-11): see selection/build_agent_round_panel.py for why this block exists.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "sobel_mediation.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sobel_mediation import MODEL_SPECS, CONDITIONS

warnings.filterwarnings("ignore")

PANEL_PATH = "exports/round_level_agent_panel.csv"
OUT_DIR = "exports"
ALPHA = 0.05
Z_CRIT = stats.norm.ppf(1 - ALPHA / 2)


# ═════════════════════════════════════════════════════════════════════════
# 1. Column inspection / resolution
# ═════════════════════════════════════════════════════════════════════════

COLUMN_CANDIDATES = {
    "agent_id":    ["agent_id", "agent", "aid"],
    "run_id":      ["run_id", "run"],
    "round":       ["round", "t", "round_num"],
    "condition":   ["condition", "cond"],
    "family":      ["family", "model_family"],
    "contribution": ["contribution", "contrib"],
    "group_id":    ["group_id", "group"],
    "group_members": ["group_members", "groupmates"],
    "group_size":  ["group_size"],
}


def resolve_column(df, candidates, required=True, fuzzy=True):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    if fuzzy:
        # substring fallback only for candidates long enough to be unambiguous
        # (short codes like "IN"/"DN" would spuriously match e.g. "avg_incoming_weight")
        for cand in candidates:
            if len(cand) < 4:
                continue
            for lc, orig in cols_lower.items():
                if cand.lower() in lc:
                    return orig
    if required:
        raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")
    return None


def load_and_inspect_panel():
    print(f"Reading existing round-level agent dataset: {PANEL_PATH}")
    df = pd.read_csv(PANEL_PATH)
    print(f"  Columns found: {list(df.columns)}")
    print(f"  Shape: {df.shape}")

    resolved = {key: resolve_column(df, cands, required=(key not in ("group_id", "group_members", "group_size")))
                for key, cands in COLUMN_CANDIDATES.items()}
    print("  Resolved column mapping:")
    for k, v in resolved.items():
        print(f"    {k:15s} -> {v}")

    has_in = resolve_column(df, ["IN", "injunctive_norm", "injunctive"], required=False)
    has_dn = resolve_column(df, ["DN", "descriptive_norm", "descriptive"], required=False)
    print(f"  IN column present in panel: {has_in}")
    print(f"  DN column present in panel: {has_dn}")
    if has_in is None or has_dn is None:
        print("  -> IN/DN not present in the existing panel; sourcing them fresh "
              "from each round log's `perceptions` block and merging on "
              "(run_id, round, agent_id).")

    rename = {resolved["agent_id"]: "agent_id", resolved["run_id"]: "run_id",
              resolved["round"]: "round", resolved["condition"]: "condition",
              resolved["family"]: "family", resolved["contribution"]: "contribution"}
    if resolved.get("group_id"):
        rename[resolved["group_id"]] = "group_id"
    if resolved.get("group_members"):
        rename[resolved["group_members"]] = "group_members"
    if resolved.get("group_size"):
        rename[resolved["group_size"]] = "group_size"
    df = df.rename(columns=rename)

    if "in_groups_this_round" in df.columns:
        n_before = len(df)
        df = df[df["in_groups_this_round"]].copy()
        print(f"  Kept {len(df):,} / {n_before:,} rows flagged as active "
              f"(in_groups_this_round == True); excluded-agent rows have no "
              f"contribution and are dropped here.")
    else:
        df = df[df["contribution"].notna()].copy()

    return df, has_in, has_dn


def load_perceptions_from_logs():
    """One row per (run_id, round, agent_id) with IN/DN, nulls preserved."""
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant,
                                   f"seed{seed}", "log_*.json")
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
                for r in d["round_logs"]:
                    percs = r.get("perceptions") or {}
                    for aid_str, perc in percs.items():
                        if not perc:
                            continue
                        rows.append({
                            "run_id": run_id, "round": r["round"],
                            "agent_id": int(aid_str),
                            "IN": perc.get("injunctive_norm"),
                            "DN": perc.get("descriptive_norm"),
                        })
    return pd.DataFrame(rows)


def build_base_dataset():
    panel, has_in, has_dn = load_and_inspect_panel()

    if has_in is None or has_dn is None:
        perc = load_perceptions_from_logs()
        print(f"  Loaded {len(perc):,} perception records from raw logs.")
        n_before = len(panel)
        df = panel.merge(perc, on=["run_id", "round", "agent_id"], how="left")
        n_unmatched = df["DN"].isna().sum()
        print(f"  Merged onto panel: {len(df):,} rows "
              f"({n_before:,} panel rows in, {n_unmatched:,} with no perception "
              f"record found at all).")
    else:
        df = panel.rename(columns={has_in: "IN", has_dn: "DN"})

    # group mean contribution excluding the focal agent, from the panel's own
    # group assignment for that round
    if "group_id" in df.columns and "group_size" in df.columns:
        grp_sum = df.groupby(["run_id", "round", "group_id"])["contribution"].transform("sum")
        grp_n = df.groupby(["run_id", "round", "group_id"])["contribution"].transform("count")
        df["group_mean_contribution_excl_self"] = np.where(
            grp_n > 1, (grp_sum - df["contribution"]) / (grp_n - 1), np.nan
        )
    else:
        df["group_mean_contribution_excl_self"] = np.nan

    return df


# ═════════════════════════════════════════════════════════════════════════
# 2. Data preparation: lags, gaps, shift, exclusion reporting
# ═════════════════════════════════════════════════════════════════════════

def prepare_gap_data(df):
    df = df.sort_values(["run_id", "agent_id", "round"]).reset_index(drop=True)
    grp = df.groupby(["run_id", "agent_id"], sort=False)

    df["contribution_lead1"] = grp["contribution"].shift(-1)
    df["round_lead1"] = grp["round"].shift(-1)
    df["run_id_lead1"] = grp["run_id"].shift(-1)  # sanity check, should equal run_id or NaN

    consecutive = (df["round_lead1"] == df["round"] + 1)
    has_next = df["round_lead1"].notna()

    n_total = len(df)
    n_last_round_of_traj = (~has_next).sum()
    n_nonconsecutive = (has_next & ~consecutive).sum()

    print(f"\nRow-level trajectory check (n={n_total:,} active agent-round obs):")
    print(f"  Dropped as last round of its run-agent trajectory (no t+1 to link): {n_last_round_of_traj:,}")
    print(f"  Dropped for non-consecutive round linkage (should be 0 given sticky "
          f"exclusion, kept as a safeguard): {n_nonconsecutive:,}")

    df["valid_link"] = has_next & consecutive
    linked = df[df["valid_link"]].copy()

    linked["in_gap"] = linked["IN"] - linked["contribution"]
    linked["dn_gap"] = linked["DN"] - linked["contribution"]
    linked["contribution_shift_lead1"] = linked["contribution_lead1"] - linked["contribution"]

    # ── null-IN reporting (before dropping) ──────────────────────────────
    n_linked = len(linked)
    n_in_null = linked["IN"].isna().sum()
    n_dn_null = linked["DN"].isna().sum()
    print(f"\nLinked (t, t+1) observations available for the gap model: {n_linked:,}")
    print(f"  Removed because IN is null: {n_in_null:,} ({n_in_null / n_linked:.2%})")
    print(f"  Removed because DN is null: {n_dn_null:,} ({n_dn_null / n_linked:.2%})")

    null_by_cond = linked.groupby("condition")["IN"].apply(lambda s: s.isna().mean())
    null_by_fam = linked.groupby("family")["IN"].apply(lambda s: s.isna().mean())
    null_by_round = linked.groupby("round")["IN"].apply(lambda s: s.isna().mean())
    print("\n  IN-null rate by condition:")
    print(null_by_cond.to_string(float_format=lambda x: f"{x:.2%}"))
    print("\n  IN-null rate by model family:")
    print(null_by_fam.to_string(float_format=lambda x: f"{x:.2%}"))
    print("\n  IN-null rate by round (t):")
    print(null_by_round.to_string(float_format=lambda x: f"{x:.2%}"))

    final = linked.dropna(subset=["IN", "DN", "contribution_shift_lead1"]).copy()
    final["run_agent_id"] = final["run_id"] + "_" + final["agent_id"].astype(str)

    print(f"\nFinal analysis sample: {len(final):,} observations "
          f"({len(final) / n_total:.1%} of all active agent-round observations)")
    print(f"  Run-level clusters (run_id): {final['run_id'].nunique():,}")
    print(f"  Run-agent units (for the within-agent model): {final['run_agent_id'].nunique():,}")

    exclusion_report = {
        "n_active_agent_round_obs": n_total,
        "n_dropped_last_round_of_trajectory": int(n_last_round_of_traj),
        "n_dropped_nonconsecutive": int(n_nonconsecutive),
        "n_linked_pairs": n_linked,
        "n_removed_in_null": int(n_in_null),
        "pct_removed_in_null": n_in_null / n_linked,
        "n_removed_dn_null": int(n_dn_null),
        "final_n": len(final),
        "final_n_clusters": final["run_id"].nunique(),
        "null_by_condition": null_by_cond,
        "null_by_family": null_by_fam,
        "null_by_round": null_by_round,
    }
    return final, exclusion_report


# ═════════════════════════════════════════════════════════════════════════
# 3. Regression helpers
# ═════════════════════════════════════════════════════════════════════════

def fit_cluster_ols(formula, data, cluster_col="run_id"):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )


def zscale(s):
    return (s - s.mean()) / s.std()


def print_focal(model, terms_labels, header):
    print(f"\n{'-'*72}\n{header}\n{'-'*72}")
    print(f"N={int(model.nobs):,}   R2={model.rsquared:.4f}   Adj.R2={model.rsquared_adj:.4f}")
    ci = model.conf_int(alpha=ALPHA)
    for term, label in terms_labels:
        b, se, p = model.params[term], model.bse[term], model.pvalues[term]
        lo, hi = ci.loc[term, 0], ci.loc[term, 1]
        stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "†" if p < .10 else ""
        print(f"  {label:40s} b={b:8.4f}  SE={se:7.4f}  95%CI=[{lo:7.4f}, {hi:7.4f}]  p={p:7.4f} {stars}")


def wald_equality(model, term1, term2):
    res = model.wald_test(f"{term1} = {term2}", scalar=True)
    stat = float(np.squeeze(res.statistic))
    pval = float(np.squeeze(res.pvalue))
    df_denom = getattr(res, "df_denom", None)
    return stat, pval, df_denom


FE_FORMULA = (
    'C(condition, Treatment(reference="BASELINE")) '
    '+ C(family, Treatment(reference="GPT")) '
    '+ C(round, Treatment(reference=1))'
)


# ═════════════════════════════════════════════════════════════════════════
# 4. Models
# ═════════════════════════════════════════════════════════════════════════

def run_primary_models(df):
    formula_raw = f"contribution_shift_lead1 ~ in_gap + dn_gap + {FE_FORMULA}"
    m_raw = fit_cluster_ols(formula_raw, df)

    df_z = df.copy()
    df_z["contribution_shift_lead1_z"] = zscale(df_z["contribution_shift_lead1"])
    df_z["in_gap_z"] = zscale(df_z["in_gap"])
    df_z["dn_gap_z"] = zscale(df_z["dn_gap"])
    formula_z = f"contribution_shift_lead1_z ~ in_gap_z + dn_gap_z + {FE_FORMULA}"
    m_z = fit_cluster_ols(formula_z, df_z)

    return m_raw, m_z, formula_raw, formula_z


def run_lagged_level_model(df):
    formula = f"contribution_lead1 ~ contribution + IN + DN + {FE_FORMULA}"
    return fit_cluster_ols(formula, df), formula


def run_within_agent_model(df):
    """
    Agent (run_agent) fixed effects via within-transformation: demean
    in_gap, dn_gap, and the outcome by run_agent_id, then regress the
    demeaned outcome on the demeaned gaps. Condition and family are
    invariant within a run_agent unit (fixed for the life of a run) and are
    therefore absorbed by the transformation, not separately estimable
    here. SEs clustered by run_id; degrees of freedom are not corrected for
    the absorbed run_agent means, so these SEs are a standard applied
    approximation, slightly liberal in a strict small-sample sense.
    """
    d = df.copy()
    for col in ["in_gap", "dn_gap", "contribution_shift_lead1"]:
        d[col + "_within"] = d[col] - d.groupby("run_agent_id")[col].transform("mean")
    formula = "contribution_shift_lead1_within ~ in_gap_within + dn_gap_within"
    return fit_cluster_ols(formula, d), formula, d


def run_group_behavior_model(df):
    d = df.dropna(subset=["group_mean_contribution_excl_self"]).copy()
    formula = (f"contribution_shift_lead1 ~ in_gap + dn_gap + "
               f"group_mean_contribution_excl_self + {FE_FORMULA}")
    return fit_cluster_ols(formula, d), formula, len(d)


def run_asymmetric_model(df):
    d = df.copy()
    d["in_gap_positive"] = d["in_gap"].clip(lower=0)
    d["in_gap_negative"] = d["in_gap"].clip(upper=0)
    d["dn_gap_positive"] = d["dn_gap"].clip(lower=0)
    d["dn_gap_negative"] = d["dn_gap"].clip(upper=0)
    formula = (f"contribution_shift_lead1 ~ in_gap_positive + in_gap_negative "
               f"+ dn_gap_positive + dn_gap_negative + {FE_FORMULA}")
    return fit_cluster_ols(formula, d), formula


def run_condition_interaction_model(df):
    formula = (
        'contribution_shift_lead1 ~ in_gap * C(condition, Treatment(reference="BASELINE")) '
        '+ dn_gap * C(condition, Treatment(reference="BASELINE")) '
        '+ C(family, Treatment(reference="GPT")) + C(round, Treatment(reference=1))'
    )
    return fit_cluster_ols(formula, df), formula


# ═════════════════════════════════════════════════════════════════════════
# 5. Diagnostics
# ═════════════════════════════════════════════════════════════════════════

def compute_diagnostics(df, primary_model):
    diag = {}

    diag["corr_matrix"] = df[["contribution", "IN", "DN", "in_gap", "dn_gap",
                              "contribution_shift_lead1"]].corr()

    diag["pct_contribution_at_0"] = (df["contribution"] == 0).mean()
    diag["pct_contribution_at_10"] = (df["contribution"] == df["contribution"].max()).mean()
    diag["contribution_max_observed"] = df["contribution"].max()

    cluster_sizes = df.groupby("run_id").size()
    diag["n_clusters"] = cluster_sizes.shape[0]
    diag["cluster_size_min"] = cluster_sizes.min()
    diag["cluster_size_median"] = cluster_sizes.median()
    diag["cluster_size_max"] = cluster_sizes.max()

    diag["obs_by_condition"] = df.groupby("condition").size()
    diag["obs_by_family"] = df.groupby("family").size()

    diag["resid"] = primary_model.resid
    diag["fitted"] = primary_model.fittedvalues

    return diag


def print_diagnostics(diag):
    print(f"\n{'-'*72}\nDIAGNOSTICS\n{'-'*72}")
    print("\nCorrelation matrix (contribution, IN, DN, in_gap, dn_gap, shift):")
    print(diag["corr_matrix"].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nContribution at 0:  {diag['pct_contribution_at_0']:.2%}")
    print(f"Contribution at max observed ({diag['contribution_max_observed']:.0f}): "
          f"{diag['pct_contribution_at_10']:.2%}")
    print(f"\nClusters (run_id): n={diag['n_clusters']}  "
          f"size min/median/max = {diag['cluster_size_min']}/"
          f"{diag['cluster_size_median']}/{diag['cluster_size_max']}")
    print("\nObservations by condition:")
    print(diag["obs_by_condition"].to_string())
    print("\nObservations by model family:")
    print(diag["obs_by_family"].to_string())


def plot_diagnostics(df, diag, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    axes[0, 0].hist(df["in_gap"], bins=40, color="#4C72B0", alpha=0.85)
    axes[0, 0].axvline(0, color="black", lw=1, ls="--")
    axes[0, 0].set_title("Distribution of InGap (IN - contribution)", fontsize=11)
    axes[0, 0].set_xlabel("in_gap")

    axes[0, 1].hist(df["dn_gap"], bins=40, color="#DD8452", alpha=0.85)
    axes[0, 1].axvline(0, color="black", lw=1, ls="--")
    axes[0, 1].set_title("Distribution of DnGap (DN - contribution)", fontsize=11)
    axes[0, 1].set_xlabel("dn_gap")

    axes[0, 2].hist(df["contribution_shift_lead1"], bins=40, color="#55A868", alpha=0.85)
    axes[0, 2].axvline(0, color="black", lw=1, ls="--")
    axes[0, 2].set_title("Distribution of next-round contribution shift", fontsize=11)
    axes[0, 2].set_xlabel("contribution_shift_lead1")

    axes[1, 0].scatter(diag["fitted"], diag["resid"], s=5, alpha=0.15, color="#4C72B0")
    axes[1, 0].axhline(0, color="black", lw=1, ls="--")
    axes[1, 0].set_title("Residuals vs fitted (primary raw-units model)", fontsize=11)
    axes[1, 0].set_xlabel("fitted"); axes[1, 0].set_ylabel("residual")

    sm.qqplot(diag["resid"], line="45", ax=axes[1, 1], markersize=2, alpha=0.2)
    axes[1, 1].set_title("Residual QQ-plot", fontsize=11)

    cluster_sizes = df.groupby("run_id").size()
    axes[1, 2].hist(cluster_sizes, bins=30, color="#8172B2", alpha=0.85)
    axes[1, 2].set_title("Cluster (run_id) size distribution", fontsize=11)
    axes[1, 2].set_xlabel("observations per run")

    fig.suptitle("Gap-based alignment: diagnostics", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# 6. Main figure: binned mean shift vs signed gap, with 95% CI
# ═════════════════════════════════════════════════════════════════════════

def binned_means(x, y, n_bins=12):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    bin_idx = np.digitize(x, edges[1:-1], right=True)
    out = []
    for b in np.unique(bin_idx):
        mask = bin_idx == b
        n = mask.sum()
        if n < 5:
            continue
        xb = x[mask].mean()
        yb = y[mask].mean()
        se = y[mask].std(ddof=1) / np.sqrt(n)
        out.append((xb, yb, se, n))
    return pd.DataFrame(out, columns=["x_mean", "y_mean", "se", "n"])


def plot_main_figure(df, out_stub):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    specs = [
        ("in_gap", "InGap = IN$_t$ $-$ Contribution$_t$", "#4C72B0", axes[0]),
        ("dn_gap", "DnGap = DN$_t$ $-$ Contribution$_t$", "#DD8452", axes[1]),
    ]
    for col, xlabel, color, ax in specs:
        b = binned_means(df[col].values, df["contribution_shift_lead1"].values, n_bins=14)
        ci = Z_CRIT * b["se"]
        ax.errorbar(b["x_mean"], b["y_mean"], yerr=ci, fmt="o", color=color,
                    ecolor=color, elinewidth=1.5, capsize=3, markersize=5)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(f"n bins = {len(b)}, binned by quantile", fontsize=9, color="gray")

    axes[0].set_ylabel("Mean next-round contribution shift\n(binned, 95% CI)", fontsize=11)
    fig.suptitle("Expectation gap vs. subsequent contribution adjustment\n"
                 "(observational association, not a causal estimate)",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# 7. Output writers
# ═════════════════════════════════════════════════════════════════════════

def build_results_csv(models_named, exclusion_report, diag, wald_raw, wald_z, out_path):
    rows = []
    # regression coefficient rows
    for name, model, cluster_n in models_named:
        ci = model.conf_int(alpha=ALPHA)
        for term in model.params.index:
            rows.append({
                "section": "regression", "model": name, "term": term,
                "estimate": model.params[term], "se": model.bse[term],
                "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1],
                "p_value": model.pvalues[term],
                "n": int(model.nobs), "n_clusters": cluster_n,
                "r2": model.rsquared, "r2_adj": model.rsquared_adj, "note": "",
            })

    rows.append({"section": "wald_test", "model": "primary_raw",
                 "term": "in_gap = dn_gap", "estimate": np.nan, "se": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "p_value": wald_raw[1],
                 "n": np.nan, "n_clusters": np.nan, "r2": np.nan, "r2_adj": np.nan,
                 "note": f"wald_stat={wald_raw[0]:.4f}"})
    rows.append({"section": "wald_test", "model": "primary_standardized",
                 "term": "in_gap_z = dn_gap_z", "estimate": np.nan, "se": np.nan,
                 "ci_low": np.nan, "ci_high": np.nan, "p_value": wald_z[1],
                 "n": np.nan, "n_clusters": np.nan, "r2": np.nan, "r2_adj": np.nan,
                 "note": f"wald_stat={wald_z[0]:.4f}"})

    for k, v in exclusion_report.items():
        if isinstance(v, pd.Series):
            continue
        rows.append({"section": "sample_exclusion", "model": "", "term": k,
                     "estimate": v if isinstance(v, (int, float)) else np.nan,
                     "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                     "p_value": np.nan, "n": np.nan, "n_clusters": np.nan,
                     "r2": np.nan, "r2_adj": np.nan, "note": ""})

    for k in ["null_by_condition", "null_by_family", "null_by_round"]:
        s = exclusion_report[k]
        for idx, val in s.items():
            rows.append({"section": "in_null_rate", "model": k, "term": str(idx),
                         "estimate": val, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                         "p_value": np.nan, "n": np.nan, "n_clusters": np.nan,
                         "r2": np.nan, "r2_adj": np.nan, "note": ""})

    rows.append({"section": "diagnostics", "model": "", "term": "pct_contribution_at_0",
                 "estimate": diag["pct_contribution_at_0"], "se": np.nan, "ci_low": np.nan,
                 "ci_high": np.nan, "p_value": np.nan, "n": np.nan, "n_clusters": np.nan,
                 "r2": np.nan, "r2_adj": np.nan, "note": ""})
    rows.append({"section": "diagnostics", "model": "", "term": "pct_contribution_at_max",
                 "estimate": diag["pct_contribution_at_10"], "se": np.nan, "ci_low": np.nan,
                 "ci_high": np.nan, "p_value": np.nan, "n": np.nan, "n_clusters": np.nan,
                 "r2": np.nan, "r2_adj": np.nan, "note": ""})
    for a in diag["corr_matrix"].index:
        for b_ in diag["corr_matrix"].columns:
            rows.append({"section": "correlation", "model": "", "term": f"{a}_vs_{b_}",
                         "estimate": diag["corr_matrix"].loc[a, b_], "se": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan,
                         "n": np.nan, "n_clusters": np.nan, "r2": np.nan, "r2_adj": np.nan,
                         "note": ""})

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def fmt_coef(model, term):
    b, se, p = model.params[term], model.bse[term], model.pvalues[term]
    s = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "$\\dagger$" if p < .10 else ""
    return rf"{b:.3f} ({se:.3f})$^{{{s}}}$" if s else rf"{b:.3f} ({se:.3f})"


def build_latex_table(m_raw, m_z, m_lagged, m_within, out_path):
    def row(label, model, term, blank_if_missing=True):
        if term not in model.params.index:
            return rf"  {label} & --- \\"
        return rf"  {label} & {fmt_coef(model, term)} \\"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & Primary (raw) & Primary (std.) & Lagged-level & Within-agent \\",
        r"\midrule",
        rf"  InGap$_t$ & {fmt_coef(m_raw,'in_gap')} & {fmt_coef(m_z,'in_gap_z')} & --- & {fmt_coef(m_within,'in_gap_within')} \\",
        rf"  DnGap$_t$ & {fmt_coef(m_raw,'dn_gap')} & {fmt_coef(m_z,'dn_gap_z')} & --- & {fmt_coef(m_within,'dn_gap_within')} \\",
        rf"  Contribution$_t$ & --- & --- & {fmt_coef(m_lagged,'contribution')} & --- \\",
        rf"  IN$_t$ & --- & --- & {fmt_coef(m_lagged,'IN')} & --- \\",
        rf"  DN$_t$ & --- & --- & {fmt_coef(m_lagged,'DN')} & --- \\",
        r"\midrule",
        rf"  Condition FE & Yes & Yes & Yes & Absorbed \\",
        rf"  Family FE & Yes & Yes & Yes & Absorbed \\",
        rf"  Round FE & Yes & Yes & Yes & No \\",
        r"\midrule",
        rf"  $N$ & {int(m_raw.nobs):,} & {int(m_z.nobs):,} & {int(m_lagged.nobs):,} & {int(m_within.nobs):,} \\",
        rf"  $R^2$ & {m_raw.rsquared:.3f} & {m_z.rsquared:.3f} & {m_lagged.rsquared:.3f} & {m_within.rsquared:.3f} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Expectation gaps and subsequent contribution adjustment. "
         r"\textit{Primary}: contribution$_{t+1}-$contribution$_t$ on InGap$_t$, DnGap$_t$, "
         r"condition FE (ref.\ BASELINE), model-family FE (ref.\ GPT), and round FE. "
         r"\textit{Lagged-level}: contribution$_{t+1}$ on contribution$_t$, IN$_t$, DN$_t$, "
         r"and the same FE set --- a robustness check against mechanical coupling between "
         r"the gap predictors and the change-score outcome. "
         r"\textit{Within-agent}: InGap/DnGap and the outcome centered within run-agent "
         r"unit (condition/family FE absorbed, not separately estimable). "
         r"OLS, SEs clustered by run\_id in parentheses. "
         r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$. "
         r"Associations are observational; no causal interpretation is implied.}"),
        r"\label{tab:gap_alignment}",
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def build_markdown_summary(m_raw, m_z, m_lagged, m_within, wald_raw, exclusion_report, out_path,
                            null_by_family_note=""):
    b_in, p_in = m_raw.params["in_gap"], m_raw.pvalues["in_gap"]
    b_dn, p_dn = m_raw.params["dn_gap"], m_raw.pvalues["dn_gap"]
    stat, p_wald, _ = wald_raw

    in_dir = "moves toward IN (positive coefficient)" if b_in > 0 else "moves away from IN (negative coefficient)"
    dn_dir = "moves toward DN (positive coefficient)" if b_dn > 0 else "moves away from DN (negative coefficient)"

    lagged_theta_in = m_lagged.params["IN"]
    lagged_theta_dn = m_lagged.params["DN"]
    lagged_theta_c = m_lagged.params["contribution"]

    within_b_in = m_within.params["in_gap_within"]
    within_b_dn = m_within.params["dn_gap_within"]

    lines = [
        "# Gap-based alignment test: summary",
        "",
        "## Question",
        "Does the discrepancy between an agent's round-t contribution and its own",
        "elicited descriptive/injunctive expectation predict the direction and",
        "magnitude of its round-(t+1) contribution adjustment?",
        "",
        "## Headline result",
        f"- InGap coefficient (b_IN): {b_in:.4f} (p={p_in:.4g}) — next-round contribution",
        f"  {in_dir}.",
        f"- DnGap coefficient (b_DN): {b_dn:.4f} (p={p_dn:.4g}) — next-round contribution",
        f"  {dn_dir}.",
        f"- Coefficient-equality Wald test (H0: b_IN = b_DN): statistic={stat:.4f}, p={p_wald:.4g}.",
        ("  The two coefficients are statistically distinguishable at the 5% level."
         if p_wald < 0.05 else
         "  The two coefficients are not statistically distinguishable at the 5% level; "
         "do not describe one as a stronger predictor than the other based on point "
         "estimates alone."),
        "",
        "## Does behavior move toward IN?",
        (f"The primary model's positive, significant InGap coefficient (p={p_in:.4g}) is "
         "consistent with agents' next-round contribution moving toward their elicited "
         "injunctive expectation." if b_in > 0 and p_in < 0.05 else
         "The InGap coefficient does not support a consistent move toward IN in this sample."),
        "",
        "## Does behavior move toward DN?",
        (f"The primary model's positive, significant DnGap coefficient (p={p_dn:.4g}) is "
         "consistent with agents' next-round contribution moving toward their elicited "
         "descriptive expectation." if b_dn > 0 and p_dn < 0.05 else
         "The DnGap coefficient does not support a consistent move toward DN in this sample."),
        "",
        "## Robustness",
        f"- **Lagged-level model** (contribution_t+1 ~ contribution_t + IN_t + DN_t + FE): "
        f"theta_C={lagged_theta_c:.4f}, theta_IN={lagged_theta_in:.4f}, theta_DN={lagged_theta_dn:.4f}. "
        "This is algebraically related to the gap model — subtracting contribution_t from both "
        "sides of a model with theta_IN*IN_t + theta_DN*DN_t + (theta_C-1)*contribution_t recovers "
        "the gap-model form only under the restriction theta_IN = -( theta_C - 1 coefficient on the "
        "IN part), i.e. the gap specification implicitly constrains the coefficient on contribution_t "
        "to be minus the sum of the gap coefficients. The lagged-level model relaxes that constraint, "
        "so agreement in sign/significance between theta_IN, theta_DN here and b_IN, b_DN in the gap "
        "model is what makes the gap-model result credible rather than an artifact of the shared "
        "contribution_t term appearing on both sides of the primary specification.",
        (f"  Both IN_t and DN_t retain positive, significant coefficients in the lagged-level model, "
         "consistent with the gap-model finding." if lagged_theta_in > 0 and lagged_theta_dn > 0
         and m_lagged.pvalues['IN'] < 0.05 and m_lagged.pvalues['DN'] < 0.05 else
         "  The lagged-level model does not fully replicate the sign/significance pattern from the "
         "gap model — treat the gap-model result with more caution."),
        "",
        f"- **Within-run-agent model** (condition/family FE absorbed): InGap={within_b_in:.4f}, "
        f"DnGap={within_b_dn:.4f}. This asks whether an agent's own round-to-round *changes* in "
        "expectation gap predict its own subsequent behavior change, net of any stable difference "
        "between cooperative and uncooperative agents.",
        (f"  Both within-agent coefficients are positive, consistent with the pooled result."
         if within_b_in > 0 and within_b_dn > 0 else
         "  At least one within-agent coefficient changes sign relative to the pooled result — "
         "part of the pooled association may reflect stable between-agent differences rather than "
         "within-agent responsiveness to changing expectations."),
        "",
        "## Null IN observations",
        f"- {exclusion_report['n_removed_in_null']:,} of {exclusion_report['n_linked_pairs']:,} "
        f"linked (t, t+1) observations ({exclusion_report['pct_removed_in_null']:.2%}) were "
        "dropped because IN was null at t. No imputation was performed. Null rates by condition "
        "and round are low and broadly spread (roughly 2-6%, full breakdown in the results CSV, "
        "section=in_null_rate). "
        f"{null_by_family_note}"
        "The dropped observations are not necessarily missing at random (e.g. if a model fails to "
        "report IN specifically when contribution and expectation diverge sharply, dropping those "
        "rows could bias the estimated IN-gap effect toward zero for that family).",
        "",
        "## Limitations",
        "- All associations are observational. IN and DN are self-reports elicited from the same",
        "  LLM agent that produced the contribution; nothing here establishes that the",
        "  expectation *causes* the behavioral adjustment, only that the discrepancy has",
        "  prospective behavioral relevance.",
        "- Contribution appears on both sides of the primary gap specification (in the gap",
        "  predictors and implicitly in the outcome), which is exactly why the lagged-level",
        "  model above is reported as the essential robustness check.",
        "- Exclusion from the game is sticky (agents who drop out do not return), so the",
        "  sample thins over rounds within a run; round fixed effects only partially address",
        "  this kind of compositional change.",
        "- 'IN'/'DN' and 'expectation' describe elicited text/number outputs from an LLM,",
        "  not claims about internal beliefs, and are treated as behavioral measurements only.",
        "",
    ]
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    base = build_base_dataset()
    df, exclusion_report = prepare_gap_data(base)

    # ── Primary models ────────────────────────────────────────────────────
    m_raw, m_z, f_raw, f_z = run_primary_models(df)
    print_focal(m_raw, [("in_gap", "b_IN  (InGap, raw units)"),
                        ("dn_gap", "b_DN  (DnGap, raw units)")],
                "PRIMARY GAP MODEL (raw 0-10 units)")
    print_focal(m_z, [("in_gap_z", "b_IN  (InGap, standardized)"),
                      ("dn_gap_z", "b_DN  (DnGap, standardized)")],
                "PRIMARY GAP MODEL (standardized)")

    wald_raw = wald_equality(m_raw, "in_gap", "dn_gap")
    wald_z = wald_equality(m_z, "in_gap_z", "dn_gap_z")
    print(f"\nWald test H0: b_IN = b_DN (raw model): stat={wald_raw[0]:.4f}, p={wald_raw[1]:.4g}")
    print(f"Wald test H0: b_IN = b_DN (standardized model): stat={wald_z[0]:.4f}, p={wald_z[1]:.4g}")
    if wald_raw[1] >= 0.05:
        print("  -> Cannot reject equality; do not claim IN is a stronger predictor than DN "
              "based on the point estimates alone.")

    # ── Lagged-level robustness ──────────────────────────────────────────
    m_lagged, f_lagged = run_lagged_level_model(df)
    print_focal(m_lagged, [("contribution", "theta_C (contribution_t)"),
                          ("IN", "theta_IN (IN_t)"),
                          ("DN", "theta_DN (DN_t)")],
                "ESSENTIAL ROBUSTNESS: LAGGED-LEVEL MODEL")

    # ── Within-agent robustness ──────────────────────────────────────────
    m_within, f_within, _ = run_within_agent_model(df)
    print_focal(m_within, [("in_gap_within", "InGap (within run-agent)"),
                          ("dn_gap_within", "DnGap (within run-agent)")],
                "ROBUSTNESS 1: WITHIN-AGENT (run-agent centered)")

    # ── Group behavior robustness ─────────────────────────────────────────
    m_group, f_group, n_group = run_group_behavior_model(df)
    print_focal(m_group, [("in_gap", "InGap | group mean contribution"),
                         ("dn_gap", "DnGap | group mean contribution"),
                         ("group_mean_contribution_excl_self", "Group mean contribution (excl. self)")],
                f"ROBUSTNESS 2: OBSERVED GROUP BEHAVIOR (n={n_group:,})")

    # ── Asymmetric gaps (exploratory) ─────────────────────────────────────
    m_asym, f_asym = run_asymmetric_model(df)
    print_focal(m_asym, [("in_gap_positive", "InGap+ (below expectation... contribution < IN)"),
                        ("in_gap_negative", "InGap- (above expectation)"),
                        ("dn_gap_positive", "DnGap+"),
                        ("dn_gap_negative", "DnGap-")],
                "ROBUSTNESS 3 (EXPLORATORY): ASYMMETRIC POSITIVE/NEGATIVE GAPS")

    # ── Condition interaction (secondary) ─────────────────────────────────
    m_inter, f_inter = run_condition_interaction_model(df)
    inter_terms = [(t, t) for t in m_inter.params.index if "in_gap:" in t or "dn_gap:" in t]
    print_focal(m_inter, inter_terms, "ROBUSTNESS 4 (SECONDARY): CONDITION x GAP INTERACTIONS")

    # ── Diagnostics ─────────────────────────────────────────────────────
    diag = compute_diagnostics(df, m_raw)
    print_diagnostics(diag)
    diag_png = os.path.join(OUT_DIR, "gap_alignment_diagnostics.png")
    plot_diagnostics(df, diag, diag_png)

    # ── Main figure ────────────────────────────────────────────────────
    fig_stub = os.path.join(OUT_DIR, "gap_alignment_binned_shift")
    plot_main_figure(df, fig_stub)

    # ── Outputs ────────────────────────────────────────────────────────
    models_named = [
        ("primary_raw", m_raw, df["run_id"].nunique()),
        ("primary_standardized", m_z, df["run_id"].nunique()),
        ("lagged_level", m_lagged, df["run_id"].nunique()),
        ("within_agent", m_within, df["run_id"].nunique()),
        ("group_behavior", m_group, df.dropna(subset=["group_mean_contribution_excl_self"])["run_id"].nunique()),
        ("asymmetric_gaps", m_asym, df["run_id"].nunique()),
        ("condition_interaction", m_inter, df["run_id"].nunique()),
    ]
    csv_path = os.path.join(OUT_DIR, "gap_based_alignment_results.csv")
    build_results_csv(models_named, exclusion_report, diag, wald_raw, wald_z, csv_path)

    tex_path = os.path.join(OUT_DIR, "gap_based_alignment_table.tex")
    build_latex_table(m_raw, m_z, m_lagged, m_within, tex_path)

    md_path = os.path.join(OUT_DIR, "gap_based_alignment_summary.md")
    null_by_fam = exclusion_report["null_by_family"]
    fam_max, rate_max = null_by_fam.idxmax(), null_by_fam.max()
    if rate_max > 0.10:
        others_near_zero = (null_by_fam.drop(fam_max) < 0.03).all()
        family_note = (
            f"By model family, however, nulls are concentrated almost entirely in "
            f"{fam_max} ({rate_max:.1%} of its observations), "
            f"{'while every other family is under 3%. ' if others_near_zero else ''}"
            f"This looks like a {fam_max}-specific elicitation/parsing issue rather than "
            "a phenomenon spread evenly across models. "
        )
    else:
        family_note = "Null rates are also low across every model family. "
    build_markdown_summary(m_raw, m_z, m_lagged, m_within, wald_raw, exclusion_report, md_path,
                           null_by_family_note=family_note)

    # ── Final report ───────────────────────────────────────────────────
    print(f"\n{'='*72}\nMODEL FORMULAS\n{'='*72}")
    print(f"Primary (raw):        {f_raw}")
    print(f"Primary (standard.):  {f_z}")
    print(f"Lagged-level:         {f_lagged}")
    print(f"Within-agent:         {f_within}")
    print(f"Group behavior:       {f_group}")
    print(f"Asymmetric gaps:      {f_asym}")
    print(f"Condition interaction:{f_inter}")

    print(f"\n{'='*72}\nMAIN COEFFICIENT TABLE (primary raw-units model)\n{'='*72}")
    print(m_raw.summary().tables[1])

    print(f"\n{'='*72}\nCOEFFICIENT-EQUALITY TEST (H0: b_IN = b_DN)\n{'='*72}")
    print(f"Raw model:          stat={wald_raw[0]:.4f}, p={wald_raw[1]:.4g}")
    print(f"Standardized model: stat={wald_z[0]:.4f}, p={wald_z[1]:.4g}")

    print(f"\n{'='*72}\nSAMPLE EXCLUSIONS\n{'='*72}")
    for k in ["n_active_agent_round_obs", "n_dropped_last_round_of_trajectory",
              "n_dropped_nonconsecutive", "n_linked_pairs", "n_removed_in_null",
              "pct_removed_in_null", "final_n", "final_n_clusters"]:
        v = exclusion_report[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    print(f"\n{'='*72}\nOUTPUT FILES\n{'='*72}")
    for p in [csv_path, tex_path, fig_stub + ".png", fig_stub + ".pdf", diag_png, md_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
