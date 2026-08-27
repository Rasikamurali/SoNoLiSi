"""
gap_based_alignment.py
-----------------------------
OLS test of whether the gap between an agent's own elicited
injunctive/descriptive expectation and its contribution predicts the shift
into next-round contribution.

    InGap_{i,t}  = IN_{i,t} - Contribution_{i,t}
    DnGap_{i,t}  = DN_{i,t} - Contribution_{i,t}
    Shift_{i,t}  = Contribution_{i,t+1} - Contribution_{i,t}

Primary model:
    Shift_{i,t} = b_IN*InGap_{i,t} + b_DN*DnGap_{i,t}
                  + ConditionFE + FamilyFE + RoundFE + eps_{i,t}

IN_t and DN_t are elicited AFTER round t's contribution, so both gap
predictors are measured no later than t, and the outcome (the move into
t+1) is the only thing that happens after elicitation. This makes the
design prospective, but it is still purely observational: a positive
coefficient describes a correlation between the expectation-behavior
discrepancy and the subsequent adjustment, not a causal effect.

Tiers (--tier flag, default 7b):
    7b        - gpt, llama-7b, mistral-7b, qwen-7b. PRIMARY/headline result.
                -> figures/MAIN_RESULTS/4_gap_based_alignment/
    s2        - gpt-5-mini, llama-70b, mistral-13b, qwen-72b (supplementary
                replication).
                -> figures/SUPPLEMENTARY_RESULTS/2_bigger_models/4_gap_based_alignment/
    pooled    - all families pooled (the OLD primary; kept as a cross-check).
                -> figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/subset_all_families/
    community - population size N in {12, 16, 20}, group size G=4 fixed,
                8 model variants (7B/13B-14B/70B-72B; no GPT -- GPT has no
                community-size sweep data). One fit per N.
                -> figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/community/4_gap_based_alignment/
    group     - interaction group size G, 7B trio only (llama/mistral/qwen),
                two matched strata (N=12: G in {3,4,6}; N=16: G in {4,8}).
                One fit per (stratum, G).
                -> figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/group/4_gap_based_alignment/

Family-tier data source: the existing round-level agent panel
(exports/round_level_agent_panel.csv, built by build_agent_round_panel.py)
merged with IN/DN pulled fresh from the raw round logs.

Community/group data source: no pre-built panel covers these runs, so a
self-contained raw-log loader (build_base_dataset_from_sources, below) reads
directly from each run's round_logs -- same InGap/DnGap/Shift definitions,
same formula, just assembled without the panel dependency. Source directory
layout (COMMUNITY_SOURCES / GROUP_N12_SOURCES / GROUP_N16_SOURCES) matches
analyze_structural_robustness.py's own source definitions.
"""

import os
import sys
import argparse
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

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_specs import MODEL_SPECS, CONDITIONS

warnings.filterwarnings("ignore")

# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/alignment/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ALPHA = 0.05
Z_CRIT = stats.norm.ppf(1 - ALPHA / 2)

# ═════════════════════════════════════════════════════════════════════════
# 0. Tier definitions
# ═════════════════════════════════════════════════════════════════════════

_SPEC_BY_FAMILY = {s[1]: s for s in MODEL_SPECS}

TIER_SPECS = {
    "7b":     {"families": ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"],
               "publish_dir": f"{BASE}/figures/MAIN_RESULTS/4_gap_based_alignment"},
    "s2":     {"families": ["GPT-5-mini", "Llama-70B", "Mistral-13B", "Qwen-72B"],
               "publish_dir": f"{BASE}/figures/SUPPLEMENTARY_RESULTS/2_bigger_models/4_gap_based_alignment"},
    "pooled": {"families": None,
               "publish_dir": f"{BASE}/figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/subset_all_families"},
    # Standalone 13B/14B and 70B/72B family groupings, distinct from the "s2"
    # tier above (which mixes gpt-5-mini/llama-70b/mistral-13b/qwen-72b into
    # one replication set).
    "13b":    {"families": ["Llama-13B", "Mistral-13B", "Qwen-14B"],
               "publish_dir": f"{BASE}/figures/SUPPLEMENTARY_RESULTS/3_13b_tier/4_gap_based_alignment"},
    "70b":    {"families": ["Llama-70B", "Qwen-72B"],
               "publish_dir": f"{BASE}/figures/SUPPLEMENTARY_RESULTS/4_70b_tier/4_gap_based_alignment"},
}
EXPORT_ROOT = f"{BASE}/code/analysis/exports/alignment_by_gap"

# Community/group-size raw sources, reused verbatim from
# analyze_structural_robustness.py's COMMUNITY_SOURCES / GROUP_N12_SOURCES /
# GROUP_N16_SOURCES (same directories, same seed ranges).
COMMUNITY_MODELS = ["llama", "mistral", "qwen", "llama_13b", "mistral_13b", "qwen_14b", "llama_70b", "qwen_72b"]
SEVENB_MODELS = ["llama", "mistral", "qwen"]
_SPEC_BY_KEY = {s[0]: s for s in MODEL_SPECS}

COMMUNITY_SOURCES = {"12": {}, "16": {}, "20": {}}
for _mk in COMMUNITY_MODELS:
    _, _fam, _rdir, _variant, _seeds = _SPEC_BY_KEY[_mk]
    COMMUNITY_SOURCES["12"][_mk] = dict(dir=os.path.join(_rdir, _mk, _variant), seeds=list(_seeds), family=_fam)
    for _n in ("16", "20"):
        COMMUNITY_SOURCES[_n][_mk] = dict(dir=os.path.join(f"{BASE}/code/results", _mk, "local", f"N{_n}_G4"),
                                           seeds=list(range(42, 52)), family=_fam)

GROUP_N12_SOURCES = {"3": {}, "4": {}, "6": {}}
for _mk in SEVENB_MODELS:
    _, _fam, _rdir, _variant, _seeds = _SPEC_BY_KEY[_mk]
    GROUP_N12_SOURCES["4"][_mk] = dict(dir=os.path.join(_rdir, _mk, _variant), seeds=list(_seeds), family=_fam)
    for _g in ("3", "6"):
        GROUP_N12_SOURCES[_g][_mk] = dict(dir=os.path.join(f"{BASE}/code/results", _mk, "local_groupsizevary",
                                                             f"N12_G{_g}_MCPR0.4"),
                                           seeds=list(range(42, 52)), family=_fam)

GROUP_N16_SOURCES = {"4": {}, "8": {}}
for _mk in SEVENB_MODELS:
    _, _fam, _rdir, _v, _s = _SPEC_BY_KEY[_mk]
    GROUP_N16_SOURCES["4"][_mk] = dict(dir=os.path.join(f"{BASE}/code/results", _mk, "local", "N16_G4"),
                                        seeds=list(range(42, 52)), family=_fam)
    GROUP_N16_SOURCES["8"][_mk] = dict(dir=os.path.join(f"{BASE}/code/results", _mk, "local_groupsizevary",
                                                          "N16_G8_MCPR0.4"),
                                        seeds=list(range(42, 52)), family=_fam)

STRUCTURAL_SETTINGS = {
    "community": [("N", n, sources) for n, sources in COMMUNITY_SOURCES.items()],
    "group": [("N12", g, GROUP_N12_SOURCES[g]) for g in ("3", "4", "6")]
            + [("N16", g, GROUP_N16_SOURCES[g]) for g in ("4", "8")],
}


# ═════════════════════════════════════════════════════════════════════════
# 1. Family-tier data loading
# ═════════════════════════════════════════════════════════════════════════

PANEL_PATH = f"{BASE}/code/analysis/exports/round_level_agent_panel.csv"

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
    print(f"  Shape: {df.shape}")

    resolved = {key: resolve_column(df, cands, required=(key not in ("group_id", "group_members", "group_size")))
                for key, cands in COLUMN_CANDIDATES.items()}

    has_in = resolve_column(df, ["IN", "injunctive_norm", "injunctive"], required=False)
    has_dn = resolve_column(df, ["DN", "descriptive_norm", "descriptive"], required=False)
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
              f"(in_groups_this_round == True).")
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
        n_before = len(panel)
        df = panel.merge(perc, on=["run_id", "round", "agent_id"], how="left")
        n_unmatched = df["DN"].isna().sum()
        print(f"  Merged onto panel: {len(df):,} rows "
              f"({n_before:,} panel rows in, {n_unmatched:,} with no perception found).")
    else:
        df = panel.rename(columns={has_in: "IN", has_dn: "DN"})

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
# 1b. Community/group-size data loading (new: raw-log loader, no panel
#     dependency -- same output shape as build_base_dataset() above so it
#     feeds the same prepare_gap_data() downstream)
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


def build_base_dataset_from_sources(sources, axis_label):
    """sources: {model_key: {dir, seeds, family}}. One row per
    (run_id, agent_id, round) with contribution/IN/DN -- same columns
    prepare_gap_data() expects from build_base_dataset()."""
    rows = []
    for model_key, src in sources.items():
        for seed in src["seeds"]:
            found = load_latest_logs_for_seed(os.path.join(src["dir"], f"seed{seed}"))
            for cond, d in found.items():
                run_id = f"{src['family']}_{axis_label}_s{seed}_{cond}"
                for r in d["round_logs"]:
                    percs = r.get("perceptions") or {}
                    contribs = r["contributions"]
                    for aid_str, contrib in contribs.items():
                        perc = percs.get(aid_str) or {}
                        rows.append({
                            "run_id": run_id, "agent_id": int(aid_str), "round": r["round"],
                            "condition": cond, "family": src["family"],
                            "contribution": float(contrib),
                            "IN": perc.get("injunctive_norm"),
                            "DN": perc.get("descriptive_norm"),
                        })
    df = pd.DataFrame(rows)
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
    df["run_id_lead1"] = grp["run_id"].shift(-1)

    consecutive = (df["round_lead1"] == df["round"] + 1)
    has_next = df["round_lead1"].notna()

    n_total = len(df)
    n_last_round_of_traj = (~has_next).sum()
    n_nonconsecutive = (has_next & ~consecutive).sum()

    print(f"\nRow-level trajectory check (n={n_total:,} active agent-round obs):")
    print(f"  Dropped as last round of its run-agent trajectory: {n_last_round_of_traj:,}")
    print(f"  Dropped for non-consecutive round linkage: {n_nonconsecutive:,}")

    df["valid_link"] = has_next & consecutive
    linked = df[df["valid_link"]].copy()

    linked["in_gap"] = linked["IN"] - linked["contribution"]
    linked["dn_gap"] = linked["DN"] - linked["contribution"]
    linked["contribution_shift_lead1"] = linked["contribution_lead1"] - linked["contribution"]

    n_linked = len(linked)
    n_in_null = linked["IN"].isna().sum()
    n_dn_null = linked["DN"].isna().sum()
    print(f"\nLinked (t, t+1) observations available for the gap model: {n_linked:,}")
    print(f"  Removed because IN is null: {n_in_null:,} ({n_in_null / max(n_linked,1):.2%})")
    print(f"  Removed because DN is null: {n_dn_null:,} ({n_dn_null / max(n_linked,1):.2%})")

    null_by_cond = linked.groupby("condition")["IN"].apply(lambda s: s.isna().mean())
    null_by_fam = linked.groupby("family")["IN"].apply(lambda s: s.isna().mean())
    null_by_round = linked.groupby("round")["IN"].apply(lambda s: s.isna().mean())

    final = linked.dropna(subset=["IN", "DN", "contribution_shift_lead1"]).copy()
    final["run_agent_id"] = final["run_id"] + "_" + final["agent_id"].astype(str)

    print(f"\nFinal analysis sample: {len(final):,} observations")
    print(f"  Run-level clusters (run_id): {final['run_id'].nunique():,}")

    exclusion_report = {
        "n_active_agent_round_obs": n_total,
        "n_dropped_last_round_of_trajectory": int(n_last_round_of_traj),
        "n_dropped_nonconsecutive": int(n_nonconsecutive),
        "n_linked_pairs": n_linked,
        "n_removed_in_null": int(n_in_null),
        "pct_removed_in_null": n_in_null / max(n_linked, 1),
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
    return stat, pval, getattr(res, "df_denom", None)


# Reference family for the family FE. Repointed in run_tier() to whichever
# family is actually present in that tier's data (a hardcoded "GPT"
# reference would make patsy raise for any tier that excludes GPT, e.g. s2).
FAMILY_REF = "GPT"


def fe_formula():
    return (
        'C(condition, Treatment(reference="BASELINE")) '
        f'+ C(family, Treatment(reference="{FAMILY_REF}")) '
        '+ C(round, Treatment(reference=1))'
    )


# ═════════════════════════════════════════════════════════════════════════
# 4. Models
# ═════════════════════════════════════════════════════════════════════════

def run_primary_models(df):
    """The headline regression: Shift ~ InGap + DnGap + Condition/Family/Round FE,
    fit twice -- once on raw units (b interpretable in contribution points),
    once with everything z-scored (b interpretable in SDs, for cross-term
    effect-size comparison)."""
    formula_raw = f"contribution_shift_lead1 ~ in_gap + dn_gap + {fe_formula()}"
    m_raw = fit_cluster_ols(formula_raw, df)

    df_z = df.copy()
    df_z["contribution_shift_lead1_z"] = zscale(df_z["contribution_shift_lead1"])
    df_z["in_gap_z"] = zscale(df_z["in_gap"])
    df_z["dn_gap_z"] = zscale(df_z["dn_gap"])
    formula_z = f"contribution_shift_lead1_z ~ in_gap_z + dn_gap_z + {fe_formula()}"
    m_z = fit_cluster_ols(formula_z, df_z)

    return m_raw, m_z, formula_raw, formula_z


def run_lagged_level_model(df):
    """Alternative to the shift specification: predicts next-round contribution
    LEVEL directly from this round's contribution level plus raw IN/DN (not
    gaps), so the gap relationship is implicit in the IN/DN coefficients
    rather than pre-differenced. See lagged_alignment_check.py for the
    independent, differently-specified cross-check of this same idea."""
    formula = f"contribution_lead1 ~ contribution + IN + DN + {fe_formula()}"
    return fit_cluster_ols(formula, df), formula


def run_within_agent_model(df):
    """Agent fixed-effects robustness check: demeans each variable within
    agent (subtracts that agent's own across-run average) before regressing,
    so the estimate comes only from within-agent variation over time, not
    from stable cross-agent differences in gap size or responsiveness."""
    d = df.copy()
    for col in ["in_gap", "dn_gap", "contribution_shift_lead1"]:
        d[col + "_within"] = d[col] - d.groupby("run_agent_id")[col].transform("mean")
    formula = "contribution_shift_lead1_within ~ in_gap_within + dn_gap_within"
    return fit_cluster_ols(formula, d), formula, d


def run_group_behavior_model(df):
    d = df.dropna(subset=["group_mean_contribution_excl_self"]).copy()
    if d.empty:
        return None, None, 0
    formula = (f"contribution_shift_lead1 ~ in_gap + dn_gap + "
               f"group_mean_contribution_excl_self + {fe_formula()}")
    return fit_cluster_ols(formula, d), formula, len(d)


def run_asymmetric_model(df):
    """Splits each gap into its positive part (expectation above own
    contribution) and negative part (expectation below), each zeroed out on
    the other side via clip(), so the model can estimate a separate slope
    for over- vs. under-expectation instead of assuming one symmetric b."""
    d = df.copy()
    d["in_gap_positive"] = d["in_gap"].clip(lower=0)
    d["in_gap_negative"] = d["in_gap"].clip(upper=0)
    d["dn_gap_positive"] = d["dn_gap"].clip(lower=0)
    d["dn_gap_negative"] = d["dn_gap"].clip(upper=0)
    formula = (f"contribution_shift_lead1 ~ in_gap_positive + in_gap_negative "
               f"+ dn_gap_positive + dn_gap_negative + {fe_formula()}")
    return fit_cluster_ols(formula, d), formula


def run_condition_interaction_model(df):
    """Interacts each gap term with condition, so the InGap/DnGap slope is
    allowed to differ by experimental condition instead of being pooled --
    condition main effects are absorbed into the interaction, so no separate
    condition FE term is added alongside it."""
    formula = (
        'contribution_shift_lead1 ~ in_gap * C(condition, Treatment(reference="BASELINE")) '
        '+ dn_gap * C(condition, Treatment(reference="BASELINE")) '
        f'+ C(family, Treatment(reference="{FAMILY_REF}")) + C(round, Treatment(reference=1))'
    )
    return fit_cluster_ols(formula, df), formula


def run_per_family_models(df, family_order):
    formula = (
        'contribution_shift_lead1 ~ in_gap + dn_gap '
        '+ C(condition, Treatment(reference="BASELINE")) + C(round, Treatment(reference=1))'
    )
    results = {}
    for family in family_order:
        d = df[df["family"] == family]
        if d.empty:
            continue
        try:
            results[family] = (fit_cluster_ols(formula, d), len(d), d["run_id"].nunique())
        except Exception as e:
            print(f"  [WARN] per-family model failed for {family}: {e}")
    return results, formula


def run_family_interaction_model(df):
    """Same idea as run_condition_interaction_model but for model family:
    lets the InGap/DnGap slope vary by family, then runs a joint Wald test
    on each set of interaction terms to check whether that variation is
    statistically distinguishable from a single pooled slope."""
    formula = (
        f'contribution_shift_lead1 ~ in_gap * C(family, Treatment(reference="{FAMILY_REF}")) '
        f'+ dn_gap * C(family, Treatment(reference="{FAMILY_REF}")) '
        '+ C(condition, Treatment(reference="BASELINE")) + C(round, Treatment(reference=1))'
    )
    model = fit_cluster_ols(formula, df)
    in_terms = [t for t in model.params.index if t.startswith("in_gap:")]
    dn_terms = [t for t in model.params.index if t.startswith("dn_gap:")]
    het_in = model.wald_test(", ".join(f"{t} = 0" for t in in_terms), scalar=True)
    het_dn = model.wald_test(", ".join(f"{t} = 0" for t in dn_terms), scalar=True)
    het_in_stat, het_in_p = float(np.squeeze(het_in.statistic)), float(np.squeeze(het_in.pvalue))
    het_dn_stat, het_dn_p = float(np.squeeze(het_dn.statistic)), float(np.squeeze(het_dn.pvalue))
    return model, formula, (het_in_stat, het_in_p), (het_dn_stat, het_dn_p)


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
# 6. Main figure
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

def build_results_csv(models_named, exclusion_report, diag, wald_raw, wald_z, het_in, het_dn, out_path):
    rows = []
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
    if het_in is not None and not np.isnan(het_in[1]):
        rows.append({"section": "wald_test", "model": "family_interaction",
                     "term": "in_gap:family jointly 0", "estimate": np.nan, "se": np.nan,
                     "ci_low": np.nan, "ci_high": np.nan, "p_value": het_in[1],
                     "n": np.nan, "n_clusters": np.nan, "r2": np.nan, "r2_adj": np.nan,
                     "note": f"wald_stat={het_in[0]:.4f}"})
        rows.append({"section": "wald_test", "model": "family_interaction",
                     "term": "dn_gap:family jointly 0", "estimate": np.nan, "se": np.nan,
                     "ci_low": np.nan, "ci_high": np.nan, "p_value": het_dn[1],
                     "n": np.nan, "n_clusters": np.nan, "r2": np.nan, "r2_adj": np.nan,
                     "note": f"wald_stat={het_dn[0]:.4f}"})

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
         r"condition FE, model-family FE, and round FE. "
         r"\textit{Lagged-level}: contribution$_{t+1}$ on contribution$_t$, IN$_t$, DN$_t$, "
         r"and the same FE set. "
         r"\textit{Within-agent}: InGap/DnGap and the outcome centered within run-agent unit. "
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


def build_family_latex_table(family_results, het_in, het_dn, family_order, out_path):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Family & InGap$_t$ & DnGap$_t$ & $N$ \\",
        r"\midrule",
    ]
    for family in family_order:
        if family not in family_results:
            continue
        model, n, n_clusters = family_results[family]
        in_cell = fmt_coef(model, "in_gap") if "in_gap" in model.params.index else "---"
        dn_cell = fmt_coef(model, "dn_gap") if "dn_gap" in model.params.index else "---"
        lines.append(rf"  {family} & {in_cell} & {dn_cell} & {n:,} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Primary gap model, refit separately per model family. "
         rf"Joint Wald test that all InGap$\times$family interaction terms are zero: "
         rf"stat={het_in[0]:.2f}, p={het_in[1]:.4f}. Same test for DnGap$\times$family: "
         rf"stat={het_dn[0]:.2f}, p={het_dn[1]:.4f}. "
         r"OLS, SEs clustered by run\_id in parentheses.}"),
        r"\label{tab:gap_alignment_by_family}",
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def build_markdown_summary(m_raw, m_z, m_lagged, m_within, wald_raw, exclusion_report, out_path,
                            family_order, family_results=None, het_in=None, het_dn=None):
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
        "# Gap-based alignment test: summary", "",
        "## Headline result",
        f"- InGap coefficient (b_IN): {b_in:.4f} (p={p_in:.4g}) — {in_dir}.",
        f"- DnGap coefficient (b_DN): {b_dn:.4f} (p={p_dn:.4g}) — {dn_dir}.",
        f"- Coefficient-equality Wald test (H0: b_IN = b_DN): statistic={stat:.4f}, p={p_wald:.4g}.",
        "",
        "## Robustness",
        f"- Lagged-level model: theta_C={lagged_theta_c:.4f}, theta_IN={lagged_theta_in:.4f}, "
        f"theta_DN={lagged_theta_dn:.4f}.",
        f"- Within-run-agent model: InGap={within_b_in:.4f}, DnGap={within_b_dn:.4f}.",
        "",
    ]
    if family_results is not None:
        fam_lines = ["## Is the effect universal across model families?", ""]
        for family in family_order:
            if family not in family_results:
                continue
            model, n, n_clusters = family_results[family]
            b_i, p_i = model.params["in_gap"], model.pvalues["in_gap"]
            b_d, p_d = model.params["dn_gap"], model.pvalues["dn_gap"]
            fam_lines.append(f"- **{family}** (N={n:,}, {n_clusters} clusters): "
                             f"InGap={b_i:+.4f} (p={p_i:.4g}), DnGap={b_d:+.4f} (p={p_d:.4g})")
        if het_in is not None and het_dn is not None:
            fam_lines += ["",
                f"InGap x family jointly zero: stat={het_in[0]:.2f}, p={het_in[1]:.4g}",
                f"DnGap x family jointly zero: stat={het_dn[0]:.2f}, p={het_dn[1]:.4g}", ""]
        lines += fam_lines
    lines += [
        "## Null IN observations",
        f"- {exclusion_report['n_removed_in_null']:,} of {exclusion_report['n_linked_pairs']:,} "
        f"linked (t, t+1) observations ({exclusion_report['pct_removed_in_null']:.2%}) were "
        "dropped because IN was null at t. No imputation was performed.",
        "",
        "## Limitations",
        "- All associations are observational.",
        "- Contribution appears on both sides of the primary gap specification, which is why "
        "the lagged-level model is reported as the essential robustness check.",
        "",
    ]
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# 8. Per-tier pipeline
# ═════════════════════════════════════════════════════════════════════════

def _run_full_pipeline(df, family_order, out_dir, publish_dir, label):
    """Fits every model (primary raw/std, lagged-level, within-agent, group
    behavior, asymmetric, condition interaction, per-family, family
    interaction) and writes the full table/figure/summary suite."""
    global FAMILY_REF
    os.makedirs(out_dir, exist_ok=True)

    families_present = sorted(df["family"].unique())
    FAMILY_REF = "GPT" if "GPT" in families_present else families_present[0]
    print(f"\n[{label}] Family FE reference level: {FAMILY_REF}")

    m_raw, m_z, f_raw, f_z = run_primary_models(df)
    print_focal(m_raw, [("in_gap", "b_IN"), ("dn_gap", "b_DN")], f"[{label}] PRIMARY GAP MODEL (raw)")
    wald_raw = wald_equality(m_raw, "in_gap", "dn_gap")
    wald_z = wald_equality(m_z, "in_gap_z", "dn_gap_z")
    print(f"Wald H0: b_IN=b_DN (raw): stat={wald_raw[0]:.4f}, p={wald_raw[1]:.4g}")

    m_lagged, f_lagged = run_lagged_level_model(df)
    print_focal(m_lagged, [("contribution", "theta_C"), ("IN", "theta_IN"), ("DN", "theta_DN")],
                f"[{label}] LAGGED-LEVEL ROBUSTNESS")

    m_within, f_within, _ = run_within_agent_model(df)
    print_focal(m_within, [("in_gap_within", "InGap (within)"), ("dn_gap_within", "DnGap (within)")],
                f"[{label}] WITHIN-AGENT ROBUSTNESS")

    m_group, f_group, n_group = run_group_behavior_model(df)
    if m_group is not None:
        print_focal(m_group, [("in_gap", "InGap | group mean"), ("dn_gap", "DnGap | group mean")],
                    f"[{label}] GROUP-BEHAVIOR ROBUSTNESS (n={n_group:,})")

    m_asym, f_asym = run_asymmetric_model(df)
    m_inter, f_inter = run_condition_interaction_model(df)

    family_results, f_family = run_per_family_models(df, family_order)
    for family in family_order:
        if family not in family_results:
            continue
        model, n, n_clusters = family_results[family]
        print_focal(model, [("in_gap", "InGap"), ("dn_gap", "DnGap")],
                    f"[{label}] {family}  (N={n:,}, {n_clusters} clusters)")

    if len(families_present) > 1:
        m_family_inter, f_family_inter, het_in, het_dn = run_family_interaction_model(df)
        print(f"\n[{label}] JOINT HETEROGENEITY: InGap x family p={het_in[1]:.4g}, "
              f"DnGap x family p={het_dn[1]:.4g}")
    else:
        m_family_inter, het_in, het_dn = None, (np.nan, np.nan), (np.nan, np.nan)

    diag = compute_diagnostics(df, m_raw)
    plot_diagnostics(df, diag, os.path.join(out_dir, "gap_alignment_diagnostics.png"))
    fig_stub = os.path.join(out_dir, "gap_alignment_binned_shift")
    plot_main_figure(df, fig_stub)

    models_named = [
        ("primary_raw", m_raw, df["run_id"].nunique()),
        ("primary_standardized", m_z, df["run_id"].nunique()),
        ("lagged_level", m_lagged, df["run_id"].nunique()),
        ("within_agent", m_within, df["run_id"].nunique()),
        ("asymmetric_gaps", m_asym, df["run_id"].nunique()),
        ("condition_interaction", m_inter, df["run_id"].nunique()),
    ]
    if m_group is not None:
        models_named.append(("group_behavior", m_group,
                             df.dropna(subset=["group_mean_contribution_excl_self"])["run_id"].nunique()))
    if m_family_inter is not None:
        models_named.append(("family_interaction", m_family_inter, df["run_id"].nunique()))
    models_named += [(f"per_family_{family}", model, n_clusters)
                      for family, (model, n, n_clusters) in family_results.items()]
    csv_path = os.path.join(out_dir, "gap_based_alignment_results.csv")
    build_results_csv(models_named, _last_exclusion_report, diag, wald_raw, wald_z, het_in, het_dn, csv_path)

    tex_path = os.path.join(out_dir, "gap_based_alignment_table.tex")
    build_latex_table(m_raw, m_z, m_lagged, m_within, tex_path)

    family_tex_path = os.path.join(out_dir, "gap_based_alignment_by_family.tex")
    build_family_latex_table(family_results, het_in, het_dn, family_order, family_tex_path)

    md_path = os.path.join(out_dir, "gap_based_alignment_summary.md")
    build_markdown_summary(m_raw, m_z, m_lagged, m_within, wald_raw, exclusion_report=_last_exclusion_report,
                           out_path=md_path, family_order=family_order,
                           family_results=family_results, het_in=het_in, het_dn=het_dn)

    print(f"\n[{label}] Output written to: {out_dir}")

    if publish_dir and os.path.abspath(publish_dir) != os.path.abspath(out_dir):
        os.makedirs(publish_dir, exist_ok=True)
        import shutil
        for fname in os.listdir(out_dir):
            src = os.path.join(out_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(publish_dir, fname))
        print(f"[{label}] Published to: {publish_dir}")

    return {"m_raw": m_raw, "m_z": m_z, "m_lagged": m_lagged, "m_within": m_within,
            "wald_raw": wald_raw, "family_results": family_results,
            "het_in": het_in, "het_dn": het_dn}


_last_exclusion_report = None


def run_family_tier(tier):
    global _last_exclusion_report
    spec = TIER_SPECS[tier]
    families = spec["families"]

    base = build_base_dataset()
    df, exclusion_report = prepare_gap_data(base)
    _last_exclusion_report = exclusion_report

    if families is not None:
        n_before = len(df)
        df = df[df["family"].isin(families)].copy()
        print(f"\n[{tier}] Restricted to {families}: {n_before:,} -> {len(df):,} obs, "
              f"{df['run_id'].nunique()} clusters")
        family_order = [f for f in families if f in set(df["family"])]
    else:
        family_order = [s[1] for s in MODEL_SPECS]

    out_dir = os.path.join(EXPORT_ROOT, f"tier_{tier}")
    return _run_full_pipeline(df, family_order, out_dir, spec["publish_dir"], tier)


def run_structural_tier(axis):
    """axis: 'community' or 'group'. One fit per setting (N, for community;
    (stratum, G) for group), each written to its own subfolder."""
    global _last_exclusion_report
    results = {}
    for stratum, val, sources in STRUCTURAL_SETTINGS[axis]:
        label = f"{axis}_{stratum}_{val}"
        base = build_base_dataset_from_sources(sources, axis_label=f"{stratum}{val}")
        if base.empty:
            print(f"\n[{label}] [WARN] no data assembled, skipping.")
            continue
        df, exclusion_report = prepare_gap_data(base)
        if df.empty:
            print(f"\n[{label}] [WARN] no linked observations, skipping.")
            continue
        _last_exclusion_report = exclusion_report
        family_order = sorted(df["family"].unique())
        out_dir = os.path.join(EXPORT_ROOT, f"tier_{axis}", label)
        publish_dir = os.path.join(
            BASE, "figures", "SUPPLEMENTARY_RESULTS", "5_community_group_mcpr",
            axis, "4_gap_based_alignment", label
        )
        results[label] = _run_full_pipeline(df, family_order, out_dir, publish_dir, label)
    return results


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gap-based alignment test (item c).")
    parser.add_argument("--tier", choices=["7b", "s2", "13b", "70b", "pooled", "community", "group", "all"],
                         default="7b", help="Which tier to run (default: 7b, the primary result).")
    args = parser.parse_args()

    tiers = ["7b", "s2", "13b", "70b", "pooled", "community", "group"] if args.tier == "all" else [args.tier]
    for tier in tiers:
        if tier in TIER_SPECS:
            run_family_tier(tier)
        else:
            run_structural_tier(tier)


if __name__ == "__main__":
    main()
