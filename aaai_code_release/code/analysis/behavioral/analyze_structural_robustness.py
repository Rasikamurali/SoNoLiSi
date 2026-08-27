"""
analyze_structural_robustness.py
----------------------------------
Supplementary structural robustness: do the paper's behavioral, convergence,
and mechanism conclusions hold as social structure changes? Two axes,
analyzed separately (they perturb different things):

  A. COMMUNITY SIZE: N in {12, 16, 20}, G=4, MPCR=0.4 fixed. 8 model variants
     (7B/13B-14B/70B-72B; no GPT -- no community-size sweep data for it).
  B. GROUP SIZE: G varies, 7B trio only, MPCR=0.4 fixed, two SEPARATE matched
     strata (G=3,4,6 at N=12 is not continuous with G=8 at N=16, since G=8
     also changes population size): N=12: G in {3,4,6}; N=16: G in {4,8}.

Run from anywhere: python3 code/analysis/behavioral/analyze_structural_robustness.py
Read-only (glob + json.load on existing logs).


Convergence metric = cross-agent SD per round per seed (ddof=1, >=2 values
required) -- a population SD that doesn't mechanically grow with N. The
social-selection coefficient reuses social_selection_feedback_analysis.py's
Step 5 (undercontribution -> evaluation) by patching its
MODEL_SPECS/REF_FAMILY per structural setting. The social-learning
coefficient reproduces gap_based_alignment.py's primary spec (Shift ~ InGap
+ DnGap + Condition/Family/Round FE) via a self-contained raw-log loader
(build_gap_dataset_from_logs), since that script's own data prep depends on
a pre-built panel that doesn't cover these group-size-variant runs.

Unit of analysis: run-level aggregate (model x setting x condition x seed);
agent-round observations are never independent replicates. Pooled
("model-balanced") estimates use a two-stage model-then-run cluster
bootstrap (5000 resamples, fixed seed) so a run with more observations
doesn't outweigh a smaller one.

Outputs (this directory): QC report, run-level means + late-round SDs,
per-setting summaries, PAPER_PAIRS/matched contrasts, mechanism
coefficients, figures C1-C3, summary tables, and structural_interpretation.md.
"""

import os
import sys
import glob
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/behavioral/) so it still resolves
# correctly if the release is moved or copied elsewhere; anchors raw
# results/ and code/results/ data paths only.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ANALYSIS_DIR separately anchors to this file's own location for locating
# sibling code modules -- same anchor-walk pattern used by every other
# script in this release.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ANALYSIS_DIR, "model_specs.py")):
    ANALYSIS_DIR = os.path.dirname(ANALYSIS_DIR)
for _p in [ANALYSIS_DIR] + [
    os.path.join(ANALYSIS_DIR, d) for d in os.listdir(ANALYSIS_DIR)
    if os.path.isdir(os.path.join(ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_specs import stars, SIG_TEX, MODEL_SPECS as MAIN_MODEL_SPECS  # noqa: E402

# ═════════════════════════════════════════════════════════════════════════
# Reused main-paper conventions
# ═════════════════════════════════════════════════════════════════════════

CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_COLORS = {
    "PURE_BASELINE": "#aaaaaa", "BASELINE": "#6baed6", "NO_SELECTION": "#fd8d3c",
    "NO_DISCUSSION": "#74c476", "FULL": "#e6550d",
}
COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline", "BASELINE": "Baseline", "NO_SELECTION": "No Selection",
    "NO_DISCUSSION": "No Discussion", "FULL": "Full",
}
COND_MARKERS = {"FULL": "o", "NO_DISCUSSION": "s", "NO_SELECTION": "^", "BASELINE": "D", "PURE_BASELINE": "v"}

PAPER_PAIRS = [
    ("PURE_BASELINE", "BASELINE"), ("BASELINE", "NO_DISCUSSION"), ("BASELINE", "NO_SELECTION"),
    ("NO_SELECTION", "FULL"), ("NO_DISCUSSION", "FULL"),
]

COMMUNITY_MODELS = ["llama", "mistral", "qwen", "llama_13b", "mistral_13b", "qwen_14b", "llama_70b", "qwen_72b"]
SEVENB_MODELS = ["llama", "mistral", "qwen"]
MODEL_TIER = {
    "llama": "7B", "mistral": "7B", "qwen": "7B",
    "llama_13b": "13B/14B", "mistral_13b": "13B/14B", "qwen_14b": "13B/14B",
    "llama_70b": "70B/72B", "qwen_72b": "70B/72B",
}
_SPEC_BY_KEY = {s[0]: s for s in MAIN_MODEL_SPECS}  # (model_key, family, results_dir, variant, seeds)
MODEL_LABELS = {k: _SPEC_BY_KEY[k][1] for k in COMMUNITY_MODELS}

LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 13, 11, 10
ENDOWMENT = 10
N_BOOT, BOOT_SEED = 5000, 12345
LATE_ROUNDS = list(range(16, 21))  # matches behavior_quantified.py's LAST_N=5 steady-state window

# ═════════════════════════════════════════════════════════════════════════
# Data sources
# ═════════════════════════════════════════════════════════════════════════
# COMMUNITY: N -> model -> dict(dir, seeds)
COMMUNITY_SOURCES = {"12": {}, "16": {}, "20": {}}
for mk in COMMUNITY_MODELS:
    _, fam, rdir, variant, seeds = _SPEC_BY_KEY[mk]
    COMMUNITY_SOURCES["12"][mk] = dict(dir=os.path.join(rdir, mk, variant), seeds=list(seeds), family=fam)
    # N16/N20 community-size sweep dirs always live under code/results/, regardless of
    # which root (results/ vs code/results/) that model's own N=12 default data uses.
    for n in ("16", "20"):
        COMMUNITY_SOURCES[n][mk] = dict(dir=os.path.join(f"{BASE}/code/results", mk, "local", f"N{n}_G4"),
                                         seeds=list(range(42, 52)), family=fam)

# GROUP SIZE, stratum N=12: G -> model -> dict(dir, seeds)
GROUP_N12_SOURCES = {"3": {}, "4": {}, "6": {}}
for mk in SEVENB_MODELS:
    _, fam, rdir, variant, seeds = _SPEC_BY_KEY[mk]
    GROUP_N12_SOURCES["4"][mk] = dict(dir=os.path.join(rdir, mk, variant), seeds=list(seeds), family=fam)
    for g in ("3", "6"):
        GROUP_N12_SOURCES[g][mk] = dict(dir=os.path.join(f"{BASE}/code/results", mk, "local_groupsizevary",
                                                           f"N12_G{g}_MCPR0.4"),
                                         seeds=list(range(42, 52)), family=fam)

# GROUP SIZE, stratum N=16: G -> model -> dict(dir, seeds)
GROUP_N16_SOURCES = {"4": {}, "8": {}}
for mk in SEVENB_MODELS:
    _, fam, rdir, _v, _s = _SPEC_BY_KEY[mk]
    GROUP_N16_SOURCES["4"][mk] = dict(dir=os.path.join(f"{BASE}/code/results", mk, "local", "N16_G4"),
                                       seeds=list(range(42, 52)), family=fam)
    GROUP_N16_SOURCES["8"][mk] = dict(dir=os.path.join(f"{BASE}/code/results", mk, "local_groupsizevary",
                                                         "N16_G8_MCPR0.4"),
                                       seeds=list(range(42, 52)), family=fam)


# ═════════════════════════════════════════════════════════════════════════
# 1. Loader: run-level contribution means + late-round convergence SDs
#    (one raw-log read per run, produces both in one pass)
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


def sd_or_nan(vals):
    return float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan


def load_axis(sources, axis_col, extra_cols_fn=None):
    """sources: {axis_value(str): {model: dict(dir, seeds, family)}}.
    Returns (df, qc) where df has one row per (axis_value, model, condition,
    seed) with mean_contribution + late-round contrib_sd/in_sd/dn_sd."""
    rows = []
    qc = {"missing_cells": [], "dup_seed_dirs": [], "n_raw_files_by_cell": {}}
    for axis_val, model_map in sources.items():
        for model, src in model_map.items():
            for seed in src["seeds"]:
                seed_dir = os.path.join(src["dir"], f"seed{seed}")
                found = load_latest_logs_for_seed(seed_dir)
                n_raw = len(glob.glob(os.path.join(seed_dir, "log_*.json")))
                qc["n_raw_files_by_cell"][(axis_val, model, seed)] = n_raw
                if n_raw > len(CONDITIONS):
                    qc["dup_seed_dirs"].append((axis_val, model, seed, n_raw))
                for cond in CONDITIONS:
                    d = found.get(cond)
                    if d is None:
                        qc["missing_cells"].append((axis_val, model, cond, seed))
                        continue
                    vals = []
                    late_contrib, late_in, late_dn = [], [], []
                    for r in d["round_logs"]:
                        cvals = [float(v) for v in r["contributions"].values()]
                        vals.extend(cvals)
                        if r["round"] in LATE_ROUNDS:
                            in_vals, dn_vals = [], []
                            for perc in (r.get("perceptions") or {}).values():
                                if not perc:
                                    continue
                                if perc.get("injunctive_norm") is not None:
                                    in_vals.append(float(perc["injunctive_norm"]))
                                if perc.get("descriptive_norm") is not None:
                                    dn_vals.append(float(perc["descriptive_norm"]))
                            late_contrib.append(sd_or_nan(cvals))
                            late_in.append(sd_or_nan(in_vals))
                            late_dn.append(sd_or_nan(dn_vals))
                    if not vals:
                        qc["missing_cells"].append((axis_val, model, cond, seed))
                        continue
                    row = {
                        axis_col: axis_val, "model": model, "tier": MODEL_TIER.get(model, "7B"),
                        "family": src["family"], "condition": cond, "seed": seed,
                        "mean_contribution": float(np.mean(vals)),
                        "n_agent_round_obs": len(vals), "n_agents": d["round_logs"][0]["contributions"].__len__(),
                        "late_contrib_sd": float(np.nanmean(late_contrib)) if late_contrib else np.nan,
                        "late_in_sd": float(np.nanmean(late_in)) if late_in else np.nan,
                        "late_dn_sd": float(np.nanmean(late_dn)) if late_dn else np.nan,
                    }
                    if extra_cols_fn:
                        row.update(extra_cols_fn(axis_val))
                    rows.append(row)
    return pd.DataFrame(rows), qc


# ═════════════════════════════════════════════════════════════════════════
# 2. QC report
# ═════════════════════════════════════════════════════════════════════════

def qc_for_axis(df, qc, sources, axis_col, axis_label, log):
    log(f"\n[{axis_label}] expected vs. found ({axis_col} x model x condition x seed):")
    total_expected = total_found = 0
    for axis_val, model_map in sources.items():
        for model, src in model_map.items():
            expected = len(src["seeds"]) * len(CONDITIONS)
            found = len(df[(df[axis_col] == axis_val) & (df.model == model)])
            unique_seeds = df[(df[axis_col] == axis_val) & (df.model == model)]["seed"].nunique()
            total_expected += expected
            total_found += found
            flag = "OK" if found == expected else f"MISSING {expected - found}"
            log(f"  {axis_col}={axis_val:<4} {model:<12}: {found:>3}/{expected:<3}  "
                f"({unique_seeds} unique seeds)  {flag}")
    log(f"  Totals: {total_found}/{total_expected} cells found.")

    dedup_events = [e for e in qc["dup_seed_dirs"]]
    log(f"  Duplicate/rerun seed-dirs requiring latest-timestamp-wins dedup: {len(dedup_events)}")
    for axis_val, model, seed, n_raw in dedup_events[:15]:
        log(f"    {axis_col}={axis_val} model={model} seed={seed}: {n_raw} raw log files "
            f"({n_raw - len(CONDITIONS)} extra rerun(s) removed)")

    if qc["missing_cells"]:
        log(f"  [WARNING] {len(qc['missing_cells'])} missing individual run x condition cells:")
        for axis_val, model, cond, seed in qc["missing_cells"][:20]:
            log(f"    {axis_col}={axis_val} model={model} condition={cond} seed={seed}")
    else:
        log("  No missing individual cells beyond whole-axis-value gaps (there are none here by design).")

    bad = df[(df.mean_contribution < 0) | (df.mean_contribution > ENDOWMENT)]
    log(f"  Range check: {len(bad)} run-level means outside [0, {ENDOWMENT}]. {'OK' if len(bad) == 0 else 'INVESTIGATE'}")
    dup_rows = df.duplicated(subset=[axis_col, "model", "condition", "seed"]).sum()
    log(f"  Duplicate-row check: {dup_rows} duplicate rows in final table. {'OK' if dup_rows == 0 else 'INVESTIGATE'}")
    assert "gpt" not in df["model"].unique(), "GPT must never appear in structural robustness data."
    log(f"  GPT-exclusion check: OK (GPT not present in {axis_label} data).")


# ═════════════════════════════════════════════════════════════════════════
# 3. Pooled model-balanced summary + cluster bootstrap
# ═════════════════════════════════════════════════════════════════════════

def model_clustered_bootstrap(df, axis_col, axis_val, condition, models_subset, value_col="mean_contribution",
                               n_boot=N_BOOT, seed=BOOT_SEED):
    cell = df[(df[axis_col] == axis_val) & (df.condition == condition) & (df.model.isin(models_subset))]
    models_present = sorted(cell["model"].unique())
    if not models_present:
        return None
    per_model = {m: cell.loc[cell.model == m, value_col].dropna().values for m in models_present}
    per_model = {m: v for m, v in per_model.items() if len(v) > 0}
    models_present = sorted(per_model.keys())
    if not models_present:
        return None
    # Point estimate: mean-of-model-means, so every model contributes equally
    # regardless of how many seeds/agents it has (model-balanced, not pooled).
    point_est = float(np.mean([arr.mean() for arr in per_model.values()]))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    models_arr = np.array(models_present)
    for b in range(n_boot):
        # Two-stage resample: first resample WHICH models are in this draw
        # (with replacement), then resample each of those models' own
        # run-level values (with replacement) -- this is what keeps a model
        # with more seeds/runs from dominating the uncertainty estimate.
        resampled_models = rng.choice(models_arr, size=len(models_arr), replace=True)
        model_means = [rng.choice(per_model[m], size=len(per_model[m]), replace=True).mean()
                       for m in resampled_models]
        boot[b] = np.mean(model_means)
    se = float(boot.std(ddof=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])  # percentile bootstrap CI
    return dict(mean=point_est, se=se, ci_lo=float(lo), ci_hi=float(hi),
                n_models=len(models_present), n_seeds=int(cell["seed"].nunique() * len(models_present)))


def seed_level_summary(df, axis_col, axis_val, condition, model, value_col="mean_contribution"):
    cell = df[(df[axis_col] == axis_val) & (df.condition == condition) & (df.model == model)]
    vals = cell[value_col].dropna().values
    if len(vals) == 0:
        return None
    n = len(vals)
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    tcrit = scipy_stats.t.ppf(0.975, max(n - 1, 1))
    return dict(mean=mean, se=se, ci_lo=mean - tcrit * se, ci_hi=mean + tcrit * se, n_models=1, n_seeds=n)


def build_summary(df, axis_col, axis_values, pool_specs, value_col="mean_contribution"):
    """pool_specs: list of (pool_name, models_subset) e.g. [("Pooled8", COMMUNITY_MODELS), ("Pooled7B", SEVENB_MODELS)]"""
    rows = []
    for axis_val in axis_values:
        for condition in CONDITIONS:
            for pool_name, models_subset in pool_specs:
                r = model_clustered_bootstrap(df, axis_col, axis_val, condition, models_subset, value_col)
                if r is not None:
                    rows.append({axis_col: axis_val, "condition": condition, "model": pool_name,
                                 "method": "model-clustered bootstrap (n=5000)", "metric": value_col, **r})
            for model in df[df[axis_col] == axis_val]["model"].unique():
                r = seed_level_summary(df, axis_col, axis_val, condition, model, value_col)
                if r is not None:
                    rows.append({axis_col: axis_val, "condition": condition, "model": model,
                                 "method": "t-based 95% CI across run-level means", "metric": value_col, **r})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 4. Regression helpers (reused pattern: OLS + Wald contrast on Treatment coding)
# ═════════════════════════════════════════════════════════════════════════

REF_COND = "BASELINE"


def fit_ols_hc3(df, formula):
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def cond_param(cond, ref=REF_COND):
    return None if cond == ref else f"C(condition, Treatment('{ref}'))[T.{cond}]"


def wald_contrast(result, key_b, key_a):
    params, cov = result.params, result.cov_params()
    # A missing key means that level IS the regression's reference level,
    # which has no coefficient of its own (implicitly 0, with 0 variance).
    def _c(k): return params[k] if k and k in params.index else 0.0
    def _v(k): return cov.loc[k, k] if k and k in cov.index else 0.0
    def _cv(a, b): return cov.loc[a, b] if a and b and a in cov.index and b in cov.columns else 0.0
    diff = _c(key_b) - _c(key_a)
    # Var(B-A) = Var(B) + Var(A) - 2*Cov(A,B); a two-sided z-test on that diff.
    se = np.sqrt(max(_v(key_b) + _v(key_a) - 2 * _cv(key_b, key_a), 0.0))
    z = diff / se if se > 0 else np.nan
    p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return diff, se, z, p


def pick_ref_model(models_present, preferred):
    return preferred if preferred in models_present else sorted(models_present)[0]


# ═════════════════════════════════════════════════════════════════════════
# 5. PART I -- COMMUNITY SIZE
# ═════════════════════════════════════════════════════════════════════════

def community_condition_contrasts(df, log):
    rows = []
    for pool_name, models_subset, ref_model in [("Pooled8", COMMUNITY_MODELS, "mistral"),
                                                  ("Pooled7B", SEVENB_MODELS, "mistral")]:
        for n in ["12", "16", "20"]:
            sub = df[(df.N == n) & (df.model.isin(models_subset))]
            if sub.empty:
                continue
            models_here = sorted(sub["model"].unique())
            rm = pick_ref_model(models_here, ref_model)
            formula = (f"mean_contribution ~ C(condition, Treatment('{REF_COND}'))"
                       + (f" + C(model, Treatment('{rm}'))" if len(models_here) > 1 else ""))
            res = fit_ols_hc3(sub, formula)
            for cA, cB in PAPER_PAIRS:
                diff, se, z, p = wald_contrast(res, cond_param(cB), cond_param(cA))
                rows.append({"analysis": "community", "N": n, "G": "4", "scope": pool_name,
                             "contrast": f"{cA} -> {cB}", "diff": diff, "se": se, "p": p, "n": len(sub)})
    n_c = len(PAPER_PAIRS)
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out["p"] * n_c).clip(upper=1.0)
    out["ci_lo"], out["ci_hi"] = out["diff"] - 1.96 * out["se"], out["diff"] + 1.96 * out["se"]
    out["sig"] = out["p_bonf"].apply(stars)
    return out


def community_interaction_models(df, log):
    log("\n" + "=" * 72 + "\nCOMMUNITY SIZE: Contribution ~ Condition * N + Model FE (Pooled8, HC3)\n" + "=" * 72)
    models_here = sorted(df["model"].unique())
    rm = pick_ref_model(models_here, "mistral")
    formula = (f"mean_contribution ~ C(condition, Treatment('{REF_COND}')) * C(N, Treatment('12')) "
               f"+ C(model, Treatment('{rm}'))")
    res = fit_ols_hc3(df, formula)
    log(res.summary().as_text())
    # Joint Wald test that every Condition x N interaction coefficient is 0 --
    # i.e. whether the condition effect is uniform across community sizes.
    inter = [t for t in res.params.index if ":" in t]
    het = None
    if inter:
        wt = res.wald_test(", ".join(f"{t} = 0" for t in inter), scalar=True)
        het = (float(np.squeeze(wt.statistic)), float(np.squeeze(wt.pvalue)), len(inter))
        log(f"\nJoint Wald test, Condition x N = 0 (df={het[2]}): stat={het[0]:.3f}, p={het[1]:.4g}")

    log("\n" + "-" * 72 + "\nHETEROGENEITY CHECK: Condition * N * ModelTier (joint interaction test only)\n" + "-" * 72)
    # Same idea one level up: does the Condition x N pattern itself differ by
    # model-scale tier? Only the 3-way (":"-count == 2) interaction terms
    # answer that -- the 2-way terms already tested above are excluded.
    formula3 = (f"mean_contribution ~ C(condition, Treatment('{REF_COND}')) * C(N, Treatment('12')) "
                f"* C(tier, Treatment('7B'))")
    res3 = fit_ols_hc3(df, formula3)
    inter3 = [t for t in res3.params.index if t.count(":") == 2]
    het3 = None
    if inter3:
        wt3 = res3.wald_test(", ".join(f"{t} = 0" for t in inter3), scalar=True)
        het3 = (float(np.squeeze(wt3.statistic)), float(np.squeeze(wt3.pvalue)), len(inter3))
        log(f"Joint Wald test, 3-way Condition x N x Tier = 0 (df={het3[2]}): "
            f"stat={het3[0]:.3f}, p={het3[1]:.4g}")
        log("  -> " + ("Reject H0: the pooled community-size effect is NOT uniform across tiers -- "
                        "inspect community_summary.csv by tier before trusting the pooled figure alone."
                        if het3[1] < 0.05 else
                        "Cannot reject H0: no strong evidence the community-size effect differs by tier."))
    return res, het, res3, het3


def community_mechanism(log):
    """Reuses social_selection_feedback_analysis.py's Step 5
    (undercontribution -> mean_evaluation_received) via monkey-patch, one
    coefficient per N in {12, 16, 20}, pooled across all 8 model variants."""
    import model_specs as sm
    import social_selection_feedback_analysis as ssfa

    variant_by_n = {"12": None, "16": "local/N16_G4", "20": "local/N20_G4"}
    rows = []
    log("\n" + "=" * 72 + "\nCOMMUNITY-SIZE MECHANISM: social-selection (Step 5 of "
        "social_selection_feedback_analysis.py)\n" + "=" * 72)
    for n, variant_override in variant_by_n.items():
        specs = []
        for mk in COMMUNITY_MODELS:
            _, fam, rdir, variant, seeds = _SPEC_BY_KEY[mk]
            # N16/N20 community-size dirs always live under code/results/, same fix as COMMUNITY_SOURCES above.
            v = variant_override if variant_override else variant
            r = f"{BASE}/code/results" if variant_override else rdir
            specs.append((mk, fam, r, v, list(range(42, 52)) if variant_override else list(seeds)))
        ssfa.MODEL_SPECS = specs
        ssfa.REF_FAMILY = "Mistral-7B"
        ssfa.N_AGENTS = int(n)  # community-size sweep changes population size; ssfa hardcodes N_AGENTS=12
        ssfa.OUT_DIR = os.path.join(OUT_DIR, f"_tmp_ssfa_N{n}")  # step5_analysis doesn't write files itself
        try:
            runs, edge_df, agent_df = ssfa.build_core_datasets()
            desc, reg_out, text = ssfa.step5_analysis(agent_df)
            r = reg_out[reg_out["model"] == "pooled_ols_with_FE"].iloc[0]
            rows.append({"N": n, "term": "undercontribution", "estimate": r["estimate"], "se": r["se"],
                         "ci_lo": r["ci_low"], "ci_hi": r["ci_high"], "p_value": r["p_value"],
                         "n_obs": r["n"], "n_clusters": r["n_clusters"],
                         "n_models": len(specs), "families": [s[1] for s in specs]})
            log(f"\nN={n} (models: {[s[0] for s in specs]}):")
            log(f"  undercontribution -> mean_evaluation_received: "
                f"b={r['estimate']:.4f} SE={r['se']:.4f} 95%CI=[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
                f"p={r['p_value']:.4g}  (n={int(r['n']):,}, clusters={int(r['n_clusters'])})")
        except Exception as e:
            log(f"\nN={n}: [WARN] mechanism model failed: {e}")
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 6. PART II -- GROUP SIZE
# ═════════════════════════════════════════════════════════════════════════

def group_matched_contrasts(df, log):
    """Delta_G = coefficient on C(G, Treatment('4'))[T.<G>] fit per condition,
    pooled across the 7B trio with model FE, HC3 robust."""
    rows = []
    strata = [("N12", ["3", "4", "6"]), ("N16", ["4", "8"])]
    for stratum_name, gvals in strata:
        sub_all = df[df.stratum == stratum_name]
        for cond in CONDITIONS:
            sub = sub_all[sub_all.condition == cond]
            if sub.empty or sub["G"].nunique() < 2:
                continue
            models_here = sorted(sub["model"].unique())
            rm = pick_ref_model(models_here, "mistral")
            formula = (f"mean_contribution ~ C(G, Treatment('4'))"
                       + (f" + C(model, Treatment('{rm}'))" if len(models_here) > 1 else ""))
            res = fit_ols_hc3(sub, formula)
            for g in gvals:
                if g == "4":
                    continue
                key = f"C(G, Treatment('4'))[T.{g}]"
                if key not in res.params.index:
                    continue
                diff, se = res.params[key], res.bse[key]
                p = res.pvalues[key]
                rows.append({"analysis": "group_size", "stratum": stratum_name, "condition": cond,
                             "contrast": f"G{g} - G4", "diff": diff, "se": se, "p": p,
                             "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se, "n": len(sub)})
    out = pd.DataFrame(rows)
    if not out.empty:
        n_c = len(CONDITIONS)
        out["p_bonf"] = (out["p"] * n_c).clip(upper=1.0)
        out["sig"] = out["p_bonf"].apply(stars)
    return out


def group_interaction_models(df, log):
    results = {}
    for stratum_name, ref_g in [("N12", "4"), ("N16", "4")]:
        sub = df[df.stratum == stratum_name]
        models_here = sorted(sub["model"].unique())
        rm = pick_ref_model(models_here, "mistral")
        log(f"\n{'-'*72}\nGROUP SIZE ({stratum_name}): Contribution ~ Condition * G + Model FE (HC3)\n{'-'*72}")
        formula = (f"mean_contribution ~ C(condition, Treatment('{REF_COND}')) * C(G, Treatment('{ref_g}')) "
                   f"+ C(model, Treatment('{rm}'))")
        res = fit_ols_hc3(sub, formula)
        log(res.summary().as_text())
        inter = [t for t in res.params.index if ":" in t]
        het = None
        if inter:
            wt = res.wald_test(", ".join(f"{t} = 0" for t in inter), scalar=True)
            het = (float(np.squeeze(wt.statistic)), float(np.squeeze(wt.pvalue)), len(inter))
            log(f"\nJoint Wald test, Condition x G = 0 (df={het[2]}): stat={het[0]:.3f}, p={het[1]:.4g}")
        results[stratum_name] = (res, het)
    return results


def build_gap_dataset_from_logs(sources_for_setting):
    """Self-contained reproduction of gap_based_alignment.py's primary
    data prep (InGap/DnGap/Shift), built directly from raw logs since that
    script's own data prep depends on a pre-built panel that doesn't cover
    group-size-variant runs (see module docstring)."""
    rows = []
    for model, src in sources_for_setting.items():
        for seed in src["seeds"]:
            found = load_latest_logs_for_seed(os.path.join(src["dir"], f"seed{seed}"))
            for cond, d in found.items():
                run_id = f"{src['family']}_s{seed}_{cond}_{src['dir']}"
                by_round = {}
                for r in d["round_logs"]:
                    by_round[r["round"]] = r
                sorted_rounds = sorted(by_round)
                for i, rnd in enumerate(sorted_rounds):
                    r = by_round[rnd]
                    percs = r.get("perceptions") or {}
                    contribs = r["contributions"]
                    next_r = by_round.get(sorted_rounds[i + 1]) if i + 1 < len(sorted_rounds) else None
                    for aid, perc in percs.items():
                        if not perc:
                            continue
                        inj, desc = perc.get("injunctive_norm"), perc.get("descriptive_norm")
                        contrib = contribs.get(aid)
                        if inj is None or desc is None or contrib is None:
                            continue
                        contrib_next = None
                        if next_r is not None:
                            contrib_next = next_r["contributions"].get(aid)
                        if contrib_next is None:
                            continue
                        rows.append({
                            "run_id": run_id, "family": src["family"], "model": model,
                            "condition": cond, "seed": seed, "round": rnd, "agent_id": aid,
                            "in_gap": float(inj) - float(contrib), "dn_gap": float(desc) - float(contrib),
                            "contribution_shift_lead1": float(contrib_next) - float(contrib),
                        })
    return pd.DataFrame(rows)


def group_mechanism(log):
    """Reproduces gap_based_alignment.py's primary specification
    (Shift ~ InGap + DnGap + Condition FE + Family FE + Round FE) for each
    group-size setting, pooled across the 7B trio."""
    settings = [("N12", "3", GROUP_N12_SOURCES["3"]), ("N12", "4", GROUP_N12_SOURCES["4"]),
                ("N12", "6", GROUP_N12_SOURCES["6"]), ("N16", "4", GROUP_N16_SOURCES["4"]),
                ("N16", "8", GROUP_N16_SOURCES["8"])]
    rows = []
    log("\n" + "=" * 72 + "\nGROUP-SIZE MECHANISM: social-learning correction "
        "(gap_based_alignment.py's primary specification)\n" + "=" * 72)
    for stratum, g, sources in settings:
        gdf = build_gap_dataset_from_logs(sources)
        if gdf.empty:
            log(f"\n{stratum} G={g}: [WARN] no data assembled, skipping.")
            continue
        families_here = sorted(gdf["family"].unique())
        ref_fam = pick_ref_model(families_here, "Mistral-7B")
        formula = (f"contribution_shift_lead1 ~ in_gap + dn_gap "
                   f"+ C(condition, Treatment('BASELINE')) + C(family, Treatment('{ref_fam}')) "
                   f"+ C(round, Treatment(1))")
        res = smf.ols(formula, data=gdf).fit(cov_type="cluster", cov_kwds={"groups": gdf["run_id"]})
        for term in ["in_gap", "dn_gap"]:
            rows.append({"stratum": stratum, "G": g, "term": term, "estimate": res.params[term],
                        "se": res.bse[term], "ci_lo": res.conf_int().loc[term, 0],
                        "ci_hi": res.conf_int().loc[term, 1], "p_value": res.pvalues[term],
                        "n_obs": int(res.nobs), "n_clusters": gdf["run_id"].nunique()})
        log(f"\n{stratum} G={g} (n={int(res.nobs):,}, clusters={gdf['run_id'].nunique()}):")
        log(f"  in_gap: b={res.params['in_gap']:.4f} SE={res.bse['in_gap']:.4f} p={res.pvalues['in_gap']:.4g}")
        log(f"  dn_gap: b={res.params['dn_gap']:.4f} SE={res.bse['dn_gap']:.4f} p={res.pvalues['dn_gap']:.4g}")
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 7. Figures
# ═════════════════════════════════════════════════════════════════════════

def _plot_lines(ax, summary_sub, x_order, x_col, annotate_n=False):
    x_pos = {v: i for i, v in enumerate(x_order)}
    for cond in CONDITIONS:
        s = summary_sub[summary_sub.condition == cond].sort_values(x_col)
        if s.empty:
            continue
        xs = [x_pos[v] for v in s[x_col]]
        ys = s["mean"].values
        errs = np.abs(np.vstack([ys - s["ci_lo"].values, s["ci_hi"].values - ys]))
        ax.errorbar(xs, ys, yerr=errs, color=COND_COLORS[cond], marker=COND_MARKERS[cond],
                    markersize=6, markeredgecolor="white", markeredgewidth=0.5, linewidth=2,
                    capsize=3, label=COND_LABELS[cond])
        if annotate_n:
            for x, y, nm in zip(xs, ys, s["n_models"]):
                ax.annotate(f"n={int(nm)}", (x, y), textcoords="offset points", xytext=(0, 7),
                            fontsize=6.5, color="gray", ha="center")
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels(x_order, fontsize=TICK_SIZE)
    ax.set_xlim(-0.4, len(x_order) - 0.6)
    ax.set_ylim(0, ENDOWMENT + 0.5)
    ax.axhline(ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.4)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)


def make_figure_c1(comm_summary, group_summary):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    panelA = comm_summary[comm_summary.model == "Pooled8"]
    _plot_lines(axes[0], panelA, ["12", "16", "20"], "N", annotate_n=True)
    axes[0].set_title("A. Community size\n(8 model variants)", fontsize=LABEL_SIZE, fontweight="bold")
    axes[0].set_xlabel("N", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("Mean Contribution (run-level)", fontsize=LABEL_SIZE)

    panelB = comm_summary[comm_summary.model == "Pooled7B"]
    _plot_lines(axes[1], panelB, ["12", "16", "20"], "N", annotate_n=True)
    axes[1].set_title("B. Community size\n(7B trio only)", fontsize=LABEL_SIZE, fontweight="bold")
    axes[1].set_xlabel("N", fontsize=LABEL_SIZE)

    panelC = group_summary[(group_summary.stratum == "N12") & (group_summary.model == "Pooled")]
    _plot_lines(axes[2], panelC, ["3", "4", "6"], "G", annotate_n=True)
    axes[2].set_title("C. Group size at N=12\n(7B trio)", fontsize=LABEL_SIZE, fontweight="bold")
    axes[2].set_xlabel("G", fontsize=LABEL_SIZE)

    panelD = group_summary[(group_summary.stratum == "N16") & (group_summary.model == "Pooled")]
    _plot_lines(axes[3], panelD, ["4", "8"], "G", annotate_n=True)
    axes[3].set_title("D. Group size at N=16\n(7B trio)", fontsize=LABEL_SIZE, fontweight="bold")
    axes[3].set_xlabel("G", fontsize=LABEL_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS), fontsize=LEGEND_SIZE,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Structural Robustness of Contributions", fontsize=LABEL_SIZE + 3, y=1.04)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(OUT_DIR, "figure_C1_structural_contributions.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figure_C1_structural_contributions.pdf/.png")


def make_tier_validation_figure(comm_summary):
    tiers = ["7B", "13B/14B", "70B/72B"]
    tier_models = {"7B": SEVENB_MODELS, "13B/14B": ["llama_13b", "mistral_13b", "qwen_14b"],
                   "70B/72B": ["llama_70b", "qwen_72b"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, tier in zip(axes, tiers):
        pool_name = f"PooledTier_{tier}"
        sub = comm_summary[comm_summary.model == pool_name]
        _plot_lines(ax, sub, ["12", "16", "20"], "N", annotate_n=True)
        ax.set_title(f"{tier} ({', '.join(MODEL_LABELS[m] for m in tier_models[tier])})",
                     fontsize=LABEL_SIZE - 1, fontweight="bold")
        ax.set_xlabel("N", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("Mean Contribution (run-level)", fontsize=LABEL_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS), fontsize=LEGEND_SIZE,
               frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Community-Size Robustness by Model Scale Tier", fontsize=LABEL_SIZE + 3, y=1.04)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    out = os.path.join(OUT_DIR, "community_model_tier_validation.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: community_model_tier_validation.pdf/.png")


def make_figure_c2(comm_conv_summary, group_conv_summary):
    outcomes = [("late_contrib_sd", "Contribution SD"), ("late_in_sd", "IN SD (normative expect.)"),
                ("late_dn_sd", "DN SD (empirical expect.)")]
    panels = [("comm8", "Community (8-model)", comm_conv_summary[comm_conv_summary.model == "Pooled8"], "N", ["12", "16", "20"]),
              ("comm7b", "Community (7B)", comm_conv_summary[comm_conv_summary.model == "Pooled7B"], "N", ["12", "16", "20"]),
              ("groupN12", "Group size, N=12", group_conv_summary[(group_conv_summary.stratum == "N12") & (group_conv_summary.model == "Pooled")], "G", ["3", "4", "6"]),
              ("groupN16", "Group size, N=16", group_conv_summary[(group_conv_summary.stratum == "N16") & (group_conv_summary.model == "Pooled")], "G", ["4", "8"])]

    fig, axes = plt.subplots(len(outcomes), len(panels), figsize=(4.3 * len(panels), 3.6 * len(outcomes)), sharey="row")
    ymax_by_row = {}
    for row, (metric_col, metric_label) in enumerate(outcomes):
        for col, (key, title, data, xcol, xorder) in enumerate(panels):
            ax = axes[row, col]
            x_pos = {v: i for i, v in enumerate(xorder)}
            for cond in CONDITIONS:
                s = data[data.condition == cond].sort_values(xcol)
                if s.empty:
                    continue
                s = s[s["metric"] == metric_col] if "metric" in s.columns else s
                if s.empty:
                    continue
                xs = [x_pos[v] for v in s[xcol]]
                ys, ci_lo, ci_hi = s["mean"].values, s["ci_lo"].values, s["ci_hi"].values
                errs = np.abs(np.vstack([ys - ci_lo, ci_hi - ys]))
                ax.errorbar(xs, ys, yerr=errs, color=COND_COLORS[cond], marker=COND_MARKERS[cond],
                            markersize=5, linewidth=1.6, capsize=2.5, label=COND_LABELS[cond])
            ax.set_xticks(list(x_pos.values()))
            ax.set_xticklabels(xorder, fontsize=TICK_SIZE - 1)
            ax.set_xlim(-0.4, len(xorder) - 0.6)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=TICK_SIZE - 1)
            if row == 0:
                ax.set_title(title, fontsize=LABEL_SIZE - 1, fontweight="bold")
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=LABEL_SIZE - 1)
            if row == len(outcomes) - 1:
                ax.set_xlabel(xcol, fontsize=LABEL_SIZE - 1)
    for row in range(len(outcomes)):
        ymax = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(0, ymax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS), fontsize=LEGEND_SIZE,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Structural Robustness of Convergence (late-round mean, rounds {LATE_ROUNDS[0]}-{LATE_ROUNDS[-1]})",
                 fontsize=LABEL_SIZE + 3, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT_DIR, "figure_C2_structural_convergence.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figure_C2_structural_convergence.pdf/.png")


def make_figure_c3(comm_mech, group_mech):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    if not comm_mech.empty:
        ys = list(range(len(comm_mech)))
        ax.errorbar(comm_mech["estimate"], ys, xerr=[comm_mech["estimate"] - comm_mech["ci_lo"],
                                                       comm_mech["ci_hi"] - comm_mech["estimate"]],
                    fmt="o", color="#e6550d", capsize=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"N={n}" for n in comm_mech["N"]], fontsize=TICK_SIZE)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("undercontribution -> evaluation (b)", fontsize=LABEL_SIZE - 1)
        ax.set_title("Community size:\nsocial-selection sanctioning", fontsize=LABEL_SIZE - 1, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

    for ax, term, title in [(axes[1], "in_gap", "InGap -> shift"), (axes[2], "dn_gap", "DnGap -> shift")]:
        sub = group_mech[group_mech.term == term].copy()
        if sub.empty:
            continue
        sub["label"] = sub["stratum"] + " G=" + sub["G"]
        ys = list(range(len(sub)))
        ax.errorbar(sub["estimate"], ys, xerr=[sub["estimate"] - sub["ci_lo"], sub["ci_hi"] - sub["estimate"]],
                    fmt="o", color="#3182bd", capsize=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["label"], fontsize=TICK_SIZE)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{title} (b)", fontsize=LABEL_SIZE - 1)
        ax.set_title(f"Group size:\nsocial-learning ({term})", fontsize=LABEL_SIZE - 1, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2)

    fig.suptitle("Mechanism Robustness", fontsize=LABEL_SIZE + 3, y=1.05)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "figure_C3_mechanism_robustness.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figure_C3_mechanism_robustness.pdf/.png")


# ═════════════════════════════════════════════════════════════════════════
# 8. Tables
# ═════════════════════════════════════════════════════════════════════════

def make_summary_table(comm_summary, group_summary):
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\caption{Structural robustness: mean contribution (SE) by structural setting and condition. "
             r"Pooled columns are model-balanced (equal weight per model variant); SEs from a two-stage "
             r"model-then-run cluster bootstrap, $n_{\text{boot}}=5000$.}",
             r"\label{tab:structural_summary}", r"\begin{tabular}{lllcc}", r"\toprule",
             r"Analysis & Setting & Condition & Pooled mean (SE) & $n_{\text{models}}$ \\", r"\midrule"]
    for label, summary, col, order in [("Community (8-model)", comm_summary, "N", ["12", "16", "20"]),
                                        ("Community (7B)", comm_summary, "N", ["12", "16", "20"])]:
        pool_name = "Pooled8" if "8-model" in label else "Pooled7B"
        for v in order:
            for cond in CONDITIONS:
                row = summary[(summary[col] == v) & (summary.condition == cond) & (summary.model == pool_name)]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(f"{label} & {col}={v} & {COND_LABELS[cond]} & {r['mean']:.2f} ({r['se']:.2f}) & {int(r['n_models'])} \\\\")
        lines.append(r"\addlinespace")
    for stratum, order in [("N12", ["3", "4", "6"]), ("N16", ["4", "8"])]:
        for g in order:
            for cond in CONDITIONS:
                row = group_summary[(group_summary.stratum == stratum) & (group_summary.G == g)
                                     & (group_summary.condition == cond) & (group_summary.model == "Pooled")]
                if row.empty:
                    continue
                r = row.iloc[0]
                lines.append(f"Group size & {stratum}, G={g} & {COND_LABELS[cond]} & {r['mean']:.2f} ({r['se']:.2f}) & {int(r['n_models'])} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "structural_summary_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved: structural_summary_table.tex")


def make_contrasts_table(contrasts):
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\caption{Structural robustness contrasts. Community-size rows: the 5 main-paper mechanism "
             r"comparisons at each N (Bonferroni over 5 contrasts). Group-size rows: matched $G$ vs.\ $G{=}4$ "
             r"contrasts per condition (Bonferroni over 5 conditions). "
             r"$^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}",
             r"\label{tab:structural_contrasts}", r"\begin{tabular}{llllc}", r"\toprule",
             r"Analysis & Setting & Scope & Contrast & $\hat{\beta}$ (SE) \\", r"\midrule"]
    for _, r in contrasts.iterrows():
        setting = f"N={r['N']}" if r["analysis"] == "community" else f"{r['stratum']}"
        scope = r.get("scope", r.get("condition", ""))
        cell = f"{r['diff']:.2f} ({r['se']:.2f}){SIG_TEX[r['sig']]}"
        lines.append(f"{r['analysis']} & {setting} & {scope} & {r['contrast']} & {cell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "structural_contrasts.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved: structural_contrasts.tex")


# ═════════════════════════════════════════════════════════════════════════
# 9. Interpretation
# ═════════════════════════════════════════════════════════════════════════

def write_interpretation(comm_df, comm_summary, comm_contrasts, comm_het, comm_het3,
                          group_df, group_summary, group_contrasts, group_het, comm_mech, group_mech):
    def fmt_het(het):
        if het is None:
            return "N/A"
        s, p, d = het
        return f"stat={s:.3f}, df={d}, p={p:.4g} ({'reject' if p < .05 else 'fail to reject'} H0 of no interaction)"

    full_pooled8 = comm_summary[(comm_summary.model == "Pooled8") & (comm_summary.condition == "FULL")]
    coop_trend = "insufficient data"
    if len(full_pooled8) >= 2:
        s = full_pooled8.sort_values("N")
        coop_trend = (f"{s.iloc[0]['mean']:.2f} at N={s.iloc[0]['N']} -> {s.iloc[-1]['mean']:.2f} at N={s.iloc[-1]['N']}")

    sig_by_n = {n: list(comm_contrasts[(comm_contrasts.N == n) & (comm_contrasts.scope == "Pooled8")
                                        & (comm_contrasts.p_bonf < 0.05)]["contrast"])
                for n in ["12", "16", "20"]}

    md = f"""# Structural Robustness — Interpretation

**Scope note:** GPT was not included in either sweep (no community-size or
group-size data exists for it). Community-size robustness draws on 8 model
variants across three scale tiers (7B/13B-14B/70B-72B). Group-size
robustness draws on the 7B trio only (Llama, Mistral, Qwen) — the only
tier with a full group-size sweep.

## Community size

**1-3. Cooperation level, condition rankings, and main contrasts across N=12/16/20**
Pooled (8-model) FULL-condition contribution: {coop_trend}.
Bonferroni-significant pooled contrasts (of the 5 main-paper mechanism comparisons) by N:
"""
    for n in ["12", "16", "20"]:
        sig = sig_by_n[n]
        md += f"- N={n}: {', '.join(sig) if sig else '(none reach significance)'}\n"
    md += f"""
Pooled Condition x N interaction test: {fmt_het(comm_het)}

**4-6. Convergence (contribution / normative-expectation / empirical-expectation)**
See `figure_C2_structural_convergence.pdf` (Community panels, columns 1-2)
and `community_convergence_run_level.csv` / `community_summary.csv`
(metric column) for the late-round (rounds {LATE_ROUNDS[0]}-{LATE_ROUNDS[-1]}) SD trends by N. Lower SD =
stronger convergence; compare the trajectory shape across N=12/16/20 rather
than absolute magnitude alone, since more agents per community does not
mechanically inflate SD, but a larger community may need more rounds to
fully converge.

**7. Social-selection mechanism**
"""
    if not comm_mech.empty:
        for _, r in comm_mech.iterrows():
            direction = "sanctioning" if r["estimate"] < 0 else "unexpected positive"
            sig = "significant" if r["p_value"] < 0.05 else "not significant"
            md += (f"- N={r['N']}: undercontribution -> evaluation b={r['estimate']:.3f} "
                   f"(SE {r['se']:.3f}, p={r['p_value']:.3g}, {sig}) — {direction} direction.\n")
    else:
        md += "- Mechanism model could not be fit (see structural_statistical_results.txt for errors).\n"
    md += f"""
**8-9. Model-tier consistency**
3-way Condition x N x ModelTier joint interaction test: {fmt_het(comm_het3)}
See `community_model_tier_validation.pdf` for the per-tier breakdown — if
the pooled panel (Figure C1, panel A) and the per-tier panels tell visually
different stories, or if the interaction test above rejects, the pooled
result should be described as tier-heterogeneous, not universal.

## Group size (7B trio only)

**1-3. Cooperation, rankings, contrasts**
See `figure_C1_structural_contributions.pdf` panels C/D and
`group_size_summary.csv`. Matched contrasts (G vs. matched G=4 reference)
by condition:
"""
    if not group_contrasts.empty:
        for _, r in group_contrasts.iterrows():
            md += (f"- {r['stratum']} {r['contrast']}, {COND_LABELS.get(r['condition'], r['condition'])}: "
                   f"{r['diff']:.2f} (SE {r['se']:.2f}), p_bonf={r['p_bonf']:.3g} {r['sig']}\n")
    md += """
Note: the clean group-size-only comparison is (N=16,G=8) vs.\\ (N=16,G=4) —
both hold population fixed. G=3/6 vs.\\ G=4 at N=12 is the other clean
comparison (population also fixed at 12 there). Do NOT compare G=8 (N=16)
directly against G=3 or G=6 (N=12) as if group size were the only thing
varying — population size differs too in that comparison.

**4-5. Convergence**
See `figure_C2_structural_convergence.pdf` (Group-size panels, columns 3-4).

**6. Social-learning mechanism**
"""
    if not group_mech.empty:
        for _, r in group_mech.iterrows():
            sig = "significant" if r["p_value"] < 0.05 else "not significant"
            md += (f"- {r['stratum']} G={r['G']}, {r['term']}: b={r['estimate']:.3f} "
                   f"(SE {r['se']:.3f}, p={r['p_value']:.3g}, {sig})\n")
    else:
        md += "- Mechanism model could not be fit for any setting (see structural_statistical_results.txt).\n"
    md += """
**7. Cross-model consistency (Llama/Mistral/Qwen)**
See `group_size_run_level_means.csv` / `group_size_summary.csv` filtered to
individual models (not the Pooled rows) to check whether any one family
drives the pooled pattern.

---
## Suggested manuscript framing
Use the "Behavior -> Convergence -> Mechanism remains robust" narrative
ONLY where the numbers above actually support each step; if any step
diverges (a significant interaction, a sign flip, a mechanism coefficient
losing significance with a materially different point estimate — not just
a p-value crossing 0.05), describe that divergence explicitly rather than
defaulting to a general robustness claim.

---
*Generated by `analyze_structural_robustness.py`. See that script's module
docstring for the full list of reused main-paper components and the
pooled-estimation procedure (model-clustered bootstrap).*
"""
    with open(os.path.join(OUT_DIR, "structural_interpretation.md"), "w") as f:
        f.write(md)
    print("Saved: structural_interpretation.md")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    log_lines = []
    def log(msg):
        log_lines.append(str(msg))
        print(msg)

    log(f"Output directory: {OUT_DIR}\n")
    log("=" * 72 + "\nQC REPORT -- Structural Robustness\n" + "=" * 72)

    comm_df, comm_qc = load_axis(COMMUNITY_SOURCES, "N")
    qc_for_axis(comm_df, comm_qc, COMMUNITY_SOURCES, "N", "COMMUNITY SIZE", log)

    g12_df, g12_qc = load_axis(GROUP_N12_SOURCES, "G")
    g12_df["stratum"] = "N12"
    g16_df, g16_qc = load_axis(GROUP_N16_SOURCES, "G")
    g16_df["stratum"] = "N16"
    group_df = pd.concat([g12_df, g16_df], ignore_index=True)
    group_qc = {"missing_cells": g12_qc["missing_cells"] + g16_qc["missing_cells"],
                "dup_seed_dirs": g12_qc["dup_seed_dirs"] + g16_qc["dup_seed_dirs"]}
    qc_for_axis(g12_df, g12_qc, GROUP_N12_SOURCES, "G", "GROUP SIZE (N=12)", log)
    qc_for_axis(g16_df, g16_qc, GROUP_N16_SOURCES, "G", "GROUP SIZE (N=16)", log)
    log("=" * 72)

    with open(os.path.join(OUT_DIR, "structural_qc_report.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    # ── Save run-level CSVs ──
    contrib_cols = ["N", "model", "tier", "family", "condition", "seed", "mean_contribution", "n_agent_round_obs"]
    comm_df[contrib_cols].to_csv(os.path.join(OUT_DIR, "community_run_level_means.csv"), index=False)
    conv_cols_c = ["N", "model", "tier", "family", "condition", "seed", "late_contrib_sd", "late_in_sd", "late_dn_sd"]
    comm_df[conv_cols_c].to_csv(os.path.join(OUT_DIR, "community_convergence_run_level.csv"), index=False)

    group_contrib_cols = ["stratum", "G", "model", "family", "condition", "seed", "mean_contribution", "n_agent_round_obs"]
    group_df[group_contrib_cols].to_csv(os.path.join(OUT_DIR, "group_size_run_level_means.csv"), index=False)
    group_conv_cols = ["stratum", "G", "model", "family", "condition", "seed", "late_contrib_sd", "late_in_sd", "late_dn_sd"]
    group_df[group_conv_cols].to_csv(os.path.join(OUT_DIR, "group_size_convergence_run_level.csv"), index=False)
    print("Saved: community_run_level_means.csv, community_convergence_run_level.csv, "
          "group_size_run_level_means.csv, group_size_convergence_run_level.csv")

    # ── Summaries (contribution) ──
    comm_pool_specs = [("Pooled8", COMMUNITY_MODELS), ("Pooled7B", SEVENB_MODELS)]
    for tier, models_ in [("7B", SEVENB_MODELS), ("13B/14B", ["llama_13b", "mistral_13b", "qwen_14b"]),
                           ("70B/72B", ["llama_70b", "qwen_72b"])]:
        comm_pool_specs.append((f"PooledTier_{tier}", models_))
    comm_summary = build_summary(comm_df, "N", ["12", "16", "20"], comm_pool_specs, "mean_contribution")

    group_summary_parts = []
    for stratum, gdf_, gvals in [("N12", g12_df, ["3", "4", "6"]), ("N16", g16_df, ["4", "8"])]:
        s = build_summary(gdf_, "G", gvals, [("Pooled", SEVENB_MODELS)], "mean_contribution")
        s["stratum"] = stratum
        group_summary_parts.append(s)
    group_summary = pd.concat(group_summary_parts, ignore_index=True)

    comm_summary.to_csv(os.path.join(OUT_DIR, "community_summary.csv"), index=False)
    group_summary.to_csv(os.path.join(OUT_DIR, "group_size_summary.csv"), index=False)
    print("Saved: community_summary.csv, group_size_summary.csv")

    # ── Convergence summaries (appended, metric-tagged) ──
    conv_metrics = ["late_contrib_sd", "late_in_sd", "late_dn_sd"]
    comm_conv_summary_parts, group_conv_summary_parts = [], []
    for metric in conv_metrics:
        s = build_summary(comm_df, "N", ["12", "16", "20"], comm_pool_specs, metric)
        s["metric"] = metric
        comm_conv_summary_parts.append(s)
        for stratum, gdf_, gvals in [("N12", g12_df, ["3", "4", "6"]), ("N16", g16_df, ["4", "8"])]:
            s2 = build_summary(gdf_, "G", gvals, [("Pooled", SEVENB_MODELS)], metric)
            s2["stratum"], s2["metric"] = stratum, metric
            group_conv_summary_parts.append(s2)
    comm_conv_summary = pd.concat(comm_conv_summary_parts, ignore_index=True)
    group_conv_summary = pd.concat(group_conv_summary_parts, ignore_index=True)

    # ── Figures ──
    make_figure_c1(comm_summary, group_summary)
    make_tier_validation_figure(comm_summary)
    make_figure_c2(comm_conv_summary, group_conv_summary)

    # ── Contrasts + regressions ──
    stats_path = os.path.join(OUT_DIR, "structural_statistical_results.txt")
    stat_log_lines = []
    def slog(msg):
        stat_log_lines.append(str(msg))
        print(msg)

    comm_contrasts = community_condition_contrasts(comm_df, slog)
    comm_res, comm_het, comm_res3, comm_het3 = community_interaction_models(comm_df, slog)
    group_contrasts = group_matched_contrasts(group_df, slog)
    group_results = group_interaction_models(group_df, slog)
    group_het = {k: v[1] for k, v in group_results.items()}

    comm_mech = community_mechanism(slog)
    group_mech = group_mechanism(slog)

    with open(stats_path, "w") as f:
        f.write("\n".join(stat_log_lines) + "\n")
    print("Saved: structural_statistical_results.txt")

    all_contrasts = pd.concat([comm_contrasts, group_contrasts], ignore_index=True, sort=False)
    all_contrasts.to_csv(os.path.join(OUT_DIR, "structural_contrasts.csv"), index=False)
    print("Saved: structural_contrasts.csv")
    make_contrasts_table(all_contrasts)
    make_summary_table(comm_summary, group_summary)

    comm_mech.to_csv(os.path.join(OUT_DIR, "community_mechanism_coefficients.csv"), index=False)
    group_mech.to_csv(os.path.join(OUT_DIR, "group_size_mechanism_coefficients.csv"), index=False)
    print("Saved: community_mechanism_coefficients.csv, group_size_mechanism_coefficients.csv")

    make_figure_c3(comm_mech, group_mech)

    write_interpretation(comm_df, comm_summary, comm_contrasts, comm_het, comm_het3,
                          group_df, group_summary, group_contrasts, group_het, comm_mech, group_mech)

    # ── Final console summary ──
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Community-size models: {COMMUNITY_MODELS}")
    print(f"Group-size models: {SEVENB_MODELS}")
    print(f"Community seeds retained: {sorted(comm_df['seed'].unique())}")
    print(f"Group-size seeds retained: {sorted(group_df['seed'].unique())}")
    print(f"Deduplicated reruns (community): {len(comm_qc['dup_seed_dirs'])}; "
          f"(group N=12): {len(g12_qc['dup_seed_dirs'])}; (group N=16): {len(g16_qc['dup_seed_dirs'])}")
    print(f"Community Condition x N interaction: {comm_het}")
    print(f"Community Condition x N x Tier interaction: {comm_het3}")
    print(f"Group-size Condition x G interactions: {group_het}")
    if not comm_mech.empty:
        print("Community-size social-selection mechanism (undercontribution -> evaluation):")
        print(comm_mech[["N", "estimate", "se", "p_value"]].to_string(index=False))
    if not group_mech.empty:
        print("Group-size social-learning mechanism (InGap/DnGap -> shift):")
        print(group_mech[["stratum", "G", "term", "estimate", "se", "p_value"]].to_string(index=False))
    print(f"\nAll outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
