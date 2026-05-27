"""
perception_quantified.py
------------------------
Statistical quantification of injunctive (IN) and descriptive (DN) norm perceptions.

Data: one row per (model, condition, seed, round, agent_id) with IN and DN values.
PURE_BASELINE is excluded — perceptions field is empty for that condition.

Analyses:
  1. Fisher z-tests  — pairwise comparison of r(IN, DN) across all (model × condition) cells
  2. OLS regression  — IN ~ DN × model × condition  (SEs clustered by seed)

Output:
  figures/2026-03-22/cross_model/perception_fisher_z_*.csv
  figures/2026-03-22/cross_model/perception_ols_*.csv
  figures/2026-03-22/<model>/perception_analysis/perception_*.csv  (per-model)
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from itertools import combinations
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS    = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT   = "/data3/rasimura/social-norm-evo/figures/2026-03-22"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "global"
SEEDS      = list(range(43, 53))
# PURE_BASELINE excluded — no perceptions recorded for that condition
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
REFERENCE_CONDITION = "BASELINE"
REFERENCE_MODEL     = "gpt"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_latest_log(model, variant, seed, condition):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def build_perception_dataframe():
    """
    Long-format DataFrame: one row per (model, condition, seed, round, agent_id).
    Columns: injunctive_norm (IN), descriptive_norm (DN).
    Rows with either value missing are dropped.
    """
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_latest_log(model, VARIANT, seed, cond)
                if d is None:
                    continue
                for r in d["round_logs"]:
                    rnd   = r["round"]
                    percs = r.get("perceptions") or {}
                    for aid_str, perc in percs.items():
                        if not perc:
                            continue
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        if inj is None or desc is None:
                            continue
                        rows.append({
                            "model":            model,
                            "condition":        cond,
                            "seed":             seed,
                            "round":            rnd,
                            "agent_id":         str(aid_str),
                            "injunctive_norm":  float(inj),
                            "descriptive_norm": float(desc),
                        })

    df = pd.DataFrame(rows)
    df["round_c"] = df["round"] - df["round"].mean()
    df["model"]     = pd.Categorical(df["model"],     categories=MODELS,      ordered=False)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS,   ordered=False)
    return df


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


def fisher_z(r):
    """Pearson r → Fisher z (arctanh)."""
    r = np.clip(r, -0.9999, 0.9999)
    return np.arctanh(r)


def fisher_z_test(r1, n1, r2, n2):
    """
    Two-sample Fisher z-test comparing two Pearson correlations.
    Returns: z-statistic, two-tailed p-value.
    """
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z  = (z1 - z2) / se
    p  = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


# ─── Analysis 1: Fisher z-tests ───────────────────────────────────────────────

def run_fisher_z(df):
    """
    For every (model, condition) cell, compute r(IN, DN) with 95% CI.
    Then run all pairwise Fisher z-tests (within-model across conditions,
    within-condition across models, and all cross-cell pairs).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 1 — Fisher z-tests on r(IN, DN)")
    print("=" * 70)

    out_dir = os.path.join(FIG_ROOT, "cross_model")

    # ── Step 1: per-cell correlations ────────────────────────────────────────
    cell_rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            sub = df[(df["model"] == model) & (df["condition"] == cond)]
            n   = len(sub)
            if n < 5:
                continue
            r, p_pearson = stats.pearsonr(sub["descriptive_norm"], sub["injunctive_norm"])
            z    = fisher_z(r)
            se_z = 1.0 / np.sqrt(n - 3)
            ci_lo = np.tanh(z - 1.96 * se_z)
            ci_hi = np.tanh(z + 1.96 * se_z)
            cell_rows.append({
                "model": model, "condition": cond,
                "n": n, "r": r, "fisher_z": z,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "p_pearson": p_pearson,
            })

    cells = pd.DataFrame(cell_rows)
    print("\n  Per-cell r(IN, DN):")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(cells.to_string(index=False))
    pd.reset_option("display.float_format")
    save_csv(cells, os.path.join(out_dir, "perception_fisher_z_cells.csv"))

    # ── Step 2: pairwise Fisher z-tests ──────────────────────────────────────
    pair_rows = []
    cell_list = cells.to_dict("records")

    for a, b in combinations(cell_list, 2):
        z_stat, p = fisher_z_test(a["r"], a["n"], b["r"], b["n"])
        pair_rows.append({
            "cell_A":   f"{a['model']}:{a['condition']}",
            "cell_B":   f"{b['model']}:{b['condition']}",
            "r_A":      a["r"],  "r_B": b["r"],
            "delta_r":  a["r"] - b["r"],
            "z_stat":   z_stat,
            "p":        p,
        })

    pairs = pd.DataFrame(pair_rows)
    n_tests = len(pairs)
    pairs["p_bonf"] = (pairs["p"] * n_tests).clip(upper=1.0)
    pairs["sig"]    = pairs["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )

    # Print only significant pairs
    sig = pairs[pairs["p_bonf"] < 0.05].sort_values("p_bonf")
    print(f"\n  Pairwise Fisher z-tests ({n_tests} total, Bonferroni-corrected):")
    print(f"  Significant pairs (p_bonf < 0.05):")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(sig[["cell_A","cell_B","r_A","r_B","delta_r","z_stat","p_bonf","sig"]].to_string(index=False))
    pd.reset_option("display.float_format")
    save_csv(pairs, os.path.join(out_dir, "perception_fisher_z_pairwise.csv"))

    # ── Step 3: within-model across-condition tests ───────────────────────────
    within_model_rows = []
    for model in MODELS:
        mcells = cells[cells["model"] == model].to_dict("records")
        for a, b in combinations(mcells, 2):
            z_stat, p = fisher_z_test(a["r"], a["n"], b["r"], b["n"])
            within_model_rows.append({
                "model": model,
                "cond_A": a["condition"], "r_A": a["r"],
                "cond_B": b["condition"], "r_B": b["r"],
                "delta_r": a["r"] - b["r"],
                "z_stat": z_stat, "p": p,
            })
    wm = pd.DataFrame(within_model_rows)
    n_wm = len(wm)
    wm["p_bonf"] = (wm["p"] * n_wm).clip(upper=1.0)
    wm["sig"]    = wm["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    print(f"\n  Within-model across-condition Fisher z-tests:")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(wm.to_string(index=False))
    pd.reset_option("display.float_format")
    save_csv(wm, os.path.join(out_dir, "perception_fisher_z_within_model.csv"))

    # ── Step 4: within-condition across-model tests ───────────────────────────
    within_cond_rows = []
    for cond in CONDITIONS:
        ccells = cells[cells["condition"] == cond].to_dict("records")
        for a, b in combinations(ccells, 2):
            z_stat, p = fisher_z_test(a["r"], a["n"], b["r"], b["n"])
            within_cond_rows.append({
                "condition": cond,
                "model_A": a["model"], "r_A": a["r"],
                "model_B": b["model"], "r_B": b["r"],
                "delta_r": a["r"] - b["r"],
                "z_stat": z_stat, "p": p,
            })
    wc = pd.DataFrame(within_cond_rows)
    n_wc = len(wc)
    wc["p_bonf"] = (wc["p"] * n_wc).clip(upper=1.0)
    wc["sig"]    = wc["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    print(f"\n  Within-condition across-model Fisher z-tests:")
    pd.set_option("display.float_format", "{:.4f}".format)
    print(wc.to_string(index=False))
    pd.reset_option("display.float_format")
    save_csv(wc, os.path.join(out_dir, "perception_fisher_z_within_condition.csv"))


# ─── Analysis 2: OLS regression IN ~ DN × model × condition ──────────────────

def run_in_dn_regression(df):
    """
    OLS: injunctive_norm ~ descriptive_norm * model * condition
         (SEs clustered by seed)

    Reference: model=gpt, condition=BASELINE.

    Tests:
      - Main effect of DN: baseline IN~DN slope (gpt, BASELINE)
      - DN:model interactions: does the IN~DN slope differ by model?
      - DN:condition interactions: does the slope differ by condition?
      - DN:model:condition 3-way: do model differences in slope vary by condition?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2 — OLS: IN ~ DN × model × condition (clustered by seed)")
    print(f"  Reference: model={REFERENCE_MODEL}, condition={REFERENCE_CONDITION}")
    print("=" * 70)

    ref_m = REFERENCE_MODEL
    ref_c = REFERENCE_CONDITION
    out_dir = os.path.join(FIG_ROOT, "cross_model")

    formula = (
        f"injunctive_norm ~ "
        f"descriptive_norm * "
        f"C(model, Treatment('{ref_m}')) * "
        f"C(condition, Treatment('{ref_c}'))"
    )
    print(f"\n  Formula: {formula}")
    print(f"  N = {len(df):,} observations")

    try:
        result = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["seed"]}
        )
    except Exception as e:
        print(f"  [WARN] fit failed: {e}")
        return

    print(result.summary())

    fe = result.params.reset_index()
    fe.columns = ["term", "coef"]
    fe["se"] = result.bse.values
    fe["z"]  = result.tvalues.values
    fe["p"]  = result.pvalues.values
    fe["sig"] = fe["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    save_csv(fe, os.path.join(out_dir, "perception_ols_IN_DN_full.csv"))

    # ── Per-model fits for interpretable per-model slopes ─────────────────────
    print("\n  Per-model OLS: IN ~ DN * condition  (clustered by seed)")
    per_model_rows = []
    for model in MODELS:
        mdf = df[df["model"] == model].copy()
        formula_m = (
            f"injunctive_norm ~ "
            f"descriptive_norm * C(condition, Treatment('{ref_c}'))"
        )
        try:
            res_m = smf.ols(formula_m, data=mdf).fit(
                cov_type="cluster", cov_kwds={"groups": mdf["seed"]}
            )
        except Exception as e:
            print(f"  [WARN] {model} fit failed: {e}")
            continue

        fe_m = res_m.params.reset_index()
        fe_m.columns = ["term", "coef"]
        fe_m["se"]    = res_m.bse.values
        fe_m["z"]     = res_m.tvalues.values
        fe_m["p"]     = res_m.pvalues.values
        fe_m["model"] = model
        fe_m["sig"]   = fe_m["p"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        )
        per_model_rows.append(fe_m)

        out_m = os.path.join(FIG_ROOT, model, "perception_analysis")
        save_csv(fe_m, os.path.join(out_m, "perception_ols_IN_DN_per_model.csv"))

    if per_model_rows:
        all_per_model = pd.concat(per_model_rows, ignore_index=True)
        save_csv(all_per_model, os.path.join(out_dir, "perception_ols_IN_DN_per_model_all.csv"))

        # Print DN slope summary across models/conditions
        dn_slopes = all_per_model[
            all_per_model["term"].str.startswith("descriptive_norm")
        ][["model","term","coef","se","p","sig"]]
        print("\n  DN slopes (IN ~ DN slope terms):")
        pd.set_option("display.float_format", "{:.4f}".format)
        print(dn_slopes.to_string(index=False))
        pd.reset_option("display.float_format")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building perception dataframe …")
    df = build_perception_dataframe()
    print(f"  Total observations: {len(df):,}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Conditions: {df['condition'].unique().tolist()}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Rounds: {df['round'].min()}–{df['round'].max()}")
    print(f"  IN range: {df['injunctive_norm'].min():.1f}–{df['injunctive_norm'].max():.1f}")
    print(f"  DN range: {df['descriptive_norm'].min():.1f}–{df['descriptive_norm'].max():.1f}")
    print()
    print(df.groupby(["model","condition"])["injunctive_norm"].count().unstack())

    run_fisher_z(df)
    run_in_dn_regression(df)

    print("\nAll done.")
