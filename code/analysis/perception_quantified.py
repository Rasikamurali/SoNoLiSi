"""
perception_quantified.py
------------------------
OLS regression of injunctive (IN) on descriptive (DN) norm perceptions.

Data: one row per (model, condition, seed, round, agent_id) with IN and DN values.
PURE_BASELINE is excluded — perceptions field is empty for that condition.

Analysis: OLS — IN ~ DN × model × condition  (SEs clustered by seed)

Output:
  figures/2026-03-22/cross_model/perception_ols_*.csv
  figures/2026-03-22/<model>/perception_analysis/perception_*.csv  (per-model)
"""

import argparse
import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS    = os.getenv("SNLS_RESULTS_DIR", "/data3/rasimura/social-norm-evo/results")
FIG_ROOT   = "/data3/rasimura/social-norm-evo/figures/2026-03-22"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "global"
SEEDS      = list(range(43, 53))
ALL_SEEDS  = list(range(43, 53))
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


# ─── Analysis: OLS regression IN ~ DN × model × condition ───────────────────

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
        f"C(condition, Treatment('{ref_c}')) + round_c"
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
            f"descriptive_norm * C(condition, Treatment('{ref_c}')) + round_c"
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
    parser = argparse.ArgumentParser(description="Perception OLS + Fisher z + LME analysis.")
    parser.add_argument("--variant", "-v", default="global", choices=["global", "local"],
                        help="Data variant to analyse (default: global).")
    parser.add_argument("--models", "-m", nargs="+",
                        default=["gpt", "llama", "mistral", "qwen"],
                        choices=["gpt", "llama", "mistral", "qwen",
                                 "llama_13b", "mistral_13b", "qwen_14b", "llama_70b", "qwen_72b"],
                        help="Models to include (default: all four).")
    parser.add_argument("--seeds", "-s", nargs="+", type=int, default=None,
                        metavar="SEED", help="Explicit seed list. Overrides --n.")
    parser.add_argument("--n", type=int, default=10, metavar="N",
                        help="Number of seeds from the default list (default: 10).")
    parser.add_argument("--fig-root", "-o", default=None,
                        help="Root figures directory (default: figures/2026-03-22).")
    args = parser.parse_args()

    VARIANT = args.variant
    MODELS  = args.models
    SEEDS   = args.seeds if args.seeds is not None else ALL_SEEDS[:args.n]
    if args.fig_root:
        FIG_ROOT = args.fig_root
    if REFERENCE_MODEL not in MODELS:
        REFERENCE_MODEL = MODELS[0]

    print(f"Variant : {VARIANT}")
    print(f"Models  : {MODELS}")
    print(f"Seeds   : {SEEDS}")
    print(f"Fig root: {FIG_ROOT}")

    print("\nBuilding perception dataframe …")
    df = build_perception_dataframe()
    print(f"  Total observations: {len(df):,}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Conditions: {df['condition'].unique().tolist()}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Rounds: {df['round'].min()}–{df['round'].max()}")
    print(f"  IN range: {df['injunctive_norm'].min():.1f}–{df['injunctive_norm'].max():.1f}")
    print(f"  DN range: {df['descriptive_norm'].min():.1f}–{df['descriptive_norm'].max():.1f}")
    print()
    print(df.groupby(["model", "condition"])["injunctive_norm"].count().unstack())

    run_in_dn_regression(df)

    print("\nAll done.")
