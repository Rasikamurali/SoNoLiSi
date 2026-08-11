"""
perception_dn_in.py
-------------------
Reverse perception regression: DN ~ IN × condition + round_c, per model.

Mirrors the IN~DN analysis in perception_quantified.py but flips the
outcome and predictor to test whether injunctive norms predict how agents
perceive group behavior (motivated perception / projection).

Saves per-tier CSVs alongside existing IN~DN outputs:
  figures/2026-03-22/perception/7b/cross_model/perception_ols_DN_IN_per_model_all.csv
  figures/2026-03-22/perception/13b/cross_model/perception_ols_DN_IN_per_model_all.csv
  figures/2026-03-22/perception/70b/cross_model/perception_ols_DN_IN_per_model_all.csv
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE     = "/data3/rasimura/social-norm-evo"
FIG_ROOT = f"{BASE}/figures/2026-03-22/perception"

TIER_SPECS = {
    "7b": [
        ("gpt",        "GPT",         f"{BASE}/results",      "local", list(range(43, 53))),
        ("llama",      "Llama-7B",    f"{BASE}/results",      "local", list(range(43, 53))),
        ("mistral",    "Mistral-7B",  f"{BASE}/results",      "local", list(range(43, 53))),
        ("qwen",       "Qwen-7B",     f"{BASE}/results",      "local", list(range(43, 53))),
    ],
    "13b": [
        ("llama_13b",  "Llama-13B",   f"{BASE}/results",      "local", list(range(42, 52))),
        ("mistral_13b","Mistral-13B", f"{BASE}/results",      "local", list(range(42, 52))),
        ("qwen_14b",   "Qwen-14B",    f"{BASE}/results",      "local", list(range(42, 52))),
    ],
    "70b": [
        ("llama_70b",  "Llama-70B",   f"{BASE}/code/results", "local", list(range(42, 52))),
        ("qwen_72b",   "Qwen-72B",    f"{BASE}/code/results", "local", list(range(42, 52))),
    ],
}

CONDITIONS       = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
REFERENCE_COND   = "BASELINE"


def load_tier(specs):
    rows = []
    for model_key, family, results_dir, variant, seeds in specs:
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
                for r in d["round_logs"]:
                    percs = r.get("perceptions") or {}
                    for aid_str, perc in percs.items():
                        if not perc:
                            continue
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        if inj is None or desc is None:
                            continue
                        rows.append({
                            "model":            model_key,
                            "condition":        cond,
                            "seed":             seed,
                            "round":            r["round"],
                            "injunctive_norm":  float(inj),
                            "descriptive_norm": float(desc),
                        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["round_c"] = df["round"] - df["round"].mean()
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=False)
    return df


def run_per_model(df, formula_tpl, ref_c):
    rows = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model].copy()
        formula = formula_tpl.format(ref_c=ref_c)
        try:
            res = smf.ols(formula, data=mdf).fit(
                cov_type="cluster", cov_kwds={"groups": mdf["seed"]}
            )
        except Exception as e:
            print(f"  [WARN] {model} fit failed: {e}")
            continue
        fe = res.params.reset_index()
        fe.columns = ["term", "coef"]
        fe["se"]    = res.bse.values
        fe["z"]     = res.tvalues.values
        fe["p"]     = res.pvalues.values
        fe["model"] = model
        rows.append(fe)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    dn_in_formula = (
        "descriptive_norm ~ "
        "injunctive_norm * C(condition, Treatment('{ref_c}')) + round_c"
    )

    for tier, specs in TIER_SPECS.items():
        print(f"\n── Tier {tier} ──")
        df = load_tier(specs)
        if df.empty:
            print("  No data found, skipping.")
            continue
        print(f"  {len(df):,} observations, models: {sorted(df['model'].unique())}")

        result = run_per_model(df, dn_in_formula, REFERENCE_COND)
        if result.empty:
            print("  No results, skipping.")
            continue

        out_dir = os.path.join(FIG_ROOT, tier, "cross_model")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "perception_ols_DN_IN_per_model_all.csv")
        result.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        # Print IN sensitivity summary
        in_slopes = result[result["term"].str.startswith("injunctive_norm")]
        print(in_slopes[["model", "term", "coef", "se", "p"]].to_string(index=False))

    print("\nDone.")
