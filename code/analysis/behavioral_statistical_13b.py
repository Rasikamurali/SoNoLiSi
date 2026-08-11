"""
behavioral_statistical_13b.py
------------------------------
Same five behavioral statistical tests as behavioral_statistical.py,
applied to the 13B/14B model family:
  Llama-13B, Mistral-13B, Qwen-14B

Data source: results/{model}/local/seed*/log_*.json  (N=12, G=4, MCPR=0.4)
Outputs  →  figures/statistical_tests/behavioral_13b/
"""

import json
import glob
import os
import sys
import numpy as np
import pandas as pd

# Import all test logic and LaTeX writers from the 7B module.
# We override its module-level globals to redirect to 13B families.
sys.path.insert(0, os.path.dirname(__file__))
import behavioral_statistical as _bs

# ── 13B config ─────────────────────────────────────────────────────────────────

MODELS_13B = ["llama_13b", "mistral_13b", "qwen_14b"]
LABELS_13B = {
    "llama_13b":  "Llama-13B",
    "mistral_13b": "Mistral-13B",
    "qwen_14b":   "Qwen-14B",
}

RESULTS_13B = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT_13B = "/data3/rasimura/social-norm-evo/figures/statistical_tests/behavioral_13b"

# Override the module-level globals that test functions reference
_bs.FAMILIES = ["Llama-13B", "Mistral-13B", "Qwen-14B"]
_bs.REF_FAM  = "Mistral-13B"


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_logs():
    best = {}
    for model in MODELS_13B:
        pattern = os.path.join(RESULTS_13B, model, "local", "seed*", "log_*.json")
        for path in sorted(glob.glob(pattern)):
            try:
                d    = json.load(open(path))
                seed = int(path.split("seed")[1].split("/")[0])
                best[(model, seed, d["condition"])] = (seed, d)
            except Exception:
                continue
    return best


def build_dataframe() -> pd.DataFrame:
    rows = []
    for (model, _, _), (s, d) in _load_logs().items():
        for r in d["round_logs"]:
            rows.append({
                "family":    LABELS_13B[model],
                "seed":      s,
                "condition": d["condition"],
                "round":     r["round"],
                "contrib":   float(np.mean(list(r["contributions"].values()))),
                "run_id":    f"{LABELS_13B[model]}_s{s}_{d['condition']}",
            })
    df = pd.DataFrame(rows)
    df["round_c"]  = df["round"] - _bs.ROUND_MEAN
    df["family_f"] = pd.Categorical(df["family"],    _bs.FAMILIES)
    df["cond_f"]   = pd.Categorical(df["condition"], _bs.CONDITIONS)
    return df


def build_agent_dataframe() -> pd.DataFrame:
    rows = []
    for (model, _, _), (s, d) in _load_logs().items():
        for r in d["round_logs"]:
            for aid, contrib in r["contributions"].items():
                rows.append({
                    "family":    LABELS_13B[model],
                    "seed":      s,
                    "condition": d["condition"],
                    "round":     r["round"],
                    "agent_id":  int(aid),
                    "contrib":   float(contrib),
                    "run_id":    f"{LABELS_13B[model]}_s{s}_{d['condition']}",
                })
    df = pd.DataFrame(rows)
    df["round_c"]  = df["round"] - _bs.ROUND_MEAN
    df["family_f"] = pd.Categorical(df["family"],    _bs.FAMILIES)
    df["cond_f"]   = pd.Categorical(df["condition"], _bs.CONDITIONS)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_ROOT_13B, exist_ok=True)

    print("=" * 70)
    print("BEHAVIORAL STATISTICAL TESTS — 13B/14B MODELS")
    print("Llama-13B | Mistral-13B | Qwen-14B")
    print("=" * 70)

    print("\nLoading data...")
    df        = build_dataframe()
    df_agents = build_agent_dataframe()
    print(f"  Run-level:   {len(df):,} rows, {df['run_id'].nunique()} runs")
    print(f"  Agent-level: {len(df_agents):,} rows")
    print(df.groupby(["family", "condition"])["run_id"].nunique().unstack().to_string())

    # ── Test 1: Level contrasts ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1 — LEVEL ANALYSIS")
    print("=" * 70)
    means_df, level_cdf, level_summ = _bs.test_level_contrasts(df)
    for fam, s in level_summ:
        print(f"  {fam}: {s}")
    sig1 = level_cdf[level_cdf["p_holm"] < .05]
    print(f"  Significant (p<.05): {len(sig1)}/{len(level_cdf)}")
    if len(sig1):
        print(sig1[["family", "contrast", "estimate", "z", "p_holm", "sig"]]
              .to_string(index=False))

    means_df.to_csv(os.path.join(FIG_ROOT_13B, "level_condition_means.csv"), index=False)
    level_cdf.to_csv(os.path.join(FIG_ROOT_13B, "level_contrasts.csv"), index=False)
    _bs.write_level_tex(means_df, level_cdf, os.path.join(FIG_ROOT_13B, "level_contrasts.tex"))

    # ── Test 2: Slope contrasts ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2 — SLOPE ANALYSIS")
    print("=" * 70)
    slope_df, slope_cdf, slope_summ = _bs.test_slope_contrasts(df)
    for fam, s in slope_summ:
        print(f"  {fam}: {s}")
    sig2 = slope_cdf[slope_cdf["p_holm"] < .05]
    print(f"  Significant (p<.05): {len(sig2)}/{len(slope_cdf)}")
    if len(sig2):
        print(sig2[["family", "contrast", "estimate", "z", "p_holm", "sig"]]
              .to_string(index=False))

    slope_df.to_csv(os.path.join(FIG_ROOT_13B, "slope_condition_estimates.csv"), index=False)
    slope_cdf.to_csv(os.path.join(FIG_ROOT_13B, "slope_contrasts.csv"), index=False)
    _bs.write_slope_tex(slope_df, slope_cdf, os.path.join(FIG_ROOT_13B, "slope_contrasts.tex"))

    # ── Test 3: Omnibus LRT ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3 — CROSS-MODEL CONSISTENCY (OMNIBUS LRT)")
    print("=" * 70)
    omni_df, omni_summ = _bs.test_omnibus(df)
    for s in omni_summ:
        print(f"  {s}")
    print(omni_df[["test", "model", "df", "chi2", "p", "sig"]].to_string(index=False))

    omni_df.to_csv(os.path.join(FIG_ROOT_13B, "omnibus_interactions.csv"), index=False)
    _bs.write_omnibus_tex(omni_df, os.path.join(FIG_ROOT_13B, "omnibus_interactions.tex"))

    # ── Test 4: Moderation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4 — INITIAL-COOPERATION MODERATION")
    print("=" * 70)
    mod_df, mod_summ = _bs.test_moderation(df_agents)
    for fam, s in mod_summ:
        print(f"  {fam}: {s}")
    if len(mod_df):
        sig4  = mod_df[mod_df["p_holm"] < .05]
        marg4 = mod_df[(mod_df["p_holm"] >= .05) & (mod_df["p_holm"] < .10)]
        print(f"  Significant (p<.05): {len(sig4)}/{len(mod_df)}")
        print(f"  Marginal   (p<.10): {len(marg4)}/{len(mod_df)}")

    mod_df.to_csv(os.path.join(FIG_ROOT_13B, "initial_level_moderation.csv"), index=False)
    _bs.write_moderation_tex(mod_df, os.path.join(FIG_ROOT_13B, "initial_level_moderation.tex"))

    # ── Test 5: Dispersion ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5 — DISPERSION (CONVERGENCE CHECK)")
    print("=" * 70)
    disp_df = _bs.test_dispersion(df_agents)
    pivot = disp_df.pivot(index="condition", columns="family",
                          values="mean").reindex(_bs.CONDITIONS)[_bs.FAMILIES]
    print(pivot.round(3).to_string())

    disp_df.to_csv(os.path.join(FIG_ROOT_13B, "dispersion.csv"), index=False)
    _bs.write_dispersion_tex(disp_df, os.path.join(FIG_ROOT_13B, "dispersion.tex"))

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {FIG_ROOT_13B}")


if __name__ == "__main__":
    main()
