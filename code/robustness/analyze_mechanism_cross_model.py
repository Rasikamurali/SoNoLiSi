"""
analyze_mechanism_cross_model.py
---------------------------------
Round-1 gap and mixed-effects slopes for prompt_sensitivity results,
run across all models. Tests whether the FULL vs PURE_BASELINE divergence
is mechanism-driven (builds over rounds) or pre-training (present at round 1).
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

BASE       = "/data3/rasimura/social-norm-evo/results/robustness/prompt_sensitivity"
MODELS     = ["openai", "llama", "mistral", "qwen"]
MODEL_LABELS = {"openai": "GPT-4o-mini", "llama": "Llama 3.1-8B",
                "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}
VARIANTS   = ["standard", "verb_only", "anchor_only", "verb_anchor"]
CONDITIONS = ["PURE_BASELINE", "FULL"]
SEEDS      = list(range(43, 48))


def load_log(model, variant, seed, cond):
    pattern = os.path.join(BASE, model, variant, f"seed{seed}", f"log_*_{cond}_seed{seed}.json")
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        if d.get("condition") == cond:
            return d
    return None


def load_agent_data(model):
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_log(model, variant, seed, cond)
                if d is None:
                    continue
                for rlog in d["round_logs"]:
                    for aid_str, c in rlog["contributions"].items():
                        rows.append({
                            "variant": variant, "condition": cond,
                            "seed": seed, "round": rlog["round"],
                            "contribution": int(c),
                        })
    return pd.DataFrame(rows)


# ─── a) Round-1 gap ──────────────────────────────────────────────────────────

def round1_gap(df, model_label):
    r1   = df[df["round"] == 1]
    rows = []
    for variant in VARIANTS:
        for cond in CONDITIONS:
            sub        = r1[(r1["variant"] == variant) & (r1["condition"] == cond)]
            seed_means = sub.groupby("seed")["contribution"].mean()
            rows.append({"variant": variant, "condition": cond,
                         "mean": round(seed_means.mean(), 3),
                         "se":   round(seed_means.sem(), 3)})
    ss = pd.DataFrame(rows)

    gaps = []
    for variant in VARIANTS:
        pb  = ss[(ss["variant"] == variant) & (ss["condition"] == "PURE_BASELINE")].iloc[0]
        fl  = ss[(ss["variant"] == variant) & (ss["condition"] == "FULL")].iloc[0]
        gap    = fl["mean"] - pb["mean"]
        se_gap = np.sqrt(fl["se"]**2 + pb["se"]**2)
        gaps.append({
            "variant": variant,
            "pb_r1": pb["mean"], "full_r1": fl["mean"],
            "gap_r1": round(gap, 3),
            "z": round(gap / se_gap, 2) if se_gap > 0 else np.nan,
        })
    return pd.DataFrame(gaps)


# ─── b) Mixed-effects slopes ─────────────────────────────────────────────────

def mixed_effects_slopes(df):
    df = df.copy()
    df["is_full"] = (df["condition"] == "FULL").astype(int)
    rows = []
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].copy()
        if sub.empty:
            continue
        try:
            mod = smf.mixedlm(
                "contribution ~ round * is_full",
                data=sub, groups=sub["seed"],
            ).fit(reml=True)
            for term, label in [
                ("round",         "PB slope (per round)"),
                ("is_full",       "FULL intercept offset"),
                ("round:is_full", "FULL slope increment"),
            ]:
                coef = mod.params.get(term, np.nan)
                se   = mod.bse.get(term, np.nan)
                pval = mod.pvalues.get(term, np.nan)
                rows.append({
                    "variant": variant, "term": label,
                    "coef": round(coef, 4), "se": round(se, 4),
                    "z":    round(coef / se, 3) if se > 0 else np.nan,
                    "p":    round(pval, 4),
                    "sig":  "***" if pval < 0.001 else ("**" if pval < 0.01 else
                            ("*"   if pval < 0.05  else ("†"  if pval < 0.10 else ""))),
                })
        except Exception as e:
            print(f"    MixedLM failed ({variant}): {e}")
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  {MODEL_LABELS[model]}")
        print(f"{'='*60}")

        df = load_agent_data(model)
        if df.empty:
            print("  No data found.")
            continue
        print(f"  {len(df)} agent-round rows loaded")

        print(f"\n  ── a) Round-1 gap: FULL − Pure Baseline ──")
        r1 = round1_gap(df, MODEL_LABELS[model])
        print(r1.to_string(index=False))

        print(f"\n  ── b) Mixed-effects: round × condition ──")
        me = mixed_effects_slopes(df)
        key_term = "FULL slope increment"
        slope_rows = me[me["term"] == key_term][["variant", "coef", "se", "z", "p", "sig"]]
        print(f"  [FULL slope increment — positive = FULL grows faster than PB]")
        print(slope_rows.to_string(index=False))

        pb_slope_rows = me[me["term"] == "PB slope (per round)"][["variant", "coef", "p"]]
        print(f"\n  [PB slope — should be ~0 if baseline is flat]")
        print(pb_slope_rows.to_string(index=False))
