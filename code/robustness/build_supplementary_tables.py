"""
build_supplementary_tables.py
------------------------------
Assembles the four supplementary robustness tables (S1-S4) from the
EXISTING robustness analysis scripts. Does not reimplement their
statistics -- it imports analyze_temperature_sweep.py,
analyze_prompt_variants.py, analyze_prompt_sensitivity.py, and
analyze_mechanism_vs_pretraining.py as modules and calls their own
load_*/steady_state_*/ols_*/treatment_gap/round1_gap/mixed_effects_slopes
functions directly, monkey-patching each module's RESULTS_DIR global to
loop over all four backends before calling in -- this is the SAME
monkey-patch-then-call pattern already used elsewhere in this repo for
model-tier orchestration (see code/analysis/cross-model/run_*_analysis.py
and CODEBASE_INVENTORY.md section 0). Those four scripts were hardcoded
to RESULTS_DIR=".../openai" only, so this is the minimal way to get
per-model-family statistics out of them without duplicating their logic.

For temperature_sweep / prompt_variants / prompt_sensitivity, model-level
steady-state + gap statistics were already available from
analyze_cross_model.py's saved CSVs (figures/robustness/cross_model/*.csv)
-- those are read directly rather than recomputed. What was NOT already
available per-model and required the monkey-patch call-in:
  - temperature_sweep: full OLS stats (SE, t, CI) per model (ts_ols.csv only
    had coef + p)
  - prompt_sensitivity: round-1 gap and the mixed-effects round:is_full
    slope term per model (analyze_mechanism_vs_pretraining.py had no
    cross-model counterpart at all)

Table S4 (parameter_sweep) is read from analyze_parameter_sweep.py's own
output (figures/robustness/parameter_sweep/param_sweep_master.csv), a new
script written for this task since none existed.

Output:
  figures/appendix and tables/table_{temperature_robustness,prompt_variants,
    prompt_sensitivity,parameter_sweep}.tex
  figures/appendix and tables/{temperature_robustness_table,
    prompt_variants_table,prompt_sensitivity_table,parameter_sweep_table}.csv
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

BASE        = "/data3/rasimura/social-norm-evo"
ROBUST_DIR  = f"{BASE}/code/robustness"
CROSS_DIR   = f"{BASE}/figures/robustness/cross_model"
TS_FIG_DIR  = f"{BASE}/figures/robustness/temperature_sweep"
PV_FIG_DIR  = f"{BASE}/figures/robustness/prompt_variants"
PS_FIG_DIR  = f"{BASE}/figures/robustness/prompt_sensitivity"
PARAM_DIR   = f"{BASE}/figures/robustness/parameter_sweep"
OUT_DIR     = f"{BASE}/figures/appendix and tables"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROBUST_DIR)

MODELS       = ["openai", "llama", "mistral", "qwen"]
MODEL_LABELS = {"openai": "GPT-4o-mini", "llama": "Llama 3.1-8B",
                "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}


def z_to_p(z):
    return 2 * (1 - norm.cdf(abs(z))) if pd.notna(z) else np.nan


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S1 — Temperature robustness
# ═══════════════════════════════════════════════════════════════════════════

def build_table_s1():
    import analyze_temperature_sweep as ts

    ss_rows, ols_rows = [], []
    for model in MODELS:
        ts.RESULTS_DIR = f"{BASE}/results/robustness/temperature_sweep/{model}"
        df       = ts.load_data()
        df_agent = ts.load_agent_level()
        if df.empty:
            print(f"  [S1] no data for {model}")
            continue

        ss = ts.steady_state_summary(df)               # condition, temperature, mean, se, sd, n_seeds
        ss["model"] = model
        ss_rows.append(ss)

        ols = ts.ols_temp_effect(df_agent)              # condition, coef_temp, se, t, p, ci_lo, ci_hi, sig
        ols["model"] = model
        ols_rows.append(ols)

    ss_all  = pd.concat(ss_rows, ignore_index=True)
    ols_all = pd.concat(ols_rows, ignore_index=True)

    # CV across the 4 temperature-specific steady-state means, per (model, condition)
    cv_rows = []
    for model in MODELS:
        for cond in ["PURE_BASELINE", "BASELINE"]:
            sub = ss_all[(ss_all["model"] == model) & (ss_all["condition"] == cond)]
            means = sub.sort_values("temperature")["mean"].to_numpy()
            cv = float(np.std(means, ddof=1) / np.mean(means)) * 100 if np.mean(means) != 0 else np.nan
            cv_rows.append({"model": model, "condition": cond, "cv_pct": cv})
    cv_df = pd.DataFrame(cv_rows)

    ss_all.to_csv(os.path.join(OUT_DIR, "s1_steady_state_by_model.csv"), index=False)
    ols_all.to_csv(os.path.join(OUT_DIR, "s1_ols_by_model.csv"), index=False)
    cv_df.to_csv(os.path.join(OUT_DIR, "s1_cv_by_model.csv"), index=False)

    # Consistency check against the pre-existing pooled (openai-only) result
    pooled = pd.read_csv(os.path.join(TS_FIG_DIR, "temp_ols_results.csv"))
    recomputed_openai = ols_all[ols_all["model"] == "openai"][["condition", "coef_temp", "p"]]
    print("  [S1] pooled temp_ols_results.csv (openai, pre-existing):")
    print(pooled[["condition", "coef_temp", "p"]].to_string(index=False))
    print("  [S1] recomputed openai row (via monkey-patched call-in):")
    print(recomputed_openai.to_string(index=False))

    return ss_all, ols_all, cv_df


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S2 — Prompt variants (structural)
# ═══════════════════════════════════════════════════════════════════════════

def build_table_s2():
    ss  = pd.read_csv(os.path.join(CROSS_DIR, "pv_steady_state.csv"))   # model, variant, condition, mean, se
    gap = pd.read_csv(os.path.join(CROSS_DIR, "pv_gaps.csv"))           # model, variant, pb_mean, full_mean, gap, se_gap, z
    gap["p"]     = gap["z"].apply(z_to_p)
    gap["ci_lo"] = gap["gap"] - 1.96 * gap["se_gap"]
    gap["ci_hi"] = gap["gap"] + 1.96 * gap["se_gap"]

    # Pooled (GPT-4o-mini-only) panel, as directly produced by analyze_prompt_variants.py
    pooled = pd.read_csv(os.path.join(PV_FIG_DIR, "pv_treatment_gap.csv"))
    pooled["p"] = pooled["z"].apply(z_to_p)

    ss.to_csv(os.path.join(OUT_DIR, "s2_steady_state_by_model.csv"), index=False)
    gap.to_csv(os.path.join(OUT_DIR, "s2_gap_by_model.csv"), index=False)

    print("  [S2] existing openai-only pv_treatment_gap.csv:")
    print(pooled.to_string(index=False))
    print("  [S2] openai row within per-model gap.csv (should match exactly):")
    print(gap[gap["model"] == "openai"].to_string(index=False))

    return ss, gap, pooled


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S3 — Word-level prompt sensitivity + mechanism-vs-pretraining
# ═══════════════════════════════════════════════════════════════════════════

def build_table_s3():
    import analyze_mechanism_vs_pretraining as mech

    ss  = pd.read_csv(os.path.join(CROSS_DIR, "ps_steady_state.csv"))
    gap = pd.read_csv(os.path.join(CROSS_DIR, "ps_gaps.csv"))
    gap["p"]     = gap["z"].apply(z_to_p)
    gap["ci_lo"] = gap["gap"] - 1.96 * gap["se_gap"]
    gap["ci_hi"] = gap["gap"] + 1.96 * gap["se_gap"]

    r1_rows, me_rows = [], []
    for model in MODELS:
        mech.RESULTS_DIR = f"{BASE}/results/robustness/prompt_sensitivity/{model}"
        df_agent = mech.load_agent_level()
        df_round = mech.load_round_means()
        if df_agent.empty:
            print(f"  [S3] no data for {model}")
            continue
        r1 = mech.round1_gap(df_agent)                # variant, pb_r1, full_r1, gap_r1, se, z
        r1["model"] = model
        r1_rows.append(r1)

        me = mech.mixed_effects_slopes(df_agent)       # variant, term, coef, se, z, p, sig
        me["model"] = model
        me_rows.append(me)

    r1_all = pd.concat(r1_rows, ignore_index=True)
    me_all = pd.concat(me_rows, ignore_index=True)
    r1_all["p"] = r1_all["z"].apply(z_to_p)

    # Isolate the round:is_full ("FULL slope increment") interaction term --
    # this is the MixedLM-based dynamic quantity analyze_mechanism_vs_pretraining.py
    # itself treats as the key test of mechanism-driven (vs. pretraining) divergence.
    slope_inc = me_all[me_all["term"] == "FULL slope increment"].rename(
        columns={"coef": "full_round_beta", "se": "full_round_se", "p": "full_round_p"}
    )[["model", "variant", "full_round_beta", "full_round_se", "full_round_p"]]

    ss.to_csv(os.path.join(OUT_DIR, "s3_steady_state_by_model.csv"), index=False)
    gap.to_csv(os.path.join(OUT_DIR, "s3_gap_by_model.csv"), index=False)
    r1_all.to_csv(os.path.join(OUT_DIR, "s3_round1_gap_by_model.csv"), index=False)
    me_all.to_csv(os.path.join(OUT_DIR, "s3_mixed_effects_by_model.csv"), index=False)

    pooled_r1 = pd.read_csv(os.path.join(PS_FIG_DIR, "mech_round1_gap.csv"))
    print("  [S3] existing openai-only mech_round1_gap.csv:")
    print(pooled_r1.to_string(index=False))
    print("  [S3] openai row within per-model round1_gap (should match exactly):")
    print(r1_all[r1_all["model"] == "openai"].to_string(index=False))

    return ss, gap, r1_all, slope_inc


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S4 — Parameter sweep (already computed by analyze_parameter_sweep.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_table_s4():
    master = pd.read_csv(os.path.join(PARAM_DIR, "param_sweep_master.csv"))
    return master


if __name__ == "__main__":
    print("=" * 70)
    print("TABLE S1 — Temperature")
    print("=" * 70)
    s1_ss, s1_ols, s1_cv = build_table_s1()

    print("\n" + "=" * 70)
    print("TABLE S2 — Prompt variants")
    print("=" * 70)
    s2_ss, s2_gap, s2_pooled = build_table_s2()

    print("\n" + "=" * 70)
    print("TABLE S3 — Prompt sensitivity")
    print("=" * 70)
    s3_ss, s3_gap, s3_r1, s3_slope = build_table_s3()

    print("\n" + "=" * 70)
    print("TABLE S4 — Parameter sweep")
    print("=" * 70)
    s4 = build_table_s4()
    print(s4.to_string(index=False))

    print(f"\nIntermediate CSVs -> {OUT_DIR}")
