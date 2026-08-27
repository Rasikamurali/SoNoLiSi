"""
behavior_quantified.py
----------------------
Q1 (within-model): OLS + Wald test on the paper's 5 main mechanism
condition pairs (see PAPER_PAIRS), round as covariate, seed-clustered SEs,
Bonferroni over 5. Q2 (per-condition): same, over all 6 model pairs.
Sign convention: "A -> B" reports coef(B)-coef(A) (positive = B higher).

Inputs: results/{model}/local/seed{N}/log_*.json (fallback:
code/results/{model}/local/additional_runs/...).
Outputs (figures/2026-03-22/paper_stats/): q1/q2_all_rounds*.csv +
q1/q2_contribution*.tex.
"""

import argparse
import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from itertools import combinations
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/behavioral/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS   = os.getenv("SNLS_RESULTS_DIR", f"{BASE}/results")
OUT_DIR   = f"{BASE}/figures/2026-03-22/paper_stats"
# Fallback base for seeds not found under RESULTS (e.g. seeds 53-92 live under
# code/results/{model}/{variant}/additional_runs/seed{s}/, not results/{model}/{variant}/seed{s}/).
ADDITIONAL_RUNS_RESULTS = f"{BASE}/code/results"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "local"
SEEDS      = list(range(43, 53))
ALL_SEEDS  = list(range(43, 53))
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
REF_COND   = "PURE_BASELINE"
REF_MODEL  = "gpt"

PAPER_PAIRS = [
    ("PURE_BASELINE", "BASELINE"),
    ("BASELINE",      "NO_DISCUSSION"),
    ("BASELINE",      "NO_SELECTION"),
    ("NO_SELECTION",  "FULL"),
    ("NO_DISCUSSION", "FULL"),
]

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_latest_log(model, seed, condition):
    # Finds the raw log file for one (model, seed, condition) combination.
    # Looks in the main results folder first, then in the "additional_runs"
    # fallback folder (used for extra seeds added later). If more than one
    # log file matches, the most recently written one wins.
    patterns = [
        os.path.join(RESULTS, model, VARIANT, f"seed{seed}", "log_*.json"),
        os.path.join(ADDITIONAL_RUNS_RESULTS, model, VARIANT, "additional_runs",
                     f"seed{seed}", "log_*.json"),
    ]
    for pattern in patterns:
        best = {}
        for p in sorted(glob.glob(pattern)):
            try:
                with open(p) as f:
                    d = json.load(f)
                best[d["condition"]] = d
            except Exception:
                continue
        if condition in best:
            return best[condition]
    return None


def build_dataframe():
    """Long-format DataFrame: one row per (model, condition, seed, round, agent)."""
    # Loads every model/seed/condition's log file and flattens each agent's
    # contribution in each round into its own row.
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                for r in d["round_logs"]:
                    rnd = r["round"]
                    for agent_id, contrib in r["contributions"].items():
                        rows.append({
                            "model":        model,
                            "condition":    cond,
                            "seed":         seed,
                            "round":        rnd,
                            "contribution": float(contrib),
                        })
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=False)
    df["model"]     = pd.Categorical(df["model"],     categories=MODELS,     ordered=False)
    return df


# ─── OLS helper ───────────────────────────────────────────────────────────────

def fit_ols(df, formula, cluster_col="seed"):
    """OLS with HC-robust SEs clustered at cluster_col level."""
    # Fits the regression and returns None (with a warning printed) instead
    # of raising, so one failed fit doesn't stop the whole script.
    try:
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
    except Exception as e:
        print(f"  [WARN] OLS failed: {e}")
        return None


def wald_contrast(result, key_a, key_b):
    """Wald test for coef[key_a]-coef[key_b] (either may be None -> 0).
    Returns (diff, se, z, p); call sites pass (key_B, key_A) per the
    module's sign convention."""
    params = result.params
    cov    = result.cov_params()

    # Small helpers: a coefficient/variance/covariance is treated as 0 if
    # its key is missing, which happens for the reference condition/model
    # (statsmodels doesn't give the reference level its own coefficient,
    # since by construction it's the baseline everything else is compared to).
    def _c(k):  return params[k]         if k and k in params.index else 0.0
    def _v(k):  return cov.loc[k, k]     if k and k in cov.index   else 0.0
    def _cv(a, b):
        return cov.loc[a, b] if a and b and a in cov.index and b in cov.columns else 0.0

    # Standard formula for the variance of a difference of two (possibly
    # correlated) estimates: Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B).
    diff = _c(key_a) - _c(key_b)
    se   = np.sqrt(_v(key_a) + _v(key_b) - 2 * _cv(key_a, key_b))
    z    = diff / se if se > 0 else np.nan
    p    = 2 * (1 - _norm.cdf(abs(z)))
    return diff, se, z, p


def cond_param(cond, ref=None):
    """Statsmodels Treatment-contrast parameter name for a condition level."""
    # Builds the exact coefficient name statsmodels assigns to a condition
    # dummy variable (or returns None if this condition IS the reference
    # level, which has no dummy of its own).
    if ref is None: ref = REF_COND
    return None if cond == ref else f"C(condition, Treatment('{ref}'))[T.{cond}]"


def model_param(mdl, ref=None):
    """Statsmodels Treatment-contrast parameter name for a model level."""
    # Same idea as cond_param(), but for the model-family dummy variable.
    if ref is None: ref = REF_MODEL
    return None if mdl == ref else f"C(model, Treatment('{ref}'))[T.{mdl}]"


# ─── Q1: Within-model condition contrasts ─────────────────────────────────────

def run_q1(df):
    """Per model: contribution ~ C(condition) + round, seed-clustered SEs.
    One row per (model, pair)."""
    formula = "contribution ~ C(condition, Treatment('{ref}')) + round".format(ref=REF_COND)

    # Fit one regression per model (not pooled across models), then read
    # off each of the 5 paper contrasts from that one fitted model.
    rows = []
    for model in MODELS:
        mdf = df[df["model"] == model].copy()
        result = fit_ols(mdf, formula)
        if result is None:
            continue
        for cA, cB in PAPER_PAIRS:
            diff, se, z, p = wald_contrast(result, cond_param(cB), cond_param(cA))
            rows.append({"model": model, "cond_A": cA, "cond_B": cB,
                         "diff": diff, "se": se, "z": z, "p": p})

    # Bonferroni-correct across the 5 contrasts tested within each model.
    out = pd.DataFrame(rows)
    n   = len(PAPER_PAIRS)
    out["p_bonf"] = (out["p"] * n).clip(upper=1.0)
    out["sig"]    = out["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    return out


# ─── Q2: Cross-model comparisons per condition ────────────────────────────────

def run_q2(df):
    """Per condition: contribution ~ C(model) + round, seed-clustered SEs.
    One row per (condition, model pair)."""
    formula = "contribution ~ C(model, Treatment('{ref}')) + round".format(ref=REF_MODEL)

    # Fit one regression per condition (not pooled across conditions), then
    # read off every pairwise model comparison from that one fitted model.
    model_pairs = list(combinations(MODELS, 2))
    rows = []
    for cond in CONDITIONS:
        cdf = df[df["condition"] == cond].copy()
        result = fit_ols(cdf, formula)
        if result is None:
            continue
        for mA, mB in model_pairs:
            diff, se, z, p = wald_contrast(result, model_param(mB), model_param(mA))
            rows.append({"condition": cond, "model_A": mA, "model_B": mB,
                         "diff": diff, "se": se, "z": z, "p": p})

    # Bonferroni-correct across all model pairs tested within each condition.
    out = pd.DataFrame(rows)
    n   = len(model_pairs)
    out["p_bonf"] = (out["p"] * n).clip(upper=1.0)
    out["sig"]    = out["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    return out


# ─── LaTeX table generation ───────────────────────────────────────────────────

COND_PAIR_LABELS = {
    ("PURE_BASELINE", "BASELINE"):      r"Pure Baseline $\to$ Baseline",
    ("BASELINE",      "NO_DISCUSSION"): r"Baseline $\to$ No Discussion",
    ("BASELINE",      "NO_SELECTION"):  r"Baseline $\to$ No Selection",
    ("NO_SELECTION",  "FULL"):          r"No Selection $\to$ Full",
    ("NO_DISCUSSION", "FULL"):          r"No Discussion $\to$ Full",
}
MODEL_PAIR_LABELS = {
    ("gpt",    "llama"):   r"GPT vs.\ Llama",
    ("gpt",    "mistral"): r"GPT vs.\ Mistral",
    ("gpt",    "qwen"):    r"GPT vs.\ Qwen",
    ("llama",  "mistral"): r"Llama vs.\ Mistral",
    ("llama",  "qwen"):    r"Llama vs.\ Qwen",
    ("mistral","qwen"):    r"Mistral vs.\ Qwen",
}
COND_TEX = {
    "PURE_BASELINE": "Pure Baseline",
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
MODEL_TEX = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen",
             "llama_13b": "Llama-13B", "mistral_13b": "Mistral-13B", "qwen_14b": "Qwen-14B",
             "llama_70b": "Llama-70B", "qwen_72b": "Qwen-72B", "gpt-5-mini": "GPT-5-mini"}
SIG_TEX = {"***": r"$^{***}$", "**": r"$^{**}$", "*": r"$^{*}$", "": ""}

CAPTION_NOTE = (r"Cells show $\hat{\beta}$ (SE) for the second-listed condition/model "
                r"minus the first (positive = second is higher). "
                r"Bonferroni-corrected: $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.")


def _cell(diff, se, sig):
    # Formats one table cell as "estimate (standard error)" plus significance stars.
    return f"{diff:.2f} ({se:.2f}){SIG_TEX[sig]}"


def _save_tex(lines, path):
    # Writes a list of LaTeX source lines to a file, creating the folder if needed.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


def make_q1_latex(q1_df, suffix=""):
    """Q1 LaTeX table: rows = 5 condition pairs, cols = models."""
    col_spec = "l" + "r" * len(MODELS)
    header   = " & ".join(MODEL_TEX[m] for m in MODELS)
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Condition Comparison & {header} \\",
        r"\midrule",
    ]
    for cA, cB in PAPER_PAIRS:
        cells = []
        for mdl in MODELS:
            r = q1_df[(q1_df["model"] == mdl) & (q1_df["cond_A"] == cA) & (q1_df["cond_B"] == cB)]
            cells.append(_cell(r.iloc[0]["diff"], r.iloc[0]["se"], r.iloc[0]["sig"])
                         if not r.empty else "--")
        lines.append(rf"{COND_PAIR_LABELS[(cA, cB)]} & {' & '.join(cells)} \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Within-model condition contrasts (Contribution). {CAPTION_NOTE}}}",
        rf"\label{{tab:q1_contribution{suffix}}}",
        r"\end{table}",
    ]
    _save_tex(lines, os.path.join(OUT_DIR, f"q1_contribution{suffix}.tex"))


def make_q2_latex(q2_df, suffix=""):
    """Q2 LaTeX table: rows = 6 model pairs, cols = 5 conditions."""
    model_pairs = list(combinations(MODELS, 2))
    col_spec = "l" + "r" * len(CONDITIONS)
    header   = " & ".join(COND_TEX[c] for c in CONDITIONS)
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Model Comparison & {header} \\",
        r"\midrule",
    ]
    for mA, mB in model_pairs:
        cells = []
        for cond in CONDITIONS:
            r = q2_df[(q2_df["model_A"] == mA) & (q2_df["model_B"] == mB) & (q2_df["condition"] == cond)]
            cells.append(_cell(r.iloc[0]["diff"], r.iloc[0]["se"], r.iloc[0]["sig"])
                         if not r.empty else "--")
        pair_label = MODEL_PAIR_LABELS.get((mA, mB), f"{MODEL_TEX.get(mA, mA)} vs.\\ {MODEL_TEX.get(mB, mB)}")
        lines.append(rf"{pair_label} & {' & '.join(cells)} \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Cross-model comparisons per condition (Contribution). {CAPTION_NOTE}}}",
        rf"\label{{tab:q2_contribution{suffix}}}",
        r"\end{table}",
    ]
    _save_tex(lines, os.path.join(OUT_DIR, f"q2_contribution{suffix}.tex"))


def save_csv(df, path):
    # Writes a DataFrame to CSV, creating the destination folder if needed.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral OLS analysis (contribution, all rounds).")
    parser.add_argument("--variant", "-v", default="local",
                        help="Data variant to analyse (default: local). Accepts any subpath "
                             "under {results_dir}/{model}/, e.g. 'local/N16_G4' or "
                             "'local_groupsizevary/N12_G3_MCPR0.4' for the "
                             "community-size/group-size/MCPR sweeps.")
    parser.add_argument("--models", "-m", nargs="+",
                        default=["gpt", "llama", "mistral", "qwen"],
                        choices=["gpt", "llama", "mistral", "qwen",
                                 "llama_13b", "mistral_13b", "qwen_14b", "llama_70b", "qwen_72b",
                                 "gpt-5-mini"],
                        help="Models to include (default: all four).")
    parser.add_argument("--seeds", "-s", nargs="+", type=int, default=None,
                        metavar="SEED", help="Explicit seed list. Overrides --n.")
    parser.add_argument("--n", type=int, default=10, metavar="N",
                        help="Number of seeds from the default list (default: 10).")
    parser.add_argument("--out-dir", "-o", default=None,
                        help="Output directory (default: figures/2026-03-22/paper_stats).")
    args = parser.parse_args()

    # Apply CLI overrides to module-level globals used by helpers
    VARIANT = args.variant
    MODELS  = args.models
    SEEDS   = args.seeds if args.seeds is not None else ALL_SEEDS[:args.n]
    if args.out_dir:
        OUT_DIR = args.out_dir
    # Sanitize for use in a filename -- variant subpaths like "local/N16_G4"
    # contain "/", which isn't valid in a bare filename component.
    variant_tag = f"_{VARIANT.replace('/', '_')}"

    # If the default reference model is not in the requested model list, use the first one
    if REF_MODEL not in MODELS:
        REF_MODEL = MODELS[0]

    print(f"Variant   : {VARIANT}")
    print(f"Models    : {MODELS}")
    print(f"Seeds     : {SEEDS}")
    print(f"Ref model : {REF_MODEL}")
    print(f"Out dir   : {OUT_DIR}")

    print("\nBuilding dataframe …")
    df = build_dataframe()
    print(f"  {len(df):,} observations | {df['seed'].nunique()} seeds | "
          f"rounds {df['round'].min()}–{df['round'].max()}")

    print("\n── OLS: Q1 (within-model condition pairs) ──")
    q1 = run_q1(df)
    save_csv(q1, os.path.join(OUT_DIR, f"q1_all_rounds{variant_tag}.csv"))
    make_q1_latex(q1, suffix=variant_tag)

    print("\n── OLS: Q2 (cross-model pairs per condition) ──")
    q2 = run_q2(df)
    save_csv(q2, os.path.join(OUT_DIR, f"q2_all_rounds{variant_tag}.csv"))
    make_q2_latex(q2, suffix=variant_tag)

    print("\nAll done.")
