"""
behavior_quantified_noexpect.py
--------------------------------
OLS with seed-clustered SEs for behavioral outcomes (contribution) in the
NO_EXPECT (perception_on=False) arm of the simulation.

Mirrors behavior_quantified.py's Q1 methodology EXACTLY (same formula shape,
same Wald-contrast machinery, same Bonferroni correction) so results are
directly comparable to the main-paper Q1 tables. Differences from the main
script are only in scope, forced by what the noexpect arm actually contains:

  - 4 models (gpt, llama, mistral, qwen). GPT-4o-mini noexpect data lives
    under code/results/gpt-4o-mini/local_noexpect/ (added 2026-08-15) —
    note the directory is "gpt-4o-mini", not "gpt", unlike the other three
    models; see MODEL_DIR below. GPT also uses a different seed range
    (43-52) than llama/mistral/qwen (42-51) — same off-by-one quirk as the
    main local-variant with-expect dataset (results/gpt/local/ also starts
    at seed43, not seed42); see MODEL_SEEDS below.
  - 4 conditions: PURE_BASELINE plus 3 noexpect conditions, no BASELINE
    analogue:
        PURE_BASELINE              (discussion off, selection off, perception off) — true zero-mechanism reference
        DISCUSSION_ONLY_NO_EXPECT  (discussion on,  selection off) — "no selection"
        SELECTION_ONLY_NO_EXPECT   (discussion off, selection on)  — "no social learning"
        FULL_NO_EXPECT             (discussion on,  selection on)
    PURE_BASELINE is pulled from the main with-expect dataset
    (results/{model}/local/, using "gpt" there — the with-expect and
    noexpect datasets use different directory names for the same model;
    see PB_MODEL_DIR vs. MODEL_DIR), NOT from local_noexpect/ — there is no
    noexpect-arm PURE_BASELINE run because it isn't needed: PURE_BASELINE
    already has perception_on=False (see make_config() in
    SoNoLiSi_v5_local_noexpect.py), so it's mechanically identical to what a
    "PURE_BASELINE_NO_EXPECT" would be. Matched to each model's own seed
    range (see MODEL_SEEDS) as the noexpect conditions for a balanced
    within-model design.
    (Plain BASELINE has no noexpect analogue for the same underlying reason
    — perception/expectation formation is mechanically gated by
    discussion_on, so with discussion off there is nothing for perception_on
    to toggle — but BASELINE isn't included here since PURE_BASELINE is the
    more informative zero-mechanism reference point and the task only asked
    to add "pure baseline".)
  - "Compare each pair" = all C(4,2) = 6 pairs (no directional PAPER_PAIRS
    subset), Bonferroni over n=6.
  - metric: contribution only (task scope).

Sign convention (matches behavior_quantified.py, flipped 2026-08-17): for
pair "A -> B", diff = coef(B) - coef(A), so positive means the second-listed
condition is higher.

Output: figures/2026-03-22/paper_stats/noexpect/
See code/analysis/MAIN_PAPER_RESULTS_NOEXPECT.md for how this fits the paper.
"""

import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from itertools import combinations
from scipy.stats import norm as _norm
import json
import glob

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS    = "/data3/rasimura/social-norm-evo/code/results"
OUT_DIR    = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats/noexpect"

# PURE_BASELINE lives in the main-paper "local" (with-expect) dataset — see
# module docstring for why that's still the correct zero-mechanism reference.
PB_RESULTS = "/data3/rasimura/social-norm-evo/results"
PB_VARIANT = "local"
PB_COND    = "PURE_BASELINE"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "local_noexpect"

# code/results/ dir name for the noexpect dataset differs from the with-expect
# ("PB_") dataset for gpt: "gpt-4o-mini" here vs. "gpt" under results/{model}/local/.
MODEL_DIR    = {"gpt": "gpt-4o-mini", "llama": "llama", "mistral": "mistral", "qwen": "qwen"}
PB_MODEL_DIR = {"gpt": "gpt", "llama": "llama", "mistral": "mistral", "qwen": "qwen"}

# gpt's noexpect run (and its with-expect PURE_BASELINE) starts at seed43, not
# seed42 — same off-by-one as the rest of the local-variant with-expect dataset.
MODEL_SEEDS = {
    "gpt":     list(range(43, 53)),
    "llama":   list(range(42, 52)),
    "mistral": list(range(42, 52)),
    "qwen":    list(range(42, 52)),
}
NOEXPECT_CONDITIONS = ["DISCUSSION_ONLY_NO_EXPECT", "SELECTION_ONLY_NO_EXPECT", "FULL_NO_EXPECT"]
CONDITIONS = [PB_COND] + NOEXPECT_CONDITIONS
REF_COND   = PB_COND
METRIC     = "contribution"
LAST_N     = 5   # rounds used for steady-state analysis

PAPER_PAIRS = list(combinations(CONDITIONS, 2))   # all 6 pairs

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_latest_log(results_dir, variant, model, seed, condition):
    pattern = os.path.join(results_dir, model, variant, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def build_dataframe():
    """Long-format DataFrame: one row per (model, condition, seed, round, agent)."""
    rows = []
    for model in MODELS:
        for seed in MODEL_SEEDS[model]:
            for cond in NOEXPECT_CONDITIONS:
                d = load_latest_log(RESULTS, VARIANT, MODEL_DIR[model], seed, cond)
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
            d = load_latest_log(PB_RESULTS, PB_VARIANT, PB_MODEL_DIR[model], seed, PB_COND)
            if d is not None:
                for r in d["round_logs"]:
                    rnd = r["round"]
                    for agent_id, contrib in r["contributions"].items():
                        rows.append({
                            "model":        model,
                            "condition":    PB_COND,
                            "seed":         seed,
                            "round":        rnd,
                            "contribution": float(contrib),
                        })
    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=False)
    df["model"]     = pd.Categorical(df["model"],     categories=MODELS,     ordered=False)
    return df


# ─── OLS + Wald helpers (identical to behavior_quantified.py) ────────────────

def fit_ols(df, formula, cluster_col="seed"):
    try:
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
    except Exception as e:
        print(f"  [WARN] OLS failed: {e}")
        return None


def wald_contrast(result, key_a, key_b):
    params = result.params
    cov    = result.cov_params()

    def _c(k):  return params[k]         if k and k in params.index else 0.0
    def _v(k):  return cov.loc[k, k]     if k and k in cov.index   else 0.0
    def _cv(a, b):
        return cov.loc[a, b] if a and b and a in cov.index and b in cov.columns else 0.0

    diff = _c(key_a) - _c(key_b)
    se   = np.sqrt(_v(key_a) + _v(key_b) - 2 * _cv(key_a, key_b))
    z    = diff / se if se > 0 else np.nan
    p    = 2 * (1 - _norm.cdf(abs(z)))
    return diff, se, z, p


def cond_param(cond, ref=REF_COND):
    return None if cond == ref else f"C(condition, Treatment('{ref}'))[T.{cond}]"


# ─── Q1: Within-model condition contrasts ─────────────────────────────────────

def run_q1(df, last_n=None):
    """
    Per model: OLS with seed-clustered SEs.
    All rounds:    contribution ~ C(condition) + round
    Final n:       contribution ~ C(condition)            (steady state)
    Returns DataFrame with one row per (model, pair).
    """
    if last_n:
        dfw     = df[df["round"] > df["round"].max() - last_n].copy()
        formula = f"contribution ~ C(condition, Treatment('{REF_COND}'))"
    else:
        dfw     = df.copy()
        formula = f"contribution ~ C(condition, Treatment('{REF_COND}')) + round"

    rows = []
    for model in MODELS:
        mdf    = dfw[dfw["model"] == model].copy()
        result = fit_ols(mdf, formula)
        if result is None:
            continue
        for cA, cB in PAPER_PAIRS:
            diff, se, z, p = wald_contrast(result, cond_param(cB), cond_param(cA))
            rows.append({"model": model, "cond_A": cA, "cond_B": cB,
                         "diff": diff, "se": se, "z": z, "p": p})

    out = pd.DataFrame(rows)
    n   = len(PAPER_PAIRS)
    out["p_bonf"] = (out["p"] * n).clip(upper=1.0)
    out["sig"]    = out["p_bonf"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )
    return out


# ─── LaTeX table generation ───────────────────────────────────────────────────

COND_PAIR_LABELS = {
    ("PURE_BASELINE", "DISCUSSION_ONLY_NO_EXPECT"):            r"Pure Baseline $\to$ Discussion Only",
    ("PURE_BASELINE", "SELECTION_ONLY_NO_EXPECT"):              r"Pure Baseline $\to$ Selection Only",
    ("PURE_BASELINE", "FULL_NO_EXPECT"):                        r"Pure Baseline $\to$ Full",
    ("DISCUSSION_ONLY_NO_EXPECT", "SELECTION_ONLY_NO_EXPECT"): r"Discussion Only $\to$ Selection Only",
    ("DISCUSSION_ONLY_NO_EXPECT", "FULL_NO_EXPECT"):           r"Discussion Only $\to$ Full",
    ("SELECTION_ONLY_NO_EXPECT",  "FULL_NO_EXPECT"):           r"Selection Only $\to$ Full",
}
COND_TEX = {
    "PURE_BASELINE":              "Pure Baseline",
    "DISCUSSION_ONLY_NO_EXPECT": "Discussion Only (No Expect)",
    "SELECTION_ONLY_NO_EXPECT":  "Selection Only (No Expect)",
    "FULL_NO_EXPECT":            "Full (No Expect)",
}
MODEL_TEX = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
SIG_TEX = {"***": r"$^{***}$", "**": r"$^{**}$", "*": r"$^{*}$", "": ""}

CAPTION_NOTE = (r"Cells show $\hat{\beta}$ (SE) for the second-listed condition minus the "
                r"first (positive = second is higher). Pure Baseline is the matched "
                r"with-expect-dataset zero-mechanism reference (discussion/selection/perception "
                r"all off), included here because it is unaffected by the expectation manipulation. "
                r"Bonferroni-corrected over 6 pairs: $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.")


def _cell(diff, se, sig):
    return f"{diff:.2f} ({se:.2f}){SIG_TEX[sig]}"


def make_q1_latex(q1_df, suffix=""):
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
        rf"\caption{{Within-model condition contrasts, no-expectation arm (Contribution). {CAPTION_NOTE}}}",
        rf"\label{{tab:q1_noexpect_contribution{suffix}}}",
        r"\end{table}",
    ]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"q1_contribution{suffix}.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Models    : {MODELS}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Seeds     : {MODEL_SEEDS}")
    print(f"Ref cond  : {REF_COND}")
    print(f"Out dir   : {OUT_DIR}")

    print("\nBuilding dataframe …")
    df = build_dataframe()
    print(f"  {len(df):,} observations | {df['seed'].nunique()} seeds | "
          f"rounds {df['round'].min()}–{df['round'].max()}")
    print(df.groupby(["model", "condition"], observed=True).size().unstack())

    print("\n── OLS: Q1 all rounds ──")
    q1_all = run_q1(df)
    print(q1_all.to_string(index=False))
    save_csv(q1_all, os.path.join(OUT_DIR, "q1_all_rounds_contribution.csv"))
    make_q1_latex(q1_all)

    print(f"\n── OLS: Q1 final {LAST_N} rounds (steady state) ──")
    q1_final = run_q1(df, last_n=LAST_N)
    print(q1_final.to_string(index=False))
    save_csv(q1_final, os.path.join(OUT_DIR, f"q1_final{LAST_N}r_contribution.csv"))
    make_q1_latex(q1_final, suffix=f"_final{LAST_N}r")

    print("\nAll done.")
