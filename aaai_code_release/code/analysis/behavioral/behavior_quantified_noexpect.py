"""
behavior_quantified_noexpect.py
--------------------------------
Same test as behavior_quantified.py's Q1 (OLS on contribution, seed-clustered
standard errors, Wald contrasts, Bonferroni correction), but run on the
no-expectations ablation arm instead of the main 5 conditions -- the arm
where perception/expectation formation is switched off, so discussion and
selection can be tested in isolation from expectation formation. Uses the
exact same formula shape and Wald-contrast machinery as the main script, so
the two sets of results are directly comparable.

The 4 conditions compared here:
    PURE_BASELINE              discussion off, selection off, perception off
                               (zero-mechanism reference; pulled from the
                               main with-expectation dataset, not re-run
                               here, since it's identical either way once
                               perception is off)
    DISCUSSION_ONLY_NO_EXPECT  discussion on,  selection off  ("no selection")
    SELECTION_ONLY_NO_EXPECT   discussion off, selection on   ("no social learning")
    FULL_NO_EXPECT             discussion on,  selection on
All C(4,2) = 6 pairs are compared (unlike the main script's 5 directional
pairs), Bonferroni-corrected over 6.

Two known data quirks this script accounts for (not simplifications, just
facts about where the raw files live):
  - GPT's no-expectations data is stored under the folder name
    "gpt-4o-mini", while the other three models keep their usual folder
    names (see MODEL_DIR).
  - GPT's seed range (43-52) differs by one from llama/mistral/qwen's
    (42-51); see MODEL_SEEDS.

Inputs: raw simulation logs under code/results/{model}/local_noexpect/
seed{N}/log_*.json for the three no-expectations conditions, plus
results/{model}/local/seed{N}/log_*.json for the PURE_BASELINE reference.
Outputs (figures/2026-03-22/paper_stats/noexpect/): q1_all_rounds_
contribution.csv and q1_final5r_contribution.csv (raw numbers, all rounds
and final-5-rounds respectively), plus the matching .tex tables.
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
# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/behavioral/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS    = f"{BASE}/code/results"
OUT_DIR    = f"{BASE}/figures/2026-03-22/paper_stats/noexpect"

# PURE_BASELINE lives in the main-paper "local" (with-expect) dataset — see
# module docstring for why that's still the correct zero-mechanism reference.
PB_RESULTS = f"{BASE}/results"
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
    # Finds and loads the log file for one (model, seed, condition)
    # combination. If more than one file matches, the last one found wins.
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
    # Loads the 3 no-expectations conditions from their own dataset, then
    # separately loads PURE_BASELINE from the main with-expectation dataset
    # (see module docstring for why PURE_BASELINE doesn't need its own
    # no-expectations run) and appends it onto the same table.
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
    # Fits the regression and returns None (with a warning) instead of
    # raising, so one failed fit doesn't stop the whole script.
    try:
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
    except Exception as e:
        print(f"  [WARN] OLS failed: {e}")
        return None


def wald_contrast(result, key_a, key_b):
    # Computes coef(key_a) - coef(key_b) and its standard error/p-value.
    # A missing key is treated as the reference level (coefficient 0).
    params = result.params
    cov    = result.cov_params()

    def _c(k):  return params[k]         if k and k in params.index else 0.0
    def _v(k):  return cov.loc[k, k]     if k and k in cov.index   else 0.0
    def _cv(a, b):
        return cov.loc[a, b] if a and b and a in cov.index and b in cov.columns else 0.0

    # Var(A-B) = Var(A) + Var(B) - 2*Cov(A,B).
    diff = _c(key_a) - _c(key_b)
    se   = np.sqrt(_v(key_a) + _v(key_b) - 2 * _cv(key_a, key_b))
    z    = diff / se if se > 0 else np.nan
    p    = 2 * (1 - _norm.cdf(abs(z)))
    return diff, se, z, p


def cond_param(cond, ref=REF_COND):
    # Statsmodels' coefficient name for one condition dummy variable
    # (None for the reference condition, which has no dummy of its own).
    return None if cond == ref else f"C(condition, Treatment('{ref}'))[T.{cond}]"


# ─── Q1: Within-model condition contrasts ─────────────────────────────────────

def run_q1(df, last_n=None):
    """
    Per model: OLS with seed-clustered SEs.
    All rounds:    contribution ~ C(condition) + round
    Final n:       contribution ~ C(condition)            (steady state)
    Returns DataFrame with one row per (model, pair).
    """
    # If last_n is given, restrict to the final last_n rounds and drop
    # "round" from the formula (a steady-state view); otherwise use every
    # round with "round" included as a linear covariate.
    if last_n:
        dfw     = df[df["round"] > df["round"].max() - last_n].copy()
        formula = f"contribution ~ C(condition, Treatment('{REF_COND}'))"
    else:
        dfw     = df.copy()
        formula = f"contribution ~ C(condition, Treatment('{REF_COND}')) + round"

    # Fit one regression per model, then read off all 6 pairwise contrasts
    # from that one fitted model.
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
    # Formats one table cell as "estimate (standard error)" plus significance stars.
    return f"{diff:.2f} ({se:.2f}){SIG_TEX[sig]}"


def make_q1_latex(q1_df, suffix=""):
    # Builds and writes the LaTeX table: one row per condition pair, one
    # column per model.
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
    # Writes a DataFrame to CSV, creating the destination folder if needed.
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
