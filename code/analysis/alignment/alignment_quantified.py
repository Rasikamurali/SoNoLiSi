"""
alignment_quantified.py
-----------------------
OLS with seed-clustered SEs for perception-action gap outcomes.

Gap metrics:
  IN_gap = injunctive_norm  − actual_contribution
  DN_gap = descriptive_norm − actual_contribution

PURE_BASELINE excluded (no perceptions recorded).
Reference condition: BASELINE.  Reference model: gpt.

Four table sets (each produces IN_gap + DN_gap LaTeX tables):

  Set 1 — Q1 all rounds:   within-model, condition pairs, round as covariate
  Set 2 — Q2 all rounds:   cross-model,  per condition,   round as covariate
  Set 3 — Q1 final 5r:     within-model, condition pairs, steady-state
  Set 4 — Q2 final 5r:     cross-model,  per condition,   steady-state

Q1 condition pairs:
  a) Baseline       → No Discussion
  b) Baseline       → No Selection
  c) No Selection   → Full
  d) No Discussion  → Full

Output: figures/2026-03-22/paper_stats/alignment_*.csv / *.tex
"""

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
RESULTS  = "/data3/rasimura/social-norm-evo/results"
OUT_DIR  = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "global"
SEEDS      = list(range(43, 53))
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
REF_COND   = "BASELINE"
REF_MODEL  = "gpt"
METRICS    = ["IN_gap", "DN_gap"]
LAST_N     = 5

PAPER_PAIRS = [
    ("BASELINE",     "NO_DISCUSSION"),
    ("BASELINE",     "NO_SELECTION"),
    ("NO_SELECTION", "FULL"),
    ("NO_DISCUSSION","FULL"),
]

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_latest_log(model, seed, condition):
    pattern = os.path.join(RESULTS, model, VARIANT, f"seed{seed}", "log_*.json")
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
    """
    Long-format DataFrame: one row per (model, condition, seed, round, agent).
    Only agents with valid perceptions in that round are included.
    """
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                for r in d["round_logs"]:
                    rnd           = r["round"]
                    contributions = {int(k): float(v) for k, v in r["contributions"].items()}
                    perceptions   = {int(k): v        for k, v in (r.get("perceptions") or {}).items()}

                    for aid, perc in perceptions.items():
                        if not perc:
                            continue
                        actual = contributions.get(aid)
                        if actual is None:
                            continue
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        if inj is None or desc is None:
                            continue
                        rows.append({
                            "model":     model,
                            "condition": cond,
                            "seed":      seed,
                            "round":     rnd,
                            "agent_id":  str(aid),
                            "IN_gap":    float(inj)  - actual,
                            "DN_gap":    float(desc) - actual,
                        })

    df = pd.DataFrame(rows)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITIONS, ordered=False)
    df["model"]     = pd.Categorical(df["model"],     categories=MODELS,     ordered=False)
    return df


# ─── OLS helper ───────────────────────────────────────────────────────────────

def fit_ols(df, formula, cluster_col="seed"):
    try:
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
    except Exception as e:
        print(f"  [WARN] OLS failed: {e}")
        return None


def wald_contrast(result, key_a, key_b):
    """Wald test for coef[key_a] - coef[key_b]. Either key may be None (reference → 0)."""
    params = result.params
    cov    = result.cov_params()

    def _c(k):     return params[k]     if k and k in params.index else 0.0
    def _v(k):     return cov.loc[k, k] if k and k in cov.index   else 0.0
    def _cv(a, b): return cov.loc[a, b] if a and b and a in cov.index and b in cov.columns else 0.0

    diff = _c(key_a) - _c(key_b)
    se   = np.sqrt(_v(key_a) + _v(key_b) - 2 * _cv(key_a, key_b))
    z    = diff / se if se > 0 else np.nan
    p    = 2 * (1 - _norm.cdf(abs(z)))
    return diff, se, z, p


def cond_param(cond, ref=REF_COND):
    return None if cond == ref else f"C(condition, Treatment('{ref}'))[T.{cond}]"

def model_param(mdl, ref=REF_MODEL):
    return None if mdl == ref else f"C(model, Treatment('{ref}'))[T.{mdl}]"


def _sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ─── Q1: Within-model condition contrasts ─────────────────────────────────────

def run_q1(df, last_n=None):
    if last_n:
        dfw     = df[df["round"] > df["round"].max() - last_n].copy()
        formula = "{{metric}} ~ C(condition, Treatment('{ref}'))".format(ref=REF_COND)
    else:
        dfw     = df.copy()
        formula = "{{metric}} ~ C(condition, Treatment('{ref}')) * round".format(ref=REF_COND)

    rows = []
    for model in MODELS:
        mdf = dfw[dfw["model"] == model].copy()
        for metric in METRICS:
            result = fit_ols(mdf, formula.replace("{metric}", metric))
            if result is None:
                continue
            for cA, cB in PAPER_PAIRS:
                diff, se, z, p = wald_contrast(result, cond_param(cA), cond_param(cB))
                rows.append({"model": model, "metric": metric,
                             "cond_A": cA, "cond_B": cB,
                             "diff": diff, "se": se, "z": z, "p": p})

    out = pd.DataFrame(rows)
    n = len(PAPER_PAIRS)
    out["p_bonf"] = (out["p"] * n).clip(upper=1.0)
    out["sig"]    = out["p_bonf"].apply(_sig)
    return out


# ─── Q2: Cross-model comparisons per condition ────────────────────────────────

def run_q2(df, last_n=None):
    if last_n:
        dfw     = df[df["round"] > df["round"].max() - last_n].copy()
        formula = "{{metric}} ~ C(model, Treatment('{ref}'))".format(ref=REF_MODEL)
    else:
        dfw     = df.copy()
        formula = "{{metric}} ~ C(model, Treatment('{ref}')) + round".format(ref=REF_MODEL)

    model_pairs = list(combinations(MODELS, 2))
    rows = []
    for cond in CONDITIONS:
        cdf = dfw[dfw["condition"] == cond].copy()
        for metric in METRICS:
            result = fit_ols(cdf, formula.replace("{metric}", metric))
            if result is None:
                continue
            for mA, mB in model_pairs:
                diff, se, z, p = wald_contrast(result, model_param(mA), model_param(mB))
                rows.append({"condition": cond, "metric": metric,
                             "model_A": mA, "model_B": mB,
                             "diff": diff, "se": se, "z": z, "p": p})

    out = pd.DataFrame(rows)
    n = len(model_pairs)
    out["p_bonf"] = (out["p"] * n).clip(upper=1.0)
    out["sig"]    = out["p_bonf"].apply(_sig)
    return out


# ─── LaTeX table generation ───────────────────────────────────────────────────

COND_PAIR_LABELS = {
    ("BASELINE",     "NO_DISCUSSION"): r"Baseline $\to$ No Discussion",
    ("BASELINE",     "NO_SELECTION"):  r"Baseline $\to$ No Selection",
    ("NO_SELECTION", "FULL"):          r"No Selection $\to$ Full",
    ("NO_DISCUSSION","FULL"):          r"No Discussion $\to$ Full",
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
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
MODEL_TEX  = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
METRIC_TEX = {"IN_gap": "IN Gap", "DN_gap": "DN Gap"}
SIG_TEX    = {"***": r"$^{***}$", "**": r"$^{**}$", "*": r"$^{*}$", "": ""}

CAPTION_NOTE = (r"Cells show $\hat{\beta}$ (SE). "
                r"Bonferroni-corrected: $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.")


def _cell(diff, se, sig):
    return f"{diff:.2f} ({se:.2f}){SIG_TEX[sig]}"


def _save_tex(lines, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


def make_q1_latex(q1_df, suffix=""):
    """Q1: rows = condition pairs, cols = 4 models."""
    for metric in METRICS:
        sub      = q1_df[q1_df["metric"] == metric]
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
                r = sub[(sub["model"] == mdl) & (sub["cond_A"] == cA) & (sub["cond_B"] == cB)]
                cells.append(_cell(r.iloc[0]["diff"], r.iloc[0]["se"], r.iloc[0]["sig"])
                             if not r.empty else "--")
            lines.append(rf"{COND_PAIR_LABELS[(cA, cB)]} & {' & '.join(cells)} \\")
        lines += [
            r"\bottomrule", r"\end{tabular}",
            rf"\caption{{Within-model condition contrasts ({METRIC_TEX[metric]}). {CAPTION_NOTE}}}",
            rf"\label{{tab:align_q1_{metric}{suffix}}}",
            r"\end{table}",
        ]
        _save_tex(lines, os.path.join(OUT_DIR, f"align_q1_{metric}{suffix}.tex"))


def make_q2_latex(q2_df, suffix=""):
    """Q2: rows = model pairs, cols = 4 conditions."""
    model_pairs = list(combinations(MODELS, 2))
    for metric in METRICS:
        sub      = q2_df[q2_df["metric"] == metric]
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
                r = sub[(sub["model_A"] == mA) & (sub["model_B"] == mB) & (sub["condition"] == cond)]
                cells.append(_cell(r.iloc[0]["diff"], r.iloc[0]["se"], r.iloc[0]["sig"])
                             if not r.empty else "--")
            lines.append(rf"{MODEL_PAIR_LABELS[(mA, mB)]} & {' & '.join(cells)} \\")
        lines += [
            r"\bottomrule", r"\end{tabular}",
            rf"\caption{{Cross-model comparisons per condition ({METRIC_TEX[metric]}). {CAPTION_NOTE}}}",
            rf"\label{{tab:align_q2_{metric}{suffix}}}",
            r"\end{table}",
        ]
        _save_tex(lines, os.path.join(OUT_DIR, f"align_q2_{metric}{suffix}.tex"))


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building perception-action gap dataframe …")
    df = build_dataframe()
    print(f"  {len(df):,} observations | {df['seed'].nunique()} seeds | "
          f"rounds {df['round'].min()}–{df['round'].max()}")
    print(df.groupby(["model", "condition"])["IN_gap"].count().unstack())

    print("\n── Set 1: Q1 all rounds ──")
    q1_all = run_q1(df)
    save_csv(q1_all, os.path.join(OUT_DIR, "align_q1_all_rounds.csv"))
    make_q1_latex(q1_all, suffix="")

    print(f"\n── Set 2: Q1 final {LAST_N} rounds ──")
    q1_final = run_q1(df, last_n=LAST_N)
    save_csv(q1_final, os.path.join(OUT_DIR, f"align_q1_final{LAST_N}r.csv"))
    make_q1_latex(q1_final, suffix=f"_final{LAST_N}r")

    print("\nAll done.")
