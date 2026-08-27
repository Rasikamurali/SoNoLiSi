"""
behavioral_statistical.py
--------------------------
Formal statistical tests for contribution (behavioral) outcomes -- the
paper's "supporting results" (mixed effects, level/slope). Single reusable
file for any model set: 7B/8B (default), 13B/14B tier, 70B/72B tier, or any
other subset, selected via --models on the CLI. Was three separate files
(behavioral_statistical.py + _13b.py + _70b.py monkey-patching this one's
globals) with the actual data-loading logic duplicated verbatim across all
three; consolidated since --models/--variant/--results-dir/--out-dir already
cover what those two variant files did.

Tests
-----
1. Level analysis       — contribution ~ condition + (1|run), pairwise contrasts
2. Slope analysis       — contribution ~ round*condition + (1+round|run), slope contrasts
3. Cross-model omnibus  — LRT for condition×family (level) and round×condition×family (slope)
4. Initial-cooperation  — round*condition*initial_level moderation per family
5. Dispersion           — SD across agents, final 5 rounds, per family×condition

Outputs  →  --out-dir (default: figures/statistical_tests/behavioral/)
"""

import argparse
import json
import glob
import os
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Defaults are the 7B/8B family; --models/--model-labels/--ref-family/
# --variant/--results-dir on the CLI (see __main__ below) repoint all of
# this to any other model set -- e.g. the 13B/14B tier, the 70B/72B tier,
# or the S2 replication set. Every test_*/write_*_tex function below reads
# FAMILIES/REF_FAM/CONDITIONS as module globals, so nothing past this
# config block needs to change per model set.
RESULTS    = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT   = "/data3/rasimura/social-norm-evo/figures/statistical_tests/behavioral"
VARIANT    = "local"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
LABELS     = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
FAMILIES   = ["GPT", "Llama", "Mistral", "Qwen"]
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
ROUND_MEAN = 10.5
REF_COND   = "PURE_BASELINE"
REF_FAM    = "Mistral"

# Fallback display names for known model keys -- --model-labels overrides these.
DEFAULT_LABELS = {
    "gpt":         "GPT",
    "llama":       "Llama",
    "mistral":     "Mistral",
    "qwen":        "Qwen",
    "llama_13b":   "Llama-13B",
    "mistral_13b": "Mistral-13B",
    "qwen_14b":    "Qwen-14B",
    "llama_70b":   "Llama-70B",
    "qwen_72b":    "Qwen-72B",
    "gpt-5-mini":  "GPT-5-mini",
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_best_logs():
    """Return dict (model, seed, condition) → (seed, log_dict), latest file wins."""
    best = {}
    for model in MODELS:
        for path in sorted(glob.glob(
                os.path.join(RESULTS, model, VARIANT, "seed*", "log_*.json"))):
            try:
                d    = json.load(open(path))
                seed = int(path.split("seed")[1].split("/")[0])
                best[(model, seed, d["condition"])] = (seed, d)
            except Exception:
                continue
    return best


def build_dataframe() -> pd.DataFrame:
    """Run-level mean contributions: one row per (run × round)."""
    rows = []
    for (model, _, _), (s, d) in _load_best_logs().items():
        for r in d["round_logs"]:
            rows.append({
                "family":    LABELS[model],
                "seed":      s,
                "condition": d["condition"],
                "round":     r["round"],
                "contrib":   float(np.mean(list(r["contributions"].values()))),
                "run_id":    f"{LABELS[model]}_s{s}_{d['condition']}",
            })
    df = pd.DataFrame(rows)
    df["round_c"]  = df["round"] - ROUND_MEAN
    df["family_f"] = pd.Categorical(df["family"],    FAMILIES)
    df["cond_f"]   = pd.Categorical(df["condition"], CONDITIONS)
    return df


def build_agent_dataframe() -> pd.DataFrame:
    """Agent-level contributions: one row per (agent × round × run)."""
    rows = []
    for (model, _, _), (s, d) in _load_best_logs().items():
        for r in d["round_logs"]:
            for aid, contrib in r["contributions"].items():
                rows.append({
                    "family":    LABELS[model],
                    "seed":      s,
                    "condition": d["condition"],
                    "round":     r["round"],
                    "agent_id":  int(aid),
                    "contrib":   float(contrib),
                    "run_id":    f"{LABELS[model]}_s{s}_{d['condition']}",
                })
    df = pd.DataFrame(rows)
    df["round_c"]  = df["round"] - ROUND_MEAN
    df["family_f"] = pd.Categorical(df["family"],    FAMILIES)
    df["cond_f"]   = pd.Categorical(df["condition"], CONDITIONS)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fit_lmm(formula, df, re_formula="~round_c", reml=True,
            method="lbfgs", maxiter=2000, fallback=True):
    """
    Fit LMM. Tries lbfgs → bfgs → nm before falling back to intercept-only RE.
    Returns (result, converged, fallback_used).
    """
    methods = [method] + [m for m in ("lbfgs", "bfgs", "nm") if m != method]
    re_options = ([re_formula, "~1"] if (fallback and re_formula != "~1")
                  else [re_formula])
    for use_re in re_options:
        for meth in methods:
            try:
                m   = smf.mixedlm(formula, data=df, groups=df["run_id"],
                                   re_formula=use_re)
                res = m.fit(method=meth, maxiter=maxiter, reml=reml)
                return res, res.converged, (use_re == "~1")
            except Exception:
                continue
    raise RuntimeError(f"LMM failed: {formula}")


def fe_cov(result):
    idx = result.fe_params.index
    return result.cov_params().loc[idx, idx].values


def get_X(result, grid_df):
    """Fixed-effects design matrix rows for grid_df."""
    di  = result.model.data.design_info
    (X,) = build_design_matrices([di], grid_df)
    return np.asarray(X)


def level_grid():
    """One row per condition for level predictions (no round_c)."""
    g = pd.DataFrame({"cond_f": CONDITIONS, "contrib": 0.0})
    g["cond_f"] = pd.Categorical(g["cond_f"], CONDITIONS)
    return g


def slope_grid():
    """Two rows per condition (round_c = ±9.5) for slope extraction."""
    rows = [{"cond_f": c, "round_c": r, "contrib": 0.0}
            for r in [-9.5, 9.5] for c in CONDITIONS]
    g = pd.DataFrame(rows)
    g["cond_f"] = pd.Categorical(g["cond_f"], CONDITIONS)
    return g


def delta_contrast(c_vec, beta, cov):
    est = float(c_vec @ beta)
    se  = float(np.sqrt(max(float(c_vec @ cov @ c_vec), 0.0)))
    z   = est / se if se > 1e-12 else np.nan
    p   = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
    return est, se, z, p


def apply_holm(pvals):
    _, padj, _, _ = multipletests(pvals, method="holm")
    return padj.tolist()


def stars(p):
    if pd.isna(p): return ""
    if p < .001:   return "***"
    if p < .01:    return "**"
    if p < .05:    return "*"
    if p < .10:    return "."
    return "ns"


# ── LaTeX formatting helpers ──

def fmt_p(p):
    if pd.isna(p):  return "---"
    if p < .001:    return r"$<$.001"
    return f".{int(round(p * 1000)):03d}"


def fmt_n(x, d=3):
    if pd.isna(x): return "---"
    s = f"{abs(x):.{d}f}"
    return (r"$-$" + s) if x < 0 else s


def sig_sup(sig):
    return rf"$^{{{sig}}}$" if sig in ("***", "**", "*", ".") else ""


def cond_tex(s):
    return s.replace("_", r"\_")


def contrast_tex(s):
    return s.replace("_", r"\_").replace("−", r"$-$")


CAPTION_STARS = (r"$^{\dagger}p < .10$, $^{*}p < .05$, "
                 r"$^{**}p < .01$, $^{***}p < .001$.")

N_CONTRASTS = len(CONDITIONS)         # 5
N_PAIRS     = N_CONTRASTS * (N_CONTRASTS - 1) // 2   # 10


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1 — LEVEL ANALYSIS
# contribution ~ condition + (1|run)  per family
# Pairwise contrasts (10 per family), Holm-corrected
# ═══════════════════════════════════════════════════════════════════════════════

def test_level_contrasts(df):
    formula = f"contrib ~ C(cond_f, Treatment('{REF_COND}'))"
    means_rows, contrast_rows, summaries = [], [], []

    for fam in FAMILIES:
        sub = df[df["family"] == fam].copy()
        sub["cond_f"] = pd.Categorical(sub["condition"], CONDITIONS)

        result, converged, fallback = fit_lmm(formula, sub, re_formula="~1", fallback=False)
        beta = result.fe_params.values
        cov  = fe_cov(result)

        X = get_X(result, level_grid())   # shape (5, n_fe)

        # Predicted level per condition
        for i, cond in enumerate(CONDITIONS):
            means_rows.append({"family": fam, "condition": cond,
                                "mean": round(float(X[i] @ beta), 4)})

        # All 10 pairwise contrasts
        raw_p, tmp = [], []
        for (i, c1), (j, c2) in combinations(enumerate(CONDITIONS), 2):
            c_vec = X[j] - X[i]
            est, se, z, p = delta_contrast(c_vec, beta, cov)
            raw_p.append(p)
            tmp.append({"family": fam, "contrast": f"{c2} − {c1}",
                         "mean_c1": round(float(X[i] @ beta), 4),
                         "mean_c2": round(float(X[j] @ beta), 4),
                         "estimate": round(est, 4), "se": round(se, 4),
                         "z": round(z, 3), "p_raw": p})

        for row, ph in zip(tmp, apply_holm(raw_p)):
            row["p_holm"] = round(ph, 4)
            row["sig"]    = stars(ph)
            contrast_rows.append(row)

        min_n = sub.groupby("condition")["run_id"].nunique().min()
        parts = [f"converged={converged}"]
        if fallback:      parts.append("intercept-only fallback")
        if min_n < 10:    parts.append(f"LOW POWER: min n/condition={min_n}")
        summaries.append((fam, ", ".join(parts)))

    return pd.DataFrame(means_rows), pd.DataFrame(contrast_rows), summaries


def write_level_tex(means_df, cdf, path):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        (r"\caption{Pairwise contrasts in mean contribution level by model family."
         r" Model per family: \texttt{contribution $\sim$ condition},"
         r" random intercept per run."
         r" All 10 pairwise contrasts across five conditions;"
         r" $p$-values Holm-corrected within each family."
         r" $\bar{c}_1$, $\bar{c}_2$ are predicted condition means."
         r" " + CAPTION_STARS + "}"),
        r"\label{tab:level_contrasts}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (r"Family & Contrast & $\bar{c}_1$ & $\bar{c}_2$"
         r" & $\Delta$ & SE & $z$ & $p_\text{Holm}$ \\"),
        r"\midrule",
    ]
    for i, fam in enumerate(FAMILIES):
        if i > 0: lines.append(r"\midrule")
        lines.append(rf"\multirow{{{N_PAIRS}}}{{*}}{{{fam}}}")
        for _, row in cdf[cdf["family"] == fam].iterrows():
            p_tex = fmt_p(row["p_holm"]) + sig_sup(row["sig"])
            lines.append(
                rf" & {contrast_tex(row['contrast'])}"
                rf" & {fmt_n(row['mean_c1'])}"
                rf" & {fmt_n(row['mean_c2'])}"
                rf" & {fmt_n(row['estimate'])}"
                rf" & {fmt_n(row['se'])}"
                rf" & {fmt_n(row['z'])}"
                rf" & {p_tex} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2 — SLOPE ANALYSIS
# Joint model: contribution ~ round*condition*family + (1+round|run)
# Per-family slopes extracted via delta method; pairwise contrasts (10 per
# family), Holm-corrected within family.
# Rationale: per-family models fail to converge (~10 runs/condition is
# insufficient to estimate random slopes); the joint model pools all ~215 runs
# to estimate the RE variance, then marginalises per family via EMMs.
# ═══════════════════════════════════════════════════════════════════════════════

def test_slope_contrasts(df):
    formula = (f"contrib ~ round_c"
               f" * C(cond_f, Treatment('{REF_COND}'))"
               f" * C(family_f, Treatment('{REF_FAM}'))")

    result, converged, fallback = fit_lmm(
        formula, df, re_formula="~round_c", fallback=True)
    beta = result.fe_params.values
    cov  = fe_cov(result)

    # Joint slope grid: 4 families × 2 round_c values × 5 conditions = 40 rows
    # Layout: for family i, condition j:
    #   low  row = i*10 + j        (round_c = -9.5)
    #   high row = i*10 + 5 + j    (round_c = +9.5)
    rows = []
    for fam in FAMILIES:
        for rnd_c in [-9.5, 9.5]:
            for cond in CONDITIONS:
                rows.append({"family_f": fam, "cond_f": cond,
                              "round_c": rnd_c, "contrib": 0.0,
                              "family": fam, "condition": cond, "seed": 43,
                              "run_id": "dummy"})
    grid = pd.DataFrame(rows)
    grid["family_f"] = pd.Categorical(grid["family_f"], FAMILIES)
    grid["cond_f"]   = pd.Categorical(grid["cond_f"],   CONDITIONS)

    X = get_X(result, grid)
    nc = len(CONDITIONS)  # 5

    slope_rows, contrast_rows = [], []

    for fi, fam in enumerate(FAMILIES):
        base = fi * 10   # start of this family's rows
        slope_vecs = [(X[base + nc + j] - X[base + j]) / 19.0
                      for j in range(nc)]
        slopes     = [float(sv @ beta)                         for sv in slope_vecs]
        slope_ses  = [float(np.sqrt(max(sv @ cov @ sv, 0.0))) for sv in slope_vecs]

        for j, cond in enumerate(CONDITIONS):
            slope_rows.append({"family": fam, "condition": cond,
                                "slope": round(slopes[j], 4),
                                "se":    round(slope_ses[j], 4)})

        raw_p, tmp = [], []
        for (i, c1), (j, c2) in combinations(enumerate(CONDITIONS), 2):
            c_diff = slope_vecs[j] - slope_vecs[i]
            est, se, z, p = delta_contrast(c_diff, beta, cov)
            raw_p.append(p)
            tmp.append({"family": fam, "contrast": f"{c2} − {c1}",
                         "slope_c1": round(slopes[i], 4),
                         "slope_c2": round(slopes[j], 4),
                         "estimate": round(est, 4), "se": round(se, 4),
                         "z": round(z, 3), "p_raw": p})

        for row, ph in zip(tmp, apply_holm(raw_p)):
            row["p_holm"] = round(ph, 4)
            row["sig"]    = stars(ph)
            contrast_rows.append(row)

    n_runs   = df["run_id"].nunique()
    min_cell = df.groupby(["family", "condition"])["run_id"].nunique().min()
    parts = [f"joint model, converged={converged}"]
    if fallback:    parts.append("intercept-only RE fallback")
    if min_cell < 10: parts.append(f"LOW POWER: min n/cell={min_cell}")
    summaries = [("all families", ", ".join(parts))]

    return pd.DataFrame(slope_rows), pd.DataFrame(contrast_rows), summaries


def write_slope_tex(slope_df, cdf, path):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        (r"\caption{Pairwise contrasts in contribution growth rate (round slope,"
         r" units/round) by model family."
         r" Single joint model: \texttt{contribution $\sim$ round $\times$ condition"
         r" $\times$ family}, random intercept and round slope per run."
         r" Per-family slopes estimated via delta method on fixed-effects covariance."
         r" All 10 pairwise contrasts across five conditions;"
         r" $p$-values Holm-corrected within each family."
         r" $\hat{\beta}_1$, $\hat{\beta}_2$ are estimated round slopes per condition."
         r" " + CAPTION_STARS + "}"),
        r"\label{tab:slope_contrasts}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (r"Family & Contrast & $\hat{\beta}_1$ & $\hat{\beta}_2$"
         r" & $\Delta\hat{\beta}$ & SE & $z$ & $p_\text{Holm}$ \\"),
        r"\midrule",
    ]
    for i, fam in enumerate(FAMILIES):
        if i > 0: lines.append(r"\midrule")
        lines.append(rf"\multirow{{{N_PAIRS}}}{{*}}{{{fam}}}")
        for _, row in cdf[cdf["family"] == fam].iterrows():
            p_tex = fmt_p(row["p_holm"]) + sig_sup(row["sig"])
            lines.append(
                rf" & {contrast_tex(row['contrast'])}"
                rf" & {fmt_n(row['slope_c1'])}"
                rf" & {fmt_n(row['slope_c2'])}"
                rf" & {fmt_n(row['estimate'])}"
                rf" & {fmt_n(row['se'])}"
                rf" & {fmt_n(row['z'])}"
                rf" & {p_tex} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3 — CROSS-MODEL CONSISTENCY (OMNIBUS LRT)
# LRT for condition×family (level) and round×condition×family (slope)
# Uses ML (reml=False) for valid likelihood comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _lrt(llf_full, llf_red, df_diff):
    chi2 = 2.0 * (llf_full - llf_red)
    p    = float(scipy_stats.chi2.sf(chi2, df=df_diff))
    return round(chi2, 3), df_diff, p


def test_omnibus(df):
    rows, summaries = [], []

    # — Level omnibus: condition×family interaction —
    f_full  = (f"contrib ~ C(cond_f, Treatment('{REF_COND}'))"
               f" * C(family_f, Treatment('{REF_FAM}'))")
    f_red   = (f"contrib ~ C(cond_f, Treatment('{REF_COND}'))"
               f" + C(family_f, Treatment('{REF_FAM}'))")
    df_diff = (len(CONDITIONS) - 1) * (len(FAMILIES) - 1)  # 4*3 = 12

    try:
        r_full, cv_f, _ = fit_lmm(f_full, df, re_formula="~1",
                                   reml=False, fallback=False)
        r_red,  cv_r, _ = fit_lmm(f_red,  df, re_formula="~1",
                                   reml=False, fallback=False)
        chi2, ddf, p = _lrt(r_full.llf, r_red.llf, df_diff)
        rows.append({"test": "condition × family (level)",
                     "model": "intercept RE", "df": ddf,
                     "chi2": chi2, "p": round(p, 4), "sig": stars(p),
                     "note": f"full conv={cv_f}, red conv={cv_r}"})
        summaries.append(f"Level omnibus: converged full={cv_f}, reduced={cv_r}")
    except Exception as e:
        rows.append({"test": "condition × family (level)",
                     "model": "FAILED", "df": df_diff,
                     "chi2": np.nan, "p": np.nan, "sig": "", "note": str(e)})
        summaries.append(f"Level omnibus: FAILED — {e}")

    # — Slope omnibus: round×condition×family interaction —
    f_full_s = (f"contrib ~ round_c * C(cond_f, Treatment('{REF_COND}'))"
                f" * C(family_f, Treatment('{REF_FAM}'))")
    f_red_s  = (f"contrib ~ round_c * C(cond_f, Treatment('{REF_COND}'))"
                f" + round_c * C(family_f, Treatment('{REF_FAM}'))"
                f" + C(cond_f, Treatment('{REF_COND}')) * C(family_f, Treatment('{REF_FAM}'))")
    df_diff_s = (len(CONDITIONS) - 1) * (len(FAMILIES) - 1)  # 12

    for re_f, re_label in [("~round_c", "slope RE"), ("~1", "intercept RE (fallback)")]:
        try:
            r_full_s, cv_f, _ = fit_lmm(f_full_s, df, re_formula=re_f,
                                         reml=False, fallback=False)
            r_red_s,  cv_r, _ = fit_lmm(f_red_s,  df, re_formula=re_f,
                                         reml=False, fallback=False)
            chi2, ddf, p = _lrt(r_full_s.llf, r_red_s.llf, df_diff_s)
            rows.append({"test": "round × condition × family (slope)",
                         "model": re_label, "df": ddf,
                         "chi2": chi2, "p": round(p, 4), "sig": stars(p),
                         "note": f"full conv={cv_f}, red conv={cv_r}"})
            summaries.append(
                f"Slope omnibus ({re_label}): conv full={cv_f}, reduced={cv_r}")
            break
        except Exception as e:
            if re_label.startswith("slope"):
                summaries.append(f"Slope omnibus (slope RE) failed, trying fallback: {e}")
                continue
            rows.append({"test": "round × condition × family (slope)",
                         "model": "FAILED", "df": df_diff_s,
                         "chi2": np.nan, "p": np.nan, "sig": "", "note": str(e)})
            summaries.append(f"Slope omnibus: FAILED — {e}")

    return pd.DataFrame(rows), summaries


def write_omnibus_tex(odf, path):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        (r"\caption{Omnibus likelihood ratio tests for cross-model consistency of"
         r" condition effects on contribution."
         r" \textit{Level}: condition $\times$ family interaction in"
         r" \texttt{contribution $\sim$ condition $\times$ family}."
         r" \textit{Slope}: round $\times$ condition $\times$ family interaction in"
         r" \texttt{contribution $\sim$ round $\times$ condition $\times$ family}."
         r" Both models use ML estimation (REML$=$False) for valid LRT."
         r" " + CAPTION_STARS + "}"),
        r"\label{tab:omnibus_interactions}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Interaction term & df & $\chi^2$ & $p$ & \\",
        r"\midrule",
    ]
    for _, row in odf.iterrows():
        test_tex  = row["test"].replace("×", r"$\times$")
        chi2_str  = fmt_n(row["chi2"], d=2)
        p_str     = fmt_p(row["p"]) + sig_sup(row["sig"])
        lines.append(
            rf"{test_tex} & {int(row['df'])} & {chi2_str} & {p_str} & \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4 — INITIAL-COOPERATION MODERATION
# contribution ~ round*condition*initial_level_c + (1+round|run)  per family
# initial_level = agent's contribution in round 1 of PURE_BASELINE,
#                 mean-centered within family
# Reports round×condition×initial_level_c terms (vs BASELINE reference)
# ═══════════════════════════════════════════════════════════════════════════════

def _add_initial_level(df_agents, fam):
    """Merge round-1 PURE_BASELINE contribution as initial_level; center within family."""
    fam_df = df_agents[df_agents["family"] == fam].copy()
    pb_r1  = fam_df[
        (fam_df["condition"] == "PURE_BASELINE") & (fam_df["round"] == 1)
    ][["seed", "agent_id", "contrib"]].rename(columns={"contrib": "initial_level"})

    merged = fam_df.merge(pb_r1, on=["seed", "agent_id"], how="left")
    fallback_mean = pb_r1["initial_level"].mean()
    n_missing = merged["initial_level"].isna().sum()
    merged["initial_level"] = merged["initial_level"].fillna(fallback_mean)
    merged["initial_level_c"] = merged["initial_level"] - merged["initial_level"].mean()
    return merged, n_missing


def test_moderation(df_agents):
    ref_cond_mod = "BASELINE"
    formula = (f"contrib ~ round_c * C(cond_f, Treatment('{ref_cond_mod}'))"
               f" * initial_level_c")
    contrast_rows, summaries = [], []

    for fam in FAMILIES:
        fam_df, n_missing = _add_initial_level(df_agents, fam)
        fam_df["cond_f"] = pd.Categorical(fam_df["condition"], CONDITIONS)

        result, converged, fallback = fit_lmm(
            formula, fam_df, re_formula="~round_c", fallback=True)
        beta   = result.fe_params.values
        params = result.fe_params
        cov    = fe_cov(result)

        # Extract three-way terms: round_c : cond_f[T.X] : initial_level_c
        three_way_pattern = re.compile(
            r"round_c.*cond_f.*initial_level_c|"
            r"initial_level_c.*cond_f.*round_c|"
            r"round_c.*initial_level_c.*cond_f", re.IGNORECASE)
        cond_pattern = re.compile(r"\[T\.([A-Z_]+)\]")

        three_way_terms = {
            name: idx for idx, name in enumerate(params.index)
            if three_way_pattern.search(name)
        }

        raw_p, tmp = [], []
        for term_name, col_idx in three_way_terms.items():
            cond_match = cond_pattern.search(term_name)
            cond_label = cond_match.group(1) if cond_match else term_name
            est = float(params.iloc[col_idx])
            se  = float(np.sqrt(max(cov[col_idx, col_idx], 0)))
            z   = est / se if se > 1e-12 else np.nan
            p   = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
            raw_p.append(p)
            tmp.append({"family": fam,
                         "condition_vs_baseline": cond_label,
                         "estimate": round(est, 4), "se": round(se, 4),
                         "z": round(z, 3), "p_raw": p})

        for row, ph in zip(tmp, apply_holm(raw_p)):
            row["p_holm"] = round(ph, 4)
            row["sig"]    = stars(ph)
            contrast_rows.append(row)

        min_n = fam_df.groupby("condition")["run_id"].nunique().min()
        parts = [f"converged={converged}"]
        if fallback:    parts.append("intercept-only fallback")
        if n_missing:   parts.append(f"{n_missing} missing initial_level → family mean")
        if min_n < 10:  parts.append(f"LOW POWER: min n/condition={min_n}")
        parts.append("exploratory — Holm-corrected across 4 three-way terms")
        summaries.append((fam, ", ".join(parts)))

    return pd.DataFrame(contrast_rows), summaries


def write_moderation_tex(mdf, path):
    conds_non_ref = [c for c in CONDITIONS if c != "BASELINE"]
    n_rows        = len(conds_non_ref)

    lines = [
        r"\begin{table}[ht]", r"\centering",
        (r"\caption{Three-way round $\times$ condition $\times$ initial cooperation"
         r" interaction terms from moderation analysis, by model family."
         r" Model per family: \texttt{contribution $\sim$ round $\times$ condition"
         r" $\times$ initial\_level\_c}, random intercept and round slope per run."
         r" \texttt{initial\_level\_c} is the agent's contribution in round~1 of"
         r" \textsc{Pure\_Baseline}, mean-centred within family."
         r" Reference condition: \textsc{Baseline}."
         r" Coefficients indicate whether the round$\times$initial-level relationship"
         r" differs from Baseline in each condition."
         r" $p$-values Holm-corrected across the four non-baseline conditions"
         r" within each family. " + CAPTION_STARS + "}"),
        r"\label{tab:initial_level_moderation}",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Family & Condition (vs.\ Baseline) & $\hat{\beta}$ & SE & $z$ & $p_\text{Holm}$ \\",
        r"\midrule",
    ]
    for i, fam in enumerate(FAMILIES):
        if i > 0: lines.append(r"\midrule")
        fam_rows = mdf[mdf["family"] == fam]
        lines.append(rf"\multirow{{{len(fam_rows)}}}{{*}}{{{fam}}}")
        for _, row in fam_rows.iterrows():
            cond_tex_s = cond_tex(row["condition_vs_baseline"])
            p_tex      = fmt_p(row["p_holm"]) + sig_sup(row["sig"])
            marginal   = r" $^{\dagger}$" if (not pd.isna(row["p_holm"])
                                               and row["p_holm"] < .10
                                               and row["p_holm"] >= .05) else ""
            lines.append(
                rf" & {cond_tex_s}"
                rf" & {fmt_n(row['estimate'])}"
                rf" & {fmt_n(row['se'])}"
                rf" & {fmt_n(row['z'])}"
                rf" & {p_tex} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5 — DISPERSION (CONVERGENCE CHECK)
# SD of contribution across agents, final 5 rounds, per family×condition
# Output: mean ± SE across runs
# ═══════════════════════════════════════════════════════════════════════════════

def test_dispersion(df_agents):
    final5 = df_agents[df_agents["round"] >= 16]

    # SD across agents per (run_id, round)
    per_round = (
        final5.groupby(["family", "condition", "run_id", "round"])["contrib"]
        .std(ddof=1)
        .reset_index()
        .rename(columns={"contrib": "sd"})
    )

    # Average SD over final 5 rounds per run
    per_run = (
        per_round.groupby(["family", "condition", "run_id"])["sd"]
        .mean()
        .reset_index()
    )

    # Mean ± SE across runs per family×condition
    agg = (
        per_run.groupby(["family", "condition"])["sd"]
        .agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        .reset_index()
    )
    agg["mean"] = agg["mean"].round(3)
    agg["se"]   = agg["se"].round(3)
    return agg


def write_dispersion_tex(disp_df, path):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        (r"\caption{Dispersion of individual contributions (SD across agents)"
         r" in the final five rounds, by model family and condition."
         r" Values are mean $\pm$ SE across runs."
         r" Lower SD indicates convergence toward a shared contribution level.}"),
        r"\label{tab:dispersion}",
        r"\small",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Family & Condition & SD (mean $\pm$ SE) \\",
        r"\midrule",
    ]
    for i, fam in enumerate(FAMILIES):
        if i > 0: lines.append(r"\midrule")
        fam_rows = disp_df[disp_df["family"] == fam]
        lines.append(rf"\multirow{{{len(CONDITIONS)}}}{{*}}{{{fam}}}")
        for cond in CONDITIONS:
            row = fam_rows[fam_rows["condition"] == cond]
            if len(row) == 0:
                cell = "---"
            else:
                m, s = float(row["mean"]), float(row["se"])
                cell = rf"{m:.3f} $\pm$ {s:.3f}"
            lines.append(rf" & {cond_tex(cond)} & {cell} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(FIG_ROOT, exist_ok=True)

    print("Loading data...")
    df        = build_dataframe()
    df_agents = build_agent_dataframe()
    print(f"  Run-level:   {len(df):,} rows, {df['run_id'].nunique()} runs")
    print(f"  Agent-level: {len(df_agents):,} rows")

    # ── Test 1: Level contrasts ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1 — LEVEL ANALYSIS")
    print("=" * 70)
    means_df, level_cdf, level_summ = test_level_contrasts(df)
    for fam, s in level_summ:
        print(f"  {fam}: {s}")
    sig1 = level_cdf[level_cdf["p_holm"] < .05]
    print(f"  Significant (p<.05): {len(sig1)}/{len(level_cdf)}")
    if len(sig1):
        print(sig1[["family", "contrast", "estimate", "z", "p_holm", "sig"]]
              .to_string(index=False))

    means_df.to_csv(os.path.join(FIG_ROOT, "level_condition_means.csv"), index=False)
    level_cdf.to_csv(os.path.join(FIG_ROOT, "level_contrasts.csv"), index=False)
    write_level_tex(means_df, level_cdf, os.path.join(FIG_ROOT, "level_contrasts.tex"))

    # ── Test 2: Slope contrasts ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2 — SLOPE ANALYSIS")
    print("=" * 70)
    slope_df, slope_cdf, slope_summ = test_slope_contrasts(df)
    for fam, s in slope_summ:
        print(f"  {fam}: {s}")
    sig2 = slope_cdf[slope_cdf["p_holm"] < .05]
    print(f"  Significant (p<.05): {len(sig2)}/{len(slope_cdf)}")
    if len(sig2):
        print(sig2[["family", "contrast", "estimate", "z", "p_holm", "sig"]]
              .to_string(index=False))

    slope_df.to_csv(os.path.join(FIG_ROOT, "slope_condition_estimates.csv"), index=False)
    slope_cdf.to_csv(os.path.join(FIG_ROOT, "slope_contrasts.csv"), index=False)
    write_slope_tex(slope_df, slope_cdf, os.path.join(FIG_ROOT, "slope_contrasts.tex"))

    # ── Test 3: Omnibus LRT ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3 — CROSS-MODEL CONSISTENCY (OMNIBUS LRT)")
    print("=" * 70)
    omni_df, omni_summ = test_omnibus(df)
    for s in omni_summ:
        print(f"  {s}")
    print(omni_df[["test", "model", "df", "chi2", "p", "sig"]].to_string(index=False))

    omni_df.to_csv(os.path.join(FIG_ROOT, "omnibus_interactions.csv"), index=False)
    write_omnibus_tex(omni_df, os.path.join(FIG_ROOT, "omnibus_interactions.tex"))

    # ── Test 4: Moderation ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4 — INITIAL-COOPERATION MODERATION")
    print("=" * 70)
    mod_df, mod_summ = test_moderation(df_agents)
    for fam, s in mod_summ:
        print(f"  {fam}: {s}")
    if len(mod_df):
        sig4 = mod_df[mod_df["p_holm"] < .05]
        marg4 = mod_df[(mod_df["p_holm"] >= .05) & (mod_df["p_holm"] < .10)]
        print(f"  Significant (p<.05): {len(sig4)}/{len(mod_df)}")
        print(f"  Marginal   (p<.10): {len(marg4)}/{len(mod_df)}")
        if len(sig4) + len(marg4):
            print((sig4 if len(sig4) else marg4)[
                ["family", "condition_vs_baseline", "estimate", "z", "p_holm", "sig"]
            ].to_string(index=False))

    mod_df.to_csv(os.path.join(FIG_ROOT, "initial_level_moderation.csv"), index=False)
    write_moderation_tex(mod_df, os.path.join(FIG_ROOT, "initial_level_moderation.tex"))

    # ── Test 5: Dispersion ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5 — DISPERSION (CONVERGENCE CHECK)")
    print("=" * 70)
    disp_df = test_dispersion(df_agents)
    pivot = disp_df.pivot(index="condition", columns="family",
                          values="mean").reindex(CONDITIONS)[FAMILIES]
    print(pivot.round(3).to_string())

    disp_df.to_csv(os.path.join(FIG_ROOT, "dispersion.csv"), index=False)
    write_dispersion_tex(disp_df, os.path.join(FIG_ROOT, "dispersion.tex"))

    print("\n" + "=" * 70)
    print(f"All outputs saved to: {FIG_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavioral mixed-effects statistical tests.")
    parser.add_argument("--models", "-m", nargs="+", default=["gpt", "llama", "mistral", "qwen"],
                        help="Model keys to include (default: the 7B/8B set).")
    parser.add_argument("--model-labels", nargs="+", default=[],
                        metavar="KEY=FAMILY",
                        help="Override family labels, e.g. --model-labels gpt=GPT. "
                             "Unlisted known keys fall back to a built-in default; "
                             "unknown keys fall back to the raw key.")
    parser.add_argument("--ref-family", default=None,
                        help="Reference family for contrasts (default: the first model's family).")
    parser.add_argument("--variant", "-v", default="local", help="Data variant (default: local).")
    parser.add_argument("--results-dir", default=RESULTS,
                        help="Results root (default: the main results/ dir).")
    parser.add_argument("--out-dir", "-o", default=FIG_ROOT, help="Output directory.")
    args = parser.parse_args()

    labels = dict(DEFAULT_LABELS)
    for pair in args.model_labels:
        key, _, label = pair.partition("=")
        labels[key] = label

    MODELS   = args.models
    LABELS   = {k: labels.get(k, k) for k in MODELS}
    FAMILIES = [LABELS[k] for k in MODELS]
    REF_FAM  = args.ref_family if args.ref_family else FAMILIES[0]
    RESULTS  = args.results_dir
    VARIANT  = args.variant
    FIG_ROOT = args.out_dir

    print(f"Models      : {MODELS}")
    print(f"Families    : {FAMILIES}")
    print(f"Ref family  : {REF_FAM}")
    print(f"Variant     : {VARIANT}")
    print(f"Results dir : {RESULTS}")
    print(f"Out dir     : {FIG_ROOT}")

    main()
