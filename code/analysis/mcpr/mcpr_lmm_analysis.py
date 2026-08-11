"""
mcpr_lmm_analysis.py
--------------------
Mixed-effects longitudinal models examining MCPR effects on contribution
in the SoNoLiSi public goods game (7B models only, N=12, G=4).

UNIT OF ANALYSIS
  One row per (run × round).  A "run" (run_id) is one simulated community:
    run_id = model_family + "_s" + seed + "_" + condition + "_MCPR" + mcpr
  Each run is observed 20 times (rounds 1-20). run_id is the random-effects
  grouping variable.

DATA SOURCES
  MCPR=0.4  results/{model}/local/seed{s}/log_*.json            (seeds 42-52)
  MCPR=0.5  code/results/{model}/local_groupsizevary/N12_G4_MCPR0.5/ (42-51)
  MCPR=0.8  same path with MCPR0.8  (Mistral + Qwen only; Llama absent)

  NOTE: in MCPR=0.4 logs, the mcpr field is absent (experiment predates the
  groupsizevary script). We impute 0.4 because G=4, multiplier=1.6,
  MCPR = multiplier/G = 1.6/4 = 0.4, verified in SoNoLiSi_v5_os_local.py.

FIXED EFFECTS (group_size=4 and community_size=12 are constant -> omitted)
  round       : continuous, centred at mean (round_c = round - 10.5)
  mcpr        : 3-level factor {0.4*, 0.5, 0.8}  (* = reference)
  model_family: 3-level factor {Llama*, Mistral, Qwen}
  condition   : 5-level factor {PURE_BASELINE*, BASELINE, NO_SELECTION,
                                NO_DISCUSSION, FULL}

RANDOM EFFECTS
  (1 + round_c | run_id)   random intercept + random slope for round

MODEL 1 (primary)
  mean_contribution ~ round_c * C(mcpr) * C(family)
                    + round_c * C(condition)
                    + (1 + round_c | run_id)

  Key terms:
    round_c                          - average temporal trend
    C(mcpr)[T.x]                     - MCPR main effect
    round_c:C(mcpr)[T.x]             - MCPR × trajectory
    round_c:C(mcpr)[T.x]:C(family)  - MCPR × trajectory × family (PRIMARY)

MODEL 2 (secondary — does MCPR × condition interact?)
  Attempt full four-way; fall back to additive MCPR×family + MCPR×condition
  if singular:
    mean_contribution ~ round_c * C(mcpr) * C(family)
                      + round_c * C(mcpr) * C(condition)
                      + (1 + round_c | run_id)

OUTPUTS  (figures/local/mcpr/stats/)
  model1_summary.txt / model1_coefs.csv
  model2_summary.txt / model2_coefs.csv  (or model2_reduced_*)
  emm_model1.csv  - estimated marginal means: mcpr × family × round
  emm_model2.csv  - estimated marginal means: mcpr × condition × round
  contrasts_mcpr_by_family.csv   (Holm-corrected slope contrasts)
  contrasts_mcpr_by_condition.csv
  diagnostics.txt
  plots/  predicted_model1_by_family.pdf
          predicted_model1_by_condition.pdf
          predicted_model2_by_family_condition.pdf
          residuals_model1.pdf
          residuals_model2.pdf
"""

import json, glob, os, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

MAIN_RESULTS = "/data3/rasimura/social-norm-evo/results"
CODE_RESULTS = "/data3/rasimura/social-norm-evo/code/results"
OUT_DIR      = "/data3/rasimura/social-norm-evo/figures/local/mcpr/stats"
PLOT_DIR     = os.path.join(OUT_DIR, "plots")

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_7B = ["llama", "mistral", "qwen"]
FAMILY_LABELS = {"llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline", "BASELINE": "Baseline",
    "NO_SELECTION":  "No Selection",  "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
SEEDS_MAIN = list(range(42, 53))
SEEDS_CODE = list(range(42, 52))

MCPR_COLORS   = {"0.4": "#555555", "0.5": "#1E88E5", "0.8": "#E53935"}
FAMILY_COLORS = {"Llama": "#1f77b4", "Mistral": "#d62728", "Qwen": "#2ca02c"}
ROUND_MEAN    = 10.5   # mean of rounds 1-20

# ─── 1. Data loading ──────────────────────────────────────────────────────────

def load_logs(pattern):
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
            best[d["condition"]] = d
        except Exception:
            continue
    return best


def extract_rows(d, model, seed, mcpr_val):
    rows = []
    for r in d["round_logs"]:
        contribs = list(r["contributions"].values())
        if not contribs:
            continue
        rows.append({
            "model_family":      FAMILY_LABELS[model],
            "seed":              int(seed),
            "condition":         d["condition"],
            "mcpr":              str(mcpr_val),
            "round":             int(r["round"]),
            "mean_contribution": float(np.mean(contribs)),
            "n_agents":          len(contribs),
        })
    return rows


def build_dataframe():
    rows = []

    # MCPR = 0.4: main results (G=4, N=12, multiplier=1.6 → MCPR=0.4)
    for model in MODELS_7B:
        for seed in SEEDS_MAIN:
            pat  = os.path.join(MAIN_RESULTS, model, "local",
                                f"seed{seed}", "log_*.json")
            logs = load_logs(pat)
            for d in logs.values():
                rows.extend(extract_rows(d, model, seed, 0.4))

    # MCPR = 0.5 and 0.8: groupsizevary experiments
    for model in MODELS_7B:
        for mcpr_str, mcpr_val in [("0.5", 0.5), ("0.8", 0.8)]:
            tag = f"N12_G4_MCPR{mcpr_str}"
            for seed in SEEDS_CODE:
                pat  = os.path.join(CODE_RESULTS, model, "local_groupsizevary",
                                    tag, f"seed{seed}", "log_*.json")
                logs = load_logs(pat)
                for d in logs.values():
                    rows.extend(extract_rows(d, model, seed, mcpr_val))

    df = pd.DataFrame(rows)

    # Categorical dtypes
    df["mcpr_f"]   = pd.Categorical(df["mcpr"],
                                     categories=["0.4", "0.5", "0.8"])
    df["family_f"] = pd.Categorical(df["model_family"],
                                     categories=["Llama", "Mistral", "Qwen"])
    df["cond_f"]   = pd.Categorical(df["condition"],
                                     categories=CONDITIONS)

    # Cell-means factor: only the 8 observed (family, mcpr) combinations.
    # Llama × MCPR=0.8 does not exist, so the full C(family)*C(mcpr) interaction
    # has a structural zero column → singular design matrix. Using this combined
    # factor avoids that: each level is a distinct, observed cell.
    FM_CATS = [
        "Llama_MCPR0.4", "Llama_MCPR0.5",
        "Mistral_MCPR0.4", "Mistral_MCPR0.5", "Mistral_MCPR0.8",
        "Qwen_MCPR0.4",   "Qwen_MCPR0.5",   "Qwen_MCPR0.8",
    ]
    df["family_mcpr_f"] = pd.Categorical(
        df["model_family"] + "_MCPR" + df["mcpr"],
        categories=FM_CATS,
    )

    # Centred round (improves LMM convergence; intercept = contribution at round 10.5)
    df["round_c"] = df["round"] - ROUND_MEAN

    # Unique run identifier (level-2 grouping variable for random effects)
    df["run_id"] = (df["model_family"] + "_s" + df["seed"].astype(str)
                    + "_" + df["condition"] + "_MCPR" + df["mcpr"].astype(str))

    print(f"Dataset: {len(df):,} rows, {df['run_id'].nunique()} unique runs")
    print(f"MCPR counts:\n{df.groupby('mcpr')['run_id'].nunique().to_string()}")
    print(f"Family × MCPR run counts:")
    print(df.groupby(["model_family", "mcpr"])["run_id"].nunique().to_string())
    return df


FM_CATS = [
    "Llama_MCPR0.4", "Llama_MCPR0.5",
    "Mistral_MCPR0.4", "Mistral_MCPR0.5", "Mistral_MCPR0.8",
    "Qwen_MCPR0.4",   "Qwen_MCPR0.5",   "Qwen_MCPR0.8",
]


# ─── 2. Model fitting ─────────────────────────────────────────────────────────

def fit_model(formula, df, label, re_formula="~round_c", method="lbfgs",
              fallback_re=None):
    """
    Fit a MixedLM. If convergence fails or random effects are singular,
    try the fallback_re formula (e.g. None for intercept-only random effects).
    Returns (result, re_used, warnings_str).
    """
    diag_notes = []
    re_used    = re_formula

    for re_f in ([re_formula] + ([fallback_re] if fallback_re else [])):
        try:
            if re_f:
                model = smf.mixedlm(formula, data=df,
                                    groups=df["run_id"],
                                    re_formula=re_f)
            else:
                model = smf.mixedlm(formula, data=df, groups=df["run_id"])

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(method=method, maxiter=2000)
                for w in caught:
                    msg = str(w.message)
                    if "Convergence" in msg or "singular" in msg.lower():
                        diag_notes.append(f"WARNING ({re_f}): {msg}")

            re_used = re_f
            # Check for singular random effects (variance close to zero)
            if hasattr(result, "cov_re"):
                cov_re = np.array(result.cov_re)
                min_eig = np.linalg.eigvalsh(cov_re).min()
                if min_eig < 1e-6:
                    diag_notes.append(
                        f"Singular RE ({re_f}): smallest eigenvalue = {min_eig:.2e}. "
                        "Random effects may not be reliably estimated.")
            print(f"  [{label}] Fitted with re_formula='{re_f}', "
                  f"AIC={result.aic:.1f}, BIC={result.bic:.1f}")
            return result, re_used, "\n".join(diag_notes)

        except Exception as e:
            diag_notes.append(f"FAILED with re='{re_f}': {e}")
            if re_f == re_formula and fallback_re:
                print(f"  [{label}] re='{re_f}' failed, trying fallback...")
                continue
            raise

    raise RuntimeError(f"[{label}] All fitting attempts failed.")


# ─── 3. Coefficient table ─────────────────────────────────────────────────────

def coef_table(result):
    """Return a clean coefficient DataFrame from a MixedLM result."""
    fe   = result.fe_params
    bse  = result.bse_fe
    zstat = fe / bse
    pvals = 2 * scipy_stats.norm.sf(np.abs(zstat))
    ci    = result.conf_int().loc[fe.index]
    df_out = pd.DataFrame({
        "coef":   fe.values,
        "se":     bse.values,
        "z":      zstat.values,
        "p":      pvals,
        "ci_lo":  ci.iloc[:, 0].values,
        "ci_hi":  ci.iloc[:, 1].values,
    }, index=fe.index)
    df_out["sig"] = df_out["p"].apply(
        lambda p: "***" if p < .001 else ("**" if p < .01 else
                  ("*" if p < .05 else ("." if p < .1 else ""))))
    return df_out


# ─── 4. Estimated marginal means ──────────────────────────────────────────────

def emm_predictions(result, grid_df):
    """
    Compute fixed-effect predictions (EMMs) for rows in grid_df.
    Uses patsy to build the design matrix from the fitted model's DesignInfo.

    Returns grid_df extended with columns: emm, se, ci_lo, ci_hi.
    NaN is inserted for any row whose variance estimate is non-finite
    (e.g., cells with no observed data → quasi-separation → huge SE).
    """
    design_info = result.model.data.design_info
    (X_new,) = build_design_matrices([design_info], grid_df)
    X = np.asarray(X_new)

    beta = result.fe_params.values

    # Extract FE-only block from the full cov_params (which includes RE variances)
    fe_idx = result.fe_params.index
    cov_fe = result.cov_params().loc[fe_idx, fe_idx].values

    pred = X @ beta
    var  = np.einsum("ij,jk,ik->i", X, cov_fe, X)   # diag(X Cov X^T)
    # Mark cells with implausibly large variance as inestimable (quasi-separation)
    bad  = (~np.isfinite(var)) | (var > 1e6)
    var  = np.where(bad, np.nan, var)
    se   = np.sqrt(np.clip(var, 0, None))
    pred = np.where(bad, np.nan, pred)

    out = grid_df.copy()
    out["emm"]   = pred
    out["se"]    = se
    out["ci_lo"] = pred - 1.96 * se
    out["ci_hi"] = pred + 1.96 * se
    n_bad = int(bad.sum())
    if n_bad:
        print(f"  NOTE: {n_bad} prediction row(s) inestimable (quasi-separation) → set to NaN")
    return out


def make_prediction_grid_m1():
    """
    EMM grid for Model 1 (cell-means coding via family_mcpr_f).
    Only the 8 observed (family, mcpr) combinations × 5 conditions × 20 rounds.
    """
    rows = []
    rounds = np.arange(1, 21)
    for fm in FM_CATS:
        fam, mcpr = fm.split("_MCPR")
        for cond in CONDITIONS:
            for rnd in rounds:
                rows.append({
                    "family_mcpr_f": fm,
                    "cond_f":        cond,
                    "round_c":       rnd - ROUND_MEAN,
                    "round":         rnd,
                    "model_family":  fam,
                    "mcpr":          mcpr,
                    "condition":     cond,
                    "mcpr_f":        mcpr,
                    "family_f":      fam,
                })
    grid = pd.DataFrame(rows)
    grid["family_mcpr_f"] = pd.Categorical(grid["family_mcpr_f"], categories=FM_CATS)
    grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   categories=["0.4","0.5","0.8"])
    grid["family_f"] = pd.Categorical(grid["family_f"], categories=["Llama","Mistral","Qwen"])
    grid["cond_f"]   = pd.Categorical(grid["cond_f"],   categories=CONDITIONS)
    return grid


def make_prediction_grid_m2():
    """
    EMM grid for Model 2 (mcpr_f × cond_f + family_f).
    All 3 MCPR × 5 conditions × 3 families × 20 rounds.
    Llama at MCPR=0.8 is estimated additively (no interaction), so it's included.
    """
    rows = []
    rounds = np.arange(1, 21)
    for mcpr in ["0.4", "0.5", "0.8"]:
        for cond in CONDITIONS:
            for fam in ["Llama", "Mistral", "Qwen"]:
                for rnd in rounds:
                    rows.append({
                        "mcpr_f":        mcpr,
                        "cond_f":        cond,
                        "family_f":      fam,
                        "round_c":       rnd - ROUND_MEAN,
                        "round":         rnd,
                        "model_family":  fam,
                        "condition":     cond,
                        "mcpr":          mcpr,
                        "family_mcpr_f": fam + "_MCPR" + mcpr,  # not in formula but harmless
                    })
    grid = pd.DataFrame(rows)
    grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   categories=["0.4","0.5","0.8"])
    grid["family_f"] = pd.Categorical(grid["family_f"], categories=["Llama","Mistral","Qwen"])
    grid["cond_f"]   = pd.Categorical(grid["cond_f"],   categories=CONDITIONS)
    return grid


def marginalize_emm(emm_full, groupby_cols):
    """Average EMMs over nuisance dimensions; skips NaN cells (inestimable)."""
    def rms_se(x):
        x = x.dropna()
        return np.sqrt(np.mean(x**2)) if len(x) > 0 else np.nan

    agg = emm_full.groupby(groupby_cols, observed=True).agg(
        emm   =("emm",   "mean"),
        se    =("se",    rms_se),
        ci_lo =("ci_lo", "mean"),
        ci_hi =("ci_hi", "mean"),
    ).reset_index()
    return agg


# ─── 5. Contrasts (slope over round by MCPR, delta-method SEs, Holm p-adj) ────

def _compute_slope(result, grid_pts, w_r1, w_r20):
    """
    Compute slope = (E[y|round=20] - E[y|round=1]) / 19 for a set of prediction
    rows via the delta method. Returns (slope, se).
    """
    design_info = result.model.data.design_info
    fe_idx      = result.fe_params.index
    cov_fe      = result.cov_params().loc[fe_idx, fe_idx].values

    (X,) = build_design_matrices([design_info], grid_pts)
    X = np.asarray(X)

    beta      = result.fe_params.values
    slope_r1  = w_r1  @ X @ beta
    slope_r20 = w_r20 @ X @ beta
    slope     = (slope_r20 - slope_r1) / 19.0

    c_vec    = (w_r20 - w_r1) @ X / 19.0
    se_slope = float(np.sqrt(np.clip(c_vec @ cov_fe @ c_vec, 0, None)))
    return float(slope), se_slope


def _holm_finish(cdf):
    """Add p_holm and sig columns in place."""
    valid = cdf["p_raw"].notna()
    if valid.any():
        _, p_adj, _, _ = multipletests(cdf.loc[valid, "p_raw"], method="holm")
        cdf.loc[valid, "p_holm"] = p_adj
    else:
        cdf["p_holm"] = np.nan
    cdf["sig"] = cdf["p_holm"].apply(
        lambda p: "***" if p < .001 else ("**" if p < .01 else
                  ("*" if p < .05 else ("." if p < .1 else "ns")))
        if pd.notna(p) else "")
    cdf["p_raw"]  = cdf["p_raw"].round(4)
    cdf["p_holm"] = cdf["p_holm"].round(4)
    return cdf


def slope_contrasts_m1(result):
    """
    Model 1 (cell-means family_mcpr_f): pairwise MCPR slope contrasts within
    each model family. Slopes are averaged over conditions.

    For each (family, mcpr) cell, the slope = (EMM[r=20] - EMM[r=1]) / 19,
    averaged over the 5 conditions.
    """
    # Map each family to its observed MCPR levels
    family_mpcrs = {
        "Llama":   ["0.4", "0.5"],
        "Mistral": ["0.4", "0.5", "0.8"],
        "Qwen":    ["0.4", "0.5", "0.8"],
    }

    slope_rows = []
    for fam, mpcrs in family_mpcrs.items():
        for mcpr in mpcrs:
            fm = f"{fam}_MCPR{mcpr}"
            # Build avg-over-condition grid for round 1 and round 20
            pts = []
            for cond in CONDITIONS:
                for rnd in [1, 20]:
                    pts.append({
                        "family_mcpr_f": fm,
                        "cond_f":        cond,
                        "round_c":       rnd - ROUND_MEAN,
                        "round":         rnd,
                        "model_family":  fam,
                        "mcpr":          mcpr,
                        "condition":     cond,
                        "mcpr_f":        mcpr,
                        "family_f":      fam,
                    })
            grid = pd.DataFrame(pts)
            grid["family_mcpr_f"] = pd.Categorical(grid["family_mcpr_f"], FM_CATS)
            grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   ["0.4","0.5","0.8"])
            grid["family_f"] = pd.Categorical(grid["family_f"], ["Llama","Mistral","Qwen"])
            grid["cond_f"]   = pd.Categorical(grid["cond_f"],   CONDITIONS)

            n  = len(grid)
            nc = len(CONDITIONS)
            w_r1  = np.zeros(n)
            w_r20 = np.zeros(n)
            for i in range(nc):
                w_r1[2*i]    = 1.0 / nc
                w_r20[2*i+1] = 1.0 / nc

            slope, se = _compute_slope(result, grid, w_r1, w_r20)
            slope_rows.append({"model_family": fam, "mcpr": mcpr,
                                "slope": slope, "se": se})

    slope_df = pd.DataFrame(slope_rows)

    # Pairwise MCPR contrasts within each family
    contrast_rows = []
    for fam, mpcrs in family_mpcrs.items():
        pairs = [(a, b) for i, a in enumerate(["0.4","0.5","0.8"])
                 for b in ["0.4","0.5","0.8"][i+1:]
                 if a in mpcrs and b in mpcrs]
        for m1, m2 in pairs:
            r1 = slope_df[(slope_df["model_family"] == fam) & (slope_df["mcpr"] == m1)]
            r2 = slope_df[(slope_df["model_family"] == fam) & (slope_df["mcpr"] == m2)]
            est = float(r2["slope"]) - float(r1["slope"])
            se  = float(np.sqrt(float(r1["se"])**2 + float(r2["se"])**2))
            z   = est / se if se > 0 else np.nan
            p   = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
            contrast_rows.append({
                "model_family": fam, "contrast": f"MCPR{m2} - MCPR{m1}",
                "slope_m1": round(float(r1["slope"]), 4),
                "slope_m2": round(float(r2["slope"]), 4),
                "estimate": round(est, 4), "se": round(se, 4),
                "z": round(z, 3) if not np.isnan(z) else np.nan,
                "p_raw": p,
            })
        # Note missing MCPR=0.8 pairs for Llama
        if fam == "Llama":
            for m2 in ["0.8"]:
                for m1 in ["0.4", "0.5"]:
                    contrast_rows.append({
                        "model_family": fam, "contrast": f"MCPR{m2} - MCPR{m1}",
                        "slope_m1": np.nan, "slope_m2": np.nan,
                        "estimate": np.nan, "se": np.nan, "z": np.nan,
                        "p_raw": np.nan,
                    })

    cdf = pd.DataFrame(contrast_rows)
    return _holm_finish(cdf)


def slope_contrasts_m2(result):
    """
    Model 2 (mcpr_f × cond_f + family_f): pairwise MCPR slope contrasts
    within each condition, averaged over families.
    """
    # For Model 2, family is additive — average over all 3 families
    # (including Llama at MCPR=0.8, which is an additive extrapolation, not a
    # missing-cell problem in this model because there's no family×mcpr interaction)
    slope_rows = []
    for cond in CONDITIONS:
        for mcpr in ["0.4", "0.5", "0.8"]:
            pts = []
            for fam in ["Llama", "Mistral", "Qwen"]:
                for rnd in [1, 20]:
                    pts.append({
                        "mcpr_f":    mcpr,
                        "cond_f":    cond,
                        "family_f":  fam,
                        "round_c":   rnd - ROUND_MEAN,
                        "round":     rnd,
                        "model_family": fam,
                        "condition": cond,
                        "mcpr":      mcpr,
                    })
            grid = pd.DataFrame(pts)
            grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   ["0.4","0.5","0.8"])
            grid["family_f"] = pd.Categorical(grid["family_f"], ["Llama","Mistral","Qwen"])
            grid["cond_f"]   = pd.Categorical(grid["cond_f"],   CONDITIONS)

            n   = len(grid)
            nf  = 3
            w_r1  = np.zeros(n)
            w_r20 = np.zeros(n)
            for i in range(nf):
                w_r1[2*i]    = 1.0 / nf
                w_r20[2*i+1] = 1.0 / nf

            slope, se = _compute_slope(result, grid, w_r1, w_r20)
            slope_rows.append({"condition": cond, "mcpr": mcpr,
                                "slope": slope, "se": se})

    slope_df = pd.DataFrame(slope_rows)

    contrast_rows = []
    mcpr_pairs = [("0.4","0.5"), ("0.4","0.8"), ("0.5","0.8")]
    for cond in CONDITIONS:
        sub = slope_df[slope_df["condition"] == cond]
        for m1, m2 in mcpr_pairs:
            r1 = sub[sub["mcpr"] == m1]
            r2 = sub[sub["mcpr"] == m2]
            est = float(r2["slope"]) - float(r1["slope"])
            se  = float(np.sqrt(float(r1["se"])**2 + float(r2["se"])**2))
            z   = est / se if se > 0 else np.nan
            p   = float(2 * scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
            contrast_rows.append({
                "condition": cond, "contrast": f"MCPR{m2} - MCPR{m1}",
                "slope_m1": round(float(r1["slope"]), 4),
                "slope_m2": round(float(r2["slope"]), 4),
                "estimate": round(est, 4), "se": round(se, 4),
                "z": round(z, 3) if not np.isnan(z) else np.nan,
                "p_raw": p,
            })

    cdf = pd.DataFrame(contrast_rows)
    return _holm_finish(cdf)


# ─── 6. Diagnostic plots ──────────────────────────────────────────────────────

def plot_residuals(result, label, out_path):
    fitted   = np.asarray(result.fittedvalues)
    resid    = np.asarray(result.resid)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Residuals vs fitted
    axes[0].scatter(fitted, resid, alpha=0.15, s=6, color="#1f77b4")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Fitted values"); axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")
    axes[0].grid(True, alpha=0.3)

    # QQ plot
    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid)
    axes[1].scatter(osm, osr, alpha=0.2, s=6, color="#d62728")
    axes[1].plot(osm, slope*np.array(osm)+intercept, color="black", lw=1.2)
    axes[1].set_xlabel("Theoretical quantiles"); axes[1].set_ylabel("Sample quantiles")
    axes[1].set_title("Normal Q-Q")
    axes[1].grid(True, alpha=0.3)

    # Residual histogram
    axes[2].hist(resid, bins=50, color="#2ca02c", alpha=0.7, edgecolor="none")
    axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Count")
    axes[2].set_title("Residual Distribution")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Residual diagnostics — {label}", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── 7. Prediction plots ──────────────────────────────────────────────────────

def _sub(emm_df, **kwargs):
    """Filter emm_df by column==value pairs, skip if emm is all NaN."""
    mask = pd.Series(True, index=emm_df.index)
    for col, val in kwargs.items():
        mask &= (emm_df[col] == val)
    sub = emm_df[mask].sort_values("round")
    if sub.empty or sub["emm"].isna().all():
        return None
    return sub


def plot_emm_by_family(emm_df, label, out_path):
    """Predicted trajectories: one panel per family, lines coloured by MCPR."""
    families = ["Llama", "Mistral", "Qwen"]
    mpcrs    = ["0.4", "0.5", "0.8"]

    fig, axes = plt.subplots(1, len(families),
                             figsize=(5 * len(families), 4.5),
                             sharey=True)
    for ax, fam in zip(axes, families):
        for mcpr in mpcrs:
            sub = _sub(emm_df, model_family=fam, mcpr=mcpr)
            if sub is None: continue
            color = MCPR_COLORS[mcpr]
            ax.plot(sub["round"].to_numpy(), sub["emm"].to_numpy(),
                    color=color, linewidth=2, label=f"MCPR={mcpr}")
            ax.fill_between(sub["round"].to_numpy(),
                            sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(),
                            color=color, alpha=0.15)
        ax.set_title(fam, fontsize=13, fontweight="bold")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylim(0, 10.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("Predicted mean contribution", fontsize=12)

    handles = [plt.Line2D([0],[0], color=MCPR_COLORS[m], lw=2, label=f"MCPR={m}")
               for m in mpcrs]
    fig.legend(handles, [f"MCPR={m}" for m in mpcrs],
               loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{label} — predicted trajectories by model family", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_emm_by_condition(emm_df, label, out_path):
    """Predicted trajectories: one panel per condition, lines coloured by MCPR."""
    mpcrs = ["0.4", "0.5", "0.8"]
    fig, axes = plt.subplots(1, len(CONDITIONS),
                             figsize=(4.5 * len(CONDITIONS), 4.5),
                             sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        for mcpr in mpcrs:
            sub = _sub(emm_df, condition=cond, mcpr=mcpr)
            if sub is None: continue
            color = MCPR_COLORS[mcpr]
            ax.plot(sub["round"].to_numpy(), sub["emm"].to_numpy(),
                    color=color, linewidth=2, label=f"MCPR={mcpr}")
            ax.fill_between(sub["round"].to_numpy(),
                            sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(),
                            color=color, alpha=0.15)
        ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylim(0, 10.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Predicted mean contribution", fontsize=12)

    handles = [plt.Line2D([0],[0], color=MCPR_COLORS[m], lw=2, label=f"MCPR={m}")
               for m in mpcrs]
    fig.legend(handles, [f"MCPR={m}" for m in mpcrs],
               loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{label} — predicted trajectories by condition", fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_emm_faceted(emm_df, label, out_path):
    """Faceted: rows=family, cols=condition, coloured by MCPR."""
    families = ["Llama", "Mistral", "Qwen"]
    mpcrs    = ["0.4", "0.5", "0.8"]

    fig, axes = plt.subplots(len(families), len(CONDITIONS),
                             figsize=(4.0 * len(CONDITIONS), 3.5 * len(families)),
                             sharex=True, sharey=True)
    for ri, fam in enumerate(families):
        for ci, cond in enumerate(CONDITIONS):
            ax = axes[ri, ci]
            for mcpr in mpcrs:
                sub = _sub(emm_df, model_family=fam, condition=cond, mcpr=mcpr)
                if sub is None: continue
                color = MCPR_COLORS[mcpr]
                ax.plot(sub["round"].to_numpy(), sub["emm"].to_numpy(),
                        color=color, linewidth=2, label=f"MCPR={mcpr}")
                ax.fill_between(sub["round"].to_numpy(),
                                sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(),
                                color=color, alpha=0.15)
            ax.set_ylim(0, 10.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=9)
            if ri == 0:
                ax.set_title(COND_LABELS[cond], fontsize=11, fontweight="bold")
            if ri == len(families) - 1:
                ax.set_xlabel("Round", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=11, fontweight="bold")

    handles = [plt.Line2D([0],[0], color=MCPR_COLORS[m], lw=2, label=f"MCPR={m}")
               for m in mpcrs]
    fig.legend(handles, [f"MCPR={m}" for m in mpcrs],
               loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{label}", fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── 8. Save helpers ──────────────────────────────────────────────────────────

def save_text(text, path):
    with open(path, "w") as f:
        f.write(text)
    print(f"  Saved: {path}")


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    diag_lines = []

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = build_dataframe()
    save_csv(df, os.path.join(OUT_DIR, "analysis_dataset.csv"))

    diag_lines.append("=== Dataset ===")
    diag_lines.append(f"Total rows  : {len(df):,}")
    diag_lines.append(f"Total runs  : {df['run_id'].nunique()}")
    diag_lines.append(f"Models      : {sorted(df['model_family'].unique())}")
    diag_lines.append(f"MCPR levels : {sorted(df['mcpr'].unique())}")
    diag_lines.append(f"Conditions  : {sorted(df['condition'].unique())}")
    diag_lines.append(f"Rounds      : {df['round'].min()} - {df['round'].max()}")
    diag_lines.append(f"\nNote: MCPR=0.8 unavailable for Llama (0 runs). "
                      "Model estimates for Llama at MCPR=0.8 extrapolate beyond observed data.")
    diag_lines.append(f"\nRun counts by family × MCPR:")
    diag_lines.append(
        df.groupby(["model_family","mcpr"])["run_id"].nunique().to_string())

    # ── Model 1: primary (cell-means coding) ─────────────────────────────────
    # C(mcpr_f) * C(family_f) has a structural zero column for Llama×MCPR=0.8
    # (no observations) → singular design matrix. Fix: use family_mcpr_f, a
    # single 8-level factor encoding only the observed (family, MCPR) cells.
    # Interpretation: each level's intercept + round slope = EMM for that cell,
    # averaged over conditions. Pairwise MCPR contrasts computed via delta method.
    print("\n[2] Fitting Model 1 (primary — cell-means family×MCPR)...")
    f1 = ("mean_contribution ~ "
          "round_c * C(family_mcpr_f, Treatment('Mistral_MCPR0.4')) "
          "+ round_c * C(cond_f, Treatment('PURE_BASELINE'))")

    result1, re1, warn1 = fit_model(f1, df, "Model1",
                                    re_formula="~round_c",
                                    fallback_re=None)

    coefs1 = coef_table(result1)
    diag_lines.append(f"\n=== Model 1 (cell-means family_mcpr × round + condition × round) ===")
    diag_lines.append(f"RE formula: {re1}")
    if warn1: diag_lines.append(warn1)
    llf1 = result1.llf
    k1   = len(result1.fe_params) + 3
    n1   = result1.nobs
    aic1 = (-2*llf1 + 2*k1) if np.isfinite(llf1) else float("nan")
    bic1 = (-2*llf1 + np.log(n1)*k1) if np.isfinite(llf1) else float("nan")
    diag_lines.append(f"AIC={aic1:.2f}, BIC={bic1:.2f}, LogLik={llf1:.4f}")
    if hasattr(result1, "cov_re"):
        diag_lines.append(f"RE covariance:\n{result1.cov_re}")

    save_text(str(result1.summary()), os.path.join(OUT_DIR, "model1_summary.txt"))
    save_csv(coefs1.reset_index().rename(columns={"index":"parameter"}),
             os.path.join(OUT_DIR, "model1_coefs.csv"))

    # ── Model 1 EMMs ─────────────────────────────────────────────────────────
    print("  Computing Model 1 EMMs...")
    grid1 = make_prediction_grid_m1()
    emm1_full = emm_predictions(result1, grid1)

    # Marginalise over condition for family × mcpr × round
    emm1_family = marginalize_emm(emm1_full, ["model_family", "mcpr", "round"])
    save_csv(emm1_family, os.path.join(OUT_DIR, "emm_model1_by_family.csv"))

    # Marginalise over family for condition × mcpr × round
    emm1_cond = marginalize_emm(emm1_full, ["condition", "mcpr", "round"])
    save_csv(emm1_cond, os.path.join(OUT_DIR, "emm_model1_by_condition.csv"))

    # ── Model 1 contrasts ─────────────────────────────────────────────────────
    print("  Computing slope contrasts for Model 1 (by family)...")
    cont1_fam = slope_contrasts_m1(result1)
    save_csv(cont1_fam, os.path.join(OUT_DIR, "contrasts_model1_mcpr_by_family.csv"))

    # ── Model 1 plots ─────────────────────────────────────────────────────────
    print("  Plotting Model 1...")
    plot_emm_by_family(emm1_family, "Model 1 — cell-means (family × MCPR)",
                       os.path.join(PLOT_DIR, "predicted_model1_by_family.pdf"))
    plot_emm_by_condition(emm1_cond, "Model 1 — cell-means",
                          os.path.join(PLOT_DIR, "predicted_model1_by_condition.pdf"))
    plot_residuals(result1, "Model 1",
                   os.path.join(PLOT_DIR, "residuals_model1.pdf"))

    # ── Model 2: secondary (MCPR × condition, family additive) ────────────────
    # C(mcpr_f) × C(cond_f) is fully estimable (all 15 cells observed).
    # C(family_f) enters additively — no mcpr×family interaction needed here.
    # For Llama at MCPR=0.8: the model extrapolates additively (Llama main effect
    # + MCPR=0.8 main effect); this is noted in diagnostics.
    print("\n[3] Fitting Model 2 (secondary — MCPR × condition + family)...")
    f2 = ("mean_contribution ~ "
          "round_c * C(mcpr_f, Treatment('0.4')) * C(cond_f, Treatment('PURE_BASELINE')) "
          "+ round_c * C(family_f, Treatment('Mistral'))")

    result2, re2, warn2 = fit_model(f2, df, "Model2",
                                    re_formula="~round_c",
                                    fallback_re=None)
    m2_label = "model2"

    llf2 = result2.llf; k2 = len(result2.fe_params) + 3; n2 = result2.nobs
    aic2 = (-2*llf2 + 2*k2) if np.isfinite(llf2) else float("nan")
    bic2 = (-2*llf2 + np.log(n2)*k2) if np.isfinite(llf2) else float("nan")
    diag_lines.append(f"\n=== Model 2 (MCPR × condition × round + family × round) ===")
    diag_lines.append(f"RE formula: {re2}")
    if warn2: diag_lines.append(warn2)
    diag_lines.append(f"AIC={aic2:.2f}, BIC={bic2:.2f}, LogLik={llf2:.4f}")
    if hasattr(result2, "cov_re"):
        diag_lines.append(f"RE covariance:\n{result2.cov_re}")

    coefs2 = coef_table(result2)
    save_text(str(result2.summary()),
              os.path.join(OUT_DIR, f"{m2_label}_summary.txt"))
    save_csv(coefs2.reset_index().rename(columns={"index":"parameter"}),
             os.path.join(OUT_DIR, f"{m2_label}_coefs.csv"))

    # ── Model 2 EMMs ─────────────────────────────────────────────────────────
    print("  Computing Model 2 EMMs...")
    grid2 = make_prediction_grid_m2()
    emm2_full = emm_predictions(result2, grid2)

    # Full EMM: family × condition × mcpr × round
    emm2_all = marginalize_emm(emm2_full, ["model_family", "condition", "mcpr", "round"])
    save_csv(emm2_all, os.path.join(OUT_DIR, "emm_model2_by_family_condition.csv"))

    # Marginalise over family for condition × mcpr view
    emm2_cond = marginalize_emm(emm2_full, ["condition", "mcpr", "round"])
    save_csv(emm2_cond, os.path.join(OUT_DIR, "emm_model2_by_condition.csv"))

    # ── Model 2 contrasts ─────────────────────────────────────────────────────
    print("  Computing slope contrasts for Model 2 (by condition)...")
    cont2_cond = slope_contrasts_m2(result2)
    save_csv(cont2_cond,
             os.path.join(OUT_DIR, f"{m2_label}_contrasts_mcpr_by_condition.csv"))

    # ── Model 2 plots ─────────────────────────────────────────────────────────
    print("  Plotting Model 2...")
    plot_emm_faceted(emm2_all,
                     "Model 2 — predicted trajectories (family × condition)",
                     os.path.join(PLOT_DIR, f"predicted_{m2_label}_by_family_condition.pdf"))
    plot_emm_by_condition(emm2_cond, "Model 2 — MCPR × condition",
                          os.path.join(PLOT_DIR, f"predicted_{m2_label}_by_condition.pdf"))
    plot_residuals(result2, "Model 2",
                   os.path.join(PLOT_DIR, f"residuals_{m2_label}.pdf"))

    # ── Diagnostics file ──────────────────────────────────────────────────────
    diag_lines.append("\n=== Contribution distribution ===")
    diag_lines.append(f"Mean: {df['mean_contribution'].mean():.3f}")
    diag_lines.append(f"SD:   {df['mean_contribution'].std():.3f}")
    diag_lines.append(f"Min:  {df['mean_contribution'].min():.3f}")
    diag_lines.append(f"Max:  {df['mean_contribution'].max():.3f}")
    diag_lines.append(
        f"Fraction at floor (≤0.5): "
        f"{(df['mean_contribution'] <= 0.5).mean():.3f}")
    diag_lines.append(
        f"Fraction at ceiling (≥9.5): "
        f"{(df['mean_contribution'] >= 9.5).mean():.3f}")
    diag_lines.append(
        "\nNote: mean_contribution is bounded [0,10]. If floor/ceiling fractions "
        "are high, a Tobit or beta-regression may be preferred over LMM.")

    diag_lines.append("\n=== Key fixed effects (Model 1) — family_mcpr × round ===")
    key_terms = [p for p in coefs1.index
                 if "round_c" in p.lower() and "family_mcpr" in p.lower()]
    if key_terms:
        diag_lines.append(coefs1.loc[key_terms].to_string())

    diag_lines.append("\n=== Key fixed effects (Model 2) — mcpr × condition × round ===")
    key_terms2 = [p for p in coefs2.index
                  if "round_c" in p.lower() and "mcpr" in p.lower()
                  and "cond" in p.lower()]
    if key_terms2:
        diag_lines.append(coefs2.loc[key_terms2].to_string())

    save_text("\n".join(diag_lines), os.path.join(OUT_DIR, "diagnostics.txt"))

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("\nKey Model 1 terms (round × family_mcpr):")
    key = [p for p in coefs1.index
           if "round_c" in p.lower() and "family_mcpr" in p.lower()]
    if key:
        print(coefs1.loc[key, ["coef","se","z","p","sig"]].to_string())


if __name__ == "__main__":
    main()
