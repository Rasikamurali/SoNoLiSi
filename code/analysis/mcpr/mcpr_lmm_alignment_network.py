"""
mcpr_lmm_alignment_network.py
------------------------------
Mixed-effects longitudinal models examining MCPR effects on
perception-behavior alignment and network structure in SoNoLiSi
(7B models only, N=12, G=4, MCPR ∈ {0.4, 0.5, 0.8}).

DEPENDENT VARIABLES
  Alignment (mean over agents per run × round):
    IN_gap   — injunctive_norm  − actual_contribution  (positive = over-aspire)
    DN_gap   — descriptive_norm − actual_contribution  (positive = over-estimate peers)

  Network (aggregated over agents per run × round):
    spearman_r — Spearman r(incoming_weight, contribution) across agents
                 > 0 → network rewards cooperators
    gini       — Gini coefficient of incoming-weight distribution
                 > 0 → weight inequality (selective network)
    top_share  — fraction of total weight held by top-25%-contributing agents

NOTES
  PURE_BASELINE has no perceptions (agents don't discuss) →
    IN_gap / DN_gap are NaN for all PURE_BASELINE rows; those rows are dropped.
  PURE_BASELINE has uniform weights (1.0) → Gini≈0, top_share≈0.25, Spearman=NaN.
    For network models, PURE_BASELINE is the structural reference.

DATA SOURCES  (identical to mcpr_lmm_analysis.py)
  MCPR=0.4  results/{model}/local/seed{42-52}/log_*.json
  MCPR=0.5  code/results/{model}/local_groupsizevary/N12_G4_MCPR0.5/seed{42-51}/
  MCPR=0.8  same path, MCPR0.8  (Mistral + Qwen only)

MODELS (fitted per DV)
  Model 1: dv ~ round_c * C(family_mcpr_f) + round_c * C(cond_f)
             + (1 + round_c | run_id)
    family_mcpr_f: 8-level cell-means factor (avoids Llama×MCPR=0.8 empty cell)
    Condition reference:
      alignment DVs → BASELINE  (PURE_BASELINE has no perception data)
      network DVs   → PURE_BASELINE (flat-network baseline)

  Model 2: dv ~ round_c * C(mcpr_f) * C(cond_f) + round_c * C(family_f)
             + (1 + round_c | run_id)
    Same condition references as Model 1.

OUTPUTS  figures/local/mcpr/stats/alignment_network/{dv}/
  model1_summary.txt, model1_coefs.csv
  model2_summary.txt, model2_coefs.csv
  emm_model1_by_family.csv, emm_model1_by_condition.csv
  emm_model2_by_condition.csv
  contrasts_model1_mcpr_by_family.csv
  contrasts_model2_mcpr_by_condition.csv
  plots/predicted_model1_by_family.pdf
        predicted_model1_by_condition.pdf
        predicted_model2_by_condition.pdf
        predicted_model2_by_family_condition.pdf
        residuals_model1.pdf
        residuals_model2.pdf
"""

import json, glob, os, warnings
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
BASE_OUT     = "/data3/rasimura/social-norm-evo/figures/local/mcpr/stats/alignment_network"

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_7B     = ["llama", "mistral", "qwen"]
FAMILY_LABELS = {"llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
CONDITIONS    = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_LABELS   = {
    "PURE_BASELINE": "Pure Baseline", "BASELINE": "Baseline",
    "NO_SELECTION": "No Selection",   "NO_DISCUSSION": "No Discussion",
    "FULL": "Full",
}
SEEDS_MAIN = list(range(42, 53))
SEEDS_CODE = list(range(42, 52))

MCPR_COLORS   = {"0.4": "#555555", "0.5": "#1E88E5", "0.8": "#E53935"}
FAMILY_COLORS = {"Llama": "#1f77b4", "Mistral": "#d62728", "Qwen": "#2ca02c"}
ROUND_MEAN    = 10.5

FM_CATS = [
    "Llama_MCPR0.4", "Llama_MCPR0.5",
    "Mistral_MCPR0.4", "Mistral_MCPR0.5", "Mistral_MCPR0.8",
    "Qwen_MCPR0.4",   "Qwen_MCPR0.5",   "Qwen_MCPR0.8",
]

DV_META = {
    "IN_gap":    {"label": "Injunctive Norm − Contribution",
                  "cond_ref": "BASELINE",   "conditions": CONDITIONS[1:]},
    "DN_gap":    {"label": "Descriptive Norm − Contribution",
                  "cond_ref": "BASELINE",   "conditions": CONDITIONS[1:]},
    "spearman_r":{"label": "Spearman r(weight, contribution)",
                  "cond_ref": "PURE_BASELINE", "conditions": CONDITIONS},
    "gini":      {"label": "Gini (incoming weight)",
                  "cond_ref": "PURE_BASELINE", "conditions": CONDITIONS},
    "top_share": {"label": "Top-quartile weight share",
                  "cond_ref": "PURE_BASELINE", "conditions": CONDITIONS},
}

# ─── 1. Metric helpers ────────────────────────────────────────────────────────

def incoming_weights(network_weights, agent_ids_int):
    iw = {a: 0.0 for a in agent_ids_int}
    for e in network_weights:
        v = int(e["v"])
        if v in iw:
            iw[v] += float(e["weight"])
    return iw


def gini_coef(values):
    x = np.array(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.dot(np.arange(1, n + 1), x)) / (n * x.sum()) - (n + 1) / n


def round_metrics(r):
    """
    Extract all 5 metrics from a single round_log entry.
    Returns dict with keys: IN_gap, DN_gap, spearman_r, gini, top_share.
    Missing / undefined values → NaN.
    """
    contribs_str = r.get("contributions") or {}
    perceptions  = r.get("perceptions")  or {}
    nw           = r.get("network_weights") or []

    agents_int  = [int(k) for k in contribs_str]
    contribs    = {int(k): float(v) for k, v in contribs_str.items()}

    # Alignment: mean gap over agents with perceptions
    in_gaps, dn_gaps = [], []
    for k, perc in perceptions.items():
        if not perc:
            continue
        aid    = int(k)
        actual = contribs.get(aid)
        if actual is None:
            continue
        inj  = perc.get("injunctive_norm")
        desc = perc.get("descriptive_norm")
        if inj  is not None: in_gaps.append(float(inj)  - actual)
        if desc is not None: dn_gaps.append(float(desc) - actual)

    IN_gap = float(np.mean(in_gaps)) if in_gaps else np.nan
    DN_gap = float(np.mean(dn_gaps)) if dn_gaps else np.nan

    # Network metrics
    iw       = incoming_weights(nw, agents_int)
    iw_vals  = np.array([iw[a]      for a in agents_int], dtype=float)
    con_vals = np.array([contribs[a] for a in agents_int], dtype=float)

    if len(set(iw_vals)) < 2 or len(set(con_vals)) < 2:
        sp_r = np.nan
    else:
        sp_r, _ = scipy_stats.spearmanr(iw_vals, con_vals)

    g   = gini_coef(iw_vals)

    total_iw = iw_vals.sum()
    if total_iw > 0 and len(con_vals) > 0:
        thresh    = np.percentile(con_vals, 75)
        top_mask  = con_vals >= thresh
        top_shr   = float(iw_vals[top_mask].sum() / total_iw)
    else:
        top_shr = np.nan

    return {"IN_gap": IN_gap, "DN_gap": DN_gap,
            "spearman_r": sp_r, "gini": g, "top_share": top_shr}


# ─── 2. Data loading ──────────────────────────────────────────────────────────

def load_logs(pattern):
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
            best[d["condition"]] = d
        except Exception:
            continue
    return best


def build_dataframe():
    rows = []

    def process_log(d, model, seed, mcpr_val):
        fam    = FAMILY_LABELS[model]
        mcpr_s = str(mcpr_val)
        cond   = d["condition"]
        run_id = f"{fam}_s{seed}_{cond}_MCPR{mcpr_s}"
        for r in d["round_logs"]:
            m = round_metrics(r)
            rows.append({
                "model_family":  fam,
                "seed":          int(seed),
                "condition":     cond,
                "mcpr":          mcpr_s,
                "round":         int(r["round"]),
                "run_id":        run_id,
                **m,
            })

    # MCPR = 0.4 (main results, MCPR imputed)
    for model in MODELS_7B:
        for seed in SEEDS_MAIN:
            pat  = os.path.join(MAIN_RESULTS, model, "local",
                                f"seed{seed}", "log_*.json")
            for d in load_logs(pat).values():
                process_log(d, model, seed, 0.4)

    # MCPR = 0.5 and 0.8 (groupsizevary)
    for model in MODELS_7B:
        for mcpr_str, mcpr_val in [("0.5", 0.5), ("0.8", 0.8)]:
            tag = f"N12_G4_MCPR{mcpr_str}"
            for seed in SEEDS_CODE:
                pat  = os.path.join(CODE_RESULTS, model, "local_groupsizevary",
                                    tag, f"seed{seed}", "log_*.json")
                for d in load_logs(pat).values():
                    process_log(d, model, seed, mcpr_val)

    df = pd.DataFrame(rows)
    df["round_c"]       = df["round"] - ROUND_MEAN
    df["mcpr_f"]        = pd.Categorical(df["mcpr"],         ["0.4","0.5","0.8"])
    df["family_f"]      = pd.Categorical(df["model_family"], ["Llama","Mistral","Qwen"])
    df["cond_f"]        = pd.Categorical(df["condition"],    CONDITIONS)
    df["family_mcpr_f"] = pd.Categorical(
        df["model_family"] + "_MCPR" + df["mcpr"], FM_CATS)

    print(f"Dataset: {len(df):,} rows, {df['run_id'].nunique()} runs")
    for dv in DV_META:
        n_valid = df[dv].notna().sum()
        print(f"  {dv:12s}: {n_valid:,} non-NaN rows ({n_valid/len(df)*100:.1f}%)")
    return df


# ─── 3. Model fitting ─────────────────────────────────────────────────────────

def fit_lmm(formula, df_sub, label, re_formula="~round_c"):
    """Fit MixedLM; return (result, diag_str)."""
    diag = []
    try:
        model  = smf.mixedlm(formula, data=df_sub, groups=df_sub["run_id"],
                              re_formula=re_formula)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(method="lbfgs", maxiter=2000)
            for w in caught:
                msg = str(w.message)
                if "Convergence" in msg or "singular" in msg.lower():
                    diag.append(f"WARNING: {msg}")
        if hasattr(result, "cov_re"):
            cov_re = np.array(result.cov_re)
            eig = np.linalg.eigvalsh(cov_re).min()
            if eig < 1e-6:
                diag.append(f"Singular RE: min eigenvalue = {eig:.2e}")
        llf = result.llf
        k   = len(result.fe_params) + 3
        n   = result.nobs
        aic = -2*llf + 2*k   if np.isfinite(llf) else float("nan")
        bic = -2*llf + np.log(n)*k if np.isfinite(llf) else float("nan")
        diag.append(f"Converged: {result.converged}  AIC={aic:.2f}  BIC={bic:.2f}  LogLik={llf:.4f}  N={n}")
        print(f"  [{label}] AIC={aic:.1f}, BIC={bic:.1f}, LogLik={llf:.4f}")
        return result, "\n".join(diag)
    except Exception as e:
        return None, f"FAILED: {e}"


# ─── 4. Coefficient table ─────────────────────────────────────────────────────

def coef_table(result):
    fe  = result.fe_params
    bse = result.bse_fe
    z   = fe / bse
    p   = 2 * scipy_stats.norm.sf(np.abs(z))
    ci  = result.conf_int().loc[fe.index]
    out = pd.DataFrame({
        "coef": fe.values, "se": bse.values, "z": z.values, "p": p,
        "ci_lo": ci.iloc[:,0].values, "ci_hi": ci.iloc[:,1].values,
    }, index=fe.index)
    out["sig"] = out["p"].apply(
        lambda v: "***" if v < .001 else ("**" if v < .01 else
                  ("*" if v < .05 else ("." if v < .1 else ""))))
    return out


# ─── 5. EMM predictions ───────────────────────────────────────────────────────

def emm_predictions(result, grid_df):
    """Fixed-effect predictions for grid_df rows; NaN for inestimable cells."""
    design_info = result.model.data.design_info
    (X_new,)    = build_design_matrices([design_info], grid_df)
    X = np.asarray(X_new)

    fe_idx = result.fe_params.index
    cov_fe = result.cov_params().loc[fe_idx, fe_idx].values
    beta   = result.fe_params.values

    pred  = X @ beta
    var   = np.einsum("ij,jk,ik->i", X, cov_fe, X)
    bad   = (~np.isfinite(var)) | (var > 1e6)
    var   = np.where(bad, np.nan, var)
    pred  = np.where(bad, np.nan, pred)
    se    = np.sqrt(np.clip(var, 0, None))

    out = grid_df.copy()
    out["emm"]   = pred
    out["se"]    = se
    out["ci_lo"] = pred - 1.96 * se
    out["ci_hi"] = pred + 1.96 * se
    return out


def make_grid_m1(active_conds):
    """Prediction grid for Model 1: family_mcpr_f × condition × round."""
    rows = []
    for fm in FM_CATS:
        fam, mcpr = fm.split("_MCPR")
        for cond in active_conds:
            for rnd in range(1, 21):
                rows.append({
                    "family_mcpr_f": fm, "cond_f": cond,
                    "round_c": rnd - ROUND_MEAN, "round": rnd,
                    "model_family": fam, "mcpr": mcpr, "condition": cond,
                    "mcpr_f": mcpr, "family_f": fam,
                })
    grid = pd.DataFrame(rows)
    grid["family_mcpr_f"] = pd.Categorical(grid["family_mcpr_f"], FM_CATS)
    grid["mcpr_f"]        = pd.Categorical(grid["mcpr_f"],   ["0.4","0.5","0.8"])
    grid["family_f"]      = pd.Categorical(grid["family_f"], ["Llama","Mistral","Qwen"])
    grid["cond_f"]        = pd.Categorical(grid["cond_f"],   active_conds)
    return grid


def make_grid_m2(active_conds):
    """Prediction grid for Model 2: mcpr_f × condition × family × round."""
    rows = []
    for mcpr in ["0.4", "0.5", "0.8"]:
        for cond in active_conds:
            for fam in ["Llama", "Mistral", "Qwen"]:
                for rnd in range(1, 21):
                    rows.append({
                        "mcpr_f": mcpr, "cond_f": cond, "family_f": fam,
                        "round_c": rnd - ROUND_MEAN, "round": rnd,
                        "model_family": fam, "condition": cond, "mcpr": mcpr,
                    })
    grid = pd.DataFrame(rows)
    grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   ["0.4","0.5","0.8"])
    grid["family_f"] = pd.Categorical(grid["family_f"], ["Llama","Mistral","Qwen"])
    grid["cond_f"]   = pd.Categorical(grid["cond_f"],   active_conds)
    return grid


def marginalize(emm_full, groupby_cols):
    def rms_se(x):
        x = x.dropna()
        return np.sqrt(np.mean(x**2)) if len(x) > 0 else np.nan
    return emm_full.groupby(groupby_cols, observed=True).agg(
        emm   =("emm",   "mean"),
        se    =("se",    rms_se),
        ci_lo =("ci_lo", "mean"),
        ci_hi =("ci_hi", "mean"),
    ).reset_index()


# ─── 6. Slope contrasts ───────────────────────────────────────────────────────

def _slope(result, pts_df, w_r1, w_r20):
    """Delta-method slope = (E[y|r=20] - E[y|r=1]) / 19."""
    di = result.model.data.design_info
    fe_idx = result.fe_params.index
    cov_fe = result.cov_params().loc[fe_idx, fe_idx].values
    (X,) = build_design_matrices([di], pts_df)
    X = np.asarray(X)
    beta  = result.fe_params.values
    slope = ((w_r20 - w_r1) @ X @ beta) / 19.0
    c_vec = (w_r20 - w_r1) @ X / 19.0
    se    = float(np.sqrt(np.clip(c_vec @ cov_fe @ c_vec, 0, None)))
    return float(slope), se


def _holm(cdf):
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
    cdf["p_raw"]  = cdf.get("p_raw",  pd.Series(dtype=float)).round(4)
    cdf["p_holm"] = cdf.get("p_holm", pd.Series(dtype=float)).round(4)
    return cdf


def contrasts_m1_by_family(result, active_conds):
    """MCPR slope contrasts within family (Model 1, cell-means)."""
    family_mpcrs = {
        "Llama":   ["0.4","0.5"],
        "Mistral": ["0.4","0.5","0.8"],
        "Qwen":    ["0.4","0.5","0.8"],
    }
    slope_rows = []
    for fam, mpcrs in family_mpcrs.items():
        for mcpr in mpcrs:
            fm = f"{fam}_MCPR{mcpr}"
            pts = []
            for cond in active_conds:
                for rnd in [1, 20]:
                    pts.append({
                        "family_mcpr_f": fm, "cond_f": cond,
                        "round_c": rnd - ROUND_MEAN, "round": rnd,
                        "model_family": fam, "mcpr": mcpr, "condition": cond,
                        "mcpr_f": mcpr, "family_f": fam,
                    })
            grid = pd.DataFrame(pts)
            grid["family_mcpr_f"] = pd.Categorical(grid["family_mcpr_f"], FM_CATS)
            grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   ["0.4","0.5","0.8"])
            grid["family_f"] = pd.Categorical(grid["family_f"], ["Llama","Mistral","Qwen"])
            grid["cond_f"]   = pd.Categorical(grid["cond_f"],   active_conds)
            nc = len(active_conds)
            w1 = np.zeros(len(grid)); w20 = np.zeros(len(grid))
            for i in range(nc):
                w1[2*i] = 1/nc; w20[2*i+1] = 1/nc
            s, se = _slope(result, grid, w1, w20)
            slope_rows.append({"model_family": fam, "mcpr": mcpr, "slope": s, "se": se})

    slope_df = pd.DataFrame(slope_rows)
    rows = []
    for fam, mpcrs in family_mpcrs.items():
        pairs = [(a, b) for i, a in enumerate(["0.4","0.5","0.8"])
                 for b in ["0.4","0.5","0.8"][i+1:]
                 if a in mpcrs and b in mpcrs]
        for m1, m2 in pairs:
            r1 = slope_df[(slope_df["model_family"]==fam) & (slope_df["mcpr"]==m1)]
            r2 = slope_df[(slope_df["model_family"]==fam) & (slope_df["mcpr"]==m2)]
            est = float(r2["slope"]) - float(r1["slope"])
            se  = float(np.sqrt(float(r1["se"])**2 + float(r2["se"])**2))
            z   = est/se if se > 0 else np.nan
            p   = float(2*scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
            rows.append({"model_family": fam, "contrast": f"MCPR{m2}-MCPR{m1}",
                         "slope_m1": round(float(r1["slope"]),4),
                         "slope_m2": round(float(r2["slope"]),4),
                         "estimate": round(est,4), "se": round(se,4),
                         "z": round(z,3) if not np.isnan(z) else np.nan,
                         "p_raw": p})
        if fam == "Llama":
            for m2 in ["0.8"]:
                for m1 in ["0.4","0.5"]:
                    rows.append({"model_family": fam, "contrast": f"MCPR{m2}-MCPR{m1}",
                                 "slope_m1": np.nan, "slope_m2": np.nan,
                                 "estimate": np.nan, "se": np.nan,
                                 "z": np.nan, "p_raw": np.nan})
    return _holm(pd.DataFrame(rows))


def contrasts_m2_by_condition(result, active_conds):
    """MCPR slope contrasts within condition (Model 2, mcpr × cond + family)."""
    slope_rows = []
    for cond in active_conds:
        for mcpr in ["0.4","0.5","0.8"]:
            pts = []
            for fam in ["Llama","Mistral","Qwen"]:
                for rnd in [1, 20]:
                    pts.append({"mcpr_f": mcpr, "cond_f": cond, "family_f": fam,
                                 "round_c": rnd-ROUND_MEAN, "round": rnd,
                                 "model_family": fam, "condition": cond, "mcpr": mcpr})
            grid = pd.DataFrame(pts)
            grid["mcpr_f"]   = pd.Categorical(grid["mcpr_f"],   ["0.4","0.5","0.8"])
            grid["family_f"] = pd.Categorical(grid["family_f"], ["Llama","Mistral","Qwen"])
            grid["cond_f"]   = pd.Categorical(grid["cond_f"],   active_conds)
            nf = 3
            w1 = np.zeros(len(grid)); w20 = np.zeros(len(grid))
            for i in range(nf):
                w1[2*i] = 1/nf; w20[2*i+1] = 1/nf
            s, se = _slope(result, grid, w1, w20)
            slope_rows.append({"condition": cond, "mcpr": mcpr, "slope": s, "se": se})

    slope_df = pd.DataFrame(slope_rows)
    rows = []
    mcpr_pairs = [("0.4","0.5"),("0.4","0.8"),("0.5","0.8")]
    for cond in active_conds:
        sub = slope_df[slope_df["condition"]==cond]
        for m1, m2 in mcpr_pairs:
            r1 = sub[sub["mcpr"]==m1]; r2 = sub[sub["mcpr"]==m2]
            est = float(r2["slope"]) - float(r1["slope"])
            se  = float(np.sqrt(float(r1["se"])**2 + float(r2["se"])**2))
            z   = est/se if se > 0 else np.nan
            p   = float(2*scipy_stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan
            rows.append({"condition": cond, "contrast": f"MCPR{m2}-MCPR{m1}",
                         "slope_m1": round(float(r1["slope"]),4),
                         "slope_m2": round(float(r2["slope"]),4),
                         "estimate": round(est,4), "se": round(se,4),
                         "z": round(z,3) if not np.isnan(z) else np.nan,
                         "p_raw": p})
    return _holm(pd.DataFrame(rows))


# ─── 7. Plots ─────────────────────────────────────────────────────────────────

def _sub(df, **kw):
    mask = pd.Series(True, index=df.index)
    for col, val in kw.items():
        mask &= (df[col] == val)
    sub = df[mask].sort_values("round")
    return None if (sub.empty or sub["emm"].isna().all()) else sub


def plot_by_family(emm_df, dv_label, out_path):
    families = ["Llama","Mistral","Qwen"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, fam in zip(axes, families):
        for mcpr in ["0.4","0.5","0.8"]:
            sub = _sub(emm_df, model_family=fam, mcpr=mcpr)
            if sub is None: continue
            c = MCPR_COLORS[mcpr]
            ax.plot(sub["round"].to_numpy(), sub["emm"].to_numpy(),
                    color=c, linewidth=2, label=f"MCPR={mcpr}")
            ax.fill_between(sub["round"].to_numpy(),
                            sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(),
                            color=c, alpha=0.15)
        ax.set_title(fam, fontsize=13, fontweight="bold")
        ax.set_xlabel("Round", fontsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=11)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    axes[0].set_ylabel(dv_label, fontsize=11)
    handles = [plt.Line2D([0],[0], color=MCPR_COLORS[m], lw=2, label=f"MCPR={m}")
               for m in ["0.4","0.5","0.8"]]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_by_condition(emm_df, active_conds, dv_label, out_path):
    fig, axes = plt.subplots(1, len(active_conds),
                             figsize=(4.5*len(active_conds), 4.5), sharey=True)
    if len(active_conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, active_conds):
        for mcpr in ["0.4","0.5","0.8"]:
            sub = _sub(emm_df, condition=cond, mcpr=mcpr)
            if sub is None: continue
            c = MCPR_COLORS[mcpr]
            ax.plot(sub["round"].to_numpy(), sub["emm"].to_numpy(),
                    color=c, linewidth=2, label=f"MCPR={mcpr}")
            ax.fill_between(sub["round"].to_numpy(),
                            sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(),
                            color=c, alpha=0.15)
        ax.set_title(COND_LABELS.get(cond, cond), fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=10)
        ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
    axes[0].set_ylabel(dv_label, fontsize=11)
    handles = [plt.Line2D([0],[0], color=MCPR_COLORS[m], lw=2, label=f"MCPR={m}")
               for m in ["0.4","0.5","0.8"]]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.07))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_faceted(emm_df, active_conds, dv_label, out_path):
    """rows=family, cols=condition, lines=MCPR."""
    families = ["Llama","Mistral","Qwen"]
    fig, axes = plt.subplots(len(families), len(active_conds),
                             figsize=(4.0*len(active_conds), 3.5*len(families)),
                             sharex=True, sharey=True)
    if len(families) == 1:
        axes = np.array([axes])
    if len(active_conds) == 1:
        axes = axes[:, np.newaxis]
    for ri, fam in enumerate(families):
        for ci, cond in enumerate(active_conds):
            ax = axes[ri, ci]
            for mcpr in ["0.4","0.5","0.8"]:
                sub = _sub(emm_df, model_family=fam, condition=cond, mcpr=mcpr)
                if sub is None: continue
                c = MCPR_COLORS[mcpr]
                ax.plot(sub["round"].to_numpy(), sub["emm"].to_numpy(),
                        color=c, linewidth=2)
                ax.fill_between(sub["round"].to_numpy(),
                                sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy(),
                                color=c, alpha=0.15)
            ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=9)
            if ri == 0:
                ax.set_title(COND_LABELS.get(cond, cond), fontsize=11, fontweight="bold")
            if ri == len(families)-1:
                ax.set_xlabel("Round", fontsize=10)
            if ci == 0:
                ax.set_ylabel(fam, fontsize=11, fontweight="bold")
    handles = [plt.Line2D([0],[0], color=MCPR_COLORS[m], lw=2, label=f"MCPR={m}")
               for m in ["0.4","0.5","0.8"]]
    fig.legend(handles, [h.get_label() for h in handles],
               loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(dv_label, fontsize=12, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_residuals(result, label, out_path):
    fitted = np.asarray(result.fittedvalues)
    resid  = np.asarray(result.resid)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(fitted, resid, alpha=0.12, s=5, color="#1f77b4")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Fitted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted"); axes[0].grid(True, alpha=0.3)

    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(resid)
    axes[1].scatter(osm, osr, alpha=0.2, s=5, color="#d62728")
    axes[1].plot(osm, slope*np.array(osm)+intercept, color="black", lw=1.2)
    axes[1].set_xlabel("Theoretical"); axes[1].set_ylabel("Sample")
    axes[1].set_title("Q-Q"); axes[1].grid(True, alpha=0.3)

    axes[2].hist(resid, bins=50, color="#2ca02c", alpha=0.7, edgecolor="none")
    axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Count")
    axes[2].set_title("Distribution"); axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Residuals — {label}", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── 8. Per-DV analysis ───────────────────────────────────────────────────────

def save_text(text, path):
    with open(path, "w") as f:
        f.write(text)
    print(f"  Saved: {path}")


def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def run_dv(dv, df_all):
    meta    = DV_META[dv]
    label   = meta["label"]
    cond_ref = meta["cond_ref"]
    active  = meta["conditions"]

    out_dir  = os.path.join(BASE_OUT, dv)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DV: {dv}  ({label})")
    print(f"  Conditions: {active}  |  Cond ref: {cond_ref}")
    print(f"{'='*60}")

    # Subset: only rows with this condition and non-NaN DV
    df = df_all[df_all["condition"].isin(active)].copy()
    df = df.dropna(subset=[dv])
    df["cond_f"] = pd.Categorical(df["condition"], categories=active)
    df = df.rename(columns={dv: "dv"})

    n_runs  = df["run_id"].nunique()
    n_rows  = len(df)
    print(f"  {n_rows:,} rows, {n_runs} runs after NA drop")
    if n_rows < 100 or n_runs < 10:
        print(f"  SKIP: insufficient data for {dv}")
        return

    # ── Model 1 ────────────────────────────────────────────────────────────────
    f1 = (f"dv ~ round_c * C(family_mcpr_f, Treatment('Mistral_MCPR0.4'))"
          f" + round_c * C(cond_f, Treatment('{cond_ref}'))")
    print(f"\n  [Model 1] {f1[:80]}...")
    result1, diag1 = fit_lmm(f1, df, f"{dv}/M1")

    if result1 is None:
        save_text(diag1, os.path.join(out_dir, "model1_FAILED.txt"))
    else:
        coefs1 = coef_table(result1)
        save_text(str(result1.summary()) + "\n\n" + diag1,
                  os.path.join(out_dir, "model1_summary.txt"))
        save_csv(coefs1.reset_index().rename(columns={"index":"parameter"}),
                 os.path.join(out_dir, "model1_coefs.csv"))

        grid1    = make_grid_m1(active)
        emm1_raw = emm_predictions(result1, grid1)
        emm1_fam = marginalize(emm1_raw, ["model_family","mcpr","round"])
        emm1_cnd = marginalize(emm1_raw, ["condition","mcpr","round"])
        save_csv(emm1_fam, os.path.join(out_dir, "emm_model1_by_family.csv"))
        save_csv(emm1_cnd, os.path.join(out_dir, "emm_model1_by_condition.csv"))

        cont1 = contrasts_m1_by_family(result1, active)
        save_csv(cont1, os.path.join(out_dir, "contrasts_model1_mcpr_by_family.csv"))

        plot_by_family(emm1_fam, label,
                       os.path.join(plot_dir, "predicted_model1_by_family.pdf"))
        plot_by_condition(emm1_cnd, active, label,
                          os.path.join(plot_dir, "predicted_model1_by_condition.pdf"))
        plot_residuals(result1, f"Model 1 — {dv}",
                       os.path.join(plot_dir, "residuals_model1.pdf"))

    # ── Model 2 ────────────────────────────────────────────────────────────────
    f2 = (f"dv ~ round_c * C(mcpr_f, Treatment('0.4'))"
          f" * C(cond_f, Treatment('{cond_ref}'))"
          f" + round_c * C(family_f, Treatment('Mistral'))")
    print(f"\n  [Model 2] {f2[:80]}...")
    result2, diag2 = fit_lmm(f2, df, f"{dv}/M2")

    if result2 is None:
        save_text(diag2, os.path.join(out_dir, "model2_FAILED.txt"))
    else:
        coefs2 = coef_table(result2)
        save_text(str(result2.summary()) + "\n\n" + diag2,
                  os.path.join(out_dir, "model2_summary.txt"))
        save_csv(coefs2.reset_index().rename(columns={"index":"parameter"}),
                 os.path.join(out_dir, "model2_coefs.csv"))

        grid2    = make_grid_m2(active)
        emm2_raw = emm_predictions(result2, grid2)
        emm2_fam = marginalize(emm2_raw, ["model_family","condition","mcpr","round"])
        emm2_cnd = marginalize(emm2_raw, ["condition","mcpr","round"])
        save_csv(emm2_fam, os.path.join(out_dir, "emm_model2_by_family_condition.csv"))
        save_csv(emm2_cnd, os.path.join(out_dir, "emm_model2_by_condition.csv"))

        cont2 = contrasts_m2_by_condition(result2, active)
        save_csv(cont2, os.path.join(out_dir, "contrasts_model2_mcpr_by_condition.csv"))

        plot_by_condition(emm2_cnd, active, label,
                          os.path.join(plot_dir, "predicted_model2_by_condition.pdf"))
        plot_faceted(emm2_fam, active, label,
                     os.path.join(plot_dir, "predicted_model2_by_family_condition.pdf"))
        plot_residuals(result2, f"Model 2 — {dv}",
                       os.path.join(plot_dir, "residuals_model2.pdf"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1] Loading data ...")
    df = build_dataframe()

    # Brief descriptive stats per DV
    desc_lines = ["=== DV descriptive statistics ==="]
    for dv in DV_META:
        s = df[dv].dropna()
        desc_lines.append(
            f"{dv:12s}: N={len(s):,} mean={s.mean():.3f} sd={s.std():.3f} "
            f"min={s.min():.3f} max={s.max():.3f}")
    print("\n".join(desc_lines))

    os.makedirs(BASE_OUT, exist_ok=True)
    save_text("\n".join(desc_lines), os.path.join(BASE_OUT, "dv_descriptives.txt"))

    print("\n[2] Running per-DV analyses ...")
    for dv in DV_META:
        try:
            run_dv(dv, df)
        except Exception as e:
            import traceback
            print(f"\nERROR in {dv}: {e}")
            traceback.print_exc()

    print(f"\nAll outputs saved under: {BASE_OUT}")


if __name__ == "__main__":
    main()
