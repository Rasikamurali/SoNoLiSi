"""
eval_network_quantified.py
--------------------------
Analysis 1 & 2 — OLS of round-aggregated network metrics, early vs. late
  Metrics: alignment (Spearman r), gini, top_share
  Spec:    metric ~ C(period) * C(family)  [+C(condition) in pooled]
  Models:  GPT, Llama-7B, Mistral-7B, Qwen-7B  (global variant)
  Output:  eval_network_ols_by_cond.tex, eval_network_ols_pooled.tex

Analysis 3 — Reward-loop OLS (lagged agent-level)
  Does contributing more → more weight next round (Step 1)?
  Does having more weight → higher contribution next round (Step 2)?
  Step 1: weight_z      ~ contribution_lag1_z * family
  Step 2: contribution_z ~ weight_lag1_z      * family
  Models:  GPT + 7B + 13B + 70B/72B  (all variants)
  Output:  reward_loop_no_discussion.tex, reward_loop_full.tex
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

BASE    = "/data3/rasimura/social-norm-evo"
OUT_DIR = f"{BASE}/figures/2026-03-22/paper_stats"

# ── Config for Analyses 1 & 2 (round-aggregated, 7B global) ──────────────────
RESULTS    = f"{BASE}/results"
MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "global"
SEEDS      = list(range(43, 53))
CONDITIONS = ["NO_DISCUSSION", "FULL"]

FAMILY_LABELS = {
    "gpt":     "GPT",
    "llama":   "Llama-7B",
    "mistral": "Mistral-7B",
    "qwen":    "Qwen-7B",
}
REF_FAMILY = "GPT"
REF_COND   = "NO_DISCUSSION"
METRICS    = ["alignment", "gini", "top_share"]
METRIC_LABELS = {
    "alignment": r"Alignment (Spearman $r$)",
    "gini":      "Gini (weight inequality)",
    "top_share": "Top-quartile share",
}

# ── Config for Analysis 3 (agent-level reward loop, all model sizes) ──────────
MODEL_SPECS_ALL = [
    ("gpt",         "GPT",         f"{BASE}/results",      "global", list(range(43, 53))),
    ("llama",       "Llama-7B",    f"{BASE}/results",      "global", list(range(43, 53))),
    ("mistral",     "Mistral-7B",  f"{BASE}/results",      "global", list(range(43, 53))),
    ("qwen",        "Qwen-7B",     f"{BASE}/results",      "global", list(range(43, 53))),
    ("llama_13b",   "Llama-13B",   f"{BASE}/results",      "local",  list(range(42, 52))),
    ("mistral_13b", "Mistral-13B", f"{BASE}/results",      "local",  list(range(42, 52))),
    ("qwen_14b",    "Qwen-14B",    f"{BASE}/results",      "local",  list(range(42, 52))),
    ("llama_70b",   "Llama-70B",   f"{BASE}/code/results", "local",  list(range(42, 52))),
    ("qwen_72b",    "Qwen-72B",    f"{BASE}/code/results", "local",  list(range(42, 52))),
]
ALL_FAMILIES     = [spec[1] for spec in MODEL_SPECS_ALL]
NON_REF_FAMILIES_ALL = [f for f in ALL_FAMILIES if f != REF_FAMILY]


# ── Metric / weight helpers ───────────────────────────────────────────────────

def incoming_weights(network_weights, agent_ids):
    iw = {a: 0.0 for a in agent_ids}
    for e in network_weights:
        v = int(e["v"])
        if v in iw:
            iw[v] += float(e["weight"])
    return iw


def gini(values):
    x = np.array(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.dot(np.arange(1, n + 1), x)) / (n * x.sum()) - (n + 1) / n


def compute_round_metrics(r):
    """Round-level metrics (alignment, gini, top_share) for Analyses 1 & 2."""
    nw   = r.get("network_weights", [])
    cont = r.get("contributions", {})
    if not nw or not cont:
        return None
    agent_ids = [int(k) for k in cont]
    iw        = incoming_weights(nw, agent_ids)
    contribs  = {int(k): float(v) for k, v in cont.items()}
    iw_vals   = np.array([iw[a]        for a in agent_ids])
    con_vals  = np.array([contribs[a]  for a in agent_ids])

    alignment = (np.nan if len(set(iw_vals)) < 2 or len(set(con_vals)) < 2
                 else spearmanr(iw_vals, con_vals)[0])
    g         = gini(iw_vals)
    thresh    = np.percentile(con_vals, 75)
    total_iw  = iw_vals.sum()
    top_share = iw_vals[con_vals >= thresh].sum() / total_iw if total_iw > 0 else np.nan

    return {"alignment": alignment, "gini": g, "top_share": top_share}


# ── OLS helpers (shared) ──────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""

SIG_TEX = {
    "***": r"$^{***}$", "**": r"$^{**}$",
    "*":   r"$^{*}$",   "†":  r"$^{\dagger}$", "": "",
}

def cell(result, term):
    c = result.params.get(term, np.nan)
    s = result.bse.get(term, np.nan)
    p = result.pvalues.get(term, np.nan)
    if np.isnan(c):
        return ""
    return rf"{c:.3f} ({s:.3f}){SIG_TEX[stars(p)]}"

def fit_ols(df, formula, cluster_col="run_id"):
    """OLS with cluster-robust SEs. Drops rows with NaN in formula variables."""
    used_cols = [cluster_col, "metric_val", "period", "family", "condition"]
    clean = df[[c for c in used_cols if c in df.columns]].dropna().reset_index(drop=True)
    return smf.ols(formula, data=clean).fit(
        cov_type="cluster", cov_kwds={"groups": clean[cluster_col]}
    )

def fit_ols_clean(df, formula, cluster_col="run_id"):
    """OLS for pre-cleaned dataframes (reward loop). Assumes no NaN."""
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSES 1 & 2 — round-aggregated period OLS (7B global models)
# ═══════════════════════════════════════════════════════════════════════════════

def load_all():
    """Load round-level metric data for 7B global models (Analyses 1 & 2)."""
    rows = []
    for model in MODELS:
        family = FAMILY_LABELS[model]
        for seed in SEEDS:
            pattern = os.path.join(RESULTS, model, VARIANT, f"seed{seed}", "log_*.json")
            cond_data = {}
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                    cond_data[d["condition"]] = d
                except Exception:
                    continue
            for cond in CONDITIONS:
                d = cond_data.get(cond)
                if d is None:
                    continue
                run_id  = f"{model}_s{seed}_{cond}"
                max_rnd = max(r["round"] for r in d["round_logs"])
                for r in d["round_logs"]:
                    m = compute_round_metrics(r)
                    if m is None:
                        continue
                    rows.append({"family": family, "model": model, "condition": cond,
                                 "seed": seed, "run_id": run_id,
                                 "round": r["round"], "max_round": max_rnd, **m})
    df = pd.DataFrame(rows)
    early = df["round"] <= 3
    late  = df["round"] >= df["max_round"] - 2
    df    = df[early | late].copy()
    df["period"] = np.where(df["round"] <= 3, "early", "late")
    return df


def period_term():
    return "C(period, Treatment('early'))[T.late]"

def family_period_term(fam):
    return (f"C(period, Treatment('early'))[T.late]:"
            f"C(family, Treatment('{REF_FAMILY}'))[T.{fam}]")

def cond_term():
    return f"C(condition, Treatment('{REF_COND}'))[T.FULL]"

NON_REF_FAMILIES = [FAMILY_LABELS[m] for m in MODELS if FAMILY_LABELS[m] != REF_FAMILY]


def header_row_12():
    return (r"  & " + " & ".join(rf"\textbf{{{METRIC_LABELS[m]}}}" for m in METRICS)
            + r" \\")


def make_table_by_cond(models_by_cond):
    def block(cond, results):
        lines = [
            rf"  \multicolumn{{4}}{{l}}{{\textit{{Condition: {cond.replace('_',' ').title()}}}}}\\ [2pt]",
            (r"  \quad Late (GPT ref.) & "
             + " & ".join(cell(results[m], period_term()) for m in METRICS) + r" \\"),
        ]
        for fam in NON_REF_FAMILIES:
            short = fam.replace("-7B", "")
            lines.append(
                rf"  \quad Late $\times$ {short} & "
                + " & ".join(cell(results[m], family_period_term(fam)) for m in METRICS)
                + r" \\"
            )
        r2 = " & ".join(f"{results[m].rsquared:.3f}" for m in METRICS)
        lines.append(rf"  \quad $R^2$ & {r2} \\")
        return lines

    def get_n_cl(results):
        df = results[METRICS[0]].model.data.frame
        return len(df), df["run_id"].nunique()

    out = [r"\begin{table}[ht]", r"\centering", r"\small",
           r"\begin{tabular}{lrrr}", r"\toprule", header_row_12(), r"\midrule"]
    for cond, results in models_by_cond.items():
        out += block(cond, results)
        n, cl = get_n_cl(results)
        out += [rf"  \quad $N$ & \multicolumn{{3}}{{c}}{{{n:,}}} \\",
                rf"  \quad Clusters & \multicolumn{{3}}{{c}}{{{cl}}} \\",
                r"  \midrule"]
    out += [r"\bottomrule",
            r"\end{tabular}",
            (r"\caption{OLS of network-development metrics, early (rounds 1--3) vs.\ late "
             r"(last 3 rounds, dynamic). Reference: period=early, family=GPT. "
             r"SEs clustered by run. "
             r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}"),
            r"\label{tab:eval_network_by_cond}", r"\end{table}"]
    return "\n".join(out) + "\n"


def make_table_pooled(pooled_results):
    df = pooled_results[METRICS[0]].model.data.frame
    n, cl = len(df), df["run_id"].nunique()
    rows = [(r"  Late (GPT ref.) & "
             + " & ".join(cell(pooled_results[m], period_term()) for m in METRICS) + r" \\")]
    for fam in NON_REF_FAMILIES:
        rows.append(rf"  Late $\times$ {fam.replace('-7B','')} & "
                    + " & ".join(cell(pooled_results[m], family_period_term(fam)) for m in METRICS)
                    + r" \\")
    rows += [(r"  FULL (vs.\ No Discussion) & "
              + " & ".join(cell(pooled_results[m], cond_term()) for m in METRICS) + r" \\"),
             r"\midrule",
             (r"  $R^2$ & "
              + " & ".join(f"{pooled_results[m].rsquared:.3f}" for m in METRICS) + r" \\"),
             rf"  $N$ & \multicolumn{{3}}{{c}}{{{n:,}}} \\",
             rf"  Clusters & \multicolumn{{3}}{{c}}{{{cl}}} \\"]
    lines = [r"\begin{table}[ht]", r"\centering", r"\small",
             r"\begin{tabular}{lrrr}", r"\toprule", header_row_12(), r"\midrule"
             ] + rows + [r"\bottomrule", r"\end{tabular}",
             (r"\caption{OLS of network-development metrics, pooled across conditions. "
              r"Reference: period=early, family=GPT, condition=NO\_DISCUSSION. "
              r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}"),
             r"\label{tab:eval_network_pooled}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Reward-loop OLS (agent-level lagged, all model sizes)
# ═══════════════════════════════════════════════════════════════════════════════

def load_agent_data_all():
    """Load per-agent per-round data for all 9 model specs."""
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS_ALL:
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant,
                                   f"seed{seed}", "log_*.json")
            cond_data = {}
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                    cond_data[d["condition"]] = d
                except Exception:
                    continue
            for cond in CONDITIONS:
                d = cond_data.get(cond)
                if d is None:
                    continue
                run_id = f"{model_key}_s{seed}"
                for r in d["round_logs"]:
                    nw      = r.get("network_weights", [])
                    cont    = r.get("contributions", {})
                    if not cont:
                        continue
                    agent_ids = [int(k) for k in cont]
                    contribs  = {int(k): float(v) for k, v in cont.items()}
                    iw        = (incoming_weights(nw, agent_ids) if nw
                                 else {a: np.nan for a in agent_ids})
                    for aid in agent_ids:
                        rows.append({
                            "family":          family,
                            "model":           model_key,
                            "condition":       cond,
                            "seed":            seed,
                            "run_id":          run_id,
                            "round":           r["round"],
                            "agent_id":        aid,
                            "contribution":    contribs[aid],
                            "incoming_weight": iw[aid],
                        })
    return pd.DataFrame(rows)


def build_lagged_agent(df):
    """
    Sort by (run_id, condition, agent_id, round); create lag-1 columns within
    each (run_id, condition, agent_id) group. Drop rows where lag is NaN
    (first round per agent per run), which respects all boundaries.
    """
    df = df.sort_values(["run_id", "condition", "agent_id", "round"]).copy()
    grp = df.groupby(["run_id", "condition", "agent_id"])
    df["contribution_lag1"] = grp["contribution"].shift(1)
    df["weight_lag1"]       = grp["incoming_weight"].shift(1)
    df = df.dropna(subset=["contribution_lag1", "weight_lag1",
                            "contribution", "incoming_weight"]).reset_index(drop=True)
    return df


def zscale_cond(df, cond):
    """Z-score the four key columns within condition (across all families/seeds)."""
    sub = df[df["condition"] == cond].copy()
    for col in ["contribution", "incoming_weight", "contribution_lag1", "weight_lag1"]:
        mu  = sub[col].mean()
        sig = sub[col].std()
        sub[col + "_z"] = (sub[col] - mu) / sig if sig > 0 else 0.0
    return sub


# ── Stability flag ────────────────────────────────────────────────────────────

def weight_stability_flags(df, threshold_cv=0.05):
    """
    Return dict {family: bool} flagging families with near-uniform incoming
    weights (CV = std/mean < threshold). Unstable estimates expected there.
    """
    flags = {}
    for fam in df["family"].unique():
        w = df.loc[df["family"] == fam, "incoming_weight"].dropna()
        cv = w.std() / abs(w.mean()) if abs(w.mean()) > 1e-10 else np.inf
        flags[fam] = cv < threshold_cv
    return flags


# ── Term name helpers ─────────────────────────────────────────────────────────

def rl_main_term(predictor):
    return predictor

def rl_int_term(predictor, fam):
    return f"{predictor}:C(family, Treatment('{REF_FAMILY}'))[T.{fam}]"


# ── OLS for one condition ──────────────────────────────────────────────────────

def run_reward_loop_cond(df_lagged, cond):
    """
    Fit Step 1 and Step 2 for a single condition.
    Returns (mS1, mS2, sub_df).
    """
    sub = zscale_cond(df_lagged, cond)

    formula_s1 = (
        f"incoming_weight_z ~ (contribution_lag1_z + weight_lag1_z) * "
        f"C(family, Treatment('{REF_FAMILY}'))"
    )
    formula_s2 = (
        f"contribution_z ~ (weight_lag1_z + contribution_lag1_z) * "
        f"C(family, Treatment('{REF_FAMILY}'))"
    )

    mS1 = fit_ols_clean(sub, formula_s1)
    mS2 = fit_ols_clean(sub, formula_s2)
    return mS1, mS2, sub


# ── LaTeX table for one condition ─────────────────────────────────────────────

def make_reward_loop_table(mS1, mS2, sub, cond, flags):
    n_obs  = len(sub)
    n_runs = sub["run_id"].nunique()

    pred1 = "contribution_lag1_z"
    pred2 = "weight_lag1_z"

    flag_note_parts = [f for f, flagged in flags.items() if flagged]
    flag_note = ("")
    if flag_note_parts:
        flag_note = (r" \textbf{Note:} near-uniform weights detected for "
                     + ", ".join(flag_note_parts)
                     + r" (CV $<$ 5\%); Step~1 estimates unreliable for these families.")

    def short(fam):
        return fam

    rows = []
    # ── Step 1 block: contribution_lag1_z ────────────────────────────────────
    pred1w = "weight_lag1_z"
    rows += [
        r"  \multicolumn{3}{l}{\textit{Step 1: $w_z \sim (c_{t-1,z} + w_{t-1,z}) \times \text{family}$}} \\[2pt]",
        r"  \quad \textit{Effect of $c_{t-1,z}$ (controlling for $w_{t-1,z}$):} & & \\",
        rf"  \quad\quad Main effect (GPT ref.) & {cell(mS1, pred1)} & \\",
    ]
    for fam in NON_REF_FAMILIES_ALL:
        rows.append(rf"  \quad\quad $\times$ {short(fam)} & {cell(mS1, rl_int_term(pred1, fam))} & \\")

    # ── Step 1 block: weight_lag1_z (autoregressive control) ─────────────────
    rows += [
        r"  \quad \textit{Autoregressive control ($w_{t-1,z}$):} & & \\",
        rf"  \quad\quad Main effect (GPT ref.) & {cell(mS1, pred1w)} & \\",
    ]
    for fam in NON_REF_FAMILIES_ALL:
        rows.append(rf"  \quad\quad $\times$ {short(fam)} & {cell(mS1, rl_int_term(pred1w, fam))} & \\")

    rows.append(rf"  \quad $R^2$ & {mS1.rsquared:.3f} & \\")
    rows.append(r"  \midrule")

    # ── Step 2 block: weight_lag1_z ──────────────────────────────────────────
    pred2c = "contribution_lag1_z"
    rows += [
        r"  \multicolumn{3}{l}{\textit{Step 2: $c_z \sim (w_{t-1,z} + c_{t-1,z}) \times \text{family}$}} \\[2pt]",
        r"  \quad \textit{Effect of $w_{t-1,z}$ (controlling for $c_{t-1,z}$):} & & \\",
        rf"  \quad\quad Main effect (GPT ref.) & & {cell(mS2, pred2)} \\",
    ]
    for fam in NON_REF_FAMILIES_ALL:
        rows.append(rf"  \quad\quad $\times$ {short(fam)} & & {cell(mS2, rl_int_term(pred2, fam))} \\")

    # ── Step 2 block: contribution_lag1_z (autoregressive control) ───────────
    rows += [
        r"  \quad \textit{Autoregressive control ($c_{t-1,z}$):} & & \\",
        rf"  \quad\quad Main effect (GPT ref.) & & {cell(mS2, pred2c)} \\",
    ]
    for fam in NON_REF_FAMILIES_ALL:
        rows.append(rf"  \quad\quad $\times$ {short(fam)} & & {cell(mS2, rl_int_term(pred2c, fam))} \\")

    rows.append(rf"  \quad $R^2$ & & {mS2.rsquared:.3f} \\")
    rows.append(r"  \midrule")

    rows += [
        rf"  $N$ (agent $\times$ round) & \multicolumn{{2}}{{c}}{{{n_obs:,}}} \\",
        rf"  Clusters (runs)            & \multicolumn{{2}}{{c}}{{{n_runs}}} \\",
    ]

    cond_label = cond.replace("_", " ").replace("NO DISCUSSION", "No Discussion")
    label_key  = cond.lower().replace("_", "")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        rf"  & Step 1 ($\beta$ on $c_{{t-1,z}}$ and $w_{{t-1,z}}$) & Step 2 ($\beta$ on $w_{{t-1,z}}$ and $c_{{t-1,z}}$) \\",
        r"\midrule",
    ] + rows + [
        r"\bottomrule",
        r"\end{tabular}",
        (rf"\caption{{Reward-loop OLS for condition: {cond_label}. "
         r"Step~1: does contributing more at $t-1$ increase incoming weight at $t$, "
         r"controlling for lagged weight $w_{t-1}$ (autoregressive term)? "
         r"Step~2: does having more weight at $t-1$ increase contribution at $t$, "
         r"controlling for lagged contribution $c_{t-1}$ (autoregressive term)? "
         r"All variables z-scored within condition. "
         r"OLS with SEs clustered by run (model $\times$ seed). "
         r"$\dagger p{{<}}.10$, $*p{{<}}.05$, $**p{{<}}.01$, $***p{{<}}.001$."
         + flag_note + r"}"),
        rf"\label{{tab:reward_loop_{label_key}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Plain-text summary ────────────────────────────────────────────────────────

def reward_loop_summary(mS1, mS2, sub, cond, flags):
    pred1 = "contribution_lag1_z"
    pred2 = "weight_lag1_z"
    sep   = "-" * 60
    print(f"\n{sep}")
    print(f"REWARD LOOP SUMMARY — {cond}")
    print(sep)
    n_obs  = len(sub)
    n_runs = sub["run_id"].nunique()
    print(f"  N obs (after lag): {n_obs:,}  |  N runs: {n_runs}")

    # Per-family effective slope (main effect + interaction)
    for fam in ALL_FAMILIES:
        if fam == REF_FAMILY:
            s1_b = mS1.params.get(pred1, np.nan)
            s1_p = mS1.pvalues.get(pred1, np.nan)
            s2_b = mS2.params.get(pred2, np.nan)
            s2_p = mS2.pvalues.get(pred2, np.nan)
        else:
            t1   = rl_int_term(pred1, fam)
            t2   = rl_int_term(pred2, fam)
            s1_b = mS1.params.get(pred1, np.nan) + mS1.params.get(t1, 0.0)
            s1_p = mS1.pvalues.get(t1, np.nan)  # interaction p-value
            s2_b = mS2.params.get(pred2, np.nan) + mS2.params.get(t2, 0.0)
            s2_p = mS2.pvalues.get(t2, np.nan)

        step1_sig = not np.isnan(s1_p) and s1_p < 0.05
        step2_sig = not np.isnan(s2_p) and s2_p < 0.05

        if fam == REF_FAMILY:
            step1_sig = not np.isnan(mS1.pvalues.get(pred1, np.nan)) and mS1.pvalues.get(pred1) < 0.05
            step2_sig = not np.isnan(mS2.pvalues.get(pred2, np.nan)) and mS2.pvalues.get(pred2) < 0.05

        loop_verdict = ("FULL LOOP"    if step1_sig and step2_sig
                        else "partial (step 1 only)" if step1_sig
                        else "partial (step 2 only)" if step2_sig
                        else "NO evidence")

        pred1w = "weight_lag1_z"
        pred2c = "contribution_lag1_z"
        # weight AR control in Step 1
        if fam == REF_FAMILY:
            war_b = mS1.params.get(pred1w, np.nan)
            war_p = mS1.pvalues.get(pred1w, np.nan)
        else:
            war_b = (mS1.params.get(pred1w, np.nan)
                     + mS1.params.get(rl_int_term(pred1w, fam), 0.0))
            war_p = mS1.pvalues.get(rl_int_term(pred1w, fam), np.nan)
        # contribution AR control in Step 2
        if fam == REF_FAMILY:
            ar_b = mS2.params.get(pred2c, np.nan)
            ar_p = mS2.pvalues.get(pred2c, np.nan)
        else:
            ar_b = (mS2.params.get(pred2c, np.nan)
                    + mS2.params.get(rl_int_term(pred2c, fam), 0.0))
            ar_p = mS2.pvalues.get(rl_int_term(pred2c, fam), np.nan)

        flag_str = "  ⚠ low-variance weights" if flags.get(fam) else ""
        print(f"\n  {fam}")
        print(f"    Step 1 (c_lag → w|w_lag): β={s1_b:+.3f}  {stars(mS1.pvalues.get(pred1 if fam==REF_FAMILY else rl_int_term(pred1,fam), 1.0)) or 'n.s.'}")
        print(f"    Step 1 (w_lag → w, AR):   β={war_b:+.3f}  {stars(war_p) or 'n.s.'}")
        print(f"    Step 2 (w_lag → c|c_lag): β={s2_b:+.3f}  {stars(mS2.pvalues.get(pred2 if fam==REF_FAMILY else rl_int_term(pred2,fam), 1.0)) or 'n.s.'}")
        print(f"    Step 2 (c_lag → c, AR):   β={ar_b:+.3f}  {stars(ar_p) or 'n.s.'}")
        print(f"    → {loop_verdict}{flag_str}")

    print(f"\n  Stability flags (CV < 5% for incoming_weight):")
    for fam, flagged in flags.items():
        print(f"    {fam:15s}: {'⚠ FLAGGED' if flagged else 'OK'}")
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_analyses_12():
    print("\n" + "=" * 60)
    print("ANALYSES 1 & 2 — round-aggregated period OLS (7B global)")
    print("=" * 60)
    df = load_all()
    print(f"  {len(df):,} round-obs | period counts:")
    print(df.groupby(["condition", "period"])["round"].count().unstack())

    formula_single = (
        "metric_val ~ C(period, Treatment('early')) * "
        f"C(family, Treatment('{REF_FAMILY}'))"
    )
    formula_pooled = (
        "metric_val ~ C(period, Treatment('early')) * "
        f"C(family, Treatment('{REF_FAMILY}')) + "
        f"C(condition, Treatment('{REF_COND}'))"
    )

    by_cond = {}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].copy()
        results = {}
        for metric in METRICS:
            sub["metric_val"] = sub[metric]
            results[metric] = fit_ols(sub, formula_single)
        by_cond[cond] = results

    tex = make_table_by_cond(by_cond)
    path = os.path.join(OUT_DIR, "eval_network_ols_by_cond.tex")
    with open(path, "w") as f: f.write(tex)
    print(f"\n  Saved → {path}")

    pooled = {}
    for metric in METRICS:
        df["metric_val"] = df[metric]
        pooled[metric] = fit_ols(df, formula_pooled)
    tex2 = make_table_pooled(pooled)
    path2 = os.path.join(OUT_DIR, "eval_network_ols_pooled.tex")
    with open(path2, "w") as f: f.write(tex2)
    print(f"  Saved → {path2}")


def run_analysis_3():
    print("\n" + "=" * 60)
    print("ANALYSIS 3 — reward-loop OLS (agent-level, all model sizes)")
    print("=" * 60)

    print("  Loading agent-level data …")
    raw = load_agent_data_all()
    print(f"  Raw: {len(raw):,} agent-round obs | {raw['run_id'].nunique()} runs")

    df_lag = build_lagged_agent(raw)
    print(f"  After lag/drop: {len(df_lag):,} obs | {df_lag['run_id'].nunique()} runs")
    print("  Obs by condition:")
    print(df_lag.groupby("condition")["agent_id"].count().to_string())

    os.makedirs(OUT_DIR, exist_ok=True)

    for cond in CONDITIONS:
        fname = ("reward_loop_no_discussion.tex" if cond == "NO_DISCUSSION"
                 else "reward_loop_full.tex")

        mS1, mS2, sub = run_reward_loop_cond(df_lag, cond)
        flags = weight_stability_flags(sub)

        tex  = make_reward_loop_table(mS1, mS2, sub, cond, flags)
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w") as f:
            f.write(tex)
        print(f"\n  Saved → {path}")
        print(tex)

        reward_loop_summary(mS1, mS2, sub, cond, flags)


def main():
    run_analyses_12()
    run_analysis_3()
    print("\nAll done.")


if __name__ == "__main__":
    main()
