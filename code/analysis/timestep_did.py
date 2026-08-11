"""
timestep_did.py
---------------
Tests temporal ordering between DN, IN, and contribution using lagged predictors.
Contemporaneous correlations cannot establish sequence; lags can.

Lag structure: within each (run_id, agent_id) group, t-1 values of IN and DN
predict t values of IN or contribution. Round-1 rows (no valid t-1) are dropped.

Models (all cluster SEs by run_id):
  E: IN_z          ~ DN_lag1_z              — does past DN shape current IN?
  F: contribution_z ~ IN_lag1_z             — does past IN predict contribution?
  G: contribution_z ~ DN_lag1_z             — does past DN predict contribution?
  H: contribution_z ~ IN_lag1_z + DN_lag1_z — joint; Wald test beta_IN = beta_DN

Reference (contemporaneous, from in_dn_predictive.py, Model C):
  contribution_z ~ IN_z + DN_z: IN_z β=0.560, DN_z β=1.039  (DN >> IN)
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

BASE    = "/data3/rasimura/social-norm-evo"
OUT_DIR = f"{BASE}/figures/2026-03-22/paper_stats"

MODEL_SPECS = [
    ("gpt",         "GPT",         f"{BASE}/results",      "local", list(range(43, 53))),
    ("llama",       "Llama-7B",    f"{BASE}/results",      "local", list(range(43, 53))),
    ("mistral",     "Mistral-7B",  f"{BASE}/results",      "local", list(range(43, 53))),
    ("qwen",        "Qwen-7B",     f"{BASE}/results",      "local", list(range(43, 53))),
    ("llama_13b",   "Llama-13B",   f"{BASE}/results",      "local", list(range(42, 52))),
    ("mistral_13b", "Mistral-13B", f"{BASE}/results",      "local", list(range(42, 52))),
    ("qwen_14b",    "Qwen-14B",    f"{BASE}/results",      "local", list(range(42, 52))),
    ("llama_70b",   "Llama-70B",   f"{BASE}/code/results", "local", list(range(42, 52))),
    ("qwen_72b",    "Qwen-72B",    f"{BASE}/code/results", "local", list(range(42, 52))),
]
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]

# Contemporaneous reference coefficients (from in_dn_predictive.py Model C)
CONTEMP_IN_BETA = 0.560
CONTEMP_DN_BETA = 1.039


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
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
                run_id = f"{family}_s{seed}_{cond}"
                for r in d["round_logs"]:
                    contribs = {int(k): float(v)
                                for k, v in r["contributions"].items()}
                    percs = {int(k): v
                             for k, v in (r.get("perceptions") or {}).items()}
                    for aid, perc in percs.items():
                        if not perc:
                            continue
                        actual = contribs.get(int(aid))
                        inj    = perc.get("injunctive_norm")
                        desc   = perc.get("descriptive_norm")
                        if actual is None or inj is None or desc is None:
                            continue
                        rows.append({
                            "family":       family,
                            "model":        model_key,
                            "condition":    cond,
                            "seed":         seed,
                            "round":        r["round"],
                            "agent_id":     int(aid),
                            "run_id":       run_id,
                            "contribution": float(actual),
                            "IN":           float(inj),
                            "DN":           float(desc),
                        })
    return pd.DataFrame(rows)


# ── Lagging ───────────────────────────────────────────────────────────────────

def build_lagged(df):
    """
    Sort by (run_id, agent_id, round), shift IN and DN within each
    (run_id, agent_id) group. Drop rows where lag is NaN (first round
    of each group) — this automatically respects run and agent boundaries.
    """
    df = df.sort_values(["run_id", "agent_id", "round"]).copy()
    grp = df.groupby(["run_id", "agent_id"])
    df["IN_lag1"] = grp["IN"].shift(1)
    df["DN_lag1"] = grp["DN"].shift(1)
    df = df.dropna(subset=["IN_lag1", "DN_lag1"]).copy()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    raw = load_all()
    print(f"  Raw: {len(raw):,} agent-round obs | {raw['run_id'].nunique()} run_ids")

    df = build_lagged(raw)
    print(f"  After lagging (round 1 dropped): {len(df):,} obs | "
          f"{df['run_id'].nunique()} run_ids")

    # Z-score within the lagged filtered sample
    for col in ["IN", "DN", "IN_lag1", "DN_lag1", "contribution"]:
        z_col = col + "_z" if col != "contribution" else "contribution_z"
        if col == "contribution":
            df["contribution_z"] = (df[col] - df[col].mean()) / df[col].std()
        else:
            df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()

    def fit(formula):
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["run_id"]}
        )

    print("\nFitting models …")
    mE = fit("IN_z ~ DN_lag1_z")
    mF = fit("contribution_z ~ IN_lag1_z")
    mG = fit("contribution_z ~ DN_lag1_z")
    mH = fit("contribution_z ~ IN_lag1_z + DN_lag1_z")

    # ── Wald test (Model H): H0: beta_IN_lag1 = beta_DN_lag1 ─────────────────
    wt        = mH.wald_test("IN_lag1_z = DN_lag1_z", use_f=False)
    wald_chi2 = float(np.squeeze(wt.statistic))
    wald_df   = 1
    wald_p    = float(wt.pvalue)

    # ── VIF (Model H) ─────────────────────────────────────────────────────────
    X_h    = sm.add_constant(df[["IN_lag1_z", "DN_lag1_z"]])
    vif_in = variance_inflation_factor(X_h.values, 1)
    vif_dn = variance_inflation_factor(X_h.values, 2)

    # ── Extract key coefficients ──────────────────────────────────────────────
    def coef(m, term):
        return m.params.get(term, np.nan)
    def se(m, term):
        return m.bse.get(term, np.nan)
    def pv(m, term):
        return m.pvalues.get(term, np.nan)

    def cell(m, term):
        c, s, p = coef(m, term), se(m, term), pv(m, term)
        if np.isnan(c):
            return ""
        return rf"{c:.3f} ({s:.3f}){SIG_TEX[stars(p)]}"

    in_lag_F  = coef(mF, "IN_lag1_z")
    dn_lag_G  = coef(mG, "DN_lag1_z")
    in_lag_H  = coef(mH, "IN_lag1_z")
    dn_lag_H  = coef(mH, "DN_lag1_z")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    n_obs = len(df)
    n_cl  = df["run_id"].nunique()

    wald_note = (
        rf"Model~H Wald test (H$_0$: $\hat{{\beta}}_\mathrm{{IN\_lag1}}"
        rf"=\hat{{\beta}}_\mathrm{{DN\_lag1}}$): "
        rf"$\chi^2({wald_df})={wald_chi2:.2f}$, $p={wald_p:.3f}$."
    )
    vif_note = (
        f"VIF (Model~H): IN$_{{\\text{{lag1}}}}={vif_in:.2f}$, "
        f"DN$_{{\\text{{lag1}}}}={vif_dn:.2f}$"
        + (" --- both $>5$, interpret jointly with caution." if vif_in > 5 or vif_dn > 5
           else ".")
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"  & Model E & Model F & Model G & Model H \\",
        r"  & (outcome: IN$_z$) & \multicolumn{3}{c}{(outcome: contribution$_z$)} \\",
        r"\midrule",
        # DN_lag1_z row: appears in E, G, H
        rf"  DN$_{{t-1,z}}$ & {cell(mE,'DN_lag1_z')} & & {cell(mG,'DN_lag1_z')} & {cell(mH,'DN_lag1_z')} \\",
        # IN_lag1_z row: appears in F, H
        rf"  IN$_{{t-1,z}}$ & & {cell(mF,'IN_lag1_z')} & & {cell(mH,'IN_lag1_z')} \\",
        r"\midrule",
        rf"  $R^2$ & {mE.rsquared:.3f} & {mF.rsquared:.3f} & {mG.rsquared:.3f} & {mH.rsquared:.3f} \\",
        rf"  $N$ & \multicolumn{{4}}{{c}}{{{n_obs:,}}} \\",
        rf"  Clusters & \multicolumn{{4}}{{c}}{{{n_cl:,}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Lagged predictors of injunctive norm and contribution. "
         r"All predictors are $t-1$ values; outcomes are at time $t$. "
         r"Round 1 dropped (no valid lag). "
         r"OLS with SEs clustered by run (model $\times$ seed $\times$ condition). "
         r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$. "
         + wald_note + " " + vif_note + "}"),
        r"\label{tab:lagged_predictive}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "lagged_predictive_power.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\n  Saved → {out_path}")
    print("\n" + tex)

    # ── Plain-text summary ────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("SUMMARY")
    print(sep)
    print(f"  N obs (after lag): {n_obs:,}  |  N runs: {n_cl:,}")

    print(f"\n  Model E: DN_lag1_z → IN_z")
    print(f"    β = {coef(mE,'DN_lag1_z'):.3f}  SE={se(mE,'DN_lag1_z'):.3f}"
          f"  p={pv(mE,'DN_lag1_z'):.4f}  R²={mE.rsquared:.4f}")
    print(f"    → Past descriptive norm {'does' if pv(mE,'DN_lag1_z')<0.05 else 'does NOT'} "
          f"predict current injunctive norm.")

    print(f"\n  Model F: IN_lag1_z → contribution_z")
    print(f"    β = {in_lag_F:.3f}  SE={se(mF,'IN_lag1_z'):.3f}"
          f"  p={pv(mF,'IN_lag1_z'):.4f}  R²={mF.rsquared:.4f}")

    print(f"\n  Model G: DN_lag1_z → contribution_z")
    print(f"    β = {dn_lag_G:.3f}  SE={se(mG,'DN_lag1_z'):.3f}"
          f"  p={pv(mG,'DN_lag1_z'):.4f}  R²={mG.rsquared:.4f}")

    print(f"\n  Model F vs G (lagged predictors of contribution):")
    if abs(in_lag_F) > abs(dn_lag_G):
        lag_winner = "IN_lag1"
        lag_loser  = "DN_lag1"
    else:
        lag_winner = "DN_lag1"
        lag_loser  = "IN_lag1"
    print(f"    {lag_winner} has a larger standardised coefficient "
          f"({abs(in_lag_F):.3f} vs {abs(dn_lag_G):.3f})")
    print(f"    R²: Model F={mF.rsquared:.4f}  Model G={mG.rsquared:.4f}  "
          f"(higher = {('F (IN_lag)' if mF.rsquared > mG.rsquared else 'G (DN_lag)')})")

    # Compare to contemporaneous
    contemp_winner = "DN" if CONTEMP_DN_BETA > CONTEMP_IN_BETA else "IN"
    lag_winner_short = "IN" if lag_winner == "IN_lag1" else "DN"
    flip = (contemp_winner != lag_winner_short)
    print(f"\n  Contemporaneous (Model C): DN_z β={CONTEMP_DN_BETA:.3f} > IN_z β={CONTEMP_IN_BETA:.3f}  → DN wins")
    print(f"  Lagged          (F vs G):  {lag_winner_short}_lag1 β={abs(max(in_lag_F,dn_lag_G,key=abs)):.3f}"
          f" > {lag_loser.replace('_lag1','')} β={abs(min(in_lag_F,dn_lag_G,key=abs)):.3f}  → {lag_winner_short} wins")
    if flip:
        print(f"  *** RANKING FLIPS once lagged: contemporaneous DN > IN, but lagged IN_lag1 > DN_lag1 ***")
    else:
        print(f"  Ranking is CONSISTENT: {lag_winner_short} dominates both contemporaneously and with a lag.")

    print(f"\n  Model H (joint): IN_lag1_z β={in_lag_H:.3f}  DN_lag1_z β={dn_lag_H:.3f}")
    print(f"  Wald H0 beta_IN_lag1 = beta_DN_lag1: chi2({wald_df})={wald_chi2:.3f}  p={wald_p:.4f}")
    print(f"  VIF: IN_lag1={vif_in:.3f}  DN_lag1={vif_dn:.3f}"
          + ("  ← both >5, multicollinearity present" if vif_in > 5 else ""))
    print(sep)


if __name__ == "__main__":
    main()
