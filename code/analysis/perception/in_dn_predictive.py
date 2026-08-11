"""
in_dn_predictive.py
-------------------
Tests whether injunctive norm (IN) or descriptive norm (DN) better predicts
actual contribution. Models A-D with z-scored predictors, clustered SEs by run_id.

Pooled across all model families and conditions (PURE_BASELINE excluded).
run_id = family × seed × condition (clustering unit).
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

BASE = "/data3/rasimura/social-norm-evo"
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


# ── Analysis ──────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = load_all()
    print(f"  {len(df):,} observations | {df['run_id'].nunique()} run_ids")
    print(f"  Families: {sorted(df['family'].unique())}")
    print(f"  'family' column present: {'family' in df.columns}  ← ready for follow-up")

    # Z-score within the full filtered sample
    df["IN_z"] = (df["IN"] - df["IN"].mean()) / df["IN"].std()
    df["DN_z"] = (df["DN"] - df["DN"].mean()) / df["DN"].std()

    def fit(formula):
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["run_id"]}
        )

    print("\nFitting models …")
    mA = fit("contribution ~ IN_z")
    mB = fit("contribution ~ DN_z")
    mC = fit("contribution ~ IN_z + DN_z")
    mD = fit("contribution ~ IN_z * DN_z")

    # ── Wald test (Model C): H0: beta_IN_z = beta_DN_z ───────────────────────
    wt        = mC.wald_test("IN_z = DN_z", use_f=False)
    wald_chi2 = float(np.squeeze(wt.statistic))
    wald_df   = 1  # single linear restriction
    wald_p    = float(wt.pvalue)

    # ── VIF (Model C) ─────────────────────────────────────────────────────────
    X_c    = sm.add_constant(df[["IN_z", "DN_z"]])
    vif_in = variance_inflation_factor(X_c.values, 1)
    vif_dn = variance_inflation_factor(X_c.values, 2)

    # ── Interaction significance (Model D) ────────────────────────────────────
    int_term = "IN_z:DN_z"
    int_coef = mD.params.get(int_term, np.nan)
    int_se   = mD.bse.get(int_term, np.nan)
    int_p    = mD.pvalues.get(int_term, np.nan)
    int_sig  = int_p < 0.05

    # ── LaTeX table ───────────────────────────────────────────────────────────

    def cell(result, term):
        if term not in result.params.index:
            return ""
        coef = result.params[term]
        se   = result.bse[term]
        p    = result.pvalues[term]
        return rf"{coef:.3f} ({se:.3f}){SIG_TEX[stars(p)]}"

    term_rows = [
        ("IN_z",      r"IN$_z$"),
        ("DN_z",      r"DN$_z$"),
        ("IN_z:DN_z", r"IN$_z$$\times$DN$_z$"),
    ]

    body = []
    for term, label in term_rows:
        cells = " & ".join(cell(m, term) for m in [mA, mB, mC, mD])
        body.append(rf"    {label} & {cells} \\")

    n_obs = len(df)
    n_cl  = df["run_id"].nunique()

    wald_note = (
        rf"Model~C Wald test (H$_0$: $\hat{{\beta}}_\mathrm{{IN}}="
        rf"\hat{{\beta}}_\mathrm{{DN}}$): "
        rf"$\chi^2({wald_df})={wald_chi2:.2f}$, $p={wald_p:.3f}$."
    )
    int_note = (
        "" if int_sig
        else r" Interaction in Model~D non-significant; Model~C is the primary specification."
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"  & Model A & Model B & Model C & Model D \\",
        r"\midrule",
    ] + body + [
        r"\midrule",
        rf"  $R^2$      & {mA.rsquared:.3f} & {mB.rsquared:.3f}"
        rf" & {mC.rsquared:.3f} & {mD.rsquared:.3f} \\",
        rf"  $N$        & \multicolumn{{4}}{{c}}{{{n_obs:,}}} \\",
        rf"  Clusters   & \multicolumn{{4}}{{c}}{{{n_cl:,}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Predictive power of injunctive norm (IN$_z$) and descriptive "
         r"norm (DN$_z$) on actual contribution. OLS with SEs clustered by run "
         r"(model $\times$ seed $\times$ condition). All models and conditions "
         r"pooled (PURE\_BASELINE excluded). "
         r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$. "
         + wald_note + int_note + "}"),
        r"\label{tab:in_dn_predictive}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "in_dn_predictive_power.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\n  Saved → {out_path}")
    print("\n" + tex)

    # ── Plain-text summary ────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("SUMMARY")
    print(sep)
    print(f"  Model A  R² = {mA.rsquared:.4f}   IN_z only")
    print(f"  Model B  R² = {mB.rsquared:.4f}   DN_z only")
    print(f"  Model C  R² = {mC.rsquared:.4f}   IN_z + DN_z")
    print(f"  Model D  R² = {mD.rsquared:.4f}   IN_z * DN_z")

    print(f"\n  VIF (Model C):")
    flag_in = "  ← FLAG >5" if vif_in > 5 else ""
    flag_dn = "  ← FLAG >5" if vif_dn > 5 else ""
    print(f"    IN_z : {vif_in:.3f}{flag_in}")
    print(f"    DN_z : {vif_dn:.3f}{flag_dn}")

    print(f"\n  Wald test  H0: beta_IN_z = beta_DN_z")
    print(f"    chi2({wald_df}) = {wald_chi2:.3f},  p = {wald_p:.4f}")
    sig_str = "REJECTED" if wald_p < 0.05 else "not rejected"
    print(f"    H0 {sig_str} at α=0.05")

    print(f"\n  Model D interaction (IN_z × DN_z):")
    print(f"    coef = {int_coef:.4f},  SE = {int_se:.4f},  p = {int_p:.4f}")
    if int_sig:
        print("    → Significant: report Model D alongside C.")
    else:
        print("    → NOT significant: Model C is the primary spec to report.")

    print(f"\n  'family' column present: {('family' in df.columns)}")
    print("  Follow-up spec ready: contribution ~ (IN_z + DN_z) * family, cluster(run_id)")
    print(sep)


if __name__ == "__main__":
    main()
