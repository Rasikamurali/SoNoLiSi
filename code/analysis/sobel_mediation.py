"""
sobel_mediation.py
------------------
Sobel mediation test: does IN mediate the DN → contribution path?

Path diagram: DN --a--> IN --b--> contribution
                  \___________c'_________/  (direct)

Indirect effect = a * b
Sobel SE        = sqrt(b² * se_a² + a² * se_b²)
Sobel z         = a*b / Sobel SE
Proportion med. = a*b / c  (c = total effect of DN on contribution)

Two versions:
  Contemporaneous: X = DN_z, M = IN_z, Y = contribution_z
  Lagged:          X = DN_z, M = IN_z, Y = contribution_lead1_z
      (t DN and t IN, both elicited after round t's contribution, predict
       round t+1 contribution)

NOTE on timing: IN/DN are elicited AFTER contributions each round, so IN(t)
and DN(t) cannot causally explain contribution(t) - only contribution(t+1).
The contemporaneous model is therefore purely correlational. The lagged
model instead uses DN(t) and IN(t) (elicited together, after round t) to
predict contribution(t+1), so every arrow in the path respects elicitation
order.

All regressions use OLS with SEs clustered by run_id.
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm as _norm

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


def add_lags(df):
    df = df.sort_values(["run_id", "agent_id", "round"]).copy()
    grp = df.groupby(["run_id", "agent_id"])
    df["contribution_lead1"] = grp["contribution"].shift(-1)
    return df


def zscale(df, cols):
    for col in cols:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    return df


# ── Sobel test ────────────────────────────────────────────────────────────────

def sobel(a, se_a, b, se_b):
    """
    Returns indirect effect, Sobel SE, z-statistic, and two-tailed p-value.
    Goodman (1960) formula: SE = sqrt(b²·se_a² + a²·se_b²)
    """
    indirect = a * b
    se       = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    z        = indirect / se
    p        = float(2 * (1 - _norm.cdf(abs(z))))
    return indirect, se, z, p


def run_mediation(df, x_col, m_col, y_col, cluster_col):
    """
    Fits the three OLS models for Baron-Kenny / Sobel mediation.

    Returns a dict with all coefficients, SEs, and Sobel statistics.
    """
    def fit(formula):
        return smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )

    # Path a: M ~ X
    mA = fit(f"{m_col} ~ {x_col}")
    a     = mA.params[x_col]
    se_a  = mA.bse[x_col]
    p_a   = mA.pvalues[x_col]

    # Total effect: Y ~ X
    mC = fit(f"{y_col} ~ {x_col}")
    c     = mC.params[x_col]
    se_c  = mC.bse[x_col]
    p_c   = mC.pvalues[x_col]

    # Path b + direct: Y ~ M + X
    mB = fit(f"{y_col} ~ {m_col} + {x_col}")
    b      = mB.params[m_col]
    se_b   = mB.bse[m_col]
    p_b    = mB.pvalues[m_col]
    c_d    = mB.params[x_col]   # direct effect c'
    se_c_d = mB.bse[x_col]
    p_c_d  = mB.pvalues[x_col]

    indirect, se_ind, z_ind, p_ind = sobel(a, se_a, b, se_b)

    prop_mediated = indirect / c if abs(c) > 1e-10 else np.nan

    return {
        "a": a, "se_a": se_a, "p_a": p_a,                    # X → M
        "b": b, "se_b": se_b, "p_b": p_b,                    # M → Y | X
        "c": c, "se_c": se_c, "p_c": p_c,                    # total
        "c_d": c_d, "se_c_d": se_c_d, "p_c_d": p_c_d,       # direct c'
        "indirect": indirect, "se_ind": se_ind,
        "z_ind": z_ind, "p_ind": p_ind,
        "prop_mediated": prop_mediated,
        "n_obs": len(df), "n_clusters": df[cluster_col].nunique(),
    }


# ── LaTeX table ───────────────────────────────────────────────────────────────

def fmt_coef(val, se, p):
    s = stars(p)
    return rf"{val:.3f} ({se:.3f}){SIG_TEX[s]}"

def fmt_val(val, p=None):
    if p is not None:
        return rf"{val:.3f}{SIG_TEX[stars(p)]}"
    return rf"{val:.3f}"


def make_table(contemp, lagged):
    n_obs_c = f"{contemp['n_obs']:,}"
    n_obs_l = f"{lagged['n_obs']:,}"
    n_cl_c  = f"{contemp['n_clusters']:,}"
    n_cl_l  = f"{lagged['n_clusters']:,}"
    pm_c    = f"{contemp['prop_mediated']:.1%}"
    pm_l    = f"{lagged['prop_mediated']:.1%}"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"  & Contemporaneous & Lagged ($t \to t+1$) \\",
        r"  & (X = DN$_{t,z}$, M = IN$_{t,z}$, Y = contribution$_{t,z}$) & (X = DN$_{t,z}$, M = IN$_{t,z}$, Y = contribution$_{t+1,z}$) \\",
        r"\midrule",
        r"  \multicolumn{3}{l}{\textit{Path a: X $\to$ M}} \\",
        rf"  \quad $a$ (X $\to$ IN$_z$) & {fmt_coef(contemp['a'], contemp['se_a'], contemp['p_a'])} & {fmt_coef(lagged['a'], lagged['se_a'], lagged['p_a'])} \\",
        r"  \multicolumn{3}{l}{\textit{Path b: M $\to$ Y $\mid$ X}} \\",
        rf"  \quad $b$ (IN$_z$ $\to$ contribution$_z$) & {fmt_coef(contemp['b'], contemp['se_b'], contemp['p_b'])} & {fmt_coef(lagged['b'], lagged['se_b'], lagged['p_b'])} \\",
        r"  \multicolumn{3}{l}{\textit{Effects}} \\",
        rf"  \quad Total $c$ (X $\to$ Y) & {fmt_coef(contemp['c'], contemp['se_c'], contemp['p_c'])} & {fmt_coef(lagged['c'], lagged['se_c'], lagged['p_c'])} \\",
        rf"  \quad Direct $c'$ (X $\to$ Y $\mid$ IN$_z$) & {fmt_coef(contemp['c_d'], contemp['se_c_d'], contemp['p_c_d'])} & {fmt_coef(lagged['c_d'], lagged['se_c_d'], lagged['p_c_d'])} \\",
        rf"  \quad Indirect $a \cdot b$ & {fmt_coef(contemp['indirect'], contemp['se_ind'], contemp['p_ind'])} & {fmt_coef(lagged['indirect'], lagged['se_ind'], lagged['p_ind'])} \\",
        rf"  \quad Sobel $z$ & {contemp['z_ind']:.3f} & {lagged['z_ind']:.3f} \\",
        rf"  \quad \% mediated & {pm_c} & {pm_l} \\",
        r"\midrule",
        rf"  $N$ & {n_obs_c} & {n_obs_l} \\",
        rf"  Clusters & {n_cl_c} & {n_cl_l} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Sobel mediation test: IN as mediator of DN $\to$ contribution. "
         r"\textit{Contemporaneous}: X, M, Y all from round $t$ (correlational only, "
         r"since IN/DN are elicited after contribution each round). "
         r"\textit{Lagged}: X = DN and M = IN, both elicited after round $t$'s "
         r"contribution; Y = contribution at $t+1$ (last round dropped), so every "
         r"path respects elicitation order. "
         r"Path coefficients from OLS with SEs clustered by run "
         r"(model $\times$ seed $\times$ condition); all variables z-scored. "
         r"Indirect effect SE uses the Goodman (1960) formula: "
         r"$\sqrt{b^2 \cdot \widehat{\mathrm{Var}}(a) + a^2 \cdot \widehat{\mathrm{Var}}(b)}$. "
         r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}"),
        r"\label{tab:sobel_mediation}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    raw = load_all()
    print(f"  Raw: {len(raw):,} obs | {raw['run_id'].nunique()} run_ids")

    # ── Contemporaneous ───────────────────────────────────────────────────────
    df_c = raw.copy()
    df_c = zscale(df_c, ["IN", "DN", "contribution"])
    df_c = df_c.rename(columns={"IN_z": "IN_z", "DN_z": "DN_z",
                                  "contribution_z": "contribution_z"})

    print("\nRunning contemporaneous mediation …")
    contemp = run_mediation(df_c, x_col="DN_z", m_col="IN_z",
                            y_col="contribution_z", cluster_col="run_id")

    # ── Lagged ────────────────────────────────────────────────────────────────
    df_l = add_lags(raw)
    df_l = df_l.dropna(subset=["contribution_lead1"]).copy()
    df_l = zscale(df_l, ["IN", "DN", "contribution", "contribution_lead1"])

    print("Running lagged mediation (DN_t → IN_t → contribution_{t+1}) …")
    lagged = run_mediation(df_l, x_col="DN_z", m_col="IN_z",
                           y_col="contribution_lead1_z", cluster_col="run_id")

    # ── Output ────────────────────────────────────────────────────────────────
    tex = make_table(contemp, lagged)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "sobel_mediation.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\n  Saved → {out_path}")
    print("\n" + tex)

    # ── Plain-text summary ────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("SUMMARY")
    print(sep)

    for label, res in [("Contemporaneous", contemp), ("Lagged (t → t+1)", lagged)]:
        print(f"\n  [{label}]  N={res['n_obs']:,}  clusters={res['n_clusters']}")
        print(f"    Path a  (DN → IN):          β={res['a']:.3f}  SE={res['se_a']:.3f}  p={res['p_a']:.4f}  {stars(res['p_a'])}")
        print(f"    Path b  (IN → contrib|DN):  β={res['b']:.3f}  SE={res['se_b']:.3f}  p={res['p_b']:.4f}  {stars(res['p_b'])}")
        print(f"    Total c (DN → contrib):     β={res['c']:.3f}  SE={res['se_c']:.3f}  p={res['p_c']:.4f}  {stars(res['p_c'])}")
        print(f"    Direct c' (DN → contrib|IN):β={res['c_d']:.3f}  SE={res['se_c_d']:.3f}  p={res['p_c_d']:.4f}  {stars(res['p_c_d'])}")
        print(f"    Indirect a·b:               {res['indirect']:.3f}  SE={res['se_ind']:.3f}")
        print(f"    Sobel z = {res['z_ind']:.3f}   p = {res['p_ind']:.4f}  {stars(res['p_ind'])}")
        print(f"    Proportion mediated: {res['prop_mediated']:.1%}")

        if res["p_ind"] < 0.05:
            if abs(res["c_d"]) < 0.05 or res["prop_mediated"] > 0.80:
                med_type = "full"
            else:
                med_type = "partial"
            print(f"    → Significant {med_type} mediation by IN")
        else:
            print("    → Mediation NOT significant")

    print(sep)


if __name__ == "__main__":
    main()
