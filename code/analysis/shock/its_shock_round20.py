"""
its_shock_round20.py
--------------------
Interrupted time series (ITS) analysis of the round-20 adversarial shock.

DID is not identified for the round-20 injection because the unshocked control
arm ends at round 20, leaving no control observations in the post-shock window
(rounds 21–25).  ITS uses the shocked arm's own pre-injection trend as the
counterfactual.

Analysis window: rounds 11–25 (drop rounds 1–10 norm-formation transient).
  pre-shock:  rounds 11–20  →  round_c ∈ [−9, 0]
  post-shock: rounds 21–25  →  round_c ∈ [1, 5]
  round_c = round − 20  (injection point = 0)
  post_shock = 1 if round > 20, else 0

Per-family model (clustered SEs by run_id):
  contribution ~ round_c * post_shock

Pooled model (cross-family heterogeneity):
  contribution ~ round_c * post_shock * C(family, Treatment('Llama-7B'))

Outputs
-------
  figures/2026-03-22/paper_stats/its_shock_round20.tex
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

BASE     = "/data3/rasimura/social-norm-evo"
CODE_RES = f"{BASE}/code/results"
OUT_DIR  = f"{BASE}/figures/2026-03-22/paper_stats"

CONDITION       = "FULL"
ADV_ID          = 12
INTRO_ROUND     = 20
ANALYSIS_ROUNDS = list(range(11, 26))   # 11–25 inclusive

SPECS = [
    ("llama",       "Llama-7B"),
    ("mistral",     "Mistral-7B"),
    ("qwen",        "Qwen-7B"),
    ("llama_13b",   "Llama-13B"),
    ("mistral_13b", "Mistral-13B"),
    ("qwen_14b",    "Qwen-14B"),
    ("llama_70b",   "Llama-70B"),
    ("qwen_72b",    "Qwen-72B"),
]
SEEDS        = list(range(42, 52))
FAMILY_ORDER = [f for _, f in SPECS]
REF_FAMILY   = "Llama-7B"


# ── Helpers ───────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""

SIG_TEX = {"***": r"$^{***}$", "**": r"$^{**}$",
           "*":   r"$^{*}$",   "†":  r"$^{\dagger}$", "": ""}

def fmt(b, se, p):
    return rf"{b:.3f} ({se:.3f}){SIG_TEX[stars(p)]}"

def get3(res, term):
    if res is None or term not in res.params:
        return np.nan, np.nan, np.nan
    return float(res.params[term]), float(res.bse[term]), float(res.pvalues[term])

def p0(b, se):
    if np.isnan(b) or np.isnan(se) or se <= 0:
        return np.nan
    return float(2 * _norm.sf(abs(b / se)))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(pattern):
    for path in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(path))
            if d.get("condition") == CONDITION:
                return d
        except Exception:
            continue
    return None


def load_data():
    rows = []
    for model_key, family in SPECS:
        for seed in SEEDS:
            pat = os.path.join(
                CODE_RES, model_key, "local_newintro",
                f"intro{INTRO_ROUND}_adversarial",
                f"seed{seed}", "log_*.json")
            d = load_log(pat)
            if d is None:
                continue
            run_id = f"{family}_s{seed}"
            for r in d.get("round_logs", []):
                rnum = r["round"]
                if rnum not in ANALYSIS_ROUNDS:
                    continue
                for aid_str, val in r["contributions"].items():
                    if int(aid_str) == ADV_ID:
                        continue
                    rows.append({
                        "family":       family,
                        "run_id":       run_id,
                        "round":        rnum,
                        "agent_id":     int(aid_str),
                        "contribution": float(val),
                    })

    df = pd.DataFrame(rows)
    df["round_c"]    = df["round"] - INTRO_ROUND
    df["post_shock"] = (df["round"] > INTRO_ROUND).astype(float)
    df["family"]     = pd.Categorical(df["family"], categories=FAMILY_ORDER)
    return df


# ── Models ────────────────────────────────────────────────────────────────────

def fit_ols(df, formula):
    clean = df.dropna(subset=["contribution", "round_c", "post_shock", "run_id"]
                      ).reset_index(drop=True)
    if len(clean) < 10 or clean["run_id"].nunique() < 2:
        return None
    return smf.ols(formula, data=clean).fit(
        cov_type="cluster", cov_kwds={"groups": clean["run_id"]})


def run_family_its(df, family):
    sub = df[df["family"] == family].copy()
    return fit_ols(sub, "contribution ~ round_c * post_shock"), sub


def run_pooled_its(df):
    FAM = f"C(family, Treatment('{REF_FAMILY}'))"
    return fit_ols(df.copy(), f"contribution ~ round_c * post_shock * {FAM}")


# ── Wald test for cross-family heterogeneity ──────────────────────────────────

def wald_heterogeneity(pooled_res):
    if pooled_res is None:
        return None, None, None
    params = list(pooled_res.params.index)

    level_terms = [p for p in params
                   if "post_shock" in p and "C(family" in p and "round_c" not in p]
    slope_terms = [p for p in params
                   if "round_c" in p and "post_shock" in p and "C(family" in p]
    all_terms   = level_terms + slope_terms

    def wald(terms):
        if not terms:
            return None
        R = np.zeros((len(terms), len(params)))
        for i, t in enumerate(terms):
            R[i, params.index(t)] = 1.0
        wt = pooled_res.wald_test(R, use_f=False)
        return float(wt.statistic), float(wt.pvalue), len(terms)

    return wald(level_terms), wald(slope_terms), wald(all_terms)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def make_tex(family_results, n_obs, n_runs):
    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"  Family & Pre-shock slope & Level change & Slope change \\",
        r"  & (round\_c, rounds 11--20) & (post\_shock) "
        r"& (round\_c $\times$ post\_shock) \\",
        r"\midrule",
    ]
    for fam in FAMILY_ORDER:
        res, _ = family_results[fam]
        if res is None:
            lines.append(rf"  {fam} & --- & --- & --- \\")
            continue
        pb, pse, _  = get3(res, "round_c")
        lb, lse, _  = get3(res, "post_shock")
        sb, sse, _  = get3(res, "round_c:post_shock")
        pc = fmt(pb, pse, p0(pb, pse)) if not np.isnan(pb) else "---"
        lc = fmt(lb, lse, p0(lb, lse)) if not np.isnan(lb) else "---"
        sc = fmt(sb, sse, p0(sb, sse)) if not np.isnan(sb) else "---"
        lines.append(rf"  {fam} & {pc} & {lc} & {sc} \\")

    lines += [
        r"\midrule",
        rf"  $N$ (obs)  & \multicolumn{{3}}{{l}}{{{n_obs:,}}} \\",
        rf"  $N$ (runs) & \multicolumn{{3}}{{l}}{{{n_runs}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Interrupted time series (ITS) estimates of the round-20 adversarial-shock "
         r"effect on non-adversarial agent contributions. "
         r"Analysis window: rounds 11--25; round\_c $= $ round $- 20$ (injection at 0). "
         r"\emph{Pre-shock slope}: trend per round during rounds 11--20 "
         r"(expected $\approx 0$ if contribution norms are fully established). "
         r"\emph{Level change} (post\_shock): discontinuity in the level of contributions "
         r"at the injection point. "
         r"\emph{Slope change} (round\_c $\times$ post\_shock): change in trajectory "
         r"per round after injection relative to before. "
         r"The pre-injection trend serves as the counterfactual (no unshocked control "
         r"arm extends past round 20). "
         r"OLS per family, SEs clustered by run (family $\times$ seed). "
         r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}"),
        r"\label{tab:its_shock_round20}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Plain-text summary ────────────────────────────────────────────────────────

def print_summary(family_results, wald_level, wald_slope, wald_all, df):
    sep = "=" * 72
    print(f"\n{sep}")
    print("ITS ANALYSIS — Round-20 adversarial shock (rounds 11–25)")
    print(sep)

    print("\n  Sample sizes:")
    for fam in FAMILY_ORDER:
        sub  = df[df["family"] == fam]
        pre  = sub[sub["post_shock"] == 0]
        post = sub[sub["post_shock"] == 1]
        print(f"    {fam:14s}: {len(sub):5,} obs / {sub['run_id'].nunique():2d} runs "
              f"  (pre {len(pre):,} | post {len(post):,})")

    # ── Pre-shock trend check ─────────────────────────────────────────────────
    print("\n  PRE-SHOCK TREND (round_c, rounds 11–20)")
    print("  Expected ≈ 0 if norms fully established by round 10.")
    any_concern = False
    for fam in FAMILY_ORDER:
        res, _ = family_results[fam]
        if res is None:
            print(f"    {fam:14s}: ---")
            continue
        b, se, _ = get3(res, "round_c")
        pp = p0(b, se)
        if np.isnan(pp):
            note = ""
        elif pp < 0.05:
            note = "  ← NON-FLAT — weakens ITS counterfactual"
            any_concern = True
        elif pp < 0.10:
            note = "  ← marginal trend"
            any_concern = True
        else:
            note = ""
        print(f"    {fam:14s}: β={b:+.4f}  SE={se:.4f}  p={pp:.4f}  {stars(pp):4s}{note}")
    if not any_concern:
        print("  → All families: pre-shock trend ≈ flat. ITS counterfactual is plausible.")

    # ── Level change at injection ─────────────────────────────────────────────
    print("\n  LEVEL CHANGE AT INJECTION (post_shock)")
    print("  Discontinuity in contribution level at round 20→21.")
    for fam in FAMILY_ORDER:
        res, _ = family_results[fam]
        if res is None:
            print(f"    {fam:14s}: ---")
            continue
        b, se, _ = get3(res, "post_shock")
        pp = p0(b, se)
        if np.isnan(pp):     effect = "---"
        elif pp < 0.05:      effect = ("↓ DROP" if b < 0 else "↑ JUMP")
        elif pp < 0.10:      effect = ("↓ marginal drop" if b < 0 else "↑ marginal jump")
        else:                effect = "no sig. change"
        print(f"    {fam:14s}: β={b:+.4f}  SE={se:.4f}  p={pp:.4f}  {stars(pp):4s}  {effect}")

    # ── Slope change post-shock ───────────────────────────────────────────────
    print("\n  SLOPE CHANGE POST-SHOCK (round_c × post_shock)")
    print("  Change in trend per round after injection vs. before.")
    for fam in FAMILY_ORDER:
        res, _ = family_results[fam]
        if res is None:
            print(f"    {fam:14s}: ---")
            continue
        b, se, _    = get3(res, "round_c:post_shock")
        bp, sep_, _ = get3(res, "round_c")
        pp = p0(b, se)
        post_slope = bp + b if not np.isnan(bp) else np.nan
        if np.isnan(pp):    effect = "---"
        elif pp < 0.05:     effect = ("steeper ↑" if b > 0 else "flattened/reversed ↓")
        elif pp < 0.10:     effect = "marginal"
        else:               effect = "no sig. change"
        print(f"    {fam:14s}: β={b:+.4f}  SE={se:.4f}  p={pp:.4f}  {stars(pp):4s}  "
              f"{effect}  (post-shock slope ≈ {post_slope:+.4f})")

    # ── Wald tests ────────────────────────────────────────────────────────────
    print("\n  CROSS-FAMILY HETEROGENEITY (pooled Wald tests, chi-squared, clustered)")
    for label, wt in [
        ("Level  (post_shock × family)        ", wald_level),
        ("Slope  (round_c:post_shock × family)", wald_slope),
        ("Joint  (level + slope × family)     ", wald_all),
    ]:
        if wt is None:
            print(f"    {label}: ---")
        else:
            chi2, pval, df_ = wt
            print(f"    {label}: χ²({df_}) = {chi2:.2f},  p = {pval:.4f}  {stars(pval)}")

    # ── Comparison to round-10 DiD ────────────────────────────────────────────
    print("\n  COMPARISON TO ROUND-10 DiD")
    print("  Round-10 DiD (from shock_analysis.py):")
    print("    • Magnitude DiD: all n.s. — no significant immediate disruption for any family.")
    print("    • Full-recovery check: statistical full recovery for all families;")
    print("      Mistral-7B marginal (β=−1.19, p=0.077†).")
    print("  Round-20 ITS (this analysis):")
    print("    • See level/slope estimates above.")
    print("  Hypothesis: if longer norm-establishment → stronger resilience,")
    print("    expect smaller/absent disruption at round-20 vs. round-10.")
    print("  Hypothesis: if longer norm-establishment → stronger vulnerability,")
    print("    expect larger disruption at round-20.")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading ITS data …")
    df = load_data()
    print(f"  {len(df):,} observations  |  "
          f"{df['run_id'].nunique()} runs  |  "
          f"{df['family'].nunique()} families")

    family_results = {fam: run_family_its(df, fam) for fam in FAMILY_ORDER}

    pooled_res = run_pooled_its(df)
    wald_level, wald_slope, wald_all = wald_heterogeneity(pooled_res)

    os.makedirs(OUT_DIR, exist_ok=True)
    tex = make_tex(family_results, len(df), df["run_id"].nunique())
    out = os.path.join(OUT_DIR, "its_shock_round20.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"  Saved → {out}")

    print_summary(family_results, wald_level, wald_slope, wald_all, df)


if __name__ == "__main__":
    main()
