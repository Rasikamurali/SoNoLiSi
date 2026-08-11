"""
shock_analysis.py
-----------------
Difference-in-differences: effect of adversarial-agent introduction on
non-adversarial agent contributions, by model family.

Two separate analyses:
  Round-10 injection: adversarial agent (ID 12) joins at round 11; runs = 15 rounds
  Round-20 injection: adversarial agent (ID 12) joins at round 21; runs = 25 rounds

Arms:
  Shocked  — code/results/<model>/local_newintro/intro{10,20}_adversarial/  seeds 42–51
  Control  — results/<model>/local/  (or code/results/<model>/local/ for 70B/72B)
             20-round unshocked runs, seeds 42–51, FULL condition only

GPT excluded (no shocked runs).

Period windows
  intro10:  pre=7–9, post_immediate=11–13, post_late=13–15
  intro20:  pre=17–19, post_immediate=21–23, post_late=23–25
  NOTE for intro20: control runs end at round 20 → post_immediate and post_late
  windows do NOT exist in the control arm.  The magnitude/recovery DID is therefore
  not identified for intro20.  We report:
    (a) a pre-period balance check per family (shocked vs control, rounds 17–19)
    (b) a within-shocked-arm before/after per family (pre → post_imm → post_late)

Condition: FULL throughout.
Clustering: by run_id = "{family}_{arm}_s{seed}".

Output
------
  figures/2026-03-22/paper_stats/shock_magnitude_round10.tex
  figures/2026-03-22/paper_stats/shock_recovery_round10.tex
  figures/2026-03-22/paper_stats/shock_magnitude_round20.tex   (fallback tables)
  figures/2026-03-22/paper_stats/shock_recovery_round20.tex
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

BASE      = "/data3/rasimura/social-norm-evo"
CODE_RES  = f"{BASE}/code/results"
MAIN_RES  = f"{BASE}/results"
OUT_DIR   = f"{BASE}/figures/2026-03-22/paper_stats"
CONDITION = "FULL"
ADV_ID    = 12

# ── Model specs ───────────────────────────────────────────────────────────────
# (model_key, family_label, shock_base, ctrl_base)
# Shocked arm always under: shock_base/<model>/local_newintro/intro{N}_adversarial/
# Control arm always under: ctrl_base/<model>/local/
SPECS = [
    ("llama",       "Llama-7B",    CODE_RES, MAIN_RES),
    ("mistral",     "Mistral-7B",  CODE_RES, MAIN_RES),
    ("qwen",        "Qwen-7B",     CODE_RES, MAIN_RES),
    ("llama_13b",   "Llama-13B",   CODE_RES, MAIN_RES),
    ("mistral_13b", "Mistral-13B", CODE_RES, MAIN_RES),
    ("qwen_14b",    "Qwen-14B",    CODE_RES, MAIN_RES),
    ("llama_70b",   "Llama-70B",   CODE_RES, CODE_RES),
    ("qwen_72b",    "Qwen-72B",    CODE_RES, CODE_RES),
]
SEEDS        = list(range(42, 52))
FAMILY_ORDER = ["Llama-7B", "Mistral-7B", "Qwen-7B",
                "Llama-13B", "Mistral-13B", "Qwen-14B",
                "Llama-70B", "Qwen-72B"]
REF_FAMILY   = "Llama-7B"
NON_REF      = [f for f in FAMILY_ORDER if f != REF_FAMILY]

CONFIGS = {
    10: dict(inject=11, total=15,
             pre=list(range(7,10)),
             post_imm=list(range(11,14)),
             post_late=list(range(13,16)),
             ctrl_max=20),
    20: dict(inject=21, total=25,
             pre=list(range(17,20)),
             post_imm=list(range(21,24)),
             post_late=list(range(23,26)),
             ctrl_max=20),
}


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
    return res.params[term], res.bse[term], res.pvalues[term]

def cell(res, term):
    b, se, p = get3(res, term)
    return "---" if np.isnan(b) else fmt(b, se, p)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(pattern, condition):
    for path in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(path))
            if d["condition"] == condition:
                return d
        except Exception:
            continue
    return None


def extract_rows(d, family, seed, arm):
    run_id = f"{family}_{arm}_s{seed}"
    for r in d["round_logs"]:
        rnum = r["round"]
        for aid_str, val in r["contributions"].items():
            if int(aid_str) == ADV_ID:
                continue
            yield {"family": family, "arm": arm, "seed": seed,
                   "run_id": run_id, "round": rnum,
                   "agent_id": int(aid_str), "contribution": float(val)}


def load_data(intro_round):
    cfg  = CONFIGS[intro_round]
    rows = []
    for model_key, family, shock_base, ctrl_base in SPECS:
        for seed in SEEDS:
            # Shocked arm
            shock_pat = os.path.join(
                shock_base, model_key, "local_newintro",
                f"intro{intro_round}_adversarial", f"seed{seed}", "log_*.json")
            d = load_log(shock_pat, CONDITION)
            if d:
                rows.extend(extract_rows(d, family, seed, "shocked"))

            # Control arm
            ctrl_pat = os.path.join(
                ctrl_base, model_key, "local", f"seed{seed}", "log_*.json")
            d = load_log(ctrl_pat, CONDITION)
            if d:
                rows.extend(extract_rows(d, family, seed, "control"))

    df = pd.DataFrame(rows)

    # Assign period labels; drop rows outside all windows
    all_rounds = set(cfg["pre"]) | set(cfg["post_imm"]) | set(cfg["post_late"])
    df = df[df["round"].isin(all_rounds)].copy()

    def period_of(r):
        if r in set(cfg["pre"]):      return "pre"
        if r in set(cfg["post_imm"]): return "post_immediate"
        return "post_late"

    df["period"] = df["round"].apply(period_of)
    df["family"] = pd.Categorical(df["family"], categories=FAMILY_ORDER)
    df["arm"]    = pd.Categorical(df["arm"],    categories=["control", "shocked"])
    return df


# ── OLS fitter ────────────────────────────────────────────────────────────────

def fit_ols(df, formula):
    clean = df.dropna(subset=["contribution","arm","family","period","run_id"]
                      ).reset_index(drop=True)
    if len(clean) < 2 or clean["run_id"].nunique() < 2:
        return None
    return smf.ols(formula, data=clean).fit(
        cov_type="cluster", cov_kwds={"groups": clean["run_id"]})


# ── Term builders ─────────────────────────────────────────────────────────────

def _did(period_val):
    return (f"C(period, Treatment('pre'))[T.{period_val}]"
            f":C(arm, Treatment('control'))[T.shocked]")

def _did_fam(period_val, fam):
    return (_did(period_val) +
            f":C(family, Treatment('{REF_FAMILY}'))[T.{fam}]")

def _rec(period_val="post_late"):
    return (f"C(period, Treatment('post_immediate'))[T.{period_val}]"
            f":C(arm, Treatment('control'))[T.shocked]")

def _rec_fam(fam):
    return _rec() + f":C(family, Treatment('{REF_FAMILY}'))[T.{fam}]"

def _arm():
    return f"C(arm, Treatment('control'))[T.shocked]"

def _arm_fam(fam):
    return _arm() + f":C(family, Treatment('{REF_FAMILY}'))[T.{fam}]"

def _ba(period_val):
    return f"C(period, Treatment('pre'))[T.{period_val}]"

def _ba_fam(period_val, fam):
    return (_ba(period_val) +
            f":C(family, Treatment('{REF_FAMILY}'))[T.{fam}]")

def _ba_late():
    return f"C(period, Treatment('post_immediate'))[T.post_late]"

def _ba_late_fam(fam):
    return _ba_late() + f":C(family, Treatment('{REF_FAMILY}'))[T.{fam}]"


# ── Model runners ─────────────────────────────────────────────────────────────

def fml(outcome, predictors):
    return f"{outcome} ~ {predictors}"

FAM = f"C(family, Treatment('{REF_FAMILY}'))"

def run_magnitude(df):
    sub = df[df["period"].isin(["pre","post_immediate"])].copy()
    f   = fml("contribution",
               f"C(period, Treatment('pre')) * C(arm, Treatment('control')) * {FAM}")
    return fit_ols(sub, f), sub

def run_recovery(df):
    sub = df[df["period"].isin(["post_immediate","post_late"])].copy()
    f   = fml("contribution",
               f"C(period, Treatment('post_immediate')) * C(arm, Treatment('control')) * {FAM}")
    return fit_ols(sub, f), sub

def run_full_check(df):
    sub = df[df["period"] == "post_late"].copy()
    f   = fml("contribution",
               f"C(arm, Treatment('control')) * {FAM}")
    return fit_ols(sub, f), sub

def run_ba(df):
    """Within-shocked arm: pre vs post_immediate vs post_late."""
    sub = df[df["arm"] == "shocked"].copy()
    f   = fml("contribution",
               f"C(period, Treatment('pre')) * {FAM}")
    return fit_ols(sub, f), sub

def run_ba_late(df):
    """Within-shocked arm: post_immediate vs post_late."""
    sub = df[(df["arm"] == "shocked") &
             df["period"].isin(["post_immediate","post_late"])].copy()
    f   = fml("contribution",
               f"C(period, Treatment('post_immediate')) * {FAM}")
    return fit_ols(sub, f), sub

def run_balance(df):
    """Pre-period only: shocked vs control."""
    sub = df[df["period"] == "pre"].copy()
    f   = fml("contribution",
               f"C(arm, Treatment('control')) * {FAM}")
    return fit_ols(sub, f), sub


# ── Cell-count helper ─────────────────────────────────────────────────────────

def counts(df):
    """Return {(arm, period): (n_obs, n_runs)}."""
    out = {}
    for (arm, period), g in df.groupby(["arm","period"]):
        out[(arm, period)] = (len(g), g["run_id"].nunique())
    return out

def nc(d, arm, period):
    obs, runs = d.get((arm, period), (0, 0))
    flag = r"\textsuperscript{!}" if 0 < runs < 5 else ""
    return f"{obs:,}/{runs}{flag}"


# ── Per-family extractor ──────────────────────────────────────────────────────

def per_family(res, ref_term_fn, delta_term_fn):
    """
    Returns list of (family, b_total, se_total, p_delta, p_vs_zero).

    b_total, se_total — point estimate and conservative SE of the total effect
                         for each family (ref + delta; SE ignores cross-covariance)
    p_delta           — p-value of the delta (interaction) term; tests whether
                         this family differs from the reference (Llama-7B)
    p_vs_zero         — large-sample two-sided p-value for the total effect vs. 0;
                         use this for direction significance labels
    """
    out = []
    b_ref, se_ref, p_ref = get3(res, ref_term_fn())
    p0_ref = (2 * _norm.sf(abs(b_ref / se_ref))
              if (se_ref > 0 and not np.isnan(b_ref)) else np.nan)
    out.append((REF_FAMILY, b_ref, se_ref, p_ref, p0_ref))
    for fam in NON_REF:
        db, dse, dp = get3(res, delta_term_fn(fam))
        if np.isnan(db):
            out.append((fam, np.nan, np.nan, np.nan, np.nan))
        else:
            b_tot  = b_ref + db
            se_tot = np.sqrt(se_ref**2 + dse**2) if not np.isnan(se_ref) else np.nan
            p0_tot = (2 * _norm.sf(abs(b_tot / se_tot))
                      if (se_tot > 0 and not np.isnan(b_tot)) else np.nan)
            out.append((fam, b_tot, se_tot, dp, p0_tot))
    return out


# ── LaTeX tables (intro10, full DID) ─────────────────────────────────────────

def make_magnitude_tex(mag_res, mag_sub, rec_res, rec_sub,
                       chk_res, chk_sub, intro_round):
    cfg = CONFIGS[intro_round]
    ct  = counts(mag_sub)

    # Per-family DID estimates
    mag_fam  = per_family(mag_res,
                           lambda: _did("post_immediate"),
                           lambda f: _did_fam("post_immediate", f))
    chk_fam  = per_family(chk_res, _arm, _arm_fam)
    # unpack 5-tuples; use p_vs_zero for display significance in table


    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"  Family & Magnitude DiD & Full-recovery check \\",
        rf"  & (pre $\to$ post-imm.) & (post-late only) \\",
        r"  & \multicolumn{1}{c}{$\Delta$contrib (shocked$-$control)} "
        r"& \multicolumn{1}{c}{arm[shocked]} \\",
        r"\midrule",
    ]
    for (fam, mb, mse, mp, mp0), (_, cb, cse, cp, cp0) in zip(mag_fam, chk_fam):
        mc = fmt(mb, mse, mp0) if not np.isnan(mb) else "---"
        cc = fmt(cb, cse, cp0) if not np.isnan(cb) else "---"
        lines.append(rf"  {fam} & {mc} & {cc} \\")

    r2_mag = f"{mag_res.rsquared:.3f}" if mag_res else "---"
    r2_chk = f"{chk_res.rsquared:.3f}" if chk_res else "---"
    n_mag  = len(mag_sub)
    n_chk  = len(chk_sub)
    cl_mag = mag_sub["run_id"].nunique()
    cl_chk = chk_sub["run_id"].nunique()

    lines += [
        r"\midrule",
        rf"  $R^2$     & {r2_mag} & {r2_chk} \\",
        rf"  $N$       & {n_mag:,} & {n_chk:,} \\",
        rf"  Clusters  & {cl_mag} & {cl_chk} \\",
        r"\midrule",
        r"  \multicolumn{3}{l}{\textit{Obs per cell (obs/runs)}} \\",
        rf"  \quad Control pre    & \multicolumn{{2}}{{l}}{{{nc(ct,'control','pre')}}} \\",
        rf"  \quad Control post-imm. & \multicolumn{{2}}{{l}}{{{nc(ct,'control','post_immediate')}}} \\",
        rf"  \quad Shocked  pre    & \multicolumn{{2}}{{l}}{{{nc(ct,'shocked','pre')}}} \\",
        rf"  \quad Shocked  post-imm. & \multicolumn{{2}}{{l}}{{{nc(ct,'shocked','post_immediate')}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (rf"\caption{{Shock magnitude and full-recovery check --- round-{intro_round} injection "
         rf"(adversarial agent joins round {cfg['inject']}). "
         rf"\textit{{Magnitude DiD}}: difference-in-differences estimator "
         rf"(period[post-imm.] $\times$ arm[shocked]) per family; "
         rf"pre=rounds {cfg['pre'][0]}--{cfg['pre'][-1]}, "
         rf"post-imm.=rounds {cfg['post_imm'][0]}--{cfg['post_imm'][-1]}. "
         rf"\textit{{Full-recovery check}}: shocked$-$control gap in post-late window "
         rf"(rounds {cfg['post_late'][0]}--{cfg['post_late'][-1]}); "
         rf"non-significant = statistical full recovery. "
         rf"OLS, SEs clustered by run (family $\times$ arm $\times$ seed). "
         rf"Reference family: {REF_FAMILY}. "
         rf"$\dagger p{{<}}.10$, $*p{{<}}.05$, $**p{{<}}.01$, $***p{{<}}.001$.}}"),
        rf"\label{{tab:shock_magnitude_r{intro_round}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def make_recovery_tex(rec_res, rec_sub, intro_round):
    cfg = CONFIGS[intro_round]
    ct  = counts(rec_sub)

    rec_fam = per_family(rec_res, _rec, _rec_fam)

    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"  Family & Recovery DiD \\",
        r"  & (post-imm.\ $\to$ post-late) \\",
        r"\midrule",
    ]
    for fam, b, se, p, p0 in rec_fam:
        c = fmt(b, se, p0) if not np.isnan(b) else "---"
        lines.append(rf"  {fam} & {c} \\")

    r2 = f"{rec_res.rsquared:.3f}" if rec_res else "---"
    lines += [
        r"\midrule",
        rf"  $R^2$    & {r2} \\",
        rf"  $N$      & {len(rec_sub):,} \\",
        rf"  Clusters & {rec_sub['run_id'].nunique()} \\",
        r"\midrule",
        r"  \multicolumn{2}{l}{\textit{Obs per cell (obs/runs)}} \\",
        rf"  \quad Control post-imm.  & {nc(ct,'control','post_immediate')} \\",
        rf"  \quad Control post-late  & {nc(ct,'control','post_late')} \\",
        rf"  \quad Shocked  post-imm. & {nc(ct,'shocked','post_immediate')} \\",
        rf"  \quad Shocked  post-late & {nc(ct,'shocked','post_late')} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (rf"\caption{{Shock recovery --- round-{intro_round} injection. "
         rf"DiD estimator (period[post-late] $\times$ arm[shocked]) per family. "
         rf"Positive = shocked arm closing gap with control from post-imm.\ to post-late. "
         rf"OLS, SEs clustered by run. "
         rf"$\dagger p{{<}}.10$, $*p{{<}}.05$, $**p{{<}}.01$, $***p{{<}}.001$.}}"),
        rf"\label{{tab:shock_recovery_r{intro_round}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── LaTeX tables (intro20, fallback) ─────────────────────────────────────────

def make_magnitude_tex_fallback(ba_res, ba_sub, bal_res, bal_sub, intro_round):
    cfg = CONFIGS[intro_round]
    ct_ba  = counts(ba_sub)
    ct_bal = counts(bal_sub)

    ba_fam  = per_family(ba_res,
                          lambda: _ba("post_immediate"),
                          lambda f: _ba_fam("post_immediate", f))
    bal_fam = per_family(bal_res, _arm, _arm_fam)

    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"  Family & Within-shocked B/A & Pre-period balance \\",
        r"  & (pre $\to$ post-imm., shocked only) & (shocked$-$control, pre only) \\",
        r"\midrule",
        rf"  \multicolumn{{3}}{{l}}{{\textbf{{NOTE: DID not identified "
        r"(control ends at round 20; post-shock windows missing)}}}}\\ ",
    ]
    for (fam, bb, bse, bp, bp0), (_, lb, lse, lp, lp0) in zip(ba_fam, bal_fam):
        bc = fmt(bb, bse, bp0) if not np.isnan(bb) else "---"
        lc = fmt(lb, lse, lp0) if not np.isnan(lb) else "---"
        lines.append(rf"  {fam} & {bc} & {lc} \\")

    r2_ba  = f"{ba_res.rsquared:.3f}"  if ba_res  else "---"
    r2_bal = f"{bal_res.rsquared:.3f}" if bal_res else "---"
    lines += [
        r"\midrule",
        rf"  $R^2$    & {r2_ba} & {r2_bal} \\",
        rf"  $N$      & {len(ba_sub):,} & {len(bal_sub):,} \\",
        rf"  Clusters & {ba_sub['run_id'].nunique()} & {bal_sub['run_id'].nunique()} \\",
        r"\midrule",
        r"  \multicolumn{3}{l}{\textit{Obs per cell (obs/runs)}} \\",
        rf"  \quad Shocked pre      & {nc(ct_ba,'shocked','pre')} & {nc(ct_bal,'shocked','pre')} \\",
        rf"  \quad Shocked post-imm.& {nc(ct_ba,'shocked','post_immediate')} & --- \\",
        rf"  \quad Control pre      & --- & {nc(ct_bal,'control','pre')} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (rf"\caption{{Shock analysis (fallback) --- round-{intro_round} injection "
         rf"(adversarial joins round {cfg['inject']}). "
         rf"DID is not identified: the unshocked control arm ends at round 20, "
         rf"so post-shock windows (rounds 21--25) have no control observations. "
         rf"\textit{{Within-shocked B/A}}: OLS on shocked arm only, "
         rf"period[post-imm.] vs.\ pre; "
         rf"pre=rounds {cfg['pre'][0]}--{cfg['pre'][-1]}, "
         rf"post-imm.=rounds {cfg['post_imm'][0]}--{cfg['post_imm'][-1]}. "
         rf"\textit{{Pre-period balance}}: shocked$-$control in pre-window (rounds "
         rf"{cfg['pre'][0]}--{cfg['pre'][-1]}); non-significant = arms comparable at baseline. "
         rf"OLS, SEs clustered by run. "
         rf"$\dagger p{{<}}.10$, $*p{{<}}.05$, $**p{{<}}.01$, $***p{{<}}.001$.}}"),
        rf"\label{{tab:shock_magnitude_r{intro_round}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def make_recovery_tex_fallback(ba_late_res, ba_late_sub, intro_round):
    cfg = CONFIGS[intro_round]
    ct  = counts(ba_late_sub)

    late_fam = per_family(ba_late_res,
                           _ba_late,
                           _ba_late_fam)

    lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"  Family & Within-shocked: post-imm.\ $\to$ post-late \\",
        r"\midrule",
        rf"  \multicolumn{{2}}{{l}}{{\textbf{{NOTE: DID not identified "
        r"(control ends at round 20)}}}}\\ ",
    ]
    for fam, b, se, p, p0 in late_fam:
        c = fmt(b, se, p0) if not np.isnan(b) else "---"
        lines.append(rf"  {fam} & {c} \\")

    r2 = f"{ba_late_res.rsquared:.3f}" if ba_late_res else "---"
    lines += [
        r"\midrule",
        rf"  $R^2$    & {r2} \\",
        rf"  $N$      & {len(ba_late_sub):,} \\",
        rf"  Clusters & {ba_late_sub['run_id'].nunique()} \\",
        r"\midrule",
        r"  \multicolumn{2}{l}{\textit{Obs per cell (obs/runs), shocked only}} \\",
        rf"  \quad post-imm.  & {nc(ct,'shocked','post_immediate')} \\",
        rf"  \quad post-late  & {nc(ct,'shocked','post_late')} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (rf"\caption{{Shock recovery (within-shocked arm) --- round-{intro_round} injection. "
         rf"Coefficient = change from post-imm.\ to post-late within the shocked arm; "
         rf"post-imm.=rounds {cfg['post_imm'][0]}--{cfg['post_imm'][-1]}, "
         rf"post-late=rounds {cfg['post_late'][0]}--{cfg['post_late'][-1]}. "
         rf"Positive = contributions continued to rise after the initial aftermath. "
         rf"OLS, SEs clustered by run. "
         rf"$\dagger p{{<}}.10$, $*p{{<}}.05$, $**p{{<}}.01$, $***p{{<}}.001$.}}"),
        rf"\label{{tab:shock_recovery_r{intro_round}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Plain-text summary ────────────────────────────────────────────────────────

def print_summary(intro_round, df, identified,
                  mag_res=None, rec_res=None, chk_res=None,
                  ba_res=None, ba_late_res=None, bal_res=None):
    cfg = CONFIGS[intro_round]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"SHOCK ANALYSIS — Round-{intro_round} injection "
          f"(adversarial joins round {cfg['inject']})")
    print(sep)

    ct = counts(df)
    print("\n  Cell sizes (obs/runs):")
    for arm in ["control","shocked"]:
        for period in ["pre","post_immediate","post_late"]:
            obs, runs = ct.get((arm, period), (0, 0))
            flag = "  ← MISSING" if runs == 0 else ("  ← thin" if runs < 5 else "")
            print(f"    {arm:8s} × {period:15s}: {obs:5,} / {runs:3d} runs{flag}")

    if not identified:
        print("\n  *** DID NOT IDENTIFIED: control ends at round 20; "
              "post-shock windows have no control observations.")
        print("  *** Reporting within-shocked before/after and pre-period balance.\n")

    # p legend: p_delta = differs from Llama-7B ref; p0 = total effect vs. zero (conserv. SE)
    if identified:
        print("\n  MAGNITUDE DiD (pre → post-immediate), per family:")
        print("  [β = total shocked−control DiD; * = p_vs_zero; Δ = sig. diff. from Llama-7B ref]")
        mag_fam = per_family(mag_res,
                              lambda: _did("post_immediate"),
                              lambda f: _did_fam("post_immediate", f))
        for fam, b, se, p_delta, p0 in mag_fam:
            if np.isnan(b):
                print(f"    {fam:14s}: ---")
            else:
                direction = ("↓" if b < 0 and p0 < 0.05
                             else "↑" if b > 0 and p0 < 0.05
                             else "n.s.")
                delta_flag = f"  Δref{stars(p_delta)}" if p_delta < 0.10 else ""
                print(f"    {fam:14s}: β={b:+.3f}  SE={se:.3f}  "
                      f"p_vs0={p0:.4f} {stars(p0):4s}  {direction}{delta_flag}")
    else:
        print("\n  WITHIN-SHOCKED B/A (pre → post-immediate), per family:")
        print("  [β = total effect; * = p_vs_zero; Δ = sig. diff. from Llama-7B ref]")
        ba_fam = per_family(ba_res,
                             lambda: _ba("post_immediate"),
                             lambda f: _ba_fam("post_immediate", f))
        for fam, b, se, p_delta, p0 in ba_fam:
            if np.isnan(b):
                print(f"    {fam:14s}: ---")
            else:
                delta_flag = f"  Δref{stars(p_delta)}" if p_delta < 0.10 else ""
                print(f"    {fam:14s}: β={b:+.3f}  SE={se:.3f}  "
                      f"p_vs0={p0:.4f} {stars(p0):4s}{delta_flag}")

        print("\n  PRE-PERIOD BALANCE (shocked vs control, rounds "
              f"{cfg['pre'][0]}–{cfg['pre'][-1]}), per family:")
        bal_fam = per_family(bal_res, _arm, _arm_fam)
        for fam, b, se, p_delta, p0 in bal_fam:
            if np.isnan(b):
                print(f"    {fam:14s}: ---")
            else:
                flag = "  *** IMBALANCED" if p0 < 0.05 else ""
                print(f"    {fam:14s}: β={b:+.3f}  SE={se:.3f}  "
                      f"p_vs0={p0:.4f} {stars(p0):4s}{flag}")

    # ── Recovery ──────────────────────────────────────────────────────────────
    if identified:
        print("\n  RECOVERY DiD (post-immediate → post-late), per family:")
        print("  [β = total (shocked gap widening = negative); * = p_vs_zero]")
        rec_fam = per_family(rec_res, _rec, _rec_fam)
        for fam, b, se, p_delta, p0 in rec_fam:
            if np.isnan(b):
                print(f"    {fam:14s}: ---")
            else:
                direction = ("closing" if b > 0 and p0 < 0.05
                             else "widening" if b < 0 and p0 < 0.05
                             else "stable")
                delta_flag = f"  Δref{stars(p_delta)}" if p_delta < 0.10 else ""
                print(f"    {fam:14s}: β={b:+.3f}  SE={se:.3f}  "
                      f"p_vs0={p0:.4f} {stars(p0):4s}  gap {direction}{delta_flag}")

        print("\n  FULL-RECOVERY CHECK (post-late, shocked vs control), per family:")
        chk_fam = per_family(chk_res, _arm, _arm_fam)
        for fam, b, se, p_delta, p0 in chk_fam:
            if np.isnan(b):
                print(f"    {fam:14s}: ---")
            else:
                status = ("FULL" if p0 >= 0.05 else "PARTIAL/NONE")
                print(f"    {fam:14s}: β={b:+.3f}  SE={se:.3f}  "
                      f"p_vs0={p0:.4f} {stars(p0):4s}  → {status} recovery")
    else:
        print("\n  WITHIN-SHOCKED (post-immediate → post-late), per family:")
        print("  [β = total; * = p_vs_zero; Δ = sig. diff. from Llama-7B ref]")
        late_fam = per_family(ba_late_res, _ba_late, _ba_late_fam)
        for fam, b, se, p_delta, p0 in late_fam:
            if np.isnan(b):
                print(f"    {fam:14s}: ---")
            else:
                direction = ("↑" if b > 0 and p0 < 0.05
                             else "↓" if b < 0 and p0 < 0.05
                             else "flat")
                delta_flag = f"  Δref{stars(p_delta)}" if p_delta < 0.10 else ""
                print(f"    {fam:14s}: β={b:+.3f}  SE={se:.3f}  "
                      f"p_vs0={p0:.4f} {stars(p0):4s}  {direction}{delta_flag}")

    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_one(intro_round):
    cfg = CONFIGS[intro_round]
    print(f"\nLoading data for intro-round {intro_round} …")
    df = load_data(intro_round)
    print(f"  Rows: {len(df):,}  |  Runs: {df['run_id'].nunique()}")

    ct = counts(df)
    ctrl_post = sum(obs for (arm, period), (obs, _) in ct.items()
                    if arm == "control" and period != "pre")
    identified = (ctrl_post > 0)

    os.makedirs(OUT_DIR, exist_ok=True)

    if identified:
        mag_res, mag_sub = run_magnitude(df)
        rec_res, rec_sub = run_recovery(df)
        chk_res, chk_sub = run_full_check(df)

        mag_tex = make_magnitude_tex(mag_res, mag_sub, rec_res, rec_sub,
                                     chk_res, chk_sub, intro_round)
        rec_tex = make_recovery_tex(rec_res, rec_sub, intro_round)

        out_mag = os.path.join(OUT_DIR, f"shock_magnitude_round{intro_round}.tex")
        out_rec = os.path.join(OUT_DIR, f"shock_recovery_round{intro_round}.tex")
        with open(out_mag, "w") as f: f.write(mag_tex)
        with open(out_rec, "w") as f: f.write(rec_tex)
        print(f"  Saved → {out_mag}")
        print(f"  Saved → {out_rec}")

        print_summary(intro_round, df, identified=True,
                      mag_res=mag_res, rec_res=rec_res, chk_res=chk_res)
    else:
        ba_res,      ba_sub      = run_ba(df)
        ba_late_res, ba_late_sub = run_ba_late(df)
        bal_res,     bal_sub     = run_balance(df)

        mag_tex = make_magnitude_tex_fallback(ba_res, ba_sub,
                                               bal_res, bal_sub, intro_round)
        rec_tex = make_recovery_tex_fallback(ba_late_res, ba_late_sub, intro_round)

        out_mag = os.path.join(OUT_DIR, f"shock_magnitude_round{intro_round}.tex")
        out_rec = os.path.join(OUT_DIR, f"shock_recovery_round{intro_round}.tex")
        with open(out_mag, "w") as f: f.write(mag_tex)
        with open(out_rec, "w") as f: f.write(rec_tex)
        print(f"  Saved (fallback) → {out_mag}")
        print(f"  Saved (fallback) → {out_rec}")

        print_summary(intro_round, df, identified=False,
                      ba_res=ba_res, ba_late_res=ba_late_res, bal_res=bal_res)


def main():
    run_one(10)
    run_one(20)


if __name__ == "__main__":
    main()
