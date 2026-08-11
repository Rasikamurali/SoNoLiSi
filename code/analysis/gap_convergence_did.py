"""
gap_convergence_did.py
----------------------
DiD test: does the IN-behavior gap close faster than the DN-behavior gap?

IN_gap = abs(injunctive_norm  - contribution)
DN_gap = abs(descriptive_norm - contribution)

Primary model: gap_value ~ period * gap_type, cluster(run_id)
  period:   'early' (rounds 1-3) vs 'late' (last 3 rounds per run, dynamic)
  gap_type: 'IN' vs 'DN'  (reference: 'early', 'IN')

Robustness: gap_value ~ round_c * gap_type, rounds 1-6, cluster(run_id)
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
                            "family":    family,
                            "model":     model_key,
                            "condition": cond,
                            "seed":      seed,
                            "round":     r["round"],
                            "agent_id":  int(aid),
                            "run_id":    run_id,
                            "actual":    float(actual),
                            "IN":        float(inj),
                            "DN":        float(desc),
                        })
    df = pd.DataFrame(rows)
    df["IN_gap"] = (df["IN"] - df["actual"]).abs()
    df["DN_gap"] = (df["DN"] - df["actual"]).abs()
    return df


# ── Reshape ───────────────────────────────────────────────────────────────────

def to_long(df_wide):
    """Wide → long: one row per agent × round × gap_type."""
    id_cols = ["run_id", "family", "condition", "seed", "round", "agent_id"]
    in_rows = df_wide[id_cols + ["IN_gap"]].rename(columns={"IN_gap": "gap_value"})
    in_rows["gap_type"] = "IN"
    dn_rows = df_wide[id_cols + ["DN_gap"]].rename(columns={"DN_gap": "gap_value"})
    dn_rows["gap_type"] = "DN"
    return pd.concat([in_rows, dn_rows], ignore_index=True)


def assign_period(long_df):
    """
    Label 'early' (rounds 1-3) and 'late' (max_round-2 to max_round per run_id).
    Drop all other rounds. Handles variable run lengths.
    """
    max_round = (long_df.groupby("run_id")["round"]
                 .max().rename("max_round").reset_index())
    df = long_df.merge(max_round, on="run_id")

    early_mask = df["round"] <= 3
    late_mask  = df["round"] >= df["max_round"] - 2

    df = df[early_mask | late_mask].copy()
    df["period"] = np.where(df["round"] <= 3, "early", "late")
    return df.drop(columns=["max_round"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    wide = load_all()
    print(f"  {len(wide):,} agent-round obs | {wide['run_id'].nunique()} run_ids")

    long_all = to_long(wide)
    long     = assign_period(long_all)

    print(f"  Long (early+late only): {len(long):,} rows")
    print("\n  Observations per cell:")
    print(long.groupby(["gap_type", "period"])["gap_value"].count().unstack())

    # ── Cell means ────────────────────────────────────────────────────────────
    cell_means = (long.groupby(["gap_type", "period"])["gap_value"]
                  .mean().unstack())
    in_early = cell_means.loc["IN", "early"]
    in_late  = cell_means.loc["IN", "late"]
    dn_early = cell_means.loc["DN", "early"]
    dn_late  = cell_means.loc["DN", "late"]

    print(f"\n  Cell means:")
    print(f"    IN: early={in_early:.3f}  late={in_late:.3f}  Δ={in_late - in_early:+.3f}")
    print(f"    DN: early={dn_early:.3f}  late={dn_late:.3f}  Δ={dn_late - dn_early:+.3f}")

    # ── Primary DiD model ─────────────────────────────────────────────────────
    print("\nFitting DiD model …")
    did = smf.ols(
        "gap_value ~ C(period, Treatment('early')) * C(gap_type, Treatment('IN'))",
        data=long
    ).fit(cov_type="cluster", cov_kwds={"groups": long["run_id"]})

    # Coefficient keys
    k_late = "C(period, Treatment('early'))[T.late]"
    k_dn   = "C(gap_type, Treatment('IN'))[T.DN]"
    k_did  = ("C(period, Treatment('early'))[T.late]:"
               "C(gap_type, Treatment('IN'))[T.DN]")

    # IN within-type change: directly the period[T.late] coef
    in_chg    = did.params[k_late]
    in_chg_se = did.bse[k_late]
    in_chg_p  = did.pvalues[k_late]

    # DN within-type change: period + interaction (delta method)
    dn_chg = did.params[k_late] + did.params[k_did]
    cov    = did.cov_params()
    dn_se  = np.sqrt(cov.loc[k_late, k_late] + cov.loc[k_did, k_did]
                     + 2 * cov.loc[k_late, k_did])
    dn_z   = dn_chg / dn_se
    dn_p   = float(2 * (1 - _norm.cdf(abs(dn_z))))

    # DiD coefficient: interaction term
    did_coef = did.params[k_did]
    did_se   = did.bse[k_did]
    did_t    = did.tvalues[k_did]
    did_p    = did.pvalues[k_did]

    # ── Robustness: continuous round, rounds 1-6 ──────────────────────────────
    print("Fitting robustness model (rounds 1-6) …")
    long_r6 = long_all[long_all["round"] <= 6].copy()
    long_r6["round_c"] = long_r6["round"] - long_r6["round"].mean()

    rob = smf.ols(
        "gap_value ~ round_c * C(gap_type, Treatment('IN'))",
        data=long_r6
    ).fit(cov_type="cluster", cov_kwds={"groups": long_r6["run_id"]})

    k_rob    = "round_c:C(gap_type, Treatment('IN'))[T.DN]"
    rob_coef = rob.params[k_rob]
    rob_se   = rob.bse[k_rob]
    rob_p    = rob.pvalues[k_rob]

    # ── LaTeX table ───────────────────────────────────────────────────────────

    def fmt(val, se, p):
        return rf"{val:+.3f} ({se:.3f}){SIG_TEX[stars(p)]}"

    rob_note = (
        rf"Robustness (rounds 1--6, continuous): "
        rf"round\textsubscript{{c}}$\times$DN $= {rob_coef:+.3f}$ "
        rf"(SE $= {rob_se:.3f}$), $p = {rob_p:.3f}${SIG_TEX[stars(rob_p)]}."
    )

    n_obs = len(long)
    n_cl  = long["run_id"].nunique()

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"  & Early (rounds 1--3) & Late (last 3 rounds) \\",
        r"\midrule",
        rf"  IN gap & {in_early:.3f} & {in_late:.3f} \\",
        rf"  DN gap & {dn_early:.3f} & {dn_late:.3f} \\",
        r"\midrule",
        r"  \multicolumn{3}{l}{\textit{DiD estimates (OLS, SEs clustered by run)}} \\[2pt]",
        rf"  IN change (late $-$ early)  & \multicolumn{{2}}{{c}}{{{fmt(in_chg, in_chg_se, in_chg_p)}}} \\",
        rf"  DN change (late $-$ early)  & \multicolumn{{2}}{{c}}{{{fmt(dn_chg, dn_se, dn_p)}}} \\",
        rf"  DiD: DN change $-$ IN change & \multicolumn{{2}}{{c}}{{{fmt(did_coef, did_se, did_p)}}} \\",
        r"\midrule",
        rf"  $N$ (agent $\times$ period $\times$ type) & \multicolumn{{2}}{{c}}{{{n_obs:,}}} \\",
        rf"  Clusters (runs) & \multicolumn{{2}}{{c}}{{{n_cl:,}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Convergence of perception-behavior gaps. Cell means (top) show "
         r"raw absolute IN-gap and DN-gap in early (rounds 1--3) and late (last 3 rounds) "
         r"windows. DiD estimates (bottom) from "
         r"\texttt{gap\_value $\sim$ period $\times$ gap\_type}, SEs clustered by run "
         r"(model $\times$ seed $\times$ condition). Reference levels: \textit{early}, \textit{IN}. "
         r"$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$. "
         + rob_note + "}"),
        r"\label{tab:gap_convergence_did}",
        r"\end{table}",
    ]

    tex = "\n".join(lines) + "\n"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "gap_convergence_did.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\n  Saved → {out_path}")
    print("\n" + tex)

    # ── Plain-text summary ────────────────────────────────────────────────────
    sep = "=" * 60
    print(sep)
    print("SUMMARY")
    print(sep)

    in_delta = in_late - in_early
    dn_delta = dn_late - dn_early

    print(f"  IN gap: {in_early:.3f} → {in_late:.3f}  (Δ = {in_delta:+.3f},"
          f" {'closed' if in_delta < 0 else 'widened'})")
    print(f"  DN gap: {dn_early:.3f} → {dn_late:.3f}  (Δ = {dn_delta:+.3f},"
          f" {'closed' if dn_delta < 0 else 'widened'})")

    faster = "DN" if dn_delta < in_delta else "IN"
    slower = "IN" if faster == "DN" else "DN"
    print(f"\n  {faster} gap closed more  "
          f"(Δ{faster} = {min(in_delta, dn_delta):+.3f} "
          f"vs Δ{slower} = {max(in_delta, dn_delta):+.3f}, "
          f"difference = {did_coef:+.3f})")

    print(f"\n  DiD coefficient: {did_coef:+.3f}  SE={did_se:.3f}  p={did_p:.4f}"
          f"  ({stars(did_p) or 'n.s.'})")
    print(f"  IN within-type:  {in_chg:+.3f}  SE={in_chg_se:.3f}  p={in_chg_p:.4f}"
          f"  ({stars(in_chg_p) or 'n.s.'})")
    print(f"  DN within-type:  {dn_chg:+.3f}  SE={dn_se:.3f}  p={dn_p:.4f}"
          f"  ({stars(dn_p) or 'n.s.'})")

    print(f"\n  Robustness (rounds 1-6, continuous):")
    print(f"    round_c × DN: {rob_coef:+.4f}  SE={rob_se:.4f}  p={rob_p:.4f}"
          f"  ({stars(rob_p) or 'n.s.'})")
    agree = np.sign(rob_coef) == np.sign(did_coef)
    print(f"    Direction {'AGREES' if agree else 'DISAGREES'} with DiD")
    print(sep)


if __name__ == "__main__":
    main()
