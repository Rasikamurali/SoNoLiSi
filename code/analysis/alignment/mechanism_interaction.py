"""
mechanism_interaction.py
------------------------
Test hindrance/synergy between three mechanisms on agent contribution:

  S  — Selection   (binary): on in NO_DISCUSSION + FULL
  D  — Discussion  (binary): on in NO_SELECTION  + FULL
  P  — Perception  (continuous): standardised injunctive norm (IN_z),
       always active; provides the third dimension via within-condition
       variation

Analysis restricted to conditions with ≥ 2 mechanisms active:
  NO_SELECTION  (D=1, P varies) — discussion + perception
  NO_DISCUSSION (S=1, P varies) — selection  + perception
  FULL          (S=1, D=1, P varies) — all three

Because S × D alone is collinear with intercept in these 3 conditions
(BASELINE is excluded), the key interaction is the three-way S:D:IN_z.
Model (per family, OLS with run-clustered SEs):

  contribution ~ S + D + IN_z + S:IN_z + D:IN_z + S:D:IN_z + round_c

  S:D is omitted — it is collinear with {Intercept, S, D} over the three
  conditions and would be unidentified without BASELINE.

Key quantities:
  β_S         — selection lift on intercept (vs NO_SELECTION base)
  β_D         — discussion lift on intercept (vs NO_DISCUSSION base)
  β_{IN}      — perception → contribution slope (in NO_SELECTION)
  β_{S:IN}    — does selection amplify/dampen perception slope?
  β_{D:IN}    — does discussion amplify/dampen perception slope?
  β_{S:D:IN}  — three-way: does FULL condition change the perception slope
                 beyond what S:IN + D:IN predict?
                 < 0  hindrance   (perception less impactful in FULL)
                 > 0  synergy     (perception more impactful in FULL)

IN_z is the same-round injunctive norm standardised within each
family × condition (zero-mean, unit-variance). Perception happens after
contribution in the same round; this is a concurrent association, not
causal. Round 1 IN_z is NaN for models with no prior context — rows with
missing IN are dropped.

Holm correction applied across 9 families on β_{S:D:IN} p-values.

Output:
  figures/2026-03-22/paper_stats/mechanism_interaction_all.tex
  figures/2026-03-22/paper_stats/mechanism_interaction_late.tex
"""

import json, glob, os, sys, warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

BASE      = "/data3/rasimura/social-norm-evo"
OUT_DIR   = f"{BASE}/figures/2026-03-22/paper_stats"
GROUP_SIZE = 4

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

FAMILY_ORDER = [
    "GPT", "Llama-7B", "Mistral-7B", "Qwen-7B",
    "Llama-13B", "Mistral-13B", "Qwen-14B", "Llama-70B", "Qwen-72B",
]

# Conditions with ≥ 2 mechanisms active
ANALYSIS_CONDITIONS = {"NO_SELECTION", "NO_DISCUSSION", "FULL"}
S_CONDITIONS        = {"NO_DISCUSSION", "FULL"}
D_CONDITIONS        = {"NO_SELECTION",  "FULL"}

COEF_NAMES = ["Intercept", "S", "D", "IN_z", "S:IN_z", "D:IN_z", "S:D:IN_z", "round_c"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(model_key, family, results_dir, variant, seeds):
    """
    Load agent-level rows for the three analysis conditions,
    filtered to groups of exactly GROUP_SIZE.
    Includes same-round IN as the perception measure.
    """
    rows = []
    for seed in seeds:
        pattern = os.path.join(
            results_dir, model_key, variant, f"seed{seed}", "log_*.json"
        )
        for path in sorted(glob.glob(pattern)):
            try:
                d = json.load(open(path))
            except Exception:
                continue
            condition = d.get("condition", "")
            if condition not in ANALYSIS_CONDITIONS:
                continue

            round_logs = d.get("round_logs", [])
            if not round_logs:
                continue
            max_round = max(r["round"] for r in round_logs)
            run_id    = f"{family}_s{seed}_{condition}"

            for r in round_logs:
                rnum    = r["round"]
                contribs = {int(k): float(v) for k, v in r["contributions"].items()}
                percs    = {int(k): v for k, v in (r.get("perceptions") or {}).items()
                            if v and v.get("injunctive_norm") is not None}

                # group-size map for this round
                gsz = {}
                for g in r.get("groups", []):
                    for aid in g:
                        gsz[int(aid)] = len(g)

                for aid, contrib in contribs.items():
                    if gsz.get(aid) != GROUP_SIZE:
                        continue
                    perc = percs.get(aid, {})
                    in_val = perc.get("injunctive_norm") if perc else None
                    rows.append({
                        "family":       family,
                        "condition":    condition,
                        "seed":         seed,
                        "run_id":       run_id,
                        "round":        rnum,
                        "max_round":    max_round,
                        "agent":        aid,
                        "contribution": contrib,
                        "IN":           in_val,
                        "S":            int(condition in S_CONDITIONS),
                        "D":            int(condition in D_CONDITIONS),
                    })
    return rows


def assign_period(df):
    df = df.copy()
    df["period"] = "mid"
    df.loc[df["round"] <= 3, "period"] = "early"
    df.loc[df["round"] >= df["max_round"] - 2, "period"] = "late"
    return df


def standardise_IN(df):
    """Z-score IN within each family × condition."""
    df = df.copy()
    grp = df.groupby(["family", "condition"])["IN"]
    df["IN_z"] = (df["IN"] - grp.transform("mean")) / grp.transform("std")
    return df


# ── Regression ────────────────────────────────────────────────────────────────

MAX_COEF = 1e6  # flag numerical blowup

def run_regression(df):
    """
    OLS: contribution ~ S + D + IN_z + S:IN_z + D:IN_z + S:D:IN_z + round_c
    Clustered by run_id. Drops rows with missing IN_z.
    Returns empty dict if the model is numerically unstable (near-zero IN variance
    in any condition causes blowup in the three-way term).
    """
    sub = df.dropna(subset=["IN_z"]).copy()
    if len(sub) < 30:
        return {}, 0

    # Guard: if IN_z has near-zero variance in any condition×S×D cell, the
    # interaction terms are not identifiable — skip rather than report garbage.
    for (s, d), grp in sub.groupby(["S", "D"]):
        if grp["IN_z"].std() < 1e-4:
            print(f"  Skipping: near-zero IN_z variance in S={s}, D={d} cell")
            return {}, 0

    sub["round_c"] = sub["round"] - sub["round"].mean()
    formula = ("contribution ~ S + D + IN_z"
               " + S:IN_z + D:IN_z + S:D:IN_z"
               " + round_c")
    try:
        res = smf.ols(formula, data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["run_id"]}
        )
        params = {}
        for name in COEF_NAMES:
            if name not in res.params:
                continue
            c, se, p = res.params[name], res.bse[name], res.pvalues[name]
            # Discard if numerically blown up
            if abs(c) > MAX_COEF or np.isnan(c) or np.isnan(se):
                print(f"  Unstable coefficient for {name}: coef={c:.3g}, se={se:.3g} — dropping")
                return {}, 0
            params[name] = (c, se, p)
        return params, int(res.nobs)
    except Exception as e:
        print(f"  Regression failed: {e}")
        return {}, 0


# ── Formatting ────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return r"^{***}"
    if p < 0.01:  return r"^{**}"
    if p < 0.05:  return r"^{*}"
    if p < 0.10:  return r"^{\dagger}"
    return ""


def fmt_cell(coef, se, p, p_corr=None):
    s = stars(p_corr if p_corr is not None else p)
    return rf"${coef:+.3f}\ ({se:.3f}){s}$"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    all_rows = []
    for spec in MODEL_SPECS:
        rows = load_data(*spec)
        all_rows.extend(rows)
        print(f"  {spec[1]}: {len(rows):,} rows")

    df = assign_period(standardise_IN(pd.DataFrame(all_rows)))
    print(f"\nTotal: {len(df):,} obs  |  with IN: {df['IN'].notna().sum():,}")
    print(f"Conditions: {sorted(df['condition'].unique())}")

    results_all  = {}
    results_late = {}
    nobs_all     = {}
    nobs_late    = {}

    for fam in FAMILY_ORDER:
        sub      = df[df["family"] == fam]
        sub_late = sub[sub["period"] == "late"]

        n_in = sub["IN"].notna().sum()
        print(f"\n{fam}: {len(sub):,} obs ({n_in:,} with IN), "
              f"{sub['run_id'].nunique()} runs")

        params_all,  n_all  = run_regression(sub)
        params_late, n_late = run_regression(sub_late)

        results_all[fam]  = params_all
        results_late[fam] = params_late
        nobs_all[fam]     = n_all
        nobs_late[fam]    = n_late

        for label, params in [("all", params_all), ("late", params_late)]:
            key = "S:D:IN_z"
            if key in params:
                c, se, p = params[key]
                direction = "HINDRANCE" if c < 0 else "synergy  "
                print(f"  [{label:4s}] S:D:IN = {c:+.3f} ({se:.3f}), p={p:.3f}  → {direction}")

    # ── Holm correction on three-way p-values ─────────────────────────────────

    def holm_correct(results):
        key = "S:D:IN_z"
        fams = [f for f in FAMILY_ORDER if key in results.get(f, {})]
        p_arr = np.array([results[f][key][2] for f in fams])
        _, p_corr, _, _ = multipletests(p_arr, alpha=0.05, method="holm")
        return dict(zip(fams, p_corr))

    p_corr_all  = holm_correct(results_all)
    p_corr_late = holm_correct(results_late)

    for label, results, p_corr in [("all", results_all, p_corr_all),
                                   ("late", results_late, p_corr_late)]:
        print(f"\n── Holm-corrected S:D:IN p-values ({label} rounds) ──")
        for fam in FAMILY_ORDER:
            if fam in p_corr:
                c, se, p = results[fam]["S:D:IN_z"]
                print(f"  {fam:<14}: raw p={p:.3f}  corrected p={p_corr[fam]:.3f}"
                      f"  coef={c:+.3f}")

    # ── Condition means per family (for table reference) ──────────────────────

    cond_means = (
        df.groupby(["family", "condition"])["contribution"]
        .mean()
        .unstack(fill_value=np.nan)
    )

    # ── LaTeX table ───────────────────────────────────────────────────────────

    def make_table(results, p_corr_dict, label):
        lines = [
            r"\begin{table}[ht]",
            r"\centering\small",
            (rf"\caption{{Mechanism interaction ({label} rounds): "
             r"\textit{contribution} $\sim$ $S + D + \mathrm{IN}_z$"
             r"$ + S{\cdot}\mathrm{IN}_z + D{\cdot}\mathrm{IN}_z$"
             r"$ + S{\cdot}D{\cdot}\mathrm{IN}_z + \mathrm{round}_c$. "
             r"OLS with run-clustered SE. "
             r"Restricted to conditions with $\geq 2$ mechanisms: "
             r"No Selection ($D{=}1$), No Discussion ($S{=}1$), Full ($S{=}D{=}1$). "
             r"$S$: Selection indicator. $D$: Discussion indicator. "
             r"$\mathrm{IN}_z$: injunctive norm standardised within family $\times$ condition "
             r"(concurrent perception measure). "
             r"$\beta_{S \cdot D \cdot \mathrm{IN}}$: three-way interaction --- "
             r"does the Full condition shift the perception$\to$contribution slope "
             r"beyond what $S{\cdot}\mathrm{IN} + D{\cdot}\mathrm{IN}$ predict? "
             r"$< 0$ hindrance; $> 0$ synergy. "
             r"Stars on $\beta_{S \cdot D \cdot \mathrm{IN}}$: Holm-corrected across 9 families. "
             r"All others uncorrected. "
             r"$^{\dagger}p{<}0.10$, $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
            r"\label{tab:mechanism_interaction_" + label + r"}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{l ccc ccc c c}",
            r"\toprule",
            (r"  & $\beta_S$ & $\beta_D$ & $\beta_{\mathrm{IN}}$"
             r" & $\beta_{S \cdot \mathrm{IN}}$ & $\beta_{D \cdot \mathrm{IN}}$"
             r" & $\beta_{S \cdot D \cdot \mathrm{IN}}$"
             r" & $\beta_{\mathrm{round}}$ & $N$ \\"),
            r"\midrule",
        ]

        for fam in FAMILY_ORDER:
            params = results.get(fam, {})

            def cell(name, p_override=None):
                if name not in params:
                    return "---"
                c, se, p = params[name]
                return fmt_cell(c, se, p, p_override)

            p3 = p_corr_dict.get(fam)
            row = (
                rf"  {fam}"
                rf" & {cell('S')}"
                rf" & {cell('D')}"
                rf" & {cell('IN_z')}"
                rf" & {cell('S:IN_z')}"
                rf" & {cell('D:IN_z')}"
                rf" & {cell('S:D:IN_z', p3)}"
                rf" & {cell('round_c')}"
                rf" & {nobs_all[fam] if label == 'all' else nobs_late.get(fam, 0)}"
                r" \\"
            )
            lines.append(row)

        lines += [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
        return "\n".join(lines) + "\n"

    os.makedirs(OUT_DIR, exist_ok=True)

    for label, results, p_corr in [
        ("all",  results_all,  p_corr_all),
        ("late", results_late, p_corr_late),
    ]:
        fname    = f"mechanism_interaction_{label}.tex"
        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "w") as f:
            f.write(make_table(results, p_corr, label))
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
