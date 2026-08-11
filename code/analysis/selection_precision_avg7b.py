"""
Averaged version of selection_precision_combined.tex: pool round-level
observations across GPT and the three 7B models (Llama-7B, Mistral-7B,
Qwen-7B) into a single group, instead of one row per model. Plain
descriptive pooling (mean +/- SD across all pooled seed x round
observations) -- no OLS/regression, matching the underlying methodology of
the original per-model table (which itself is just grouped mean/SD, not a
significance test).

Reuses load_all()/assign_period() from selection_precision.py so the pooled
numbers are computed from the exact same round-level rows as the per-model
table, not from re-averaging the four already-rounded per-model cells.

Output: figures/2026-03-22/paper_stats/selection_precision_combined_avg.tex
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from selection_precision import load_all, assign_period, OUT_DIR  # noqa: E402

SMALL_FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
OUT_FNAME      = "selection_precision_combined_avg.tex"
N_THIN         = 15


def _mean_sd(sub, metric):
    vals = sub[metric].dropna()
    if len(vals) == 0:
        return np.nan, np.nan
    return float(vals.mean()), float(vals.std())


def fmt_pm(mean, sd):
    if np.isnan(mean):
        return "---"
    return rf"${mean:.2f} \pm {sd:.2f}$"


def fmt_pm_n(mean, sd, n):
    if n == 0 or np.isnan(mean):
        return "---"
    s = rf"${mean:.2f} \pm {sd:.2f}$"
    if n < N_THIN:
        s += r"$^{\dagger}$"
    s += rf"$_{{{n}}}$"
    return s


def main():
    print("Loading selection-precision data ...")
    df = load_all()
    df = assign_period(df)
    small = df[df["family"].isin(SMALL_FAMILIES)].copy()
    print(f"  Pooled round-observations (GPT + 7B): {len(small):,}")

    # ── No Discussion: Early / Late, Seed / Group ──────────────────────────
    nodisc       = small[small["condition"] == "NO_DISCUSSION"]
    nd_trivial   = int(nodisc["all_equal"].sum())
    nd_total     = len(nodisc)
    nd_early     = nodisc[nodisc["period"] == "early"]
    nd_late      = nodisc[nodisc["period"] == "late"]

    nd_e_seed = fmt_pm(*_mean_sd(nd_early, "seed_prec"))
    nd_e_grp  = fmt_pm(*_mean_sd(nd_early, "group_prec"))
    nd_l_seed = fmt_pm(*_mean_sd(nd_late,  "seed_prec"))
    nd_l_grp  = fmt_pm(*_mean_sd(nd_late,  "group_prec"))

    # ── Full: Early, Late (all incl.), Late (excl. trivial) ────────────────
    full       = small[small["condition"] == "FULL"]
    f_trivial  = int(full["all_equal"].sum())
    f_total    = len(full)
    f_early    = full[full["period"] == "early"]
    f_late     = full[full["period"] == "late"]
    f_late_nt  = f_late[f_late["all_equal"] == 0]
    n_nt       = len(f_late_nt)

    f_e_seed  = fmt_pm(*_mean_sd(f_early, "seed_prec"))
    f_e_grp   = fmt_pm(*_mean_sd(f_early, "group_prec"))
    f_lnt_seed_m, f_lnt_seed_s = _mean_sd(f_late_nt, "seed_prec")
    f_lnt_grp_m,  f_lnt_grp_s  = _mean_sd(f_late_nt, "group_prec")
    f_lnt_seed = fmt_pm_n(f_lnt_seed_m, f_lnt_seed_s, n_nt)
    f_lnt_grp  = fmt_pm_n(f_lnt_grp_m,  f_lnt_grp_s,  n_nt)

    pct_nd = 100.0 * nd_trivial / nd_total if nd_total else 0.0
    pct_f  = 100.0 * f_trivial  / f_total  if f_total  else 0.0

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r"  & \multicolumn{4}{c}{\textit{No Discussion}} & \multicolumn{4}{c}{\textit{Full}} \\",
        r"  \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"  & \multicolumn{2}{c}{Early (rounds 1--3)} & \multicolumn{2}{c}{Late ($T{-}2$ to $T$)} & \multicolumn{2}{c}{Early (rounds 1--3)} & \multicolumn{2}{c}{Late (excl.\ trivial)} \\",
        r"  \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        r"  Model & Seed & Group & Seed & Group & Seed & Group & Seed & Group \\",
        r"\midrule",
        rf"  GPT + 7B (pooled) & {nd_e_seed} & {nd_e_grp} & {nd_l_seed} & {nd_l_grp}"
        rf" & {f_e_seed} & {f_e_grp} & {f_lnt_seed} & {f_lnt_grp} \\",
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Selection precision under the \emph{No Discussion} and \emph{Full} "
         r"conditions, pooled across GPT and the three 7B models (Llama-7B, Mistral-7B, "
         r"Qwen-7B) -- every seed $\times$ round observation from all four models combined "
         r"into one distribution, rather than averaging the four per-model means. "
         r"\emph{Seed precision}: fraction of seed agents (group$_k[0]$, selected by weighted "
         r"random draw proportional to average incoming weight) whose round-$t{-}1$ "
         r"contribution was at or above the 75th percentile of that round's contributions. "
         r"\emph{Group precision}: same fraction for all group members. "
         r"Baseline (random selection) $\approx 0.25$. Cells report mean $\pm$ SD across the "
         r"pooled seeds and rounds in that period. "
         r"Full-Late excludes trivially all-equal rounds (all agents cooperate fully, which "
         r"would give precision $= 1.0$ by construction and inflate the mean); subscript shows "
         r"$N$ (number of non-trivial round-observations, pooled). `---' indicates $N=0$; "
         rf"$\dagger$ indicates $N{{<}}{N_THIN}$ (interpret with caution). "
         rf"Trivial round-pairs dropped (pooled): {pct_nd:.1f}\% (No Discussion, not excluded "
         rf"above), {pct_f:.1f}\% (Full, excluded above).}}"),
        r"\label{tab:selection_precision_combined_avg}",
        r"\end{table}",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, OUT_FNAME)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
