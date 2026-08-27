"""
render_supplementary_tables.py
-------------------------------
Formats the dataframes assembled by build_supplementary_tables.py into
the four supplementary tables (S1-S4): booktabs LaTeX + companion CSVs.

Run after build_supplementary_tables.py (imports and re-calls its
build_table_s{1,2,3,4} functions directly -- no data is recomputed here,
only formatted).

Output: figures/appendix and tables/
  table_temperature_robustness.tex   / temperature_robustness_table.csv
  table_prompt_variants.tex          / prompt_variants_table.csv
  table_prompt_sensitivity.tex       / prompt_sensitivity_table.csv
  table_parameter_sweep.tex          / parameter_sweep_table.csv
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/data3/rasimura/social-norm-evo/code/robustness")
from build_supplementary_tables import (  # noqa: E402
    BASE, OUT_DIR, MODELS, MODEL_LABELS,
    build_table_s1, build_table_s2, build_table_s3, build_table_s4,
)

MODEL_ORDER = ["openai", "llama", "mistral", "qwen"]


# ─── Formatting helpers ────────────────────────────────────────────────────────

def ms(mean, se, dec=2):
    if pd.isna(mean):
        return "---"
    return f"{mean:.{dec}f} ({se:.{dec}f})" if pd.notna(se) else f"{mean:.{dec}f} (---)"


def signed(x, dec=2):
    if pd.isna(x):
        return "---"
    return f"{x:+.{dec}f}"


def pfmt(p):
    if pd.isna(p):
        return "---"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi, dec=2):
    if pd.isna(lo) or pd.isna(hi):
        return "---"
    return f"[{lo:.{dec}f}, {hi:.{dec}f}]"


def esc(s):
    return str(s).replace("_", r"\_")


TEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=0.7in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{caption}

\begin{document}
\pagestyle{empty}
"""
TEX_END = r"""
\end{document}
"""


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S1 — Temperature robustness
# ═══════════════════════════════════════════════════════════════════════════

def render_s1(ss_all, ols_all, cv_df):
    temps = [0.3, 0.5, 0.7, 1.0]
    csv_rows = []

    def panel_rows(cond):
        rows_tex = []
        for model in MODEL_ORDER:
            cells = []
            for t in temps:
                r = ss_all[(ss_all.model == model) & (ss_all.condition == cond) & (ss_all.temperature == t)]
                mean = r["mean"].values[0] if len(r) else np.nan
                se   = r["se"].values[0] if len(r) else np.nan
                cells.append(ms(mean, se))
                csv_rows.append({"panel": cond, "model": model, "temperature": t,
                                  "mean": mean, "se": se})
            cv_r = cv_df[(cv_df.model == model) & (cv_df.condition == cond)]
            cv_val = cv_r["cv_pct"].values[0] if len(cv_r) else np.nan
            o = ols_all[(ols_all.model == model) & (ols_all.condition == cond)]
            beta = o["coef_temp"].values[0] if len(o) else np.nan
            se_b = o["se"].values[0] if len(o) else np.nan
            p_b  = o["p"].values[0] if len(o) else np.nan
            rows_tex.append(
                f"{MODEL_LABELS[model]} & " + " & ".join(cells) +
                f" & {cv_val:.1f} & {beta:.3f} & {se_b:.3f} & {pfmt(p_b)} \\\\"
            )
            csv_rows.append({"panel": cond, "model": model, "temperature": "CV_pct/beta/se/p",
                              "mean": cv_val, "se": se_b, "beta": beta, "p": p_b})
        return rows_tex

    pb_rows = panel_rows("PURE_BASELINE")
    bl_rows = panel_rows("BASELINE")

    tex = r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Table S1. Temperature robustness.} Mean contribution (SE across
5 seeds) at each decision temperature, coefficient of variation across the
four temperature-specific means (CV, \%), and the temperature coefficient
$\beta$ from contribution $\sim$ temperature $+$ round (seed-clustered SEs)
per model family. Steady state = last 5 of 20 rounds, seed-then-mean
aggregation (one mean per seed, then mean/SE across seeds).}
\label{tab:s1_temperature}
\begin{tabular}{lccccrrrr}
\toprule
\textbf{Model} & \textbf{T=.3} & \textbf{T=.5} & \textbf{T=.7} & \textbf{T=1.0} & \textbf{CV (\%)} & $\boldsymbol{\beta}_{\text{temp}}$ & \textbf{SE} & \textbf{p} \\
\midrule
\multicolumn{9}{l}{\textit{Panel A: Pure Baseline}} \\[2pt]
""" + "\n".join(pb_rows) + r"""
\midrule
\multicolumn{9}{l}{\textit{Panel B: Baseline}} \\[2pt]
""" + "\n".join(bl_rows) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\footnotesize
\textit{Notes.} $N=5$ seeds (43--47) per model $\times$ temperature $\times$ condition cell.
Backend: GPT-4o-mini (OpenAI API) and Llama 3.1-8B / Mistral-7B / Qwen2.5-7B (vLLM, local).
Endowment $=10$; steady-state window $=$ rounds 16--20.
Pre-existing pooled (GPT-4o-mini-only) temperature test, as originally reported and
reproduced exactly here as the \textit{openai} row above:
Pure Baseline $\beta=-0.108$, $p=0.303$; Baseline $\beta=-0.048$, $p=0.822$.
CV computed across the four steady-state temperature means within each model
$\times$ condition cell, not across seeds or rounds.
\end{table*}
"""
    with open(os.path.join(OUT_DIR, "table_temperature_robustness.tex"), "w") as f:
        f.write(TEX_PREAMBLE + tex + TEX_END)

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "temperature_robustness_table.csv"), index=False)
    return tex


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S2 — Structural prompt variability
# ═══════════════════════════════════════════════════════════════════════════

def render_s2(ss, gap, pooled):
    variants = ["standard", "minimal", "tendency_free"]
    variant_label = {"standard": "Standard", "minimal": "Minimal", "tendency_free": "Tendency-Free"}
    rows_tex = []
    csv_rows = []

    for model in MODEL_ORDER:
        for v in variants:
            pb = ss[(ss.model == model) & (ss.variant == v) & (ss.condition == "PURE_BASELINE")]
            fl = ss[(ss.model == model) & (ss.variant == v) & (ss.condition == "FULL")]
            g  = gap[(gap.model == model) & (gap.variant == v)]
            pb_m, pb_s = (pb["mean"].values[0], pb["se"].values[0]) if len(pb) else (np.nan, np.nan)
            fl_m, fl_s = (fl["mean"].values[0], fl["se"].values[0]) if len(fl) else (np.nan, np.nan)
            gap_v = g["gap"].values[0] if len(g) else np.nan
            se_g  = g["se_gap"].values[0] if len(g) else np.nan
            z_v   = g["z"].values[0] if len(g) else np.nan
            p_v   = g["p"].values[0] if len(g) else np.nan
            lo, hi = (g["ci_lo"].values[0], g["ci_hi"].values[0]) if len(g) else (np.nan, np.nan)

            rows_tex.append(
                f"{MODEL_LABELS[model]} & {variant_label[v]} & {ms(pb_m, pb_s)} & {ms(fl_m, fl_s)} & "
                f"{signed(gap_v)} & {ci(lo, hi)} & {z_v:.2f} & {pfmt(p_v)} \\\\"
            )
            csv_rows.append({"model": model, "variant": v, "pb_mean": pb_m, "pb_se": pb_s,
                              "full_mean": fl_m, "full_se": fl_s, "gap": gap_v, "se_gap": se_g,
                              "ci_lo": lo, "ci_hi": hi, "z": z_v, "p": p_v})

    pooled_rows = []
    for _, r in pooled.iterrows():
        pooled_rows.append(
            f"GPT-4o-mini (existing, pooled report) & {variant_label[r['variant']]} & "
            f"{r['pb_mean']:.2f} & {r['full_mean']:.2f} & {signed(r['gap'])} & --- & {r['z']:.2f} & {pfmt(r['p'])} \\\\"
        )

    tex = r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Table S2. Structural prompt-variant robustness.}
Steady-state (last 5 of 20 rounds) Pure Baseline and FULL contribution means,
and the FULL $-$ Pure Baseline treatment gap, per model $\times$ system-prompt
variant. \emph{Standard} presents cooperation tendency as an explicit numeric
value (the main-simulation prompt); \emph{Minimal} omits internal-state
guidance entirely; \emph{Tendency-Free} describes the same tendency
qualitatively rather than numerically. The gap is not expected to be
uniformly present -- it should be read for where it survives, where it
attenuates, and where it does not, by model family.}
\label{tab:s2_prompt_variants}
\footnotesize
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\textbf{Model} & \textbf{Prompt} & \textbf{PB Mean (SE)} & \textbf{Full Mean (SE)} & \textbf{Full$-$PB} & \textbf{95\% CI} & \textbf{z} & \textbf{p} \\
\midrule
""" + "\n".join(rows_tex) + r"""
\midrule
\multicolumn{8}{l}{\textit{Pooled panel (GPT-4o-mini only -- see note)}} \\[2pt]
""" + "\n".join(pooled_rows) + r"""
\bottomrule
\end{tabular}%
}

\vspace{4pt}
\footnotesize
\textit{Notes.} $N=5$ seeds (43--47) per cell. Gap SE $=\sqrt{SE_{\text{Full}}^2+SE_{\text{PB}}^2}$;
95\% CI $=$ gap $\pm\,1.96\times$SE; $z=$gap/SE, two-sided normal $p$.
\textbf{The ``pooled'' panel is GPT-4o-mini only, not a genuine estimate pooled across
model families} -- it reproduces the numbers originally reported from
analyze\_prompt\_variants.py, whose \texttt{RESULTS\_DIR} was hardcoded to the
\texttt{openai} backend. It is retained here for continuity with that prior report but
should not be read as a cross-model summary; the per-model rows above are the
genuine multi-family picture. No aggregate CI is reported for it because it was
not computed by the original script.
\end{table*}
"""
    with open(os.path.join(OUT_DIR, "table_prompt_variants.tex"), "w") as f:
        f.write(TEX_PREAMBLE + tex + TEX_END)

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "prompt_variants_table.csv"), index=False)
    return tex


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S3 — Word-level prompt sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def render_s3(ss, gap, r1, slope):
    variants = ["standard", "verb_only", "anchor_only", "verb_anchor"]
    variant_label = {"standard": "Standard", "verb_only": "Verb Only",
                      "anchor_only": "Anchor Only", "verb_anchor": "Verb + Anchor"}
    rows_tex = []
    csv_rows = []

    for model in MODEL_ORDER:
        for v in variants:
            pb = ss[(ss.model == model) & (ss.variant == v) & (ss.condition == "PURE_BASELINE")]
            fl = ss[(ss.model == model) & (ss.variant == v) & (ss.condition == "FULL")]
            g  = gap[(gap.model == model) & (gap.variant == v)]
            r1c = r1[(r1.model == model) & (r1.variant == v)]
            sc  = slope[(slope.model == model) & (slope.variant == v)]

            pb_m, pb_s = (pb["mean"].values[0], pb["se"].values[0]) if len(pb) else (np.nan, np.nan)
            fl_m, fl_s = (fl["mean"].values[0], fl["se"].values[0]) if len(fl) else (np.nan, np.nan)
            gap_v = g["gap"].values[0] if len(g) else np.nan
            z_v   = g["z"].values[0] if len(g) else np.nan
            p_v   = g["p"].values[0] if len(g) else np.nan
            r1_gap = r1c["gap_r1"].values[0] if len(r1c) else np.nan
            r1_p   = r1c["p"].values[0] if len(r1c) else np.nan
            beta   = sc["full_round_beta"].values[0] if len(sc) else np.nan
            beta_p = sc["full_round_p"].values[0] if len(sc) else np.nan

            rows_tex.append(
                f"{MODEL_LABELS[model]} & {variant_label[v]} & {ms(pb_m, pb_s)} & {ms(fl_m, fl_s)} & "
                f"{signed(gap_v)} & {z_v:.2f} & {pfmt(p_v)} & {signed(r1_gap)} & {beta:.3f} & {pfmt(beta_p)} \\\\"
            )
            csv_rows.append({"model": model, "variant": v, "pb_mean": pb_m, "pb_se": pb_s,
                              "full_mean": fl_m, "full_se": fl_s, "gap": gap_v, "z": z_v, "p": p_v,
                              "round1_gap": r1_gap, "round1_p": r1_p,
                              "full_round_beta_mixedlm": beta, "full_round_p_mixedlm": beta_p})

    tex = r"""
\begin{table*}[t]
\centering
\small
\caption{\textbf{Table S3. Word-level prompt sensitivity and mechanism-vs-pretraining dynamics.}
Built after structural variants showed the no-guidance prompt defaults to the
exact numeric midpoint (5.0) while the standard prompt does not ($\sim$3.7):
this isolates which specific words are responsible. \emph{Verb Only} replaces
the contribution/contributed verb pair with motivation-neutral phrasing;
\emph{Anchor Only} replaces the 0/1 scale description with motivation-neutral
behavioral language; \emph{Verb + Anchor} combines both. Round-1 gap and the
Full$\times$Round MixedLM slope increment (contribution $\sim$ round
$\times$ is\_full with a random intercept per seed) distinguish an
immediately-present gap from one that builds over repeated interaction.}
\label{tab:s3_prompt_sensitivity}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
\textbf{Model} & \textbf{Variant} & \textbf{PB Mean (SE)} & \textbf{Full Mean (SE)} & \textbf{Full$-$PB} & \textbf{z} & \textbf{p} & \textbf{Round-1 Gap} & $\boldsymbol{\beta}_{\text{Full}\times\text{Round}}$ & \textbf{p} \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}%
}

\vspace{4pt}
\footnotesize
\textit{Notes.} $N=5$ seeds (43--47) per cell. Full$-$PB gap SE $=\sqrt{SE_{\text{Full}}^2+SE_{\text{PB}}^2}$,
$z=$gap/SE. Round-1 gap uses the same unpaired SE combination on round-1-only
seed means. $\beta_{\text{Full}\times\text{Round}}$ is the \texttt{round:is\_full}
term from a per-(model, variant) mixed-effects model
(contribution $\sim$ round $\times$ is\_full, random intercept by seed, REML),
from \texttt{analyze\_mechanism\_vs\_pretraining.py} -- used in place of a
plain OLS Full$\times$Round interaction because it already accounts for the
repeated-measures structure. \textbf{A substantive result to note explicitly:}
for GPT-4o-mini, the \emph{Verb Only} ($z=6.97$) and \emph{Anchor Only}
($z=5.68$) gaps are both larger and more significant than the \emph{Standard}
gap ($z=1.22$) -- removing the fairness/anchor wording strengthens, rather
than eliminates, the FULL treatment effect. This is not a mechanism failure;
it is evidence the standard wording's own language partially competes with
the ostracism/selection mechanism rather than purely enabling it.
\end{table*}
"""
    with open(os.path.join(OUT_DIR, "table_prompt_sensitivity.tex"), "w") as f:
        f.write(TEX_PREAMBLE + tex + TEX_END)

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "prompt_sensitivity_table.csv"), index=False)
    return tex


# ═══════════════════════════════════════════════════════════════════════════
# TABLE S4 — Network-update parameter robustness
# ═══════════════════════════════════════════════════════════════════════════

def render_s4(master):
    pairs = ["pos0p5_neg0p5", "pos0p4_neg0p6", "pos0p2_neg0p8"]
    rows_tex, rows_secondary = [], []
    csv_rows = []

    for model in ["llama", "mistral", "qwen"]:
        for pair in pairs:
            r = master[(master.model == model) & (master.pair == pair)].iloc[0]
            ref_tag = r"\;(ref.)" if r["is_default"] else ""
            delta_pt = "--- (ref.)" if r["is_default"] else signed(r["delta_vs_default"])
            align_str = "---" if pd.isna(r["alignment_mean"]) else f"{r['alignment_mean']:.3f}"
            rows_tex.append(
                f"{MODEL_LABELS[model]} & {r['pos_rate']:.1f} & {r['neg_rate']:.1f}{ref_tag} & "
                f"{ms(r['contrib_mean'], r['contrib_se'])} & {delta_pt} & "
                f"{r['traj_beta']:.3f} & {align_str} & "
                f"{r['gini_mean']:.3f} & {r['top_share_mean']:.2f}$^{{\\text{{a}}}}$ \\\\"
            )
            delta_ci = "--- (ref.)" if r["is_default"] else ci(r["ci_lo"], r["ci_hi"])
            delta_p  = "---" if r["is_default"] else pfmt(r["p"])
            rows_secondary.append(
                f"{MODEL_LABELS[model]} & {r['pos_rate']:.1f}/{r['neg_rate']:.1f} & "
                f"{delta_ci} & {delta_p} & {r['traj_se']:.3f} & {pfmt(r['traj_p'])} & "
                f"{r['pct_tied_rounds']*100:.0f}\\% \\\\"
            )
            csv_rows.append(r.to_dict())

    tex = r"""
\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{\textbf{Table S4. Network-update rate-asymmetry robustness (FULL condition only).}
The run script's docstring describes a stale one-factor-at-a-time design over
three separate parameters; what is actually on disk is a joint sweep over
three (positive, negative) tie-update-rate pairs, FULL condition only, no
GPT-4o-mini run, no Pure Baseline arm -- reported as such, not backfilled.
This is a magnitude-sensitivity test, not an invariance test: the network
update rate directly parameterizes the social-selection mechanism, so
magnitude shifts are expected. Contribution and network metrics from the
last 5 of 20 rounds. Inferential statistics (95\% CI, $p$) for $\Delta$ and
trajectory $\beta$ are reported compactly in the lower panel and in full in
the companion CSV, to keep the primary panel focused on effect sizes.}
\label{tab:s4_parameter_sweep}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
\textbf{Model} & \textbf{Pos.\ rate} & \textbf{Neg.\ rate} & \textbf{Contribution Mean (SE)} & \textbf{$\Delta$ vs.\ Default} & \textbf{Traj.\ $\beta$} & \textbf{Weight--Contrib.\ $\rho$} & \textbf{Gini} & \textbf{Top-Q Share} \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}%
}

\vspace{6pt}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Pos/Neg} & \textbf{$\Delta$ 95\% CI} & \textbf{$\Delta$ p} & \textbf{Traj.\ SE} & \textbf{Traj.\ p} & \textbf{\% Tied Rounds} \\
\midrule
""" + "\n".join(rows_secondary) + r"""
\bottomrule
\end{tabular}

\vspace{4pt}
\footnotesize
\textit{Notes.} $N=5$ seeds (43--47) per model $\times$ pair, Llama 3.1-8B /
Mistral-7B / Qwen2.5-7B only (no GPT-4o-mini run exists for this experiment).
\textbf{Default-pair caveat:} the main simulation's actual default is
(pos$=0.10$, neg$=0.80$; see main-text Table of simulation parameters), which
is \emph{not} one of the three swept pairs. \texttt{pos0.2\_neg0.8} is the
nearest tested pair (exact match on the negative rate; 2$\times$ the positive
rate) and is used as the reference cell per instruction -- it is a nearby
analogue, not a replication, of the main-paper default.
$\Delta$ vs.\ Default uses an unpaired SE combination
($\sqrt{SE_{\text{pair}}^2+SE_{\text{default}}^2}$); \textbf{seed identifiers are
shared across pairs but are not matched replicates for behavioral outcomes}:
round-1 group formation is identical across pairs for a given seed
(confirming the Python/NumPy RNG stream is shared), but round-1
\emph{contributions} already diverge across pairs for the same seed because
agent decisions are sampled from the LLM backend, which is not controlled by
that seed. Trajectory $\beta$ is the contribution $\sim$ round coefficient
(seed-clustered SE, all 20 rounds, FULL only -- no baseline arm to
difference against). Weight--Contribution $\rho$ (Spearman), Gini, and
Top-Q Share use the exact definitions from
\texttt{eval\_network\_quantified.py} (the main-paper network analysis).
\% Tied Rounds is the share of (seed $\times$ round) observations in the
steady-state window where all 12 agents contributed the identical amount.
$^{\text{a}}$\textbf{Top-Q Share $=1.00$ in most cells is a mechanical
artifact, not evidence of concentrated influence:} within-round contributions
are tied across all 12 agents in 84--100\% of last-5-round observations in
every cell, which forces the 75th-percentile threshold to select every agent
and leaves Weight--Contribution $\rho$ undefined (\,---\,) whenever
contributions have zero within-round variance in every sampled round for a
cell. Gini remains interpretable throughout since edge weights still vary
even when contributions tie.
\end{table*}
"""
    with open(os.path.join(OUT_DIR, "table_parameter_sweep.tex"), "w") as f:
        f.write(TEX_PREAMBLE + tex + TEX_END)

    pd.DataFrame(csv_rows).to_csv(os.path.join(OUT_DIR, "parameter_sweep_table.csv"), index=False)
    return tex


if __name__ == "__main__":
    s1_ss, s1_ols, s1_cv = build_table_s1()
    s2_ss, s2_gap, s2_pooled = build_table_s2()
    s3_ss, s3_gap, s3_r1, s3_slope = build_table_s3()
    s4 = build_table_s4()

    render_s1(s1_ss, s1_ols, s1_cv)
    render_s2(s2_ss, s2_gap, s2_pooled)
    render_s3(s3_ss, s3_gap, s3_r1, s3_slope)
    render_s4(s4)

    print(f"Wrote 4 .tex + 4 .csv files to: {OUT_DIR}")
