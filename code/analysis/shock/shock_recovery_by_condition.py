"""
shock_recovery_by_condition.py
--------------------------------
Follow-up to dissociation_recovery_stats.py, restructured per the actual
question of interest: for each mechanism condition (Baseline, No Discussion,
No Selection, Full) and each injection round (10, 20), compare NE (belief)
and Contribution (behavior) on three things:

  (a) Shock propagation to incumbents -- how much do the 8 non-replaced
      ("incumbent") agents' own NE/contribution move, from their pre-shock
      baseline to the injection round, just from 4 of their groupmates being
      hijacked into adversarial behavior? Single-timepoint comparison at
      offset 1: |incumbent value - incumbent pre-shock baseline|, NE vs
      Contribution, OLS with C(model) controls (no time dimension -- this
      isn't a recovery-rate question, it's "how hard did incumbents get
      hit").

  (b) Restoration of incumbents -- do incumbents' own NE/contribution levels
      recover back toward their pre-shock baseline over the 5 post-injection
      rounds? |incumbent gap| ~ offset * NE + C(model), offsets 1..5,
      clustered by run. This is the "Level" framing from the original
      dissociation figure.

  (c) Reconvergence between adversarial and non-adversarial groups -- does
      the non-replaced-minus-replaced gap (adversarial vs incumbent agents)
      close over the same window? Same regression spec, but on the
      replaced-vs-non-replaced GAP (not incumbent levels). This is the "Gap"
      framing from the original figure.

For (b) and (c) the reported quantities are the same b1/b3 recovery-slope
decomposition as dissociation_recovery_stats.py's main model:
    |value|_{r,t} ~ b0 + b1*Time_t + b2*NE + b3*(Time_t x NE) + C(model) + e
b1 = Contribution's slope (reference level), b3 = NE's slope minus
Contribution's slope. For (a) there's no Time term (single timepoint):
    |value|_{r,1} ~ b0 + b2*NE + C(model) + e
b2 = NE's shock magnitude minus Contribution's.
All SEs clustered by run (model x seed).

Scope: pools llama/mistral/qwen (7-8B tier) only, seeds 42-51. GPT has no
round-10 shock data and its round-20 shock design (1 new agent joining, 12
incumbents) is structurally different from these three models' (4-of-12
agents hijacked, 8 incumbents) -- see conversation; excluded from pooling
rather than silently combined.

Output: one .tex table per condition,
  figures/2026-03-22/paper_stats/shock_recovery_{condition}.tex
copied into figures/MAIN_RESULTS/8_shock_treatment/.
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = "/data3/rasimura/social-norm-evo/code/results"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"
MAIN_RESULTS_DIR = "/data3/rasimura/social-norm-evo/figures/MAIN_RESULTS/8_shock_treatment"

MODELS = ["llama", "mistral", "qwen"]
SEEDS = list(range(42, 52))
CONDITIONS = ["BASELINE", "NO_DISCUSSION", "NO_SELECTION", "FULL"]
COND_LABELS = {
    "BASELINE": "Baseline", "NO_DISCUSSION": "No Discussion",
    "NO_SELECTION": "No Selection", "FULL": "Full",
}
ROUND_CONFIGS = [("round10", "replace10_random_adversarial", 10),
                  ("round20", "replace20_random_adversarial", 20)]


def _load(model, seed, variant, cond):
    pattern = os.path.join(BASE, model, "local_newgroup", variant,
                            f"seed{seed}", f"log_*_{cond}_seed{seed}.json")
    paths = glob.glob(pattern)
    return json.load(open(paths[0])) if paths else None


def build_gap(variant, pivot, cond):
    """Non-replaced-minus-replaced gap, offsets -1..5."""
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            d = _load(model, seed, variant, cond)
            if d is None:
                continue
            rl = d["round_logs"]
            replaced_ids = set(str(x) for x in
                                next(r["agents_replaced"] for r in rl if "agents_replaced" in r))
            for r in rl:
                offset = r["round"] - pivot
                if offset < -1 or offset > 5:
                    continue
                contrib = r["contributions"]
                perc = r["perceptions"]
                groups = r["groups"]

                def per_agent_gap(getter):
                    per_agent = {}
                    for g in groups:
                        g = [str(a) for a in g]
                        vals = [getter(i) for i in g if getter(i) is not None]
                        if not vals:
                            continue
                        gm = np.mean(vals)
                        for i in g:
                            v = getter(i)
                            if v is not None:
                                per_agent[i] = v - gm
                    return per_agent

                def gap(getter):
                    pa = per_agent_gap(getter)
                    nr = [pa[i] for i in pa if i not in replaced_ids]
                    rp = [pa[i] for i in pa if i in replaced_ids]
                    if not nr or not rp:
                        return None
                    return float(np.mean(nr) - np.mean(rp))

                rows.append(dict(
                    model=model, seed=seed, offset=offset,
                    belief=gap(lambda i: perc.get(i, {}).get("injunctive_norm")),
                    contribution=gap(lambda i: contrib.get(i)),
                ))
    return pd.DataFrame(rows)


def build_level(variant, pivot, cond):
    """Incumbents' (non-replaced agents') own raw levels, offsets -1..5."""
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            d = _load(model, seed, variant, cond)
            if d is None:
                continue
            rl = d["round_logs"]
            replaced_ids = set(str(x) for x in
                                next(r["agents_replaced"] for r in rl if "agents_replaced" in r))
            for r in rl:
                offset = r["round"] - pivot
                if offset < -1 or offset > 5:
                    continue
                contrib = r["contributions"]
                perc = r["perceptions"]
                nonrep = [i for i in contrib.keys() if i not in replaced_ids]
                c_vals = [contrib[i] for i in nonrep if i in contrib]
                b_vals = [perc[i]["injunctive_norm"] for i in nonrep
                          if i in perc and "injunctive_norm" in perc[i]]
                rows.append(dict(model=model, seed=seed, offset=offset,
                                  belief=np.mean(b_vals) if b_vals else None,
                                  contribution=np.mean(c_vals) if c_vals else None))
    return pd.DataFrame(rows)


def to_long(df):
    long = df.melt(id_vars=["model", "seed", "offset"],
                    value_vars=["belief", "contribution"],
                    var_name="outcome_type", value_name="value")
    long = long.dropna(subset=["value"])
    long["run_id"] = long["model"] + "_s" + long["seed"].astype(str)
    return long


def add_abs_gap(long):
    base = (long[long["offset"].isin([-1, 0])]
            .groupby(["run_id", "outcome_type"])["value"].mean()
            .rename("baseline").reset_index())
    post = long[long["offset"] >= 1].merge(base, on=["run_id", "outcome_type"], how="left")
    post["abs_gap"] = (post["value"] - post["baseline"]).abs()
    post["NE"] = (post["outcome_type"] == "belief").astype(int)
    return post


def fit_propagation(post):
    """(a) |gap|_{r,1} ~ NE + C(model), clustered by run -- single timepoint,
    no time dimension."""
    sub = post[post["offset"] == 1]
    res = smf.ols("abs_gap ~ NE + C(model)", data=sub).fit(
        cov_type="cluster", cov_kwds={"groups": sub["run_id"]})
    contrib_mag = res.params["Intercept"] + res.params[[k for k in res.params.index if k.startswith("C(model)")]].mean()
    # report raw group means directly (more transparent than intercept algebra)
    means = sub.groupby("NE")["abs_gap"].mean()
    return dict(
        ne_mag=means.get(1, float("nan")), contrib_mag=means.get(0, float("nan")),
        diff=res.params["NE"], diff_se=res.bse["NE"], diff_p=res.pvalues["NE"],
        n=int(res.nobs), clusters=sub["run_id"].nunique(),
    )


def fit_recovery(post, col="abs_gap"):
    """(b)/(c) <col> ~ offset * NE + C(model), clustered by run, offsets 1..5."""
    res = smf.ols(f"{col} ~ offset * NE + C(model)", data=post).fit(
        cov_type="cluster", cov_kwds={"groups": post["run_id"]})
    return dict(
        contrib_slope=res.params["offset"], contrib_se=res.bse["offset"],
        ne_diff=res.params["offset:NE"], ne_diff_se=res.bse["offset:NE"],
        ne_diff_p=res.pvalues["offset:NE"],
        n=int(res.nobs), clusters=post["run_id"].nunique(),
    )


def add_normalized(post):
    """normalized_t = |Y_t-Y_pre| / |Y_offset1-Y_pre|, rescaled by the
    POPULATION-mean (not each run's own) shock at offset 1 per outcome:
    ~15-20% of individual runs have abs_gap[offset==1] approx 0 by chance
    (checked empirically), so a per-run denominator blows up/divides by zero
    for a nontrivial share of the sample. The population mean is a fixed
    linear rescaling per outcome -- same significance-testing logic as the
    raw model, just in comparable 'fraction of shock' units."""
    gap0 = post.loc[post["offset"] == 1].groupby("outcome_type")["abs_gap"].mean().to_dict()
    post = post.copy()
    post["normalized"] = post.apply(lambda row: row["abs_gap"] / gap0[row["outcome_type"]], axis=1)
    return post


def stars(p):
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    if p < 0.10:
        return "$^{\\dagger}$"
    return ""


def main():
    for cond in CONDITIONS:
        rows_tex = []
        for tag, variant, pivot in ROUND_CONFIGS:
            gap_post = add_abs_gap(to_long(build_gap(variant, pivot, cond)))
            level_post = add_abs_gap(to_long(build_level(variant, pivot, cond)))

            prop = fit_propagation(level_post)
            restore = fit_recovery(level_post)
            reconv = fit_recovery(gap_post)
            restore_n = fit_recovery(add_normalized(level_post), col="normalized")
            reconv_n = fit_recovery(add_normalized(gap_post), col="normalized")

            round_label = "10" if tag == "round10" else "20"
            rows_tex.append(
                f"  {round_label} & (a) Shock propagation & "
                f"{prop['ne_mag']:.3f} & {prop['contrib_mag']:.3f} & "
                f"{prop['diff']:.3f} ({prop['diff_se']:.3f}){stars(prop['diff_p'])} \\\\"
            )
            rows_tex.append(
                f"   & (b) Restoration, raw slope & "
                f"{restore['contrib_slope']+restore['ne_diff']:.3f} & {restore['contrib_slope']:.3f} & "
                f"{restore['ne_diff']:.3f} ({restore['ne_diff_se']:.3f}){stars(restore['ne_diff_p'])} \\\\"
            )
            rows_tex.append(
                f"   & (b$'$) Restoration, normalized & "
                f"{restore_n['contrib_slope']+restore_n['ne_diff']:.3f} & {restore_n['contrib_slope']:.3f} & "
                f"{restore_n['ne_diff']:.3f} ({restore_n['ne_diff_se']:.3f}){stars(restore_n['ne_diff_p'])} \\\\"
            )
            rows_tex.append(
                f"   & (c) Reconvergence, raw slope & "
                f"{reconv['contrib_slope']+reconv['ne_diff']:.3f} & {reconv['contrib_slope']:.3f} & "
                f"{reconv['ne_diff']:.3f} ({reconv['ne_diff_se']:.3f}){stars(reconv['ne_diff_p'])} \\\\"
            )
            rows_tex.append(
                f"   & (c$'$) Reconvergence, normalized & "
                f"{reconv_n['contrib_slope']+reconv_n['ne_diff']:.3f} & {reconv_n['contrib_slope']:.3f} & "
                f"{reconv_n['ne_diff']:.3f} ({reconv_n['ne_diff_se']:.3f}){stars(reconv_n['ne_diff_p'])} \\\\"
            )
            print(f"[{cond} / {tag}] propagation: NE={prop['ne_mag']:.3f} Contrib={prop['contrib_mag']:.3f} "
                  f"diff={prop['diff']:.3f} p={prop['diff_p']:.4f} | "
                  f"restoration raw: NE={restore['contrib_slope']+restore['ne_diff']:.3f} "
                  f"Contrib={restore['contrib_slope']:.3f} diff={restore['ne_diff']:.3f} p={restore['ne_diff_p']:.4f} | "
                  f"restoration norm: NE={restore_n['contrib_slope']+restore_n['ne_diff']:.3f} "
                  f"Contrib={restore_n['contrib_slope']:.3f} diff={restore_n['ne_diff']:.3f} p={restore_n['ne_diff_p']:.4f} | "
                  f"reconvergence raw: NE={reconv['contrib_slope']+reconv['ne_diff']:.3f} "
                  f"Contrib={reconv['contrib_slope']:.3f} diff={reconv['ne_diff']:.3f} p={reconv['ne_diff_p']:.4f} | "
                  f"reconvergence norm: NE={reconv_n['contrib_slope']+reconv_n['ne_diff']:.3f} "
                  f"Contrib={reconv_n['contrib_slope']:.3f} diff={reconv_n['ne_diff']:.3f} p={reconv_n['ne_diff_p']:.4f}")

        tex = r"""\begin{table}[ht]
\centering\small
\begin{tabular}{llccc}
\toprule
  Round & Metric & NE & Contribution & Diff.\ (SE) \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\caption{Shock propagation, restoration, and reconvergence: NE (belief) vs.\ Contribution (behavior),
""" + COND_LABELS[cond] + r""" condition, pooled Llama/Mistral/Qwen (7--8B), seeds 42--51.
\textit{(a) Shock propagation}: $|{\rm gap}_{r,1}| \sim \beta_0+\beta_2\mathrm{NE}+C(\mathrm{model})+\epsilon$
at the injection round only (offset 1), gap $=$ incumbent value $-$ that run's own pre-shock baseline
(offsets $-1,0$) -- magnitude of disruption transmitted to the 8 non-replaced incumbents.
\textit{(b) Restoration}: same incumbent-level gap, now $\sim\beta_0+\beta_1\mathrm{Time}+\beta_2\mathrm{NE}+\beta_3(\mathrm{Time}\times\mathrm{NE})+C(\mathrm{model})+\epsilon$
over offsets 1--5 -- reported slopes are $\beta_1$ (Contribution) and $\beta_1+\beta_3$ (NE), Diff.\ is $\beta_3$.
\textit{(c) Reconvergence}: same slope model, but on the non-replaced-minus-replaced gap (adversarial vs.\
incumbent agents) instead of incumbent levels. \textit{(b$'$)/(c$'$)}: the same two models
re-fit on $|Y_t-Y_{\rm pre}|/|Y_{\rm offset1}-Y_{\rm pre}|$ instead of raw units -- rescaling each outcome
by its own shock magnitude (=1 at offset 1, $\to$0 as it fully recovers) so slopes are comparable
fractions-of-shock-closed-per-round rather than raw-unit slopes conflated with how far each outcome had
to travel. The denominator is the \textit{population-mean} $|{\rm gap}|$ at offset 1 for that outcome, not
each run's own: $\sim$15--20\% of individual runs have $|{\rm gap}_{r,1}|\approx0$ by chance, which makes a
per-run ratio unstable. All OLS, SEs clustered by run (model $\times$ seed).
$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}
\label{tab:shock_recovery_""" + cond.lower() + r"""}
\end{table}
"""
        out_path = os.path.join(OUT_DIR, f"shock_recovery_{cond.lower()}.tex")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"Saved -> {out_path}\n")


if __name__ == "__main__":
    main()
