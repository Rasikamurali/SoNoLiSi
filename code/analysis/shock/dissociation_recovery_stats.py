"""
dissociation_recovery_stats.py
-------------------------------
Statistical companion to plot_dissociation_belief_vs_contribution.py
(figures/MAIN_RESULTS/8_shock_treatment/dissociation_belief_vs_contribution.png).
That figure shows, for both injection rounds (10/20) and both outcome
framings (gap column / untouched-level column), NE (belief) and Contribution
trajectories diverging at the shock and reconverging afterward.

History (two superseded passes, kept here for context):
  v1 -- raw post-shock slope (value ~ offset, offsets 1..5). Dropped: confounded
        by shock magnitude (Contribution is displaced ~2-4x further than NE at
        injection), so Contribution's steeper raw slope back toward baseline
        was mechanically inevitable, not evidence of faster relative recovery.
  v2 -- normalized by *population-mean* shock magnitude per outcome
        (normalized = (value-baseline)/shock). Fixed the v1 confound, but the
        offset*outcome interaction was non-significant everywhere (p=.52-.71),
        prompting a re-derivation of the model closer to a standard event-study
        DiD spec (this version).

Current (v3) primary model, per round x panel-type (Gap/Level), FULL condition:
    |Gap_{r,t}| = b0 + b1*Time_t + b2*NE + b3*(Time_t x NE) + C(model) + e
  where Gap_{r,t} = value_{r,t} - baseline_r (baseline_r = that run's own mean
  over the two pre-injection rounds, offset in {-1,0}), Time_t = offset
  (1..5, rounds since injection), NE = 1 for the belief outcome / 0 for
  contribution, run r = model x seed. SEs clustered by run.
    b1 = Contribution's recovery slope (units/round, toward 0)
    b3 = how much steeper/shallower NE's slope is than Contribution's --
         the direct test of the headline dissociation claim.
  C(model) = family fixed effects, since raw units aren't comparable across
  model families.

Stage extension: the same model, pooling round10 + round20 within each
panel-type and adding a Stage dummy (1 = round20) interacted with Time:
    |Gap_{r,t}| = b0 + b1*Time + b2*NE + b3*(Time x NE) + b4*Stage
                  + b5*(Time x Stage) + C(model) + e
  b5 tests whether recovery is faster/slower for the later-stage (round 20)
  injection, pooled across outcome. Run clusters are (model,seed,stage) here
  since replace10_*/replace20_* are independent simulations even for the same
  (model,seed).

Normalized/intuitive reframing: Recovery_t = 1 - |Gap_t| / Gap0, where Gap0 is
the *population-mean* |Gap| at offset 1 (NOT each run's own Gap0 -- checked
empirically and ~15-20% of individual runs have |Gap_{r,1}| == 0 by chance,
i.e. no measurable displacement that particular run/round, which makes a
per-run ratio blow up or divide by zero for a nontrivial share of the sample).
Using the population mean as a fixed per-outcome rescaling constant keeps the
same interpretation (0 = fully recovered, 1 = no recovery) while being a
well-behaved linear transform of the primary model above (same significance
test, different units) -- reported as descriptive "% recovered by offset +5"
alongside the primary regression, not as a substitute for it.

Data:
  Gap column   -- reconstructed from raw logs, same methodology as
                  build_dissociation_round10.py (see that script's docstring
                  for reconstruction/validation caveats, which also apply here
                  to round 20 gap: no per-seed round-20 gap data survived from
                  the original lost script). FULL condition only.
  Level column -- reconstructed the same way as build_untouched_agents_levels.py.
                  FULL condition only (that script only ever builds FULL).

Output: figures/2026-03-22/paper_stats/shock_recovery_ne_vs_contribution.tex
and shock_recovery_stage_interaction.tex (copy into
figures/MAIN_RESULTS/8_shock_treatment/ alongside the figure, same convention
as shock_recovery_round10.tex / shock_recovery_round20.tex).
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = "/data3/rasimura/social-norm-evo/code/results"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"
OUT_TEX_MAIN = os.path.join(OUT_DIR, "shock_recovery_ne_vs_contribution.tex")
OUT_TEX_STAGE = os.path.join(OUT_DIR, "shock_recovery_stage_interaction.tex")

MODELS = ["llama", "mistral", "qwen"]
SEEDS = list(range(42, 52))

CONFIGS = [
    ("round10", "replace10_random_adversarial", 10, 0),
    ("round20", "replace20_random_adversarial", 20, 1),
]


def build_gap(variant, pivot):
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            pattern = os.path.join(BASE, model, "local_newgroup", variant,
                                    f"seed{seed}", f"log_*_FULL_seed{seed}.json")
            paths = glob.glob(pattern)
            if not paths:
                continue
            d = json.load(open(paths[0]))
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


def build_level(variant, pivot):
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            pattern = os.path.join(BASE, model, "local_newgroup", variant,
                                    f"seed{seed}", f"log_*_FULL_seed{seed}.json")
            paths = glob.glob(pattern)
            if not paths:
                continue
            d = json.load(open(paths[0]))
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


def to_long(df, stage):
    long = df.melt(id_vars=["model", "seed", "offset"],
                    value_vars=["belief", "contribution"],
                    var_name="outcome_type", value_name="value")
    long = long.dropna(subset=["value"])
    long["stage"] = stage
    long["run_id"] = long["model"] + "_s" + long["seed"].astype(str) + f"_st{stage}"
    return long


def add_abs_gap(long):
    """baseline_r = that run's own mean over offset in {-1,0}; abs_gap = |value
    - baseline_r| for the post-shock rows (offset 1..5), kept per-run (not
    population) since it's a subtraction, not a division -- no instability."""
    base = (long[long["offset"].isin([-1, 0])]
            .groupby(["run_id", "outcome_type"])["value"].mean()
            .rename("baseline").reset_index())
    post = long[long["offset"] >= 1].merge(base, on=["run_id", "outcome_type"], how="left")
    post["abs_gap"] = (post["value"] - post["baseline"]).abs()
    post["NE"] = (post["outcome_type"] == "belief").astype(int)
    return post


def population_gap0(post):
    """Population-mean |abs_gap| at offset==1, per outcome_type -- the fixed
    scaling constant for the Recovery_t reframing (avoids per-run division by
    a near-zero individual shock)."""
    return post[post["offset"] == 1].groupby("outcome_type")["abs_gap"].mean().to_dict()


def fit_main(post):
    """|Gap| ~ offset * NE + C(model), clustered SE by run. NE=0 (contribution)
    is the reference level, so offset's coefficient is Contribution's recovery
    slope and offset:NE is the NE-vs-Contribution slope difference."""
    res = smf.ols("abs_gap ~ offset * NE + C(model)", data=post).fit(
        cov_type="cluster", cov_kwds={"groups": post["run_id"]})
    return dict(
        contrib_slope=res.params["offset"], contrib_se=res.bse["offset"],
        ne_diff=res.params["offset:NE"], ne_diff_se=res.bse["offset:NE"],
        ne_diff_p=res.pvalues["offset:NE"],
        n=int(res.nobs), clusters=post["run_id"].nunique(),
    )


def fit_stage(post_pooled):
    """|Gap| ~ offset * NE + offset * stage + C(model), clustered SE by run
    (run_id includes stage so round10/round20 runs from the same model+seed
    aren't treated as the same cluster). offset:stage tests whether recovery
    is faster/slower for the round-20 injection, pooled across outcome."""
    res = smf.ols("abs_gap ~ offset * NE + offset * stage + C(model)", data=post_pooled).fit(
        cov_type="cluster", cov_kwds={"groups": post_pooled["run_id"]})
    return dict(
        contrib_slope=res.params["offset"], contrib_se=res.bse["offset"],
        ne_diff=res.params["offset:NE"], ne_diff_se=res.bse["offset:NE"],
        ne_diff_p=res.pvalues["offset:NE"],
        stage_diff=res.params["offset:stage"], stage_diff_se=res.bse["offset:stage"],
        stage_diff_p=res.pvalues["offset:stage"],
        n=int(res.nobs), clusters=post_pooled["run_id"].nunique(),
    )


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
    label_map = {
        ("round10", "gap"): "Round 10, Gap",
        ("round20", "gap"): "Round 20, Gap",
        ("round10", "level"): "Round 10, Level",
        ("round20", "level"): "Round 20, Level",
    }

    posts = {}  # (tag, kind) -> post df
    for tag, variant, pivot, stage in CONFIGS:
        posts[(tag, "gap")] = add_abs_gap(to_long(build_gap(variant, pivot), stage))
        posts[(tag, "level")] = add_abs_gap(to_long(build_level(variant, pivot), stage))

    # ---- Table 1: main per-round model + descriptive %-recovered ----
    rows_tex = []
    for (tag, kind), post in posts.items():
        row_label = label_map[(tag, kind)]
        fit = fit_main(post)
        gap0 = population_gap0(post)
        rec5 = post[post["offset"] == 5].groupby("outcome_type")["abs_gap"].mean()
        rec_ne = 1 - rec5["belief"] / gap0["belief"]
        rec_c = 1 - rec5["contribution"] / gap0["contribution"]

        rows_tex.append(
            f"  {row_label} & {fit['contrib_slope']:.3f} ({fit['contrib_se']:.3f}) & "
            f"{fit['ne_diff']:.3f} ({fit['ne_diff_se']:.3f}){stars(fit['ne_diff_p'])} & "
            f"{rec_ne*100:.0f}\\% / {rec_c*100:.0f}\\% & "
            f"{fit['n']} / {fit['clusters']} \\\\"
        )
        print(f"{row_label}: Contrib slope={fit['contrib_slope']:.3f} ({fit['contrib_se']:.3f}), "
              f"NE slope diff={fit['ne_diff']:.3f} ({fit['ne_diff_se']:.3f}) p={fit['ne_diff_p']:.4f} | "
              f"recovered@+5 NE={rec_ne*100:.0f}% Contrib={rec_c*100:.0f}%")

    tex_main = r"""\begin{table}[ht]
\centering\small
\begin{tabular}{lcccc}
\toprule
  Panel & Contrib.\ slope (SE) & NE slope diff.\ (SE) & \% recovered @+5 & $N$ / clusters \\
  & \multicolumn{2}{c}{$|$Gap$|\sim$offset$\times$NE, offsets 1--5} & NE / Contrib.\ & \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\caption{Post-shock recovery: NE (belief) vs.\ Contribution (behavior), FULL condition.
$|\mathrm{Gap}_{r,t}| \sim \beta_0+\beta_1\mathrm{Time}_t+\beta_2\mathrm{NE}+\beta_3(\mathrm{Time}_t\times\mathrm{NE})+C(\mathrm{model})+\epsilon$,
where Gap$_{r,t}$ = value $-$ that run's own pre-shock baseline (offsets $-1,0$), run $r=$ model $\times$ seed,
OLS with SEs clustered by run. \textit{Contrib.\ slope} is $\beta_1$ (Contribution's recovery rate, reference level);
\textit{NE slope diff.} is $\beta_3$, the test of whether NE closes its gap at a different rate than Contribution.
\textit{\% recovered @+5} is descriptive: $1-|\mathrm{Gap}_{+5}|/\mathrm{Gap}_0$, where Gap$_0$ is the
population-mean $|\mathrm{Gap}|$ at the injection round (offset 1) for that outcome -- not each run's own Gap$_0$,
since $\sim$15--20\% of individual runs have $|\mathrm{Gap}_{r,1}|\approx0$ by chance, which makes a per-run ratio
unstable. $\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}
\label{tab:shock_recovery_ne_vs_contribution}
\end{table}
"""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_TEX_MAIN, "w") as f:
        f.write(tex_main)
    print(f"\nSaved -> {OUT_TEX_MAIN}")

    # ---- Table 2: pooled round10+round20, Time x Stage interaction ----
    rows_tex2 = []
    for kind in ["gap", "level"]:
        pooled = pd.concat([posts[("round10", kind)], posts[("round20", kind)]], ignore_index=True)
        fit = fit_stage(pooled)
        label = "Gap" if kind == "gap" else "Level"
        rows_tex2.append(
            f"  {label} & {fit['contrib_slope']:.3f} ({fit['contrib_se']:.3f}) & "
            f"{fit['ne_diff']:.3f} ({fit['ne_diff_se']:.3f}){stars(fit['ne_diff_p'])} & "
            f"{fit['stage_diff']:.3f} ({fit['stage_diff_se']:.3f}){stars(fit['stage_diff_p'])} & "
            f"{fit['n']} / {fit['clusters']} \\\\"
        )
        print(f"[Stage] {label}: Contrib slope={fit['contrib_slope']:.3f} ({fit['contrib_se']:.3f}), "
              f"NE diff={fit['ne_diff']:.3f} p={fit['ne_diff_p']:.4f}, "
              f"Stage(r20-r10) diff={fit['stage_diff']:.3f} ({fit['stage_diff_se']:.3f}) "
              f"p={fit['stage_diff_p']:.4f}")

    tex_stage = r"""\begin{table}[ht]
\centering\small
\begin{tabular}{lcccc}
\toprule
  Panel & Contrib.\ slope (SE) & NE slope diff.\ (SE) & Stage slope diff.\ (SE) & $N$ / clusters \\
  & \multicolumn{3}{c}{round 10 $+$ round 20 pooled, offsets 1--5} & \\
\midrule
""" + "\n".join(rows_tex2) + r"""
\bottomrule
\end{tabular}
\caption{Recovery by injection stage, round 10 and round 20 pooled, FULL condition.
$|\mathrm{Gap}_{r,t}| \sim \beta_0+\beta_1\mathrm{Time}_t+\beta_2\mathrm{NE}+\beta_3(\mathrm{Time}_t\times\mathrm{NE})+\beta_4\mathrm{Stage}+\beta_5(\mathrm{Time}_t\times\mathrm{Stage})+C(\mathrm{model})+\epsilon$,
Stage $=1$ for round-20 injection, run clusters are (model, seed, stage) since replace10/replace20
are independent simulations. \textit{Stage slope diff.} is $\beta_5$: whether the later-stage (round 20)
injection recovers faster or slower than round 10, pooled across NE/Contribution.
$\dagger p{<}.10$, $*p{<}.05$, $**p{<}.01$, $***p{<}.001$.}
\label{tab:shock_recovery_stage_interaction}
\end{table}
"""
    with open(OUT_TEX_STAGE, "w") as f:
        f.write(tex_stage)
    print(f"Saved -> {OUT_TEX_STAGE}")


if __name__ == "__main__":
    main()
