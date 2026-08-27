"""
build_dissociation_round10.py
------------------------------
Extends the dissociation_belief_vs_contribution analysis (see
plot_dissociation_belief_vs_contribution.py docstring -- original generating
script for the round-20 data is lost, only dissociation_pooled_agg.csv
survives) to the round-10 injection condition, so the figure can become a
2x2 grid: rows = injection round (20 existing / 10 new), columns =
discussion off/on.

Methodology (reverse-engineered from raw logs under code/results/*/local_newgroup/,
validated by rebuilding the round-20 numbers and comparing to the saved
dissociation_pooled_agg.csv -- same shape/spike-at-offset-1/decay pattern,
mean abs deviation ~0.18 vs values spanning 0-5, i.e. reconstruction is not
a bit-exact match to whatever the original script did, but structurally
matches closely enough to trust for the new round-10 panels):

  - Uses the `random`-selection adversarial variant (replace{10,20}_random_adversarial),
    which fit the saved round-20 numbers better than the `top` variant.
  - FULL condition -> discussion_on=1, NO_DISCUSSION condition -> discussion_on=0.
  - Agent IDs are stable across rounds; `agents_replaced` (recorded once, at the
    injection round) marks which agents get hijacked into adversarial behavior
    from the following round on. "Gap to group mean" = each agent's value minus
    the mean of its round's randomly-reshuffled 4-person group (all members,
    replaced + non-replaced), averaged separately over replaced/non-replaced
    agents, then differenced (non-replaced minus replaced).
  - offset = round - pivot_round, where pivot_round is the last pre-injection
    round (10 for replace10_*, 20 for replace20_*) so offset 0 is last-normal
    round and offset 1 is first-post-injection round.

Outputs exports/dissociations/dissociation_round10_pooled_agg.csv in the same
schema as dissociation_pooled_agg.csv, for use by the 2x2 plotting script.
"""

import json
import glob
import os
import numpy as np
import pandas as pd

BASE = "/data3/rasimura/social-norm-evo/code/results"
OUT = "/data3/rasimura/social-norm-evo/code/analysis/exports/dissociations/dissociation_round10_pooled_agg.csv"
MODELS = ["llama", "mistral", "qwen"]
SEEDS = list(range(42, 52))
COND_MAP = {"FULL": 1, "NO_DISCUSSION": 0}
VARIANT = "replace10_random_adversarial"
PIVOT = 10


def build():
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            for cond, disc_on in COND_MAP.items():
                pattern = os.path.join(BASE, model, "local_newgroup", VARIANT,
                                        f"seed{seed}", f"log_*_{cond}_seed{seed}.json")
                paths = glob.glob(pattern)
                if not paths:
                    continue
                d = json.load(open(paths[0]))
                rl = d["round_logs"]
                replaced_ids = set(str(x) for x in
                                    next(r["agents_replaced"] for r in rl if "agents_replaced" in r))
                for r in rl:
                    offset = r["round"] - PIVOT
                    if offset < -5 or offset > 5:
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
                        model=model, seed=seed, discussion_on=disc_on, offset=offset,
                        belief=gap(lambda i: perc.get(i, {}).get("injunctive_norm")),
                        contribution=gap(lambda i: contrib.get(i)),
                    ))
    return pd.DataFrame(rows)


def main():
    df = build()
    long = df.melt(id_vars=["model", "seed", "discussion_on", "offset"],
                    value_vars=["belief", "contribution"],
                    var_name="outcome_type", value_name="value")
    agg = long.groupby(["discussion_on", "outcome_type", "offset"])["value"].agg(
        mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count").reset_index()
    agg.to_csv(OUT, index=False)
    print(f"Saved -> {OUT}")
    print(agg)


if __name__ == "__main__":
    main()
