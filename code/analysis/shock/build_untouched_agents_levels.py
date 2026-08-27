"""
build_untouched_agents_levels.py
----------------------------------
Companion to build_dissociation_round10.py. Instead of the non-replaced-minus-
replaced GAP, this tracks the 8 non-replaced ("untouched") agents' own raw
contribution and injunctive-norm levels around the injection of 4 adversarial
agents (FULL condition only -- discussion + selection both on), for both
injection rounds (20 and 10).

Question: after 4 agents carrying a conflicting (low-contribution) expectation
enter the population, does the untouched majority hold its own contribution/IN
level, or get pulled down toward the adversarial agents?

Same data/methodology as build_dissociation_round10.py (random-selection
variant, agent IDs stable across rounds, `agents_replaced` marks which 4 of
the 12 agents get hijacked into adversarial behavior from the following round
on) -- see that script's docstring for the validation/caveats that carry over
here. This script does NOT subtract a group mean; it reports the untouched
agents' raw mean contribution / injunctive_norm per round.

Outputs exports/dissociations/untouched_levels_full_{round20,round10}.csv
with columns: offset, outcome_type (contribution/belief), mean, sem, n.
"""

import json
import glob
import os
import numpy as np
import pandas as pd

BASE = "/data3/rasimura/social-norm-evo/code/results"
OUT_DIR = "/data3/rasimura/social-norm-evo/code/analysis/exports/dissociations"
MODELS = ["llama", "mistral", "qwen"]
SEEDS = list(range(42, 52))
COND = "FULL"

CONFIGS = [
    ("round20", "replace20_random_adversarial", 20),
    ("round10", "replace10_random_adversarial", 10),
]


def build(variant, pivot):
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            pattern = os.path.join(BASE, model, "local_newgroup", variant,
                                    f"seed{seed}", f"log_*_{COND}_seed{seed}.json")
            paths = glob.glob(pattern)
            if not paths:
                continue
            d = json.load(open(paths[0]))
            rl = d["round_logs"]
            replaced_ids = set(str(x) for x in
                                next(r["agents_replaced"] for r in rl if "agents_replaced" in r))
            for r in rl:
                offset = r["round"] - pivot
                if offset < -5 or offset > 5:
                    continue
                contrib = r["contributions"]
                perc = r["perceptions"]
                nonrep = [i for i in contrib.keys() if i not in replaced_ids]
                c_vals = [contrib[i] for i in nonrep if i in contrib]
                b_vals = [perc[i]["injunctive_norm"] for i in nonrep
                          if i in perc and "injunctive_norm" in perc[i]]
                rows.append(dict(model=model, seed=seed, offset=offset,
                                  contribution=np.mean(c_vals) if c_vals else None,
                                  belief=np.mean(b_vals) if b_vals else None))
    return pd.DataFrame(rows)


def main():
    for tag, variant, pivot in CONFIGS:
        df = build(variant, pivot)
        long = df.melt(id_vars=["model", "seed", "offset"],
                        value_vars=["belief", "contribution"],
                        var_name="outcome_type", value_name="value")
        agg = long.groupby(["outcome_type", "offset"])["value"].agg(
            mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count").reset_index()
        out_path = os.path.join(OUT_DIR, f"untouched_levels_full_{tag}.csv")
        agg.to_csv(out_path, index=False)
        print(f"Saved -> {out_path}")
        print(agg)


if __name__ == "__main__":
    main()
