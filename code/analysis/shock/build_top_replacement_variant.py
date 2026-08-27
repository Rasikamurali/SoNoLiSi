"""
build_top_replacement_variant.py
-----------------------------------
Same as build_dissociation_round10.py + build_untouched_agents_levels.py,
but using the `top`-selection adversarial variant (replace{10,20}_top_adversarial
-- the 4 most-connected agents get hijacked into adversarial behavior, instead
of 4 random agents) rather than `random`. FULL condition only, both injection
rounds (20 and 10).

Same methodology/caveats as the `random` versions -- see those scripts'
docstrings. This is purely a selection-strategy robustness variant, kept
as a separate output/figure rather than replacing the `random` one, since
`random` was the variant that best matched the one piece of ground truth
available (the saved round-20 dissociation_pooled_agg.csv).

Outputs (exports/dissociations/):
  dissociation_gap_full_top_round20.csv, dissociation_gap_full_top_round10.csv
  untouched_levels_full_top_round20.csv, untouched_levels_full_top_round10.csv
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
    ("round20", "replace20_top_adversarial", 20),
    ("round10", "replace10_top_adversarial", 10),
]


def load_runs(variant):
    runs = []
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
            runs.append((model, seed, rl, replaced_ids))
    return runs


def build_gap(runs, pivot):
    rows = []
    for model, seed, rl, replaced_ids in runs:
        for r in rl:
            offset = r["round"] - pivot
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
                model=model, seed=seed, offset=offset,
                belief=gap(lambda i: perc.get(i, {}).get("injunctive_norm")),
                contribution=gap(lambda i: contrib.get(i)),
            ))
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["model", "seed", "offset"], value_vars=["belief", "contribution"],
                    var_name="outcome_type", value_name="value")
    return long.groupby(["outcome_type", "offset"])["value"].agg(
        mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count").reset_index()


def build_levels(runs, pivot):
    rows = []
    for model, seed, rl, replaced_ids in runs:
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
    df = pd.DataFrame(rows)
    long = df.melt(id_vars=["model", "seed", "offset"], value_vars=["belief", "contribution"],
                    var_name="outcome_type", value_name="value")
    return long.groupby(["outcome_type", "offset"])["value"].agg(
        mean="mean", sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count").reset_index()


def main():
    for tag, variant, pivot in CONFIGS:
        runs = load_runs(variant)
        gap_agg = build_gap(runs, pivot)
        gap_path = os.path.join(OUT_DIR, f"dissociation_gap_full_top_{tag}.csv")
        gap_agg.to_csv(gap_path, index=False)
        print(f"Saved -> {gap_path}")

        level_agg = build_levels(runs, pivot)
        level_path = os.path.join(OUT_DIR, f"untouched_levels_full_top_{tag}.csv")
        level_agg.to_csv(level_path, index=False)
        print(f"Saved -> {level_path}")


if __name__ == "__main__":
    main()
