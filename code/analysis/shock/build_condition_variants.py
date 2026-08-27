"""
build_condition_variants.py
------------------------------
Same methodology as build_dissociation_round10.py + build_untouched_agents_levels.py
(random-selection replacement variant, agent IDs stable across rounds,
`agents_replaced` marks the 4 of 12 hijacked into adversarial behavior the
round after injection), but run per-condition instead of FULL only:
BASELINE, NO_SELECTION, NO_DISCUSSION ("no social learning" -- the discussion
mechanism is the vehicle for social learning in this sim).

For NO_DISCUSSION, the round-20 gap already has a saved, trusted ground-truth
source (exports/dissociations/dissociation_pooled_agg.csv, discussion_on=0)
and the round-10 gap has the already-validated reconstruction
(dissociation_round10_pooled_agg.csv, discussion_on=0) -- both reused as-is
rather than rebuilt. BASELINE and NO_SELECTION have no saved ground truth for
either round, so both are reconstructed from raw logs here. The "non-adversarial
agents' own level" metric has no saved ground truth for any condition, so it's
always reconstructed.

Outputs (exports/dissociations/):
  dissociation_gap_{cond}_round{20,10}.csv
  nonadversarial_levels_{cond}_round{20,10}.csv
for cond in {baseline, no_selection, no_discussion}.
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

CONDITIONS = {
    "baseline": "BASELINE",
    "no_selection": "NO_SELECTION",
    "no_discussion": "NO_DISCUSSION",
}
PIVOTS = [("round20", "replace20_random_adversarial", 20),
          ("round10", "replace10_random_adversarial", 10)]

TRUSTED_GAP = {
    ("no_discussion", "round20"): ("exports/dissociations/dissociation_pooled_agg.csv", 0),
    ("no_discussion", "round10"): ("exports/dissociations/dissociation_round10_pooled_agg.csv", 0),
}


def load_runs(variant, cond_name):
    runs = []
    for model in MODELS:
        for seed in SEEDS:
            pattern = os.path.join(BASE, model, "local_newgroup", variant,
                                    f"seed{seed}", f"log_*_{cond_name}_seed{seed}.json")
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
    for cond_tag, cond_name in CONDITIONS.items():
        for pivot_tag, variant, pivot in PIVOTS:
            runs = load_runs(variant, cond_name)

            gap_out = os.path.join(OUT_DIR, f"dissociation_gap_{cond_tag}_{pivot_tag}.csv")
            trusted = TRUSTED_GAP.get((cond_tag, pivot_tag))
            if trusted:
                path, disc_val = trusted
                full_path = path if os.path.isabs(path) else \
                    os.path.join("/data3/rasimura/social-norm-evo/code/analysis", path)
                d = pd.read_csv(full_path)
                gap_agg = d[d["discussion_on"] == disc_val][["outcome_type", "offset", "mean", "sem", "n"]]
            else:
                gap_agg = build_gap(runs, pivot)
            gap_agg.to_csv(gap_out, index=False)
            print(f"Saved -> {gap_out}")

            level_agg = build_levels(runs, pivot)
            level_out = os.path.join(OUT_DIR, f"nonadversarial_levels_{cond_tag}_{pivot_tag}.csv")
            level_agg.to_csv(level_out, index=False)
            print(f"Saved -> {level_out}")


if __name__ == "__main__":
    main()
