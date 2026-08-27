"""
exclusion_diagnostics.py
-------------------------
How many agents get excluded from their group (via the selection mechanism)
over the course of a run, and in which round does each exclusion first occur?

Reads `excluded_agents` (a list of agent ids) off each round log. Exclusion
appears to be sticky: once an agent shows up in `excluded_agents`, it keeps
showing up in every later round of that run. So the metric that matters is
each agent's *first* excluded round, not a per-round count.
"""

import json
import glob
import os
import sys
from collections import Counter

import pandas as pd

# Reorg (2026-08-11): see build_agent_round_panel.py for why this block exists.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_specs import MODEL_SPECS, CONDITIONS, BASE


def load_first_exclusions():
    """
    Returns a DataFrame with one row per agent that was ever excluded:
    family, model, condition, seed, run_id, agent_id, first_excluded_round,
    n_rounds (total rounds in that run), n_agents (total agents in that run).

    Also returns a DataFrame with one row per run: family, model, condition,
    seed, run_id, n_rounds, n_agents, n_excluded.
    """
    exclusion_rows = []
    run_rows = []

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
                n_agents = len(d.get("final_agents") or [])
                n_rounds = len(d["round_logs"])

                first_excluded_round = {}
                for r in d["round_logs"]:
                    for aid in (r.get("excluded_agents") or []):
                        if aid not in first_excluded_round:
                            first_excluded_round[aid] = r["round"]

                for aid, rnd in first_excluded_round.items():
                    exclusion_rows.append({
                        "family": family, "model": model_key,
                        "condition": cond, "seed": seed, "run_id": run_id,
                        "agent_id": aid, "first_excluded_round": rnd,
                        "n_rounds": n_rounds, "n_agents": n_agents,
                    })

                run_rows.append({
                    "family": family, "model": model_key,
                    "condition": cond, "seed": seed, "run_id": run_id,
                    "n_rounds": n_rounds, "n_agents": n_agents,
                    "n_excluded": len(first_excluded_round),
                })

    return pd.DataFrame(exclusion_rows), pd.DataFrame(run_rows)


def main():
    excl, runs = load_first_exclusions()

    sep = "=" * 60
    print(sep)
    print("EXCLUSION DIAGNOSTICS")
    print(sep)

    total_agents = runs["n_agents"].sum()
    total_excluded = len(excl)
    n_runs = len(runs)
    n_runs_with_excl = (runs["n_excluded"] > 0).sum()

    print(f"\nRuns loaded: {n_runs}")
    print(f"Runs with >=1 exclusion: {n_runs_with_excl} "
          f"({n_runs_with_excl / n_runs:.1%})")
    print(f"Total agent-slots across all runs: {total_agents:,}")
    print(f"Total agents ever excluded: {total_excluded:,} "
          f"({total_excluded / total_agents:.2%} of all agent-slots)")

    # ── By condition ─────────────────────────────────────────────────────────
    print(f"\n{'-'*60}\nBy condition\n{'-'*60}")
    by_cond = runs.groupby("condition").agg(
        n_runs=("run_id", "nunique"),
        n_agents=("n_agents", "sum"),
        n_excluded=("n_excluded", "sum"),
    )
    by_cond["pct_excluded"] = by_cond["n_excluded"] / by_cond["n_agents"]
    print(by_cond.to_string(float_format=lambda x: f"{x:.3f}"))

    # ── By model family ──────────────────────────────────────────────────────
    print(f"\n{'-'*60}\nBy model family\n{'-'*60}")
    by_fam = runs.groupby("family").agg(
        n_runs=("run_id", "nunique"),
        n_agents=("n_agents", "sum"),
        n_excluded=("n_excluded", "sum"),
    )
    by_fam["pct_excluded"] = by_fam["n_excluded"] / by_fam["n_agents"]
    print(by_fam.to_string(float_format=lambda x: f"{x:.3f}"))

    # ── When: round of first exclusion ──────────────────────────────────────
    print(f"\n{'-'*60}\nWhen: round of first exclusion (raw round #)\n{'-'*60}")
    if len(excl):
        by_round = excl["first_excluded_round"].value_counts().sort_index()
        for rnd, n in by_round.items():
            print(f"  round {rnd:>3}: {n:>4} agents excluded for the first time")

        print(f"\n{'-'*60}\nWhen: normalized position in the run "
              f"(first_excluded_round / n_rounds)\n{'-'*60}")
        excl["frac_through_run"] = (excl["first_excluded_round"]
                                     / excl["n_rounds"])
        bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
        binned = pd.cut(excl["frac_through_run"], bins, include_lowest=True)
        print(binned.value_counts().sort_index().to_string())
    else:
        print("  No exclusions found.")

    print(sep)
    return excl, runs


if __name__ == "__main__":
    main()
