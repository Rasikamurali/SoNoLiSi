"""
build_agent_round_panel.py
---------------------------
Long-format panel: one row per (run, round, agent) with which group the
agent was in that round, its contribution/payoff/cooperation_tendency, its
average incoming network weight (the quantity the selection mechanism
thresholds on), and whether it was excluded that round.

Built to check whether exclusion tracks poor contribution or reflects other
dynamics (e.g. group composition, network position) — see
`get_excluded_agents()` in SoNoLiSi_v5_os_local.py: an agent is excluded when
its average incoming edge weight drops below `participation_threshold` (0.3).
Missing edges count as weight 0, matching the simulation's own computation.

Output: exports/round_level_agent_panel.csv
"""

import json
import glob
import os
import sys

import numpy as np
import pandas as pd

# Reorg (2026-08-11): this file now lives in code/analysis/selection/, but
# model_specs.py stays at code/analysis/ (shared across many themes) --
# walk up to find it and add every code/analysis/ subfolder to sys.path so
# bare local imports keep working regardless of which theme folder a module
# ended up in.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_specs import MODEL_SPECS, CONDITIONS

PARTICIPATION_THRESHOLD = 0.3  # from SimConfig in SoNoLiSi_v5_os_local.py


def build_panel():
    rows = []

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
                all_ids = sorted({int(a["id"]) for a in d["final_agents"]})

                for r in d["round_logs"]:
                    rnd = r["round"]

                    group_of = {}
                    for gi, grp in enumerate(r.get("groups") or []):
                        members = [int(x) for x in grp]
                        for aid in members:
                            group_of[aid] = (gi, [m for m in members if m != aid])

                    contributions = {int(k): v for k, v in (r.get("contributions") or {}).items()}
                    payoffs = {int(k): v for k, v in (r.get("payoffs") or {}).items()}
                    excluded_this_round = {int(x) for x in (r.get("excluded_agents") or [])}

                    coop_tendency = {int(s["id"]): s.get("cooperation_tendency")
                                      for s in (r.get("agent_states") or [])}

                    # avg incoming network weight per agent (mirrors get_excluded_agents)
                    incoming = {aid: [] for aid in all_ids}
                    for edge in (r.get("network_weights") or []):
                        v = int(edge["v"])
                        if v in incoming:
                            incoming[v].append(edge["weight"])
                    others_n = max(len(all_ids) - 1, 0)
                    avg_incoming = {
                        aid: (sum(incoming[aid]) / others_n if others_n else np.nan)
                        for aid in all_ids
                    }

                    for aid in all_ids:
                        gi, mates = group_of.get(aid, (np.nan, []))
                        in_group = aid in group_of
                        rows.append({
                            "family": family, "model": model_key, "condition": cond,
                            "seed": seed, "run_id": run_id, "round": rnd,
                            "agent_id": aid,
                            "group_id": gi,
                            "group_members": ";".join(str(m) for m in mates),
                            "group_size": (len(mates) + 1) if in_group else np.nan,
                            "contribution": contributions.get(aid, np.nan),
                            "payoff": payoffs.get(aid, np.nan),
                            "cooperation_tendency": coop_tendency.get(aid, np.nan),
                            "avg_incoming_weight": round(avg_incoming[aid], 4),
                            "excluded_this_round": aid in excluded_this_round,
                            "in_groups_this_round": in_group,
                        })

    panel = pd.DataFrame(rows)

    first_excl = (panel[panel["excluded_this_round"]]
                  .groupby(["run_id", "agent_id"])["round"].min()
                  .rename("first_excluded_round"))
    panel = panel.merge(first_excl, on=["run_id", "agent_id"], how="left")
    panel["ever_excluded"] = panel["first_excluded_round"].notna()

    return panel


def main():
    panel = build_panel()
    panel.to_csv("exports/round_level_agent_panel.csv", index=False)
    print(f"Wrote exports/round_level_agent_panel.csv  shape={panel.shape}")


if __name__ == "__main__":
    main()
