"""
extract_agent_data.py
---------------------
Extracts agent-level data per round across all models, variants, seeds, conditions.

Output: results/agent_level_data.csv

Columns:
  model, variant, condition, seed, round,
  group_id, agent_id,
  payoff, cooperation_tendency,
  injunctive_norm, descriptive_norm,
  preferred_partners, agents_to_avoid
"""

import json, os, glob
import pandas as pd

CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

MODELS   = ["gpt", "llama", "mistral", "qwen"]
VARIANTS = ["global", "local"]


def extract_file(path: str, model: str, variant: str) -> list:
    with open(path) as f:
        data = json.load(f)

    condition = data["condition"]
    seed      = data["seed"]
    rows      = []

    for r in data["round_logs"]:
        round_num = r["round"]

        # Build agent → group_id lookup
        agent_group = {}
        for g_idx, group in enumerate(r.get("groups") or []):
            for aid in group:
                agent_group[int(aid)] = g_idx

        # Payoffs
        payoffs = {int(k): v for k, v in r.get("payoffs", {}).items()}

        # Cooperation tendency
        ct_map = {a["id"]: a["cooperation_tendency"] for a in r.get("agent_states", [])}

        # Perceptions
        perceptions = r.get("perceptions") or {}

        # All agent IDs present this round
        all_agents = set(ct_map.keys())

        for agent_id in sorted(all_agents):
            perc = perceptions.get(str(agent_id)) or perceptions.get(agent_id)

            inj_norm  = None
            desc_norm = None
            preferred = None
            avoid     = None

            if perc:
                raw_inj = perc.get("injunctive_norm")
                if raw_inj is not None:
                    try:
                        inj_norm = float(raw_inj)
                    except (TypeError, ValueError):
                        pass

                raw_desc = perc.get("descriptive_norm")
                if raw_desc is not None:
                    try:
                        desc_norm = float(raw_desc)
                    except (TypeError, ValueError):
                        pass

                preferred = perc.get("preferred_partners")
                avoid     = perc.get("agents_to_avoid")

            rows.append({
                "model":              model,
                "variant":            variant,
                "condition":          condition,
                "seed":               seed,
                "round":              round_num,
                "group_id":           agent_group.get(agent_id),
                "agent_id":           agent_id,
                "payoff":             payoffs.get(agent_id),
                "cooperation_tendency": ct_map.get(agent_id),
                "injunctive_norm":    inj_norm,
                "descriptive_norm":   desc_norm,
                "preferred_partners": str(preferred) if preferred is not None else None,
                "agents_to_avoid":    str(avoid)     if avoid     is not None else None,
            })

    return rows


def main():
    all_rows = []
    total_files = 0

    for model in MODELS:
        for variant in VARIANTS:
            base = os.path.join(RESULTS_DIR, model, variant)
            if not os.path.isdir(base):
                continue

            for seed_dir in sorted(glob.glob(os.path.join(base, "seed*"))):
                # Latest timestamp wins per condition within each seed folder
                cond_best: dict = {}
                for path in sorted(glob.glob(os.path.join(seed_dir, "log_*.json"))):
                    with open(path) as f:
                        cond = json.load(f)["condition"]
                    cond_best[cond] = path

                for cond, path in cond_best.items():
                    rows = extract_file(path, model, variant)
                    all_rows.extend(rows)
                    total_files += 1

            print(f"  {model}/{variant}: done")

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "agent_level_data.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows → {out_path}")
    print(f"Files processed: {total_files}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
