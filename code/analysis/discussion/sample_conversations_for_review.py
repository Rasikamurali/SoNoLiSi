"""
sample_conversations_for_review.py
-------------------------------------
Samples ENTIRE discussion transcripts (every round, every group, in order)
for full runs, for manual/qualitative review. The sampling unit is a whole
run (family x seed x condition) - randomized by seed, not by round or
group - so each output file is the complete, original conversation history
for that run rather than a single-round excerpt.

Current settings: 3 random runs per family, restricted to GPT and the 7B
open-weight models (Llama-7B, Mistral-7B, Qwen-7B), pooled across both
discussion-enabled conditions (FULL/NO_SELECTION) - so the condition mix
per family is whatever the random draw gives, not a fixed split. True
random draw each run (no fixed seed); the drawn entropy seed is printed
and saved to the manifest for reproducibility if needed.

Output: exports/discussion_mechanism/manual_conversation_samples/
    <condition>/<family>_seed<seed>_FULLTRANSCRIPT.txt
    _manifest.csv  (one row per sampled run, for tracking what's been reviewed)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# Reorg (2026-08-11): see discussion_mechanism_analysis.py for why this block exists.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from discussion_talk_vs_behavior import load_messages

OUT_DIR = "exports/discussion_mechanism/manual_conversation_samples"
FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
RUNS_PER_FAMILY = 3


def format_full_transcript(run_id, condition, family, seed, run_df):
    lines = []
    lines.append(f"run_id:    {run_id}")
    lines.append(f"condition: {condition}")
    lines.append(f"family:    {family}")
    lines.append(f"seed:      {seed}")
    lines.append(f"rounds:    {sorted(run_df['round'].unique())}")
    lines.append("=" * 78)

    for rnd in sorted(run_df["round"].unique()):
        rdf = run_df[run_df["round"] == rnd]
        lines.append("")
        lines.append(f"########## ROUND {rnd} ##########")
        for gid in sorted(rdf["group_id"].dropna().unique()):
            gdf = rdf[rdf.group_id == gid].sort_values("position_in_group")
            members = sorted(gdf["agent_id"].unique())
            lines.append("")
            lines.append(f"--- Group {int(gid)} (members: {members}) ---")
            for _, row in gdf.iterrows():
                num = f"  [extracted target: {row['extracted_number']}]" if pd.notna(row["extracted_number"]) else ""
                lines.append(f"Agent {row['agent_id']} (turn {row['position_in_group']}):{num}")
                lines.append(f"  {row['message']}")
            contrib_str = ", ".join(f"Agent {int(r.agent_id)}={r.actual_contribution:g}"
                                    for r in gdf.drop_duplicates("agent_id").itertuples()
                                    if pd.notna(r.actual_contribution))
            lines.append(f"  [contributions: {contrib_str}]")
    return "\n".join(lines)


def clear_previous_samples():
    for f in glob.glob(os.path.join(OUT_DIR, "**", "*.txt"), recursive=True):
        os.remove(f)
    manifest_path = os.path.join(OUT_DIR, "_manifest.csv")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)


def main():
    print("Loading discussion messages...")
    msgs = load_messages()
    msgs = msgs[msgs["family"].isin(FAMILIES)]
    print(f"  {len(msgs):,} messages across families: {FAMILIES}")

    clear_previous_samples()

    seed_entropy = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed_entropy)
    print(f"  Random draw seed (for reproducibility if needed): {seed_entropy}")

    manifest_rows = []

    for family in FAMILIES:
        sub = msgs[msgs["family"] == family]
        run_keys = sub[["run_id", "condition", "seed"]].drop_duplicates()
        n = min(RUNS_PER_FAMILY, len(run_keys))
        picked = run_keys.sample(n=n, random_state=rng.integers(0, 1_000_000))
        for _, k in picked.iterrows():
            run_df = msgs[msgs.run_id == k.run_id]
            text = format_full_transcript(k.run_id, k.condition, family, k.seed, run_df)

            cond_dir = os.path.join(OUT_DIR, k.condition)
            os.makedirs(cond_dir, exist_ok=True)
            fname = f"{family}_seed{k.seed}_FULLTRANSCRIPT.txt"
            fpath = os.path.join(cond_dir, fname)
            with open(fpath, "w") as f:
                f.write(text)

            manifest_rows.append({
                "file": os.path.relpath(fpath, OUT_DIR), "family": family, "condition": k.condition,
                "run_id": k.run_id, "seed": k.seed, "n_rounds": run_df["round"].nunique(),
                "n_messages": len(run_df), "reviewed": "", "notes": "",
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest["_draw_seed"] = seed_entropy
    manifest_path = os.path.join(OUT_DIR, "_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"\nSampled {len(manifest)} full run transcripts "
         f"({RUNS_PER_FAMILY} per family x {len(FAMILIES)} families)")
    print(manifest[["family", "condition", "seed", "n_rounds", "n_messages"]].to_string(index=False))
    print(f"\nSaved to: {OUT_DIR}/")
    print(f"Manifest (with blank 'reviewed'/'notes' columns to fill in): {manifest_path}")


if __name__ == "__main__":
    main()
