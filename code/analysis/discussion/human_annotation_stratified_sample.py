"""
human_annotation_stratified_sample.py
------------------------------------------------------------
Draws a stratified sample of ~80 group-round discussions for human hand-
coding with the social-learning annotation codebook (see
gpt5_annotation_pilot/social_learning_annotation_codebook.json). Unlike
GPT_s43_FULL_hand_coded_sample.csv (a single GPT run), this spans model
family and condition so the ground-truth set isn't limited to one model.

Sampling unit: one group's discussion transcript in one round of one run
(family x seed x condition x round x group) -- same granularity as
discussion_id in the existing hand-coded sample.

Strata: family (GPT, Llama-7B, Mistral-7B, Qwen-7B) x condition (FULL,
NO_SELECTION) x round period (Early 1-7, Middle 8-14, Late 15-20 -- the
user's stated "Middle 8-15 / Late 15-20" overlapped at round 15; cut
Middle off at 14 to make the bins non-overlapping and cover all 20 rounds).

That's 24 strata. Each of the 8 (family, condition) pairs contributes 3
discussions to two periods and 4 to a third, with which period gets the
"+1" rotated across the 8 pairs. That makes family totals (20 each) and
condition totals (40 each) come out exactly even, and period totals land
within 1 of each other (27/27/26) rather than skewing toward one period.
Total = 80.

Output (exports/human_annotation_stratified_sample/):
    stratified_sample_for_coding.csv - same column template as
        GPT_s43_FULL_hand_coded_sample.csv (blank Human_*/LLM_* columns,
        ready to hand-code), one row per message across the 80 sampled
        discussions.
    sample_manifest.csv - one row per sampled discussion (family,
        condition, seed, round, group, period, n_turns) for tracking.
"""

import os
import sys

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

FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
CONDITIONS = ["FULL", "NO_SELECTION"]
PERIODS = ["Early", "Middle", "Late"]
BASE_PER_STRATUM = 3
RANDOM_SEED = 42  # fixed, not re-drawn each run -- this sample is meant to be handed out for coding once, reproducibly

OUT_DIR = "exports/human_annotation_stratified_sample"


def period_of(round_num: int) -> str:
    if round_num <= 7:
        return "Early"
    if round_num <= 14:
        return "Middle"
    return "Late"


def stratum_targets() -> dict:
    """See module docstring: each (family, condition) pair sends its "+1"
    discussion to a different period, rotating across the 8 pairs."""
    pairs = [(f, c) for f in FAMILIES for c in CONDITIONS]
    extra_period = {pair: PERIODS[i % 3] for i, pair in enumerate(pairs)}
    targets = {}
    for f, c in pairs:
        for p in PERIODS:
            targets[(f, c, p)] = BASE_PER_STRATUM + (1 if extra_period[(f, c)] == p else 0)
    return targets


def build_discussion_index(msgs: pd.DataFrame) -> pd.DataFrame:
    """One row per group-round discussion, with its period and a stable id."""
    msgs = msgs.copy()
    msgs["period"] = msgs["round"].apply(period_of)
    disc = (
        msgs.groupby(["family", "condition", "seed", "run_id", "round", "group_id", "period"])
        .size()
        .reset_index(name="n_turns")
    )
    disc["discussion_id"] = disc.apply(
        lambda r: f"{r.run_id}_R{int(r['round']):02d}_G{int(r.group_id)}", axis=1
    )
    return disc


def draw_sample(disc_index: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    targets = stratum_targets()
    picked = []
    for (family, condition, period), n in targets.items():
        pool = disc_index[
            (disc_index.family == family)
            & (disc_index.condition == condition)
            & (disc_index.period == period)
        ]
        if len(pool) < n:
            raise ValueError(
                f"stratum (family={family}, condition={condition}, period={period}) "
                f"has only {len(pool)} discussions available, need {n}"
            )
        picked.append(pool.sample(n=n, random_state=rng.integers(0, 1_000_000)))
    return pd.concat(picked, ignore_index=True)


def format_coding_rows(msgs: pd.DataFrame, sampled_disc: pd.DataFrame) -> pd.DataFrame:
    msgs = msgs.copy()
    msgs["period"] = msgs["round"].apply(period_of)
    msgs["discussion_id"] = msgs.apply(
        lambda r: f"{r.run_id}_R{int(r['round']):02d}_G{int(r.group_id)}", axis=1
    )
    sub = msgs[msgs.discussion_id.isin(sampled_disc.discussion_id)].copy()

    contrib_lookup, member_lookup = {}, {}
    for disc_id, gdf in sub.groupby("discussion_id"):
        contrib_lookup[disc_id] = ", ".join(
            f"Agent {int(r.agent_id)}={r.actual_contribution:g}"
            for r in gdf.drop_duplicates("agent_id").itertuples()
            if pd.notna(r.actual_contribution)
        )
        member_lookup[disc_id] = ", ".join(str(a) for a in sorted(gdf.agent_id.unique()))

    blank_labels = {
        "Human_Speech_Act": "", "Human_Reference": "", "Human_Agreement": "",
        "Human_Directionality": "", "Human_Appeal_Justification": "", "Human_Notes": "",
        "LLM_Speech_Act": "", "LLM_Reference": "", "LLM_Agreement": "",
        "LLM_Directionality": "", "LLM_Appeal_Justification": "", "LLM_Notes": "",
    }

    rows = []
    for _, r in sub.sort_values(["discussion_id", "position_in_group"]).iterrows():
        rows.append({
            "run_id": r.run_id, "family": r.family, "condition": r.condition, "seed": r.seed,
            "period": r.period, "group": int(r.group_id), "discussion_id": r.discussion_id,
            "group_members": member_lookup[r.discussion_id],
            "agent": r.agent_id, "turn": r.position_in_group,
            "extracted_target": r.extracted_number, "contribution": r.actual_contribution,
            "group_contributions": contrib_lookup[r.discussion_id], "round": r["round"],
            "utterance": r.message,
            **blank_labels,
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Loading discussion messages...")
    msgs = load_messages()
    msgs = msgs[msgs.family.isin(FAMILIES) & msgs.condition.isin(CONDITIONS)]
    print(f"  {len(msgs):,} messages across {FAMILIES} x {CONDITIONS}")

    disc_index = build_discussion_index(msgs)
    sampled_disc = draw_sample(disc_index, rng)

    print(f"\nSampled {len(sampled_disc)} group-discussions ({sampled_disc.n_turns.sum()} total turns)")
    print(sampled_disc.groupby(["family", "condition", "period"]).size().unstack().fillna(0).astype(int))

    coding_df = format_coding_rows(msgs, sampled_disc)

    os.makedirs(OUT_DIR, exist_ok=True)
    coding_path = os.path.join(OUT_DIR, "stratified_sample_for_coding.csv")
    coding_df.to_csv(coding_path, index=False)

    manifest_path = os.path.join(OUT_DIR, "sample_manifest.csv")
    sampled_disc.sort_values(["family", "condition", "period", "seed", "round", "group_id"]).to_csv(
        manifest_path, index=False
    )

    print(f"\nSaved {len(coding_df)} rows ({len(sampled_disc)} discussions) to {coding_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
