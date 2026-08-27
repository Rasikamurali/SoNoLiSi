"""
estimate_full_corpus_cost.py
------------------------------------------------------------
Estimates the cost of running gpt_5_annotation_sl.py's per-group,
8-example-discussion, stratified-shot-pool condition (the one just
validated on the 80-discussion hand-coded sample) across the FULL
corpus of discussions for GPT + the three -7B families (Llama-7B,
Mistral-7B, Qwen-7B), FULL and NO_SELECTION conditions only.

Builds the actual system prompt (codebook + 8 stratified shots) and the
actual per-discussion user prompt for every discussion in scope, and
tokenizes them for real with tiktoken -- not a word-count approximation.
Output tokens are estimated by scaling the empirical completion-tokens/
turn rate observed in the 288-turn validation run (14,897 / 288 =
51.7 tokens/turn) rather than guessed from scratch.
"""

import json
import os
import sys

import pandas as pd
import tiktoken

# Reorg (2026-08-11): this file moved one directory deeper
# (code/analysis/gpt5_annotation_pilot/ -> code/analysis/discussion/gpt5_annotation_pilot/).
# discussion_talk_vs_behavior.py stays at code/analysis/ (shared across
# themes) -- anchor to it and add every subfolder to sys.path.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from discussion_talk_vs_behavior import load_messages  # noqa: E402

from gpt_5_annotation_sl import (  # noqa: E402
    CODEBOOK_PATH, DATA_PATH, SHOT_POOL_DISCUSSION_IDS_STRATIFIED,
    SHOT_CONDITIONS, build_codebook_prompt, build_shots_block,
    build_group_user_prompt, discussion_transcript,
)

FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
CONDITIONS = ["FULL", "NO_SELECTION"]

IN_PRICE = 0.75 / 1e6
OUT_PRICE = 4.50 / 1e6
EMPIRICAL_OUT_TOKENS_PER_TURN = 14_897 / 288  # from the 72-discussion / 288-turn validation run

enc = tiktoken.get_encoding("o200k_base")


def ntok(s: str) -> int:
    return len(enc.encode(s))


def main():
    msgs = load_messages()
    scoped = msgs[msgs.family.isin(FAMILIES) & msgs.condition.isin(CONDITIONS)].copy()
    scoped["discussion_id"] = scoped.apply(
        lambda r: f"{r.run_id}_R{int(r['round']):02d}_G{int(r.group_id)}", axis=1
    )

    n_discussions = scoped.discussion_id.nunique()
    n_turns = len(scoped)
    print(f"Scope: family in {FAMILIES}, condition in {CONDITIONS}")
    print(f"  {n_discussions:,} discussions, {n_turns:,} turns")
    print(scoped.groupby(["family", "condition"]).discussion_id.nunique().unstack())

    codebook = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            os.path.basename(CODEBOOK_PATH))))
    shot_source_df = pd.read_csv(os.path.join(
        _ANALYSIS_DIR,
        "exports", "human_annotation_stratified_sample", "stratified_sample_hand_coded.csv"))
    pool_df = shot_source_df[shot_source_df.discussion_id.isin(SHOT_POOL_DISCUSSION_IDS_STRATIFIED)]

    condition_spec = next(c for c in SHOT_CONDITIONS if c["name"] == "8_example_discussions")
    system_prompt = build_codebook_prompt(codebook, per_group=True)
    shots = build_shots_block(condition_spec, codebook, pool_df, SHOT_POOL_DISCUSSION_IDS_STRATIFIED)
    system_prompt += "\n\n" + shots
    sys_tok = ntok(system_prompt)
    print(f"\nSystem prompt + 8 shots: {sys_tok:,} tokens (sent on every call)")

    user_tok_total = 0
    for did, d in scoped.groupby("discussion_id"):
        d = d.rename(columns={"agent_id": "agent", "message": "utterance"}).sort_values("turn"
                     if "turn" in d.columns else "position_in_group")
        if "turn" not in d.columns:
            d = d.rename(columns={"position_in_group": "turn"})
        user_prompt = build_group_user_prompt(d)
        user_tok_total += ntok(user_prompt)

    input_tokens = n_discussions * sys_tok + user_tok_total
    output_tokens = round(n_turns * EMPIRICAL_OUT_TOKENS_PER_TURN)
    cost = input_tokens * IN_PRICE + output_tokens * OUT_PRICE

    print(f"\nUser-prompt tokens (transcripts, summed over all {n_discussions:,} discussions): {user_tok_total:,}")
    print(f"Total input tokens (system+shots x {n_discussions:,} calls + transcripts): {input_tokens:,}")
    print(f"Estimated output tokens ({EMPIRICAL_OUT_TOKENS_PER_TURN:.1f} tok/turn x {n_turns:,} turns): {output_tokens:,}")
    print(f"\nEstimated cost, no caching: ${cost:.2f}")

    cacheable = sys_tok >= 1024
    if cacheable:
        cached_sys_tokens = sys_tok + (n_discussions - 1) * sys_tok * 0.1
        cached_input_tokens = cached_sys_tokens + user_tok_total
        cached_cost = cached_input_tokens * IN_PRICE + output_tokens * OUT_PRICE
        print(f"Estimated cost with OpenAI automatic prompt caching (~90% off repeated system+shots prefix): ${cached_cost:.2f}")


if __name__ == "__main__":
    main()
