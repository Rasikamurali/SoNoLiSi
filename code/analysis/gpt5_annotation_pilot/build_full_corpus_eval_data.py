"""
build_full_corpus_eval_data.py
------------------------------------------------------------
Builds the eval-data CSV (same column schema as stratified_sample_hand_coded.csv,
Human_*/LLM_* columns blank) for the FULL corpus of discussions in scope for the
production annotation run: GPT + the three -7B families (Llama-7B, Mistral-7B,
Qwen-7B), FULL and NO_SELECTION conditions -- no sampling, every discussion.

This mirrors human_annotation_stratified_sample.py's row-formatting exactly
(reuses its build_discussion_index/format_coding_rows), just without the
stratified draw_sample() step -- every discussion in scope is included.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # code/analysis/
from discussion_talk_vs_behavior import load_messages  # noqa: E402
from human_annotation_stratified_sample import build_discussion_index, format_coding_rows  # noqa: E402

FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
CONDITIONS = ["FULL", "NO_SELECTION"]

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exports", "gpt5_annotation_pilot", "full_corpus_eval_data.csv",
)


def main():
    msgs = load_messages()
    scoped = msgs[msgs.family.isin(FAMILIES) & msgs.condition.isin(CONDITIONS)].copy()

    disc_index = build_discussion_index(scoped)
    print(f"{len(disc_index):,} discussions, {disc_index.n_turns.sum():,} turns in scope")

    coding_df = format_coding_rows(scoped, disc_index)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    coding_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(coding_df):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
