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

# Reorg (2026-08-11): this file moved one directory deeper
# (code/analysis/gpt5_annotation_pilot/ -> code/analysis/discussion/gpt5_annotation_pilot/).
# discussion_talk_vs_behavior.py stays at code/analysis/ (shared across
# themes); human_annotation_stratified_sample.py now lives at
# code/analysis/discussion/ (a sibling of this file's parent, not this file's
# own directory). Anchor to code/analysis/ and add every subfolder to
# sys.path so both bare imports keep working.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "sobel_mediation.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from discussion_talk_vs_behavior import load_messages  # noqa: E402
from human_annotation_stratified_sample import build_discussion_index, format_coding_rows  # noqa: E402

FAMILIES = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]
CONDITIONS = ["FULL", "NO_SELECTION"]

OUT_PATH = os.path.join(_ANALYSIS_DIR, "exports", "gpt5_annotation_pilot", "full_corpus_eval_data.csv")


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
