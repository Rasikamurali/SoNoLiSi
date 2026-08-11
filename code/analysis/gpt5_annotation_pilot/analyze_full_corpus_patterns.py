"""
analyze_full_corpus_patterns.py
------------------------------------------------------------
Descriptive patterns in the LLM-coded full-corpus discussions
(exports/gpt5_annotation_pilot/full_corpus_run/8_example_discussions_annotated.csv):
counterproposal rate by round/family, directionality mix over time
(does "maintain" come to dominate), and agreement/follow behavior by
speaker position and round.

The file's own "round" column is corrupted (human_annotation_stratified_
sample.py's r.round attribute-access bug, fixed separately) -- round
number is recovered here from discussion_id instead (format is
"..._R<round>_G<group>", reliable and unaffected by that bug).
"""

import os
import re

import pandas as pd

IN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exports", "gpt5_annotation_pilot", "full_corpus_run", "8_example_discussions_annotated.csv",
)

ROUND_RE = re.compile(r"_R(\d+)_G\d+$")


def recover_round(discussion_id: str) -> int:
    m = ROUND_RE.search(discussion_id)
    if not m:
        raise ValueError(f"couldn't parse round from discussion_id: {discussion_id}")
    return int(m.group(1))


def period_of(round_num: int) -> str:
    if round_num <= 7:
        return "Early"
    if round_num <= 14:
        return "Middle"
    return "Late"


def pct(series: pd.Series, value) -> float:
    return (series == value).mean()


def main():
    df = pd.read_csv(IN_PATH)
    df["round_n"] = df.discussion_id.apply(recover_round)
    df["period_n"] = df.round_n.apply(period_of)
    df["is_opener"] = df.turn == 0
    n = len(df)
    print(f"n={n:,} turns, {df.discussion_id.nunique():,} discussions\n")

    # ------------------------------------------------------------
    print("=" * 70)
    print("1. Counterproposal rate by round-period (all turns)")
    print("=" * 70)
    t = df.groupby("period_n").LLM_Speech_Act.apply(lambda s: pct(s, "Counterproposal"))
    print(t.reindex(["Early", "Middle", "Late"]).to_string())

    print("\nCounterproposal rate by round-period, followers only (turn>0 -- openers can't")
    print("counter-propose against nothing said yet in their own group's round)")
    t = df[~df.is_opener].groupby("period_n").LLM_Speech_Act.apply(lambda s: pct(s, "Counterproposal"))
    print(t.reindex(["Early", "Middle", "Late"]).to_string())

    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. Counterproposal rate by model family (followers only)")
    print("=" * 70)
    t = df[~df.is_opener].groupby("family").LLM_Speech_Act.apply(lambda s: pct(s, "Counterproposal"))
    print(t.sort_values(ascending=False).to_string())

    print("\nFamily x period (followers only)")
    t = (df[~df.is_opener].groupby(["family", "period_n"]).LLM_Speech_Act
         .apply(lambda s: pct(s, "Counterproposal")).unstack()[["Early", "Middle", "Late"]])
    print(t.to_string())

    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. Directionality mix over time (share of turns with a directionality label)")
    print("=" * 70)
    labeled = df[df.LLM_Directionality.notna()]
    t = (labeled.groupby("period_n").LLM_Directionality
         .value_counts(normalize=True).unstack().reindex(["Early", "Middle", "Late"]))
    print(t.to_string())

    print("\nSame, by round number (Maintain share specifically)")
    t = labeled.groupby("round_n").LLM_Directionality.apply(lambda s: pct(s, "Maintain"))
    print(t.to_string())

    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Agreement rate by period and speaker position (do followers increasingly agree?)")
    print("=" * 70)
    t = (df.assign(position=df.is_opener.map({True: "opener", False: "follower"}))
         .groupby(["period_n", "position"]).LLM_Agreement
         .apply(lambda s: pct(s, "Yes")).unstack().reindex(["Early", "Middle", "Late"]))
    print(t.to_string())

    print("\nAgreement=Yes rate by round number, followers only")
    t = df[~df.is_opener].groupby("round_n").LLM_Agreement.apply(lambda s: pct(s, "Yes"))
    print(t.to_string())

    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. Speech_Act by position (openers propose, followers agree/counter)")
    print("=" * 70)
    t = (df.assign(position=df.is_opener.map({True: "opener", False: "follower"}))
         .groupby("position").LLM_Speech_Act.value_counts(normalize=True, dropna=False).unstack())
    print(t.to_string())

    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("6. Counterproposal rate by condition (FULL vs NO_SELECTION), by period")
    print("=" * 70)
    t = (df[~df.is_opener].groupby(["condition", "period_n"]).LLM_Speech_Act
         .apply(lambda s: pct(s, "Counterproposal")).unstack()[["Early", "Middle", "Late"]])
    print(t.to_string())


if __name__ == "__main__":
    main()
