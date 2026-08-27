"""
build_pattern_tables.py
------------------------------------------------------------
Turns the descriptive patterns from analyze_full_corpus_patterns.py into
two compact tables for write-up:

  Table 1 -- the time-trend story (counterproposal / maintain / agreement
             rate by round period), the core "local stabilization" evidence.
  Table 2 -- the family-difference story (counterproposal rate by model
             family x period), kept separate from Table 1 so the time
             trend isn't buried under a family x period x metric grid.

Saves both as CSV (exports/gpt5_annotation_pilot/full_corpus_run/) and
prints them formatted for direct use in chat/write-up.
"""

import os
import re

import pandas as pd

# Reorg (2026-08-11): this file moved one directory deeper
# (code/analysis/gpt5_annotation_pilot/ -> code/analysis/discussion/gpt5_annotation_pilot/),
# so anchoring to code/analysis/ needs a 3rd dirname() hop now, not 2.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)

RUN_DIR = os.path.join(_ANALYSIS_DIR, "exports", "gpt5_annotation_pilot", "full_corpus_run")
IN_PATH = os.path.join(RUN_DIR, "8_example_discussions_annotated.csv")

ROUND_RE = re.compile(r"_R(\d+)_G\d+$")


def recover_round(discussion_id: str) -> int:
    return int(ROUND_RE.search(discussion_id).group(1))


def period_of(round_num: int) -> str:
    if round_num <= 7:
        return "Early (R1-7)"
    if round_num <= 14:
        return "Middle (R8-14)"
    return "Late (R15-20)"


PERIOD_ORDER = ["Early (R1-7)", "Middle (R8-14)", "Late (R15-20)"]


def pct(series: pd.Series, value) -> float:
    return (series == value).mean() * 100


def main():
    df = pd.read_csv(IN_PATH)
    df["round_n"] = df.discussion_id.apply(recover_round)
    df["period_n"] = df.round_n.apply(period_of)
    followers = df[df.turn != 0]
    labeled_dir = df[df.LLM_Directionality.notna()]

    # ---------------- Table 1: time-trend ----------------
    t1 = pd.DataFrame({
        "Counterproposal rate (followers)": followers.groupby("period_n").LLM_Speech_Act
            .apply(lambda s: pct(s, "Counterproposal")),
        "Directionality = Maintain (of labeled turns)": labeled_dir.groupby("period_n").LLM_Directionality
            .apply(lambda s: pct(s, "Maintain")),
        "Agreement = Yes (followers)": followers.groupby("period_n").LLM_Agreement
            .apply(lambda s: pct(s, "Yes")),
    }).reindex(PERIOD_ORDER).T
    t1.index.name = "Metric (% of turns)"

    # ---------------- Table 2: family x period ----------------
    t2 = (followers.groupby(["family", "period_n"]).LLM_Speech_Act
          .apply(lambda s: pct(s, "Counterproposal")).unstack()[PERIOD_ORDER])
    t2["Overall"] = followers.groupby("family").LLM_Speech_Act.apply(lambda s: pct(s, "Counterproposal"))
    t2 = t2.sort_values("Overall", ascending=False)
    t2.index.name = "Family"
    t2.columns.name = "Counterproposal rate (%, followers)"

    os.makedirs(RUN_DIR, exist_ok=True)
    t1.round(1).to_csv(os.path.join(RUN_DIR, "table1_time_trend.csv"))
    t2.round(1).to_csv(os.path.join(RUN_DIR, "table2_family_counterproposal.csv"))

    pd.set_option("display.width", 120)
    print("TABLE 1 -- Discussion-mechanism indicators by round period (% of turns)")
    print(t1.round(1).to_string())
    print(f"\nn = {len(followers):,} follower turns; {len(labeled_dir):,} turns with a directionality label")
    print("\n\nTABLE 2 -- Counterproposal rate by model family x period (followers only, %)")
    print(t2.round(1).to_string())

    print(f"\nSaved to {RUN_DIR}/table1_time_trend.csv and table2_family_counterproposal.csv")


if __name__ == "__main__":
    main()
