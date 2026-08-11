"""
select_shot_discussions.py
------------------------------------------------------------
Picks 8 discussions from exports/human_annotation_stratified_sample/
stratified_sample_hand_coded.csv to serve as few-shot exemplars for
gpt_5_annotation_sl.py, greedily maximizing label-value coverage
(Human_Speech_Act/Reference/Agreement/Directionality) with a tie-break
toward introducing a new model family, so the shot pool isn't all one
family the way the original GPT_s43_FULL pool was.

Prints the chosen discussion_ids (paste into gpt_5_annotation_sl.py's
SHOT_POOL_DISCUSSION_IDS_STRATIFIED, or pass via --shot-discussion-ids).
"""

import pandas as pd

DATA_PATH = "../exports/human_annotation_stratified_sample/stratified_sample_hand_coded.csv"
FIELDS = ["Human_Speech_Act", "Human_Reference", "Human_Agreement", "Human_Directionality"]
N_SHOTS = 8


def label_set(g: pd.DataFrame) -> set:
    vals = set()
    for col in FIELDS:
        vals |= set(g[col].dropna().unique())
    return vals


def main():
    df = pd.read_csv(DATA_PATH)
    diversity = {did: label_set(g) for did, g in df.groupby("discussion_id")}
    family_of = df.groupby("discussion_id")["family"].first()

    all_labels = set().union(*diversity.values())
    print("All label values present across the sample:", sorted(all_labels))

    chosen, covered, families_used = [], set(), set()
    remaining = dict(diversity)
    while len(chosen) < N_SHOTS and remaining:
        def score(did):
            new_labels = len(diversity[did] - covered)
            new_fam = 1 if family_of[did] not in families_used else 0
            return (new_labels, new_fam)
        best = max(remaining, key=score)
        chosen.append(best)
        covered |= diversity[best]
        families_used.add(family_of[best])
        del remaining[best]

    print(f"\nChosen {len(chosen)} shot discussions:")
    for did in chosen:
        print(f"  {did}  family={family_of[did]}  labels={sorted(diversity[did])}")

    print("\nLabel values NOT covered by these:", sorted(all_labels - covered))
    print("Families used:", sorted(families_used))
    print("\nPython list:")
    print(chosen)


if __name__ == "__main__":
    main()
