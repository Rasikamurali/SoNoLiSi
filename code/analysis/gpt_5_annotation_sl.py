"""
gpt_5_annotation_sl.py
------------------------------------------------------------
Pilot test: can gpt-5.4-mini follow the social-learning discussion-coding
codebook (gpt5_annotation_pilot/social_learning_annotation_codebook.json),
and if so, how many worked examples does it need?

Ground truth is the hand-coded sample gpt5_annotation_pilot/
GPT_s43_FULL_hand_coded_sample.csv (20 group-discussions x 4 turns each,
from run GPT_s43_FULL). Four of those discussions are held out as a
few-shot exemplar pool (never scored); the remaining 16 are the eval set.

For each shot condition, the target model annotates every eval turn (given
that turn's own discussion up through its own position, as prior context)
and the predictions are scored against the Human_* columns per field.
Outputs land in exports/gpt5_annotation_pilot/: one *_annotated.csv per
condition (predictions filled into LLM_* columns, same shape as the input
template) plus condition_comparison_summary.csv (per-field accuracy/kappa
and row-level exact-match rate by condition, to see whether accuracy rises
with more examples and where it plateaus).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm

load_dotenv()
client = OpenAI()

# ============================================================
# Config
# ============================================================

MODEL_NAME = "gpt-5.4-mini"
# Carried over from SoNoLiSi_v5_local.py: gpt-5-mini variants only accept
# temperature=1 (the API default). Assuming gpt-5.4-mini follows suit;
# drop it from this set if the API rejects/ignores that assumption.
FIXED_TEMPERATURE_MODELS = {"gpt-5-mini", "gpt-5-mini-2025-08-01", "gpt-5.4-mini"}

PILOT_DIR = "gpt5_annotation_pilot"
CODEBOOK_PATH = os.path.join(PILOT_DIR, "social_learning_annotation_codebook.json")
DATA_PATH = os.path.join(PILOT_DIR, "GPT_s43_FULL_hand_coded_sample.csv")
OUT_DIR = os.path.join("exports", "gpt5_annotation_pilot")

# Fields scored against ground truth (short names; data columns are
# Human_<field> / LLM_<field>). "Notes" is requested from the model too,
# purely as a diagnostic rationale -- it is never scored.
# Appeal_Justification dropped from the codebook and from scoring (2026-08-10):
# not part of the analysis going forward.
FIELDS = ["Speech_Act", "Reference", "Agreement", "Directionality"]

# The hand-coded sample leaves a field blank instead of writing "No" when
# a value doesn't hold. The codebook defines an explicit "No" for
# Reference/Agreement, and the sample never once writes "No" out for
# those two fields -- so blank is treated as "No" there for scoring.
# Speech_Act/Directionality have no "does not apply" value in the codebook,
# so blank there is treated as null/not-applicable.
# ASSUMPTION: check this reading against intent before trusting the scores.
BLANK_MEANS_NO = {"Reference", "Agreement"}

# Discussions reserved as few-shot exemplars, never scored -- capped at 12
# of the 20 total so a fixed 8-discussion (32-turn) eval set can be held
# out and reused identically across every condition below, including the
# original 1/2/4 ones (re-run on this smaller eval set for apples-to-apples
# comparability with the new 8/12 conditions). Ordered so smaller shot
# counts still cover diverse labels: R02 alone covers Proposal/Counter-
# proposal/Reference/Agreement/Maintain/Increase; R04 adds Decrease; R09
# adds Mixed/Qualified; R01 adds Appeal (the only Appeal instance in the
# whole hand-coded sample -- Justification never occurs). R06/R08/R11/R12
# and R14/R16/R18/R20 extend coverage into more middle- and late-round
# discussions for the 8- and 12-example conditions.
SHOT_POOL_DISCUSSION_IDS = [
    "GPT_s43_FULL_R02_G0",
    "GPT_s43_FULL_R04_G0",
    "GPT_s43_FULL_R09_G0",
    "GPT_s43_FULL_R01_G0",
    "GPT_s43_FULL_R06_G0",
    "GPT_s43_FULL_R08_G0",
    "GPT_s43_FULL_R11_G0",
    "GPT_s43_FULL_R12_G0",
    "GPT_s43_FULL_R14_G0",
    "GPT_s43_FULL_R16_G0",
    "GPT_s43_FULL_R18_G0",
    "GPT_s43_FULL_R20_G0",
]

# Alternate shot pool drawn from the cross-family stratified_sample_hand_coded.csv
# instead of the single-run GPT_s43_FULL pool above. Chosen by
# select_shot_discussions.py (greedy label-value coverage, tie-break toward a
# new model family) -- covers every Human_* label value seen anywhere in that
# 319-row sample using one discussion from each of GPT/Llama-7B/Mistral-7B/Qwen-7B
# plus 4 more (2 Mistral-7B, 3 more GPT) to reach 8 total.
SHOT_POOL_DISCUSSION_IDS_STRATIFIED = [
    "Mistral-7B_s45_FULL_R05_G2",
    "Mistral-7B_s49_FULL_R08_G0",
    "GPT_s43_FULL_R06_G1",
    "Llama-7B_s43_NO_SELECTION_R07_G0",
    "Qwen-7B_s43_FULL_R04_G1",
    "GPT_s43_FULL_R14_G2",
    "GPT_s43_FULL_R18_G1",
    "GPT_s43_NO_SELECTION_R06_G1",
]

SHOT_CONDITIONS = [
    {"name": "zero_shot", "kind": "none"},
    {"name": "codebook_examples", "kind": "codebook"},           # the 3 examples baked into the codebook JSON
    {"name": "1_example_discussion", "kind": "pool", "n_discussions": 1},
    {"name": "2_example_discussions", "kind": "pool", "n_discussions": 2},
    {"name": "4_example_discussions", "kind": "pool", "n_discussions": 4},
    {"name": "8_example_discussions", "kind": "pool", "n_discussions": 8},
    {"name": "12_example_discussions", "kind": "pool", "n_discussions": 12},
]


# ============================================================
# Prompt construction
# ============================================================

def build_codebook_prompt(codebook: dict, per_group: bool = False) -> str:
    if per_group:
        task_line = (
            "You will be given a full discussion (every turn in one group's round, in order). "
            "Assign one value for each of the four fields below to EVERY turn (or null if a "
            "field cannot be determined from the codebook's own definitions). Base your answer "
            "strictly on the definitions given -- do not use outside judgment about what seems "
            "reasonable."
        )
    else:
        task_line = (
            "For the TARGET turn only, assign one value for each of the four fields below (or null "
            "if the field cannot be determined from the codebook's own definitions). Base your "
            "answer strictly on the definitions given -- do not use outside judgment about what "
            "seems reasonable."
        )
    lines = [
        "You are an expert discourse coder annotating turns from multi-agent public-goods-game "
        "discussions using a fixed codebook. Each turn is one agent's message proposing or "
        "responding to a contribution level for the current round.",
        "",
        task_line,
        "",
    ]
    for field, spec in codebook["annotation_fields"].items():
        short = field.replace("Human_", "")
        lines.append(f"## {short}")
        for value, vspec in spec["values"].items():
            lines.append(f'- "{value}": {vspec["definition"]}')
        lines.append("")
    if per_group:
        lines.append(
            "Respond with a single JSON object whose keys are the turn numbers (as strings, e.g. "
            '"0", "1") and whose values are label objects with exactly these keys: '
            + ", ".join(f'"{f}"' for f in FIELDS)
            + ', "Notes". Include an entry for every turn shown. Use null for any label field '
            "that does not apply to that turn. \"Notes\" is a one-sentence rationale."
        )
    else:
        lines.append(
            "Respond with a single JSON object with exactly these keys: "
            + ", ".join(f'"{f}"' for f in FIELDS)
            + ', "Notes". Use null for any of the five label fields that does not apply to the '
            "target turn. \"Notes\" is a one-sentence rationale for your labels."
        )
    return "\n".join(lines)


def format_codebook_builtin_examples(codebook: dict) -> str:
    blocks = []
    for ex in codebook.get("coding_examples", []):
        ctx = f"\nContext: {ex['context']}" if "context" in ex else ""
        blocks.append(
            f'Message: "{ex["text"]}"{ctx}\n'
            f"Labels: {ex['labels']} -- {ex['explanation']}"
        )
    return "### Codebook examples\n\n" + "\n\n".join(blocks)


def discussion_transcript(df: pd.DataFrame, discussion_id: str, upto_turn: int | None = None) -> pd.DataFrame:
    d = df[df.discussion_id == discussion_id].sort_values("turn")
    if upto_turn is not None:
        d = d[d.turn <= upto_turn]
    return d


def format_transcript_lines(d: pd.DataFrame) -> str:
    return "\n".join(
        f'Turn {int(r.turn)} -- Agent {r.agent}: "{r.utterance}"' for r in d.itertuples()
    )


def row_ground_truth(row: pd.Series) -> dict:
    labels = {}
    for f in FIELDS:
        val = row.get(f"Human_{f}")
        if pd.isna(val) or val == "":
            val = "No" if f in BLANK_MEANS_NO else None
        labels[f] = val
    return labels


def format_pool_discussion_example(df: pd.DataFrame, discussion_id: str) -> str:
    d = discussion_transcript(df, discussion_id)
    lines = [format_transcript_lines(d), "", "Correct labels per turn:"]
    for _, r in d.iterrows():
        gt = row_ground_truth(r)
        lines.append(f"Turn {int(r['turn'])} (Agent {r['agent']}): {json.dumps(gt)}")
    return "\n".join(lines)


def build_shots_block(condition: dict, codebook: dict, pool_df: pd.DataFrame,
                       shot_discussion_ids: list[str]) -> str:
    kind = condition["kind"]
    if kind == "none":
        return ""
    if kind == "codebook":
        return format_codebook_builtin_examples(codebook)
    if kind == "pool":
        chosen = shot_discussion_ids[: condition["n_discussions"]]
        blocks = [format_pool_discussion_example(pool_df, did) for did in chosen]
        return "### Example discussions (hand-coded)\n\n" + "\n\n".join(blocks)
    raise ValueError(f"unknown shot kind: {kind}")


def build_user_prompt(df: pd.DataFrame, row: pd.Series) -> str:
    d = discussion_transcript(df, row["discussion_id"], upto_turn=row["turn"])
    transcript = format_transcript_lines(d)
    return (
        f"Discussion so far (this group's messages so far this round, in order):\n{transcript}\n\n"
        f'TARGET turn to label: Turn {int(row["turn"])} -- Agent {row["agent"]}.\n'
        "Return the JSON object of labels for this target turn only."
    )


def build_group_user_prompt(d: pd.DataFrame) -> str:
    transcript = format_transcript_lines(d)
    turns = ", ".join(str(int(t)) for t in sorted(d.turn.unique()))
    return (
        f"Full discussion (every turn in this group's round, in order):\n{transcript}\n\n"
        f"Label every turn listed above (turns {turns}). Return the JSON object of "
        "per-turn labels as specified in the system instructions."
    )


# ============================================================
# LLM call
# ============================================================

_usage: dict = defaultdict(int)


def annotate(system_prompt: str, user_prompt: str, model: str) -> dict:
    call_kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    if model not in FIXED_TEMPERATURE_MODELS:
        call_kwargs["temperature"] = 0
    try:
        resp = client.chat.completions.create(**call_kwargs)
        parsed = json.loads(resp.choices[0].message.content)
        u = resp.usage
        _usage["prompt_tokens"] += u.prompt_tokens
        _usage["completion_tokens"] += u.completion_tokens
    except Exception as e:
        parsed = {f: None for f in FIELDS}
        parsed["Notes"] = None
        parsed["_error"] = str(e)
    return parsed


def annotate_group(system_prompt: str, user_prompt: str, model: str) -> dict:
    """Like annotate(), but the model returns one object per turn number
    (string keys) instead of a single flat label object."""
    call_kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    if model not in FIXED_TEMPERATURE_MODELS:
        call_kwargs["temperature"] = 0
    try:
        resp = client.chat.completions.create(**call_kwargs)
        parsed = json.loads(resp.choices[0].message.content)
        if not isinstance(parsed, dict):
            raise ValueError(f"expected a JSON object keyed by turn, got {type(parsed)}")
        u = resp.usage
        _usage["prompt_tokens"] += u.prompt_tokens
        _usage["completion_tokens"] += u.completion_tokens
    except Exception as e:
        parsed = {"_error": str(e)}
    return parsed


# ============================================================
# Run a condition / score results
# ============================================================

def run_condition(condition: dict, codebook: dict, eval_source_df: pd.DataFrame,
                   pool_df: pd.DataFrame, eval_df: pd.DataFrame, model: str,
                   shot_discussion_ids: list[str]) -> pd.DataFrame:
    system_prompt = build_codebook_prompt(codebook)
    shots = build_shots_block(condition, codebook, pool_df, shot_discussion_ids)
    if shots:
        system_prompt += "\n\n" + shots

    records = []
    for _, row in tqdm(list(eval_df.iterrows()), desc=condition["name"]):
        user_prompt = build_user_prompt(eval_source_df, row)
        pred = annotate(system_prompt, user_prompt, model)
        rec = row.to_dict()
        for f in FIELDS:
            rec[f"LLM_{f}"] = pred.get(f)
        rec["LLM_Notes"] = pred.get("Notes")
        if "_error" in pred:
            rec["LLM_error"] = pred["_error"]
        records.append(rec)
    return pd.DataFrame(records)


def run_group_condition(condition: dict, codebook: dict, eval_source_df: pd.DataFrame,
                         pool_df: pd.DataFrame, model: str,
                         shot_discussion_ids: list[str]) -> pd.DataFrame:
    """One API call per discussion (group-round), labeling every turn in it at
    once, instead of one call per utterance."""
    system_prompt = build_codebook_prompt(codebook, per_group=True)
    shots = build_shots_block(condition, codebook, pool_df, shot_discussion_ids)
    if shots:
        system_prompt += "\n\n" + shots

    records = []
    discussion_ids = eval_source_df.discussion_id.unique()
    for did in tqdm(list(discussion_ids), desc=condition["name"]):
        d = discussion_transcript(eval_source_df, did)
        user_prompt = build_group_user_prompt(d)
        pred_by_turn = annotate_group(system_prompt, user_prompt, model)
        error = pred_by_turn.get("_error")
        for _, row in d.iterrows():
            pred = pred_by_turn.get(str(int(row["turn"])), {}) if not error else {}
            rec = row.to_dict()
            for f in FIELDS:
                rec[f"LLM_{f}"] = pred.get(f)
            rec["LLM_Notes"] = pred.get("Notes")
            if error:
                rec["LLM_error"] = error
            records.append(rec)
    return pd.DataFrame(records)


def score(results_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    field_rows = []
    for f in FIELDS:
        y_true, y_pred = [], []
        for _, r in results_df.iterrows():
            gt = r[f"Human_{f}"]
            if pd.isna(gt) or gt == "":
                gt = "No" if f in BLANK_MEANS_NO else None
            pred = r[f"LLM_{f}"]
            if pred in ("", "null", "None"):
                pred = None
            y_true.append(str(gt))
            y_pred.append(str(pred))
        acc = accuracy_score(y_true, y_pred)
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except ValueError:
            kappa = float("nan")
        field_rows.append({"field": f, "n": len(y_true), "accuracy": acc, "cohen_kappa": kappa})

    def row_match(r):
        for f in FIELDS:
            gt = r[f"Human_{f}"]
            if pd.isna(gt) or gt == "":
                gt = "No" if f in BLANK_MEANS_NO else None
            pred = r[f"LLM_{f}"]
            if pred in ("", "null", "None"):
                pred = None
            if str(gt) != str(pred):
                return False
        return True

    exact_match_rate = results_df.apply(row_match, axis=1).mean()
    return pd.DataFrame(field_rows), exact_match_rate


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions", type=str, default="all",
        help="Comma-separated condition names to run, or 'all' (default). "
             f"Choices: {', '.join(c['name'] for c in SHOT_CONDITIONS)}",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max eval turns per condition. Default: 32 (the fixed 8-discussion eval set) "
             "when --eval-data is not given; no cap (score every row) when it is.",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument(
        "--eval-data", type=str, default=None,
        help="Path to an external hand-coded CSV (same column schema) to use as the eval set "
             "instead of the held-out slice of DATA_PATH. Shots still come from the "
             "GPT_s43_FULL pool by default regardless of this flag -- pass --shot-data to draw "
             "shots from --eval-data's own file instead. The discussion-so-far context for each "
             "eval row is built from this same file, so it must contain every turn (not just "
             "the target turns) for its discussions.",
    )
    parser.add_argument(
        "--shot-data", type=str, default=None,
        help="Path to a hand-coded CSV to draw few-shot example discussions from, instead of "
             "DATA_PATH (GPT_s43_FULL_hand_coded_sample.csv). Pair with --shot-discussion-ids "
             "(defaults to SHOT_POOL_DISCUSSION_IDS_STRATIFIED if this points at "
             "stratified_sample_hand_coded.csv, otherwise you must pass --shot-discussion-ids "
             "explicitly). Discussions used as shots are always excluded from the eval set.",
    )
    parser.add_argument(
        "--shot-discussion-ids", type=str, default=None,
        help="Comma-separated discussion_ids to use as the shot pool (overrides the default "
             "for whichever file --shot-data/DATA_PATH points at).",
    )
    parser.add_argument(
        "--per-group", action="store_true",
        help="Annotate one whole discussion (group-round) per API call, asking for labels on "
             "every turn at once, instead of one call per utterance.",
    )
    args = parser.parse_args()

    conditions = SHOT_CONDITIONS if args.conditions == "all" else [
        c for c in SHOT_CONDITIONS if c["name"] in args.conditions.split(",")
    ]
    if not conditions:
        raise SystemExit(f"No matching conditions for --conditions={args.conditions!r}")

    codebook = json.load(open(CODEBOOK_PATH))

    shot_path = args.shot_data or DATA_PATH
    shot_source_df = pd.read_csv(shot_path)
    if args.shot_discussion_ids:
        shot_discussion_ids = args.shot_discussion_ids.split(",")
    elif shot_path == DATA_PATH:
        shot_discussion_ids = SHOT_POOL_DISCUSSION_IDS
    elif os.path.basename(shot_path) == "stratified_sample_hand_coded.csv":
        shot_discussion_ids = SHOT_POOL_DISCUSSION_IDS_STRATIFIED
    else:
        raise SystemExit(f"--shot-data={shot_path!r} has no default shot pool -- pass --shot-discussion-ids")
    pool_df = shot_source_df[shot_source_df.discussion_id.isin(shot_discussion_ids)]
    missing = set(shot_discussion_ids) - set(pool_df.discussion_id)
    if missing:
        raise SystemExit(f"shot_discussion_ids not found in --shot-data: {missing}")

    if args.eval_data:
        eval_source_df = pd.read_csv(args.eval_data)
    else:
        eval_source_df = pd.read_csv(DATA_PATH)

    # Shots are only excluded from eval when both come from the same underlying
    # file (by path); an eval file distinct from the shot file needs no overlap
    # check since group-round discussion_ids are unique per source run anyway,
    # but we guard explicitly in case the same discussion_id shows up in both.
    overlap = set(shot_discussion_ids) & set(eval_source_df.discussion_id)
    eval_df = eval_source_df[~eval_source_df.discussion_id.isin(overlap)]
    if not args.eval_data and not args.shot_data:
        eval_df = eval_df.head(args.limit or 32)
    elif args.limit:
        eval_df = eval_df.head(args.limit)

    os.makedirs(args.out_dir, exist_ok=True)

    summaries = []
    for condition in conditions:
        if args.per_group:
            results_df = run_group_condition(condition, codebook, eval_df, pool_df, args.model,
                                               shot_discussion_ids)
        else:
            results_df = run_condition(condition, codebook, eval_source_df, pool_df, eval_df,
                                        args.model, shot_discussion_ids)
        results_df.to_csv(os.path.join(args.out_dir, f"{condition['name']}_annotated.csv"), index=False)

        field_summary, exact_match_rate = score(results_df)
        field_summary["condition"] = condition["name"]
        field_summary["row_exact_match"] = exact_match_rate
        summaries.append(field_summary)

        print(f"\n=== {condition['name']} (n={len(results_df)}, row exact match={exact_match_rate:.1%}) ===")
        print(field_summary.drop(columns=["condition", "row_exact_match"]).to_string(index=False))

    overall = pd.concat(summaries, ignore_index=True)
    overall.to_csv(os.path.join(args.out_dir, "condition_comparison_summary.csv"), index=False)

    print(f"\nPrompt tokens: {_usage['prompt_tokens']:,}  Completion tokens: {_usage['completion_tokens']:,}")
    print(f"Saved per-condition annotations and condition_comparison_summary.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
