"""
Repeated phrase (n-gram) finder across discussion transcripts.

First step toward measuring anchoring/stabilization: which multi-word
phrases recur most often across agent discussion messages? A high-count
templated phrase (e.g. "given agent_id's suggestion") is a candidate
signal that later speakers are anchoring on earlier ones rather than
generating independent language each round.

Scope: only conditions with non-empty discussion_transcript data
(FULL, NO_SELECTION as of 2026-08). BASELINE / NO_DISCUSSION /
PURE_BASELINE have none and are skipped automatically.

Output: one CSV per normalization mode in code/analysis/exports/,
listing every n-gram (n=2..5) with count >= MIN_COUNT, sorted by count
descending within each n, plus one example (model/seed/round) where it
occurred.
"""

import json
import glob
import os
import re
import csv
from collections import Counter

from nltk.tokenize import word_tokenize

RESULTS = "/data3/rasimura/social-norm-evo/results"
EXPORT_DIR = "/data3/rasimura/social-norm-evo/code/analysis/exports"

CONDITIONS_WITH_DISCUSSION = {"FULL", "NO_SELECTION"}
N_RANGE = range(2, 6)   # 2-gram .. 5-gram
MIN_COUNT = 50

_AGENT_RE = re.compile(r'\bagent\s+\d+\b')


def normalize(msg: str, collapse_agent_ids: bool) -> str:
    msg = msg.lower()
    if collapse_agent_ids:
        msg = _AGENT_RE.sub('agent_id', msg)
    return msg


def tokenize(msg: str):
    return [t for t in word_tokenize(msg) if any(c.isalnum() for c in t)]


def iter_messages(root=RESULTS):
    pattern = os.path.join(root, "*", "*", "seed*", "log_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        cond = d.get("condition")
        if cond not in CONDITIONS_WITH_DISCUSSION:
            continue
        parts = path.split(os.sep)
        model, variant = parts[-4], parts[-3]
        seed = d.get("seed")
        for r in d["round_logs"]:
            rnd = r["round"]
            pos_counter = {}
            for entry in r.get("discussion_transcript") or []:
                msg = entry.get("message", "")
                if not msg:
                    continue
                g = entry.get("group")
                pos = pos_counter.get(g, 0)
                pos_counter[g] = pos + 1
                yield {
                    "model": model, "variant": variant, "seed": seed,
                    "condition": cond, "round": rnd, "group": g,
                    "position_in_group": pos,
                    "agent_id": entry.get("agent_id"), "message": msg,
                }


def count_ngrams(messages, n, collapse_agent_ids):
    counts = Counter()
    example = {}
    for m in messages:
        toks = tokenize(normalize(m["message"], collapse_agent_ids))
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i:i + n])
            counts[gram] += 1
            if gram not in example:
                example[gram] = (m["model"], m["variant"], m["seed"], m["round"])
    return counts, example


def main():
    print("Loading messages...")
    messages = list(iter_messages())
    by_cond = Counter(m["condition"] for m in messages)
    by_variant = Counter(m["variant"] for m in messages)
    print(f"  {len(messages)} messages loaded")
    print(f"  by condition: {dict(by_cond)}")
    print(f"  by variant:   {dict(by_variant)}")

    os.makedirs(EXPORT_DIR, exist_ok=True)

    for collapse in (True, False):
        tag = "agentid_collapsed" if collapse else "raw"
        out_path = os.path.join(EXPORT_DIR, f"repeated_phrases_{tag}.csv")
        rows = []
        print(f"\n--- normalization: {tag} ---")
        for n in N_RANGE:
            counts, example = count_ngrams(messages, n, collapse)
            hits = sorted(
                ((gram, c) for gram, c in counts.items() if c >= MIN_COUNT),
                key=lambda x: -x[1],
            )
            print(f"  n={n}: {len(hits)} phrases with count >= {MIN_COUNT} "
                  f"(out of {len(counts)} distinct {n}-grams)")
            for gram, c in hits[:5]:
                print(f"      {c:>6}  {gram!r}")
            for gram, c in hits:
                model, variant, seed, rnd = example[gram]
                rows.append({
                    "n": n, "phrase": gram, "count": c,
                    "example_model": model, "example_variant": variant,
                    "example_seed": seed, "example_round": rnd,
                })
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "n", "phrase", "count",
                "example_model", "example_variant", "example_seed", "example_round",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
