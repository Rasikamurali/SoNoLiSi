"""
Phrase-in-context analysis for discussion transcripts. Builds on
repeated_phrases.py in two ways:

  1. Threshold sweep — at count-size (79k messages), a >=50 cutoff barely
     filters anything (thousands of hits per n). This sweeps several much
     higher thresholds and reports how many phrases survive at each, then
     reports a deduplicated "maximal phrase" list at one chosen threshold
     (a short n-gram that is just a truncation of a longer, near-equally
     common phrase is dropped in favor of the longer one).

  2. Context tagging — a bare phrase + count says nothing about *when* or
     *by whom* it's used. For the maximal phrases, this attaches:
       - mean round, and share of occurrences in early/mid/late rounds
       - share of occurrences by first speaker vs. later speaker in their
         group-round (position_in_group == 0 vs > 0) — directly relevant
         to the anchoring question, since agreement/reference language
         should skew toward later speakers if anchoring is happening
       - 2 verbatim example messages, so the phrase can be read in situ
         rather than as an isolated string
"""

import os
import csv
from collections import Counter, defaultdict

from repeated_phrases import iter_messages, normalize, tokenize, N_RANGE

EXPORT_DIR = "/data3/rasimura/social-norm-evo/code/analysis/exports"
THRESHOLD_SWEEP = [50, 100, 250, 500, 1000, 2000, 5000]
MAXIMAL_THRESHOLD = 500     # phrase set used for the deduped/context pass
ABSORPTION_RATIO = 0.7      # short phrase dropped if a longer one covers this much of its count
ROUND_BUCKETS = [(1, 5, "early (1-5)"), (6, 15, "mid (6-15)"), (16, 20, "late (16-20)")]
COLLAPSE_AGENT_IDS = True   # normalize "agent 11" -> "agent_id" before counting


def bucket_round(rnd):
    for lo, hi, label in ROUND_BUCKETS:
        if lo <= rnd <= hi:
            return label
    return "other"


def tokenize_corpus(messages):
    """Tokenize every message once; returns list of (message_dict, token_tuple)."""
    return [(m, tuple(tokenize(normalize(m["message"], COLLAPSE_AGENT_IDS))))
            for m in messages]


def build_counts(tokenized):
    """counts[n][gram] = count, computed from a pre-tokenized corpus."""
    counts = {n: Counter() for n in N_RANGE}
    for _, toks in tokenized:
        for n in N_RANGE:
            for i in range(len(toks) - n + 1):
                counts[n][toks[i:i + n]] += 1
    return counts


def threshold_sweep(counts):
    print("\n=== Threshold sweep: distinct phrases surviving at each cutoff ===")
    header = "n".rjust(3) + "".join(f"{t:>8}" for t in THRESHOLD_SWEEP)
    print(header)
    for n in N_RANGE:
        row = f"{n:>3}"
        for t in THRESHOLD_SWEEP:
            row += f"{sum(1 for c in counts[n].values() if c >= t):>8}"
        print(row)


def maximal_phrases(counts, threshold):
    """
    Drop any n-gram that is (near-)fully absorbed into a surviving (n+1)-gram,
    i.e. count(n+1 gram) >= ABSORPTION_RATIO * count(n gram) and the n-gram is
    a prefix or suffix of it. Returns list of (n, gram_tuple, count).
    """
    kept_by_n = {}
    for n in sorted(N_RANGE, reverse=True):
        candidates = [(g, c) for g, c in counts[n].items() if c >= threshold]
        if n == max(N_RANGE):
            kept_by_n[n] = candidates
            continue
        longer_grams = [g for g, _ in kept_by_n.get(n + 1, [])]
        survivors = []
        for g, c in candidates:
            absorbed = False
            for lg in longer_grams:
                if len(lg) != n + 1:
                    continue
                if (lg[:n] == g or lg[1:] == g) and counts[n + 1][lg] >= ABSORPTION_RATIO * c:
                    absorbed = True
                    break
            if not absorbed:
                survivors.append((g, c))
        kept_by_n[n] = survivors

    out = []
    for n in N_RANGE:
        for g, c in kept_by_n.get(n, []):
            out.append((n, g, c))
    out.sort(key=lambda x: -x[2])
    return out


def context_stats(tokenized, gram_tuple, n, bucket_totals):
    """Scan pre-tokenized corpus for this exact gram; return round/position aggregates + examples."""
    hits = []
    for m, toks in tokenized:
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == gram_tuple:
                hits.append(m)
                break  # count each message once for position/round purposes
    if not hits:
        return None
    round_bucket_counts = Counter(bucket_round(m["round"]) for m in hits)
    round_bucket_rate = {
        label: round_bucket_counts.get(label, 0) / bucket_totals[label]
        for _, _, label in ROUND_BUCKETS
    }
    first_speaker = sum(1 for m in hits if m["position_in_group"] == 0)
    later_speaker = len(hits) - first_speaker
    examples = hits[:2]
    return {
        "n_messages": len(hits),
        "round_buckets": dict(round_bucket_counts),
        "round_bucket_rate": round_bucket_rate,
        "pct_first_speaker": first_speaker / len(hits),
        "pct_later_speaker": later_speaker / len(hits),
        "examples": examples,
    }


def main():
    print("Loading messages...")
    messages = list(iter_messages())
    print(f"  {len(messages)} messages loaded")

    print("Tokenizing corpus once...")
    tokenized = tokenize_corpus(messages)

    print("Counting n-grams (n=2..5)...")
    counts = build_counts(tokenized)

    bucket_totals = Counter(bucket_round(m["round"]) for m in messages)

    threshold_sweep(counts)

    print(f"\n=== Maximal deduplicated phrases at count >= {MAXIMAL_THRESHOLD} ===")
    maximal = maximal_phrases(counts, MAXIMAL_THRESHOLD)
    print(f"  {len(maximal)} maximal phrases survive (after dropping truncated sub-phrases)")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    rows = []
    report_lines = [
        "# Discussion phrase context report\n",
        f"\nCorpus: {len(messages)} discussion messages (FULL + NO_SELECTION conditions, "
        "all models/variants). Agent-ID references collapsed to `agent_id` before counting.\n",
        f"\n## Threshold sweep\n",
        "\nDistinct n-grams surviving at each count cutoff:\n\n",
        "| n | " + " | ".join(str(t) for t in THRESHOLD_SWEEP) + " |\n",
        "|---" * (len(THRESHOLD_SWEEP) + 1) + "|\n",
    ]
    for n in N_RANGE:
        row = [str(n)] + [str(sum(1 for c in counts[n].values() if c >= t)) for t in THRESHOLD_SWEEP]
        report_lines.append("| " + " | ".join(row) + " |\n")

    report_lines.append(
        f"\n## Maximal phrases at count >= {MAXIMAL_THRESHOLD}, with context\n"
        "\nShort phrases that are near-fully absorbed into a longer surviving phrase "
        "(a truncation, not a distinct pattern) are dropped.\n"
    )

    print(f"\nComputing round/position context for top {min(30, len(maximal))} phrases "
          f"(this rescans the corpus per phrase)...")
    for n, gram, c in maximal[:30]:
        stats = context_stats(tokenized, gram, n, bucket_totals)
        phrase_str = " ".join(gram)
        print(f"\n  [{c:>6}] \"{phrase_str}\"")
        if stats is None:
            continue
        rate_str = ", ".join(f"{label}={stats['round_bucket_rate'][label]:.1%}"
                              for _, _, label in ROUND_BUCKETS)
        print(f"      usage rate by round bucket (share of that bucket's messages): {rate_str}")
        print(f"      first-speaker share: {stats['pct_first_speaker']:.1%}  "
              f"| later-speaker share: {stats['pct_later_speaker']:.1%}")
        for ex in stats["examples"]:
            print(f"      e.g. ({ex['model']}/{ex['variant']}/seed{ex['seed']}/r{ex['round']}): "
                  f"{ex['message'][:140]}")

        rows.append({
            "n": n, "phrase": phrase_str, "count": c,
            "pct_first_speaker": round(stats["pct_first_speaker"], 4),
            "pct_later_speaker": round(stats["pct_later_speaker"], 4),
            "rate_early": round(stats["round_bucket_rate"]["early (1-5)"], 4),
            "rate_mid": round(stats["round_bucket_rate"]["mid (6-15)"], 4),
            "rate_late": round(stats["round_bucket_rate"]["late (16-20)"], 4),
            "example_1": stats["examples"][0]["message"] if stats["examples"] else "",
        })

        report_lines.append(
            f"\n### `{phrase_str}` (n={n}, count={c})\n"
            f"- first-speaker share: {stats['pct_first_speaker']:.1%}, "
            f"later-speaker share: {stats['pct_later_speaker']:.1%}\n"
            f"- usage rate by round bucket (share of that bucket's messages): "
            f"early={stats['round_bucket_rate']['early (1-5)']:.1%}, "
            f"mid={stats['round_bucket_rate']['mid (6-15)']:.1%}, "
            f"late={stats['round_bucket_rate']['late (16-20)']:.1%}\n"
        )
        for ex in stats["examples"]:
            report_lines.append(
                f"- example ({ex['model']}/{ex['variant']}/seed{ex['seed']}/round{ex['round']}): "
                f"\"{ex['message']}\"\n"
            )

    csv_path = os.path.join(EXPORT_DIR, "discussion_phrase_context.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "phrase", "count", "pct_first_speaker", "pct_later_speaker",
            "rate_early", "rate_mid", "rate_late", "example_1",
        ])
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(EXPORT_DIR, "discussion_phrase_context.md")
    with open(md_path, "w") as f:
        f.writelines(report_lines)

    print(f"\nSaved -> {csv_path}")
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()
