"""
discussion_talk_vs_behavior.py
--------------------------------
Three linked questions about the FULL / NO_SELECTION discussion transcripts
(the only two conditions with discussion_on=True; see make_config() in
SoNoLiSi_v5_os_local.py):

1. Talk vs. behavior: does the contribution number an agent states in
   discussion match what it actually contributes that round?
2. First proposal: who speaks first in a group's discussion each round
   (speaking order is randomly shuffled every round - see run_sim(),
   `random.shuffle(agent_order)` - so this is never the same agent's
   "trait", only a per-round event), what number do they propose, and do
   later speakers converge to it?
3. Cross-group heterogeneity: multiple groups discuss independently within
   the same round. Do their discussed targets (and actual contributions)
   look similar to each other, and does that cross-group spread shrink or
   stay flat across rounds?

Number extraction from free text is necessarily approximate. It is
validated against a random sample (see conversation history / dev notes)
at roughly 48/50 sensible matches, with one known soft spot: the "percent"
method assumes "X%" means X% of the 10-token ENDOWMENT, which is not
always the base the model actually meant (e.g. one Llama-13B message says
"20% of our individual payoffs", a different and much larger quantity than
the endowment). Percent-derived numbers are flagged separately in the
output so they can be excluded or inspected on their own.

Outputs (exports/):
    discussion_messages.csv                  - every extracted message, for manual spot-checking
    discussion_talk_vs_behavior_summary.txt  - gap stats (talk number vs actual contribution)
    discussion_anchoring_summary.txt         - first-proposer / convergence stats
    discussion_cross_group_heterogeneity.csv/.png - round-level cross-group spread trend
"""

import os
import re
import json
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sobel_mediation import MODEL_SPECS

warnings.filterwarnings("ignore")

DISCUSSION_CONDITIONS = ["FULL", "NO_SELECTION"]  # the only conditions with discussion_on=True
ENDOWMENT = 10
OUT_DIR = "exports"


# ═════════════════════════════════════════════════════════════════════════
# 1. Number extraction
# ═════════════════════════════════════════════════════════════════════════

def extract_number(text):
    """Best-effort extraction of a proposed/stated contribution number.
    Returns (value_in_0_10_units, method) or (None, None)."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10\b', text, re.I)
    if m:
        return float(m.group(1)), "direct_over_10"
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if m:
        return float(m.group(1)) / 100 * ENDOWMENT, "percent"
    m = re.search(r'contribut\w*\D{0,30}?(\d+(?:\.\d+)?)', text, re.I)
    if m:
        return float(m.group(1)), "raw"
    m = re.search(r'\b(?:around|about|approximately)\s+(\d+(?:\.\d+)?)\b', text, re.I)
    if m:
        return float(m.group(1)), "around"
    return None, None


SELF_COMMIT_RE = re.compile(
    r"\bmy contribution (?:will be|is)\b|\bi will contribute\b|\bi'll (?:give|contribute)\b|"
    r"\bi plan to contribute\b|\bi propose (?:a )?contribut\w*\b", re.I)
GROUP_TARGET_RE = re.compile(
    r"\bwe should\b|\blet'?s\b|\bwe each\b|\bour group\b|\bwe all\b|\bwe agree\b", re.I)


def message_features(msg):
    val, method = extract_number(msg)
    return {
        "extracted_number": val,
        "extraction_method": method,
        "has_self_commit": bool(SELF_COMMIT_RE.search(msg)),
        "has_group_target": bool(GROUP_TARGET_RE.search(msg)),
        "message_len": len(msg),
    }


# ═════════════════════════════════════════════════════════════════════════
# 2. Load message-level dataset
# ═════════════════════════════════════════════════════════════════════════

def flag_implausible(df):
    """A handful of messages state a number wildly outside [0, ENDOWMENT]
    (e.g. Llama-13B: 'I will contribute $93', Mistral-7B: '64 units') - the
    text is extracted correctly, but the model itself is using the wrong
    scale (dollars, or some other unit entirely), almost always in round 1
    before any observed behavior anchors the scale. These are flagged and
    excluded from the aggregate stats below (kept in the raw CSV)."""
    df = df.copy()
    df["number_in_range"] = df["extracted_number"].isna() | df["extracted_number"].between(0, ENDOWMENT)
    return df


def load_messages():
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant,
                                   f"seed{seed}", "log_*.json")
            # dedupe reruns exactly like sobel_mediation.load_all() /
            # build_agent_round_panel.build_panel(): several seed+condition
            # combos have 2-3 separately-timestamped reruns in the same
            # directory (genuinely different completions, not copies); keep
            # only the lexicographically-last one per condition, matching
            # every other script in this analysis suite.
            cond_data = {}
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                    cond_data[d["condition"]] = d
                except Exception:
                    continue
            for cond, d in cond_data.items():
                if cond not in DISCUSSION_CONDITIONS:
                    continue
                run_id = f"{family}_s{seed}_{cond}"
                for r in d["round_logs"]:
                    dt = r.get("discussion_transcript")
                    if not dt:
                        continue
                    contribs = {int(k): float(v) for k, v in r["contributions"].items()}
                    # position within each group, preserving original transcript order
                    pos_counter = {}
                    for m in dt:
                        gid = m["group"]
                        pos = pos_counter.get(gid, 0)
                        pos_counter[gid] = pos + 1
                        feats = message_features(m["message"])
                        rows.append({
                            "family": family, "model": model_key, "condition": cond,
                            "seed": seed, "run_id": run_id, "round": r["round"],
                            "group_id": gid, "position_in_group": pos,
                            "agent_id": m["agent_id"],
                            "actual_contribution": contribs.get(int(m["agent_id"])),
                            "message": m["message"],
                            **feats,
                        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
# 3. Talk vs. behavior gap
# ═════════════════════════════════════════════════════════════════════════

def talk_vs_behavior(msgs):
    # each agent's FINAL stated message in a given run/round/group (in case
    # of multiple discussion turns) is their "closing position"
    d = msgs.dropna(subset=["extracted_number", "actual_contribution"]).copy()
    d = d.sort_values(["run_id", "round", "group_id", "agent_id", "position_in_group"])
    last = d.groupby(["run_id", "round", "group_id", "agent_id"], as_index=False).tail(1)

    last["talk_behavior_gap"] = last["actual_contribution"] - last["extracted_number"]

    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("TALK vs. BEHAVIOR")
    lines.append("gap = actual_contribution - extracted_number "
                 "(positive = did MORE than said; negative = did LESS)")
    lines.append(sep)
    lines.append(f"\nMessages with an extractable number and a matched contribution: "
                 f"{len(last):,} (out of {len(d):,} agent-round-group messages with "
                 f"any number, out of {len(msgs):,} discussion messages total)")

    corr = last[["extracted_number", "actual_contribution"]].corr().iloc[0, 1]
    lines.append(f"Correlation(talk number, actual contribution): r={corr:.3f}")
    lines.append(f"\nOverall gap: mean={last['talk_behavior_gap'].mean():.3f}  "
                 f"median={last['talk_behavior_gap'].median():.3f}  "
                 f"sd={last['talk_behavior_gap'].std():.3f}")
    lines.append(f"  % exact match (gap==0):     {(last['talk_behavior_gap']==0).mean():.1%}")
    lines.append(f"  % did LESS than stated (<0): {(last['talk_behavior_gap']<0).mean():.1%}")
    lines.append(f"  % did MORE than stated (>0): {(last['talk_behavior_gap']>0).mean():.1%}")

    lines.append(f"\n{'-'*70}\nBy model family (mean gap, n)\n{'-'*70}")
    by_fam = last.groupby("family")["talk_behavior_gap"].agg(["mean", "median", "std", "count"])
    lines.append(by_fam.to_string(float_format=lambda x: f"{x:.3f}"))

    lines.append(f"\n{'-'*70}\nBy extraction method (mean gap, n) - 'percent' is lower-confidence,"
                 f" see module docstring\n{'-'*70}")
    by_method = last.groupby("extraction_method")["talk_behavior_gap"].agg(["mean", "median", "count"])
    lines.append(by_method.to_string(float_format=lambda x: f"{x:.3f}"))

    lines.append(f"\n{'-'*70}\nBy message framing\n{'-'*70}")
    for flag, label in [("has_self_commit", "self-commitment phrasing ('I will contribute...')"),
                        ("has_group_target", "group-target phrasing ('we should...')")]:
        sub = last[last[flag]]
        other = last[~last[flag]]
        lines.append(f"  {label}: n={len(sub):,}  mean gap={sub['talk_behavior_gap'].mean():.3f}  "
                     f"(vs. n={len(other):,} without this phrasing, mean gap="
                     f"{other['talk_behavior_gap'].mean():.3f})")

    lines.append(f"\n{'-'*70}\n'percent'-flagged messages by family (share of that family's "
                 f"extractions using the lower-confidence percent method)\n{'-'*70}")
    pct_share = last.groupby("family")["extraction_method"].apply(lambda s: (s == "percent").mean())
    lines.append(pct_share.to_string(float_format=lambda x: f"{x:.1%}"))

    return "\n".join(lines), last


# ═════════════════════════════════════════════════════════════════════════
# 4. First proposal / anchoring
# ═════════════════════════════════════════════════════════════════════════

def anchoring_analysis(msgs):
    d = msgs.copy()
    groups = d.groupby(["run_id", "round", "group_id"])

    opener_rows = []
    follower_rows = []
    for (run_id, rnd, gid), g in groups:
        g = g.sort_values("position_in_group")
        opener = g.iloc[0]
        opener_rows.append({
            "run_id": run_id, "round": rnd, "group_id": gid,
            "opener_agent_id": opener["agent_id"],
            "opening_number": opener["extracted_number"],
            "family": opener["family"], "condition": opener["condition"],
        })
        opening_number = opener["extracted_number"]
        if pd.isna(opening_number):
            continue
        for _, row in g.iloc[1:].iterrows():
            if pd.isna(row["extracted_number"]):
                continue
            follower_rows.append({
                "run_id": run_id, "round": rnd, "group_id": gid,
                "position_in_group": row["position_in_group"],
                "family": row["family"], "condition": row["condition"],
                "opening_number": opening_number,
                "follower_number": row["extracted_number"],
                "matches_opener": abs(row["extracted_number"] - opening_number) < 1e-6,
                "abs_diff_from_opener": abs(row["extracted_number"] - opening_number),
            })

    openers = pd.DataFrame(opener_rows)
    followers = pd.DataFrame(follower_rows)

    # placebo: for each round, pair each follower with a RANDOMLY DRAWN
    # opening_number from a different group in the same run+round, to get
    # a chance-level "match rate" given that round's distribution of
    # opening numbers (some rounds are just more homogeneous than others).
    rng = np.random.default_rng(0)
    placebo_matches = []
    for (run_id, rnd), sub in followers.groupby(["run_id", "round"]):
        pool = openers[(openers.run_id == run_id) & (openers["round"] == rnd)]
        pool = pool.dropna(subset=["opening_number"])
        if len(pool) < 2:
            continue
        for _, row in sub.iterrows():
            other_pool = pool[pool["group_id"] != row["group_id"]]["opening_number"].values
            if len(other_pool) == 0:
                continue
            draw = rng.choice(other_pool)
            placebo_matches.append(abs(draw - row["follower_number"]) < 1e-6)

    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("FIRST PROPOSAL / ANCHORING")
    lines.append("Speaking order within a group is reshuffled every round "
                 "(random.shuffle(agent_order) in run_sim), so 'who opens' is a "
                 "per-round event, not a stable trait of any agent.")
    lines.append(sep)

    n_opener_groups = len(openers)
    n_opener_with_number = openers["opening_number"].notna().sum()
    lines.append(f"\nGroup-rounds with a discussion: {n_opener_groups:,}")
    lines.append(f"Openers with an extractable number: {n_opener_with_number:,} "
                 f"({n_opener_with_number/n_opener_groups:.1%})")

    lines.append(f"\nFollower messages with an extractable number, opener also had one: "
                 f"{len(followers):,}")
    lines.append(f"Exact match to opener's number: {followers['matches_opener'].mean():.1%}")
    lines.append(f"Mean |follower - opener|: {followers['abs_diff_from_opener'].mean():.3f}")
    if placebo_matches:
        lines.append(f"\nPlacebo (follower matched against a random OTHER group's opener, "
                     f"same run+round): exact match rate = {np.mean(placebo_matches):.1%} "
                     f"(n={len(placebo_matches):,})")
        lines.append("  -> the gap between the real match rate and this placebo is the part "
                     "of 'anchoring' not just explained by everyone converging on a similar "
                     "number that round anyway (e.g. because the descriptive norm was similar).")

    lines.append(f"\n{'-'*70}\nAnchoring rate by position in the discussion (2nd speaker vs 3rd vs "
                 f"4th...)\n{'-'*70}")
    by_pos = followers.groupby("position_in_group")["matches_opener"].agg(["mean", "count"])
    lines.append(by_pos.to_string(float_format=lambda x: f"{x:.3f}"))

    lines.append(f"\n{'-'*70}\nAnchoring rate by model family\n{'-'*70}")
    by_fam = followers.groupby("family")["matches_opener"].agg(["mean", "count"])
    lines.append(by_fam.to_string(float_format=lambda x: f"{x:.3f}"))

    lines.append(f"\n{'-'*70}\nAnchoring rate by round (does it strengthen or weaken over the run?)"
                 f"\n{'-'*70}")
    by_round = followers.groupby("round")["matches_opener"].agg(["mean", "count"])
    lines.append(by_round.to_string(float_format=lambda x: f"{x:.3f}"))

    return "\n".join(lines), openers, followers


# ═════════════════════════════════════════════════════════════════════════
# 5. Cross-group heterogeneity within a round, trend across rounds
# ═════════════════════════════════════════════════════════════════════════

def cross_group_heterogeneity(msgs):
    group_level = (msgs.groupby(["run_id", "round", "group_id", "family", "condition"])
                        .agg(discussed_target=("extracted_number", "mean"),
                             actual_mean_contribution=("actual_contribution", "mean"))
                        .reset_index())

    def spread(s):
        return s.std() if len(s) > 1 else np.nan

    round_level = (group_level.groupby(["run_id", "round", "family", "condition"])
                              .agg(n_groups=("group_id", "nunique"),
                                   discussed_target_std=("discussed_target", spread),
                                   actual_contribution_std=("actual_mean_contribution", spread))
                              .reset_index())
    round_level = round_level[round_level["n_groups"] > 1]

    trend = (round_level.groupby(["round", "condition"])
                        .agg(mean_discussed_target_std=("discussed_target_std", "mean"),
                             mean_actual_contribution_std=("actual_contribution_std", "mean"),
                             n_run_rounds=("run_id", "nunique"))
                        .reset_index())
    return group_level, round_level, trend


def plot_heterogeneity_trend(trend, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    colors = {"FULL": "#4C72B0", "NO_SELECTION": "#DD8452"}
    for cond in ["FULL", "NO_SELECTION"]:
        sub = trend[trend["condition"] == cond].sort_values("round")
        axes[0].plot(sub["round"].to_numpy(), sub["mean_discussed_target_std"].to_numpy(),
                    marker="o", color=colors[cond], label=cond)
        axes[1].plot(sub["round"].to_numpy(), sub["mean_actual_contribution_std"].to_numpy(),
                    marker="o", color=colors[cond], label=cond)
    axes[0].set_title("Cross-group spread in DISCUSSED target", fontsize=11)
    axes[1].set_title("Cross-group spread in ACTUAL mean contribution", fontsize=11)
    for ax in axes:
        ax.set_xlabel("round")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Std. dev. across groups within a run-round\n(averaged across runs)")
    fig.suptitle("Do independently-discussing groups within the same run converge to a "
                "similar norm over time?", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading discussion messages (FULL / NO_SELECTION conditions only)...")
    msgs = load_messages()
    print(f"  {len(msgs):,} messages across {msgs['run_id'].nunique()} runs")
    print(f"  Extractable number: {msgs['extracted_number'].notna().mean():.1%}")

    msgs = flag_implausible(msgs)
    n_oob = (~msgs["number_in_range"]).sum()
    n_with_num = msgs["extracted_number"].notna().sum()
    print(f"  Out-of-[0,{ENDOWMENT}]-range extractions (kept in CSV, excluded from stats "
          f"below): {n_oob:,} ({n_oob/n_with_num:.2%} of numbered messages)")
    oob_by_round1 = msgs[~msgs["number_in_range"] & (msgs["round"] == 1)].shape[0]
    print(f"    of which in round 1: {oob_by_round1:,} ({oob_by_round1/max(n_oob,1):.0%}) - "
          f"before any observed behavior anchors the scale")

    msgs.to_csv(os.path.join(OUT_DIR, "discussion_messages.csv"), index=False)

    msgs_clean = msgs.copy()
    msgs_clean.loc[~msgs_clean["number_in_range"], "extracted_number"] = np.nan

    tvb_text, tvb_df = talk_vs_behavior(msgs_clean)
    print("\n" + tvb_text)
    with open(os.path.join(OUT_DIR, "discussion_talk_vs_behavior_summary.txt"), "w") as f:
        f.write(tvb_text + "\n")

    anchor_text, openers, followers = anchoring_analysis(msgs_clean)
    print("\n" + anchor_text)
    with open(os.path.join(OUT_DIR, "discussion_anchoring_summary.txt"), "w") as f:
        f.write(anchor_text + "\n")

    group_level, round_level, trend = cross_group_heterogeneity(msgs_clean)
    trend.to_csv(os.path.join(OUT_DIR, "discussion_cross_group_heterogeneity.csv"), index=False)
    fig_path = os.path.join(OUT_DIR, "discussion_cross_group_heterogeneity.png")
    plot_heterogeneity_trend(trend, fig_path)

    print(f"\n{'='*70}\nCROSS-GROUP HETEROGENEITY (first/last round in each condition)\n{'='*70}")
    for cond in ["FULL", "NO_SELECTION"]:
        sub = trend[trend["condition"] == cond].sort_values("round")
        if len(sub) == 0:
            continue
        first, last = sub.iloc[0], sub.iloc[-1]
        print(f"\n[{cond}]")
        print(f"  round {int(first['round'])}: discussed_target_std={first['mean_discussed_target_std']:.3f}  "
              f"actual_contribution_std={first['mean_actual_contribution_std']:.3f}")
        print(f"  round {int(last['round'])}: discussed_target_std={last['mean_discussed_target_std']:.3f}  "
              f"actual_contribution_std={last['mean_actual_contribution_std']:.3f}")

    print(f"\n{'='*70}\nOUTPUT FILES\n{'='*70}")
    for p in ["discussion_messages.csv", "discussion_talk_vs_behavior_summary.txt",
              "discussion_anchoring_summary.txt", "discussion_cross_group_heterogeneity.csv",
              "discussion_cross_group_heterogeneity.png"]:
        print(f"  {os.path.join(OUT_DIR, p)}")


if __name__ == "__main__":
    main()
