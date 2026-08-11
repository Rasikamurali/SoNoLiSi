"""
Analysis of discussion dynamics in the NO_SELECTION condition.

Focus:
  - How does discussion evolve from round 1 to round 20?
  - How do violators (CT < 0.3) vs cooperators (CT > 0.7) talk differently?

Methods:
  1. LIWC-style lexical features (VADER sentiment + custom word categories)
  2. Monroe et al. log-odds ratio (Fightin' Words) — violators vs cooperators

Plots:
  A. Per-seed: feature trajectories over rounds, split by CT group
  B. Aggregated per model: mean ± SD across seeds
  C. Log-odds heatmap: word z-scores × rounds (aggregated per model+variant)
  D. Log-odds top-words bar chart at key rounds
"""

import json
import glob
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict, Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"
MODELS   = ["gpt", "llama", "mistral", "qwen"]
VARIANTS = ["global"]
SEEDS    = list(range(43, 53))
CONDITION = "NO_SELECTION"

CT_GROUPS = {
    "Violators (CT<0.3)":    (0.0, 0.3),
    "Normal (0.3–0.7)":      (0.3, 0.7),
    "Cooperators (CT>0.7)":  (0.7, 1.01),
}
CT_COLORS = {
    "Violators (CT<0.3)":    "#d62728",
    "Normal (0.3–0.7)":      "#ff7f0e",
    "Cooperators (CT>0.7)":  "#2ca02c",
}

# ─── LIWC-style lexicons ───────────────────────────────────────────────────────

LEXICONS = {
    "coop_words": {
        "contribute", "contributing", "contribution", "contributions",
        "cooperate", "cooperating", "cooperation", "cooperative",
        "together", "collective", "collectively", "share", "sharing",
        "group", "common", "joint", "help", "support", "supporting",
        "fair", "trust", "trusting", "trusted", "benefit", "mutual",
        "everyone", "everybody", "all", "pool", "pooling",
    },
    "defect_words": {
        "withhold", "withholding", "keep", "keeping", "selfish",
        "individual", "myself", "own", "personal", "alone",
        "maximize", "profit", "advantage", "less", "reduce", "lower",
        "nothing", "zero", "minimal", "minimum",
    },
    "norm_lang": {
        "should", "ought", "must", "expect", "expected", "norm",
        "rule", "standard", "right", "wrong", "fair", "unfair",
        "appropriate", "proper", "acceptable", "reasonable",
        "responsible", "obligated", "supposed",
    },
    "collective_pron": {
        "we", "us", "our", "ours", "ourselves",
    },
    "individual_pron": {
        "i", "me", "my", "mine", "myself",
    },
    "certainty": {
        "always", "definitely", "certain", "certainly", "sure",
        "clearly", "obviously", "will", "absolutely", "undoubtedly",
        "guaranteed", "must", "indeed",
    },
    "hedging": {
        "maybe", "perhaps", "might", "could", "possibly", "possibly",
        "uncertain", "unsure", "think", "believe", "suppose", "guess",
        "seem", "appears", "likely", "probably", "somewhat",
    },
    "numbers": None,  # handled separately via regex
}

_NUMBER_RE = re.compile(r'\b\d+(\.\d+)?\b')


# ─── Helpers ──────────────────────────────────────────────────────────────────

_vader = None

def get_vader():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def load_latest_log(model, variant, seed, condition=CONDITION):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    cond_best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            if d.get("condition") == condition:
                cond_best[p] = d
        except Exception:
            continue
    if not cond_best:
        return None
    # return the latest (by filename timestamp)
    return cond_best[sorted(cond_best)[-1]]


def savefig(fig, model, fname):
    out_dir = os.path.join(FIG_ROOT, model, "discussion_analysis")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def ct_group(ct):
    for name, (lo, hi) in CT_GROUPS.items():
        if lo <= ct < hi:
            return name
    return None


def extract_features(message: str) -> dict:
    """Return a dict of normalized lexical features for a message."""
    vader = get_vader()
    scores = vader.polarity_scores(message)

    tokens = word_tokenize(message.lower())
    n = max(len(tokens), 1)

    feats = {
        "vader_compound": scores["compound"],
        "vader_pos":      scores["pos"],
        "vader_neg":      scores["neg"],
    }

    for cat, wordset in LEXICONS.items():
        if cat == "numbers":
            feats["numbers"] = len(_NUMBER_RE.findall(message)) / n
        else:
            feats[cat] = sum(1 for t in tokens if t in wordset) / n

    # derived: collective / individual pronoun ratio (safe)
    coll = feats["collective_pron"]
    indiv = feats["individual_pron"]
    denom = coll + indiv
    feats["we_i_ratio"] = (coll - indiv) / denom if denom > 0 else 0.0

    return feats


FEATURE_LABELS = {
    "vader_compound":  "Sentiment (VADER compound)",
    "coop_words":      "Cooperation words",
    "defect_words":    "Defection words",
    "norm_lang":       "Norm language",
    "we_i_ratio":      "Collective − Individual pronoun ratio",
    "certainty":       "Certainty words",
    "hedging":         "Hedging words",
    "numbers":         "Number mentions",
}

PLOT_FEATURES = list(FEATURE_LABELS.keys())


# ─── Data collection ──────────────────────────────────────────────────────────

def collect_messages(model, variant):
    """
    Returns a list of records:
      {seed, round, agent_id, ct_initial, ct_current, group, message, **features}

    group is assigned from the agent's round-1 CT so that violator/cooperator
    labels remain stable across all 20 rounds.
    """
    records = []
    for seed in SEEDS:
        d = load_latest_log(model, variant, seed)
        if d is None:
            continue
        # Build initial CT from round 1
        r1 = next((r for r in d["round_logs"] if r["round"] == 1), None)
        if r1 is None:
            continue
        initial_ct = {a["id"]: a["cooperation_tendency"] for a in r1["agent_states"]}

        for r in d["round_logs"]:
            rnd = r["round"]
            current_ct = {a["id"]: a["cooperation_tendency"] for a in r["agent_states"]}
            for entry in r.get("discussion_transcript") or []:
                aid = entry["agent_id"]
                msg = entry.get("message", "")
                if not msg:
                    continue
                ct0 = initial_ct.get(aid)
                if ct0 is None:
                    continue
                grp = ct_group(ct0)
                feats = extract_features(msg)
                records.append({
                    "seed": seed, "round": rnd, "agent_id": aid,
                    "ct_initial": ct0, "ct_current": current_ct.get(aid),
                    "group": grp, "message": msg,
                    **feats,
                })
    return records


# ─── Plot A: Per-seed feature trajectories ────────────────────────────────────

def plot_per_seed_trajectories(model):
    print(f"\n[{model.upper()}] Plot A — per-seed feature trajectories (NO_SELECTION)")

    for variant in VARIANTS:
        records = collect_messages(model, variant)
        if not records:
            print(f"  No data for {variant}")
            continue

        out_dir = os.path.join(FIG_ROOT, model, "discussion_analysis", "per_seed", variant)
        os.makedirs(out_dir, exist_ok=True)

        seeds_present = sorted({r["seed"] for r in records})
        rounds = list(range(1, 21))
        n_feats = len(PLOT_FEATURES)
        ncols = 4
        nrows = math.ceil(n_feats / ncols)

        for seed in seeds_present:
            seed_recs = [r for r in records if r["seed"] == seed]

            # group_round_vals[group][feat][round] = list of values
            grv = {g: {f: defaultdict(list) for f in PLOT_FEATURES} for g in CT_GROUPS}
            for rec in seed_recs:
                g = rec["group"]
                if g is None:
                    continue
                for feat in PLOT_FEATURES:
                    grv[g][feat][rec["round"]].append(rec[feat])

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                                     sharex=True)
            axes = np.array(axes).flatten()
            fig.suptitle(
                f"{model.upper()} — {variant.capitalize()} — seed {seed} — NO_SELECTION\n"
                f"Discussion features over rounds by CT group",
                fontsize=10
            )

            for i, feat in enumerate(PLOT_FEATURES):
                ax = axes[i]
                for grp, color in CT_COLORS.items():
                    means = [np.mean(grv[grp][feat][rnd]) if grv[grp][feat].get(rnd) else np.nan
                             for rnd in rounds]
                    ax.plot(rounds, means, color=color, linewidth=1.8,
                            marker="o", markersize=3, label=grp.split(" ")[0])
                ax.set_title(FEATURE_LABELS[feat], fontsize=8)
                ax.set_xlim(0.5, 20.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=7)

            # hide unused axes
            for j in range(len(PLOT_FEATURES), len(axes)):
                axes[j].set_visible(False)

            for ax in axes[(nrows - 1) * ncols:]:
                ax.set_xlabel("Round", fontsize=8)

            plt.tight_layout()
            path = os.path.join(out_dir, f"seed{seed:02d}_features_by_CT.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {path}")


# ─── Plot B: Aggregated per model ─────────────────────────────────────────────

def plot_aggregated_trajectories(model):
    print(f"\n[{model.upper()}] Plot B — aggregated feature trajectories (NO_SELECTION)")

    for variant in VARIANTS:
        records = collect_messages(model, variant)
        if not records:
            print(f"  No data for {variant}")
            continue

        rounds = list(range(1, 21))
        n_feats = len(PLOT_FEATURES)
        ncols = 4
        nrows = math.ceil(n_feats / ncols)

        # For each seed × group × feat × round → mean, then aggregate across seeds
        seeds_present = sorted({r["seed"] for r in records})

        # seed_means[seed][group][feat][round] = mean over agents in that seed/round
        seed_means = {}
        for seed in seeds_present:
            seed_recs = [r for r in records if r["seed"] == seed]
            sm = {g: {f: {} for f in PLOT_FEATURES} for g in CT_GROUPS}
            for rnd in rounds:
                for grp in CT_GROUPS:
                    vals_by_feat = defaultdict(list)
                    for rec in seed_recs:
                        if rec["round"] == rnd and rec["group"] == grp:
                            for feat in PLOT_FEATURES:
                                vals_by_feat[feat].append(rec[feat])
                    for feat in PLOT_FEATURES:
                        if vals_by_feat[feat]:
                            sm[grp][feat][rnd] = np.mean(vals_by_feat[feat])
            seed_means[seed] = sm

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3),
                                 sharex=True)
        axes = np.array(axes).flatten()
        fig.suptitle(
            f"{model.upper()} — {variant.capitalize()} — NO_SELECTION\n"
            f"Discussion features over rounds (mean ± SD across {len(seeds_present)} seeds)",
            fontsize=10
        )

        for i, feat in enumerate(PLOT_FEATURES):
            ax = axes[i]
            for grp, color in CT_COLORS.items():
                # collect per-seed means for this group+feat across rounds
                per_seed = []
                for seed in seeds_present:
                    row = [seed_means[seed][grp][feat].get(rnd, np.nan) for rnd in rounds]
                    per_seed.append(row)
                per_seed = np.array(per_seed, dtype=float)  # shape (n_seeds, n_rounds)

                means = np.nanmean(per_seed, axis=0)
                sds   = np.nanstd(per_seed, axis=0)

                ax.plot(rounds, means, color=color, linewidth=2,
                        marker="o", markersize=3,
                        label=grp.split(" ")[0])
                ax.fill_between(rounds, means - sds, means + sds,
                                color=color, alpha=0.15)

            ax.set_title(FEATURE_LABELS[feat], fontsize=8)
            ax.set_xlim(0.5, 20.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)

        for j in range(len(PLOT_FEATURES), len(axes)):
            axes[j].set_visible(False)
        for ax in axes[(nrows - 1) * ncols:]:
            ax.set_xlabel("Round", fontsize=8)

        plt.tight_layout()
        savefig(fig, model, f"discussion_features_aggregated_{variant}.png")


# ─── Log-odds (Monroe et al.) ─────────────────────────────────────────────────

def _tokenize(text: str):
    return [t for t in word_tokenize(text.lower())
            if t.isalpha() and len(t) > 2]


STOPWORDS = {
    "the", "and", "that", "for", "this", "with", "are", "has",
    "have", "was", "its", "but", "not", "you", "they", "can",
    "will", "all", "would", "each", "been", "more", "from",
    "our", "who", "what", "out", "about", "one", "your", "when",
    "into", "also", "than", "then", "other", "there", "their",
    "which", "some", "these", "those", "were",
}


def log_odds_ratio(count_a: Counter, count_b: Counter,
                   prior: float = 0.01) -> dict:
    """
    Monroe et al. log-odds ratio with Dirichlet prior.
    Returns {word: z_score}, positive = favours group A.
    """
    vocab = set(count_a) | set(count_b)
    vocab = {w for w in vocab if w not in STOPWORDS}
    n_a = sum(count_a.values())
    n_b = sum(count_b.values())
    V   = len(vocab)

    results = {}
    for w in vocab:
        f_a = count_a.get(w, 0)
        f_b = count_b.get(w, 0)
        # log-odds
        log_a = math.log(f_a + prior) - math.log(n_a - f_a + prior * (V - 1))
        log_b = math.log(f_b + prior) - math.log(n_b - f_b + prior * (V - 1))
        delta = log_a - log_b
        var   = 1.0 / (f_a + prior) + 1.0 / (f_b + prior)
        results[w] = delta / math.sqrt(var)

    return results


def build_round_corpora(records, group_a, group_b):
    """
    Returns {round: (Counter_a, Counter_b)} using all messages from all seeds.
    """
    corpora = {}
    rounds = sorted({r["round"] for r in records})
    for rnd in rounds:
        cnt_a = Counter()
        cnt_b = Counter()
        for rec in records:
            if rec["round"] != rnd:
                continue
            tokens = _tokenize(rec["message"])
            if rec["group"] == group_a:
                cnt_a.update(tokens)
            elif rec["group"] == group_b:
                cnt_b.update(tokens)
        corpora[rnd] = (cnt_a, cnt_b)
    return corpora


# ─── Plot C: Log-odds heatmap (violators vs cooperators) ──────────────────────

def plot_logodds_heatmap(model):
    """
    Heatmap: top-N words × rounds.
    Colour = z-score (positive = violator-favoured, negative = cooperator-favoured).
    Aggregated across all seeds for this model+variant.
    """
    print(f"\n[{model.upper()}] Plot C — log-odds heatmap (Violators vs Cooperators)")

    GROUP_A = "Violators (CT<0.3)"
    GROUP_B = "Cooperators (CT>0.7)"
    TOP_N   = 20  # top N words by |z| (pooled across rounds)
    rounds  = list(range(1, 21))

    for variant in VARIANTS:
        records = collect_messages(model, variant)
        if not records:
            continue

        corpora = build_round_corpora(records, GROUP_A, GROUP_B)

        # Compute z-scores per round
        round_z = {}
        for rnd, (cnt_a, cnt_b) in corpora.items():
            if sum(cnt_a.values()) < 5 or sum(cnt_b.values()) < 5:
                continue
            round_z[rnd] = log_odds_ratio(cnt_a, cnt_b)

        if not round_z:
            print(f"  No data for {variant}")
            continue

        # Select top-N words by mean |z| across rounds
        all_words = set()
        for zs in round_z.values():
            all_words |= set(zs.keys())

        mean_abs_z = {}
        for w in all_words:
            vals = [abs(round_z[rnd].get(w, 0)) for rnd in round_z]
            mean_abs_z[w] = np.mean(vals)

        top_words = sorted(mean_abs_z, key=mean_abs_z.get, reverse=True)[:TOP_N]
        # Sort: violator-favoured (positive mean z) on top, cooperator-favoured below
        mean_z = {w: np.mean([round_z[rnd].get(w, 0) for rnd in round_z])
                  for w in top_words}
        top_words = sorted(top_words, key=lambda w: mean_z[w], reverse=True)

        # Build matrix: words × rounds
        matrix = np.zeros((len(top_words), len(rounds)))
        for j, rnd in enumerate(rounds):
            if rnd in round_z:
                for i, w in enumerate(top_words):
                    matrix[i, j] = round_z[rnd].get(w, 0.0)

        vmax = np.nanpercentile(np.abs(matrix[matrix != 0]), 95) if matrix.any() else 1.0

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels([str(r) for r in rounds], fontsize=8)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=8)
        ax.set_xlabel("Round", fontsize=9)
        ax.set_title(
            f"{model.upper()} — {variant.capitalize()} — NO_SELECTION\n"
            f"Log-odds z-score: Violators vs Cooperators (top {TOP_N} words by |z|)\n"
            f"Red = Violator-favoured  |  Blue = Cooperator-favoured",
            fontsize=9
        )
        plt.colorbar(im, ax=ax, label="z-score", shrink=0.8)
        plt.tight_layout()
        savefig(fig, model, f"discussion_logodds_heatmap_{variant}.png")


# ─── Plot D: Log-odds top words at key rounds ──────────────────────────────────

def plot_logodds_key_rounds(model):
    """
    Horizontal bar charts of top-K words at key rounds
    (R1, R5, R10, R15, R20), violators vs cooperators.
    """
    print(f"\n[{model.upper()}] Plot D — log-odds key rounds bar charts")

    GROUP_A  = "Violators (CT<0.3)"
    GROUP_B  = "Cooperators (CT>0.7)"
    KEY_RNDS = [1, 5, 10, 15, 20]
    TOP_K    = 12

    for variant in VARIANTS:
        records = collect_messages(model, variant)
        if not records:
            continue

        corpora = build_round_corpora(records, GROUP_A, GROUP_B)

        fig, axes = plt.subplots(1, len(KEY_RNDS), figsize=(4 * len(KEY_RNDS), 6),
                                 sharey=False)
        fig.suptitle(
            f"{model.upper()} — {variant.capitalize()} — NO_SELECTION\n"
            f"Log-odds top words: Violators vs Cooperators at key rounds",
            fontsize=10
        )

        for ax, rnd in zip(axes, KEY_RNDS):
            cnt_a, cnt_b = corpora.get(rnd, (Counter(), Counter()))
            if sum(cnt_a.values()) < 3 or sum(cnt_b.values()) < 3:
                ax.set_title(f"Round {rnd}\n(no data)")
                continue

            z_scores = log_odds_ratio(cnt_a, cnt_b)
            # top K in each direction
            sorted_z = sorted(z_scores.items(), key=lambda x: x[1])
            top_neg = sorted_z[:TOP_K // 2]           # cooperator-favoured
            top_pos = sorted_z[-(TOP_K // 2):][::-1]  # violator-favoured
            items   = top_pos + top_neg
            words   = [w for w, _ in items]
            zvals   = [z for _, z in items]
            colors  = ["#d62728" if z > 0 else "#2ca02c" for z in zvals]

            ax.barh(range(len(words)), zvals, color=colors, alpha=0.8)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("z-score", fontsize=8)
            ax.set_title(f"Round {rnd}", fontsize=9)
            ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        savefig(fig, model, f"discussion_logodds_key_rounds_{variant}.png")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for model in MODELS:
        plot_per_seed_trajectories(model)
        plot_aggregated_trajectories(model)
        plot_logodds_heatmap(model)
        plot_logodds_key_rounds(model)
    print("\nAll done.")
