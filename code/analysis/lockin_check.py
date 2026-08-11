"""
lockin_check.py
---------------
Checks whether the local simulation results exhibit lock-in:
do certain agent pairs dominate group membership across rounds?

For each (model, condition, seed), extracts the groups per round from
the simulation logs and computes co-occurrence statistics.

Key metrics:
  - Co-occurrence Gini: 0 = all pairs grouped equally, 1 = one pair always together
  - Max co-occurrence fraction: highest pair / total rounds
  - Pairs never grouped: agents who were never in the same group
  - Expected co-occurrence (random baseline): rounds × (G-1) / (N-1)

Conditions compared:
  - FULL / NO_DISCUSSION: network-based selection active → potential lock-in
  - NO_SELECTION / BASELINE: random grouping → should be near random baseline
  - PURE_BASELINE: random grouping, no perception → cleanest random baseline

Output: figures/lockin_check/
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
OUT_DIR  = "/data3/rasimura/social-norm-evo/figures/lockin_check"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS     = ["gpt", "llama", "mistral", "qwen"]
MODEL_LABELS = {"gpt": "GPT-4o-mini", "llama": "Llama 3.1-8B",
                "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}
VARIANT    = "local"
SEEDS      = list(range(43, 53))

# Conditions split by selection mechanism
SELECTION_CONDITIONS = ["FULL", "NO_DISCUSSION"]          # weight-based grouping
RANDOM_CONDITIONS    = ["NO_SELECTION", "BASELINE", "PURE_BASELINE"]  # random grouping
ALL_CONDITIONS       = SELECTION_CONDITIONS + RANDOM_CONDITIONS

COND_COLORS = {
    "FULL":           "#e6550d",
    "NO_DISCUSSION":  "#74c476",
    "NO_SELECTION":   "#fd8d3c",
    "BASELINE":       "#6baed6",
    "PURE_BASELINE":  "#aaaaaa",
}
COND_LABELS = {
    "FULL":           "Full",
    "NO_DISCUSSION":  "No Discussion",
    "NO_SELECTION":   "No Selection",
    "BASELINE":       "Baseline",
    "PURE_BASELINE":  "Pure Baseline",
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(model, seed, condition):
    pattern = os.path.join(RESULTS, model, VARIANT, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


# ─── Co-occurrence analysis ───────────────────────────────────────────────────

def cooccurrence_from_log(d):
    """
    Build co-occurrence matrix from a simulation log.
    Returns (N, matrix) where N = number of agents.
    """
    agents = set()
    for r in d["round_logs"]:
        for g in r["groups"]:
            agents.update(g)
    n      = len(agents)
    agent_list = sorted(agents)
    idx    = {a: i for i, a in enumerate(agent_list)}
    mat    = np.zeros((n, n), dtype=int)

    for r in d["round_logs"]:
        for g in r["groups"]:
            for i in g:
                for j in g:
                    if i != j:
                        mat[idx[i], idx[j]] += 1
    return n, mat


def gini(values):
    vals = np.sort(np.array(values, dtype=float))
    n    = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    idx  = np.arange(1, n + 1)
    return float((2 * (idx * vals).sum() / (n * vals.sum())) - (n + 1) / n)


def lockin_stats(n, mat, rounds):
    """Compute lock-in metrics from a co-occurrence matrix."""
    # Upper-triangle only (undirected pairs)
    pairs = [mat[i, j] for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return {}
    group_size  = 4   # assumed
    expected    = rounds * (group_size - 1) / (n - 1)
    return {
        "n_agents":         n,
        "n_pairs":          len(pairs),
        "rounds":           rounds,
        "expected_random":  round(expected, 2),
        "mean_cooc":        round(np.mean(pairs), 3),
        "sd_cooc":          round(np.std(pairs), 3),
        "min_cooc":         int(min(pairs)),
        "max_cooc":         int(max(pairs)),
        "gini_cooc":        round(gini(pairs), 4),
        "pairs_never":      int(sum(1 for p in pairs if p == 0)),
        "pairs_2x_expected": int(sum(1 for p in pairs if p >= 2 * expected)),
        "max_frac_rounds":  round(max(pairs) / rounds, 3),
    }


# ─── Build full results table ─────────────────────────────────────────────────

def build_results():
    rows = []
    for model in MODELS:
        for cond in ALL_CONDITIONS:
            for seed in SEEDS:
                d = load_log(model, seed, cond)
                if d is None:
                    continue
                rounds = len(d["round_logs"])
                n, mat = cooccurrence_from_log(d)
                stats  = lockin_stats(n, mat, rounds)
                rows.append({
                    "model": model, "condition": cond, "seed": seed,
                    **stats,
                })
    return pd.DataFrame(rows)


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_gini_by_condition(df):
    """Box plots of co-occurrence Gini per condition, all models combined."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 5),
                             sharey=True)
    fig.suptitle("Co-occurrence Gini by condition\n"
                 "(higher = more concentrated grouping = more lock-in risk)",
                 fontsize=13, y=1.02)

    for ax, model in zip(axes, MODELS):
        data   = [df[(df["model"] == model) & (df["condition"] == c)]["gini_cooc"].values
                  for c in ALL_CONDITIONS]
        labels = [COND_LABELS[c] for c in ALL_CONDITIONS]
        colors = [COND_COLORS[c] for c in ALL_CONDITIONS]

        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        # Random baseline reference
        ax.axhline(df[df["condition"].isin(RANDOM_CONDITIONS)]["gini_cooc"].mean(),
                   color="gray", linestyle="--", linewidth=1.5,
                   label="Random baseline mean")

        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_ylim(0, 0.8)
        ax.grid(True, alpha=0.3, axis="y")
        if ax == axes[0]:
            ax.set_ylabel("Co-occurrence Gini", fontsize=11)
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "gini_by_condition.png"),
                dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "gini_by_condition.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  Saved: gini_by_condition.png/.pdf")


def plot_pairs_never(df):
    """Bar chart: mean number of pairs that never co-occur per condition."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 4.5),
                             sharey=True)
    fig.suptitle("Mean number of agent pairs that NEVER appear in the same group\n"
                 "(> 0 = some pairs always excluded from each other)", fontsize=13, y=1.02)

    for ax, model in zip(axes, MODELS):
        means = [df[(df["model"] == model) & (df["condition"] == c)]["pairs_never"].mean()
                 for c in ALL_CONDITIONS]
        bars  = ax.bar(range(len(ALL_CONDITIONS)), means,
                       color=[COND_COLORS[c] for c in ALL_CONDITIONS], alpha=0.85)
        ax.set_xticks(range(len(ALL_CONDITIONS)))
        ax.set_xticklabels([COND_LABELS[c] for c in ALL_CONDITIONS],
                           rotation=30, ha="right", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", linewidth=0.8)
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                        f"{val:.1f}", ha="center", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Mean pairs never co-grouped", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pairs_never_grouped.png"),
                dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "pairs_never_grouped.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  Saved: pairs_never_grouped.png/.pdf")


def plot_max_cooccurrence(df):
    """Max co-occurrence fraction: how much one pair dominates relative to random."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 4.5),
                             sharey=True)
    fig.suptitle("Max co-occurrence fraction (highest pair / total rounds)\n"
                 "Dashed line = random expectation", fontsize=13, y=1.02)

    for ax, model in zip(axes, MODELS):
        for cond in ALL_CONDITIONS:
            sub  = df[(df["model"] == model) & (df["condition"] == cond)]["max_frac_rounds"]
            x    = ALL_CONDITIONS.index(cond)
            ax.scatter([x] * len(sub), sub, color=COND_COLORS[cond],
                       alpha=0.6, s=40, zorder=3)
            ax.plot([x - 0.3, x + 0.3], [sub.mean(), sub.mean()],
                    color=COND_COLORS[cond], linewidth=2.5, zorder=4)

        # random expectation
        expected_frac = df[df["condition"] == "PURE_BASELINE"]["max_frac_rounds"].mean()
        ax.axhline(expected_frac, color="gray", linestyle="--", linewidth=1.5,
                   label=f"PB mean: {expected_frac:.2f}")

        ax.set_xticks(range(len(ALL_CONDITIONS)))
        ax.set_xticklabels([COND_LABELS[c] for c in ALL_CONDITIONS],
                           rotation=30, ha="right", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.0)
        if ax == axes[0]:
            ax.set_ylabel("Max co-occurrence / rounds", fontsize=11)
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "max_cooccurrence.png"),
                dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "max_cooccurrence.pdf"),
                bbox_inches="tight")
    plt.close()
    print("  Saved: max_cooccurrence.png/.pdf")


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary(df):
    print("\n── Co-occurrence lock-in summary ─────────────────────────────────")
    print(f"  {'Model':12s} {'Condition':18s} {'Gini':>6} {'Pairs=0':>8} "
          f"{'MaxFrac':>8} {'Mean':>6} {'SD':>6}")
    print("  " + "-" * 68)
    for model in MODELS:
        for cond in ALL_CONDITIONS:
            sub = df[(df["model"] == model) & (df["condition"] == cond)]
            if sub.empty:
                continue
            flag = "⚠" if sub["gini_cooc"].mean() > 0.35 or sub["pairs_never"].mean() > 2 else " "
            print(f"  {MODEL_LABELS[model]:12s} {COND_LABELS[cond]:18s} "
                  f"{sub['gini_cooc'].mean():>6.3f} "
                  f"{sub['pairs_never'].mean():>8.1f} "
                  f"{sub['max_frac_rounds'].mean():>8.3f} "
                  f"{sub['mean_cooc'].mean():>6.2f} "
                  f"{sub['sd_cooc'].mean():>6.2f} {flag}")
        print()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading simulation logs …")
    df = build_results()
    print(f"  {len(df)} runs loaded ({df['model'].nunique()} models × "
          f"{df['condition'].nunique()} conditions × ~{df['seed'].nunique()} seeds)")

    print_summary(df)

    df.to_csv(os.path.join(OUT_DIR, "lockin_stats.csv"), index=False)
    print(f"\n  Saved: lockin_stats.csv")

    print("\nGenerating plots …")
    plot_gini_by_condition(df)
    plot_pairs_never(df)
    plot_max_cooccurrence(df)

    print(f"\nAll outputs → {OUT_DIR}")
