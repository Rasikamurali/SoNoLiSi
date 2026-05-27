"""
temp_eval_alignment2.py
-----------------------
Perception-action gap trajectories over 20 rounds.

Gap metrics (per agent per round):
  IN_gap = injunctive_norm  − actual_contribution
  DN_gap = descriptive_norm − actual_contribution

Aggregation: mean across agents within each seed → seed-level trajectory.
Conditions with perceptions: BASELINE, NO_SELECTION, NO_DISCUSSION, FULL.

Analysis 1 — Per model, per condition:
  Seed-level trajectories (thin lines) + cross-seed mean (thick line).
  One figure per model: 2 rows (IN_gap / DN_gap) × 4 cols (conditions).

Analysis 2 — Per condition, across models:
  Cross-seed mean ± SE per model, all 4 models overlaid.
  One figure: 2 rows (IN_gap / DN_gap) × 4 cols (conditions).

Output: figures/2026-03-22/<model>/alignment_analysis/  (Analysis 1)
        figures/2026-03-22/cross_model/                  (Analysis 2)
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "global"
SEEDS      = list(range(43, 53))
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
ROUNDS     = list(range(1, 21))

COND_LABELS = {
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
MODEL_COLORS = {
    "gpt":     "#1f77b4",
    "llama":   "#d62728",
    "mistral": "#2ca02c",
    "qwen":    "#9467bd",
}
MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}

GAP_LABELS = {"IN_gap": "Injunctive Norm − Actual", "DN_gap": "Descriptive Norm − Actual"}
GAP_COLORS = {"IN_gap": "#e377c2", "DN_gap": "#17becf"}

LABEL_SIZE  = 14
TICK_SIZE   = 14
LEGEND_SIZE = 16


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_latest_log(model, seed, condition):
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


def compute_seed_gaps(model, seed, condition):
    """
    Returns dict: round → {"IN_gap": float, "DN_gap": float}
    Each value is the mean gap across all agents with valid perceptions that round.
    """
    d = load_latest_log(model, seed, condition)
    if d is None:
        return {}

    result = {}
    for r in d["round_logs"]:
        rnd          = r["round"]
        contributions = {int(k): float(v) for k, v in r["contributions"].items()}
        perceptions   = {int(k): v        for k, v in (r.get("perceptions") or {}).items()}

        in_gaps, dn_gaps = [], []
        for aid, perc in perceptions.items():
            if not perc:
                continue
            actual = contributions.get(aid)
            if actual is None:
                continue
            inj  = perc.get("injunctive_norm")
            desc = perc.get("descriptive_norm")
            if inj  is not None: in_gaps.append(float(inj)  - actual)
            if desc is not None: dn_gaps.append(float(desc) - actual)

        result[rnd] = {
            "IN_gap": float(np.mean(in_gaps)) if in_gaps else np.nan,
            "DN_gap": float(np.mean(dn_gaps)) if dn_gaps else np.nan,
        }
    return result


def build_store():
    """
    store[model][condition][seed][round] = {"IN_gap": float, "DN_gap": float}
    """
    store = {m: {c: {} for c in CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                gaps = compute_seed_gaps(model, seed, cond)
                if gaps:
                    store[model][cond][seed] = gaps
    return store


def seed_series(store, model, cond, gap_key):
    """
    Returns list of round-series arrays (one per seed) for a given gap metric.
    Each array has length len(ROUNDS), NaN where data is missing.
    """
    series = []
    for seed, round_data in store[model][cond].items():
        arr = np.array([round_data.get(r, {}).get(gap_key, np.nan) for r in ROUNDS])
        series.append(arr)
    return series


def mean_se(series_list):
    """Returns (mean, se) arrays across seeds."""
    arr  = np.array(series_list, dtype=float)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    return mean, se


def savefig(fig, subdir, fname):
    out_dir = os.path.join(FIG_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, fname.replace(".png", f".{ext}"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Analysis 1: Per model — seed trajectories + mean ─────────────────────────

def plot_per_model(store):
    """
    One figure per model.
    Layout: 2 rows (IN_gap, DN_gap) × 4 cols (conditions).
    Thin semi-transparent lines = individual seeds.
    Thick line = cross-seed mean.
    """
    print("\nAnalysis 1 — Per-model perception-action gap trajectories")

    rounds = np.array(ROUNDS)

    for model in MODELS:
        fig, axes = plt.subplots(2, len(CONDITIONS),
                                 figsize=(4 * len(CONDITIONS), 7),
                                 sharex=True, sharey="row")

        for col, cond in enumerate(CONDITIONS):
            for row, gap_key in enumerate(["IN_gap", "DN_gap"]):
                ax      = axes[row, col]
                series  = seed_series(store, model, cond, gap_key)
                color   = GAP_COLORS[gap_key]

                if not series:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                else:
                    for arr in series:
                        ax.plot(rounds, arr, color=color, linewidth=0.8,
                                alpha=0.35, zorder=1)
                    mean, _ = mean_se(series)
                    ax.plot(rounds, mean, color=color, linewidth=2.5,
                            zorder=2, label=GAP_LABELS[gap_key])

                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.tick_params(labelsize=TICK_SIZE)
                ax.grid(True, alpha=0.25)

                if col == 0:
                    ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LABEL_SIZE)
                if row == 0:
                    ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
                if row == len(["IN_gap", "DN_gap"]) - 1:
                    ax.set_xlabel("Round", fontsize=LABEL_SIZE)

        plt.tight_layout()
        savefig(fig, os.path.join(MODEL_LABELS[model].lower(), "alignment_analysis"),
                "perception_action_gap_per_seed.png")


# ─── Analysis 2: Per condition — across models ─────────────────────────────────

def plot_across_models(store):
    """
    One figure: 2 rows (IN_gap, DN_gap) × 4 cols (conditions).
    Each subplot: 4 model lines (cross-seed mean ± SE).
    """
    print("\nAnalysis 2 — Per-condition perception-action gap across models")

    rounds = np.array(ROUNDS)

    fig, axes = plt.subplots(2, len(CONDITIONS),
                             figsize=(4 * len(CONDITIONS), 7),
                             sharex=True, sharey="row")

    for col, cond in enumerate(CONDITIONS):
        for row, gap_key in enumerate(["IN_gap", "DN_gap"]):
            ax = axes[row, col]

            for model in MODELS:
                series = seed_series(store, model, cond, gap_key)
                if not series:
                    continue
                mean, se = mean_se(series)
                color    = MODEL_COLORS[model]
                ax.plot(rounds, mean, color=color, linewidth=2,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=color, alpha=0.12)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
            if row == len(["IN_gap", "DN_gap"]) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.align_ylabels(axes[:, 0])
    savefig(fig, "cross_model", "perception_action_gap_across_models.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building perception-action gap store …")
    store = build_store()

    # Quick data summary
    for model in MODELS:
        for cond in CONDITIONS:
            n = len(store[model][cond])
            print(f"  {model:8s} {cond:15s}: {n} seeds")

    plot_per_model(store)
    plot_across_models(store)

    print("\nAll done.")
