"""
CT.py
-----
Cooperative tendency (CT) trajectories over 20 rounds.

CT is recorded per agent per round in agent_states.
Aggregation: mean CT across agents within each seed → seed-level trajectory.

Analysis 1 — Within model, across conditions:
  One figure per model: all 5 conditions overlaid.
  Thin seed lines + thick cross-seed mean, with ± SE ribbon.

Analysis 2 — Across models, per condition:
  One figure: 2×3 subplots (or 1×5), one per condition.
  4 model lines per subplot (mean ± SE across seeds).

Output:
  figures/2026-03-22/<model>/CT/ct_within_model.png/pdf
  figures/2026-03-22/cross_model/ct_across_models.png/pdf
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
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
ROUNDS     = list(range(1, 21))

COND_LABELS = {
    "PURE_BASELINE":  "Pure Baseline",
    "BASELINE":       "Baseline",
    "NO_SELECTION":   "No Selection",
    "NO_DISCUSSION":  "No Discussion",
    "FULL":           "Full",
}
COND_COLORS = {
    "PURE_BASELINE":  "#aaaaaa",
    "BASELINE":       "#6baed6",
    "NO_SELECTION":   "#fd8d3c",
    "NO_DISCUSSION":  "#74c476",
    "FULL":           "#e6550d",
}
MODEL_COLORS = {"gpt": "#1f77b4", "llama": "#d62728",
                "mistral": "#2ca02c", "qwen": "#9467bd"}
MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}

LABEL_SIZE = 13
TICK_SIZE  = 11

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


def compute_seed_ct(model, seed, condition):
    """
    Returns {round: mean_CT} averaged across agents whose round-1 CT < 0.5.
    """
    d = load_latest_log(model, seed, condition)
    if d is None:
        return {}

    # Identify agents with initial CT < 0.5
    r1 = next((r for r in d["round_logs"] if r["round"] == 1), None)
    if r1 is None:
        return {}
    low_ct_agents = {a["id"] for a in r1["agent_states"]
                     if a.get("cooperation_tendency", 1.0) < 0.5}
    if not low_ct_agents:
        return {}

    result = {}
    for r in d["round_logs"]:
        vals = [a["cooperation_tendency"] for a in r.get("agent_states", [])
                if a["id"] in low_ct_agents and "cooperation_tendency" in a]
        if vals:
            result[r["round"]] = float(np.mean(vals))
    return result


def build_store():
    """store[model][condition][seed] = {round: mean_CT}"""
    store = {m: {c: {} for c in CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                s = compute_seed_ct(model, seed, cond)
                if s:
                    store[model][cond][seed] = s
    return store


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def seed_series(store, model, cond):
    """List of round-series arrays (one per seed)."""
    series = []
    for seed_data in store[model][cond].values():
        arr = np.array([seed_data.get(r, np.nan) for r in ROUNDS])
        series.append(arr)
    return series


def mean_se(series_list):
    arr  = np.array(series_list, dtype=float)
    n    = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, se


def savefig(fig, subdir, fname):
    out_dir = os.path.join(FIG_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, fname.replace(".png", f".{ext}"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Analysis 1: Within model, across conditions ──────────────────────────────

def plot_within_model(store):
    """
    Single figure: 1 row × 4 cols, one subplot per model.
    All 5 conditions overlaid in each subplot.
    Thin seed lines + thick mean with ± SE ribbon.
    """
    print("\nAnalysis 1 — CT within model, across conditions")
    rounds = np.array(ROUNDS)

    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5),
                             sharey=True, sharex=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]

        for cond in CONDITIONS:
            series = seed_series(store, model, cond)
            if not series:
                continue
            color = COND_COLORS[cond]

            for arr in series:
                ax.plot(rounds, arr, color=color, linewidth=0.7,
                        alpha=0.25, zorder=1)

            mean, se = mean_se(series)
            ax.plot(rounds, mean, color=color, linewidth=2.5,
                    label=COND_LABELS[cond], zorder=2)
            ax.fill_between(rounds, mean - se, mean + se,
                            color=color, alpha=0.15, zorder=1)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.25)
        ax.text(0.97, 0.97, MODEL_LABELS[model], transform=ax.transAxes,
                fontsize=LABEL_SIZE, ha="right", va="top")

        if idx == 0:
            ax.set_ylabel("Mean Cooperation Tendency", fontsize=LABEL_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS),
               fontsize=LABEL_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, "cross_model", "ct_within_model.png")


# ─── Analysis 2: Across models, per condition ─────────────────────────────────

def plot_across_models(store):
    """
    One figure: 2 rows × 3 cols subplots (one per condition, last slot empty).
    Arranged as 5 subplots with the 6th hidden.
    4 model lines per subplot (mean ± SE).
    """
    print("\nAnalysis 2 — CT across models, per condition")
    rounds = np.array(ROUNDS)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, cond in enumerate(CONDITIONS):
        ax = axes_flat[idx]

        for model in MODELS:
            series = seed_series(store, model, cond)
            if not series:
                continue
            mean, se = mean_se(series)
            color    = MODEL_COLORS[model]
            ax.plot(rounds, mean, color=color, linewidth=2,
                    marker="o", markersize=3, label=MODEL_LABELS[model])
            ax.fill_between(rounds, mean - se, mean + se,
                            color=color, alpha=0.12)

        ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.25)

        if idx % 3 == 0:
            ax.set_ylabel("Mean Cooperation Tendency", fontsize=LABEL_SIZE)
        if idx >= 3:
            ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    # hide unused 6th subplot
    axes_flat[-1].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right",
               bbox_to_anchor=(0.98, 0.05),
               fontsize=LABEL_SIZE, frameon=False)

    plt.tight_layout()
    savefig(fig, "cross_model", "ct_across_models.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building CT store …")
    store = build_store()

    for model in MODELS:
        for cond in CONDITIONS:
            n = len(store[model][cond])
            print(f"  {model:8s} {cond:15s}: {n} seeds")

    plot_within_model(store)
    plot_across_models(store)
    print("\nAll done.")
