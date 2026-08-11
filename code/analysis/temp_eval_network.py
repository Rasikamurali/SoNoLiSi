"""
temp_eval_network.py
--------------------
Network development trajectories over 20 rounds.

Two metrics computed per seed per round:
  1. gini         — Gini coefficient of incoming-weight distribution
                    Does weight become more concentrated (unequal) over time?
  2. top_share    — fraction of total incoming weight held by top-25% contributors
                    Do the highest contributors capture more network influence?

Conditions with selection: NO_DISCUSSION, FULL.

Analysis 1 — Per model: NO_DISCUSSION vs FULL
  One figure per model: 3 rows (metrics) × 2 cols (conditions).
  Thin lines = individual seeds; thick line = cross-seed mean.

Analysis 2 — Across models: per condition
  One figure: 3 rows (metrics) × 2 cols (NO_DISCUSSION / FULL).
  4 model lines per subplot (mean ± SE across seeds).

Output:
  figures/2026-03-22/<model>/selection_analysis/network_development.png/pdf
  figures/2026-03-22/cross_model/network_development_across_models.png/pdf
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
VARIANT    = "local"
SEEDS      = list(range(43, 53))
CONDITIONS = ["NO_DISCUSSION", "FULL"]
ROUNDS     = list(range(1, 21))

COND_LABELS  = {"NO_DISCUSSION": "No Discussion", "FULL": "Full"}
COND_COLORS  = {"NO_DISCUSSION": "#9467bd", "FULL": "#2ca02c"}
MODEL_COLORS = {"gpt": "#1f77b4", "llama": "#d62728",
                "mistral": "#2ca02c", "qwen": "#9467bd"}
MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}

METRIC_LABELS = {
    "gini":      "Gini (incoming weight)",
    "top_share": "Top-quartile weight share",
}

LABEL_SIZE  = 16
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


# ─── Metric computation ───────────────────────────────────────────────────────

def incoming_weights(network_weights, agent_ids):
    """Return dict {agent_id: total_incoming_weight}."""
    iw = {a: 0.0 for a in agent_ids}
    for e in network_weights:
        v = int(e["v"])
        if v in iw:
            iw[v] += float(e["weight"])
    return iw


def gini(values):
    """Gini coefficient of a non-negative array."""
    x = np.array(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.dot(np.arange(1, n + 1), x)) / (n * x.sum()) - (n + 1) / n


def compute_round_metrics(r):
    """
    Given a single round_log entry, return dict with alignment, gini, top_share.
    Returns None if network_weights or contributions are missing.
    """
    nw   = r.get("network_weights", [])
    cont = r.get("contributions", {})
    if not nw or not cont:
        return None

    agent_ids = [int(k) for k in cont]
    iw        = incoming_weights(nw, agent_ids)
    contribs  = {int(k): float(v) for k, v in cont.items()}

    iw_vals   = np.array([iw[a]      for a in agent_ids])
    con_vals  = np.array([contribs[a] for a in agent_ids])

    g = gini(iw_vals)

    thresh    = np.percentile(con_vals, 75)
    top_mask  = con_vals >= thresh
    total_iw  = iw_vals.sum()
    top_share = iw_vals[top_mask].sum() / total_iw if total_iw > 0 else np.nan

    return {"gini": g, "top_share": top_share}


def compute_seed_metrics(model, seed, condition):
    """Returns {round: {metric: float}} for one seed/condition."""
    d = load_latest_log(model, seed, condition)
    if d is None:
        return {}
    result = {}
    for r in d["round_logs"]:
        m = compute_round_metrics(r)
        if m is not None:
            result[r["round"]] = m
    return result


def build_store():
    """store[model][condition][seed] = {round: {metric: float}}"""
    store = {m: {c: {} for c in CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                s = compute_seed_metrics(model, seed, cond)
                if s:
                    store[model][cond][seed] = s
    return store


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def seed_series(store, model, cond, metric):
    """List of round-series arrays (one per seed), NaN where missing."""
    series = []
    for seed, round_data in store[model][cond].items():
        arr = np.array([round_data.get(r, {}).get(metric, np.nan) for r in ROUNDS])
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


# ─── Analysis 1: Per model ────────────────────────────────────────────────────

def plot_per_model(store):
    """
    Per model: NO_DISCUSSION vs FULL.
    3 rows (metrics) × 2 cols (conditions).
    Thin seed lines + thick mean.
    """
    print("\nAnalysis 1 — Per-model network development")
    metrics = ["gini", "top_share"]
    rounds  = np.array(ROUNDS)

    for model in MODELS:
        fig, axes = plt.subplots(len(metrics), len(CONDITIONS),
                                 figsize=(4 * len(CONDITIONS), 4 * len(metrics)),
                                 sharex=True)

        for col, cond in enumerate(CONDITIONS):
            color = COND_COLORS[cond]
            for row, metric in enumerate(metrics):
                ax     = axes[row, col]
                series = seed_series(store, model, cond, metric)

                if not series:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                else:
                    for arr in series:
                        ax.plot(rounds, arr, color=color, linewidth=0.8,
                                alpha=0.3, zorder=1)
                    mean, _ = mean_se(series)
                    ax.plot(rounds, mean, color=color, linewidth=2.5, zorder=2)

                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.tick_params(labelsize=TICK_SIZE)
                ax.grid(True, alpha=0.25)

                if col == 0:
                    ax.set_ylabel(METRIC_LABELS[metric], fontsize=LABEL_SIZE)
                if row == 0:
                    ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
                if row == len(metrics) - 1:
                    ax.set_xlabel("Round", fontsize=LABEL_SIZE)

        plt.tight_layout()
        savefig(fig, os.path.join(MODEL_LABELS[model].lower(), "selection_analysis"),
                "network_development.png")


# ─── Analysis 2: Across models ────────────────────────────────────────────────

def plot_across_models(store):
    """
    Per condition, across models.
    2 rows (NO_DISCUSSION / FULL) × 3 cols (metrics).
    4 model lines (mean ± SE).
    """
    print("\nAnalysis 2 — Network development across models")
    metrics = ["gini", "top_share"]
    rounds  = np.array(ROUNDS)

    fig, axes = plt.subplots(len(CONDITIONS), len(metrics),
                             figsize=(4 * len(metrics), 4 * len(CONDITIONS)),
                             sharex=True)

    for row, cond in enumerate(CONDITIONS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            for model in MODELS:
                series = seed_series(store, model, cond, metric)
                if not series:
                    continue
                mean, se = mean_se(series)
                color    = MODEL_COLORS[model]
                ax.plot(rounds, mean, color=color, linewidth=2,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=color, alpha=0.12)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(COND_LABELS[cond], fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=LABEL_SIZE)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    savefig(fig, "cross_model", "network_development_across_models.png")


# ─── Analysis 2b: Across models — horizontal (3 rows × 2 cols) ───────────────

def plot_across_models_horizontal(store):
    """
    3 rows (metrics) × 2 cols (conditions).
    Same data as plot_across_models, transposed layout.
    """
    print("\nAnalysis 2b — Network development across models (horizontal)")
    metrics = ["gini", "top_share"]
    rounds  = np.array(ROUNDS)

    fig, axes = plt.subplots(len(metrics), len(CONDITIONS),
                             figsize=(4 * len(CONDITIONS), 4 * len(metrics)),
                             sharex=True)

    for row, metric in enumerate(metrics):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]

            for model in MODELS:
                series = seed_series(store, model, cond, metric)
                if not series:
                    continue
                mean, se = mean_se(series)
                color    = MODEL_COLORS[model]
                ax.plot(rounds, mean, color=color, linewidth=2,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=color, alpha=0.12)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
            if row == len(metrics) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    savefig(fig, "cross_model", "network_development_across_models_horizontal.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building network metric store …")
    store = build_store()

    for model in MODELS:
        for cond in CONDITIONS:
            n = len(store[model][cond])
            print(f"  {model:8s} {cond:15s}: {n} seeds")

    plot_per_model(store)
    plot_across_models(store)
    plot_across_models_horizontal(store)
    print("\nAll done.")
