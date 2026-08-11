"""
network_development_all_models.py
----------------------------------
Network development trajectories across ALL models (GPT + 7B + 13B + 70B),
NO_DISCUSSION and FULL conditions, 20 rounds.

Metrics (Spearman alignment dropped):
  gini      — Gini coefficient of incoming-weight distribution
  top_share — fraction of total incoming weight held by top-25% contributors

Output:
  figures/2026-03-22/cross_model/network_development_across_models.png/pdf
  figures/2026-03-22/cross_model/network_development_across_models_horizontal.png/pdf
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE      = "/data3/rasimura/social-norm-evo"
RESULTS   = f"{BASE}/results"
CODE_RES  = f"{BASE}/code/results"
FIG_ROOT  = f"{BASE}/figures/2026-03-22"

CONDITIONS = ["NO_DISCUSSION", "FULL"]
ROUNDS     = list(range(1, 21))

COND_LABELS = {"NO_DISCUSSION": "No Discussion", "FULL": "Full"}

# ── Per-model loading config ──────────────────────────────────────────────────
# (model_key, label, base_dir, variant, seeds)
MODEL_SPECS = [
    ("gpt",        "GPT",         RESULTS,  "global", list(range(43, 53))),
    ("llama",      "Llama-7B",    RESULTS,  "global", list(range(43, 53))),
    ("mistral",    "Mistral-7B",  RESULTS,  "global", list(range(43, 53))),
    ("qwen",       "Qwen-7B",     RESULTS,  "global", list(range(43, 53))),
    ("llama_13b",  "Llama-13B",   RESULTS,  "local",  list(range(42, 52))),
    ("mistral_13b","Mistral-13B", RESULTS,  "local",  list(range(42, 52))),
    ("qwen_14b",   "Qwen-14B",    RESULTS,  "local",  list(range(42, 52))),
    ("llama_70b",  "Llama-70B",   CODE_RES, "local",  list(range(42, 52))),
    ("qwen_72b",   "Qwen-72B",    CODE_RES, "local",  list(range(42, 52))),
]
MODEL_KEYS   = [s[0] for s in MODEL_SPECS]
MODEL_LABELS = {s[0]: s[1] for s in MODEL_SPECS}

# Colour palette — one per model, visually distinct
MODEL_COLORS = {
    "gpt":        "#000000",
    "llama":      "#1f77b4",
    "mistral":    "#d62728",
    "qwen":       "#2ca02c",
    "llama_13b":  "#9467bd",
    "mistral_13b":"#8c564b",
    "qwen_14b":   "#e377c2",
    "llama_70b":  "#bcbd22",
    "qwen_72b":   "#17becf",
}

METRICS = ["gini", "top_share"]
METRIC_LABELS = {
    "gini":      "Gini (incoming weight)",
    "top_share": "Top-quartile weight share",
}

LS = 15   # label size
TS = 13   # tick size
LL = 13   # legend size
LW = 2.0  # line width


# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(base, model_key, variant, seed, condition):
    pattern = os.path.join(base, model_key, variant, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


# ── Metric computation ────────────────────────────────────────────────────────

def incoming_weights(nw, agent_ids):
    iw = {a: 0.0 for a in agent_ids}
    for e in nw:
        v = int(e["v"])
        if v in iw:
            iw[v] += float(e["weight"])
    return iw


def gini(values):
    x = np.array(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0 or x.sum() == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.dot(np.arange(1, n + 1), x)) / (n * x.sum()) - (n + 1) / n


def compute_round_metrics(r):
    nw   = r.get("network_weights", [])
    cont = r.get("contributions", {})
    if not nw or not cont:
        return None
    agent_ids = [int(k) for k in cont]
    iw        = incoming_weights(nw, agent_ids)
    iw_vals   = np.array([iw[a] for a in agent_ids])
    con_vals  = np.array([float(cont[str(a)]) for a in agent_ids])

    g         = gini(iw_vals)
    thresh    = np.percentile(con_vals, 75)
    top_mask  = con_vals >= thresh
    total_iw  = iw_vals.sum()
    top_share = iw_vals[top_mask].sum() / total_iw if total_iw > 0 else np.nan

    return {"gini": g, "top_share": top_share}


def build_store():
    """store[model_key][condition][seed] = {round: {metric: float}}"""
    store = {m: {c: {} for c in CONDITIONS} for m in MODEL_KEYS}
    for model_key, _, base, variant, seeds in MODEL_SPECS:
        for cond in CONDITIONS:
            for seed in seeds:
                d = load_log(base, model_key, variant, seed, cond)
                if d is None:
                    continue
                result = {}
                for r in d["round_logs"]:
                    rnd = r["round"]
                    if rnd not in ROUNDS:
                        continue
                    m = compute_round_metrics(r)
                    if m is not None:
                        result[rnd] = m
                if result:
                    store[model_key][cond][seed] = result
    return store


# ── Aggregation ───────────────────────────────────────────────────────────────

def seed_series(store, model_key, cond, metric):
    series = []
    for seed_data in store[model_key][cond].values():
        arr = np.array([seed_data.get(r, {}).get(metric, np.nan) for r in ROUNDS])
        series.append(arr)
    return series


def mean_se(series_list):
    arr  = np.array(series_list, dtype=float)
    n    = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, se


# ── Save ──────────────────────────────────────────────────────────────────────

def savefig(fig, subdir, fname_stem):
    out_dir = os.path.join(FIG_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{fname_stem}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ── Plot: 2 rows (conditions) × 2 cols (metrics) ─────────────────────────────

def plot_across_models(store):
    """Rows = conditions, cols = metrics."""
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(len(CONDITIONS), len(METRICS),
                             figsize=(5 * len(METRICS), 4 * len(CONDITIONS)),
                             sharex=True)

    for row, cond in enumerate(CONDITIONS):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for model_key in MODEL_KEYS:
                series = seed_series(store, model_key, cond, metric)
                if not series:
                    continue
                mean, se = mean_se(series)
                color    = MODEL_COLORS[model_key]
                ax.plot(rounds, mean, color=color, linewidth=LW,
                        marker="o", markersize=3, label=MODEL_LABELS[model_key])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=color, alpha=0.12)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TS)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(COND_LABELS[cond], fontsize=LS)
            if row == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=LS)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("Round", fontsize=LS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(MODEL_KEYS), 5),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    savefig(fig, "cross_model", "network_development_across_models")


# ── Plot: 2 rows (metrics) × 2 cols (conditions) — horizontal ─────────────────

def plot_across_models_horizontal(store):
    """Rows = metrics, cols = conditions."""
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(len(METRICS), len(CONDITIONS),
                             figsize=(5 * len(CONDITIONS), 4 * len(METRICS)),
                             sharex=True)

    for row, metric in enumerate(METRICS):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            for model_key in MODEL_KEYS:
                series = seed_series(store, model_key, cond, metric)
                if not series:
                    continue
                mean, se = mean_se(series)
                color    = MODEL_COLORS[model_key]
                ax.plot(rounds, mean, color=color, linewidth=LW,
                        marker="o", markersize=3, label=MODEL_LABELS[model_key])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=color, alpha=0.12)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TS)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=LS)
            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LS)
            if row == len(METRICS) - 1:
                ax.set_xlabel("Round", fontsize=LS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(MODEL_KEYS), 5),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    savefig(fig, "cross_model", "network_development_across_models_horizontal")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building network metric store …")
    store = build_store()

    print("\n  Seed counts per model × condition:")
    for model_key in MODEL_KEYS:
        for cond in CONDITIONS:
            n = len(store[model_key][cond])
            print(f"    {MODEL_LABELS[model_key]:14s} × {cond:15s}: {n} seeds")

    print("\nPlotting …")
    plot_across_models(store)
    plot_across_models_horizontal(store)
    print("\nDone.")
