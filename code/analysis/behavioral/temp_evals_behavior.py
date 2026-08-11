"""
temp_evals.py
-------------
1. Per run (seed), across conditions, contribution & payoff over rounds — global vs local.
2. Across seeds, per model, contribution & payoff across conditions (mean ± SE).

Output: figures/2026-03-20/<model>/
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"

MODELS   = ["gpt", "llama", "mistral", "qwen"]
VARIANTS = ["global"]
SEEDS    = list(range(43, 53))
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]

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
VARIANT_STYLE = {"global": "-", "local": "--"}
VARIANT_COLORS = {"global": "#1f77b4", "local": "#d62728"}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_latest_log(model: str, variant: str, seed: int, condition: str):
    """Return the latest log dict for model/variant/seed/condition, or None."""
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    cond_best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            cond_best[d["condition"]] = d   # latest timestamp wins (sorted)
        except Exception:
            continue
    return cond_best.get(condition)


def round_series(data: dict, key: str):
    """Extract per-round mean of 'contributions' or 'payoffs'."""
    out = []
    for r in data["round_logs"]:
        vals = list(r[key].values())
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def savefig(fig, model: str, fname: str):
    out_dir = os.path.join(FIG_ROOT, model, "behavioral_analysis")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Per seed: contribution & payoff over rounds, global vs local
#           One figure per model per seed, 2 rows (contrib / payoff) × 5 cols (conditions)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_per_seed(model: str):
    print(f"\n[{model.upper()}] Plot 1 — per-seed, all conditions overlaid (global / local)")
    for seed in SEEDS:
        for variant in VARIANTS:
            # collect data for all conditions
            cond_data = {}
            for cond in CONDITIONS:
                d = load_latest_log(model, variant, seed, cond)
                if d is not None:
                    cond_data[cond] = d

            if not cond_data:
                continue

            fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            fig.suptitle(
                f"{model.upper()} — seed {seed} — {variant.capitalize()}\n"
                f"Contribution & Payoff across Conditions",
                fontsize=11
            )

            for cond, d in cond_data.items():
                rounds  = [r["round"] for r in d["round_logs"]]
                contrib = round_series(d, "contributions")
                payoff  = round_series(d, "payoffs")
                color   = COND_COLORS[cond]
                label   = COND_LABELS[cond]
                ax_c.plot(rounds, contrib, color=color, linewidth=2,
                          marker="o", markersize=3, label=label)
                ax_p.plot(rounds, payoff,  color=color, linewidth=2,
                          marker="o", markersize=3, label=label)

            ax_c.set_ylim(0, 16)
            ax_c.set_ylabel("Mean Contribution (0–10)", fontsize=9)
            ax_c.legend(fontsize=8, loc="lower right")
            ax_c.grid(True, alpha=0.3)
            ax_c.xaxis.set_major_locator(ticker.MultipleLocator(5))

            ax_p.set_ylabel("Mean Payoff", fontsize=9)
            ax_p.set_xlabel("Round", fontsize=9)
            ax_p.legend(fontsize=8, loc="lower right")
            ax_p.grid(True, alpha=0.3)
            ax_p.xaxis.set_major_locator(ticker.MultipleLocator(5))

            plt.tight_layout()
            savefig(fig, model, f"per_seed_{seed:02d}_{variant}_contribution_payoff.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Across seeds: contribution & payoff per condition (mean ± SE)
#           One figure per model, 2 rows (contrib / payoff) × 5 cols (conditions)
#           global and local as separate lines
# ═══════════════════════════════════════════════════════════════════════════════

def plot_across_seeds(model: str):
    print(f"\n[{model.upper()}] Plot 2 — across-seed mean ± SE per condition")

    # Collect arrays: data[variant][cond][metric] = list of round-series (one per seed)
    from collections import defaultdict
    store = {v: {c: {"contributions": [], "payoffs": []} for c in CONDITIONS}
             for v in VARIANTS}

    for variant in VARIANTS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_latest_log(model, variant, seed, cond)
                if d is None:
                    continue
                store[variant][cond]["contributions"].append(round_series(d, "contributions"))
                store[variant][cond]["payoffs"].append(round_series(d, "payoffs"))

    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 7),
                             sharey="row", sharex=True)
    fig.suptitle(f"{model.upper()} — Mean ± SE across seeds — Contribution & Payoff\n"
                 f"(solid=global, dashed=local)", fontsize=11)

    for col, cond in enumerate(CONDITIONS):
        ax_c = axes[0, col]
        ax_p = axes[1, col]
        ax_c.set_title(COND_LABELS[cond], fontsize=9)

        for variant in VARIANTS:
            for metric, ax in [("contributions", ax_c), ("payoffs", ax_p)]:
                series_list = store[variant][cond][metric]
                if not series_list:
                    continue
                arr = np.array(series_list, dtype=float)   # (n_seeds, n_rounds)
                n_rounds = arr.shape[1]
                rounds = np.arange(1, n_rounds + 1)
                mean = np.nanmean(arr, axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
                ls   = VARIANT_STYLE[variant]
                col_v = VARIANT_COLORS[variant]
                ax.plot(rounds, mean, ls=ls, color=col_v, linewidth=2,
                        marker="o", markersize=3, label=variant.capitalize())
                ax.fill_between(rounds, mean - se, mean + se,
                                color=col_v, alpha=0.15)

        ax_c.set_ylim(0, 10.5)
        ax_c.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax_p.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax_p.set_xlabel("Round", fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_p.grid(True, alpha=0.3)
        if col == 0:
            ax_c.set_ylabel("Mean Contribution (0–10)", fontsize=8)
            ax_p.set_ylabel("Mean Payoff", fontsize=8)
        if col == len(CONDITIONS) - 1:
            ax_c.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    savefig(fig, model, "across_seeds_contribution_payoff_by_condition.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Across seeds: all conditions on one axes (global only, then local only)
#           Easier to compare conditions directly
# ═══════════════════════════════════════════════════════════════════════════════

def plot_across_seeds_conditions_overlay(model: str):
    print(f"\n[{model.upper()}] Plot 3 — all conditions overlaid, global panels")

    from collections import defaultdict
    store = {v: {c: {"contributions": [], "payoffs": []} for c in CONDITIONS}
             for v in VARIANTS}

    for variant in VARIANTS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_latest_log(model, variant, seed, cond)
                if d is None:
                    continue
                store[variant][cond]["contributions"].append(round_series(d, "contributions"))
                store[variant][cond]["payoffs"].append(round_series(d, "payoffs"))

    fig, axes = plt.subplots(2, 1, figsize=(8, 9))
    fig.suptitle(f"{model.upper()} — All Conditions — Mean ± SE across {len(SEEDS)} seeds", fontsize=12)

    titles = {
        (0, "contributions"): "Contribution — Global",
        (1, "payoffs"):       "Payoff — Global",
    }

    for (row, metric), title in titles.items():
        ax = axes[row]
        variant = "global"
        ax.set_title(title, fontsize=10)

        for cond in CONDITIONS:
            series_list = store[variant][cond][metric]
            if not series_list:
                continue
            arr    = np.array(series_list, dtype=float)
            rounds = np.arange(1, arr.shape[1] + 1)
            mean   = np.nanmean(arr, axis=0)
            se     = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2,
                    marker="o", markersize=3, label=COND_LABELS[cond])
            ax.fill_between(rounds, mean - se, mean + se,
                            color=COND_COLORS[cond], alpha=0.15)

        if row == 0:
            ax.set_ylim(0, 10.5)
            ax.set_ylabel("Mean Contribution (0–10)", fontsize=9)
        else:
            ax.set_ylabel("Mean Payoff", fontsize=9)
        ax.set_xlabel("Round", fontsize=9)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, model, "across_seeds_all_conditions_overlay.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Distribution of agent contributions per round (violin + jitter)
#           One figure per model, 5 subplots (one per condition).
#           Data pooled across all seeds and agents.
# ═══════════════════════════════════════════════════════════════════════════════

def plot_contribution_violins(model: str):
    print(f"\n[{model.upper()}] Plot 4 — contribution & payoff violin + jitter per round")

    variant  = "global"
    metrics  = [("contributions", "Contribution", (-0.5, 10.5)),
                ("payoffs",       "Payoff",        (None, None))]

    fig, axes = plt.subplots(len(CONDITIONS), 2,
                             figsize=(14, 4 * len(CONDITIONS)),
                             sharey="col")
    rng = np.random.default_rng(42)

    for row, cond in enumerate(CONDITIONS):
        # collect per-round data for both metrics
        round_data = {key: {} for key, _, _ in metrics}
        for seed in SEEDS:
            d = load_latest_log(model, variant, seed, cond)
            if d is None:
                continue
            for r in d["round_logs"]:
                rnd = r["round"]
                for key, _, _ in metrics:
                    vals = [float(v) for v in r[key].values()]
                    round_data[key].setdefault(rnd, []).extend(vals)

        for col, (key, ylabel, ylim) in enumerate(metrics):
            ax = axes[row, col]
            rv = round_data[key]
            if not rv:
                continue

            rounds = sorted(rv)
            data   = [rv[rnd] for rnd in rounds]

            parts = ax.violinplot(data, positions=rounds, widths=0.7,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(COND_COLORS[cond])
                pc.set_alpha(0.45)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

            for rnd, vals in zip(rounds, data):
                jitter = rng.uniform(-0.25, 0.25, size=len(vals))
                ax.scatter(np.array([rnd] * len(vals)) + jitter, vals,
                           color=COND_COLORS[cond], alpha=0.15, s=4, linewidths=0)

            if ylim[0] is not None:
                ax.set_ylim(*ylim)
            ax.set_ylabel(f"{COND_LABELS[cond]}\n{ylabel}", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.2, axis="y")
            if row == 0:
                ax.set_title(ylabel, fontsize=13)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("Round", fontsize=12)

    plt.tight_layout()
    savefig(fig, model, "contribution_violins.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5 — All models, all conditions: two separate figures
#           Contribution figure: 2×2 subplots (one per model)
#           Payoff figure:       2×2 subplots (one per model)
#           Each subplot: mean ± SE across 10 seeds, 5 condition lines
#           No titles or subplot labels; larger axis labels
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}

def plot_all_models_by_metric():
    print("\nPlot 5 — all-models overlay, separate contribution & payoff figures")

    # Build store for all models
    store = {m: {c: {"contributions": [], "payoffs": []} for c in CONDITIONS}
             for m in MODELS}

    for model in MODELS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_latest_log(model, "global", seed, cond)
                if d is None:
                    continue
                store[model][cond]["contributions"].append(round_series(d, "contributions"))
                store[model][cond]["payoffs"].append(round_series(d, "payoffs"))

    LABEL_SIZE  = 16
    TICK_SIZE   = 14
    LEGEND_SIZE = 16
    LW          = 2.0
    MS          = 3

    for metric, ylabel, ylim, fname in [
        ("contributions", "Mean Contribution (0–10)", (0, 10.5), "all_models_contribution.png"),
        ("payoffs",       "Mean Payoff",               (None, None), "all_models_payoff.png"),
    ]:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5),
                                 sharex=True, sharey=True)

        for idx, model in enumerate(MODELS):
            ax = axes[idx]

            for cond in CONDITIONS:
                series_list = store[model][cond][metric]
                if not series_list:
                    continue
                arr    = np.array(series_list, dtype=float)
                rounds = np.arange(1, arr.shape[1] + 1)
                mean   = np.nanmean(arr, axis=0)
                se     = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
                ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=LW,
                        marker="o", markersize=MS, label=COND_LABELS[cond])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=COND_COLORS[cond], alpha=0.15)

            if ylim[0] is not None:
                ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(axis="both", labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Round", fontsize=LABEL_SIZE)

            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

            ax.text(0.97, 0.97, MODEL_LABELS[model],
                    transform=ax.transAxes, fontsize=LABEL_SIZE,
                    ha="right", va="top")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS),
                   fontsize=LEGEND_SIZE, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        for ext in ("png", "pdf"):
            out_path = os.path.join(FIG_ROOT, fname.replace(".png", f".{ext}"))
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    for model in MODELS:
        plot_per_seed(model)
        plot_across_seeds(model)
        plot_across_seeds_conditions_overlay(model)
        plot_contribution_violins(model)
    plot_all_models_by_metric()
    print("\nAll done.")
