"""
temp_evals_behavior.py
-----------------------
Per-model behavioral plots (contribution & payoff over rounds), local
variant only.

  plot_per_seed(model)    — one figure per seed, all 5 conditions overlaid,
                             2 rows (contribution / payoff). This is the only
                             function reused by the run_*_analysis.py
                             orchestrators (run_local/70b/13b_analysis.py),
                             which import this module and patch FIG_ROOT/
                             SEEDS/MODELS before calling it.
  plot_across_seeds(model) — one figure per model, cross-seed mean ± SE per
                             condition. Not currently called by any
                             orchestrator; kept as a standalone QA plot
                             (see __main__).

Output: figures/<FIG_ROOT>/<model>/behavioral_analysis/
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

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_latest_log(model: str, seed: int, condition: str):
    """Return the latest log dict for model/seed/condition (local variant), or None."""
    pattern = os.path.join(RESULTS, model, "local", f"seed{seed}", "log_*.json")
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
# Plot 1 — Per seed: contribution & payoff over rounds, all conditions overlaid
#           One figure per model per seed, 2 rows (contrib / payoff)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_per_seed(model: str):
    print(f"\n[{model.upper()}] Plot 1 — per-seed, all conditions overlaid")
    for seed in SEEDS:
        cond_data = {}
        for cond in CONDITIONS:
            d = load_latest_log(model, seed, cond)
            if d is not None:
                cond_data[cond] = d

        if not cond_data:
            continue

        fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(
            f"{model.upper()} — seed {seed}\n"
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
        savefig(fig, model, f"per_seed_{seed:02d}_contribution_payoff.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Across seeds: contribution & payoff per condition (mean ± SE)
#           One figure per model, 2 rows (contrib / payoff) × 5 cols (conditions)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_across_seeds(model: str):
    print(f"\n[{model.upper()}] Plot 2 — across-seed mean ± SE per condition")

    store = {c: {"contributions": [], "payoffs": []} for c in CONDITIONS}
    for seed in SEEDS:
        for cond in CONDITIONS:
            d = load_latest_log(model, seed, cond)
            if d is None:
                continue
            store[cond]["contributions"].append(round_series(d, "contributions"))
            store[cond]["payoffs"].append(round_series(d, "payoffs"))

    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 7),
                             sharey="row", sharex=True)
    fig.suptitle(f"{model.upper()} — Mean ± SE across seeds — Contribution & Payoff",
                 fontsize=11)

    for col, cond in enumerate(CONDITIONS):
        ax_c = axes[0, col]
        ax_p = axes[1, col]
        ax_c.set_title(COND_LABELS[cond], fontsize=9)

        for metric, ax in [("contributions", ax_c), ("payoffs", ax_p)]:
            series_list = store[cond][metric]
            if not series_list:
                continue
            arr = np.array(series_list, dtype=float)   # (n_seeds, n_rounds)
            rounds = np.arange(1, arr.shape[1] + 1)
            mean = np.nanmean(arr, axis=0)
            se   = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2,
                    marker="o", markersize=3)
            ax.fill_between(rounds, mean - se, mean + se,
                            color=COND_COLORS[cond], alpha=0.15)

        ax_c.set_ylim(0, 10.5)
        ax_c.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax_p.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax_p.set_xlabel("Round", fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_p.grid(True, alpha=0.3)
        if col == 0:
            ax_c.set_ylabel("Mean Contribution (0–10)", fontsize=8)
            ax_p.set_ylabel("Mean Payoff", fontsize=8)

    plt.tight_layout()
    savefig(fig, model, "across_seeds_contribution_payoff_by_condition.png")


if __name__ == "__main__":
    for model in MODELS:
        plot_per_seed(model)
        plot_across_seeds(model)
    print("\nAll done.")
