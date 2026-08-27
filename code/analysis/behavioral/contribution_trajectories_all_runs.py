"""
contribution_trajectories_all_runs.py
--------------------------------------
Contribution trajectories for the main-paper "local" dataset, llama/mistral/
qwen only: one figure, 1x3 subplots (one per model). Within each subplot,
one line per condition — mean +/- 95% CI (t-distribution) across that
condition's seeds.

Seeds come from TWO locations, merged:
  - 10 canonical seeds (43-52):   results/{model}/local/seed{s}/log_*.json
  - 40 additional seeds (53-92):  code/results/{model}/local/additional_runs/seed{s}/log_*.json
50 seeds/condition total once both locations are fully populated. As of
this run, llama and mistral have all 40 additional seeds; qwen's
additional_runs only has an (empty) seed53 so far — the per-model n in each
legend/title reports the actual count loaded, not an assumed 50.

Output: figures/2026-03-22/additional_runs_results/01_contribution_trajectories/
"""

import os
import json
import glob
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "/data3/rasimura/social-norm-evo"
OUT_DIR = f"{BASE}/figures/2026-03-22/additional_runs_results/01_contribution_trajectories"

MODELS     = ["llama", "mistral", "qwen"]
MODEL_TEX  = {"llama": "Llama-7B", "mistral": "Mistral-7B", "qwen": "Qwen-7B"}
VARIANT    = "local"
SEEDS      = list(range(43, 93))   # 43-52 canonical + 53-92 additional = 50
# Where to look for a given seed's log files, tried in order until one has data.
SEED_SEARCH_DIRS = [
    lambda model, seed: os.path.join(BASE, "results", model, VARIANT, f"seed{seed}"),
    lambda model, seed: os.path.join(BASE, "code", "results", model, VARIANT,
                                     "additional_runs", f"seed{seed}"),
]
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_COLORS = {
    "PURE_BASELINE": "#8172B2",
    "BASELINE":      "#55A868",
    "NO_SELECTION":  "#DD8452",
    "NO_DISCUSSION": "#C44E52",
    "FULL":          "#4C72B0",
}
COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline", "BASELINE": "Baseline",
    "NO_SELECTION": "No Selection", "NO_DISCUSSION": "No Discussion",
    "FULL": "Full",
}
ROUNDS    = list(range(1, 21))
ENDOWMENT = 10


def load_latest_log(model, seed, condition):
    for seed_dir_fn in SEED_SEARCH_DIRS:
        pattern = os.path.join(seed_dir_fn(model, seed), "log_*.json")
        best = {}
        for p in sorted(glob.glob(pattern)):
            try:
                d = json.load(open(p))
                best[d["condition"]] = d
            except Exception:
                continue
        if condition in best:
            return best[condition]
    return None


def build_store():
    """store[model][condition][seed] = [mean contribution per round]"""
    store = {m: {c: {} for c in CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                series = []
                for r in d["round_logs"]:
                    vals = list(r["contributions"].values())
                    series.append(float(np.mean(vals)) if vals else np.nan)
                if len(series) == len(ROUNDS):
                    store[model][cond][seed] = series
    return store


def mean_ci95(arr):
    """
    arr: one row per seed/run.
    Returns (mean, half-width) for a t-based 95% CI, df = n_seeds - 1 per round.
    """
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    tcrit = scipy_stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return mean, tcrit * se


def plot_by_condition(store):
    """One figure, 1x3 subplots (one per model). Within each subplot, one line
    per condition: mean contribution +/- 95% CI across that condition's 10 seeds."""
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 5), sharey=True)

    for ax, model in zip(axes, MODELS):
        for cond in CONDITIONS:
            seed_data = store[model][cond]
            if not seed_data:
                continue
            arr = np.array(list(seed_data.values()), dtype=float)
            mean, ci = mean_ci95(arr)
            color = COND_COLORS[cond]
            ax.plot(rounds, mean, color=color, linewidth=2,
                    label=f"{COND_LABELS[cond]} (n={arr.shape[0]})")
            ax.fill_between(rounds, mean - ci, mean + ci, color=color, alpha=0.15)

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.grid(True, alpha=0.2)
        ax.set_title(MODEL_TEX[model], fontsize=13)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Mean contribution", fontsize=12)
    fig.suptitle("Contribution trajectories by condition "
                "(mean $\\pm$ 95% CI across seeds per condition; n in legend)",
                fontsize=14, y=1.03)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"contribution_trajectories_by_condition.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


def main():
    print("Loading contribution trajectories …")
    store = build_store()
    for model in MODELS:
        n = sum(len(store[model][c]) for c in CONDITIONS)
        print(f"  {model}: {n} runs loaded (expect up to {len(CONDITIONS) * len(SEEDS)})")
        for cond in CONDITIONS:
            print(f"    {cond}: {len(store[model][cond])} seeds")

    plot_by_condition(store)
    print("\nDone.")


if __name__ == "__main__":
    main()
