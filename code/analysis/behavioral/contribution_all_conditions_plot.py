"""
contribution_all_conditions_plot.py
-------------------------------------
Draws the paper's contribution-trajectory figure: mean contribution per
round over all 20 rounds, with all 5 experimental conditions overlaid as
separate lines, one panel per model. The same script (with different
--models/--variant/--out-dir arguments) produces every version of this
figure used in the paper -- the main 7B/8B result, the bigger-model
replication, the 13B and 70B tiers, and the community-size/group-size/MCPR
structural sweeps.

Colors/markers/legend labels/sizing are fixed -- this is meant to always
produce the same visual style, not a general-purpose plotting tool.

Inputs: raw simulation logs under {results_dir}/{model}/{variant}/seed{N}/
log_*.json for each requested model.
Outputs: {out_dir}/{out_name}.pdf and {out_dir}/{out_name}.png.
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as _stats

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/behavioral/) so the default results
# path still works if the release is moved or copied elsewhere. --results-dir
# (or the SNLS_RESULTS_DIR environment variable) can override this default.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.getenv("SNLS_RESULTS_DIR", f"{BASE}/results")

CONDITIONS = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
ROUNDS     = list(range(1, 21))
ENDOWMENT  = 10
ALL_SEEDS  = list(range(42, 53))

LABEL_SIZE = 14
TICK_SIZE  = 12

# Fixed style, matching MAIN_RESULTS/1_contribution_trajectories/all_conditions_local exactly.
COND_COLORS = {
    "FULL":          "#2a78d6",  # blue
    "NO_DISCUSSION": "#eb6834",  # orange
    "NO_SELECTION":  "#1baf7a",  # aqua
    "BASELINE":      "#eda100",  # yellow
    "PURE_BASELINE": "#e87ba4",  # magenta
}
COND_MARKERS = {
    "FULL":          "o",
    "NO_DISCUSSION": "s",
    "NO_SELECTION":  "^",
    "BASELINE":      "D",
    "PURE_BASELINE": "v",
}
MECHANISM_LEGEND_LABELS = {
    "PURE_BASELINE": r"$\emptyset$",
    "BASELINE":      "E",
    "NO_SELECTION":  "E + SL",
    "NO_DISCUSSION": "E + SS",
    "FULL":          "E + SS + SL",
}
ADDITIVE_ORDER = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]

# Fallback display names for known model keys -- --model-labels overrides these.
DEFAULT_DISPLAY_LABELS = {
    "gpt":         "GPT-4o-mini",
    "llama":       "Llama 3.1-8B",
    "mistral":     "Mistral-7B",
    "qwen":        "Qwen2.5-7B",
    "llama_13b":   "Llama 2 13B",
    "mistral_13b": "Mistral-13B",
    "qwen_14b":    "Qwen 2.5 14B",
    "llama_70b":   "Llama 3.1-70B",
    "qwen_72b":    "Qwen 2.5-72B",
    "gpt-5-mini":  "GPT-5-mini",
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_latest_log(results_dir, model, variant, seed, condition):
    # Finds and loads the log file for one (model, seed, condition)
    # combination. If more than one file matches, the last one found wins.
    pattern = os.path.join(results_dir, model, variant, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def build_contribution_store(results_dir, variant, models, seeds):
    """store[model][cond][seed] = [mean_contribution per round]"""
    # For every model/condition/seed, averages contribution across all
    # agents in each round, giving one trajectory (one value per round)
    # per seed. Only keeps a seed's trajectory if it has a value for every
    # round (an incomplete run is dropped rather than plotted with gaps).
    store = {m: {c: {} for c in CONDITIONS} for m in models}
    for model in models:
        for cond in CONDITIONS:
            for seed in seeds:
                d = load_latest_log(results_dir, model, variant, seed, cond)
                if d is None:
                    continue
                series = [np.mean(list(r["contributions"].values()))
                          for r in d["round_logs"]]
                if len(series) == len(ROUNDS):
                    store[model][cond][seed] = series
    return store


def mean_ci95(arr):
    """
    arr: one row per seed/run (run-clustered). Returns (mean, half-width) for a
    t-based 95% CI, df = n_seeds - 1 per round.
    """
    # Standard across-seed mean and 95% confidence interval, computed
    # separately for each round (each column of arr).
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    tcrit = _stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return mean, tcrit * se


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_all_conditions(store, models, display_labels, out_dir, out_name):
    """1 row x len(models) cols: all conditions overlaid per model."""
    # One panel per model. Within each panel, draws one line per condition
    # (mean contribution per round, with a shaded 95% confidence band),
    # then adds a single shared legend below the whole figure and saves
    # both a PDF and a PNG.
    rounds = np.array(ROUNDS)
    big_label  = LABEL_SIZE + 16
    big_tick   = TICK_SIZE + 14
    title_size = LABEL_SIZE + 10
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 5.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        for cond in CONDITIONS:
            seed_data = store[model][cond]
            if not seed_data:
                continue
            arr = np.array(list(seed_data.values()), dtype=float)
            mean, ci = mean_ci95(arr)
            ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2,
                    marker=COND_MARKERS[cond], markevery=2, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=MECHANISM_LEGEND_LABELS[cond])
            ax.fill_between(rounds, mean - ci, mean + ci,
                            color=COND_COLORS[cond], alpha=0.15)
        ax.axhline(ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.4)
        ax.set_title(display_labels.get(model, model), fontsize=title_size)
        ax.set_xlabel("Round", fontsize=big_label)
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=big_tick)
        ax.grid(True, alpha=0.2)
        if ax is axes[0]:
            ax.set_ylabel("Mean Contribution", fontsize=big_label)

    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_labels = [MECHANISM_LEGEND_LABELS[c] for c in ADDITIVE_ORDER
                      if MECHANISM_LEGEND_LABELS[c] in label_to_handle]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]
    fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=len(CONDITIONS),
               fontsize=big_label, frameon=False, markerscale=1.6,
               handlelength=2.4, handletextpad=0.6, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout(rect=[0, 0.09, 1, 1])

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{out_name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{out_name}.png/.pdf")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contribution trajectories, all conditions overlaid.")
    parser.add_argument("--variant", "-v", default="local",
                        help="Data variant (default: local). Accepts any subpath under "
                             "{results_dir}/{model}/, e.g. 'local/N16_G4' or "
                             "'local_groupsizevary/N12_G3_MCPR0.4'.")
    parser.add_argument("--models", "-m", nargs="+", required=True,
                        help="Model keys to include, e.g. gpt llama mistral qwen.")
    parser.add_argument("--model-labels", nargs="+", default=[],
                        metavar="KEY=LABEL",
                        help="Override display labels, e.g. --model-labels gpt='GPT-4o-mini'. "
                             "Unlisted known keys fall back to a built-in default; "
                             "unknown keys fall back to the raw key.")
    parser.add_argument("--seeds", "-s", nargs="+", type=int, default=None,
                        metavar="SEED", help="Explicit seed list. Overrides --n.")
    parser.add_argument("--n", type=int, default=10, metavar="N",
                        help="Number of seeds from the default list (default: 10).")
    parser.add_argument("--results-dir", default=RESULTS,
                        help="Results root (default: $SNLS_RESULTS_DIR or the main results/ dir).")
    parser.add_argument("--out-dir", "-o", required=True, help="Output directory.")
    parser.add_argument("--out-name", default="all_conditions",
                        help="Output filename stem (default: all_conditions).")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else ALL_SEEDS[:args.n]
    display_labels = dict(DEFAULT_DISPLAY_LABELS)
    for pair in args.model_labels:
        key, _, label = pair.partition("=")
        display_labels[key] = label

    print(f"Variant     : {args.variant}")
    print(f"Models      : {args.models}")
    print(f"Seeds       : {seeds}")
    print(f"Results dir : {args.results_dir}")
    print(f"Out         : {args.out_dir}/{args.out_name}")

    store = build_contribution_store(args.results_dir, args.variant, args.models, seeds)
    for model in args.models:
        for cond in CONDITIONS:
            print(f"  {display_labels.get(model, model):18s} / {cond:20s}: {len(store[model][cond])} seeds")

    plot_all_conditions(store, args.models, display_labels, args.out_dir, args.out_name)
