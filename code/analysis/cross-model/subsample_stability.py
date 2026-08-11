"""
subsample_stability.py
----------------------
Checks whether the main conclusions are robust to sample size by repeatedly
subsampling from the 10 available seeds without replacement.

Subsample sizes: 5, 6, 7, 8 seeds  (100 iterations each)

For each subsample the following metrics are computed from the last 5 rounds:

  full_mean    — mean contribution in FULL condition
  gap          — FULL mean − PURE_BASELINE mean  (primary treatment effect)
  ordering     — does the expected hierarchy hold?
                 FULL > NO_DISCUSSION > NO_SELECTION > BASELINE > PURE_BASELINE
  full_highest — is FULL the highest-contributing condition?

Summary statistics across iterations:
  - mean and SD of the gap estimate  (magnitude stability)
  - % iterations where gap > 0       (directional consistency)
  - % iterations where ordering holds (structural consistency)
  - coverage: does the full-sample CI contain the subsample mean?

Outputs:
  figures/2026-03-22/paper_stats/subsample_stability_summary.csv
  figures/2026-03-22/paper_stats/subsample_stability_gap_distributions.png/.pdf
  figures/2026-03-22/paper_stats/subsample_stability_curve.png/.pdf
"""

import argparse
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

RESULTS   = os.getenv("SNLS_RESULTS_DIR", "/data3/rasimura/social-norm-evo/results")
OUT_DIR   = "/data3/rasimura/social-norm-evo/figures/2026-03-22/paper_stats"

MODELS       = ["gpt", "llama", "mistral", "qwen"]
MODEL_LABELS = {"gpt":         "GPT-4o-mini",
                "llama":       "Llama 3.1-8B",
                "mistral":     "Mistral-7B",
                "qwen":        "Qwen2.5-7B",
                "llama_13b":   "Llama 2-13B",
                "mistral_13b": "Mistral Nemo 12B",
                "qwen_14b":    "Qwen2.5-14B",
                "llama_70b":   "Llama 3.1-70B",
                "qwen_72b":    "Qwen2.5-72B"}
VARIANT      = "global"
ALL_SEEDS    = list(range(42, 53))   # seeds 42–52 (11 seeds)

SUBSAMPLE_SIZES = [5, 6, 7, 8]
N_ITER          = 100

CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]

LAST_N = 5   # rounds used for steady-state

MODEL_COLORS = {
    "gpt":         "#1f77b4",
    "llama":       "#ff7f0e",
    "mistral":     "#2ca02c",
    "qwen":        "#d62728",
    "llama_13b":   "#9467bd",
    "mistral_13b": "#8c564b",
    "qwen_14b":    "#e377c2",
    "llama_70b":   "#17becf",
    "qwen_72b":    "#bcbd22",
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


def load_seed_means(model):
    """
    Returns seed_means[seed][condition] = mean contribution (last LAST_N rounds).
    Only includes seeds with all 5 conditions available.
    """
    seed_means = {}
    for seed in ALL_SEEDS:
        row = {}
        ok  = True
        for cond in CONDITIONS:
            d = load_log(model, seed, cond)
            if d is None:
                ok = False
                break
            contribs = []
            for r in d["round_logs"]:
                vals = list(r["contributions"].values())
                if vals:
                    contribs.append(float(np.mean(vals)))
            row[cond] = float(np.mean(contribs[-LAST_N:])) if contribs else np.nan
        if ok:
            seed_means[seed] = row
    return seed_means


# ─── Metrics per subsample ────────────────────────────────────────────────────

def compute_metrics(seed_means, subsample):
    """
    For a given list of seed indices, compute per-condition means and
    the key stability metrics.
    """
    cond_means = {}
    for cond in CONDITIONS:
        vals = [seed_means[s][cond] for s in subsample if not np.isnan(seed_means[s][cond])]
        cond_means[cond] = np.mean(vals) if vals else np.nan

    full_mean = cond_means["FULL"]
    pb_mean   = cond_means["PURE_BASELINE"]
    bl_mean   = cond_means["BASELINE"]
    gap       = full_mean - pb_mean

    # Core paper claims — these are what must hold for conclusions to replicate
    full_beats_pb       = full_mean > pb_mean          # primary treatment effect
    full_beats_baseline = full_mean > bl_mean          # FULL > BASELINE
    full_beats_both_pb_and_bl = full_beats_pb and full_beats_baseline

    # FULL in top 2 conditions (more lenient — acknowledges NO_DISCUSSION can compete)
    sorted_means = sorted(cond_means.values(), reverse=True)
    full_top2    = full_mean >= sorted_means[1] if len(sorted_means) >= 2 else False

    return {
        "full_mean":               full_mean,
        "pb_mean":                 pb_mean,
        "bl_mean":                 bl_mean,
        "gap":                     gap,
        "gap_vs_baseline":         full_mean - bl_mean,
        "full_beats_pb":           full_beats_pb,
        "full_beats_baseline":     full_beats_baseline,
        "full_beats_both":         full_beats_both_pb_and_bl,
        "full_top2":               full_top2,
        **{f"mean_{c}": cond_means[c] for c in CONDITIONS},
    }


# ─── Main subsampling loop ────────────────────────────────────────────────────

def run_subsampling(seed_means, rng):
    """
    For each subsample size and iteration, compute metrics.
    Returns a DataFrame with one row per (size, iteration).
    """
    seeds     = list(seed_means.keys())
    full_data = compute_metrics(seed_means, seeds)   # full-sample reference

    rows = []
    for n in SUBSAMPLE_SIZES:
        for i in range(N_ITER):
            subsample = rng.choice(seeds, size=n, replace=False).tolist()
            m         = compute_metrics(seed_means, subsample)
            rows.append({"n_seeds": n, "iteration": i, **m})

    return pd.DataFrame(rows), full_data


# ─── Summary statistics ───────────────────────────────────────────────────────

def summarise(df, full_data, model):
    """Per subsample size: mean, SD, directional consistency, ordering rate."""
    rows = []
    for n in SUBSAMPLE_SIZES:
        sub = df[df["n_seeds"] == n]
        rows.append({
            "model":                model,
            "n_seeds":              n,
            "gap_mean":             round(sub["gap"].mean(), 4),
            "gap_sd":               round(sub["gap"].std(), 4),
            "gap_full":             round(full_data["gap"], 4),
            "gap_positive_pct":     round(100 * (sub["gap"] > 0).mean(), 1),
            "beats_pb_pct":         round(100 * sub["full_beats_pb"].mean(), 1),
            "beats_baseline_pct":   round(100 * sub["full_beats_baseline"].mean(), 1),
            "beats_both_pct":       round(100 * sub["full_beats_both"].mean(), 1),
            "full_top2_pct":        round(100 * sub["full_top2"].mean(), 1),
            "full_mean_mean":       round(sub["full_mean"].mean(), 4),
            "full_mean_sd":         round(sub["full_mean"].std(), 4),
            "full_mean_full":       round(full_data["full_mean"], 4),
        })
    return pd.DataFrame(rows)


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_gap_distributions(all_results, all_full):
    """
    Violin plot of FULL − PB gap distribution per subsample size,
    one panel per model. Horizontal line = full-sample estimate.
    """
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5),
                             sharey=False)

    for ax, model in zip(axes, MODELS):
        df        = all_results[model]
        full_gap  = all_full[model]["gap"]
        color     = MODEL_COLORS[model]

        positions = range(len(SUBSAMPLE_SIZES))
        data      = [df[df["n_seeds"] == n]["gap"].values for n in SUBSAMPLE_SIZES]

        parts = ax.violinplot(data, positions=list(positions),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.axhline(full_gap, color=color, linestyle="--", linewidth=1.8,
                   label=f"Full-sample ({full_gap:.2f})")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)

        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(n) for n in SUBSAMPLE_SIZES], fontsize=11)
        ax.set_xlabel("Subsample size (seeds)", fontsize=11)
        ax.set_title(MODEL_LABELS.get(model, model), fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend(fontsize=9, loc="lower right")

        if ax == axes[0]:
            ax.set_ylabel("FULL − Pure Baseline\n(mean contribution, last 5 rounds)",
                          fontsize=11)

    fig.suptitle(f"Subsample Stability: FULL − PB Gap ({N_ITER} iterations per size)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"subsample_stability_gap_distributions.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: subsample_stability_gap_distributions.png/.pdf")


def plot_stability_curve(all_summaries):
    """
    Stability curve: x = subsample size, y = SD of gap estimate.
    Lower SD = more stable. One line per model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric, ylabel, title in [
        (axes[0], "gap_sd",          "SD of gap estimate",              "Precision of FULL − PB Gap"),
        (axes[1], "beats_both_pct",  "% iterations FULL > PB and BL",   "Core Claim Consistency (%)"),
    ]:
        for model in MODELS:
            sub = all_summaries[all_summaries["model"] == model]
            ax.plot(sub["n_seeds"].to_numpy(), sub[metric].to_numpy(),
                    marker="o", color=MODEL_COLORS[model],
                    linewidth=2, label=MODEL_LABELS.get(model, model))
        ax.set_xlabel("Subsample size (seeds)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(SUBSAMPLE_SIZES)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"subsample_stability_curve.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: subsample_stability_curve.png/.pdf")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subsample stability analysis for main conclusions."
    )
    parser.add_argument("--variant", "-v", default="global",
                        choices=["global", "local"],
                        help="Data variant (default: global).")
    parser.add_argument("--models", "-m", nargs="+",
                        default=["gpt", "llama", "mistral", "qwen"],
                        choices=["gpt", "llama", "mistral", "qwen",
                                 "llama_13b", "mistral_13b", "qwen_14b", "llama_70b", "qwen_72b"])
    parser.add_argument("--seeds", "-s", nargs="+", type=int, default=None,
                        metavar="SEED",
                        help="Explicit seed list to use. Overrides --n.")
    parser.add_argument("--n", type=int, default=len(ALL_SEEDS), metavar="N",
                        help=f"Use first N seeds from ALL_SEEDS (default: {len(ALL_SEEDS)}).")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="Iterations per subsample size (default: 100).")
    parser.add_argument("--rng-seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42).")
    parser.add_argument("--out-dir", "-o", default=None)
    args = parser.parse_args()

    VARIANT   = args.variant
    MODELS    = args.models
    N_ITER    = args.n_iter
    ALL_SEEDS = args.seeds if args.seeds is not None else ALL_SEEDS[:args.n]
    if args.out_dir:
        OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    rng = np.random.default_rng(args.rng_seed)

    print(f"Variant  : {VARIANT}")
    print(f"Models   : {MODELS}")
    print(f"Seeds    : {ALL_SEEDS}")
    print(f"Sizes    : {SUBSAMPLE_SIZES}  ×  {N_ITER} iterations each")
    print(f"RNG seed : {args.rng_seed}")

    all_results  = {}   # model → DataFrame of iteration results
    all_full     = {}   # model → full-sample metrics
    all_summaries_list = []

    for model in MODELS:
        print(f"\n── {MODEL_LABELS.get(model, model)} ──────────────────────────────────")

        print("  Loading seed means …")
        seed_means = load_seed_means(model)
        n_loaded   = len(seed_means)
        print(f"  {n_loaded}/{len(ALL_SEEDS)} seeds loaded with all conditions")

        if n_loaded < max(SUBSAMPLE_SIZES):
            print(f"  [WARN] Only {n_loaded} seeds — skipping subsample sizes > {n_loaded}")
            valid_sizes = [n for n in SUBSAMPLE_SIZES if n <= n_loaded]
        else:
            valid_sizes = SUBSAMPLE_SIZES

        if not valid_sizes:
            print(f"  [SKIP] Not enough seeds.")
            continue

        df_iter, full_data = run_subsampling(seed_means, rng)
        all_results[model] = df_iter
        all_full[model]    = full_data

        summ = summarise(df_iter, full_data, model)
        all_summaries_list.append(summ)

        print(f"\n  Full-sample: FULL mean = {full_data['full_mean']:.3f}, "
              f"gap = {full_data['gap']:.3f}")
        print(f"\n  {'n':>6}  {'gap_mean':>9}  {'gap_sd':>7}  "
              f"{'gap>0 %':>8}  {'> PB %':>7}  {'> BL %':>7}  {'> both %':>9}  {'top2 %':>7}")
        print("  " + "-" * 70)
        for _, row in summ.iterrows():
            print(f"  {int(row['n_seeds']):>6}  "
                  f"{row['gap_mean']:>9.3f}  "
                  f"{row['gap_sd']:>7.3f}  "
                  f"{row['gap_positive_pct']:>7.1f}%  "
                  f"{row['beats_pb_pct']:>6.1f}%  "
                  f"{row['beats_baseline_pct']:>6.1f}%  "
                  f"{row['beats_both_pct']:>8.1f}%  "
                  f"{row['full_top2_pct']:>6.1f}%")

    if not all_summaries_list:
        print("\nNo data loaded.")
    else:
        all_summaries = pd.concat(all_summaries_list, ignore_index=True)
        csv_path = os.path.join(OUT_DIR, "subsample_stability_summary.csv")
        all_summaries.to_csv(csv_path, index=False)
        print(f"\n  Saved: subsample_stability_summary.csv")

        print("\nGenerating plots …")
        plot_gap_distributions(all_results, all_full)
        plot_stability_curve(all_summaries)

    print(f"\nAll outputs → {OUT_DIR}")
