"""
temp_evals_selection.py
=======================
Weight-based selection metrics comparing FULL vs NO_DISCUSSION conditions.

Metrics computed per (model, variant, condition, seed):
  1. Weight gap      ΔW_t = mean_incoming(coop,t) − mean_incoming(viol,t)  per round
  2. Final-round gap ΔW_T = gap at the last round
  3. CT-group slope  β from OLS: incoming_weight ~ round, per agent, grouped by CT
  4. Separation_t    = ΔW_t / SD(all incoming weights at round t)

CT groups (initial round-1 CT):
  Violators:    CT < 0.3
  Normal:       0.3 ≤ CT < 0.7
  Cooperators:  CT ≥ 0.7

Figures (saved to FIG_ROOT/{model}/selection_analysis/):
  sel_main_{model}.png
    Main figure: mean incoming weight per CT group over rounds.
    2 rows (FULL / NO_DISCUSSION) × 2 cols (global / local).
    Mean ± 95% CI ribbon across seeds.

  sel_weight_gap_{model}.png
    Weight gap ΔW_t over rounds, same 2×2 layout.

  sel_separation_{model}.png
    Separation (normalised gap) over rounds, same layout.

  sel_final_gap_{model}.png
    Secondary figure: boxplot + bar of final-round ΔW by condition.
    One figure per model, global vs local side by side.

  sel_slopes_{model}.png
    CT-group slope distributions: boxplot of β per group per condition.
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as _stats

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANTS   = ["global"]
SEEDS      = list(range(43, 53))
CONDITIONS = ["FULL", "NO_DISCUSSION"]          # main comparison
ALL_CONDITIONS = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION"]  # for secondary figs

N_BOOT = 1000
np.random.seed(0)

CT_THRESHOLDS = {"Violators": (0.0, 0.3), "Normal": (0.3, 0.7), "Cooperators": (0.7, 1.01)}
CT_ORDER      = ["Violators", "Normal", "Cooperators"]
CT_COLORS     = {"Violators": "#d62728", "Normal": "#ff7f0e", "Cooperators": "#2ca02c"}

COND_COLORS = {"FULL": "#2ca02c", "NO_DISCUSSION": "#9467bd",
               "BASELINE": "#1f77b4", "NO_SELECTION": "#8c564b"}
VARIANT_LS  = {"global": "-", "local": "--"}


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_latest_log(model, variant, seed, condition):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    cond_best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            cond_best[d["condition"]] = d
        except Exception:
            continue
    return cond_best.get(condition)


def savefig(fig, model, fname):
    out_dir = os.path.join(FIG_ROOT, model, "selection_analysis")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── CT grouping ──────────────────────────────────────────────────────────────

def ct_group(ct):
    for name, (lo, hi) in CT_THRESHOLDS.items():
        if lo <= ct < hi:
            return name
    return None


def initial_ct_map(data):
    """Return {agent_id: ct_group} using round-1 cooperation_tendency."""
    r1 = next((r for r in data["round_logs"] if r["round"] == 1), None)
    if r1 is None:
        return {}
    return {
        a["id"]: ct_group(a["cooperation_tendency"])
        for a in r1["agent_states"]
    }


# ─── Per-run extraction ───────────────────────────────────────────────────────

def incoming_weight(network_weights, agent_id):
    return sum(e["weight"] for e in network_weights if int(e["v"]) == agent_id)


def extract_run(data):
    """
    Returns:
      groups      : {agent_id: ct_group_name}
      iw_series   : {agent_id: [float per round]}   incoming weight
      n_rounds    : int
    """
    groups    = initial_ct_map(data)
    agent_ids = sorted(groups)
    n_rounds  = len(data["round_logs"])

    iw_series = {a: [] for a in agent_ids}
    for rlog in data["round_logs"]:
        nw = rlog.get("network_weights", [])
        for a in agent_ids:
            iw_series[a].append(incoming_weight(nw, a))

    return groups, iw_series, n_rounds


# ─── Metric computation ───────────────────────────────────────────────────────

def compute_group_means(groups, iw_series, n_rounds):
    """
    Returns {ct_group: np.array(n_rounds)} of mean incoming weight per round.
    """
    result = {}
    for grp in CT_ORDER:
        agents = [a for a, g in groups.items() if g == grp]
        if not agents:
            result[grp] = np.full(n_rounds, np.nan)
        else:
            arr = np.array([iw_series[a] for a in agents])   # (n_agents, n_rounds)
            result[grp] = arr.mean(axis=0)
    return result


def compute_weight_gap(group_means):
    """ΔW_t = mean_coop_t − mean_viol_t  (array over rounds)."""
    c = group_means.get("Cooperators", np.array([np.nan]))
    v = group_means.get("Violators",   np.array([np.nan]))
    return c - v


def compute_separation(groups, iw_series, n_rounds):
    """Separation_t = ΔW_t / SD(all incoming weights at round t)."""
    all_agents = list(iw_series)
    all_arr    = np.array([iw_series[a] for a in all_agents])  # (n_agents, n_rounds)
    sd_t       = all_arr.std(axis=0)
    sd_t[sd_t < 1e-9] = np.nan                                 # avoid division by zero
    grp_means  = compute_group_means(groups, iw_series, n_rounds)
    gap        = compute_weight_gap(grp_means)
    return gap / sd_t


def compute_slopes(groups, iw_series, n_rounds):
    """
    For each agent, regress incoming_weight ~ round (OLS slope β).
    Returns {ct_group: [β per agent]}.
    """
    rounds = np.arange(1, n_rounds + 1)
    slopes = defaultdict(list)
    for a, grp in groups.items():
        if grp is None:
            continue
        y = np.array(iw_series[a], dtype=float)
        if np.isnan(y).all() or y.std() < 1e-12:
            continue
        slope, *_ = np.polyfit(rounds, y, 1)
        slopes[grp].append(slope)
    return dict(slopes)


# ─── Aggregation across seeds ─────────────────────────────────────────────────

def collect_seed_series(model, variant, condition):
    """
    Returns a list of dicts, one per seed that has data:
      {seed, group_means: {grp: array}, gap: array,
       separation: array, slopes: {grp: [β]}, n_rounds}
    """
    runs = []
    for seed in SEEDS:
        d = load_latest_log(model, variant, seed, condition)
        if d is None:
            continue
        groups, iw_series, n_rounds = extract_run(d)
        if not groups:
            continue
        gm  = compute_group_means(groups, iw_series, n_rounds)
        runs.append({
            "seed":        seed,
            "group_means": gm,
            "gap":         compute_weight_gap(gm),
            "separation":  compute_separation(groups, iw_series, n_rounds),
            "slopes":      compute_slopes(groups, iw_series, n_rounds),
            "n_rounds":    n_rounds,
        })
    return runs


def mean_ci(series_list, n_rounds, n_boot=N_BOOT):
    """
    Given a list of 1-D arrays (one per seed), compute mean and 95% bootstrap CI
    at each round. Returns (mean, ci_lo, ci_hi), each shape (n_rounds,).
    """
    if not series_list:
        nan = np.full(n_rounds, np.nan)
        return nan, nan, nan
    max_len = max(len(s) for s in series_list)
    arr = np.full((len(series_list), max_len), np.nan)
    for i, s in enumerate(series_list):
        arr[i, :len(s)] = s
    mean = np.nanmean(arr, axis=0)

    boot = np.full((n_boot, max_len), np.nan)
    idx  = np.arange(len(series_list))
    for b in range(n_boot):
        chosen = arr[np.random.choice(idx, len(idx), replace=True)]
        boot[b] = np.nanmean(chosen, axis=0)
    ci_lo = np.nanpercentile(boot, 2.5,  axis=0)
    ci_hi = np.nanpercentile(boot, 97.5, axis=0)
    return mean, ci_lo, ci_hi


# ─── Main figure ──────────────────────────────────────────────────────────────
# Mean incoming weight per CT group over rounds.
# 2 rows (conditions) × 2 cols (variants). Mean + 95% CI ribbon across seeds.

def plot_main(model):
    print(f"  [main] {model}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

    variant = "global"
    for col, cond in enumerate(CONDITIONS):
        ax   = axes[col]
        runs = collect_seed_series(model, variant, cond)

        n_rounds = runs[0]["n_rounds"] if runs else 20
        rounds   = np.arange(1, n_rounds + 1)

        for grp in CT_ORDER:
            series = [r["group_means"][grp] for r in runs
                      if not np.isnan(r["group_means"][grp]).all()]
            if not series:
                continue
            m, lo, hi = mean_ci(series, n_rounds)
            col_c = CT_COLORS[grp]
            ax.plot(rounds, m, color=col_c, lw=2.5, label=grp)
            ax.fill_between(rounds, lo, hi, color=col_c, alpha=0.15)

        ax.axhline(0, color="grey", lw=0.7, linestyle=":")
        ax.set_xlim(0.5, n_rounds + 0.5)
        ax.grid(True, alpha=0.2)
        ax.set_title(cond.replace("_", " "), fontsize=10)
        ax.set_xlabel("Round", fontsize=9)
        if col == 0:
            ax.set_ylabel("Mean incoming weight", fontsize=9)
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{model.upper()} — Mean incoming weight by CT group (FULL vs NO DISCUSSION)\n"
        f"Mean ± 95% CI across seeds  |  Red=Violators  Orange=Normal  Green=Cooperators",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "sel_main.png")


# ─── Weight gap figure ────────────────────────────────────────────────────────
# ΔW_t = mean_coop − mean_viol over rounds. Same 2×2 layout.

def plot_weight_gap(model):
    print(f"  [weight gap] {model}")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    variant   = "global"
    colors    = {"FULL": "#1f77b4", "NO_DISCUSSION": "#ff7f0e"}
    n_rounds  = 20
    for cond in CONDITIONS:
        runs = collect_seed_series(model, variant, cond)
        if not runs:
            continue
        n_rounds = runs[0]["n_rounds"]
        rounds   = np.arange(1, n_rounds + 1)
        gaps     = [r["gap"] for r in runs]
        m, lo, hi = mean_ci(gaps, n_rounds)
        color = colors[cond]
        ax.plot(rounds, m, color=color, lw=2.5,
                label=cond.replace("_", " "))
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.2)

    ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax.set_xlim(0.5, n_rounds + 0.5)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("ΔW (coop − viol)", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    savefig(fig, model, "sel_weight_gap.png")


# ─── Separation figure ────────────────────────────────────────────────────────

def plot_separation(model):
    print(f"  [separation] {model}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

    variant = "global"
    for col, cond in enumerate(CONDITIONS):
        ax   = axes[col]
        runs = collect_seed_series(model, variant, cond)
        if not runs:
            continue

        n_rounds = runs[0]["n_rounds"]
        rounds   = np.arange(1, n_rounds + 1)

        seps   = [r["separation"] for r in runs]
        m, lo, hi = mean_ci(seps, n_rounds)

        ax.plot(rounds, m, color=COND_COLORS[cond], lw=2.5)
        ax.fill_between(rounds, lo, hi, color=COND_COLORS[cond], alpha=0.2)
        ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
        ax.set_xlim(0.5, n_rounds + 0.5)
        ax.grid(True, alpha=0.2)
        ax.set_title(cond.replace("_", " "), fontsize=10)
        ax.set_xlabel("Round", fontsize=9)
        if col == 0:
            ax.set_ylabel("Separation (ΔW / SD)", fontsize=9)

    fig.suptitle(
        f"{model.upper()} — Normalised separation = ΔW / SD(all weights)\n"
        f"Mean ± 95% CI across seeds",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "sel_separation.png")


# ─── Secondary figure: final-round gap by condition ──────────────────────────
# Boxplot + mean bar of ΔW_T across seeds, all conditions, global vs local.

def plot_final_gap(model):
    print(f"  [final gap] {model}")
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    variant = "global"
    x     = np.arange(len(ALL_CONDITIONS))
    final_gaps = []
    for cond in ALL_CONDITIONS:
        runs = collect_seed_series(model, variant, cond)
        final_gaps.append([float(r["gap"][-1]) for r in runs
                           if not np.isnan(r["gap"][-1])])

    bp = ax.boxplot(
        final_gaps,
        positions=x,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
    )
    for patch, cond in zip(bp["boxes"], ALL_CONDITIONS):
        patch.set_facecolor(COND_COLORS.get(cond, "grey"))
        patch.set_alpha(0.6)

    for xi, vals in zip(x, final_gaps):
        if vals:
            ax.scatter(xi, np.mean(vals), color="black", s=40, zorder=5, marker="D")

    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in ALL_CONDITIONS], fontsize=8)
    ax.set_ylabel("Final-round ΔW (coop − viol)", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"{model.upper()} — Final-round weight gap by condition\n"
        f"Box = seed distribution  |  Diamond = mean",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "sel_final_gap.png")


# ─── Slopes figure ────────────────────────────────────────────────────────────
# Boxplot of OLS slope β per CT group per condition.

def plot_slopes(model):
    print(f"  [slopes] {model}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    variant = "global"
    for col, cond in enumerate(CONDITIONS):
        ax   = axes[col]
        runs = collect_seed_series(model, variant, cond)

        pooled = defaultdict(list)
        for r in runs:
            for grp, betas in r["slopes"].items():
                pooled[grp].extend(betas)

        x     = np.arange(len(CT_ORDER))
        data  = [pooled.get(grp, []) for grp in CT_ORDER]
        colors = [CT_COLORS[grp] for grp in CT_ORDER]

        bp = ax.boxplot(
            data, positions=x, widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
        )
        for patch, col_c in zip(bp["boxes"], colors):
            patch.set_facecolor(col_c)
            patch.set_alpha(0.65)

        for xi, vals in zip(x, data):
            if vals:
                ax.scatter(xi, np.mean(vals), color="black", s=40, zorder=5, marker="D")

        ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(CT_ORDER, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(cond.replace("_", " "), fontsize=10)
        ax.set_xlabel("CT group", fontsize=9)
        if col == 0:
            ax.set_ylabel("OLS slope β (weight ~ round)", fontsize=9)

    fig.suptitle(
        f"{model.upper()} — OLS slope of incoming weight over rounds by CT group\n"
        f"Positive β = weight growing  |  Diamond = mean across agents × seeds",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "sel_slopes.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    for model in MODELS:
        print(f"\n=== {model} ===")
        plot_main(model)
        plot_weight_gap(model)
        plot_separation(model)
        plot_final_gap(model)
        plot_slopes(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
