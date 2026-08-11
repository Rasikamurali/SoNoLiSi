"""
run_local_analysis.py
---------------------
Runs the existing analysis scripts against the local variant results.
Patches VARIANT, SEEDS, FIG_ROOT/OUT_DIR in each module before calling
the core functions — no permanent changes to the original scripts.

Steps:
  1.  Completeness check — verify 10 seeds × 5 conditions per model
  2.  stability_analysis — SD convergence plots
  3.  Contribution trajectory — mean contribution over rounds (custom)
  4.  Cooperation tendency — CT over rounds (custom)
  5.  Per-seed behavioral plots — via temp_evals_behavior.plot_per_seed
  6.  Network development — via temp_eval_network
  7.  Perception-action gap — via temp_eval_alignment2
  8.  OLS + LME stats — behavior_quantified and perception_quantified (subprocess)
"""

import subprocess
import sys
import os
import importlib
import glob
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS   = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT  = "/data3/rasimura/social-norm-evo/figures/local"
CODE_DIR  = os.path.dirname(os.path.abspath(__file__))

# Reorg (2026-08-11): see run_13b_analysis.py for why this block exists.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ANALYSIS_DIR, "sobel_mediation.py")):
    ANALYSIS_DIR = os.path.dirname(ANALYSIS_DIR)
SCRIPT_SUBFOLDER = {"behavior_quantified.py": "behavioral", "perception_quantified.py": "perception"}

MODELS     = ["gpt", "llama", "mistral", "qwen"]
MODEL_LABELS = {"gpt": "GPT-4o-mini", "llama": "Llama 3.1-8B",
                "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}
VARIANT    = "local"
SEEDS      = list(range(43, 53))
CONDITIONS = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(os.path.join(FIG_ROOT, "cross_model"), exist_ok=True)
os.makedirs(os.path.join(FIG_ROOT, "paper_stats"), exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COMPLETENESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

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


def check_completeness():
    print("=" * 60)
    print("COMPLETENESS CHECK — local results")
    print("=" * 60)
    all_ok = True
    for model in MODELS:
        print(f"\n  {MODEL_LABELS[model]}")
        for cond in CONDITIONS:
            found, missing = [], []
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is not None:
                    found.append(seed)
                else:
                    missing.append(seed)
            status = "✓" if len(found) == 10 else f"✗ {len(found)}/10"
            msg = f"    {cond:20s}: {status}"
            if missing:
                msg += f"  missing seeds: {missing}"
                all_ok = False
            print(msg)
    print()
    if all_ok:
        print("  All models have 10 seeds for all conditions.")
    else:
        print("  WARNING: some seeds are missing — check above.")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STABILITY ANALYSIS (patch and run)
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability():
    print("=" * 60)
    print("STABILITY ANALYSIS — local variant")
    print("=" * 60)

    for _p in [ANALYSIS_DIR] + [
        os.path.join(ANALYSIS_DIR, d) for d in os.listdir(ANALYSIS_DIR)
        if os.path.isdir(os.path.join(ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
    ]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import stability_analysis as sa

    # Patch module globals
    sa.VARIANT  = VARIANT
    sa.SEEDS    = SEEDS
    sa.FIG_ROOT = FIG_ROOT
    sa.MODELS   = MODELS

    store = sa.build_store()

    # Count loaded seeds
    for model in MODELS:
        for cond in ["FULL", "BASELINE"]:
            n = len(store[model].get(cond, {}))
            print(f"  {MODEL_LABELS[model]} / {cond}: {n} seeds loaded")

    sa.plot_stability(store)
    print(f"  Saved: {os.path.join(FIG_ROOT, 'cross_model', 'stability_analysis.png')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTRIBUTION TRAJECTORIES — custom plot, same style as existing figures
# ═══════════════════════════════════════════════════════════════════════════════

COND_COLORS = {
    "FULL":          "#1f77b4",
    "NO_DISCUSSION": "#ff7f0e",
    "NO_SELECTION":  "#2ca02c",
    "BASELINE":      "#9467bd",
    "PURE_BASELINE": "#7f7f7f",
}
COND_LABELS = {
    "FULL":          "Full",
    "NO_DISCUSSION": "No Discussion",
    "NO_SELECTION":  "No Selection",
    "BASELINE":      "Baseline",
    "PURE_BASELINE": "Pure Baseline",
}
PLOT_CONDITIONS = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
ROUNDS = list(range(1, 21))
LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 11
ENDOWMENT   = 10


def build_contribution_store():
    """store[model][cond][seed] = [mean_contribution per round]"""
    store = {m: {c: {} for c in PLOT_CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                series = []
                for rlog in d["round_logs"]:
                    contribs = list(rlog["contributions"].values())
                    series.append(np.mean(contribs) if contribs else np.nan)
                if len(series) == len(ROUNDS):
                    store[model][cond][seed] = series
    return store


def plot_contribution_trajectories(store):
    """
    4 rows (models) × 5 cols (conditions).
    Mean ± SE contribution over rounds across seeds.
    """
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(
        len(MODELS), len(PLOT_CONDITIONS),
        figsize=(4 * len(PLOT_CONDITIONS), 3.5 * len(MODELS)),
        sharex=True, sharey=True,
    )

    for row, model in enumerate(MODELS):
        for col, cond in enumerate(PLOT_CONDITIONS):
            ax = axes[row, col]
            seed_data = store[model][cond]

            if seed_data:
                arr  = np.array(list(seed_data.values()), dtype=float)
                mean = np.nanmean(arr, axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
                ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2)
                ax.fill_between(rounds, mean - se, mean + se,
                                color=COND_COLORS[cond], alpha=0.2)
                ax.text(0.97, 0.07, f"n={len(seed_data)}",
                        transform=ax.transAxes, ha="right", fontsize=9, color="gray")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")

            ax.axhline(ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.4)
            ax.set_ylim(0, ENDOWMENT + 0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE, fontweight="bold")
            if col == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=LABEL_SIZE)
            if row == len(MODELS) - 1:
                ax.set_xlabel("Round", fontsize=TICK_SIZE)

    fig.suptitle("Mean Contribution Over Rounds — Local Variant",
                 fontsize=LABEL_SIZE + 2, y=1.01)
    plt.tight_layout()

    out = os.path.join(FIG_ROOT, "cross_model", "contribution_trajectories_local.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: contribution_trajectories_local.png/.pdf")


def plot_contribution_all_conditions(store):
    """
    Single plot: all 5 conditions on same axes, one panel per model.
    Easier to compare conditions within a model.
    """
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 4.5), sharey=True)

    for ax, model in zip(axes, MODELS):
        for cond in PLOT_CONDITIONS:
            seed_data = store[model][cond]
            if not seed_data:
                continue
            arr  = np.array(list(seed_data.values()), dtype=float)
            mean = np.nanmean(arr, axis=0)
            se   = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2,
                    label=COND_LABELS[cond])
            ax.fill_between(rounds, mean - se, mean + se,
                            color=COND_COLORS[cond], alpha=0.15)

        ax.axhline(ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.4)
        ax.set_title(MODEL_LABELS[model], fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xlabel("Round", fontsize=TICK_SIZE)
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel("Mean Contribution", fontsize=LABEL_SIZE)
            ax.legend(fontsize=LEGEND_SIZE, loc="lower right")

    fig.suptitle("Contribution by Condition — Local Variant",
                 fontsize=LABEL_SIZE + 2, y=1.01)
    plt.tight_layout()

    out = os.path.join(FIG_ROOT, "cross_model", "all_conditions_local.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: all_conditions_local.png/.pdf")


def run_contribution_plots():
    print("=" * 60)
    print("CONTRIBUTION TRAJECTORIES — local variant")
    print("=" * 60)
    store = build_contribution_store()

    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            n = len(store[model][cond])
            print(f"  {MODEL_LABELS[model]:15s} / {cond:20s}: {n} seeds")

    print()
    plot_contribution_trajectories(store)
    plot_contribution_all_conditions(store)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COOPERATION TENDENCY — CT over rounds per condition
# ═══════════════════════════════════════════════════════════════════════════════

def build_ct_store():
    store = {m: {c: {} for c in PLOT_CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                series = []
                for rlog in d["round_logs"]:
                    states = rlog.get("agent_states", [])
                    cts    = [s["cooperation_tendency"] for s in states]
                    series.append(np.mean(cts) if cts else np.nan)
                if len(series) == len(ROUNDS):
                    store[model][cond][seed] = series
    return store


def plot_ct_trajectories(store):
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 4.5), sharey=True)

    for ax, model in zip(axes, MODELS):
        for cond in ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE"]:
            seed_data = store[model][cond]
            if not seed_data:
                continue
            arr  = np.array(list(seed_data.values()), dtype=float)
            mean = np.nanmean(arr, axis=0)
            se   = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2,
                    label=COND_LABELS[cond])
            ax.fill_between(rounds, mean - se, mean + se,
                            color=COND_COLORS[cond], alpha=0.15)

        ax.set_title(MODEL_LABELS[model], fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_xlabel("Round", fontsize=TICK_SIZE)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel("Mean Cooperation Tendency", fontsize=LABEL_SIZE)
            ax.legend(fontsize=LEGEND_SIZE, loc="lower right")

    fig.suptitle("Cooperation Tendency Over Rounds — Local Variant",
                 fontsize=LABEL_SIZE + 2, y=1.01)
    plt.tight_layout()

    out = os.path.join(FIG_ROOT, "cross_model", "ct_trajectories_local.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: ct_trajectories_local.png/.pdf")


def run_ct_plots():
    print("=" * 60)
    print("COOPERATION TENDENCY — local variant")
    print("=" * 60)
    store = build_ct_store()
    plot_ct_trajectories(store)


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. ALL-MODELS CONTRIBUTION + PAYOFF (replicates plot_all_models_by_metric)
# ═══════════════════════════════════════════════════════════════════════════════

def _round_series(data: dict, key: str):
    """Per-round mean of 'contributions' or 'payoffs'."""
    out = []
    for r in data["round_logs"]:
        vals = list(r[key].values())
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def plot_all_models_by_metric_local():
    store = {m: {c: {"contributions": [], "payoffs": []} for c in PLOT_CONDITIONS}
             for m in MODELS}

    for model in MODELS:
        for seed in SEEDS:
            for cond in PLOT_CONDITIONS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                store[model][cond]["contributions"].append(_round_series(d, "contributions"))
                store[model][cond]["payoffs"].append(_round_series(d, "payoffs"))

    LS, TS, LL = 16, 14, 16
    LW, MS     = 2.0, 3

    for metric, ylabel, ylim, fname in [
        ("contributions", "Mean Contribution (0–10)", (0, 10.5),   "all_models_contribution.png"),
        ("payoffs",       "Mean Payoff",               (None, None), "all_models_payoff.png"),
    ]:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5),
                                 sharex=True, sharey=True)
        for idx, model in enumerate(MODELS):
            ax = axes[idx]
            for cond in PLOT_CONDITIONS:
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
            ax.tick_params(axis="both", labelsize=TS)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Round", fontsize=LS)
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=LS)
            ax.text(0.97, 0.97, MODEL_LABELS[model],
                    transform=ax.transAxes, fontsize=LS, ha="right", va="top")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(PLOT_CONDITIONS),
                   fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(rect=[0, 0.07, 1, 1])

        for ext in ("png", "pdf"):
            out = os.path.join(FIG_ROOT, fname.replace(".png", f".{ext}"))
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {os.path.basename(out)}")
        plt.close(fig)


def run_all_models_plots():
    print("=" * 60)
    print("ALL-MODELS CONTRIBUTION + PAYOFF — local variant")
    print("=" * 60)
    plot_all_models_by_metric_local()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PER-SEED BEHAVIORAL PLOTS (temp_evals_behavior — plot_per_seed only)
#    across_seeds / violins / all_models all hardcode "global" internally;
#    the equivalent cross-seed plots come from run_contribution_plots() above.
# ═══════════════════════════════════════════════════════════════════════════════

def run_per_seed_plots():
    print("=" * 60)
    print("PER-SEED BEHAVIORAL PLOTS — local variant")
    print("=" * 60)

    import temp_evals_behavior as teb

    teb.VARIANTS = ["local"]
    teb.FIG_ROOT = FIG_ROOT
    teb.SEEDS    = SEEDS
    teb.MODELS   = MODELS

    for model in MODELS:
        print(f"  {MODEL_LABELS[model]}")
        teb.plot_per_seed(model)
    print(f"  Saved to {FIG_ROOT}/{{model}}/behavioral_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NETWORK DEVELOPMENT (temp_eval_network)
# ═══════════════════════════════════════════════════════════════════════════════

def run_network_plots():
    print("=" * 60)
    print("NETWORK DEVELOPMENT — local variant")
    print("=" * 60)

    import temp_eval_network as ten

    ten.VARIANT  = "local"
    ten.FIG_ROOT = FIG_ROOT
    ten.SEEDS    = SEEDS
    ten.MODELS   = MODELS

    store = ten.build_store()
    for model in MODELS:
        for cond in ["FULL", "NO_DISCUSSION"]:
            n = len(store[model].get(cond, {}))
            print(f"  {MODEL_LABELS[model]} / {cond}: {n} seeds")

    ten.plot_per_model(store)
    ten.plot_across_models(store)
    ten.plot_across_models_horizontal(store)
    print(f"  Saved to {FIG_ROOT}/cross_model/ and per-model selection_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PERCEPTION-ACTION GAP (temp_eval_alignment2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_plots():
    print("=" * 60)
    print("PERCEPTION-ACTION GAP — local variant")
    print("=" * 60)

    import temp_eval_alignment2 as tea

    tea.VARIANT  = "local"
    tea.FIG_ROOT = FIG_ROOT
    tea.SEEDS    = SEEDS
    tea.MODELS   = MODELS

    store = tea.build_store()
    for model in MODELS:
        for cond in ["FULL", "BASELINE"]:
            n = len(store[model].get(cond, {}))
            print(f"  {MODEL_LABELS[model]} / {cond}: {n} seeds")

    tea.plot_per_model(store)
    tea.plot_across_models(store)
    print(f"  Saved to {FIG_ROOT}/cross_model/ and per-model alignment_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. STATS — behavior_quantified + perception_quantified via subprocess
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats():
    print("=" * 60)
    print("OLS + LME STATS — local variant")
    print("=" * 60)

    stats_out = os.path.join(FIG_ROOT, "paper_stats")
    os.makedirs(stats_out, exist_ok=True)

    scripts = [
        (
            "behavior_quantified.py",
            ["--variant", "local", "--out-dir", stats_out],
        ),
        (
            "perception_quantified.py",
            ["--variant", "local", "--fig-root", FIG_ROOT],
        ),
    ]

    for script, extra_args in scripts:
        script_path = os.path.join(ANALYSIS_DIR, SCRIPT_SUBFOLDER.get(script, ""), script)
        cmd = [sys.executable, script_path] + extra_args
        print(f"\n  Running: {script} {' '.join(extra_args)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] {script} exited with code {result.returncode}")
            if result.stderr:
                print(result.stderr[-1000:])
        else:
            # Print last few lines of stdout as confirmation
            lines = [l for l in result.stdout.splitlines() if l.strip()]
            for l in lines[-6:]:
                print(f"    {l}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_completeness()
    run_stability()
    run_contribution_plots()
    run_all_models_plots()
    run_ct_plots()
    run_per_seed_plots()
    run_network_plots()
    run_alignment_plots()
    run_stats()
    print(f"\nAll outputs → {FIG_ROOT}")
