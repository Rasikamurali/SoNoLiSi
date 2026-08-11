"""
run_13b_analysis.py
--------------------
Runs the standard analysis pipeline for the ~13B open-source models
(Llama 2 13B, Mistral Nemo 12B, Qwen 2.5 14B) using the local variant results.

Mirrors run_local_analysis.py — patches module-level globals in the shared
analysis scripts rather than duplicating logic.

Output: figures/local/13b_models/
"""

import subprocess
import sys
import os
import glob
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/local/13b_models"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reorg (2026-08-11): this file now lives in code/analysis/cross-model/, but
# the modules it imports (stability_analysis, temp_eval_alignment2,
# temp_eval_network, temp_evals_behavior) are split across cross-model/,
# alignment/, selection/, and behavioral/. ANALYSIS_DIR anchors to
# code/analysis/ regardless of nesting depth; SCRIPT_SUBFOLDER records where
# each subprocess-invoked script (behavior_quantified.py,
# perception_quantified.py) now lives, for the os.path.join(CODE_DIR, script)
# calls below that used to assume everything was co-located.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ANALYSIS_DIR, "sobel_mediation.py")):
    ANALYSIS_DIR = os.path.dirname(ANALYSIS_DIR)
SCRIPT_SUBFOLDER = {"behavior_quantified.py": "behavioral", "perception_quantified.py": "perception"}

MODELS   = ["llama_13b", "mistral_13b", "qwen_14b"]

# Used as directory names by analysis scripts (must be path-safe, no spaces)
MODEL_LABELS = {
    "llama_13b":   "llama_13b",
    "mistral_13b": "mistral_13b",
    "qwen_14b":    "qwen_14b",
}

# Human-readable names for plot titles in custom plots
DISPLAY_LABELS = {
    "llama_13b":   "Llama 2 13B",
    "mistral_13b": "Mistral Nemo 12B",
    "qwen_14b":    "Qwen 2.5 14B",
}

# Colors for the new models
MODEL_COLORS_13B = {
    "llama_13b":   "#9467bd",
    "mistral_13b": "#8c564b",
    "qwen_14b":    "#e377c2",
}
VARIANT  = "local"
SEEDS    = list(range(42, 52))   # seeds 42–51 (10 seeds)

CONDITIONS    = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
PLOT_CONDITIONS = CONDITIONS
LAST_N        = 5
ROUNDS        = list(range(1, 21))
ENDOWMENT     = 10

COND_COLORS = {
    "PURE_BASELINE":  "#aaaaaa",
    "BASELINE":       "#6baed6",
    "NO_SELECTION":   "#fd8d3c",
    "NO_DISCUSSION":  "#74c476",
    "FULL":           "#e6550d",
}
COND_LABELS = {
    "PURE_BASELINE":  "Pure Baseline",
    "BASELINE":       "Baseline",
    "NO_SELECTION":   "No Selection",
    "NO_DISCUSSION":  "No Discussion",
    "FULL":           "Full",
}
MODEL_COLORS = MODEL_COLORS_13B

LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 11

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(os.path.join(FIG_ROOT, "cross_model"), exist_ok=True)
for m in MODELS:
    os.makedirs(os.path.join(FIG_ROOT, m, "behavioral_analysis"), exist_ok=True)
    os.makedirs(os.path.join(FIG_ROOT, m, "selection_analysis"),  exist_ok=True)
    os.makedirs(os.path.join(FIG_ROOT, m, "alignment_analysis"),  exist_ok=True)
    os.makedirs(os.path.join(FIG_ROOT, m, "perception_analysis"), exist_ok=True)

for _p in [ANALYSIS_DIR] + [
    os.path.join(ANALYSIS_DIR, d) for d in os.listdir(ANALYSIS_DIR)
    if os.path.isdir(os.path.join(ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─── Shared loader ────────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COMPLETENESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_completeness():
    print("=" * 60)
    print("COMPLETENESS CHECK — 13B models / local")
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
            status = "✓" if len(found) == len(SEEDS) else f"✗ {len(found)}/{len(SEEDS)}"
            msg = f"    {cond:20s}: {status}"
            if missing:
                msg += f"  missing: {missing}"
                all_ok = False
            print(msg)
    print()
    if all_ok:
        print(f"  All models have {len(SEEDS)} seeds for all conditions.")
    else:
        print("  WARNING: some seeds are missing.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability():
    print("=" * 60)
    print("STABILITY ANALYSIS — 13B models")
    print("=" * 60)

    import stability_analysis as sa
    sa.VARIANT     = VARIANT
    sa.SEEDS       = SEEDS
    sa.FIG_ROOT    = FIG_ROOT
    sa.MODELS      = MODELS
    sa.MODEL_LABELS = MODEL_LABELS

    store = sa.build_store()
    for model in MODELS:
        for cond in ["FULL", "BASELINE"]:
            n = len(store[model].get(cond, {}))
            print(f"  {MODEL_LABELS[model]} / {cond}: {n} seeds")

    sa.plot_stability(store)
    print(f"  Saved: stability_analysis.png/.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTRIBUTION TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_contribution_store():
    store = {m: {c: {} for c in PLOT_CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                series = [np.mean(list(r["contributions"].values()))
                          for r in d["round_logs"]]
                if len(series) == len(ROUNDS):
                    store[model][cond][seed] = series
    return store


def plot_contribution_trajectories(store):
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
                ax.set_ylabel(DISPLAY_LABELS[model], fontsize=LABEL_SIZE)
            if row == len(MODELS) - 1:
                ax.set_xlabel("Round", fontsize=TICK_SIZE)
    fig.suptitle("Mean Contribution Over Rounds — 13B Models (Local)", fontsize=LABEL_SIZE + 2, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_ROOT, "cross_model", "contribution_trajectories.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: contribution_trajectories.png/.pdf")


def plot_all_models_by_metric(store):
    def _round_series(data, key):
        return [np.mean(list(r[key].values())) for r in data["round_logs"]]

    metric_store = {m: {c: {"contributions": [], "payoffs": []} for c in PLOT_CONDITIONS}
                    for m in MODELS}
    for model in MODELS:
        for seed in SEEDS:
            for cond in PLOT_CONDITIONS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                metric_store[model][cond]["contributions"].append(_round_series(d, "contributions"))
                metric_store[model][cond]["payoffs"].append(_round_series(d, "payoffs"))

    for metric, ylabel, ylim, fname in [
        ("contributions", "Mean Contribution (0–10)", (0, 10.5),   "all_models_contribution.png"),
        ("payoffs",       "Mean Payoff",               (None, None), "all_models_payoff.png"),
    ]:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 5),
                                 sharex=True, sharey=True)
        for idx, model in enumerate(MODELS):
            ax = axes[idx]
            for cond in PLOT_CONDITIONS:
                series_list = metric_store[model][cond][metric]
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
            if ylim[0] is not None:
                ax.set_ylim(*ylim)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Round", fontsize=LABEL_SIZE)
            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            ax.text(0.97, 0.97, DISPLAY_LABELS[model],
                    transform=ax.transAxes, fontsize=LABEL_SIZE, ha="right", va="top")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(PLOT_CONDITIONS),
                   fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.04))
        plt.tight_layout(rect=[0, 0.07, 1, 1])
        for ext in ("png", "pdf"):
            out = os.path.join(FIG_ROOT, fname.replace(".png", f".{ext}"))
            fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def run_contribution_plots():
    print("=" * 60)
    print("CONTRIBUTION TRAJECTORIES — 13B models")
    print("=" * 60)
    store = build_contribution_store()
    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            print(f"  {MODEL_LABELS[model]:18s} / {cond:20s}: {len(store[model][cond])} seeds")
    print()
    plot_contribution_trajectories(store)
    plot_all_models_by_metric(store)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-SEED BEHAVIORAL PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_per_seed_plots():
    print("=" * 60)
    print("PER-SEED BEHAVIORAL PLOTS — 13B models")
    print("=" * 60)

    import temp_evals_behavior as teb
    teb.VARIANTS     = [VARIANT]
    teb.FIG_ROOT     = FIG_ROOT
    teb.SEEDS        = SEEDS
    teb.MODELS       = MODELS
    teb.MODEL_LABELS = MODEL_LABELS

    for model in MODELS:
        print(f"  {DISPLAY_LABELS[model]}")
        teb.plot_per_seed(model)
    print(f"  Saved to {FIG_ROOT}/{{model}}/behavioral_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NETWORK DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_network_plots():
    print("=" * 60)
    print("NETWORK DEVELOPMENT — 13B models")
    print("=" * 60)

    import temp_eval_network as ten
    ten.VARIANT       = VARIANT
    ten.FIG_ROOT      = FIG_ROOT
    ten.SEEDS         = SEEDS
    ten.MODELS        = MODELS
    ten.MODEL_LABELS  = MODEL_LABELS
    ten.MODEL_COLORS  = MODEL_COLORS_13B

    store = ten.build_store()
    for model in MODELS:
        n = len(store[model].get("FULL", {}))
        print(f"  {MODEL_LABELS[model]} / FULL: {n} seeds")

    ten.plot_per_model(store)
    ten.plot_across_models(store)
    ten.plot_across_models_horizontal(store)
    print(f"  Saved to {FIG_ROOT}/cross_model/ and per-model selection_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PERCEPTION-ACTION GAP
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_plots():
    print("=" * 60)
    print("PERCEPTION-ACTION GAP — 13B models")
    print("=" * 60)

    import temp_eval_alignment2 as tea
    tea.VARIANT       = VARIANT
    tea.FIG_ROOT      = FIG_ROOT
    tea.SEEDS         = SEEDS
    tea.MODELS        = MODELS
    tea.MODEL_LABELS  = MODEL_LABELS
    tea.MODEL_COLORS  = MODEL_COLORS_13B

    store = tea.build_store()
    for model in MODELS:
        n = len(store[model].get("FULL", {}))
        print(f"  {MODEL_LABELS[model]} / FULL: {n} seeds")

    tea.plot_per_model(store)
    tea.plot_across_models(store)
    print(f"  Saved to {FIG_ROOT}/cross_model/ and per-model alignment_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. OLS + LME STATS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats():
    print("=" * 60)
    print("OLS + LME STATS — 13B models")
    print("=" * 60)

    stats_out = os.path.join(FIG_ROOT, "paper_stats")
    os.makedirs(stats_out, exist_ok=True)

    seeds_str = [str(s) for s in SEEDS]

    for script, extra in [
        ("behavior_quantified.py",
         ["--variant", VARIANT, "--models"] + MODELS + ["--seeds"] + seeds_str + ["--out-dir", stats_out]),
        ("perception_quantified.py",
         ["--variant", VARIANT, "--models"] + MODELS + ["--seeds"] + seeds_str + ["--fig-root", FIG_ROOT]),
    ]:
        script_path = os.path.join(ANALYSIS_DIR, SCRIPT_SUBFOLDER.get(script, ""), script)
        cmd = [sys.executable, script_path] + extra
        print(f"\n  Running: {script}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [WARN] {script} exited with code {result.returncode}")
            if result.stderr:
                print(result.stderr[-800:])
        else:
            for line in [l for l in result.stdout.splitlines() if l.strip()][-5:]:
                print(f"    {line}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_completeness()
    run_stability()
    run_contribution_plots()
    run_per_seed_plots()
    run_network_plots()
    run_alignment_plots()
    run_stats()
    print(f"\nAll outputs → {FIG_ROOT}")
