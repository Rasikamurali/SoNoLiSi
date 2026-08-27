"""
run_70b_analysis.py
--------------------
Runs the standard analysis pipeline for the 70B models
(Llama 3.1-70B, Qwen 2.5-72B) — local variant, seeds 42–51.

Results live in code/results/ (not the main results/ folder);
RESULTS is patched in each analysis module accordingly.

Output: figures/local/70b_models/
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
from scipy import stats as _stats

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/code/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/local/70b_models"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reorg (2026-08-11): see run_13b_analysis.py for why this block exists.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ANALYSIS_DIR, "model_specs.py")):
    ANALYSIS_DIR = os.path.dirname(ANALYSIS_DIR)
SCRIPT_SUBFOLDER = {"behavior_quantified.py": "behavioral"}

MODELS = ["llama_70b", "qwen_72b"]

# Used as directory names by patched analysis scripts (must be path-safe)
MODEL_LABELS = {
    "llama_70b": "llama_70b",
    "qwen_72b":  "qwen_72b",
}

# Human-readable names for custom plots
DISPLAY_LABELS = {
    "llama_70b": "Llama 3.1-70B",
    "qwen_72b":  "Qwen 2.5-72B",
}

MODEL_COLORS_70B = {
    "llama_70b": "#17becf",
    "qwen_72b":  "#bcbd22",
}

VARIANT = "local"
SEEDS   = list(range(42, 52))

CONDITIONS      = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
PLOT_CONDITIONS = CONDITIONS
LAST_N          = 5
ROUNDS          = list(range(1, 21))
ENDOWMENT       = 10

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

LABEL_SIZE  = 14
TICK_SIZE   = 12
LEGEND_SIZE = 11

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(os.path.join(FIG_ROOT, "cross_model"), exist_ok=True)
for m in MODELS:
    for sub in ("behavioral_analysis", "selection_analysis",
                "alignment_analysis", "perception_analysis"):
        os.makedirs(os.path.join(FIG_ROOT, m, sub), exist_ok=True)

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
    print("COMPLETENESS CHECK — 70B models / local")
    print("=" * 60)
    all_ok = True
    for model in MODELS:
        print(f"\n  {DISPLAY_LABELS[model]}")
        for cond in CONDITIONS:
            found   = [s for s in SEEDS if load_latest_log(model, s, cond) is not None]
            missing = [s for s in SEEDS if s not in found]
            status  = "✓" if len(found) == len(SEEDS) else f"✗ {len(found)}/{len(SEEDS)}"
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
    """Superseded 2026-08-24 by perception_consensus.py --tier 70b (item a of
    the perception/alignment consolidation)."""
    print("=" * 60)
    print("PERCEPTUAL CONSENSUS (perception_consensus.py) — 70b tier")
    print("=" * 60)
    script_path = os.path.join(ANALYSIS_DIR, "perception", "perception_consensus.py")
    result = subprocess.run([sys.executable, script_path, "--tier", "70b"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] perception_consensus.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
    else:
        for l in [l for l in result.stdout.splitlines() if l.strip()][-10:]:
            print(f"    {l}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTRIBUTION TRAJECTORIES + ALL-MODELS PLOTS
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


def mean_ci95(arr):
    """
    arr: one row per seed/run (run-clustered). Returns (mean, half-width) for a
    t-based 95% CI, df = n_seeds - 1 per round.
    """
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    tcrit = _stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return mean, tcrit * se


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
                arr = np.array(list(seed_data.values()), dtype=float)
                mean, ci = mean_ci95(arr)
                ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2)
                ax.fill_between(rounds, mean - ci, mean + ci,
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
    fig.suptitle("Mean Contribution Over Rounds — 70B Models (Local)",
                 fontsize=LABEL_SIZE + 2, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_ROOT, "cross_model", "contribution_trajectories.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: contribution_trajectories.png/.pdf")


def plot_all_models_by_metric(store):
    def _round_series(d, key):
        return [np.mean(list(r[key].values())) for r in d["round_logs"]]

    metric_store = {m: {c: {"contributions": [], "payoffs": []} for c in PLOT_CONDITIONS}
                    for m in MODELS}
    for model in MODELS:
        for seed in SEEDS:
            for cond in PLOT_CONDITIONS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                metric_store[model][cond]["contributions"].append(
                    _round_series(d, "contributions"))
                metric_store[model][cond]["payoffs"].append(
                    _round_series(d, "payoffs"))

    for metric, ylabel, ylim, fname in [
        ("contributions", "Mean Contribution (0–10)", (0, 10.5),    "all_models_contribution.png"),
        ("payoffs",       "Mean Payoff",               (None, None), "all_models_payoff.png"),
    ]:
        fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5),
                                 sharex=True, sharey=True)
        for idx, model in enumerate(MODELS):
            ax = axes[idx]
            for cond in PLOT_CONDITIONS:
                sl = metric_store[model][cond][metric]
                if not sl:
                    continue
                arr    = np.array(sl, dtype=float)
                rounds = np.arange(1, arr.shape[1] + 1)
                mean, ci = mean_ci95(arr)
                ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=2,
                        marker="o", markersize=3, label=COND_LABELS[cond])
                ax.fill_between(rounds, mean - ci, mean + ci,
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
            fig.savefig(os.path.join(FIG_ROOT, fname.replace(".png", f".{ext}")),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


def run_contribution_all_conditions_plot():
    print("=" * 60)
    print("CONTRIBUTION TRAJECTORIES (all conditions overlaid) — 70B models")
    print("=" * 60)
    out_dir = "/data3/rasimura/social-norm-evo/figures/SUPPLEMENTARY_RESULTS/4_70b_tier/1_contribution_trajectories"
    script_path = os.path.join(ANALYSIS_DIR, "behavioral", "contribution_all_conditions_plot.py")
    cmd = ([sys.executable, script_path,
            "--variant", VARIANT, "--models"] + MODELS
           + ["--seeds"] + [str(s) for s in SEEDS]
           + ["--results-dir", RESULTS,
              "--out-dir", out_dir,
              "--out-name", "all_conditions_70b",
              "--model-labels"] + [f"{k}={v}" for k, v in DISPLAY_LABELS.items()])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] contribution_all_conditions_plot.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1500:])
    else:
        for line in [l for l in result.stdout.splitlines() if l.strip()]:
            print(f"    {line}")


def run_contribution_plots():
    print("=" * 60)
    print("CONTRIBUTION TRAJECTORIES — 70B models")
    print("=" * 60)
    store = build_contribution_store()
    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            print(f"  {DISPLAY_LABELS[model]:16s} / {cond:20s}: {len(store[model][cond])} seeds")
    print()
    plot_contribution_trajectories(store)
    plot_all_models_by_metric(store)
    run_contribution_all_conditions_plot()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-SEED BEHAVIORAL PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_per_seed_plots():
    print("=" * 60)
    print("PER-SEED BEHAVIORAL PLOTS — 70B models")
    print("=" * 60)
    import temp_evals_behavior as teb
    teb.RESULTS      = RESULTS
    teb.FIG_ROOT     = FIG_ROOT
    teb.SEEDS        = SEEDS
    teb.MODELS       = MODELS
    for model in MODELS:
        print(f"  {DISPLAY_LABELS[model]}")
        teb.plot_per_seed(model)
    print(f"  Saved to {FIG_ROOT}/{{model}}/behavioral_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NETWORK DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_network_plots():
    print("=" * 60)
    print("NETWORK DEVELOPMENT — 70B models")
    print("=" * 60)
    import temp_eval_network as ten
    ten.RESULTS      = RESULTS
    ten.VARIANT      = VARIANT
    ten.FIG_ROOT     = FIG_ROOT
    ten.SEEDS        = SEEDS
    ten.MODELS       = MODELS
    ten.MODEL_LABELS = MODEL_LABELS
    ten.MODEL_COLORS = MODEL_COLORS_70B
    store = ten.build_store()
    for model in MODELS:
        print(f"  {DISPLAY_LABELS[model]} / FULL: {len(store[model].get('FULL', {}))} seeds")
    ten.plot_per_model(store)
    ten.plot_across_models(store)
    ten.plot_across_models_horizontal(store)
    print(f"  Saved to {FIG_ROOT}/cross_model/ and per-model selection_analysis/")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PERCEPTION-ACTION GAP
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_plots():
    """Superseded 2026-08-24 by perception_action_gap_plot.py --tier 70b
    (item b of the perception/alignment consolidation)."""
    print("=" * 60)
    print("PERCEPTION-ACTION GAP (perception_action_gap_plot.py) — 70b tier")
    print("=" * 60)
    script_path = os.path.join(ANALYSIS_DIR, "alignment", "perception_action_gap_plot.py")
    result = subprocess.run([sys.executable, script_path, "--tier", "70b"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] perception_action_gap_plot.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
    else:
        for l in [l for l in result.stdout.splitlines() if l.strip()][-10:]:
            print(f"    {l}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. OLS + LME STATS
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats():
    print("=" * 60)
    print("OLS + LME STATS — 70B models")
    print("=" * 60)
    stats_out = os.path.join(FIG_ROOT, "paper_stats")
    os.makedirs(stats_out, exist_ok=True)
    seeds_str = [str(s) for s in SEEDS]

    # behavior_quantified.py's Q1/Q2 tables are this tier's canonical
    # pairwise-OLS+Wald result -- write directly to SUPPLEMENTARY_RESULTS.
    pairwise_out = "/data3/rasimura/social-norm-evo/figures/SUPPLEMENTARY_RESULTS/4_70b_tier/2_pairwise_ols_wald"
    os.makedirs(pairwise_out, exist_ok=True)

    # behavior_quantified reads from RESULTS via its own load_latest_log —
    # pass --results-dir so it uses code/results
    for script, extra in [
        ("behavior_quantified.py",
         ["--variant", VARIANT, "--models"] + MODELS
         + ["--seeds"] + seeds_str + ["--out-dir", pairwise_out]),
    ]:
        script_path = os.path.join(ANALYSIS_DIR, SCRIPT_SUBFOLDER.get(script, ""), script)
        cmd = [sys.executable, script_path] + extra
        print(f"\n  Running: {script}")
        # Inject RESULTS path via env var so the script's load_latest_log finds code/results
        env = os.environ.copy()
        env["SNLS_RESULTS_DIR"] = RESULTS
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"  [WARN] {script} exited with code {result.returncode}")
            if result.stderr:
                print(result.stderr[-600:])
        else:
            for line in [l for l in result.stdout.splitlines() if l.strip()][-5:]:
                print(f"    {line}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SUBSAMPLE STABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_subsample_stability():
    print("=" * 60)
    print("SUBSAMPLE STABILITY — 70B models")
    print("=" * 60)
    stats_out = os.path.join(FIG_ROOT, "paper_stats")
    os.makedirs(stats_out, exist_ok=True)
    seeds_str = [str(s) for s in SEEDS]
    cmd = (
        [sys.executable, os.path.join(CODE_DIR, "subsample_stability.py"),
         "--variant", VARIANT,
         "--models"] + MODELS
        + ["--seeds"] + seeds_str
        + ["--out-dir", stats_out]
    )
    env = os.environ.copy()
    env["SNLS_RESULTS_DIR"] = RESULTS
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  [WARN] subsample_stability.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-600:])
    else:
        for line in [l for l in result.stdout.splitlines() if l.strip()][-10:]:
            print(f"  {line}")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SUPPORTING RESULTS — mixed-effects level/slope tests (behavioral_statistical.py)
# ═══════════════════════════════════════════════════════════════════════════════

def run_behavioral_statistical():
    print("=" * 60)
    print("BEHAVIORAL STATISTICAL TESTS (mixed effects, level/slope) — 70B/72B")
    print("=" * 60)
    import behavioral_statistical as bs
    bs.MODELS   = ["llama_70b", "qwen_72b"]
    bs.LABELS   = {"llama_70b": "Llama-70B", "qwen_72b": "Qwen-72B"}
    bs.FAMILIES = ["Llama-70B", "Qwen-72B"]
    bs.REF_FAM  = "Llama-70B"
    bs.RESULTS  = "/data3/rasimura/social-norm-evo/code/results"
    bs.VARIANT  = "local"
    bs.FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/SUPPLEMENTARY_RESULTS/4_70b_tier/behavioral_mixed_effects"
    bs.main()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_completeness()
    run_stability()
    run_contribution_plots()
    run_per_seed_plots()
    run_network_plots()
    run_alignment_plots()
    run_stats()
    run_subsample_stability()
    run_behavioral_statistical()
    print(f"\nAll outputs → {FIG_ROOT}")
