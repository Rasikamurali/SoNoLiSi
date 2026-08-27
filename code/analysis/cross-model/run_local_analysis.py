"""
run_local_analysis.py
---------------------
Runs the existing analysis scripts against the local variant results.
Patches VARIANT, SEEDS, FIG_ROOT/OUT_DIR in each module before calling
the core functions — no permanent changes to the original scripts.

Steps:
  1.  Completeness check — verify 10 seeds × 5 conditions per model
  2.  Perceptual consensus — SD convergence, via perception_consensus.py (subprocess, 2026-08-24)
  3.  Contribution trajectory — mean contribution over rounds (custom)
  5.  Per-seed behavioral plots — via temp_evals_behavior.plot_per_seed
  7.  Perception-action gap — via perception_action_gap_plot.py (subprocess, 2026-08-24)
  8.  OLS + LME stats — behavior_quantified (subprocess)
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
from scipy import stats as _stats

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS   = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT  = "/data3/rasimura/social-norm-evo/figures/local"
CODE_DIR  = os.path.dirname(os.path.abspath(__file__))

# Reorg (2026-08-11): see run_13b_analysis.py for why this block exists.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ANALYSIS_DIR, "model_specs.py")):
    ANALYSIS_DIR = os.path.dirname(ANALYSIS_DIR)
SCRIPT_SUBFOLDER = {"behavior_quantified.py": "behavioral"}

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
    """Superseded 2026-08-24 by perception_consensus.py (item a of the
    perception/alignment consolidation) -- run as a subprocess since that
    script is tier-parametrized via --tier rather than monkeypatched
    globals. Writes the SD-convergence plot + early/late and continuous-
    slope tables straight to figures/MAIN_RESULTS/3_dn_in_convergence/."""
    print("=" * 60)
    print("PERCEPTUAL CONSENSUS (perception_consensus.py) — 7b tier")
    print("=" * 60)

    script_path = os.path.join(ANALYSIS_DIR, "perception", "perception_consensus.py")
    result = subprocess.run([sys.executable, script_path, "--tier", "7b"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] perception_consensus.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
    else:
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        for l in lines[-10:]:
            print(f"    {l}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTRIBUTION TRAJECTORIES — custom plot, same style as existing figures
# ═══════════════════════════════════════════════════════════════════════════════

# Colorblind-safe categorical palette (dataviz skill reference palette, slots 1-5,
# validated: worst adjacent CVD ΔE 9.1, worst adjacent normal-vision ΔE 19.6).
COND_COLORS = {
    "FULL":          "#2a78d6",  # blue
    "NO_DISCUSSION": "#eb6834",  # orange
    "NO_SELECTION":  "#1baf7a",  # aqua
    "BASELINE":      "#eda100",  # yellow
    "PURE_BASELINE": "#e87ba4",  # magenta
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


def mean_ci95(arr):
    """
    arr: one row per seed/run (run-clustered). Returns (mean, half-width) for a
    t-based 95% CI, df = n_seeds - 1 per round (n_seeds ~10, so notably wider
    than a z=1.96 approximation).
    """
    n = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    tcrit = _stats.t.ppf(0.975, np.maximum(n - 1, 1))
    return mean, tcrit * se


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


def run_contribution_all_conditions_plot():
    print("=" * 60)
    print("CONTRIBUTION TRAJECTORIES (all conditions overlaid) — local variant")
    print("=" * 60)
    out_dir = "/data3/rasimura/social-norm-evo/figures/MAIN_RESULTS/1_contribution_trajectories"
    script_path = os.path.join(ANALYSIS_DIR, "behavioral", "contribution_all_conditions_plot.py")
    cmd = ([sys.executable, script_path,
            "--variant", VARIANT, "--models"] + MODELS
           + ["--seeds"] + [str(s) for s in SEEDS]
           + ["--results-dir", RESULTS,
              "--out-dir", out_dir,
              "--out-name", "all_conditions_local",
              "--model-labels"] + [f"{k}={v}" for k, v in MODEL_LABELS.items()])
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
    print("CONTRIBUTION TRAJECTORIES — local variant")
    print("=" * 60)
    store = build_contribution_store()

    for model in MODELS:
        for cond in PLOT_CONDITIONS:
            n = len(store[model][cond])
            print(f"  {MODEL_LABELS[model]:15s} / {cond:20s}: {n} seeds")

    print()
    plot_contribution_trajectories(store)
    run_contribution_all_conditions_plot()




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

    teb.FIG_ROOT = FIG_ROOT
    teb.SEEDS    = SEEDS
    teb.MODELS   = MODELS

    for model in MODELS:
        print(f"  {MODEL_LABELS[model]}")
        teb.plot_per_seed(model)
    print(f"  Saved to {FIG_ROOT}/{{model}}/behavioral_analysis/")



# ═══════════════════════════════════════════════════════════════════════════════
# 7. PERCEPTION-ACTION GAP (temp_eval_alignment2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_plots():
    """Superseded 2026-08-24 by perception_action_gap_plot.py (item b of the
    perception/alignment consolidation) -- run as a subprocess since that
    script is tier-parametrized via --tier rather than monkeypatched
    globals. Publishes straight to figures/MAIN_RESULTS/5_perception_action_gap_evolution/."""
    print("=" * 60)
    print("PERCEPTION-ACTION GAP (perception_action_gap_plot.py) — 7b tier")
    print("=" * 60)

    script_path = os.path.join(ANALYSIS_DIR, "alignment", "perception_action_gap_plot.py")
    result = subprocess.run([sys.executable, script_path, "--tier", "7b"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] perception_action_gap_plot.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
    else:
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        for l in lines[-10:]:
            print(f"    {l}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. STATS — behavior_quantified via subprocess
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats():
    print("=" * 60)
    print("OLS + LME STATS — local variant")
    print("=" * 60)

    stats_out = os.path.join(FIG_ROOT, "paper_stats")
    os.makedirs(stats_out, exist_ok=True)

    # behavior_quantified.py's Q1/Q2 tables are the paper's canonical
    # pairwise-OLS+Wald result -- write directly to MAIN_RESULTS instead of
    # a scratch dir that then needs manually copying over.
    main_pairwise_out = "/data3/rasimura/social-norm-evo/figures/MAIN_RESULTS/2_pairwise_ols_wald/main"
    os.makedirs(main_pairwise_out, exist_ok=True)

    scripts = [
        (
            "behavior_quantified.py",
            ["--variant", "local", "--out-dir", main_pairwise_out],
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


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SUPPORTING RESULTS — mixed-effects level/slope tests (behavioral_statistical)
# ═══════════════════════════════════════════════════════════════════════════════

def run_behavioral_statistical():
    print("=" * 60)
    print("BEHAVIORAL STATISTICAL TESTS (mixed effects, level/slope) — 7B/8B")
    print("=" * 60)
    for _p in [ANALYSIS_DIR] + [
        os.path.join(ANALYSIS_DIR, d) for d in os.listdir(ANALYSIS_DIR)
        if os.path.isdir(os.path.join(ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
    ]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import behavioral_statistical as bs
    bs.MODELS   = ["gpt", "llama", "mistral", "qwen"]
    bs.LABELS   = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
    bs.FAMILIES = ["GPT", "Llama", "Mistral", "Qwen"]
    bs.REF_FAM  = "Mistral"
    bs.RESULTS  = "/data3/rasimura/social-norm-evo/results"
    bs.VARIANT  = "local"
    bs.FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/SUPPORTING_MAIN_RESULTS/behavioral_mixed_effects_7b"
    bs.main()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_completeness()
    run_stability()
    run_contribution_plots()
    run_all_models_plots()
    run_per_seed_plots()
    run_alignment_plots()
    run_stats()
    run_behavioral_statistical()
    print(f"\nAll outputs → {FIG_ROOT}")
