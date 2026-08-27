"""
perception_action_gap_plot.py
------------------------------
Item (b) of the perception/alignment consolidation (2026-08-24):
perception-action gap trajectories over 20 rounds (descriptive, not a
statistical test -- see gap_based_alignment.py for the regression). Supersedes
temp_eval_alignment2.py (archived to code/analysis/archive/alignment/), which
this file ports wholesale for the plotting logic; only tier dispatch,
output-path routing, and the new community/group-size tiers are new.

Gap metrics (per agent per round):
  IN_gap = injunctive_norm  - actual_contribution
  DN_gap = descriptive_norm - actual_contribution

Aggregation: mean across agents within each seed -> seed-level trajectory.
Conditions with perceptions: BASELINE, NO_SELECTION, NO_DISCUSSION, FULL.

Analysis 1 - Per model, per condition: seed-level trajectories (thin lines)
  + cross-seed mean (thick line). One figure per model.
Analysis 2 - Per condition, across models: cross-seed mean +/- SE per model,
  all models overlaid in one figure. This is the MAIN_RESULTS/SUPPLEMENTARY_
  RESULTS figure (perception_action_gap_across_models.png/.pdf).

Tiers (--tier flag, default 7b): 7b / s2 (family-based, RESULTS/VARIANT/
SEEDS/MODELS exactly as previously monkeypatched into temp_eval_alignment2.py
by run_local_analysis.py / run_s2_analysis.py) and community / group (new --
same COMMUNITY_SOURCES / GROUP_N12_SOURCES / GROUP_N16_SOURCES as
gap_based_alignment.py, one figure per (stratum, axis-value) setting, models
overlaid the same way the family tiers overlay model families).
"""

import os
import sys
import argparse
import json
import glob
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gap_based_alignment import (  # noqa: E402
    COMMUNITY_SOURCES, GROUP_N12_SOURCES, GROUP_N16_SOURCES, STRUCTURAL_SETTINGS,
)

BASE = "/data3/rasimura/social-norm-evo"
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
ROUNDS = list(range(1, 21))

COND_LABELS = {
    "BASELINE": "E", "NO_SELECTION": "E + SL",
    "NO_DISCUSSION": "E + SS", "FULL": "E + SS + SL",
}
GAP_LABELS = {"IN_gap": "NE − Contribution", "DN_gap": "EE − Contribution"}
GAP_COLORS = {"IN_gap": "#e377c2", "DN_gap": "#17becf"}
LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 20, 18, 20

_QUAL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728",
                "#bcbd22", "#8c564b", "#17becf"]
_QUAL_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

TIER_SPECS = {
    "7b": dict(
        results=f"{BASE}/results", variant="local", seeds=list(range(43, 53)),
        models=["gpt", "llama", "mistral", "qwen"],
        labels={"gpt": "GPT-4o-mini", "llama": "Llama-3.1-8b", "mistral": "Mistral-7B", "qwen": "Qwen-2.5-7B"},
        colors={"gpt": "#1f77b4", "llama": "#ff7f0e", "mistral": "#2ca02c", "qwen": "#9467bd"},
        markers={"gpt": "o", "llama": "s", "mistral": "^", "qwen": "D"},
        stage_root=f"{BASE}/figures/local",
        publish_dir=f"{BASE}/figures/MAIN_RESULTS/5_perception_action_gap_evolution",
    ),
    "s2": dict(
        results=f"{BASE}/code/results", variant="local", seeds=list(range(42, 52)),
        models=["gpt-5-mini", "llama_70b", "mistral_13b", "qwen_72b"],
        labels={"gpt-5-mini": "GPT-5-mini", "llama_70b": "Llama 3.1-70B",
                "mistral_13b": "Mistral-13B", "qwen_72b": "Qwen 2.5-72B"},
        colors={"gpt-5-mini": "#d62728", "llama_70b": "#bcbd22",
                "mistral_13b": "#8c564b", "qwen_72b": "#17becf"},
        markers={"gpt-5-mini": "o", "llama_70b": "s", "mistral_13b": "^", "qwen_72b": "D"},
        stage_root=f"{BASE}/figures/local/s2_models",
        publish_dir=f"{BASE}/figures/SUPPLEMENTARY_RESULTS/2_bigger_models/5_perception_action_gap_evolution",
    ),
    # Standalone 13B/14B and 70B/72B family groupings (distinct from "s2"
    # above) -- matches run_13b_analysis.py / run_70b_analysis.py's own
    # MODELS lists, kept so those orchestrators' perception-action-gap step
    # still has a tier to point at after temp_eval_alignment2.py is archived.
    "13b": dict(
        results=f"{BASE}/results", variant="local", seeds=list(range(42, 52)),
        models=["llama_13b", "mistral_13b", "qwen_14b"],
        labels={"llama_13b": "llama_13b", "mistral_13b": "mistral_13b", "qwen_14b": "qwen_14b"},
        colors={"llama_13b": "#9467bd", "mistral_13b": "#8c564b", "qwen_14b": "#e377c2"},
        markers={"llama_13b": "o", "mistral_13b": "s", "qwen_14b": "^"},
        stage_root=f"{BASE}/figures/local/13b_models",
        publish_dir=f"{BASE}/figures/SUPPLEMENTARY_RESULTS/3_13b_tier/5_perception_action_gap_evolution",
    ),
    "70b": dict(
        results=f"{BASE}/code/results", variant="local", seeds=list(range(42, 52)),
        models=["llama_70b", "qwen_72b"],
        labels={"llama_70b": "Llama 3.1-70B", "qwen_72b": "Qwen 2.5-72B"},
        colors={"llama_70b": "#17becf", "qwen_72b": "#bcbd22"},
        markers={"llama_70b": "o", "qwen_72b": "s"},
        stage_root=f"{BASE}/figures/local/70b_models",
        publish_dir=f"{BASE}/figures/SUPPLEMENTARY_RESULTS/4_70b_tier/5_perception_action_gap_evolution",
    ),
}


# ─── Data loading ───────────────────────────────────────────────────────────

def load_latest_log(base_dir, seed, condition):
    pattern = os.path.join(base_dir, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def compute_seed_gaps(base_dir, seed, condition):
    """round -> {"IN_gap": float, "DN_gap": float}, mean across agents that round."""
    d = load_latest_log(base_dir, seed, condition)
    if d is None:
        return {}
    result = {}
    for r in d["round_logs"]:
        rnd = r["round"]
        contributions = {int(k): float(v) for k, v in r["contributions"].items()}
        perceptions = {int(k): v for k, v in (r.get("perceptions") or {}).items()}
        in_gaps, dn_gaps = [], []
        for aid, perc in perceptions.items():
            if not perc:
                continue
            actual = contributions.get(aid)
            if actual is None:
                continue
            inj, desc = perc.get("injunctive_norm"), perc.get("descriptive_norm")
            if inj is not None:
                in_gaps.append(float(inj) - actual)
            if desc is not None:
                dn_gaps.append(float(desc) - actual)
        result[rnd] = {
            "IN_gap": float(np.mean(in_gaps)) if in_gaps else np.nan,
            "DN_gap": float(np.mean(dn_gaps)) if dn_gaps else np.nan,
        }
    return result


def build_store(model_dirs, seeds_by_model):
    """model_dirs: {model_key: base_dir}. store[model][cond][seed] = {round: gaps}."""
    store = {m: {c: {} for c in CONDITIONS} for m in model_dirs}
    for model, base_dir in model_dirs.items():
        for cond in CONDITIONS:
            for seed in seeds_by_model[model]:
                gaps = compute_seed_gaps(base_dir, seed, cond)
                if gaps:
                    store[model][cond][seed] = gaps
    return store


def seed_series(store, model, cond, gap_key):
    series = []
    for seed, round_data in store[model][cond].items():
        arr = np.array([round_data.get(r, {}).get(gap_key, np.nan) for r in ROUNDS])
        series.append(arr)
    return series


def mean_se(series_list):
    arr = np.array(series_list, dtype=float)
    mean = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    return mean, se


def savefig(fig, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, fname.replace(".png", f".{ext}"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {path}")
    plt.close(fig)


# ─── Analysis 1: per model ──────────────────────────────────────────────────

def plot_per_model(store, models, stage_root):
    print("\nAnalysis 1 - Per-model perception-action gap trajectories")
    rounds = np.array(ROUNDS)
    for model in models:
        fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 7),
                                 sharex=True, sharey="row")
        for col, cond in enumerate(CONDITIONS):
            for row, gap_key in enumerate(["IN_gap", "DN_gap"]):
                ax = axes[row, col]
                series = seed_series(store, model, cond, gap_key)
                color = GAP_COLORS[gap_key]
                if not series:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=9, color="gray")
                else:
                    for arr in series:
                        ax.plot(rounds, arr, color=color, linewidth=0.8, alpha=0.35, zorder=1)
                    mean, _ = mean_se(series)
                    ax.plot(rounds, mean, color=color, linewidth=2.5, zorder=2, label=GAP_LABELS[gap_key])
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.tick_params(labelsize=TICK_SIZE)
                ax.grid(True, alpha=0.25)
                if col == 0:
                    ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LABEL_SIZE)
                if row == 0:
                    ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
                if row == 1:
                    ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        plt.tight_layout()
        savefig(fig, os.path.join(stage_root, model, "alignment_analysis"),
                "perception_action_gap_per_seed.png")


# ─── Analysis 2: across models/settings ─────────────────────────────────────

def plot_across_models(store, models, labels, colors, markers, out_dir, fname):
    print("\nAnalysis 2 - Per-condition perception-action gap across models")
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(4 * len(CONDITIONS), 7),
                             sharex=True, sharey="row")
    for col, cond in enumerate(CONDITIONS):
        for row, gap_key in enumerate(["IN_gap", "DN_gap"]):
            ax = axes[row, col]
            for model in models:
                series = seed_series(store, model, cond, gap_key)
                if not series:
                    continue
                mean, se = mean_se(series)
                color = colors[model]
                ax.plot(rounds, mean, color=color, linewidth=2, marker=markers[model],
                        markersize=5, label=labels[model])
                ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.12)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)
            if col == 0:
                ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
            if row == 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=len(models),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.align_ylabels(axes[:, 0])
    savefig(fig, out_dir, fname)


def publish(stage_dir, publish_dir, fnames):
    """Copy only the named files -- stage_dir (e.g. figures/local/cross_model/)
    is a directory SHARED with other orchestrator scripts, so copying its
    entire contents would leak unrelated files into publish_dir."""
    if os.path.abspath(stage_dir) == os.path.abspath(publish_dir):
        return
    os.makedirs(publish_dir, exist_ok=True)
    for fname in fnames:
        src = os.path.join(stage_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(publish_dir, fname))
    print(f"  Published -> {publish_dir}")


# ─── Tier runners ────────────────────────────────────────────────────────────

def run_family_tier(tier):
    spec = TIER_SPECS[tier]
    model_dirs = {m: os.path.join(spec["results"], m, spec["variant"]) for m in spec["models"]}
    seeds_by_model = {m: spec["seeds"] for m in spec["models"]}

    print(f"[{tier}] Building perception-action gap store …")
    store = build_store(model_dirs, seeds_by_model)
    for model in spec["models"]:
        for cond in ["FULL", "BASELINE"]:
            n = len(store[model].get(cond, {}))
            print(f"  {spec['labels'][model]} / {cond}: {n} seeds")

    plot_per_model(store, spec["models"], spec["stage_root"])
    cross_dir = os.path.join(spec["stage_root"], "cross_model")
    fname = "perception_action_gap_across_models.png"
    plot_across_models(store, spec["models"], spec["labels"], spec["colors"], spec["markers"],
                       cross_dir, fname)
    publish(cross_dir, spec["publish_dir"], [fname, fname.replace(".png", ".pdf")])


def run_structural_tier(axis):
    for stratum, val, sources in STRUCTURAL_SETTINGS[axis]:
        label = f"{axis}_{stratum}_{val}"
        models = sorted(sources.keys())
        if not models:
            continue
        model_dirs = {m: sources[m]["dir"] for m in models}
        seeds_by_model = {m: sources[m]["seeds"] for m in models}
        display_labels = {m: sources[m]["family"] for m in models}
        colors = {m: _QUAL_COLORS[i % len(_QUAL_COLORS)] for i, m in enumerate(models)}
        markers = {m: _QUAL_MARKERS[i % len(_QUAL_MARKERS)] for i, m in enumerate(models)}

        print(f"\n[{label}] Building perception-action gap store …")
        store = build_store(model_dirs, seeds_by_model)
        for model in models:
            n = len(store[model].get("FULL", {}))
            print(f"  {display_labels[model]} / FULL: {n} seeds")

        stage_dir = os.path.join(BASE, "figures", "local", "structural", label, "cross_model")
        fname = "perception_action_gap_across_models.png"
        plot_across_models(store, models, display_labels, colors, markers, stage_dir, fname)
        publish_dir = os.path.join(BASE, "figures", "SUPPLEMENTARY_RESULTS", "5_community_group_mcpr",
                                   axis, "5_perception_action_gap_evolution", label)
        publish(stage_dir, publish_dir, [fname, fname.replace(".png", ".pdf")])


def main():
    parser = argparse.ArgumentParser(description="Perception-action gap trajectory plots (item b).")
    parser.add_argument("--tier", choices=["7b", "s2", "13b", "70b", "community", "group", "all"], default="7b")
    args = parser.parse_args()

    tiers = ["7b", "s2", "13b", "70b", "community", "group"] if args.tier == "all" else [args.tier]
    for tier in tiers:
        if tier in TIER_SPECS:
            run_family_tier(tier)
        else:
            run_structural_tier(tier)
    print("\nAll done.")


if __name__ == "__main__":
    main()
