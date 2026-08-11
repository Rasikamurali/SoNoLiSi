"""
new_intro_adversarial_plots.py
------------------------------
Behavioral trajectory plots for the adversarial new-agent introduction experiments.

Data: code/results/{model}/local_newintro/intro{10|20}_adversarial/
Models: llama, mistral, qwen, llama_13b, mistral_13b, qwen_14b, llama_70b, qwen_72b

Plots produced
--------------
1. cross_model_intro10.pdf / cross_model_intro20.pdf
   One figure per intro-round setting. X-axis = round number.
   One line per condition (mean contribution across all agents and seeds,
   averaged across all 8 models). Pre-intro segment: solid, full colour.
   Post-intro segment: lighter shade of same colour, dashed.
   Vertical dashed line marks the intro round.

2. per_model/{model}_intro10.pdf / per_model/{model}_intro20.pdf
   Same design as above but for a single model (mean ± SE across seeds).
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import colorsys

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/code/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/new_intro/adversarial"

MODELS = ["llama", "mistral", "qwen", "llama_13b", "mistral_13b",
          "qwen_14b", "llama_70b", "qwen_72b"]
MODEL_LABELS = {
    "llama":       "Llama-8B",
    "mistral":     "Mistral-7B",
    "qwen":        "Qwen-7B",
    "llama_13b":   "Llama-13B",
    "mistral_13b": "Mistral-13B",
    "qwen_14b":    "Qwen-14B",
    "llama_70b":   "Llama-70B",
    "qwen_72b":    "Qwen-72B",
}

TAGS = {
    "intro10": "intro10_adversarial",
    "intro20": "intro20_adversarial",
}
INTRO_ROUNDS = {"intro10": 11, "intro20": 21}   # round where new agent first plays
SEEDS = list(range(42, 52))

CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline",
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}

# Conditions that have perception data (IN/DN)
PERC_CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]

# Condition colours (pre-intro)
COND_COLORS = {
    "PURE_BASELINE": "#9E9E9E",   # grey
    "BASELINE":      "#2196F3",   # blue
    "NO_SELECTION":  "#4CAF50",   # green
    "NO_DISCUSSION": "#FF9800",   # orange
    "FULL":          "#9C27B0",   # purple
}

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(os.path.join(FIG_ROOT, "per_model"), exist_ok=True)


# ─── Colour helpers ───────────────────────────────────────────────────────────

def lighten(hex_color, factor=0.45):
    """Return a lightened version of hex_color by blending with white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    r2 = r + (1 - r) * factor
    g2 = g + (1 - g) * factor
    b2 = b + (1 - b) * factor
    return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(model, tag_key, seed, condition):
    tag   = TAGS[tag_key]
    pat   = os.path.join(RESULTS, model, "local_newintro", tag,
                         f"seed{seed}", "log_*.json")
    best  = {}
    for p in sorted(glob.glob(pat)):
        try:
            d = json.load(open(p))
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def seed_metric_series(model, tag_key, seed, condition):
    """
    Returns {round: {contribution, payoff, in_gap, dn_gap}} per round for one seed.
    Gaps are NaN if perceptions are unavailable (e.g. PURE_BASELINE).
    """
    d = load_log(model, tag_key, seed, condition)
    if d is None:
        return {}
    result = {}
    for r in d["round_logs"]:
        rnd = r["round"]
        contribs = list(r.get("contributions", {}).values())
        payoffs  = list(r.get("payoffs", {}).values())
        percs    = r.get("perceptions") or {}

        ins, dns = [], []
        for p in percs.values():
            if not p: continue
            iv = p.get("injunctive_norm")
            dv = p.get("descriptive_norm")
            if iv is not None: ins.append(float(iv))
            if dv is not None: dns.append(float(dv))

        mean_c = float(np.mean(contribs)) if contribs else np.nan
        result[rnd] = {
            "contribution": mean_c,
            "payoff":       float(np.mean(payoffs)) if payoffs else np.nan,
            "in_gap":       float(np.mean(ins)) - mean_c if ins and not np.isnan(mean_c) else np.nan,
            "dn_gap":       float(np.mean(dns)) - mean_c if dns and not np.isnan(mean_c) else np.nan,
        }
    return result


def build_series(model, tag_key, condition, metric):
    """
    Returns (rounds_array, mean_array, se_array) for one metric, aggregated across seeds.
    """
    all_rounds = set()
    seed_data  = {}
    for seed in SEEDS:
        s = seed_metric_series(model, tag_key, seed, condition)
        if s:
            seed_data[seed] = s
            all_rounds |= set(s.keys())

    if not seed_data:
        return np.array([]), np.array([]), np.array([])

    rounds = np.array(sorted(all_rounds))
    matrix = np.full((len(seed_data), len(rounds)), np.nan)
    for i, sd in enumerate(seed_data.values()):
        for j, r in enumerate(rounds):
            if r in sd:
                matrix[i, j] = sd[r].get(metric, np.nan)

    mean = np.nanmean(matrix, axis=0)
    n    = np.sum(~np.isnan(matrix), axis=0)
    se   = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    return rounds, mean, se


# ─── Plotting core ────────────────────────────────────────────────────────────

def plot_condition_lines(ax, rounds, mean, se, intro_round,
                         color_pre, color_post, label=None, show_se=True):
    """
    Draw one condition's trajectory on ax.
    Pre-intro: solid, color_pre.
    Post-intro (including intro round itself): dashed, color_post.
    """
    pre_mask  = rounds <  intro_round
    post_mask = rounds >= intro_round

    # pre-intro segment
    if pre_mask.any():
        ax.plot(rounds[pre_mask], mean[pre_mask],
                color=color_pre, linewidth=2, solid_capstyle="round",
                label=label)
        if show_se:
            ax.fill_between(rounds[pre_mask],
                            (mean - se)[pre_mask], (mean + se)[pre_mask],
                            color=color_pre, alpha=0.15)

    # bridge the gap (last pre point → first post point) so the line is continuous
    if pre_mask.any() and post_mask.any():
        bridge_x = [rounds[pre_mask][-1], rounds[post_mask][0]]
        bridge_y = [mean[pre_mask][-1],   mean[post_mask][0]]
        ax.plot(bridge_x, bridge_y, color=color_post,
                linewidth=2, linestyle="--", solid_capstyle="round")

    # post-intro segment
    if post_mask.any():
        ax.plot(rounds[post_mask], mean[post_mask],
                color=color_post, linewidth=2, linestyle="--",
                solid_capstyle="round")
        if show_se:
            ax.fill_between(rounds[post_mask],
                            (mean - se)[post_mask], (mean + se)[post_mask],
                            color=color_post, alpha=0.15)


def style_ax(ax, intro_round, total_rounds, xlabel=True):
    ax.axvline(intro_round, color="black", linewidth=1.2,
               linestyle=":", alpha=0.7, label="_nolegend_")
    ax.axvspan(intro_round, total_rounds + 0.5, color="gray", alpha=0.06)
    ax.set_xlim(0.5, total_rounds + 0.5)
    ax.set_ylim(0, 10.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)
    if xlabel:
        ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean contribution", fontsize=11)


METRIC_CONFIG = {
    "contribution": {"ylabel": "Mean contribution",    "ylim": (0, 10.5), "conditions": CONDITIONS},
    "payoff":       {"ylabel": "Mean payoff",           "ylim": None,      "conditions": CONDITIONS},
    "in_gap":       {"ylabel": "IN gap (IN − contrib)", "ylim": None,      "conditions": PERC_CONDITIONS},
    "dn_gap":       {"ylabel": "DN gap (DN − contrib)", "ylim": None,      "conditions": PERC_CONDITIONS},
}

from matplotlib.lines import Line2D

def _legend_entries(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.2,
                          linestyle=":", label="New agent introduced"))
    labels.append("New agent introduced")
    return handles, labels


# ─── 1. Cross-model plots ─────────────────────────────────────────────────────

def cross_model_avg(tag_key, condition, metric):
    """Average mean series across all models that have data."""
    all_series = []
    for model in MODELS:
        rounds, mean, _ = build_series(model, tag_key, condition, metric)
        if len(rounds):
            all_series.append((rounds, mean))

    if not all_series:
        return np.array([]), np.array([]), np.array([])

    all_rounds = sorted(set(r for rounds, _ in all_series for r in rounds))
    arr = np.full((len(all_series), len(all_rounds)), np.nan)
    for i, (rounds, mean) in enumerate(all_series):
        r2idx = {r: j for j, r in enumerate(all_rounds)}
        for r, v in zip(rounds, mean):
            arr[i, r2idx[r]] = v

    rounds_arr = np.array(all_rounds)
    mean_arr   = np.nanmean(arr, axis=0)
    n          = np.sum(~np.isnan(arr), axis=0)
    se_arr     = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    return rounds_arr, mean_arr, se_arr


def make_cross_model_plot(tag_key, metric):
    intro_round  = INTRO_ROUNDS[tag_key]
    total_rounds = (intro_round - 1) + 5
    cfg          = METRIC_CONFIG[metric]
    conds        = cfg["conditions"]

    fig, ax = plt.subplots(figsize=(7, 4))

    for cond in conds:
        rounds, mean, se = cross_model_avg(tag_key, cond, metric)
        if not len(rounds):
            continue
        color_pre  = COND_COLORS[cond]
        color_post = lighten(color_pre, factor=0.45)
        plot_condition_lines(ax, rounds, mean, se, intro_round,
                             color_pre, color_post,
                             label=COND_LABELS[cond], show_se=True)

    style_ax(ax, intro_round, total_rounds)
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    if cfg["ylim"]:
        ax.set_ylim(cfg["ylim"])
    if metric in ("in_gap", "dn_gap"):
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)

    handles, labels = _legend_entries(ax)
    ax.legend(handles, labels, fontsize=9, frameon=True, loc="best", ncol=1)
    ax.set_title(f"Cross-model {cfg['ylabel'].lower()} — adversarial agent after "
                 f"round {intro_round - 1}", fontsize=11)

    plt.tight_layout()
    fname = f"cross_model_{tag_key}_{metric}"
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_ROOT, f"{fname}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── 2. Per-model combined (intro10 + intro20 side by side) ──────────────────

def make_per_model_combined(model, metric):
    cfg          = METRIC_CONFIG[metric]
    conds        = cfg["conditions"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for ax, tag_key in zip(axes, ["intro10", "intro20"]):
        intro_round  = INTRO_ROUNDS[tag_key]
        total_rounds = (intro_round - 1) + 5

        for cond in conds:
            rounds, mean, se = build_series(model, tag_key, cond, metric)
            if not len(rounds):
                continue
            color_pre  = COND_COLORS[cond]
            color_post = lighten(color_pre, factor=0.45)
            plot_condition_lines(ax, rounds, mean, se, intro_round,
                                 color_pre, color_post,
                                 label=COND_LABELS[cond], show_se=True)

        style_ax(ax, intro_round, total_rounds, xlabel=True)
        if cfg["ylim"]:
            ax.set_ylim(cfg["ylim"])
        if metric in ("in_gap", "dn_gap"):
            ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
        ax.set_title(f"Intro after round {intro_round - 1}", fontsize=12)
        ax.set_ylabel(cfg["ylabel"] if ax is axes[0] else "", fontsize=11)

    handles, labels = _legend_entries(axes[0])
    n_cols = len(conds) + 1
    fig.legend(handles, labels, loc="lower center", ncol=n_cols,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f"{MODEL_LABELS[model]} — {cfg['ylabel'].lower()} — adversarial new agent",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    fname = f"{model}_{metric}_combined"
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_ROOT, "per_model", f"{fname}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics = list(METRIC_CONFIG.keys())

    print("Cross-model plots …")
    for tag_key in ["intro10", "intro20"]:
        for metric in metrics:
            make_cross_model_plot(tag_key, metric)

    print("\nPer-model combined plots …")
    for model in MODELS:
        for metric in metrics:
            make_per_model_combined(model, metric)

    print("\nDone.")
