"""
robustness_split.py
-------------------
Splits the 16-agent and 20-agent cross-model figures by model size and model family.

For each of N=16 and N=20, generates:
  figures/local/{N}agents/by_size/7b_8b/
  figures/local/{N}agents/by_size/13b_14b/
  figures/local/{N}agents/by_size/70b_72b/
  figures/local/{N}agents/by_family/llama/
  figures/local/{N}agents/by_family/mistral/
  figures/local/{N}agents/by_family/qwen/

Each subfolder contains:
  all_models_contribution.pdf/png
  all_models_payoff.pdf/png
  cross_model/network_development_across_models.pdf/png
  cross_model/perception_action_gap_across_models.pdf/png
"""

import json, glob, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

CODE_RESULTS = "/data3/rasimura/social-norm-evo/code/results"
FIG_BASE     = "/data3/rasimura/social-norm-evo/figures/local"

# ─── Model groups ─────────────────────────────────────────────────────────────

BY_SIZE = {
    "7b_8b":   ["llama", "mistral", "qwen"],
    "13b_14b": ["llama_13b", "mistral_13b", "qwen_14b"],
    "70b_72b": ["llama_70b", "qwen_72b"],
}

BY_FAMILY = {
    "llama":   ["llama", "llama_13b", "llama_70b"],
    "mistral": ["mistral", "mistral_13b"],
    "qwen":    ["qwen", "qwen_14b", "qwen_72b"],
}

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
MODEL_COLORS = {
    "llama":       "#1f77b4",
    "mistral":     "#d62728",
    "qwen":        "#2ca02c",
    "llama_13b":   "#9467bd",
    "mistral_13b": "#8c564b",
    "qwen_14b":    "#e377c2",
    "llama_70b":   "#bcbd22",
    "qwen_72b":    "#17becf",
}

CONDITIONS   = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
PERC_CONDS   = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
NET_CONDS    = ["NO_DISCUSSION", "FULL"]

COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline",
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
COND_COLORS = {
    "PURE_BASELINE": "#7f7f7f",
    "BASELINE":      "#9467bd",
    "NO_SELECTION":  "#2ca02c",
    "NO_DISCUSSION": "#ff7f0e",
    "FULL":          "#1f77b4",
}

SEEDS = list(range(42, 52))
LS, TS, LL, LW = 14, 12, 13, 2.0
ROUNDS = np.arange(1, 21)

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(root, model, variant, tag, seed, condition):
    pattern = os.path.join(root, model, variant, tag, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def round_series(d, key):
    out = []
    for r in d["round_logs"]:
        vals = list(r[key].values())
        out.append(float(np.mean(vals)) if vals else np.nan)
    return out


def savefig(fig, path_no_ext):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_no_ext}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path_no_ext}.pdf")


# ─── 1 & 2: all_models_contribution / all_models_payoff ──────────────────────

def plot_all_models_metric(root, variant, tag, models, fig_root,
                           metric_key, ylabel, ylim, fname):
    store = {m: {c: [] for c in CONDITIONS} for m in models}
    for model in models:
        for cond in CONDITIONS:
            for seed in SEEDS:
                d = load_log(root, model, variant, tag, seed, cond)
                if d is None: continue
                store[model][cond].append(round_series(d, metric_key))

    fig, axes = plt.subplots(1, len(models),
                             figsize=(4 * len(models), 5),
                             sharex=True, sharey=True)
    if len(models) == 1: axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        for cond in CONDITIONS:
            sl = store[model][cond]
            if not sl: continue
            arr    = np.array(sl, dtype=float)
            rounds = np.arange(1, arr.shape[1] + 1)
            mean   = np.nanmean(arr, axis=0)
            se     = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(rounds, mean, color=COND_COLORS[cond], linewidth=LW,
                    marker="o", markersize=3, label=COND_LABELS[cond])
            ax.fill_between(rounds, mean - se, mean + se,
                            color=COND_COLORS[cond], alpha=0.15)
        if ylim: ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=TS)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Round", fontsize=LS)
        if idx == 0: ax.set_ylabel(ylabel, fontsize=LS)
        ax.text(0.97, 0.97, MODEL_LABELS[model],
                transform=ax.transAxes, fontsize=LS, ha="right", va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    savefig(fig, os.path.join(fig_root, fname))


# ─── 3: network_development_across_models ────────────────────────────────────

def incoming_weights_map(nw, agent_ids):
    iw = {a: 0.0 for a in agent_ids}
    for e in nw:
        v = int(e["v"])
        if v in iw: iw[v] += float(e["weight"])
    return iw


def gini_coeff(values):
    x = np.sort(np.array(values, dtype=float))
    n = len(x); s = x.sum()
    if n == 0 or s == 0: return np.nan
    return (2 * np.dot(np.arange(1, n+1), x)) / (n * s) - (n+1)/n


def compute_network_round(r):
    nw   = r.get("network_weights", [])
    cont = r.get("contributions", {})
    if not nw or not cont: return None
    agent_ids = [int(k) for k in cont]
    iw        = incoming_weights_map(nw, agent_ids)
    contribs  = {int(k): float(v) for k, v in cont.items()}
    iw_vals   = np.array([iw[a] for a in agent_ids])
    con_vals  = np.array([contribs[a] for a in agent_ids])
    if len(set(iw_vals)) < 2 or len(set(con_vals)) < 2:
        alignment = np.nan
    else:
        alignment, _ = scipy_stats.spearmanr(iw_vals, con_vals)
    g      = gini_coeff(iw_vals)
    thresh = np.percentile(con_vals, 75)
    total  = iw_vals.sum()
    topq   = iw_vals[con_vals >= thresh].sum() / total if total > 0 else np.nan
    return {"alignment": float(alignment), "gini": g, "top_share": topq}


def build_network_store(root, variant, tag, models):
    store = {m: {c: {} for c in NET_CONDS} for m in models}
    for model in models:
        for cond in NET_CONDS:
            for seed in SEEDS:
                d = load_log(root, model, variant, tag, seed, cond)
                if d is None: continue
                result = {}
                for r in d["round_logs"]:
                    m = compute_network_round(r)
                    if m: result[r["round"]] = m
                if result: store[model][cond][seed] = result
    return store


METRIC_LABELS = {
    "alignment": r"Spearman $r$(weight, contrib)",
    "gini":      "Gini (incoming weight)",
    "top_share": "Top-quartile weight share",
}


def plot_network_across_models(store, models, fig_root):
    metrics = ["alignment", "gini", "top_share"]
    fig, axes = plt.subplots(len(NET_CONDS), len(metrics),
                             figsize=(4*len(metrics), 4*len(NET_CONDS)),
                             sharex=True)
    if len(NET_CONDS) == 1: axes = axes[np.newaxis, :]

    for row, cond in enumerate(NET_CONDS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for model in models:
                series = [
                    np.array([sd.get(r, {}).get(metric, np.nan) for r in ROUNDS])
                    for sd in store[model][cond].values()
                ]
                if not series: continue
                arr  = np.array(series, dtype=float)
                mean = np.nanmean(arr, axis=0)
                n    = np.sum(~np.isnan(arr), axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n>0, n, np.nan))
                color = MODEL_COLORS[model]
                ax.plot(ROUNDS, mean, color=color, linewidth=LW,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(ROUNDS, mean-se, mean+se, color=color, alpha=0.12)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TS)
            ax.grid(True, alpha=0.25)
            if col == 0:              ax.set_ylabel(COND_LABELS[cond], fontsize=LS)
            if row == 0:              ax.set_title(METRIC_LABELS[metric], fontsize=LS)
            if row == len(NET_CONDS)-1: ax.set_xlabel("Round", fontsize=LS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    cdir = os.path.join(fig_root, "cross_model")
    os.makedirs(cdir, exist_ok=True)
    savefig(fig, os.path.join(cdir, "network_development_across_models"))


# ─── 4: perception_action_gap_across_models ───────────────────────────────────

GAP_LABELS = {"IN_gap": "Injunctive Norm - Actual", "DN_gap": "Descriptive Norm - Actual"}
GAP_COLORS = {"IN_gap": "#e377c2", "DN_gap": "#17becf"}


def build_gap_store(root, variant, tag, models):
    store = {m: {c: {} for c in PERC_CONDS} for m in models}
    for model in models:
        for cond in PERC_CONDS:
            for seed in SEEDS:
                d = load_log(root, model, variant, tag, seed, cond)
                if d is None: continue
                result = {}
                for r in d["round_logs"]:
                    rnd   = r["round"]
                    conts = {int(k): float(v) for k, v in r["contributions"].items()}
                    percs = {int(k): v for k, v in (r.get("perceptions") or {}).items()}
                    in_gaps, dn_gaps = [], []
                    for aid, perc in percs.items():
                        if not perc: continue
                        actual = conts.get(aid)
                        if actual is None: continue
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        if inj  is not None: in_gaps.append(float(inj)  - actual)
                        if desc is not None: dn_gaps.append(float(desc) - actual)
                    result[rnd] = {
                        "IN_gap": float(np.mean(in_gaps)) if in_gaps else np.nan,
                        "DN_gap": float(np.mean(dn_gaps)) if dn_gaps else np.nan,
                    }
                if result: store[model][cond][seed] = result
    return store


def plot_alignment_across_models(store, models, fig_root):
    gap_keys = ["IN_gap", "DN_gap"]
    fig, axes = plt.subplots(len(gap_keys), len(PERC_CONDS),
                             figsize=(4*len(PERC_CONDS), 7),
                             sharex=True, sharey="row")
    for col, cond in enumerate(PERC_CONDS):
        for row, gap_key in enumerate(gap_keys):
            ax = axes[row, col]
            for model in models:
                series = [
                    np.array([sd.get(r, {}).get(gap_key, np.nan) for r in ROUNDS])
                    for sd in store[model][cond].values()
                ]
                if not series: continue
                arr  = np.array(series, dtype=float)
                mean = np.nanmean(arr, axis=0)
                n    = np.sum(~np.isnan(arr), axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n>0, n, np.nan))
                color = MODEL_COLORS[model]
                ax.plot(ROUNDS, mean, color=color, linewidth=LW,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(ROUNDS, mean-se, mean+se, color=color, alpha=0.12)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TS)
            ax.grid(True, alpha=0.25)
            if col == 0: ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LS)
            if row == 0: ax.set_title(COND_LABELS[cond], fontsize=LS)
            if row == len(gap_keys)-1: ax.set_xlabel("Round", fontsize=LS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.align_ylabels(axes[:, 0])
    cdir = os.path.join(fig_root, "cross_model")
    os.makedirs(cdir, exist_ok=True)
    savefig(fig, os.path.join(cdir, "perception_action_gap_across_models"))


# ─── Runner for one group ─────────────────────────────────────────────────────

def run_group(root, variant, tag, models, fig_root, group_label):
    print(f"  [{group_label}] {[MODEL_LABELS[m] for m in models]}")
    os.makedirs(fig_root, exist_ok=True)

    plot_all_models_metric(root, variant, tag, models, fig_root,
                           "contributions", "Mean Contribution (0-10)",
                           (0, 10.5), "all_models_contribution")

    plot_all_models_metric(root, variant, tag, models, fig_root,
                           "payoffs", "Mean Payoff",
                           None, "all_models_payoff")

    net_store = build_network_store(root, variant, tag, models)
    plot_network_across_models(net_store, models, fig_root)

    gap_store = build_gap_store(root, variant, tag, models)
    plot_alignment_across_models(gap_store, models, fig_root)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for n_val in [16, 20]:
        root    = CODE_RESULTS
        variant = "local"
        tag     = f"N{n_val}_G4"
        base    = os.path.join(FIG_BASE, f"{n_val}agents")

        print(f"\n{'='*60}\nN={n_val} agents — by size\n{'='*60}")
        for size_label, models in BY_SIZE.items():
            run_group(root, variant, tag, models,
                      os.path.join(base, "by_size", size_label), size_label)

        print(f"\n{'='*60}\nN={n_val} agents — by family\n{'='*60}")
        for family_label, models in BY_FAMILY.items():
            run_group(root, variant, tag, models,
                      os.path.join(base, "by_family", family_label), family_label)

    print("\nAll done.")
