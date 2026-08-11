"""
robustness_analysis.py
----------------------
Produces the same figure set as figures/local/ for each robustness experiment:

  1. all_models_contribution.pdf/png
  2. all_models_payoff.pdf/png
  3. cross_model/network_development_across_models.pdf/png
  4. cross_model/perception_action_gap_across_models.pdf/png
  5. {model}/behavioral_analysis/per_seed_*_contribution_payoff.png
  6. {model}/selection_analysis/network_development.pdf/png
  7. {model}/alignment_analysis/perception_action_gap_per_seed.pdf/png
  8. {model}/perception_analysis/perception_ols.csv

Experiments (each gets its own folder under figures/local/):
  N=16 agents       -> figures/local/16agents/      (8 models, seeds 42-51)
  N=20 agents       -> figures/local/20agents/      (8 models, seeds 42-51)
  Group size G=3    -> figures/local/groupsize/G3/  (3 models, seeds 42-51)
  Group size G=6    -> figures/local/groupsize/G6/  (3 models, seeds 42-51)
  Group size G=8    -> figures/local/groupsize/G8/  (3 models, seeds 42-51)
  MCPR=0.5          -> figures/local/mcpr/MCPR0.5/  (3 models, seeds 42-51)
  MCPR=0.8          -> figures/local/mcpr/MCPR0.8/  (3 models, seeds 42-51)
"""

import json, glob, os, warnings, csv
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

# ─── Models ───────────────────────────────────────────────────────────────────

MODELS_7B  = ["llama", "mistral", "qwen"]
MODELS_ALL = ["llama", "mistral", "qwen", "llama_13b", "mistral_13b",
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

SEEDS = list(range(42, 52))   # seeds 42-51 for all code/results experiments

LS = 14   # label size
TS = 12   # tick size
LL = 13   # legend size
LW = 2.0  # line width

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
    """Per-round mean of 'contributions' or 'payoffs'."""
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
    """One panel per model, all conditions as colored lines. Matches reference style."""
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
            series_list = store[model][cond]
            if not series_list: continue
            arr    = np.array(series_list, dtype=float)
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
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=LS)
        ax.text(0.97, 0.97, MODEL_LABELS[model],
                transform=ax.transAxes, fontsize=LS, ha="right", va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CONDITIONS),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    savefig(fig, os.path.join(fig_root, fname))


# ─── 3: cross_model/network_development_across_models ────────────────────────

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
    # Spearman alignment
    if len(set(iw_vals)) < 2 or len(set(con_vals)) < 2:
        alignment = np.nan
    else:
        alignment, _ = scipy_stats.spearmanr(iw_vals, con_vals)
    # Gini
    g = gini_coeff(iw_vals)
    # Top-quartile share
    thresh    = np.percentile(con_vals, 75)
    top_mask  = con_vals >= thresh
    total_iw  = iw_vals.sum()
    top_share = iw_vals[top_mask].sum() / total_iw if total_iw > 0 else np.nan
    return {"alignment": float(alignment), "gini": g, "top_share": top_share}


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
                    if m is not None: result[r["round"]] = m
                if result: store[model][cond][seed] = result
    return store


def net_seed_series(store, model, cond, metric, rounds):
    series = []
    for seed_data in store[model][cond].values():
        arr = np.array([seed_data.get(r, {}).get(metric, np.nan) for r in rounds])
        series.append(arr)
    return series


METRIC_LABELS = {
    "alignment": r"Spearman $r$(weight, contrib)",
    "gini":      "Gini (incoming weight)",
    "top_share": "Top-quartile weight share",
}


def plot_network_across_models(store, models, rounds, fig_root):
    metrics = ["alignment", "gini", "top_share"]
    fig, axes = plt.subplots(len(NET_CONDS), len(metrics),
                             figsize=(4*len(metrics), 4*len(NET_CONDS)),
                             sharex=True)
    if len(NET_CONDS) == 1: axes = axes[np.newaxis, :]

    for row, cond in enumerate(NET_CONDS):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for model in models:
                series = net_seed_series(store, model, cond, metric, rounds)
                if not series: continue
                arr  = np.array(series, dtype=float)
                mean = np.nanmean(arr, axis=0)
                n    = np.sum(~np.isnan(arr), axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n>0, n, np.nan))
                color = MODEL_COLORS[model]
                ax.plot(rounds, mean, color=color, linewidth=LW,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(rounds, mean-se, mean+se, color=color, alpha=0.12)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TS)
            ax.grid(True, alpha=0.25)
            if col == 0:     ax.set_ylabel(COND_LABELS[cond], fontsize=LS)
            if row == 0:     ax.set_title(METRIC_LABELS[metric], fontsize=LS)
            if row == len(NET_CONDS)-1: ax.set_xlabel("Round", fontsize=LS)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               fontsize=LL, frameon=False, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    cdir = os.path.join(fig_root, "cross_model")
    os.makedirs(cdir, exist_ok=True)
    savefig(fig, os.path.join(cdir, "network_development_across_models"))


def plot_network_per_model(store, models, rounds, fig_root):
    metrics = ["alignment", "gini", "top_share"]
    for model in models:
        fig, axes = plt.subplots(len(metrics), len(NET_CONDS),
                                 figsize=(4*len(NET_CONDS), 4*len(metrics)),
                                 sharex=True)
        if len(NET_CONDS) == 1: axes = axes[:, np.newaxis]
        for col, cond in enumerate(NET_CONDS):
            color = COND_COLORS[cond]
            for row, metric in enumerate(metrics):
                ax     = axes[row, col]
                series = net_seed_series(store, model, cond, metric, rounds)
                if not series:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=9)
                else:
                    arr = np.array(series, dtype=float)
                    for row_arr in arr:
                        ax.plot(rounds, row_arr, color=color, linewidth=0.8, alpha=0.3)
                    mean = np.nanmean(arr, axis=0)
                    ax.plot(rounds, mean, color=color, linewidth=2.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.tick_params(labelsize=TS)
                ax.grid(True, alpha=0.25)
                if col == 0: ax.set_ylabel(METRIC_LABELS[metric], fontsize=LS)
                if row == 0: ax.set_title(COND_LABELS[cond], fontsize=LS)
                if row == len(metrics)-1: ax.set_xlabel("Round", fontsize=LS)

        fig.suptitle(MODEL_LABELS[model], fontsize=LS+2, y=1.005)
        plt.tight_layout()
        sdir = os.path.join(fig_root, model, "selection_analysis")
        os.makedirs(sdir, exist_ok=True)
        savefig(fig, os.path.join(sdir, "network_development"))


# ─── 4: cross_model/perception_action_gap_across_models ──────────────────────

def compute_gap_seed(root, model, variant, tag, seed, cond):
    d = load_log(root, model, variant, tag, seed, cond)
    if d is None: return {}
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
    return result


def build_gap_store(root, variant, tag, models):
    store = {m: {c: {} for c in PERC_CONDS} for m in models}
    for model in models:
        for cond in PERC_CONDS:
            for seed in SEEDS:
                gaps = compute_gap_seed(root, model, variant, tag, seed, cond)
                if gaps: store[model][cond][seed] = gaps
    return store


GAP_LABELS = {"IN_gap": "Injunctive Norm - Actual", "DN_gap": "Descriptive Norm - Actual"}
GAP_COLORS = {"IN_gap": "#e377c2", "DN_gap": "#17becf"}


def gap_seed_series(store, model, cond, gap_key, rounds):
    series = []
    for seed_data in store[model][cond].values():
        arr = np.array([seed_data.get(r, {}).get(gap_key, np.nan) for r in rounds])
        series.append(arr)
    return series


def plot_alignment_across_models(store, models, rounds, fig_root):
    gap_keys = ["IN_gap", "DN_gap"]
    fig, axes = plt.subplots(len(gap_keys), len(PERC_CONDS),
                             figsize=(4*len(PERC_CONDS), 7),
                             sharex=True, sharey="row")
    for col, cond in enumerate(PERC_CONDS):
        for row, gap_key in enumerate(gap_keys):
            ax = axes[row, col]
            for model in models:
                series = gap_seed_series(store, model, cond, gap_key, rounds)
                if not series: continue
                arr  = np.array(series, dtype=float)
                mean = np.nanmean(arr, axis=0)
                n    = np.sum(~np.isnan(arr), axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n>0, n, np.nan))
                color = MODEL_COLORS[model]
                ax.plot(rounds, mean, color=color, linewidth=LW,
                        marker="o", markersize=3, label=MODEL_LABELS[model])
                ax.fill_between(rounds, mean-se, mean+se, color=color, alpha=0.12)
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


def plot_alignment_per_model(store, models, rounds, fig_root):
    gap_keys = ["IN_gap", "DN_gap"]
    for model in models:
        fig, axes = plt.subplots(len(gap_keys), len(PERC_CONDS),
                                 figsize=(4*len(PERC_CONDS), 7),
                                 sharex=True, sharey="row")
        for col, cond in enumerate(PERC_CONDS):
            for row, gap_key in enumerate(gap_keys):
                ax     = axes[row, col]
                series = gap_seed_series(store, model, cond, gap_key, rounds)
                color  = GAP_COLORS[gap_key]
                if not series:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=9)
                else:
                    arr = np.array(series, dtype=float)
                    for row_arr in arr:
                        ax.plot(rounds, row_arr, color=color, linewidth=0.8, alpha=0.35)
                    mean = np.nanmean(arr, axis=0)
                    ax.plot(rounds, mean, color=color, linewidth=2.5,
                            label=GAP_LABELS[gap_key])
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.tick_params(labelsize=TS)
                ax.grid(True, alpha=0.25)
                if col == 0: ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LS)
                if row == 0: ax.set_title(COND_LABELS[cond], fontsize=LS)
                if row == len(gap_keys)-1: ax.set_xlabel("Round", fontsize=LS)

        fig.suptitle(MODEL_LABELS[model], fontsize=LS+2, y=1.005)
        plt.tight_layout()
        adir = os.path.join(fig_root, model, "alignment_analysis")
        os.makedirs(adir, exist_ok=True)
        savefig(fig, os.path.join(adir, "perception_action_gap_per_seed"))


# ─── 5: {model}/behavioral_analysis/per_seed_*_contribution_payoff.png ────────

def plot_per_seed_behavioral(root, variant, tag, models, fig_root):
    for model in models:
        bdir = os.path.join(fig_root, model, "behavioral_analysis")
        os.makedirs(bdir, exist_ok=True)
        for seed in SEEDS:
            cond_data = {}
            for cond in CONDITIONS:
                d = load_log(root, model, variant, tag, seed, cond)
                if d is not None: cond_data[cond] = d
            if not cond_data: continue

            fig, (ax_c, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            fig.suptitle(
                f"{MODEL_LABELS[model]} — seed {seed}\nContribution & Payoff",
                fontsize=11)
            for cond, d in cond_data.items():
                rounds  = [r["round"] for r in d["round_logs"]]
                contrib = round_series(d, "contributions")
                payoff  = round_series(d, "payoffs")
                color   = COND_COLORS[cond]
                label   = COND_LABELS[cond]
                ax_c.plot(rounds, contrib, color=color, linewidth=2,
                          marker="o", markersize=3, label=label)
                ax_p.plot(rounds, payoff,  color=color, linewidth=2,
                          marker="o", markersize=3)
            ax_c.set_ylim(0, 11)
            ax_c.set_ylabel("Mean Contribution", fontsize=10)
            ax_c.legend(fontsize=8, loc="lower right")
            ax_c.grid(True, alpha=0.3)
            ax_c.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax_p.set_ylabel("Mean Payoff", fontsize=10)
            ax_p.set_xlabel("Round", fontsize=10)
            ax_p.grid(True, alpha=0.3)
            ax_p.xaxis.set_major_locator(ticker.MultipleLocator(5))
            plt.tight_layout()
            fname = os.path.join(bdir, f"per_seed_{seed}_contribution_payoff.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"    -> {bdir}/per_seed_*.png")


# ─── 6: {model}/perception_analysis/ — OLS csv ───────────────────────────────

def write_perception_stats(root, variant, tag, models, fig_root):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        print("    [SKIP] perception stats: pandas/statsmodels not available")
        return

    for model in models:
        rows = []
        for cond in PERC_CONDS:
            for seed in SEEDS:
                d = load_log(root, model, variant, tag, seed, cond)
                if d is None: continue
                for r in d["round_logs"]:
                    rnd   = r["round"]
                    percs = r.get("perceptions") or {}
                    conts = {int(k): float(v) for k, v in r["contributions"].items()}
                    for aid_str, perc in percs.items():
                        if not perc: continue
                        aid  = int(aid_str)
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        if inj is None or desc is None: continue
                        rows.append({
                            "condition": cond, "seed": seed,
                            "round": rnd, "agent_id": aid,
                            "IN": float(inj), "DN": float(desc),
                            "contribution": conts.get(aid, np.nan),
                        })

        if not rows:
            continue

        pdir = os.path.join(fig_root, model, "perception_analysis")
        os.makedirs(pdir, exist_ok=True)

        df = pd.DataFrame(rows).dropna()
        results = []
        for cond, grp in df.groupby("condition"):
            if len(grp) < 10: continue
            try:
                m = smf.ols("IN ~ DN", data=grp).fit()
                results.append({
                    "condition": cond,
                    "slope_DN": round(m.params.get("DN", np.nan), 4),
                    "intercept": round(m.params.get("Intercept", np.nan), 4),
                    "r2": round(m.rsquared, 4),
                    "p_DN": round(m.pvalues.get("DN", np.nan), 4),
                    "n": len(grp),
                })
            except Exception:
                pass

        if results:
            csv_path = os.path.join(pdir, "perception_ols.csv")
            pd.DataFrame(results).to_csv(csv_path, index=False)
            print(f"    -> {csv_path}")


# ─── Master runner ────────────────────────────────────────────────────────────

def run_experiment(root, variant, tag, models, fig_root, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    os.makedirs(os.path.join(fig_root, "cross_model"), exist_ok=True)

    # Determine max rounds from first available log
    n_rounds = 20
    for model in models:
        for cond in CONDITIONS:
            d = load_log(root, model, variant, tag, SEEDS[0], cond)
            if d:
                n_rounds = len(d["round_logs"])
                break
        else:
            continue
        break
    rounds = list(range(1, n_rounds + 1))
    rounds_arr = np.array(rounds)

    print(f"  Rounds: {n_rounds}, Models: {[MODEL_LABELS[m] for m in models]}")

    # 1 & 2: all_models contribution + payoff
    print("  [1] all_models_contribution")
    plot_all_models_metric(root, variant, tag, models, fig_root,
                           "contributions", "Mean Contribution (0-10)",
                           (0, 10.5), "all_models_contribution")
    print("  [2] all_models_payoff")
    plot_all_models_metric(root, variant, tag, models, fig_root,
                           "payoffs", "Mean Payoff",
                           None, "all_models_payoff")

    # 3: cross_model network
    print("  [3] network_development_across_models")
    net_store = build_network_store(root, variant, tag, models)
    plot_network_across_models(net_store, models, rounds_arr, fig_root)

    # per-model selection_analysis
    print("  [selection_analysis] network per model")
    plot_network_per_model(net_store, models, rounds_arr, fig_root)

    # 4: cross_model alignment
    print("  [4] perception_action_gap_across_models")
    gap_store = build_gap_store(root, variant, tag, models)
    plot_alignment_across_models(gap_store, models, rounds_arr, fig_root)

    # per-model alignment_analysis
    print("  [alignment_analysis] per model")
    plot_alignment_per_model(gap_store, models, rounds_arr, fig_root)

    # per-model behavioral_analysis
    print("  [behavioral_analysis] per seed per model")
    plot_per_seed_behavioral(root, variant, tag, models, fig_root)

    # per-model perception_analysis
    print("  [perception_analysis] OLS stats per model")
    write_perception_stats(root, variant, tag, models, fig_root)

    print(f"  Done -> {fig_root}")


# ─── Experiment definitions ───────────────────────────────────────────────────

EXPERIMENTS = [
    (CODE_RESULTS, "local",               "N16_G4",         MODELS_ALL,
     os.path.join(FIG_BASE, "16agents"),
     "N=16 agents"),

    (CODE_RESULTS, "local",               "N20_G4",         MODELS_ALL,
     os.path.join(FIG_BASE, "20agents"),
     "N=20 agents"),

    (CODE_RESULTS, "local_groupsizevary", "N12_G3_MCPR0.4", MODELS_7B,
     os.path.join(FIG_BASE, "groupsize", "G3"),
     "Group size G=3 (N=12, MCPR=0.4)"),

    (CODE_RESULTS, "local_groupsizevary", "N12_G6_MCPR0.4", MODELS_7B,
     os.path.join(FIG_BASE, "groupsize", "G6"),
     "Group size G=6 (N=12, MCPR=0.4)"),

    (CODE_RESULTS, "local_groupsizevary", "N16_G8_MCPR0.4", MODELS_7B,
     os.path.join(FIG_BASE, "groupsize", "G8"),
     "Group size G=8 (N=16, MCPR=0.4)"),

    (CODE_RESULTS, "local_groupsizevary", "N12_G4_MCPR0.5", MODELS_7B,
     os.path.join(FIG_BASE, "mcpr", "MCPR0.5"),
     "MCPR=0.5 (G=4, N=12)"),

    (CODE_RESULTS, "local_groupsizevary", "N12_G4_MCPR0.8", MODELS_7B,
     os.path.join(FIG_BASE, "mcpr", "MCPR0.8"),
     "MCPR=0.8 (G=4, N=12)"),
]


if __name__ == "__main__":
    for args in EXPERIMENTS:
        run_experiment(*args)
    print("\nAll experiments done.")
