"""
evaluation_outcomes.py
----------------------
Analyses SoNoLiSi_v5 simulation logs across conditions.

Figures produced:
  1. Behavior and Expectations Over Time
       - mean contribution c̄_t
       - mean injunctive norm Ī_t
       - mean descriptive norm D̄_t
       - mean payoff p̄_t (secondary axis)

  2. Perception Convergence
       - injunctive norm variance Var(I_t)
       - descriptive norm error |D̄_t - c̄_t|

  3. Discussion vs Behavior
       - mean proposed contribution m̄_t  (extracted from discussion messages)
       - mean contribution c̄_t
       - behavior–discussion alignment |m̄_t - c̄_t|

  4. Selection Enforcement
       - correlation(c_i,t, W_i,t) where W_i,t = sum of incoming edge weights

  5. Network Assortativity
       - numeric_assortativity_coefficient(G_t, contribution)

Summary table: c̄, p̄, Ī, D̄, Var(I), E_t, m̄, A_t, Corr_t, Assortativity_t
"""

import json
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Configuration
# ============================================================

LOG_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR    = os.path.join(os.path.dirname(LOG_DIR), "results")
LOG_PATTERN_V5    = os.path.join(LOG_DIR, "log_v5_20260311_173502_*_seed43.json")
LOG_PATTERN_LOCAL = os.path.join(LOG_DIR, "log_v5local_*_seed43.json")
LOG_PATTERN_LLAMA   = os.path.join(RESULTS_DIR, "log_v5os_llama_*_seed43.json")
LOG_PATTERN_MISTRAL = os.path.join(RESULTS_DIR, "log_v5os_mistral_*_seed43.json")
LOG_PATTERN_QWEN    = os.path.join(RESULTS_DIR, "log_v5os_qwen_*_seed43.json")
ENDOWMENT = 10

CONDITION_ORDER = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
CONDITION_LABELS = {
    "PURE_BASELINE":  "Pure Baseline",
    "BASELINE":       "Baseline (perception)",
    "NO_SELECTION":   "No Selection",
    "NO_DISCUSSION":  "No Discussion",
    "FULL":           "Full",
}
COLORS = {
    "PURE_BASELINE":  "#aaaaaa",
    "BASELINE":       "#6baed6",
    "NO_SELECTION":   "#fd8d3c",
    "NO_DISCUSSION":  "#74c476",
    "FULL":           "#e6550d",
}

# ============================================================
# Load logs
# ============================================================

def load_logs(pattern: str) -> dict:
    logs = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        cond = data["condition"]
        logs[cond] = data
    return logs


# ============================================================
# Helpers
# ============================================================

def _extract_proposals(messages: list) -> list:
    """Extract numbers in [0, ENDOWMENT] from discussion message strings."""
    nums = []
    for m in messages:
        for tok in re.findall(r'\b(\d+(?:\.\d+)?)\b', m):
            v = float(tok)
            if 0.0 <= v <= ENDOWMENT:
                nums.append(v)
    return nums


def _incoming_weights(network_weights: list, agent_ids: list) -> dict:
    """Sum incoming edge weights per agent: W_i = sum_j w_{j->i}."""
    totals = {aid: 0.0 for aid in agent_ids}
    for e in network_weights:
        v = e["v"]
        if v in totals:
            totals[v] += e["weight"]
    return totals


def _assortativity(network_weights: list, contributions: dict) -> float:
    """Numeric assortativity of the directed network by contribution."""
    G = nx.DiGraph()
    for aid, c in contributions.items():
        G.add_node(aid, contribution=c)
    for e in network_weights:
        G.add_edge(e["u"], e["v"], weight=e["weight"])
    if G.number_of_edges() < 2:
        return np.nan
    try:
        return nx.numeric_assortativity_coefficient(G, "contribution")
    except Exception:
        return np.nan


# ============================================================
# Extract round-level metrics
# ============================================================

def extract_round_metrics(data: dict) -> pd.DataFrame:
    rows = []
    for r in data["round_logs"]:
        contribs_by_id = {int(k): v for k, v in r["contributions"].items()}
        contribs        = list(contribs_by_id.values())
        payoffs         = {int(k): v for k, v in r.get("payoffs", {}).items()}
        mean_contrib    = np.mean(contribs) if contribs else np.nan
        mean_payoff     = np.mean(list(payoffs.values())) if payoffs else np.nan

        # --- Perception metrics ---
        injunctive_vals, descriptive_vals = [], []
        if r.get("perceptions"):
            for perc in r["perceptions"].values():
                if perc is None:
                    continue
                i_raw = perc.get("injunctive_norm")
                if i_raw is not None:
                    try:
                        injunctive_vals.append(float(i_raw))
                    except (TypeError, ValueError):
                        pass
                d_raw = perc.get("descriptive_norm")
                if d_raw is not None:
                    try:
                        descriptive_vals.append(float(d_raw))
                    except (TypeError, ValueError):
                        pass

        mean_injunctive  = np.mean(injunctive_vals)  if injunctive_vals  else np.nan
        mean_descriptive = np.mean(descriptive_vals) if descriptive_vals else np.nan
        var_injunctive   = np.var(injunctive_vals)   if len(injunctive_vals) > 1 else np.nan
        desc_error       = abs(mean_descriptive - mean_contrib) \
                           if not (np.isnan(mean_descriptive) or np.isnan(mean_contrib)) else np.nan

        # --- Discussion metrics ---
        transcript = r.get("discussion_transcript") or []
        messages   = [e["message"] for e in transcript if isinstance(e.get("message"), str)]
        proposals  = _extract_proposals(messages)
        mean_proposed = np.mean(proposals) if proposals else np.nan
        disc_alignment = abs(mean_proposed - mean_contrib) \
                         if not (np.isnan(mean_proposed) or np.isnan(mean_contrib)) else np.nan

        # --- Selection / network metrics ---
        net_weights = r.get("network_weights", [])
        agent_ids   = list(contribs_by_id.keys())
        W           = _incoming_weights(net_weights, agent_ids)

        c_vec = np.array([contribs_by_id[aid] for aid in agent_ids])
        w_vec = np.array([W[aid]               for aid in agent_ids])
        if len(c_vec) > 1 and np.std(c_vec) > 0 and np.std(w_vec) > 0:
            corr = float(np.corrcoef(c_vec, w_vec)[0, 1])
        else:
            corr = np.nan

        assortativity = _assortativity(net_weights, contribs_by_id)

        rows.append({
            "round":            r["round"],
            "mean_contribution": mean_contrib,
            "mean_payoff":       mean_payoff,
            "mean_injunctive":   mean_injunctive,
            "mean_descriptive":  mean_descriptive,
            "var_injunctive":    var_injunctive,
            "desc_error":        desc_error,
            "mean_proposed":     mean_proposed,
            "disc_alignment":    disc_alignment,
            "contrib_weight_corr": corr,
            "assortativity":     assortativity,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def _line(ax, df, col, label, color, **kwargs):
    vals = df[col].to_numpy()
    if not np.all(np.isnan(vals)):
        ax.plot(df["round"].to_numpy(), vals, label=label, color=color,
                linewidth=2, marker="o", markersize=4, **kwargs)


def make_plots(metrics_by_cond: dict, out_dir: str, suffix: str = ""):
    rounds_label = "Round"

    # ----------------------------------------------------------
    # Figure 1 — Behavior and Expectations Over Time
    # ----------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle(f"Figure 1 — Behavior and Expectations Over Time{suffix}", fontsize=12)

    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df  = metrics_by_cond[cond]
        col = COLORS[cond]
        lbl = CONDITION_LABELS[cond]
        _line(axes1[0], df, "mean_contribution", lbl, col)
        _line(axes1[1], df, "mean_injunctive",   lbl, col)
        _line(axes1[2], df, "mean_descriptive",  lbl, col)

    axes1[0].set_title("Mean Contribution c̄_t")
    axes1[1].set_title("Mean Injunctive Norm Ī_t")
    axes1[2].set_title("Mean Descriptive Norm D̄_t")
    for ax in axes1:
        ax.set_xlabel(rounds_label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # secondary axis: mean payoff on contribution panel
    ax1b = axes1[0].twinx()
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        vals = df["mean_payoff"].to_numpy()
        if not np.all(np.isnan(vals)):
            ax1b.plot(df["round"].to_numpy(), vals,
                      color=COLORS[cond], linewidth=1, linestyle="--", alpha=0.5)
    ax1b.set_ylabel("Mean payoff (dashed)", fontsize=8)

    plt.tight_layout()
    _save(fig1, out_dir, f"fig1_behavior_expectations{suffix}")

    # ----------------------------------------------------------
    # Figure 2 — Perception Convergence
    # ----------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle(f"Figure 2 — Perception Convergence{suffix}", fontsize=12)

    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df  = metrics_by_cond[cond]
        col = COLORS[cond]
        lbl = CONDITION_LABELS[cond]
        _line(axes2[0], df, "var_injunctive", lbl, col)
        _line(axes2[1], df, "desc_error",     lbl, col)

    axes2[0].set_title("Injunctive Norm Variance Var(I_t)")
    axes2[1].set_title("Descriptive Norm Error |D̄_t − c̄_t|")
    for ax in axes2:
        ax.set_xlabel(rounds_label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig2, out_dir, f"fig2_perception_convergence{suffix}")

    # ----------------------------------------------------------
    # Figure 3 — Discussion vs Behavior
    # ----------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle(f"Figure 3 — Discussion vs Behavior{suffix}", fontsize=12)

    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df  = metrics_by_cond[cond]
        col = COLORS[cond]
        lbl = CONDITION_LABELS[cond]
        _line(axes3[0], df, "mean_contribution", lbl, col)
        _line(axes3[0], df, "mean_proposed",     f"{lbl} (proposed)", col, linestyle="--")
        _line(axes3[1], df, "disc_alignment",    lbl, col)

    axes3[0].set_title("Mean Contribution vs Mean Proposed m̄_t")
    axes3[1].set_title("Behavior–Discussion Alignment |m̄_t − c̄_t|")
    for ax in axes3:
        ax.set_xlabel(rounds_label)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig3, out_dir, f"fig3_discussion_behavior{suffix}")

    # ----------------------------------------------------------
    # Figure 4 — Selection Enforcement
    # ----------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    fig4.suptitle(f"Figure 4 — Selection Enforcement{suffix}", fontsize=12)

    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        _line(ax4, df, "contrib_weight_corr", CONDITION_LABELS[cond], COLORS[cond])

    ax4.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax4.set_title("Corr(c_i,t, W_i,t) — Contribution vs Incoming Weight")
    ax4.set_xlabel(rounds_label)
    ax4.set_ylabel("Pearson r")
    ax4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig4, out_dir, f"fig4_selection_enforcement{suffix}")

    # ----------------------------------------------------------
    # Figure 5 — Network Assortativity
    # ----------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    fig5.suptitle(f"Figure 5 — Network Assortativity by Contribution{suffix}", fontsize=12)

    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        _line(ax5, df, "assortativity", CONDITION_LABELS[cond], COLORS[cond])

    ax5.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax5.set_title("Assortativity_t (numeric, by contribution)")
    ax5.set_xlabel(rounds_label)
    ax5.set_ylabel("Assortativity coefficient")
    ax5.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig5, out_dir, f"fig5_assortativity{suffix}")


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ============================================================
# Summary table
# ============================================================

def print_summary(metrics_by_cond: dict):
    def _m(df, col):
        if col not in df.columns or df[col].isna().all():
            return "     nan"
        return f"{df[col].mean():8.3f}"

    print("\n" + "=" * 110)
    print(f"{'Condition':<20} {'c̄':>8} {'p̄':>8} {'Ī':>8} {'D̄':>8} "
          f"{'Var(I)':>8} {'E_t':>8} {'m̄':>8} {'A_t':>8} {'Corr':>8} {'Assort':>8}")
    print("=" * 110)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        print(
            f"{CONDITION_LABELS[cond]:<20}"
            f"{_m(df, 'mean_contribution')}"
            f"{_m(df, 'mean_payoff')}"
            f"{_m(df, 'mean_injunctive')}"
            f"{_m(df, 'mean_descriptive')}"
            f"{_m(df, 'var_injunctive')}"
            f"{_m(df, 'desc_error')}"
            f"{_m(df, 'mean_proposed')}"
            f"{_m(df, 'disc_alignment')}"
            f"{_m(df, 'contrib_weight_corr')}"
            f"{_m(df, 'assortativity')}"
        )
    print("=" * 110)


# ============================================================
# Cross-model seed43 comparison
# ============================================================

MODEL_LINE_STYLES = {
    "gpt":     {"color": "#2166ac", "linestyle": "-"},
    "llama":   {"color": "#d73027", "linestyle": "-"},
    "mistral": {"color": "#1a9641", "linestyle": "-"},
    "qwen":    {"color": "#f46d43", "linestyle": "-"},
}
VARIANT_DASH = {"global": "-", "local": "--"}


def load_seed43_all_models() -> dict:
    """
    For each model/variant, load the latest log per condition for seed43.
    Returns { (model, variant, condition): round_metrics_df }
    """
    from datetime import date
    results: dict = {}
    for model in ["gpt", "llama", "mistral", "qwen"]:
        for variant in ["global", "local"]:
            seed_dir = os.path.join(RESULTS_DIR, model, variant, "seed43")
            if not os.path.isdir(seed_dir):
                continue
            cond_best: dict = {}
            for path in sorted(glob.glob(os.path.join(seed_dir, "log_*.json"))):
                with open(path) as f:
                    data = json.load(f)
                cond_best[data["condition"]] = data
            for cond, data in cond_best.items():
                key = (model, variant, cond)
                results[key] = extract_round_metrics(data)
    return results


def make_compare_plots(metrics: dict):
    """
    One figure per condition, one line per model/variant.
    One figure per metric across all conditions (grid: conditions × models).
    Saved to figures/{today}/.
    """
    from datetime import date
    out_dir = os.path.join(os.path.dirname(RESULTS_DIR), "figures", str(date.today()))
    os.makedirs(out_dir, exist_ok=True)

    # Group by condition
    cond_data: dict = {}
    for (model, variant, cond), df in metrics.items():
        cond_data.setdefault(cond, {})[f"{model}_{variant}"] = df

    plot_specs = [
        ("mean_contribution",  "Mean Contribution",        "Contribution (0–10)"),
        ("mean_payoff",        "Mean Payoff",              "Payoff"),
        ("mean_injunctive",    "Mean Injunctive Norm",     "Injunctive norm (0–10)"),
        ("mean_descriptive",   "Mean Descriptive Norm",    "Descriptive norm (0–10)"),
        ("disc_alignment",     "Discussion Alignment",     "|m̄_t − c̄_t|"),
        ("contrib_weight_corr","Selection Correlation",    "corr(c, W_incoming)"),
    ]

    for cond in CONDITION_ORDER:
        if cond not in cond_data:
            continue
        model_dfs = cond_data[cond]

        n = len(plot_specs)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        fig.suptitle(
            f"Seed 43 — All Models — {CONDITION_LABELS.get(cond, cond)}",
            fontsize=13
        )

        for ax, (metric, title, ylabel) in zip(axes, plot_specs):
            for label, df in sorted(model_dfs.items()):
                model, variant = label.rsplit("_", 1)
                if metric not in df.columns or df[metric].isna().all():
                    continue
                color = MODEL_LINE_STYLES.get(model, {}).get("color", "#888")
                ls    = VARIANT_DASH.get(variant, "-")
                ax.plot(df["round"].to_numpy(), df[metric].to_numpy(),
                        label=label, color=color, linestyle=ls,
                        linewidth=2, marker="o", markersize=3)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"seed43_all_models_{cond}.png"
        path  = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
        plt.close(fig)

    # Also one figure per metric across all conditions (2×3 grid: conditions)
    for metric, title, ylabel in plot_specs:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()
        fig.suptitle(f"Seed 43 — {title} — All Conditions & Models", fontsize=13)

        for ax, cond in zip(axes_flat, CONDITION_ORDER):
            ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
            model_dfs = cond_data.get(cond, {})
            for label, df in sorted(model_dfs.items()):
                model, variant = label.rsplit("_", 1)
                if metric not in df.columns or df[metric].isna().all():
                    continue
                color = MODEL_LINE_STYLES.get(model, {}).get("color", "#888")
                ls    = VARIANT_DASH.get(variant, "-")
                ax.plot(df["round"].to_numpy(), df[metric].to_numpy(),
                        label=label, color=color, linestyle=ls,
                        linewidth=2, marker="o", markersize=3)
            ax.set_xlabel("Round")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

        # hide unused panels
        for ax in axes_flat[len(CONDITION_ORDER):]:
            ax.set_visible(False)

        plt.tight_layout()
        safe = metric.replace("/", "_")
        fname = f"seed43_all_models_by_condition_{safe}.png"
        path  = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        print(f"Saved → {path}")
        plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local",   action="store_true", help="Run on v5local logs")
    parser.add_argument("--llama",   action="store_true", help="Run on llama results")
    parser.add_argument("--mistral", action="store_true", help="Run on mistral results")
    parser.add_argument("--qwen",    action="store_true", help="Run on qwen results")
    parser.add_argument("--compare", action="store_true",
                        help="Plot seed43 across all models and variants")
    args = parser.parse_args()

    if args.compare:
        print("Loading seed43 logs for all models/variants...")
        metrics = load_seed43_all_models()
        if not metrics:
            print("No seed43 logs found.")
            exit(1)
        print(f"Loaded {len(metrics)} (model, variant, condition) combinations.")
        make_compare_plots(metrics)
    else:
        if args.local:
            pattern, suffix, out_dir = LOG_PATTERN_LOCAL,   "_local",   LOG_DIR
        elif args.llama:
            pattern, suffix, out_dir = LOG_PATTERN_LLAMA,   "_llama",   RESULTS_DIR
        elif args.mistral:
            pattern, suffix, out_dir = LOG_PATTERN_MISTRAL, "_mistral", RESULTS_DIR
        elif args.qwen:
            pattern, suffix, out_dir = LOG_PATTERN_QWEN,    "_qwen",    RESULTS_DIR
        else:
            pattern, suffix, out_dir = LOG_PATTERN_V5,      "",         LOG_DIR

        logs = load_logs(pattern)
        if not logs:
            print(f"No log files found matching: {pattern}")
            exit(1)

        print(f"Loaded conditions: {sorted(logs.keys())}")
        metrics_by_cond = {cond: extract_round_metrics(data) for cond, data in logs.items()}
        print_summary(metrics_by_cond)
        make_plots(metrics_by_cond, out_dir=out_dir, suffix=suffix)
