"""
appendix_evaluations.py
-----------------------
Appendix diagnostics for SoNoLiSi_v5 simulation logs.

Figures produced:
  A1. Behavioral Stability
        - contribution variance Var(c_t)
        - round-to-round drift |c̄_t - c̄_{t-1}|

  A2. Expectation Alignment
        - injunctive–descriptive gap |Ī_t - D̄_t|

  A3. Internalization
        - internalization gap G_t = mean_i |CT_i,t - I_i,t/E|
        - CT–behavior alignment B_t = mean_i |CT_i,t - c_i,t/E|

  A4. Partner Choice Signals
        - mean contribution of preferred partners Pref_t
        - mean contribution of avoided agents Avoid_t

  A5. Exclusion Dynamics
        - P(edge_loss | low_contribution) per round

  A6. Group-Level Cooperation
        - distribution of group mean contributions per round (box plots)

  A7. Cooperation Tendency Dynamics
        - mean CT̄_t
        - variance Var(CT_t)

Summary table: Var(c), Drift, Gap, G_t, B_t, Pref, Avoid, P(loss|low), CT̄, Var(CT)
"""

import json
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Configuration  (mirrors evaluation_outcomes.py)
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

def _incoming_weights(network_weights: list, agent_ids: list) -> dict:
    """Sum of incoming edge weights per agent."""
    totals = {aid: 0.0 for aid in agent_ids}
    for e in network_weights:
        v = e["v"]
        if v in totals:
            totals[v] += e["weight"]
    return totals


# ============================================================
# Extract per-round metrics
# ============================================================

def extract_round_metrics(data: dict) -> pd.DataFrame:
    round_logs = data["round_logs"]
    rows = []

    for idx, r in enumerate(round_logs):
        contribs = {int(k): v for k, v in r["contributions"].items()}
        c_vals   = list(contribs.values())
        mean_c   = np.mean(c_vals) if c_vals else np.nan

        # --- A1: Behavioral Stability ---
        var_c  = np.var(c_vals)  if len(c_vals) > 1 else np.nan
        if idx > 0:
            prev_c_vals = list({int(k): v for k, v in round_logs[idx - 1]["contributions"].items()}.values())
            drift = abs(mean_c - np.mean(prev_c_vals)) if prev_c_vals else np.nan
        else:
            drift = np.nan

        # --- A2 / A3: Perception, internalization ---
        inj_vals, desc_vals = [], []
        intern_gaps, ct_behav_aligns = [], []
        agent_cts = {a["id"]: a["cooperation_tendency"] for a in r["agent_states"]}

        if r.get("perceptions"):
            for aid_str, perc in r["perceptions"].items():
                if perc is None:
                    continue
                aid = int(aid_str)
                ct  = agent_cts.get(aid, np.nan)

                i_raw = perc.get("injunctive_norm")
                i_hat = None
                if i_raw is not None:
                    try:
                        i_hat = float(i_raw)
                        inj_vals.append(i_hat)
                    except (TypeError, ValueError):
                        pass

                d_raw = perc.get("descriptive_norm")
                if d_raw is not None:
                    try:
                        desc_vals.append(float(d_raw))
                    except (TypeError, ValueError):
                        pass

                # Internalization gap: |CT - I/E|
                if i_hat is not None and not np.isnan(ct):
                    intern_gaps.append(abs(ct - i_hat / ENDOWMENT))

                # CT–behavior alignment: |CT - c/E|
                c_i = contribs.get(aid)
                if c_i is not None and not np.isnan(ct):
                    ct_behav_aligns.append(abs(ct - c_i / ENDOWMENT))

        mean_inj  = np.mean(inj_vals)  if inj_vals  else np.nan
        mean_desc = np.mean(desc_vals) if desc_vals else np.nan
        inj_desc_gap    = abs(mean_inj - mean_desc) \
                          if not (np.isnan(mean_inj) or np.isnan(mean_desc)) else np.nan
        intern_gap      = np.mean(intern_gaps)    if intern_gaps    else np.nan
        ct_behav_align  = np.mean(ct_behav_aligns) if ct_behav_aligns else np.nan

        # --- A4: Partner Choice Signals ---
        pref_contribs, avoid_contribs = [], []
        if r.get("perceptions"):
            for aid_str, perc in r["perceptions"].items():
                if perc is None:
                    continue
                for j in (perc.get("preferred_partners") or []):
                    try:
                        c_j = contribs.get(int(j))
                        if c_j is not None:
                            pref_contribs.append(c_j)
                    except (TypeError, ValueError):
                        pass
                for j in (perc.get("agents_to_avoid") or []):
                    try:
                        c_j = contribs.get(int(j))
                        if c_j is not None:
                            avoid_contribs.append(c_j)
                    except (TypeError, ValueError):
                        pass

        mean_pref_contrib  = np.mean(pref_contribs)  if pref_contribs  else np.nan
        mean_avoid_contrib = np.mean(avoid_contribs) if avoid_contribs else np.nan

        # --- A5: Exclusion / Edge-Loss Dynamics ---
        edge_loss_prob = np.nan
        if idx + 1 < len(round_logs) and c_vals:
            q25 = np.percentile(c_vals, 25)
            low_agents = [aid for aid, c in contribs.items() if c <= q25]
            if low_agents:
                agent_ids = list(contribs.keys())
                w_now  = _incoming_weights(r["network_weights"], agent_ids)
                w_next = _incoming_weights(round_logs[idx + 1]["network_weights"], agent_ids)
                losses = [1 if w_next.get(aid, 0.0) < w_now.get(aid, 0.0) else 0
                          for aid in low_agents]
                edge_loss_prob = np.mean(losses)

        # --- A6: Group-Level Cooperation — store per-group means as a list ---
        groups      = r.get("groups", [])
        group_means = []
        for grp in groups:
            grp_ids = [int(x) for x in grp]
            vals    = [contribs[i] for i in grp_ids if i in contribs]
            if vals:
                group_means.append(np.mean(vals))

        # --- A7: CT Dynamics ---
        ct_vals  = list(agent_cts.values())
        mean_ct  = np.mean(ct_vals)  if ct_vals else np.nan
        var_ct   = np.var(ct_vals)   if len(ct_vals) > 1 else np.nan

        rows.append({
            "round":            r["round"],
            # A1
            "var_contrib":      var_c,
            "drift":            drift,
            # A2
            "inj_desc_gap":     inj_desc_gap,
            # A3
            "intern_gap":       intern_gap,
            "ct_behav_align":   ct_behav_align,
            # A4
            "mean_pref_contrib":  mean_pref_contrib,
            "mean_avoid_contrib": mean_avoid_contrib,
            # A5
            "edge_loss_prob":   edge_loss_prob,
            # A6 — stored as object column for box plot
            "group_means":      group_means,
            # A7
            "mean_ct":          mean_ct,
            "var_ct":           var_ct,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting helpers
# ============================================================

def _line(ax, df, col, label, color, **kwargs):
    vals = df[col].to_numpy()
    if not np.all(np.isnan(vals)):
        ax.plot(df["round"].to_numpy(), vals, label=label, color=color,
                linewidth=2, marker="o", markersize=4, **kwargs)


def _fmt_ax(ax, title, ylabel="", xlabel="Round"):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _save(fig, out_dir, name):
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ============================================================
# Make all appendix figures
# ============================================================

def make_plots(metrics_by_cond: dict, out_dir: str, suffix: str = ""):

    # ----------------------------------------------------------
    # A1 — Behavioral Stability
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"A1 — Behavioral Stability{suffix}", fontsize=12)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df, col, lbl = metrics_by_cond[cond], COLORS[cond], CONDITION_LABELS[cond]
        _line(axes[0], df, "var_contrib", lbl, col)
        _line(axes[1], df, "drift",       lbl, col)
    _fmt_ax(axes[0], "Contribution Variance Var(c_t)")
    _fmt_ax(axes[1], "Round-to-Round Drift |c̄_t − c̄_{t-1}|")
    plt.tight_layout()
    _save(fig, out_dir, f"appA1_behavioral_stability{suffix}")

    # ----------------------------------------------------------
    # A2 — Expectation Alignment
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"A2 — Expectation Alignment{suffix}", fontsize=12)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        _line(ax, df, "inj_desc_gap", CONDITION_LABELS[cond], COLORS[cond])
    _fmt_ax(ax, "Injunctive–Descriptive Gap |Ī_t − D̄_t|")
    plt.tight_layout()
    _save(fig, out_dir, f"appA2_expectation_alignment{suffix}")

    # ----------------------------------------------------------
    # A3 — Internalization
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"A3 — Internalization{suffix}", fontsize=12)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df, col, lbl = metrics_by_cond[cond], COLORS[cond], CONDITION_LABELS[cond]
        _line(axes[0], df, "intern_gap",     lbl, col)
        _line(axes[1], df, "ct_behav_align", lbl, col)
    _fmt_ax(axes[0], "Internalization Gap G_t = mean|CT − I/E|")
    _fmt_ax(axes[1], "CT–Behavior Alignment B_t = mean|CT − c/E|")
    plt.tight_layout()
    _save(fig, out_dir, f"appA3_internalization{suffix}")

    # ----------------------------------------------------------
    # A4 — Partner Choice Signals
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"A4 — Partner Choice Signals{suffix}", fontsize=12)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df, col, lbl = metrics_by_cond[cond], COLORS[cond], CONDITION_LABELS[cond]
        _line(ax, df, "mean_pref_contrib",  f"{lbl} (preferred)", col)
        _line(ax, df, "mean_avoid_contrib", f"{lbl} (avoided)",   col, linestyle="--")
    _fmt_ax(ax, "Mean Contribution: Preferred vs Avoided Partners",
            ylabel="Mean contribution")
    plt.tight_layout()
    _save(fig, out_dir, f"appA4_partner_choice{suffix}")

    # ----------------------------------------------------------
    # A5 — Exclusion Dynamics
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"A5 — Exclusion Dynamics{suffix}", fontsize=12)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        _line(ax, df, "edge_loss_prob", CONDITION_LABELS[cond], COLORS[cond])
    ax.set_ylim(-0.05, 1.05)
    _fmt_ax(ax, "P(edge loss | low contribution)",
            ylabel="Probability")
    plt.tight_layout()
    _save(fig, out_dir, f"appA5_exclusion_dynamics{suffix}")

    # ----------------------------------------------------------
    # A6 — Group-Level Cooperation (box plots per round, one subplot per condition)
    # ----------------------------------------------------------
    n_conds = sum(1 for c in CONDITION_ORDER if c in metrics_by_cond)
    fig, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 5), sharey=True)
    if n_conds == 1:
        axes = [axes]
    fig.suptitle(f"A6 — Group-Level Cooperation{suffix}", fontsize=12)
    ax_idx = 0
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df  = metrics_by_cond[cond]
        ax  = axes[ax_idx]; ax_idx += 1
        rounds      = df["round"].tolist()
        group_data  = df["group_means"].tolist()
        # only keep rounds that have group data
        valid = [(t, gm) for t, gm in zip(rounds, group_data) if gm]
        if valid:
            ts, gms = zip(*valid)
            ax.boxplot(gms, positions=list(ts), widths=0.6,
                       patch_artist=True,
                       boxprops=dict(facecolor=COLORS[cond], alpha=0.6),
                       medianprops=dict(color="black"))
        ax.set_title(CONDITION_LABELS[cond], fontsize=9)
        ax.set_xlabel("Round")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Group mean contribution")
    plt.tight_layout()
    _save(fig, out_dir, f"appA6_group_cooperation{suffix}")

    # ----------------------------------------------------------
    # A7 — Cooperation Tendency Dynamics
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"A7 — Cooperation Tendency Dynamics{suffix}", fontsize=12)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df, col, lbl = metrics_by_cond[cond], COLORS[cond], CONDITION_LABELS[cond]
        _line(axes[0], df, "mean_ct", lbl, col)
        _line(axes[1], df, "var_ct",  lbl, col)
    _fmt_ax(axes[0], "Mean Cooperation Tendency CT̄_t")
    _fmt_ax(axes[1], "CT Variance Var(CT_t)")
    plt.tight_layout()
    _save(fig, out_dir, f"appA7_ct_dynamics{suffix}")


# ============================================================
# Summary table
# ============================================================

def print_summary(metrics_by_cond: dict):
    def _m(df, col):
        if col not in df.columns or df[col].isna().all():
            return "     nan"
        return f"{df[col].mean():8.3f}"

    print("\n" + "=" * 120)
    print(f"{'Condition':<20} {'Var(c)':>8} {'Drift':>8} {'Gap':>8} "
          f"{'G_t':>8} {'B_t':>8} {'Pref':>8} {'Avoid':>8} "
          f"{'P(loss)':>9} {'CT̄':>8} {'Var(CT)':>8}")
    print("=" * 120)
    for cond in CONDITION_ORDER:
        if cond not in metrics_by_cond:
            continue
        df = metrics_by_cond[cond]
        print(
            f"{CONDITION_LABELS[cond]:<20}"
            f"{_m(df, 'var_contrib')}"
            f"{_m(df, 'drift')}"
            f"{_m(df, 'inj_desc_gap')}"
            f"{_m(df, 'intern_gap')}"
            f"{_m(df, 'ct_behav_align')}"
            f"{_m(df, 'mean_pref_contrib')}"
            f"{_m(df, 'mean_avoid_contrib')}"
            f"{_m(df, 'edge_loss_prob'):>9}"
            f"{_m(df, 'mean_ct')}"
            f"{_m(df, 'var_ct')}"
        )
    print("=" * 120)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local",   action="store_true", help="Run on v5local logs")
    parser.add_argument("--llama",   action="store_true", help="Run on llama results")
    parser.add_argument("--mistral", action="store_true", help="Run on mistral results")
    parser.add_argument("--qwen",    action="store_true", help="Run on qwen results")
    args = parser.parse_args()

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
