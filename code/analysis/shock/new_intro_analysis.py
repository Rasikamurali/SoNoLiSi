"""
new_intro_analysis.py
---------------------
Analyzes how a newly introduced agent (round 21) adapts to an established group,
and how incumbents respond.

Three analyses:
  1. Behavior     — contribution and cooperation tendency over rounds
  2. Perception   — injunctive norm (IN) and descriptive norm (DN) over rounds
  3. Alignment    — IN-gap (IN − contribution) and DN-gap (DN − contribution)

For each metric, trajectories are split by agent type:
  new      — the single newly introduced agent
  incumbent — all pre-existing agents

Aggregation: mean across agents within each seed → seed-level series → cross-seed mean ± SE.
A vertical dashed line marks round 21 (intro round).

Output:
  figures/new_intro/behavior_trajectories.pdf/.png
  figures/new_intro/perception_trajectories.pdf/.png
  figures/new_intro/alignment_trajectories.pdf/.png
  figures/new_intro/paper_stats/new_intro_summary.tex
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/new_intro"
STAT_DIR = os.path.join(FIG_ROOT, "paper_stats")

MODEL       = "gpt"
SEEDS       = list(range(46, 53))
CONDITIONS  = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
PERC_CONDS  = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
ALL_CONDS   = CONDITIONS  # PURE_BASELINE excluded (no perceptions, no norm formation)
ROUNDS      = list(range(1, 26))   # 20 normal + 5 extension
INTRO_ROUND = 21
NORMAL_ROUNDS    = list(range(1, INTRO_ROUND))
EXTENSION_ROUNDS = list(range(INTRO_ROUND, 26))

COND_LABELS = {
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
COND_SHORT = {
    "BASELINE":      "B",
    "NO_SELECTION":  "NS",
    "NO_DISCUSSION": "ND",
    "FULL":          "F",
}

COLORS = {"new": "#e74c3c", "incumbent": "#2980b9"}
LABELS = {"new": "New agent", "incumbent": "Incumbents"}

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(seed, condition):
    pattern = os.path.join(RESULTS, MODEL, "new_intro", f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(p))
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def get_new_agent_id(d):
    for rlog in d["round_logs"]:
        new_id = rlog.get("new_agent_introduced")
        if new_id is not None:
            return str(new_id)
    return None


def compute_seed_series(seed, condition):
    """
    Returns dict: {round: {"new": {metric: float}, "incumbent": {metric: float}}}
    Metrics: contribution, cooperation_tendency, injunctive_norm, descriptive_norm,
             in_gap, dn_gap
    """
    d = load_log(seed, condition)
    if d is None:
        return {}
    new_id = get_new_agent_id(d)
    result = {}

    for rlog in d["round_logs"]:
        rnd   = rlog["round"]
        contribs = {str(k): float(v) for k, v in rlog["contributions"].items()}
        percs    = rlog.get("perceptions") or {}
        states   = {str(s["id"]): s for s in (rlog.get("agent_states") or [])}

        # Split agent IDs: new agent only exists from intro round onward
        if rnd < INTRO_ROUND:
            inc_ids = list(contribs.keys())
            new_ids = []
        else:
            new_ids = [new_id] if new_id in contribs else []
            inc_ids = [k for k in contribs if k != new_id]

        def agent_metrics(ids):
            if not ids:
                return None
            cs, ins, dns, cts = [], [], [], []
            for aid in ids:
                c = contribs.get(aid)
                if c is not None:
                    cs.append(c)
                p = percs.get(aid) or {}
                inj  = p.get("injunctive_norm")
                desc = p.get("descriptive_norm")
                if inj  is not None: ins.append(float(inj))
                if desc is not None: dns.append(float(desc))
                st = states.get(aid)
                if st and "cooperation_tendency" in st:
                    cts.append(float(st["cooperation_tendency"]))

            out = {}
            if cs:
                out["contribution"] = float(np.mean(cs))
            if ins:
                out["injunctive_norm"]  = float(np.mean(ins))
                if cs:
                    out["in_gap"] = float(np.mean(ins)) - float(np.mean(cs))
            if dns:
                out["descriptive_norm"] = float(np.mean(dns))
                if cs:
                    out["dn_gap"] = float(np.mean(dns)) - float(np.mean(cs))
            if cts:
                out["cooperation_tendency"] = float(np.mean(cts))
            return out

        result[rnd] = {
            "incumbent": agent_metrics(inc_ids),
            "new":       agent_metrics(new_ids),
        }
    return result


def build_store():
    store = {c: {} for c in ALL_CONDS}
    for cond in ALL_CONDS:
        for seed in SEEDS:
            s = compute_seed_series(seed, cond)
            if s:
                store[cond][seed] = s
    return store


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def extract_series(store, cond, agent_type, metric):
    """Returns list of arrays (one per seed), length = len(ROUNDS), NaN where missing."""
    series = []
    for seed_data in store[cond].values():
        arr = []
        for r in ROUNDS:
            rdata = seed_data.get(r, {}).get(agent_type)
            arr.append(rdata.get(metric, np.nan) if rdata else np.nan)
        series.append(np.array(arr, dtype=float))
    return series


def mean_se(series_list):
    arr  = np.array(series_list, dtype=float)
    n    = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.where(n > 1, n, np.nan))
    return mean, se


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def make_trajectory_figure(store, metric_pairs, row_labels, fname, suptitle):
    """
    metric_pairs: list of (metric_key, y_label) per row
    One row per metric pair, one column per condition.
    """
    n_rows = len(metric_pairs)
    n_cols = len(ALL_CONDS)
    rounds = np.array(ROUNDS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows),
                             sharex=True, sharey="row")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, cond in enumerate(ALL_CONDS):
        for row, (metric, ylabel) in enumerate(metric_pairs):
            ax = axes[row, col]
            for atype in ["incumbent", "new"]:
                series = extract_series(store, cond, atype, metric)
                if not series:
                    continue
                mean, se = mean_se(series)
                valid = ~np.isnan(mean)
                if not valid.any():
                    continue
                color = COLORS[atype]
                ax.plot(rounds[valid], mean[valid],
                        color=color, linewidth=2, label=LABELS[atype])
                ax.fill_between(rounds[valid],
                                (mean - se)[valid], (mean + se)[valid],
                                color=color, alpha=0.15)

            ax.axvline(INTRO_ROUND, color="black", linestyle="--",
                       linewidth=1.2, alpha=0.6)
            ax.axvspan(INTRO_ROUND, ROUNDS[-1] + 0.5, alpha=0.05, color="gray")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=13)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel("Round", fontsize=11)

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_ROOT, f"{fname}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Summary statistics table ─────────────────────────────────────────────────

def phase_mean(store, cond, agent_type, metric, rounds_subset):
    vals = []
    for seed_data in store[cond].values():
        rvals = []
        for r in rounds_subset:
            rdata = seed_data.get(r, {}).get(agent_type)
            v = rdata.get(metric, np.nan) if rdata else np.nan
            if not np.isnan(v):
                rvals.append(v)
        if rvals:
            vals.append(float(np.mean(rvals)))
    return np.array(vals)


def ttest_new_vs_inc(store, cond, metric, rounds_subset):
    """Paired t-test (per-seed means) comparing new agent to incumbents."""
    new_vals, inc_vals = [], []
    for seed_data in store[cond].values():
        n_rvals, i_rvals = [], []
        for r in rounds_subset:
            nd = seed_data.get(r, {}).get("new")
            id_ = seed_data.get(r, {}).get("incumbent")
            if nd:
                v = nd.get(metric, np.nan)
                if not np.isnan(v): n_rvals.append(v)
            if id_:
                v = id_.get(metric, np.nan)
                if not np.isnan(v): i_rvals.append(v)
        if n_rvals and i_rvals:
            new_vals.append(np.mean(n_rvals))
            inc_vals.append(np.mean(i_rvals))
    if len(new_vals) < 3:
        return np.nan, np.nan
    diff = np.array(new_vals) - np.array(inc_vals)
    t, p = stats.ttest_1samp(diff, 0)
    return float(diff.mean()), float(p)


def make_summary_table(store):
    metrics = [
        ("contribution",        "Contribution"),
        ("cooperation_tendency","Coop. Tendency"),
        ("injunctive_norm",     "IN"),
        ("descriptive_norm",    "DN"),
        ("in_gap",              "IN gap"),
        ("dn_gap",              "DN gap"),
    ]

    def stars(p):
        if np.isnan(p):    return ""
        if p < 0.001:      return r"$^{***}$"
        if p < 0.01:       return r"$^{**}$"
        if p < 0.05:       return r"$^{*}$"
        return ""

    def cell(vals):
        if len(vals) == 0: return "---"
        return f"{np.mean(vals):.3f}"

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll" + "r" * (len(ALL_CONDS) * 2) + "}")
    lines.append(r"\toprule")

    # Header: condition pairs
    cond_hdr = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{COND_LABELS[c]}}}}}"
        for c in ALL_CONDS
    )
    lines.append(r"Metric & Agent & " + cond_hdr + r" \\")

    subcol_hdr = " & ".join([r"New & Inc."] * len(ALL_CONDS))
    lines.append(r" & & " + subcol_hdr + r" \\")
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}")

    # Normal phase rows
    lines.append(r"\multicolumn{" + str(2 + len(ALL_CONDS) * 2) + r"}{l}{\textit{Normal phase (rounds 1–20, incumbents only)}} \\")
    lines.append(r"\midrule")
    for metric, mlabel in metrics:
        if metric in ("injunctive_norm", "descriptive_norm", "in_gap", "dn_gap"):
            pass  # include all
        cells = []
        for cond in ALL_CONDS:
            inc = phase_mean(store, cond, "incumbent", metric, NORMAL_ROUNDS)
            cells.append("--- & " + cell(inc))
        lines.append(rf"{mlabel} & Inc. & " + " & ".join(cells) + r" \\")

    # Extension phase rows
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(2 + len(ALL_CONDS) * 2) + r"}{l}{\textit{Extension phase (rounds 21–25)}} \\")
    lines.append(r"\midrule")
    for metric, mlabel in metrics:
        cells_new, cells_inc, cells_diff = [], [], []
        for cond in ALL_CONDS:
            nw  = phase_mean(store, cond, "new",      metric, EXTENSION_ROUNDS)
            inc = phase_mean(store, cond, "incumbent", metric, EXTENSION_ROUNDS)
            diff, p = ttest_new_vs_inc(store, cond, metric, EXTENSION_ROUNDS)
            cells_new.append(cell(nw))
            cells_inc.append(cell(inc))
        # New agent row
        row_new = " & ".join(
            f"{cells_new[i]} & {cells_inc[i]}"
            for i in range(len(ALL_CONDS))
        )
        lines.append(rf"{mlabel} & New vs Inc. & " + row_new + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{New-agent introduction analysis. "
                 r"\textit{Normal phase}: mean over rounds 1--20 for incumbents. "
                 r"\textit{Extension phase}: mean over rounds 21--25 for new agent and incumbents. "
                 r"IN = injunctive norm; DN = descriptive norm; gap = norm $-$ contribution.}")
    lines.append(r"\label{tab:new_intro_summary}")
    lines.append(r"\end{table*}")

    path = os.path.join(STAT_DIR, "new_intro_summary.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


def make_gap_table(store):
    """
    Compact table: extension-phase mean for new vs incumbent, with t-test stars
    for new ≠ incumbent difference.
    Rows: conditions. Cols: metrics × {new, inc, diff*}.
    """
    metrics = [
        ("contribution",    "Contrib."),
        ("injunctive_norm", "IN"),
        ("descriptive_norm","DN"),
        ("in_gap",          "IN gap"),
        ("dn_gap",          "DN gap"),
    ]

    def stars(p):
        if np.isnan(p):  return ""
        if p < 0.001:    return r"***"
        if p < 0.01:     return r"**"
        if p < 0.05:     return r"*"
        return ""

    def cell(vals):
        return f"{np.mean(vals):.2f}" if len(vals) else "---"

    n_metrics = len(metrics)
    col_spec = "l" + "rrr" * n_metrics

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Metric header
    mhdr = " & ".join(
        rf"\multicolumn{{3}}{{c}}{{\textbf{{{ml}}}}}"
        for _, ml in metrics
    )
    lines.append(r"Condition & " + mhdr + r" \\")

    # Sub-header: New / Inc. / Δ*
    subhdr = " & ".join([r"New & Inc. & $\Delta$"] * n_metrics)
    lines.append(r" & " + subhdr + r" \\")

    # Cmidrule per metric block (cols 2-4, 5-7, ...)
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}"
        for i in range(n_metrics)
    )
    lines.append(cmidrules)

    for cond in ALL_CONDS:
        row_cells = []
        for metric, _ in metrics:
            nw  = phase_mean(store, cond, "new",       metric, EXTENSION_ROUNDS)
            inc = phase_mean(store, cond, "incumbent",  metric, EXTENSION_ROUNDS)
            diff, p = ttest_new_vs_inc(store, cond, metric, EXTENSION_ROUNDS)
            delta_str = (f"{diff:+.2f}{stars(p)}" if not np.isnan(diff) else "---")
            row_cells.append(f"{cell(nw)} & {cell(inc)} & {delta_str}")
        lines.append(rf"\textbf{{{COND_LABELS[cond]}}} & " + " & ".join(row_cells) + r" \\")

    lines.append(r"\midrule")
    # Normal phase incumbents for reference
    lines.append(r"\multicolumn{" + str(1 + n_metrics * 3) + r"}{l}{\textit{Incumbent baseline (normal phase, rounds 1--20)}} \\")
    for cond in ALL_CONDS:
        row_cells = []
        for metric, _ in metrics:
            inc = phase_mean(store, cond, "incumbent", metric, NORMAL_ROUNDS)
            row_cells.append(rf"--- & {cell(inc)} & ---")
        lines.append(rf"\textit{{{COND_LABELS[cond]}}} & " + " & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Extension-phase means (rounds 21--25) for newly introduced agent (New) "
                 r"and incumbents (Inc.), with difference $\Delta = \text{New} - \text{Inc.}$ "
                 r"tested via paired $t$-test across seeds. "
                 r"Incumbent baseline (normal phase) shown for reference. "
                 r"$^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.}")
    lines.append(r"\label{tab:new_intro_gap}")
    lines.append(r"\end{table*}")

    path = os.path.join(STAT_DIR, "new_intro_gap.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading new_intro data …")
    store = build_store()
    for cond in ALL_CONDS:
        n = len(store[cond])
        print(f"  {cond:20s}: {n} seeds")

    print("\nPlotting behavioral trajectories …")
    make_trajectory_figure(
        store,
        metric_pairs=[
            ("contribution",         "Mean contribution"),
            ("cooperation_tendency", "Cooperation tendency"),
        ],
        row_labels=["Contribution", "Coop. tendency"],
        fname="behavior_trajectories",
        suptitle="Behavioral trajectories: new agent vs. incumbents",
    )

    print("\nPlotting perception trajectories …")
    make_trajectory_figure(
        store,
        metric_pairs=[
            ("injunctive_norm",  "Injunctive norm (IN)"),
            ("descriptive_norm", "Descriptive norm (DN)"),
        ],
        row_labels=["IN", "DN"],
        fname="perception_trajectories",
        suptitle="Perception trajectories: new agent vs. incumbents",
    )

    print("\nPlotting alignment trajectories …")
    make_trajectory_figure(
        store,
        metric_pairs=[
            ("in_gap", "IN gap (IN − contribution)"),
            ("dn_gap", "DN gap (DN − contribution)"),
        ],
        row_labels=["IN gap", "DN gap"],
        fname="alignment_trajectories",
        suptitle="Perception–behavior alignment: new agent vs. incumbents",
    )

    print("\nGenerating summary tables …")
    make_gap_table(store)

    print("\nDone.")
