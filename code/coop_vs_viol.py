"""
coop_vs_viol.py
---------------
Trajectory analysis for Cooperators vs Violators (CT-based groups).

Groups assigned from initial (round-1) CT — stable across rounds:
  Violators   CT ∈ [0.0, 0.6)
  Cooperators CT ∈ [0.6, 1.0]

Trajectory plots (cross-model, 2 × 4 or 1 × 4):
  A — Contribution & Payoff (2 × 4: row = metric, col = model)
  B — Perception gap        (2 × 4: row = IN/DN gap, col = model)
  C — Incoming weights      (1 × 4: col = model)

Per-model diagnostic plots:
  Plot 1 — Perception vs Actual scatter
  Plot 2 — Bias by CT group (bar chart)
  Plot 3 — Error distribution by CT group (histogram)
  Figure 1 — Perception alignment trajectory (IN/DN/actual per CT group)

Output: figures/2026-03-22/coop_viol/
        figures/2026-03-22/coop_viol/<model>/  (per-model diagnostics)
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"
MODELS   = ["gpt", "llama", "mistral", "qwen"]
VARIANT  = "global"
SEEDS    = list(range(43, 53))
ROUNDS   = list(range(1, 21))
ENDOWMENT = 10.0

ALL_CONDITIONS  = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
PERC_CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
SEL_CONDITIONS  = ["NO_DISCUSSION", "FULL"]
DISC_CONDITIONS = ["NO_SELECTION", "FULL"]   # conditions with discussion (for per-model figs)

CT_GROUPS = {
    "Violators":   (0.0, 0.6),
    "Cooperators": (0.6, 1.01),
}
CT_ORDER  = ["Violators", "Cooperators"]
CT_COLORS = {"Violators": "#d62728", "Cooperators": "#2ca02c"}

COND_COLORS = {
    "PURE_BASELINE": "#7f7f7f",
    "BASELINE":      "#bcbd22",
    "NO_SELECTION":  "#1f77b4",
    "NO_DISCUSSION": "#9467bd",
    "FULL":          "#2ca02c",
}
COND_LS = {
    "PURE_BASELINE": ":",
    "BASELINE":      "-.",
    "NO_SELECTION":  "--",
    "NO_DISCUSSION": (0, (5, 2)),
    "FULL":          "-",
}
COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline",
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}

LABEL_SIZE  = 13
TICK_SIZE   = 11
LEGEND_SIZE = 10

# ─── Core helpers ─────────────────────────────────────────────────────────────

def load_log(model, seed, condition):
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


def load_log_variant(model, variant, seed, condition):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            if d.get("condition") == condition:
                best[p] = d
        except Exception:
            continue
    if not best:
        return None
    return best[sorted(best)[-1]]


def ct_group(ct):
    for name, (lo, hi) in CT_GROUPS.items():
        if lo <= ct < hi:
            return name
    return None


def get_initial_ct(d):
    r1 = next((r for r in d["round_logs"] if r["round"] == 1), None)
    if r1 is None:
        return {}
    return {a["id"]: a["cooperation_tendency"] for a in r1["agent_states"]}


def mean_se(series_list):
    arr  = np.array(series_list, dtype=float)
    n    = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, se


def savefig(fig, fname):
    out_dir = os.path.join(FIG_ROOT, "coop_viol")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, fname.replace(".png", f".{ext}"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


def savefig_model(fig, model, fname):
    out_dir = os.path.join(FIG_ROOT, "coop_viol", model)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, fname.replace(".png", f".{ext}"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Store builders ───────────────────────────────────────────────────────────

def build_behavior_store(conditions=ALL_CONDITIONS):
    """
    store[model][cond][metric][group] = list of seed-series arrays (one per seed, len=ROUNDS)
    metric: 'contribution', 'payoff'
    """
    store = {m: {c: {"contribution": {g: [] for g in CT_ORDER},
                      "payoff":       {g: [] for g in CT_ORDER}}
                 for c in conditions}
             for m in MODELS}

    for model in MODELS:
        for cond in conditions:
            for seed in SEEDS:
                d = load_log(model, seed, cond)
                if d is None:
                    continue
                ict = get_initial_ct(d)
                if not ict:
                    continue

                grp_cont = defaultdict(lambda: defaultdict(list))
                grp_pay  = defaultdict(lambda: defaultdict(list))

                for r in d["round_logs"]:
                    rnd  = r["round"]
                    cont = {int(k): float(v) for k, v in r["contributions"].items()}
                    pays = {int(k): float(v) for k, v in (r.get("payoffs") or {}).items()}
                    for aid, ct0 in ict.items():
                        grp = ct_group(ct0)
                        if grp is None:
                            continue
                        if aid in cont:
                            grp_cont[grp][rnd].append(cont[aid])
                        if aid in pays:
                            grp_pay[grp][rnd].append(pays[aid])

                for grp in CT_ORDER:
                    arr_c = np.array([np.nanmean(grp_cont[grp][r]) if grp_cont[grp][r] else np.nan
                                      for r in ROUNDS])
                    arr_p = np.array([np.nanmean(grp_pay[grp][r])  if grp_pay[grp][r]  else np.nan
                                      for r in ROUNDS])
                    store[model][cond]["contribution"][grp].append(arr_c)
                    store[model][cond]["payoff"][grp].append(arr_p)

    return store


def build_gap_store(conditions=PERC_CONDITIONS):
    """
    store[model][cond][gap][group] = list of seed-series arrays
    gap: 'IN_gap', 'DN_gap'
    """
    store = {m: {c: {"IN_gap": {g: [] for g in CT_ORDER},
                      "DN_gap": {g: [] for g in CT_ORDER}}
                 for c in conditions}
             for m in MODELS}

    for model in MODELS:
        for cond in conditions:
            for seed in SEEDS:
                d = load_log(model, seed, cond)
                if d is None:
                    continue
                ict = get_initial_ct(d)
                if not ict:
                    continue

                grp_in = defaultdict(lambda: defaultdict(list))
                grp_dn = defaultdict(lambda: defaultdict(list))

                for r in d["round_logs"]:
                    rnd  = r["round"]
                    cont = {int(k): float(v) for k, v in r["contributions"].items()}
                    perc = {int(k): v for k, v in (r.get("perceptions") or {}).items()}
                    for aid, ct0 in ict.items():
                        grp    = ct_group(ct0)
                        actual = cont.get(aid)
                        if grp is None or actual is None:
                            continue
                        p   = perc.get(aid) or {}
                        inj  = p.get("injunctive_norm")
                        desc = p.get("descriptive_norm")
                        if inj  is not None:
                            grp_in[grp][rnd].append(float(inj) - actual)
                        if desc is not None:
                            grp_dn[grp][rnd].append(float(desc) - actual)

                for grp in CT_ORDER:
                    arr_in = np.array([np.nanmean(grp_in[grp][r]) if grp_in[grp][r] else np.nan
                                       for r in ROUNDS])
                    arr_dn = np.array([np.nanmean(grp_dn[grp][r]) if grp_dn[grp][r] else np.nan
                                       for r in ROUNDS])
                    store[model][cond]["IN_gap"][grp].append(arr_in)
                    store[model][cond]["DN_gap"][grp].append(arr_dn)

    return store


def build_weight_store(conditions=SEL_CONDITIONS):
    """
    store[model][cond][group] = list of seed-series arrays (mean incoming weight per round)
    """
    store = {m: {c: {g: [] for g in CT_ORDER} for c in conditions} for m in MODELS}

    for model in MODELS:
        for cond in conditions:
            for seed in SEEDS:
                d = load_log(model, seed, cond)
                if d is None:
                    continue
                ict = get_initial_ct(d)
                if not ict:
                    continue

                grp_wt = defaultdict(lambda: defaultdict(list))

                for r in d["round_logs"]:
                    rnd = r["round"]
                    nw  = r.get("network_weights") or []
                    iw  = defaultdict(float)
                    for e in nw:
                        iw[int(e["v"])] += float(e["weight"])
                    for aid, ct0 in ict.items():
                        grp = ct_group(ct0)
                        if grp is None:
                            continue
                        grp_wt[grp][rnd].append(iw.get(aid, 0.0))

                for grp in CT_ORDER:
                    arr = np.array([np.nanmean(grp_wt[grp][r]) if grp_wt[grp][r] else np.nan
                                    for r in ROUNDS])
                    store[model][cond][grp].append(arr)

    return store


# ─── Shared legend builder ────────────────────────────────────────────────────

def make_shared_legend(conditions):
    """
    Returns list of proxy artists for a shared figure legend:
      • CT group colour patches
      • Condition line style entries
    """
    handles = []
    for grp in CT_ORDER:
        handles.append(mpatches.Patch(color=CT_COLORS[grp], label=grp))
    handles.append(mpatches.Patch(color="none", label=""))   # spacer
    for cond in conditions:
        handles.append(mlines.Line2D([], [], color="grey", ls=COND_LS[cond],
                                     lw=1.8, label=COND_LABELS[cond]))
    return handles


# ─── Plot A: Contribution & Payoff Trajectories (2 × 4) ──────────────────────

def plot_contribution_payoff(bstore):
    """
    2 rows (contribution / payoff) × 4 cols (models).
    Lines: CT group colour × condition linestyle, ALL_CONDITIONS.
    """
    print("\nPlotting contribution & payoff trajectories …")
    metrics = [("contribution", "Contribution"), ("payoff", "Payoff")]
    rounds  = np.array(ROUNDS)

    fig, axes = plt.subplots(len(metrics), len(MODELS),
                             figsize=(4 * len(MODELS), 4 * len(metrics)),
                             sharex=True)

    for row, (metric, ylabel) in enumerate(metrics):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            for grp in CT_ORDER:
                color = CT_COLORS[grp]
                for cond in ALL_CONDITIONS:
                    series = bstore[model][cond][metric][grp]
                    if not series:
                        continue
                    mean, se = mean_se(series)
                    ax.plot(rounds, mean, color=color, ls=COND_LS[cond], lw=1.8)
                    ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.08)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=LABEL_SIZE)
            if row == len(metrics) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles = make_shared_legend(ALL_CONDITIONS)
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CT_ORDER) + 1 + len(ALL_CONDITIONS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, "contribution_payoff.png")


# ─── Plot B: Perception Gap Trajectories (2 × 4) ─────────────────────────────

def plot_perception_gaps(gstore):
    """
    2 rows (IN_gap / DN_gap) × 4 cols (models).
    Lines: CT group colour × condition linestyle, PERC_CONDITIONS.
    """
    print("\nPlotting perception gap trajectories …")
    gaps   = [("IN_gap", "IN − Actual"), ("DN_gap", "DN − Actual")]
    rounds = np.array(ROUNDS)

    fig, axes = plt.subplots(len(gaps), len(MODELS),
                             figsize=(4 * len(MODELS), 4 * len(gaps)),
                             sharex=True)

    for row, (gap, ylabel) in enumerate(gaps):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            for grp in CT_ORDER:
                color = CT_COLORS[grp]
                for cond in PERC_CONDITIONS:
                    series = gstore[model][cond][gap][grp]
                    if not series:
                        continue
                    mean, se = mean_se(series)
                    ax.plot(rounds, mean, color=color, ls=COND_LS[cond], lw=1.8)
                    ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.08)

            ax.axhline(0, color="black", lw=0.8, alpha=0.5, linestyle=":")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=LABEL_SIZE)
            if row == len(gaps) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles = make_shared_legend(PERC_CONDITIONS)
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CT_ORDER) + 1 + len(PERC_CONDITIONS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, "perception_gaps.png")


# ─── Plot C: Incoming Weight Trajectories (1 × 4) ────────────────────────────

def plot_incoming_weights(wstore):
    """
    1 row × 4 cols (models).
    Lines: CT group colour × condition linestyle, SEL_CONDITIONS.
    """
    print("\nPlotting incoming weight trajectories …")
    rounds = np.array(ROUNDS)

    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(4 * len(MODELS), 4),
                             sharex=True, sharey=True)

    for col, model in enumerate(MODELS):
        ax = axes[col]
        for grp in CT_ORDER:
            color = CT_COLORS[grp]
            for cond in SEL_CONDITIONS:
                series = wstore[model][cond][grp]
                if not series:
                    continue
                mean, se = mean_se(series)
                ax.plot(rounds, mean, color=color, ls=COND_LS[cond], lw=2)
                ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.1)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, alpha=0.25)
        ax.set_title(MODEL_LABELS[model], fontsize=LABEL_SIZE)
        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        if col == 0:
            ax.set_ylabel("Mean Incoming Weight", fontsize=LABEL_SIZE)

    handles = make_shared_legend(SEL_CONDITIONS)
    fig.legend(handles=handles, loc="lower center",
               ncol=len(CT_ORDER) + 1 + len(SEL_CONDITIONS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    savefig(fig, "incoming_weights.png")


# ─── Per-model diagnostic plots ───────────────────────────────────────────────

PERC_CONDITIONS_DIAG = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION"]
PERC_COND_LABELS_DIAG = {
    "BASELINE": "Baseline", "FULL": "Full",
    "NO_DISCUSSION": "No Discussion", "NO_SELECTION": "No Selection",
}
CT_GROUP_ORDER  = ["Violators", "Cooperators"]
CT_GROUP_COLORS = {"injunctive_norm": "#1f77b4", "descriptive_norm": "#ff7f0e"}
BIAS_LABELS     = {"injunctive_norm": "IN − Actual", "descriptive_norm": "DN − Actual"}
CT_ERR_COLORS   = {"Violators": "#d62728", "Cooperators": "#2ca02c"}
NORM_STYLE      = {
    "injunctive_norm":  {"color": "#1f77b4", "label": "Injunctive norm"},
    "descriptive_norm": {"color": "#ff7f0e", "label": "Descriptive norm"},
}


def collect_alignment_data(model, variant, condition):
    records = []
    for seed in SEEDS:
        d = load_log_variant(model, variant, seed, condition)
        if d is None:
            continue
        r1 = next((r for r in d["round_logs"] if r["round"] == 1), None)
        if r1 is None:
            continue
        initial_ct = {a["id"]: a["cooperation_tendency"] for a in r1["agent_states"]}
        for r in d["round_logs"]:
            rnd           = r["round"]
            contributions = {int(k): v for k, v in r["contributions"].items()}
            perceptions   = {int(k): v for k, v in (r.get("perceptions") or {}).items()}
            for aid, ct0 in initial_ct.items():
                grp    = ct_group(ct0)
                actual = contributions.get(aid)
                if actual is None:
                    continue
                perc = perceptions.get(aid) or {}
                records.append({
                    "seed": seed, "round": rnd, "agent_id": aid, "group": grp,
                    "actual":           float(actual),
                    "injunctive_norm":  perc.get("injunctive_norm"),
                    "descriptive_norm": perc.get("descriptive_norm"),
                })
    return records


def aggregate_by_seed_round(records, field, group):
    data = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec["group"] != group:
            continue
        val = rec[field]
        if val is not None:
            data[rec["seed"]][rec["round"]].append(float(val))
    per_seed = {}
    for seed, round_vals in data.items():
        per_seed[seed] = {rnd: np.mean(v) for rnd, v in round_vals.items() if v}
    return per_seed


def rounds_mean_sd(per_seed_means):
    arr = np.full((len(per_seed_means), len(ROUNDS)), np.nan)
    for i, (_, rm) in enumerate(per_seed_means.items()):
        for j, rnd in enumerate(ROUNDS):
            if rnd in rm:
                arr[i, j] = rm[rnd]
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _scatter_norm_vs_actual(ax, records, norm_key):
    from scipy import stats as _stats
    xs = [r[norm_key] for r in records if r[norm_key] is not None]
    ys = [r["actual"] for r in records if r[norm_key] is not None]
    if not xs:
        return None
    style = NORM_STYLE[norm_key]
    ax.scatter(xs, ys, color=style["color"], alpha=0.15, s=8, edgecolors="none",
               label=style["label"])
    if len(xs) > 2 and np.std(xs) > 0:
        slope, intercept, r, *_ = _stats.linregress(xs, ys)
        x_line = np.linspace(0, ENDOWMENT, 100)
        ax.plot(x_line, slope * x_line + intercept, color=style["color"],
                lw=1.8, linestyle="--", label=f"fit r={r:.2f}")
        return r
    return None


def plot_perception_vs_actual(model):
    print(f"  [Plot 1] perception vs actual — {model}")
    nrows = len(PERC_CONDITIONS_DIAG)
    fig, axes = plt.subplots(nrows, 1, figsize=(5, nrows * 4),
                             sharex=True, sharey=True, squeeze=False)
    for row, cond in enumerate(PERC_CONDITIONS_DIAG):
        ax   = axes[row, 0]
        recs = collect_alignment_data(model, "global", cond)
        any_data = False
        for norm_key in ("injunctive_norm", "descriptive_norm"):
            if _scatter_norm_vs_actual(ax, recs, norm_key) is not None:
                any_data = True
        ax.plot([0, ENDOWMENT], [0, ENDOWMENT], color="grey", lw=1, linestyle=":", alpha=0.6)
        if not any_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="grey")
        ax.set_xlim(-0.5, ENDOWMENT + 0.5)
        ax.set_ylim(-0.5, ENDOWMENT + 0.5)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel(f"{PERC_COND_LABELS_DIAG[cond]}\nActual contribution", fontsize=9)
        if row == nrows - 1:
            ax.set_xlabel("Perception value", fontsize=9)
        if row == 0 and any_data:
            ax.legend(fontsize=7, markerscale=2)
    fig.suptitle(f"{model.upper()} — Perception vs Actual", fontsize=11)
    plt.tight_layout()
    savefig_model(fig, model, "alignment_perception_vs_actual.png")


def _bias_by_group_seed(records, norm_key):
    data = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec[norm_key] is None or rec["group"] is None:
            continue
        data[rec["group"]][rec["seed"]].append(float(rec[norm_key]) - rec["actual"])
    return {g: {s: np.mean(v) for s, v in sd.items()} for g, sd in data.items()}


def plot_bias_by_group(model):
    print(f"  [Plot 2] bias by group — {model}")
    nrows = len(PERC_CONDITIONS_DIAG)
    fig, axes = plt.subplots(nrows, 1, figsize=(5.5, nrows * 3.5), sharey=True, squeeze=False)
    x = np.arange(len(CT_GROUP_ORDER))
    w = 0.35
    for row, cond in enumerate(PERC_CONDITIONS_DIAG):
        ax   = axes[row, 0]
        recs = collect_alignment_data(model, "global", cond)
        for i, norm_key in enumerate(("injunctive_norm", "descriptive_norm")):
            bias_data = _bias_by_group_seed(recs, norm_key)
            means, errs = [], []
            for grp in CT_GROUP_ORDER:
                seed_vals = list(bias_data.get(grp, {}).values())
                means.append(np.mean(seed_vals) if seed_vals else np.nan)
                errs.append(np.std(seed_vals)   if seed_vals else 0)
            offset = (i - 0.5) * w
            ax.bar(x + offset, means, w, label=BIAS_LABELS[norm_key],
                   color=CT_GROUP_COLORS[norm_key], alpha=0.8, edgecolor="white")
            ax.errorbar(x + offset, means, yerr=errs,
                        fmt="none", color="black", capsize=3, lw=1)
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CT_GROUP_ORDER, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(f"{PERC_COND_LABELS_DIAG[cond]}\nBias", fontsize=9)
        if row == 0:
            ax.legend(fontsize=8)
        if row == nrows - 1:
            ax.set_xlabel("CT group", fontsize=9)
    fig.suptitle(f"{model.upper()} — Perception Bias by CT Group", fontsize=11)
    plt.tight_layout()
    savefig_model(fig, model, "alignment_bias_by_group.png")


def plot_error_distribution(model):
    print(f"  [Plot 3] error distribution — {model}")
    nrows     = len(PERC_CONDITIONS_DIAG)
    norm_keys = [("injunctive_norm", "IN − Actual"), ("descriptive_norm", "DN − Actual")]
    fig, axes = plt.subplots(nrows, 2, figsize=(10, nrows * 3.2))
    for row, cond in enumerate(PERC_CONDITIONS_DIAG):
        for col, (norm_key, norm_label) in enumerate(norm_keys):
            ax   = axes[row, col]
            recs = collect_alignment_data(model, "global", cond)
            any_data = False
            for grp in CT_GROUP_ORDER:
                errors = [float(rec[norm_key]) - rec["actual"]
                          for rec in recs
                          if rec[norm_key] is not None and rec["group"] == grp]
                if errors:
                    ax.hist(errors, bins=20, alpha=0.5, color=CT_ERR_COLORS[grp],
                            label=grp, edgecolor="none", density=True)
                    any_data = True
            ax.axvline(0, color="black", lw=1, linestyle="--", alpha=0.7)
            ax.grid(True, alpha=0.2)
            if not any_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="grey")
            if row == 0:
                ax.set_title(norm_label, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{PERC_COND_LABELS_DIAG[cond]}\nDensity", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            if row == 0 and any_data:
                ax.legend(fontsize=10)
    plt.tight_layout()
    savefig_model(fig, model, "alignment_error_distribution.png")


def plot_perception_alignment(model):
    """
    2 rows (CT groups) × 2 cols (conditions).
    IN (blue), DN (orange), actual (black) trajectories per CT group.
    """
    print(f"  [Fig 1] perception alignment — {model}")
    ncols = len(DISC_CONDITIONS)
    nrows = len(CT_GROUP_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                             sharex=True, sharey=True)
    fig.suptitle(
        f"{model.upper()} — Perception → Action Alignment\n"
        f"IN (blue --) | DN (orange --) | Actual (black —)",
        fontsize=10
    )
    for col, cond in enumerate(DISC_CONDITIONS):
        recs = collect_alignment_data(model, "global", cond)
        for row, grp in enumerate(CT_GROUP_ORDER):
            ax = axes[row, col]
            for field, color, ls, label in [
                ("injunctive_norm",  "#1f77b4", "--", "Injunctive norm"),
                ("descriptive_norm", "#ff7f0e", "-.", "Descriptive norm"),
                ("actual",           "#2c2c2c", "-",  "Actual"),
            ]:
                psm = aggregate_by_seed_round(recs, field, grp)
                if not psm:
                    continue
                m, s = rounds_mean_sd(psm)
                ax.plot(ROUNDS, m, color=color, ls=ls, lw=2, marker="o", ms=3, label=label)
                ax.fill_between(ROUNDS, m - s, m + s, color=color, alpha=0.12)
            ax.axhline(ENDOWMENT / 2, color="grey", ls=":", lw=1, alpha=0.5)
            ax.set_ylim(-0.5, ENDOWMENT + 0.5)
            ax.set_xlim(0.5, 20.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(cond.replace("_", " "), fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{grp}\nContribution (0–{int(ENDOWMENT)})", fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel("Round", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    savefig_model(fig, model, "alignment_perception.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building stores …")
    bstore = build_behavior_store()
    gstore = build_gap_store()
    wstore = build_weight_store()

    print("\n── Trajectory plots ──")
    plot_contribution_payoff(bstore)
    plot_perception_gaps(gstore)
    plot_incoming_weights(wstore)

    print("\n── Per-model diagnostics ──")
    for model in MODELS:
        print(f"\n[{model.upper()}]")
        plot_perception_vs_actual(model)
        plot_bias_by_group(model)
        plot_error_distribution(model)
        plot_perception_alignment(model)

    print("\nAll done.")
