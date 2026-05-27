"""
Alignment analysis: do agents walk the talk?

Two alignment checks, for NO_SELECTION and FULL conditions:
  a) Discussion → Action: stated contribution amount vs actual contribution
  b) Perception → Action: injunctive/descriptive norm vs actual contribution

Groups based on initial (round-1) CT so labels are stable across rounds.

Plots per model (global and local variants, both conditions):
  Figure 1: Discussion alignment — stated vs actual, mean ± SD across seeds
  Figure 2: Perception alignment — IN/DN vs actual, mean ± SD across seeds
  Figure 3: Gap analysis — (stated−actual), (IN−actual), (DN−actual) over rounds
"""

import json
import glob
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS   = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT  = "/data3/rasimura/social-norm-evo/figures/2026-03-22"
MODELS    = ["gpt", "llama", "mistral", "qwen"]
VARIANTS  = ["global"]
SEEDS     = list(range(43, 53))
CONDITIONS = ["NO_SELECTION", "FULL"]
ENDOWMENT  = 10.0
ROUNDS     = list(range(1, 21))

CT_GROUPS = {
    "Violators":    (0.0, 0.3),
    "Normal":       (0.3, 0.7),
    "Cooperators":  (0.7, 1.01),
}
CT_COLORS = {
    "Violators":   "#d62728",
    "Normal":      "#ff7f0e",
    "Cooperators": "#2ca02c",
}
COND_COLORS = {"NO_SELECTION": "#1f77b4", "FULL": "#9467bd"}
COND_LS     = {"NO_SELECTION": "-",       "FULL": "--"}

# ─── Stated-contribution extractor ────────────────────────────────────────────

_STATED_PATTERNS = [
    # "contribute/contributing [around] X" or "I will/plan/propose contributing X"
    r'contribut(?:e|ing)\s+(?:around|about|approximately|roughly|at\s+least|up\s+to|at\s+most)?\s*(\d+(?:\.\d+)?)',
    # "contribution(s) [of/at] [around] X"
    r'contributions?\s+(?:of\s+|at\s+)?(?:around|about|approximately|roughly)?\s*(\d+(?:\.\d+)?)',
    # "aim(ing) for [a contribution of] [around] X"
    r'aim(?:ing)?\s+for\s+(?:a\s+)?(?:contribution\s+(?:of\s+)?)?(?:around|about|approximately)?\s*(\d+(?:\.\d+)?)',
    # "I will/plan/intend/commit to contribute [around] X"
    r'I\s+(?:will|plan|intend|commit)\s+(?:to\s+)?contribut(?:e|ing)\s+(?:around|about)?\s*(\d+(?:\.\d+)?)',
    # "propose/suggest/recommend contributing [around] X"
    r'(?:propose|suggest|recommend)\s+(?:contributing\s+)?(?:around|about|approximately)?\s*(\d+(?:\.\d+)?)',
    # "maintaining [contributions] at [around] X"
    r'maintaining\s+(?:contributions?\s+)?(?:at|to)\s+(?:around|about)?\s*(\d+(?:\.\d+)?)',
    # "X/10" fraction form (e.g. "4/10")
    r'(\d+(?:\.\d+)?)/10\b',
    # percentage form: "around X%" → convert to absolute (X/100 * ENDOWMENT)
    r'(?:contribut\w+|around|about|maintaining)\s+(?:around\s+)?(\d+)\s*%',
    # range form "X-Y" → take midpoint (e.g. "around 7-8")
    r'(?:around|about|approximately|between)\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)',
]


_PCT_IDX   = len(_STATED_PATTERNS) - 2   # percentage pattern index
_RANGE_IDX = len(_STATED_PATTERNS) - 1  # range pattern index


def extract_stated(message: str) -> float | None:
    """Return the first contribution amount mentioned in the message, or None."""
    cleaned = re.sub(r'\bAgents?\s+[\d,\s]+', '', message, flags=re.IGNORECASE)
    for i, pat in enumerate(_STATED_PATTERNS):
        m = re.search(pat, cleaned, re.IGNORECASE)
        if m:
            if i == _PCT_IDX:
                val = float(m.group(1)) / 100.0 * ENDOWMENT
            elif i == _RANGE_IDX:
                val = (float(m.group(1)) + float(m.group(2))) / 2.0
            else:
                val = float(m.group(1))
            if 0.0 <= val <= ENDOWMENT:
                return val
    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_latest_log(model, variant, seed, condition):
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


def savefig(fig, model, fname):
    out_dir = os.path.join(FIG_ROOT, model, "alignment_analysis")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def ct_group(ct):
    for name, (lo, hi) in CT_GROUPS.items():
        if lo <= ct < hi:
            return name
    return None


# ─── Data collection ──────────────────────────────────────────────────────────

def collect_alignment_data(model, variant, condition):
    """
    Returns a list of per-agent-per-round records:
      {seed, round, agent_id, group,
       stated,           # float | None — stated contribution from discussion
       actual,           # float — actual contribution
       injunctive_norm,  # float | None
       descriptive_norm} # float | None
    """
    records = []
    for seed in SEEDS:
        d = load_latest_log(model, variant, seed, condition)
        if d is None:
            continue

        # Initial CT from round 1
        r1 = next((r for r in d["round_logs"] if r["round"] == 1), None)
        if r1 is None:
            continue
        initial_ct = {a["id"]: a["cooperation_tendency"] for a in r1["agent_states"]}

        for r in d["round_logs"]:
            rnd          = r["round"]
            contributions = {int(k): v for k, v in r["contributions"].items()}
            perceptions   = {int(k): v for k, v in (r.get("perceptions") or {}).items()}

            # Aggregate stated contributions per agent across their discussion messages
            stated_by_agent = defaultdict(list)
            for entry in r.get("discussion_transcript") or []:
                aid = entry["agent_id"]
                val = extract_stated(entry.get("message", ""))
                if val is not None:
                    stated_by_agent[aid].append(val)

            for aid, ct0 in initial_ct.items():
                grp    = ct_group(ct0)
                actual = contributions.get(aid)
                if actual is None:
                    continue
                perc = perceptions.get(aid) or {}
                stated_vals = stated_by_agent.get(aid)
                stated = float(np.mean(stated_vals)) if stated_vals else None

                records.append({
                    "seed":             seed,
                    "round":            rnd,
                    "agent_id":         aid,
                    "group":            grp,
                    "stated":           stated,
                    "actual":           float(actual),
                    "injunctive_norm":  perc.get("injunctive_norm"),
                    "descriptive_norm": perc.get("descriptive_norm"),
                })
    return records


# ─── Aggregation helper ───────────────────────────────────────────────────────

def aggregate_by_seed_round(records, field, group):
    """
    Returns per_seed_means[seed][round] = mean of `field` over agents in `group`.
    None values in `field` are excluded.
    """
    # seed → round → list of values
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


def rounds_mean_sd(per_seed_means, rounds):
    """
    Given {seed: {round: mean}}, return arrays (means, sds) over seeds for each round.
    """
    arr = np.full((len(per_seed_means), len(rounds)), np.nan)
    for i, (_, rm) in enumerate(per_seed_means.items()):
        for j, rnd in enumerate(rounds):
            if rnd in rm:
                arr[i, j] = rm[rnd]
    means = np.nanmean(arr, axis=0)
    sds   = np.nanstd(arr,  axis=0)
    return means, sds


# ─── Plot 1: Perception vs Actual scatter ─────────────────────────────────────
#
# One figure per model.
# Rows: conditions  |  Cols: global / local
# Each panel: scatter of IN (blue) and DN (orange) vs actual contribution.
#   • diagonal = perfect alignment
#   • dashed regression line per norm type
#   • pooled across all seeds and agents

PERC_CONDITIONS = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION"]
PERC_COND_LABELS = {
    "BASELINE": "Baseline", "FULL": "Full",
    "NO_DISCUSSION": "No Discussion", "NO_SELECTION": "No Selection",
}

NORM_STYLE = {
    "injunctive_norm":  {"color": "#1f77b4", "label": "Injunctive norm",  "marker": "o"},
    "descriptive_norm": {"color": "#ff7f0e", "label": "Descriptive norm", "marker": "s"},
}


def _scatter_norm_vs_actual(ax, records, norm_key):
    """Scatter of norm_key (x) vs actual (y) with regression line. Returns r or None."""
    from scipy import stats as _stats
    xs = [r[norm_key] for r in records if r[norm_key] is not None]
    ys = [r["actual"] for r in records if r[norm_key] is not None]
    if not xs:
        return None
    style = NORM_STYLE[norm_key]
    ax.scatter(xs, ys, color=style["color"], alpha=0.15, s=8,
               edgecolors="none", label=style["label"])
    if len(xs) > 2 and np.std(xs) > 0:
        slope, intercept, r, *_ = _stats.linregress(xs, ys)
        x_line = np.linspace(0, ENDOWMENT, 100)
        ax.plot(x_line, slope * x_line + intercept,
                color=style["color"], lw=1.8, linestyle="--",
                label=f"fit r={r:.2f}")
        return r
    return None


def plot_perception_vs_actual(model):
    """Plot 1 — one figure per model: rows=conditions, cols=variants."""
    print(f"  [Plot 1] perception vs actual — {model}")
    nrows, ncols = len(PERC_CONDITIONS), len(VARIANTS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4),
                             sharex=True, sharey=True, squeeze=False)

    for row, cond in enumerate(PERC_CONDITIONS):
        for col, variant in enumerate(VARIANTS):
            ax = axes[row, col]
            recs = collect_alignment_data(model, variant, cond)

            any_data = False
            for norm_key in ("injunctive_norm", "descriptive_norm"):
                r = _scatter_norm_vs_actual(ax, recs, norm_key)
                if r is not None:
                    any_data = True

            # diagonal: perfect alignment
            ax.plot([0, ENDOWMENT], [0, ENDOWMENT], color="grey",
                    lw=1, linestyle=":", alpha=0.6, label="Perfect alignment")

            if not any_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="grey")

            ax.set_xlim(-0.5, ENDOWMENT + 0.5)
            ax.set_ylim(-0.5, ENDOWMENT + 0.5)
            ax.grid(True, alpha=0.2)
            ax.axhline(ENDOWMENT / 2, color="grey", lw=0.6, linestyle="--", alpha=0.4)
            ax.axvline(ENDOWMENT / 2, color="grey", lw=0.6, linestyle="--", alpha=0.4)

            if row == 0:
                ax.set_title(f"{variant.capitalize()}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{PERC_COND_LABELS[cond]}\nActual contribution", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Perception value", fontsize=9)
            if row == 0 and col == 0 and any_data:
                ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(
        f"{model.upper()} — Perception vs Actual Contribution\n"
        f"Blue = Injunctive norm  |  Orange = Descriptive norm  |  dotted = perfect alignment",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "alignment_perception_vs_actual.png")


# ─── Plot 2: Bias by CT group ─────────────────────────────────────────────────
#
# One figure per model.
# Rows: conditions  |  Cols: global / local
# Each panel: grouped bar chart — x=CT group, bars=IN bias & DN bias
# Bias = mean(Perception − Actual), error bars = ± std across seeds

CT_GROUP_ORDER  = ["Violators", "Normal", "Cooperators"]
CT_GROUP_COLORS = {
    "injunctive_norm":  "#1f77b4",
    "descriptive_norm": "#ff7f0e",
}
BIAS_LABELS = {
    "injunctive_norm":  "IN − Actual",
    "descriptive_norm": "DN − Actual",
}


def _bias_by_group_seed(records, norm_key):
    """
    Returns {group: {seed: mean_bias}} for norm_key.
    bias = norm_value − actual (only where norm is not None).
    """
    data = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec[norm_key] is None or rec["group"] is None:
            continue
        bias = float(rec[norm_key]) - rec["actual"]
        data[rec["group"]][rec["seed"]].append(bias)
    return {
        grp: {seed: np.mean(vals) for seed, vals in seed_data.items()}
        for grp, seed_data in data.items()
    }


def plot_bias_by_group(model):
    """Plot 2 — bias bar charts by CT group."""
    print(f"  [Plot 2] bias by group — {model}")
    nrows, ncols = len(PERC_CONDITIONS), len(VARIANTS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.5),
                             sharey=True, squeeze=False)

    x      = np.arange(len(CT_GROUP_ORDER))
    w      = 0.35

    for row, cond in enumerate(PERC_CONDITIONS):
        for col, variant in enumerate(VARIANTS):
            ax   = axes[row, col]
            recs = collect_alignment_data(model, variant, cond)

            for i, norm_key in enumerate(("injunctive_norm", "descriptive_norm")):
                bias_data = _bias_by_group_seed(recs, norm_key)
                means, errs = [], []
                for grp in CT_GROUP_ORDER:
                    seed_vals = list(bias_data.get(grp, {}).values())
                    if seed_vals:
                        means.append(np.mean(seed_vals))
                        errs.append(np.std(seed_vals))
                    else:
                        means.append(np.nan)
                        errs.append(0)

                offset = (i - 0.5) * w
                col_c  = CT_GROUP_COLORS[norm_key]
                ax.bar(x + offset, means, w, label=BIAS_LABELS[norm_key],
                       color=col_c, alpha=0.8, edgecolor="white")
                ax.errorbar(x + offset, means, yerr=errs,
                            fmt="none", color="black", capsize=3, lw=1)

            ax.axhline(0, color="black", lw=0.8, alpha=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(CT_GROUP_ORDER, fontsize=8)
            ax.grid(axis="y", alpha=0.2)

            if row == 0:
                ax.set_title(f"{variant.capitalize()}", fontsize=10)
                ax.legend(fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{PERC_COND_LABELS[cond]}\nBias (Perception − Actual)", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("CT group", fontsize=9)

    fig.suptitle(
        f"{model.upper()} — Perception Bias by CT Group\n"
        f"Blue = IN−Actual  |  Orange = DN−Actual  |  Error bars = ±std across seeds\n"
        f"Positive = overestimated  |  Negative = underestimated",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "alignment_bias_by_group.png")


# ─── Plot 3: Distribution of errors ──────────────────────────────────────────
#
# One figure per model.
# Rows: conditions  |  Cols: global / local
# Each panel: overlaid histogram of (IN − actual) and (DN − actual),
#   split by CT group using colour.

CT_ERR_COLORS = {
    "Violators":   "#d62728",
    "Normal":      "#ff7f0e",
    "Cooperators": "#2ca02c",
}


def plot_error_distribution(model):
    """Plot 3 — histogram of perception errors split by CT group."""
    print(f"  [Plot 3] error distribution — {model}")
    nrows = len(PERC_CONDITIONS)
    norm_keys = [("injunctive_norm", "IN − Actual"),
                 ("descriptive_norm", "DN − Actual")]

    fig, axes = plt.subplots(nrows, 2, figsize=(10, nrows * 3.2), sharey=False)

    variant = "global"
    for row, cond in enumerate(PERC_CONDITIONS):
        for col, (norm_key, norm_label) in enumerate(norm_keys):
            ax   = axes[row, col]
            recs = collect_alignment_data(model, variant, cond)

            any_data = False
            for grp in CT_GROUP_ORDER:
                errors = [
                    float(rec[norm_key]) - rec["actual"]
                    for rec in recs
                    if rec[norm_key] is not None and rec["group"] == grp
                ]
                if errors:
                    ax.hist(errors, bins=20, alpha=0.5,
                            color=CT_ERR_COLORS[grp], label=grp,
                            edgecolor="none", density=True)
                    any_data = True

            ax.axvline(0, color="black", lw=1, linestyle="--", alpha=0.7)
            ax.grid(True, alpha=0.2)

            if not any_data:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="grey")

            if row == 0:
                ax.set_title(norm_label, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{PERC_COND_LABELS[cond]}\nDensity", fontsize=12)
            ax.tick_params(axis="both", labelsize=11)
            if row == 0 and any_data:
                ax.legend(fontsize=10)

    plt.tight_layout()
    savefig(fig, model, "alignment_error_distribution.png")


# ─── Figure 1: Discussion alignment (stated vs actual) ────────────────────────

def plot_discussion_alignment(model, variant):
    """
    3 rows (CT groups) × 2 cols (conditions).
    Each subplot: stated (dashed) vs actual (solid), mean ± SD across seeds.
    """
    print(f"  [Fig 1] discussion alignment — {variant}")

    groups = list(CT_GROUPS.keys())
    ncols  = len(CONDITIONS)
    nrows  = len(groups)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                             sharex=True, sharey=True)
    fig.suptitle(
        f"{model.upper()} — {variant.capitalize()} — Discussion → Action Alignment\n"
        f"Stated contribution (dashed) vs Actual contribution (solid), mean ± SD",
        fontsize=10
    )

    for col, cond in enumerate(CONDITIONS):
        recs = collect_alignment_data(model, variant, cond)
        for row, grp in enumerate(groups):
            ax    = axes[row][col]
            color = CT_COLORS[grp]

            stated_psm = aggregate_by_seed_round(recs, "stated", grp)
            actual_psm = aggregate_by_seed_round(recs, "actual", grp)

            if stated_psm:
                sm, ss = rounds_mean_sd(stated_psm, ROUNDS)
                ax.plot(ROUNDS, sm, color=color, ls="--", lw=2,
                        marker="o", ms=3, label="Stated")
                ax.fill_between(ROUNDS, sm - ss, sm + ss, color=color, alpha=0.15)

            if actual_psm:
                am, as_ = rounds_mean_sd(actual_psm, ROUNDS)
                ax.plot(ROUNDS, am, color=color, ls="-", lw=2,
                        marker="s", ms=3, label="Actual")
                ax.fill_between(ROUNDS, am - as_, am + as_, color=color, alpha=0.1)

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
                ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(fig, model, f"alignment_discussion_{variant}.png")


# ─── Figure 2: Perception alignment (IN/DN vs actual) ─────────────────────────

def plot_perception_alignment(model, variant):
    """
    3 rows (CT groups) × 2 cols (conditions).
    Each subplot: injunctive norm (blue dashed), descriptive norm (orange dashed),
    actual contribution (black solid).
    """
    print(f"  [Fig 2] perception alignment — {variant}")

    groups = list(CT_GROUPS.keys())
    ncols  = len(CONDITIONS)
    nrows  = len(groups)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.2),
                             sharex=True, sharey=True)
    fig.suptitle(
        f"{model.upper()} — {variant.capitalize()} — Perception → Action Alignment\n"
        f"Injunctive norm (blue --) | Descriptive norm (orange --) | Actual (black —)",
        fontsize=10
    )

    for col, cond in enumerate(CONDITIONS):
        recs = collect_alignment_data(model, variant, cond)
        for row, grp in enumerate(groups):
            ax = axes[row][col]

            for field, color, ls, label in [
                ("injunctive_norm",  "#1f77b4", "--", "Injunctive norm"),
                ("descriptive_norm", "#ff7f0e", "-.", "Descriptive norm"),
                ("actual",           "#2c2c2c", "-",  "Actual"),
            ]:
                psm = aggregate_by_seed_round(recs, field, grp)
                if not psm:
                    continue
                m, s = rounds_mean_sd(psm, ROUNDS)
                ax.plot(ROUNDS, m, color=color, ls=ls, lw=2,
                        marker="o", ms=3, label=label)
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
    savefig(fig, model, f"alignment_perception_{variant}.png")


# ─── Figure 3: Gap analysis ────────────────────────────────────────────────────

def plot_gap_analysis(model, variant):
    """
    2 rows (conditions) × 3 cols (CT groups).
    Three gap lines per subplot:
      stated − actual  (discussion gap, red)
      IN − actual      (injunctive gap, blue)
      DN − actual      (descriptive gap, orange)
    Positive = over-stated / over-expected; negative = under-stated / under-expected.
    """
    print(f"  [Fig 3] gap analysis — {variant}")

    groups = list(CT_GROUPS.keys())
    ncols  = len(groups)
    nrows  = len(CONDITIONS)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2),
                             sharex=True, sharey=True)
    fig.suptitle(
        f"{model.upper()} — {variant.capitalize()} — Alignment Gaps Over Rounds\n"
        f"stated−actual (red) | injunctive−actual (blue) | descriptive−actual (orange)\n"
        f"Positive = overstated / over-expected",
        fontsize=10
    )

    gap_specs = [
        ("stated_gap",    "#d62728", "-",  "Stated − Actual"),
        ("inj_gap",       "#1f77b4", "--", "Injunctive − Actual"),
        ("desc_gap",      "#ff7f0e", "-.", "Descriptive − Actual"),
    ]

    for row, cond in enumerate(CONDITIONS):
        recs = collect_alignment_data(model, variant, cond)

        # Pre-compute gap fields
        gap_recs = []
        for rec in recs:
            gap_rec = {k: rec[k] for k in ("seed", "round", "group")}
            a = rec["actual"]
            gap_rec["stated_gap"] = (rec["stated"] - a)       if rec["stated"]           is not None else None
            gap_rec["inj_gap"]    = (rec["injunctive_norm"] - a)  if rec["injunctive_norm"]  is not None else None
            gap_rec["desc_gap"]   = (rec["descriptive_norm"] - a) if rec["descriptive_norm"] is not None else None
            gap_recs.append(gap_rec)

        for col, grp in enumerate(groups):
            ax = axes[row][col]

            for field, color, ls, label in gap_specs:
                psm = aggregate_by_seed_round(gap_recs, field, grp)
                if not psm:
                    continue
                m, s = rounds_mean_sd(psm, ROUNDS)
                ax.plot(ROUNDS, m, color=color, ls=ls, lw=2,
                        marker="o", ms=3, label=label)
                ax.fill_between(ROUNDS, m - s, m + s, color=color, alpha=0.12)

            ax.axhline(0, color="black", lw=1, alpha=0.6)
            ax.set_xlim(0.5, 20.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.grid(True, alpha=0.3)

            if row == 0:
                ax.set_title(grp, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{cond.replace('_',' ')}\nGap", fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel("Round", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    savefig(fig, model, f"alignment_gaps_{variant}.png")


# ─── Figure 4: Per-seed alignment (all agents pooled, no CT split) ────────────

def _round_means(recs, field, rnd):
    """Mean of `field` across all agents in `recs` for a given round, ignoring None."""
    vals = [rec[field] for rec in recs if rec["round"] == rnd and rec[field] is not None]
    return float(np.mean(vals)) if vals else np.nan


def plot_per_seed_alignment(model, variant):
    """
    One figure per seed: 2 rows (conditions) × 3 cols (discussion / perception / gaps).
    All agents pooled — no CT-group split.
    """
    print(f"  [Fig 4] per-seed alignment — {variant}")

    out_dir = os.path.join(FIG_ROOT, model, "alignment_analysis", "per_seed", variant)
    os.makedirs(out_dir, exist_ok=True)

    for seed in SEEDS:
        # Load records for both conditions for this seed
        cond_recs = {}
        for cond in CONDITIONS:
            d = load_latest_log(model, variant, seed, cond)
            if d is None:
                continue
            r1 = next((r for r in d["round_logs"] if r["round"] == 1), None)
            if r1 is None:
                continue
            initial_ct = {a["id"]: a["cooperation_tendency"] for a in r1["agent_states"]}

            recs = []
            for r in d["round_logs"]:
                rnd           = r["round"]
                contributions = {int(k): v for k, v in r["contributions"].items()}
                perceptions   = {int(k): v for k, v in (r.get("perceptions") or {}).items()}
                stated_by_agent = defaultdict(list)
                for entry in r.get("discussion_transcript") or []:
                    val = extract_stated(entry.get("message", ""))
                    if val is not None:
                        stated_by_agent[entry["agent_id"]].append(val)
                for aid, ct0 in initial_ct.items():
                    actual = contributions.get(aid)
                    if actual is None:
                        continue
                    perc = perceptions.get(aid) or {}
                    sv   = stated_by_agent.get(aid)
                    recs.append({
                        "round":            rnd,
                        "actual":           float(actual),
                        "stated":           float(np.mean(sv)) if sv else None,
                        "injunctive_norm":  perc.get("injunctive_norm"),
                        "descriptive_norm": perc.get("descriptive_norm"),
                    })
            cond_recs[cond] = recs

        if not cond_recs:
            continue

        fig, axes = plt.subplots(len(CONDITIONS), 3,
                                 figsize=(14, len(CONDITIONS) * 3.5),
                                 sharex=True)
        fig.suptitle(
            f"{model.upper()} — {variant.capitalize()} — seed {seed} — Alignment\n"
            f"All agents pooled",
            fontsize=10
        )

        col_titles = ["Discussion: Stated vs Actual",
                      "Perception: IN / DN vs Actual",
                      "Gaps (stated−actual, IN−actual, DN−actual)"]

        for row, cond in enumerate(CONDITIONS):
            recs = cond_recs.get(cond, [])

            # ── col 0: stated vs actual ──────────────────────────────────────
            ax = axes[row][0]
            stated_m = [_round_means(recs, "stated", rnd) for rnd in ROUNDS]
            actual_m = [_round_means(recs, "actual", rnd) for rnd in ROUNDS]
            ax.plot(ROUNDS, stated_m, color="#1f77b4", ls="--", lw=2,
                    marker="o", ms=3, label="Stated")
            ax.plot(ROUNDS, actual_m, color="#1f77b4", ls="-",  lw=2,
                    marker="s", ms=3, label="Actual")
            ax.axhline(ENDOWMENT / 2, color="grey", ls=":", lw=1, alpha=0.5)
            ax.set_ylim(-0.5, ENDOWMENT + 0.5)
            if row == 0:
                ax.legend(fontsize=7)
                ax.set_title(col_titles[0], fontsize=8)
            ax.set_ylabel(f"{cond.replace('_',' ')}\nContribution", fontsize=8)

            # ── col 1: IN / DN vs actual ─────────────────────────────────────
            ax = axes[row][1]
            in_m   = [_round_means(recs, "injunctive_norm",  rnd) for rnd in ROUNDS]
            dn_m   = [_round_means(recs, "descriptive_norm", rnd) for rnd in ROUNDS]
            ax.plot(ROUNDS, in_m,   color="#1f77b4", ls="--", lw=2,
                    marker="o", ms=3, label="Injunctive norm")
            ax.plot(ROUNDS, dn_m,   color="#ff7f0e", ls="-.", lw=2,
                    marker="o", ms=3, label="Descriptive norm")
            ax.plot(ROUNDS, actual_m, color="#2c2c2c", ls="-", lw=2,
                    marker="s", ms=3, label="Actual")
            ax.axhline(ENDOWMENT / 2, color="grey", ls=":", lw=1, alpha=0.5)
            ax.set_ylim(-0.5, ENDOWMENT + 0.5)
            if row == 0:
                ax.legend(fontsize=7)
                ax.set_title(col_titles[1], fontsize=8)

            # ── col 2: gaps ──────────────────────────────────────────────────
            ax = axes[row][2]
            stated_gap = [s - a if not (np.isnan(s) or np.isnan(a)) else np.nan
                          for s, a in zip(stated_m, actual_m)]
            in_gap     = [i - a if not (np.isnan(i) or np.isnan(a)) else np.nan
                          for i, a in zip(in_m, actual_m)]
            dn_gap     = [dn - a if not (np.isnan(dn) or np.isnan(a)) else np.nan
                          for dn, a in zip(dn_m, actual_m)]
            ax.plot(ROUNDS, stated_gap, color="#d62728", ls="-",  lw=2,
                    marker="o", ms=3, label="Stated−Actual")
            ax.plot(ROUNDS, in_gap,     color="#1f77b4", ls="--", lw=2,
                    marker="o", ms=3, label="IN−Actual")
            ax.plot(ROUNDS, dn_gap,     color="#ff7f0e", ls="-.", lw=2,
                    marker="o", ms=3, label="DN−Actual")
            ax.axhline(0, color="black", lw=1, alpha=0.6)
            if row == 0:
                ax.legend(fontsize=7)
                ax.set_title(col_titles[2], fontsize=8)

            for ax in axes[row]:
                ax.set_xlim(0.5, 20.5)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.grid(True, alpha=0.3)

        for ax in axes[-1]:
            ax.set_xlabel("Round", fontsize=8)

        plt.tight_layout()
        path = os.path.join(out_dir, f"seed{seed:02d}_alignment.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved → {path}")


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for model in MODELS:
        print(f"\n[{model.upper()}]")
        plot_perception_vs_actual(model)
        plot_bias_by_group(model)
        plot_error_distribution(model)
    print("\nAll done.")
