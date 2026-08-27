"""
four_gap_perception_action_test.py
-----------------------------------
EXPLORATORY, NOT CANONICAL. Standalone script — does not import or modify
temp_eval_alignment2.py or gap_based_alignment_test.py, and does not write
into their output locations (figures/local/cross_model/, exports/alignment_by_gap/,
or figures/MAIN_RESULTS/). Requested to test whether extending the IN/DN
perception-action-gap analysis to all 4 elicited perception fields is
worthwhile before committing to it as a paper result.

The round log's `perceptions` block actually carries FOUR fields per agent
per round (see SoNoLiSi_v5_os_local.py's norm-reflection prompt), not just
the two (IN/DN) used everywhere else in the codebase:
    injunctive_norm         - numeric already (0-10 float)
    descriptive_norm        - numeric already (0-10 float)
    expectation_of_others   - FREE TEXT, e.g. "I expect others will
                               contribute around 6-7 in the next round."
    others_expectation_of_me - FREE TEXT, e.g. "Others likely expect me
                               to contribute around 5."

The two free-text fields are parsed to a single float via a first-number
heuristic (see parse_expectation_number docstring below). Spot-checked on a
random sample of ~31.7k real (model, seed, round, agent) values pulled from
the local-variant logs: 99.24% yield a parsed number, the rest are pure
qualitative statements with no number ("around the group average") and are
left as NaN (excluded from that agent-round's gap, same convention already
used for missing IN/DN). This has NOT been hand-validated against human
coding the way the discourse-coding pipeline was
(code/analysis/discussion/gpt_5_annotation_sl.py) - treat it as a first-pass
heuristic sufficient for an exploratory look, not as validated as the
IN/DN fields.

Gap convention (matches temp_eval_alignment2.py / gap_based_alignment_test.py,
both of which use the SIGNED gap, not absolute value):
    IN_gap  = injunctive_norm            - actual_contribution
    DN_gap  = descriptive_norm           - actual_contribution
    EO_gap  = parsed(expectation_of_others)    - actual_contribution
    OEM_gap = parsed(others_expectation_of_me) - actual_contribution

Scope: matches figures/MAIN_RESULTS' 4-family/10-seed local-variant scope
(gpt-4o-mini, Llama-7B, Mistral-7B, Qwen-7B; seeds 43-52; BASELINE/
NO_SELECTION/NO_DISCUSSION/FULL) - NOT model_specs.py's shared MODEL_SPECS
(which as of 2026-08-18 also carries 13B/70B tiers and GPT-5-mini for the
S2 replication; see project memory on that scope-drift risk). Hardcoded
here rather than imported, deliberately, so this script can't inherit that
drift either.

Outputs (all under figures/exploratory/four_gap_test/ - a new location,
nothing here overwrites any existing figure or export):
    perception_action_gap_four_measures.png/.pdf  - 4 rows (IN/DN/EO/OEM
        gap) x 4 cols (condition) figure, mirroring
        temp_eval_alignment2.py::plot_across_models's layout and style
    four_gap_alignment_results.csv  - regression coefficients, long format
    four_gap_alignment_summary.md   - plain-language summary
"""

import os
import re
import json
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ─── Config (matches figures/MAIN_RESULTS' 4-family/10-seed local scope) ──────
RESULTS    = "/data3/rasimura/social-norm-evo/results"
FIG_OUT    = "/data3/rasimura/social-norm-evo/figures/exploratory/four_gap_test"
VARIANT    = "local"
MODELS     = ["gpt", "llama", "mistral", "qwen"]
MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
MODEL_COLORS = {"gpt": "#1f77b4", "llama": "#ff7f0e", "mistral": "#2ca02c", "qwen": "#9467bd"}
MODEL_MARKERS = {"gpt": "o", "llama": "s", "mistral": "^", "qwen": "D"}
SEEDS      = list(range(43, 53))
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
COND_LABELS = {
    "BASELINE":      "E",
    "NO_SELECTION":  "E + SL",
    "NO_DISCUSSION": "E + SS",
    "FULL":          "E + SS + SL",
}
ROUNDS = list(range(1, 21))
FAMILY_REF = "GPT"

GAP_KEYS   = ["IN_gap", "DN_gap", "EO_gap", "OEM_gap"]
GAP_LABELS = {
    "IN_gap":  "Injunctive norm − Contribution",
    "DN_gap":  "Descriptive norm − Contribution",
    "EO_gap":  "Expect-others − Contribution",
    "OEM_gap": "Others'-expectation-of-me − Contribution",
}

LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 16, 14, 15


# ─── Free-text -> number parser ────────────────────────────────────────────────

def parse_expectation_number(text):
    """
    Heuristic first-number extractor for the expectation_of_others /
    others_expectation_of_me free-text fields.

    Priority:
      1. "X / 10" or "X/10" (agents phrase answers as "6 / 10") -> take X,
         not the denominator.
      2. "X-Y" or "X to Y" range -> average of the two endpoints.
      3. First standalone number in the text.
      4. None if no number is present (pure qualitative answer, e.g.
         "around the group average") - caller should drop it, same as a
         missing injunctive_norm/descriptive_norm.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    t = str(text)
    m = re.search(r"(\d+\.?\d*)\s*/\s*10\b", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*(?:-|to)\s*(\d+\.?\d*)", t)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.search(r"(\d+\.?\d*)", t)
    if m:
        return float(m.group(1))
    return None


# ─── Data loading ───────────────────────────────────────────────────────────────

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


def build_gap_store():
    """
    store[model][condition][seed][round] = {gap_key: mean-across-agents float}
    Mirrors temp_eval_alignment2.py::compute_seed_gaps/build_store, extended
    to all 4 gap keys.
    """
    store = {m: {c: {} for c in CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                per_round = {}
                for r in d["round_logs"]:
                    rnd = r["round"]
                    contributions = {int(k): float(v) for k, v in r["contributions"].items()}
                    perceptions = {int(k): v for k, v in (r.get("perceptions") or {}).items()}
                    gaps = {k: [] for k in GAP_KEYS}
                    for aid, perc in perceptions.items():
                        if not perc:
                            continue
                        actual = contributions.get(aid)
                        if actual is None:
                            continue
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        eo   = parse_expectation_number(perc.get("expectation_of_others"))
                        oem  = parse_expectation_number(perc.get("others_expectation_of_me"))
                        if inj  is not None: gaps["IN_gap"].append(float(inj) - actual)
                        if desc is not None: gaps["DN_gap"].append(float(desc) - actual)
                        if eo   is not None: gaps["EO_gap"].append(eo - actual)
                        if oem  is not None: gaps["OEM_gap"].append(oem - actual)
                    per_round[rnd] = {k: (float(np.mean(v)) if v else np.nan) for k, v in gaps.items()}
                if per_round:
                    store[model][cond][seed] = per_round
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


# ─── Figure: 4 rows (gap types) x 4 cols (conditions) ──────────────────────────

def plot_four_gaps(store):
    rounds = np.array(ROUNDS)
    fig, axes = plt.subplots(len(GAP_KEYS), len(CONDITIONS),
                             figsize=(4.2 * len(CONDITIONS), 3.2 * len(GAP_KEYS)),
                             sharex=True, sharey="row")

    for col, cond in enumerate(CONDITIONS):
        for row, gap_key in enumerate(GAP_KEYS):
            ax = axes[row, col]
            for model in MODELS:
                series = seed_series(store, model, cond, gap_key)
                if not series:
                    continue
                mean, se = mean_se(series)
                color = MODEL_COLORS[model]
                ax.plot(rounds, mean, color=color, linewidth=2,
                        marker=MODEL_MARKERS[model], markersize=5,
                        label=MODEL_LABELS[model])
                ax.fill_between(rounds, mean - se, mean + se, color=color, alpha=0.12)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(GAP_LABELS[gap_key], fontsize=LABEL_SIZE - 3)
            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
            if row == len(GAP_KEYS) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODELS),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.015))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.align_ylabels(axes[:, 0])

    os.makedirs(FIG_OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIG_OUT, f"perception_action_gap_four_measures.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {out}")
    plt.close(fig)


# ─── Statistical test: extend gap_based_alignment_test.py's primary spec ──────

def build_agent_round_panel():
    """
    One row per (model, condition, seed, agent, round): contribution_t,
    contribution_{t+1}, the 4 gaps, plus condition/family/round/run_id for
    the FE spec. Built fresh from raw logs (round_level_agent_panel.csv
    doesn't carry EO/OEM) - mirrors gap_based_alignment_test.py's variable
    definitions exactly for the 2 fields it already has.
    """
    rows = []
    for model in MODELS:
        family = MODEL_LABELS[model]
        for cond in CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, seed, cond)
                if d is None:
                    continue
                run_id = f"{family}_s{seed}_{cond}"
                round_logs = {r["round"]: r for r in d["round_logs"]}
                sorted_rounds = sorted(round_logs)
                for i, rnd in enumerate(sorted_rounds[:-1]):
                    r_t = round_logs[rnd]
                    r_t1 = round_logs[sorted_rounds[i + 1]]
                    contrib_t = {int(k): float(v) for k, v in r_t["contributions"].items()}
                    contrib_t1 = {int(k): float(v) for k, v in r_t1["contributions"].items()}
                    percs = {int(k): v for k, v in (r_t.get("perceptions") or {}).items()}
                    for aid, perc in percs.items():
                        if not perc:
                            continue
                        c_t = contrib_t.get(aid)
                        c_t1 = contrib_t1.get(aid)
                        if c_t is None or c_t1 is None:
                            continue
                        inj  = perc.get("injunctive_norm")
                        desc = perc.get("descriptive_norm")
                        eo   = parse_expectation_number(perc.get("expectation_of_others"))
                        oem  = parse_expectation_number(perc.get("others_expectation_of_me"))
                        if inj is None or desc is None or eo is None or oem is None:
                            continue
                        rows.append({
                            "run_id": run_id, "family": family, "condition": cond,
                            "seed": seed, "round": rnd, "agent_id": aid,
                            "contribution": c_t, "contribution_shift_lead1": c_t1 - c_t,
                            "in_gap":  float(inj)  - c_t,
                            "dn_gap":  float(desc) - c_t,
                            "eo_gap":  eo  - c_t,
                            "oem_gap": oem - c_t,
                        })
    return pd.DataFrame(rows)


def run_four_gap_regression(df):
    formula = (
        "contribution_shift_lead1 ~ in_gap + dn_gap + eo_gap + oem_gap "
        f'+ C(condition, Treatment(reference="BASELINE")) '
        f'+ C(family, Treatment(reference="{FAMILY_REF}")) '
        "+ C(round, Treatment(reference=1))"
    )
    m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["run_id"]})
    return m, formula


def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""


def write_outputs(m, formula, df):
    os.makedirs(FIG_OUT, exist_ok=True)

    rows = []
    for term in ["in_gap", "dn_gap", "eo_gap", "oem_gap"]:
        rows.append({
            "term": term, "estimate": m.params[term], "se": m.bse[term],
            "p_value": m.pvalues[term], "n": int(m.nobs),
            "n_clusters": df["run_id"].nunique(),
        })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(FIG_OUT, "four_gap_alignment_results.csv"), index=False)

    lines = [
        "# Four-gap alignment test (EXPLORATORY, not canonical)",
        "",
        "Extends gap_based_alignment_test.py's primary specification "
        "(`contribution_shift_lead1 ~ in_gap + dn_gap + condition/family/round FE, "
        "cluster by run`) with two more gap terms built from the "
        "`expectation_of_others` / `others_expectation_of_me` free-text perception "
        "fields, numerically parsed via a first-number heuristic "
        "(see script docstring; ~99.2% parse rate on a spot-check sample, "
        "not hand-validated).",
        "",
        f"Formula: `{formula}`",
        f"N = {int(m.nobs):,}, clusters (run_id) = {df['run_id'].nunique()}",
        "",
        "| Term | Estimate | SE | p | |",
        "|---|---|---|---|---|",
    ]
    for _, row in res.iterrows():
        lines.append(f"| {row['term']} | {row['estimate']:+.4f} | {row['se']:.4f} | "
                     f"{row['p_value']:.4g} | {stars(row['p_value'])} |")

    lines += [
        "",
        "**This has not replaced or modified gap_based_alignment_test.py, "
        "its outputs in exports/alignment_by_gap/, or MAIN_PAPER_RESULTS.md.** "
        "Purely exploratory, per request.",
    ]
    out_md = os.path.join(FIG_OUT, "four_gap_alignment_summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved -> {out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    print("=" * 74)
    print("Building 4-gap store (IN/DN/EO/OEM) for the perception-action figure ...")
    print("=" * 74)
    store = build_gap_store()
    for model in MODELS:
        n_full = len(store[model].get("FULL", {}))
        print(f"  {MODEL_LABELS[model]}: {n_full} seeds loaded for FULL")
    plot_four_gaps(store)

    print("\n" + "=" * 74)
    print("Building agent-round panel for the 4-gap alignment regression ...")
    print("=" * 74)
    df = build_agent_round_panel()
    print(f"  {len(df):,} agent-round rows, {df['run_id'].nunique()} run_ids")

    m, formula = run_four_gap_regression(df)
    write_outputs(m, formula, df)
