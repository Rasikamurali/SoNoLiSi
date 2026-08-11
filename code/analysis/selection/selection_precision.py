"""
selection_precision.py
----------------------
Selection-precision metrics for the weighted group-formation mechanism.

For each round t >= 2, the previous round's contributions define a
top-quartile threshold. We then ask:

  Seed precision   — fraction of seeds (groups[k][0]) that were top-quartile
                     contributors in round t-1
  Group precision  — fraction of ALL group members that were top-quartile
                     contributors in round t-1

Computed for conditions with weighted selection: NO_DISCUSSION and FULL.
Aggregated by model family × period (early: rounds 1-3, late: max-2..max).

Output
------
  figures/2026-03-22/paper_stats/selection_precision_no_discussion.tex
  figures/2026-03-22/paper_stats/selection_precision_full.tex
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE    = "/data3/rasimura/social-norm-evo"
OUT_DIR = f"{BASE}/figures/2026-03-22/paper_stats"

# ── Model specs ───────────────────────────────────────────────────────────────
# (model_key, family_label, results_dir, variant, seeds)
MODEL_SPECS = [
    ("gpt",         "GPT",         f"{BASE}/results",      "global", list(range(43, 53))),
    ("llama",       "Llama-7B",    f"{BASE}/results",      "global", list(range(43, 53))),
    ("mistral",     "Mistral-7B",  f"{BASE}/results",      "global", list(range(43, 53))),
    ("qwen",        "Qwen-7B",     f"{BASE}/results",      "global", list(range(43, 53))),
    ("llama_13b",   "Llama-13B",   f"{BASE}/results",      "local",  list(range(42, 52))),
    ("mistral_13b", "Mistral-13B", f"{BASE}/results",      "local",  list(range(42, 52))),
    ("qwen_14b",    "Qwen-14B",    f"{BASE}/results",      "local",  list(range(42, 52))),
    ("llama_70b",   "Llama-70B",   f"{BASE}/code/results", "local",  list(range(42, 52))),
    ("qwen_72b",    "Qwen-72B",    f"{BASE}/code/results", "local",  list(range(42, 52))),
]
SELECT_CONDITIONS = ["NO_DISCUSSION", "FULL"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_log(results_dir, model_key, variant, seed):
    """Return dict of {condition: log_dict} for one seed."""
    pattern = os.path.join(results_dir, model_key, variant, f"seed{seed}", "log_*.json")
    best = {}
    for path in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(path))
            best[d["condition"]] = d
        except Exception:
            continue
    return best


def precision_rows(d, family, model_key, seed, condition):
    """
    Given a single log_dict for one (seed, condition), yield one row per
    round t >= 2 with seed_precision and group_precision relative to round t-1.
    """
    rounds = d.get("round_logs", [])
    if not rounds:
        return

    # Build lookup: round → contributions dict
    contrib_by_round = {}
    for r in rounds:
        rnum = r["round"]
        cont = {int(k): float(v) for k, v in r["contributions"].items()}
        contrib_by_round[rnum] = cont

    run_id    = f"{family}_s{seed}_{condition}"
    max_round = max(contrib_by_round)

    for r in rounds:
        rnum   = r["round"]
        groups = r.get("groups", [])
        if not groups or rnum < 2:
            continue  # round 1 has no prior contributions

        prior = contrib_by_round.get(rnum - 1, {})
        if not prior:
            continue

        # Top-quartile threshold from prior round
        prior_vals = np.array(list(prior.values()))
        thresh     = np.percentile(prior_vals, 75)

        # Identify all seeds and all group members (excluding small leftover groups)
        seed_ids    = []
        all_members = []
        for g in groups:
            if len(g) < 2:
                continue
            seed_ids.append(g[0])
            all_members.extend(g)

        if not seed_ids:
            continue

        def top_q(agent_id):
            return prior.get(agent_id, np.nan) >= thresh

        seed_top  = [top_q(a) for a in seed_ids   if not np.isnan(prior.get(a, np.nan))]
        group_top = [top_q(a) for a in all_members if not np.isnan(prior.get(a, np.nan))]

        seed_prec  = np.mean(seed_top)  if seed_top  else np.nan
        group_prec = np.mean(group_top) if group_top else np.nan

        # All-equal flag: contributions trivially tie → top_share/top_q is uninformative
        all_equal = float(np.std(prior_vals) < 1e-6)

        yield {
            "family":      family,
            "model":       model_key,
            "seed":        seed,
            "condition":   condition,
            "run_id":      run_id,
            "round":       rnum,
            "max_round":   max_round,
            "seed_prec":   seed_prec,
            "group_prec":  group_prec,
            "all_equal":   all_equal,
            "n_seeds":     len(seed_ids),
            "n_members":   len(all_members),
        }


def load_all():
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        for seed in seeds:
            logs = load_log(results_dir, model_key, variant, seed)
            for cond in SELECT_CONDITIONS:
                d = logs.get(cond)
                if d is None:
                    continue
                for row in precision_rows(d, family, model_key, seed, cond):
                    rows.append(row)
    return pd.DataFrame(rows)


# ── Period assignment ─────────────────────────────────────────────────────────

def assign_period(df):
    """early: rounds 1-3; late: max_round-2 .. max_round (per run_id)."""
    df = df.copy()
    df["period"] = "mid"
    df.loc[df["round"] <= 3, "period"] = "early"
    late_mask = df["round"] >= df["max_round"] - 2
    df.loc[late_mask, "period"] = "late"
    return df


# ── Summary statistics ────────────────────────────────────────────────────────

def summarize(df, condition):
    """Return a summary DataFrame indexed by (family, period)."""
    sub = df[df["condition"] == condition].copy()

    # Optionally flag trivial rounds (all-equal prior contributions)
    n_trivial = sub["all_equal"].sum()
    n_total   = len(sub)

    # Aggregate: mean and SD across seeds × rounds, per family × period
    agg = (sub.groupby(["family", "period"])[["seed_prec", "group_prec"]]
              .agg(["mean", "std"])
              .reset_index())
    agg.columns = ["family", "period",
                   "seed_mean", "seed_sd",
                   "group_mean", "group_sd"]

    # Add N observations per cell
    n_obs = (sub.groupby(["family", "period"])
                .size()
                .reset_index(name="n_obs"))
    agg = agg.merge(n_obs, on=["family", "period"])

    return agg, n_trivial, n_total


# ── LaTeX table ───────────────────────────────────────────────────────────────

FAMILY_ORDER = [
    "GPT", "Llama-7B", "Mistral-7B", "Qwen-7B",
    "Llama-13B", "Mistral-13B", "Qwen-14B",
    "Llama-70B", "Qwen-72B",
]
COND_LABEL = {"NO_DISCUSSION": "No Discussion", "FULL": "Full"}


def fmt_cell(mean, sd):
    if np.isnan(mean):
        return "---"
    return rf"{mean:.2f} ({sd:.2f})"


def make_table(agg, condition, n_trivial, n_total):
    cond_label = COND_LABEL[condition]

    # Pivot so rows = family, period columns side-by-side
    pivot = agg.set_index(["family", "period"])

    rows_early = {fam: pivot.loc[(fam, "early")] if (fam, "early") in pivot.index else None
                  for fam in FAMILY_ORDER}
    rows_late  = {fam: pivot.loc[(fam, "late")]  if (fam, "late")  in pivot.index else None
                  for fam in FAMILY_ORDER}

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "cc" * 2 + r"}",
        r"\toprule",
        r"  & \multicolumn{2}{c}{Early (rounds 1--3)} "
        r"& \multicolumn{2}{c}{Late (rounds $T{-}2$ to $T$)} \\",
        r"  \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"  Model & Seed prec. & Group prec. & Seed prec. & Group prec. \\",
        r"\midrule",
    ]

    for fam in FAMILY_ORDER:
        e = rows_early[fam]
        l = rows_late[fam]

        def cell(row, metric):
            if row is None:
                return "---"
            return fmt_cell(row[f"{metric}_mean"], row[f"{metric}_sd"])

        e_seed  = cell(e, "seed")
        e_group = cell(e, "group")
        l_seed  = cell(l, "seed")
        l_group = cell(l, "group")

        lines.append(rf"  {fam} & {e_seed} & {e_group} & {l_seed} & {l_group} \\")

    pct_trivial = 100.0 * n_trivial / n_total if n_total > 0 else 0.0

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Selection precision under the \emph{" + cond_label + r"} condition. "
         r"\emph{Seed precision}: fraction of seed agents (groups$_k[0]$, selected "
         r"by weighted random draw proportional to average incoming weight) whose "
         r"round-$t-1$ contribution was at or above the 75th percentile of that round's "
         r"contributions. "
         r"\emph{Group precision}: same fraction for all group members. "
         r"Baseline (random selection) $\approx 0.25$. "
         r"Cells report mean (SD) across seeds and rounds in that period. "
         rf"Note: {pct_trivial:.1f}\% of round-pairs have perfectly equal prior contributions "
         r"(all agents cooperate fully), making the 75th-percentile threshold trivially "
         r"met by everyone; these rounds yield precision = 1.0 by construction and are "
         r"included in the reported means.}"),
        rf"\label{{tab:selection_precision_{condition.lower()}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df = load_all()
    print(f"  {len(df):,} round observations across "
          f"{df['run_id'].nunique()} runs")

    df = assign_period(df)

    os.makedirs(OUT_DIR, exist_ok=True)

    for cond in SELECT_CONDITIONS:
        agg, n_trivial, n_total = summarize(df, cond)
        tex = make_table(agg, cond, n_trivial, n_total)
        fname = f"selection_precision_{cond.lower()}.tex"
        out_path = os.path.join(OUT_DIR, fname)
        with open(out_path, "w") as f:
            f.write(tex)
        print(f"\n  [{cond}] → {out_path}")

        # Plain-text summary
        print(f"\n  {COND_LABEL[cond]}:  {n_trivial}/{n_total} "
              f"({100*n_trivial/n_total:.1f}%) trivial all-equal rounds")
        sub = df[df["condition"] == cond]
        for period in ["early", "late"]:
            psub = sub[sub["period"] == period]
            if psub.empty:
                continue
            print(f"    {period.upper():5s} | seed_prec={psub['seed_prec'].mean():.3f} "
                  f"group_prec={psub['group_prec'].mean():.3f}  "
                  f"(N={len(psub):,} rounds)")
            for fam in FAMILY_ORDER:
                fsub = psub[psub["family"] == fam]
                if fsub.empty:
                    continue
                print(f"           {fam:14s}  seed={fsub['seed_prec'].mean():.3f}  "
                      f"group={fsub['group_prec'].mean():.3f}  "
                      f"(n_rounds={len(fsub)})")


if __name__ == "__main__":
    main()
