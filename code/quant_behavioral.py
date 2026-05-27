"""
quant_behavioral.py
===================
Run-level behavioral metrics with bootstrap statistical comparisons.

Step 1 — Aggregate to run-level metrics
  For each (model, seed, condition, setting):
    mean_contribution   : mean contribution across all agents and rounds
    final_contribution  : mean contribution in the final round
    mean_payoff         : mean payoff across all agents and rounds
    exclusion_rate      : fraction of (agent, round) pairs where agent was excluded

Step 2 — Statistical comparisons (bootstrap, n=1000)
  A. Across conditions — for each (model, setting): all condition pairs
  B. Global vs Local  — for each (model, condition): paired by seed

Step 3 — Output
  Summary tables (mean ± CI) and pairwise comparison tables to CSV/print

Step 4 — Visualization
  Bar plots (CI as error bars), one figure per model per metric

Step 5 — Interpretation helper
  Brief factual summary printed to stdout
"""

import glob
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS   = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT  = "/data3/rasimura/social-norm-evo/figures/2026-03-20"
TABLE_DIR = os.path.join(FIG_ROOT, "behavioral_tables")
os.makedirs(TABLE_DIR, exist_ok=True)

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANTS   = ["global", "local"]
SEEDS      = list(range(43, 53))
CONDITIONS = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION", "PURE_BASELINE"]

N_BOOT = 1000
np.random.seed(0)

COND_COLORS = {
    "BASELINE":       "#1f77b4",
    "FULL":           "#2ca02c",
    "NO_DISCUSSION":  "#9467bd",
    "NO_SELECTION":   "#8c564b",
    "PURE_BASELINE":  "#7f7f7f",
}
METRICS = ["mean_contribution", "final_contribution", "mean_payoff", "exclusion_rate"]
METRIC_LABELS = {
    "mean_contribution":  "Mean contribution",
    "final_contribution": "Final-round contribution",
    "mean_payoff":        "Mean payoff",
    "exclusion_rate":     "Exclusion rate",
}


# ─── Step 1: Data loading & run-level metrics ─────────────────────────────────

def load_latest_log(model: str, variant: str, seed: int, condition: str):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    cond_best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            cond_best[d["condition"]] = d
        except Exception:
            continue
    return cond_best.get(condition)


def compute_run_metrics(data: dict) -> dict:
    """
    Given a single run log, return a dict of scalar metrics.
    exclusion_rate = fraction of (agent, round) observations where agent was excluded.
    """
    all_contributions = []
    all_payoffs       = []
    excluded_obs      = 0
    total_obs         = 0
    final_round_contribs = []

    round_logs = data["round_logs"]
    n_rounds   = len(round_logs)

    for rlog in round_logs:
        contribs  = {int(k): v for k, v in rlog["contributions"].items()}
        payoffs   = {int(k): v for k, v in rlog["payoffs"].items()}
        excluded  = set(rlog.get("excluded_agents") or [])
        all_agents = set(contribs) | set(payoffs)

        all_contributions.extend(contribs.values())
        all_payoffs.extend(payoffs.values())

        for agent_id in all_agents:
            total_obs += 1
            if agent_id in excluded:
                excluded_obs += 1

        if rlog["round"] == n_rounds:
            final_round_contribs = list(contribs.values())

    return {
        "mean_contribution":  float(np.mean(all_contributions)) if all_contributions else np.nan,
        "final_contribution": float(np.mean(final_round_contribs)) if final_round_contribs else np.nan,
        "mean_payoff":        float(np.mean(all_payoffs)) if all_payoffs else np.nan,
        "exclusion_rate":     excluded_obs / total_obs if total_obs > 0 else np.nan,
    }


def build_dataframe() -> pd.DataFrame:
    """Load all runs and return a tidy DataFrame with one row per run."""
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    data = load_latest_log(model, variant, seed, condition)
                    if data is None:
                        continue
                    metrics = compute_run_metrics(data)
                    rows.append({
                        "model":     model,
                        "seed":      seed,
                        "condition": condition,
                        "setting":   variant,
                        **metrics,
                    })
    return pd.DataFrame(rows)


# ─── Step 2: Bootstrap helpers ────────────────────────────────────────────────

def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT):
    """
    Unpaired bootstrap: resample a and b independently.
    Returns (mean_diff, ci_low, ci_high) where diff = mean(b) - mean(a).
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan
    diffs = np.array([
        np.mean(np.random.choice(b, len(b), replace=True)) -
        np.mean(np.random.choice(a, len(a), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_paired(local_vals: np.ndarray, global_vals: np.ndarray, n_boot: int = N_BOOT):
    """
    Paired bootstrap resampling seeds with replacement.
    Returns (mean_diff, ci_low, ci_high) where diff = local − global.
    """
    diffs = local_vals - global_vals
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    boot = np.array([
        np.mean(np.random.choice(diffs, len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(boot)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ─── Step 2A: Condition comparisons ──────────────────────────────────────────

def condition_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, setting, metric): all pairwise condition comparisons.
    Returns a DataFrame with columns:
      [model, setting, metric, cond_a, cond_b, mean_diff, ci_low, ci_high]
    where diff = mean(cond_b) - mean(cond_a).
    """
    rows = []
    for model in MODELS:
        for setting in VARIANTS:
            sub = df[(df["model"] == model) & (df["setting"] == setting)]
            available_conds = sub["condition"].unique()
            for ca, cb in combinations(sorted(available_conds), 2):
                for metric in METRICS:
                    a = sub[sub["condition"] == ca][metric].values
                    b = sub[sub["condition"] == cb][metric].values
                    md, ci_lo, ci_hi = bootstrap_diff(a, b)
                    rows.append({
                        "model":     model,
                        "setting":   setting,
                        "metric":    metric,
                        "cond_a":    ca,
                        "cond_b":    cb,
                        "mean_diff": md,
                        "ci_low":    ci_lo,
                        "ci_high":   ci_hi,
                    })
    return pd.DataFrame(rows)


# ─── Step 2B: Global vs Local paired comparison ───────────────────────────────

def global_local_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, condition, metric): paired diff = local − global, by seed.
    Returns DataFrame with columns:
      [model, condition, metric, mean_diff, ci_low, ci_high]
    """
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            sub = df[(df["model"] == model) & (df["condition"] == condition)]
            for metric in METRICS:
                g = sub[sub["setting"] == "global"].set_index("seed")[metric]
                l = sub[sub["setting"] == "local"].set_index("seed")[metric]
                shared = g.index.intersection(l.index)
                if len(shared) == 0:
                    continue
                gv = g.loc[shared].values.astype(float)
                lv = l.loc[shared].values.astype(float)
                md, ci_lo, ci_hi = bootstrap_paired(lv, gv)
                rows.append({
                    "model":     model,
                    "condition": condition,
                    "metric":    metric,
                    "mean_diff": md,
                    "ci_low":    ci_lo,
                    "ci_high":   ci_hi,
                })
    return pd.DataFrame(rows)


# ─── Step 3: Summary tables ───────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± CI (bootstrap) for each (model, condition, setting, metric)."""
    rows = []
    for model in MODELS:
        for condition in CONDITIONS:
            for setting in VARIANTS:
                sub = df[
                    (df["model"] == model) &
                    (df["condition"] == condition) &
                    (df["setting"] == setting)
                ]
                if sub.empty:
                    continue
                for metric in METRICS:
                    vals = sub[metric].dropna().values
                    if len(vals) == 0:
                        continue
                    boot = np.array([
                        np.mean(np.random.choice(vals, len(vals), replace=True))
                        for _ in range(N_BOOT)
                    ])
                    rows.append({
                        "model":     model,
                        "condition": condition,
                        "setting":   setting,
                        "metric":    metric,
                        "mean":      float(np.mean(vals)),
                        "ci_low":    float(np.percentile(boot, 2.5)),
                        "ci_high":   float(np.percentile(boot, 97.5)),
                        "n_seeds":   len(vals),
                    })
    return pd.DataFrame(rows)


def print_summary(summ: pd.DataFrame) -> None:
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model.upper()}")
        print(f"{'='*60}")
        sub = summ[summ["model"] == model]
        for metric in METRICS:
            print(f"\n  {METRIC_LABELS[metric]}:")
            tbl = sub[sub["metric"] == metric].pivot_table(
                index="condition", columns="setting",
                values=["mean", "ci_low", "ci_high"], aggfunc="first"
            )
            for cond in CONDITIONS:
                if cond not in tbl.index:
                    continue
                parts = []
                for setting in VARIANTS:
                    try:
                        m  = tbl.loc[cond, ("mean",   setting)]
                        lo = tbl.loc[cond, ("ci_low", setting)]
                        hi = tbl.loc[cond, ("ci_high",setting)]
                        parts.append(f"{setting}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
                    except KeyError:
                        pass
                print(f"    {cond:16s}  " + "   |   ".join(parts))


def print_condition_comparisons(comp: pd.DataFrame) -> None:
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model.upper()} — Condition comparisons")
        print(f"{'='*60}")
        sub = comp[comp["model"] == model]
        for setting in VARIANTS:
            print(f"\n  [{setting}]")
            ssub = sub[sub["setting"] == setting]
            for metric in METRICS:
                print(f"    {METRIC_LABELS[metric]}:")
                msub = ssub[ssub["metric"] == metric]
                for _, row in msub.iterrows():
                    sig = "*" if (row["ci_low"] > 0 or row["ci_high"] < 0) else " "
                    print(f"      {sig} {row['cond_a']:16s} vs {row['cond_b']:16s}"
                          f"  Δ={row['mean_diff']:+.3f}  CI=[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")


def print_global_local(gl: pd.DataFrame) -> None:
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model.upper()} — Global vs Local (Δ = local − global)")
        print(f"{'='*60}")
        sub = gl[gl["model"] == model]
        for metric in METRICS:
            print(f"\n  {METRIC_LABELS[metric]}:")
            msub = sub[sub["metric"] == metric]
            for _, row in msub.iterrows():
                sig = "*" if (row["ci_low"] > 0 or row["ci_high"] < 0) else " "
                print(f"    {sig} {row['condition']:16s}"
                      f"  Δ={row['mean_diff']:+.3f}  CI=[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")


# ─── Step 4: Visualization ────────────────────────────────────────────────────

def plot_bar_charts(summ: pd.DataFrame, model: str) -> None:
    """
    2 subplots (mean_contribution, final_contribution), x=condition,
    two bar groups (global/local), error bars = 95% CI.
    """
    plot_metrics = ["mean_contribution", "final_contribution"]
    sub = summ[summ["model"] == model]
    conds = [c for c in CONDITIONS if c in sub["condition"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes.flat, plot_metrics):
        msub = sub[sub["metric"] == metric]
        x    = np.arange(len(conds))
        w    = 0.35

        for i, setting in enumerate(VARIANTS):
            ssub = msub[msub["setting"] == setting].set_index("condition")
            means  = [ssub.loc[c, "mean"]    if c in ssub.index else np.nan for c in conds]
            ci_lo  = [ssub.loc[c, "ci_low"]  if c in ssub.index else np.nan for c in conds]
            ci_hi  = [ssub.loc[c, "ci_high"] if c in ssub.index else np.nan for c in conds]
            errs   = [
                [m - lo for m, lo in zip(means, ci_lo)],
                [hi - m for m, hi in zip(means, ci_hi)],
            ]
            offset = (i - 0.5) * w
            col    = "#2c7bb6" if setting == "global" else "#d7191c"
            ax.bar(x + offset, means, w, label=setting.capitalize(),
                   color=col, alpha=0.8, edgecolor="white")
            ax.errorbar(x + offset, means, yerr=errs, fmt="none",
                        color="black", capsize=3, lw=1.2)

        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{model.upper()} — Behavioral metrics by condition & setting\n"
                 f"(error bars = 95% bootstrap CI across seeds)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG_ROOT, model, "selection_analysis", "behavioral_bars.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def plot_seed_distributions(df: pd.DataFrame, model: str) -> None:
    """Boxplots of seed-level distributions per condition and setting."""
    plot_metrics = ["mean_contribution", "final_contribution"]
    sub   = df[df["model"] == model]
    conds = [c for c in CONDITIONS if c in sub["condition"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes.flat, plot_metrics):
        data_g = [sub[(sub["condition"] == c) & (sub["setting"] == "global")][metric].dropna().values
                  for c in conds]
        data_l = [sub[(sub["condition"] == c) & (sub["setting"] == "local")][metric].dropna().values
                  for c in conds]

        x = np.arange(len(conds))
        w = 0.35
        bp_kw = dict(widths=w * 0.85, patch_artist=True, medianprops=dict(color="black", lw=2))

        for offset, data, col, label in [
            (-w / 2, data_g, "#2c7bb6", "global"),
            ( w / 2, data_l, "#d7191c", "local"),
        ]:
            positions = x + offset
            bp = ax.boxplot(data, positions=positions, **bp_kw)
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_alpha(0.6)
            bp["boxes"][0].set_label(label.capitalize())   # legend entry

        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{model.upper()} — Seed-level distributions by condition & setting",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(FIG_ROOT, model, "selection_analysis", "behavioral_boxplots.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ─── Step 5: Interpretation helper ───────────────────────────────────────────

def interpret(df: pd.DataFrame, summ: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("  INTERPRETATION SUMMARY")
    print("="*60)

    for model in MODELS:
        sub = summ[(summ["model"] == model) & (summ["metric"] == "mean_contribution")]
        if sub.empty:
            continue
        best_row = sub.loc[sub["mean"].idxmax()]
        print(f"\n  {model.upper()}:")
        print(f"    Highest cooperation: {best_row['condition']} [{best_row['setting']}]"
              f"  (mean = {best_row['mean']:.3f})")

        # global vs local: is there a systematic difference in mean_contribution?
        gl_diff = summ[
            (summ["model"] == model) & (summ["metric"] == "mean_contribution")
        ].groupby("setting")["mean"].mean()
        if "global" in gl_diff and "local" in gl_diff:
            delta = gl_diff["local"] - gl_diff["global"]
            direction = "higher" if delta > 0 else "lower"
            print(f"    Local vs global contribution: local is {abs(delta):.3f} {direction} on average")

        # exclusion rate: which condition excludes most?
        excl_sub = summ[(summ["model"] == model) & (summ["metric"] == "exclusion_rate")]
        if not excl_sub.empty and excl_sub["mean"].notna().any():
            best_excl = excl_sub.loc[excl_sub["mean"].idxmax()]
            print(f"    Highest exclusion: {best_excl['condition']} [{best_excl['setting']}]"
                  f"  (rate = {best_excl['mean']:.4f})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = build_dataframe()
    print(f"  Loaded {len(df)} runs across {df['model'].nunique()} models, "
          f"{df['seed'].nunique()} seeds, {df['condition'].nunique()} conditions, "
          f"{df['setting'].nunique()} settings.")

    print("\nComputing summary statistics...")
    summ = summary_table(df)

    print("\nRunning condition comparisons (bootstrap)...")
    comp = condition_comparisons(df)

    print("\nRunning global vs local comparisons (paired bootstrap)...")
    gl = global_local_comparisons(df)

    # ── Save tables to CSV ────────────────────────────────────────────────────
    df.to_csv(  os.path.join(TABLE_DIR, "run_metrics.csv"),        index=False)
    summ.to_csv(os.path.join(TABLE_DIR, "summary_table.csv"),      index=False)
    comp.to_csv(os.path.join(TABLE_DIR, "condition_comparisons.csv"), index=False)
    gl.to_csv(  os.path.join(TABLE_DIR, "global_local_comparisons.csv"), index=False)
    print(f"\n  Tables saved to {TABLE_DIR}/")

    # ── Print results ─────────────────────────────────────────────────────────
    print_summary(summ)
    print_condition_comparisons(comp)
    print_global_local(gl)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    for model in MODELS:
        plot_bar_charts(summ, model)
        plot_seed_distributions(df, model)

    # ── Interpretation ────────────────────────────────────────────────────────
    interpret(df, summ)

    print("\nDone.")


if __name__ == "__main__":
    main()
