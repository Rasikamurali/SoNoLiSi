"""
stability_analysis.py
---------------------
Rate of convergence: how quickly agents adopt the same IN, DN, and contribution.

Metric: cross-agent standard deviation per round per seed.
  contrib_sd — SD of contributions across agents
  in_sd      — SD of injunctive norm across agents (agents with valid perceptions)
  dn_sd      — SD of descriptive norm across agents

Lower SD over time = faster / stronger consensus toward the shared norm and behaviour.

Conditions: BASELINE, NO_SELECTION, NO_DISCUSSION, FULL  (PURE_BASELINE skipped).

Layout: 4 rows (models) × 4 cols (conditions).
Each subplot: 3 lines (contrib SD, IN SD, DN SD) — cross-seed mean ± SE ribbon.

Output: figures/2026-03-22/cross_model/stability_analysis.png / .pdf
"""

import json
import glob
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Config ───────────────────────────────────────────────────────────────────
RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANT    = "global"
SEEDS      = list(range(43, 53))
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
ROUNDS     = list(range(1, 21))

COND_LABELS = {
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
MODEL_LABELS = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}

METRIC_COLORS = {
    "contrib_sd": "#1f77b4",
    "in_sd":      "#e377c2",
    "dn_sd":      "#17becf",
}
METRIC_LABELS_PLOT = {
    "contrib_sd": "Contribution SD",
    "in_sd":      "IN SD",
    "dn_sd":      "DN SD",
}

LABEL_SIZE  = 16
TICK_SIZE   = 14
LEGEND_SIZE = 16

# ─── Data loading ─────────────────────────────────────────────────────────────

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


# ─── Per-seed metric computation ──────────────────────────────────────────────

def compute_seed_sds(model, seed, condition):
    """
    Returns {round: {contrib_sd, in_sd, dn_sd}} for one seed.
    in_sd and dn_sd are NaN if fewer than 2 agents have valid perceptions.
    """
    d = load_latest_log(model, seed, condition)
    if d is None:
        return {}

    result = {}
    for r in d["round_logs"]:
        rnd   = r["round"]
        cont  = [float(v) for v in r["contributions"].values()]
        percs = r.get("perceptions") or {}

        in_vals, dn_vals = [], []
        for perc in percs.values():
            if not perc:
                continue
            inj  = perc.get("injunctive_norm")
            desc = perc.get("descriptive_norm")
            if inj  is not None: in_vals.append(float(inj))
            if desc is not None: dn_vals.append(float(desc))

        result[rnd] = {
            "contrib_sd": float(np.std(cont,     ddof=1)) if len(cont)    >= 2 else np.nan,
            "in_sd":      float(np.std(in_vals,  ddof=1)) if len(in_vals) >= 2 else np.nan,
            "dn_sd":      float(np.std(dn_vals,  ddof=1)) if len(dn_vals) >= 2 else np.nan,
        }
    return result


def build_store():
    """store[model][condition][seed] = {round: {metric: float}}"""
    store = {m: {c: {} for c in CONDITIONS} for m in MODELS}
    for model in MODELS:
        for cond in CONDITIONS:
            for seed in SEEDS:
                s = compute_seed_sds(model, seed, cond)
                if s:
                    store[model][cond][seed] = s
    return store


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def seed_series(store, model, cond, metric):
    series = []
    for seed_data in store[model][cond].values():
        arr = np.array([seed_data.get(r, {}).get(metric, np.nan) for r in ROUNDS])
        series.append(arr)
    return series


def mean_se(series_list):
    arr  = np.array(series_list, dtype=float)
    n    = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    se   = np.nanstd(arr, axis=0) / np.sqrt(np.where(n > 0, n, np.nan))
    return mean, se


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_stability(store):
    """
    4 rows (models) × 4 cols (conditions).
    Each subplot: contrib SD, IN SD, DN SD trajectories (mean ± SE across seeds).
    """
    rounds  = np.array(ROUNDS)
    metrics = ["contrib_sd", "in_sd", "dn_sd"]

    fig, axes = plt.subplots(len(MODELS), len(CONDITIONS),
                             figsize=(4 * len(CONDITIONS), 4 * len(MODELS)),
                             sharex=True, sharey=True)

    for row, model in enumerate(MODELS):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]

            for metric in metrics:
                series = seed_series(store, model, cond, metric)
                if not series:
                    continue
                mean, se = mean_se(series)
                color    = METRIC_COLORS[metric]
                ax.plot(rounds, mean, color=color, linewidth=2,
                        label=METRIC_LABELS_PLOT[metric])
                ax.fill_between(rounds, mean - se, mean + se,
                                color=color, alpha=0.15)

            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=TICK_SIZE)
            ax.grid(True, alpha=0.25)

            if col == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=LABEL_SIZE)
            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=LABEL_SIZE)
            if row == len(MODELS) - 1:
                ax.set_xlabel("Round", fontsize=LABEL_SIZE)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(metrics),
               fontsize=LEGEND_SIZE, frameon=False, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir = os.path.join(FIG_ROOT, "cross_model")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"stability_analysis.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
    plt.close(fig)


# ─── Quantification ───────────────────────────────────────────────────────────

METRICS     = ["contrib_sd", "in_sd", "dn_sd"]
METRIC_TEX  = {"contrib_sd": "Contribution", "in_sd": "IN", "dn_sd": "DN"}
MODEL_TEX   = {"gpt": "GPT", "llama": "Llama", "mistral": "Mistral", "qwen": "Qwen"}
COND_TEX    = {
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}
LAST_N = 5
OUT_DIR = os.path.join(FIG_ROOT, "paper_stats")


def per_seed_slope(store, model, cond, metric):
    """
    OLS slope of SD ~ round fitted per seed.
    Returns array of slopes (one per seed), ignoring seeds with <3 valid rounds.
    """
    rounds = np.array(ROUNDS, dtype=float)
    slopes = []
    for seed_data in store[model][cond].values():
        arr = np.array([seed_data.get(r, {}).get(metric, np.nan) for r in ROUNDS])
        mask = ~np.isnan(arr)
        if mask.sum() < 3:
            continue
        x = rounds[mask] - rounds[mask].mean()   # center round
        y = arr[mask]
        slope = np.dot(x, y) / np.dot(x, x)
        slopes.append(slope)
    return np.array(slopes)


def per_seed_final_sd(store, model, cond, metric):
    """
    Mean SD over last LAST_N rounds, per seed.
    Returns array of final-SD values (one per seed).
    """
    final_rounds = ROUNDS[-LAST_N:]
    vals = []
    for seed_data in store[model][cond].values():
        rvals = [seed_data.get(r, {}).get(metric, np.nan) for r in final_rounds]
        rvals = [v for v in rvals if not np.isnan(v)]
        if rvals:
            vals.append(float(np.mean(rvals)))
    return np.array(vals)


def compute_summary(store):
    """
    Returns two DataFrames (slope_df, final_df) with columns:
      model, condition, metric, mean, se
    """
    slope_rows, final_rows = [], []
    for model in MODELS:
        for cond in CONDITIONS:
            for metric in METRICS:
                s = per_seed_slope(store, model, cond, metric)
                if len(s) > 0:
                    slope_rows.append({
                        "model": model, "condition": cond, "metric": metric,
                        "mean": float(np.mean(s)),
                        "se":   float(np.std(s, ddof=1) / np.sqrt(len(s))),
                        "n":    len(s),
                    })
                f = per_seed_final_sd(store, model, cond, metric)
                if len(f) > 0:
                    final_rows.append({
                        "model": model, "condition": cond, "metric": metric,
                        "mean": float(np.mean(f)),
                        "se":   float(np.std(f, ddof=1) / np.sqrt(len(f))),
                        "n":    len(f),
                    })
    import pandas as pd
    return pd.DataFrame(slope_rows), pd.DataFrame(final_rows)


def _cell(mean, se):
    return f"{mean:.3f} ({se:.3f})"


def make_latex_table(df, caption, label, fname):
    """
    Rows = conditions (4), sub-rows = metrics (3) → 12 data rows total.
    Cols = models (4).
    Uses \\multirow for condition labels.
    Requires: booktabs, multirow LaTeX packages.
    """
    import pandas as pd
    col_spec  = "ll" + "r" * len(MODELS)
    model_hdr = " & ".join(MODEL_TEX[m] for m in MODELS)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Condition & Metric & {model_hdr} \\",
        r"\midrule",
    ]

    for ci, cond in enumerate(CONDITIONS):
        n_metrics = len(METRICS)
        for mi, metric in enumerate(METRICS):
            cells = []
            for model in MODELS:
                row = df[(df["model"] == model) &
                         (df["condition"] == cond) &
                         (df["metric"] == metric)]
                cells.append(_cell(row.iloc[0]["mean"], row.iloc[0]["se"])
                             if not row.empty else "--")

            cond_cell = (rf"\multirow{{{n_metrics}}}{{*}}{{{COND_TEX[cond]}}}"
                         if mi == 0 else "")
            lines.append(
                rf"{cond_cell} & {METRIC_TEX[metric]} & {' & '.join(cells)} \\"
            )

        if ci < len(CONDITIONS) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption} Cells show mean (SE) across seeds.}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


def save_csv(df, fname):
    import pandas as pd
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ─── Mixed effects regression ─────────────────────────────────────────────────

def run_mixed_effects(store):
    """
    Mixed effects model as a robustness check for the OLS slopes.

    Model (per model × condition × metric):
        SD_ij ~ β0 + β1 * round_c_j + b_i + ε_ij
    where:
        SD_ij    = cross-agent SD at round j for seed i
        round_c  = round centred on its mean (removes intercept collinearity)
        b_i      ~ N(0, σ²_b)  random intercept per seed
        ε_ij     ~ N(0, σ²)    residual

    Random intercept only (not slope) because N_seeds = 10 is too small for
    reliable random-slope estimation.

    Returns a DataFrame with columns:
        model, condition, metric,
        fe_slope, fe_se, fe_pval,   ← fixed-effect estimate for round_c
        re_var,                      ← estimated random-intercept variance
        n_obs, n_groups,
        converged
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    round_mean = float(np.mean(ROUNDS))
    rows = []

    for model in MODELS:
        for cond in CONDITIONS:
            for metric in METRICS:
                # Build long-format data
                records = []
                for seed, seed_data in store[model][cond].items():
                    for r in ROUNDS:
                        val = seed_data.get(r, {}).get(metric, np.nan)
                        if not np.isnan(val):
                            records.append({
                                "seed":    str(seed),
                                "round_c": r - round_mean,
                                "sd":      val,
                            })

                if len(records) < 10:
                    rows.append({
                        "model": model, "condition": cond, "metric": metric,
                        "fe_slope": np.nan, "fe_se": np.nan, "fe_pval": np.nan,
                        "re_var": np.nan,
                        "n_obs": len(records), "n_groups": 0, "converged": False,
                    })
                    continue

                df = pd.DataFrame(records)
                converged = False
                fe_slope = fe_se = fe_pval = re_var = np.nan

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        me = smf.mixedlm("sd ~ round_c", df,
                                         groups=df["seed"]).fit(reml=True,
                                                                method="lbfgs")
                    fe_slope  = float(me.fe_params["round_c"])
                    fe_se     = float(me.bse["round_c"])
                    fe_pval   = float(me.pvalues["round_c"])
                    re_var    = float(me.cov_re.iloc[0, 0])
                    converged = me.converged
                except Exception as exc:
                    print(f"    ME failed {model} {cond} {metric}: {exc}")

                rows.append({
                    "model": model, "condition": cond, "metric": metric,
                    "fe_slope":  fe_slope,
                    "fe_se":     fe_se,
                    "fe_pval":   fe_pval,
                    "re_var":    re_var,
                    "n_obs":     len(records),
                    "n_groups":  df["seed"].nunique(),
                    "converged": converged,
                })

    return pd.DataFrame(rows)


def _pstar(p):
    """Significance stars for p-value."""
    if np.isnan(p):  return ""
    if p < 0.001:    return "***"
    if p < 0.01:     return "**"
    if p < 0.05:     return "*"
    return ""


def make_latex_table_me(ols_df, me_df, fname):
    """
    Side-by-side OLS slope vs ME fixed-effect slope.
    Each cell: value (SE)[stars].
    Stars derived from ME p-values.
    """
    col_spec  = "ll" + "r" * len(MODELS)
    model_hdr = " & ".join(MODEL_TEX[m] for m in MODELS)

    def ols_cell(model, cond, metric):
        row = ols_df[(ols_df["model"] == model) &
                     (ols_df["condition"] == cond) &
                     (ols_df["metric"] == metric)]
        if row.empty:
            return "--"
        return f"{row.iloc[0]['mean']:.3f} ({row.iloc[0]['se']:.3f})"

    def me_cell(model, cond, metric):
        row = me_df[(me_df["model"] == model) &
                    (me_df["condition"] == cond) &
                    (me_df["metric"] == metric)]
        if row.empty or np.isnan(row.iloc[0]["fe_slope"]):
            return "--"
        r   = row.iloc[0]
        return (f"{r['fe_slope']:.3f} ({r['fe_se']:.3f})"
                + _pstar(r["fe_pval"]))

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Condition & Metric & {model_hdr} \\",
        r"\midrule",
    ]

    for ci, cond in enumerate(CONDITIONS):
        n_metrics = len(METRICS)
        # OLS block
        for mi, metric in enumerate(METRICS):
            cells = [ols_cell(m, cond, metric) for m in MODELS]
            prefix = (rf"\multirow{{{n_metrics * 2}}}{{*}}{{{COND_TEX[cond]}}}"
                      if mi == 0 else "")
            metric_label = METRIC_TEX[metric] + r" \textit{(OLS)}"
            lines.append(rf"{prefix} & {metric_label} & {' & '.join(cells)} \\")

        # ME block
        for mi, metric in enumerate(METRICS):
            cells = [me_cell(m, cond, metric) for m in MODELS]
            metric_label = METRIC_TEX[metric] + r" \textit{(ME)}"
            lines.append(rf" & {metric_label} & {' & '.join(cells)} \\")

        if ci < len(CONDITIONS) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\multicolumn{" + str(2 + len(MODELS)) + r"}{l}{"
        r"\footnotesize $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$ (ME fixed-effect slope).} \\",
        r"\end{tabular}",
        r"\caption{Convergence rate: OLS slope (mean $\pm$ SE across seeds) and "
        r"mixed-effects fixed-effect slope (SE) for SD $\sim$ round (centred). "
        r"ME model includes a random intercept per seed.}",
        r"\label{tab:stability_slope_robustness}",
        r"\end{table}",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved → {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building stability store …")
    store = build_store()

    for model in MODELS:
        for cond in CONDITIONS:
            n = len(store[model][cond])
            print(f"  {model:8s} {cond:15s}: {n} seeds")

    print("\nPlotting …")
    plot_stability(store)

    print("\nComputing OLS summary statistics …")
    slope_df, final_df = compute_summary(store)

    save_csv(slope_df, "stability_slopes.csv")
    save_csv(final_df, "stability_final_sd.csv")

    make_latex_table(
        slope_df,
        caption=(r"Convergence rate: OLS slope of SD $\sim$ round (centred), "
                 r"per model, condition, and metric."),
        label="tab:stability_slope",
        fname="stability_slope.tex",
    )
    make_latex_table(
        final_df,
        caption=(rf"Stability: mean SD over final {LAST_N} rounds, "
                 r"per model, condition, and metric."),
        label="tab:stability_final_sd",
        fname="stability_final_sd.tex",
    )

    print("\nRunning mixed effects robustness check …")
    me_df = run_mixed_effects(store)

    save_csv(me_df, "stability_slopes_me.csv")

    make_latex_table_me(
        slope_df, me_df,
        fname="stability_slope_robustness.tex",
    )

    # Print a quick console summary comparing OLS vs ME slopes
    print("\n  OLS vs ME slope comparison (mean across conditions per model/metric):")
    import pandas as pd
    merged = slope_df.merge(
        me_df[["model", "condition", "metric", "fe_slope", "fe_pval", "converged"]],
        on=["model", "condition", "metric"], how="left",
    )
    for model in MODELS:
        for metric in METRICS:
            sub = merged[(merged["model"] == model) & (merged["metric"] == metric)]
            ols_mean = sub["mean"].mean()
            me_mean  = sub["fe_slope"].mean()
            print(f"    {model:8s} {metric:12s}  OLS={ols_mean:+.4f}  ME={me_mean:+.4f}")

    print("\nAll done.")
