"""
compare_global_local.py
-----------------------
Direct comparison of global vs local simulation variants.

The only structural difference between global and local:
  global : Discussion (all 12 agents) → Group formation → Contributions
  local  : Group formation → Discussion (per group) → Contributions

Analyses:
  1. Steady-state table — mean contribution + payoff (last 5 rounds),
     global vs local per (model × condition).

  2. Overlay plots — global (solid) vs local (dashed) on the same axes,
     one panel per model, separate figures for contribution and payoff.

  3. OLS: contribution ~ C(condition) * variant + round
     The variant:condition interaction tests whether the condition gradient
     differs between global and local.

  4. Paired t-test per (model × condition) — seed-matched since both
     variants use seeds 43–52.

Output: figures/comparison_global_local/
"""

import json
import glob
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/results"
OUT_DIR  = "/data3/rasimura/social-norm-evo/figures/comparison_global_local"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS     = ["gpt", "llama", "mistral", "qwen"]
MODEL_LABELS = {"gpt": "GPT-4o-mini", "llama": "Llama 3.1-8B",
                "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}
VARIANTS   = ["global", "local"]
SEEDS      = list(range(43, 53))
CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
LAST_N     = 5

COND_COLORS = {
    "PURE_BASELINE":  "#aaaaaa",
    "BASELINE":       "#6baed6",
    "NO_SELECTION":   "#fd8d3c",
    "NO_DISCUSSION":  "#74c476",
    "FULL":           "#e6550d",
}
COND_LABELS = {
    "PURE_BASELINE":  "Pure Baseline",
    "BASELINE":       "Baseline",
    "NO_SELECTION":   "No Selection",
    "NO_DISCUSSION":  "No Discussion",
    "FULL":           "Full",
}
VARIANT_STYLE = {"global": "-", "local": "--"}
VARIANT_ALPHA = {"global": 1.0, "local": 0.75}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(model, variant, seed, condition):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


def build_store():
    """store[model][variant][cond][seed] = {contributions: [], payoffs: []}"""
    store = {m: {v: {c: {} for c in CONDITIONS} for v in VARIANTS} for m in MODELS}
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for cond in CONDITIONS:
                    d = load_log(model, variant, seed, cond)
                    if d is None:
                        continue
                    contribs = []
                    payoffs  = []
                    for r in d["round_logs"]:
                        cv = list(r["contributions"].values())
                        pv = list(r["payoffs"].values())
                        contribs.append(np.mean(cv) if cv else np.nan)
                        payoffs.append(np.mean(pv) if pv else np.nan)
                    store[model][variant][cond][seed] = {
                        "contributions": contribs,
                        "payoffs":       payoffs,
                    }
    return store


def build_agent_df():
    """Long-format: one row per (model, variant, condition, seed, round, agent)."""
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for cond in CONDITIONS:
                    d = load_log(model, variant, seed, cond)
                    if d is None:
                        continue
                    for r in d["round_logs"]:
                        rnd = r["round"]
                        for aid_str, c in r["contributions"].items():
                            rows.append({
                                "model": model, "variant": variant,
                                "condition": cond, "seed": seed,
                                "round": rnd, "contribution": float(c),
                                "payoff": float(r["payoffs"].get(aid_str, np.nan)),
                            })
    df = pd.DataFrame(rows)
    df["is_local"] = (df["variant"] == "local").astype(int)
    return df


# ─── 1. Steady-state table ────────────────────────────────────────────────────

def steady_state_table(store):
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            for cond in CONDITIONS:
                seed_means_c, seed_means_p = [], []
                for seed, data in store[model][variant][cond].items():
                    c_arr = np.array(data["contributions"])
                    p_arr = np.array(data["payoffs"])
                    seed_means_c.append(np.nanmean(c_arr[-LAST_N:]))
                    seed_means_p.append(np.nanmean(p_arr[-LAST_N:]))
                if not seed_means_c:
                    continue
                rows.append({
                    "model": model, "variant": variant, "condition": cond,
                    "n_seeds":   len(seed_means_c),
                    "contrib_mean": round(np.mean(seed_means_c), 3),
                    "contrib_se":   round(np.std(seed_means_c) / np.sqrt(len(seed_means_c)), 3),
                    "payoff_mean":  round(np.mean(seed_means_p), 3),
                    "payoff_se":    round(np.std(seed_means_p) / np.sqrt(len(seed_means_p)), 3),
                })
    return pd.DataFrame(rows)


def print_comparison_table(ss):
    print("\n── Steady-state comparison (last 5 rounds) ──────────────────")
    print(f"  {'Model':12s} {'Condition':18s} {'Global contrib':15s} {'Local contrib':15s} {'Diff':8s}")
    print("  " + "-" * 72)
    for model in MODELS:
        for cond in CONDITIONS:
            g = ss[(ss["model"] == model) & (ss["variant"] == "global") & (ss["condition"] == cond)]
            l = ss[(ss["model"] == model) & (ss["variant"] == "local")  & (ss["condition"] == cond)]
            if g.empty or l.empty:
                continue
            gm = g.iloc[0]["contrib_mean"]
            lm = l.iloc[0]["contrib_mean"]
            gs = g.iloc[0]["contrib_se"]
            ls_v = l.iloc[0]["contrib_se"]
            diff = lm - gm
            print(f"  {MODEL_LABELS[model]:12s} {cond:18s} "
                  f"{gm:.2f} ± {gs:.2f}   {lm:.2f} ± {ls_v:.2f}   "
                  f"{'▲' if diff > 0 else '▼'}{abs(diff):.2f}")


# ─── 2. Overlay plots ─────────────────────────────────────────────────────────

def plot_overlay(store, metric, ylabel, ylim, fname):
    """Global (solid) vs local (dashed) on same axes, one panel per model."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 5),
                             sharex=True, sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for cond in CONDITIONS:
            for variant in VARIANTS:
                seed_data = store[model][variant][cond]
                if not seed_data:
                    continue
                arr    = np.array([d[metric] for d in seed_data.values()], dtype=float)
                rounds = np.arange(1, arr.shape[1] + 1)
                mean   = np.nanmean(arr, axis=0)
                se     = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
                lbl    = f"{COND_LABELS[cond]} ({variant})" if idx == 0 else None
                ax.plot(rounds, mean,
                        color=COND_COLORS[cond],
                        linestyle=VARIANT_STYLE[variant],
                        linewidth=1.8 if variant == "global" else 1.5,
                        alpha=VARIANT_ALPHA[variant],
                        label=lbl)
                ax.fill_between(rounds, mean - se, mean + se,
                                color=COND_COLORS[cond], alpha=0.08)

        if ylim[0] is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=12)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Round", fontsize=13)
        if idx == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        ax.text(0.97, 0.97, MODEL_LABELS[model],
                transform=ax.transAxes, fontsize=13, ha="right", va="top")

    # Legend: one entry per condition + line style indicator
    handles, labels = axes[0].get_legend_handles_labels()
    # Add style legend manually
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2, label="Global"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Local"),
    ]
    fig.legend(handles + style_handles,
               labels + ["Global", "Local"],
               loc="lower center", ncol=len(CONDITIONS) + 2,
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, fname.replace(".png", f".{ext}"))
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# ─── 3. OLS: variant × condition interaction ──────────────────────────────────

def ols_variant_interaction(df):
    """
    OLS per model: contribution ~ C(condition) * is_local + round
    Clustered by seed.
    is_local = 1 for local, 0 for global.
    The is_local:condition interaction tests whether the condition gradient
    differs between variants — i.e., whether the information structure changes
    how much norms emerge.
    """
    print("\n── OLS: contribution ~ condition * is_local + round ─────────")
    ref_cond = "PURE_BASELINE"
    rows = []
    for model in MODELS:
        sub = df[df["model"] == model].copy()
        try:
            formula = f"contribution ~ C(condition, Treatment('{ref_cond}')) * is_local + round"
            res = smf.ols(formula, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["seed"]}
            )
            # Extract is_local main effect and key interactions
            for term, label in [
                ("is_local", "Local main effect"),
                (f"C(condition, Treatment('{ref_cond}'))[T.FULL]:is_local",
                 "FULL × local"),
                (f"C(condition, Treatment('{ref_cond}'))[T.NO_DISCUSSION]:is_local",
                 "No Discussion × local"),
                (f"C(condition, Treatment('{ref_cond}'))[T.NO_SELECTION]:is_local",
                 "No Selection × local"),
                (f"C(condition, Treatment('{ref_cond}'))[T.BASELINE]:is_local",
                 "Baseline × local"),
            ]:
                if term not in res.params.index:
                    continue
                coef = res.params[term]
                se   = res.bse[term]
                pval = res.pvalues[term]
                rows.append({
                    "model": model, "term": label,
                    "coef": round(coef, 3), "se": round(se, 3),
                    "z":    round(coef / se, 2) if se > 0 else np.nan,
                    "p":    round(pval, 4),
                    "sig":  "***" if pval < 0.001 else ("**" if pval < 0.01
                            else ("*" if pval < 0.05 else "")),
                })
        except Exception as e:
            print(f"  [WARN] OLS failed for {model}: {e}")

    result = pd.DataFrame(rows)
    if not result.empty:
        for model in MODELS:
            print(f"\n  [{MODEL_LABELS[model]}]")
            sub = result[result["model"] == model][["term", "coef", "se", "z", "p", "sig"]]
            print(sub.to_string(index=False))
    result.to_csv(os.path.join(OUT_DIR, "ols_variant_interaction.csv"), index=False)
    return result


# ─── 4. Paired t-tests (seed-matched) ────────────────────────────────────────

def paired_tests(store):
    """
    For each (model, condition): paired t-test of mean contribution,
    global vs local, matched on seed.
    """
    print("\n── Paired t-test: global vs local (seed-matched, last 5 rounds) ──")
    rows = []
    for model in MODELS:
        for cond in CONDITIONS:
            g_vals, l_vals = [], []
            for seed in SEEDS:
                g = store[model]["global"][cond].get(seed)
                l = store[model]["local"][cond].get(seed)
                if g is None or l is None:
                    continue
                g_vals.append(np.nanmean(g["contributions"][-LAST_N:]))
                l_vals.append(np.nanmean(l["contributions"][-LAST_N:]))
            if len(g_vals) < 3:
                continue
            g_arr, l_arr = np.array(g_vals), np.array(l_vals)
            diff   = l_arr - g_arr
            t, p   = stats.ttest_rel(l_arr, g_arr)
            rows.append({
                "model": model, "condition": cond,
                "global_mean": round(g_arr.mean(), 3),
                "local_mean":  round(l_arr.mean(), 3),
                "diff_mean":   round(diff.mean(), 3),
                "diff_se":     round(diff.std() / np.sqrt(len(diff)), 3),
                "t":    round(t, 3),
                "p":    round(p, 4),
                "sig":  "***" if p < 0.001 else ("**" if p < 0.01
                        else ("*" if p < 0.05 else "")),
                "n_seeds": len(g_vals),
            })

    result = pd.DataFrame(rows)
    print(f"\n  {'Model':12s} {'Condition':18s} {'Global':8s} {'Local':8s} {'Diff':8s} {'p':8s} sig")
    print("  " + "-" * 72)
    for _, row in result.iterrows():
        print(f"  {MODEL_LABELS[row['model']]:12s} {row['condition']:18s} "
              f"{row['global_mean']:7.3f}  {row['local_mean']:7.3f}  "
              f"{row['diff_mean']:+7.3f}  {row['p']:7.4f} {row['sig']}")
    result.to_csv(os.path.join(OUT_DIR, "paired_ttest_global_local.csv"), index=False)
    return result


# ─── 5. Difference-over-rounds plot ──────────────────────────────────────────

def plot_difference_over_rounds(store):
    """
    Local − Global mean contribution per round, by condition, one panel per model.
    Zero line = no difference. Positive = local is more cooperative.
    """
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 4.5), sharey=True)

    for ax, model in zip(axes, MODELS):
        for cond in CONDITIONS:
            g_dict = store[model]["global"][cond]
            l_dict = store[model]["local"][cond]
            shared = sorted(set(g_dict) & set(l_dict))
            if len(shared) < 2:
                continue
            g_arr  = np.array([g_dict[s]["contributions"] for s in shared], dtype=float)
            l_arr  = np.array([l_dict[s]["contributions"] for s in shared], dtype=float)
            diff   = l_arr - g_arr
            rounds = np.arange(1, diff.shape[1] + 1)
            mean   = np.nanmean(diff, axis=0)
            se     = np.nanstd(diff, axis=0) / np.sqrt(diff.shape[0])
            ax.plot(rounds, mean, color=COND_COLORS[cond],
                    linewidth=2, label=COND_LABELS[cond])
            ax.fill_between(rounds, mean - se, mean + se,
                            color=COND_COLORS[cond], alpha=0.15)

        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel("Local − Global\n(mean contribution)", fontsize=11)
            ax.legend(fontsize=9, loc="lower left")

    fig.suptitle("Local vs Global: Per-Round Contribution Difference",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"difference_over_rounds.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: difference_over_rounds.png/.pdf")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data …")
    store = build_store()
    df    = build_agent_df()
    print(f"  {len(df):,} agent-round observations")

    print("\n" + "=" * 60)
    print("1. STEADY-STATE TABLE")
    print("=" * 60)
    ss = steady_state_table(store)
    ss.to_csv(os.path.join(OUT_DIR, "steady_state_comparison.csv"), index=False)
    print_comparison_table(ss)

    print("\n" + "=" * 60)
    print("2. OVERLAY PLOTS")
    print("=" * 60)
    plot_overlay(store, "contributions", "Mean Contribution (0–10)",
                 (0, 10.5), "overlay_contribution.png")
    plot_overlay(store, "payoffs", "Mean Payoff",
                 (None, None), "overlay_payoff.png")

    print("\n" + "=" * 60)
    print("3. OLS: VARIANT × CONDITION INTERACTION")
    print("=" * 60)
    ols_variant_interaction(df)

    print("\n" + "=" * 60)
    print("4. PAIRED T-TESTS (seed-matched)")
    print("=" * 60)
    paired_tests(store)

    print("\n" + "=" * 60)
    print("5. DIFFERENCE OVER ROUNDS")
    print("=" * 60)
    plot_difference_over_rounds(store)

    print(f"\nAll outputs → {OUT_DIR}")
