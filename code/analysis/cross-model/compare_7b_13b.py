"""
compare_7b_13b.py
-----------------
Direct comparison of 7B vs ~13B models within each family (local variant).

Pairs:
  Llama    : Meta-Llama-3.1-8B-Instruct  vs  Llama-2-13B-chat-hf
  Mistral  : Mistral-7B-Instruct-v0.3    vs  Mistral-Nemo-Instruct-2407 (12B)
  Qwen     : Qwen2.5-7B-Instruct         vs  Qwen2.5-14B-Instruct

Both variants use the local information structure, seeds 42–51 (10 seeds),
so comparisons are cleanly matched on information structure and seed.

Analyses:
  1. Steady-state table   — mean contribution + payoff (last 5 rounds)
  2. Overlay plots        — 7B (solid) vs 13B (dashed) per family × condition
  3. Paired t-tests       — seed-matched (same 10 seeds for both sizes)
  4. OLS per family       — contribution ~ C(condition) * is_large + round
                            is_large:condition interaction tests whether
                            scale changes the condition gradient
  5. FULL − PB gap comparison — does larger scale strengthen mechanism effect?

Output: figures/comparison_7b_vs_13b/
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

RESULTS = "/data3/rasimura/social-norm-evo/results"
OUT_DIR = "/data3/rasimura/social-norm-evo/figures/comparison_7b_vs_13b"
os.makedirs(OUT_DIR, exist_ok=True)

VARIANT = "local"
SEEDS   = list(range(42, 52))   # 10 seeds, same for both sizes

# Model pairs: (7B key, 13B key, family label)
MODEL_PAIRS = [
    ("llama",   "llama_13b",   "Llama"),
    ("mistral", "mistral_13b", "Mistral"),
    ("qwen",    "qwen_14b",    "Qwen"),
]
SIZE_LABELS  = {"small": "7B", "large": "13B"}
SIZE_DISPLAY = {
    "llama":       "Llama 3.1-8B",
    "llama_13b":   "Llama 2-13B",
    "mistral":     "Mistral-7B",
    "mistral_13b": "Mistral Nemo 12B",
    "qwen":        "Qwen2.5-7B",
    "qwen_14b":    "Qwen2.5-14B",
}
FAMILY_COLORS = {"Llama": "#ff7f0e", "Mistral": "#2ca02c", "Qwen": "#d62728"}

CONDITIONS = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
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
LAST_N = 5


# ─── Data loading ─────────────────────────────────────────────────────────────

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


def build_store():
    """store[model][cond][seed] = {contributions, payoffs}"""
    all_models = [m for pair in MODEL_PAIRS for m in (pair[0], pair[1])]
    store = {m: {c: {} for c in CONDITIONS} for m in all_models}
    for model in all_models:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_log(model, seed, cond)
                if d is None:
                    continue
                store[model][cond][seed] = {
                    "contributions": [np.mean(list(r["contributions"].values()))
                                      for r in d["round_logs"]],
                    "payoffs":       [np.mean(list(r["payoffs"].values()))
                                      for r in d["round_logs"]],
                }
    return store


def build_agent_df():
    rows = []
    all_models = [m for pair in MODEL_PAIRS for m in (pair[0], pair[1])]
    for model in all_models:
        family = next(f for s, l, f in MODEL_PAIRS if s == model or l == model)
        is_large = int(model.endswith("_13b") or model.endswith("_14b"))
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_log(model, seed, cond)
                if d is None:
                    continue
                for r in d["round_logs"]:
                    for aid_str, c in r["contributions"].items():
                        rows.append({
                            "model": model, "family": family,
                            "is_large": is_large,
                            "condition": cond, "seed": seed,
                            "round": r["round"],
                            "contribution": float(c),
                            "payoff": float(r["payoffs"].get(aid_str, np.nan)),
                        })
    return pd.DataFrame(rows)


# ─── 1. Steady-state table ────────────────────────────────────────────────────

def steady_state_table(store):
    rows = []
    for small, large, family in MODEL_PAIRS:
        for size_key, model in [("7B", small), ("13B", large)]:
            for cond in CONDITIONS:
                seed_means = [np.mean(store[model][cond][s]["contributions"][-LAST_N:])
                              for s in SEEDS if s in store[model][cond]]
                if not seed_means:
                    continue
                rows.append({
                    "family": family, "size": size_key, "model": model,
                    "condition": cond,
                    "mean": round(np.mean(seed_means), 3),
                    "se":   round(np.std(seed_means) / np.sqrt(len(seed_means)), 3),
                    "n":    len(seed_means),
                })
    return pd.DataFrame(rows)


def print_steady_state(ss):
    print("\n── Steady-state: mean contribution (last 5 rounds) ─────────────")
    print(f"  {'Family':8s} {'Cond':18s} {'7B':14s} {'13B':14s} {'Diff':8s}")
    print("  " + "-" * 65)
    for family in ["Llama", "Mistral", "Qwen"]:
        for cond in CONDITIONS:
            s7  = ss[(ss["family"] == family) & (ss["size"] == "7B")  & (ss["condition"] == cond)]
            s13 = ss[(ss["family"] == family) & (ss["size"] == "13B") & (ss["condition"] == cond)]
            if s7.empty or s13.empty:
                continue
            m7, m13 = s7.iloc[0]["mean"], s13.iloc[0]["mean"]
            e7, e13 = s7.iloc[0]["se"],   s13.iloc[0]["se"]
            diff    = m13 - m7
            print(f"  {family:8s} {cond:18s} "
                  f"{m7:.2f}±{e7:.2f}   {m13:.2f}±{e13:.2f}   "
                  f"{'▲' if diff > 0 else '▼'}{abs(diff):.2f}")


# ─── 2. Overlay plots ─────────────────────────────────────────────────────────

def plot_overlay(store):
    """One row per family, 5 cols (conditions). 7B solid, 13B dashed."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 11),
                             sharex=True, sharey=True)
    rounds = np.arange(1, 21)

    for row, (small, large, family) in enumerate(MODEL_PAIRS):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            for model, ls, lw, alpha, size_lbl in [
                (small, "-",  2.2, 1.0, "7B"),
                (large, "--", 1.8, 0.8, "13B"),
            ]:
                seed_data = store[model][cond]
                if not seed_data:
                    continue
                arr  = np.array([d["contributions"] for d in seed_data.values()], dtype=float)
                mean = np.nanmean(arr, axis=0)
                se   = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
                lbl  = f"{family} {size_lbl}" if col == 0 else None
                ax.plot(rounds, mean, color=COND_COLORS[cond],
                        linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)
                ax.fill_between(rounds, mean - se, mean + se,
                                color=COND_COLORS[cond], alpha=0.1)

            ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.4)
            ax.set_ylim(0, 10.5)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.2)

            if row == 0:
                ax.set_title(COND_LABELS[cond], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(family, fontsize=13, fontweight="bold",
                              color=FAMILY_COLORS[family])
            if row == 2:
                ax.set_xlabel("Round", fontsize=11)

    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=2.2, label="7B"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="13B"),
    ]
    fig.legend(style_handles, ["7B", "13B"], loc="lower center",
               ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("7B vs ~13B: Contribution Over Rounds (Local Variant)",
                 fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"overlay_7b_vs_13b.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: overlay_7b_vs_13b.png/.pdf")


# ─── 3. Paired t-tests (seed-matched) ────────────────────────────────────────

def paired_tests(store):
    print("\n── Paired t-tests: 13B − 7B (seed-matched, last 5 rounds) ──────")
    rows = []
    for small, large, family in MODEL_PAIRS:
        for cond in CONDITIONS:
            vals_s, vals_l = [], []
            for seed in SEEDS:
                s = store[small][cond].get(seed)
                l = store[large][cond].get(seed)
                if s is None or l is None:
                    continue
                vals_s.append(np.mean(s["contributions"][-LAST_N:]))
                vals_l.append(np.mean(l["contributions"][-LAST_N:]))
            if len(vals_s) < 3:
                continue
            s_arr, l_arr = np.array(vals_s), np.array(vals_l)
            diff = l_arr - s_arr
            t, p = stats.ttest_rel(l_arr, s_arr)
            rows.append({
                "family": family, "condition": cond,
                "mean_7b":  round(s_arr.mean(), 3),
                "mean_13b": round(l_arr.mean(), 3),
                "diff":     round(diff.mean(), 3),
                "se_diff":  round(diff.std() / np.sqrt(len(diff)), 3),
                "t": round(t, 3), "p": round(p, 4),
                "sig": "***" if p < 0.001 else ("**" if p < 0.01
                       else ("*" if p < 0.05 else "")),
                "n": len(vals_s),
            })

    result = pd.DataFrame(rows)
    print(f"\n  {'Family':8s} {'Condition':18s} {'7B':7s}  {'13B':7s}  {'Diff':8s} {'p':8s} sig")
    print("  " + "-" * 65)
    for _, row in result.iterrows():
        print(f"  {row['family']:8s} {row['condition']:18s} "
              f"{row['mean_7b']:6.3f}  {row['mean_13b']:6.3f}  "
              f"{row['diff']:+7.3f}  {row['p']:7.4f} {row['sig']}")
    result.to_csv(os.path.join(OUT_DIR, "paired_ttest_7b_vs_13b.csv"), index=False)
    return result


# ─── 4. OLS: condition × scale interaction ────────────────────────────────────

def ols_scale_interaction(df):
    """
    Per family: contribution ~ C(condition) * is_large + round
    Tests whether scale changes the condition gradient.
    """
    print("\n── OLS: contribution ~ condition * is_large + round ─────────────")
    ref_cond = "PURE_BASELINE"
    rows = []
    for _, _, family in MODEL_PAIRS:
        sub = df[df["family"] == family].copy()
        try:
            formula = f"contribution ~ C(condition, Treatment('{ref_cond}')) * is_large + round"
            res = smf.ols(formula, data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["seed"]}
            )
            for cond in ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE"]:
                term = f"C(condition, Treatment('{ref_cond}'))[T.{cond}]:is_large"
                if term not in res.params.index:
                    continue
                coef = res.params[term]
                se   = res.bse[term]
                pval = res.pvalues[term]
                rows.append({
                    "family": family, "condition": cond,
                    "coef": round(coef, 3), "se": round(se, 3),
                    "z":    round(coef / se, 2) if se > 0 else np.nan,
                    "p":    round(pval, 4),
                    "sig":  "***" if pval < 0.001 else ("**" if pval < 0.01
                            else ("*" if pval < 0.05 else "")),
                })
        except Exception as e:
            print(f"  [WARN] OLS failed for {family}: {e}")

    result = pd.DataFrame(rows)
    if not result.empty:
        print(f"\n  [Positive = 13B has larger condition effect than 7B]")
        for family in ["Llama", "Mistral", "Qwen"]:
            sub = result[result["family"] == family]
            if sub.empty:
                continue
            print(f"\n  {family}:")
            print(sub[["condition", "coef", "se", "z", "p", "sig"]].to_string(index=False))
    result.to_csv(os.path.join(OUT_DIR, "ols_scale_interaction.csv"), index=False)
    return result


# ─── 5. Gap comparison plot (FULL − PB per family per size) ──────────────────

def plot_gap_comparison(ss):
    """Bar chart: FULL − PB gap for 7B vs 13B, one group per family."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(MODEL_PAIRS))
    width  = 0.3

    for offset, size, color in [(-width/2, "7B", "#aec7e8"), (width/2, "13B", "#1f77b4")]:
        vals, errs = [], []
        for _, _, family in MODEL_PAIRS:
            pb   = ss[(ss["family"] == family) & (ss["size"] == size) & (ss["condition"] == "PURE_BASELINE")]
            full = ss[(ss["family"] == family) & (ss["size"] == size) & (ss["condition"] == "FULL")]
            if pb.empty or full.empty:
                vals.append(0); errs.append(0)
                continue
            gap    = full.iloc[0]["mean"] - pb.iloc[0]["mean"]
            se_gap = np.sqrt(full.iloc[0]["se"]**2 + pb.iloc[0]["se"]**2)
            vals.append(gap); errs.append(se_gap)
        ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
               color=color, alpha=0.9, label=size)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f[2] for f in MODEL_PAIRS], fontsize=13)
    ax.set_ylabel("FULL − Pure Baseline\n(mean contribution, last 5 rounds)", fontsize=12)
    ax.set_title("Treatment Effect (FULL − PB) by Model Size", fontsize=13)
    ax.legend(fontsize=12, title="Size")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"gap_comparison.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: gap_comparison.png/.pdf")


def plot_condition_bars(ss):
    """Grouped bar chart per condition, 7B vs 13B, one panel per family."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    x     = np.arange(len(CONDITIONS))
    width = 0.35

    for ax, (small, large, family) in zip(axes, MODEL_PAIRS):
        color = FAMILY_COLORS[family]
        for offset, size, alpha in [(-width/2, "7B", 0.6), (width/2, "13B", 0.95)]:
            sub = ss[(ss["family"] == family) & (ss["size"] == size)]
            sub = sub.set_index("condition").reindex(CONDITIONS)
            ax.bar(x + offset, sub["mean"].fillna(0), width,
                   yerr=sub["se"].fillna(0), capsize=3,
                   color=color, alpha=alpha, label=f"{family} {size}")
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c].replace(" ", "\n") for c in CONDITIONS],
                           fontsize=9)
        ax.set_title(family, fontsize=13, fontweight="bold", color=color)
        ax.set_ylim(0, 11)
        ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.4)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend(fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Mean Contribution (last 5 rounds)", fontsize=11)

    fig.suptitle("7B vs ~13B: Steady-State Contribution by Condition", fontsize=13, y=1.01)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"condition_bars_7b_vs_13b.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: condition_bars_7b_vs_13b.png/.pdf")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data …")
    store = build_store()
    df    = build_agent_df()
    print(f"  {len(df):,} observations")

    # Check coverage
    for small, large, family in MODEL_PAIRS:
        n_s = len(store[small]["FULL"])
        n_l = len(store[large]["FULL"])
        print(f"  {family}: {small} {n_s} seeds, {large} {n_l} seeds")

    print("\n" + "=" * 60)
    print("1. STEADY-STATE TABLE")
    print("=" * 60)
    ss = steady_state_table(store)
    ss.to_csv(os.path.join(OUT_DIR, "steady_state_7b_vs_13b.csv"), index=False)
    print_steady_state(ss)

    print("\n" + "=" * 60)
    print("2. OVERLAY PLOTS")
    print("=" * 60)
    plot_overlay(store)

    print("\n" + "=" * 60)
    print("3. PAIRED T-TESTS")
    print("=" * 60)
    paired_tests(store)

    print("\n" + "=" * 60)
    print("4. OLS: SCALE × CONDITION INTERACTION")
    print("=" * 60)
    ols_scale_interaction(df)

    print("\n" + "=" * 60)
    print("5. GAP AND CONDITION PLOTS")
    print("=" * 60)
    plot_gap_comparison(ss)
    plot_condition_bars(ss)

    print(f"\nAll outputs → {OUT_DIR}")
