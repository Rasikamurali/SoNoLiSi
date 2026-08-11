"""
analyze_prompt_sensitivity.py
--------------------------
Analyzes robustness of baseline and full-model behavior to system prompt framing.

Tests:
  1. OLS: contribution ~ variant (dummies, ref=standard) + round, seed-clustered SEs.
     For PURE_BASELINE: coefficients on minimal and tendency_free should be non-significant.
     For FULL: checks whether treatment effect persists across prompt framings.

  2. FULL - PURE_BASELINE gap per variant (last 5 rounds).
     All three gaps should be positive and significant.

  3. Cooperation tendency evolution: does CT diverge across variants in FULL?

Output: figures/robustness/prompt_sensitivity/
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

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS_DIR = "/data3/rasimura/social-norm-evo/results/robustness/prompt_sensitivity/openai"
OUT_DIR     = "/data3/rasimura/social-norm-evo/figures/robustness/prompt_sensitivity"
os.makedirs(OUT_DIR, exist_ok=True)

VARIANTS   = ["standard", "verb_only", "anchor_only", "verb_anchor"]
CONDITIONS = ["PURE_BASELINE", "FULL"]
SEEDS      = list(range(43, 48))
ENDOWMENT  = 10
LAST_N     = 5

VARIANT_COLORS  = {"standard": "#1f77b4", "verb_only": "#e377c2", "anchor_only": "#2ca02c", "verb_anchor": "#d62728"}
VARIANT_LABELS  = {"standard": "Standard", "verb_only": "Verb Only", "anchor_only": "Anchor Only", "verb_anchor": "Verb + Anchor"}
COND_LABELS     = {"PURE_BASELINE": "Pure Baseline", "FULL": "Full"}
REF_VARIANT     = "standard"

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(variant, seed, cond):
    pattern = os.path.join(RESULTS_DIR, variant, f"seed{seed}", f"log_*_{cond}_seed{seed}.json")
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        if d.get("condition") == cond:
            return d
    return None


def load_data():
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_log(variant, seed, cond)
                if d is None:
                    print(f"  MISSING: variant={variant} cond={cond} seed={seed}")
                    continue
                for rlog in d["round_logs"]:
                    r        = rlog["round"]
                    contribs = rlog["contributions"]
                    states   = rlog.get("agent_states", [])
                    rows.append({
                        "variant":           variant,
                        "condition":         cond,
                        "seed":              seed,
                        "round":             r,
                        "mean_contribution": np.mean(list(contribs.values())),
                        "mean_ct":           np.mean([s["cooperation_tendency"] for s in states]) if states else np.nan,
                    })
    return pd.DataFrame(rows)


def load_agent_level():
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_log(variant, seed, cond)
                if d is None:
                    continue
                for rlog in d["round_logs"]:
                    for aid_str, contrib in rlog["contributions"].items():
                        rows.append({
                            "variant":      variant,
                            "condition":    cond,
                            "seed":         seed,
                            "round":        rlog["round"],
                            "agent_id":     int(aid_str),
                            "contribution": contrib,
                        })
    return pd.DataFrame(rows)


# ─── Statistics ───────────────────────────────────────────────────────────────

def steady_state_summary(df):
    late = df[df["round"] > (20 - LAST_N)]
    rows = []
    for cond in CONDITIONS:
        for variant in VARIANTS:
            sub        = late[(late["condition"] == cond) & (late["variant"] == variant)]
            seed_means = sub.groupby("seed")["mean_contribution"].mean()
            rows.append({
                "condition": cond, "variant": variant,
                "mean": round(seed_means.mean(), 3),
                "se":   round(seed_means.sem(), 3),
                "sd":   round(seed_means.std(), 3),
            })
    return pd.DataFrame(rows)


def treatment_gap(ss):
    """FULL - PURE_BASELINE gap per variant."""
    rows = []
    pb   = ss[ss["condition"] == "PURE_BASELINE"].set_index("variant")
    fl   = ss[ss["condition"] == "FULL"].set_index("variant")
    for v in VARIANTS:
        gap    = fl.loc[v, "mean"] - pb.loc[v, "mean"]
        se_gap = np.sqrt(fl.loc[v, "se"]**2 + pb.loc[v, "se"]**2)
        z      = gap / se_gap if se_gap > 0 else np.nan
        rows.append({
            "variant": v,
            "pb_mean": pb.loc[v, "mean"],
            "full_mean": fl.loc[v, "mean"],
            "gap": round(gap, 3),
            "se_gap": round(se_gap, 3),
            "z": round(z, 2),
        })
    return pd.DataFrame(rows)


def ols_variant_effect(df_agent):
    """
    OLS: contribution ~ C(variant, Treatment('standard')) + round,
    seed-clustered SEs. Run per condition.
    """
    results = []
    for cond in CONDITIONS:
        sub = df_agent[df_agent["condition"] == cond].copy()
        try:
            mod = smf.ols(
                "contribution ~ C(variant, Treatment('standard')) + round", data=sub
            ).fit(cov_type="cluster", cov_kwds={"groups": sub["seed"]})
            for term in ["C(variant, Treatment('standard'))[T.verb_only]",
                         "C(variant, Treatment('standard'))[T.anchor_only]",
                         "C(variant, Treatment('standard'))[T.verb_anchor]"]:
                label = term.split("[T.")[-1].rstrip("]")
                coef  = mod.params.get(term, np.nan)
                se    = mod.bse.get(term, np.nan)
                pval  = mod.pvalues.get(term, np.nan)
                ci    = mod.conf_int().loc[term] if term in mod.conf_int().index else [np.nan, np.nan]
                results.append({
                    "condition": cond, "variant_vs_standard": label,
                    "coef": round(coef, 4), "se": round(se, 4),
                    "t": round(coef/se, 3) if se > 0 else np.nan,
                    "p": round(pval, 4),
                    "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4),
                    "sig": "*" if pval < 0.05 else ("†" if pval < 0.10 else ""),
                })
        except Exception as e:
            print(f"  OLS failed for {cond}: {e}")
    return pd.DataFrame(results)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_contributions_over_rounds(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df["condition"] == cond]
        for variant in VARIANTS:
            ts    = sub[sub["variant"] == variant].groupby("round")["mean_contribution"]
            means = ts.mean()
            sems  = ts.sem()
            ax.plot(means.index.to_numpy(), means.values,
                    color=VARIANT_COLORS[variant], linewidth=2,
                    label=VARIANT_LABELS[variant])
            ax.fill_between(means.index.to_numpy(),
                            means.values - sems.values,
                            means.values + sems.values,
                            color=VARIANT_COLORS[variant], alpha=0.15)
        ax.axhline(ENDOWMENT / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(COND_LABELS[cond], fontsize=13, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Mean Contribution" if ax == axes[0] else "", fontsize=11)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.tick_params(labelsize=10)
        ax.legend(fontsize=10)
    fig.suptitle("Prompt Variant Robustness: Mean Contribution Over Rounds", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pv_contribution_over_rounds.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "pv_contribution_over_rounds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pv_contribution_over_rounds")


def plot_gap_bars(ss):
    """
    Grouped bar chart: PURE_BASELINE and FULL side by side per variant.
    Highlights the FULL-PB gap across prompt framings.
    """
    x      = np.arange(len(VARIANTS))
    width  = 0.3
    fig, ax = plt.subplots(figsize=(8, 4.5))

    pb_vals = ss[ss["condition"] == "PURE_BASELINE"].set_index("variant")
    fl_vals = ss[ss["condition"] == "FULL"].set_index("variant")

    ax.bar(x - width/2,
           [pb_vals.loc[v, "mean"] for v in VARIANTS],
           width, yerr=[pb_vals.loc[v, "se"] for v in VARIANTS],
           capsize=4, label="Pure Baseline", color="#aec7e8", alpha=0.9)
    ax.bar(x + width/2,
           [fl_vals.loc[v, "mean"] for v in VARIANTS],
           width, yerr=[fl_vals.loc[v, "se"] for v in VARIANTS],
           capsize=4, label="Full", color="#1f77b4", alpha=0.9)

    # annotate gap
    for i, v in enumerate(VARIANTS):
        gap = fl_vals.loc[v, "mean"] - pb_vals.loc[v, "mean"]
        ax.annotate(f"Δ{gap:.2f}", xy=(x[i], fl_vals.loc[v, "mean"] + pb_vals.loc[v, "se"] + 0.3),
                    ha="center", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=11)
    ax.set_ylabel("Mean Contribution (last 5 rounds)", fontsize=11)
    ax.set_ylim(0, ENDOWMENT + 1.5)
    ax.axhline(ENDOWMENT / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.legend(fontsize=10)
    ax.set_title("Full vs Pure Baseline Gap by Prompt Variant", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pv_gap_bars.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "pv_gap_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pv_gap_bars")


def plot_ct_over_rounds(df):
    sub = df[df["condition"] == "FULL"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in VARIANTS:
        ts    = sub[sub["variant"] == variant].groupby("round")["mean_ct"]
        means = ts.mean()
        sems  = ts.sem()
        ax.plot(means.index.to_numpy(), means.values,
                color=VARIANT_COLORS[variant], linewidth=2, label=VARIANT_LABELS[variant])
        ax.fill_between(means.index.to_numpy(),
                        means.values - sems.values,
                        means.values + sems.values,
                        color=VARIANT_COLORS[variant], alpha=0.15)
    ax.set_title("Cooperation Tendency Over Rounds — Full Condition", fontsize=13)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean Cooperation Tendency", fontsize=11)
    ax.set_xlim(1, 20)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pv_ct_over_rounds.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "pv_ct_over_rounds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pv_ct_over_rounds")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df       = load_data()
    df_agent = load_agent_level()
    print(f"  {len(df)} run-round rows | {len(df_agent)} agent-round rows")

    print("\n── Steady-state summary (last 5 rounds) ─────────────")
    ss = steady_state_summary(df)
    print(ss.to_string(index=False))
    ss.to_csv(os.path.join(OUT_DIR, "pv_steady_state.csv"), index=False)

    print("\n── FULL - PURE_BASELINE gap per variant ──────────────")
    gap = treatment_gap(ss)
    print(gap.to_string(index=False))
    gap.to_csv(os.path.join(OUT_DIR, "pv_treatment_gap.csv"), index=False)

    print("\n── OLS: contribution ~ variant + round (ref=standard) ")
    ols = ols_variant_effect(df_agent)
    print(ols.to_string(index=False))
    ols.to_csv(os.path.join(OUT_DIR, "pv_ols_results.csv"), index=False)

    print("\n── Generating plots ──────────────────────────────────")
    plot_contributions_over_rounds(df)
    plot_gap_bars(ss)
    plot_ct_over_rounds(df)

    print(f"\nAll outputs → {OUT_DIR}")
