"""
analyze_mechanism_vs_pretraining.py
-------------------------------------
Tests whether cooperation emergence in FULL is mechanism-driven or
a pre-training artifact, using the prompt_sensitivity results.

Four tests:
  a) Round-1 gap: FULL − PURE_BASELINE at round 1 only.
     If the gap is already large at round 1, pre-training is a plausible driver.
     If it is near zero, the divergence is something that unfolds over time.

  b) Mixed-effects model: contribution ~ round * condition + (1|seed).
     The round:condition interaction tests whether FULL has a steeper
     slope across rounds than PURE_BASELINE.
     Run per variant so we can see if the interaction is consistent.

  c) Stabilization round: first round where the 3-round rolling change
     in mean contribution drops below a threshold — separately for
     FULL and PURE_BASELINE per variant.

  d) Per-round difference: FULL_mean(t) − PB_mean(t) for each round,
     with SE ribbon. Reveals whether the gap opens early, late, or linearly.
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
STAB_THRESH = 0.3    # max round-over-round change to call "stable"
STAB_WINDOW = 3      # consecutive rounds below threshold

VARIANT_COLORS = {"standard": "#1f77b4", "verb_only": "#e377c2",
                  "anchor_only": "#2ca02c", "verb_anchor": "#d62728"}
VARIANT_LABELS = {"standard": "Standard", "verb_only": "Verb Only",
                  "anchor_only": "Anchor Only", "verb_anchor": "Verb + Anchor"}

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(variant, seed, cond):
    pattern = os.path.join(RESULTS_DIR, variant, f"seed{seed}", f"log_*_{cond}_seed{seed}.json")
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        if d.get("condition") == cond:
            return d
    return None


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
                            "contribution": int(contrib),
                        })
    return pd.DataFrame(rows)


def load_round_means():
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for cond in CONDITIONS:
                d = load_log(variant, seed, cond)
                if d is None:
                    continue
                for rlog in d["round_logs"]:
                    contribs = list(rlog["contributions"].values())
                    rows.append({
                        "variant":   variant,
                        "condition": cond,
                        "seed":      seed,
                        "round":     rlog["round"],
                        "mean_contribution": np.mean(contribs),
                    })
    return pd.DataFrame(rows)


# ─── a) Round-1 gap ──────────────────────────────────────────────────────────

def round1_gap(df_agent):
    r1  = df_agent[df_agent["round"] == 1]
    rows = []
    for variant in VARIANTS:
        for cond in CONDITIONS:
            sub        = r1[(r1["variant"] == variant) & (r1["condition"] == cond)]
            seed_means = sub.groupby("seed")["contribution"].mean()
            rows.append({
                "variant": variant, "condition": cond,
                "mean": round(seed_means.mean(), 3),
                "se":   round(seed_means.sem(), 3),
            })
    result = pd.DataFrame(rows)

    gaps = []
    for variant in VARIANTS:
        pb   = result[(result["variant"] == variant) & (result["condition"] == "PURE_BASELINE")].iloc[0]
        fl   = result[(result["variant"] == variant) & (result["condition"] == "FULL")].iloc[0]
        gap  = fl["mean"] - pb["mean"]
        se_g = np.sqrt(fl["se"]**2 + pb["se"]**2)
        gaps.append({
            "variant": variant,
            "pb_r1": pb["mean"], "full_r1": fl["mean"],
            "gap_r1": round(gap, 3), "se": round(se_g, 3),
            "z": round(gap / se_g, 2) if se_g > 0 else np.nan,
        })
    return pd.DataFrame(gaps)


# ─── b) Mixed-effects model ──────────────────────────────────────────────────

def mixed_effects_slopes(df_agent):
    """
    Per variant: contribution ~ round * is_full + (1|seed)
    is_full = 1 if FULL, 0 if PURE_BASELINE.
    Key coefficient: round:is_full — positive = FULL slope steeper.
    """
    df = df_agent.copy()
    df["is_full"] = (df["condition"] == "FULL").astype(int)

    rows = []
    for variant in VARIANTS:
        sub = df[df["variant"] == variant].copy()
        try:
            mod = smf.mixedlm(
                "contribution ~ round * is_full",
                data=sub,
                groups=sub["seed"],
            ).fit(reml=True)

            for term, label in [
                ("round",         "slope_PB (per round)"),
                ("is_full",       "FULL intercept offset"),
                ("round:is_full", "FULL slope increment"),
            ]:
                coef = mod.params.get(term, np.nan)
                se   = mod.bse.get(term, np.nan)
                pval = mod.pvalues.get(term, np.nan)
                rows.append({
                    "variant": variant, "term": label,
                    "coef": round(coef, 4), "se": round(se, 4),
                    "z":    round(coef / se, 3) if se > 0 else np.nan,
                    "p":    round(pval, 4),
                    "sig":  "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ("†" if pval < 0.10 else ""))),
                })
        except Exception as e:
            print(f"  MixedLM failed for {variant}: {e}")

    return pd.DataFrame(rows)


# ─── c) Stabilization round ──────────────────────────────────────────────────

def stabilization_round(df_round):
    rows = []
    for variant in VARIANTS:
        for cond in CONDITIONS:
            sub    = df_round[(df_round["variant"] == variant) & (df_round["condition"] == cond)]
            means  = sub.groupby("round")["mean_contribution"].mean().sort_index()
            deltas = means.diff().abs().dropna()

            stab_round = None
            for r in deltas.index:
                window = deltas.loc[r: r + STAB_WINDOW - 1]
                if len(window) == STAB_WINDOW and (window < STAB_THRESH).all():
                    stab_round = int(r)
                    break

            rows.append({
                "variant": variant, "condition": cond,
                "stabilization_round": stab_round if stab_round else ">20",
                "final_mean": round(means.iloc[-1], 3),
            })
    return pd.DataFrame(rows)


# ─── d) Per-round difference plot ────────────────────────────────────────────

def plot_per_round_difference(df_round):
    """
    FULL_mean(t) − PB_mean(t) per round, one line per variant.
    If mechanisms drive divergence, the gap should widen over rounds.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    for variant in VARIANTS:
        sub_full = df_round[(df_round["variant"] == variant) & (df_round["condition"] == "FULL")]
        sub_pb   = df_round[(df_round["variant"] == variant) & (df_round["condition"] == "PURE_BASELINE")]

        full_by_round = sub_full.groupby(["round", "seed"])["mean_contribution"].mean().reset_index()
        pb_by_round   = sub_pb.groupby(["round", "seed"])["mean_contribution"].mean().reset_index()
        merged = full_by_round.merge(pb_by_round, on=["round", "seed"], suffixes=("_full", "_pb"))
        merged["diff"] = merged["mean_contribution_full"] - merged["mean_contribution_pb"]

        by_round = merged.groupby("round")["diff"]
        means    = by_round.mean()
        sems     = by_round.sem()
        rounds   = means.index.to_numpy()

        ax.plot(rounds, means.values, color=VARIANT_COLORS[variant],
                linewidth=2, label=VARIANT_LABELS[variant])
        ax.fill_between(rounds, means.values - sems.values,
                        means.values + sems.values,
                        color=VARIANT_COLORS[variant], alpha=0.15)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("FULL − Pure Baseline\n(mean contribution)", fontsize=11)
    ax.set_title("Divergence of FULL from Pure Baseline Over Rounds", fontsize=13)
    ax.set_xlim(1, 20)
    ax.legend(fontsize=10, loc="upper left")
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mech_per_round_diff.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "mech_per_round_diff.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: mech_per_round_diff")


def plot_trajectories(df_round):
    """
    Contribution over rounds for FULL and PURE_BASELINE side by side,
    one panel per variant.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax, variant in zip(axes, VARIANTS):
        for cond, color, ls in [("PURE_BASELINE", "#999999", "--"), ("FULL", VARIANT_COLORS[variant], "-")]:
            sub   = df_round[(df_round["variant"] == variant) & (df_round["condition"] == cond)]
            means = sub.groupby("round")["mean_contribution"].mean()
            sems  = sub.groupby("round")["mean_contribution"].sem()
            ax.plot(means.index.to_numpy(), means.values, color=color,
                    linewidth=2, linestyle=ls,
                    label="Full" if cond == "FULL" else "Pure Baseline")
            ax.fill_between(means.index.to_numpy(),
                            means.values - sems.values,
                            means.values + sems.values,
                            color=color, alpha=0.15)
        ax.axhline(ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(VARIANT_LABELS[variant], fontsize=11, fontweight="bold")
        ax.set_xlabel("Round", fontsize=10)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.tick_params(labelsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Mean Contribution", fontsize=10)
            ax.legend(fontsize=9)

    fig.suptitle("FULL vs Pure Baseline Trajectories by Prompt Variant", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mech_trajectories.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "mech_trajectories.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: mech_trajectories")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df_agent = load_agent_level()
    df_round = load_round_means()
    print(f"  {len(df_agent)} agent-round rows")

    print("\n── a) Round-1 gap: FULL − Pure Baseline ─────────────")
    r1 = round1_gap(df_agent)
    print(r1.to_string(index=False))
    r1.to_csv(os.path.join(OUT_DIR, "mech_round1_gap.csv"), index=False)

    print("\n── b) Mixed-effects slopes: round × condition ────────")
    me = mixed_effects_slopes(df_agent)
    # Print pivoted for readability
    for variant in VARIANTS:
        print(f"\n  [{VARIANT_LABELS[variant]}]")
        sub = me[me["variant"] == variant][["term", "coef", "se", "z", "p", "sig"]]
        print(sub.to_string(index=False))
    me.to_csv(os.path.join(OUT_DIR, "mech_mixed_effects.csv"), index=False)

    print("\n── c) Stabilization round ────────────────────────────")
    stab = stabilization_round(df_round)
    print(stab.to_string(index=False))
    stab.to_csv(os.path.join(OUT_DIR, "mech_stabilization.csv"), index=False)

    print("\n── d) Generating per-round plots ─────────────────────")
    plot_per_round_difference(df_round)
    plot_trajectories(df_round)

    print(f"\nAll outputs → {OUT_DIR}")
