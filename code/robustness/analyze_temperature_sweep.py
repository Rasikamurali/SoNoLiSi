"""
analyze_temperature_sweep.py
-----------------------------
Analyzes robustness of baseline conditions to decision temperature.

Tests:
  1. OLS: contribution ~ temperature (continuous) for PURE_BASELINE and BASELINE,
     with round as covariate and seed-clustered SEs.
     Null: slope on temperature is zero — baseline is temperature-stable.

  2. Cooperation tendency drift: does CT change across rounds at different temps?
     (BASELINE only — PURE_BASELINE has no CT-updating mechanism)

  3. Plot: mean contribution over rounds, lines by temperature,
     panels by condition. Tight bunching = robust baseline.

Output: figures/robustness/temperature_sweep/
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

RESULTS_DIR = "/data3/rasimura/social-norm-evo/results/robustness/temperature_sweep/openai"
OUT_DIR     = "/data3/rasimura/social-norm-evo/figures/robustness/temperature_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

TEMPERATURES = [0.3, 0.5, 0.7, 1.0]
CONDITIONS   = ["PURE_BASELINE", "BASELINE"]
SEEDS        = list(range(43, 48))
ENDOWMENT    = 10
LAST_N       = 5

TEMP_COLORS = {0.3: "#1a6faf", 0.5: "#3ca0d0", 0.7: "#f4a942", 1.0: "#d94f3d"}
COND_LABELS = {"PURE_BASELINE": "Pure Baseline", "BASELINE": "Baseline"}

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    rows = []
    for temp in TEMPERATURES:
        temp_label = f"temp_{temp}"
        for seed in SEEDS:
            for cond in CONDITIONS:
                pattern = os.path.join(
                    RESULTS_DIR, temp_label, f"seed{seed}", f"log_*_{cond}_seed{seed}.json"
                )
                files = sorted(glob.glob(pattern))
                # validate condition field — glob "*_BASELINE_*" also matches PURE_BASELINE filenames
                d = None
                for p in files:
                    with open(p) as f:
                        candidate = json.load(f)
                    if candidate.get("condition") == cond:
                        d = candidate
                        break
                if d is None:
                    print(f"  MISSING: temp={temp} cond={cond} seed={seed}")
                    continue
                for rlog in d["round_logs"]:
                    r = rlog["round"]
                    contribs = rlog["contributions"]
                    mean_contrib = np.mean(list(contribs.values()))
                    # mean cooperation tendency this round
                    states   = rlog.get("agent_states", [])
                    mean_ct  = np.mean([s["cooperation_tendency"] for s in states]) if states else np.nan
                    rows.append({
                        "temperature": temp,
                        "condition":   cond,
                        "seed":        seed,
                        "round":       r,
                        "mean_contribution": mean_contrib,
                        "mean_ct":     mean_ct,
                        "n_agents":    len(contribs),
                    })
    return pd.DataFrame(rows)


def load_agent_level() -> pd.DataFrame:
    """One row per agent per round — for clustered SE regression."""
    rows = []
    for temp in TEMPERATURES:
        temp_label = f"temp_{temp}"
        for seed in SEEDS:
            for cond in CONDITIONS:
                pattern = os.path.join(
                    RESULTS_DIR, temp_label, f"seed{seed}", f"log_*_{cond}_seed{seed}.json"
                )
                files = sorted(glob.glob(pattern))
                d = None
                for p in files:
                    with open(p) as f:
                        candidate = json.load(f)
                    if candidate.get("condition") == cond:
                        d = candidate
                        break
                if d is None:
                    continue
                for rlog in d["round_logs"]:
                    r = rlog["round"]
                    for aid_str, contrib in rlog["contributions"].items():
                        rows.append({
                            "temperature": temp,
                            "condition":   cond,
                            "seed":        seed,
                            "round":       r,
                            "agent_id":    int(aid_str),
                            "contribution": contrib,
                        })
    return pd.DataFrame(rows)


# ─── Statistics ───────────────────────────────────────────────────────────────

def ols_temp_effect(df_agent: pd.DataFrame) -> pd.DataFrame:
    """
    OLS: contribution ~ temperature + round, clustered by seed.
    Run separately for each condition.
    H0: coefficient on temperature = 0 (baseline is temp-stable).
    """
    results = []
    for cond in CONDITIONS:
        sub = df_agent[df_agent["condition"] == cond].copy()
        try:
            mod = smf.ols("contribution ~ temperature + round", data=sub).fit(
                cov_type="cluster", cov_kwds={"groups": sub["seed"]}
            )
            coef  = mod.params["temperature"]
            se    = mod.bse["temperature"]
            tstat = mod.tvalues["temperature"]
            pval  = mod.pvalues["temperature"]
            ci_lo, ci_hi = mod.conf_int().loc["temperature"]
            results.append({
                "condition": cond,
                "coef_temp": round(coef, 4),
                "se":        round(se, 4),
                "t":         round(tstat, 3),
                "p":         round(pval, 4),
                "ci_lo":     round(ci_lo, 4),
                "ci_hi":     round(ci_hi, 4),
                "sig":       "*" if pval < 0.05 else ("†" if pval < 0.10 else ""),
            })
        except Exception as e:
            print(f"  OLS failed for {cond}: {e}")
    return pd.DataFrame(results)


def steady_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SE of contribution in last LAST_N rounds, by (temperature, condition)."""
    late = df[df["round"] > (20 - LAST_N)]
    rows = []
    for cond in CONDITIONS:
        for temp in TEMPERATURES:
            sub = late[(late["condition"] == cond) & (late["temperature"] == temp)]
            # seed-level means first, then average across seeds
            seed_means = sub.groupby("seed")["mean_contribution"].mean()
            rows.append({
                "condition":   cond,
                "temperature": temp,
                "mean":  round(seed_means.mean(), 3),
                "se":    round(seed_means.sem(), 3),
                "sd":    round(seed_means.std(), 3),
                "n_seeds": len(seed_means),
            })
    return pd.DataFrame(rows)


def cv_by_temp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coefficient of variation across seeds for each (temperature, condition, round).
    Lower CV across temperatures = more stable baseline.
    """
    rows = []
    for cond in CONDITIONS:
        for r in sorted(df["round"].unique()):
            sub   = df[(df["condition"] == cond) & (df["round"] == r)]
            means = sub.groupby("temperature")["mean_contribution"].mean()
            cv    = means.std() / means.mean() if means.mean() > 0 else np.nan
            rows.append({"condition": cond, "round": r, "cv_across_temps": round(cv, 4)})
    return pd.DataFrame(rows)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_contributions_over_rounds(df: pd.DataFrame):
    """
    Main figure: mean contribution over rounds, one line per temperature,
    two panels (PURE_BASELINE | BASELINE).
    Tight bunching of lines = temperature-robust baseline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, cond in zip(axes, CONDITIONS):
        sub = df[df["condition"] == cond]
        for temp in TEMPERATURES:
            ts = sub[sub["temperature"] == temp].groupby("round")["mean_contribution"]
            means = ts.mean()
            sems  = ts.sem()
            rounds = means.index
            ax.plot(rounds.to_numpy(), means.values, color=TEMP_COLORS[temp],
                    linewidth=2, label=f"temp={temp}")
            ax.fill_between(rounds.to_numpy(),
                            means.values - sems.values,
                            means.values + sems.values,
                            color=TEMP_COLORS[temp], alpha=0.15)

        ax.axhline(ENDOWMENT / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_title(COND_LABELS[cond], fontsize=13, fontweight="bold")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Mean Contribution" if ax == axes[0] else "", fontsize=11)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.tick_params(labelsize=10)
        if ax == axes[1]:
            ax.legend(fontsize=10, loc="lower right")

    fig.suptitle("Temperature Robustness: Mean Contribution by Decision Temperature",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temp_contribution_over_rounds.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "temp_contribution_over_rounds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: temp_contribution_over_rounds")


def plot_steady_state_bars(ss: pd.DataFrame):
    """Bar chart of final-5-round mean contribution, grouped by temperature."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, cond in zip(axes, CONDITIONS):
        sub = ss[ss["condition"] == cond]
        x   = np.arange(len(TEMPERATURES))
        bars = ax.bar(x, sub["mean"], yerr=sub["se"], capsize=4,
                      color=[TEMP_COLORS[t] for t in TEMPERATURES],
                      alpha=0.85, width=0.55, error_kw={"linewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}" for t in TEMPERATURES], fontsize=10)
        ax.set_xlabel("Decision Temperature", fontsize=11)
        ax.set_ylabel("Mean Contribution (last 5 rounds)" if ax == axes[0] else "", fontsize=11)
        ax.set_title(COND_LABELS[cond], fontsize=13, fontweight="bold")
        ax.set_ylim(0, ENDOWMENT + 0.5)
        ax.axhline(ENDOWMENT / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.tick_params(labelsize=10)

    fig.suptitle("Steady-State Contribution by Temperature (last 5 rounds, mean ± SE across seeds)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temp_steady_state_bars.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "temp_steady_state_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: temp_steady_state_bars")


def plot_ct_over_rounds(df: pd.DataFrame):
    """Cooperation tendency over rounds (BASELINE only — CT updates via norm internalization)."""
    sub = df[df["condition"] == "BASELINE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for temp in TEMPERATURES:
        ts    = sub[sub["temperature"] == temp].groupby("round")["mean_ct"]
        means = ts.mean()
        sems  = ts.sem()
        ax.plot(means.index.to_numpy(), means.values, color=TEMP_COLORS[temp],
                linewidth=2, label=f"temp={temp}")
        ax.fill_between(means.index.to_numpy(),
                        means.values - sems.values,
                        means.values + sems.values,
                        color=TEMP_COLORS[temp], alpha=0.15)

    ax.set_title("Cooperation Tendency Over Rounds — Baseline", fontsize=13)
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Mean Cooperation Tendency", fontsize=11)
    ax.set_xlim(1, 20)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "temp_ct_over_rounds.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "temp_ct_over_rounds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: temp_ct_over_rounds")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data...")
    df       = load_data()
    df_agent = load_agent_level()
    print(f"  {len(df)} run-round rows | {len(df_agent)} agent-round rows")

    print("\n── Steady-state summary ──────────────────────────────")
    ss = steady_state_summary(df)
    print(ss.to_string(index=False))
    ss.to_csv(os.path.join(OUT_DIR, "temp_steady_state.csv"), index=False)

    print("\n── OLS: contribution ~ temperature + round ───────────")
    ols = ols_temp_effect(df_agent)
    print(ols.to_string(index=False))
    ols.to_csv(os.path.join(OUT_DIR, "temp_ols_results.csv"), index=False)

    print("\n── CV across temperatures (mean over all rounds) ─────")
    cv = cv_by_temp(df)
    for cond in CONDITIONS:
        mean_cv = cv[cv["condition"] == cond]["cv_across_temps"].mean()
        print(f"  {cond}: mean CV = {mean_cv:.4f}")
    cv.to_csv(os.path.join(OUT_DIR, "temp_cv.csv"), index=False)

    print("\n── Generating plots ──────────────────────────────────")
    plot_contributions_over_rounds(df)
    plot_steady_state_bars(ss)
    plot_ct_over_rounds(df)

    print(f"\nAll outputs → {OUT_DIR}")
