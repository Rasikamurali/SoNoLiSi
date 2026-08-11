"""
analyze_cross_model.py
----------------------
Cross-model summary of all three robustness tests:
  temperature_sweep   — baseline stability across decision temperatures
  prompt_variants     — baseline stability across structural prompt changes
  prompt_sensitivity  — word-level sensitivity (verb/anchor)

For each test, produces:
  - Steady-state summary table (all models side by side)
  - OLS / gap statistics per model
  - Combined plot: one panel per model, lines/bars by variant/temp

Models: openai, llama, mistral, qwen
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

BASE     = "/data3/rasimura/social-norm-evo/results/robustness"
OUT_DIR  = "/data3/rasimura/social-norm-evo/figures/robustness/cross_model"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS      = ["openai", "llama", "mistral", "qwen"]
MODEL_LABELS = {"openai": "GPT-4o-mini", "llama": "Llama 3.1-8B",
                "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}
SEEDS       = list(range(43, 48))
LAST_N      = 5
ENDOWMENT   = 10

MODEL_COLORS = {"openai": "#1f77b4", "llama": "#ff7f0e",
                "mistral": "#2ca02c", "qwen":  "#d62728"}

# ─── Generic loader ───────────────────────────────────────────────────────────

def load_log(base_dir, label_path, seed, cond):
    """Load log matching condition from base_dir/label_path/seed{seed}/log_*.json"""
    pattern = os.path.join(base_dir, label_path, f"seed{seed}", f"log_*_{cond}_seed{seed}.json")
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        if d.get("condition") == cond:
            return d
    return None


def round_means(base_dir, label_path, seeds, conditions):
    rows = []
    for seed in seeds:
        for cond in conditions:
            d = load_log(base_dir, label_path, seed, cond)
            if d is None:
                continue
            for rlog in d["round_logs"]:
                rows.append({
                    "seed": seed, "condition": cond,
                    "round": rlog["round"],
                    "mean_contribution": np.mean(list(rlog["contributions"].values())),
                })
    return pd.DataFrame(rows)


def agent_rows(base_dir, label_path, seeds, conditions):
    rows = []
    for seed in seeds:
        for cond in conditions:
            d = load_log(base_dir, label_path, seed, cond)
            if d is None:
                continue
            for rlog in d["round_logs"]:
                for aid_str, c in rlog["contributions"].items():
                    rows.append({
                        "seed": seed, "condition": cond,
                        "round": rlog["round"], "contribution": int(c),
                    })
    return pd.DataFrame(rows)


def steady_state(df, group_cols):
    late = df[df["round"] > (20 - LAST_N)]
    rows = []
    for keys, sub in late.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        seed_means = sub.groupby("seed")["mean_contribution"].mean()
        rows.append(dict(zip(group_cols, keys)) | {
            "mean": round(seed_means.mean(), 3),
            "se":   round(seed_means.sem(), 3),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TEMPERATURE SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_temperature_sweep():
    TEMPS      = [0.3, 0.5, 0.7, 1.0]
    CONDITIONS = ["PURE_BASELINE", "BASELINE"]

    all_rows, ols_rows = [], []

    for model in MODELS:
        for temp in TEMPS:
            df = round_means(f"{BASE}/temperature_sweep",
                             f"{model}/temp_{temp}", SEEDS, CONDITIONS)
            if df.empty:
                continue
            for _, row in df.iterrows():
                all_rows.append({"model": model, "temperature": temp, **row.to_dict()})

        # OLS per model per condition
        agent_all = []
        for temp in TEMPS:
            df_a = agent_rows(f"{BASE}/temperature_sweep",
                              f"{model}/temp_{temp}", SEEDS, CONDITIONS)
            if not df_a.empty:
                df_a["temperature"] = temp
                agent_all.append(df_a)
        if not agent_all:
            continue
        df_agent = pd.concat(agent_all)

        for cond in CONDITIONS:
            sub = df_agent[df_agent["condition"] == cond]
            if sub.empty:
                continue
            try:
                mod = smf.ols("contribution ~ temperature + round", data=sub).fit(
                    cov_type="cluster", cov_kwds={"groups": sub["seed"]}
                )
                coef = mod.params["temperature"]
                pval = mod.pvalues["temperature"]
                ols_rows.append({
                    "model": model, "condition": cond,
                    "coef_temp": round(coef, 4),
                    "p": round(pval, 4),
                    "sig": "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "")),
                })
            except Exception:
                pass

    df_all  = pd.DataFrame(all_rows)
    ss      = steady_state(df_all, ["model", "temperature", "condition"])
    ols_df  = pd.DataFrame(ols_rows)

    # Summary table: PURE_BASELINE across temps, all models
    print("\n── Temperature Sweep: PURE_BASELINE steady-state (last 5 rounds) ──")
    pivot = ss[ss["condition"] == "PURE_BASELINE"].pivot_table(
        index="temperature", columns="model", values="mean"
    )[MODELS]
    print(pivot.round(3).to_string())

    print("\n── Temperature Sweep: OLS coef on temperature (p-value) ──")
    print(ols_df.to_string(index=False))

    # CV across temps per model — measure of stability
    print("\n── Temperature Sweep: CV across temperatures (PURE_BASELINE) ──")
    pb = ss[ss["condition"] == "PURE_BASELINE"]
    for model in MODELS:
        vals = pb[pb["model"] == model]["mean"]
        cv   = vals.std() / vals.mean() if vals.mean() > 0 else np.nan
        print(f"  {MODEL_LABELS[model]:15s}: CV = {cv:.4f}")

    # Save
    ss.to_csv(os.path.join(OUT_DIR, "ts_steady_state.csv"), index=False)
    ols_df.to_csv(os.path.join(OUT_DIR, "ts_ols.csv"), index=False)

    # Plot: PURE_BASELINE mean ± SE across temps, all models
    fig, ax = plt.subplots(figsize=(7, 4))
    pb = ss[ss["condition"] == "PURE_BASELINE"]
    for model in MODELS:
        sub = pb[pb["model"] == model].sort_values("temperature")
        ax.plot(sub["temperature"].to_numpy(), sub["mean"].to_numpy(), marker="o",
                color=MODEL_COLORS[model], linewidth=2, label=MODEL_LABELS[model])
        ax.fill_between(sub["temperature"].to_numpy(),
                        (sub["mean"] - sub["se"]).to_numpy(), (sub["mean"] + sub["se"]).to_numpy(),
                        color=MODEL_COLORS[model], alpha=0.12)
    ax.axhline(ENDOWMENT / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Decision Temperature", fontsize=11)
    ax.set_ylabel("Mean Contribution (last 5 rounds)", fontsize=11)
    ax.set_title("Temperature Sweep — Pure Baseline Stability", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, ENDOWMENT + 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ts_pb_stability.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "ts_pb_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ts_pb_stability")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROMPT VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_prompt_variants():
    VARIANTS   = ["standard", "minimal", "tendency_free"]
    CONDITIONS = ["PURE_BASELINE", "FULL"]

    all_rows, gap_rows = [], []

    for model in MODELS:
        for variant in VARIANTS:
            df = round_means(f"{BASE}/prompt_variants",
                             f"{model}/{variant}", SEEDS, CONDITIONS)
            if df.empty:
                continue
            for _, row in df.iterrows():
                all_rows.append({"model": model, "variant": variant, **row.to_dict()})

    df_all = pd.DataFrame(all_rows)
    ss     = steady_state(df_all, ["model", "variant", "condition"])

    # Gap: FULL - PURE_BASELINE per model × variant
    for model in MODELS:
        for variant in VARIANTS:
            pb  = ss[(ss["model"] == model) & (ss["variant"] == variant) & (ss["condition"] == "PURE_BASELINE")]
            fl  = ss[(ss["model"] == model) & (ss["variant"] == variant) & (ss["condition"] == "FULL")]
            if pb.empty or fl.empty:
                continue
            gap    = fl["mean"].values[0] - pb["mean"].values[0]
            se_gap = np.sqrt(fl["se"].values[0]**2 + pb["se"].values[0]**2)
            gap_rows.append({
                "model": model, "variant": variant,
                "pb_mean": pb["mean"].values[0], "full_mean": fl["mean"].values[0],
                "gap": round(gap, 3), "se_gap": round(se_gap, 3),
                "z": round(gap / se_gap, 2) if se_gap > 0 else np.nan,
            })
    gap_df = pd.DataFrame(gap_rows)

    print("\n── Prompt Variants: PURE_BASELINE steady-state ──")
    pivot = ss[ss["condition"] == "PURE_BASELINE"].pivot_table(
        index="variant", columns="model", values="mean"
    )[MODELS]
    print(pivot.round(3).to_string())

    print("\n── Prompt Variants: FULL − PURE_BASELINE gap ──")
    pivot_gap = gap_df.pivot_table(index="variant", columns="model", values="gap")[MODELS]
    print(pivot_gap.round(3).to_string())

    print("\n── Prompt Variants: z-scores for gap ──")
    pivot_z = gap_df.pivot_table(index="variant", columns="model", values="z")[MODELS]
    print(pivot_z.round(2).to_string())

    ss.to_csv(os.path.join(OUT_DIR, "pv_steady_state.csv"), index=False)
    gap_df.to_csv(os.path.join(OUT_DIR, "pv_gaps.csv"), index=False)

    # Plot: grouped bar — gap per variant per model
    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 4), sharey=True)
    x = np.arange(len(VARIANTS))
    for ax, model in zip(axes, MODELS):
        sub_pb = ss[(ss["model"] == model) & (ss["condition"] == "PURE_BASELINE")].set_index("variant")
        sub_fl = ss[(ss["model"] == model) & (ss["condition"] == "FULL")].set_index("variant")
        ax.bar(x - 0.2, [sub_pb.loc[v, "mean"] if v in sub_pb.index else 0 for v in VARIANTS],
               0.35, yerr=[sub_pb.loc[v, "se"] if v in sub_pb.index else 0 for v in VARIANTS],
               capsize=3, color="#aec7e8", label="Pure Baseline")
        ax.bar(x + 0.2, [sub_fl.loc[v, "mean"] if v in sub_fl.index else 0 for v in VARIANTS],
               0.35, yerr=[sub_fl.loc[v, "se"] if v in sub_fl.index else 0 for v in VARIANTS],
               capsize=3, color=MODEL_COLORS[model], label="Full", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["Standard", "Minimal", "Tend-Free"], fontsize=9, rotation=15)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold")
        ax.set_ylim(0, ENDOWMENT + 1)
        ax.axhline(ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        if ax == axes[0]:
            ax.set_ylabel("Mean Contribution (last 5 rounds)", fontsize=10)
            ax.legend(fontsize=9)
    fig.suptitle("Prompt Variants: Full vs Pure Baseline by Model", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pv_gap_bars.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "pv_gap_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pv_gap_bars")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROMPT SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def run_prompt_sensitivity():
    VARIANTS   = ["standard", "verb_only", "anchor_only", "verb_anchor"]
    CONDITIONS = ["PURE_BASELINE", "FULL"]

    all_rows, gap_rows = [], []

    for model in MODELS:
        for variant in VARIANTS:
            df = round_means(f"{BASE}/prompt_sensitivity",
                             f"{model}/{variant}", SEEDS, CONDITIONS)
            if df.empty:
                continue
            for _, row in df.iterrows():
                all_rows.append({"model": model, "variant": variant, **row.to_dict()})

    df_all = pd.DataFrame(all_rows)
    ss     = steady_state(df_all, ["model", "variant", "condition"])

    for model in MODELS:
        for variant in VARIANTS:
            pb = ss[(ss["model"] == model) & (ss["variant"] == variant) & (ss["condition"] == "PURE_BASELINE")]
            fl = ss[(ss["model"] == model) & (ss["variant"] == variant) & (ss["condition"] == "FULL")]
            if pb.empty or fl.empty:
                continue
            gap    = fl["mean"].values[0] - pb["mean"].values[0]
            se_gap = np.sqrt(fl["se"].values[0]**2 + pb["se"].values[0]**2)
            gap_rows.append({
                "model": model, "variant": variant,
                "pb_mean": pb["mean"].values[0], "full_mean": fl["mean"].values[0],
                "gap": round(gap, 3), "se_gap": round(se_gap, 3),
                "z": round(gap / se_gap, 2) if se_gap > 0 else np.nan,
            })
    gap_df = pd.DataFrame(gap_rows)

    print("\n── Prompt Sensitivity: PURE_BASELINE steady-state ──")
    pivot = ss[ss["condition"] == "PURE_BASELINE"].pivot_table(
        index="variant", columns="model", values="mean"
    )[MODELS]
    print(pivot.round(3).to_string())

    print("\n── Prompt Sensitivity: FULL − PURE_BASELINE gap ──")
    pivot_gap = gap_df.pivot_table(index="variant", columns="model", values="gap")[MODELS]
    print(pivot_gap.round(3).to_string())

    print("\n── Prompt Sensitivity: z-scores for gap ──")
    pivot_z = gap_df.pivot_table(index="variant", columns="model", values="z")[MODELS]
    print(pivot_z.round(2).to_string())

    ss.to_csv(os.path.join(OUT_DIR, "ps_steady_state.csv"), index=False)
    gap_df.to_csv(os.path.join(OUT_DIR, "ps_gaps.csv"), index=False)

    # Plot: PURE_BASELINE across variants — are local models stable?
    VARIANT_LABELS = {"standard": "Std", "verb_only": "Verb",
                      "anchor_only": "Anchor", "verb_anchor": "V+A"}
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(VARIANTS))
    width = 0.18
    for i, model in enumerate(MODELS):
        sub = ss[(ss["model"] == model) & (ss["condition"] == "PURE_BASELINE")]
        sub = sub.set_index("variant").reindex(VARIANTS)
        offset = (i - 1.5) * width
        ax.bar(x + offset, sub["mean"].fillna(0),
               width, yerr=sub["se"].fillna(0), capsize=2,
               color=MODEL_COLORS[model], label=MODEL_LABELS[model], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=11)
    ax.set_ylabel("Mean Contribution (last 5 rounds)", fontsize=11)
    ax.set_title("Prompt Sensitivity — Pure Baseline Stability Across Models", fontsize=12)
    ax.axhline(ENDOWMENT / 2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylim(0, ENDOWMENT + 1)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ps_pb_stability.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "ps_pb_stability.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ps_pb_stability")

    # Plot: gap heatmap — rows=variants, cols=models
    fig, ax = plt.subplots(figsize=(7, 4))
    gap_pivot = gap_df.pivot_table(index="variant", columns="model", values="gap")[MODELS].reindex(VARIANTS)
    im = ax.imshow(gap_pivot.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=6)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=10)
    for i in range(len(VARIANTS)):
        for j in range(len(MODELS)):
            val = gap_pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10,
                        color="black" if abs(val) < 4 else "white")
    plt.colorbar(im, ax=ax, label="FULL − PB gap")
    ax.set_title("Prompt Sensitivity: Treatment Gap by Model and Variant", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ps_gap_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "ps_gap_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: ps_gap_heatmap")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TEMPERATURE SWEEP")
    print("=" * 60)
    run_temperature_sweep()

    print("\n" + "=" * 60)
    print("PROMPT VARIANTS")
    print("=" * 60)
    run_prompt_variants()

    print("\n" + "=" * 60)
    print("PROMPT SENSITIVITY")
    print("=" * 60)
    run_prompt_sensitivity()

    print(f"\nAll outputs → {OUT_DIR}")
