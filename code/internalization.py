"""
Norm internalization analysis based on Bicchieri's definition.

An agent has *internalized* a norm when it cooperates independently of
perceived social expectations (empirical + normative).

Internalization index for agent i:
    I_i = mean(c_i) * (1 - |ρ(c_i,  DN_i + IN_i)|)

Where:
    c_i(t)  = contribution at round t  (0–10)
    DN_i(t) = perceived descriptive norm at round t
    IN_i(t) = perceived injunctive norm at round t
    ρ       = Pearson correlation across rounds

Interpretation:
    High mean(c) + low |ρ|  → internalized cooperator
    High mean(c) + high |ρ| → conditional conformist (Bicchieri conformity)
    Low  mean(c) + any  |ρ| → defector / violator
"""

import json
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-20"
OUT_DIR  = os.path.join(FIG_ROOT, "internalization")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS     = ["gpt", "llama", "mistral", "qwen"]
VARIANTS   = ["global", "local"]
SEEDS      = list(range(43, 53))
CONDITIONS = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION"]

CT_BINS   = {"Violators": (0.0, 0.3), "Normal": (0.3, 0.7), "Cooperators": (0.7, 1.01)}
CT_COLORS = {"Violators": "#d62728", "Normal": "#ff7f0e", "Cooperators": "#2ca02c"}

COND_LABELS = {
    "BASELINE":      "B",
    "FULL":          "F",
    "NO_DISCUSSION": "ND",
    "NO_SELECTION":  "NS",
}

N_BOOT = 1000

# ── log loading ───────────────────────────────────────────────────────────────
def load_latest_log(model: str, variant: str, seed: int, condition: str):
    path = os.path.join(RESULTS, model, variant, f"seed{seed}")
    pattern = os.path.join(path, f"log_*_{condition}_seed{seed}.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


# ── per-agent series extraction ───────────────────────────────────────────────
def extract_agent_series(log: dict) -> dict:
    """
    Returns dict: agent_id -> {
        'contributions': list[float],
        'dn':            list[float],   # may be shorter (only rounds with perception)
        'in':            list[float],
        'ct':            float,
        'rounds_c':      list[int],
        'rounds_dn':     list[int],
    }
    """
    agents = {}

    # cooperation tendency from round-1 agent_states
    for r in log["round_logs"]:
        for s in r.get("agent_states") or []:
            aid = int(s["id"])
            if aid not in agents:
                agents[aid] = {
                    "contributions": [], "dn": [], "in_": [],
                    "ct": float(s.get("cooperation_tendency", np.nan)),
                    "rounds_c": [], "rounds_dn": [],
                }

    for r in log["round_logs"]:
        rnd = int(r["round"])

        # contributions (dict str→int)
        for aid_s, c in (r.get("contributions") or {}).items():
            aid = int(aid_s)
            if aid not in agents:
                agents[aid] = {"contributions": [], "dn": [], "in_": [],
                               "ct": np.nan, "rounds_c": [], "rounds_dn": []}
            agents[aid]["contributions"].append(float(c))
            agents[aid]["rounds_c"].append(rnd)

        # perceptions (dict int→{descriptive_norm, injunctive_norm, ...})
        percs = r.get("perceptions")
        if not percs:
            continue
        for aid_k, p in percs.items():
            aid = int(aid_k)
            dn = p.get("descriptive_norm")
            in_ = p.get("injunctive_norm")
            if dn is None or in_ is None:
                continue
            if aid not in agents:
                agents[aid] = {"contributions": [], "dn": [], "in_": [],
                               "ct": np.nan, "rounds_c": [], "rounds_dn": []}
            agents[aid]["dn"].append(float(dn))
            agents[aid]["in_"].append(float(in_))
            agents[aid]["rounds_dn"].append(rnd)

    return agents


# ── incoming weight (final round) ─────────────────────────────────────────────
def final_incoming_weights(log: dict) -> dict:
    """Returns {agent_id: incoming_weight} from the last round's network."""
    last = log["round_logs"][-1]
    nw = last.get("network_weights") or []
    weights = {}
    for edge in nw:
        tgt = int(edge.get("target", edge.get("v")))
        weights[tgt] = weights.get(tgt, 0.0) + float(edge["weight"])
    return weights


# ── internalization index ─────────────────────────────────────────────────────
def internalization_index(c: list, dn: list, in_: list) -> float:
    """
    I_i = mean(c) * (1 - |ρ(c, DN+IN)|)

    Requires at least 3 overlapping rounds.  Returns NaN if not enough data.
    If std(c) == 0 (constant contributor), ρ is treated as 0.
    """
    if len(c) < 3 or len(dn) < 3:
        return np.nan

    # match on round index — dn/in_ may be a subset of c rounds
    # simplest: use the first min(len(c), len(dn)) overlapping rounds
    n = min(len(c), len(dn))
    c_arr  = np.array(c[:n])
    sum_arr = np.array(dn[:n]) + np.array(in_[:n])

    mean_c = c_arr.mean()
    if c_arr.std() < 1e-9 or sum_arr.std() < 1e-9:
        rho = 0.0
    else:
        rho, _ = stats.pearsonr(c_arr, sum_arr)

    return mean_c * (1.0 - abs(rho))


# ── CT group helper ───────────────────────────────────────────────────────────
def ct_group(ct: float) -> str:
    for name, (lo, hi) in CT_BINS.items():
        if lo <= ct < hi:
            return name
    return "Normal"


# ── build full dataframe ──────────────────────────────────────────────────────
def build_dataframe() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for cond in CONDITIONS:
                    log = load_latest_log(model, variant, seed, cond)
                    if log is None:
                        continue
                    series = extract_agent_series(log)
                    iw     = final_incoming_weights(log)
                    for aid, data in series.items():
                        idx = internalization_index(
                            data["contributions"], data["dn"], data["in_"])
                        rows.append({
                            "model":     model,
                            "variant":   variant,
                            "seed":      seed,
                            "condition": cond,
                            "agent_id":  aid,
                            "ct":        data["ct"],
                            "ct_group":  ct_group(data["ct"]) if not np.isnan(data["ct"]) else "Normal",
                            "I":         idx,
                            "mean_c":    np.mean(data["contributions"]) if data["contributions"] else np.nan,
                            "rho":       (internalization_index(data["contributions"], data["dn"], data["in_"])
                                          / np.mean(data["contributions"])
                                          if (np.mean(data["contributions"]) or 0) > 1e-9 else np.nan),
                            "incoming_weight": iw.get(aid, np.nan),
                        })
    df = pd.DataFrame(rows)
    # recompute rho cleanly
    def _rho(row):
        c  = row["mean_c"]
        I  = row["I"]
        if np.isnan(I) or np.isnan(c) or c < 1e-9:
            return np.nan
        return 1.0 - I / c
    df["abs_rho"] = df.apply(_rho, axis=1)
    return df


# ── bootstrap CI ─────────────────────────────────────────────────────────────
def mean_ci(values, n=N_BOOT):
    v = np.array(values)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    boots = [np.mean(np.random.choice(v, size=len(v), replace=True)) for _ in range(n)]
    return np.mean(v), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ── Plot 1: I_i distribution by CT group per model (all conditions pooled) ────
def plot_by_ct_group(df: pd.DataFrame):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 4), sharey=False)
    fig.suptitle("Internalization index by CT group (all conditions)", fontsize=13)

    groups = list(CT_BINS.keys())
    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        data = [sub[sub["ct_group"] == g]["I"].dropna().values for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(CT_COLORS[g])
        ax.set_title(model.upper())
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=15, ha="right")
        ax.set_ylabel("I_i" if model == MODELS[0] else "")
        ax.set_xlabel("CT group")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "int_by_ct_group.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ── Plot 2: I_i by condition per model (CT groups as hue) ────────────────────
def plot_by_condition(df: pd.DataFrame):
    conds = CONDITIONS
    groups = list(CT_BINS.keys())
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12, 4 * len(MODELS)), sharey=False)
    fig.suptitle("Internalization index by condition and CT group", fontsize=13)

    x = np.arange(len(conds))
    width = 0.25

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        for gi, g in enumerate(groups):
            means, lows, highs = [], [], []
            for cond in conds:
                vals = sub[(sub["condition"] == cond) & (sub["ct_group"] == g)]["I"].dropna().values
                m, lo, hi = mean_ci(vals)
                means.append(m); lows.append(m - lo); highs.append(hi - m)
            offset = (gi - 1) * width
            bars = ax.bar(x + offset, means, width,
                          color=CT_COLORS[g], label=g, alpha=0.85)
            ax.errorbar(x + offset, means,
                        yerr=[lows, highs],
                        fmt="none", color="black", capsize=3, linewidth=1)
        ax.set_title(model.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in conds])
        ax.set_ylabel("Mean I_i [95% CI]")
        ax.legend(title="CT group", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "int_by_condition.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ── Plot 3: I_i vs incoming weight scatter ────────────────────────────────────
def plot_vs_incoming_weight(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig.suptitle("Internalization index vs. final incoming network weight", fontsize=13)

    for ax, model in zip(axes, MODELS):
        sub = df[(df["model"] == model) & df["I"].notna() & df["incoming_weight"].notna()]
        for g in CT_BINS:
            g_sub = sub[sub["ct_group"] == g]
            ax.scatter(g_sub["incoming_weight"], g_sub["I"],
                       color=CT_COLORS[g], alpha=0.3, s=10, label=g)
        # overall regression
        x = sub["incoming_weight"].values
        y = sub["I"].values
        if len(x) > 10:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, slope * xr + intercept, "k--", linewidth=1.5,
                    label=f"r={r:.2f} p={p:.3f}")
        ax.set_title(model.upper())
        ax.set_xlabel("Incoming weight (final round)")
        ax.set_ylabel("I_i")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "int_vs_weight.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ── Plot 4: Global vs local I_i per model per condition ──────────────────────
def plot_global_vs_local(df: pd.DataFrame):
    conds = CONDITIONS
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle("Internalization index: global vs local", fontsize=13)

    x = np.arange(len(conds))
    width = 0.3
    var_colors = {"global": "#1f77b4", "local": "#ff7f0e"}

    for ax, model in zip(axes, MODELS):
        sub = df[df["model"] == model]
        for vi, variant in enumerate(VARIANTS):
            means, lows, highs = [], [], []
            for cond in conds:
                vals = sub[(sub["variant"] == variant) & (sub["condition"] == cond)]["I"].dropna().values
                m, lo, hi = mean_ci(vals)
                means.append(m); lows.append(m - lo); highs.append(hi - m)
            offset = (vi - 0.5) * width
            ax.bar(x + offset, means, width,
                   color=var_colors[variant], label=variant, alpha=0.85)
            ax.errorbar(x + offset, means,
                        yerr=[lows, highs],
                        fmt="none", color="black", capsize=3, linewidth=1)
        ax.set_title(model.upper())
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS[c] for c in conds])
        ax.set_ylabel("Mean I_i [95% CI]")
        ax.legend(title="Variant")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "int_global_vs_local.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ── Summary table ─────────────────────────────────────────────────────────────
def save_summary_table(df: pd.DataFrame):
    rows = []
    for model in MODELS:
        for variant in VARIANTS:
            for cond in CONDITIONS:
                for g in CT_BINS:
                    vals = df[
                        (df["model"] == model) &
                        (df["variant"] == variant) &
                        (df["condition"] == cond) &
                        (df["ct_group"] == g)
                    ]["I"].dropna().values
                    m, lo, hi = mean_ci(vals)
                    rows.append({
                        "model": model, "variant": variant,
                        "condition": cond, "ct_group": g,
                        "mean_I": round(m, 3) if not np.isnan(m) else np.nan,
                        "ci_lo":  round(lo, 3) if not np.isnan(lo) else np.nan,
                        "ci_hi":  round(hi, 3) if not np.isnan(hi) else np.nan,
                        "n":      len(vals),
                    })
    table = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "internalization_summary.csv")
    table.to_csv(out, index=False)
    print(f"Saved → {out}")
    return table


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building dataframe …")
    df = build_dataframe()
    print(f"  {len(df)} agent-run records, {df['I'].notna().sum()} with valid I_i")

    # save raw data
    raw_out = os.path.join(OUT_DIR, "internalization_raw.csv")
    df.to_csv(raw_out, index=False)
    print(f"Saved → {raw_out}")

    print("Plotting …")
    plot_by_ct_group(df)
    plot_by_condition(df)
    plot_vs_incoming_weight(df)
    plot_global_vs_local(df)
    save_summary_table(df)

    print("Done.")
