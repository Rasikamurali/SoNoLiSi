"""
evaluation_multi.py
-------------------
Multi-seed, multi-model evaluation of SoNoLiSi_v5 simulation logs.

Folder structure expected:
  results/{model}/{variant}/seed{N}/log_*.json
  results/{model}/{variant}/seed{N}/conversations_*.json
  where model   ∈ {gpt, llama, mistral, qwen}
        variant ∈ {global, local}

Outputs (CSVs → results/, figures → figures/):
  round_metrics.csv              — per model/variant/condition/seed/round
  round_metrics_aggregated.csv   — mean ± SE across seeds
  run_summary_metrics.csv        — per-run scalars
  condition_summary.csv          — mean ± SE across seeds per condition
  discussion_logodds_rounds.csv  — log-odds R1 vs R20
  discussion_logodds_quartiles.csv — log-odds cooperators vs violators

Usage:
  python evaluation_multi.py --model llama
  python evaluation_multi.py --model llama --variant local
  python evaluation_multi.py --model llama --variant both
  python evaluation_multi.py --all
  python evaluation_multi.py --all --variant both
"""

import json, os, glob, re, argparse
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.corpus import stopwords

# ============================================================
# Paths & constants
# ============================================================

CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures", str(date.today()))

MODELS   = ["gpt", "llama", "mistral", "qwen"]
VARIANTS = ["global", "local"]

ENDOWMENT      = 10
COOP_THRESHOLD = 5.5
FINAL_WINDOW   = 5

CONDITION_ORDER = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
CONDITION_LABELS = {
    "PURE_BASELINE":  "Pure Baseline",
    "BASELINE":       "Baseline",
    "NO_SELECTION":   "No Selection",
    "NO_DISCUSSION":  "No Discussion",
    "FULL":           "Full",
}
COND_COLORS = {
    "PURE_BASELINE":  "#aaaaaa",
    "BASELINE":       "#6baed6",
    "NO_SELECTION":   "#fd8d3c",
    "NO_DISCUSSION":  "#74c476",
    "FULL":           "#e6550d",
}
MODEL_COLORS = {
    "gpt":     "#2166ac",
    "llama":   "#d73027",
    "mistral": "#1a9641",
    "qwen":    "#f46d43",
}

ROUND_METRIC_COLS = [
    "mean_contribution", "mean_payoff",
    "align_norm_behavior", "align_ct_behavior", "align_inj_desc",
    "internalization_gap",
]

RUN_SUMMARY_COLS = [
    "final_cooperation", "time_to_cooperation",
    "final_align_norm_behavior", "final_align_ct_behavior",
    "final_align_inj_desc", "final_internalization_gap",
]

STOPWORDS = set(stopwords.words("english"))

# ============================================================
# Load logs and conversations
# ============================================================

def _best_per_condition(seed_dir: str, prefix: str) -> dict:
    """Within a seed folder, sort files so latest timestamp wins per condition."""
    cond_best = {}
    for path in sorted(glob.glob(os.path.join(seed_dir, f"{prefix}*.json"))):
        with open(path) as f:
            data = json.load(f)
        cond_best[data["condition"]] = data
    return cond_best


def load_model_logs(model: str, variant: str) -> dict:
    """Returns {condition: [data_per_seed, ...]}"""
    base = os.path.join(RESULTS_DIR, model, variant)
    if not os.path.isdir(base):
        return {}
    logs = {}
    for seed_dir in sorted(glob.glob(os.path.join(base, "seed*"))):
        for cond, data in _best_per_condition(seed_dir, "log_").items():
            logs.setdefault(cond, []).append(data)
    return logs


def load_model_conversations(model: str, variant: str) -> dict:
    """Returns {condition: [conv_data_per_seed, ...]}"""
    base = os.path.join(RESULTS_DIR, model, variant)
    if not os.path.isdir(base):
        return {}
    convs = {}
    for seed_dir in sorted(glob.glob(os.path.join(base, "seed*"))):
        for cond, data in _best_per_condition(seed_dir, "conversations_").items():
            convs.setdefault(cond, []).append(data)
    return convs


# ============================================================
# Per-round metrics
# ============================================================

def _norm_vals(round_data: dict, key: str) -> list:
    out = []
    for perc in (round_data.get("perceptions") or {}).values():
        if perc and perc.get(key) is not None:
            try:
                out.append(float(perc[key]))
            except (TypeError, ValueError):
                pass
    return out


def compute_round_metrics(data: dict, model: str, variant: str) -> pd.DataFrame:
    rows = []
    for r in data["round_logs"]:
        contribs  = {int(k): v for k, v in r["contributions"].items()}
        payoffs   = {int(k): v for k, v in r.get("payoffs", {}).items()}
        c_vals    = list(contribs.values())
        p_vals    = list(payoffs.values())
        agent_cts = {a["id"]: a["cooperation_tendency"] for a in r["agent_states"]}
        mean_c    = np.mean(c_vals) if c_vals else np.nan
        mean_p    = np.mean(p_vals) if p_vals else np.nan

        inj_vals  = _norm_vals(r, "injunctive_norm")
        desc_vals = _norm_vals(r, "descriptive_norm")

        # Per-agent alignment and internalization
        align_nb, align_cb, align_id, intern_gaps = [], [], [], []

        if r.get("perceptions"):
            for aid_str, perc in r["perceptions"].items():
                if not perc:
                    continue
                aid = int(aid_str)
                ct  = agent_cts.get(aid, np.nan)
                c_i = contribs.get(aid)

                i_raw = perc.get("injunctive_norm")
                d_raw = perc.get("descriptive_norm")

                i_hat = None
                d_hat = None
                try:
                    if i_raw is not None:
                        i_hat = float(i_raw)
                except (TypeError, ValueError):
                    pass
                try:
                    if d_raw is not None:
                        d_hat = float(d_raw)
                except (TypeError, ValueError):
                    pass

                # 3a) Norm vs behavior: |Î_i - c_i|  (both on 0-10 scale)
                if i_hat is not None and c_i is not None:
                    align_nb.append(abs(i_hat - c_i))

                # 3b) CT vs behavior: |CT_i * E - c_i|
                if not np.isnan(ct) and c_i is not None:
                    align_cb.append(abs(ct * ENDOWMENT - c_i))

                # 3c) Injunctive vs descriptive: |Î_i - D̂_i|
                if i_hat is not None and d_hat is not None:
                    align_id.append(abs(i_hat - d_hat))

                # 4) Internalization: |CT_i - Î_i / E|
                if i_hat is not None and not np.isnan(ct):
                    intern_gaps.append(abs(ct - i_hat / ENDOWMENT))

        rows.append({
            "model":     model,
            "variant":   variant,
            "condition": data["condition"],
            "seed":      data["seed"],
            "round":     r["round"],
            "mean_contribution":    mean_c,
            "mean_payoff":          mean_p,
            "align_norm_behavior":  np.mean(align_nb)    if align_nb    else np.nan,
            "align_ct_behavior":    np.mean(align_cb)    if align_cb    else np.nan,
            "align_inj_desc":       np.mean(align_id)    if align_id    else np.nan,
            "internalization_gap":  np.mean(intern_gaps) if intern_gaps else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# Run-level summary
# ============================================================

def compute_run_summary(round_df: pd.DataFrame, model: str,
                        variant: str, condition: str, seed: int) -> dict:
    df    = round_df.sort_values("round")
    last5 = df.tail(FINAL_WINDOW)

    reached = df[df["mean_contribution"] >= COOP_THRESHOLD]["round"]
    t_coop  = int(reached.iloc[0]) if len(reached) > 0 else np.nan

    return {
        "model":     model,
        "variant":   variant,
        "condition": condition,
        "seed":      seed,
        "final_cooperation":          last5["mean_contribution"].mean(),
        "time_to_cooperation":        t_coop,
        "final_align_norm_behavior":  last5["align_norm_behavior"].mean(),
        "final_align_ct_behavior":    last5["align_ct_behavior"].mean(),
        "final_align_inj_desc":       last5["align_inj_desc"].mean(),
        "final_internalization_gap":  last5["internalization_gap"].mean(),
    }


# ============================================================
# Aggregation
# ============================================================

def _agg_group(grp: pd.DataFrame) -> pd.Series:
    out = {}
    for col in ROUND_METRIC_COLS:
        if col not in grp.columns:
            continue
        vals = grp[col].dropna()
        n    = len(vals)
        out[f"{col}_mean"] = vals.mean()                     if n > 0 else np.nan
        out[f"{col}_std"]  = vals.std(ddof=1)               if n > 1 else np.nan
        out[f"{col}_se"]   = vals.std(ddof=1) / np.sqrt(n)  if n > 1 else np.nan
    return pd.Series(out)


def aggregate_round_metrics(round_df: pd.DataFrame) -> pd.DataFrame:
    return (round_df
            .groupby(["model", "variant", "condition", "round"], sort=False)
            .apply(_agg_group, include_groups=False)
            .reset_index())


def aggregate_run_summaries(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, variant, condition), grp in summary_df.groupby(
            ["model", "variant", "condition"]):
        for col in RUN_SUMMARY_COLS:
            vals = grp[col].dropna()
            n    = len(vals)
            rows.append({
                "model":     model,
                "variant":   variant,
                "condition": condition,
                "metric":    col,
                "mean": vals.mean()                   if n > 0 else np.nan,
                "std":  vals.std(ddof=1)              if n > 1 else np.nan,
                "se":   vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
            })
    return pd.DataFrame(rows)


# ============================================================
# Discussion: log-odds analysis
# ============================================================

def _tokenize(text: str) -> list:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def log_odds_ratio(counts_a: Counter, counts_b: Counter,
                   alpha: float = 0.01) -> pd.DataFrame:
    """
    Monroe et al. (2008) weighted log-odds ratio with Dirichlet prior.
    Positive z-score = word more characteristic of group A.
    """
    vocab  = set(counts_a) | set(counts_b)
    n_a    = sum(counts_a.values())
    n_b    = sum(counts_b.values())
    V      = len(vocab)
    rows   = []
    for w in vocab:
        a_w = counts_a.get(w, 0)
        b_w = counts_b.get(w, 0)
        log_odds = (
            np.log((a_w + alpha) / (n_a + alpha * V - a_w - alpha)) -
            np.log((b_w + alpha) / (n_b + alpha * V - b_w - alpha))
        )
        variance = 1 / (a_w + alpha) + 1 / (b_w + alpha)
        rows.append({
            "word":       w,
            "log_odds":   log_odds,
            "z_score":    log_odds / np.sqrt(variance),
            "count_a":    a_w,
            "count_b":    b_w,
        })
    return pd.DataFrame(rows).sort_values("z_score", ascending=False)


def _extract_discussion_message(raw: str) -> str:
    """Extract 'message' field from raw JSON string, fall back to raw text."""
    try:
        parsed = json.loads(raw)
        return parsed.get("message", raw)
    except (json.JSONDecodeError, TypeError):
        return str(raw)


def compute_discussion_logodds(log_data: dict, conv_data: dict) -> dict:
    """
    Returns two DataFrames:
      'rounds'    — log-odds R1 vs R20
      'quartiles' — log-odds cooperators (Q4) vs violators (Q1)
    Both keyed in returned dict.
    """
    # Build round -> list of messages
    round_messages: dict[int, list] = {}
    for entry in conv_data.get("conversations", []):
        if entry.get("call_type") != "discussion":
            continue
        rnd = entry["round"]
        msg = _extract_discussion_message(entry.get("raw_response", ""))
        round_messages.setdefault(rnd, []).append(msg)

    if not round_messages:
        return {}

    rounds_sorted = sorted(round_messages)
    r_first = rounds_sorted[0]
    r_last  = rounds_sorted[-1]

    # --- R1 vs R_last log-odds ---
    counts_r1   = Counter(_tokenize(" ".join(round_messages[r_first])))
    counts_rlast = Counter(_tokenize(" ".join(round_messages[r_last])))
    df_rounds = log_odds_ratio(counts_r1, counts_rlast)
    df_rounds["comparison"] = f"R{r_first}_vs_R{r_last}"

    # --- Quartile classification from log data ---
    # Mean contribution per agent across all rounds
    agent_mean_contrib: dict[int, float] = {}
    for r in log_data.get("round_logs", []):
        for aid_str, c in r["contributions"].items():
            aid = int(aid_str)
            agent_mean_contrib.setdefault(aid, [])
            agent_mean_contrib[aid].append(c)
    agent_mean_contrib = {k: np.mean(v) for k, v in agent_mean_contrib.items()}

    if not agent_mean_contrib:
        return {"rounds": df_rounds}

    vals = list(agent_mean_contrib.values())
    q25  = np.percentile(vals, 25)
    q75  = np.percentile(vals, 75)

    cooperators = {aid for aid, m in agent_mean_contrib.items() if m >= q75}
    violators   = {aid for aid, m in agent_mean_contrib.items() if m <= q25}

    # Collect messages by agent quartile (across all rounds)
    msgs_coop  = []
    msgs_viol  = []
    for entry in conv_data.get("conversations", []):
        if entry.get("call_type") != "discussion":
            continue
        aid = entry.get("agent_id")
        msg = _extract_discussion_message(entry.get("raw_response", ""))
        if aid in cooperators:
            msgs_coop.append(msg)
        elif aid in violators:
            msgs_viol.append(msg)

    if not msgs_coop or not msgs_viol:
        return {"rounds": df_rounds}

    counts_coop = Counter(_tokenize(" ".join(msgs_coop)))
    counts_viol = Counter(_tokenize(" ".join(msgs_viol)))
    df_quartiles = log_odds_ratio(counts_coop, counts_viol)
    df_quartiles["comparison"] = "cooperators_vs_violators"

    return {"rounds": df_rounds, "quartiles": df_quartiles}


# ============================================================
# Figures 1–4: trajectory plots
# ============================================================

TRAJ_FIGS = [
    ("mean_contribution",   "Fig 1 — Mean Contribution Over Rounds",
     "Contribution (0–10)",  "fig01_mean_contribution_over_rounds"),
    ("mean_payoff",         "Fig 2 — Mean Payoff Over Rounds",
     "Payoff",               "fig02_mean_payoff_over_rounds"),
    ("align_norm_behavior", "Fig 3a — Norm–Behavior Alignment (|injunctive norm − contribution|)",
     "|Î − c| (0–10 scale)", "fig03a_alignment_injunctive_norm_vs_contribution"),
    ("align_ct_behavior",   "Fig 3b — CT–Behavior Alignment (|cooperation tendency·E − contribution|)",
     "|CT·E − c|",           "fig03b_alignment_cooperation_tendency_vs_contribution"),
    ("align_inj_desc",      "Fig 3c — Injunctive–Descriptive Gap (|injunctive norm − descriptive norm|)",
     "|Î − D̂|",             "fig03c_alignment_injunctive_vs_descriptive_norm"),
    ("internalization_gap", "Fig 4 — Internalization Gap (|CT − injunctive norm / E|)",
     "|CT − Î/E|",           "fig04_internalization_gap"),
]


def plot_trajectories(agg_df: pd.DataFrame, model: str, variant: str):
    sub = agg_df[(agg_df["model"] == model) & (agg_df["variant"] == variant)]
    if sub.empty:
        return
    suffix = f"_{model}_{variant}"

    for metric, title, ylabel, fname_base in TRAJ_FIGS:
        mean_col = f"{metric}_mean"
        se_col   = f"{metric}_se"
        if mean_col not in sub.columns:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.suptitle(f"{title} — {model.upper()} ({variant})", fontsize=12)

        for cond in CONDITION_ORDER:
            cdf = sub[sub["condition"] == cond].sort_values("round")
            if cdf.empty or cdf[mean_col].isna().all():
                continue
            rounds = cdf["round"].to_numpy()
            mean   = cdf[mean_col].to_numpy()
            se     = cdf[se_col].to_numpy() if se_col in cdf.columns else np.zeros_like(mean)
            ax.plot(rounds, mean, label=CONDITION_LABELS[cond],
                    color=COND_COLORS[cond], linewidth=2, marker="o", markersize=3)
            ax.fill_between(rounds, mean - se, mean + se,
                            alpha=0.18, color=COND_COLORS[cond])

        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, f"{fname_base}{suffix}")


# ============================================================
# Figure 5–6: log-odds bar charts
# ============================================================

def plot_logodds(df: pd.DataFrame, title: str, label_a: str, label_b: str,
                 fname: str, top_n: int = 20):
    """Horizontal bar chart: top_n words most characteristic of A and B."""
    df = df.dropna(subset=["z_score"])
    top_a = df.nlargest(top_n // 2, "z_score")
    top_b = df.nsmallest(top_n // 2, "z_score")
    plot_df = pd.concat([top_a, top_b]).sort_values("z_score")

    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_df) * 0.35)))
    colors = ["#d73027" if z > 0 else "#4575b4" for z in plot_df["z_score"]]
    ax.barh(plot_df["word"], plot_df["z_score"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Log-odds z-score")
    ax.set_title(f"{title}\n← {label_b}   {label_a} →", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    _save(fig, fname)


# ============================================================
# Shared save helper
# ============================================================

def _save(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=150)
    print(f"Saved → {path}")
    plt.close(fig)


# ============================================================
# Figure 7: boxplots across seeds
# ============================================================

BOX_METRICS = [
    ("final_cooperation",         "Final Cooperation"),
    ("time_to_cooperation",       "Time to Cooperation"),
    ("final_internalization_gap", "Final Internalization Gap"),
]


def plot_boxplots(summary_df: pd.DataFrame, models: list, variant: str):
    df = summary_df[summary_df["variant"] == variant]
    if df.empty:
        return
    from matplotlib.patches import Patch

    n_metrics = len(BOX_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    fig.suptitle(f"Fig 7 — Across-Seed Summaries ({variant})", fontsize=12)

    conds_present = [c for c in CONDITION_ORDER if c in df["condition"].values]
    n_models = len(models)
    width = 0.8 / n_models

    for ax, (metric, title) in zip(axes, BOX_METRICS):
        for m_idx, model in enumerate(models):
            mdf = df[df["model"] == model]
            for c_idx, cond in enumerate(conds_present):
                vals = mdf[mdf["condition"] == cond][metric].dropna().tolist()
                if not vals:
                    continue
                x_pos = c_idx + (m_idx - n_models / 2 + 0.5) * width
                bp = ax.boxplot(vals, positions=[x_pos], widths=width * 0.85,
                                patch_artist=True, manage_ticks=False)
                color = MODEL_COLORS.get(model, "#888888")
                for patch in bp["boxes"]:
                    patch.set(facecolor=color, alpha=0.7)
                for element in ["whiskers", "caps", "medians", "fliers"]:
                    for line in bp[element]:
                        line.set(color=color)

        ax.set_xticks(range(len(conds_present)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds_present],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        handles = [Patch(facecolor=MODEL_COLORS.get(m, "#888"), label=m.upper())
                   for m in models]
        ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    _save(fig, f"fig07_boxplots_final_cooperation_time_internalization_{variant}")


# ============================================================
# Main pipeline
# ============================================================

def run_model_variant(model: str, variant: str,
                      all_round_rows: list,
                      all_summary_rows: list,
                      all_logodds_rounds: list,
                      all_logodds_quartiles: list):
    logs  = load_model_logs(model, variant)
    convs = load_model_conversations(model, variant)

    if not logs:
        print(f"  [{model}/{variant}] No logs found — skipping")
        return

    print(f"  [{model}/{variant}] conditions: "
          + ", ".join(f"{c}({len(v)})" for c, v in logs.items()))

    for cond, runs in logs.items():
        for data in runs:
            seed = data["seed"]

            # Round metrics
            rdf = compute_round_metrics(data, model, variant)
            all_round_rows.append(rdf)
            all_summary_rows.append(
                compute_run_summary(rdf, model, variant, cond, seed)
            )

            # Discussion log-odds (only conditions with discussion)
            conv_runs = convs.get(cond, [])
            conv_data = next(
                (c for c in conv_runs if c.get("seed") == seed), None
            )
            if conv_data:
                result = compute_discussion_logodds(data, conv_data)
                for df_key, df_val in result.items():
                    if df_val is not None and not df_val.empty:
                        df_val = df_val.copy()
                        df_val["model"]     = model
                        df_val["variant"]   = variant
                        df_val["condition"] = cond
                        df_val["seed"]      = seed
                        if df_key == "rounds":
                            all_logodds_rounds.append(df_val)
                        else:
                            all_logodds_quartiles.append(df_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=MODELS)
    parser.add_argument("--variant", choices=["global", "local", "both"],
                        default="global")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    models_to_run = MODELS if args.all else ([args.model] if args.model else None)
    if models_to_run is None:
        parser.print_help()
        raise SystemExit(0)

    variants_to_run = VARIANTS if args.variant == "both" else [args.variant]

    all_round_rows:        list = []
    all_summary_rows:      list = []
    all_logodds_rounds:    list = []
    all_logodds_quartiles: list = []

    for model in models_to_run:
        for variant in variants_to_run:
            run_model_variant(model, variant,
                              all_round_rows, all_summary_rows,
                              all_logodds_rounds, all_logodds_quartiles)

    if not all_round_rows:
        print("No data loaded — exiting.")
        raise SystemExit(1)

    round_df        = pd.concat(all_round_rows, ignore_index=True)
    summary_df      = pd.DataFrame(all_summary_rows)
    agg_round_df    = aggregate_round_metrics(round_df)
    cond_summary_df = aggregate_run_summaries(summary_df)

    # Save CSVs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    round_df.to_csv(        os.path.join(RESULTS_DIR, "round_metrics.csv"),            index=False)
    agg_round_df.to_csv(    os.path.join(RESULTS_DIR, "round_metrics_aggregated.csv"), index=False)
    summary_df.to_csv(      os.path.join(RESULTS_DIR, "run_summary_metrics.csv"),      index=False)
    cond_summary_df.to_csv( os.path.join(RESULTS_DIR, "condition_summary.csv"),        index=False)

    if all_logodds_rounds:
        pd.concat(all_logodds_rounds, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "discussion_logodds_rounds.csv"), index=False)
    if all_logodds_quartiles:
        pd.concat(all_logodds_quartiles, ignore_index=True).to_csv(
            os.path.join(RESULTS_DIR, "discussion_logodds_quartiles.csv"), index=False)

    print(f"\nCSVs saved to {RESULTS_DIR}")

    # Trajectory figures (Figs 1–4) per model/variant
    loaded_models   = round_df["model"].unique().tolist()
    loaded_variants = round_df["variant"].unique().tolist()
    for model in loaded_models:
        for variant in loaded_variants:
            plot_trajectories(agg_round_df, model, variant)

    # Log-odds figures (Figs 5–6) per model/variant/condition
    DISCUSSION_CONDITIONS = {"NO_SELECTION", "FULL"}
    if all_logodds_rounds:
        lo_rounds_df = pd.concat(all_logodds_rounds, ignore_index=True)
        for (model, variant, cond), grp in lo_rounds_df.groupby(
                ["model", "variant", "condition"]):
            if cond not in DISCUSSION_CONDITIONS:
                continue
            # Aggregate z-scores across seeds
            agg = grp.groupby("word")["z_score"].mean().reset_index()
            agg = agg.assign(log_odds=agg["z_score"], count_a=0, count_b=0)
            r_range = grp["comparison"].iloc[0] if not grp.empty else "R1_vs_Rlast"
            parts   = r_range.split("_vs_")
            plot_logodds(agg, f"Fig 5 — Discussion Word Shift: {r_range} ({cond})",
                         parts[0], parts[-1] if len(parts) > 1 else "Rlast",
                         f"fig05_discussion_logodds_early_vs_late_rounds_{model}_{variant}_{cond}")

    if all_logodds_quartiles:
        lo_q_df = pd.concat(all_logodds_quartiles, ignore_index=True)
        for (model, variant, cond), grp in lo_q_df.groupby(
                ["model", "variant", "condition"]):
            if cond not in DISCUSSION_CONDITIONS:
                continue
            agg = grp.groupby("word")["z_score"].mean().reset_index()
            agg = agg.assign(log_odds=agg["z_score"], count_a=0, count_b=0)
            plot_logodds(agg, f"Fig 6 — Discussion: Cooperators vs Violators ({cond})",
                         "Cooperators (Q4)", "Violators (Q1)",
                         f"fig06_discussion_logodds_cooperators_vs_violators_{model}_{variant}_{cond}")

    # Boxplot figure (Fig 7) per variant
    for variant in loaded_variants:
        plot_boxplots(summary_df, loaded_models, variant)

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'Model':<10} {'Variant':<8} {'Condition':<22} {'Metric':<35} "
          f"{'Mean':>10} {'Std':>10} {'SE':>10}")
    print("=" * 110)
    for _, row in cond_summary_df.iterrows():
        def _f(v): return f"{v:.3f}" if pd.notna(v) else "  nan"
        print(f"{row['model']:<10} {row['variant']:<8} "
              f"{CONDITION_LABELS.get(row['condition'], row['condition']):<22} "
              f"{row['metric']:<35} {_f(row['mean']):>10} "
              f"{_f(row['std']):>10} {_f(row['se']):>10}")
    print("=" * 110)
