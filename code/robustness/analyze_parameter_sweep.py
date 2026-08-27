"""
analyze_parameter_sweep.py
---------------------------
Robustness check: sensitivity of the FULL condition to the network-tie
update-rate asymmetry (positive vs. negative evaluation learning rates).

NOTE ON SCOPE (docstring in parameter_sweep.py is stale — see
CODEBASE_INVENTORY.md and the run script itself): the OFAT design described
there (network_update_rate_neg / tie_remove_threshold / norm_internalization_rate,
each at three levels) was NOT what was actually executed. What is on disk is a
--rate_pairs sweep over three (positive, negative) update-rate pairs applied
jointly:
    pos0.5_neg0.5   (symmetric)
    pos0.4_neg0.6
    pos0.2_neg0.8
run only for the FULL condition, only for the open-source backends
(llama, mistral, qwen — no GPT-4o-mini), 5 seeds (43-47) each. This script
analyzes exactly that data. It does not invent or backfill the OFAT
parameters, the OpenAI backend, or a PURE_BASELINE arm for this experiment,
none of which exist on disk.

DEFAULT / REFERENCE CAVEAT: the main simulation's actual default is
(pos=0.10, neg=0.80) — see SoNoLiSi_v5_os_local.py's SimConfig and
figures/appendix and tables/table_experimental_details.tex ($\alpha^+$=0.10,
$\alpha^-$=0.80). None of the three swept pairs reproduces that default
exactly. pos0.2_neg0.8 is the nearest tested pair (exact match on the
negative rate, 2x the positive rate) and is used as the reference cell per
instruction, but it is a nearby analogue, not a replication, of the
main-paper default. Flagged again in the table note.

SEED-MATCHING CHECK: the same seed identifiers are reused across the three
pairs. `random.seed(seed); np.random.seed(seed)` at the top of run_sim()
means the pure-Python RNG stream (initial network topology, round-1 group
formation) is identical across pairs for a given seed -- verified below by
comparing round-1 `groups`. However, agent decisions are sampled from the
LLM backend, whose stochastic decoding is NOT controlled by that RNG seed,
so round-1 *contributions* already diverge across pairs for the "same"
seed. Conclusion: pairs share a matched starting topology but not matched
behavioral trajectories, so comparisons across pairs are treated as
independent (unpaired) samples, not paired differences. See
param_seed_matching_check.csv.

Metrics (Table S4):
  Contribution:
    - steady-state (last 5 of 20 rounds) mean/SE, aggregated seed-first
    - trajectory beta: contribution ~ round, seed-clustered SEs (FULL only,
      no condition term -- there is no PURE_BASELINE arm here)
    - delta vs. default (pos0.2_neg0.8), unpaired SE combination
      sqrt(se_pair^2 + se_default^2), 95% CI, z, two-sided p
  Network (last 5 rounds, using the *same* definitions as the main-paper
  network analysis, imported directly from eval_network_quantified.py):
    - alignment: Spearman r between incoming edge weight and contribution
    - gini: Gini coefficient of incoming edge weight
    - top_share: share of total incoming weight held by top-contribution-
      quartile agents

Output: figures/robustness/parameter_sweep/
"""

from __future__ import annotations

import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

BASE        = "/data3/rasimura/social-norm-evo"
RESULTS_DIR = f"{BASE}/results/robustness/parameter_sweep"
OUT_DIR     = f"{BASE}/figures/robustness/parameter_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse the exact main-paper network-metric definitions rather than
# reimplementing them (per instruction: use existing definitions, don't
# invent metrics).
sys.path.insert(0, f"{BASE}/code/analysis/selection")
from eval_network_quantified import gini, compute_round_metrics  # noqa: E402

MODELS       = ["llama", "mistral", "qwen"]
MODEL_LABELS = {"llama": "Llama 3.1-8B", "mistral": "Mistral-7B", "qwen": "Qwen2.5-7B"}

PAIRS       = ["pos0p5_neg0p5", "pos0p4_neg0p6", "pos0p2_neg0p8"]
PAIR_VALUES = {
    "pos0p5_neg0p5": (0.5, 0.5),
    "pos0p4_neg0p6": (0.4, 0.6),
    "pos0p2_neg0p8": (0.2, 0.8),
}
DEFAULT_PAIR = "pos0p2_neg0p8"

SEEDS     = list(range(43, 48))
CONDITION = "FULL"
ROUNDS    = 20
LAST_N    = 5
ENDOWMENT = 10


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_log(model: str, pair: str, seed: int) -> dict | None:
    pattern = os.path.join(RESULTS_DIR, model, "rate_pairs", pair, f"seed{seed}",
                            f"log_*_{CONDITION}_seed{seed}.json")
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        if d.get("condition") == CONDITION:
            return d
    return None


def load_round_means(model: str, pair: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        d = load_log(model, pair, seed)
        if d is None:
            print(f"  MISSING: model={model} pair={pair} seed={seed}")
            continue
        for rlog in d["round_logs"]:
            rows.append({
                "seed": seed, "round": rlog["round"],
                "mean_contribution": np.mean(list(rlog["contributions"].values())),
            })
    return pd.DataFrame(rows)


def load_agent_level(model: str, pair: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        d = load_log(model, pair, seed)
        if d is None:
            continue
        for rlog in d["round_logs"]:
            for aid_str, c in rlog["contributions"].items():
                rows.append({"seed": seed, "round": rlog["round"],
                             "agent_id": int(aid_str), "contribution": float(c)})
    return pd.DataFrame(rows)


def load_network_metrics(model: str, pair: str) -> pd.DataFrame:
    """
    Per (seed, round) alignment / gini / top_share via the shared main-paper
    definitions, plus the same "all_equal" trivial-round flag used in
    selection_precision.py (std(contributions) < 1e-6): when contributions
    are fully tied within a round, the 75th-percentile threshold selects
    every agent, so top_share is mechanically forced to 1.0 and alignment
    (Spearman r) is undefined (no variance in contribution) -- not a
    computation bug, a property of the metric under a degenerate
    distribution. Flagged and reported rather than hidden.
    """
    rows = []
    for seed in SEEDS:
        d = load_log(model, pair, seed)
        if d is None:
            continue
        for rlog in d["round_logs"]:
            m = compute_round_metrics(rlog)
            if m is None:
                continue
            con_vals = np.array(list(rlog["contributions"].values()), dtype=float)
            all_equal = float(np.std(con_vals) < 1e-6)
            rows.append({"seed": seed, "round": rlog["round"], "all_equal": all_equal, **m})
    return pd.DataFrame(rows)


# ─── Seed-matching check ───────────────────────────────────────────────────────

def check_seed_matching() -> pd.DataFrame:
    """
    For each (model, seed): are round-1 groups identical across the three
    pairs (shared RNG stream), and are round-1 contributions identical
    (would require deterministic LLM decoding, which we don't have)?
    """
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            logs = {pair: load_log(model, pair, seed) for pair in PAIRS}
            if any(v is None for v in logs.values()):
                continue
            r1 = {pair: logs[pair]["round_logs"][0] for pair in PAIRS}
            groups_ref = r1[PAIRS[0]]["groups"]
            groups_match = all(r1[p]["groups"] == groups_ref for p in PAIRS[1:])
            contrib_ref = r1[PAIRS[0]]["contributions"]
            contrib_match = all(r1[p]["contributions"] == contrib_ref for p in PAIRS[1:])
            rows.append({
                "model": model, "seed": seed,
                "round1_groups_match_across_pairs": groups_match,
                "round1_contributions_match_across_pairs": contrib_match,
            })
    return pd.DataFrame(rows)


# ─── Contribution statistics ──────────────────────────────────────────────────

def steady_state_contribution(df_round: pd.DataFrame) -> tuple[float, float, int]:
    late = df_round[df_round["round"] > (ROUNDS - LAST_N)]
    seed_means = late.groupby("seed")["mean_contribution"].mean()
    return seed_means.mean(), seed_means.sem(), len(seed_means)


def trajectory_ols(df_agent: pd.DataFrame) -> tuple[float, float, float]:
    """contribution ~ round, seed-clustered SEs. FULL-only, so no condition term."""
    try:
        mod = smf.ols("contribution ~ round", data=df_agent).fit(
            cov_type="cluster", cov_kwds={"groups": df_agent["seed"]}
        )
        return mod.params["round"], mod.bse["round"], mod.pvalues["round"]
    except Exception as e:
        print(f"  trajectory OLS failed: {e}")
        return np.nan, np.nan, np.nan


# ─── Network statistics ────────────────────────────────────────────────────────

def steady_state_network(df_net: pd.DataFrame) -> dict:
    """Seed-first aggregation: mean each metric within-seed over the last 5
    rounds, then mean/SE across the 5 seeds."""
    late = df_net[df_net["round"] > (ROUNDS - LAST_N)]
    out = {}
    for metric in ["alignment", "gini", "top_share"]:
        seed_means = late.groupby("seed")[metric].mean()  # NaN-safe (pandas skips NaN in mean)
        out[f"{metric}_mean"] = seed_means.mean()
        out[f"{metric}_se"]   = seed_means.sem()
        out[f"{metric}_n"]    = seed_means.notna().sum()
    out["pct_tied_rounds"] = late["all_equal"].mean() if len(late) else np.nan
    out["n_round_obs"]     = len(late)
    return out


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Seed-matching check ──────────────────────────────")
    match_df = check_seed_matching()
    print(match_df.to_string(index=False))
    match_df.to_csv(os.path.join(OUT_DIR, "param_seed_matching_check.csv"), index=False)
    print(f"\n  round-1 groups match across pairs (same seed): "
          f"{match_df['round1_groups_match_across_pairs'].all()}")
    print(f"  round-1 contributions match across pairs (same seed): "
          f"{match_df['round1_contributions_match_across_pairs'].any()}")

    # ── Pass 1: raw stats per (model, pair) ──
    cells = {}
    for model in MODELS:
        for pair in PAIRS:
            df_round = load_round_means(model, pair)
            df_agent = load_agent_level(model, pair)
            df_net   = load_network_metrics(model, pair)

            c_mean, c_se, n_seeds = steady_state_contribution(df_round)
            traj_b, traj_se, traj_p = trajectory_ols(df_agent)
            net = steady_state_network(df_net)

            pos, neg = PAIR_VALUES[pair]
            cells[(model, pair)] = {
                "model": model, "model_label": MODEL_LABELS[model],
                "pair": pair, "pos_rate": pos, "neg_rate": neg,
                "is_default": pair == DEFAULT_PAIR,
                "n_seeds": n_seeds,
                "contrib_mean": c_mean, "contrib_se": c_se,
                "traj_beta": traj_b, "traj_se": traj_se, "traj_p": traj_p,
                **net,
            }

    steady_df = pd.DataFrame(cells.values())
    steady_df.to_csv(os.path.join(OUT_DIR, "param_steady_state.csv"), index=False)

    # ── Pass 2: delta vs. default (unpaired SE combination) ──
    delta_rows = []
    for model in MODELS:
        default_cell = cells[(model, DEFAULT_PAIR)]
        for pair in PAIRS:
            cell = cells[(model, pair)]
            delta    = cell["contrib_mean"] - default_cell["contrib_mean"]
            se_delta = np.sqrt(cell["contrib_se"]**2 + default_cell["contrib_se"]**2)
            z    = delta / se_delta if se_delta > 0 else np.nan
            pval = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_lo, ci_hi = delta - 1.96 * se_delta, delta + 1.96 * se_delta
            delta_rows.append({
                "model": model, "pair": pair,
                "delta_vs_default": delta, "se_delta": se_delta,
                "ci_lo": ci_lo, "ci_hi": ci_hi, "z": z, "p": pval,
            })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(os.path.join(OUT_DIR, "param_delta_vs_default.csv"), index=False)

    # ── Merge into master table ──
    master = steady_df.merge(delta_df, on=["model", "pair"])
    master = master.sort_values(["model", "pair"])
    master.to_csv(os.path.join(OUT_DIR, "param_sweep_master.csv"), index=False)

    print("\n── Steady-state contribution + network metrics ───────")
    print(master[["model", "pair", "pos_rate", "neg_rate", "is_default",
                   "contrib_mean", "contrib_se", "delta_vs_default", "p",
                   "traj_beta", "traj_p",
                   "alignment_mean", "alignment_n", "gini_mean", "top_share_mean",
                   "pct_tied_rounds"]].to_string(index=False))

    print(f"\nAll outputs -> {OUT_DIR}")
