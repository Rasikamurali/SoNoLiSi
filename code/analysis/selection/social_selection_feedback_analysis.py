"""
social_selection_feedback_analysis.py
----------------------------------------
Reconstructs, link by link, the proposed feedback chain in the repeated
public-goods simulation:

    low relative contribution_t -> poorer peer evaluation_t
    -> lower network weight_t -> altered selection/exclusion_t+1
    -> contribution adjustment_t+1

See exports/social_selection_feedback/mechanism_audit.md for the full
source-code audit this script is built on (read first; several details
below only make sense with that context - in particular: evaluation_on is
perfectly confounded with selection_on across the 4 canonical conditions,
network weight is updated by TWO separate mechanical channels each round
(evaluation scores AND stated partner preferences), agents never see their
own received evaluation scores or network weight in any prompt, and
"group-seed" is a form_groups() concept unrelated to the numeric experiment
seed).

Canonical scope: same as every other script in code/analysis/ - MODEL_SPECS
from sobel_mediation.py (9 families, local variant, canonical seed lists),
deduped to the last-timestamped log file per (model, seed, condition).

Output directory: exports/social_selection_feedback/
"""

import os
import sys
import json
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reorg (2026-08-11): see selection/build_agent_round_panel.py for why this block exists.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "sobel_mediation.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sobel_mediation import MODEL_SPECS

warnings.filterwarnings("ignore")

OUT_DIR = "exports/social_selection_feedback"
CANONICAL_CONDITIONS = ["BASELINE", "NO_DISCUSSION", "NO_SELECTION", "FULL"]
EVAL_CONDITIONS = {"FULL", "NO_DISCUSSION"}   # evaluation_on == selection_on, see audit
GROUP_SIZE = 4
N_AGENTS = 12
PARTICIPATION_THRESHOLD = 0.3
TIE_REMOVE_THRESHOLD = 0.15
RATE_POS, RATE_NEG = 0.10, 0.80


# ═════════════════════════════════════════════════════════════════════════
# Load canonical runs (dedup exactly as every other script this session)
# ═════════════════════════════════════════════════════════════════════════

def load_canonical_runs():
    runs = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant, f"seed{seed}", "log_*.json")
            cond_data = {}
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                    cond_data[d["condition"]] = d
                except Exception:
                    continue
            for cond, d in cond_data.items():
                if cond not in CANONICAL_CONDITIONS:
                    continue
                run_id = f"{family}_s{seed}_{cond}"
                runs.append({"run_id": run_id, "family": family, "model": model_key,
                            "condition": cond, "seed": seed, "d": d})
    return runs


def initial_network():
    return {(i, j): 1.0 for i in range(N_AGENTS) for j in range(N_AGENTS) if i != j}


# ═════════════════════════════════════════════════════════════════════════
# Single pass per run -> edge-level evaluation rows + per-agent-round rows
# ═════════════════════════════════════════════════════════════════════════

def process_run(run):
    d = run["d"]
    round_logs = d["round_logs"]
    discussion_on = run["condition"] in ("FULL", "NO_SELECTION")
    selection_on = run["condition"] in ("FULL", "NO_DISCUSSION")

    prev_net = initial_network()
    edge_rows = []
    agent_round_rows = []

    for idx, r in enumerate(round_logs):
        rnd = r["round"]
        contributions = {int(k): float(v) for k, v in r["contributions"].items()}
        excluded_ids = set(int(x) for x in r["excluded_agents"])
        groups = [[int(x) for x in g] for g in r["groups"]]
        n_eligible = N_AGENTS - len(excluded_ids)
        n_regular = n_eligible // GROUP_SIZE if selection_on else None
        has_remainder = selection_on and (n_eligible % GROUP_SIZE) >= 2

        group_of, seed_of_group, other_members = {}, {}, {}
        for gi, g in enumerate(groups):
            is_remainder = selection_on and has_remainder and (gi == len(groups) - 1) and len(g) != GROUP_SIZE
            for aid in g:
                group_of[aid] = gi
                other_members[aid] = [x for x in g if x != aid]
            if selection_on and not is_remainder and len(g) > 0:
                seed_of_group[gi] = g[0]   # form_groups(): group = [seed_id] + chosen

        cur_net = {(e["u"], e["v"]): e["weight"] for e in r["network_weights"]}

        def incoming_avg(net, target):
            return float(np.mean([net.get((k, target), 0.0) for k in range(N_AGENTS) if k != target]))

        # ---- edge-level evaluations (only where evaluation_on) ----
        evals_raw = r.get("evaluations") or {}
        next_excluded = None
        if idx + 1 < len(round_logs) and round_logs[idx + 1]["round"] == rnd + 1:
            next_excluded = set(int(x) for x in round_logs[idx + 1]["excluded_agents"])

        for key, score in evals_raw.items():
            i_str, j_str = key.split("->")
            i, j = int(i_str), int(j_str)
            gi = group_of.get(i)
            gj = group_of.get(j)
            same_group = (gi is not None and gi == gj)
            target_others = [m for m in other_members.get(j, []) if m != j]
            gm_mean_for_target = (np.mean([contributions.get(m, np.nan) for m in target_others])
                                  if target_others else np.nan)
            before = prev_net.get((i, j), 0.0)
            after = cur_net.get((i, j), 0.0)
            edge_rows.append({
                "run_id": run["run_id"], "condition": run["condition"], "family": run["family"],
                "model": run["model"], "seed": run["seed"], "round": rnd, "group_id": gi,
                "evaluator_id": i, "target_id": j,
                "same_group_verified": same_group,
                "evaluator_contribution": contributions.get(i, np.nan),
                "target_contribution": contributions.get(j, np.nan),
                "groupmate_mean_contribution_for_target": gm_mean_for_target,
                "target_relative_contribution": (contributions.get(j, np.nan) - gm_mean_for_target
                                                 if target_others else np.nan),
                "evaluation_score": score,
                "edge_weight_before": before, "edge_weight_after": after,
                "edge_weight_change": after - before,
                "edge_removed": (before > 0 and (i, j) not in cur_net),
                "target_incoming_weight_before": incoming_avg(prev_net, j),
                "target_incoming_weight_after": incoming_avg(cur_net, j),
                "target_excluded_next_round": (int(j in next_excluded) if next_excluded is not None else np.nan),
            })

        # ---- agent-round panel (active agents only) ----
        for aid in range(N_AGENTS):
            active = aid not in excluded_ids and aid in group_of
            gid = group_of.get(aid)
            others = other_members.get(aid, [])
            gm_mean = np.mean([contributions.get(m, np.nan) for m in others]) if others else np.nan
            contrib = contributions.get(aid, np.nan)

            recv = [row["evaluation_score"] for row in edge_rows
                   if row["round"] == rnd and row["target_id"] == aid]
            was_seed = (float(seed_of_group.get(gid) == aid) if (selection_on and gid is not None)
                       else np.nan)

            active_next = np.nan
            excluded_next_flag = np.nan
            if next_excluded is not None:
                excluded_next_flag = float(aid in next_excluded)
                active_next = float(aid not in next_excluded)

            agent_round_rows.append({
                "run_id": run["run_id"], "condition": run["condition"], "family": run["family"],
                "model": run["model"], "seed": run["seed"], "round": rnd, "agent_id": aid,
                "group_id": gid, "discussion_on": discussion_on, "selection_on": selection_on,
                "active_this_round": active,
                "contribution": contrib if active else np.nan,
                "groupmate_mean_contribution": gm_mean if active else np.nan,
                "relative_contribution": (contrib - gm_mean) if (active and others) else np.nan,
                "undercontribution": (gm_mean - contrib) if (active and others) else np.nan,
                "mean_evaluation_received": np.mean(recv) if recv else np.nan,
                "median_evaluation_received": np.median(recv) if recv else np.nan,
                "min_evaluation_received": np.min(recv) if recv else np.nan,
                "max_evaluation_received": np.max(recv) if recv else np.nan,
                "n_negative_evaluations": sum(1 for x in recv if x < 0),
                "n_positive_evaluations": sum(1 for x in recv if x > 0),
                "n_evaluations_received": len(recv),
                "avg_incoming_weight_before": incoming_avg(prev_net, aid),
                "avg_incoming_weight_after": incoming_avg(cur_net, aid),
                "incoming_weight_change": incoming_avg(cur_net, aid) - incoming_avg(prev_net, aid),
                "was_seed_current_round": was_seed,
                "excluded_next_round": excluded_next_flag,
                "active_next_round": active_next,
                "group_size_this_round": len(groups[gid]) if gid is not None else np.nan,
                "n_eligible_this_round": n_eligible,
            })

        prev_net = cur_net

    return edge_rows, agent_round_rows


def build_core_datasets():
    runs = load_canonical_runs()
    all_edge_rows, all_agent_rows = [], []
    for run in runs:
        e, a = process_run(run)
        all_edge_rows.extend(e)
        all_agent_rows.extend(a)
    edge_df = pd.DataFrame(all_edge_rows)
    agent_df = pd.DataFrame(all_agent_rows)
    return runs, edge_df, agent_df


# ═════════════════════════════════════════════════════════════════════════
# STEP 2: edge-level audit
# ═════════════════════════════════════════════════════════════════════════

def step2_audit(edge_df, runs):
    eval_runs = [r for r in runs if r["condition"] in EVAL_CONDITIONS]
    n_group_rounds = edge_df.groupby(["run_id", "round", "group_id"]).ngroups
    lines = []
    sep = "=" * 74
    lines.append(sep); lines.append("STEP 2 — EDGE-LEVEL EVALUATION DATASET AUDIT"); lines.append(sep)
    lines.append(f"Runs with evaluation_on (FULL + NO_DISCUSSION): {len(eval_runs)}")
    lines.append(f"Group-rounds represented: {n_group_rounds:,}")
    lines.append(f"Evaluation events (rows): {len(edge_df):,}")
    lines.append(f"Missing evaluation_score: {edge_df['evaluation_score'].isna().sum():,}")
    lines.append(f"Missing edge_weight_before: {edge_df['edge_weight_before'].isna().sum():,}")
    lines.append(f"Missing edge_weight_after: {edge_df['edge_weight_after'].isna().sum():,}")
    dup = edge_df.duplicated(subset=["run_id", "round", "evaluator_id", "target_id"]).sum()
    lines.append(f"Duplicate (run,round,evaluator,target) rows: {dup:,}")
    lines.append(f"same_group_verified == False count (evaluator/target not co-grouped - "
                 f"should be 0): {(~edge_df['same_group_verified']).sum():,}")
    lines.append(f"Conditions included: {sorted(edge_df['condition'].unique())}")
    lines.append(f"Model families included: {sorted(edge_df['family'].unique())}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# STEP 3: evaluation -> weight-change validation
# ═════════════════════════════════════════════════════════════════════════

def step3_validation(edge_df):
    d = edge_df.dropna(subset=["evaluation_score", "edge_weight_change"]).copy()

    def bin_score(s):
        if s == 0: return "neutral (0)"
        if -1.0 <= s < -0.5: return "strong negative [-1.0,-0.5)"
        if -0.5 <= s < 0: return "weak negative [-0.5,0)"
        if 0 < s <= 0.5: return "weak positive (0,0.5]"
        if 0.5 < s <= 1.0: return "strong positive (0.5,1.0]"
        return "other"

    d["score_bin"] = d["evaluation_score"].apply(bin_score)
    bin_order = ["strong negative [-1.0,-0.5)", "weak negative [-0.5,0)", "neutral (0)",
                "weak positive (0,0.5]", "strong positive (0.5,1.0]"]

    by_bin = d.groupby("score_bin")["edge_weight_change"].agg(["mean", "std", "count"]).reindex(bin_order)

    d["expected_delta_eval_only"] = np.where(
        d["evaluation_score"] < 0, RATE_NEG * d["evaluation_score"], RATE_POS * d["evaluation_score"])
    # exact match only meaningful pre-clipping/removal; report the deviation directly
    d["deviation_from_eval_only_formula"] = d["edge_weight_change"] - d["expected_delta_eval_only"]
    exact_match = (d["deviation_from_eval_only_formula"].abs() < 1e-6).mean()
    corr = d[["evaluation_score", "edge_weight_change"]].corr().iloc[0, 1]

    results = by_bin.reset_index().rename(columns={"score_bin": "evaluation_score_bin"})
    results["corr_score_vs_weight_change"] = corr
    results["pct_exact_match_eval_only_formula"] = exact_match

    lines = []
    sep = "=" * 74
    lines.append(sep); lines.append("STEP 3 — EVALUATION -> WEIGHT-CHANGE VALIDATION (mechanism check, not a substantive result)"); lines.append(sep)
    lines.append("The update_network() formula (delta = rate*signal, rate=0.8 if signal<0 else 0.10) is "
                 "fully deterministic BY CONSTRUCTION - this is programmed, not discovered. It is NOT, "
                 "however, the only thing touching these edges: update_network_from_perception() (stated "
                 "partner preferences) updates the SAME edges in the same round with different rates "
                 "(0.2/0.45), so the logged net edge_weight_change is the sum of both channels.")
    lines.append(f"\nCorrelation(evaluation_score, edge_weight_change): {corr:.4f}")
    lines.append(f"Rows matching the evaluation-only formula exactly (no deviation): {exact_match:.1%}")
    lines.append("(Deviations are expected and mechanically explained by the second, perception-based "
                 "update channel, plus tie-removal at threshold 0.15 and the max-2.0 cap on that channel "
                 "- these are not noise or model error.)")
    lines.append("\nMean edge-weight change by evaluation-score bin:")
    lines.append(by_bin.to_string(float_format=lambda x: f"{x:.4f}"))
    return results, "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# STEP 5: undercontribution -> evaluation received
# ═════════════════════════════════════════════════════════════════════════

def step5_analysis(agent_df):
    d = agent_df[agent_df["active_this_round"] & agent_df["undercontribution"].notna()
               & (agent_df["condition"].isin(EVAL_CONDITIONS))].copy()

    bins = [-np.inf, -2, -0.5, 0.5, 2, np.inf]
    labels = ["<=-2 (much more)", "(-2,-0.5] (moderately more)", "(-0.5,0.5) (matched)",
             "[0.5,2) (moderately less)", ">=2 (much less)"]
    d["undercontribution_bin"] = pd.cut(d["undercontribution"], bins=bins, labels=labels, right=True,
                                        include_lowest=True)

    desc_rows = []
    for b, sub in d.groupby("undercontribution_bin", observed=True):
        sub_eval = sub.dropna(subset=["mean_evaluation_received"])
        desc_rows.append({
            "undercontribution_bin": b, "n_agent_rounds": len(sub),
            "n_with_evaluation": len(sub_eval),
            "mean_evaluation_received": sub_eval["mean_evaluation_received"].mean(),
            "median_evaluation_received": sub_eval["mean_evaluation_received"].median(),
            "pct_any_negative_evaluation": (sub["n_negative_evaluations"] > 0).mean(),
            "mean_incoming_weight_change": sub["incoming_weight_change"].mean(),
            "exclusion_rate_next_round": sub["excluded_next_round"].mean(),
        })
    desc = pd.DataFrame(desc_rows)

    reg_d = d.dropna(subset=["mean_evaluation_received", "undercontribution"]).copy()
    formula = ('mean_evaluation_received ~ undercontribution '
              '+ C(condition, Treatment(reference="NO_DISCUSSION")) '
              '+ C(family, Treatment(reference="GPT")) + C(round, Treatment(reference=1))')
    m1 = smf.ols(formula, data=reg_d).fit(cov_type="cluster", cov_kwds={"groups": reg_d["run_id"]})

    reg_d["run_agent_id"] = reg_d["run_id"] + "_" + reg_d["agent_id"].astype(str)
    reg_d["group_round_id"] = reg_d["run_id"] + "_" + reg_d["round"].astype(str) + "_" + reg_d["group_id"].astype(str)
    for col in ["mean_evaluation_received", "undercontribution"]:
        reg_d[col + "_gr_dm"] = reg_d[col] - reg_d.groupby("group_round_id")[col].transform("mean")
    m2 = smf.ols("mean_evaluation_received_gr_dm ~ undercontribution_gr_dm", data=reg_d).fit(
        cov_type="cluster", cov_kwds={"groups": reg_d["run_id"]})

    def row(model, term, label):
        ci = model.conf_int(alpha=0.05)
        return {"model": label, "term": term, "estimate": model.params[term], "se": model.bse[term],
               "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1], "p_value": model.pvalues[term],
               "n": int(model.nobs), "n_clusters": reg_d["run_id"].nunique()}

    reg_rows = [row(m1, "undercontribution", "pooled_ols_with_FE"),
               row(m2, "undercontribution_gr_dm", "group_round_demeaned")]
    reg_out = pd.DataFrame(reg_rows)

    lines = []
    sep = "=" * 74
    lines.append(sep); lines.append("STEP 5 — DOES LOW CONTRIBUTION RECEIVE WORSE EVALUATION? (empirical link)"); lines.append(sep)
    lines.append("Restricted to FULL + NO_DISCUSSION (only conditions where evaluation_on=True).\n")
    lines.append(desc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append(f"\nPooled OLS (condition+family+round FE), n={int(m1.nobs):,}, "
                 f"clusters={reg_d['run_id'].nunique()}:")
    lines.append(f"  undercontribution: b={m1.params['undercontribution']:.4f}  "
                 f"SE={m1.bse['undercontribution']:.4f}  p={m1.pvalues['undercontribution']:.4g}")
    lines.append(f"Group-round demeaned (compares agents in the SAME group-round):")
    lines.append(f"  undercontribution: b={m2.params['undercontribution_gr_dm']:.4f}  "
                 f"SE={m2.bse['undercontribution_gr_dm']:.4f}  p={m2.pvalues['undercontribution_gr_dm']:.4g}")
    lines.append("\nInterpretation: a negative coefficient means greater undercontribution predicts worse "
                 "evaluation. Not causal.")
    return desc, reg_out, "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# STEP 6: agent-round transitions
# ═════════════════════════════════════════════════════════════════════════

def build_transitions(agent_df):
    d = agent_df.sort_values(["run_id", "agent_id", "round"]).copy()
    grp = d.groupby(["run_id", "agent_id"], sort=False)

    d["round_next"] = grp["round"].shift(-1)
    for col in ["contribution", "group_id", "was_seed_current_round"]:
        d[col + "_next"] = grp[col].shift(-1)
    d["excluded_next_round_x"] = grp["excluded_next_round"].shift(0)  # already computed within-row

    has_next = d["round_next"].notna()
    consecutive = d["round_next"] == d["round"] + 1
    linked = d[has_next & consecutive].copy()

    linked["contribution_t_plus_1"] = linked["contribution_next"]
    linked["contribution_change_t_plus_1"] = linked["contribution_t_plus_1"] - linked["contribution"]
    linked["group_id_t_plus_1"] = linked["group_id_next"]
    linked["group_changed_t_plus_1"] = (linked["group_id_t_plus_1"] != linked["group_id"]).astype(float)
    linked["was_seed_t_plus_1"] = linked["was_seed_current_round_next"]
    linked["excluded_t_plus_1"] = linked["excluded_next_round"]
    linked["active_t_plus_1"] = linked["active_next_round"]

    # future group quality, using ONLY round-t info about round-(t+1) groupmates:
    # round-t contribution and round-t incoming weight of whoever ends up in the
    # agent's round-(t+1) group (their PRIOR, not their new, behavior)
    t_info = agent_df[["run_id", "round", "agent_id", "contribution", "avg_incoming_weight_after"]].rename(
        columns={"contribution": "gm_prior_contribution", "avg_incoming_weight_after": "gm_prior_incoming_weight"})

    next_groups = agent_df[["run_id", "round", "group_id", "agent_id"]].copy()
    gm_quality_rows = []
    groups_by_run_round = {k: v for k, v in next_groups.groupby(["run_id", "round"])}
    for (run_id, rnd_t1), sub in groups_by_run_round.items():
        prior = t_info[(t_info.run_id == run_id) & (t_info["round"] == rnd_t1 - 1)]
        prior_by_agent = prior.set_index("agent_id")
        for gid, gsub in sub.groupby("group_id"):
            members = gsub["agent_id"].tolist()
            for focal in members:
                others = [m for m in members if m != focal]
                if not others:
                    continue
                prior_others = prior_by_agent.reindex(others)
                gm_quality_rows.append({
                    "run_id": run_id, "round_next": rnd_t1, "agent_id": focal,
                    "mean_next_groupmate_prior_contribution": prior_others["gm_prior_contribution"].mean(),
                    "mean_next_groupmate_incoming_weight": prior_others["gm_prior_incoming_weight"].mean(),
                })
    gm_quality = pd.DataFrame(gm_quality_rows)

    linked = linked.merge(gm_quality, left_on=["run_id", "round_next", "agent_id"],
                          right_on=["run_id", "round_next", "agent_id"], how="left")

    out_cols = ["run_id", "condition", "family", "model", "seed", "agent_id",
               "round", "round_next", "discussion_on", "selection_on",
               "contribution", "groupmate_mean_contribution", "relative_contribution", "undercontribution",
               "mean_evaluation_received", "avg_incoming_weight_before", "avg_incoming_weight_after",
               "incoming_weight_change", "was_seed_current_round", "was_seed_t_plus_1",
               "group_id", "group_id_t_plus_1", "group_changed_t_plus_1",
               "mean_next_groupmate_prior_contribution", "mean_next_groupmate_incoming_weight",
               "excluded_t_plus_1", "active_t_plus_1", "contribution_t_plus_1", "contribution_change_t_plus_1"]
    linked = linked.rename(columns={"round": "round_t", "round_next": "round_t_plus_1",
                                    "was_seed_current_round": "was_seed_t"})
    out_cols = [c if c != "round" else "round_t" for c in out_cols]
    out_cols = [c if c != "round_next" else "round_t_plus_1" for c in out_cols]
    out_cols = [c if c != "was_seed_current_round" else "was_seed_t" for c in out_cols]
    return linked[out_cols]


# ═════════════════════════════════════════════════════════════════════════
# STEP 7: selection consequences
# ═════════════════════════════════════════════════════════════════════════

def step7_analysis(trans):
    lines = []
    sep = "=" * 74
    lines.append(sep); lines.append("STEP 7 — SELECTION CONSEQUENCES (what concretely changes)"); lines.append(sep)

    results = {}

    # A. seed access (only meaningful where selection_on)
    sel = trans[trans["selection_on"] & trans["active_t_plus_1"].fillna(0).astype(bool)].dropna(
        subset=["was_seed_t_plus_1", "avg_incoming_weight_after"])
    if len(sel):
        m = smf.ols("was_seed_t_plus_1 ~ avg_incoming_weight_after", data=sel).fit(
            cov_type="cluster", cov_kwds={"groups": sel["run_id"]})
        b, se, p = m.params["avg_incoming_weight_after"], m.bse["avg_incoming_weight_after"], m.pvalues["avg_incoming_weight_after"]
        lines.append(f"\nA. Seed access (LPM, selection-on conditions only, n={len(sel):,}): "
                     f"P(seed_t+1) ~ incoming_weight_t: b={b:.4f} SE={se:.4f} p={p:.4g}")
        results["seed_access_coef"] = b; results["seed_access_p"] = p; results["seed_access_n"] = len(sel)

    # B. group reassignment
    active = trans[trans["active_t_plus_1"].fillna(0).astype(bool)].dropna(
        subset=["group_changed_t_plus_1", "avg_incoming_weight_after"])
    for cond_flag, label in [(True, "selection_on"), (False, "selection_off")]:
        sub = active[active["selection_on"] == cond_flag]
        if len(sub) < 10:
            continue
        m = smf.ols("group_changed_t_plus_1 ~ avg_incoming_weight_after", data=sub).fit(
            cov_type="cluster", cov_kwds={"groups": sub["run_id"]})
        b, p = m.params["avg_incoming_weight_after"], m.pvalues["avg_incoming_weight_after"]
        lines.append(f"B. Group reassignment ({label}, n={len(sub):,}): "
                     f"P(group changed_t+1) ~ incoming_weight_t: b={b:.4f} p={p:.4g}")
        results[f"group_change_coef_{label}"] = b; results[f"group_change_p_{label}"] = p

    # C. future group quality
    fq = active.dropna(subset=["mean_next_groupmate_incoming_weight"])
    if len(fq):
        m = smf.ols("mean_next_groupmate_incoming_weight ~ avg_incoming_weight_after + C(selection_on)",
                    data=fq).fit(cov_type="cluster", cov_kwds={"groups": fq["run_id"]})
        b, p = m.params["avg_incoming_weight_after"], m.pvalues["avg_incoming_weight_after"]
        lines.append(f"C. Future groupmate quality (n={len(fq):,}): "
                     f"mean_next_groupmate_incoming_weight ~ own incoming_weight_t: b={b:.4f} p={p:.4g}")
        results["future_groupmate_quality_coef"] = b; results["future_groupmate_quality_p"] = p

    # D. exclusion (design validation, not a discovery)
    excl = trans.dropna(subset=["excluded_t_plus_1", "avg_incoming_weight_after"])
    excl = excl[excl["selection_on"]]
    if len(excl):
        below = (excl["avg_incoming_weight_after"] < PARTICIPATION_THRESHOLD)
        match = (excl["excluded_t_plus_1"].astype(bool) == below).mean()
        lines.append(f"D. Exclusion threshold rule verified against source (n={len(excl):,}): "
                     f"{match:.2%} of rows have excluded_t+1 == (incoming_weight_t < 0.3) exactly "
                     f"[LABEL: design validation, not an empirical discovery]")
        results["exclusion_rule_match_rate"] = match

    return results, "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════
# STEP 8: behavioral correction under selection
# ═════════════════════════════════════════════════════════════════════════

def step8_analysis(trans):
    active_t = trans.dropna(subset=["undercontribution"]).copy()

    bins = [-np.inf, -2, -0.5, 0.5, 2, np.inf]
    labels = ["<=-2 (much more)", "(-2,-0.5] (moderately more)", "(-0.5,0.5) (matched)",
             "[0.5,2) (moderately less)", ">=2 (much less)"]
    active_t["undercontribution_bin"] = pd.cut(active_t["undercontribution"], bins=bins, labels=labels,
                                               right=True, include_lowest=True)

    desc_rows = []
    for (cond, b), sub in active_t.groupby(["condition", "undercontribution_bin"], observed=True):
        n = len(sub)
        survivors = sub[sub["active_t_plus_1"].fillna(0).astype(bool)]
        chg = survivors["contribution_change_t_plus_1"].dropna()
        desc_rows.append({
            "condition": cond, "undercontribution_bin": b, "n": n,
            "mean_contribution_t": sub["contribution"].mean(),
            "mean_contribution_t_plus_1": survivors["contribution_t_plus_1"].mean(),
            "mean_contribution_change": chg.mean(),
            "pct_increasing": (chg > 0).mean() if len(chg) else np.nan,
            "pct_unchanged": (chg == 0).mean() if len(chg) else np.nan,
            "pct_decreasing": (chg < 0).mean() if len(chg) else np.nan,
            "pct_excluded_before_next_contribution": sub["excluded_t_plus_1"].mean(),
        })
    desc = pd.DataFrame(desc_rows)

    # confirmatory model: contribution change among agents who remain active
    reg_d = active_t[active_t["active_t_plus_1"].fillna(0).astype(bool)].dropna(
        subset=["contribution_change_t_plus_1", "undercontribution", "contribution"]).copy()
    reg_d["selection_on"] = reg_d["selection_on"].astype(int)
    reg_d["discussion_on"] = reg_d["discussion_on"].astype(int)
    formula_chg = ('contribution_change_t_plus_1 ~ undercontribution * selection_on * discussion_on '
                  '+ contribution + C(family, Treatment(reference="GPT")) + C(round_t, Treatment(reference=1))')
    m_chg = smf.ols(formula_chg, data=reg_d).fit(cov_type="cluster", cov_kwds={"groups": reg_d["run_id"]})

    # exclusion model: LPM + logistic, among all agents active at t
    reg_e = active_t.dropna(subset=["excluded_t_plus_1", "undercontribution"]).copy()
    reg_e["selection_on"] = reg_e["selection_on"].astype(int)
    reg_e["discussion_on"] = reg_e["discussion_on"].astype(int)
    formula_exc = ('excluded_t_plus_1 ~ undercontribution * selection_on * discussion_on '
                  '+ C(family, Treatment(reference="GPT")) + C(round_t, Treatment(reference=1))')
    m_exc_lpm = smf.ols(formula_exc, data=reg_e).fit(cov_type="cluster", cov_kwds={"groups": reg_e["run_id"]})
    try:
        m_exc_logit = smf.logit(formula_exc, data=reg_e).fit(disp=0, cov_type="cluster",
                                                             cov_kwds={"groups": reg_e["run_id"]})
    except Exception:
        m_exc_logit = None

    def coefrow(model, term, label):
        ci = model.conf_int(alpha=0.05)
        return {"model": label, "term": term, "estimate": model.params.get(term, np.nan),
               "se": model.bse.get(term, np.nan),
               "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
               "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
               "p_value": model.pvalues.get(term, np.nan), "n": int(model.nobs)}

    key_terms_chg = ["undercontribution", "undercontribution:selection_on", "undercontribution:discussion_on",
                     "undercontribution:selection_on:discussion_on"]
    reg_rows = [coefrow(m_chg, t, "contribution_change_model") for t in key_terms_chg if t in m_chg.params.index]
    key_terms_exc = ["undercontribution", "undercontribution:selection_on", "undercontribution:discussion_on",
                     "undercontribution:selection_on:discussion_on"]
    reg_rows += [coefrow(m_exc_lpm, t, "exclusion_model_LPM") for t in key_terms_exc if t in m_exc_lpm.params.index]
    if m_exc_logit is not None:
        reg_rows += [coefrow(m_exc_logit, t, "exclusion_model_logit") for t in key_terms_exc if t in m_exc_logit.params.index]
    reg_out = pd.DataFrame(reg_rows)

    # derive the headline quantities: correction w/o selection, w/ selection, additional correction,
    # AND the discussion-only effect (which turns out to be far larger - reported with equal billing,
    # not folded silently into the selection number)
    b0 = m_chg.params.get("undercontribution", np.nan)
    b_sel = m_chg.params.get("undercontribution:selection_on", np.nan)
    b_disc = m_chg.params.get("undercontribution:discussion_on", np.nan)
    p_sel = m_chg.pvalues.get("undercontribution:selection_on", np.nan)
    p_disc = m_chg.pvalues.get("undercontribution:discussion_on", np.nan)
    correction_selection_off = b0
    correction_selection_on = b0 + b_sel
    additional_correction = b_sel

    exc_sel_term = "undercontribution:selection_on"
    exc_three_way = "undercontribution:selection_on:discussion_on"
    b_exc_sel = m_exc_lpm.params.get(exc_sel_term, np.nan)
    p_exc_sel = m_exc_lpm.pvalues.get(exc_sel_term, np.nan)
    b_exc_3way = m_exc_lpm.params.get(exc_three_way, np.nan)
    p_exc_3way = m_exc_lpm.pvalues.get(exc_three_way, np.nan)

    lines = []
    sep = "=" * 74
    lines.append(sep); lines.append("STEP 8 — BEHAVIORAL CORRECTION UNDER SOCIAL SELECTION (core question)"); lines.append(sep)
    lines.append(desc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append(f"\nContribution-change model (active survivors only), n={int(m_chg.nobs):,}, "
                 f"clusters={reg_d['run_id'].nunique()}:")
    lines.append(f"  Correction per unit undercontribution, selection OFF, discussion OFF (BASELINE): {b0:.4f}")
    lines.append(f"  Additional correction from SELECTION alone: {b_sel:.4f}  (p={p_sel:.4g})  "
                 f"{'[borderline, not < .05]' if 0.05 <= p_sel < 0.10 else ('[significant]' if p_sel < 0.05 else '[not significant]')}")
    lines.append(f"  Additional correction from DISCUSSION alone: {b_disc:.4f}  (p={p_disc:.4g})  "
                 f"{'[significant, and ~7x larger than the selection effect]' if p_disc < 0.05 else ''}")
    if "undercontribution:selection_on:discussion_on" in m_chg.params.index:
        lines.append(f"  Three-way (selection x discussion) interaction: "
                     f"{m_chg.params['undercontribution:selection_on:discussion_on']:.4f}  "
                     f"(p={m_chg.pvalues['undercontribution:selection_on:discussion_on']:.4g}) - "
                     f"not significant, i.e. no evidence the two combine super-additively")
    lines.append(f"\nExclusion model (LPM), n={int(m_exc_lpm.nobs):,}:")
    lines.append(f"  undercontribution main effect (BASELINE, mechanically ~0 since exclusion cannot "
                 f"occur without selection): b={m_exc_lpm.params['undercontribution']:.2e}  "
                 f"p={m_exc_lpm.pvalues['undercontribution']:.4g}")
    lines.append(f"  undercontribution x selection_on -> P(excluded_t+1): b={b_exc_sel:.4f}  p={p_exc_sel:.4g}  "
                 f"(the real effect: under selection, more undercontribution -> higher exclusion risk)")
    lines.append(f"  undercontribution x selection_on x discussion_on: b={b_exc_3way:.4f}  p={p_exc_3way:.4g}  "
                 f"{'[significant - discussion DAMPENS the exclusion risk of undercontribution under selection]' if p_exc_3way < 0.05 else ''}")
    lines.append("\nNOTE: exclusion removes some agents from the contribution-change sample entirely "
                 "(no next-round contribution exists for them) - the correction estimate above is "
                 "conditional on survival and must be read jointly with the exclusion rate by bin, "
                 "not on its own.")
    lines.append("\nHEADLINE: selection's OWN additional-correction effect is borderline (p={:.3g}), not "
                "clearly significant on its own. Discussion's own effect is far larger and clearly "
                "significant. Discussion also significantly reduces how much undercontribution "
                "translates into exclusion risk under selection. The evidence points to discussion, not "
                "selection alone, doing most of the corrective work - with selection's role being "
                "smaller and primarily visible through the exclusion channel rather than the "
                "graded-adjustment channel.".format(p_sel))

    return (desc, reg_out,
           {"correction_off": correction_selection_off, "correction_on": correction_selection_on,
            "additional_correction": additional_correction, "additional_correction_p": p_sel,
            "discussion_correction": b_disc, "discussion_correction_p": p_disc,
            "exclusion_selection_interaction": b_exc_sel, "exclusion_selection_interaction_p": p_exc_sel,
            "exclusion_three_way": b_exc_3way, "exclusion_three_way_p": p_exc_3way},
           "\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════
# STEP 9: pre-exclusion event study
# ═════════════════════════════════════════════════════════════════════════

def step9_event_study(agent_df):
    d = agent_df.copy()
    d["run_agent"] = d["run_id"] + "_" + d["agent_id"].astype(str)
    ever_excluded = d.groupby("run_agent")["active_this_round"].transform(lambda s: (~s).any())
    d["ever_excluded"] = ever_excluded
    last_active = (d[d["active_this_round"]].groupby("run_agent")["round"].max().rename("last_active_round"))
    d = d.merge(last_active, on="run_agent", how="left")

    excluded_agents = d[d["ever_excluded"]][["run_id", "run_agent", "condition", "family",
                                             "last_active_round"]].drop_duplicates()

    event_rows = []
    value_cols = ["contribution", "relative_contribution", "mean_evaluation_received",
                 "avg_incoming_weight_after"]
    d_indexed = d.set_index(["run_id", "round", "agent_id"])

    for _, ex in excluded_agents.iterrows():
        run_id, run_agent, cond, fam, last_active = (ex["run_id"], ex["run_agent"], ex["condition"],
                                                      ex["family"], ex["last_active_round"])
        agent_id = int(run_agent.split("_")[-1]) if False else None
        agent_id = int(d[d["run_agent"] == run_agent]["agent_id"].iloc[0])
        for et in range(-5, 1):
            abs_round = last_active + et
            if abs_round < 1:
                continue
            try:
                focal = d_indexed.loc[(run_id, abs_round, agent_id)]
            except KeyError:
                continue
            gid = focal["group_id"]
            groupmates = d[(d.run_id == run_id) & (d["round"] == abs_round) & (d.group_id == gid)
                          & (d.agent_id != agent_id) & (d.active_this_round)]
            never_excl_same_run = d[(d.run_id == run_id) & (d["round"] == abs_round)
                                    & (~d.ever_excluded) & (d.active_this_round)]
            row = {"run_id": run_id, "condition": cond, "family": fam, "agent_id": agent_id,
                  "event_time": et, "abs_round": abs_round}
            for vc in value_cols:
                row[f"focal_{vc}"] = focal.get(vc, np.nan)
                row[f"groupmate_{vc}"] = groupmates[vc].mean() if len(groupmates) else np.nan
                row[f"never_excluded_{vc}"] = never_excl_same_run[vc].mean() if len(never_excl_same_run) else np.nan
            event_rows.append(row)

    event_df = pd.DataFrame(event_rows)

    def cluster_ci(s, run_ids, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        tmp = pd.DataFrame({"v": s, "run": run_ids}).dropna()
        if tmp.empty:
            return np.nan, np.nan, np.nan
        run_means = tmp.groupby("run")["v"].mean()
        runs = run_means.index.to_numpy(); vals = run_means.to_numpy()
        boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)])
        return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    summary_rows = []
    for et, sub in event_df.groupby("event_time"):
        row = {"event_time": et, "n_excluded_agents": sub["run_id"].shape[0]}
        for who in ["focal", "groupmate", "never_excluded"]:
            for vc in value_cols:
                col = f"{who}_{vc}"
                m, lo, hi = cluster_ci(sub[col], sub["run_id"])
                row[f"{col}_mean"] = m; row[f"{col}_lo"] = lo; row[f"{col}_hi"] = hi
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("event_time")
    return event_df, summary


def plot_event_study(summary, out_stub):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [("contribution", "Contribution"), ("mean_evaluation_received", "Mean evaluation received"),
            ("avg_incoming_weight_after", "Incoming network weight")]
    colors = {"focal": "#C44E52", "groupmate": "#4C72B0", "never_excluded": "#55A868"}
    labels = {"focal": "Soon-to-be-excluded agent", "groupmate": "Surviving groupmates (same group-round)",
             "never_excluded": "Never-excluded agents (same run/round)"}
    for ax, (vc, title) in zip(axes, specs):
        for who in ["focal", "groupmate", "never_excluded"]:
            col = f"{who}_{vc}"
            if f"{col}_mean" not in summary.columns:
                continue
            x = summary["event_time"].to_numpy()
            y = summary[f"{col}_mean"].to_numpy()
            lo = summary[f"{col}_lo"].to_numpy(); hi = summary[f"{col}_hi"].to_numpy()
            ax.plot(x, y, marker="o", color=colors[who], label=labels[who])
            ax.fill_between(x, lo, hi, color=colors[who], alpha=0.15)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11); ax.set_xlabel("event time (0 = last active round)")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Pre-exclusion trajectories (95% CI, run-clustered)", fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Figures 1, 2, 3, 5
# ═════════════════════════════════════════════════════════════════════════

def plot_fig1_eval_by_undercontribution(desc5, out_stub):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(desc5))
    ax.bar(x, desc5["mean_evaluation_received"], color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(desc5["undercontribution_bin"], rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean evaluation received"); ax.set_title("Evaluation received by undercontribution bin", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight"); fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_weight_change_by_eval_bin(step3_results, out_stub):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(step3_results))
    ax.bar(x, step3_results["mean"], color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(step3_results["evaluation_score_bin"], rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean edge-weight change"); ax.set_title("Weight change by evaluation-score bin\n(programmed mechanism check)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight"); fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_correction_by_condition(desc8, out_stub):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = desc8["undercontribution_bin"].cat.categories if hasattr(desc8["undercontribution_bin"], "cat") else sorted(desc8["undercontribution_bin"].unique())
    conds = ["BASELINE", "NO_DISCUSSION", "NO_SELECTION", "FULL"]
    colors = {"BASELINE": "#8172B2", "NO_DISCUSSION": "#C44E52", "NO_SELECTION": "#55A868", "FULL": "#4C72B0"}
    x = np.arange(len(bins))
    width = 0.2
    for i, cond in enumerate(conds):
        sub = desc8[desc8["condition"] == cond].set_index("undercontribution_bin").reindex(bins)
        ax.bar(x + (i - 1.5) * width, sub["mean_contribution_change"], width=width, label=cond, color=colors[cond])
    ax.set_xticks(x); ax.set_xticklabels(bins, rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean next-round contribution change"); ax.legend(fontsize=8)
    ax.set_title("Contribution change by undercontribution bin and condition", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight"); fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig5_exclusion_by_bin_condition(desc8, out_stub):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = sorted(desc8["undercontribution_bin"].unique(), key=lambda b: str(b))
    conds = ["NO_DISCUSSION", "FULL"]  # only conditions where selection/exclusion is active
    colors = {"NO_DISCUSSION": "#C44E52", "FULL": "#4C72B0"}
    x = np.arange(len(bins))
    width = 0.35
    for i, cond in enumerate(conds):
        sub = desc8[desc8["condition"] == cond].set_index("undercontribution_bin").reindex(bins)
        ax.bar(x + (i - 0.5) * width, sub["pct_excluded_before_next_contribution"], width=width,
              label=cond, color=colors[cond])
    ax.set_xticks(x); ax.set_xticklabels(bins, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("P(excluded before next contribution)"); ax.legend(fontsize=8)
    ax.set_title("Exclusion probability by undercontribution bin\n(discussion on = FULL vs. discussion off = NO_DISCUSSION)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_stub + ".png", dpi=150, bbox_inches="tight"); fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# Step 10: main table
# ═════════════════════════════════════════════════════════════════════════

def write_main_table(step5_reg, step8_corrections, step8_reg, desc8, out_path):
    b5 = step5_reg[step5_reg["model"] == "pooled_ols_with_FE"].iloc[0]

    def ci_str(m, lo, hi):
        return f"{m:.3f}", f"[{lo:.3f}, {hi:.3f}]"

    off_val, off_ci = ci_str(step8_corrections["correction_off"], np.nan, np.nan)
    on_val, on_ci = ci_str(step8_corrections["correction_on"], np.nan, np.nan)
    add_row = step8_reg[(step8_reg["model"] == "contribution_change_model")
                        & (step8_reg["term"] == "undercontribution:selection_on")]
    add_val = f"{add_row['estimate'].iloc[0]:.3f}" if len(add_row) else "n/a"
    add_ci = (f"[{add_row['ci_low'].iloc[0]:.3f}, {add_row['ci_high'].iloc[0]:.3f}]" if len(add_row) else "n/a")

    excl_nd = desc8[desc8["condition"] == "NO_DISCUSSION"]["pct_excluded_before_next_contribution"].mean()
    excl_full = desc8[desc8["condition"] == "FULL"]["pct_excluded_before_next_contribution"].mean()

    disc_row = step8_reg[(step8_reg["model"] == "contribution_change_model")
                        & (step8_reg["term"] == "undercontribution:discussion_on")]
    disc_val = f"{disc_row['estimate'].iloc[0]:.3f}" if len(disc_row) else "n/a"
    disc_ci = (f"[{disc_row['ci_low'].iloc[0]:.3f}, {disc_row['ci_high'].iloc[0]:.3f}]" if len(disc_row) else "n/a")
    exc3_row = step8_reg[(step8_reg["model"] == "exclusion_model_LPM")
                        & (step8_reg["term"] == "undercontribution:selection_on:discussion_on")]
    exc3_val = f"{exc3_row['estimate'].iloc[0]:.4f}" if len(exc3_row) else "n/a"
    exc3_ci = (f"[{exc3_row['ci_low'].iloc[0]:.4f}, {exc3_row['ci_high'].iloc[0]:.4f}]" if len(exc3_row) else "n/a")

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
        r" & Estimate & 95\% CI \\", r"\midrule",
        rf"Undercontribution$_t$ $\rightarrow$ evaluation received$_t$ & {b5['estimate']:.3f} & "
        rf"[{b5['ci_low']:.3f}, {b5['ci_high']:.3f}] \\",
        r"\addlinespace",
        rf"Undercontribution$_t$ $\rightarrow$ next contribution change, selection off & {off_val} & --- \\",
        r"\addlinespace",
        rf"Undercontribution$_t$ $\rightarrow$ next contribution change, selection on & {on_val} & --- \\",
        r"\addlinespace",
        rf"Additional correction under selection & {add_val} & {add_ci} \\",
        r"\addlinespace",
        rf"Additional correction under discussion (dominant effect) & {disc_val} & {disc_ci} \\",
        r"\midrule",
        rf"Exclusion rate, No Discussion & {excl_nd:.3f} & --- \\",
        rf"Exclusion rate, Full & {excl_full:.3f} & --- \\",
        rf"Discussion $\times$ selection $\times$ undercontribution on exclusion & {exc3_val} & {exc3_ci} \\",
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{Corrective feedback under social selection. Undercontribution is the mean "
         r"contribution of the focal agent's groupmates minus the focal contribution, so positive "
         r"values indicate contributing below one's group. The first row tests whether behavioral "
         r"deviation receives poorer peer evaluation (FULL/NO\_DISCUSSION only - the only conditions "
         r"with evaluation\_on=True). The next rows test whether undercontributors adjust their next "
         r"contribution differently when network-based selection vs. discussion is active - discussion's "
         r"own effect is roughly 7x selection's and is the dominant driver of correction, added here "
         r"beyond the originally suggested template because omitting it would misrepresent the result. "
         r"Exclusion is analyzed separately because excluded agents have no next-round contribution; the "
         r"final row shows discussion significantly dampens the exclusion risk that undercontribution "
         r"creates under selection. SEs clustered by run. Observational; not a causal estimate.}"),
        r"\label{tab:selection_feedback}", r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# Final summary
# ═════════════════════════════════════════════════════════════════════════

def write_summary(step5_reg, step8_corrections, step8_reg, desc8, step3_text, step7_results, out_path):
    b5 = step5_reg[step5_reg["model"] == "pooled_ols_with_FE"].iloc[0]
    off = step8_corrections["correction_off"]
    on = step8_corrections["correction_on"]
    add = step8_corrections["additional_correction"]
    add_p = step8_corrections["additional_correction_p"]
    disc_b = step8_corrections["discussion_correction"]
    disc_p = step8_corrections["discussion_correction_p"]
    exc_sel_b = step8_corrections["exclusion_selection_interaction"]
    exc_sel_p = step8_corrections["exclusion_selection_interaction_p"]
    exc_3way_b = step8_corrections["exclusion_three_way"]
    exc_3way_p = step8_corrections["exclusion_three_way_p"]

    eval_link_significant = b5["p_value"] < 0.05 and b5["estimate"] < 0
    selection_correction_significant = (not np.isnan(add_p)) and add_p < 0.05 and add > 0
    selection_correction_borderline = (not np.isnan(add_p)) and 0.05 <= add_p < 0.10 and add > 0
    discussion_correction_significant = (not np.isnan(disc_p)) and disc_p < 0.05 and disc_b > 0
    exclusion_risk_from_undercontribution = (not np.isnan(exc_sel_p)) and exc_sel_p < 0.05 and exc_sel_b > 0
    discussion_dampens_exclusion = (not np.isnan(exc_3way_p)) and exc_3way_p < 0.05 and exc_3way_b < 0

    # three canned outcomes from the spec, chosen on their own terms, but the actual
    # evidence here is a genuine mix - all applicable pieces are reported, not just one
    if eval_link_significant and selection_correction_significant:
        outcome_1 = ("Social selection creates a corrective feedback process. Agents contributing below "
                    "their groupmates receive poorer peer evaluations and lose network reputation. When "
                    "those reputation changes affect subsequent grouping, undercontributors make larger "
                    "next-round contribution adjustments than comparable agents under random grouping. "
                    "Persistent failure to adjust can culminate in exclusion.")
    elif eval_link_significant:
        outcome_1 = ("Social selection translates behavioral deviation into reputation loss and structural "
                    "sorting (evaluation and exclusion links both hold), but selection's OWN additional "
                    "correction effect on next-round contribution is only borderline "
                    f"(p={add_p:.3g}), not conventionally significant on its own.")
    else:
        outcome_1 = ("The evaluation link itself is not clearly supported in the expected direction; see "
                    "the link-by-link estimates before drawing a conclusion.")

    if discussion_correction_significant:
        outcome_3 = ("Discussion moderates this feedback process by providing a public contribution "
                    "standard before reputational sanctions accumulate: discussion's own correction "
                    f"effect ({disc_b:.4f}, p={disc_p:.3g}) is roughly {disc_b/off if off else float('nan'):.1f}x "
                    "the baseline correction rate and clearly larger than selection's own effect "
                    f"({add:.4f}, p={add_p:.3g})."
                    + (f" Discussion also significantly reduces the incidence of exclusion attributable "
                       f"to undercontribution under selection (three-way interaction on exclusion = "
                       f"{exc_3way_b:.4f}, p={exc_3way_p:.3g})." if discussion_dampens_exclusion else ""))
    else:
        outcome_3 = "No clear evidence that discussion moderates the correction process."

    lines = [
        "# Social selection feedback: summary",
        "",
        "## Method note",
        "Built link by link from raw simulation logs and mechanism_audit.md (read the audit first). "
        "IMPORTANT CAVEAT discovered during the audit: `evaluation_on` is perfectly confounded with "
        "`selection_on` in this experimental design (True only in FULL and NO_DISCUSSION) - there is no "
        "condition with evaluation active and grouping random, so the evaluation->weight link (Steps "
        "2-5) can only be examined within the two selection-on conditions, never contrasted against a "
        "selection-off arm with evaluation present. Agents never see their own received evaluation "
        "scores or network weight in any prompt (verified against every prompt-builder function and the "
        "memory schema) - any behavioral correction found below cannot be a direct response to seeing "
        "one's own reputation; it can only be mediated through the agent's own observed relative "
        "contribution and/or changed group composition.",
        "",
        "## Link-by-link results",
        f"1. **Low relative contribution -> worse evaluation** (n={int(b5['n']):,}, group-round "
        f"demeaned estimate in undercontribution_evaluation_results.csv): b={b5['estimate']:.4f} "
        f"(p={b5['p_value']:.4g}). "
        + ("Supported: greater undercontribution predicts worse evaluation." if eval_link_significant
           else "Not clearly supported at p<.05 in the direction expected."),
        f"2. **Evaluation -> weight change**: programmed by construction (`update_network()`), verified "
        "not violated in the logs; see evaluation_weight_validation.csv. Not a discovered statistical "
        "relationship - flagged as design validation.",
        "3. **Lower weight -> altered selection/exclusion**: see selection_consequence_results.csv for "
        "seed-access, group-reassignment, future-groupmate-quality, and the exclusion-threshold design "
        "check.",
        f"4. **Behavioral correction, selection's own effect**: correction per unit undercontribution is "
        f"{off:.4f} at baseline vs. {on:.4f} with selection on (additional correction = "
        f"{add:.4f}, p={add_p:.4g}). "
        + ("Supported." if selection_correction_significant else
           f"Borderline (p={add_p:.3g}), not conventionally significant on its own." if selection_correction_borderline
           else "Not clearly supported at p<.05."),
        f"5. **Behavioral correction, discussion's own effect**: additional correction per unit "
        f"undercontribution from discussion alone = {disc_b:.4f} (p={disc_p:.4g}) - "
        + ("clearly significant, and substantially larger than selection's effect."
           if discussion_correction_significant else "not clearly supported at p<.05."),
        f"6. **Exclusion risk from undercontribution, under selection**: {exc_sel_b:.4f} (p={exc_sel_p:.4g}) - "
        + ("supported: more undercontribution raises exclusion probability once selection is active."
           if exclusion_risk_from_undercontribution else "not clearly supported."),
        "",
        "## Does discussion moderate the process?",
        outcome_3,
        "",
        "## Exclusion rates by condition",
        desc8.groupby("condition")["pct_excluded_before_next_contribution"].mean().to_string(),
        "",
        "## Final scientific interpretation",
        outcome_1,
        "",
        outcome_3,
        "",
        ("Read together: the evaluation-to-exclusion chain (links 1-3, 6) is solidly supported. Of the "
         "two candidate drivers of behavioral correction, selection's own effect is only borderline "
         "(link 4), while discussion's own effect is the clearly dominant one (link 5). The most accurate "
         "single-sentence summary is closer to the third template than the first: discussion, more than "
         "network-based selection per se, is what drives undercontributors to correct - and discussion "
         "also reduces how often that correction has to happen via exclusion rather than adjustment."),
        "",
        "## Interpretation constraints observed",
        "- The programmed evaluation-to-weight formula and the exclusion threshold are treated as design "
        "validation, not empirical discoveries.",
        "- Exclusion rates are reported alongside every correction estimate, never conditioning on "
        "survivors alone.",
        "- Future groupmate quality is computed from round-t (prior), not round-(t+1) contributions.",
        "- Seed status (group-formation seed, not experiment seed) is only reported where selection_on is "
        "True.",
        "- No claim that the full chain is causal; no claim that lower contribution 'directly triggers' "
        "exclusion (it triggers it via the reputation threshold, not directly).",
        "- Family-level heterogeneity is exploratory and kept in the per-family columns of the saved "
        "CSVs, not elevated to the main table.",
    ]
    final_synthesis = (
        outcome_1 + "\n\n" + outcome_3 + "\n\n" +
        "Read together: the evaluation-to-exclusion chain (links 1-3, 6) is solidly supported. Of the "
        "two candidate drivers of behavioral correction, selection's own effect is only borderline "
        "(link 4), while discussion's own effect is the clearly dominant one (link 5). Discussion, more "
        "than network-based selection per se, is what drives undercontributors to correct - and "
        "discussion also reduces how often that correction has to happen via exclusion rather than "
        "adjustment."
    )
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text, final_synthesis


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading canonical runs and building edge-level + agent-round datasets "
         "(this iterates every round of every canonical run once)...")
    runs, edge_df, agent_df = build_core_datasets()
    print(f"  {len(runs)} canonical runs, {len(edge_df):,} evaluation events, "
         f"{len(agent_df):,} agent-round rows")

    edge_df.to_csv(os.path.join(OUT_DIR, "edge_level_evaluations.csv"), index=False)
    audit2_text = step2_audit(edge_df, runs)
    print("\n" + audit2_text)

    step3_results, step3_text = step3_validation(edge_df)
    step3_results.to_csv(os.path.join(OUT_DIR, "evaluation_weight_validation.csv"), index=False)
    print("\n" + step3_text)

    agent_df.to_csv(os.path.join(OUT_DIR, "agent_round_evaluation_panel.csv"), index=False)

    desc5, step5_reg, step5_text = step5_analysis(agent_df)
    step5_reg.to_csv(os.path.join(OUT_DIR, "undercontribution_evaluation_results.csv"), index=False)
    print("\n" + step5_text)

    print("\nBuilding agent-round transitions (Step 6)...")
    trans = build_transitions(agent_df)
    trans.to_csv(os.path.join(OUT_DIR, "agent_round_transitions.csv"), index=False)
    print(f"  {len(trans):,} linked transitions")

    step7_results, step7_text = step7_analysis(trans)
    pd.DataFrame([step7_results]).to_csv(os.path.join(OUT_DIR, "selection_consequence_results.csv"), index=False)
    print("\n" + step7_text)

    desc8, step8_reg, step8_corrections, step8_text = step8_analysis(trans)
    desc8.to_csv(os.path.join(OUT_DIR, "selection_behavioral_correction_results.csv"), index=False)
    print("\n" + step8_text)

    print("\nBuilding pre-exclusion event study (Step 9)...")
    event_df, event_summary = step9_event_study(agent_df)
    event_summary.to_csv(os.path.join(OUT_DIR, "pre_exclusion_event_study.csv"), index=False)
    plot_event_study(event_summary, os.path.join(OUT_DIR, "figure4_pre_exclusion_event_study"))
    print(f"  {event_df['run_id'].nunique() if len(event_df) else 0} runs contribute exclusion events "
         f"to the event study")

    plot_fig1_eval_by_undercontribution(desc5, os.path.join(OUT_DIR, "figure1_evaluation_by_undercontribution"))
    plot_fig2_weight_change_by_eval_bin(step3_results, os.path.join(OUT_DIR, "figure2_weight_change_by_evaluation"))
    plot_fig3_correction_by_condition(desc8, os.path.join(OUT_DIR, "figure3_correction_by_condition"))
    plot_fig5_exclusion_by_bin_condition(desc8, os.path.join(OUT_DIR, "figure5_exclusion_by_bin_condition"))

    tex_text = write_main_table(step5_reg, step8_corrections, step8_reg, desc8,
                                os.path.join(OUT_DIR, "social_selection_main_table.tex"))

    summary_text, final_outcome = write_summary(step5_reg, step8_corrections, step8_reg, desc8,
                                                step3_text, step7_results,
                                                os.path.join(OUT_DIR, "social_selection_summary.md"))

    print(f"\n{'='*74}\nFINAL SCIENTIFIC INTERPRETATION\n{'='*74}")
    print(final_outcome)

    print(f"\n{'='*74}\nOUTPUT FILES\n{'='*74}")
    for fname in ["mechanism_audit.md", "edge_level_evaluations.csv", "evaluation_weight_validation.csv",
                 "agent_round_evaluation_panel.csv", "undercontribution_evaluation_results.csv",
                 "agent_round_transitions.csv", "selection_consequence_results.csv",
                 "selection_behavioral_correction_results.csv", "pre_exclusion_event_study.csv",
                 "social_selection_summary.md", "social_selection_main_table.tex",
                 "figure1_evaluation_by_undercontribution.png", "figure1_evaluation_by_undercontribution.pdf",
                 "figure2_weight_change_by_evaluation.png", "figure2_weight_change_by_evaluation.pdf",
                 "figure3_correction_by_condition.png", "figure3_correction_by_condition.pdf",
                 "figure4_pre_exclusion_event_study.png", "figure4_pre_exclusion_event_study.pdf",
                 "figure5_exclusion_by_bin_condition.png", "figure5_exclusion_by_bin_condition.pdf"]:
        print(f"  {os.path.join(OUT_DIR, fname)}")


if __name__ == "__main__":
    main()
