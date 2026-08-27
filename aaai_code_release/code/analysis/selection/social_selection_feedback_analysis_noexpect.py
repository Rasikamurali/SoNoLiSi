"""
social_selection_feedback_analysis_noexpect.py
------------------------------------------------
Per-link test of the feedback chain

    contribution_t -> evaluation_t -> network update_t
    -> selection/exclusion consequences_t+1 -> contribution adjustment_t+1

The condition here is where the chain's mechanics are cleanest to read because there is no discussion channel muddying the
evaluation/exclusion signal.

Each link is its own OLS model, standard errors clustered by run
(run_id = family_seed, one run per model x seed since only one condition is
in scope). This is a lighter, per-link version of the original
social_selection_feedback_analysis.py (which pools E/E+SS/
E+SL/E+SL+SS into single models with selection_on x discussion_on
interaction terms) — that pooled design doesn't apply here because the
no-expect arm only has one condition with evaluation_on=True and one with
discussion_on=True; there's nothing to interact against within-arm.


Output: exports/social_selection_feedback_noexpect/
"""

import os
import json
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/selection/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Anchored to this file's own location (not cwd) so this resolves correctly
# regardless of the caller's working directory.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports", "social_selection_feedback_noexpect")

MODELS = [("llama", "Llama-7B"), ("mistral", "Mistral-7B"), ("qwen", "Qwen-7B")]
SEEDS = list(range(42, 52))   # matched across both arms
GROUP_SIZE = 4
N_AGENTS = 12
PARTICIPATION_THRESHOLD = 0.3

# Each arm: (label, results_dir, variant, condition, discussion_on, selection_on)
ARMS = {
    "noexpect": {
        "label": "SELECTION_ONLY_NO_EXPECT (no social learning, no expectations)",
        "results_dir": f"{BASE}/code/results",
        "variant": "local_noexpect",
        "condition": "SELECTION_ONLY_NO_EXPECT",
        "discussion_on": False,
        "selection_on": True,
    },
    "expect": {
        "label": "NO_DISCUSSION (matched with-expectation condition, main paper)",
        "results_dir": f"{BASE}/results",
        "variant": "local",
        "condition": "NO_DISCUSSION",
        "discussion_on": False,
        "selection_on": True,
    },
}


# ═════════════════════════════════════════════════════════════════════════
# Load + process runs (single condition per arm, by construction)
# ═════════════════════════════════════════════════════════════════════════

def load_runs(arm_key):
    arm = ARMS[arm_key]
    runs = []
    for model_key, family in MODELS:
        for seed in SEEDS:
            pattern = os.path.join(arm["results_dir"], model_key, arm["variant"],
                                    f"seed{seed}", "log_*.json")
            best = None
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                except Exception:
                    continue
                if d.get("condition") == arm["condition"]:
                    best = d
            if best is None:
                continue
            run_id = f"{family}_s{seed}_{arm['condition']}"
            runs.append({"run_id": run_id, "family": family, "model": model_key,
                         "seed": seed, "d": best})
    return runs


def initial_network():
    return {(i, j): 1.0 for i in range(N_AGENTS) for j in range(N_AGENTS) if i != j}


def process_run(run, discussion_on, selection_on):
    d = run["d"]
    round_logs = d["round_logs"]
    prev_net = initial_network()
    edge_rows, agent_round_rows = [], []

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
                seed_of_group[gi] = g[0]

        cur_net = {(e["u"], e["v"]): e["weight"] for e in r["network_weights"]}

        def incoming_avg(net, target):
            return float(np.mean([net.get((k, target), 0.0) for k in range(N_AGENTS) if k != target]))

        evals_raw = r.get("evaluations") or {}
        next_excluded = None
        if idx + 1 < len(round_logs) and round_logs[idx + 1]["round"] == rnd + 1:
            next_excluded = set(int(x) for x in round_logs[idx + 1]["excluded_agents"])

        for key, score in evals_raw.items():
            i_str, j_str = key.split("->")
            i, j = int(i_str), int(j_str)
            target_others = [m for m in other_members.get(j, []) if m != j]
            gm_mean_for_target = (np.mean([contributions.get(m, np.nan) for m in target_others])
                                  if target_others else np.nan)
            before = prev_net.get((i, j), 0.0)
            after = cur_net.get((i, j), 0.0)
            edge_rows.append({
                "run_id": run["run_id"], "family": run["family"], "model": run["model"],
                "seed": run["seed"], "round": rnd,
                "target_relative_contribution": (contributions.get(j, np.nan) - gm_mean_for_target
                                                 if target_others else np.nan),
                "evaluation_score": score,
                "edge_weight_before": before, "edge_weight_after": after,
                "edge_weight_change": after - before,
            })

        for aid in range(N_AGENTS):
            active = aid not in excluded_ids and aid in group_of
            gid = group_of.get(aid)
            others = other_members.get(aid, [])
            gm_mean = np.mean([contributions.get(m, np.nan) for m in others]) if others else np.nan
            contrib = contributions.get(aid, np.nan)

            recv = [row["evaluation_score"] for row in edge_rows
                   if row["round"] == rnd and row.get("_target_id") == aid]
            # (target_id not stored above to keep edge_rows lean; recompute directly)
            recv = [float(sc) for key, sc in evals_raw.items()
                   if int(key.split("->")[1]) == aid]
            was_seed = (float(seed_of_group.get(gid) == aid) if (selection_on and gid is not None)
                       else np.nan)

            active_next = np.nan
            excluded_next_flag = np.nan
            if next_excluded is not None:
                excluded_next_flag = float(aid in next_excluded)
                active_next = float(aid not in next_excluded)

            agent_round_rows.append({
                "run_id": run["run_id"], "family": run["family"], "model": run["model"],
                "seed": run["seed"], "round": rnd, "agent_id": aid, "group_id": gid,
                "active_this_round": active,
                "contribution": contrib if active else np.nan,
                "relative_contribution": (contrib - gm_mean) if (active and others) else np.nan,
                "undercontribution": (gm_mean - contrib) if (active and others) else np.nan,
                "mean_evaluation_received": np.mean(recv) if recv else np.nan,
                "avg_incoming_weight_before": incoming_avg(prev_net, aid),
                "avg_incoming_weight_after": incoming_avg(cur_net, aid),
                "was_seed_current_round": was_seed,
                "excluded_next_round": excluded_next_flag,
                "active_next_round": active_next,
            })

        prev_net = cur_net

    return edge_rows, agent_round_rows


def build_core_datasets(arm_key):
    arm = ARMS[arm_key]
    runs = load_runs(arm_key)
    all_edge, all_agent = [], []
    for run in runs:
        e, a = process_run(run, arm["discussion_on"], arm["selection_on"])
        all_edge.extend(e)
        all_agent.extend(a)
    return runs, pd.DataFrame(all_edge), pd.DataFrame(all_agent)


def build_transitions(agent_df):
    d = agent_df.sort_values(["run_id", "agent_id", "round"]).copy()
    grp = d.groupby(["run_id", "agent_id"], sort=False)
    d["round_next"] = grp["round"].shift(-1)
    for col in ["contribution", "group_id", "was_seed_current_round"]:
        d[col + "_next"] = grp[col].shift(-1)

    has_next = d["round_next"].notna()
    consecutive = d["round_next"] == d["round"] + 1
    linked = d[has_next & consecutive].copy()

    linked["contribution_t_plus_1"] = linked["contribution_next"]
    linked["contribution_change_t_plus_1"] = linked["contribution_t_plus_1"] - linked["contribution"]
    linked["group_changed_t_plus_1"] = (linked["group_id_next"] != linked["group_id"]).astype(float)
    linked["was_seed_t_plus_1"] = linked["was_seed_current_round_next"]
    linked = linked.rename(columns={"round": "round_t", "was_seed_current_round": "was_seed_t"})
    return linked


# ═════════════════════════════════════════════════════════════════════════
# Per-link OLS models (identical formula shape across arms)
# ═════════════════════════════════════════════════════════════════════════

def coefrow(link, model, term, label):
    ci = model.conf_int(alpha=0.05)
    return {"link": link, "arm": label, "term": term,
           "estimate": model.params.get(term, np.nan), "se": model.bse.get(term, np.nan),
           "ci_low": ci.loc[term, 0] if term in ci.index else np.nan,
           "ci_high": ci.loc[term, 1] if term in ci.index else np.nan,
           "p_value": model.pvalues.get(term, np.nan),
           "n": int(model.nobs), "n_clusters": model.model.data.orig_exog is not None and None}


def run_links(arm_key):
    arm = ARMS[arm_key]
    label = arm_key
    runs, edge_df, agent_df = build_core_datasets(arm_key)
    trans = build_transitions(agent_df)
    rows = []

    # Link 1: contribution -> evaluation received
    d1 = agent_df[agent_df["active_this_round"] & agent_df["undercontribution"].notna()
                 & agent_df["mean_evaluation_received"].notna()].copy()
    if len(d1) >= 30:
        f1 = ('mean_evaluation_received ~ undercontribution '
              '+ C(family, Treatment(reference="Llama-7B")) + C(round, Treatment(reference=1))')
        m1 = smf.ols(f1, data=d1).fit(cov_type="cluster", cov_kwds={"groups": d1["run_id"]})
        r = coefrow("1_contribution_to_evaluation", m1, "undercontribution", label)
        r["n_clusters"] = d1["run_id"].nunique()
        rows.append(r)

    # Link 2: evaluation -> network (edge weight change) [mechanism check]
    d2 = edge_df.dropna(subset=["evaluation_score", "edge_weight_change"]).copy()
    if len(d2) >= 30:
        f2 = "edge_weight_change ~ evaluation_score"
        m2 = smf.ols(f2, data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2["run_id"]})
        r = coefrow("2_evaluation_to_network_update", m2, "evaluation_score", label)
        r["n_clusters"] = d2["run_id"].nunique()
        rows.append(r)

    # Link 3a: network update -> selection consequences (seed access)
    active_next = trans[trans["active_next_round"].fillna(0).astype(bool)]
    d3a = active_next.dropna(subset=["was_seed_t_plus_1", "avg_incoming_weight_after"])
    if len(d3a) >= 30:
        f3a = "was_seed_t_plus_1 ~ avg_incoming_weight_after"
        m3a = smf.ols(f3a, data=d3a).fit(cov_type="cluster", cov_kwds={"groups": d3a["run_id"]})
        r = coefrow("3a_network_to_seed_access", m3a, "avg_incoming_weight_after", label)
        r["n_clusters"] = d3a["run_id"].nunique()
        rows.append(r)

    # Link 3b: network update -> group reassignment
    d3b = active_next.dropna(subset=["group_changed_t_plus_1", "avg_incoming_weight_after"])
    if len(d3b) >= 30:
        f3b = "group_changed_t_plus_1 ~ avg_incoming_weight_after"
        m3b = smf.ols(f3b, data=d3b).fit(cov_type="cluster", cov_kwds={"groups": d3b["run_id"]})
        r = coefrow("3b_network_to_group_reassignment", m3b, "avg_incoming_weight_after", label)
        r["n_clusters"] = d3b["run_id"].nunique()
        rows.append(r)

    # Link 4: contribution adjustment (undercontribution_t -> contribution_change_t+1, survivors only)
    d4 = trans[trans["active_next_round"].fillna(0).astype(bool)].dropna(
        subset=["contribution_change_t_plus_1", "undercontribution", "contribution"]).copy()
    if len(d4) >= 30:
        f4 = ('contribution_change_t_plus_1 ~ undercontribution + contribution '
              '+ C(family, Treatment(reference="Llama-7B")) + C(round_t, Treatment(reference=1))')
        m4 = smf.ols(f4, data=d4).fit(cov_type="cluster", cov_kwds={"groups": d4["run_id"]})
        r = coefrow("4_contribution_adjustment", m4, "undercontribution", label)
        r["n_clusters"] = d4["run_id"].nunique()
        rows.append(r)

    # Link 5: exclusion (undercontribution_t -> excluded_t+1)
    d5 = trans.dropna(subset=["undercontribution", "contribution"]).copy()
    d5["excluded_t_plus_1"] = agent_df.set_index(["run_id", "agent_id", "round"]).reindex(
        pd.MultiIndex.from_arrays([d5["run_id"], d5["agent_id"], d5["round_t"]])
    )["excluded_next_round"].values
    d5 = d5.dropna(subset=["excluded_t_plus_1"])
    if len(d5) >= 30:
        f5 = ('excluded_t_plus_1 ~ undercontribution '
              '+ C(family, Treatment(reference="Llama-7B")) + C(round_t, Treatment(reference=1))')
        m5 = smf.ols(f5, data=d5).fit(cov_type="cluster", cov_kwds={"groups": d5["run_id"]})
        r = coefrow("5_exclusion", m5, "undercontribution", label)
        r["n_clusters"] = d5["run_id"].nunique()
        rows.append(r)

    return pd.DataFrame(rows), runs, edge_df, agent_df, trans


LINK_LABELS = {
    "1_contribution_to_evaluation": "Undercontribution -> evaluation received",
    "2_evaluation_to_network_update": "Evaluation score -> edge-weight change",
    "3a_network_to_seed_access": "Incoming weight -> P(group seed, t+1)",
    "3b_network_to_group_reassignment": "Incoming weight -> P(group changed, t+1)",
    "4_contribution_adjustment": "Undercontribution -> next contribution change",
    "5_exclusion": "Undercontribution -> P(excluded, t+1)",
}
LINK_ORDER = list(LINK_LABELS.keys())


def write_comparison_table(noexpect_rows, expect_rows, out_path):
    ne = noexpect_rows.set_index("link")
    ex = expect_rows.set_index("link")
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Link & $\hat{\beta}$ (No Expect) & $p$ & $\hat{\beta}$ (Expect, matched) & $p$ \\",
        r"\midrule",
    ]
    for link in LINK_ORDER:
        label = LINK_LABELS[link]
        ne_str = ex_str = r"--- & ---"
        if link in ne.index:
            row = ne.loc[link]
            ne_str = rf"{row['estimate']:.4f} & {row['p_value']:.3g}"
        else:
            ne_str = r"--- & ---"
        if link in ex.index:
            row = ex.loc[link]
            ex_str = rf"{row['estimate']:.4f} & {row['p_value']:.3g}"
        else:
            ex_str = r"--- & ---"
        lines.append(rf"{label} & {ne_str} & {ex_str} \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{Per-link OLS estimates of the contribution$\to$evaluation$\to$network update"
         r"$\to$selection consequences$\to$contribution adjustment chain, SELECTION\_ONLY\_NO\_EXPECT"
         r" vs.\ the matched with-expectation condition (NO\_DISCUSSION), same 3 models "
         r"(Llama/Mistral/Qwen-7B) and same seed range (42--51). SEs clustered by run "
         r"(family $\times$ seed). Identical formula per link across arms.}"),
        r"\label{tab:selection_feedback_noexpect}", r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def write_summary_md(noexpect_rows, expect_rows, out_path):
    ne = noexpect_rows.set_index("link")
    ex = expect_rows.set_index("link")
    lines = ["# Feedback-chain per-link comparison: SELECTION_ONLY_NO_EXPECT vs. NO_DISCUSSION", ""]
    lines.append("Same per-link OLS formulas fit separately in each arm (not a pooled/interaction "
                 "model like the main-paper social_selection_feedback_analysis.py — there's only one "
                 "evaluation-on, selection-on condition per arm here, so there's nothing to interact "
                 "against within-arm). SEs clustered by run (family x seed). Models: Llama/Mistral/"
                 "Qwen-7B. Seeds: 42-51, matched across arms.\n")
    for link in LINK_ORDER:
        lines.append(f"## {LINK_LABELS[link]}")
        if link in ne.index:
            r = ne.loc[link]
            lines.append(f"- No-expect:  b={r['estimate']:.4f}  SE={r['se']:.4f}  "
                         f"p={r['p_value']:.4g}  n={int(r['n']):,}  clusters={int(r['n_clusters'])}")
        else:
            lines.append("- No-expect:  insufficient data")
        if link in ex.index:
            r = ex.loc[link]
            lines.append(f"- Expect (matched NO_DISCUSSION):  b={r['estimate']:.4f}  SE={r['se']:.4f}  "
                         f"p={r['p_value']:.4g}  n={int(r['n']):,}  clusters={int(r['n_clusters'])}")
        else:
            lines.append("- Expect: insufficient data")
        lines.append("")
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 74)
    print("NO-EXPECT ARM: SELECTION_ONLY_NO_EXPECT")
    print("=" * 74)
    ne_rows, ne_runs, ne_edge, ne_agent, ne_trans = run_links("noexpect")
    print(f"  Runs: {len(ne_runs)}  Edge rows: {len(ne_edge):,}  Agent-round rows: {len(ne_agent):,}")
    print(ne_rows[["link", "estimate", "se", "p_value", "n", "n_clusters"]].to_string(index=False))

    print("\n" + "=" * 74)
    print("MATCHED EXPECT ARM: NO_DISCUSSION")
    print("=" * 74)
    ex_rows, ex_runs, ex_edge, ex_agent, ex_trans = run_links("expect")
    print(f"  Runs: {len(ex_runs)}  Edge rows: {len(ex_edge):,}  Agent-round rows: {len(ex_agent):,}")
    print(ex_rows[["link", "estimate", "se", "p_value", "n", "n_clusters"]].to_string(index=False))

    ne_rows.to_csv(os.path.join(OUT_DIR, "links_noexpect.csv"), index=False)
    ex_rows.to_csv(os.path.join(OUT_DIR, "links_expect_matched.csv"), index=False)

    tex = write_comparison_table(ne_rows, ex_rows, os.path.join(OUT_DIR, "feedback_chain_comparison.tex"))
    md = write_summary_md(ne_rows, ex_rows, os.path.join(OUT_DIR, "feedback_chain_comparison_summary.md"))

    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
