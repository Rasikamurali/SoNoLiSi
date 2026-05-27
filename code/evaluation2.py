# ============================================================
# evaluation2.py — Deep analysis of SoNoLiSi simulation logs
#
# Analyses (all saved as PNGs):
#   1. Payoff distribution by role over rounds
#   2. Ostracism tracking — incoming weight trajectories for
#      violators vs cooperators; fraction of near-zero edges
#   3. Network structure — degree/strength distributions,
#      clustering, in-strength by role over time
#   4. Condition comparison — cumulative payoff, mean contribution,
#      cooperation rate, mean edge weight across conditions
#   5. Norm emergence — inferred_norm and agents_to_avoid
#      sentiment over rounds; contribution variance over time
#
# Usage (single file):
#   python evaluation2.py log_v5_*_FULL_seed42.json
#
# Usage (multi-file / condition comparison):
#   python evaluation2.py log_v5_*_FULL_seed42.json \
#                         log_v5_*_NO_DISCUSSION_seed42.json \
#                         log_v5_*_NO_SELECTION_seed42.json \
#                         --out-dir ./plots
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np


# ============================================================
# Shared helpers
# ============================================================

ROLE_COLORS = {"violator": "#d62728", "cooperator": "#1f77b4"}
COND_COLORS  = {"FULL": "#2ca02c", "NO_DISCUSSION": "#9467bd", "NO_SELECTION": "#8c564b"}
OSTRACISM_THRESHOLD = 0.2   # edge weight below this → "near-ostracism"


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def role_map(data: dict) -> Dict[int, str]:
    """
    Returns {agent_id: "violator"|"cooperator"}.
    Supports logs with a "role" field or a "cooperation_tendency" field.
    For tendency-based logs, uses initial tendency from round 1 agent_states:
    tendency < 0.3 → violator, else → cooperator.
    """
    final_agents = data["final_agents"]
    if "role" in final_agents[0]:
        return {a["id"]: a["role"] for a in final_agents}
    init_states = data["round_logs"][0]["agent_states"]
    return {
        s["id"]: ("violator" if s["cooperation_tendency"] < 0.3 else "cooperator")
        for s in init_states
    }


def int_id(x) -> int:
    return int(x)


# ============================================================
# 1. Payoff distribution
# ============================================================

def plot_payoff_distribution(data: dict, condition: str, out_dir: str, base: str) -> None:
    roles   = role_map(data)
    n_rounds = len(data["round_logs"])
    rounds   = list(range(1, n_rounds + 1))

    # cumulative payoff per agent per round
    cum: Dict[int, List[float]] = defaultdict(list)
    for rlog in data["round_logs"]:
        payoffs = {int_id(k): v for k, v in rlog["payoffs"].items()}
        for aid in sorted(roles):
            prev = cum[aid][-1] if cum[aid] else 0.0
            cum[aid].append(prev + payoffs.get(aid, 0.0))

    # per-round payoff per agent
    per_round: Dict[int, List[float]] = defaultdict(list)
    for rlog in data["round_logs"]:
        payoffs = {int_id(k): v for k, v in rlog["payoffs"].items()}
        for aid in sorted(roles):
            per_round[aid].append(payoffs.get(aid, 0.0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: cumulative payoff trajectories
    ax = axes[0]
    for aid, vals in cum.items():
        role  = roles[aid]
        color = ROLE_COLORS[role]
        ax.plot(rounds, vals, color=color, alpha=0.6,
                linewidth=2, label=f"Agent {aid} ({role[0].upper()})")
    ax.set_xlabel("Round"); ax.set_ylabel("Cumulative payoff")
    ax.set_title("Cumulative payoff trajectories")
    ax.grid(True, alpha=0.3)
    handles = [
        plt.Line2D([0],[0], color=ROLE_COLORS["violator"],   lw=2, label="Violator"),
        plt.Line2D([0],[0], color=ROLE_COLORS["cooperator"], lw=2, label="Cooperator"),
    ]
    ax.legend(handles=handles, fontsize=9)

    # Panel B: mean per-round payoff by role
    ax = axes[1]
    for role in ("violator", "cooperator"):
        ids = [aid for aid, r in roles.items() if r == role]
        mean_vals = [np.mean([per_round[aid][t] for aid in ids]) for t in range(n_rounds)]
        std_vals  = [np.std( [per_round[aid][t] for aid in ids]) for t in range(n_rounds)]
        ax.plot(rounds, mean_vals, color=ROLE_COLORS[role], lw=2.5, label=role.capitalize())
        ax.fill_between(rounds,
                        [m - s for m, s in zip(mean_vals, std_vals)],
                        [m + s for m, s in zip(mean_vals, std_vals)],
                        color=ROLE_COLORS[role], alpha=0.15)
    ax.set_xlabel("Round"); ax.set_ylabel("Mean per-round payoff")
    ax.set_title("Mean payoff by role ± std")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Panel C: final total payoff distribution (violin / strip)
    ax = axes[2]
    for i, role in enumerate(("violator", "cooperator")):
        ids    = [aid for aid, r in roles.items() if r == role]
        totals = [data["final_agents"][aid]["material_payoff"] for aid in ids]
        ax.scatter([i] * len(totals), totals,
                   color=ROLE_COLORS[role], s=80, zorder=3, alpha=0.9)
        ax.plot([i - 0.2, i + 0.2], [np.mean(totals)] * 2,
                color="black", lw=2, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Violators", "Cooperators"])
    ax.set_ylabel("Total payoff"); ax.set_title("Final total payoff by role")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Payoff analysis — {condition}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"payoff_{base}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved payoff distribution → {path}")


# ============================================================
# 2. Ostracism tracking
# ============================================================

def _incoming_strength(network_weights: List[dict], target: int) -> float:
    return sum(e["weight"] for e in network_weights if int_id(e["v"]) == target)


def _ostracism_fraction(network_weights: List[dict]) -> float:
    """Fraction of directed edges below OSTRACISM_THRESHOLD."""
    if not network_weights:
        return 0.0
    return sum(1 for e in network_weights if e["weight"] < OSTRACISM_THRESHOLD) / len(network_weights)


def plot_ostracism(data: dict, condition: str, out_dir: str, base: str) -> None:
    roles    = role_map(data)
    n_rounds = len(data["round_logs"])
    rounds   = list(range(1, n_rounds + 1))

    # incoming strength per agent per round
    incoming: Dict[int, List[float]] = defaultdict(list)
    ost_frac: List[float] = []

    for rlog in data["round_logs"]:
        nw = rlog.get("network_weights", [])
        for aid in sorted(roles):
            incoming[aid].append(_incoming_strength(nw, aid))
        ost_frac.append(_ostracism_fraction(nw))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: incoming strength over time per agent
    ax = axes[0]
    for aid, vals in incoming.items():
        role  = roles[aid]
        ax.plot(rounds, vals, color=ROLE_COLORS[role], alpha=0.65, lw=2,
                label=f"Agent {aid}")
    ax.set_xlabel("Round"); ax.set_ylabel("Total incoming weight")
    ax.set_title("Incoming network strength per agent")
    handles = [
        plt.Line2D([0],[0], color=ROLE_COLORS["violator"],   lw=2, label="Violator"),
        plt.Line2D([0],[0], color=ROLE_COLORS["cooperator"], lw=2, label="Cooperator"),
    ]
    ax.legend(handles=handles, fontsize=9); ax.grid(True, alpha=0.3)

    # Panel B: mean incoming strength by role
    ax = axes[1]
    for role in ("violator", "cooperator"):
        ids = [aid for aid, r in roles.items() if r == role]
        mean_vals = [np.mean([incoming[aid][t] for aid in ids]) for t in range(n_rounds)]
        ax.plot(rounds, mean_vals, color=ROLE_COLORS[role], lw=2.5, label=role.capitalize())
    ax.axhline(y=0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Round"); ax.set_ylabel("Mean incoming strength")
    ax.set_title("Mean incoming strength by role")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Panel C: fraction of near-ostracism edges over time
    ax = axes[2]
    ax.fill_between(rounds, ost_frac, alpha=0.3, color="#d62728")
    ax.plot(rounds, ost_frac, color="#d62728", lw=2.5)
    ax.axhline(y=0, color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Round")
    ax.set_ylabel(f"Fraction of edges < {OSTRACISM_THRESHOLD}")
    ax.set_title(f"Near-ostracism edge fraction\n(weight < {OSTRACISM_THRESHOLD})")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Ostracism tracking — {condition}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"ostracism_{base}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved ostracism tracking → {path}")


# ============================================================
# 3. Network structure
# ============================================================

def _build_graph(network_weights: List[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for e in network_weights:
        G.add_edge(int_id(e["u"]), int_id(e["v"]), weight=e["weight"])
    return G


def plot_network_structure(data: dict, condition: str, out_dir: str, base: str) -> None:
    roles    = role_map(data)
    n_rounds = len(data["round_logs"])
    rounds   = list(range(1, n_rounds + 1))

    # per-round graph metrics
    density_series:      List[float] = []
    mean_strength_series: List[float] = []
    reciprocity_series:  List[float] = []

    # in-strength by role per round
    in_strength_role: Dict[str, List[float]] = {"violator": [], "cooperator": []}

    for rlog in data["round_logs"]:
        G = _build_graph(rlog.get("network_weights", []))
        density_series.append(nx.density(G))
        weights = [d["weight"] for _, _, d in G.edges(data=True)]
        mean_strength_series.append(np.mean(weights) if weights else 0.0)
        reciprocity_series.append(nx.overall_reciprocity(G) if G.number_of_edges() > 0 else 0.0)

        for role in ("violator", "cooperator"):
            ids = [aid for aid, r in roles.items() if r == role]
            strengths = [
                sum(d["weight"] for _, _, d in G.in_edges(aid, data=True))
                for aid in ids if aid in G
            ]
            in_strength_role[role].append(np.mean(strengths) if strengths else 0.0)

    # Final-round weight distribution by role pair
    final_nw = data["round_logs"][-1].get("network_weights", [])
    vv_weights, vc_weights, cc_weights = [], [], []
    for e in final_nw:
        u, v, w = int_id(e["u"]), int_id(e["v"]), e["weight"]
        ru, rv  = roles.get(u, "cooperator"), roles.get(v, "cooperator")
        if ru == "violator" and rv == "violator":
            vv_weights.append(w)
        elif ru == "cooperator" and rv == "cooperator":
            cc_weights.append(w)
        else:
            vc_weights.append(w)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: graph-level metrics over time
    ax = axes[0, 0]
    ax.plot(rounds, density_series,       lw=2, color="#2ca02c", label="Density")
    ax.plot(rounds, reciprocity_series,   lw=2, color="#9467bd", label="Reciprocity")
    ax2 = ax.twinx()
    ax2.plot(rounds, mean_strength_series, lw=2, color="#ff7f0e", linestyle="--", label="Mean weight")
    ax2.set_ylabel("Mean edge weight", color="#ff7f0e")
    ax.set_xlabel("Round"); ax.set_ylabel("Metric (0–1)")
    ax.set_title("Network-level metrics over time")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: in-strength by role over time
    ax = axes[0, 1]
    for role in ("violator", "cooperator"):
        ax.plot(rounds, in_strength_role[role],
                color=ROLE_COLORS[role], lw=2.5, label=role.capitalize())
    ax.set_xlabel("Round"); ax.set_ylabel("Mean in-strength")
    ax.set_title("Mean incoming strength by role")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Panel C: final-round weight histograms by pair type
    ax = axes[1, 0]
    bins = np.linspace(0, max(vv_weights + vc_weights + cc_weights) + 0.1, 20)
    for weights, label, color in [
        (vv_weights, "V–V", "#d62728"),
        (vc_weights, "V–C", "#ff7f0e"),
        (cc_weights, "C–C", "#1f77b4"),
    ]:
        if weights:
            ax.hist(weights, bins=bins, alpha=0.5, color=color, label=label, edgecolor="black", lw=0.4)
    ax.set_xlabel("Edge weight"); ax.set_ylabel("Count")
    ax.set_title("Final-round weight distribution by pair type")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    # Panel D: final network heatmap (adjacency matrix sorted by role)
    ax = axes[1, 1]
    sorted_ids = (
        sorted(aid for aid, r in roles.items() if r == "violator") +
        sorted(aid for aid, r in roles.items() if r == "cooperator")
    )
    n = len(sorted_ids)
    idx = {aid: i for i, aid in enumerate(sorted_ids)}
    mat = np.zeros((n, n))
    for e in final_nw:
        u, v = int_id(e["u"]), int_id(e["v"])
        if u in idx and v in idx:
            mat[idx[u], idx[v]] = e["weight"]
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="Weight")
    tick_labels = [f"{aid}({'V' if roles[aid]=='violator' else 'C'})" for aid in sorted_ids]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=7)
    n_v = sum(1 for r in roles.values() if r == "violator")
    ax.axhline(n_v - 0.5, color="white", lw=1.5); ax.axvline(n_v - 0.5, color="white", lw=1.5)
    ax.set_title("Final adjacency matrix (V then C)")

    fig.suptitle(f"Network structure — {condition}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"network_structure_{base}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved network structure → {path}")


# ============================================================
# 4. Condition comparison (multi-file)
# ============================================================

def plot_condition_comparison(datasets: List[Tuple[str, dict]], out_dir: str, tag: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for cond, data in datasets:
        roles     = role_map(data)
        n_rounds  = len(data["round_logs"])
        rounds    = list(range(1, n_rounds + 1))
        color     = COND_COLORS.get(cond, "gray")

        # cumulative mean payoff
        mean_cum: List[float] = []
        acc = 0.0
        for rlog in data["round_logs"]:
            payoffs = list({int_id(k): v for k, v in rlog["payoffs"].items()}.values())
            acc += np.mean(payoffs) if payoffs else 0.0
            mean_cum.append(acc)
        axes[0, 0].plot(rounds, mean_cum, color=color, lw=2.5, label=cond)

        # mean contribution per round
        mean_contrib: List[float] = []
        coop_rate: List[float] = []
        for rlog in data["round_logs"]:
            contribs = list({int_id(k): v for k, v in rlog["contributions"].items()}.values())
            mean_contrib.append(np.mean(contribs) if contribs else 0.0)
            coop_rate.append(np.mean([1 if c >= 5 else 0 for c in contribs]) if contribs else 0.0)
        axes[0, 1].plot(rounds, mean_contrib, color=color, lw=2.5, label=cond)
        axes[1, 0].plot(rounds, coop_rate,    color=color, lw=2.5, label=cond)

        # mean edge weight per round
        mean_w: List[float] = []
        for rlog in data["round_logs"]:
            nw = rlog.get("network_weights", [])
            mean_w.append(np.mean([e["weight"] for e in nw]) if nw else 0.0)
        axes[1, 1].plot(rounds, mean_w, color=color, lw=2.5, label=cond)

    titles  = ["Cumulative mean payoff", "Mean contribution",
               "Cooperation rate (contrib ≥ 5)", "Mean network edge weight"]
    ylabels = ["Cumulative payoff", "Mean contribution",
               "Fraction contributing ≥ 5", "Mean weight"]
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title); ax.set_xlabel("Round"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle("Condition comparison", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"condition_comparison_{tag}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved condition comparison → {path}")


# ============================================================
# 5. Norm emergence
# ============================================================

_AVOID_RE = re.compile(r"agent\s+(\d+)", re.IGNORECASE)


def _count_avoid_mentions(perceptions: dict) -> Dict[int, int]:
    """Count how many times each agent is mentioned in agents_to_avoid."""
    counts: Dict[int, int] = defaultdict(int)
    for agent_perc in perceptions.values():
        text = agent_perc.get("agents_to_avoid", "")
        for m in _AVOID_RE.finditer(text):
            counts[int(m.group(1))] += 1
    return counts


def plot_norm_emergence(data: dict, condition: str, out_dir: str, base: str) -> None:
    roles    = role_map(data)
    n_rounds = len(data["round_logs"])
    rounds   = list(range(1, n_rounds + 1))

    # contribution variance over time
    contrib_var:  List[float] = []
    mean_contrib: List[float] = []

    # avoid-mention counts per agent per round
    avoid_counts: Dict[int, List[int]] = defaultdict(list)

    # cooperation rate (fraction contributing >= 5)
    coop_rate: List[float] = []

    # violator exposure: fraction of violators explicitly named in agents_to_avoid
    violator_ids = {aid for aid, r in roles.items() if r == "violator"}
    viol_exposure: List[float] = []

    for rlog in data["round_logs"]:
        contribs = [v for v in {int_id(k): v for k, v in rlog["contributions"].items()}.values()]
        contrib_var.append(float(np.var(contribs)) if contribs else 0.0)
        mean_contrib.append(float(np.mean(contribs)) if contribs else 0.0)
        coop_rate.append(np.mean([1 if c >= 5 else 0 for c in contribs]) if contribs else 0.0)

        perceptions = rlog.get("perceptions", {})
        avoid = _count_avoid_mentions(perceptions)

        all_ids = sorted(roles)
        for aid in all_ids:
            avoid_counts[aid].append(avoid.get(aid, 0))

        if violator_ids:
            named = sum(1 for vid in violator_ids if avoid.get(vid, 0) > 0)
            viol_exposure.append(named / len(violator_ids))
        else:
            viol_exposure.append(0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: contribution mean and variance over time
    ax = axes[0, 0]
    ax.plot(rounds, mean_contrib, color="#2ca02c", lw=2.5, label="Mean contribution")
    ax2 = ax.twinx()
    ax2.fill_between(rounds, contrib_var, alpha=0.2, color="#d62728")
    ax2.plot(rounds, contrib_var, color="#d62728", lw=1.5, linestyle="--", label="Variance")
    ax2.set_ylabel("Contribution variance", color="#d62728")
    ax.set_xlabel("Round"); ax.set_ylabel("Mean contribution", color="#2ca02c")
    ax.set_title("Contribution mean and variance")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: cooperation rate over time
    ax = axes[0, 1]
    ax.fill_between(rounds, coop_rate, alpha=0.2, color="#1f77b4")
    ax.plot(rounds, coop_rate, color="#1f77b4", lw=2.5)
    ax.axhline(0.5, color="gray", lw=0.8, linestyle="--", label="50% threshold")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Round"); ax.set_ylabel("Fraction contributing ≥ 5")
    ax.set_title("Cooperation rate over time")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel C: avoid-mention counts per agent over time
    ax = axes[1, 0]
    for aid in sorted(roles):
        role  = roles[aid]
        ax.plot(rounds, avoid_counts[aid],
                color=ROLE_COLORS[role], alpha=0.65, lw=1.8,
                label=f"Agent {aid} ({'V' if role=='violator' else 'C'})")
    ax.set_xlabel("Round"); ax.set_ylabel("Times named in agents_to_avoid")
    ax.set_title("Ostracism nominations per agent")
    handles = [
        plt.Line2D([0],[0], color=ROLE_COLORS["violator"],   lw=2, label="Violator"),
        plt.Line2D([0],[0], color=ROLE_COLORS["cooperator"], lw=2, label="Cooperator"),
    ]
    ax.legend(handles=handles, fontsize=9); ax.grid(True, alpha=0.3)

    # Panel D: violator exposure fraction over time
    ax = axes[1, 1]
    ax.fill_between(rounds, viol_exposure, alpha=0.25, color="#d62728")
    ax.plot(rounds, viol_exposure, color="#d62728", lw=2.5,
            label="Fraction of violators named")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of violators in avoid-lists")
    ax.set_title("Norm enforcement signal\n(violators named as agents to avoid)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Norm emergence — {condition}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"norm_emergence_{base}.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved norm emergence → {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deep analysis of SoNoLiSi logs — payoff, ostracism, network, norms."
    )
    parser.add_argument(
        "log_files", nargs="+",
        help="One or more log JSON files. Multiple files enable condition comparison."
    )
    parser.add_argument(
        "--out-dir", default=".", metavar="DIR",
        help="Output directory for plots (default: current directory)"
    )
    parser.add_argument(
        "--skip-single", action="store_true",
        help="Skip per-file analyses (payoff, ostracism, network structure, norm emergence)"
    )
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip condition comparison plot (requires multiple files)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    loaded: List[Tuple[str, dict, str]] = []   # (condition, data, base)
    for path in args.log_files:
        print(f"Loading {path} ...")
        data      = load_log(path)
        condition = data.get("condition", "unknown")
        base      = os.path.splitext(os.path.basename(path))[0]
        loaded.append((condition, data, base))

        if not args.skip_single:
            print(f"  Condition: {condition}  |  Rounds: {len(data['round_logs'])}")
            plot_payoff_distribution(data, condition, args.out_dir, base)
            plot_ostracism(data, condition, args.out_dir, base)
            plot_network_structure(data, condition, args.out_dir, base)
            plot_norm_emergence(data, condition, args.out_dir, base)

    if not args.skip_comparison and len(loaded) > 1:
        datasets = [(cond, data) for cond, data, _ in loaded]
        bases    = [b for _, _, b in loaded]
        # derive a short shared tag from the file names
        tag = "_vs_".join(c for c, _, _ in loaded)
        plot_condition_comparison(datasets, args.out_dir, tag)

    print("Done.")


if __name__ == "__main__":
    main()
