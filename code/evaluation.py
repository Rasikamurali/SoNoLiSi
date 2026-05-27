# ============================================================
# evaluation.py — Network evaluation for SoNoLiSi simulations
#
# Analyses:
#   1. Mean edge weight over time by pair type:
#      violator-violator, violator-cooperator, cooperator-cooperator
#   2. Animated network with fixed layout showing weight evolution per round
#
# Expects log files where final_agents has a "role" field: "cooperator" | "violator"
#
# Usage:
#   python evaluation.py log_v4_<timestamp>_FULL_seed42.json
#   python evaluation.py log_v4_<timestamp>_FULL_seed42.json --out-dir ./plots
# ============================================================

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D


# ============================================================
# Loading and classification
# ============================================================

def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def classify_agents(data: dict) -> Dict[int, str]:
    """
    Returns {agent_id: "violator" | "cooperator"}.
    Supports logs with a "role" field or a "cooperation_tendency" field.
    For tendency-based logs, classification uses initial tendency from round 1
    agent_states: tendency < 0.3 → violator, else → cooperator.
    """
    final_agents = data["final_agents"]
    if "role" in final_agents[0]:
        return {a["id"]: a["role"] for a in final_agents}
    # tendency-based: classify from round 1 initial states
    init_states = data["round_logs"][0]["agent_states"]
    return {
        s["id"]: ("violator" if s["cooperation_tendency"] < 0.3 else "cooperator")
        for s in init_states
    }


def get_tendency(data: dict) -> Dict[int, float]:
    """Returns {agent_id: float} for node coloring (0.1=violator, 0.9=cooperator)."""
    agent_class = classify_agents(data)
    return {aid: (0.1 if role == "violator" else 0.9) for aid, role in agent_class.items()}


def edge_type(u: int, v: int, agent_class: Dict[int, str]) -> str:
    cu, cv = agent_class.get(u, "cooperator"), agent_class.get(v, "cooperator")
    if cu == "violator" and cv == "violator":
        return "violator-violator"
    elif cu == "cooperator" and cv == "cooperator":
        return "cooperator-cooperator"
    else:
        return "violator-cooperator"


# ============================================================
# Weight series analysis
# ============================================================

def compute_weight_series(
    round_logs: List[dict],
    agent_class: Dict[int, str],
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    For each round, compute mean edge weight by pair type.
    Returns (rounds, {type: [mean_weight_per_round]}).
    """
    rounds = []
    series: Dict[str, List[float]] = defaultdict(list)

    for rlog in round_logs:
        rounds.append(rlog["round"])
        type_weights: Dict[str, List[float]] = defaultdict(list)

        for edge in rlog.get("network_weights", []):
            u, v, w = edge["u"], edge["v"], edge["weight"]
            et = edge_type(u, v, agent_class)
            type_weights[et].append(w)

        for et in ["violator-violator", "violator-cooperator", "cooperator-cooperator"]:
            vals = type_weights.get(et, [])
            series[et].append(float(np.mean(vals)) if vals else float("nan"))

    return rounds, dict(series)


def plot_weight_series(
    rounds: List[int],
    series: Dict[str, List[float]],
    condition: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    style = {
        "violator-violator":   {"color": "#d62728", "label": "Violator–Violator"},
        "violator-cooperator": {"color": "#ff7f0e", "label": "Violator–Cooperator"},
        "cooperator-cooperator": {"color": "#1f77b4", "label": "Cooperator–Cooperator"},
    }

    for et, vals in series.items():
        ax.plot(rounds, vals, linewidth=2.5, **style[et])

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Mean edge weight", fontsize=12)
    ax.set_title(f"Edge weight dynamics by pair type — {condition}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(rounds), max(rounds))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved weight series → {save_path}")
    plt.close()


# ============================================================
# Network animation
# ============================================================

NODE_COLOR = "#4878cf"   # uniform blue for all nodes
EDGE_COLOR = "#888888"   # uniform gray for all edges


def animate_network(
    round_logs: List[dict],
    agent_class: Dict[int, str],
    tendency: Dict[int, float],
    condition: str,
    save_path: str,
) -> None:
    all_nodes = sorted(agent_class.keys())

    # Fixed layout — computed once from the initial complete graph
    G_init = nx.DiGraph()
    G_init.add_nodes_from(all_nodes)
    for edge in round_logs[0].get("network_weights", []):
        G_init.add_edge(edge["u"], edge["v"], weight=edge["weight"])
    pos = nx.spring_layout(G_init, seed=42, k=2.0)

    node_labels = {n: str(n) for n in all_nodes}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [2, 1]})
    ax_net, ax_bar = axes

    # Pre-compute overall mean weight per round for the right panel
    mean_weights = []
    for rlog in round_logs:
        nw = rlog.get("network_weights", [])
        mean_weights.append(np.mean([e["weight"] for e in nw]) if nw else 0.0)

    def update(frame: int) -> None:
        ax_net.clear()
        ax_bar.clear()

        rlog = round_logs[frame]
        round_num = rlog["round"]

        # Rebuild graph for this round
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes)
        edge_weights = {}
        for edge in rlog.get("network_weights", []):
            u, v, w = edge["u"], edge["v"], edge["weight"]
            G.add_edge(u, v, weight=w)
            edge_weights[(u, v)] = w

        # Draw nodes — uniform color
        nx.draw_networkx_nodes(
            G, pos, ax=ax_net,
            nodelist=all_nodes,
            node_color=NODE_COLOR,
            node_size=700,
            alpha=0.95,
        )
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax_net,
                                font_size=9, font_color="white", font_weight="bold")

        # Draw edges — uniform color, width and alpha scaled by weight
        for (u, v) in G.edges():
            w = edge_weights.get((u, v), 0)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax_net,
                width=max(0.3, w * 2.5),
                alpha=min(0.85, max(0.05, w * 0.5)),
                edge_color=EDGE_COLOR,
                arrows=True, arrowsize=10,
                connectionstyle="arc3,rad=0.12",
            )

        ax_net.set_title(f"{condition} — Round {round_num} / {len(round_logs)}", fontsize=13)
        ax_net.axis("off")

        # Right panel: mean weight over time up to current round
        ax_bar.plot(range(1, frame + 2), mean_weights[:frame + 1],
                    color=NODE_COLOR, linewidth=2.5)
        ax_bar.scatter([round_num], [mean_weights[frame]],
                       color=NODE_COLOR, s=60, zorder=5)
        all_w = [e["weight"] for e in rlog.get("network_weights", [])]
        if all_w:
            ax_bar.axhline(mean_weights[frame], color="gray", linewidth=0.8,
                           linestyle="--", alpha=0.5)
        ax_bar.set_xlim(1, len(round_logs))
        ax_bar.set_ylim(0, max(mean_weights) * 1.2 + 0.1)
        ax_bar.set_xlabel("Round", fontsize=10)
        ax_bar.set_ylabel("Mean edge weight", fontsize=10)
        ax_bar.set_title("Network mean weight over time", fontsize=11)
        ax_bar.grid(alpha=0.3)
        ax_bar.text(round_num, mean_weights[frame] + max(mean_weights) * 0.05,
                    f"{mean_weights[frame]:.3f}", ha="center", fontsize=9)

    ani = animation.FuncAnimation(
        fig, update, frames=len(round_logs), interval=900, repeat=True
    )

    plt.tight_layout()
    writer = animation.FFMpegWriter(fps=1.2, bitrate=1800)
    ani.save(save_path, writer=writer, dpi=120)
    print(f"  Saved animation → {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SoNoLiSi simulation logs — network weight dynamics and animation."
    )
    parser.add_argument("log_file", help="Path to log JSON file (e.g. log_v4_*.json)")
    parser.add_argument(
        "--out-dir", default=".", metavar="DIR",
        help="Directory for output files (default: current directory)"
    )
    parser.add_argument(
        "--no-animation", action="store_true",
        help="Skip the network animation (faster, just produce the weight series plot)"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.log_file} ...")
    data = load_log(args.log_file)

    condition    = data.get("condition", "unknown")
    final_agents = data["final_agents"]
    round_logs   = data["round_logs"]

    agent_class = classify_agents(data)
    tendency    = get_tendency(data)

    violator_ids   = sorted(aid for aid, cls in agent_class.items() if cls == "violator")
    cooperator_ids = sorted(aid for aid, cls in agent_class.items() if cls == "cooperator")

    print(f"Condition      : {condition}")
    print(f"Rounds         : {len(round_logs)}")
    print(f"Violators      : {violator_ids}")
    print(f"Cooperators    : {cooperator_ids}")

    # 1. Weight series plot
    rounds, series = compute_weight_series(round_logs, agent_class)
    base = os.path.splitext(os.path.basename(args.log_file))[0]
    plot_path = os.path.join(args.out_dir, f"weights_{base}.png")
    plot_weight_series(rounds, series, condition, save_path=plot_path)

    # 2. Network animation
    if not args.no_animation:
        anim_path = os.path.join(args.out_dir, f"network_{base}.mp4")
        animate_network(round_logs, agent_class, tendency, condition, save_path=anim_path)

    print("Done.")


if __name__ == "__main__":
    main()
