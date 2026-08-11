"""
weight_update_check.py
----------------------
Self-contained demo of local network weight updating.

Setup:
  - 12 nodes, fully connected directed graph (all pairs have an initial weight of 1.0)
  - Each timestep: nodes are split randomly into 3 groups of 4
  - Within each group, every node assigns a random evaluation in [-1, +1] to each other
  - Weight update rule:
      if evaluation < 0:  weight += 0.8 * evaluation   (drops fast)
      if evaluation >= 0: weight += 0.2 * evaluation   (grows slowly)
      weight is clamped to [0.0, 2.0]
  - Edges with weight below 0.15 are considered severed (drawn dashed/invisible)

Animation: one frame per round, saved as weight_update_check.mp4
  - Fixed circular layout
  - Edge thickness ∝ weight
  - Edge color: green (strong) → grey → red (weak) → invisible (severed)
  - Current-round groups highlighted with matching node colours
  - Per-round weight distribution shown in a subplot
"""

import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

# ─── Config ───────────────────────────────────────────────────────────────────

N           = 12
GROUP_SIZE  = 4
ROUNDS      = 20
RATE_NEG    = 0.2
RATE_POS    = 0.8
W_INIT      = 1.0
W_MIN       = 0.0
W_MAX       = 2.0
SEVER_THRESH = 0.15   # edges below this are drawn as faint dashed lines

SEED        = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "weight_update_check.mp4")

GROUP_COLORS = ["#e6194b", "#3cb44b", "#4363d8",  # 3 groups
                "#f58231", "#911eb4", "#42d4f4"]

# ─── Simulation ───────────────────────────────────────────────────────────────

def split_groups(nodes, group_size, rng):
    shuffled = nodes[:]
    rng.shuffle(shuffled)
    return [shuffled[i: i + group_size] for i in range(0, len(shuffled), group_size)]


def update_weights(weights, group, rng):
    """In-place weight update for one group."""
    for i in group:
        for j in group:
            if i == j:
                continue
            eval_val = rng.uniform(-1.0, 1.0)
            rate  = RATE_NEG if eval_val < 0 else RATE_POS
            delta = rate * eval_val
            weights[i][j] = float(np.clip(weights[i][j] + delta, W_MIN, W_MAX))


def run_simulation(seed=SEED):
    rng = random.Random(seed)
    nodes = list(range(N))

    # weights[i][j] = directed weight from i to j
    weights = {i: {j: W_INIT for j in range(N) if j != i} for i in range(N)}

    history = []   # list of (groups, weight_snapshot) per round

    for _ in range(ROUNDS):
        groups  = split_groups(nodes, GROUP_SIZE, rng)
        for g in groups:
            update_weights(weights, g, rng)
        # snapshot: dict of (i,j) → weight for all directed edges
        snap = {(i, j): weights[i][j]
                for i in range(N) for j in range(N) if i != j}
        history.append((groups, snap))

    return history


# ─── Visualisation helpers ────────────────────────────────────────────────────

def weight_to_color(w, w_max=W_MAX):
    """Map weight to a colour: low=red, mid=grey, high=green."""
    t = np.clip(w / w_max, 0, 1)
    if t < 0.5:
        r = 1.0;  g = 2 * t;       b = 2 * t * 0.3
    else:
        r = 2 * (1 - t);  g = 1.0; b = 0.3 * (1 - t)
    return (r, g, b)


def symmetrise(snap):
    """Average directed weights to get an undirected weight for display."""
    sym = {}
    nodes = range(N)
    for i in nodes:
        for j in nodes:
            if i < j:
                sym[(i, j)] = (snap.get((i, j), 0) + snap.get((j, i), 0)) / 2
    return sym


def build_graph(sym_weights):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (i, j), w in sym_weights.items():
        G.add_edge(i, j, weight=w)
    return G


# ─── Animation ────────────────────────────────────────────────────────────────

def animate(history):
    fig = plt.figure(figsize=(14, 7), facecolor="#1a1a2e")
    gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.05)
    ax_net  = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    for ax in (ax_net, ax_hist):
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # Fixed circular layout
    pos = nx.circular_layout(list(range(N)))

    # Pre-compute initial symmetric weights for layout
    _, snap0 = history[0]
    sym0 = symmetrise(snap0)

    def draw_frame(t):
        ax_net.clear()
        ax_hist.clear()
        for ax in (ax_net, ax_hist):
            ax.set_facecolor("#1a1a2e")

        groups, snap = history[t]
        sym = symmetrise(snap)
        G   = build_graph(sym)

        # Node colour = which group they're in this round
        node_color_map = {}
        for g_idx, grp in enumerate(groups):
            for node in grp:
                node_color_map[node] = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
        node_colors = [node_color_map.get(n, "#aaaaaa") for n in G.nodes()]

        # Edges: separate active from severed
        active_edges  = [(i, j) for (i, j), w in sym.items() if w >= SEVER_THRESH]
        severed_edges = [(i, j) for (i, j), w in sym.items() if w < SEVER_THRESH]

        active_weights = [sym[(i, j)] for (i, j) in active_edges]
        active_colors  = [weight_to_color(w) for w in active_weights]
        active_widths  = [max(0.3, w * 3.5) for w in active_weights]

        # Draw severed edges faintly
        if severed_edges:
            nx.draw_networkx_edges(G, pos, edgelist=severed_edges,
                                   edge_color="#333355", width=0.5,
                                   style="dashed", alpha=0.3, ax=ax_net)

        # Draw active edges
        if active_edges:
            nx.draw_networkx_edges(G, pos, edgelist=active_edges,
                                   edge_color=active_colors,
                                   width=active_widths,
                                   alpha=0.85, ax=ax_net,
                                   arrows=False)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=600, ax=ax_net,
                               linewidths=2, edgecolors="white")
        nx.draw_networkx_labels(G, pos, font_color="white",
                                font_size=10, font_weight="bold", ax=ax_net)

        ax_net.set_title(f"Round {t + 1} / {ROUNDS}",
                         color="white", fontsize=16, pad=12)
        ax_net.axis("off")

        # Group legend
        legend_patches = [
            mpatches.Patch(color=GROUP_COLORS[g_idx],
                           label=f"Group {g_idx + 1}: {groups[g_idx]}")
            for g_idx in range(len(groups))
        ]
        ax_net.legend(handles=legend_patches, loc="lower left",
                      facecolor="#2a2a4e", edgecolor="#444466",
                      labelcolor="white", fontsize=9)

        # Weight distribution histogram
        all_weights = list(sym.values())
        ax_hist.hist(all_weights, bins=20, range=(0, W_MAX),
                     color="#4363d8", edgecolor="#7777cc", alpha=0.85)
        ax_hist.axvline(np.mean(all_weights), color="#ffdd57",
                        linestyle="--", linewidth=2,
                        label=f"mean={np.mean(all_weights):.2f}")
        ax_hist.axvline(SEVER_THRESH, color="#ff4444",
                        linestyle=":", linewidth=1.5,
                        label=f"sever<{SEVER_THRESH}")
        n_severed = sum(1 for w in all_weights if w < SEVER_THRESH)
        ax_hist.set_title(f"Weight distribution\n"
                          f"severed: {n_severed} / {len(all_weights)} edges",
                          color="white", fontsize=11)
        ax_hist.set_xlabel("Edge weight", color="#aaaacc", fontsize=10)
        ax_hist.set_ylabel("Count", color="#aaaacc", fontsize=10)
        ax_hist.tick_params(colors="#aaaacc")
        ax_hist.legend(facecolor="#2a2a4e", edgecolor="#444466",
                       labelcolor="white", fontsize=9)
        ax_hist.set_xlim(0, W_MAX)
        for spine in ax_hist.spines.values():
            spine.set_edgecolor("#444466")

    ani = animation.FuncAnimation(
        fig, draw_frame,
        frames=ROUNDS,
        interval=700,    # ms per frame
        repeat=True,
    )

    writer = animation.FFMpegWriter(fps=2, bitrate=1800,
                                    metadata={"title": "Network Weight Update Demo"})
    ani.save(OUTPUT_PATH, writer=writer, dpi=120,
             savefig_kwargs={"facecolor": "#1a1a2e"})
    plt.close(fig)
    print(f"Saved → {OUTPUT_PATH}")


# ─── Weight-based group formation ────────────────────────────────────────────

def form_groups_weighted(nodes, weights, group_size, rng):
    """
    Mirrors the simulation: probabilistic group formation from edge weights.
    Seed is sampled proportional to average incoming weight.
    Partners are sampled proportional to seed's outgoing weights.
    """
    unassigned = nodes[:]
    groups = []
    while len(unassigned) >= group_size:
        # Seed: proportional to average incoming weight from unassigned pool
        seed_w = []
        for aid in unassigned:
            others = [j for j in unassigned if j != aid]
            if others:
                avg_in = np.mean([weights[j].get(aid, 1e-4) for j in others])
            else:
                avg_in = 1e-4
            seed_w.append(max(avg_in, 1e-4))
        seed_probs = np.array(seed_w) / sum(seed_w)
        seed_id    = rng.choices(unassigned, weights=seed_probs.tolist(), k=1)[0]

        # Partners: proportional to seed's outgoing weights
        candidates = [j for j in unassigned if j != seed_id]
        if candidates:
            part_w     = [max(weights[seed_id].get(j, 1e-4), 1e-4) for j in candidates]
            part_probs = np.array(part_w) / sum(part_w)
            n_pick     = min(group_size - 1, len(candidates))
            chosen     = rng.choices(candidates, weights=part_probs.tolist(), k=n_pick)
            # deduplicate (rng.choices can repeat)
            seen, unique = set(), []
            for c in chosen:
                if c not in seen:
                    seen.add(c); unique.append(c)
            # top up if deduplication removed entries
            extras = [c for c in candidates if c not in seen]
            while len(unique) < n_pick and extras:
                unique.append(extras.pop(0))
            group = [seed_id] + unique
        else:
            group = [seed_id]

        for g in group:
            unassigned.remove(g)
        groups.append(group)
    if unassigned:
        groups.append(unassigned)
    return [g for g in groups if len(g) >= 2]


def run_simulation_weighted(seed=SEED):
    """Same as run_simulation but uses weight-based group formation."""
    rng   = random.Random(seed)
    nodes = list(range(N))
    weights = {i: {j: W_INIT for j in range(N) if j != i} for i in range(N)}
    history = []
    for _ in range(ROUNDS):
        groups = form_groups_weighted(nodes, weights, GROUP_SIZE, rng)
        for g in groups:
            update_weights(weights, g, rng)
        snap = {(i, j): weights[i][j]
                for i in range(N) for j in range(N) if i != j}
        history.append((groups, snap))
    return history


# ─── Co-occurrence and lock-in analysis ──────────────────────────────────────

def cooccurrence_matrix(history):
    """Count how many rounds each pair appeared in the same group."""
    mat = np.zeros((N, N), dtype=int)
    for groups, _ in history:
        for g in groups:
            for i in g:
                for j in g:
                    if i != j:
                        mat[i][j] += 1
    return mat


def gini(values):
    """Gini coefficient: 0 = perfectly equal, 1 = maximally unequal."""
    vals = np.sort(np.array(values, dtype=float))
    n    = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    idx  = np.arange(1, n + 1)
    return float((2 * (idx * vals).sum() / (n * vals.sum())) - (n + 1) / n)


def lockin_report(label, history):
    comat = cooccurrence_matrix(history)
    pair_counts = [comat[i][j] for i in range(N) for j in range(i + 1, N)]
    expected    = ROUNDS * (GROUP_SIZE - 1) / (N - 1)   # expected under random

    print(f"\n── {label} ──────────────────────────────────────────────────")
    print(f"  Expected co-occurrences per pair (random): {expected:.2f}")
    print(f"  Actual   co-occurrences — mean: {np.mean(pair_counts):.2f}  "
          f"min: {min(pair_counts)}  max: {max(pair_counts)}  "
          f"SD: {np.std(pair_counts):.2f}")
    print(f"  Gini coefficient of pair co-occurrences: {gini(pair_counts):.4f}  "
          f"(0=uniform, 1=one pair dominates)")
    print(f"  Pairs never co-occurring: "
          f"{sum(1 for c in pair_counts if c == 0)} / {len(pair_counts)}")
    print(f"  Pairs co-occurring ≥ 2× expected ({2*expected:.1f}+): "
          f"{sum(1 for c in pair_counts if c >= 2 * expected)} / {len(pair_counts)}")

    # Final weight stats
    _, snap = history[-1]
    sym  = symmetrise(snap)
    wvals = list(sym.values())
    print(f"  Final weights — mean: {np.mean(wvals):.3f}  "
          f"SD: {np.std(wvals):.3f}  min: {min(wvals):.3f}  max: {max(wvals):.3f}")
    print(f"  Weight Gini: {gini(wvals):.4f}")


def plot_lockin_comparison(hist_random, hist_weighted):
    """Side-by-side heatmaps of co-occurrence matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    fig.suptitle("Co-occurrence heatmap: how often each pair was grouped together",
                 color="white", fontsize=13, y=1.01)

    for ax, hist, title in [
        (axes[0], hist_random,  f"Random grouping\n(N={N}, {ROUNDS} rounds)"),
        (axes[1], hist_weighted, f"Weight-based grouping\n(N={N}, {ROUNDS} rounds)"),
    ]:
        mat = cooccurrence_matrix(hist)
        im  = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=ROUNDS)
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("Agent", color="#aaaacc"); ax.set_ylabel("Agent", color="#aaaacc")
        ax.tick_params(colors="#aaaacc")
        ax.set_facecolor("#1a1a2e")
        plt.colorbar(im, ax=ax, label="Rounds co-grouped")
        # Annotate cells
        for i in range(N):
            for j in range(N):
                if i != j and mat[i, j] > 0:
                    ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                            fontsize=7, color="black" if mat[i, j] < ROUNDS * 0.6 else "white")

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "weight_lockin_heatmap.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"\nSaved heatmap → {out}")


def plot_gini_over_time(hist_random, hist_weighted):
    """Gini coefficient of co-occurrence (up to round t) over time."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#1a1a2e")
    fig.suptitle("Lock-in over rounds: Gini of pair co-occurrence and edge weights",
                 color="white", fontsize=13, y=1.01)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#444466")
        ax.tick_params(colors="#aaaacc")

    rounds_range = range(1, ROUNDS + 1)
    for hist, label, color in [
        (hist_random,   "Random grouping",       "#4363d8"),
        (hist_weighted, "Weight-based grouping", "#e6194b"),
    ]:
        gini_cooc, gini_wt = [], []
        comat_cum = np.zeros((N, N), dtype=int)
        for t, (groups, snap) in enumerate(hist):
            for g in groups:
                for i in g:
                    for j in g:
                        if i != j: comat_cum[i][j] += 1
            pairs = [comat_cum[i][j] for i in range(N) for j in range(i+1, N)]
            gini_cooc.append(gini(pairs))
            sym  = symmetrise(snap)
            gini_wt.append(gini(list(sym.values())))

        axes[0].plot(list(rounds_range), gini_cooc, color=color, linewidth=2, label=label)
        axes[1].plot(list(rounds_range), gini_wt,   color=color, linewidth=2, label=label)

    for ax, title, ylabel in [
        (axes[0], "Co-occurrence inequality (Gini)", "Gini coefficient"),
        (axes[1], "Edge weight inequality (Gini)",   "Gini coefficient"),
    ]:
        ax.set_title(title, color="white", fontsize=12)
        ax.set_xlabel("Round", color="#aaaacc"); ax.set_ylabel(ylabel, color="#aaaacc")
        ax.legend(facecolor="#2a2a4e", edgecolor="#444466", labelcolor="white", fontsize=10)
        ax.axhline(0, color="#444466", linewidth=1, linestyle=":")
        ax.set_xlim(1, ROUNDS)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "weight_lockin_gini.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"Saved Gini plot → {out}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running {ROUNDS}-round simulation  (N={N}, groups of {GROUP_SIZE})")
    print(f"  Neg rate: {RATE_NEG}  |  Pos rate: {RATE_POS}")
    print(f"  Sever threshold: weight < {SEVER_THRESH}")

    hist_random   = run_simulation(seed=SEED)
    hist_weighted = run_simulation_weighted(seed=SEED)

    # Per-round weight summary for random grouping
    print(f"\n{'Round':>6}  {'Mean W':>8}  {'Min W':>8}  {'Severed':>8}")
    print("  " + "-" * 36)
    for t, (groups, snap) in enumerate(hist_random):
        sym  = symmetrise(snap)
        vals = list(sym.values())
        sev  = sum(1 for w in vals if w < SEVER_THRESH)
        print(f"  {t+1:>4}   {np.mean(vals):>8.3f}  {min(vals):>8.3f}  {sev:>7}")

    # Lock-in analysis
    lockin_report("Random grouping",       hist_random)
    lockin_report("Weight-based grouping", hist_weighted)

    # Comparison plots
    plot_lockin_comparison(hist_random, hist_weighted)
    plot_gini_over_time(hist_random, hist_weighted)

    print("\nRendering animation (random grouping) …")
    animate(hist_random)
