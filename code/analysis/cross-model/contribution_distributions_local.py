"""
contribution_distributions_local.py
------------------------------------
Supplementary companion to plot_contribution_all_conditions() in
run_local_analysis.py (Figure 1 of Results).

Figure 1 shows the run-clustered mean ± 95% CI trajectory only. This script
shows the underlying agent-level distribution at a few checkpoint rounds
(box + jittered strip), so a reader can see the individual-agent spread that
the mean/CI band summarizes — e.g. whether a flat mean hides two clusters of
agents (some cooperating, some not) versus genuine agent-level convergence.

Box = IQR + median (no fliers — the jittered points already show every value).
Underlying thin line = the same run-clustered mean trajectory as Figure 1,
included for continuity/context only.

Output: figures/local/cross_model/contribution_distributions_local.pdf/.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import run_local_analysis as base

FIG_ROOT     = base.FIG_ROOT
CHECKPOINTS  = [1, 10, 20]
DODGE        = {cond: (i - 2) * 0.55 for i, cond in enumerate(base.PLOT_CONDITIONS)}
BOX_WIDTH    = 0.42
LABEL_SIZE   = base.LABEL_SIZE
TICK_SIZE    = base.TICK_SIZE
_rng         = np.random.default_rng(0)


def build_agent_level_store():
    """store[model][cond][round] = list of individual agent contributions (all seeds pooled)."""
    store = {m: {c: {r: [] for r in base.ROUNDS} for c in base.PLOT_CONDITIONS} for m in base.MODELS}
    for model in base.MODELS:
        for cond in base.PLOT_CONDITIONS:
            for seed in base.SEEDS:
                d = base.load_latest_log(model, seed, cond)
                if d is None:
                    continue
                for rlog in d["round_logs"]:
                    r = rlog["round"]
                    if r in store[model][cond]:
                        store[model][cond][r].extend(rlog["contributions"].values())
    return store


def plot_one_model(agent_store, mean_store, model, ax):
    rounds = np.array(base.ROUNDS)

    # background mean trajectory (thin, all 20 rounds) — same series as Figure 1
    for cond in base.PLOT_CONDITIONS:
        seed_data = mean_store[model][cond]
        arr = np.array(list(seed_data.values()), dtype=float)
        mean, _ = base.mean_ci95(arr)
        ax.plot(rounds, mean, color=base.COND_COLORS[cond], linewidth=0.9,
                 alpha=0.3, zorder=3)

    # box (IQR + median only — no whiskers) + jitter at checkpoint rounds.
    # Whiskers are deliberately omitted: with this data's natural 0-10 bound,
    # 1.5*IQR whiskers read as a near min-max bar, which visually equalizes
    # conditions of very different noisiness. The jittered points already show
    # the full sample; the box is left as the dominant, uncluttered mark of
    # central tendency (25th-75th pctile + median).
    for cond in base.PLOT_CONDITIONS:
        color = base.COND_COLORS[cond]
        for r in CHECKPOINTS:
            vals = np.array(agent_store[model][cond][r], dtype=float)
            if len(vals) < 2:
                continue
            x0 = r + DODGE[cond]
            jitter = _rng.uniform(-BOX_WIDTH * 0.32, BOX_WIDTH * 0.32, size=len(vals))
            ax.scatter(x0 + jitter, vals, s=6, color=color, alpha=0.4,
                       linewidths=0, zorder=4)
            ax.boxplot(
                [vals], positions=[x0], widths=BOX_WIDTH,
                patch_artist=True, showfliers=False, showcaps=False,
                manage_ticks=False, zorder=5,
                medianprops=dict(color="black", linewidth=1.5),
                boxprops=dict(facecolor=color, edgecolor="black", alpha=0.6, linewidth=1.0),
                whiskerprops=dict(linewidth=0),
            )
            ax.scatter([x0], [vals.mean()], marker=base.COND_MARKERS[cond], s=60,
                       color=color, edgecolor="black", linewidths=0.8, zorder=6)

    ax.axhline(base.ENDOWMENT / 2, color="gray", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_ylim(-0.5, base.ENDOWMENT + 0.5)
    ax.set_xlim(-1.5, 22.5)
    ax.set_xticks(CHECKPOINTS)
    ax.set_xticklabels([str(r) for r in CHECKPOINTS])
    ax.set_xlabel("Round", fontsize=TICK_SIZE)
    ax.set_title(base.MODEL_LABELS[model], fontsize=LABEL_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.2)


def plot_contribution_distributions(agent_store, mean_store):
    fig, axes = plt.subplots(1, len(base.MODELS), figsize=(4.3 * len(base.MODELS), 4.8),
                             sharey=True)
    for ax, model in zip(axes, base.MODELS):
        plot_one_model(agent_store, mean_store, model, ax)
    axes[0].set_ylabel("Contribution", fontsize=LABEL_SIZE)

    handles = [
        Line2D([0], [0], marker=base.COND_MARKERS[c], color=base.COND_COLORS[c],
               linestyle="none", markersize=8, markeredgecolor="black",
               label=base.COND_LABELS[c])
        for c in base.PLOT_CONDITIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(base.PLOT_CONDITIONS),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Agent-level contribution distributions across interaction conditions",
                 fontsize=LABEL_SIZE + 1, y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_dir = os.path.join(FIG_ROOT, "cross_model")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "contribution_distributions_local.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: contribution_distributions_local.png/.pdf -> {out_dir}")


if __name__ == "__main__":
    print("Building agent-level store ...")
    agent_store = build_agent_level_store()
    print("Building run-clustered mean store ...")
    mean_store = base.build_contribution_store()
    plot_contribution_distributions(agent_store, mean_store)
