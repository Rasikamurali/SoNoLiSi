"""
Here, we want to do a deep dive into the perception of agents.

Perception captures:
a) injunctive norms (what should be done)
b) descriptive norms (what is done)
c) Social preferences

First, we will look at the distribution of injunctive norm.
In only the baseline condition, we look at the distribution per agent of IN and DN.
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ─── Paths ────────────────────────────────────────────────────────────────────
RESULTS  = "/data3/rasimura/social-norm-evo/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/2026-03-22"
MODELS   = ["gpt", "llama", "mistral", "qwen"]
VARIANTS = ["global"]
SEEDS    = list(range(43, 53))
ENDOWMENT = 10.0

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_latest_log(model, variant, seed, condition):
    pattern = os.path.join(RESULTS, model, variant, f"seed{seed}", "log_*.json")
    cond_best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            cond_best[d["condition"]] = d
        except Exception:
            continue
    return cond_best.get(condition)


def savefig(fig, model, fname):
    out_dir = os.path.join(FIG_ROOT, model, "perception_analysis")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Data collection ──────────────────────────────────────────────────────────

def collect_perception_data(model, variant, condition="BASELINE"):
    """
    Returns:
      per_agent_inj[agent_id]  = list of injunctive norm values across all rounds & seeds
      per_agent_desc[agent_id] = list of descriptive norm values across all rounds & seeds
      per_round_inj[round]     = list of injunctive norm values across all agents & seeds
      per_round_desc[round]    = list of descriptive norm values across all agents & seeds
    """
    per_agent_inj  = defaultdict(list)
    per_agent_desc = defaultdict(list)
    per_round_inj  = defaultdict(list)
    per_round_desc = defaultdict(list)

    for seed in SEEDS:
        d = load_latest_log(model, variant, seed, condition)
        if d is None:
            continue
        for r in d["round_logs"]:
            rnd = r["round"]
            percs = r.get("perceptions") or {}
            for aid_str, perc in percs.items():
                if not perc:
                    continue
                aid = int(aid_str)
                inj  = perc.get("injunctive_norm")
                desc = perc.get("descriptive_norm")
                if inj is not None:
                    per_agent_inj[aid].append(float(inj))
                    per_round_inj[rnd].append(float(inj))
                if desc is not None:
                    per_agent_desc[aid].append(float(desc))
                    per_round_desc[rnd].append(float(desc))

    return per_agent_inj, per_agent_desc, per_round_inj, per_round_desc


# ─── Plot A: Distribution of IN and DN per agent (violin) ─────────────────────

def plot_per_agent_distribution(model):
    """
    For BASELINE condition: violin plot of injunctive and descriptive norm
    values per agent, pooled across all seeds and rounds.
    One figure per variant (global / local).
    """
    print(f"\n[{model.upper()}] Plot A — per-agent IN & DN distribution (BASELINE)")

    for variant in VARIANTS:
        inj_data, desc_data, _, _ = collect_perception_data(model, variant, "BASELINE")

        if not inj_data and not desc_data:
            print(f"  No data for {variant}")
            continue

        agent_ids = sorted(set(list(inj_data.keys()) + list(desc_data.keys())))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle(
            f"{model.upper()} — {variant.capitalize()} — BASELINE\n"
            f"Distribution of Injunctive & Descriptive Norm per Agent (pooled across seeds & rounds)",
            fontsize=10
        )

        for ax, data_dict, label, color in [
            (axes[0], inj_data,  "Injunctive Norm",  "#4e79a7"),
            (axes[1], desc_data, "Descriptive Norm", "#f28e2b"),
        ]:
            plot_data = [data_dict.get(aid, []) for aid in agent_ids]
            plot_data = [d for d in plot_data if d]  # drop empties
            if not plot_data:
                ax.set_title(f"{label} — no data")
                continue

            vp = ax.violinplot(plot_data, positions=range(len(plot_data)),
                               showmedians=True, showextrema=True)
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.6)
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.5)

            # overlay mean per agent
            means = [np.mean(d) for d in plot_data]
            ax.scatter(range(len(plot_data)), means, color="black",
                       zorder=5, s=20, label="Mean")

            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels([f"A{aid}" for aid in agent_ids[:len(plot_data)]],
                               rotation=45, fontsize=7)
            ax.set_ylim(-0.5, ENDOWMENT + 0.5)
            ax.set_ylabel(f"{label} (0–{int(ENDOWMENT)})", fontsize=9)
            ax.set_xlabel("Agent", fontsize=9)
            ax.set_title(label, fontsize=10)
            ax.axhline(ENDOWMENT / 2, color="grey", linestyle="--", alpha=0.4,
                       label=f"Midpoint ({int(ENDOWMENT/2)})")
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        savefig(fig, model, f"perception_per_agent_IN_DN_baseline_{variant}.png")


# ─── Plot B: IN and DN over rounds (mean ± SD across agents & seeds) ──────────

def plot_norm_trajectories(model):
    """
    For BASELINE: mean ± SD of injunctive and descriptive norm over rounds,
    global vs local on the same axes.
    """
    print(f"\n[{model.upper()}] Plot B — IN & DN trajectories over rounds (BASELINE)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        f"{model.upper()} — BASELINE — Injunctive & Descriptive Norm over Rounds\n"
        f"(mean ± SD across agents & seeds)",
        fontsize=10
    )

    variant_colors = {"global": "#1f77b4", "local": "#d62728"}
    variant_ls     = {"global": "-",       "local": "--"}

    for variant in VARIANTS:
        _, _, per_round_inj, per_round_desc = collect_perception_data(model, variant, "BASELINE")
        color = variant_colors[variant]
        ls    = variant_ls[variant]

        for ax, per_round, label in [
            (axes[0], per_round_inj,  "Injunctive Norm"),
            (axes[1], per_round_desc, "Descriptive Norm"),
        ]:
            if not per_round:
                continue
            rounds = sorted(per_round.keys())
            means  = [np.mean(per_round[r]) for r in rounds]
            sds    = [np.std(per_round[r])  for r in rounds]
            ax.plot(rounds, means, color=color, ls=ls, linewidth=2,
                    marker="o", markersize=3, label=variant.capitalize())
            ax.fill_between(rounds,
                            [m - s for m, s in zip(means, sds)],
                            [m + s for m, s in zip(means, sds)],
                            color=color, alpha=0.15)

    for ax, label in [(axes[0], "Injunctive Norm"), (axes[1], "Descriptive Norm")]:
        ax.set_ylim(-0.5, ENDOWMENT + 0.5)
        ax.axhline(ENDOWMENT / 2, color="grey", linestyle="--", alpha=0.4)
        ax.set_xlabel("Round", fontsize=9)
        ax.set_ylabel(f"{label} (0–{int(ENDOWMENT)})", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.tight_layout()
    savefig(fig, model, "perception_IN_DN_trajectory_baseline_global_vs_local.png")


# ─── Plot C: IN vs DN scatter per agent per round ─────────────────────────────

def plot_in_vs_dn_scatter(model):
    """
    Scatter of injunctive vs descriptive norm, one point per agent per round per seed,
    BASELINE condition. Global and local as separate panels.
    Shows whether agents' descriptive and injunctive norms track each other.
    """
    print(f"\n[{model.upper()}] Plot C — IN vs DN scatter (BASELINE)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{model.upper()} — BASELINE — Injunctive vs Descriptive Norm\n"
        f"(each point = one agent × round × seed)",
        fontsize=10
    )

    for ax, variant in zip(axes, VARIANTS):
        inj_vals, desc_vals = [], []
        for seed in SEEDS:
            d = load_latest_log(model, variant, seed, "BASELINE")
            if d is None:
                continue
            for r in d["round_logs"]:
                percs = r.get("perceptions") or {}
                for perc in percs.values():
                    if not perc:
                        continue
                    inj  = perc.get("injunctive_norm")
                    desc = perc.get("descriptive_norm")
                    if inj is not None and desc is not None:
                        inj_vals.append(float(inj))
                        desc_vals.append(float(desc))

        if inj_vals:
            ax.scatter(desc_vals, inj_vals, alpha=0.15, s=10, color="#4e79a7")
            # diagonal reference line
            ax.plot([0, ENDOWMENT], [0, ENDOWMENT], "k--", linewidth=1, alpha=0.5,
                    label="IN = DN")
            # regression line
            if len(inj_vals) > 2:
                m, b = np.polyfit(desc_vals, inj_vals, 1)
                x_line = np.linspace(0, ENDOWMENT, 100)
                ax.plot(x_line, m * x_line + b, "r-", linewidth=1.5,
                        label=f"fit: slope={m:.2f}")
            corr = np.corrcoef(desc_vals, inj_vals)[0, 1]
            ax.set_title(f"{variant.capitalize()}  (r={corr:.3f})", fontsize=10)
        else:
            ax.set_title(f"{variant.capitalize()} — no data")

        ax.set_xlim(-0.5, ENDOWMENT + 0.5)
        ax.set_ylim(-0.5, ENDOWMENT + 0.5)
        ax.set_xlabel("Descriptive Norm (perceived behavior)", fontsize=9)
        ax.set_ylabel("Injunctive Norm (perceived ought)", fontsize=9)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    savefig(fig, model, "perception_IN_vs_DN_scatter_baseline.png")


def plot_per_seed_per_agent(model):
    """
    For BASELINE condition, for each seed:
      - One figure: injunctive norm per agent across rounds (one line per agent)
      - One figure: descriptive norm per agent across rounds (one line per agent)
    Saved under perception_analysis/per_seed/
    """
    print(f"\n[{model.upper()}] Plot D — per-seed per-agent IN & DN across rounds (BASELINE)")

    cmap = plt.get_cmap("tab20")

    for variant in VARIANTS:
        out_dir = os.path.join(FIG_ROOT, model, "perception_analysis", "per_seed", variant)
        os.makedirs(out_dir, exist_ok=True)

        for seed in SEEDS:
            d = load_latest_log(model, variant, seed, "BASELINE")
            if d is None:
                continue

            # Build per-agent per-round series
            agent_inj  = defaultdict(dict)   # agent_inj[aid][round] = value
            agent_desc = defaultdict(dict)

            for r in d["round_logs"]:
                rnd   = r["round"]
                percs = r.get("perceptions") or {}
                for aid_str, perc in percs.items():
                    if not perc:
                        continue
                    aid = int(aid_str)
                    inj  = perc.get("injunctive_norm")
                    desc = perc.get("descriptive_norm")
                    if inj  is not None: agent_inj[aid][rnd]  = float(inj)
                    if desc is not None: agent_desc[aid][rnd] = float(desc)

            all_agents = sorted(set(list(agent_inj.keys()) + list(agent_desc.keys())))
            if not all_agents:
                continue

            rounds = sorted({rnd for r in d["round_logs"] for rnd in [r["round"]]})

            for norm_data, norm_label, fname_part in [
                (agent_inj,  "Injunctive Norm",  "injunctive"),
                (agent_desc, "Descriptive Norm", "descriptive"),
            ]:
                fig, ax = plt.subplots(figsize=(11, 5))
                fig.suptitle(
                    f"{model.upper()} — {variant.capitalize()} — seed {seed} — BASELINE\n"
                    f"{norm_label} per Agent across Rounds",
                    fontsize=10
                )

                for idx, aid in enumerate(all_agents):
                    vals = [norm_data[aid].get(r, np.nan) for r in rounds]
                    color = cmap(idx % 20)
                    ax.plot(rounds, vals, color=color, linewidth=1.5,
                            marker="o", markersize=3, label=f"Agent {aid}")

                ax.set_ylim(-0.5, ENDOWMENT + 0.5)
                ax.axhline(ENDOWMENT / 2, color="grey", linestyle="--",
                           alpha=0.4, linewidth=1)
                ax.set_xlabel("Round", fontsize=9)
                ax.set_ylabel(f"{norm_label} (0–{int(ENDOWMENT)})", fontsize=9)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.legend(fontsize=14, bbox_to_anchor=(1.01, 1), loc="upper left",
                          ncol=1, framealpha=0.7)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                path = os.path.join(out_dir, f"seed{seed:02d}_{norm_label.lower().replace(' ','_')}_per_agent.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved → {path}")



# ─── Plot E: Per seed — IN & DN over rounds by CT group ───────────────────────

CT_GROUPS = {
    "Violators (0–0.3)": (0.0, 0.3),
    "Normal (0.3–0.7)":  (0.3, 0.7),
    "Cooperators (0.7–1.0)": (0.7, 1.0),
}
CT_COLORS = {
    "Violators (0–0.3)":     "#d62728",
    "Normal (0.3–0.7)":      "#ff7f0e",
    "Cooperators (0.7–1.0)": "#2ca02c",
}


def plot_norms_by_ct_group(model):
    """
    For BASELINE condition, per seed:
    Plot mean injunctive norm and descriptive norm across rounds,
    split by CT group (violators / normal / cooperators).
    One figure per seed per variant: 2 panels (IN top, DN bottom), 3 lines per panel.
    Saved under perception_analysis/per_seed/ct_groups/<variant>/
    """
    print(f"\n[{model.upper()}] Plot E — IN & DN by CT group per seed (BASELINE)")

    for variant in VARIANTS:
        out_dir = os.path.join(
            FIG_ROOT, model, "perception_analysis", "per_seed", "ct_groups", variant
        )
        os.makedirs(out_dir, exist_ok=True)

        for seed in SEEDS:
            d = load_latest_log(model, variant, seed, "BASELINE")
            if d is None:
                continue

            # per round, per CT group: collect IN and DN values
            group_inj  = {g: defaultdict(list) for g in CT_GROUPS}
            group_desc = {g: defaultdict(list) for g in CT_GROUPS}

            for r in d["round_logs"]:
                rnd   = r["round"]
                cts   = {a["id"]: a["cooperation_tendency"] for a in r["agent_states"]}
                percs = r.get("perceptions") or {}

                for aid_str, perc in percs.items():
                    if not perc:
                        continue
                    aid = int(aid_str)
                    ct  = cts.get(aid)
                    if ct is None:
                        continue
                    inj  = perc.get("injunctive_norm")
                    desc = perc.get("descriptive_norm")

                    for group_name, (lo, hi) in CT_GROUPS.items():
                        if lo <= ct < hi or (hi == 1.0 and ct == 1.0):
                            if inj  is not None: group_inj[group_name][rnd].append(float(inj))
                            if desc is not None: group_desc[group_name][rnd].append(float(desc))
                            break

            rounds = sorted({r["round"] for r in d["round_logs"]})

            fig, (ax_inj, ax_desc) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            fig.suptitle(
                f"{model.upper()} — {variant.capitalize()} — seed {seed} — BASELINE\n"
                f"Injunctive & Descriptive Norm by CT Group across Rounds",
                fontsize=10
            )

            for group_name, color in CT_COLORS.items():
                for ax, group_data, norm_label in [
                    (ax_inj,  group_inj[group_name],  "Injunctive Norm"),
                    (ax_desc, group_desc[group_name], "Descriptive Norm"),
                ]:
                    means = [np.mean(group_data[r]) if group_data.get(r) else np.nan for r in rounds]
                    # only plot if we have at least some data
                    if any(not np.isnan(m) for m in means):
                        ax.plot(rounds, means, color=color, linewidth=2,
                                marker="o", markersize=4, label=group_name)

            for ax, norm_label in [(ax_inj, "Injunctive Norm"), (ax_desc, "Descriptive Norm")]:
                ax.set_ylim(-0.5, ENDOWMENT + 0.5)
                ax.axhline(ENDOWMENT / 2, color="grey", linestyle="--", alpha=0.4, linewidth=1)
                ax.set_ylabel(f"{norm_label} (0–{int(ENDOWMENT)})", fontsize=9)
                ax.legend(fontsize=14, loc="upper left")
                ax.grid(True, alpha=0.3)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

            ax_desc.set_xlabel("Round", fontsize=9)
            plt.tight_layout()

            path = os.path.join(out_dir, f"seed{seed:02d}_IN_DN_by_CT_group.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved → {path}")


# ─── Plot G: DN vs IN scatter — models × conditions ──────────────────────────
#
# One figure: 4 rows (models) × 4 cols (conditions)
# x = descriptive norm, y = injunctive norm
# each point = (agent, round) pooled across seeds
# global (blue) vs local (red) as separate scatter series
# diagonal IN=DN reference + per-variant regression line + Pearson r

SCATTER_CONDITIONS = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION"]
SCATTER_COND_LABELS = {
    "BASELINE":      "Baseline",
    "FULL":          "Full",
    "NO_DISCUSSION": "No Discussion",
    "NO_SELECTION":  "No Selection",
}

def plot_dn_vs_in_scatter(save_dir=None):
    """
    Single figure: 4 rows (models) × 4 cols (conditions).
    Each panel: scatter DN (x) vs IN (y), global=blue, local=red.
    Points pooled across all seeds and agents.
    """
    print(f"\nPlot G — DN vs IN scatter (models × conditions)")

    nrows = len(MODELS)
    ncols = len(SCATTER_CONDITIONS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.5),
                             sharex=True, sharey=True)

    variant_style = {
        "global": {"color": "#2c7bb6", "label": "Global", "marker": "o"},
        "local":  {"color": "#d7191c", "label": "Local",  "marker": "s"},
    }

    for row, model in enumerate(MODELS):
        for col, cond in enumerate(SCATTER_CONDITIONS):
            ax = axes[row, col]

            any_data = False
            for variant in VARIANTS:
                dn_vals, in_vals = [], []
                for seed in SEEDS:
                    d = load_latest_log(model, variant, seed, cond)
                    if d is None:
                        continue
                    for r in d["round_logs"]:
                        percs = r.get("perceptions") or {}
                        for perc in percs.values():
                            if not perc:
                                continue
                            inj  = perc.get("injunctive_norm")
                            desc = perc.get("descriptive_norm")
                            if inj is not None and desc is not None:
                                dn_vals.append(float(desc))
                                in_vals.append(float(inj))

                if not dn_vals:
                    continue
                any_data = True
                style = variant_style[variant]
                ax.scatter(dn_vals, in_vals, color=style["color"],
                           alpha=0.15, s=8, edgecolors="none",
                           label=style["label"], marker=style["marker"])

                # regression line
                if len(dn_vals) > 2 and np.std(dn_vals) > 0:
                    from scipy import stats as _stats
                    r_val, _ = _stats.pearsonr(dn_vals, in_vals)
                    m, b = np.polyfit(dn_vals, in_vals, 1)
                    x_line = np.linspace(0, ENDOWMENT, 100)
                    ax.plot(x_line, m * x_line + b, color=style["color"],
                            lw=1.5, linestyle="--",
                            label=f"r={r_val:.2f}")

            if any_data:
                # diagonal IN = DN
                ax.plot([0, ENDOWMENT], [0, ENDOWMENT], color="grey",
                        lw=1, linestyle=":", alpha=0.6)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="grey")

            ax.set_xlim(-0.5, ENDOWMENT + 0.5)
            ax.set_ylim(-0.5, ENDOWMENT + 0.5)
            ax.grid(True, alpha=0.2)

            # labels
            if row == 0:
                ax.set_title(SCATTER_COND_LABELS[cond], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{model.upper()}\nInjunctive norm", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("Descriptive norm", fontsize=9)
            if row == 0 and col == 0 and any_data:
                ax.legend(fontsize=14, markerscale=2)

    fig.suptitle(
        "DN (x) vs IN (y) — each point = (agent, round) pooled across seeds\n"
        "Blue = Global  |  Red = Local  |  dashed line = regression  |  grey dotted = IN=DN",
        fontsize=11,
    )
    plt.tight_layout()

    out_dir = save_dir or os.path.join(FIG_ROOT, "perception_analysis")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "perception_in_dn_scatter_final.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─── Plot F: Final summary — IN & DN global vs local, all conditions pooled ───

ALL_CONDITIONS = ["BASELINE", "FULL", "NO_DISCUSSION", "NO_SELECTION", "PURE_BASELINE"]

def plot_final_summary(model):
    """
    One figure per model.
    Layout: 1 row × 2 cols  (Injunctive Norm | Descriptive Norm)
    Lines:  global (blue solid) vs local (red dashed)
    Data:   mean ± std across all conditions, seeds, and agents at each round.
    Method: per seed per condition, compute mean over agents at each round,
            then pool all (seed × condition) series and take mean ± std.
    """
    print(f"\n[{model.upper()}] Plot F — final summary IN & DN (global vs local, all conditions)")

    variant_style = {
        "global": {"color": "#2c7bb6", "ls": "-",  "label": "Global"},
        "local":  {"color": "#d7191c", "ls": "--", "label": "Local"},
    }

    # collect: {variant: {norm: [ [series per round] per (seed×condition) ]}}
    series = {v: {"injunctive_norm": [], "descriptive_norm": []} for v in VARIANTS}

    for variant in VARIANTS:
        for condition in ALL_CONDITIONS:
            for seed in SEEDS:
                d = load_latest_log(model, variant, seed, condition)
                if d is None:
                    continue
                rounds = sorted(r["round"] for r in d["round_logs"])
                # per round: mean over agents
                inj_by_round  = {}
                desc_by_round = {}
                for r in d["round_logs"]:
                    rnd   = r["round"]
                    percs = r.get("perceptions") or {}
                    inj_vals, desc_vals = [], []
                    for perc in percs.values():
                        if not perc:
                            continue
                        v = perc.get("injunctive_norm")
                        if v is not None:
                            inj_vals.append(float(v))
                        v = perc.get("descriptive_norm")
                        if v is not None:
                            desc_vals.append(float(v))
                    if inj_vals:
                        inj_by_round[rnd]  = np.mean(inj_vals)
                    if desc_vals:
                        desc_by_round[rnd] = np.mean(desc_vals)

                max_round = max(rounds) if rounds else 0
                if inj_by_round:
                    series[variant]["injunctive_norm"].append(
                        [inj_by_round.get(r, np.nan) for r in range(1, max_round + 1)]
                    )
                if desc_by_round:
                    series[variant]["descriptive_norm"].append(
                        [desc_by_round.get(r, np.nan) for r in range(1, max_round + 1)]
                    )

    # determine round range
    max_rounds = max(
        (len(s) for v in series.values() for ss in v.values() for s in ss),
        default=0
    )
    if max_rounds == 0:
        print(f"  No data for {model}")
        return

    rounds = list(range(1, max_rounds + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    norm_info = [
        ("injunctive_norm",  "Injunctive Norm"),
        ("descriptive_norm", "Descriptive Norm"),
    ]

    for ax, (norm_key, norm_label) in zip(axes, norm_info):
        for variant in VARIANTS:
            slist = series[variant][norm_key]
            if not slist:
                continue
            # pad shorter series with nan so array is rectangular
            max_len = max(len(s) for s in slist)
            arr = np.full((len(slist), max_len), np.nan)
            for i, s in enumerate(slist):
                arr[i, :len(s)] = s

            mean = np.nanmean(arr, axis=0)
            std  = np.nanstd(arr,  axis=0)
            r    = list(range(1, max_len + 1))

            style = variant_style[variant]
            ax.plot(r, mean, color=style["color"], ls=style["ls"],
                    lw=2.5, marker="o", ms=3, label=style["label"])
            ax.fill_between(r, mean - std, mean + std,
                            color=style["color"], alpha=0.15)

        ax.axhline(ENDOWMENT / 2, color="grey", ls=":", lw=1, alpha=0.5,
                   label=f"Midpoint ({int(ENDOWMENT/2)})")
        ax.set_ylim(-0.5, ENDOWMENT + 0.5)
        ax.set_xlim(0.5, max_rounds + 0.5)
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel(f"{norm_label} (0–{int(ENDOWMENT)})", fontsize=10)
        ax.set_title(norm_label, fontsize=11)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    fig.suptitle(
        f"{model.upper()} — Injunctive & Descriptive Norm over Rounds\n"
        f"Global (solid) vs Local (dashed) | mean ± std across all conditions, seeds & agents",
        fontsize=11,
    )
    plt.tight_layout()
    savefig(fig, model, "perception_final_summary.png")


# ─── Table helpers ────────────────────────────────────────────────────────────

TABLE_DIR = os.path.join(FIG_ROOT, "perception_tables")

def _collect_in_dn_pairs(model, variant, condition):
    """
    Returns (in_vals, dn_vals, total_obs, in_count, dn_count) pooled across seeds.
    total_obs = number of (agent, round) observations with a perceptions entry.
    in_count  = observations where injunctive_norm is not None.
    dn_count  = observations where descriptive_norm is not None.
    """
    in_vals, dn_vals = [], []
    total_obs = in_count = dn_count = 0
    for seed in SEEDS:
        d = load_latest_log(model, variant, seed, condition)
        if d is None:
            continue
        for r in d["round_logs"]:
            percs = r.get("perceptions") or {}
            for perc in percs.values():
                if not isinstance(perc, dict):
                    continue
                total_obs += 1
                inj  = perc.get("injunctive_norm")
                desc = perc.get("descriptive_norm")
                if inj  is not None: in_count  += 1
                if desc is not None: dn_count  += 1
                if inj is not None and desc is not None:
                    in_vals.append(float(inj))
                    dn_vals.append(float(desc))
    return in_vals, dn_vals, total_obs, in_count, dn_count


# ─── Table 1: DN–IN correlation per model × condition ─────────────────────────
#
# Columns: Model | Condition | r_global | p_global | r_local | p_local
# Pooled across all seeds and agents.

def make_correlation_table():
    from scipy import stats as _stats
    from tabulate import tabulate

    os.makedirs(TABLE_DIR, exist_ok=True)
    rows = []

    for model in MODELS:
        for cond in SCATTER_CONDITIONS:
            row = {"Model": model.upper(), "Condition": SCATTER_COND_LABELS[cond]}
            for variant in VARIANTS:
                in_vals, dn_vals, _, _, _ = _collect_in_dn_pairs(model, variant, cond)
                if len(in_vals) > 2 and np.std(dn_vals) > 0 and np.std(in_vals) > 0:
                    r, p = _stats.pearsonr(dn_vals, in_vals)
                    sig  = "*" if p < 0.05 else " "
                    row[f"r ({variant})"] = f"{sig}{r:.3f}"
                    row[f"p ({variant})"] = f"{p:.3f}" if p >= 0.001 else "<.001"
                else:
                    row[f"r ({variant})"] = "—"
                    row[f"p ({variant})"] = "—"
            rows.append(row)

    cols = ["Model", "Condition",
            "r (global)", "p (global)", "r (local)", "p (local)"]
    df = pd.DataFrame(rows, columns=cols)

    # print
    print("\n" + "="*70)
    print("  Table 1 — DN–IN Pearson correlation  (* p < .05)")
    print("="*70)
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    # save
    csv_path = os.path.join(TABLE_DIR, "correlation_in_dn.csv")
    tex_path = os.path.join(TABLE_DIR, "correlation_in_dn.tex")
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, escape=True,
                        caption="Pearson correlation between descriptive norm (DN) and injunctive norm (IN) "
                                "per model and condition, pooled across all seeds and agents. "
                                "Asterisk marks $p < .05$.",
                        label="tab:corr_in_dn",
                        column_format="llrrrr")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"\n  Saved CSV  → {csv_path}")
    print(f"  Saved LaTeX → {tex_path}")
    return df


# ─── Table 2: IN / DN expression rate per model × condition ───────────────────
#
# Columns: Model | Condition | Setting | Total obs | IN expressed (%) | DN expressed (%)
#          | Both (%) | Neither (%)

def make_expression_table():
    from tabulate import tabulate

    os.makedirs(TABLE_DIR, exist_ok=True)
    rows = []

    for model in MODELS:
        for cond in SCATTER_CONDITIONS:
            for variant in VARIANTS:
                in_vals, dn_vals, total, in_cnt, dn_cnt = \
                    _collect_in_dn_pairs(model, variant, cond)

                if total == 0:
                    rows.append({
                        "Model":       model.upper(),
                        "Condition":   SCATTER_COND_LABELS[cond],
                        "Setting":     variant.capitalize(),
                        "Total obs":   0,
                        "IN expr. (%)": "—",
                        "DN expr. (%)": "—",
                        "Both (%)":    "—",
                        "Neither (%)": "—",
                    })
                    continue

                both    = len(in_vals)           # pairs where both non-None
                neither = total - in_cnt - dn_cnt + both  # inclusion-exclusion
                rows.append({
                    "Model":        model.upper(),
                    "Condition":    SCATTER_COND_LABELS[cond],
                    "Setting":      variant.capitalize(),
                    "Total obs":    total,
                    "IN expr. (%)": f"{100 * in_cnt / total:.1f}",
                    "DN expr. (%)": f"{100 * dn_cnt / total:.1f}",
                    "Both (%)":     f"{100 * both    / total:.1f}",
                    "Neither (%)":  f"{100 * neither / total:.1f}",
                })

    cols = ["Model", "Condition", "Setting", "Total obs",
            "IN expr. (%)", "DN expr. (%)", "Both (%)", "Neither (%)"]
    df = pd.DataFrame(rows, columns=cols)

    # print
    print("\n" + "="*70)
    print("  Table 2 — IN / DN expression rate (not expressed = field is missing/None)")
    print("="*70)
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    # save
    csv_path = os.path.join(TABLE_DIR, "expression_in_dn.csv")
    tex_path = os.path.join(TABLE_DIR, "expression_in_dn.tex")
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(index=False, escape=True,
                        caption="Expression rate of injunctive norm (IN) and descriptive norm (DN) "
                                "per model, condition, and setting. "
                                "Not expressed means the field is absent or null in the perception output. "
                                "Total obs = number of (agent, round) perception entries.",
                        label="tab:expr_in_dn",
                        column_format="lllrrrrrr")
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"\n  Saved CSV  → {csv_path}")
    print(f"  Saved LaTeX → {tex_path}")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for model in MODELS:
        plot_per_agent_distribution(model)
        plot_norm_trajectories(model)
        plot_in_vs_dn_scatter(model)
        plot_per_seed_per_agent(model)
        plot_norms_by_ct_group(model)
        plot_final_summary(model)
    plot_dn_vs_in_scatter()
    make_correlation_table()
    make_expression_table()
    print("\nAll done.")
