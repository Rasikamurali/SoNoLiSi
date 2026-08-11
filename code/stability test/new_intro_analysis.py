"""
new_intro_analysis.py
---------------------
Analysis of the new-agent introduction experiment (new_intro_SoNoLiFi_v5).

20 normal rounds → new agent introduced → 5 extension rounds (21–25).

Three figures:

  A — Extension rounds: original 12 vs new agent
        contribution (left) and payoff (right)

  B — Extension rounds: original 12 vs new agent
        injunctive norm IN (left) and descriptive norm DN (right)
        [perception-active conditions only; PURE_BASELINE excluded]

  C — Pre and post introduction: original 12 agents only
        contribution, payoff, IN, DN across all 25 rounds
        vertical line marks round 20 (introduction point)

In A and B:  solid lines  = original 12 agents (mean across agents × seeds)
             dashed lines = new agent (mean across seeds)
             colours      = conditions

In C:        colours = conditions, full 1–25 trajectory, shaded SE

Output: figures/2026-03-31/new_intro/
"""

import json
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS_DIR = "/data3/rasimura/social-norm-evo/results/gpt/new_intro"
FIG_ROOT    = "/data3/rasimura/social-norm-evo/figures/2026-03-31/new_intro"

SEEDS       = list(range(46, 53))
NORM_ROUNDS = list(range(1, 21))
EXT_ROUNDS  = list(range(21, 26))
ALL_ROUNDS  = NORM_ROUNDS + EXT_ROUNDS

ALL_CONDITIONS  = ["PURE_BASELINE", "BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]
PERC_CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]

COND_LS = {
    "PURE_BASELINE": ":",
    "BASELINE":      "-.",
    "NO_SELECTION":  "--",
    "NO_DISCUSSION": (0, (5, 2)),
    "FULL":          "-",
}
COND_COLORS = {
    "PURE_BASELINE": "#7f7f7f",
    "BASELINE":      "#bcbd22",
    "NO_SELECTION":  "#1f77b4",
    "NO_DISCUSSION": "#9467bd",
    "FULL":          "#2ca02c",
}
COND_LABELS = {
    "PURE_BASELINE": "Pure Baseline",
    "BASELINE":      "Baseline",
    "NO_SELECTION":  "No Selection",
    "NO_DISCUSSION": "No Discussion",
    "FULL":          "Full",
}

LABEL_SIZE = 16
TICK_SIZE  = 12
TITLE_SIZE = 13

os.makedirs(FIG_ROOT, exist_ok=True)


# ─── I/O ──────────────────────────────────────────────────────────────────────

def load_logs(condition: str) -> list:
    pattern = os.path.join(RESULTS_DIR, "**", "log_v5_newintro_*.json")
    out = []
    for fpath in glob.glob(pattern, recursive=True):
        with open(fpath) as f:
            run = json.load(f)
        if run.get("condition") == condition:
            out.append(run)
    return out


def savefig(fig, fname: str) -> None:
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_ROOT, fname.replace(".pdf", f".{ext}"))
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
    plt.close(fig)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def mean_se(series_list):
    """Mean ± SE across a list of equal-length 1-D arrays."""
    arr = np.array(series_list, dtype=float)
    m   = np.nanmean(arr, axis=0)
    se  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    return m, se


def get_new_agent_id(run: dict) -> int:
    for rlog in run["round_logs"]:
        nai = rlog.get("new_agent_introduced")
        if nai is not None:
            return int(nai)
    raise ValueError("No new_agent_introduced found")


def get_incumbent_ids(run: dict) -> set:
    new_id = get_new_agent_id(run)
    r20    = next(r for r in run["round_logs"] if r["round"] == 20)
    return {a["id"] for a in r20["agent_states"] if a["id"] != new_id}


def rlog_by_round(run: dict) -> dict:
    return {r["round"]: r for r in run["round_logs"]}


# ─── Store builders ───────────────────────────────────────────────────────────

def build_ext_store(conditions=ALL_CONDITIONS) -> dict:
    """
    Extension rounds (21–25) per-seed series for:
      orig_contribution, orig_payoff — mean over 12 incumbent agents
      new_contribution,  new_payoff  — new agent value

    store[cond][metric] = list of arrays, one per seed, length 5
    """
    store = {}
    for cond in conditions:
        store[cond] = defaultdict(list)
        for run in load_logs(cond):
            new_id  = get_new_agent_id(run)
            inc_ids = get_incumbent_ids(run)
            logs    = rlog_by_round(run)

            orig_c, orig_p, new_c, new_p = [], [], [], []
            for t in EXT_ROUNDS:
                if t not in logs:
                    orig_c.append(np.nan); orig_p.append(np.nan)
                    new_c.append(np.nan);  new_p.append(np.nan)
                    continue
                rl = logs[t]
                c  = rl["contributions"]
                p  = rl["payoffs"]

                inc_c = [float(v) for k, v in c.items() if int(k) in inc_ids and v is not None]
                inc_p = [float(v) for k, v in p.items() if int(k) in inc_ids and v is not None]
                orig_c.append(np.mean(inc_c) if inc_c else np.nan)
                orig_p.append(np.mean(inc_p) if inc_p else np.nan)

                new_c.append(float(c.get(str(new_id), c.get(new_id, np.nan))))
                new_p.append(float(p.get(str(new_id), p.get(new_id, np.nan))))

            store[cond]["orig_contribution"].append(np.array(orig_c))
            store[cond]["orig_payoff"].append(np.array(orig_p))
            store[cond]["new_contribution"].append(np.array(new_c))
            store[cond]["new_payoff"].append(np.array(new_p))

    return store


def build_perc_store(conditions=PERC_CONDITIONS) -> dict:
    """
    Extension rounds (21–25) per-seed series for IN and DN.

    store[cond][metric] = list of arrays, one per seed, length 5
    metrics: orig_IN, orig_DN, new_IN, new_DN
    """
    store = {}
    for cond in conditions:
        store[cond] = defaultdict(list)
        for run in load_logs(cond):
            new_id  = get_new_agent_id(run)
            inc_ids = get_incumbent_ids(run)
            logs    = rlog_by_round(run)

            orig_in, orig_dn, new_in, new_dn = [], [], [], []
            for t in EXT_ROUNDS:
                if t not in logs:
                    orig_in.append(np.nan); orig_dn.append(np.nan)
                    new_in.append(np.nan);  new_dn.append(np.nan)
                    continue
                perc = logs[t].get("perceptions") or {}

                inc_in = [v["injunctive_norm"]  for k, v in perc.items()
                          if int(k) in inc_ids and v.get("injunctive_norm") is not None]
                inc_dn = [v["descriptive_norm"] for k, v in perc.items()
                          if int(k) in inc_ids and v.get("descriptive_norm") is not None]
                orig_in.append(np.mean(inc_in) if inc_in else np.nan)
                orig_dn.append(np.mean(inc_dn) if inc_dn else np.nan)

                new_entry = perc.get(str(new_id), perc.get(new_id, {})) or {}
                new_in.append(new_entry.get("injunctive_norm", np.nan))
                new_dn.append(new_entry.get("descriptive_norm", np.nan))

            store[cond]["orig_IN"].append(np.array(orig_in, dtype=float))
            store[cond]["orig_DN"].append(np.array(orig_dn, dtype=float))
            store[cond]["new_IN"].append(np.array(new_in,  dtype=float))
            store[cond]["new_DN"].append(np.array(new_dn,  dtype=float))

    return store


def build_full_store(conditions=ALL_CONDITIONS) -> dict:
    """
    All 25 rounds, original 12 agents only.

    store[cond][metric] = list of arrays, one per seed, length 25
    metrics: contribution, payoff, IN, DN
             (IN/DN will be NaN for PURE_BASELINE and rounds without perceptions)
    """
    store = {}
    for cond in conditions:
        store[cond] = defaultdict(list)
        for run in load_logs(cond):
            inc_ids = get_incumbent_ids(run)
            logs    = rlog_by_round(run)

            c_series, p_series, in_series, dn_series = [], [], [], []
            for t in ALL_ROUNDS:
                if t not in logs:
                    c_series.append(np.nan); p_series.append(np.nan)
                    in_series.append(np.nan); dn_series.append(np.nan)
                    continue
                rl   = logs[t]
                c    = rl["contributions"]
                p    = rl["payoffs"]
                perc = rl.get("perceptions") or {}

                inc_c  = [float(v) for k, v in c.items() if int(k) in inc_ids and v is not None]
                inc_p  = [float(v) for k, v in p.items() if int(k) in inc_ids and v is not None]
                inc_in = [v["injunctive_norm"]  for k, v in perc.items()
                          if int(k) in inc_ids and v.get("injunctive_norm") is not None]
                inc_dn = [v["descriptive_norm"] for k, v in perc.items()
                          if int(k) in inc_ids and v.get("descriptive_norm") is not None]

                c_series.append(np.mean(inc_c)  if inc_c  else np.nan)
                p_series.append(np.mean(inc_p)  if inc_p  else np.nan)
                in_series.append(np.mean(inc_in) if inc_in else np.nan)
                dn_series.append(np.mean(inc_dn) if inc_dn else np.nan)

            store[cond]["contribution"].append(np.array(c_series))
            store[cond]["payoff"].append(np.array(p_series))
            store[cond]["IN"].append(np.array(in_series, dtype=float))
            store[cond]["DN"].append(np.array(dn_series, dtype=float))

    return store


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _cond_legend(conditions, extra_handles=None):
    handles = [
        mlines.Line2D([], [], color=COND_COLORS[c], ls=COND_LS[c],
                      lw=1.8, label=COND_LABELS[c])
        for c in conditions
    ]
    if extra_handles:
        handles += extra_handles
    return handles


# ─── Figure A: contribution and payoff, ext rounds, orig vs new ───────────────

def plot_A(estore):
    print("Plotting A: extension contribution & payoff …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.array(EXT_ROUNDS)

    for ax, orig_key, new_key, ylabel, title in [
        (axes[0], "orig_contribution", "new_contribution",
         "Mean contribution", "Contribution (rounds 21–25)"),
        (axes[1], "orig_payoff",       "new_payoff",
         "Mean payoff",      "Payoff (rounds 21–25)"),
    ]:
        for cond in ALL_CONDITIONS:
            d = estore[cond]
            m_o, se_o = mean_se(d[orig_key])
            m_n, se_n = mean_se(d[new_key])
            col = COND_COLORS[cond]
            ls  = COND_LS[cond]

            # original 12 — solid
            ax.plot(x, m_o, color=col, ls=ls, lw=2.0)
            ax.fill_between(x, m_o - se_o, m_o + se_o, color=col, alpha=0.12)
            # new agent — dashed
            ax.plot(x, m_n, color=col, ls=ls, lw=1.2, alpha=0.65,
                    marker="o", markersize=3)

        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xticks(EXT_ROUNDS)
        ax.tick_params(labelsize=TICK_SIZE)

    orig_h = mlines.Line2D([], [], color="black", lw=2.0, label="Original 12")
    new_h  = mlines.Line2D([], [], color="black", lw=1.2, alpha=0.65,
                            marker="o", markersize=3, label="New agent")
    fig.legend(
        handles=_cond_legend(ALL_CONDITIONS) + [orig_h, new_h],
        loc="lower center", ncol=4, fontsize=TICK_SIZE,
        frameon=True, bbox_to_anchor=(0.5, -0.10),
    )
    fig.tight_layout()
    savefig(fig, "A_ext_contribution_payoff.pdf")


# ─── Figure B: IN and DN, ext rounds, orig vs new ─────────────────────────────

def plot_B(pstore):
    print("Plotting B: extension IN & DN …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.array(EXT_ROUNDS)

    for ax, orig_key, new_key, ylabel, title in [
        (axes[0], "orig_IN", "new_IN",
         "Injunctive norm (IN)",  "IN (rounds 21–25)"),
        (axes[1], "orig_DN", "new_DN",
         "Descriptive norm (DN)", "DN (rounds 21–25)"),
    ]:
        for cond in PERC_CONDITIONS:
            d = pstore[cond]
            m_o, se_o = mean_se(d[orig_key])
            m_n, se_n = mean_se(d[new_key])
            col = COND_COLORS[cond]
            ls  = COND_LS[cond]

            ax.plot(x, m_o, color=col, ls=ls, lw=2.0)
            ax.fill_between(x, m_o - se_o, m_o + se_o, color=col, alpha=0.12)
            ax.plot(x, m_n, color=col, ls=ls, lw=1.2, alpha=0.65,
                    marker="o", markersize=3)

        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xticks(EXT_ROUNDS)
        ax.tick_params(labelsize=TICK_SIZE)

    orig_h = mlines.Line2D([], [], color="black", lw=2.0, label="Original 12")
    new_h  = mlines.Line2D([], [], color="black", lw=1.2, alpha=0.65,
                            marker="o", markersize=3, label="New agent")
    fig.legend(
        handles=_cond_legend(PERC_CONDITIONS) + [orig_h, new_h],
        loc="lower center", ncol=3, fontsize=TICK_SIZE,
        frameon=True, bbox_to_anchor=(0.5, -0.10),
    )
    fig.tight_layout()
    savefig(fig, "B_ext_norms.pdf")


# ─── Figure C: full trajectory, original 12 only ─────────────────────────────

def plot_C(fstore):
    print("Plotting C: full trajectory original 12 …")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=False)
    x = np.array(ALL_ROUNDS)

    metrics = [
        ("contribution", "Mean contribution",    "Contribution",    ALL_CONDITIONS),
        ("payoff",       "Mean payoff",           "Payoff",          ALL_CONDITIONS),
        ("IN",           "Injunctive norm (IN)",  "IN",              PERC_CONDITIONS),
        ("DN",           "Descriptive norm (DN)", "DN",              PERC_CONDITIONS),
    ]

    for ax, (key, ylabel, title, conds) in zip(axes, metrics):
        for cond in conds:
            d = fstore[cond]
            m, se = mean_se(d[key])
            col = COND_COLORS[cond]
            ls  = COND_LS[cond]
            ax.plot(x, m, color=col, ls=ls, lw=1.8, label=COND_LABELS[cond])
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.12)

        ax.axvline(20.5, color="black", ls="--", lw=1.1, alpha=0.55)
        ymin, ymax = ax.get_ylim()
        ax.text(20.7, ymin + (ymax - ymin) * 0.03,
                "new agent\nintro", fontsize=8, va="bottom", alpha=0.6)

        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)

    # shared legend below
    handles_all  = _cond_legend(ALL_CONDITIONS)
    handles_perc = _cond_legend(PERC_CONDITIONS)
    # use all-conditions handles (superset)
    fig.legend(
        handles=handles_all,
        loc="lower center", ncol=5, fontsize=TICK_SIZE,
        frameon=True, bbox_to_anchor=(0.5, -0.10),
    )
    #fig.suptitle("Original 12 Agents — Pre & Post Introduction (rounds 1–25)",
     #            fontsize=TITLE_SIZE + 1, y=1.01)
    fig.tight_layout()
    savefig(fig, "C_full_trajectory.pdf")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building stores …")
    estore = build_ext_store()
    pstore = build_perc_store()
    fstore = build_full_store()

    print("Generating plots …")
    plot_A(estore)
    plot_B(pstore)
    plot_C(fstore)

    print("Done.")
