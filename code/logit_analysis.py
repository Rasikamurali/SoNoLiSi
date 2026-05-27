"""
logit_analysis.py
-----------------
Compares the point-estimate perception (what agents state) against the
token-level logprob distribution (what the model internally considers)
captured in logit_SoNoLiFi_v5 runs.

Three figures:

  A — Entropy trajectory (rounds 1–20)
        How certain is the model when it commits to each norm value?
        1×2: IN entropy (left), DN entropy (right), by condition.
        Lower entropy = more confident; rising entropy = growing uncertainty.

  B — Point estimate vs distribution expected value (EV)
        Scatter of (point estimate, EV from probs) per agent-round,
        coloured by entropy. One panel per norm (IN, DN).
        Points far from the diagonal = stated answer diverges from the
        model's internal probability mass centre.

  C — Perception accuracy: point estimate vs EV vs actual contribution
        Per condition, compare two error measures across rounds:
          |point_IN  − actual_contribution|  (what the agent said)
          |EV_IN     − actual_contribution|  (centre of mass of distribution)
        1×2 (IN left, DN right); solid = point estimate error,
        dashed = distribution EV error.

Output: figures/2026-03-31/logit/
"""

import json
import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
from collections import defaultdict

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS_DIR = "/data3/rasimura/social-norm-evo/results/gpt/logit"
FIG_ROOT    = "/data3/rasimura/social-norm-evo/figures/2026-03-31/logit"

ROUNDS          = list(range(1, 21))
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
    pattern = os.path.join(RESULTS_DIR, "**", "log_v5_logit_*.json")
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


# ─── Utilities ────────────────────────────────────────────────────────────────

def mean_se(series_list):
    arr = np.array(series_list, dtype=float)
    m   = np.nanmean(arr, axis=0)
    se  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
    return m, se


def dist_ev(probs: dict) -> float:
    """Expected value of a probability distribution over levels 0–10."""
    if probs is None:
        return np.nan
    return sum(int(k) * v for k, v in probs.items())


# ─── Store builders ───────────────────────────────────────────────────────────

def build_entropy_store(conditions=PERC_CONDITIONS) -> dict:
    """
    Per-round mean entropy across agents, one value per seed per round.

    store[cond]["IN"]  = list of arrays length 20 (one per seed)
    store[cond]["DN"]  = same
    """
    store = {}
    for cond in conditions:
        store[cond] = {"IN": [], "DN": []}
        for run in load_logs(cond):
            logs = {r["round"]: r for r in run["round_logs"]}
            in_series, dn_series = [], []
            for t in ROUNDS:
                perc = (logs[t].get("perceptions") or {}) if t in logs else {}
                in_vals = [p["injunctive_norm_entropy"]  for p in perc.values()
                           if p.get("injunctive_norm_entropy") is not None]
                dn_vals = [p["descriptive_norm_entropy"] for p in perc.values()
                           if p.get("descriptive_norm_entropy") is not None]
                in_series.append(np.mean(in_vals) if in_vals else np.nan)
                dn_series.append(np.mean(dn_vals) if dn_vals else np.nan)
            store[cond]["IN"].append(np.array(in_series))
            store[cond]["DN"].append(np.array(dn_series))
    return store


def build_scatter_data(conditions=PERC_CONDITIONS) -> dict:
    """
    All (point_estimate, EV, entropy) triples pooled across seeds × rounds × agents.

    store[cond]["IN"] = {"pt": [...], "ev": [...], "ent": [...]}
    store[cond]["DN"] = same
    """
    store = {}
    for cond in conditions:
        store[cond] = {
            "IN": {"pt": [], "ev": [], "ent": []},
            "DN": {"pt": [], "ev": [], "ent": []},
        }
        for run in load_logs(cond):
            for rlog in run["round_logs"]:
                for p in (rlog.get("perceptions") or {}).values():
                    for norm, key in [("IN", "injunctive"), ("DN", "descriptive")]:
                        pt  = p.get(f"{key}_norm")
                        prb = p.get(f"{key}_norm_probs")
                        ent = p.get(f"{key}_norm_entropy")
                        ev  = dist_ev(prb)
                        if pt is not None and not np.isnan(ev):
                            store[cond][norm]["pt"].append(float(pt))
                            store[cond][norm]["ev"].append(float(ev))
                            store[cond][norm]["ent"].append(float(ent) if ent else 0.0)
    return store


def build_accuracy_store(conditions=PERC_CONDITIONS) -> dict:
    """
    Per-round mean absolute error: |point_estimate − actual| and |EV − actual|,
    where 'actual' is the mean contribution of the agent's group that round.

    store[cond]["IN_pt"]  = list of arrays length 20  (point estimate error)
    store[cond]["IN_ev"]  = same using distribution EV
    store[cond]["DN_pt"]  = ...
    store[cond]["DN_ev"]  = ...
    """
    store = {}
    for cond in conditions:
        store[cond] = {"IN_pt": [], "IN_ev": [], "DN_pt": [], "DN_ev": []}
        for run in load_logs(cond):
            logs = {r["round"]: r for r in run["round_logs"]}

            in_pt_s, in_ev_s, dn_pt_s, dn_ev_s = [], [], [], []
            for t in ROUNDS:
                if t not in logs:
                    for s in [in_pt_s, in_ev_s, dn_pt_s, dn_ev_s]:
                        s.append(np.nan)
                    continue

                rl       = logs[t]
                contribs = rl["contributions"]
                groups   = rl["groups"]
                perc     = rl.get("perceptions") or {}

                # Build group-mean contribution lookup per agent
                agent_group_mean: dict = {}
                for group in groups:
                    gvals = [float(contribs.get(str(i), contribs.get(i, 0)))
                             for i in group]
                    gmean = np.mean(gvals)
                    for i in group:
                        agent_group_mean[i] = gmean

                in_pt_r, in_ev_r, dn_pt_r, dn_ev_r = [], [], [], []
                for aid_str, p in perc.items():
                    aid = int(aid_str)
                    if aid not in agent_group_mean:
                        continue

                    actual = agent_group_mean[aid]

                    for errs_pt, errs_ev, pt_key, prb_key in [
                        (in_pt_r, in_ev_r, "injunctive_norm",  "injunctive_norm_probs"),
                        (dn_pt_r, dn_ev_r, "descriptive_norm", "descriptive_norm_probs"),
                    ]:
                        pt  = p.get(pt_key)
                        prb = p.get(prb_key)
                        ev  = dist_ev(prb)
                        if pt is not None:
                            errs_pt.append(abs(float(pt) - actual))
                        if not np.isnan(ev):
                            errs_ev.append(abs(ev - actual))

                in_pt_s.append(np.mean(in_pt_r) if in_pt_r else np.nan)
                in_ev_s.append(np.mean(in_ev_r) if in_ev_r else np.nan)
                dn_pt_s.append(np.mean(dn_pt_r) if dn_pt_r else np.nan)
                dn_ev_s.append(np.mean(dn_ev_r) if dn_ev_r else np.nan)

            store[cond]["IN_pt"].append(np.array(in_pt_s))
            store[cond]["IN_ev"].append(np.array(in_ev_s))
            store[cond]["DN_pt"].append(np.array(dn_pt_s))
            store[cond]["DN_ev"].append(np.array(dn_ev_s))

    return store


# ─── Figure A: Entropy trajectory ─────────────────────────────────────────────

def plot_A(estore):
    print("Plotting A: entropy trajectory …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.array(ROUNDS)

    for ax, key, title, ylabel in [
        (axes[0], "IN", "IN — Token-Level Entropy",  "Mean entropy (nats)"),
        (axes[1], "DN", "DN — Token-Level Entropy",  "Mean entropy (nats)"),
    ]:
        for cond in PERC_CONDITIONS:
            m, se = mean_se(estore[cond][key])
            ax.plot(x, m, color=COND_COLORS[cond], ls=COND_LS[cond],
                    lw=1.8, label=COND_LABELS[cond])
            ax.fill_between(x, m - se, m + se,
                            color=COND_COLORS[cond], alpha=0.15)

        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=TICK_SIZE)

    handles = [
        mlines.Line2D([], [], color=COND_COLORS[c], ls=COND_LS[c],
                      lw=1.8, label=COND_LABELS[c])
        for c in PERC_CONDITIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=TICK_SIZE, frameon=True,
               bbox_to_anchor=(0.5, -0.08))
    #fig.suptitle("Model Uncertainty in Perception: Shannon Entropy of Token Logprobs",
    #             fontsize=TITLE_SIZE + 1, y=1.01)
    fig.tight_layout()
    savefig(fig, "A_entropy_trajectory.pdf")


# ─── Figure B: Point estimate vs distribution EV scatter ─────────────────────

def plot_B(sdata):
    print("Plotting B: point estimate vs distribution EV …")

    # Pool across all conditions for cleaner scatter; highlight divergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, norm, title in [
        (axes[0], "IN", "Injunctive Norm (IN)"),
        (axes[1], "DN", "Descriptive Norm (DN)"),
    ]:
        all_pt, all_ev, all_ent = [], [], []
        for cond in PERC_CONDITIONS:
            all_pt.extend(sdata[cond][norm]["pt"])
            all_ev.extend(sdata[cond][norm]["ev"])
            all_ent.extend(sdata[cond][norm]["ent"])

        pt  = np.array(all_pt)
        ev  = np.array(all_ev)
        ent = np.array(all_ent)

        # Colour by entropy
        sc = ax.scatter(pt, ev, c=ent, cmap="YlOrRd", s=8, alpha=0.5,
                        vmin=0, vmax=np.percentile(ent, 95))
        plt.colorbar(sc, ax=ax, label="Entropy (nats)")

        # Identity line
        lo = min(pt.min(), ev.min()) - 0.3
        hi = max(pt.max(), ev.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", alpha=0.5)

        # Annotation: % within ±0.5 of diagonal
        pct = np.mean(np.abs(pt - ev) < 0.5) * 100
        ax.text(0.04, 0.96, f"{pct:.0f}% within ±0.5",
                transform=ax.transAxes, fontsize=TICK_SIZE,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_xlabel("Point estimate", fontsize=LABEL_SIZE)
        ax.set_ylabel("Distribution EV", fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=TICK_SIZE)

    #fig.suptitle(
    #    "Stated Perception (point estimate) vs Model's Internal Distribution (EV)\n"
    #    "Colour = entropy — high entropy means the model was uncertain at the token level",
    #    fontsize=TITLE_SIZE, y=1.02,
    #)
    fig.tight_layout()
    savefig(fig, "B_point_vs_ev_scatter.pdf")


# ─── Figure C: Accuracy — point estimate vs EV ────────────────────────────────

def plot_C(astore):
    print("Plotting C: perception accuracy — point vs EV …")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.array(ROUNDS)

    for ax, pt_key, ev_key, title, ylabel in [
        (axes[0], "IN_pt", "IN_ev",
         "Injunctive Norm — Perception Error",  "|Perceived IN − actual contribution|"),
        (axes[1], "DN_pt", "DN_ev",
         "Descriptive Norm — Perception Error", "|Perceived DN − actual contribution|"),
    ]:
        for cond in PERC_CONDITIONS:
            col = COND_COLORS[cond]
            ls  = COND_LS[cond]

            m_pt, se_pt = mean_se(astore[cond][pt_key])
            m_ev, se_ev = mean_se(astore[cond][ev_key])

            # Point estimate error — solid line
            ax.plot(x, m_pt, color=col, ls=ls, lw=2.0,
                    label=COND_LABELS[cond])
            ax.fill_between(x, m_pt - se_pt, m_pt + se_pt,
                            color=col, alpha=0.10)

            # Distribution EV error — same colour, dashed
            ax.plot(x, m_ev, color=col, ls=ls, lw=1.0,
                    alpha=0.55, marker="x", markersize=3, markevery=3)

        ax.set_xlabel("Round", fontsize=LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=TICK_SIZE)

    cond_handles = [
        mlines.Line2D([], [], color=COND_COLORS[c], ls=COND_LS[c],
                      lw=1.8, label=COND_LABELS[c])
        for c in PERC_CONDITIONS
    ]
    pt_h  = mlines.Line2D([], [], color="black", lw=2.0,  label="Point estimate error")
    ev_h  = mlines.Line2D([], [], color="black", lw=1.0, alpha=0.55,
                           marker="x", markersize=5, label="Distribution EV error")
    fig.legend(
        handles=cond_handles + [pt_h, ev_h],
        loc="lower center", ncol=3, fontsize=TICK_SIZE,
        frameon=True, bbox_to_anchor=(0.5, -0.10),
    )
    #fig.suptitle(
    #    "Perception Accuracy: Stated Answer vs Distribution Expected Value\n"
    #    "Both measured against actual group mean contribution",
    #    fontsize=TITLE_SIZE, y=1.02,
    #)
    fig.tight_layout()
    savefig(fig, "C_accuracy_pt_vs_ev.pdf")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building stores …")
    estore = build_entropy_store()
    sdata  = build_scatter_data()
    astore = build_accuracy_store()

    print("Generating plots …")
    plot_A(estore)
    plot_B(sdata)
    plot_C(astore)

    print("Done.")
