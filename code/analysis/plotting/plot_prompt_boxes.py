"""
plot_prompt_boxes.py
====================
Renders all five LLM prompt templates as paper-style color boxes.
Output: figures/2026-03-22/combined/prompt_boxes.pdf  (+ .png)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

OUT_DIR = "/data3/rasimura/social-norm-evo/figures/2026-03-22/combined"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
PROMPTS = [
    {
        "label":    "System Prompt",
        "tag":      "SYSTEM",
        "color":    "#2c5f8a",      # slate blue
        "bg":       "#eaf2f8",
        "body": (
            "You are an autonomous agent in a repeated group interaction.\n"
            "You make decisions about contributing to a shared group fund.\n\n"
            "You have a cooperation tendency of [t] (0–1):\n"
            "  0 → strongly prioritize own material payoff\n"
            "  1 → strongly prioritize fairness and group wellbeing\n\n"
            "Your tendency influences decisions but does not rigidly determine them.\n"
            "Adjust behavior based on: others' contributions, evaluations,\n"
            "group expectations, and payoff consequences of past actions.\n\n"
            "Output STRICT JSON only with the exact keys requested. No extra text."
        ),
    },
    {
        "label":    "Discussion Prompt",
        "tag":      "DISCUSSION",
        "color":    "#276e47",      # forest green
        "bg":       "#eafaf1",
        "body": (
            "Round: [t]\n"
            "Task: Group discussion before partner selection and contribution.\n\n"
            "Your recent observations and reflections:\n"
            "  [memory — last N rounds]\n\n"
            "Discussion so far:\n"
            "  Agent i: ...\n\n"
            "Share what you think the group should do this round and why.\n"
            "Refer to what others said if relevant. Keep it to 1–3 sentences.\n\n"
            'Output JSON:  { "message": "<your statement>" }'
        ),
    },
    {
        "label":    "Decision Prompt",
        "tag":      "DECISION",
        "color":    "#8a4a00",      # amber
        "bg":       "#fef6ec",
        "body": (
            "Round: [t]\n"
            "Task: Choose your contribution to the group fund.\n\n"
            "Rules:\n"
            "  Endowment = E tokens;  contribution c ∈ [0, E]\n"
            "  Group fund = Σ(contributions) × multiplier\n"
            "  Payoff = (E − c) + fund / group_size\n\n"
            "Your group this round: [agent IDs]\n"
            "Discussion this round: [transcript]\n"
            "Norm reflection from last round:\n"
            "  Injunctive norm: [value];  Descriptive norm: [value]\n"
            "  Expected from others: [value];  Others expect from me: [value]\n\n"
            "Earlier rounds summary: [compressed memory]\n"
            "Recent rounds (detailed): [last N rounds]\n\n"
            'Output JSON:  { "contribution": <int 0–E> }'
        ),
    },
    {
        "label":    "Evaluation Prompt",
        "tag":      "EVALUATION",
        "color":    "#6b2d8a",      # violet
        "bg":       "#f5eefa",
        "body": (
            "Round: [t]\n"
            "Task: Evaluate each group member based on their contribution.\n\n"
            "Your contribution: [c] / E\n"
            "Group average contribution: [avg] / E\n\n"
            "Partners and their contributions:\n"
            "  Agent i: contributed [c_i] / E\n\n"
            "Rate each partner from −1.0 (strongly disapprove)\n"
            "to +1.0 (strongly approve).\n"
            "Ratings influence who you play with in future rounds.\n\n"
            'Output JSON:  { "evaluations": { "i": <float −1 to 1>, ... } }'
        ),
    },
    {
        "label":    "Perception Prompt",
        "tag":      "PERCEPTION",
        "color":    "#1a6b6b",      # teal
        "bg":       "#e8f8f8",
        "body": (
            "Round [t] — Post-round reflection.\n\n"
            "This round:\n"
            "  Your contribution: [c] / E;  Group average: [avg] / E\n"
            "  Your payoff: [π];  Cooperation tendency: [t] (0–1)\n"
            "  Partner contributions: Agent i: [c_i] / E, ...\n\n"
            "Earlier rounds summary: [compressed memory]\n"
            "Recent rounds (detailed): [last N rounds]\n\n"
            "Output JSON:\n"
            '  { "injunctive_norm": <float|null>,\n'
            '    "descriptive_norm": <float>,\n'
            '    "expectation_of_others": "<text>",\n'
            '    "others_expectation_of_me": "<text>",\n'
            '    "preferred_partners": [ids],\n'
            '    "agents_to_avoid": [ids] }'
        ),
    },
]

# ── Layout ─────────────────────────────────────────────────────────────────────
N = len(PROMPTS)
FIG_W = 7.2          # inches (single-column paper width)
ROW_H = 2.05         # inches per box
GAP   = 0.18         # gap between boxes
FIG_H = N * ROW_H + (N - 1) * GAP + 0.15

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")

# We'll draw each box as axes with manual positioning
for i, p in enumerate(PROMPTS):
    # y position from top (matplotlib y goes bottom→top, so flip)
    y_top = 1.0 - (i * (ROW_H + GAP)) / FIG_H
    y_bot = y_top - ROW_H / FIG_H
    ax = fig.add_axes([0.0, y_bot, 1.0, ROW_H / FIG_H])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    MARGIN = 0.018

    # ── background box ────────────────────────────────────────────────────────
    bg = FancyBboxPatch(
        (MARGIN, 0.01), 1 - 2 * MARGIN, 0.98,
        boxstyle="round,pad=0.01",
        facecolor=p["bg"], edgecolor=p["color"], linewidth=1.2,
        transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(bg)

    # ── header bar ────────────────────────────────────────────────────────────
    HEADER_H = 0.185
    header = FancyBboxPatch(
        (MARGIN, 1 - HEADER_H - 0.01), 1 - 2 * MARGIN, HEADER_H,
        boxstyle="round,pad=0.005",
        facecolor=p["color"], edgecolor="none",
        transform=ax.transAxes, zorder=2, clip_on=False,
    )
    ax.add_patch(header)

    # ── tag pill ──────────────────────────────────────────────────────────────
    ax.text(
        MARGIN + 0.012, 1 - HEADER_H / 2 - 0.01,
        p["tag"],
        transform=ax.transAxes,
        fontsize=7.2, fontweight="bold", color="white",
        fontfamily="monospace",
        va="center", ha="left", zorder=3,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white", alpha=0.18,
            edgecolor="none",
        ),
    )

    # ── header label ──────────────────────────────────────────────────────────
    ax.text(
        0.5, 1 - HEADER_H / 2 - 0.01,
        p["label"],
        transform=ax.transAxes,
        fontsize=9, fontweight="bold", color="white",
        va="center", ha="center", zorder=3,
    )

    # ── body text ─────────────────────────────────────────────────────────────
    ax.text(
        MARGIN + 0.02, 1 - HEADER_H - 0.065,
        p["body"],
        transform=ax.transAxes,
        fontsize=7.0, color="#1a1a1a",
        fontfamily="monospace",
        va="top", ha="left", zorder=3,
        linespacing=1.45,
    )

plt.savefig(os.path.join(OUT_DIR, "prompt_boxes.pdf"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(OUT_DIR, "prompt_boxes.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {OUT_DIR}/prompt_boxes.pdf")
print(f"Saved → {OUT_DIR}/prompt_boxes.png")
